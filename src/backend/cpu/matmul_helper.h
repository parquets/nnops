#pragma once
/// @file matmul_helper.h
/// @brief Inner matrix multiplication — pack + MMA micro-kernel dispatch.
///
/// All arch-specific pack/mma headers share an identical API in namrspaces
///   nnops::backend::cpu::x86_64   and   nnops::backend::cpu::aarch64.
/// This header picks the right set via NNOPS_ARCH_* and exposes the active
/// namrspace as `arch`, so consumrr code stays clean of #ifdef.

#include "nnops/detail/simd/cpu_features.hpp"  // NNOPS_ARCH_X86_64 / NNOPS_ARCH_AARCH64
#include "nnops/ops/matmul.hpp"                // MatMulAttributes
#include "nnops/core/tensor_view.hpp"          // TensorDesc

#include <algorithm>
#include <cstddef>
#include <type_traits>

// ---- arch-specific includes --------------------------------------------

#ifdef NNOPS_ARCH_X86_64
#include "x86_64/pack_f16.hpp"
#include "x86_64/pack_f32.hpp"
#include "x86_64/mma_direct_f16.hpp"
#include "x86_64/mma_direct_f32.hpp"
#include "x86_64/mma_pack_f16.hpp"
#include "x86_64/mma_pack_f32.hpp"
#elif defined(NNOPS_ARCH_AARCH64)
#include "aarch64/pack_f16.hpp"
#include "aarch64/pack_f32.hpp"
#include "aarch64/mma_direct_f16.hpp"
#include "aarch64/mma_direct_f32.hpp"
#include "aarch64/mma_pack_f16.hpp"
#include "aarch64/mma_pack_f32.hpp"
#endif

#include "nnops/detail/half.hpp"

// 64-byte cache-line alignment for packed panel strides
#define PANEL_ALIGN_BYTES 64

namespace nnops::backend::cpu {

// ---- namrspace alias ---------------------------------------------------

#ifdef NNOPS_ARCH_X86_64
namespace arch = x86_64;
#elif defined(NNOPS_ARCH_AARCH64)
namespace arch = aarch64;
#endif

// Panel size arrays (largest-first decomposition)
constexpr int MR_F32[3] = {arch::mr_f32[0], arch::mr_f32[1], arch::mr_f32[2]};
constexpr int NR_F32[3] = {arch::nr_f32[0], arch::nr_f32[1], arch::nr_f32[2]};
constexpr int MR_F16[3] = {arch::mr_f16[0], arch::mr_f16[1], arch::mr_f16[2]};
constexpr int NR_F16[3] = {arch::nr_f16[0], arch::nr_f16[1], arch::nr_f16[2]};

constexpr int MR_MAX_F32 = arch::mr_f32[0];
constexpr int NR_MAX_F32 = arch::nr_f32[0];
constexpr int MR_MAX_F16 = arch::mr_f16[0];
constexpr int NR_MAX_F16 = arch::nr_f16[0];

template <class T>
constexpr int mr_max_flt() {
    if constexpr (std::is_same_v<T, float>) {
        return MR_MAX_F32;
    } else if constexpr (std::is_same_v<T, half>) {
        return MR_MAX_F16;
    } else {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>, "Unsupported type");
    }
}

template <class T>
constexpr int nr_max_flt() {
    if constexpr (std::is_same_v<T, float>) {
        return NR_MAX_F32;
    } else if constexpr (std::is_same_v<T, half>) {
        return NR_MAX_F16;
    } else {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T  , half>, "Unsupported type");
    }
}

template <int AlignBytes>
constexpr int align_up(int n) {
    return (n + AlignBytes - 1) & ~(AlignBytes - 1);
}

// =========================================================================
//  Tiling constants (shared by matmul.cpp kernels and workspace sizing)
// =========================================================================

constexpr int KC_F32   = 128;
constexpr int KC_F16   = 256;
constexpr int KC_I8    = 512;   // int8: larger Kc since elements are 1 byte
constexpr int KC_F16I4 = 256;   // fp16×int4: placeholder (future hardware)

constexpr int MC_TARGET = 144;  // 144/6=24 (x86), 144/8=18 (aarch64)

// 64-byte aligned panel strides (in elements): ldd = align_up(mr_max * kc * sizeof(T), 64) / sizeof(T)
constexpr int LDD_A_F32 = (MR_MAX_F32 * KC_F32 * 4 + 63) / 64 * 16;
constexpr int LDD_B_F32 = (NR_MAX_F32 * KC_F32 * 4 + 63) / 64 * 16;
constexpr int LDD_A_F16 = (MR_MAX_F16 * KC_F16 * 2 + 63) / 64 * 32;
constexpr int LDD_B_F16 = (NR_MAX_F16 * KC_F16 * 2 + 63) / 64 * 32;

// =========================================================================
//  Nc from L2 constraint
// =========================================================================
// From: (mr × Kc + Nc × Kc + mr × Nc) × 2 × elem_size < L2_SIZE
// → Nc < (L2_SIZE / (2 × elem_size) - mr × Kc) / (Kc + mr)

inline int compute_nc(int mr_max, int kc, int elem_bytes, size_t l2_size) noexcept {
    int denom = 2 * elem_bytes;
    int rhs   = static_cast<int>(l2_size / denom) - mr_max * kc;
    denom = kc + mr_max;
    return (rhs / denom) - 1;  // -1 for safety margin
}

inline int round_down_nc(int nc, int nr_max) noexcept {
    return (nc / nr_max) * nr_max;
}

/// Clamp nc into [nr_max, N] so the NKM loop always makes progress and the
/// workspace sizing matches the kernel exactly (N >= 1 guaranteed by shape).
inline int clamp_nc(int nc, int nr_max, int N) noexcept {
    if (nc < nr_max) { nc = nr_max; }
    if (nc > N)      { nc = N; }
    return nc;
}

/// Resolve the tile sizes (mc, nc) for a GEMM of M×N with the given panel
/// maxima and Kc. Single source of truth for both kernels and sizing.
inline void resolve_tile_sizes(int M, int N, int mr_max, int nr_max, int kc,
                               int elem_bytes, int& mc, int& nc) noexcept {
    mc = std::min(MC_TARGET, M);
    size_t l2_size = simd::CpuFeatures::get().l2_cache_size();
    nc = clamp_nc(round_down_nc(compute_nc(mr_max, kc, elem_bytes, l2_size), nr_max),
                  nr_max, N);
}

// =========================================================================
//  Pack decision — static cost model (single source of truth)
// =========================================================================

/// Static pack decision for one GEMM computation scope.
struct PackPlan {
    bool pack_a = false;     ///< A packed into the stack buffer (transpose_a or wide stride)
    bool pack_b = false;     ///< B packed into the workspace (transpose_b or wide stride)
    bool mkn_order = false;  ///< MKN loop order (only when pack_a && !pack_b)
};

/// Page-size heuristic (elements): 4096 / sizeof(T) → f32 1024, f16 2048.
/// A row stride at or above one page makes the NKM/MKN panel reads page-scattered,
/// so packing into a contiguous 64-byte-aligned panel wins — but only when the
/// matrix is actually re-read across tile blocks (see the reuse guards below).
template <class T>
constexpr int pack_threshold_elems() {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>,
                  "pack_threshold_elems: only float/half are supported");
    return 4096 / static_cast<int>(sizeof(T));
}

/// Decide whether A and B should be packed for a GEMM over the row/column range
/// M_s × N_s, given the physical row strides lda/ldb (elements).
///
/// M_s / N_s are the range *participating in this computation scope*: the full
/// M×N for the global plan (workspace sizing, split-dimension choice), or a
/// per-slab range for the stack-resident A pack refinement (reuse counts must be
/// measured against what the slab actually reads). Rules:
///   1. correctness: transpose_a ⇒ pack_a, transpose_b ⇒ pack_b
///   2. stride+reuse: pack only when the stride is wide AND the panel is re-read
///      across multiple tile blocks (A re-read N_s/nc times, B re-read M_s/mc
///      times); a single-pass read is never worth the extra copy
///   3. loop order: MKN (k{m{n}}) amortizes the A pack; it is only chosen when A
///      is packed and B is raw (packing B too would amplify B-pack traffic)
template <class T>
inline PackPlan make_pack_plan(const MatMulAttributes& attrs,
                               int64_t M_s, int64_t N_s,
                               int64_t lda, int64_t ldb) {
    constexpr int mr_max = mr_max_flt<T>();
    constexpr int nr_max = nr_max_flt<T>();
    constexpr int kc     = std::is_same_v<T, float> ? KC_F32 : KC_F16;
    constexpr int elem   = static_cast<int>(sizeof(T));

    int mc, nc;
    resolve_tile_sizes(static_cast<int>(M_s), static_cast<int>(N_s),
                       mr_max, nr_max, kc, elem, mc, nc);

    const int thr = pack_threshold_elems<T>();

    PackPlan p;
    p.pack_a    = attrs.transpose_a || (N_s > nc && lda > thr);
    p.pack_b    = attrs.transpose_b || (M_s > mc && ldb > thr);
    p.mkn_order = p.pack_a && !p.pack_b;
    return p;
}

/// Number of panels for the largest-first decomposition of `n` into the three
/// sizes nr[0] >= nr[1] >= nr[2] (== 1). This is what the pack/MMA loops
/// actually emit, so workspace sizing must match it exactly (a simple
/// ceil(n / nr[0]) under-counts when the tail decomposes into nr=1 panels).
constexpr int num_panels(int n, const int* nr) noexcept {
    int count = 0;
    int i = 0;
    for (int s = 0; s < 3; ++s) {
        for (; i + nr[s] <= n; i += nr[s]) { ++count; }
    }
    return count;
}

/// Stack-resident packed-A buffer size (elements) for one MC_TARGET×Kc tile at
/// the full-Kc uniform stride — an upper bound for every per-k-block pack:
///   f32: 144×128×4 = 72 KB, f16: 144×256×2 = 72 KB (KC_F16 = 256), both arches.
/// Declared with alignas in the kernel.
template <class T>
constexpr int pack_a_stack_elems() {
    constexpr const int* mr    = std::is_same_v<T, float> ? MR_F32 : MR_F16;
    constexpr int       ldd_a  = std::is_same_v<T, float> ? LDD_A_F32 : LDD_A_F16;
    return num_panels(MC_TARGET, mr) * ldd_a;
}

/// Compute the workspace size (bytes) required by matmul_kernel for the given
/// tensor descriptors and operator attributes.
///
/// A workspace is needed iff the *global* pack plan decides pack_b (transpose_b
/// or a wide non-transposed B row stride). Packed A lives on the kernel stack,
/// so every pack_a-only / direct path returns 0. The size is computed from the
/// global M×N (not the thread count): the per-block panel slices tile the buffer
/// exactly, so the formula below is thread-count invariant and always covers the
/// kernel's pack_b buffer.
size_t matmul_get_workspace_size(const MatMulAttributes& attrs,
                                 const TensorDesc& a_desc,
                                 const TensorDesc& b_desc,
                                 const TensorDesc& c_desc);


// ---- public API --------------------------------------------------------
//
// Pack entry points: panels are laid out at a uniform 64-byte-aligned stride
// `ldd` (elements), so pack and MMA agree on panel placement for all panel
// sizes. The caller computes ldd from the actual k-block length (actual_kc).
//
// MMA entry points: `tile_mma_pack` accumulates C[mr][nr] += A_packed[mr][K]
// × B[nr][K], where `maybe_packed_b`/`ldb` describe B:
//   ldb < 0  → B is packed ([K][nr], row stride nr, panels advance by the
//              uniform aligned stride ldd_b = align_up(nr_max*Kc*elem,64)/elem)
//   ldb >= 0 → B is raw (row stride ldb, advance nr per panel)
// `tile_mma_direct` reads A raw via lda; B follows the same ldb < 0 /
//   ldb >= 0 convention (packed vs raw) as `tile_mma_pack`.

void tile_pack_rhs(bool trans, int nc, int kc, float* dst, int ldd, const float* src, int lds, float scale);
void tile_pack_lhs(bool trans, int mc, int kc, float* dst, int ldd, const float* src, int lds, float scale);
void tile_pack_rhs(bool trans, int nc, int kc, half* dst, int ldd, const half* src, int lds, float scale);
void tile_pack_lhs(bool trans, int mc, int kc, half* dst, int ldd, const half* src, int lds, float scale);

void tile_mma_pack(int mc, int nc, int kc, float* c, int ldc, const float* packed_a, const float* maybe_packed_b, int ldb, float clamp_min, float clamp_max);
void tile_mma_direct(int mc, int nc, int kc, float* c, int ldc, const float* a, int lda, const float* b, int ldb, float clamp_min, float clamp_max);
void tile_mma_pack(int mc, int nc, int kc, half* c, int ldc, const half* packed_a, const half* maybe_packed_b, int ldb, float clamp_min, float clamp_max);
void tile_mma_direct(int mc, int nc, int kc, half* c, int ldc, const half* a, int lda, const half* b, int ldb, float clamp_min, float clamp_max);

// SIMD-accelerated in-place scale: C[i] *= scale
void tile_scale(float* c, int ldc, float scale, int M, int N);
void tile_scale(half* c, int ldc, float scale, int M, int N);

}  // namespace nnops::backend::cpu

