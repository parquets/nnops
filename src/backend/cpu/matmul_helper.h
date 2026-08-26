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

constexpr int MC_TARGET = 192;  // 192/6=32 (x86), 192/8=24 (aarch64)

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

/// Number of panels for the largest-first decomposition of `n` into the three
/// sizes nr[0] >= nr[1] >= nr[2] (== 1). This is what the pack/MMA loops
/// actually emit, so workspace sizing must match it exactly (a simple
/// ceil(n / nr[0]) under-counts when the tail decomposes into nr=1 panels).
inline int num_panels(int n, const int* nr) noexcept {
    int count = 0;
    int i = 0;
    for (int s = 0; s < 3; ++s) {
        for (; i + nr[s] <= n; i += nr[s]) { ++count; }
    }
    return count;
}

/// Workspace = packed A + packed B, each panel laid out at a uniform
/// 64-byte-aligned stride (ldd_a / ldd_b in elements). ldd_a/b are computed
/// from the full Kc, upper-bounding any per-k-block stride.
inline size_t workspace_bytes(int mc, int nc, int ldd_a, int ldd_b, size_t elem,
                              const int* mr, const int* nr) noexcept {
    return (static_cast<size_t>(num_panels(mc, mr)) * static_cast<size_t>(ldd_a) +
            static_cast<size_t>(num_panels(nc, nr)) * static_cast<size_t>(ldd_b)) * elem;
}

/// Compute the workspace size (bytes) required by matmul_kernel for the given
/// tensor descriptors and operator attributes.
///
/// Only the packed path (transpose_a or transpose_b, f32/f16) needs a
/// workspace; all other paths return 0. The kernel uses the same tiling here,
/// so the reported size always covers the kernel's pack buffers.
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

