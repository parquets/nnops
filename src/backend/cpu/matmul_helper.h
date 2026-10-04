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
#include "x86_64/pack_dp4a_i8.hpp"
#include "x86_64/mma_pack_vnni_i8.hpp"
#elif defined(NNOPS_ARCH_AARCH64)
#include "aarch64/pack_f16.hpp"
#include "aarch64/pack_f32.hpp"
#include "aarch64/mma_direct_f16.hpp"
#include "aarch64/mma_direct_f32.hpp"
#include "aarch64/mma_pack_f16.hpp"
#include "aarch64/mma_pack_f32.hpp"
#include "aarch64/pack_dp4a_i8.hpp"
#include "aarch64/mma_pack_i8_dot.hpp"
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

// int8 (s8×s8) panel sizes. The NR list has four entries (extra nr=4 level),
// so int8 pack/mma dispatch uses a 4-level decomposition (num_panels4).
#ifdef NNOPS_ARCH_X86_64
constexpr int MR_I8[3] = {6, 4, 1};
constexpr int NR_I8[4] = {16, 8, 4, 1};
#elif defined(NNOPS_ARCH_AARCH64)
constexpr int MR_I8[3] = {8, 4, 1};
constexpr int NR_I8[4] = {12, 8, 4, 1};
#endif

constexpr int MR_MAX_I8 = MR_I8[0];
constexpr int NR_MAX_I8 = NR_I8[0];

// Hardware int8 dot-product is always used — no compile-time or runtime checks:
//   x86_64: VNNI (vpdpbusd), aarch64: SDOT (vdotq_s32). The build enables the
//   matching ISA (see CMakeLists.txt: /arch:AVXVNNI / -mavxvnni).
//
// x86 VNNI computes u8×s8 (A stored as qa+128); the epilogue folds the −128·Σqw
// correction. aarch64 SDOT computes s8×s8 directly (no offset).
#ifdef NNOPS_ARCH_X86_64
constexpr bool INT8_USE_U8_OFFSET = true;
#else
constexpr bool INT8_USE_U8_OFFSET = false;
#endif

/// M panel height the chosen route will actually run.
///
/// Largest M-panel height a float tile uses. Both routes now share one MR table:
/// packing interleaves four rows per vector (mr=8 on aarch64), and the f32 direct
/// path reads A row-major one vector per row but its widest kernel (8x12) runs
/// un-unrolled, which is what keeps mr=8 inside the register file. Callers that
/// model a working set must charge this height.
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

constexpr int64_t KC_F32   = 128;
constexpr int64_t KC_F16   = 128;   // k-block length also bounds per-block fp16 accumulation error
constexpr int64_t KC_I8    = 512;   // int8: larger Kc since elements are 1 byte
constexpr int64_t KC_F16I4 = 256;   // fp16×int4: placeholder (future hardware)

constexpr int64_t MC_TARGET = 144;  // 144/6=24 (x86), 144/8=18 (aarch64)

// 64-byte aligned panel strides (in elements): ldd = align_up(mr_max * kc * sizeof(T), 64) / sizeof(T)
constexpr int64_t LDD_A_F32 = (MR_MAX_F32 * KC_F32 * 4 + 63) / 64 * 16;
constexpr int64_t LDD_B_F32 = (NR_MAX_F32 * KC_F32 * 4 + 63) / 64 * 16;
constexpr int64_t LDD_A_F16 = (MR_MAX_F16 * KC_F16 * 2 + 63) / 64 * 32;
constexpr int64_t LDD_B_F16 = (NR_MAX_F16 * KC_F16 * 2 + 63) / 64 * 32;

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
inline void resolve_tile_sizes(int64_t  M, int64_t N, int64_t mr_max, int64_t nr_max, int64_t kc,
                               int elem_bytes, int& mc, int& nc) noexcept {
    mc = std::min(MC_TARGET, M);
    size_t l2_size = simd::CpuFeatures::get().l2_cache_size();
    nc = clamp_nc(round_down_nc(compute_nc(mr_max, kc, elem_bytes, l2_size), nr_max),
                  nr_max, N);
}

// =========================================================================
//  Parallel split — block count for the larger-dimension partition
// =========================================================================
//
// The GEMM is split on its larger dimension (N when N > M, else M) and the
// blocks are handed to ctx.cpu.run as equal contiguous slabs. The
// block count is ceil(larger / mc), one slab per A-panel tile — deliberately
// coarser than MLAS's complexity/64K target-thread formula (Complexity / 64K +
// 1 clamped to the pool size). Finer blocks do not help in practice: the GEMM
// is already limited by packed-B bandwidth and, on hybrid P/E-core CPUs, by
// the slow E-cores, so more/smaller blocks only add per-block pack overhead and
// one-panel seams without more parallelism. ceil(larger / mc) fills a modest
// thread pool (8 blocks for M=1024) without those costs.

inline int64_t split_block_count(int64_t larger, int64_t mc) noexcept {
    return std::max<int64_t>(1, (larger + mc - 1) / mc);
}

// =========================================================================
//  Pack decision — simplified (mirrors onnxruntime MLAS sgemm)
// =========================================================================
//
// B (rhs) is always packed into the workspace. A (lhs) is packed only when
// transpose_a forces it (correctness) or when its row stride exceeds one
// cache-friendly page so the panel reads would be page-scattered. The loop
// order is always NKM (n-block outer, k, then m); the MKN and direct orders
// are gone. Tiny GEMMs (M*N*K below the fast-path threshold) skip the tiled
// path entirely and go straight to the reference.

/// A is packed when its row stride exceeds this many elements (wide stride →
/// page-scattered reads), independent of element type.
constexpr int64_t PACK_A_STRIDE_THRESHOLD = 1024;

/// GEMMs with fewer than this many MACs take the fast path (naive reference),
/// skipping pack/tile/threading overhead.
constexpr int64_t GEMM_FAST_PATH_THRESHOLD = 1024;

inline bool gemm_is_small(int64_t M, int64_t N, int64_t K) noexcept {
    return M * N * K < GEMM_FAST_PATH_THRESHOLD;
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

/// 4-level variant for the int8 NR lists ({16,8,4,1} / {12,8,4,1}).
constexpr int num_panels4(int n, const int* nr) noexcept {
    int count = 0;
    int i = 0;
    for (int s = 0; s < 4; ++s) {
        for (; i + nr[s] <= n; i += nr[s]) { ++count; }
    }
    return count;
}

/// Largest `num_panels()` over every mc in [1, n].
///
/// The greedy decomposition above is NOT monotonic in n: with {8,4,1} a 143-row
/// tile needs 21 panels (17×8 + 4 + 1 + 1 + 1) while a 144-row tile needs only
/// 18 (18×8). So sizing a buffer at num_panels(MC_TARGET, mr) is wrong — the
/// *smaller* tile is the one that overflows it:
///
///   aarch64 MR = {8,4,1}: 18 at mc=144, worst 21 at mc=143
///   x86_64  MR = {6,4,1}: 24 at mc=144, worst 26 at mc=141
///
/// Anything holding the panels of an arbitrary mc <= MC_TARGET (the on-stack
/// packed-A tile) must size with this maximum, not with num_panels(n, mr).
constexpr int num_panels_max(int n, const int* mr) noexcept {
    int worst = 0;
    for (int mc = 1; mc <= n; ++mc) {
        const int c = num_panels(mc, mr);
        worst = c > worst ? c : worst;
    }
    return worst;
}

/// Stack-resident packed-A buffer size (elements) for one MC_TARGET×Kc tile at
/// the full-Kc uniform stride — an upper bound for every per-k-block pack:
///   f32 84 KB (aarch64) / 78 KB (x86_64), f16 exactly half of that.
/// The panel count is num_panels_max, not num_panels(MC_TARGET, mr): the tile
/// the kernel actually packs is a per-block mc <= MC_TARGET, whose greedy
/// decomposition can need more panels than MC_TARGET's does (see
/// num_panels_max). Declared with alignas in the kernel.
template <class T>
constexpr int pack_a_stack_elems() {
    constexpr const int* mr    = std::is_same_v<T, float> ? MR_F32 : MR_F16;
    constexpr int       ldd_a  = std::is_same_v<T, float> ? LDD_A_F32 : LDD_A_F16;
    return num_panels_max(MC_TARGET, mr) * ldd_a;
}

// int8 packed-A stack buffer: the panel stride is in bytes (int8 = 1 byte/elem),
// so LDD_A_I8 = align_up(MR_MAX_I8 * KC_I8, 64) and the buffer holds
// num_panels_max(MC_TARGET, MR_I8) panels at that full-Kc stride (21 on
// aarch64 / 26 on x86_64 — not the 18 / 24 that MC_TARGET itself decomposes
// into, which several smaller mc would overflow; see num_panels_max).
constexpr int LDD_A_I8 = align_up<PANEL_ALIGN_BYTES>(MR_MAX_I8 * KC_I8);
constexpr int PACK_A_STACK_I8 = num_panels_max(MC_TARGET, MR_I8) * LDD_A_I8;

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

// `zero_mode` selects the kernel variant: true overwrites C and never reads
// it (first k-block with beta == 0 — the accumulator starts at zero), false
// accumulates into C (C += A·B). The kernels' ZeroMode template is
// compile-time; this runtime flag picks between the two pre-instantiated
// dispatch tables.
void tile_mma_pack(int mc, int nc, int kc, float* c, int ldc, const float* packed_a, const float* maybe_packed_b, int ldb, float clamp_min, float clamp_max, bool zero_mode);
void tile_mma_direct(int mc, int nc, int kc, float* c, int ldc, const float* a, int lda, const float* b, int ldb, float clamp_min, float clamp_max, bool zero_mode);
void tile_mma_pack(int mc, int nc, int kc, half* c, int ldc, const half* packed_a, const half* maybe_packed_b, int ldb, float clamp_min, float clamp_max, bool zero_mode);
void tile_mma_direct(int mc, int nc, int kc, half* c, int ldc, const half* a, int lda, const half* b, int ldb, float clamp_min, float clamp_max, bool zero_mode);

// SIMD-accelerated in-place scale: C[i] *= scale
void tile_scale(float* c, int ldc, float scale, int M, int N);
void tile_scale(half* c, int ldc, float scale, int M, int N);

// ---- int8 (s8×s8) pack + MMA --------------------------------------------
//
// Packed layouts (shared with the VNNI/SDOT micro-kernels): both A and B are
// "group-major" — 4 consecutive int8 packed per int32, groups of 4 int8 along
// the K dimension. For a panel of mr rows (A) or nr columns (B):
//   A: [group g: row0(4B) row1(4B) ... row(mr-1)(4B)] ...
//   B: [group g: col0(4B) col1(4B) ... col(nr-1)(4B)] ...
// `kc` is the number of int8 elements along K (the pack kernels consume 4 at a
// time; the x86 VNNI kernels take K in elements, aarch64 SDOT takes K/4 groups).

// Pack A (LHS) / B (RHS) into the 4-int8-per-int32 layout. On x86 with VNNI the
// A pack also XORs every byte with 0x80 (s8 → u8 two's-complement bias); the
// correction is folded back in the epilogue (see INT8_USE_U8_OFFSET).
void tile_pack_lhs_i8(bool trans, int mc, int kc, int8_t* dst, int ldd, const int8_t* src, int lds);
void tile_pack_rhs_i8(bool trans, int nc, int kc, int8_t* dst, int ldd, const int8_t* src, int lds);

// Accumulate C[mc][nc] += A_packed[mc][K] × B_packed[K][nc] (int32).
// `kc` is in int8 elements; the caller packs A/B with tile_pack_{lhs,rhs}_i8.
void tile_mma_pack_i8(int mc, int nc, int kc,
                      int32_t* c, int ldc,
                      const int8_t* packed_a, const int8_t* packed_b,
                      int32_t clamp_min, int32_t clamp_max);

// XOR every byte of a packed-A panel with 0x80 (s8 → u8 for VNNI).
void xor0x80_i8(int8_t* p, int nbytes);

struct MatMulPlan {
    int64_t mc, nc, kc;       // tile sizes
    int64_t ldd_b;            // packed-B panel stride (elements) at full Kc
    bool    pack_a;           // A is packed (transpose_a, wide row stride, or int8)
    bool    split_n;          // split on N (else on M)
    int64_t num_slots;        // packed-B slices allocated (<= num_blocks)
    int64_t workspace_size;   // scratch bytes for packed B
};

/// Resolve the tile sizes, the split direction and the packed-B scratch layout.
///
/// Covers f32, f16 and s8×s8. Only `kc`, the panel lists and `num_slots` differ
/// per dtype; the thread-aware mc/nc shrink is shared, which is what makes the
/// block count (and so the parallelism) grow with the pool. Returns an all-zero
/// plan for any other dtype — every field must be checked before use.
///
/// @p use_thread_slots mirrors the conv2d/attention plans: when the backend
/// reports worker ids, the scratch is one slot per worker thread (indexed by
/// current_thread_id() and reused across every block a thread claims) rather
/// than one slot per parallel block. Blocks are only ever live one at a time
/// per thread, so this is the same buffer the dispatch would have handed each
/// block on its own — it just stops paying for one copy per block.
MatMulPlan get_matmul_plan(const MatMulAttributes& attrs,
                          const TensorDesc& a_desc,
                          const TensorDesc& b_desc,
                          int num_threads = 1,
                          bool use_thread_slots = false);

}  // namespace nnops::backend::cpu

