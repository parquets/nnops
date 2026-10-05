#pragma once
/// @file matmul_helper.h
/// @brief Inner matrix multiplication — pack + MMA micro-kernel dispatch.
///
/// All arch-specific pack/mma headers share an identical API in the namespaces
///   nnops::backend::cpu::x86_64   and   nnops::backend::cpu::aarch64.
/// This header picks the right set via NNOPS_ARCH_* and exposes the active
/// namespace as `arch`, so consumer code stays clean of #ifdef.

#include "nnops/detail/simd/cpu_features.hpp"  // NNOPS_ARCH_X86_64 / NNOPS_ARCH_AARCH64
#include "nnops/ops/matmul.hpp"                // MatMulAttributes
#include "nnops/core/tensor_view.hpp"          // TensorDesc

#include <algorithm>
#include <cmath>
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

// ---- namespace alias ---------------------------------------------------

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
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>, "Unsupported type");
    }
}

template <int AlignBytes>
constexpr int align_up(int n) {
    return (n + AlignBytes - 1) & ~(AlignBytes - 1);
}

// =========================================================================
//  Tiling constants (shared by matmul.cpp kernels and workspace sizing)
// =========================================================================

constexpr int64_t KC_F32   = 128;   // attention's k-block length (see attention.cpp)
constexpr int64_t KC_F16   = 128;   // attention's f16 k-block length
constexpr int64_t KC_I8    = 512;   // int8: larger Kc since elements are 1 byte
constexpr int64_t KC_F16I4 = 256;   // fp16×int4: placeholder (future hardware)

// ---- The tile problem ---------------------------------------------------
//
// mc, nc and kc all come out of ONE objective (see choose_matmul_tile): maximise
// the tile's compute-to-traffic ratio
//
//     2·mc·nc·kc / (mc·kc + nc·kc + mc·nc)
//
// subject to
//
//     nt · kc · (mc + nc) · elem_bytes  <=  L2_shared / 2   (aggregate residency)
//     (mr_max + 2·nr_max) · kc · elem_bytes  <  L1          (next-B prefetch)
//     mc <= MC_MAX,  nc <= NC_MAX
//
// These are limits on one problem, not three independent heuristics. Derivation
// and the measured A/B: skills/how_to_optimize_gemm.md §2.1.

/// Ceiling on both tile dimensions.
constexpr int64_t MC_MAX = 768;
constexpr int     NC_MAX = 768;

/// mc decomposes into mr_max panels; lcm(6, 8) = 24 covers x86_64 and aarch64, so
/// a 24-aligned mc leaves no partial A panel on either. (48, the nc grid, would
/// break M-split exactness: M=1024 on 8 workers wants mc=128, and 128 snaps to 96.)
constexpr int64_t MC_ALIGN = 24;

/// nc decomposes into nr_max panels: lcm(12, 16) = 48 is a multiple of every
/// nr_max the pack can be given — fp or int8.
constexpr int NC_ALIGN = 48;

constexpr int64_t KC_MIN   = 128;
constexpr int64_t KC_ALIGN = 32;

/// Per-dtype kc ceilings (int8 keeps its historical 512; the aggregate model
/// would otherwise ask for 1344+ there, since elem_bytes == 1 understates what
/// the pack and the s8 accumulator use).
///
/// 128 on the float paths is a length limit, not a budget one: the aarch64 f16
/// kernel accumulates in f16, so kc is how many products round into the
/// accumulator before the C read-modify-write. At K = 4096, kc 128 -> 256 moved
/// the worst deviation 0.183 -> 0.230 (~0.23% -> 0.29% of an output of ~80).
constexpr int64_t KC_CAP_F32 = 128;
constexpr int64_t KC_CAP_F16 = 128;
constexpr int64_t KC_CAP_I8  = 512;

// =========================================================================
//  Panel-grid snapping
// =========================================================================

inline int round_down_nc(int nc, int nr_max) noexcept {
    return (nc / nr_max) * nr_max;
}

/// Snap an n-tile target down to the tile grid: to an NC_ALIGN multiple while
/// that still leaves an aligned tile, else to the nr_max multiple the pack
/// requires. Called before clamp_nc, so an N-driven short tile can still end up
/// below NC_ALIGN (down to N itself when N < nr_max).
inline int round_nc_target(int nc, int nr_max) noexcept {
    return (nc >= NC_ALIGN) ? round_down_nc(nc, NC_ALIGN) : round_down_nc(nc, nr_max);
}

/// Clamp nc into [nr_max, N] so the NKM loop always makes progress and the
/// workspace sizing matches the kernel exactly (N >= 1 guaranteed by shape).
inline int clamp_nc(int nc, int nr_max, int N) noexcept {
    if (nc < nr_max) { nc = nr_max; }
    if (nc > N)      { nc = N; }
    return nc;
}

inline int64_t snap_down(int64_t v, int64_t align) noexcept {
    return (v / align) * align;
}

/// Integer square root, exact (std::sqrt on doubles is not reliable near the
/// boundaries this feeds).
inline int64_t isqrt_i64(int64_t v) noexcept {
    if (v <= 0) { return 0; }
    int64_t r = static_cast<int64_t>(std::sqrt(static_cast<double>(v)));
    while (r > 0 && r * r > v) { --r; }
    while ((r + 1) * (r + 1) <= v) { ++r; }
    return r;
}

/// kc bound from the prefetch residency requirement: the kernel holds the
/// current A micro-panel (mr_max × kc) plus TWO B micro-panels (nr_max × kc) —
/// the one being consumed and the one being prefetched for the next n-step. All
/// three must sit in L1 together or the prefetch evicts what it is feeding.
inline int64_t kc_prefetch_bound(int mr_max, int nr_max, int elem_bytes,
                                 size_t l1_bytes) noexcept {
    const int64_t per_kc = static_cast<int64_t>(mr_max + 2 * nr_max) * elem_bytes;
    return (per_kc > 0) ? snap_down(static_cast<int64_t>(l1_bytes) / per_kc, KC_ALIGN)
                        : 0;
}

/// Tile height/width for one dimension of a split, chosen so the pool can fill
/// every worker. @p cap is the largest tile the reuse objective allows; the result
/// never exceeds it, so this trades reuse away only as far as balance needs.
///
/// Maximising reuse alone picks a tile so wide that the dimension splits into
/// fewer blocks than there are workers: at 4 workers a 1024-row M resolved to
/// mc = 768 (2 blocks), so two workers idled while two ran double — 0.72x
/// against the previous rule, and the same at 2 workers with 3 blocks.
/// Targeting a block count that is a multiple of @p nt keeps every worker at
/// the same number of blocks instead.
///
/// One worker is exempt: there is nothing to balance, so the tile stays at the
/// cap and the dimension is only clamped to fit.
inline int64_t balance_tile(int64_t dim, int64_t cap, int64_t align,
                            int64_t nt) noexcept {
    if (dim <= 0) { return 1; }
    const int64_t cap_a = std::max<int64_t>(snap_down(std::min(cap, dim), align), align);
    if (nt <= 1) { return std::min<int64_t>(cap_a, dim); }

    // Fewest blocks the cap allows, rounded up to a whole number of blocks per
    // worker; the tile is then the aligned even share of that many blocks.
    const int64_t nb_min = std::max<int64_t>((dim + cap_a - 1) / cap_a, 1);
    const int64_t nb = ((nb_min + nt - 1) / nt) * nt;
    int64_t tile = (((dim + nb - 1) / nb) + align - 1) / align * align;
    tile = std::min<int64_t>(tile, cap_a);
    return std::min<int64_t>(std::max<int64_t>(tile, 1), dim);
}

/// Shared core of the two entry points below. @p kc_fixed > 0 pins kc and sizes
/// mc/nc around it; 0 lets the solver choose kc too.
inline void choose_tile_impl(int64_t M, int64_t N, int64_t K,
                             int mr_max, int nr_max, int elem_bytes,
                             int64_t num_threads, bool split_n,
                             int64_t kc_cap, int64_t kc_fixed,
                             size_t l1_bytes, size_t l2_bytes,
                             int64_t& mc, int64_t& nc, int64_t& kc) noexcept {
    const int64_t nt  = std::max<int64_t>(num_threads, 1);
    const int64_t esz = std::max<int64_t>(elem_bytes, 1);

    // kc·(mc+nc) <= C is the aggregate residency budget: all nt concurrent blocks
    // hold their own packed A and packed B for the k-block at the same time.
    const int64_t C = static_cast<int64_t>(l2_bytes) / (2 * nt * esz);

    if (kc_fixed > 0) {
        // Fixed kc (attention pins its own): spend what is left on the tile.
        kc = std::min<int64_t>(std::max<int64_t>(kc_fixed, 1), std::max<int64_t>(K, 1));
    }

    // With a free kc, maximising 2·mc·nc·kc/(mc·kc+nc·kc+mc·nc) over mc == nc at
    // kc·(mc+nc) == C gives mc = nc = sqrt(C), kc = sqrt(C)/2 — the point where
    // one more row/column and one more k are worth the same. With kc pinned the
    // objective is monotonic in mc == nc, so the tile takes the whole budget.
    const int64_t s_cap = 2 * MC_MAX;
    int64_t s = (kc_fixed > 0) ? (kc > 0 ? C / kc : s_cap) : 2 * isqrt_i64(C);
    s = std::min(s, s_cap);
    s = std::max(s, MC_ALIGN + NC_ALIGN);

    int64_t mc_t = std::max<int64_t>(snap_down(std::min(s / 2, MC_MAX), MC_ALIGN), MC_ALIGN);
    int64_t nc_t = std::max<int64_t>(
        snap_down(std::min(s - mc_t, static_cast<int64_t>(NC_MAX)), NC_ALIGN), NC_ALIGN);

    // Routing. The split dimension is the one the pool divides, so it is the one
    // that has to be balanced across workers; the other spans whole and is only
    // bounded by the ceiling and the shape.
    int64_t mc_raw, nc_raw;
    if (split_n) {
        mc_raw = snap_down(std::min(mc_t, std::max<int64_t>(M, 1)), MC_ALIGN);
        nc_raw = balance_tile(std::max<int64_t>(N, 1), nc_t, NC_ALIGN, nt);
    } else {
        mc_raw = balance_tile(std::max<int64_t>(M, 1), mc_t, MC_ALIGN, nt);
        nc_raw = std::min(nc_t, std::max<int64_t>(N, 1));
    }
    mc = std::max<int64_t>(mc_raw, 1);

    nc = clamp_nc(round_nc_target(static_cast<int>(std::min(nc_raw, std::max<int64_t>(N, 1))),
                                  nr_max),
                  nr_max, static_cast<int>(std::max<int64_t>(N, 1)));

    if (kc_fixed > 0) { return; }

    // Routing may have left budget on the table (a short M, a small nt); hand it
    // back to kc rather than wasting it.
    const int64_t span = std::max<int64_t>(mc + nc, 1);
    int64_t lim = (C > 0) ? C / span : 0;
    lim = std::min(lim, kc_prefetch_bound(mr_max, nr_max, elem_bytes, l1_bytes));
    lim = std::min(lim, kc_cap);
    lim = std::min(lim, std::max<int64_t>(K, 1));

    kc = snap_down(lim, KC_ALIGN);
    const int64_t floor_kc = std::min<int64_t>(KC_MIN, std::max<int64_t>(K, 1));
    if (kc < floor_kc) { kc = floor_kc; }
    kc = std::min(kc, std::max<int64_t>(K, 1));
    if (kc < 1) { kc = 1; }
}

/// Resolve (mc, nc, kc) for a GEMM of M×N, K deep. Single source of truth for
/// both kernel routing and workspace sizing.
inline void choose_matmul_tile(int64_t M, int64_t N, int64_t K,
                               int mr_max, int nr_max, int elem_bytes,
                               int64_t num_threads, bool split_n, int64_t kc_cap,
                               size_t l1_bytes, size_t l2_bytes,
                               int64_t& mc, int64_t& nc, int64_t& kc) noexcept {
    choose_tile_impl(M, N, K, mr_max, nr_max, elem_bytes, num_threads, split_n,
                     kc_cap, /*kc_fixed=*/0, l1_bytes, l2_bytes, mc, nc, kc);
}

/// Resolve the tile sizes (mc, nc) for a GEMM of M×N with the caller's own fixed
/// Kc — attention pins its k-block and only wants the two spatial dimensions.
/// Same rule as choose_matmul_tile, so the two stay in step.
inline void resolve_tile_sizes(int64_t M, int64_t N, int64_t mr_max, int64_t nr_max,
                               int64_t kc, int elem_bytes, int64_t num_threads,
                               int& mc, int& nc) noexcept {
    const auto& f = simd::CpuFeatures::get();
    int64_t mc64 = 0, nc64 = 0, kc_out = 0;
    choose_tile_impl(M, N, /*K=*/kc, static_cast<int>(mr_max), static_cast<int>(nr_max),
                     elem_bytes, num_threads, /*split_n=*/N > M, /*kc_cap=*/kc,
                     /*kc_fixed=*/kc, f.l1_cache_size(), f.l2_shared_cache_size(),
                     mc64, nc64, kc_out);
    mc = static_cast<int>(mc64);
    nc = static_cast<int>(nc64);
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
// B (rhs) is always packed into the workspace. A (lhs) is packed when
// transpose_a forces it (correctness), when its row stride exceeds one
// cache-friendly page so the panel reads would be page-scattered, or for int8.
// The loop order follows the split: NKM when split on N, MKN when split on M.
// Tiny GEMMs (M*N*K below the fast-path threshold) skip the tiled path
// entirely and go straight to the reference.

/// A is packed when its row stride exceeds this many elements (wide stride →
/// page-scattered reads), independent of element type.
constexpr int64_t PACK_A_STRIDE_THRESHOLD = 1024;

/// GEMMs with fewer than this many MACs take the fast path (naive reference),
/// skipping pack/tile/threading overhead.
constexpr int64_t GEMM_FAST_PATH_THRESHOLD = 1024;

inline bool gemm_is_small(int64_t M, int64_t N, int64_t K) noexcept {
    return M * N * K < GEMM_FAST_PATH_THRESHOLD;
}

/// Number of panels for the largest-first decomposition of `n` into the
/// `levels` sizes nr[0] >= ... >= nr[levels-1] (the last == 1; 3 by default,
/// 4 for the int8 lists). This is what the pack/MMA loops
/// actually emit, so workspace sizing must match it exactly (a simple
/// ceil(n / nr[0]) under-counts when the tail decomposes into nr=1 panels).
/// Written as the quotient chain rather than the stepwise loop the pack runs:
/// taking nr[s] panels while they fit leaves `rem % nr[s]` behind, so the two are
/// the same count, in O(levels) instead of O(n / nr[0]) steps.
constexpr int num_panels(int n, const int* nr, int levels = 3) noexcept {
    int count = 0;
    int rem   = n;
    for (int s = 0; s < levels; ++s) {
        count += rem / nr[s];
        rem   %= nr[s];
    }
    return count;
}

/// 4-level variant for the int8 NR lists ({16,8,4,1} / {12,8,4,1}).
constexpr int num_panels4(int n, const int* nr) noexcept {
    return num_panels(n, nr, 4);
}

/// Largest `num_panels()` over every mc in [1, n].
///
/// The greedy decomposition above is NOT monotonic in n: with {8,4,1} a 143-row
/// tile needs 21 panels (17×8 + 4 + 1 + 1 + 1) while a 144-row tile needs only
/// 18 (18×8). So sizing a buffer at num_panels(mc, mr) is wrong — the
/// *smaller* tile is the one that overflows it:
///
///   aarch64 MR = {8,4,1}: 18 at mc=144, worst 21 at mc=143
///   x86_64  MR = {6,4,1}: 24 at mc=144, worst 26 at mc=141
///
/// Anything holding the panels of an arbitrary height (the packed-A tile) must
/// size with this maximum, not with num_panels(n, mr).
///
/// Closed form rather than a scan of [1, n]: the scan is O(n*levels) on the
/// per-plan path (~2.6 us for {12,4,1} at n=1024, more than a small GEMM's whole
/// kernel), while this is O(levels). The maximum sits at one of the two branches
/// below — the quotient, or one step under it — and the remainder is the same
/// question one level down.
constexpr int num_panels_max(int n, const int* nr, int levels = 3) noexcept {
    if (n <= 0 || levels <= 0) { return 0; }
    const int step = nr[0];
    const int q    = n / step;
    if (q == 0) { return num_panels_max(n, nr + 1, levels - 1); }
    const int at_q    = q + num_panels_max(n - q * step, nr + 1, levels - 1);
    const int below_q = (q - 1) + num_panels_max(step - 1, nr + 1, levels - 1);
    return at_q > below_q ? at_q : below_q;
}

/// 4-level `num_panels_max` for the int8 NR lists ({16,8,4,1} / {12,8,4,1}) —
/// same non-monotonicity, one more level to decompose (see num_panels4).
constexpr int num_panels_max4(int n, const int* nr) noexcept {
    return num_panels_max(n, nr, 4);
}

// ---- public API --------------------------------------------------------
//
// Pack entry points: panels are laid out at a uniform 64-byte-aligned stride
// `ldd` (elements), so pack and MMA agree on panel placement for all panel
// sizes. The caller computes ldd from the actual k-block length (actual_kc).
//
// MMA entry points: `tile_mma_pack` accumulates C[mr][nr] += A_packed[mr][K]
// × B[K][nr], where `maybe_packed_b`/`ldb` describe B:
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
    int64_t np_slice;         // panels one packed-B slice holds (see get_matmul_plan)
    bool    pack_a;           // A is packed (transpose_a, wide row stride, or int8)
    bool    split_n;          // split on N (else on M)
    int64_t num_slots;        // packed-B slices allocated (<= num_blocks)
    int64_t workspace_size;   // total scratch bytes: packed B, then packed A

    // Packed-A region, which begins at pack_a_offset — the end of the packed-B
    // region — and runs to workspace_size. It holds num_slots_a slots of np_a
    // panels each at the ldd_a stride; the stride is in elements for fp and in
    // bytes for int8, and both go through align_up<PANEL_ALIGN_BYTES>, so every
    // slot and the region are 64-byte aligned, which is what the pack/MMA
    // kernels assume. Sized from the plan's own mc and kc, so it is a bound and
    // not a worst case: any block has actual_mc <= mc and actual_kc <= kc.
    // ldd_a, np_a and num_slots_a are all zero when !pack_a: the direct-A route
    // never touches the buffer, so pack_a_offset is then just the end of B.
    int64_t ldd_a;            // panel stride at full kc (elements; bytes for int8)
    int64_t np_a;             // panels one packed-A tile holds (num_panels_max(mc, MR))
    int64_t num_slots_a;      // packed-A slots (see get_matmul_plan)
    int64_t pack_a_offset;    // byte offset of the region (== the packed-B size)
};

/// Resolve the tile sizes, the split direction and the packed-A/B scratch layout.
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
///
/// num_slots_a follows the same idea but with its own predicate, and that
/// difference is load-bearing: the int8 M-split hoists its *B* pack out of the
/// parallel region (so num_slots == 1 there) while every m-block still packs its
/// own A. The dispatch must therefore decide the A slot from `num_slots_a`, not
/// from `num_slots`. A single thread always gets one A slot regardless of the
/// block count — a sequential run cannot have two A tiles live at once, and
/// sizing it per block is what would push a large M-split past the memory
/// pool's 16 MiB ceiling and turn a pooled allocation into a per-call OS one.
MatMulPlan get_matmul_plan(const MatMulAttributes& attrs,
                          const TensorDesc& a_desc,
                          const TensorDesc& b_desc,
                          int num_threads = 1,
                          bool use_thread_slots = false);

}  // namespace nnops::backend::cpu

