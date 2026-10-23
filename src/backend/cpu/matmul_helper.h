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
#include "backend/cpu/common/common_utils.hpp" // align_up

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>

// ---- arch-specific includes --------------------------------------------

#ifdef NNOPS_ARCH_X86_64
#include "x86_64/pack_f16.hpp"
#include "x86_64/pack_f32.hpp"
#include "x86_64/pack_q5_1.hpp"
#include "x86_64/mma_f16.hpp"
#include "x86_64/mma_f32.hpp"
#include "x86_64/mma_q5_q.hpp"
#include "x86_64/pack_dp4a_i8.hpp"
#include "x86_64/mma_i8.hpp"
#include "x86_64/matmul_naive.hpp"
#elif defined(NNOPS_ARCH_AARCH64)
#include "aarch64/pack_f16.hpp"
#include "aarch64/pack_f32.hpp"
#include "aarch64/mma_f16.hpp"
#include "aarch64/mma_f32.hpp"
#include "aarch64/pack_dp4a_i8.hpp"
#include "aarch64/mma_i8.hpp"
#include "aarch64/matmul_naive.hpp"
#endif

#include "nnops/detail/half.hpp"

namespace nnops::backend::cpu {

// ---- namespace alias ---------------------------------------------------

#ifdef NNOPS_ARCH_X86_64
namespace arch = x86_64;
#elif defined(NNOPS_ARCH_AARCH64)
namespace arch = aarch64;
#endif

constexpr int MR_F32[3] = {arch::mr_f32[0], arch::mr_f32[1], arch::mr_f32[2]};
constexpr int NR_F32[3] = {arch::nr_f32[0], arch::nr_f32[1], arch::nr_f32[2]};
constexpr int MR_F16[3] = {arch::mr_f16[0], arch::mr_f16[1], arch::mr_f16[2]};
constexpr int NR_F16[3] = {arch::nr_f16[0], arch::nr_f16[1], arch::nr_f16[2]};

constexpr int MR_MAX_F32 = arch::mr_f32[0];
constexpr int NR_MAX_F32 = arch::nr_f32[0];
constexpr int MR_MAX_F16 = arch::mr_f16[0];
constexpr int NR_MAX_F16 = arch::nr_f16[0];

#ifdef NNOPS_ARCH_X86_64
// The Q5_1 RHS kernel writes the same uniform panels as the float packers, so
// its hard-coded width must track both NR tables (16 for f32 and f16 here).
static_assert(arch::q5_1_nr_max == NR_MAX_F32 && arch::q5_1_nr_max == NR_MAX_F16,
              "Q5_1 RHS panel width must match the float RHS panel width");
#endif

// int8 panel sizes: A keeps the arch's MR cascade, B and the MMA are uniform.
#ifdef NNOPS_ARCH_X86_64
constexpr int MR_I8[3] = {6, 4, 1};
#elif defined(NNOPS_ARCH_AARCH64)
constexpr int MR_I8[3] = {8, 4, 1};
#endif

constexpr int MR_MAX_I8 = MR_I8[0];
constexpr int NR_MAX_I8 = arch::mma_i8_nr_max;

// int8 always uses the hardware dot product: VNNI (u8×s8) on x86, SDOT
// (s8×s8) on aarch64. Only the pack layout differs by arch.
#ifdef NNOPS_ARCH_X86_64
static_assert(arch::mma_i8_b_panel_tail == 16 * 4, "x86 int8 B panel must be 16 wide");
static_assert(arch::mma_i8_folds_a_bias, "x86 int8 kernels fold the A bias per panel");
#else
static_assert(arch::mma_i8_b_panel_tail == 0, "aarch64 int8 B panel must be 12 wide");
static_assert(!arch::mma_i8_folds_a_bias, "aarch64 int8 kernels have no panel tail");
#endif

// ---- the A-side bias: signed/unsigned mismatch, per dtype and arch ----------
//
// s8 A on x86 shifts +128 (to u8), u8 A on aarch64 shifts -128 (to s8); the
// other pairs are used as stored. One XOR 0x80 in the A pack, so `bias` is
// {-128, 0, +128}. The -bias·Σ_k B[k][n] it leaves behind is folded out by
// the B panel's correction block (x86) or by the epilogue (aarch64); see
// mma_i8_folds_a_bias.
constexpr int int8_a_pack_bias(DataType dt_a) noexcept {
#ifdef NNOPS_ARCH_X86_64
    return (dt_a == DataType::s8) ? 128 : 0;
#else
    return (dt_a == DataType::s8) ? 0 : -128;
#endif
}

/// A-side shift not already folded out by the kernels; goes to
/// matmul_int8_compensate's u8_offset.
constexpr int int8_epilogue_offset(DataType dt_a) noexcept {
    return arch::mma_i8_folds_a_bias ? 0 : int8_a_pack_bias(dt_a);
}

/// Bytes of one int8 packed-B panel slot of `kc` elements: nr_max columns x
/// ceil(kc/4) k groups, plus x86's correction block, 64-byte aligned.
constexpr int64_t int8_panel_stride(int64_t kc) noexcept {
    const int64_t kbytes = (kc + 3) & ~int64_t{3};
    return static_cast<int64_t>(mma_pack_align(
        static_cast<size_t>(NR_MAX_I8) * static_cast<size_t>(kbytes)
        + static_cast<size_t>(arch::mma_i8_b_panel_tail)));
}

/// Largest M-panel height a float tile runs; callers modelling a working set
/// must charge this.
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

// =========================================================================
//  Tiling constants (shared by matmul.cpp kernels and workspace sizing)
// =========================================================================

constexpr int64_t KC_F32   = 128;   // attention's k-block length (see attention.cpp)
constexpr int64_t KC_F16   = 128;   // attention's f16 k-block length
constexpr int64_t KC_F16I4 = 256;   // fp16×int4: placeholder (future hardware)

// ---- The tile problem ---------------------------------------------------
//
// mc, nc and kc come out of ONE objective (choose_matmul_tile): maximise
// 2·mc·nc·kc / (mc·kc + nc·kc + mc·nc), subject to
//     nt · kc · (mc + nc) · elem_bytes <= L2_shared / 2   (residency)
//     (mr_max + 2·nr_max) · kc · elem_bytes < L1          (next-B prefetch)
//     mc <= MC_MAX, nc <= NC_MAX
// Derivation and the measured A/B: skills/how_to_optimize_gemm.md §2.1.

/// Ceiling on both tile dimensions.
constexpr int64_t MC_MAX = 768;
constexpr int     NC_MAX = 768;

/// lcm(6, 8) = 24: a 24-aligned mc leaves no partial A panel on either arch.
/// (48 would break M-split exactness: M=1024 on 8 workers wants mc=128.)
constexpr int64_t MC_ALIGN = 24;

/// lcm(12, 16) = 48: a multiple of every nr_max the pack can be given.
constexpr int NC_ALIGN = 48;

constexpr int64_t KC_MIN   = 128;
constexpr int64_t KC_ALIGN = 32;

/// Per-dtype kc ceilings, all below what the residency budget affords.
///
/// 128 on fp is a precision limit: the aarch64 f16 kernel accumulates in f16,
/// and kc 256 at K=4096 moved the worst deviation 0.183 -> 0.230. f32 shares it.
///
/// 512 on int8 is a tuning choice (its accumulator is int32). Capping at 256
/// measured 3-13% slower (a second C read-modify-write per k-block) and never
/// faster, so the plan is left to follow the cache it runs in.
constexpr int64_t KC_CAP_F32 = 128;
constexpr int64_t KC_CAP_F16 = 128;
constexpr int64_t KC_CAP_I8  = 512;

// =========================================================================
//  Per-dtype plan geometry
// =========================================================================
//
// Everything the plan (and matmul.cpp's tile loops) forks on for a dtype.

struct MatmulGeom {
    int      mr_max;            // widest A micro-panel (head of the MR cascade)
    int      nr_max;            // uniform B panel width
    int      elem;              // bytes per element (1 for int8)
    int64_t  kc_cap;            // per-dtype kc ceiling (see above)
    bool     packs_a_always;    // int8 has no direct-A route
    bool     hoists_b_m_split;  // int8 packs B once per k-block on the M-split
    const int* mr_levels;       // the A-panel cascade

    /// Packed panel slot stride at a k-block of `kc` (elements for fp, bytes
    /// for int8).
    int64_t (*ldd_b)(int64_t kc);
    int64_t (*ldd_a)(int64_t kc);
};

template <class T>
constexpr int64_t geom_ldd_flt(int rows, int64_t kc) noexcept {
    return static_cast<int64_t>(
        mma_pack_align(static_cast<size_t>(rows) * static_cast<size_t>(kc) * sizeof(T))
        / sizeof(T));
}

constexpr int64_t geom_ldd_f32_b(int64_t kc) noexcept { return geom_ldd_flt<float>(NR_MAX_F32, kc); }
constexpr int64_t geom_ldd_f32_a(int64_t kc) noexcept { return geom_ldd_flt<float>(MR_MAX_F32, kc); }
constexpr int64_t geom_ldd_f16_b(int64_t kc) noexcept { return geom_ldd_flt<half>(NR_MAX_F16, kc); }
constexpr int64_t geom_ldd_f16_a(int64_t kc) noexcept { return geom_ldd_flt<half>(MR_MAX_F16, kc); }

constexpr int64_t geom_ldd_i8_a(int64_t kc) noexcept {
    const int64_t kbytes = (kc + 3) & ~int64_t{3};
    return static_cast<int64_t>(
        mma_pack_align(static_cast<size_t>(MR_MAX_I8) * static_cast<size_t>(kbytes)));
}
constexpr int64_t geom_ldd_i8_b(int64_t kc) noexcept { return int8_panel_stride(kc); }

inline constexpr MatmulGeom GEOM_F32 = {
    MR_MAX_F32, NR_MAX_F32, static_cast<int>(sizeof(float)), KC_CAP_F32,
    /*packs_a_always=*/false, /*hoists_b_m_split=*/false, MR_F32,
    geom_ldd_f32_b, geom_ldd_f32_a,
};

inline constexpr MatmulGeom GEOM_F16 = {
    MR_MAX_F16, NR_MAX_F16, static_cast<int>(sizeof(half)), KC_CAP_F16,
    /*packs_a_always=*/false, /*hoists_b_m_split=*/false, MR_F16,
    geom_ldd_f16_b, geom_ldd_f16_a,
};

inline constexpr MatmulGeom GEOM_I8 = {
    MR_MAX_I8, NR_MAX_I8, 1, KC_CAP_I8,
    /*packs_a_always=*/true, /*hoists_b_m_split=*/true, MR_I8,
    geom_ldd_i8_b, geom_ldd_i8_a,
};

// =========================================================================
//  Panel-grid snapping
// =========================================================================

inline int round_down_nc(int nc, int nr_max) noexcept {
    return (nc / nr_max) * nr_max;
}

/// Snap the n-tile target to the tile grid, or to nr_max if too small for it.
inline int round_nc_target(int nc, int nr_max) noexcept {
    return (nc >= NC_ALIGN) ? round_down_nc(nc, NC_ALIGN) : round_down_nc(nc, nr_max);
}

/// Clamp nc into [nr_max, N] so the NKM loop always makes progress.
inline int clamp_nc(int nc, int nr_max, int N) noexcept {
    if (nc < nr_max) { nc = nr_max; }
    if (nc > N)      { nc = N; }
    return nc;
}

inline int64_t snap_down(int64_t v, int64_t align) noexcept {
    return (v / align) * align;
}

/// Exact integer square root (std::sqrt on doubles can be off by one here).
inline int64_t isqrt_i64(int64_t v) noexcept {
    if (v <= 0) { return 0; }
    int64_t r = static_cast<int64_t>(std::sqrt(static_cast<double>(v)));
    while (r > 0 && r * r > v) { --r; }
    while ((r + 1) * (r + 1) <= v) { ++r; }
    return r;
}

/// L1 bound: one A panel (mr_max × kc) plus two B panels (nr_max × kc) — the
/// one in use and the one being prefetched — must fit together.
inline int64_t kc_prefetch_bound(int mr_max, int nr_max, int elem_bytes,
                                 size_t l1_bytes) noexcept {
    const int64_t per_kc = static_cast<int64_t>(mr_max + 2 * nr_max) * elem_bytes;
    return (per_kc > 0) ? snap_down(static_cast<int64_t>(l1_bytes) / per_kc, KC_ALIGN)
                        : 0;
}

/// Tile size for one dimension of a split, ≤ @p cap, picked so the block count
/// is a multiple of @p nt. Pure reuse maximisation split a 1024-row M into 2
/// blocks at 4 workers (0.72x). One worker keeps the cap.
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

    // Free kc: the optimum is mc = nc = sqrt(C), kc = sqrt(C)/2. kc pinned: the
    // objective is monotonic, so the tile takes the whole budget.
    const int64_t s_cap = 2 * MC_MAX;
    int64_t s = (kc_fixed > 0) ? (kc > 0 ? C / kc : s_cap) : 2 * isqrt_i64(C);
    s = std::min(s, s_cap);
    s = std::max(s, MC_ALIGN + NC_ALIGN);

    int64_t mc_t = std::max<int64_t>(snap_down(std::min(s / 2, MC_MAX), MC_ALIGN), MC_ALIGN);
    int64_t nc_t = std::max<int64_t>(
        snap_down(std::min(s - mc_t, static_cast<int64_t>(NC_MAX)), NC_ALIGN), NC_ALIGN);

    // The split dimension is balanced; the other spans whole.
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
// The GEMM is split on its larger dimension and the blocks run as contiguous
// slabs. The count is ceil(larger / mc) — deliberately coarser than MLAS's
// complexity/64K target: the GEMM is packed-B-bandwidth bound, so smaller
// blocks only add pack overhead and seams.

inline int64_t split_block_count(int64_t larger, int64_t mc) noexcept {
    return std::max<int64_t>(1, (larger + mc - 1) / mc);
}

// =========================================================================
//  Pack decision (mirrors onnxruntime MLAS sgemm)
// =========================================================================
//
// B is always packed. A is packed for correctness (transpose_a), for int8, or
// when K is deep enough that its rows no longer fit a page — then a panel's
// direct row reads scatter, and packing makes them sequential. The parallel
// split follows MLAS: split N when N > M, else split M.

/// A is packed when K exceeds this many elements.
constexpr int64_t PACK_A_K_THRESHOLD = 1024;

/// GEMMs with fewer than this many MACs take the fast path (naive reference),
/// skipping pack/tile/threading overhead.
constexpr int64_t GEMM_FAST_PATH_THRESHOLD = 1024;

inline bool gemm_is_small(int64_t M, int64_t N, int64_t K) noexcept {
    return M * N * K < GEMM_FAST_PATH_THRESHOLD;
}

/// Panels the largest-first decomposition emits for `n` (quotient chain, O(levels)).
constexpr int num_panels(int n, const int* nr, int levels = 3) noexcept {
    int count = 0;
    int rem   = n;
    for (int s = 0; s < levels; ++s) {
        count += rem / nr[s];
        rem   %= nr[s];
    }
    return count;
}

/// Largest `num_panels()` over every mc in [1, n]. The decomposition is NOT
/// monotonic: with {8,4,1}, 143 rows need 21 panels but 144 need 18. So the
/// packed-A buffer must size by this maximum, not num_panels(n, mr). Closed
/// form (one step down from the quotient, recursing) instead of an O(n) scan.
constexpr int num_panels_max(int n, const int* nr, int levels = 3) noexcept {
    if (n <= 0 || levels <= 0) { return 0; }
    const int step = nr[0];
    const int q    = n / step;
    if (q == 0) { return num_panels_max(n, nr + 1, levels - 1); }
    const int at_q    = q + num_panels_max(n - q * step, nr + 1, levels - 1);
    const int below_q = (q - 1) + num_panels_max(step - 1, nr + 1, levels - 1);
    return at_q > below_q ? at_q : below_q;
}

// ---- public API --------------------------------------------------------
//
// Panels sit at a uniform 64-byte-aligned stride `ldd`, computed from the
// actual k-block length. A decomposes into the arch's MR cascade; B and the
// MMA are uniform-width.
void tile_pack_lhs(bool trans, int mc, int kc, float* dst, int ldd, const float* src, int lds, float scale);
void tile_pack_lhs(bool trans, int mc, int kc, half* dst, int ldd, const half* src, int lds, float scale);

// ---- uniform-width RHS panels — the only B/MMA route ---------------------
//
// The MMA reads nr_max B lanes at every column, so B is ceil(nc/nr_max)
// panels, each nr_max wide (zero-filled past nc) at one aligned stride.
// `zero_mode` picks the kernel: true overwrites C (beta == 0 on the first
// k-block), false accumulates.
void tile_pack_rhs(bool trans, int nc, int kc, float* dst, int ldd, const float* src, int lds, float scale);
void tile_pack_rhs(bool trans, int nc, int kc, half* dst, int ldd, const half* src, int lds, float scale);

// ---- Q5_1 (llama.cpp block format) RHS ----------------------------------
//
// A Q5_1 B is a 24-byte-block stream; the pack fuses unpack + dequantize into
// the uniform panels above, so the MMA side is unchanged. The tile origin is
// taken in block units (a panel can start mid-block), and `src_row_blocks` is
// the row pitch in blocks. x86_64 only; get_matmul_plan rejects it elsewhere.
void tile_pack_rhs_q5_1(bool trans, int n_start, int nc, int k_start, int kc,
                        float* dst, int ldd, const void* src, int64_t src_row_blocks);
void tile_pack_rhs_q5_1(bool trans, int n_start, int nc, int k_start, int kc,
                        half* dst, int ldd, const void* src, int64_t src_row_blocks);

// B is always packed here; panels advance by the aligned nr_max*kc stride.
void tile_mma_pack(int mc, int nc, int kc, float* c, int ldc, const float* packed_a, const float* packed_b, float clamp_min, float clamp_max, bool zero_mode);
void tile_mma_direct(int mc, int nc, int kc, float* c, int ldc, const float* a, int lda, const float* packed_b, float clamp_min, float clamp_max, bool zero_mode);
void tile_mma_pack(int mc, int nc, int kc, half* c, int ldc, const half* packed_a, const half* packed_b, float clamp_min, float clamp_max, bool zero_mode);
void tile_mma_direct(int mc, int nc, int kc, half* c, int ldc, const half* a, int lda, const half* packed_b, float clamp_min, float clamp_max, bool zero_mode);

// SIMD-accelerated in-place scale: C[i] *= scale
void tile_scale(float* c, int ldc, float scale, int M, int N);
void tile_scale(half* c, int ldc, float scale, int M, int N);

// ---- int8 (s8×s8) pack + MMA --------------------------------------------
//
// Same shape as the float route. Panels are group-major, 4 int8 per int32
// along K:
//   A: [group g: row0(4B) row1(4B) ... row(MR-1)(4B)] ...
//   B: [group g: col0(4B) col1(4B) ... col(NR-1)(4B)] ...
// A partial final group is zero-filled. On x86 each B panel carries a trailing
// correction block (its column sums) at int8_panel_stride; aarch64 has none.
//
// `a_biased` (from int8_a_pack_bias) makes the A pack XOR 0x80 and the B pack
// write or zero that correction block.
void tile_pack_lhs_i8(bool trans, int mc, int kc, int8_t* dst, int ldd,
                      const int8_t* src, int lds, bool a_biased);

// fp activations straight into the packed int8 A panel: every panel row is
// quantised with its own offline (scale, zero_point) into a small stack buffer
// and then packed. Produces exactly the bytes of quantise-then-pack, without
// the full M×K intermediate. Row-major A only (the transposing pack).
//
// `scale` / `zero` are indexed from the panel's first row and may be null, in
// which case the scalar applies. `u8` selects the clamp range (0..255 vs
// -128..127) — storage only, the panel stays signed.
void tile_pack_quant_lhs_i8(int mc, int kc, int8_t* dst, int ldd,
                            const float* src, int lds,
                            const float* scale, const int32_t* zero,
                            float scale_scalar, int32_t zero_scalar,
                            bool u8, bool a_biased);
void tile_pack_quant_lhs_i8(int mc, int kc, int8_t* dst, int ldd,
                            const half* src, int lds,
                            const float* scale, const int32_t* zero,
                            float scale_scalar, int32_t zero_scalar,
                            bool u8, bool a_biased);

void tile_pack_rhs_i8(bool trans, int nc, int kc, int8_t* dst, int ldd,
                      const int8_t* src, int lds, bool a_biased);

// Accumulate C[mc][nc] += A_packed[mc][K] × B_packed[K][nc] (int32), one kernel
// call per M panel covering the whole N side.
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

    // Packed-A region: num_slots_a slots of np_a panels at the ldd_a stride
    // (elements for fp, bytes for int8; both 64-byte aligned). Sized from the
    // plan's mc/kc, so it is a bound. All three fields stay 0 when !pack_a.
    int64_t ldd_a;            // panel stride at full kc (elements; bytes for int8)
    int64_t np_a;             // panels one packed-A tile holds (num_panels_max(mc, MR))
    int64_t num_slots_a;      // packed-A slots (see get_matmul_plan)
    int64_t pack_a_offset;    // byte offset of the region (== the packed-B size)
};

/// Resolve the tile sizes, the split direction and the packed-A/B scratch layout.
/// Covers f32, f16, s8|u8 A × s8 B, and a f32/f16 A with a Q5_1 B. Returns an
/// all-zero plan for any other dtype.
///
/// @p use_thread_slots: when the backend reports worker ids, one scratch slot
/// per worker thread instead of one per block (a thread has one block live).
///
/// num_slots_a has its own predicate, and the difference is load-bearing: int8
/// hoists its B pack on the M-split (num_slots == 1) but still packs A per
/// block, so the dispatch must index A by `num_slots_a`. One thread always
/// gets one A slot — sizing per block would push a large M-split past the
/// memory pool's 16 MiB ceiling into a per-call OS allocation.
MatMulPlan get_matmul_plan(const MatMulAttributes& attrs,
                          const TensorDesc& a_desc,
                          const TensorDesc& b_desc,
                          int num_threads = 1,
                          bool use_thread_slots = false);

#ifdef NNOPS_ARCH_X86_64
/// Diagnostic cap on the fused Q5_1 small-M route (matmul_q5_fused.cpp):
/// negative keeps the built-in per-layout M limits, 0 disables the route.
/// Benchmarks use it to A/B the fused and packed routes inside one process.
extern int q5_fused_max_m_override;
#endif

}  // namespace nnops::backend::cpu

