#pragma once
/// @file mma_bf16.hpp
/// @brief AArch64 bfloat16 MMA kernels over uniform 16-wide B panels.
///
/// The tile is the same runtime-M shape as mma_f16.hpp: one templated rung per
/// 2-row pair count (`PAIRS` ∈ {4, 2, 1} is the {8, 4, 1}-row rungs), `nc`
/// columns in 16-wide panels, and rows past `mr` alias the last live row in
/// both A and C, so a partial-M tile needs no guards.
///
/// A is dense rows `lda` apart (the packed route's panel stride is its `kc`,
/// the direct route's is the caller's), read as row pairs: the BFMMLA A operand
/// is one row pair's four k values in each half of an 8H register. B is the
/// panel layout pack_bf16.hpp writes — ⌈kc/4⌉ k-groups of [colpair][8], a
/// colpair holding {col 2c k0..3, col 2c+1 k0..3} — and the direct route also
/// takes B in place through a row pitch (`ldb > 0`), which is the 1×1
/// convolution's filter.
///
/// FEAT_BF16 (Armv8.6) supplies BFMMLA, accumulating in f32; cpu_has_bf16()
/// picks it at the entry point and the same blocking runs as f32 FMA otherwise,
/// so a pre-v8.6 core reads the same buffers at f32 speed rather than refusing
/// the dtype. Both paths clamp and narrow at the store only.
///
/// The accumulator convention is BFMMLA's per column pair — [r0c0, r0c1, r1c0,
/// r1c1] — on both paths, which is what lets one store serve them. The store
/// side reads every live row's C before writing any of them: rows past `mr`
/// share their address with the last live row, and their accumulators are that
/// row's, so the repeated write is idempotent instead of a second C add.
///
/// Shape of each kernel: one rung serves both block widths — eight columns (4
/// column pairs) for the 8-row rung, a whole 16-column panel (8 pairs) for the
/// narrower ones — and the k-group body is written out once with literal
/// indices into `acc[PAIRS][pairs]`. The last k-group, short of four, goes
/// through the same body with the zero-padded operand builders. The k-group
/// loop itself stays a loop: BFMMLA threads every multiply-add into one
/// accumulator, so there is no independent work to interleave and unrolling
/// only lengthens the dependent chain.
///
/// Reference: onnxruntime MLAS SbgemmKernelNeon.S (8 rows × 8 columns of
/// 2×2 BFMMLA accumulators, bfcvtn on the A side).

#include <algorithm>
#include <array>
#include <arm_neon.h>
#include <utility>

#include "nnops/detail/bf16.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "backend/cpu/common/common_utils.hpp"
#include "backend/cpu/aarch64/pack_bf16.hpp"

namespace nnops::backend::cpu::aarch64 {

using bfloat16 = ::nnops::backend::cpu::bfloat16;
namespace sims = ::nnops::simd;

// =========================================================================
//  Operand builders
// =========================================================================
//
// These are their own functions: a lambda body does not inherit the enclosing
// function's target attribute, so the bf16 intrinsics have to live somewhere
// that carries it.

/// A operand, full k-group: one row pair's four k values in each half.
__attribute__((target("bf16"))) inline bfloat16x8_t bf16_a_pair(const bfloat16* r0,
                                                                const bfloat16* r1) noexcept
{
    return vreinterpretq_bf16_u16(vcombine_u16(vld1_u16(reinterpret_cast<const uint16_t*>(r0)),
                                               vld1_u16(reinterpret_cast<const uint16_t*>(r1))));
}

/// A operand for the short final k-group: @p rem (< 4) values, zero past them.
__attribute__((target("bf16"))) inline bfloat16x8_t
bf16_a_pair_tail(const bfloat16* r0, const bfloat16* r1, int rem) noexcept
{
    uint16_t tmp[8] = {};
    for (int j = 0; j < rem; ++j) {
        tmp[j] = r0[j].bits;
        tmp[4 + j] = r1[j].bits;
    }
    return vreinterpretq_bf16_u16(vld1q_u16(tmp));
}

/// B operand out of a packed panel: k-group @p g, colpair @p cp (0..7, i.e. the
/// whole 16-column panel; the low 4 are the block's first 8 columns).
__attribute__((target("bf16"))) inline bfloat16x8_t bf16_b_panel(const bfloat16* b, int g,
                                                                 int cp) noexcept
{
    return vreinterpretq_bf16_u16(
        vld1q_u16(reinterpret_cast<const uint16_t*>(b) + g * bf16_b_k4_elems + cp * 8));
}

/// B operand out of a caller's matrix, rows @p ldb apart, full k-group and
/// in-range columns: only the gather is left.
__attribute__((target("bf16"))) inline bfloat16x8_t bf16_b_stride(const bfloat16* b, int64_t ldb,
                                                                  int g, int cp) noexcept
{
    uint16_t tmp[8];
    for (int lane = 0; lane < 8; ++lane) {
        tmp[lane] = b[static_cast<int64_t>(g * 4 + (lane & 3)) * ldb + 2 * cp + (lane >> 2)].bits;
    }
    return vreinterpretq_bf16_u16(vld1q_u16(tmp));
}

/// The same gather with the bounds checks: rows past @p kc and columns past
/// @p nloc read as zero.
__attribute__((target("bf16"))) inline bfloat16x8_t
bf16_b_stride_tail(const bfloat16* b, int64_t ldb, int g, int cp, int kc, int nloc) noexcept
{
    uint16_t tmp[8] = {};
    for (int lane = 0; lane < 8; ++lane) {
        const int k = g * 4 + (lane & 3);
        const int col = 2 * cp + (lane >> 2);
        if (k < kc && col < nloc) {
            tmp[lane] = b[static_cast<int64_t>(k) * ldb + col].bits;
        }
    }
    return vreinterpretq_bf16_u16(vld1q_u16(tmp));
}

/// One k-group's B operand, out of the packed panel or the caller's matrix.
template <bool Strided, bool TAIL>
__attribute__((target("bf16"))) __attribute__((always_inline)) inline bfloat16x8_t
bf16_bop(const bfloat16* b, int64_t ldb, int g, int cp, [[maybe_unused]] int kc,
         [[maybe_unused]] int nloc) noexcept
{
    if constexpr (Strided) {
        if constexpr (TAIL) {
            return bf16_b_stride_tail(b, ldb, g, cp, kc, nloc);
        } else {
            return bf16_b_stride(b, ldb, g, cp);
        }
    } else {
        return bf16_b_panel(b, g, cp);
    }
}

// =========================================================================
//  Accumulator seeding, gathering and storing
// =========================================================================

/// One column pair's accumulator, seeded. Lane order is BFMMLA's: the row
/// pair's two columns for row 0, then the same two for row 1. A row past `mr`
/// goes through the same alias index as A and C, so `bias[r]` stays in range.
template <int seed>
__attribute__((target("bf16"))) inline float32x4_t bf16_seed(const bfloat16* bias, int r0, int r1,
                                                             int col) noexcept
{
    if constexpr (seed == 1) {
        return vcombine_f32(vdup_n_f32(sims::s_load(bias + r0)),
                            vdup_n_f32(sims::s_load(bias + r1)));
    } else if constexpr (seed == 2) {
        // Per-column: the pair's two columns land in both rows' halves.
        float32x2_t p = vdup_n_f32(sims::s_load(bias + col));
        p = vset_lane_f32(sims::s_load(bias + col + 1), p, 1);
        return vcombine_f32(p, p);
    } else {
        return vdupq_n_f32(0.0f);
    }
}

/// Join four column pairs' accumulators into one 8-column row: @p hi picks the
/// row of each pair (0 = the pair's first row, 1 = its second).
__attribute__((target("bf16"))) inline sims::v_f32x8
bf16_gather(bool hi, float32x4_t a0, float32x4_t a1, float32x4_t a2, float32x4_t a3) noexcept
{
    const float32x2_t x0 = hi ? vget_high_f32(a0) : vget_low_f32(a0);
    const float32x2_t x1 = hi ? vget_high_f32(a1) : vget_low_f32(a1);
    const float32x2_t x2 = hi ? vget_high_f32(a2) : vget_low_f32(a2);
    const float32x2_t x3 = hi ? vget_high_f32(a3) : vget_low_f32(a3);
    return sims::v_f32x8(vcombine_f32(x0, x1), vcombine_f32(x2, x3));
}

/// The C read that has to happen before any store of the block.
__attribute__((target("bf16"))) inline sims::v_f32x8 bf16_get8(const bfloat16* p) noexcept
{
    return sims::v_cvt_bf16_to_f32(reinterpret_cast<const uint16_t*>(p));
}

/// C add, clamp, narrow, store, for one row's 8 columns.
template <bool zero_mode>
__attribute__((target("bf16"))) inline void
bf16_put8(bfloat16* NNOPS_RESTRICT p, sims::v_f32x8 v, const sims::v_f32x8& cur,
          const sims::v_f32x8& vmin, const sims::v_f32x8& vmax) noexcept
{
    if constexpr (!zero_mode) {
        v = sims::v_add(v, cur);
    }
    sims::v_cvt_f32_to_bf16(reinterpret_cast<uint16_t*>(p),
                            sims::v_min(sims::v_max(v, vmin), vmax));
}

/// The alias index of every row: rows past `mr` reuse the last live one.
template <int PAIRS>
__attribute__((always_inline)) inline void
bf16_row_alias(int mr, int (&rr)[2 * PAIRS]) noexcept
{
    rr[0] = 0;
    rr[1] = std::min(1, mr - 1);
    if constexpr (PAIRS > 1) {
        rr[2] = std::min(2, mr - 1);
        rr[3] = std::min(3, mr - 1);
    }
    if constexpr (PAIRS > 2) {
        rr[4] = std::min(4, mr - 1);
        rr[5] = std::min(5, mr - 1);
    }
    if constexpr (PAIRS > 3) {
        rr[6] = std::min(6, mr - 1);
        rr[7] = std::min(7, mr - 1);
    }
}

// =========================================================================
//  One k group, the seed and the store — literal indices throughout
// =========================================================================

/// Seed every accumulator of a block: pair `p` holds rows 2p and 2p+1, column
/// pair `cp` the columns n + 2cp and n + 2cp + 1.
template <int PAIRS, int NCP, int seed, std::size_t... I>
__attribute__((target("bf16"))) __attribute__((always_inline)) inline void
bf16_seed_all(float32x4_t (&acc)[PAIRS][NCP], const bfloat16* bias, const int (&rr)[2 * PAIRS],
              int n, std::index_sequence<I...>) noexcept
{
    ((acc[I / NCP][I % NCP] =
          bf16_seed<seed>(bias, rr[2 * (I / NCP)], rr[2 * (I / NCP) + 1], n + 2 * (I % NCP))),
     ...);
}

/// One row pair's BFMMLAs against all of the block's column pairs.
template <int NCP>
__attribute__((target("bf16"))) __attribute__((always_inline)) inline void
bf16_fma_row(float32x4_t (&acc)[NCP], bfloat16x8_t aop,
             const bfloat16x8_t (&bop)[NCP]) noexcept
{
    acc[0] = vbfmmlaq_f32(acc[0], aop, bop[0]);
    acc[1] = vbfmmlaq_f32(acc[1], aop, bop[1]);
    acc[2] = vbfmmlaq_f32(acc[2], aop, bop[2]);
    acc[3] = vbfmmlaq_f32(acc[3], aop, bop[3]);
    if constexpr (NCP > 4) {
        acc[4] = vbfmmlaq_f32(acc[4], aop, bop[4]);
        acc[5] = vbfmmlaq_f32(acc[5], aop, bop[5]);
        acc[6] = vbfmmlaq_f32(acc[6], aop, bop[6]);
        acc[7] = vbfmmlaq_f32(acc[7], aop, bop[7]);
    }
}

/// One k group of a block: @p TAIL selects the zero-padded A operand for the
/// short final group (which does not advance the row pointers).
template <int PAIRS, bool Strided, int NCP, bool TAIL>
__attribute__((target("bf16"))) __attribute__((always_inline)) inline void
bf16_kstep(const bfloat16* (&a)[2 * PAIRS], const bfloat16* b, int64_t ldb, int g, int blk,
           int kc, int nloc, [[maybe_unused]] int ktail, float32x4_t (&acc)[PAIRS][NCP]) noexcept
{
    bfloat16x8_t bop[NCP];
    bop[0] = bf16_bop<Strided, TAIL>(b, ldb, g, blk * 4 + 0, kc, nloc);
    bop[1] = bf16_bop<Strided, TAIL>(b, ldb, g, blk * 4 + 1, kc, nloc);
    bop[2] = bf16_bop<Strided, TAIL>(b, ldb, g, blk * 4 + 2, kc, nloc);
    bop[3] = bf16_bop<Strided, TAIL>(b, ldb, g, blk * 4 + 3, kc, nloc);
    if constexpr (NCP > 4) {
        bop[4] = bf16_bop<Strided, TAIL>(b, ldb, g, blk * 4 + 4, kc, nloc);
        bop[5] = bf16_bop<Strided, TAIL>(b, ldb, g, blk * 4 + 5, kc, nloc);
        bop[6] = bf16_bop<Strided, TAIL>(b, ldb, g, blk * 4 + 6, kc, nloc);
        bop[7] = bf16_bop<Strided, TAIL>(b, ldb, g, blk * 4 + 7, kc, nloc);
    }
    if constexpr (TAIL) {
        bf16_fma_row<NCP>(acc[0], bf16_a_pair_tail(a[0], a[1], ktail), bop);
        if constexpr (PAIRS > 1) {
            bf16_fma_row<NCP>(acc[1], bf16_a_pair_tail(a[2], a[3], ktail), bop);
        }
        if constexpr (PAIRS > 2) {
            bf16_fma_row<NCP>(acc[2], bf16_a_pair_tail(a[4], a[5], ktail), bop);
        }
        if constexpr (PAIRS > 3) {
            bf16_fma_row<NCP>(acc[3], bf16_a_pair_tail(a[6], a[7], ktail), bop);
        }
    } else {
        bf16_fma_row<NCP>(acc[0], bf16_a_pair(a[0], a[1]), bop);
        if constexpr (PAIRS > 1) {
            bf16_fma_row<NCP>(acc[1], bf16_a_pair(a[2], a[3]), bop);
        }
        if constexpr (PAIRS > 2) {
            bf16_fma_row<NCP>(acc[2], bf16_a_pair(a[4], a[5]), bop);
        }
        if constexpr (PAIRS > 3) {
            bf16_fma_row<NCP>(acc[3], bf16_a_pair(a[6], a[7]), bop);
        }
        a[0] += 4;
        a[1] += 4;
        if constexpr (PAIRS > 1) {
            a[2] += 4;
            a[3] += 4;
        }
        if constexpr (PAIRS > 2) {
            a[4] += 4;
            a[5] += 4;
        }
        if constexpr (PAIRS > 3) {
            a[6] += 4;
            a[7] += 4;
        }
    }
}

/// Store a full block: every row's C read before any of them is written.
/// `I` walks (row, 8-column half) pairs, `NH` the halves per row.
template <int PAIRS, int NCP, bool zero_mode, std::size_t... I>
__attribute__((target("bf16"))) __attribute__((always_inline)) inline void
bf16_store_full_impl(int n, bfloat16* const (&c)[2 * PAIRS],
                     const float32x4_t (&acc)[PAIRS][NCP],
                     sims::v_f32x8 (&cur)[2 * PAIRS][NCP / 4], float lo, float hi,
                     std::index_sequence<I...>) noexcept
{
    constexpr int NH = NCP / 4;
    const sims::v_f32x8 vmin = sims::v_set1_f32x8(lo);
    const sims::v_f32x8 vmax = sims::v_set1_f32x8(hi);
    if constexpr (!zero_mode) {
        ((cur[I / NH][I % NH] = bf16_get8(c[I / NH] + n + 8 * (I % NH))), ...);
    }
    ((bf16_put8<zero_mode>(c[I / NH] + n + 8 * (I % NH),
                           bf16_gather(((I / NH) & 1) != 0, acc[(I / NH) / 2][4 * (I % NH) + 0],
                                       acc[(I / NH) / 2][4 * (I % NH) + 1],
                                       acc[(I / NH) / 2][4 * (I % NH) + 2],
                                       acc[(I / NH) / 2][4 * (I % NH) + 3]),
                           cur[I / NH][I % NH], vmin, vmax)),
     ...);
}

// =========================================================================
//  Tail blocks — generic, since at most one panel per call lands here
// =========================================================================

/// One live row of a store block: bias seed, the C read, clamp, narrow.
template <bool zero_mode, int seed>
__attribute__((target("bf16"))) inline void
bf16_store_row(bfloat16* NNOPS_RESTRICT cp, int n, int rem, int r, const bfloat16* bias,
               sims::v_f32x8 v, const sims::v_f32x8& vmin, const sims::v_f32x8& vmax) noexcept
{
    if constexpr (seed == 1) {
        v = sims::v_add(v, sims::v_set1_f32x8(static_cast<float>(bias[r])));
    } else if constexpr (seed == 2) {
        v = sims::v_add(v, bf16_get8(bias + n));
    }
    if constexpr (!zero_mode) {
        if (rem >= 8) {
            v = sims::v_add(v, bf16_get8(cp));
        } else {
            // A short block reads only its live columns; the panel's padding
            // past `rem` must not reach C.
            float cur[8] = {};
            for (int i = 0; i < rem; ++i) {
                cur[i] = static_cast<float>(cp[i]);
            }
            v = sims::v_add(v, sims::v_load(cur));
        }
    }
    v = sims::v_min(sims::v_max(v, vmin), vmax);

    if (rem >= 8) {
        sims::v_cvt_f32_to_bf16(reinterpret_cast<uint16_t*>(cp), v);
    } else {
        float lane[8];
        sims::v_store(lane, v);
        for (int i = 0; i < rem; ++i) {
            cp[i] = bfloat16(lane[i]);
        }
    }
}

/// Store one 8-column block, both rows of every pair spelled out rather than
/// looped over: @p rem is the block's live width (the panel's columns past it
/// are dropped).
template <int PAIRS, bool zero_mode, int seed>
__attribute__((target("bf16"))) inline void
bf16_store_block(int mr, int n, int rem, bfloat16* const (&c_rows)[2 * PAIRS],
                 const float32x4_t (&acc)[PAIRS][4], float lo, float hi,
                 const bfloat16* bias) noexcept
{
    const sims::v_f32x8 vmin = sims::v_set1_f32x8(lo);
    const sims::v_f32x8 vmax = sims::v_set1_f32x8(hi);
    for (int p = 0; p < PAIRS; ++p) {
        // Each accumulator holds a column pair for both rows of the pair: an
        // 8-column block is four halves joined, the low one for the pair's
        // first row and the high one for its second.
        if (2 * p < mr) {
            bf16_store_row<zero_mode, seed>(
                c_rows[2 * p] + n, n, rem, 2 * p, bias,
                bf16_gather(false, acc[p][0], acc[p][1], acc[p][2], acc[p][3]), vmin, vmax);
        }
        if (2 * p + 1 < mr) {
            bf16_store_row<zero_mode, seed>(
                c_rows[2 * p + 1] + n, n, rem, 2 * p + 1, bias,
                bf16_gather(true, acc[p][0], acc[p][1], acc[p][2], acc[p][3]), vmin, vmax);
        }
    }
}

/// The two routes' row pointers: a row past `mr` aliases the last live one, so
/// a partial panel stays inside its own rows.
template <int PAIRS>
inline void bf16_rows(const bfloat16* a, int lda, int mr, bfloat16* c, int ldc,
                      const bfloat16* (&a_rows)[2 * PAIRS], bfloat16* (&c_rows)[2 * PAIRS]) noexcept
{
    for (int r = 0; r < 2 * PAIRS; ++r) {
        const int i = (r < mr) ? r : mr - 1;
        a_rows[r] = a + static_cast<int64_t>(i) * lda;
        c_rows[r] = c + static_cast<int64_t>(i) * ldc;
    }
}

/// One 8-column block, array form. The kernels hand their leftover columns
/// (fewer than 8 for the 8-row rung, fewer than 16 for the narrow ones) to this.
template <int PAIRS, bool Strided, bool zero_mode, int seed>
__attribute__((target("bf16"))) inline void
bf16_block8(int mr, int n, int nc, int kc, bfloat16* c, int ldc, const bfloat16* a, int lda,
            const bfloat16* b, int64_t ldb, float lo, float hi, const bfloat16* bias) noexcept
{
    const bfloat16* a_rows[2 * PAIRS];
    bfloat16* c_rows[2 * PAIRS];
    bf16_rows<PAIRS>(a, lda, mr, c, ldc, a_rows, c_rows);

    const int blk = (n & 8) ? 1 : 0;
    // Columns this block may read out of a caller's matrix: b sits at the panel
    // holding n, so its first column is n & ~8 and the tile stops at nc.
    const int nloc = nc - (n & ~8);
    const int nblk = nloc - blk * 8;
    const int kgroups = kc >> 2;
    const int ktail = kc & 3;

    float32x4_t acc[PAIRS][4];
    for (int p = 0; p < PAIRS; ++p) {
        for (int cp = 0; cp < 4; ++cp) {
            acc[p][cp] = vdupq_n_f32(0.0f);
        }
    }

    for (int g = 0; g < kgroups; ++g) {
        bfloat16x8_t bop[4];
        for (int cp = 0; cp < 4; ++cp) {
            if constexpr (Strided) {
                bop[cp] = (nblk < 8) ? bf16_b_stride_tail(b, ldb, g, blk * 4 + cp, kc, nloc) :
                                       bf16_b_stride(b, ldb, g, blk * 4 + cp);
            } else {
                bop[cp] = bf16_b_panel(b, g, blk * 4 + cp);
            }
        }
        for (int p = 0; p < PAIRS; ++p) {
            const bfloat16x8_t aop = bf16_a_pair(a_rows[2 * p], a_rows[2 * p + 1]);
            for (int cp = 0; cp < 4; ++cp) {
                acc[p][cp] = vbfmmlaq_f32(acc[p][cp], aop, bop[cp]);
            }
            a_rows[2 * p] += 4;
            a_rows[2 * p + 1] += 4;
        }
    }
    if (ktail) {
        bfloat16x8_t bop[4];
        for (int cp = 0; cp < 4; ++cp) {
            if constexpr (Strided) {
                bop[cp] = bf16_b_stride_tail(b, ldb, kgroups, blk * 4 + cp, kc, nloc);
            } else {
                bop[cp] = bf16_b_panel(b, kgroups, blk * 4 + cp);
            }
        }
        for (int p = 0; p < PAIRS; ++p) {
            const bfloat16x8_t aop = bf16_a_pair_tail(a_rows[2 * p], a_rows[2 * p + 1], ktail);
            for (int cp = 0; cp < 4; ++cp) {
                acc[p][cp] = vbfmmlaq_f32(acc[p][cp], aop, bop[cp]);
            }
        }
    }

    bf16_store_block<PAIRS, zero_mode, seed>(mr, n, std::min(8, nc - n), c_rows, acc, lo, hi, bias);
}

/// Step from one packed panel to the next. Below -1 a negative ldb is how far
/// apart the caller's panels sit, which is the stride of the blob they were
/// packed into — for bf16 that is the whole-K slot, so it differs from this
/// call's own pack on a short k block. `-1` (and a non-negative ldb, which only
/// the in-place route sees) means the panels are this call's.
inline int64_t bf16_panel_step(int64_t ldb, int64_t kc) noexcept
{
    return (ldb < -1) ? -ldb : bf16_b_panel_stride(kc);
}

/// The leftover columns of a tile: 8-column blocks from @p n, starting at the
/// panel @p b already points at. One panel is 16 columns, so an in-place B
/// (Strided) steps 16 elements to the next one, not 16 k-rows.
template <int PAIRS, bool Strided, bool zero_mode, int seed>
__attribute__((target("bf16"))) inline void
bf16_tail(int mr, int n, int nc, int kc, bfloat16* c, int ldc, const bfloat16* a, int lda,
          const bfloat16* b, int64_t ldb, float lo, float hi, const bfloat16* bias) noexcept
{
    const int64_t advance = Strided ? 16 : bf16_panel_step(ldb, kc);
    for (; n < nc; n += 8) {
        bf16_block8<PAIRS, Strided, zero_mode, seed>(mr, n, nc, kc, c, ldc, a, lda, b, ldb, lo, hi,
                                                     bias);
        if (n & 8) {
            b += advance;
        }
    }
}

// =========================================================================
//  One rung: 8 columns per pass at 8 rows, a 16-column panel below that
// =========================================================================

template <int PAIRS, bool Strided, bool zero_mode, int seed>
__attribute__((target("bf16"))) inline void
bf16_kernel(int mr, int nc, int kc, bfloat16* NNOPS_RESTRICT C, int ldc,
            const bfloat16* NNOPS_RESTRICT A, int lda, const bfloat16* NNOPS_RESTRICT B,
            int64_t ldb, float lo, float hi, const bfloat16* NNOPS_RESTRICT bias) noexcept
{
    // Four column pairs (eight columns) keep the 8-row rung's 16 accumulators
    // and their A operands in registers; the narrow rungs have room for a whole
    // 16-column panel, which halves the number of passes.
    constexpr int NCP = (PAIRS == 4) ? 4 : 8;
    constexpr int BLK = 2 * NCP;

    int rr[2 * PAIRS];
    bf16_row_alias<PAIRS>(mr, rr);
    const bfloat16* a_rows[2 * PAIRS];
    bfloat16* c_rows[2 * PAIRS];
    bf16_rows<PAIRS>(A, lda, mr, C, ldc, a_rows, c_rows);

    const int kgroups = kc >> 2;
    const int ktail = kc & 3;
    const int64_t advance = Strided ? 16 : bf16_panel_step(ldb, kc);

    int n = 0;
    for (; n + BLK <= nc; n += BLK) {
        // A 16-column panel is two 8-column blocks for the 8-row rung; the
        // narrow ones take it whole.
        const int blk = (n & 8) ? 1 : 0;

        float32x4_t acc[PAIRS][NCP];
        bf16_seed_all<PAIRS, NCP, seed>(acc, bias, rr, n,
                                        std::make_index_sequence<PAIRS * NCP>{});

        const bfloat16* a[2 * PAIRS];
        a[0] = a_rows[0];
        a[1] = a_rows[1];
        if constexpr (PAIRS > 1) {
            a[2] = a_rows[2];
            a[3] = a_rows[3];
        }
        if constexpr (PAIRS > 2) {
            a[4] = a_rows[4];
            a[5] = a_rows[5];
        }
        if constexpr (PAIRS > 3) {
            a[6] = a_rows[6];
            a[7] = a_rows[7];
        }

        for (int g = 0; g < kgroups; ++g) {
            bf16_kstep<PAIRS, Strided, NCP, false>(a, B, ldb, g, blk, kc, nc, ktail, acc);
        }
        if (ktail) {
            bf16_kstep<PAIRS, Strided, NCP, true>(a, B, ldb, kgroups, blk, kc, nc, ktail, acc);
        }

        sims::v_f32x8 cur[2 * PAIRS][NCP / 4];
        bf16_store_full_impl<PAIRS, NCP, zero_mode>(
            n, c_rows, acc, cur, lo, hi, std::make_index_sequence<2 * PAIRS * (NCP / 4)>{});

        // The 8-row rung takes half a panel per pass, the narrow ones a whole
        // panel, so only the former steps B every other block.
        if constexpr (BLK == 16) {
            B += advance;
        } else if (blk) {
            B += advance;
        }
    }

    if (n < nc) {
        bf16_tail<PAIRS, Strided, zero_mode, seed>(mr, n, nc, kc, C, ldc, A, lda, B, ldb, lo, hi,
                                                   bias);
    }
}

// =========================================================================
//  Widening f32 fallback — the same blocking without FEAT_BF16
// =========================================================================

template <int PAIRS, bool Strided, bool zero_mode, int seed>
inline void bf16_gemm_wide(int mr, int nc, int kc, bfloat16* c, int ldc, const bfloat16* a, int lda,
                           const bfloat16* b, int64_t ldb, float lo, float hi,
                           const bfloat16* bias) noexcept
{
    const bfloat16* a_rows[2 * PAIRS];
    bfloat16* c_rows[2 * PAIRS];
    bf16_rows<PAIRS>(a, lda, mr, c, ldc, a_rows, c_rows);

    const int64_t advance = Strided ? 16 : bf16_panel_step(ldb, kc);

    for (int n = 0; n < nc; n += 8) {
        const int rem = std::min(8, nc - n);
        const int blk = (n & 8) ? 1 : 0;
        // Columns live in the panel holding n; a caller's matrix ends at nc.
        const int nloc = Strided ? nc - (n & ~8) : nc;

        float32x4_t acc[PAIRS][4];
        for (int p = 0; p < PAIRS; ++p) {
            for (int cp = 0; cp < 4; ++cp) {
                acc[p][cp] = vdupq_n_f32(0.0f);
            }
        }

        for (int k = 0; k < kc; ++k) {
            float bcol[8];
            if constexpr (!Strided) {
                const uint16_t* p = reinterpret_cast<const uint16_t*>(b);
                p += (k >> 2) * bf16_b_k4_elems + blk * (bf16_b_k4_elems / 2) + (k & 3);
                for (int cp = 0; cp < 4; ++cp) {
                    bcol[2 * cp] = ::nnops::bf16_to_float(p[cp * 8]);
                    bcol[2 * cp + 1] = ::nnops::bf16_to_float(p[cp * 8 + 4]);
                }
            } else {
                for (int c = 0; c < 8; ++c) {
                    const int col = std::min(blk * 8 + c, nloc - 1);
                    bcol[c] = static_cast<float>(b[static_cast<int64_t>(k) * ldb + col]);
                }
            }
            for (int p = 0; p < PAIRS; ++p) {
                const float a0 = static_cast<float>(a_rows[2 * p][k]);
                const float a1 = static_cast<float>(a_rows[2 * p + 1][k]);
                const float av[4] = {a0, a0, a1, a1};
                const float32x4_t avv = vld1q_f32(av);
                for (int cp = 0; cp < 4; ++cp) {
                    const float bv[4] = {bcol[2 * cp], bcol[2 * cp + 1], bcol[2 * cp],
                                         bcol[2 * cp + 1]};
                    acc[p][cp] = vfmaq_f32(acc[p][cp], avv, vld1q_f32(bv));
                }
            }
        }

        bf16_store_block<PAIRS, zero_mode, seed>(mr, n, rem, c_rows, acc, lo, hi, bias);
        if (blk == 1) {
            b += advance;
        }
    }
}

/// Test/bench hook: force the widening path even on a FEAT_BF16 core, so the
/// fallback is covered on hardware that has BFMMLA.
inline bool bf16_force_wide = false;

template <int PAIRS, bool zero_mode, int seed>
inline void bf16_gemm(int mr, int nc, int kc, bfloat16* c, int ldc, const bfloat16* a, int lda,
                      const bfloat16* b, int64_t ldb, float lo, float hi,
                      const bfloat16* bias) noexcept
{
    if (sims::cpu_has_bf16() && !bf16_force_wide) {
        if (ldb > 0) {
            bf16_kernel<PAIRS, true, zero_mode, seed>(mr, nc, kc, c, ldc, a, lda, b, ldb, lo, hi,
                                                      bias);
        } else {
            bf16_kernel<PAIRS, false, zero_mode, seed>(mr, nc, kc, c, ldc, a, lda, b, ldb, lo, hi,
                                                       bias);
        }
    } else {
        if (ldb > 0) {
            bf16_gemm_wide<PAIRS, true, zero_mode, seed>(mr, nc, kc, c, ldc, a, lda, b, ldb, lo, hi,
                                                         bias);
        } else {
            bf16_gemm_wide<PAIRS, false, zero_mode, seed>(mr, nc, kc, c, ldc, a, lda, b, ldb, lo,
                                                          hi, bias);
        }
    }
}

// =========================================================================
//  Public entry points — one per B form, as the other dtypes' kernels
// =========================================================================

/// A packed: the panel base walks `kc` apart inside its slot.
template <int PAIRS, bool zero_mode = false, int seed = 0>
inline void mma_pack_mrxncxkc_bf16(int mr, int nc, int kc, bfloat16* C, int ldc, const bfloat16* A,
                                   const bfloat16* B, float clamp_min, float clamp_max,
                                   const bfloat16* bias = nullptr) noexcept
{
    bf16_gemm<PAIRS, zero_mode, seed>(mr, nc, kc, C, ldc, A, kc, B, -1, clamp_min, clamp_max, bias);
}

/// A read row-major through @p lda; `ldb` is the packed panel (<= -1) or B's
/// own row pitch (> 0).
template <int PAIRS, bool zero_mode = false, int seed = 0>
inline void mma_direct_mrxncxkc_bf16(int mr, int nc, int kc, bfloat16* C, int ldc,
                                     const bfloat16* A, int lda, const bfloat16* B, int64_t ldb,
                                     float clamp_min, float clamp_max,
                                     const bfloat16* bias = nullptr) noexcept
{
    bf16_gemm<PAIRS, zero_mode, seed>(mr, nc, kc, C, ldc, A, lda, B, ldb, clamp_min, clamp_max,
                                      bias);
}

// ---- dispatch tables, consumed by matmul_helper.cpp ---------------------
template <bool Z, int Seed = 0>
inline constexpr auto mma_pack_fns_bf16 =
    std::array{&mma_pack_mrxncxkc_bf16<4, Z, Seed>, &mma_pack_mrxncxkc_bf16<2, Z, Seed>,
               &mma_pack_mrxncxkc_bf16<1, Z, Seed>};

template <bool Z, int Seed = 0>
inline constexpr auto mma_direct_fns_bf16 =
    std::array{&mma_direct_mrxncxkc_bf16<4, Z, Seed>, &mma_direct_mrxncxkc_bf16<2, Z, Seed>,
               &mma_direct_mrxncxkc_bf16<1, Z, Seed>};

} // namespace nnops::backend::cpu::aarch64
