#pragma once
/// @file mma_f16_sme.hpp
/// @brief AArch64 SME float16 mrxncxkc MMA kernel with f32 accumulation.
///
/// The four ZA tiles are a 2x2 of 16x16 faces — 32 rows by 32 columns — the
/// same tile as mma_f32_sme.hpp: one K step is two A vectors, two B vectors and
/// four `FMOPA ZA.S` with `.H` operands. The hardware two-way dot consumes two
/// K per instruction, so each vector already holds a K pair per lane; the pack
/// pre-interleaved them (pack_f16_sme.hpp): A at [row][k,k+1], B at [col][k,k+1].
///
/// With A stored in 32-row blocks, a full tile's two A vectors and a B tile's
/// two vectors are each one SVE2 pair load (`ld1h {z,z}`), so the K loop is two
/// loads and four MOPA instructions.
///
/// The MOPA predicates are **b16**: the operand's lane 2i and 2i+1 are entry
/// i's two K values, so a row is live when both lanes are. A b32 predicate
/// activates only the even lanes, which silently drops half the products.
///
/// C is f16 storage; the accumulator and the bias seed are f32. @p zero_mode
/// overwrites C instead of accumulating; @p col_bias (per output column) seeds
/// the accumulator and overwrites C. Every stored element is clamped to
/// [@p lo, @p hi] before narrowing.
///
/// @p b_block_stride is the element distance between 32-column blocks: a
/// per-call pack lays them ceil(kc/2)*64 apart, a whole-K prepacked blob
/// ceil(K/2)*64. 0 means the per-call form.

#if defined(NNOPS_ARCH_SME)

#include <algorithm>
#include <cstdint>

#include <arm_neon.h>
#include <arm_sme.h>

#include "nnops/detail/half.hpp"

#include "backend/cpu/aarch64/sme/mma_sme.hpp"

namespace nnops::backend::cpu::aarch64 {

namespace sme_detail {

/// One pre-interleaved K pair for 16 lanes: 32 f16, one full register.
__arm_locally_streaming __attribute__((target("+sme2"), always_inline))
inline svfloat16_t load_h32(svbool_t pg, const half* p) noexcept
{
    return svld1_f16(pg, reinterpret_cast<const float16_t*>(p));
}

/// Widen 16 f16 lanes to 16 f32 lanes in order (the bias seed's path).
__arm_locally_streaming __attribute__((target("+sme2"), always_inline))
inline svfloat32_t widen_h16(svfloat16_t h) noexcept
{
    const svbool_t pgh = svwhilelt_b16(0, sme_za_n);
    return svzip1_f32(svcvt_f32_f16_x(pgh, h), svcvtlt_f32_f16_x(pgh, h));
}

__arm_locally_streaming __attribute__((target("+sme2"), always_inline))
inline void store_op(half* p, svbool_t pg32, svfloat32_t v, int live) noexcept
{
    // The SVE narrow puts the even source lanes in the low half and the odd
    // ones in the high half; uzp1(h, h) interleaves them back into order. The
    // b16 predicate stores exactly the live columns, so a short last panel
    // never runs into the next row of C.
    const svfloat16_t h = svcvt_f16_f32_x(pg32, v);
    svst1_f16(svwhilelt_b16(0, live), reinterpret_cast<float16_t*>(p), svuzp1_f16(h, h));
}

} // namespace sme_detail

// The four MOPA instructions of one K-pair step, from the two pair loads. A
// macro for the same reason as NNOPS_SME_MOPA4_F32.
// One full-width C row from the two column-group tiles' row vectors: both
// groups narrow to one 64-byte store, and the accumulate read is a single
// 64-byte load.
#define NNOPS_SME_STORE_WIDE_F16(v0_, v1_, row_)                               \
    do {                                                                        \
        svfloat32_t nno_v0 = v0_;                                               \
        svfloat32_t nno_v1 = v1_;                                               \
        if (accumulate) {                                                       \
            const svbool_t nno_ph = svwhilelt_b16(0, sme_za_n);                  \
            const svfloat16_t nno_c =                                           \
                svld1_f16(svptrue_b16(), reinterpret_cast<const float16_t*>(row_)); \
            nno_v0 = svadd_f32_x(                                                \
                pn0, nno_v0,                                                     \
                svzip1_f32(svcvt_f32_f16_x(nno_ph, nno_c),                       \
                           svcvtlt_f32_f16_x(nno_ph, nno_c)));                   \
            const svfloat16_t nno_hi = svext_f16(nno_c, nno_c, sme_za_n);        \
            nno_v1 = svadd_f32_x(                                                \
                pn1, nno_v1,                                                     \
                svzip1_f32(svcvt_f32_f16_x(nno_ph, nno_hi),                      \
                           svcvtlt_f32_f16_x(nno_ph, nno_hi)));                  \
        }                                                                       \
        nno_v0 = sme_detail::clamp_v(pn0, nno_v0, vlo, vhi);                     \
        nno_v1 = sme_detail::clamp_v(pn1, nno_v1, vlo, vhi);                     \
        svst1_f16(svptrue_b16(), reinterpret_cast<float16_t*>(row_),             \
                  svuzp1_f16(svcvt_f16_f32_x(pn0, nno_v0),                       \
                             svcvt_f16_f32_x(pn1, nno_v1)));                     \
    } while (0)

#define NNOPS_SME_MOPA4_F16(pm0_, pm1_, qn0_, qn1_, av_, bv_)      \
    do {                                                           \
        const svfloat16_t nno_av0 = svget2_f16(av_, 0);            \
        const svfloat16_t nno_av1 = svget2_f16(av_, 1);            \
        const svfloat16_t nno_bv0 = svget2_f16(bv_, 0);            \
        const svfloat16_t nno_bv1 = svget2_f16(bv_, 1);            \
        svmopa_za32_f16_m(0, pm0_, qn0_, nno_av0, nno_bv0);        \
        svmopa_za32_f16_m(1, pm1_, qn0_, nno_av1, nno_bv0);        \
        svmopa_za32_f16_m(2, pm0_, qn1_, nno_av0, nno_bv1);        \
        svmopa_za32_f16_m(3, pm1_, qn1_, nno_av1, nno_bv1);        \
    } while (0)

__arm_locally_streaming __arm_new("za") __attribute__((target("+sme2")))
inline void mma_f16_sme_skinny(int mc, int nc, int kc, half* c, int ldc, const half* a,
                               const half* b, bool zero_mode, float lo, float hi,
                               const half* col_bias, int64_t b_block_stride = 0) noexcept
{
    constexpr int kRows = 2 * sme_mr;   // 32 rows: two A groups
    constexpr int kCols = 2 * sme_za_n; // 32 columns: two B groups
    const int n_tiles = (nc + kCols - 1) / kCols;
    const int k_pairs = (kc + 1) / 2;
    const svfloat32_t vlo = svdup_n_f32(lo);
    const svfloat32_t vhi = svdup_n_f32(hi);
    // A bias seed builds the tile from the bias; the destination's old value
    // is not part of the sum, so the store overwrites even when zero_mode
    // is off.
    const bool accumulate = !zero_mode && col_bias == nullptr;

    // A single 16-row tile: the four ZA tiles are four column groups, so one
    // K-pair step is one A vector, two B pair loads and four MOPAs — as in the
    // f32 kernel.
    const svbool_t pmh = svwhilelt_b16(0, 2 * mc);
    const svbool_t pg = svptrue_b32();
    // Both column groups of both tiles are live here; the wide-store
    // macro names them.
    const svbool_t pn0 = pg;
    const svbool_t pn1 = pg;
    const svbool_t qg = svptrue_b16();
    const svcount_t q16 = svptrue_c16();
    const int64_t block_stride =
        b_block_stride != 0 ? b_block_stride :
                              static_cast<int64_t>(k_pairs) * (2 * sme_fp_nr);
    for (int nt = 0; nt < n_tiles; nt += 2) {
        const int na = nt * kCols;
        const int nb = na + kCols;
        const half* const bpa =
            b + static_cast<int64_t>(na / sme_fp_nr) * block_stride;
        const half* const bpb =
            b + static_cast<int64_t>(nb / sme_fp_nr) * block_stride;
        svzero_za();
        const half* arow = a;
        const half* browa = bpa;
        const half* browb = bpb;
        for (int kp = 0; kp < k_pairs;
             ++kp, arow += 2 * kRows, browa += 2 * sme_fp_nr,
                  browb += 2 * sme_fp_nr) {
            const svfloat16_t av = sme_detail::load_h32(pmh, arow);
            const svfloat16x2_t bva = svld1_f16_x2(
                q16, reinterpret_cast<const float16_t*>(browa));
            const svfloat16x2_t bvb = svld1_f16_x2(
                q16, reinterpret_cast<const float16_t*>(browb));
            svmopa_za32_f16_m(0, pmh, qg, av, svget2_f16(bva, 0));
            svmopa_za32_f16_m(2, pmh, qg, av, svget2_f16(bva, 1));
            svmopa_za32_f16_m(1, pmh, qg, av, svget2_f16(bvb, 0));
            svmopa_za32_f16_m(3, pmh, qg, av, svget2_f16(bvb, 1));
        }
        for (int r = 0; r < mc; ++r) {
            half* const rowa = c + static_cast<int64_t>(r) * ldc + na;
            NNOPS_SME_STORE_WIDE_F16(
                svread_hor_za32_f32_m(svundef_f32(), pg, 0, r),
                svread_hor_za32_f32_m(svundef_f32(), pg, 2, r), rowa);
            half* const rowb = c + static_cast<int64_t>(r) * ldc + nb;
            NNOPS_SME_STORE_WIDE_F16(
                svread_hor_za32_f32_m(svundef_f32(), pg, 1, r),
                svread_hor_za32_f32_m(svundef_f32(), pg, 3, r), rowb);
        }
    }
}

__arm_locally_streaming __arm_new("za") __attribute__((target("+sme2")))
inline void mma_f16_sme_tiles(int mc, int nc, int kc, half* c, int ldc, const half* a,
                              const half* b, bool zero_mode, float lo, float hi,
                              const half* col_bias, int64_t b_block_stride = 0) noexcept
{
    constexpr int kRows = 2 * sme_mr;   // 32 rows: two A groups
    constexpr int kCols = 2 * sme_za_n; // 32 columns: two B groups
    const int m_tiles = (mc + kRows - 1) / kRows;
    const int n_tiles = (nc + kCols - 1) / kCols;
    const int k_pairs = (kc + 1) / 2;
    const svfloat32_t vlo = svdup_n_f32(lo);
    const svfloat32_t vhi = svdup_n_f32(hi);
    // A bias seed builds the tile from the bias; the destination's old value
    // is not part of the sum, so the store overwrites even when zero_mode
    // is off.
    const bool accumulate = !zero_mode && col_bias == nullptr;

    // Same loop order as the f32 kernel: n tile outside, M tiles inside.
    for (int nt = 0; nt < n_tiles; ++nt) {
        const int n0 = nt * kCols;
        const int nlive = std::min(kCols, nc - n0);
        const bool t1 = nlive > sme_za_n;
        const int live0 = std::min(sme_za_n, nlive);
        const int live1 = t1 ? nlive - sme_za_n : 0;
        const svbool_t pn0 = svwhilelt_b32(0, live0);
        const svbool_t pn1 = t1 ? svwhilelt_b32(0, live1) : svpfalse_b();
        const svbool_t qn0 = svwhilelt_b16(0, 2 * live0);
        const svbool_t qn1 = t1 ? svwhilelt_b16(0, 2 * live1) : svpfalse_b();
        // The 32-column B block is exactly this n tile.
        const half* const bp =
            b + static_cast<int64_t>(n0 / sme_fp_nr) *
                    (b_block_stride != 0 ? b_block_stride :
                                           static_cast<int64_t>(k_pairs) * (2 * sme_fp_nr));
        const half* const bias0 = col_bias != nullptr ? col_bias + n0 : nullptr;

        for (int mt = 0; mt < m_tiles; ++mt) {
            const int m0 = mt * kRows;
            const int mlive = std::min(kRows, mc - m0);
            const bool full = mlive == kRows;
            const bool r1 = mlive > sme_mr;
            const int lo_rows = std::min(sme_mr, mlive);
            const svbool_t pmh0 = svwhilelt_b16(0, 2 * lo_rows);
            const svbool_t pmh1 = r1 ? svwhilelt_b16(0, 2 * (mlive - sme_mr)) : svpfalse_b();
            // [m/32][k/2][64]: m0 is a multiple of 32, so the block base is
            // m0 * k_pairs * 2 and the two 32-half vectors are contiguous.
            const half* const ap = a + static_cast<int64_t>(m0) * k_pairs * 2;

            svzero_za();
            if (bias0 != nullptr) {
                const svfloat32_t bv0 =
                    sme_detail::widen_h16(svld1_f16(svwhilelt_b16(0, live0), bias0));
                const svfloat32_t bv1 =
                    t1 ? sme_detail::widen_h16(
                             svld1_f16(svwhilelt_b16(0, live1), bias0 + sme_za_n)) :
                         svdup_n_f32(0.0f);
                // Tiles 0/2 are the low row group and 1/3 the high one;
                // 0/1 are the low column group and 2/3 the high one. The ZA
                // tile index is an immediate, hence the split.
                for (int r = 0; r < lo_rows; ++r) {
                    svwrite_hor_za32_f32_m(0, r, pn0, bv0);
                    if (t1) {
                        svwrite_hor_za32_f32_m(2, r, pn1, bv1);
                    }
                }
                for (int r = 0; r < mlive - sme_mr; ++r) {
                    svwrite_hor_za32_f32_m(1, r, pn0, bv0);
                    if (t1) {
                        svwrite_hor_za32_f32_m(3, r, pn1, bv1);
                    }
                }
            }

            const half* brow = bp;
            int kp = 0;
            const svcount_t q16 = svptrue_c16();
            if (full) {
                // One pair load for the 32 rows, one for B's 32 columns,
                // unrolled four K pairs as in the f32 kernel.
                for (; kp + 4 <= k_pairs; kp += 4) {
                    const half* const ak = ap + static_cast<int64_t>(kp) * 2 * kRows;
                    const half* const bk = brow;
                    const svfloat16x2_t a0 = svld1_f16_x2(
                        q16, reinterpret_cast<const float16_t*>(ak));
                    const svfloat16x2_t a1 = svld1_f16_x2(
                        q16, reinterpret_cast<const float16_t*>(ak + 2 * kRows));
                    const svfloat16x2_t a2 = svld1_f16_x2(
                        q16, reinterpret_cast<const float16_t*>(ak + 4 * kRows));
                    const svfloat16x2_t a3 = svld1_f16_x2(
                        q16, reinterpret_cast<const float16_t*>(ak + 6 * kRows));
                    NNOPS_SME_MOPA4_F16(
                        pmh0, pmh1, qn0, qn1, a0,
                        svld1_f16_x2(q16, reinterpret_cast<const float16_t*>(bk)));
                    NNOPS_SME_MOPA4_F16(
                        pmh0, pmh1, qn0, qn1, a1,
                        svld1_f16_x2(q16,
                                     reinterpret_cast<const float16_t*>(bk + 2 * sme_fp_nr)));
                    NNOPS_SME_MOPA4_F16(
                        pmh0, pmh1, qn0, qn1, a2,
                        svld1_f16_x2(q16,
                                     reinterpret_cast<const float16_t*>(bk + 4 * sme_fp_nr)));
                    NNOPS_SME_MOPA4_F16(
                        pmh0, pmh1, qn0, qn1, a3,
                        svld1_f16_x2(q16,
                                     reinterpret_cast<const float16_t*>(bk + 6 * sme_fp_nr)));
                    brow += 8 * sme_fp_nr;
                }
                for (; kp < k_pairs; ++kp, brow += 2 * sme_fp_nr) {
                    NNOPS_SME_MOPA4_F16(
                        pmh0, pmh1, qn0, qn1,
                        svld1_f16_x2(q16, reinterpret_cast<const float16_t*>(
                                               ap + static_cast<int64_t>(kp) * 2 * kRows)),
                        svld1_f16_x2(q16, reinterpret_cast<const float16_t*>(brow)));
                }
            } else {
                for (; kp < k_pairs; ++kp, brow += 2 * sme_fp_nr) {
                    const svfloat16_t av0 = sme_detail::load_h32(
                        pmh0, ap + static_cast<int64_t>(kp) * 2 * kRows);
                    const svfloat16_t av1 =
                        r1 ? sme_detail::load_h32(pmh1,
                                                  ap + static_cast<int64_t>(kp) * 2 * kRows +
                                                      2 * sme_mr) :
                             svdup_n_f16(static_cast<float16_t>(0.0f));
                    const svfloat16x2_t bv =
                        svld1_f16_x2(q16, reinterpret_cast<const float16_t*>(brow));
                    const svfloat16_t bv0 = svget2_f16(bv, 0);
                    const svfloat16_t bv1 = svget2_f16(bv, 1);
                    svmopa_za32_f16_m(0, pmh0, qn0, av0, bv0);
                    svmopa_za32_f16_m(1, pmh1, qn0, av1, bv0);
                    svmopa_za32_f16_m(2, pmh0, qn1, av0, bv1);
                    svmopa_za32_f16_m(3, pmh1, qn1, av1, bv1);
                }
            }

            // A full-width row is 32 halfs = one 64-byte vector: both column
            // groups narrow into one store, and the accumulate read is one
            // 64-byte load split by svext. Two 32-byte halves were two
            // partial-line writes.
            const bool wide = t1 && nlive == kCols;
            if (wide) {
                // Four ZA rows per MOVA, as in the f32 kernel: the horizontal
                // read dominates this loop.
                int r = 0;
                for (; r + 4 <= lo_rows; r += 4) {
                    const svfloat32x4_t g0 = svread_hor_za32_f32_vg4(0, r);
                    const svfloat32x4_t g2 = svread_hor_za32_f32_vg4(2, r);
                    half* row = c + static_cast<int64_t>(m0 + r) * ldc + n0;
                    NNOPS_SME_STORE_WIDE_F16(svget4_f32(g0, 0), svget4_f32(g2, 0), row);
                    NNOPS_SME_STORE_WIDE_F16(svget4_f32(g0, 1), svget4_f32(g2, 1),
                                             row + ldc);
                    NNOPS_SME_STORE_WIDE_F16(svget4_f32(g0, 2), svget4_f32(g2, 2),
                                             row + 2 * ldc);
                    NNOPS_SME_STORE_WIDE_F16(svget4_f32(g0, 3), svget4_f32(g2, 3),
                                             row + 3 * ldc);
                }
                for (; r < lo_rows; ++r) {
                    half* const row = c + static_cast<int64_t>(m0 + r) * ldc + n0;
                    NNOPS_SME_STORE_WIDE_F16(
                        svread_hor_za32_f32_m(svundef_f32(), pn0, 0, r),
                        svread_hor_za32_f32_m(svundef_f32(), pn1, 2, r), row);
                }
                const int hi_rows = mlive - sme_mr;
                for (r = 0; r + 4 <= hi_rows; r += 4) {
                    const svfloat32x4_t g1 = svread_hor_za32_f32_vg4(1, r);
                    const svfloat32x4_t g3 = svread_hor_za32_f32_vg4(3, r);
                    half* row = c + static_cast<int64_t>(m0 + sme_mr + r) * ldc + n0;
                    NNOPS_SME_STORE_WIDE_F16(svget4_f32(g1, 0), svget4_f32(g3, 0), row);
                    NNOPS_SME_STORE_WIDE_F16(svget4_f32(g1, 1), svget4_f32(g3, 1),
                                             row + ldc);
                    NNOPS_SME_STORE_WIDE_F16(svget4_f32(g1, 2), svget4_f32(g3, 2),
                                             row + 2 * ldc);
                    NNOPS_SME_STORE_WIDE_F16(svget4_f32(g1, 3), svget4_f32(g3, 3),
                                             row + 3 * ldc);
                }
                for (; r < hi_rows; ++r) {
                    half* const row =
                        c + static_cast<int64_t>(m0 + sme_mr + r) * ldc + n0;
                    NNOPS_SME_STORE_WIDE_F16(
                        svread_hor_za32_f32_m(svundef_f32(), pn0, 1, r),
                        svread_hor_za32_f32_m(svundef_f32(), pn1, 3, r), row);
                }
            } else {
                for (int r = 0; r < lo_rows; ++r) {
                    half* const row = c + static_cast<int64_t>(m0 + r) * ldc + n0;
                    {
                        svfloat32_t v = svread_hor_za32_f32_m(svundef_f32(), pn0, 0, r);
                        if (accumulate) {
                            v = svadd_f32_x(
                                pn0, v,
                                sme_detail::widen_h16(svld1_f16(svwhilelt_b16(0, live0), row)));
                        }
                        sme_detail::store_op(row, pn0, sme_detail::clamp_v(pn0, v, vlo, vhi),
                                             live0);
                    }
                    if (t1) {
                        half* const p = row + sme_za_n;
                        svfloat32_t w = svread_hor_za32_f32_m(svundef_f32(), pn1, 2, r);
                        if (accumulate) {
                            w = svadd_f32_x(
                                pn1, w,
                                sme_detail::widen_h16(svld1_f16(svwhilelt_b16(0, live1), p)));
                        }
                        sme_detail::store_op(p, pn1, sme_detail::clamp_v(pn1, w, vlo, vhi),
                                             live1);
                    }
                }
                for (int r = 0; r < mlive - sme_mr; ++r) {
                    half* const row =
                        c + static_cast<int64_t>(m0 + sme_mr + r) * ldc + n0;
                    {
                        svfloat32_t v = svread_hor_za32_f32_m(svundef_f32(), pn0, 1, r);
                        if (accumulate) {
                            v = svadd_f32_x(
                                pn0, v,
                                sme_detail::widen_h16(svld1_f16(svwhilelt_b16(0, live0), row)));
                        }
                        sme_detail::store_op(row, pn0, sme_detail::clamp_v(pn0, v, vlo, vhi),
                                             live0);
                    }
                    if (t1) {
                        half* const p = row + sme_za_n;
                        svfloat32_t w = svread_hor_za32_f32_m(svundef_f32(), pn1, 3, r);
                        if (accumulate) {
                            w = svadd_f32_x(
                                pn1, w,
                                sme_detail::widen_h16(svld1_f16(svwhilelt_b16(0, live1), p)));
                        }
                        sme_detail::store_op(p, pn1, sme_detail::clamp_v(pn1, w, vlo, vhi),
                                             live1);
                    }
                }
            }
        }
    }
}

/// Dispatch the single 16-row tile (four column groups) to its own kernel;
/// see mma_f32_sme: both callees are ZA kernels, so the split costs one
/// branch, not a second streaming-mode transition.
inline void mma_f16_sme(int mc, int nc, int kc, half* c, int ldc, const half* a,
                        const half* b, bool zero_mode, float lo, float hi,
                        const half* col_bias, int64_t b_block_stride = 0) noexcept
{
    constexpr int kCols = 2 * sme_za_n;
    if (mc <= sme_mr && col_bias == nullptr && nc % (2 * kCols) == 0) {
        mma_f16_sme_skinny(mc, nc, kc, c, ldc, a, b, zero_mode, lo, hi, col_bias,
                           b_block_stride);
        return;
    }
    mma_f16_sme_tiles(mc, nc, kc, c, ldc, a, b, zero_mode, lo, hi, col_bias,
                      b_block_stride);
}


} // namespace nnops::backend::cpu::aarch64

#undef NNOPS_SME_MOPA4_F16
#undef NNOPS_SME_STORE_WIDE_F16

#endif // NNOPS_ARCH_SME
