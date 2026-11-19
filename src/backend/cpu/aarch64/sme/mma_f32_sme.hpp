#pragma once
/// @file mma_f32_sme.hpp
/// @brief AArch64 SME float32 mrxncxkc MMA kernel.
///
/// C[m][n] += A[m][k] * B[k][n] for one k-block, mc x nc x kc. The four ZA
/// tiles are arranged as a 2x2 of 16x16 faces — a 32-row x 32-column block — so
/// one k-step is two A vectors and two B vectors feeding four independent
/// `FMOPA ZA.S` instructions. That is the balanced shape: a 16x64 block would
/// load one A vector and four B vectors for the same 1024 outputs, and B is the
/// stream every M tile re-reads. This is the tile KleidiAI's SME2 MOPA kernels
/// use, and it roughly doubles the kernel against the 16x64 form.
///
/// @p a and @p b are the packed layouts pack_f32_sme.hpp writes (A in 16-row
/// groups, B in 64-column blocks); a 32-row tile consumes two A groups and a
/// 32-column tile is half a B block, both read in place.
///
/// @p zero_mode overwrites C instead of accumulating; @p col_bias (per output
/// column) seeds the accumulator and overwrites C. Every stored element is
/// clamped to [@p lo, @p hi].
///
/// @p b_block_stride is the element distance between 64-column blocks: a
/// per-call pack lays them kc*64 apart, a whole-K prepacked blob K*64. 0 means
/// the per-call form.

#if defined(NNOPS_ARCH_SME)

#include <algorithm>
#include <cstdint>

#include <arm_sme.h>

#include "backend/cpu/aarch64/sme/mma_sme.hpp"

namespace nnops::backend::cpu::aarch64 {

namespace sme_detail {

__arm_locally_streaming __attribute__((target("+sme2"), always_inline))
inline svfloat32_t load_op(svbool_t pg, const float* p) noexcept
{
    return svld1(pg, p);
}

} // namespace sme_detail

// The four MOPA instructions of one K step, from the two pair loads. A macro,
// not a helper: the ZA-state builtins require the ZA annotation on a
// definition, which the compiler refuses to pair with always_inline.
#define NNOPS_SME_MOPA4_F32(pm0_, pm1_, pn0_, pn1_, av_, bv_)      \
    do {                                                           \
        const svfloat32_t nno_av0 = svget2_f32(av_, 0);            \
        const svfloat32_t nno_av1 = svget2_f32(av_, 1);            \
        const svfloat32_t nno_bv0 = svget2_f32(bv_, 0);            \
        const svfloat32_t nno_bv1 = svget2_f32(bv_, 1);            \
        svmopa_za32_f32_m(0, pm0_, pn0_, nno_av0, nno_bv0);        \
        svmopa_za32_f32_m(1, pm1_, pn0_, nno_av1, nno_bv0);        \
        svmopa_za32_f32_m(2, pm0_, pn1_, nno_av0, nno_bv1);        \
        svmopa_za32_f32_m(3, pm1_, pn1_, nno_av1, nno_bv1);        \
    } while (0)

// The two column groups of the low row group only: a tile with at most 16
// live rows would otherwise issue two MOPAs that a false predicate discards.
#define NNOPS_SME_MOPA2_F32(pm0_, pn0_, pn1_, av0_, bv_)           \
    do {                                                           \
        const svfloat32_t nno_bv0 = svget2_f32(bv_, 0);            \
        const svfloat32_t nno_bv1 = svget2_f32(bv_, 1);            \
        svmopa_za32_f32_m(0, pm0_, pn0_, av0_, nno_bv0);           \
        svmopa_za32_f32_m(2, pm0_, pn1_, av0_, nno_bv1);           \
    } while (0)

// One full-width C row from the two column-group tiles' row vectors. Both
// halves in one 128-byte access; the accumulate read is the same pair.
#define NNOPS_SME_STORE_WIDE_ROW_F32(v0_, v1_, row_)                           \
    do {                                                                       \
        svfloat32_t nno_v0 = v0_;                                              \
        svfloat32_t nno_v1 = v1_;                                              \
        if (accumulate) {                                                      \
            const svfloat32x2_t nno_c = svld1_f32_x2(svptrue_c32(), row_);     \
            nno_v0 = svadd_f32_x(pn0, nno_v0, svget2_f32(nno_c, 0));           \
            nno_v1 = svadd_f32_x(pn1, nno_v1, svget2_f32(nno_c, 1));           \
        }                                                                      \
        nno_v0 = sme_detail::clamp_v(pn0, nno_v0, vlo, vhi);                   \
        nno_v1 = sme_detail::clamp_v(pn1, nno_v1, vlo, vhi);                   \
        svst1_f32_x2(svptrue_c32(), row_, svcreate2_f32(nno_v0, nno_v1));      \
    } while (0)

__arm_locally_streaming __arm_new("za") __attribute__((target("+sme2")))
inline void mma_f32_sme(int mc, int nc, int kc, float* c, int ldc, const float* a,
                        const float* b, bool zero_mode, float lo, float hi,
                        const float* col_bias, int64_t b_block_stride = 0) noexcept
{
    constexpr int kRows = 2 * sme_mr;   // 32 rows: two A groups
    constexpr int kCols = 2 * sme_za_n; // 32 columns: two B groups
    const int m_tiles = (mc + kRows - 1) / kRows;
    const int n_tiles = (nc + kCols - 1) / kCols;
    const int64_t block_stride = b_block_stride != 0 ? b_block_stride :
                                                       static_cast<int64_t>(kc) * sme_fp_nr;
    const svfloat32_t vlo = svdup_n_f32(lo);
    const svfloat32_t vhi = svdup_n_f32(hi);
    // A bias seed builds the tile from the bias; the destination's old value
    // is not part of the sum, so the store overwrites even when zero_mode
    // is off.
    const bool accumulate = !zero_mode && col_bias == nullptr;

    // A single 16-row tile: the four ZA tiles are four column groups, so one
    // K step is one A vector, two B pair loads and four MOPAs. With only two
    // tiles the same K step would issue two MOPAs, and the loop overhead —
    // not the MOPA — became the limit (half the throughput).
    if (mc <= sme_mr && col_bias == nullptr && nc % (2 * kCols) == 0) {
        const svbool_t pm = svwhilelt_b32(0, mc);
        const svbool_t pg = svptrue_b32();
        // The wide-store macro names the column predicates; here every column
        // of both tiles is live.
        const svbool_t pn0 = pg;
        const svbool_t pn1 = pg;
        const svcount_t p32 = svptrue_c32();
        const float* const ap = a;
        for (int nt = 0; nt < n_tiles; nt += 2) {
            const int na = nt * kCols;
            const int nb = na + kCols;
            const float* const bpa = b + static_cast<int64_t>(na / sme_fp_nr) * block_stride;
            const float* const bpb = b + static_cast<int64_t>(nb / sme_fp_nr) * block_stride;
            svzero_za();
            const float* arow = ap;
            const float* browa = bpa;
            const float* browb = bpb;
            int k = 0;
            for (; k + 4 <= kc; k += 4) {
                for (int u = 0; u < 4; ++u) {
                    const float* const ak = arow + static_cast<int64_t>(u) * kRows;
                    const float* const ba = browa + u * sme_fp_nr;
                    const float* const bb = browb + u * sme_fp_nr;
                    const svfloat32_t av = sme_detail::load_op(pm, ak);
                    const svfloat32x2_t bva = svld1_f32_x2(p32, ba);
                    const svfloat32x2_t bvb = svld1_f32_x2(p32, bb);
                    svmopa_za32_f32_m(0, pm, pg, av, svget2_f32(bva, 0));
                    svmopa_za32_f32_m(2, pm, pg, av, svget2_f32(bva, 1));
                    svmopa_za32_f32_m(1, pm, pg, av, svget2_f32(bvb, 0));
                    svmopa_za32_f32_m(3, pm, pg, av, svget2_f32(bvb, 1));
                }
                arow += 4 * kRows;
                browa += 4 * sme_fp_nr;
                browb += 4 * sme_fp_nr;
            }
            for (; k < kc; ++k) {
                const svfloat32_t av = sme_detail::load_op(pm, arow);
                const svfloat32x2_t bva = svld1_f32_x2(p32, browa);
                const svfloat32x2_t bvb = svld1_f32_x2(p32, browb);
                svmopa_za32_f32_m(0, pm, pg, av, svget2_f32(bva, 0));
                svmopa_za32_f32_m(2, pm, pg, av, svget2_f32(bva, 1));
                svmopa_za32_f32_m(1, pm, pg, av, svget2_f32(bvb, 0));
                svmopa_za32_f32_m(3, pm, pg, av, svget2_f32(bvb, 1));
                arow += kRows;
                browa += sme_fp_nr;
                browb += sme_fp_nr;
            }
            for (int r = 0; r < mc; ++r) {
                float* const rowa = c + static_cast<int64_t>(r) * ldc + na;
                NNOPS_SME_STORE_WIDE_ROW_F32(
                    svread_hor_za32_f32_m(svundef_f32(), pg, 0, r),
                    svread_hor_za32_f32_m(svundef_f32(), pg, 2, r), rowa);
                float* const rowb = c + static_cast<int64_t>(r) * ldc + nb;
                NNOPS_SME_STORE_WIDE_ROW_F32(
                    svread_hor_za32_f32_m(svundef_f32(), pg, 1, r),
                    svread_hor_za32_f32_m(svundef_f32(), pg, 3, r), rowb);
            }
        }
        return;
    }

    // The n tile is the outer loop: its 32 columns per k stay in L1 across all
    // M tiles, while an M tile is only 32 rows.
    for (int nt = 0; nt < n_tiles; ++nt) {
        const int n0 = nt * kCols;
        const int nlive = std::min(kCols, nc - n0);
        const bool t1 = nlive > sme_za_n;
        const svbool_t pn0 = svwhilelt_b32(0, std::min(sme_za_n, nlive));
        const svbool_t pn1 = t1 ? svwhilelt_b32(0, nlive - sme_za_n) : svpfalse_b();
        // The 32-column B block is exactly this n tile.
        const float* const bp = b + static_cast<int64_t>(n0 / sme_fp_nr) * block_stride;
        const float* const bias0 = col_bias != nullptr ? col_bias + n0 : nullptr;

        for (int mt = 0; mt < m_tiles; ++mt) {
            const int m0 = mt * kRows;
            const int mlive = std::min(kRows, mc - m0);
            const bool r1 = mlive > sme_mr;
            const svbool_t pm0 = svwhilelt_b32(0, std::min(sme_mr, mlive));
            const svbool_t pm1 = r1 ? svwhilelt_b32(0, mlive - sme_mr) : svpfalse_b();
            const float* const ap = a + static_cast<int64_t>(m0) * kc;

            svzero_za();
            if (bias0 != nullptr) {
                const svfloat32_t bv0 = sme_detail::load_op(pn0, bias0);
                const svfloat32_t bv1 =
                    t1 ? sme_detail::load_op(pn1, bias0 + sme_za_n) : svdup_n_f32(0.0f);
                // Tiles 0/2 are the low row group and 1/3 the high one;
                // 0/1 are the low column group and 2/3 the high one. The ZA
                // tile index is an immediate, hence the split.
                const int lo_rows = std::min(sme_mr, mlive);
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

            const bool full = mlive == kRows;
            const float* brow = bp;
            int k = 0;
            if (!r1) {
                // At most 16 live rows: two ZA tiles carry the tile.
                const svcount_t p32 = svptrue_c32();
                for (; k + 4 <= kc; k += 4) {
                    const float* const ak = ap + static_cast<int64_t>(k) * kRows;
                    const svfloat32_t a0 = sme_detail::load_op(pm0, ak);
                    const svfloat32_t a1 = sme_detail::load_op(pm0, ak + kRows);
                    const svfloat32_t a2 = sme_detail::load_op(pm0, ak + 2 * kRows);
                    const svfloat32_t a3 = sme_detail::load_op(pm0, ak + 3 * kRows);
                    const float* const bk = brow;
                    NNOPS_SME_MOPA2_F32(pm0, pn0, pn1, a0, svld1_f32_x2(p32, bk));
                    NNOPS_SME_MOPA2_F32(pm0, pn0, pn1, a1,
                                        svld1_f32_x2(p32, bk + sme_fp_nr));
                    NNOPS_SME_MOPA2_F32(pm0, pn0, pn1, a2,
                                        svld1_f32_x2(p32, bk + 2 * sme_fp_nr));
                    NNOPS_SME_MOPA2_F32(pm0, pn0, pn1, a3,
                                        svld1_f32_x2(p32, bk + 3 * sme_fp_nr));
                    brow += 4 * sme_fp_nr;
                }
                for (; k < kc; ++k, brow += sme_fp_nr) {
                    NNOPS_SME_MOPA2_F32(
                        pm0, pn0, pn1,
                        sme_detail::load_op(pm0, ap + static_cast<int64_t>(k) * kRows),
                        svld1_f32_x2(p32, brow));
                }
            } else if (full) {
                // The 32-row A block is one pair load, B's 32 columns the
                // other. Unrolled four K steps: sixteen MOPA between eight
                // pair loads, so the loads of the next step are in flight
                // while this one issues, as in KleidiAI's SME2 kernel.
                const svcount_t p32 = svptrue_c32();
                for (; k + 4 <= kc; k += 4) {
                    const float* const ak = ap + static_cast<int64_t>(k) * kRows;
                    const svfloat32x2_t a0 = svld1_f32_x2(p32, ak);
                    const svfloat32x2_t a1 = svld1_f32_x2(p32, ak + kRows);
                    const svfloat32x2_t a2 = svld1_f32_x2(p32, ak + 2 * kRows);
                    const svfloat32x2_t a3 = svld1_f32_x2(p32, ak + 3 * kRows);
                    const float* const bk = brow;
                    NNOPS_SME_MOPA4_F32(pm0, pm1, pn0, pn1, a0,
                                          svld1_f32_x2(p32, bk));
                    NNOPS_SME_MOPA4_F32(pm0, pm1, pn0, pn1, a1,
                                          svld1_f32_x2(p32, bk + sme_fp_nr));
                    NNOPS_SME_MOPA4_F32(pm0, pm1, pn0, pn1, a2,
                                          svld1_f32_x2(p32, bk + 2 * sme_fp_nr));
                    NNOPS_SME_MOPA4_F32(pm0, pm1, pn0, pn1, a3,
                                          svld1_f32_x2(p32, bk + 3 * sme_fp_nr));
                    brow += 4 * sme_fp_nr;
                }
                for (; k < kc; ++k, brow += sme_fp_nr) {
                    NNOPS_SME_MOPA4_F32(
                        pm0, pm1, pn0, pn1,
                        svld1_f32_x2(p32, ap + static_cast<int64_t>(k) * kRows),
                        svld1_f32_x2(p32, brow));
                }
            } else {
                // Same four-K unroll, A read under the row predicates: a
                // short M tile is the whole product when M is under 32, so it
                // cannot afford the un-unrolled loop.
                const svcount_t p32 = svptrue_c32();
                const svfloat32_t vzero = svdup_n_f32(0.0f);
                for (; k + 4 <= kc; k += 4) {
                    const float* const ak = ap + static_cast<int64_t>(k) * kRows;
                    const svfloat32x2_t a0 = svcreate2_f32(
                        sme_detail::load_op(pm0, ak),
                        r1 ? sme_detail::load_op(pm1, ak + sme_mr) : vzero);
                    const svfloat32x2_t a1 = svcreate2_f32(
                        sme_detail::load_op(pm0, ak + kRows),
                        r1 ? sme_detail::load_op(pm1, ak + kRows + sme_mr) : vzero);
                    const svfloat32x2_t a2 = svcreate2_f32(
                        sme_detail::load_op(pm0, ak + 2 * kRows),
                        r1 ? sme_detail::load_op(pm1, ak + 2 * kRows + sme_mr) : vzero);
                    const svfloat32x2_t a3 = svcreate2_f32(
                        sme_detail::load_op(pm0, ak + 3 * kRows),
                        r1 ? sme_detail::load_op(pm1, ak + 3 * kRows + sme_mr) : vzero);
                    const float* const bk = brow;
                    NNOPS_SME_MOPA4_F32(pm0, pm1, pn0, pn1, a0,
                                        svld1_f32_x2(p32, bk));
                    NNOPS_SME_MOPA4_F32(pm0, pm1, pn0, pn1, a1,
                                        svld1_f32_x2(p32, bk + sme_fp_nr));
                    NNOPS_SME_MOPA4_F32(pm0, pm1, pn0, pn1, a2,
                                        svld1_f32_x2(p32, bk + 2 * sme_fp_nr));
                    NNOPS_SME_MOPA4_F32(pm0, pm1, pn0, pn1, a3,
                                        svld1_f32_x2(p32, bk + 3 * sme_fp_nr));
                    brow += 4 * sme_fp_nr;
                }
                for (; k < kc; ++k, brow += sme_fp_nr) {
                    const float* const ak = ap + static_cast<int64_t>(k) * kRows;
                    NNOPS_SME_MOPA4_F32(
                        pm0, pm1, pn0, pn1,
                        svcreate2_f32(sme_detail::load_op(pm0, ak),
                                      r1 ? sme_detail::load_op(pm1, ak + sme_mr) : vzero),
                        svld1_f32_x2(p32, brow));
                }
            }

            const int lo_rows = std::min(sme_mr, mlive);
            // A full-width row is 32 f32 = 128 B = one cache line, so both
            // column groups go out as one pair store and, when accumulating,
            // come in as one pair load. Two 64-byte halves are partial-line
            // writes, and the read-back after them costs about as much as the
            // whole K loop.
            const bool wide = t1 && nlive == kCols;
            if (wide) {
                // Four ZA rows per MOVA: the horizontal read is the latency in
                // this loop, so a quarter of them is a quarter of the waits.
                int r = 0;
                for (; r + 4 <= lo_rows; r += 4) {
                    const svfloat32x4_t g0 = svread_hor_za32_f32_vg4(0, r);
                    const svfloat32x4_t g2 = svread_hor_za32_f32_vg4(2, r);
                    float* row = c + static_cast<int64_t>(m0 + r) * ldc + n0;
                    NNOPS_SME_STORE_WIDE_ROW_F32(svget4_f32(g0, 0), svget4_f32(g2, 0),
                                                 row);
                    NNOPS_SME_STORE_WIDE_ROW_F32(svget4_f32(g0, 1), svget4_f32(g2, 1),
                                                 row + ldc);
                    NNOPS_SME_STORE_WIDE_ROW_F32(svget4_f32(g0, 2), svget4_f32(g2, 2),
                                                 row + 2 * ldc);
                    NNOPS_SME_STORE_WIDE_ROW_F32(svget4_f32(g0, 3), svget4_f32(g2, 3),
                                                 row + 3 * ldc);
                }
                for (; r < lo_rows; ++r) {
                    float* const row = c + static_cast<int64_t>(m0 + r) * ldc + n0;
                    NNOPS_SME_STORE_WIDE_ROW_F32(
                        svread_hor_za32_f32_m(svundef_f32(), pn0, 0, r),
                        svread_hor_za32_f32_m(svundef_f32(), pn1, 2, r), row);
                }
                const int hi_rows = mlive - sme_mr;
                for (r = 0; r + 4 <= hi_rows; r += 4) {
                    const svfloat32x4_t g1 = svread_hor_za32_f32_vg4(1, r);
                    const svfloat32x4_t g3 = svread_hor_za32_f32_vg4(3, r);
                    float* row = c + static_cast<int64_t>(m0 + sme_mr + r) * ldc + n0;
                    NNOPS_SME_STORE_WIDE_ROW_F32(svget4_f32(g1, 0), svget4_f32(g3, 0),
                                                 row);
                    NNOPS_SME_STORE_WIDE_ROW_F32(svget4_f32(g1, 1), svget4_f32(g3, 1),
                                                 row + ldc);
                    NNOPS_SME_STORE_WIDE_ROW_F32(svget4_f32(g1, 2), svget4_f32(g3, 2),
                                                 row + 2 * ldc);
                    NNOPS_SME_STORE_WIDE_ROW_F32(svget4_f32(g1, 3), svget4_f32(g3, 3),
                                                 row + 3 * ldc);
                }
                for (; r < hi_rows; ++r) {
                    float* const row =
                        c + static_cast<int64_t>(m0 + sme_mr + r) * ldc + n0;
                    NNOPS_SME_STORE_WIDE_ROW_F32(
                        svread_hor_za32_f32_m(svundef_f32(), pn0, 1, r),
                        svread_hor_za32_f32_m(svundef_f32(), pn1, 3, r), row);
                }
            } else {
                for (int r = 0; r < lo_rows; ++r) {
                    float* const row = c + static_cast<int64_t>(m0 + r) * ldc + n0;
                    {
                        svfloat32_t v = svread_hor_za32_f32_m(svundef_f32(), pn0, 0, r);
                        if (accumulate) { v = svadd_f32_x(pn0, v, svld1(pn0, row)); }
                        svst1(pn0, row, sme_detail::clamp_v(pn0, v, vlo, vhi));
                    }
                    if (t1) {
                        float* const p = row + sme_za_n;
                        svfloat32_t v = svread_hor_za32_f32_m(svundef_f32(), pn1, 2, r);
                        if (accumulate) { v = svadd_f32_x(pn1, v, svld1(pn1, p)); }
                        svst1(pn1, p, sme_detail::clamp_v(pn1, v, vlo, vhi));
                    }
                }
                for (int r = 0; r < mlive - sme_mr; ++r) {
                    float* const row =
                        c + static_cast<int64_t>(m0 + sme_mr + r) * ldc + n0;
                    {
                        svfloat32_t v = svread_hor_za32_f32_m(svundef_f32(), pn0, 1, r);
                        if (accumulate) { v = svadd_f32_x(pn0, v, svld1(pn0, row)); }
                        svst1(pn0, row, sme_detail::clamp_v(pn0, v, vlo, vhi));
                    }
                    if (t1) {
                        float* const p = row + sme_za_n;
                        svfloat32_t v = svread_hor_za32_f32_m(svundef_f32(), pn1, 3, r);
                        if (accumulate) { v = svadd_f32_x(pn1, v, svld1(pn1, p)); }
                        svst1(pn1, p, sme_detail::clamp_v(pn1, v, vlo, vhi));
                    }
                }
            }
        }
    }
}

} // namespace nnops::backend::cpu::aarch64

#undef NNOPS_SME_MOPA4_F32
#undef NNOPS_SME_MOPA2_F32
#undef NNOPS_SME_STORE_WIDE_ROW_F32

#endif // NNOPS_ARCH_SME
