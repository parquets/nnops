#pragma once
/// @file mma_f16_sme.hpp
/// @brief AArch64 SME float16 mrxncxkc MMA kernel with f32 accumulation.
///
/// The hardware two-way dot (`FMOPA ZA.S` with `.H` operands) consumes two K
/// per instruction, so a K step is one 64-byte load per operand and one MOPA,
/// the same shape as the f32 loop. The operands are pre-interleaved at pack
/// time (pack_f16_sme.hpp): A at [row][k,k+1], B at [col][k,k+1].
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
/// @p b_block_stride is the element distance between 64-column blocks: a
/// per-call pack lays them ceil(kc/2)*128 apart, a whole-K prepacked blob
/// ceil(K/2)*128. 0 means the per-call form.

#if defined(NNOPS_ARCH_SME)

#include <algorithm>
#include <cstdint>

#include <arm_neon.h>
#include <arm_sme.h>

#include "nnops/detail/half.hpp"

#include "backend/cpu/aarch64/mma_sme.hpp"

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

/// The hardware two-way dot (`FMOPA ZA.S` with `.H` operands) consumes two K
/// per instruction. The pack already interleaved them — A at [row][k,k+1] and
/// B at [col][k,k+1] — so a K step is one 64-byte load per operand and one
/// MOPA, the same shape as the f32 loop. Interleaving here instead would cost
/// five permutes per K pair that contend with the MOPA issue slot.
///
/// The MOPA predicates are **b16**: the operand's lane 2i and 2i+1 are entry
/// i's two K values, so a row is live when both lanes are. A b32 predicate
/// activates only the even lanes, which silently drops half the products.
__arm_locally_streaming __arm_new("za") __attribute__((target("+sme2")))
inline void mma_f16_sme(int mc, int nc, int kc, half* c, int ldc, const half* a,
                         const half* b, bool zero_mode, float lo, float hi,
                         const half* col_bias, int64_t b_block_stride = 0) noexcept
{
    const int m_tiles = (mc + sme_mr - 1) / sme_mr;
    const int n_blocks = (nc + sme_nr_max - 1) / sme_nr_max;
    const int k_pairs = (kc + 1) / 2;
    const svfloat32_t vlo = svdup_n_f32(lo);
    const svfloat32_t vhi = svdup_n_f32(hi);
    // Same loop order as the f32 kernel: n block outside, M tiles inside.
    for (int nb = 0; nb < n_blocks; ++nb) {
        const int n0 = nb * sme_nr_max;
        const bool t1 = n0 + sme_za_n < nc;
        const bool t2 = n0 + 2 * sme_za_n < nc;
        const bool t3 = n0 + 3 * sme_za_n < nc;
        const int live0 = std::min(sme_za_n, nc - n0);
        const int live1 = t1 ? std::min(sme_za_n, nc - n0 - sme_za_n) : 0;
        const int live2 = t2 ? std::min(sme_za_n, nc - n0 - 2 * sme_za_n) : 0;
        const int live3 = t3 ? std::min(sme_za_n, nc - n0 - 3 * sme_za_n) : 0;
        const svbool_t pn0 = svwhilelt_b32(0, live0);
        const svbool_t pn1 = t1 ? svwhilelt_b32(0, live1) : svpfalse_b();
        const svbool_t pn2 = t2 ? svwhilelt_b32(0, live2) : svpfalse_b();
        const svbool_t pn3 = t3 ? svwhilelt_b32(0, live3) : svpfalse_b();
        const svbool_t qn0 = svwhilelt_b16(0, 2 * live0);
        const svbool_t qn1 = t1 ? svwhilelt_b16(0, 2 * live1) : svpfalse_b();
        const svbool_t qn2 = t2 ? svwhilelt_b16(0, 2 * live2) : svpfalse_b();
        const svbool_t qn3 = t3 ? svwhilelt_b16(0, 2 * live3) : svpfalse_b();
        const half* bp =
            b + static_cast<int64_t>(nb) * (b_block_stride != 0 ?
                                                b_block_stride :
                                                static_cast<int64_t>(k_pairs) * (2 * sme_nr_max));
        const half* bias0 = col_bias != nullptr ? col_bias + n0 : nullptr;

        for (int m0 = 0; m0 < m_tiles; ++m0) {
            const int mlive = std::min(sme_mr, mc - m0 * sme_mr);
            const svbool_t pmh = svwhilelt_b16(0, 2 * mlive);
            const half* ap = a + static_cast<int64_t>(m0) * k_pairs * (2 * sme_mr);

            svzero_za();
            if (col_bias != nullptr) {
                const svfloat32_t bv0 =
                    sme_detail::widen_h16(svld1_f16(svwhilelt_b16(0, live0), bias0));
                for (int i = 0; i < mlive; ++i) { svwrite_hor_za32_f32_m(0, i, pn0, bv0); }
                if (t1) {
                    const svfloat32_t bv1 = sme_detail::widen_h16(
                        svld1_f16(svwhilelt_b16(0, live1), bias0 + sme_za_n));
                    for (int i = 0; i < mlive; ++i) { svwrite_hor_za32_f32_m(1, i, pn1, bv1); }
                }
                if (t2) {
                    const svfloat32_t bv2 = sme_detail::widen_h16(
                        svld1_f16(svwhilelt_b16(0, live2), bias0 + 2 * sme_za_n));
                    for (int i = 0; i < mlive; ++i) { svwrite_hor_za32_f32_m(2, i, pn2, bv2); }
                }
                if (t3) {
                    const svfloat32_t bv3 = sme_detail::widen_h16(
                        svld1_f16(svwhilelt_b16(0, live3), bias0 + 3 * sme_za_n));
                    for (int i = 0; i < mlive; ++i) { svwrite_hor_za32_f32_m(3, i, pn3, bv3); }
                }
            }

            const half* arow = ap;
            const half* brow = bp;
            for (int kp = 0; kp < k_pairs; ++kp, arow += 2 * sme_mr, brow += 2 * sme_nr_max) {
                const svfloat16_t av = sme_detail::load_h32(pmh, arow);
                svmopa_za32_f16_m(0, pmh, qn0, av, sme_detail::load_h32(qn0, brow));
                svmopa_za32_f16_m(1, pmh, qn1, av,
                                  sme_detail::load_h32(qn1, brow + 2 * sme_za_n));
                svmopa_za32_f16_m(2, pmh, qn2, av,
                                  sme_detail::load_h32(qn2, brow + 4 * sme_za_n));
                svmopa_za32_f16_m(3, pmh, qn3, av,
                                  sme_detail::load_h32(qn3, brow + 6 * sme_za_n));
            }

            for (int i = 0; i < mlive; ++i) {
                half* row = c + static_cast<int64_t>(m0 * sme_mr + i) * ldc + n0;
                {
                    svfloat32_t v = svread_hor_za32_f32_m(svundef_f32(), pn0, 0, i);
                    if (!zero_mode) {
                        v = svadd_f32_x(
                            pn0, v,
                            sme_detail::widen_h16(svld1_f16(svwhilelt_b16(0, live0), row)));
                    }
                    sme_detail::store_op(row, pn0, sme_detail::clamp_v(pn0, v, vlo, vhi),
                                         live0);
                }
                if (t1) {
                    half* p = row + sme_za_n;
                    svfloat32_t w = svread_hor_za32_f32_m(svundef_f32(), pn1, 1, i);
                    if (!zero_mode) {
                        w = svadd_f32_x(
                            pn1, w,
                            sme_detail::widen_h16(svld1_f16(svwhilelt_b16(0, live1), p)));
                    }
                    sme_detail::store_op(p, pn1, sme_detail::clamp_v(pn1, w, vlo, vhi), live1);
                }
                if (t2) {
                    half* p = row + 2 * sme_za_n;
                    svfloat32_t w = svread_hor_za32_f32_m(svundef_f32(), pn2, 2, i);
                    if (!zero_mode) {
                        w = svadd_f32_x(
                            pn2, w,
                            sme_detail::widen_h16(svld1_f16(svwhilelt_b16(0, live2), p)));
                    }
                    sme_detail::store_op(p, pn2, sme_detail::clamp_v(pn2, w, vlo, vhi), live2);
                }
                if (t3) {
                    half* p = row + 3 * sme_za_n;
                    svfloat32_t w = svread_hor_za32_f32_m(svundef_f32(), pn3, 3, i);
                    if (!zero_mode) {
                        w = svadd_f32_x(
                            pn3, w,
                            sme_detail::widen_h16(svld1_f16(svwhilelt_b16(0, live3), p)));
                    }
                    sme_detail::store_op(p, pn3, sme_detail::clamp_v(pn3, w, vlo, vhi), live3);
                }
            }
        }
    }
}

} // namespace nnops::backend::cpu::aarch64

#endif // NNOPS_ARCH_SME
