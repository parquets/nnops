#pragma once
/// @file mma_f32_sme.hpp
/// @brief AArch64 SME float32 mrxncxkc MMA kernel.
///
/// C[m][n] += A[m][k] * B[k][n] for one k-block, mc x nc x kc. Four ZA tiles
/// share one K loop, so a k-step is four independent `FMOPA ZA.S` instructions
/// (one per 16-column panel) and the pipeline never waits on one tile's
/// dependency chain. The B block's 64 columns per k are one contiguous run, so
/// each k is two cache lines rather than four scattered streams.
///
/// @p a and @p b are the packed layouts pack_f32_sme.hpp writes. @p zero_mode
/// overwrites C instead of accumulating; @p col_bias (per output column) seeds
/// the accumulator and overwrites C. Every stored element is clamped to
/// [@p lo, @p hi].
///
/// @p b_block_stride is the element distance between 64-column blocks: a
/// per-call pack lays them kc*64 apart, a whole-K prepacked blob K*64. 0 means
/// the per-call form.

#if defined(NNOPS_ARCH_SME)

#include <algorithm>
#include <cstdint>

#include <arm_sme.h>

#include "backend/cpu/aarch64/mma_sme.hpp"

namespace nnops::backend::cpu::aarch64 {

namespace sme_detail {

__arm_locally_streaming __attribute__((target("+sme2"), always_inline))
inline svfloat32_t load_op(svbool_t pg, const float* p) noexcept
{
    return svld1(pg, p);
}

} // namespace sme_detail

__arm_locally_streaming __arm_new("za") __attribute__((target("+sme2")))
inline void mma_f32_sme(int mc, int nc, int kc, float* c, int ldc, const float* a,
                         const float* b, bool zero_mode, float lo, float hi,
                         const float* col_bias, int64_t b_block_stride = 0) noexcept
{
    const int m_tiles = (mc + sme_mr - 1) / sme_mr;
    const int n_blocks = (nc + sme_nr_max - 1) / sme_nr_max;
    const svfloat32_t vlo = svdup_n_f32(lo);
    const svfloat32_t vhi = svdup_n_f32(hi);
    // The n block is the outer loop. Its 64 columns per k are one contiguous
    // 256-byte run, small enough to stay in L1 across every M tile, whereas an
    // M tile is only 16 rows: the other order re-reads the whole B slab once
    // per tile, four times the traffic for the same work.
    for (int nb = 0; nb < n_blocks; ++nb) {
        const int n0 = nb * sme_nr_max;
        const bool t1 = n0 + sme_za_n < nc;
        const bool t2 = n0 + 2 * sme_za_n < nc;
        const bool t3 = n0 + 3 * sme_za_n < nc;
        const svbool_t pn0 = svwhilelt_b32(0, std::min(sme_za_n, nc - n0));
        const svbool_t pn1 =
            t1 ? svwhilelt_b32(0, std::min(sme_za_n, nc - n0 - sme_za_n)) : svpfalse_b();
        const svbool_t pn2 =
            t2 ? svwhilelt_b32(0, std::min(sme_za_n, nc - n0 - 2 * sme_za_n)) : svpfalse_b();
        const svbool_t pn3 =
            t3 ? svwhilelt_b32(0, std::min(sme_za_n, nc - n0 - 3 * sme_za_n)) : svpfalse_b();
        const float* bp =
            b + static_cast<int64_t>(nb) * (b_block_stride != 0 ?
                                                b_block_stride :
                                                static_cast<int64_t>(kc) * sme_nr_max);
        const float* bias0 = col_bias != nullptr ? col_bias + n0 : nullptr;

        for (int m0 = 0; m0 < m_tiles; ++m0) {
            const int mlive = std::min(sme_mr, mc - m0 * sme_mr);
            const svbool_t pm = svwhilelt_b32(0, mlive);
            const float* ap = a + static_cast<int64_t>(m0) * kc * sme_mr;

            svzero_za();
            if (col_bias != nullptr) {
                const svfloat32_t bv0 = sme_detail::load_op(pn0, bias0);
                for (int i = 0; i < mlive; ++i) { svwrite_hor_za32_f32_m(0, i, pn0, bv0); }
                if (t1) {
                    const svfloat32_t bv1 = sme_detail::load_op(pn1, bias0 + sme_za_n);
                    for (int i = 0; i < mlive; ++i) { svwrite_hor_za32_f32_m(1, i, pn1, bv1); }
                }
                if (t2) {
                    const svfloat32_t bv2 = sme_detail::load_op(pn2, bias0 + 2 * sme_za_n);
                    for (int i = 0; i < mlive; ++i) { svwrite_hor_za32_f32_m(2, i, pn2, bv2); }
                }
                if (t3) {
                    const svfloat32_t bv3 = sme_detail::load_op(pn3, bias0 + 3 * sme_za_n);
                    for (int i = 0; i < mlive; ++i) { svwrite_hor_za32_f32_m(3, i, pn3, bv3); }
                }
            }

            const float* brow = bp;
            for (int k = 0; k < kc; ++k, brow += sme_nr_max) {
                const svfloat32_t av =
                    sme_detail::load_op(pm, ap + static_cast<int64_t>(k) * sme_mr);
                svmopa_za32_f32_m(0, pm, pn0, av, sme_detail::load_op(pn0, brow));
                svmopa_za32_f32_m(1, pm, pn1, av, sme_detail::load_op(pn1, brow + sme_za_n));
                svmopa_za32_f32_m(2, pm, pn2, av, sme_detail::load_op(pn2, brow + 2 * sme_za_n));
                svmopa_za32_f32_m(3, pm, pn3, av, sme_detail::load_op(pn3, brow + 3 * sme_za_n));
            }

            for (int i = 0; i < mlive; ++i) {
                float* row = c + static_cast<int64_t>(m0 * sme_mr + i) * ldc + n0;
                {
                    svfloat32_t v = svread_hor_za32_f32_m(svundef_f32(), pn0, 0, i);
                    if (!zero_mode) { v = svadd_f32_x(pn0, v, svld1(pn0, row)); }
                    svst1(pn0, row, sme_detail::clamp_v(pn0, v, vlo, vhi));
                }
                if (t1) {
                    float* p = row + sme_za_n;
                    svfloat32_t v = svread_hor_za32_f32_m(svundef_f32(), pn1, 1, i);
                    if (!zero_mode) { v = svadd_f32_x(pn1, v, svld1(pn1, p)); }
                    svst1(pn1, p, sme_detail::clamp_v(pn1, v, vlo, vhi));
                }
                if (t2) {
                    float* p = row + 2 * sme_za_n;
                    svfloat32_t v = svread_hor_za32_f32_m(svundef_f32(), pn2, 2, i);
                    if (!zero_mode) { v = svadd_f32_x(pn2, v, svld1(pn2, p)); }
                    svst1(pn2, p, sme_detail::clamp_v(pn2, v, vlo, vhi));
                }
                if (t3) {
                    float* p = row + 3 * sme_za_n;
                    svfloat32_t v = svread_hor_za32_f32_m(svundef_f32(), pn3, 3, i);
                    if (!zero_mode) { v = svadd_f32_x(pn3, v, svld1(pn3, p)); }
                    svst1(pn3, p, sme_detail::clamp_v(pn3, v, vlo, vhi));
                }
            }
        }
    }
}

} // namespace nnops::backend::cpu::aarch64

#endif // NNOPS_ARCH_SME
