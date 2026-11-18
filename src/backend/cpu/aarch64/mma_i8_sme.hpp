#pragma once
/// @file mma_i8_sme.hpp
/// @brief AArch64 SME int8 (s8 x s8 -> int32) mrxncxkc MMA kernel.
///
/// One `SMOPA ZA.S` is a 4-way dot: 16 rows x 16 columns x 4 K = 1024 MACs,
/// four times the f32 form. The pack interleaves K in groups of four, which is
/// how the instruction reads its lanes: operand lane 4i..4i+3 are row (column)
/// i's four K values.
///
/// Because the lane groups are the tile rows, the MOPA predicate is b8 and the
/// row extent is 4*mlive lanes — a b32 predicate would leave most of the tile
/// inactive and silently drop products.
///
/// @p a and @p b are the packed layouts pack_i8_sme.hpp writes. The accumulator
/// is int32, and every stored element is clamped to [@p lo, @p hi]. @p zero_mode
/// overwrites C instead of accumulating, which is what lets the int8 route's
/// epilogue keep doing its zero-point compensation and requantize unchanged.
///
/// @p b_block_stride is the element distance between 64-column blocks: a
/// per-call pack lays them ceil(kc/4)*256 apart, a whole-K prepacked blob
/// ceil(K/4)*256. 0 means the per-call form.

#if defined(NNOPS_ARCH_SME)

#include <algorithm>
#include <cstdint>

#include <arm_sme.h>

#include "backend/cpu/aarch64/mma_sme.hpp"

namespace nnops::backend::cpu::aarch64 {

/// One SMOPA per K group of four per tile. The lane groups are the tile rows (a
/// 4-way dot over the group's four K), so the predicate is b8 and the row
/// extent is 4*mlive lanes — a b32 predicate would leave most of the tile
/// inactive and silently drop products.
__arm_locally_streaming __arm_new("za") __attribute__((target("+sme2")))
inline void mma_i8_sme(int mc, int nc, int kc, int32_t* c, int ldc, const int8_t* a,
                        const int8_t* b, bool zero_mode, int32_t lo, int32_t hi,
                        int64_t b_block_stride = 0) noexcept
{
    const int m_tiles = (mc + sme_mr - 1) / sme_mr;
    const int n_blocks = (nc + sme_nr_max - 1) / sme_nr_max;
    const int k_groups = (kc + 3) / 4;
    const svint32_t vlo = svdup_n_s32(lo);
    const svint32_t vhi = svdup_n_s32(hi);
    for (int nb = 0; nb < n_blocks; ++nb) {
        const int n0 = nb * sme_nr_max;
        const bool t1 = n0 + sme_za_n < nc;
        const bool t2 = n0 + 2 * sme_za_n < nc;
        const bool t3 = n0 + 3 * sme_za_n < nc;
        const int live0 = std::min(sme_za_n, nc - n0);
        const int live1 = t1 ? std::min(sme_za_n, nc - n0 - sme_za_n) : 0;
        const int live2 = t2 ? std::min(sme_za_n, nc - n0 - 2 * sme_za_n) : 0;
        const int live3 = t3 ? std::min(sme_za_n, nc - n0 - 3 * sme_za_n) : 0;
        const svbool_t pn0 = svwhilelt_b8(0, 4 * live0);
        const svbool_t pn1 = t1 ? svwhilelt_b8(0, 4 * live1) : svpfalse_b();
        const svbool_t pn2 = t2 ? svwhilelt_b8(0, 4 * live2) : svpfalse_b();
        const svbool_t pn3 = t3 ? svwhilelt_b8(0, 4 * live3) : svpfalse_b();
        const svbool_t s0 = svwhilelt_b32(0, live0);
        const svbool_t s1 = t1 ? svwhilelt_b32(0, live1) : svpfalse_b();
        const svbool_t s2 = t2 ? svwhilelt_b32(0, live2) : svpfalse_b();
        const svbool_t s3 = t3 ? svwhilelt_b32(0, live3) : svpfalse_b();
        const int8_t* bp =
            b + static_cast<int64_t>(nb) * (b_block_stride != 0 ?
                                                b_block_stride :
                                                static_cast<int64_t>(k_groups) * (4 * sme_nr_max));

        for (int m0 = 0; m0 < m_tiles; ++m0) {
            const int mlive = std::min(sme_mr, mc - m0 * sme_mr);
            const svbool_t pm = svwhilelt_b8(0, 4 * mlive);
            const int8_t* ap = a + static_cast<int64_t>(m0) * k_groups * 64;

            svzero_za();
            const int8_t* brow = bp;
            for (int g = 0; g < k_groups; ++g, brow += 4 * sme_nr_max) {
                const svint8_t av = svld1_s8(pm, ap + static_cast<int64_t>(g) * 64);
                svmopa_za32_s8_m(0, pm, pn0, av, svld1_s8(pn0, brow));
                svmopa_za32_s8_m(1, pm, pn1, av, svld1_s8(pn1, brow + sme_za_n * 4));
                svmopa_za32_s8_m(2, pm, pn2, av, svld1_s8(pn2, brow + 2 * sme_za_n * 4));
                svmopa_za32_s8_m(3, pm, pn3, av, svld1_s8(pn3, brow + 3 * sme_za_n * 4));
            }

            for (int i = 0; i < mlive; ++i) {
                int32_t* row = c + static_cast<int64_t>(m0 * sme_mr + i) * ldc + n0;
                {
                    svint32_t v = svread_hor_za32_s32_m(svundef_s32(), s0, 0, i);
                    if (!zero_mode) { v = svadd_s32_x(s0, v, svld1_s32(s0, row)); }
                    svst1_s32(s0, row, svmin_s32_x(s0, svmax_s32_x(s0, v, vlo), vhi));
                }
                if (t1) {
                    int32_t* p = row + sme_za_n;
                    svint32_t v = svread_hor_za32_s32_m(svundef_s32(), s1, 1, i);
                    if (!zero_mode) { v = svadd_s32_x(s1, v, svld1_s32(s1, p)); }
                    svst1_s32(s1, p, svmin_s32_x(s1, svmax_s32_x(s1, v, vlo), vhi));
                }
                if (t2) {
                    int32_t* p = row + 2 * sme_za_n;
                    svint32_t v = svread_hor_za32_s32_m(svundef_s32(), s2, 2, i);
                    if (!zero_mode) { v = svadd_s32_x(s2, v, svld1_s32(s2, p)); }
                    svst1_s32(s2, p, svmin_s32_x(s2, svmax_s32_x(s2, v, vlo), vhi));
                }
                if (t3) {
                    int32_t* p = row + 3 * sme_za_n;
                    svint32_t v = svread_hor_za32_s32_m(svundef_s32(), s3, 3, i);
                    if (!zero_mode) { v = svadd_s32_x(s3, v, svld1_s32(s3, p)); }
                    svst1_s32(s3, p, svmin_s32_x(s3, svmax_s32_x(s3, v, vlo), vhi));
                }
            }
        }
    }
}

} // namespace nnops::backend::cpu::aarch64

#endif // NNOPS_ARCH_SME
