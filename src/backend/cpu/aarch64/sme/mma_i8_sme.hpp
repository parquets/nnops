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

#include "backend/cpu/aarch64/sme/mma_sme.hpp"

namespace nnops::backend::cpu::aarch64 {

// One full 64-column row from the four column-group tiles: two 128-byte pair
// stores, and the same pair reads when accumulating.
#define NNOPS_I8_STORE_WIDE_ROW(g0_, g1_, g2_, g3_, idx_, row_)                 \
    do {                                                                        \
        svint32_t nno_v0 = svget4_s32(g0_, idx_);                               \
        svint32_t nno_v1 = svget4_s32(g1_, idx_);                               \
        svint32_t nno_v2 = svget4_s32(g2_, idx_);                               \
        svint32_t nno_v3 = svget4_s32(g3_, idx_);                               \
        if (!zero_mode) {                                                       \
            const svint32x2_t nno_c0 = svld1_s32_x2(svptrue_c32(), row_);       \
            nno_v0 = svadd_s32_x(pg, nno_v0, svget2_s32(nno_c0, 0));            \
            nno_v1 = svadd_s32_x(pg, nno_v1, svget2_s32(nno_c0, 1));            \
            const svint32x2_t nno_c1 =                                          \
                svld1_s32_x2(svptrue_c32(), (row_) + 2 * sme_za_n);             \
            nno_v2 = svadd_s32_x(pg, nno_v2, svget2_s32(nno_c1, 0));            \
            nno_v3 = svadd_s32_x(pg, nno_v3, svget2_s32(nno_c1, 1));            \
        }                                                                       \
        nno_v0 = svmin_s32_x(pg, svmax_s32_x(pg, nno_v0, vlo), vhi);            \
        nno_v1 = svmin_s32_x(pg, svmax_s32_x(pg, nno_v1, vlo), vhi);            \
        nno_v2 = svmin_s32_x(pg, svmax_s32_x(pg, nno_v2, vlo), vhi);            \
        nno_v3 = svmin_s32_x(pg, svmax_s32_x(pg, nno_v3, vlo), vhi);            \
        svst1_s32_x2(svptrue_c32(), (row_), svcreate2_s32(nno_v0, nno_v1));    \
        svst1_s32_x2(svptrue_c32(), (row_) + 2 * sme_za_n,                     \
                      svcreate2_s32(nno_v2, nno_v3));                           \
    } while (0)

#define NNOPS_I8_STORE_WIDE_ROW1(v0_, v1_, v2_, v3_, row_)                      \
    do {                                                                        \
        svint32_t nno_v0 = v0_, nno_v1 = v1_, nno_v2 = v2_, nno_v3 = v3_;       \
        if (!zero_mode) {                                                       \
            const svint32x2_t nno_c0 = svld1_s32_x2(svptrue_c32(), row_);       \
            nno_v0 = svadd_s32_x(pg, nno_v0, svget2_s32(nno_c0, 0));            \
            nno_v1 = svadd_s32_x(pg, nno_v1, svget2_s32(nno_c0, 1));            \
            const svint32x2_t nno_c1 =                                          \
                svld1_s32_x2(svptrue_c32(), (row_) + 2 * sme_za_n);             \
            nno_v2 = svadd_s32_x(pg, nno_v2, svget2_s32(nno_c1, 0));            \
            nno_v3 = svadd_s32_x(pg, nno_v3, svget2_s32(nno_c1, 1));            \
        }                                                                       \
        nno_v0 = svmin_s32_x(pg, svmax_s32_x(pg, nno_v0, vlo), vhi);            \
        nno_v1 = svmin_s32_x(pg, svmax_s32_x(pg, nno_v1, vlo), vhi);            \
        nno_v2 = svmin_s32_x(pg, svmax_s32_x(pg, nno_v2, vlo), vhi);            \
        nno_v3 = svmin_s32_x(pg, svmax_s32_x(pg, nno_v3, vlo), vhi);            \
        svst1_s32_x2(svptrue_c32(), (row_), svcreate2_s32(nno_v0, nno_v1));    \
        svst1_s32_x2(svptrue_c32(), (row_) + 2 * sme_za_n,                     \
                      svcreate2_s32(nno_v2, nno_v3));                           \
    } while (0)

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

            // Full 64-column block: four ZA rows per MOVA and a 128-byte pair
            // store per two column groups, as in the float kernels.
            const bool wide = t3 && live0 == sme_za_n;
            const svbool_t pg = svptrue_b32();
            if (wide) {
                int i = 0;
                for (; i + 4 <= mlive; i += 4) {
                    const svint32x4_t g0 = svread_hor_za32_s32_vg4(0, i);
                    const svint32x4_t g1 = svread_hor_za32_s32_vg4(1, i);
                    const svint32x4_t g2 = svread_hor_za32_s32_vg4(2, i);
                    const svint32x4_t g3 = svread_hor_za32_s32_vg4(3, i);
                    int32_t* const r0 =
                        c + static_cast<int64_t>(m0 * sme_mr + i) * ldc + n0;
                    const int64_t l = ldc;
                    NNOPS_I8_STORE_WIDE_ROW(g0, g1, g2, g3, 0, r0);
                    NNOPS_I8_STORE_WIDE_ROW(g0, g1, g2, g3, 1, r0 + l);
                    NNOPS_I8_STORE_WIDE_ROW(g0, g1, g2, g3, 2, r0 + 2 * l);
                    NNOPS_I8_STORE_WIDE_ROW(g0, g1, g2, g3, 3, r0 + 3 * l);
                }
                for (; i < mlive; ++i) {
                    int32_t* const row = c + static_cast<int64_t>(m0 * sme_mr + i) * ldc + n0;
                    NNOPS_I8_STORE_WIDE_ROW1(
                        svread_hor_za32_s32_m(svundef_s32(), pg, 0, i),
                        svread_hor_za32_s32_m(svundef_s32(), pg, 1, i),
                        svread_hor_za32_s32_m(svundef_s32(), pg, 2, i),
                        svread_hor_za32_s32_m(svundef_s32(), pg, 3, i), row);
                }
            } else {
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
}

} // namespace nnops::backend::cpu::aarch64

#endif // NNOPS_ARCH_SME
