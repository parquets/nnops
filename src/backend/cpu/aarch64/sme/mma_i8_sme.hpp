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
#include <cmath>
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
        const bool full = live0 == sme_za_n && live1 == sme_za_n && live2 == sme_za_n &&
                          live3 == sme_za_n;
        const bool half4 = !full && live0 == sme_za_n && live1 == sme_za_n;
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

// The epilogue's three steps, as macros: SVE intrinsics inside a helper would
// need their own streaming attribute, which this compiler refuses to take on a
// definition here (same reason the MOPA loops are macros).
#define NNOPS_I8_EPI_SCALE(pg_, n_, out_)                                                 \
    const svfloat32_t out_ =                                                             \
        svmul_f32_x(pg_, vsa, s_b ? svld1_f32(pg_, s_b + (n_)) : svdup_n_f32(s_b_scalar))

// One column group of one row in the float domain: the scale product, the
// optional bias and the clip. The arithmetic is the dequantizing pass's, lane
// for lane, so the two agree bit for bit.
#define NNOPS_I8_EPI_VAL(v_, sc_, bi_, pg_, out_)                                         \
    svfloat32_t out_ = svmul_f32_x(pg_, svcvt_f32_s32_x(pg_, (v_)), (sc_));                \
    if (col_bias != nullptr) {                                                            \
        out_ = svadd_f32_x(pg_, out_, (bi_));                                             \
    }                                                                                     \
    if (clip_lo && clip_hi) {                                                                 \
        out_ = svclamp_f32(out_, vlo, vhi);                                                   \
    } else if (clip_lo) {                                                                     \
        out_ = svmax_f32_x(pg_, out_, vlo);                                                   \
    } else if (clip_hi) {                                                                     \
        out_ = svmin_f32_x(pg_, out_, vhi);                                                   \
    }

// One column group out on its own, for the partial blocks and f16.
#define NNOPS_I8_EPI_PUT(f_, pg_, cnt_, dst_)                                             \
    if constexpr (std::is_same_v<TOut, half>) {                                           \
        svst1_f16(svwhilelt_b16(0, (cnt_)), reinterpret_cast<float16_t*>(dst_),           \
                  svcvt_f16_x(pg_, f_));                                                  \
    } else {                                                                              \
        svst1_f32(pg_, (dst_), f_);                                                       \
    }

#define NNOPS_I8_EPI_LANE(v_, sc_, bi_, pg_, cnt_, dst_)                                  \
    do {                                                                                  \
        NNOPS_I8_EPI_VAL(v_, sc_, bi_, pg_, nno_f);                                       \
        NNOPS_I8_EPI_PUT(nno_f, pg_, cnt_, dst_);                                         \
    } while (0)

// A whole row: four column groups, live0..live3 of them populated. A full
// 64-column block stores as two 128-byte pairs — a 64-byte store is half a
// cache line, and the read-back it costs is most of this store's time.
#define NNOPS_I8_EPI_ROW(g0_, g1_, g2_, g3_, idx_, row_)                                  \
    do {                                                                                  \
        if constexpr (std::is_same_v<TOut, float>) {                                       \
            if (half4) {                                                                   \
                NNOPS_I8_EPI_VAL(svget4_s32(g0_, idx_), sc0, bi0, s0, nno_e0);             \
                NNOPS_I8_EPI_VAL(svget4_s32(g1_, idx_), sc1, bi1, s1, nno_e1);             \
                svst1_f32_x2(svptrue_c32(), (row_), svcreate2_f32(nno_e0, nno_e1));        \
            } else if (full) {                                                             \
                NNOPS_I8_EPI_VAL(svget4_s32(g0_, idx_), sc0, bi0, s0, nno_e0);             \
                NNOPS_I8_EPI_VAL(svget4_s32(g1_, idx_), sc1, bi1, s1, nno_e1);             \
                svst1_f32_x2(svptrue_c32(), (row_), svcreate2_f32(nno_e0, nno_e1));        \
                NNOPS_I8_EPI_VAL(svget4_s32(g2_, idx_), sc2, bi2, s2, nno_e2);             \
                NNOPS_I8_EPI_VAL(svget4_s32(g3_, idx_), sc3, bi3, s3, nno_e3);             \
                svst1_f32_x2(svptrue_c32(), (row_) + 2 * sme_za_n,                         \
                             svcreate2_f32(nno_e2, nno_e3));                               \
            } else {                                                                       \
                NNOPS_I8_EPI_LANE(svget4_s32(g0_, idx_), sc0, bi0, s0, live0, (row_));     \
                if (live1) {                                                               \
                    NNOPS_I8_EPI_LANE(svget4_s32(g1_, idx_), sc1, bi1, s1, live1,          \
                                      (row_) + sme_za_n);                                  \
                }                                                                          \
                if (live2) {                                                               \
                    NNOPS_I8_EPI_LANE(svget4_s32(g2_, idx_), sc2, bi2, s2, live2,          \
                                      (row_) + 2 * sme_za_n);                              \
                }                                                                          \
                if (live3) {                                                               \
                    NNOPS_I8_EPI_LANE(svget4_s32(g3_, idx_), sc3, bi3, s3, live3,          \
                                      (row_) + 3 * sme_za_n);                              \
                }                                                                          \
            }                                                                              \
        } else {                                                                           \
            NNOPS_I8_EPI_LANE(svget4_s32(g0_, idx_), sc0, bi0, s0, live0, (row_));         \
            if (live1) {                                                                   \
                NNOPS_I8_EPI_LANE(svget4_s32(g1_, idx_), sc1, bi1, s1, live1,              \
                                  (row_) + sme_za_n);                                      \
            }                                                                              \
            if (live2) {                                                                   \
                NNOPS_I8_EPI_LANE(svget4_s32(g2_, idx_), sc2, bi2, s2, live2,              \
                                  (row_) + 2 * sme_za_n);                                  \
            }                                                                              \
            if (live3) {                                                                   \
                NNOPS_I8_EPI_LANE(svget4_s32(g3_, idx_), sc3, bi3, s3, live3,              \
                                  (row_) + 3 * sme_za_n);                                  \
            }                                                                              \
        }                                                                                  \
    } while (0)

/// The requantizing store: the same accumulation as mma_i8_sme, but the tile is
/// dequantized and written out as it leaves ZA, so the int32 accumulator never
/// reaches memory (XNNPACK's KleidiAI kernels are shaped the same way).
///
///     out[m,n] = clip((float)acc * (s_a * s_b[n]) + bias[n], lo, hi)
///
/// The clip bounds carry the activation too — a Relu or Relu6 composes into
/// them — and the caller only takes this path when nothing is compensated per
/// row or per column and the whole K is one block, so every tile starts from a
/// zeroed ZA.
template <class TOut>
__arm_locally_streaming __arm_new("za") __attribute__((target("+sme2")))
inline void mma_i8_sme_epi(int mc, int nc, int kc, TOut* out, int64_t ldm, const int8_t* a,
                           const int8_t* b, int64_t b_block_stride, const float* s_b,
                           float s_b_scalar, float s_a, const float* col_bias, float lo,
                           float hi) noexcept
{
    const int m_tiles = (mc + sme_mr - 1) / sme_mr;
    const int n_blocks = (nc + sme_nr_max - 1) / sme_nr_max;
    const int k_groups = (kc + 3) / 4;
    const svfloat32_t vlo = svdup_n_f32(lo);
    const svfloat32_t vhi = svdup_n_f32(hi);
    const svfloat32_t vsa = svdup_n_f32(s_a);
    // An unbounded side — the default epilogue has both — costs no clamp op.
    const bool clip_lo = std::isfinite(lo) || lo > 0.0f;
    const bool clip_hi = std::isfinite(hi) || hi < 0.0f;

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
        const bool full = live0 == sme_za_n && live1 == sme_za_n && live2 == sme_za_n &&
                          live3 == sme_za_n;
        const bool half4 = !full && live0 == sme_za_n && live1 == sme_za_n;
        const int8_t* bp =
            b + static_cast<int64_t>(nb) * (b_block_stride != 0 ?
                                                b_block_stride :
                                                static_cast<int64_t>(k_groups) * (4 * sme_nr_max));
        // The scale product and the bias are per output column: one load per
        // 64-column block, not one per row.
        const int c1 = n0 + sme_za_n;
        const int c2 = n0 + 2 * sme_za_n;
        const int c3 = n0 + 3 * sme_za_n;
        NNOPS_I8_EPI_SCALE(s0, n0, sc0);
        NNOPS_I8_EPI_SCALE(s1, c1, sc1);
        NNOPS_I8_EPI_SCALE(s2, c2, sc2);
        NNOPS_I8_EPI_SCALE(s3, c3, sc3);
        const svfloat32_t bi0 = col_bias ? svld1_f32(s0, col_bias + n0) : svdup_n_f32(0.0f);
        const svfloat32_t bi1 =
            (col_bias && t1) ? svld1_f32(s1, col_bias + c1) : svdup_n_f32(0.0f);
        const svfloat32_t bi2 =
            (col_bias && t2) ? svld1_f32(s2, col_bias + c2) : svdup_n_f32(0.0f);
        const svfloat32_t bi3 =
            (col_bias && t3) ? svld1_f32(s3, col_bias + c3) : svdup_n_f32(0.0f);

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

            TOut* const obase = out + static_cast<int64_t>(m0 * sme_mr) * ldm + n0;
            int i = 0;
            for (; i + 4 <= mlive; i += 4) {
                const svint32x4_t g0 = svread_hor_za32_s32_vg4(0, static_cast<uint32_t>(i));
                const svint32x4_t g1 = svread_hor_za32_s32_vg4(1, static_cast<uint32_t>(i));
                const svint32x4_t g2 = svread_hor_za32_s32_vg4(2, static_cast<uint32_t>(i));
                const svint32x4_t g3 = svread_hor_za32_s32_vg4(3, static_cast<uint32_t>(i));
                TOut* const r0 = obase + static_cast<int64_t>(i) * ldm;
                const int64_t l = ldm;
                NNOPS_I8_EPI_ROW(g0, g1, g2, g3, 0, r0);
                NNOPS_I8_EPI_ROW(g0, g1, g2, g3, 1, r0 + l);
                NNOPS_I8_EPI_ROW(g0, g1, g2, g3, 2, r0 + 2 * l);
                NNOPS_I8_EPI_ROW(g0, g1, g2, g3, 3, r0 + 3 * l);
            }
            for (; i < mlive; ++i) {
                const svbool_t pg = svptrue_b32();
                TOut* const row = obase + static_cast<int64_t>(i) * ldm;
                NNOPS_I8_EPI_LANE(svread_hor_za32_s32_m(svundef_s32(), pg, 0, i), sc0, bi0, s0,
                                  live0, row);
                if (live1) {
                    NNOPS_I8_EPI_LANE(svread_hor_za32_s32_m(svundef_s32(), pg, 1, i), sc1, bi1,
                                      s1, live1, row + sme_za_n);
                }
                if (live2) {
                    NNOPS_I8_EPI_LANE(svread_hor_za32_s32_m(svundef_s32(), pg, 2, i), sc2, bi2,
                                      s2, live2, row + 2 * sme_za_n);
                }
                if (live3) {
                    NNOPS_I8_EPI_LANE(svread_hor_za32_s32_m(svundef_s32(), pg, 3, i), sc3, bi3,
                                      s3, live3, row + 3 * sme_za_n);
                }
            }
        }
    }
}

// ---- the 8-bit requantizing store -----------------------------------------
//
// XNNPACK's packed conv ends in an int8 tile, so the store is the comparison
// that matters there: saturate on the way out and write a quarter of the bytes.

// The requantize itself, without the store: float scale, round to nearest ties
// away, offset, then one SVE2 clamp. Clamping the integer afterwards is the
// same as clamping the float first — the bounds are integers — and it folds the
// two float clamps into one instruction whose result is already a byte.
#define NNOPS_I8_EPI8_Q(v_, sc_, bi_, pg_, out_)                                          \
    svint32_t out_ = (v_);                                                                \
    if (col_bias != nullptr) {                                                            \
        out_ = svadd_s32_x(pg_, out_, (bi_));                                             \
    }                                                                                     \
    if (relu) {                                                                           \
        out_ = svmax_s32_x(pg_, out_, vzero);                                             \
    }                                                                                     \
    out_ = svadd_s32_x(                                                                   \
        pg_,                                                                              \
        svcvt_s32_f32_x(pg_, svrinta_f32_x(                                               \
                                 pg_, svmul_f32_x(pg_, svcvt_f32_s32_x(pg_, out_), (sc_)))), \
        vzp);                                                                             \
    out_ = svclamp_s32(out_, vkmin, vkmax)

// Two column groups of one row, clamped, packed by taking the low byte of each
// lane — `uzp1` on the byte view gathers 32 of them in one instruction.
#define NNOPS_I8_EPI8_STORE2(q0_, q1_, dst_)                                              \
    do {                                                                                  \
        if constexpr (std::is_same_v<TOut, uint8_t>) {                                    \
            svst1_u8(svwhilelt_b8(0, 2 * sme_za_n), reinterpret_cast<uint8_t*>(dst_),     \
                     svuzp1_u8(svreinterpret_u8_s32(q0_), svreinterpret_u8_s32(q1_)));    \
        } else {                                                                          \
            svst1_s8(svwhilelt_b8(0, 2 * sme_za_n), reinterpret_cast<int8_t*>(dst_),      \
                     svuzp1_s8(svreinterpret_s8_s32(q0_), svreinterpret_s8_s32(q1_)));    \
        }                                                                                 \
    } while (0)

// One group on its own, for the partial blocks.
#define NNOPS_I8_EPI8_STORE1(q_, pg_, cnt_, dst_)                                         \
    do {                                                                                  \
        if constexpr (std::is_same_v<TOut, uint8_t>) {                                    \
            svst1_u8(svwhilelt_b8(0, (cnt_)), reinterpret_cast<uint8_t*>(dst_),           \
                     svuzp1_u8(svreinterpret_u8_s32(q_), svreinterpret_u8_s32(q_)));      \
        } else {                                                                          \
            svst1_s8(svwhilelt_b8(0, (cnt_)), reinterpret_cast<int8_t*>(dst_),            \
                     svuzp1_s8(svreinterpret_s8_s32(q_), svreinterpret_s8_s32(q_)));      \
        }                                                                                 \
    } while (0)

// A whole row: four column groups, 32 bytes per store on the full blocks.
#define NNOPS_I8_EPI8_ROW(g0_, g1_, g2_, g3_, idx_, row_)                                 \
    do {                                                                                  \
        if (full) {                                                                       \
            NNOPS_I8_EPI8_Q(svget4_s32(g0_, idx_), sc0, bi0, s0, nno_q0);                 \
            NNOPS_I8_EPI8_Q(svget4_s32(g1_, idx_), sc1, bi1, s1, nno_q1);                 \
            NNOPS_I8_EPI8_STORE2(nno_q0, nno_q1, (row_));                                 \
            NNOPS_I8_EPI8_Q(svget4_s32(g2_, idx_), sc2, bi2, s2, nno_q2);                 \
            NNOPS_I8_EPI8_Q(svget4_s32(g3_, idx_), sc3, bi3, s3, nno_q3);                 \
            NNOPS_I8_EPI8_STORE2(nno_q2, nno_q3, (row_) + 2 * sme_za_n);                  \
        } else {                                                                          \
            NNOPS_I8_EPI8_Q(svget4_s32(g0_, idx_), sc0, bi0, s0, nno_q0);                 \
            NNOPS_I8_EPI8_STORE1(nno_q0, s0, live0, (row_));                              \
            if (live1) {                                                                  \
                NNOPS_I8_EPI8_Q(svget4_s32(g1_, idx_), sc1, bi1, s1, nno_q1);             \
                NNOPS_I8_EPI8_STORE1(nno_q1, s1, live1, (row_) + sme_za_n);               \
            }                                                                             \
            if (live2) {                                                                  \
                NNOPS_I8_EPI8_Q(svget4_s32(g2_, idx_), sc2, bi2, s2, nno_q2);             \
                NNOPS_I8_EPI8_STORE1(nno_q2, s2, live2, (row_) + 2 * sme_za_n);           \
            }                                                                             \
            if (live3) {                                                                  \
                NNOPS_I8_EPI8_Q(svget4_s32(g3_, idx_), sc3, bi3, s3, nno_q3);             \
                NNOPS_I8_EPI8_STORE1(nno_q3, s3, live3, (row_) + 3 * sme_za_n);           \
            }                                                                             \
        }                                                                                 \
    } while (0)

/// The requantizing store for an 8-bit destination:
///
///     out[m,n] = sat(round_away((acc + bias[n]) * (s_a/s_out * s_b[n])) + zp_out)
///
/// with Relu folded into a floor of zero on the integer side, exactly as the
/// int8 row pass does it. Same preconditions as the float form.
template <class TOut>
__arm_locally_streaming __arm_new("za") __attribute__((target("+sme2")))
inline void mma_i8_sme_epi_i8(int mc, int nc, int kc, TOut* out, int64_t ldm, const int8_t* a,
                              const int8_t* b, int64_t b_block_stride, const float* s_b,
                              float s_b_scalar, float row_k, const int32_t* col_bias,
                              int32_t zp_out, bool relu) noexcept
{
    constexpr bool kU8 = std::is_same_v<TOut, uint8_t>;
    // The destination's integer range, clamped after the zero point instead of
    // before the rounding — the same result, one instruction.
    const svint32_t vkmin = svdup_n_s32(kU8 ? 0 : -128);
    const svint32_t vkmax = svdup_n_s32(kU8 ? 255 : 127);
    const svfloat32_t vrowk = svdup_n_f32(row_k);
    const svint32_t vzp = svdup_n_s32(zp_out);
    const svint32_t vzero = svdup_n_s32(0);
    const int m_tiles = (mc + sme_mr - 1) / sme_mr;
    const int n_blocks = (nc + sme_nr_max - 1) / sme_nr_max;
    const int k_groups = (kc + 3) / 4;

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
        const bool full = live0 == sme_za_n && live1 == sme_za_n && live2 == sme_za_n &&
                          live3 == sme_za_n;
        const bool half4 = !full && live0 == sme_za_n && live1 == sme_za_n;
        const int8_t* bp =
            b + static_cast<int64_t>(nb) * (b_block_stride != 0 ?
                                                b_block_stride :
                                                static_cast<int64_t>(k_groups) * (4 * sme_nr_max));
        const int c1 = n0 + sme_za_n;
        const int c2 = n0 + 2 * sme_za_n;
        const int c3 = n0 + 3 * sme_za_n;
        const svfloat32_t sc0 = svmul_f32_x(s0, vrowk,
                                            s_b ? svld1_f32(s0, s_b + n0) : svdup_n_f32(s_b_scalar));
        const svfloat32_t sc1 =
            svmul_f32_x(s1, vrowk, s_b ? svld1_f32(s1, s_b + c1) : svdup_n_f32(s_b_scalar));
        const svfloat32_t sc2 =
            svmul_f32_x(s2, vrowk, s_b ? svld1_f32(s2, s_b + c2) : svdup_n_f32(s_b_scalar));
        const svfloat32_t sc3 =
            svmul_f32_x(s3, vrowk, s_b ? svld1_f32(s3, s_b + c3) : svdup_n_f32(s_b_scalar));
        const svint32_t bi0 = col_bias ? svld1_s32(s0, col_bias + n0) : svdup_n_s32(0);
        const svint32_t bi1 =
            (col_bias && t1) ? svld1_s32(s1, col_bias + c1) : svdup_n_s32(0);
        const svint32_t bi2 =
            (col_bias && t2) ? svld1_s32(s2, col_bias + c2) : svdup_n_s32(0);
        const svint32_t bi3 =
            (col_bias && t3) ? svld1_s32(s3, col_bias + c3) : svdup_n_s32(0);

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

            TOut* const obase = out + static_cast<int64_t>(m0 * sme_mr) * ldm + n0;
            int i = 0;
            for (; i + 4 <= mlive; i += 4) {
                const svint32x4_t g0 = svread_hor_za32_s32_vg4(0, static_cast<uint32_t>(i));
                const svint32x4_t g1 = svread_hor_za32_s32_vg4(1, static_cast<uint32_t>(i));
                const svint32x4_t g2 = svread_hor_za32_s32_vg4(2, static_cast<uint32_t>(i));
                const svint32x4_t g3 = svread_hor_za32_s32_vg4(3, static_cast<uint32_t>(i));
                TOut* const r0 = obase + static_cast<int64_t>(i) * ldm;
                const int64_t l = ldm;
                NNOPS_I8_EPI8_ROW(g0, g1, g2, g3, 0, r0);
                NNOPS_I8_EPI8_ROW(g0, g1, g2, g3, 1, r0 + l);
                NNOPS_I8_EPI8_ROW(g0, g1, g2, g3, 2, r0 + 2 * l);
                NNOPS_I8_EPI8_ROW(g0, g1, g2, g3, 3, r0 + 3 * l);
            }
            for (; i < mlive; ++i) {
                const svbool_t pg = svptrue_b32();
                TOut* const row = obase + static_cast<int64_t>(i) * ldm;
                NNOPS_I8_EPI8_Q(svread_hor_za32_s32_m(svundef_s32(), pg, 0, i), sc0, bi0, s0,
                                nno_q0);
                NNOPS_I8_EPI8_STORE1(nno_q0, s0, live0, row);
                if (live1) {
                    NNOPS_I8_EPI8_Q(svread_hor_za32_s32_m(svundef_s32(), pg, 1, i), sc1, bi1,
                                    s1, nno_q1);
                    NNOPS_I8_EPI8_STORE1(nno_q1, s1, live1, row + sme_za_n);
                }
                if (live2) {
                    NNOPS_I8_EPI8_Q(svread_hor_za32_s32_m(svundef_s32(), pg, 2, i), sc2, bi2,
                                    s2, nno_q2);
                    NNOPS_I8_EPI8_STORE1(nno_q2, s2, live2, row + 2 * sme_za_n);
                }
                if (live3) {
                    NNOPS_I8_EPI8_Q(svread_hor_za32_s32_m(svundef_s32(), pg, 3, i), sc3, bi3,
                                    s3, nno_q3);
                    NNOPS_I8_EPI8_STORE1(nno_q3, s3, live3, row + 3 * sme_za_n);
                }
            }
        }
    }
}

#undef NNOPS_I8_EPI8_ROW
#undef NNOPS_I8_EPI8_STORE1
#undef NNOPS_I8_EPI8_STORE2
#undef NNOPS_I8_EPI8_Q

#undef NNOPS_I8_EPI_ROW
#undef NNOPS_I8_EPI_LANE
#undef NNOPS_I8_EPI_PUT
#undef NNOPS_I8_EPI_VAL
#undef NNOPS_I8_EPI_SCALE

} // namespace nnops::backend::cpu::aarch64

#endif // NNOPS_ARCH_SME
