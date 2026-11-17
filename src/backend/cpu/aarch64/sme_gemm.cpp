/// @file sme_gemm.cpp
/// @brief The SME GEMM micro-kernels; see sme_gemm.hpp for the layouts.
///
/// This translation unit is the only one built with `+sme2`; the rest of the
/// library stays baseline AArch64 and reaches these functions behind
/// available(). Four ZA tiles hold a 16(M) x 64(N) block, and the K loop issues
/// four independent FMOPAs per k so the pipeline never waits on one tile's
/// dependency chain.

#include "backend/cpu/aarch64/sme_gemm.hpp"

#if defined(NNOPS_ARCH_SME)

#include "nnops/detail/simd/cpu_features.hpp"
#include "detail/bench_knobs.hpp"

#include <algorithm>

#include <arm_neon.h>
#include <arm_sme.h>

namespace nnops::backend::cpu::sme {

namespace {

constexpr int kMr = 16;      // SVL 64 B / sizeof(f32)
constexpr int kPanel = 16;   // one ZA tile's N width

/// The streaming vector length, asked of the hardware in streaming mode. Safe
/// only after cpu_has_sme().
__arm_locally_streaming
size_t probe_svl() noexcept
{
    return static_cast<size_t>(svcntsb());
}

/// One operand vector of 16 lanes widened to f32.
__arm_locally_streaming __attribute__((always_inline))
inline svfloat32_t load_op(svbool_t pg, const float* p) noexcept
{
    return svld1(pg, p);
}

/// One pre-interleaved K pair for 16 lanes: 32 f16, one full register.
__arm_locally_streaming __attribute__((always_inline))
inline svfloat16_t load_h32(svbool_t pg, const half* p) noexcept
{
    return svld1_f16(pg, reinterpret_cast<const float16_t*>(p));
}

/// Widen 16 f16 lanes to 16 f32 lanes in order (the bias seed's path).
__arm_locally_streaming __attribute__((always_inline))
inline svfloat32_t widen_h16(svfloat16_t h) noexcept
{
    const svbool_t pgh = svwhilelt_b16(0, 16);
    return svzip1_f32(svcvt_f32_f16_x(pgh, h), svcvtlt_f32_f16_x(pgh, h));
}

__arm_locally_streaming __attribute__((always_inline))
inline void store_op(float* p, svbool_t pg, svfloat32_t v, int /*live*/) noexcept
{
    svst1(pg, p, v);
}

__arm_locally_streaming __attribute__((always_inline))
inline void store_op(half* p, svbool_t pg32, svfloat32_t v, int live) noexcept
{
    // The SVE narrow puts the even source lanes in the low half and the odd
    // ones in the high half; uzp1(h, h) interleaves them back into order. The
    // b16 predicate stores exactly the live columns, so a short last panel
    // never runs into the next row of C.
    const svfloat16_t h = svcvt_f16_f32_x(pg32, v);
    svst1_f16(svwhilelt_b16(0, live), reinterpret_cast<float16_t*>(p), svuzp1_f16(h, h));
}

__arm_locally_streaming __attribute__((always_inline))
inline svfloat32_t clamp_v(svbool_t pg, svfloat32_t v, svfloat32_t lo, svfloat32_t hi) noexcept
{
    return svmin_f32_x(pg, svmax_f32_x(pg, v, lo), hi);
}

/// f32: four N panels (16 columns each) share one K loop; four independent ZA
/// tiles keep the MOPA pipeline fed. The B block's 64 columns per k are one
/// contiguous run, so each k is two cache lines rather than four scattered
/// streams. Panels past the N extent keep an all-false predicate, so a short
/// tail costs no branch in the loop.
__arm_locally_streaming __arm_new("za")
void gemm_f32_impl(int mc, int nc, int kc, float* c, int ldc, const float* a, const float* b,
                   bool zero_mode, float lo, float hi, const float* col_bias,
                   int64_t b_block_stride)
{
    const int m_tiles = (mc + kMr - 1) / kMr;
    const int n_blocks = (nc + kNB - 1) / kNB;
    const svfloat32_t vlo = svdup_n_f32(lo);
    const svfloat32_t vhi = svdup_n_f32(hi);
    // The n block is the outer loop. Its 64 columns per k are one contiguous
    // 256-byte run, small enough to stay in L1 across every M tile, whereas an
    // M tile is only 16 rows: the other order re-reads the whole B slab once
    // per tile, four times the traffic for the same work.
    for (int nb = 0; nb < n_blocks; ++nb) {
        const int n0 = nb * kNB;
        const bool t1 = n0 + kPanel < nc;
        const bool t2 = n0 + 2 * kPanel < nc;
        const bool t3 = n0 + 3 * kPanel < nc;
        const svbool_t pn0 = svwhilelt_b32(0, std::min(kPanel, nc - n0));
        const svbool_t pn1 =
            t1 ? svwhilelt_b32(0, std::min(kPanel, nc - n0 - kPanel)) : svpfalse_b();
        const svbool_t pn2 =
            t2 ? svwhilelt_b32(0, std::min(kPanel, nc - n0 - 2 * kPanel)) : svpfalse_b();
        const svbool_t pn3 =
            t3 ? svwhilelt_b32(0, std::min(kPanel, nc - n0 - 3 * kPanel)) : svpfalse_b();
        const float* bp = b + static_cast<int64_t>(nb) * (b_block_stride != 0 ? b_block_stride
                                                                          : static_cast<int64_t>(kc) * kNB);
        const float* bias0 = col_bias != nullptr ? col_bias + n0 : nullptr;

        for (int m0 = 0; m0 < m_tiles; ++m0) {
            const int mlive = std::min(kMr, mc - m0 * kMr);
            const svbool_t pm = svwhilelt_b32(0, mlive);
            const float* ap = a + static_cast<int64_t>(m0) * kc * kMr;

            svzero_za();
            if (col_bias != nullptr) {
                const svfloat32_t bv0 = load_op(pn0, bias0);
                for (int i = 0; i < mlive; ++i) { svwrite_hor_za32_f32_m(0, i, pn0, bv0); }
                if (t1) {
                    const svfloat32_t bv1 = load_op(pn1, bias0 + kPanel);
                    for (int i = 0; i < mlive; ++i) { svwrite_hor_za32_f32_m(1, i, pn1, bv1); }
                }
                if (t2) {
                    const svfloat32_t bv2 = load_op(pn2, bias0 + 2 * kPanel);
                    for (int i = 0; i < mlive; ++i) { svwrite_hor_za32_f32_m(2, i, pn2, bv2); }
                }
                if (t3) {
                    const svfloat32_t bv3 = load_op(pn3, bias0 + 3 * kPanel);
                    for (int i = 0; i < mlive; ++i) { svwrite_hor_za32_f32_m(3, i, pn3, bv3); }
                }
            }

            const float* brow = bp;
            for (int k = 0; k < kc; ++k, brow += kNB) {
                const svfloat32_t av = load_op(pm, ap + static_cast<int64_t>(k) * kMr);
                svmopa_za32_f32_m(0, pm, pn0, av, load_op(pn0, brow));
                svmopa_za32_f32_m(1, pm, pn1, av, load_op(pn1, brow + kPanel));
                svmopa_za32_f32_m(2, pm, pn2, av, load_op(pn2, brow + 2 * kPanel));
                svmopa_za32_f32_m(3, pm, pn3, av, load_op(pn3, brow + 3 * kPanel));
            }

            for (int i = 0; i < mlive; ++i) {
                float* row = c + static_cast<int64_t>(m0 * kMr + i) * ldc + n0;
                {
                    svfloat32_t v = svread_hor_za32_f32_m(svundef_f32(), pn0, 0, i);
                    if (!zero_mode) { v = svadd_f32_x(pn0, v, svld1(pn0, row)); }
                    svst1(pn0, row, clamp_v(pn0, v, vlo, vhi));
                }
                if (t1) {
                    float* p = row + kPanel;
                    svfloat32_t v = svread_hor_za32_f32_m(svundef_f32(), pn1, 1, i);
                    if (!zero_mode) { v = svadd_f32_x(pn1, v, svld1(pn1, p)); }
                    svst1(pn1, p, clamp_v(pn1, v, vlo, vhi));
                }
                if (t2) {
                    float* p = row + 2 * kPanel;
                    svfloat32_t v = svread_hor_za32_f32_m(svundef_f32(), pn2, 2, i);
                    if (!zero_mode) { v = svadd_f32_x(pn2, v, svld1(pn2, p)); }
                    svst1(pn2, p, clamp_v(pn2, v, vlo, vhi));
                }
                if (t3) {
                    float* p = row + 3 * kPanel;
                    svfloat32_t v = svread_hor_za32_f32_m(svundef_f32(), pn3, 3, i);
                    if (!zero_mode) { v = svadd_f32_x(pn3, v, svld1(pn3, p)); }
                    svst1(pn3, p, clamp_v(pn3, v, vlo, vhi));
                }
            }
        }
    }
}

/// f16: the hardware two-way dot (`FMOPA ZA.S` with `.H` operands) consumes two
/// K per instruction. The pack already interleaved them — A at [row][k,k+1] and
/// B at [col][k,k+1] — so a K step is one 64-byte load per operand and one
/// MOPA, the same shape as the f32 loop. Interleaving here instead would cost
/// five permutes per K pair that contend with the MOPA issue slot.
///
/// The MOPA predicates are **b16**: the operand's lane 2i and 2i+1 are entry
/// i's two K values, so a row is live when both lanes are. A b32 predicate
/// activates only the even lanes, which silently drops half the products.
__arm_locally_streaming __arm_new("za")
void gemm_f16_impl(int mc, int nc, int kc, half* c, int ldc, const half* a, const half* b,
                   bool zero_mode, float lo, float hi, const half* col_bias,
                   int64_t b_block_stride)
{
    const int m_tiles = (mc + kMr - 1) / kMr;
    const int n_blocks = (nc + kNB - 1) / kNB;
    const int k_pairs = (kc + 1) / 2;
    const svfloat32_t vlo = svdup_n_f32(lo);
    const svfloat32_t vhi = svdup_n_f32(hi);
    // Same loop order as the f32 kernel: n block outside, M tiles inside.
    for (int nb = 0; nb < n_blocks; ++nb) {
        const int n0 = nb * kNB;
        const bool t1 = n0 + kPanel < nc;
        const bool t2 = n0 + 2 * kPanel < nc;
        const bool t3 = n0 + 3 * kPanel < nc;
        const int live0 = std::min(kPanel, nc - n0);
        const int live1 = t1 ? std::min(kPanel, nc - n0 - kPanel) : 0;
        const int live2 = t2 ? std::min(kPanel, nc - n0 - 2 * kPanel) : 0;
        const int live3 = t3 ? std::min(kPanel, nc - n0 - 3 * kPanel) : 0;
        const svbool_t pn0 = svwhilelt_b32(0, live0);
        const svbool_t pn1 = t1 ? svwhilelt_b32(0, live1) : svpfalse_b();
        const svbool_t pn2 = t2 ? svwhilelt_b32(0, live2) : svpfalse_b();
        const svbool_t pn3 = t3 ? svwhilelt_b32(0, live3) : svpfalse_b();
        const svbool_t qn0 = svwhilelt_b16(0, 2 * live0);
        const svbool_t qn1 = t1 ? svwhilelt_b16(0, 2 * live1) : svpfalse_b();
        const svbool_t qn2 = t2 ? svwhilelt_b16(0, 2 * live2) : svpfalse_b();
        const svbool_t qn3 = t3 ? svwhilelt_b16(0, 2 * live3) : svpfalse_b();
        const half* bp = b + static_cast<int64_t>(nb) * (b_block_stride != 0 ? b_block_stride
                                                                          : static_cast<int64_t>(k_pairs) * (2 * kNB));
        const half* bias0 = col_bias != nullptr ? col_bias + n0 : nullptr;

        for (int m0 = 0; m0 < m_tiles; ++m0) {
            const int mlive = std::min(kMr, mc - m0 * kMr);
            const svbool_t pmh = svwhilelt_b16(0, 2 * mlive);
            const half* ap = a + static_cast<int64_t>(m0) * k_pairs * (2 * kMr);

            svzero_za();
            if (col_bias != nullptr) {
                const svfloat32_t bv0 = widen_h16(svld1_f16(svwhilelt_b16(0, live0), bias0));
                for (int i = 0; i < mlive; ++i) { svwrite_hor_za32_f32_m(0, i, pn0, bv0); }
                if (t1) {
                    const svfloat32_t bv1 =
                        widen_h16(svld1_f16(svwhilelt_b16(0, live1), bias0 + kPanel));
                    for (int i = 0; i < mlive; ++i) { svwrite_hor_za32_f32_m(1, i, pn1, bv1); }
                }
                if (t2) {
                    const svfloat32_t bv2 =
                        widen_h16(svld1_f16(svwhilelt_b16(0, live2), bias0 + 2 * kPanel));
                    for (int i = 0; i < mlive; ++i) { svwrite_hor_za32_f32_m(2, i, pn2, bv2); }
                }
                if (t3) {
                    const svfloat32_t bv3 =
                        widen_h16(svld1_f16(svwhilelt_b16(0, live3), bias0 + 3 * kPanel));
                    for (int i = 0; i < mlive; ++i) { svwrite_hor_za32_f32_m(3, i, pn3, bv3); }
                }
            }

            const half* arow = ap;
            const half* brow = bp;
            for (int kp = 0; kp < k_pairs; ++kp, arow += 2 * kMr, brow += 2 * kNB) {
                const svfloat16_t av = load_h32(pmh, arow);
                svmopa_za32_f16_m(0, pmh, qn0, av, load_h32(qn0, brow));
                svmopa_za32_f16_m(1, pmh, qn1, av, load_h32(qn1, brow + 2 * kPanel));
                svmopa_za32_f16_m(2, pmh, qn2, av, load_h32(qn2, brow + 4 * kPanel));
                svmopa_za32_f16_m(3, pmh, qn3, av, load_h32(qn3, brow + 6 * kPanel));
            }

            for (int i = 0; i < mlive; ++i) {
                half* row = c + static_cast<int64_t>(m0 * kMr + i) * ldc + n0;
                {
                    svfloat32_t v = svread_hor_za32_f32_m(svundef_f32(), pn0, 0, i);
                    if (!zero_mode) {
                        v = svadd_f32_x(pn0, v,
                                        widen_h16(svld1_f16(svwhilelt_b16(0, live0), row)));
                    }
                    store_op(row, pn0, clamp_v(pn0, v, vlo, vhi), live0);
                }
                if (t1) {
                    half* p = row + kPanel;
                    svfloat32_t w = svread_hor_za32_f32_m(svundef_f32(), pn1, 1, i);
                    if (!zero_mode) {
                        w = svadd_f32_x(pn1, w,
                                        widen_h16(svld1_f16(svwhilelt_b16(0, live1), p)));
                    }
                    store_op(p, pn1, clamp_v(pn1, w, vlo, vhi), live1);
                }
                if (t2) {
                    half* p = row + 2 * kPanel;
                    svfloat32_t w = svread_hor_za32_f32_m(svundef_f32(), pn2, 2, i);
                    if (!zero_mode) {
                        w = svadd_f32_x(pn2, w,
                                        widen_h16(svld1_f16(svwhilelt_b16(0, live2), p)));
                    }
                    store_op(p, pn2, clamp_v(pn2, w, vlo, vhi), live2);
                }
                if (t3) {
                    half* p = row + 3 * kPanel;
                    svfloat32_t w = svread_hor_za32_f32_m(svundef_f32(), pn3, 3, i);
                    if (!zero_mode) {
                        w = svadd_f32_x(pn3, w,
                                        widen_h16(svld1_f16(svwhilelt_b16(0, live3), p)));
                    }
                    store_op(p, pn3, clamp_v(pn3, w, vlo, vhi), live3);
                }
            }
        }
    }
}

template <typename T>
void pack_a_generic(int mc, int kc, T* dst, const T* src, int64_t lds, bool trans) noexcept
{
    for (int m = 0; m < mc; ++m) {
        T* out = dst + (static_cast<int64_t>(m / kMr) * kc) * kMr + (m % kMr);
        if (trans) {
            // A's physical layout is K×M: row k, column m.
            for (int k = 0; k < kc; ++k) {
                out[static_cast<int64_t>(k) * kMr] = src[static_cast<int64_t>(k) * lds + m];
            }
        } else {
            const T* in = src + static_cast<int64_t>(m) * lds;
            for (int k = 0; k < kc; ++k) {
                out[static_cast<int64_t>(k) * kMr] = in[k];
            }
        }
    }
}

/// f16 A: [m/16][ceil(kc/2)][32], lane 2i and 2i+1 holding row i's K pair.
void pack_a_f16_generic(int mc, int kc, half* dst, const half* src, int64_t lds,
                        bool trans) noexcept
{
    const int k_pairs = (kc + 1) / 2;
    for (int m = 0; m < mc; ++m) {
        half* out = dst + static_cast<int64_t>(m / kMr) * k_pairs * (2 * kMr) + (m % kMr) * 2;
        const half* in = src + static_cast<int64_t>(m) * lds;
        for (int kp = 0; kp < k_pairs; ++kp) {
            const int k = 2 * kp;
            half* lane = out + static_cast<int64_t>(kp) * 2 * kMr;
            lane[0] = trans ? src[static_cast<int64_t>(k) * lds + m] : in[k];
            lane[1] = k + 1 < kc ? (trans ? src[static_cast<int64_t>(k + 1) * lds + m]
                                          : in[k + 1])
                                 : half{};
        }
    }
}

template <typename T>
void pack_b_generic(int nc, int kc, T* dst, const T* src, int64_t ldb, bool trans) noexcept
{
    const int n_blocks = (nc + kNB - 1) / kNB;
    for (int nb = 0; nb < n_blocks; ++nb) {
        const int n0 = nb * kNB;
        const int cols = std::min(kNB, nc - n0);
        T* blk = dst + static_cast<int64_t>(nb) * kc * kNB;
        for (int k = 0; k < kc; ++k) {
            T* out = blk + static_cast<int64_t>(k) * kNB;
            if (trans) {
                // B's physical layout is N×K: row n, column k.
                for (int j = 0; j < cols; ++j) {
                    out[j] = src[static_cast<int64_t>(n0 + j) * ldb + k];
                }
            } else {
                const T* in = src + static_cast<int64_t>(k) * ldb + n0;
                for (int j = 0; j < cols; ++j) {
                    out[j] = in[j];
                }
            }
            for (int j = cols; j < kNB; ++j) {
                out[j] = T{};
            }
        }
    }
}

/// f16 B: [n/64][ceil(kc/2)][128], column j's K pair at j*2 and j*2+1.
void pack_b_f16_generic(int nc, int kc, half* dst, const half* src, int64_t ldb,
                        bool trans) noexcept
{
    const int n_blocks = (nc + kNB - 1) / kNB;
    const int k_pairs = (kc + 1) / 2;
    for (int nb = 0; nb < n_blocks; ++nb) {
        const int n0 = nb * kNB;
        const int cols = std::min(kNB, nc - n0);
        half* blk = dst + static_cast<int64_t>(nb) * k_pairs * (2 * kNB);
        for (int kp = 0; kp < k_pairs; ++kp) {
            const int k = 2 * kp;
            half* out = blk + static_cast<int64_t>(kp) * 2 * kNB;
            if (trans) {
                for (int j = 0; j < cols; ++j) {
                    out[2 * j] = src[static_cast<int64_t>(n0 + j) * ldb + k];
                    out[2 * j + 1] =
                        k + 1 < kc ? src[static_cast<int64_t>(n0 + j) * ldb + k + 1] : half{};
                }
            } else {
                // Rows k and k+1 are contiguous, so the pair interleave is one
                // zip of 8 columns at a time. This pack is on the critical path
                // of a prefill (small M), where B's traffic is not amortised.
                const half* in0 = src + static_cast<int64_t>(k) * ldb + n0;
                const half* in1 = k + 1 < kc ? in0 + ldb : nullptr;
                int j = 0;
                for (; j + 8 <= cols; j += 8) {
                    const float16x8_t a =
                        vld1q_f16(reinterpret_cast<const float16_t*>(in0 + j));
                    const float16x8_t b =
                        in1 ? vld1q_f16(reinterpret_cast<const float16_t*>(in1 + j)) :
                              vdupq_n_f16(0.0f);
                    vst1q_f16(reinterpret_cast<float16_t*>(out + 2 * j), vzip1q_f16(a, b));
                    vst1q_f16(reinterpret_cast<float16_t*>(out + 2 * j + 8), vzip2q_f16(a, b));
                }
                for (; j < cols; ++j) {
                    out[2 * j] = in0[j];
                    out[2 * j + 1] = in1 ? in1[j] : half{};
                }
            }
            for (int j = cols; j < kNB; ++j) {
                out[2 * j] = half{};
                out[2 * j + 1] = half{};
            }
        }
    }
}

/// int8: one SMOPA per K group of four per tile. The lane groups are the tile
/// rows (a 4-way dot over the group's four K), so the predicate is b8 and the
/// row extent is 4*mlive lanes — a b32 predicate would leave most of the tile
/// inactive and silently drop products.
__arm_locally_streaming __arm_new("za")
void gemm_i8_impl(int mc, int nc, int kc, int32_t* c, int ldc, const int8_t* a, const int8_t* b,
                  bool zero_mode, int32_t lo, int32_t hi)
{
    const int m_tiles = (mc + kMr - 1) / kMr;
    const int n_blocks = (nc + kNB - 1) / kNB;
    const int k_groups = (kc + 3) / 4;
    const svint32_t vlo = svdup_n_s32(lo);
    const svint32_t vhi = svdup_n_s32(hi);
    for (int nb = 0; nb < n_blocks; ++nb) {
        const int n0 = nb * kNB;
        const bool t1 = n0 + kPanel < nc;
        const bool t2 = n0 + 2 * kPanel < nc;
        const bool t3 = n0 + 3 * kPanel < nc;
        const int live0 = std::min(kPanel, nc - n0);
        const int live1 = t1 ? std::min(kPanel, nc - n0 - kPanel) : 0;
        const int live2 = t2 ? std::min(kPanel, nc - n0 - 2 * kPanel) : 0;
        const int live3 = t3 ? std::min(kPanel, nc - n0 - 3 * kPanel) : 0;
        const svbool_t pn0 = svwhilelt_b8(0, 4 * live0);
        const svbool_t pn1 = t1 ? svwhilelt_b8(0, 4 * live1) : svpfalse_b();
        const svbool_t pn2 = t2 ? svwhilelt_b8(0, 4 * live2) : svpfalse_b();
        const svbool_t pn3 = t3 ? svwhilelt_b8(0, 4 * live3) : svpfalse_b();
        const svbool_t s0 = svwhilelt_b32(0, live0);
        const svbool_t s1 = t1 ? svwhilelt_b32(0, live1) : svpfalse_b();
        const svbool_t s2 = t2 ? svwhilelt_b32(0, live2) : svpfalse_b();
        const svbool_t s3 = t3 ? svwhilelt_b32(0, live3) : svpfalse_b();
        const int8_t* bp = b + static_cast<int64_t>(nb) * k_groups * (4 * kNB);

        for (int m0 = 0; m0 < m_tiles; ++m0) {
            const int mlive = std::min(kMr, mc - m0 * kMr);
            const svbool_t pm = svwhilelt_b8(0, 4 * mlive);
            const int8_t* ap = a + static_cast<int64_t>(m0) * k_groups * 64;

            svzero_za();
            const int8_t* brow = bp;
            for (int g = 0; g < k_groups; ++g, brow += 4 * kNB) {
                const svint8_t av = svld1_s8(pm, ap + static_cast<int64_t>(g) * 64);
                svmopa_za32_s8_m(0, pm, pn0, av, svld1_s8(pn0, brow));
                svmopa_za32_s8_m(1, pm, pn1, av, svld1_s8(pn1, brow + kPanel * 4));
                svmopa_za32_s8_m(2, pm, pn2, av, svld1_s8(pn2, brow + 2 * kPanel * 4));
                svmopa_za32_s8_m(3, pm, pn3, av, svld1_s8(pn3, brow + 3 * kPanel * 4));
            }

            for (int i = 0; i < mlive; ++i) {
                int32_t* row = c + static_cast<int64_t>(m0 * kMr + i) * ldc + n0;
                {
                    svint32_t v = svread_hor_za32_s32_m(svundef_s32(), s0, 0, i);
                    if (!zero_mode) { v = svadd_s32_x(s0, v, svld1_s32(s0, row)); }
                    svst1_s32(s0, row, svmin_s32_x(s0, svmax_s32_x(s0, v, vlo), vhi));
                }
                if (t1) {
                    int32_t* p = row + kPanel;
                    svint32_t v = svread_hor_za32_s32_m(svundef_s32(), s1, 1, i);
                    if (!zero_mode) { v = svadd_s32_x(s1, v, svld1_s32(s1, p)); }
                    svst1_s32(s1, p, svmin_s32_x(s1, svmax_s32_x(s1, v, vlo), vhi));
                }
                if (t2) {
                    int32_t* p = row + 2 * kPanel;
                    svint32_t v = svread_hor_za32_s32_m(svundef_s32(), s2, 2, i);
                    if (!zero_mode) { v = svadd_s32_x(s2, v, svld1_s32(s2, p)); }
                    svst1_s32(s2, p, svmin_s32_x(s2, svmax_s32_x(s2, v, vlo), vhi));
                }
                if (t3) {
                    int32_t* p = row + 3 * kPanel;
                    svint32_t v = svread_hor_za32_s32_m(svundef_s32(), s3, 3, i);
                    if (!zero_mode) { v = svadd_s32_x(s3, v, svld1_s32(s3, p)); }
                    svst1_s32(s3, p, svmin_s32_x(s3, svmax_s32_x(s3, v, vlo), vhi));
                }
            }
        }
    }
}

} // namespace

bool available() noexcept
{
    static const bool on = [] {
        if (!simd::cpu_has_sme()) {
            return false;
        }
        // The kernels assume 16 f32 lanes; a wider SVL needs a retiled kernel.
        if (simd::cpu_sme_svl() != 0 && simd::cpu_sme_svl() != 64) {
            return false;
        }
        return probe_svl() == 64;
    }();
    // The experiment knob is read per call so a bench or test can flip it
    // between calls on one binary; the hardware verdict above is cached.
    return on && ::nnops::detail::disable_sme.load(std::memory_order_relaxed) == 0;
}

void pack_a_f32(int mc, int kc, float* dst, const float* src, int64_t lds, bool trans) noexcept
{
    pack_a_generic(mc, kc, dst, src, lds, trans);
}

void pack_a_f16(int mc, int kc, half* dst, const half* src, int64_t lds, bool trans) noexcept
{
    pack_a_f16_generic(mc, kc, dst, src, lds, trans);
}

void pack_b_f32(int nc, int kc, float* dst, const float* src, int64_t ldb, bool trans) noexcept
{
    pack_b_generic(nc, kc, dst, src, ldb, trans);
}

void pack_b_f16(int nc, int kc, half* dst, const half* src, int64_t ldb, bool trans) noexcept
{
    pack_b_f16_generic(nc, kc, dst, src, ldb, trans);
}

void pack_a_i8(int mc, int kc, int8_t* dst, const int8_t* src, int64_t lds, bool trans,
               bool xor_bias) noexcept
{
    const int k_groups = (kc + 3) / 4;
    for (int m = 0; m < mc; ++m) {
        int8_t* out = dst + static_cast<int64_t>(m / kMr) * k_groups * 64 + (m % kMr) * 4;
        for (int g = 0; g < k_groups; ++g) {
            for (int p = 0; p < 4; ++p) {
                const int k = 4 * g + p;
                int8_t v = 0;
                if (k < kc) {
                    v = trans ? src[static_cast<int64_t>(k) * lds + m] :
                                src[static_cast<int64_t>(m) * lds + k];
                    if (xor_bias) { v = static_cast<int8_t>(static_cast<uint8_t>(v) ^ 0x80u); }
                }
                out[static_cast<int64_t>(g) * 64 + p] = v;
            }
        }
    }
}

void pack_b_i8(int nc, int kc, int8_t* dst, const int8_t* src, int64_t ldb, bool trans) noexcept
{
    const int n_blocks = (nc + kNB - 1) / kNB;
    const int k_groups = (kc + 3) / 4;
    for (int nb = 0; nb < n_blocks; ++nb) {
        const int n0 = nb * kNB;
        const int cols = std::min(kNB, nc - n0);
        int8_t* blk = dst + static_cast<int64_t>(nb) * k_groups * (4 * kNB);
        for (int g = 0; g < k_groups; ++g) {
            int8_t* out = blk + static_cast<int64_t>(g) * (4 * kNB);
            for (int j = 0; j < kNB; ++j) {
                int8_t* cell = out + static_cast<int64_t>(j / kPanel) * (4 * kPanel) +
                               (j % kPanel) * 4;
                for (int p = 0; p < 4; ++p) {
                    const int k = 4 * g + p;
                    int8_t v = 0;
                    if (j < cols && k < kc) {
                        v = trans ? src[static_cast<int64_t>(n0 + j) * ldb + k] :
                                    src[static_cast<int64_t>(k) * ldb + n0 + j];
                    }
                    cell[p] = v;
                }
            }
        }
    }
}

void gemm_i8(int mc, int nc, int kc, int32_t* c, int ldc, const int8_t* a, const int8_t* b,
             bool zero_mode, int32_t lo, int32_t hi) noexcept
{
    gemm_i8_impl(mc, nc, kc, c, ldc, a, b, zero_mode, lo, hi);
}

void gemm_f32(int mc, int nc, int kc, float* c, int ldc, const float* a, const float* b,
              bool zero_mode, float lo, float hi, const float* col_bias,
              int64_t b_block_stride) noexcept
{
    gemm_f32_impl(mc, nc, kc, c, ldc, a, b, zero_mode, lo, hi, col_bias, b_block_stride);
}

void gemm_f16(int mc, int nc, int kc, half* c, int ldc, const half* a, const half* b,
              bool zero_mode, float lo, float hi, const half* col_bias,
              int64_t b_block_stride) noexcept
{
    gemm_f16_impl(mc, nc, kc, c, ldc, a, b, zero_mode, lo, hi, col_bias, b_block_stride);
}

} // namespace nnops::backend::cpu::sme

#endif // NNOPS_ARCH_SME
