#pragma once
/// @file mma_bf16.hpp
/// @brief AArch64 bfloat16 MMA kernels over uniform 16-wide B panels.
///
/// The tile is the same runtime-M shape as mma_f16.hpp: one kernel per row-pair
/// count (the {8, 4, 1} rungs are 4, 2 and 1 pairs), `nc` columns in 16-wide
/// panels, and rows past `mr` alias the last live row in both A and C, so a
/// partial-M tile needs no guards.
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
/// Shape of each kernel, mirroring mma_f16.hpp: named accumulators and the
/// whole k-group body written out. mr8 runs one 8-column block of 4×4
/// accumulators — 16 vectors, MLAS's Sbgemm tile — while mr4 and mr1 take a
/// full 16-column panel (2×8 and 1×8 accumulators) so a panel costs one pass.
/// The k-group loop itself stays a loop: BFMMLA threads every multiply-add into
/// one accumulator, so there is no independent work to interleave and
/// unrolling only lengthens the dependent chain.
///
/// Reference: onnxruntime MLAS SbgemmKernelNeon.S (8 rows × 8 columns of
/// 2×2 BFMMLA accumulators, bfcvtn on the A side).

#include <algorithm>
#include <array>
#include <arm_neon.h>

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
__attribute__((target("bf16")))
inline bfloat16x8_t bf16_a_pair(const bfloat16* r0, const bfloat16* r1) noexcept {
    return vreinterpretq_bf16_u16(
        vcombine_u16(vld1_u16(reinterpret_cast<const uint16_t*>(r0)),
                     vld1_u16(reinterpret_cast<const uint16_t*>(r1))));
}

/// A operand for the short final k-group: @p rem (< 4) values, zero past them.
__attribute__((target("bf16")))
inline bfloat16x8_t bf16_a_pair_tail(const bfloat16* r0, const bfloat16* r1,
                                     int rem) noexcept {
    uint16_t tmp[8] = {};
    for (int j = 0; j < rem; ++j) {
        tmp[j] = r0[j].bits;
        tmp[4 + j] = r1[j].bits;
    }
    return vreinterpretq_bf16_u16(vld1q_u16(tmp));
}

/// B operand out of a packed panel: k-group @p g, colpair @p cp (0..7, i.e. the
/// whole 16-column panel; the low 4 are the block's first 8 columns).
__attribute__((target("bf16")))
inline bfloat16x8_t bf16_b_panel(const bfloat16* b, int g, int cp) noexcept {
    return vreinterpretq_bf16_u16(
        vld1q_u16(reinterpret_cast<const uint16_t*>(b) + g * bf16_b_k4_elems + cp * 8));
}

/// B operand out of a caller's matrix, rows @p ldb apart, full k-group and
/// in-range columns: only the gather is left.
__attribute__((target("bf16")))
inline bfloat16x8_t bf16_b_stride(const bfloat16* b, int64_t ldb, int g,
                                  int cp) noexcept {
    uint16_t tmp[8];
    for (int lane = 0; lane < 8; ++lane) {
        tmp[lane] =
            b[static_cast<int64_t>(g * 4 + (lane & 3)) * ldb + 2 * cp + (lane >> 2)].bits;
    }
    return vreinterpretq_bf16_u16(vld1q_u16(tmp));
}

/// The same gather with the bounds checks: rows past @p kc and columns past
/// @p nc read as zero.
__attribute__((target("bf16")))
inline bfloat16x8_t bf16_b_stride_tail(const bfloat16* b, int64_t ldb, int g,
                                       int cp, int kc, int nc) noexcept {
    uint16_t tmp[8] = {};
    for (int lane = 0; lane < 8; ++lane) {
        const int k = g * 4 + (lane & 3);
        const int col = 2 * cp + (lane >> 2);
        if (k < kc && col < nc) {
            tmp[lane] = b[static_cast<int64_t>(k) * ldb + col].bits;
        }
    }
    return vreinterpretq_bf16_u16(vld1q_u16(tmp));
}

// =========================================================================
//  Accumulator seeding, gathering and storing
// =========================================================================

/// One column pair's accumulator, seeded. Lane order is BFMMLA's: the row
/// pair's two columns for row 0, then the same two for row 1. A row past `mr`
/// goes through the same alias index as A and C, so `bias[r]` stays in range.
template <int seed>
__attribute__((target("bf16")))
inline float32x4_t bf16_seed(const bfloat16* bias, int r0, int r1,
                             int col) noexcept {
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
__attribute__((target("bf16")))
inline sims::v_f32x8 bf16_gather(bool hi, float32x4_t a0, float32x4_t a1,
                                 float32x4_t a2, float32x4_t a3) noexcept {
    const float32x2_t x0 = hi ? vget_high_f32(a0) : vget_low_f32(a0);
    const float32x2_t x1 = hi ? vget_high_f32(a1) : vget_low_f32(a1);
    const float32x2_t x2 = hi ? vget_high_f32(a2) : vget_low_f32(a2);
    const float32x2_t x3 = hi ? vget_high_f32(a3) : vget_low_f32(a3);
    return sims::v_f32x8(vcombine_f32(x0, x1), vcombine_f32(x2, x3));
}

/// The C read that has to happen before any store of the block.
__attribute__((target("bf16")))
inline sims::v_f32x8 bf16_get8(const bfloat16* p) noexcept {
    return sims::v_cvt_bf16_to_f32(reinterpret_cast<const uint16_t*>(p));
}

/// C add, clamp, narrow, store, for one row's 8 columns.
template <bool zero_mode>
__attribute__((target("bf16")))
inline void bf16_put8(bfloat16* NNOPS_RESTRICT p, sims::v_f32x8 v,
                      const sims::v_f32x8& cur, const sims::v_f32x8& vmin,
                      const sims::v_f32x8& vmax) noexcept {
    if constexpr (!zero_mode) { v = sims::v_add(v, cur); }
    sims::v_cvt_f32_to_bf16(reinterpret_cast<uint16_t*>(p),
                            sims::v_min(sims::v_max(v, vmin), vmax));
}

// =========================================================================
//  Tail blocks — generic, since at most one panel per call lands here
// =========================================================================

/// One live row of a store block: bias seed, the C read, clamp, narrow.
template <bool zero_mode, int seed>
__attribute__((target("bf16")))
inline void bf16_store_row(bfloat16* NNOPS_RESTRICT cp, int n, int rem, int r,
                           const bfloat16* bias, sims::v_f32x8 v,
                           const sims::v_f32x8& vmin,
                           const sims::v_f32x8& vmax) noexcept {
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
            for (int i = 0; i < rem; ++i) { cur[i] = static_cast<float>(cp[i]); }
            v = sims::v_add(v, sims::v_load(cur));
        }
    }
    v = sims::v_min(sims::v_max(v, vmin), vmax);

    if (rem >= 8) {
        sims::v_cvt_f32_to_bf16(reinterpret_cast<uint16_t*>(cp), v);
    } else {
        float lane[8];
        sims::v_store(lane, v);
        for (int i = 0; i < rem; ++i) { cp[i] = bfloat16(lane[i]); }
    }
}

/// Store one 8-column block, both rows of every pair spelled out rather than
/// looped over: @p rem is the block's live width (the panel's columns past it
/// are dropped).
template <int PAIRS, bool zero_mode, int seed>
__attribute__((target("bf16")))
inline void bf16_store_block(int mr, int n, int rem,
                             bfloat16* const (&c_rows)[2 * PAIRS],
                             const float32x4_t (&acc)[PAIRS][4],
                             float lo, float hi, const bfloat16* bias) noexcept {
    const sims::v_f32x8 vmin = sims::v_set1_f32x8(lo);
    const sims::v_f32x8 vmax = sims::v_set1_f32x8(hi);
    for (int p = 0; p < PAIRS; ++p) {
        // Each accumulator holds a column pair for both rows of the pair: an
        // 8-column block is four halves joined, the low one for the pair's
        // first row and the high one for its second.
        if (2 * p < mr) {
            bf16_store_row<zero_mode, seed>(
                c_rows[2 * p] + n, n, rem, 2 * p, bias,
                bf16_gather(false, acc[p][0], acc[p][1], acc[p][2], acc[p][3]),
                vmin, vmax);
        }
        if (2 * p + 1 < mr) {
            bf16_store_row<zero_mode, seed>(
                c_rows[2 * p + 1] + n, n, rem, 2 * p + 1, bias,
                bf16_gather(true, acc[p][0], acc[p][1], acc[p][2], acc[p][3]),
                vmin, vmax);
        }
    }
}

/// The two routes' row pointers: a row past `mr` aliases the last live one, so
/// a partial panel stays inside its own rows.
template <int PAIRS>
inline void bf16_rows(const bfloat16* a, int lda, int mr,
                      bfloat16* c, int ldc,
                      const bfloat16* (&a_rows)[2 * PAIRS],
                      bfloat16* (&c_rows)[2 * PAIRS]) noexcept {
    for (int r = 0; r < 2 * PAIRS; ++r) {
        const int i = (r < mr) ? r : mr - 1;
        a_rows[r] = a + static_cast<int64_t>(i) * lda;
        c_rows[r] = c + static_cast<int64_t>(i) * ldc;
    }
}

/// One 8-column block, array form. The unrolled kernels hand their leftover
/// columns (fewer than 8 for mr8, fewer than 16 for mr4/mr1) to this.
template <int PAIRS, bool Strided, bool zero_mode, int seed>
__attribute__((target("bf16")))
inline void bf16_block8(int mr, int n, int nc, int kc, bfloat16* c, int ldc,
                        const bfloat16* a, int lda,
                        const bfloat16* b, int64_t ldb,
                        float lo, float hi, const bfloat16* bias) noexcept {
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
        for (int cp = 0; cp < 4; ++cp) { acc[p][cp] = vdupq_n_f32(0.0f); }
    }

    for (int g = 0; g < kgroups; ++g) {
        bfloat16x8_t bop[4];
        for (int cp = 0; cp < 4; ++cp) {
            if constexpr (Strided) {
                bop[cp] = (nblk < 8)
                              ? bf16_b_stride_tail(b, ldb, g, blk * 4 + cp, kc, nloc)
                              : bf16_b_stride(b, ldb, g, blk * 4 + cp);
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
            const bfloat16x8_t aop =
                bf16_a_pair_tail(a_rows[2 * p], a_rows[2 * p + 1], ktail);
            for (int cp = 0; cp < 4; ++cp) {
                acc[p][cp] = vbfmmlaq_f32(acc[p][cp], aop, bop[cp]);
            }
        }
    }

    bf16_store_block<PAIRS, zero_mode, seed>(mr, n, std::min(8, nc - n), c_rows,
                                             acc, lo, hi, bias);
}

/// The leftover columns of a tile: 8-column blocks from @p n, starting at the
/// panel @p b already points at. One panel is 16 columns, so an in-place B
/// (Strided) steps 16 elements to the next one, not 16 k-rows.
template <int PAIRS, bool Strided, bool zero_mode, int seed>
__attribute__((target("bf16")))
inline void bf16_tail(int mr, int n, int nc, int kc, bfloat16* c, int ldc,
                      const bfloat16* a, int lda, const bfloat16* b, int64_t ldb,
                      float lo, float hi, const bfloat16* bias) noexcept {
    const int64_t advance = Strided ? 16 : bf16_b_panel_stride(kc);
    for (; n < nc; n += 8) {
        bf16_block8<PAIRS, Strided, zero_mode, seed>(mr, n, nc, kc, c, ldc, a, lda,
                                                     b, ldb, lo, hi, bias);
        if (n & 8) { b += advance; }
    }
}

// =========================================================================
//  mr8 — 8 rows × 8 columns, 16 accumulators
// =========================================================================

template <bool Strided, bool zero_mode, int seed>
__attribute__((target("bf16")))
inline void bf16_kernel_mr8(int mr, int nc, int kc, bfloat16* NNOPS_RESTRICT C,
                            int ldc, const bfloat16* NNOPS_RESTRICT A, int lda,
                            const bfloat16* NNOPS_RESTRICT B, int64_t ldb,
                            float lo, float hi,
                            const bfloat16* NNOPS_RESTRICT bias) noexcept {
    bfloat16* NNOPS_RESTRICT c0 = C + 0 * ldc;
    bfloat16* NNOPS_RESTRICT c1 = (mr > 1) ? C + 1 * ldc : c0;
    bfloat16* NNOPS_RESTRICT c2 = (mr > 2) ? C + 2 * ldc : c1;
    bfloat16* NNOPS_RESTRICT c3 = (mr > 3) ? C + 3 * ldc : c2;
    bfloat16* NNOPS_RESTRICT c4 = (mr > 4) ? C + 4 * ldc : c3;
    bfloat16* NNOPS_RESTRICT c5 = (mr > 5) ? C + 5 * ldc : c4;
    bfloat16* NNOPS_RESTRICT c6 = (mr > 6) ? C + 6 * ldc : c5;
    bfloat16* NNOPS_RESTRICT c7 = (mr > 7) ? C + 7 * ldc : c6;

    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;
    const int r4 = (mr > 4) ? 4 : r3;
    const int r5 = (mr > 5) ? 5 : r4;
    const int r6 = (mr > 6) ? 6 : r5;
    const int r7 = (mr > 7) ? 7 : r6;

    const int kgroups = kc >> 2;
    const int ktail = kc & 3;
    const int64_t advance = Strided ? 16 : bf16_b_panel_stride(kc);
    const sims::v_f32x8 vmin = sims::v_set1_f32x8(lo);
    const sims::v_f32x8 vmax = sims::v_set1_f32x8(hi);

    int n = 0;
    for (; n + 8 <= nc; n += 8) {
        const int blk = (n & 8) ? 1 : 0;

        float32x4_t c00 = bf16_seed<seed>(bias, 0, r1, n + 0);
        float32x4_t c01 = bf16_seed<seed>(bias, 0, r1, n + 2);
        float32x4_t c02 = bf16_seed<seed>(bias, 0, r1, n + 4);
        float32x4_t c03 = bf16_seed<seed>(bias, 0, r1, n + 6);
        float32x4_t c10 = bf16_seed<seed>(bias, r2, r3, n + 0);
        float32x4_t c11 = bf16_seed<seed>(bias, r2, r3, n + 2);
        float32x4_t c12 = bf16_seed<seed>(bias, r2, r3, n + 4);
        float32x4_t c13 = bf16_seed<seed>(bias, r2, r3, n + 6);
        float32x4_t c20 = bf16_seed<seed>(bias, r4, r5, n + 0);
        float32x4_t c21 = bf16_seed<seed>(bias, r4, r5, n + 2);
        float32x4_t c22 = bf16_seed<seed>(bias, r4, r5, n + 4);
        float32x4_t c23 = bf16_seed<seed>(bias, r4, r5, n + 6);
        float32x4_t c30 = bf16_seed<seed>(bias, r6, r7, n + 0);
        float32x4_t c31 = bf16_seed<seed>(bias, r6, r7, n + 2);
        float32x4_t c32 = bf16_seed<seed>(bias, r6, r7, n + 4);
        float32x4_t c33 = bf16_seed<seed>(bias, r6, r7, n + 6);

        const bfloat16* NNOPS_RESTRICT b = B;
        const bfloat16* NNOPS_RESTRICT p0 = A + 0 * lda;
        const bfloat16* NNOPS_RESTRICT p1 = A + r1 * lda;
        const bfloat16* NNOPS_RESTRICT p2 = A + r2 * lda;
        const bfloat16* NNOPS_RESTRICT p3 = A + r3 * lda;
        const bfloat16* NNOPS_RESTRICT p4 = A + r4 * lda;
        const bfloat16* NNOPS_RESTRICT p5 = A + r5 * lda;
        const bfloat16* NNOPS_RESTRICT p6 = A + r6 * lda;
        const bfloat16* NNOPS_RESTRICT p7 = A + r7 * lda;

        for (int g = 0; g < kgroups; ++g) {
            bfloat16x8_t b0, b1, b2, b3;
            if constexpr (Strided) {
                b0 = bf16_b_stride(b, ldb, g, blk * 4 + 0);
                b1 = bf16_b_stride(b, ldb, g, blk * 4 + 1);
                b2 = bf16_b_stride(b, ldb, g, blk * 4 + 2);
                b3 = bf16_b_stride(b, ldb, g, blk * 4 + 3);
            } else {
                b0 = bf16_b_panel(b, g, blk * 4 + 0);
                b1 = bf16_b_panel(b, g, blk * 4 + 1);
                b2 = bf16_b_panel(b, g, blk * 4 + 2);
                b3 = bf16_b_panel(b, g, blk * 4 + 3);
            }

            const bfloat16x8_t a01 = bf16_a_pair(p0, p1);
            c00 = vbfmmlaq_f32(c00, a01, b0);
            c01 = vbfmmlaq_f32(c01, a01, b1);
            c02 = vbfmmlaq_f32(c02, a01, b2);
            c03 = vbfmmlaq_f32(c03, a01, b3);

            const bfloat16x8_t a23 = bf16_a_pair(p2, p3);
            c10 = vbfmmlaq_f32(c10, a23, b0);
            c11 = vbfmmlaq_f32(c11, a23, b1);
            c12 = vbfmmlaq_f32(c12, a23, b2);
            c13 = vbfmmlaq_f32(c13, a23, b3);

            const bfloat16x8_t a45 = bf16_a_pair(p4, p5);
            c20 = vbfmmlaq_f32(c20, a45, b0);
            c21 = vbfmmlaq_f32(c21, a45, b1);
            c22 = vbfmmlaq_f32(c22, a45, b2);
            c23 = vbfmmlaq_f32(c23, a45, b3);

            const bfloat16x8_t a67 = bf16_a_pair(p6, p7);
            c30 = vbfmmlaq_f32(c30, a67, b0);
            c31 = vbfmmlaq_f32(c31, a67, b1);
            c32 = vbfmmlaq_f32(c32, a67, b2);
            c33 = vbfmmlaq_f32(c33, a67, b3);

            p0 += 4; p1 += 4; p2 += 4; p3 += 4;
            p4 += 4; p5 += 4; p6 += 4; p7 += 4;
        }

        if (ktail) {
            bfloat16x8_t b0, b1, b2, b3;
            if constexpr (Strided) {
                b0 = bf16_b_stride_tail(b, ldb, kgroups, blk * 4 + 0, kc, nc);
                b1 = bf16_b_stride_tail(b, ldb, kgroups, blk * 4 + 1, kc, nc);
                b2 = bf16_b_stride_tail(b, ldb, kgroups, blk * 4 + 2, kc, nc);
                b3 = bf16_b_stride_tail(b, ldb, kgroups, blk * 4 + 3, kc, nc);
            } else {
                b0 = bf16_b_panel(b, kgroups, blk * 4 + 0);
                b1 = bf16_b_panel(b, kgroups, blk * 4 + 1);
                b2 = bf16_b_panel(b, kgroups, blk * 4 + 2);
                b3 = bf16_b_panel(b, kgroups, blk * 4 + 3);
            }

            const bfloat16x8_t a01 = bf16_a_pair_tail(p0, p1, ktail);
            c00 = vbfmmlaq_f32(c00, a01, b0);
            c01 = vbfmmlaq_f32(c01, a01, b1);
            c02 = vbfmmlaq_f32(c02, a01, b2);
            c03 = vbfmmlaq_f32(c03, a01, b3);

            const bfloat16x8_t a23 = bf16_a_pair_tail(p2, p3, ktail);
            c10 = vbfmmlaq_f32(c10, a23, b0);
            c11 = vbfmmlaq_f32(c11, a23, b1);
            c12 = vbfmmlaq_f32(c12, a23, b2);
            c13 = vbfmmlaq_f32(c13, a23, b3);

            const bfloat16x8_t a45 = bf16_a_pair_tail(p4, p5, ktail);
            c20 = vbfmmlaq_f32(c20, a45, b0);
            c21 = vbfmmlaq_f32(c21, a45, b1);
            c22 = vbfmmlaq_f32(c22, a45, b2);
            c23 = vbfmmlaq_f32(c23, a45, b3);

            const bfloat16x8_t a67 = bf16_a_pair_tail(p6, p7, ktail);
            c30 = vbfmmlaq_f32(c30, a67, b0);
            c31 = vbfmmlaq_f32(c31, a67, b1);
            c32 = vbfmmlaq_f32(c32, a67, b2);
            c33 = vbfmmlaq_f32(c33, a67, b3);
        }

        // Every live row's C is read before the first store: the rows past
        // `mr` share the last live row's address and write its value back.
        sims::v_f32x8 k0 = sims::v_set1_f32x8(0.0f);
        sims::v_f32x8 k1 = sims::v_set1_f32x8(0.0f);
        sims::v_f32x8 k2 = sims::v_set1_f32x8(0.0f);
        sims::v_f32x8 k3 = sims::v_set1_f32x8(0.0f);
        sims::v_f32x8 k4 = sims::v_set1_f32x8(0.0f);
        sims::v_f32x8 k5 = sims::v_set1_f32x8(0.0f);
        sims::v_f32x8 k6 = sims::v_set1_f32x8(0.0f);
        sims::v_f32x8 k7 = sims::v_set1_f32x8(0.0f);
        if constexpr (!zero_mode) {
            k0 = bf16_get8(c0 + n);
            k1 = bf16_get8(c1 + n);
            k2 = bf16_get8(c2 + n);
            k3 = bf16_get8(c3 + n);
            k4 = bf16_get8(c4 + n);
            k5 = bf16_get8(c5 + n);
            k6 = bf16_get8(c6 + n);
            k7 = bf16_get8(c7 + n);
        }

        bf16_put8<zero_mode>(c0 + n, bf16_gather(false, c00, c01, c02, c03), k0, vmin, vmax);
        bf16_put8<zero_mode>(c1 + n, bf16_gather(true,  c00, c01, c02, c03), k1, vmin, vmax);
        bf16_put8<zero_mode>(c2 + n, bf16_gather(false, c10, c11, c12, c13), k2, vmin, vmax);
        bf16_put8<zero_mode>(c3 + n, bf16_gather(true,  c10, c11, c12, c13), k3, vmin, vmax);
        bf16_put8<zero_mode>(c4 + n, bf16_gather(false, c20, c21, c22, c23), k4, vmin, vmax);
        bf16_put8<zero_mode>(c5 + n, bf16_gather(true,  c20, c21, c22, c23), k5, vmin, vmax);
        bf16_put8<zero_mode>(c6 + n, bf16_gather(false, c30, c31, c32, c33), k6, vmin, vmax);
        bf16_put8<zero_mode>(c7 + n, bf16_gather(true,  c30, c31, c32, c33), k7, vmin, vmax);

        if (blk) { B += advance; }
    }

    if (n < nc) {
        bf16_tail<4, Strided, zero_mode, seed>(mr, n, nc, kc, C, ldc, A, lda, B,
                                               ldb, lo, hi, bias);
    }
}

// =========================================================================
//  mr4 / mr1 — a whole 16-column panel per pass, 16 and 8 accumulators
// =========================================================================

/// Shared body of the two narrow-M kernels: @p PAIRS row pairs over all 8
/// column pairs of a panel. mr1 keeps one pair and stores one row.
template <int PAIRS, bool Strided, bool zero_mode, int seed>
__attribute__((target("bf16")))
inline void bf16_kernel_wide(int mr, int nc, int kc, bfloat16* NNOPS_RESTRICT C,
                             int ldc, const bfloat16* NNOPS_RESTRICT A, int lda,
                             const bfloat16* NNOPS_RESTRICT B, int64_t ldb,
                             float lo, float hi,
                             const bfloat16* NNOPS_RESTRICT bias) noexcept {
    // A row past `mr` aliases the last live row, so a pair's second half is
    // that row's own value and its store repeats it.
    bfloat16* NNOPS_RESTRICT c0 = C;
    bfloat16* NNOPS_RESTRICT c1 = (mr > 1) ? C + 1 * ldc : c0;
    bfloat16* NNOPS_RESTRICT c2 = (mr > 2) ? C + 2 * ldc : c1;
    bfloat16* NNOPS_RESTRICT c3 = (mr > 3) ? C + 3 * ldc : c2;

    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;

    const int kgroups = kc >> 2;
    const int ktail = kc & 3;
    const int64_t advance = Strided ? 16 : bf16_b_panel_stride(kc);
    const sims::v_f32x8 vmin = sims::v_set1_f32x8(lo);
    const sims::v_f32x8 vmax = sims::v_set1_f32x8(hi);

    int n = 0;
    for (; n + 16 <= nc; n += 16) {
        float32x4_t c00 = bf16_seed<seed>(bias, 0, r1, n + 0);
        float32x4_t c01 = bf16_seed<seed>(bias, 0, r1, n + 2);
        float32x4_t c02 = bf16_seed<seed>(bias, 0, r1, n + 4);
        float32x4_t c03 = bf16_seed<seed>(bias, 0, r1, n + 6);
        float32x4_t c04 = bf16_seed<seed>(bias, 0, r1, n + 8);
        float32x4_t c05 = bf16_seed<seed>(bias, 0, r1, n + 10);
        float32x4_t c06 = bf16_seed<seed>(bias, 0, r1, n + 12);
        float32x4_t c07 = bf16_seed<seed>(bias, 0, r1, n + 14);
        // The second pair is mr1's dead weight: the store branch discards it.
        [[maybe_unused]] float32x4_t c10 = bf16_seed<seed>(bias, r2, r3, n + 0);
        [[maybe_unused]] float32x4_t c11 = bf16_seed<seed>(bias, r2, r3, n + 2);
        [[maybe_unused]] float32x4_t c12 = bf16_seed<seed>(bias, r2, r3, n + 4);
        [[maybe_unused]] float32x4_t c13 = bf16_seed<seed>(bias, r2, r3, n + 6);
        [[maybe_unused]] float32x4_t c14 = bf16_seed<seed>(bias, r2, r3, n + 8);
        [[maybe_unused]] float32x4_t c15 = bf16_seed<seed>(bias, r2, r3, n + 10);
        [[maybe_unused]] float32x4_t c16 = bf16_seed<seed>(bias, r2, r3, n + 12);
        [[maybe_unused]] float32x4_t c17 = bf16_seed<seed>(bias, r2, r3, n + 14);

        const bfloat16* NNOPS_RESTRICT b = B;
        const bfloat16* NNOPS_RESTRICT p0 = A;
        const bfloat16* NNOPS_RESTRICT p1 = A + static_cast<int64_t>(r1) * lda;
        [[maybe_unused]] const bfloat16* NNOPS_RESTRICT p2 =
            A + static_cast<int64_t>(r2) * lda;
        [[maybe_unused]] const bfloat16* NNOPS_RESTRICT p3 =
            A + static_cast<int64_t>(r3) * lda;

        for (int g = 0; g < kgroups; ++g) {
            bfloat16x8_t b0, b1, b2, b3, b4, b5, b6, b7;
            if constexpr (Strided) {
                b0 = bf16_b_stride(b, ldb, g, 0);
                b1 = bf16_b_stride(b, ldb, g, 1);
                b2 = bf16_b_stride(b, ldb, g, 2);
                b3 = bf16_b_stride(b, ldb, g, 3);
                b4 = bf16_b_stride(b, ldb, g, 4);
                b5 = bf16_b_stride(b, ldb, g, 5);
                b6 = bf16_b_stride(b, ldb, g, 6);
                b7 = bf16_b_stride(b, ldb, g, 7);
            } else {
                b0 = bf16_b_panel(b, g, 0);
                b1 = bf16_b_panel(b, g, 1);
                b2 = bf16_b_panel(b, g, 2);
                b3 = bf16_b_panel(b, g, 3);
                b4 = bf16_b_panel(b, g, 4);
                b5 = bf16_b_panel(b, g, 5);
                b6 = bf16_b_panel(b, g, 6);
                b7 = bf16_b_panel(b, g, 7);
            }

            const bfloat16x8_t a01 = bf16_a_pair(p0, p1);
            c00 = vbfmmlaq_f32(c00, a01, b0);
            c01 = vbfmmlaq_f32(c01, a01, b1);
            c02 = vbfmmlaq_f32(c02, a01, b2);
            c03 = vbfmmlaq_f32(c03, a01, b3);
            c04 = vbfmmlaq_f32(c04, a01, b4);
            c05 = vbfmmlaq_f32(c05, a01, b5);
            c06 = vbfmmlaq_f32(c06, a01, b6);
            c07 = vbfmmlaq_f32(c07, a01, b7);

            if constexpr (PAIRS > 1) {
                const bfloat16x8_t a23 = bf16_a_pair(p2, p3);
                c10 = vbfmmlaq_f32(c10, a23, b0);
                c11 = vbfmmlaq_f32(c11, a23, b1);
                c12 = vbfmmlaq_f32(c12, a23, b2);
                c13 = vbfmmlaq_f32(c13, a23, b3);
                c14 = vbfmmlaq_f32(c14, a23, b4);
                c15 = vbfmmlaq_f32(c15, a23, b5);
                c16 = vbfmmlaq_f32(c16, a23, b6);
                c17 = vbfmmlaq_f32(c17, a23, b7);
            }

            p0 += 4; p1 += 4;
            if constexpr (PAIRS > 1) { p2 += 4; p3 += 4; }
        }

        if (ktail) {
            bfloat16x8_t b0, b1, b2, b3, b4, b5, b6, b7;
            if constexpr (Strided) {
                b0 = bf16_b_stride_tail(b, ldb, kgroups, 0, kc, nc);
                b1 = bf16_b_stride_tail(b, ldb, kgroups, 1, kc, nc);
                b2 = bf16_b_stride_tail(b, ldb, kgroups, 2, kc, nc);
                b3 = bf16_b_stride_tail(b, ldb, kgroups, 3, kc, nc);
                b4 = bf16_b_stride_tail(b, ldb, kgroups, 4, kc, nc);
                b5 = bf16_b_stride_tail(b, ldb, kgroups, 5, kc, nc);
                b6 = bf16_b_stride_tail(b, ldb, kgroups, 6, kc, nc);
                b7 = bf16_b_stride_tail(b, ldb, kgroups, 7, kc, nc);
            } else {
                b0 = bf16_b_panel(b, kgroups, 0);
                b1 = bf16_b_panel(b, kgroups, 1);
                b2 = bf16_b_panel(b, kgroups, 2);
                b3 = bf16_b_panel(b, kgroups, 3);
                b4 = bf16_b_panel(b, kgroups, 4);
                b5 = bf16_b_panel(b, kgroups, 5);
                b6 = bf16_b_panel(b, kgroups, 6);
                b7 = bf16_b_panel(b, kgroups, 7);
            }

            const bfloat16x8_t a01 = bf16_a_pair_tail(p0, p1, ktail);
            c00 = vbfmmlaq_f32(c00, a01, b0);
            c01 = vbfmmlaq_f32(c01, a01, b1);
            c02 = vbfmmlaq_f32(c02, a01, b2);
            c03 = vbfmmlaq_f32(c03, a01, b3);
            c04 = vbfmmlaq_f32(c04, a01, b4);
            c05 = vbfmmlaq_f32(c05, a01, b5);
            c06 = vbfmmlaq_f32(c06, a01, b6);
            c07 = vbfmmlaq_f32(c07, a01, b7);

            if constexpr (PAIRS > 1) {
                const bfloat16x8_t a23 = bf16_a_pair_tail(p2, p3, ktail);
                c10 = vbfmmlaq_f32(c10, a23, b0);
                c11 = vbfmmlaq_f32(c11, a23, b1);
                c12 = vbfmmlaq_f32(c12, a23, b2);
                c13 = vbfmmlaq_f32(c13, a23, b3);
                c14 = vbfmmlaq_f32(c14, a23, b4);
                c15 = vbfmmlaq_f32(c15, a23, b5);
                c16 = vbfmmlaq_f32(c16, a23, b6);
                c17 = vbfmmlaq_f32(c17, a23, b7);
            }
        }

        // Both halves of every row read before any store: alias rows share
        // the live row's address and write its value back.
        sims::v_f32x8 ka0 = sims::v_set1_f32x8(0.0f);
        sims::v_f32x8 ka1 = sims::v_set1_f32x8(0.0f);
        sims::v_f32x8 kb0 = sims::v_set1_f32x8(0.0f);
        sims::v_f32x8 kb1 = sims::v_set1_f32x8(0.0f);
        sims::v_f32x8 kc0 = sims::v_set1_f32x8(0.0f);
        sims::v_f32x8 kc1 = sims::v_set1_f32x8(0.0f);
        sims::v_f32x8 kd0 = sims::v_set1_f32x8(0.0f);
        sims::v_f32x8 kd1 = sims::v_set1_f32x8(0.0f);
        if constexpr (!zero_mode) {
            ka0 = bf16_get8(c0 + n);
            ka1 = bf16_get8(c0 + n + 8);
            kb0 = bf16_get8(c1 + n);
            kb1 = bf16_get8(c1 + n + 8);
            if constexpr (PAIRS > 1) {
                kc0 = bf16_get8(c2 + n);
                kc1 = bf16_get8(c2 + n + 8);
                kd0 = bf16_get8(c3 + n);
                kd1 = bf16_get8(c3 + n + 8);
            }
        }

        bf16_put8<zero_mode>(c0 + n, bf16_gather(false, c00, c01, c02, c03),
                             ka0, vmin, vmax);
        bf16_put8<zero_mode>(c0 + n + 8, bf16_gather(false, c04, c05, c06, c07),
                             ka1, vmin, vmax);
        bf16_put8<zero_mode>(c1 + n, bf16_gather(true, c00, c01, c02, c03),
                             kb0, vmin, vmax);
        bf16_put8<zero_mode>(c1 + n + 8, bf16_gather(true, c04, c05, c06, c07),
                             kb1, vmin, vmax);
        if constexpr (PAIRS > 1) {
            bf16_put8<zero_mode>(c2 + n, bf16_gather(false, c10, c11, c12, c13),
                                 kc0, vmin, vmax);
            bf16_put8<zero_mode>(c2 + n + 8, bf16_gather(false, c14, c15, c16, c17),
                                 kc1, vmin, vmax);
            bf16_put8<zero_mode>(c3 + n, bf16_gather(true, c10, c11, c12, c13),
                                 kd0, vmin, vmax);
            bf16_put8<zero_mode>(c3 + n + 8, bf16_gather(true, c14, c15, c16, c17),
                                 kd1, vmin, vmax);
        }

        B += advance;
    }

    if (n < nc) {
        bf16_tail<PAIRS, Strided, zero_mode, seed>(mr, n, nc, kc, C, ldc, A, lda, B,
                                                    ldb, lo, hi, bias);
    }
}

// =========================================================================
//  Widening f32 fallback — the same blocking without FEAT_BF16
// =========================================================================

template <int PAIRS, bool Strided, bool zero_mode, int seed>
inline void bf16_gemm_wide(int mr, int nc, int kc, bfloat16* c, int ldc,
                           const bfloat16* a, int lda,
                           const bfloat16* b, int64_t ldb,
                           float lo, float hi, const bfloat16* bias) noexcept {
    const bfloat16* a_rows[2 * PAIRS];
    bfloat16* c_rows[2 * PAIRS];
    bf16_rows<PAIRS>(a, lda, mr, c, ldc, a_rows, c_rows);

    const int64_t advance = Strided ? 16 : bf16_b_panel_stride(kc);

    for (int n = 0; n < nc; n += 8) {
        const int rem = std::min(8, nc - n);
        const int blk = (n & 8) ? 1 : 0;
        // Columns live in the panel holding n; a caller's matrix ends at nc.
        const int nloc = Strided ? nc - (n & ~8) : nc;

        float32x4_t acc[PAIRS][4];
        for (int p = 0; p < PAIRS; ++p) {
            for (int cp = 0; cp < 4; ++cp) { acc[p][cp] = vdupq_n_f32(0.0f); }
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
                    const float bv[4] = {bcol[2 * cp], bcol[2 * cp + 1],
                                         bcol[2 * cp], bcol[2 * cp + 1]};
                    acc[p][cp] = vfmaq_f32(acc[p][cp], avv, vld1q_f32(bv));
                }
            }
        }

        bf16_store_block<PAIRS, zero_mode, seed>(mr, n, rem, c_rows, acc, lo, hi, bias);
        if (blk == 1) { b += advance; }
    }
}

/// Test/bench hook: force the widening path even on a FEAT_BF16 core, so the
/// fallback is covered on hardware that has BFMMLA.
inline bool bf16_force_wide = false;

template <int PAIRS, bool zero_mode, int seed>
inline void bf16_gemm(int mr, int nc, int kc, bfloat16* c, int ldc,
                      const bfloat16* a, int lda,
                      const bfloat16* b, int64_t ldb,
                      float lo, float hi, const bfloat16* bias) noexcept {
    if (sims::cpu_has_bf16() && !bf16_force_wide) {
        if constexpr (PAIRS == 4) {
            if (ldb > 0) {
                bf16_kernel_mr8<true, zero_mode, seed>(mr, nc, kc, c, ldc, a, lda,
                                                       b, ldb, lo, hi, bias);
            } else {
                bf16_kernel_mr8<false, zero_mode, seed>(mr, nc, kc, c, ldc, a, lda,
                                                        b, ldb, lo, hi, bias);
            }
        } else if constexpr (PAIRS == 2) {
            if (ldb > 0) {
                bf16_kernel_wide<2, true, zero_mode, seed>(mr, nc, kc, c, ldc, a,
                                                           lda, b, ldb, lo, hi, bias);
            } else {
                bf16_kernel_wide<2, false, zero_mode, seed>(mr, nc, kc, c, ldc, a,
                                                            lda, b, ldb, lo, hi, bias);
            }
        } else {
            if (ldb > 0) {
                bf16_kernel_wide<1, true, zero_mode, seed>(mr, nc, kc, c, ldc, a,
                                                           lda, b, ldb, lo, hi, bias);
            } else {
                bf16_kernel_wide<1, false, zero_mode, seed>(mr, nc, kc, c, ldc, a,
                                                            lda, b, ldb, lo, hi, bias);
            }
        }
    } else {
        if (ldb > 0) {
            bf16_gemm_wide<PAIRS, true, zero_mode, seed>(
                mr, nc, kc, c, ldc, a, lda, b, ldb, lo, hi, bias);
        } else {
            bf16_gemm_wide<PAIRS, false, zero_mode, seed>(
                mr, nc, kc, c, ldc, a, lda, b, ldb, lo, hi, bias);
        }
    }
}

// =========================================================================
//  Public entry points — one per B form, as the other dtypes' kernels
// =========================================================================

/// A packed: the panel base walks `kc` apart inside its slot.
template <int PAIRS, bool zero_mode = false, int seed = 0>
inline void mma_pack_mrxncxkc_bf16(int mr, int nc, int kc, bfloat16* C, int ldc,
                                   const bfloat16* A, const bfloat16* B,
                                   float clamp_min, float clamp_max,
                                   const bfloat16* bias = nullptr) noexcept {
    bf16_gemm<PAIRS, zero_mode, seed>(mr, nc, kc, C, ldc, A, kc, B, -1,
                                      clamp_min, clamp_max, bias);
}

/// A read row-major through @p lda; `ldb` is the packed panel (<= -1) or B's
/// own row pitch (> 0).
template <int PAIRS, bool zero_mode = false, int seed = 0>
inline void mma_direct_mrxncxkc_bf16(int mr, int nc, int kc, bfloat16* C, int ldc,
                                     const bfloat16* A, int lda,
                                     const bfloat16* B, int64_t ldb,
                                     float clamp_min, float clamp_max,
                                     const bfloat16* bias = nullptr) noexcept {
    bf16_gemm<PAIRS, zero_mode, seed>(mr, nc, kc, C, ldc, A, lda, B, ldb,
                                      clamp_min, clamp_max, bias);
}

// ---- dispatch tables, consumed by matmul_helper.cpp ---------------------
template <bool Z, int Seed = 0>
inline constexpr auto mma_pack_fns_bf16 = std::array{
    &mma_pack_mrxncxkc_bf16<4, Z, Seed>,
    &mma_pack_mrxncxkc_bf16<2, Z, Seed>,
    &mma_pack_mrxncxkc_bf16<1, Z, Seed>};

template <bool Z, int Seed = 0>
inline constexpr auto mma_direct_fns_bf16 = std::array{
    &mma_direct_mrxncxkc_bf16<4, Z, Seed>,
    &mma_direct_mrxncxkc_bf16<2, Z, Seed>,
    &mma_direct_mrxncxkc_bf16<1, Z, Seed>};

}  // namespace nnops::backend::cpu::aarch64
