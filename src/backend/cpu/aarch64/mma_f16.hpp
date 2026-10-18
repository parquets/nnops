#pragma once
/// @file mma_f16.hpp
/// @brief AArch64 NEON float16 mrxncxkc MMA kernels over uniform 16-wide B.
///
/// The runtime-M kernel: the tile height comes in as `mr` (1..8) and the
/// column count as `nc`, so one kernel serves every panel the largest-first M
/// decomposition emits ({8, 4, 1}) instead of one per width.
///
/// A and B may both be packed (`mma_packed_ab_*`) or only B packed while A is
/// read row-major (`mma_packed_b_*`). Packed B is uniform: ceil(nc / 16)
/// panels, every one a full 16 columns wide with the columns past `nc`
/// zero-filled, each in a mma_pack_align(16*kc*sizeof(float16_t))-byte slot
/// (16*kc halfs). That lets the kernel read 16 halfs per k at every column,
/// so a tail wider than 8 costs one k-loop instead of two.
///
/// Accumulation is in f16, matching the rest of the aarch64 fp16 kernels.
/// Rows past `mr` alias the last live row — the convention pack_trans_* fills
/// on the A side and the C row pointers repeat here — so a partial-M tile
/// needs no per-row guards.

#include <array>
#include <arm_neon.h>
#include <algorithm>
#include "backend/cpu/common/common_utils.hpp"

namespace nnops::backend::cpu::aarch64 {

// =========================================================================
//  mr=8  kernels
// =========================================================================

template <bool zero_mode = false>
inline void mma_packed_ab_mr8_nr16_mrxncxkc_f16(
    int mr, int nc, int kc, float16_t* NNOPS_RESTRICT C, int ldc,
    const float16_t* NNOPS_RESTRICT A,
    const float16_t* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max) noexcept {

    float16_t* NNOPS_RESTRICT c0_ptr = C + 0 * ldc;
    float16_t* NNOPS_RESTRICT c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    float16_t* NNOPS_RESTRICT c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    float16_t* NNOPS_RESTRICT c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;
    float16_t* NNOPS_RESTRICT c4_ptr = (mr > 4) ? C + 4 * ldc : c3_ptr;
    float16_t* NNOPS_RESTRICT c5_ptr = (mr > 5) ? C + 5 * ldc : c4_ptr;
    float16_t* NNOPS_RESTRICT c6_ptr = (mr > 6) ? C + 6 * ldc : c5_ptr;
    float16_t* NNOPS_RESTRICT c7_ptr = (mr > 7) ? C + 7 * ldc : c6_ptr;

    const int ldb = static_cast<int>(mma_pack_align(16 * kc * sizeof(float16_t)) / sizeof(float16_t));
    const float16_t cmin = static_cast<float16_t>(clamp_min);
    const float16_t cmax = static_cast<float16_t>(clamp_max);
    const float16x8_t vmin = vdupq_n_f16(cmin);
    const float16x8_t vmax = vdupq_n_f16(cmax);

    int n = 0;
    for (; n < nc - 15; n += 16) {
        float16x8_t c00 = vdupq_n_f16(0.0f16), c01 = c00;
        float16x8_t c10 = vdupq_n_f16(0.0f16), c11 = c10;
        float16x8_t c20 = vdupq_n_f16(0.0f16), c21 = c20;
        float16x8_t c30 = vdupq_n_f16(0.0f16), c31 = c30;
        float16x8_t c40 = vdupq_n_f16(0.0f16), c41 = c40;
        float16x8_t c50 = vdupq_n_f16(0.0f16), c51 = c50;
        float16x8_t c60 = vdupq_n_f16(0.0f16), c61 = c60;
        float16x8_t c70 = vdupq_n_f16(0.0f16), c71 = c70;

        const float16_t* NNOPS_RESTRICT a_ptr = A;
        const float16_t* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kc; ++k) {
            const float16x8_t b0 = vld1q_f16(b_ptr + 0 * 8);
            const float16x8_t b1 = vld1q_f16(b_ptr + 1 * 8);
            b_ptr += 16;

            const float16x8_t a = vld1q_f16(a_ptr);
            a_ptr += 8;

            c00 = vfmaq_laneq_f16(c00, b0, a, 0);
            c01 = vfmaq_laneq_f16(c01, b1, a, 0);
            c10 = vfmaq_laneq_f16(c10, b0, a, 1);
            c11 = vfmaq_laneq_f16(c11, b1, a, 1);
            c20 = vfmaq_laneq_f16(c20, b0, a, 2);
            c21 = vfmaq_laneq_f16(c21, b1, a, 2);
            c30 = vfmaq_laneq_f16(c30, b0, a, 3);
            c31 = vfmaq_laneq_f16(c31, b1, a, 3);
            c40 = vfmaq_laneq_f16(c40, b0, a, 4);
            c41 = vfmaq_laneq_f16(c41, b1, a, 4);
            c50 = vfmaq_laneq_f16(c50, b0, a, 5);
            c51 = vfmaq_laneq_f16(c51, b1, a, 5);
            c60 = vfmaq_laneq_f16(c60, b0, a, 6);
            c61 = vfmaq_laneq_f16(c61, b1, a, 6);
            c70 = vfmaq_laneq_f16(c70, b0, a, 7);
            c71 = vfmaq_laneq_f16(c71, b1, a, 7);
        }

        if constexpr (!zero_mode) {
            c00 = vaddq_f16(c00, vld1q_f16(c0_ptr + 0 * 8));
            c01 = vaddq_f16(c01, vld1q_f16(c0_ptr + 1 * 8));
            c10 = vaddq_f16(c10, vld1q_f16(c1_ptr + 0 * 8));
            c11 = vaddq_f16(c11, vld1q_f16(c1_ptr + 1 * 8));
            c20 = vaddq_f16(c20, vld1q_f16(c2_ptr + 0 * 8));
            c21 = vaddq_f16(c21, vld1q_f16(c2_ptr + 1 * 8));
            c30 = vaddq_f16(c30, vld1q_f16(c3_ptr + 0 * 8));
            c31 = vaddq_f16(c31, vld1q_f16(c3_ptr + 1 * 8));
            c40 = vaddq_f16(c40, vld1q_f16(c4_ptr + 0 * 8));
            c41 = vaddq_f16(c41, vld1q_f16(c4_ptr + 1 * 8));
            c50 = vaddq_f16(c50, vld1q_f16(c5_ptr + 0 * 8));
            c51 = vaddq_f16(c51, vld1q_f16(c5_ptr + 1 * 8));
            c60 = vaddq_f16(c60, vld1q_f16(c6_ptr + 0 * 8));
            c61 = vaddq_f16(c61, vld1q_f16(c6_ptr + 1 * 8));
            c70 = vaddq_f16(c70, vld1q_f16(c7_ptr + 0 * 8));
            c71 = vaddq_f16(c71, vld1q_f16(c7_ptr + 1 * 8));
        }

        vst1q_f16(c0_ptr + 0 * 8, vminq_f16(vmaxq_f16(c00, vmin), vmax));
        vst1q_f16(c0_ptr + 1 * 8, vminq_f16(vmaxq_f16(c01, vmin), vmax));
        vst1q_f16(c1_ptr + 0 * 8, vminq_f16(vmaxq_f16(c10, vmin), vmax));
        vst1q_f16(c1_ptr + 1 * 8, vminq_f16(vmaxq_f16(c11, vmin), vmax));
        vst1q_f16(c2_ptr + 0 * 8, vminq_f16(vmaxq_f16(c20, vmin), vmax));
        vst1q_f16(c2_ptr + 1 * 8, vminq_f16(vmaxq_f16(c21, vmin), vmax));
        vst1q_f16(c3_ptr + 0 * 8, vminq_f16(vmaxq_f16(c30, vmin), vmax));
        vst1q_f16(c3_ptr + 1 * 8, vminq_f16(vmaxq_f16(c31, vmin), vmax));
        vst1q_f16(c4_ptr + 0 * 8, vminq_f16(vmaxq_f16(c40, vmin), vmax));
        vst1q_f16(c4_ptr + 1 * 8, vminq_f16(vmaxq_f16(c41, vmin), vmax));
        vst1q_f16(c5_ptr + 0 * 8, vminq_f16(vmaxq_f16(c50, vmin), vmax));
        vst1q_f16(c5_ptr + 1 * 8, vminq_f16(vmaxq_f16(c51, vmin), vmax));
        vst1q_f16(c6_ptr + 0 * 8, vminq_f16(vmaxq_f16(c60, vmin), vmax));
        vst1q_f16(c6_ptr + 1 * 8, vminq_f16(vmaxq_f16(c61, vmin), vmax));
        vst1q_f16(c7_ptr + 0 * 8, vminq_f16(vmaxq_f16(c70, vmin), vmax));
        vst1q_f16(c7_ptr + 1 * 8, vminq_f16(vmaxq_f16(c71, vmin), vmax));

        c0_ptr += 16;
        c1_ptr += 16;
        c2_ptr += 16;
        c3_ptr += 16;
        c4_ptr += 16;
        c5_ptr += 16;
        c6_ptr += 16;
        c7_ptr += 16;

        B += ldb;
    }

    const int rem = nc - n;
    if (rem > 0) {
        float16x8_t c00 = vdupq_n_f16(0.0f16), c01 = c00;
        float16x8_t c10 = vdupq_n_f16(0.0f16), c11 = c10;
        float16x8_t c20 = vdupq_n_f16(0.0f16), c21 = c20;
        float16x8_t c30 = vdupq_n_f16(0.0f16), c31 = c30;
        float16x8_t c40 = vdupq_n_f16(0.0f16), c41 = c40;
        float16x8_t c50 = vdupq_n_f16(0.0f16), c51 = c50;
        float16x8_t c60 = vdupq_n_f16(0.0f16), c61 = c60;
        float16x8_t c70 = vdupq_n_f16(0.0f16), c71 = c70;

        const float16_t* NNOPS_RESTRICT a_ptr = A;
        const float16_t* NNOPS_RESTRICT b_ptr = B;
        if (rem > 8) {
            for (int k = 0; k < kc; ++k) {
                const float16x8_t b0 = vld1q_f16(b_ptr + 0 * 8);
                const float16x8_t b1 = vld1q_f16(b_ptr + 1 * 8);
                b_ptr += 16;

                const float16x8_t a = vld1q_f16(a_ptr);
                a_ptr += 8;

                c00 = vfmaq_laneq_f16(c00, b0, a, 0);
                c01 = vfmaq_laneq_f16(c01, b1, a, 0);
                c10 = vfmaq_laneq_f16(c10, b0, a, 1);
                c11 = vfmaq_laneq_f16(c11, b1, a, 1);
                c20 = vfmaq_laneq_f16(c20, b0, a, 2);
                c21 = vfmaq_laneq_f16(c21, b1, a, 2);
                c30 = vfmaq_laneq_f16(c30, b0, a, 3);
                c31 = vfmaq_laneq_f16(c31, b1, a, 3);
                c40 = vfmaq_laneq_f16(c40, b0, a, 4);
                c41 = vfmaq_laneq_f16(c41, b1, a, 4);
                c50 = vfmaq_laneq_f16(c50, b0, a, 5);
                c51 = vfmaq_laneq_f16(c51, b1, a, 5);
                c60 = vfmaq_laneq_f16(c60, b0, a, 6);
                c61 = vfmaq_laneq_f16(c61, b1, a, 6);
                c70 = vfmaq_laneq_f16(c70, b0, a, 7);
                c71 = vfmaq_laneq_f16(c71, b1, a, 7);
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const float16x8_t b0 = vld1q_f16(b_ptr);
                b_ptr += 16;

                const float16x8_t a = vld1q_f16(a_ptr);
                a_ptr += 8;

                c00 = vfmaq_laneq_f16(c00, b0, a, 0);
                c10 = vfmaq_laneq_f16(c10, b0, a, 1);
                c20 = vfmaq_laneq_f16(c20, b0, a, 2);
                c30 = vfmaq_laneq_f16(c30, b0, a, 3);
                c40 = vfmaq_laneq_f16(c40, b0, a, 4);
                c50 = vfmaq_laneq_f16(c50, b0, a, 5);
                c60 = vfmaq_laneq_f16(c60, b0, a, 6);
                c70 = vfmaq_laneq_f16(c70, b0, a, 7);
            }
        }

        const float16x8_t acc[8][2] = {
            {c00, c01}, {c10, c11}, {c20, c21}, {c30, c31},
            {c40, c41}, {c50, c51}, {c60, c61}, {c70, c71},
        };
        float16_t* NNOPS_RESTRICT c_rows[8] = {c0_ptr, c1_ptr, c2_ptr, c3_ptr,
                                               c4_ptr, c5_ptr, c6_ptr, c7_ptr};
        for (int r = 0; r < mr; ++r) {
            float16_t* NNOPS_RESTRICT cp = c_rows[r];
            int rr = rem;
            int g = 0;
            for (; rr >= 8; rr -= 8, ++g) {
                float16x8_t v = acc[r][g];
                if constexpr (!zero_mode) {
                    v = vaddq_f16(v, vld1q_f16(cp));
                }
                vst1q_f16(cp, vminq_f16(vmaxq_f16(v, vmin), vmax));
                cp += 8;
            }
            if (rr > 0) {
                float16_t lane[8];
                vst1q_f16(lane, acc[r][g]);
                for (int i = 0; i < rr; ++i) {
                    float16_t v = lane[i];
                    if constexpr (!zero_mode) {
                        v = v + cp[i];
                    }
                    cp[i] = std::min(std::max(v, cmin), cmax);
                }
            }
        }
    }
}

template <bool zero_mode = false>
inline void mma_packed_b_mr8_nr16_mrxncxkc_f16(
    int mr, int nc, int kc, float16_t* NNOPS_RESTRICT C, int ldc,
    const float16_t* NNOPS_RESTRICT A, int lda,
    const float16_t* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max) noexcept {

    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;
    const int r4 = (mr > 4) ? 4 : r3;
    const int r5 = (mr > 5) ? 5 : r4;
    const int r6 = (mr > 6) ? 6 : r5;
    const int r7 = (mr > 7) ? 7 : r6;

    float16_t* NNOPS_RESTRICT c0_ptr = C + 0 * ldc;
    float16_t* NNOPS_RESTRICT c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    float16_t* NNOPS_RESTRICT c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    float16_t* NNOPS_RESTRICT c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;
    float16_t* NNOPS_RESTRICT c4_ptr = (mr > 4) ? C + 4 * ldc : c3_ptr;
    float16_t* NNOPS_RESTRICT c5_ptr = (mr > 5) ? C + 5 * ldc : c4_ptr;
    float16_t* NNOPS_RESTRICT c6_ptr = (mr > 6) ? C + 6 * ldc : c5_ptr;
    float16_t* NNOPS_RESTRICT c7_ptr = (mr > 7) ? C + 7 * ldc : c6_ptr;

    const int ldb = static_cast<int>(mma_pack_align(16 * kc * sizeof(float16_t)) / sizeof(float16_t));
    const float16_t cmin = static_cast<float16_t>(clamp_min);
    const float16_t cmax = static_cast<float16_t>(clamp_max);
    const float16x8_t vmin = vdupq_n_f16(cmin);
    const float16x8_t vmax = vdupq_n_f16(cmax);

    int n = 0;
    for (; n < nc - 15; n += 16) {
        float16x8_t c00 = vdupq_n_f16(0.0f16), c01 = c00;
        float16x8_t c10 = vdupq_n_f16(0.0f16), c11 = c10;
        float16x8_t c20 = vdupq_n_f16(0.0f16), c21 = c20;
        float16x8_t c30 = vdupq_n_f16(0.0f16), c31 = c30;
        float16x8_t c40 = vdupq_n_f16(0.0f16), c41 = c40;
        float16x8_t c50 = vdupq_n_f16(0.0f16), c51 = c50;
        float16x8_t c60 = vdupq_n_f16(0.0f16), c61 = c60;
        float16x8_t c70 = vdupq_n_f16(0.0f16), c71 = c70;

        const float16_t* NNOPS_RESTRICT a0 = A + 0 * lda;
        const float16_t* NNOPS_RESTRICT a1 = A + r1 * lda;
        const float16_t* NNOPS_RESTRICT a2 = A + r2 * lda;
        const float16_t* NNOPS_RESTRICT a3 = A + r3 * lda;
        const float16_t* NNOPS_RESTRICT a4 = A + r4 * lda;
        const float16_t* NNOPS_RESTRICT a5 = A + r5 * lda;
        const float16_t* NNOPS_RESTRICT a6 = A + r6 * lda;
        const float16_t* NNOPS_RESTRICT a7 = A + r7 * lda;
        const float16_t* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kc; ++k) {
            const float16x8_t b0 = vld1q_f16(b_ptr + 0 * 8);
            const float16x8_t b1 = vld1q_f16(b_ptr + 1 * 8);
            b_ptr += 16;

            const float16_t v0 = *a0++;
            const float16_t v1 = *a1++;
            const float16_t v2 = *a2++;
            const float16_t v3 = *a3++;
            const float16_t v4 = *a4++;
            const float16_t v5 = *a5++;
            const float16_t v6 = *a6++;
            const float16_t v7 = *a7++;

            c00 = vfmaq_n_f16(c00, b0, v0);
            c01 = vfmaq_n_f16(c01, b1, v0);
            c10 = vfmaq_n_f16(c10, b0, v1);
            c11 = vfmaq_n_f16(c11, b1, v1);
            c20 = vfmaq_n_f16(c20, b0, v2);
            c21 = vfmaq_n_f16(c21, b1, v2);
            c30 = vfmaq_n_f16(c30, b0, v3);
            c31 = vfmaq_n_f16(c31, b1, v3);
            c40 = vfmaq_n_f16(c40, b0, v4);
            c41 = vfmaq_n_f16(c41, b1, v4);
            c50 = vfmaq_n_f16(c50, b0, v5);
            c51 = vfmaq_n_f16(c51, b1, v5);
            c60 = vfmaq_n_f16(c60, b0, v6);
            c61 = vfmaq_n_f16(c61, b1, v6);
            c70 = vfmaq_n_f16(c70, b0, v7);
            c71 = vfmaq_n_f16(c71, b1, v7);
        }

        if constexpr (!zero_mode) {
            c00 = vaddq_f16(c00, vld1q_f16(c0_ptr + 0 * 8));
            c01 = vaddq_f16(c01, vld1q_f16(c0_ptr + 1 * 8));
            c10 = vaddq_f16(c10, vld1q_f16(c1_ptr + 0 * 8));
            c11 = vaddq_f16(c11, vld1q_f16(c1_ptr + 1 * 8));
            c20 = vaddq_f16(c20, vld1q_f16(c2_ptr + 0 * 8));
            c21 = vaddq_f16(c21, vld1q_f16(c2_ptr + 1 * 8));
            c30 = vaddq_f16(c30, vld1q_f16(c3_ptr + 0 * 8));
            c31 = vaddq_f16(c31, vld1q_f16(c3_ptr + 1 * 8));
            c40 = vaddq_f16(c40, vld1q_f16(c4_ptr + 0 * 8));
            c41 = vaddq_f16(c41, vld1q_f16(c4_ptr + 1 * 8));
            c50 = vaddq_f16(c50, vld1q_f16(c5_ptr + 0 * 8));
            c51 = vaddq_f16(c51, vld1q_f16(c5_ptr + 1 * 8));
            c60 = vaddq_f16(c60, vld1q_f16(c6_ptr + 0 * 8));
            c61 = vaddq_f16(c61, vld1q_f16(c6_ptr + 1 * 8));
            c70 = vaddq_f16(c70, vld1q_f16(c7_ptr + 0 * 8));
            c71 = vaddq_f16(c71, vld1q_f16(c7_ptr + 1 * 8));
        }

        vst1q_f16(c0_ptr + 0 * 8, vminq_f16(vmaxq_f16(c00, vmin), vmax));
        vst1q_f16(c0_ptr + 1 * 8, vminq_f16(vmaxq_f16(c01, vmin), vmax));
        vst1q_f16(c1_ptr + 0 * 8, vminq_f16(vmaxq_f16(c10, vmin), vmax));
        vst1q_f16(c1_ptr + 1 * 8, vminq_f16(vmaxq_f16(c11, vmin), vmax));
        vst1q_f16(c2_ptr + 0 * 8, vminq_f16(vmaxq_f16(c20, vmin), vmax));
        vst1q_f16(c2_ptr + 1 * 8, vminq_f16(vmaxq_f16(c21, vmin), vmax));
        vst1q_f16(c3_ptr + 0 * 8, vminq_f16(vmaxq_f16(c30, vmin), vmax));
        vst1q_f16(c3_ptr + 1 * 8, vminq_f16(vmaxq_f16(c31, vmin), vmax));
        vst1q_f16(c4_ptr + 0 * 8, vminq_f16(vmaxq_f16(c40, vmin), vmax));
        vst1q_f16(c4_ptr + 1 * 8, vminq_f16(vmaxq_f16(c41, vmin), vmax));
        vst1q_f16(c5_ptr + 0 * 8, vminq_f16(vmaxq_f16(c50, vmin), vmax));
        vst1q_f16(c5_ptr + 1 * 8, vminq_f16(vmaxq_f16(c51, vmin), vmax));
        vst1q_f16(c6_ptr + 0 * 8, vminq_f16(vmaxq_f16(c60, vmin), vmax));
        vst1q_f16(c6_ptr + 1 * 8, vminq_f16(vmaxq_f16(c61, vmin), vmax));
        vst1q_f16(c7_ptr + 0 * 8, vminq_f16(vmaxq_f16(c70, vmin), vmax));
        vst1q_f16(c7_ptr + 1 * 8, vminq_f16(vmaxq_f16(c71, vmin), vmax));

        c0_ptr += 16;
        c1_ptr += 16;
        c2_ptr += 16;
        c3_ptr += 16;
        c4_ptr += 16;
        c5_ptr += 16;
        c6_ptr += 16;
        c7_ptr += 16;

        B += ldb;
    }

    const int rem = nc - n;
    if (rem > 0) {
        float16x8_t c00 = vdupq_n_f16(0.0f16), c01 = c00;
        float16x8_t c10 = vdupq_n_f16(0.0f16), c11 = c10;
        float16x8_t c20 = vdupq_n_f16(0.0f16), c21 = c20;
        float16x8_t c30 = vdupq_n_f16(0.0f16), c31 = c30;
        float16x8_t c40 = vdupq_n_f16(0.0f16), c41 = c40;
        float16x8_t c50 = vdupq_n_f16(0.0f16), c51 = c50;
        float16x8_t c60 = vdupq_n_f16(0.0f16), c61 = c60;
        float16x8_t c70 = vdupq_n_f16(0.0f16), c71 = c70;

        const float16_t* NNOPS_RESTRICT a0 = A + 0 * lda;
        const float16_t* NNOPS_RESTRICT a1 = A + r1 * lda;
        const float16_t* NNOPS_RESTRICT a2 = A + r2 * lda;
        const float16_t* NNOPS_RESTRICT a3 = A + r3 * lda;
        const float16_t* NNOPS_RESTRICT a4 = A + r4 * lda;
        const float16_t* NNOPS_RESTRICT a5 = A + r5 * lda;
        const float16_t* NNOPS_RESTRICT a6 = A + r6 * lda;
        const float16_t* NNOPS_RESTRICT a7 = A + r7 * lda;
        const float16_t* NNOPS_RESTRICT b_ptr = B;
        if (rem > 8) {
            for (int k = 0; k < kc; ++k) {
                const float16x8_t b0 = vld1q_f16(b_ptr + 0 * 8);
                const float16x8_t b1 = vld1q_f16(b_ptr + 1 * 8);
                b_ptr += 16;

                const float16_t v0 = *a0++;
                const float16_t v1 = *a1++;
                const float16_t v2 = *a2++;
                const float16_t v3 = *a3++;
                const float16_t v4 = *a4++;
                const float16_t v5 = *a5++;
                const float16_t v6 = *a6++;
                const float16_t v7 = *a7++;

                c00 = vfmaq_n_f16(c00, b0, v0);
                c01 = vfmaq_n_f16(c01, b1, v0);
                c10 = vfmaq_n_f16(c10, b0, v1);
                c11 = vfmaq_n_f16(c11, b1, v1);
                c20 = vfmaq_n_f16(c20, b0, v2);
                c21 = vfmaq_n_f16(c21, b1, v2);
                c30 = vfmaq_n_f16(c30, b0, v3);
                c31 = vfmaq_n_f16(c31, b1, v3);
                c40 = vfmaq_n_f16(c40, b0, v4);
                c41 = vfmaq_n_f16(c41, b1, v4);
                c50 = vfmaq_n_f16(c50, b0, v5);
                c51 = vfmaq_n_f16(c51, b1, v5);
                c60 = vfmaq_n_f16(c60, b0, v6);
                c61 = vfmaq_n_f16(c61, b1, v6);
                c70 = vfmaq_n_f16(c70, b0, v7);
                c71 = vfmaq_n_f16(c71, b1, v7);
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const float16x8_t b0 = vld1q_f16(b_ptr);
                b_ptr += 16;

                const float16_t v0 = *a0++;
                const float16_t v1 = *a1++;
                const float16_t v2 = *a2++;
                const float16_t v3 = *a3++;
                const float16_t v4 = *a4++;
                const float16_t v5 = *a5++;
                const float16_t v6 = *a6++;
                const float16_t v7 = *a7++;

                c00 = vfmaq_n_f16(c00, b0, v0);
                c10 = vfmaq_n_f16(c10, b0, v1);
                c20 = vfmaq_n_f16(c20, b0, v2);
                c30 = vfmaq_n_f16(c30, b0, v3);
                c40 = vfmaq_n_f16(c40, b0, v4);
                c50 = vfmaq_n_f16(c50, b0, v5);
                c60 = vfmaq_n_f16(c60, b0, v6);
                c70 = vfmaq_n_f16(c70, b0, v7);
            }
        }

        const float16x8_t acc[8][2] = {
            {c00, c01}, {c10, c11}, {c20, c21}, {c30, c31},
            {c40, c41}, {c50, c51}, {c60, c61}, {c70, c71},
        };
        float16_t* NNOPS_RESTRICT c_rows[8] = {c0_ptr, c1_ptr, c2_ptr, c3_ptr,
                                               c4_ptr, c5_ptr, c6_ptr, c7_ptr};
        for (int r = 0; r < mr; ++r) {
            float16_t* NNOPS_RESTRICT cp = c_rows[r];
            int rr = rem;
            int g = 0;
            for (; rr >= 8; rr -= 8, ++g) {
                float16x8_t v = acc[r][g];
                if constexpr (!zero_mode) {
                    v = vaddq_f16(v, vld1q_f16(cp));
                }
                vst1q_f16(cp, vminq_f16(vmaxq_f16(v, vmin), vmax));
                cp += 8;
            }
            if (rr > 0) {
                float16_t lane[8];
                vst1q_f16(lane, acc[r][g]);
                for (int i = 0; i < rr; ++i) {
                    float16_t v = lane[i];
                    if constexpr (!zero_mode) {
                        v = v + cp[i];
                    }
                    cp[i] = std::min(std::max(v, cmin), cmax);
                }
            }
        }
    }
}

// =========================================================================
//  mr=4  kernels
// =========================================================================

template <bool zero_mode = false>
inline void mma_packed_ab_mr4_nr16_mrxncxkc_f16(
    int mr, int nc, int kc, float16_t* NNOPS_RESTRICT C, int ldc,
    const float16_t* NNOPS_RESTRICT A,
    const float16_t* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max) noexcept {

    float16_t* NNOPS_RESTRICT c0_ptr = C + 0 * ldc;
    float16_t* NNOPS_RESTRICT c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    float16_t* NNOPS_RESTRICT c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    float16_t* NNOPS_RESTRICT c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;

    const int ldb = static_cast<int>(mma_pack_align(16 * kc * sizeof(float16_t)) / sizeof(float16_t));
    const float16_t cmin = static_cast<float16_t>(clamp_min);
    const float16_t cmax = static_cast<float16_t>(clamp_max);
    const float16x8_t vmin = vdupq_n_f16(cmin);
    const float16x8_t vmax = vdupq_n_f16(cmax);

    int n = 0;
    for (; n < nc - 15; n += 16) {
        float16x8_t c00 = vdupq_n_f16(0.0f16), c01 = c00;
        float16x8_t c10 = vdupq_n_f16(0.0f16), c11 = c10;
        float16x8_t c20 = vdupq_n_f16(0.0f16), c21 = c20;
        float16x8_t c30 = vdupq_n_f16(0.0f16), c31 = c30;

        const float16_t* NNOPS_RESTRICT a_ptr = A;
        const float16_t* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kc; ++k) {
            const float16x8_t b0 = vld1q_f16(b_ptr + 0 * 8);
            const float16x8_t b1 = vld1q_f16(b_ptr + 1 * 8);
            b_ptr += 16;

            const float16x4_t a = vld1_f16(a_ptr);
            a_ptr += 4;

            c00 = vfmaq_lane_f16(c00, b0, a, 0);
            c01 = vfmaq_lane_f16(c01, b1, a, 0);
            c10 = vfmaq_lane_f16(c10, b0, a, 1);
            c11 = vfmaq_lane_f16(c11, b1, a, 1);
            c20 = vfmaq_lane_f16(c20, b0, a, 2);
            c21 = vfmaq_lane_f16(c21, b1, a, 2);
            c30 = vfmaq_lane_f16(c30, b0, a, 3);
            c31 = vfmaq_lane_f16(c31, b1, a, 3);
        }

        if constexpr (!zero_mode) {
            c00 = vaddq_f16(c00, vld1q_f16(c0_ptr + 0 * 8));
            c01 = vaddq_f16(c01, vld1q_f16(c0_ptr + 1 * 8));
            c10 = vaddq_f16(c10, vld1q_f16(c1_ptr + 0 * 8));
            c11 = vaddq_f16(c11, vld1q_f16(c1_ptr + 1 * 8));
            c20 = vaddq_f16(c20, vld1q_f16(c2_ptr + 0 * 8));
            c21 = vaddq_f16(c21, vld1q_f16(c2_ptr + 1 * 8));
            c30 = vaddq_f16(c30, vld1q_f16(c3_ptr + 0 * 8));
            c31 = vaddq_f16(c31, vld1q_f16(c3_ptr + 1 * 8));
        }

        vst1q_f16(c0_ptr + 0 * 8, vminq_f16(vmaxq_f16(c00, vmin), vmax));
        vst1q_f16(c0_ptr + 1 * 8, vminq_f16(vmaxq_f16(c01, vmin), vmax));
        vst1q_f16(c1_ptr + 0 * 8, vminq_f16(vmaxq_f16(c10, vmin), vmax));
        vst1q_f16(c1_ptr + 1 * 8, vminq_f16(vmaxq_f16(c11, vmin), vmax));
        vst1q_f16(c2_ptr + 0 * 8, vminq_f16(vmaxq_f16(c20, vmin), vmax));
        vst1q_f16(c2_ptr + 1 * 8, vminq_f16(vmaxq_f16(c21, vmin), vmax));
        vst1q_f16(c3_ptr + 0 * 8, vminq_f16(vmaxq_f16(c30, vmin), vmax));
        vst1q_f16(c3_ptr + 1 * 8, vminq_f16(vmaxq_f16(c31, vmin), vmax));

        c0_ptr += 16;
        c1_ptr += 16;
        c2_ptr += 16;
        c3_ptr += 16;

        B += ldb;
    }

    const int rem = nc - n;
    if (rem > 0) {
        float16x8_t c00 = vdupq_n_f16(0.0f16), c01 = c00;
        float16x8_t c10 = vdupq_n_f16(0.0f16), c11 = c10;
        float16x8_t c20 = vdupq_n_f16(0.0f16), c21 = c20;
        float16x8_t c30 = vdupq_n_f16(0.0f16), c31 = c30;

        const float16_t* NNOPS_RESTRICT a_ptr = A;
        const float16_t* NNOPS_RESTRICT b_ptr = B;
        if (rem > 8) {
            for (int k = 0; k < kc; ++k) {
                const float16x8_t b0 = vld1q_f16(b_ptr + 0 * 8);
                const float16x8_t b1 = vld1q_f16(b_ptr + 1 * 8);
                b_ptr += 16;

                const float16x4_t a = vld1_f16(a_ptr);
                a_ptr += 4;

                c00 = vfmaq_lane_f16(c00, b0, a, 0);
                c01 = vfmaq_lane_f16(c01, b1, a, 0);
                c10 = vfmaq_lane_f16(c10, b0, a, 1);
                c11 = vfmaq_lane_f16(c11, b1, a, 1);
                c20 = vfmaq_lane_f16(c20, b0, a, 2);
                c21 = vfmaq_lane_f16(c21, b1, a, 2);
                c30 = vfmaq_lane_f16(c30, b0, a, 3);
                c31 = vfmaq_lane_f16(c31, b1, a, 3);
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const float16x8_t b0 = vld1q_f16(b_ptr);
                b_ptr += 16;

                const float16x4_t a = vld1_f16(a_ptr);
                a_ptr += 4;

                c00 = vfmaq_lane_f16(c00, b0, a, 0);
                c10 = vfmaq_lane_f16(c10, b0, a, 1);
                c20 = vfmaq_lane_f16(c20, b0, a, 2);
                c30 = vfmaq_lane_f16(c30, b0, a, 3);
            }
        }

        const float16x8_t acc[4][2] = {
            {c00, c01}, {c10, c11}, {c20, c21}, {c30, c31},
        };
        float16_t* NNOPS_RESTRICT c_rows[4] = {c0_ptr, c1_ptr, c2_ptr, c3_ptr};
        for (int r = 0; r < mr; ++r) {
            float16_t* NNOPS_RESTRICT cp = c_rows[r];
            int rr = rem;
            int g = 0;
            for (; rr >= 8; rr -= 8, ++g) {
                float16x8_t v = acc[r][g];
                if constexpr (!zero_mode) {
                    v = vaddq_f16(v, vld1q_f16(cp));
                }
                vst1q_f16(cp, vminq_f16(vmaxq_f16(v, vmin), vmax));
                cp += 8;
            }
            if (rr > 0) {
                float16_t lane[8];
                vst1q_f16(lane, acc[r][g]);
                for (int i = 0; i < rr; ++i) {
                    float16_t v = lane[i];
                    if constexpr (!zero_mode) {
                        v = v + cp[i];
                    }
                    cp[i] = std::min(std::max(v, cmin), cmax);
                }
            }
        }
    }
}

template <bool zero_mode = false>
inline void mma_packed_b_mr4_nr16_mrxncxkc_f16(
    int mr, int nc, int kc, float16_t* NNOPS_RESTRICT C, int ldc,
    const float16_t* NNOPS_RESTRICT A, int lda,
    const float16_t* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max) noexcept {

    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;

    float16_t* NNOPS_RESTRICT c0_ptr = C + 0 * ldc;
    float16_t* NNOPS_RESTRICT c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    float16_t* NNOPS_RESTRICT c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    float16_t* NNOPS_RESTRICT c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;

    const int ldb = static_cast<int>(mma_pack_align(16 * kc * sizeof(float16_t)) / sizeof(float16_t));
    const float16_t cmin = static_cast<float16_t>(clamp_min);
    const float16_t cmax = static_cast<float16_t>(clamp_max);
    const float16x8_t vmin = vdupq_n_f16(cmin);
    const float16x8_t vmax = vdupq_n_f16(cmax);

    int n = 0;
    for (; n < nc - 15; n += 16) {
        float16x8_t c00 = vdupq_n_f16(0.0f16), c01 = c00;
        float16x8_t c10 = vdupq_n_f16(0.0f16), c11 = c10;
        float16x8_t c20 = vdupq_n_f16(0.0f16), c21 = c20;
        float16x8_t c30 = vdupq_n_f16(0.0f16), c31 = c30;

        const float16_t* NNOPS_RESTRICT a0 = A + 0 * lda;
        const float16_t* NNOPS_RESTRICT a1 = A + r1 * lda;
        const float16_t* NNOPS_RESTRICT a2 = A + r2 * lda;
        const float16_t* NNOPS_RESTRICT a3 = A + r3 * lda;
        const float16_t* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kc; ++k) {
            const float16x8_t b0 = vld1q_f16(b_ptr + 0 * 8);
            const float16x8_t b1 = vld1q_f16(b_ptr + 1 * 8);
            b_ptr += 16;

            const float16_t v0 = *a0++;
            const float16_t v1 = *a1++;
            const float16_t v2 = *a2++;
            const float16_t v3 = *a3++;

            c00 = vfmaq_n_f16(c00, b0, v0);
            c01 = vfmaq_n_f16(c01, b1, v0);
            c10 = vfmaq_n_f16(c10, b0, v1);
            c11 = vfmaq_n_f16(c11, b1, v1);
            c20 = vfmaq_n_f16(c20, b0, v2);
            c21 = vfmaq_n_f16(c21, b1, v2);
            c30 = vfmaq_n_f16(c30, b0, v3);
            c31 = vfmaq_n_f16(c31, b1, v3);
        }

        if constexpr (!zero_mode) {
            c00 = vaddq_f16(c00, vld1q_f16(c0_ptr + 0 * 8));
            c01 = vaddq_f16(c01, vld1q_f16(c0_ptr + 1 * 8));
            c10 = vaddq_f16(c10, vld1q_f16(c1_ptr + 0 * 8));
            c11 = vaddq_f16(c11, vld1q_f16(c1_ptr + 1 * 8));
            c20 = vaddq_f16(c20, vld1q_f16(c2_ptr + 0 * 8));
            c21 = vaddq_f16(c21, vld1q_f16(c2_ptr + 1 * 8));
            c30 = vaddq_f16(c30, vld1q_f16(c3_ptr + 0 * 8));
            c31 = vaddq_f16(c31, vld1q_f16(c3_ptr + 1 * 8));
        }

        vst1q_f16(c0_ptr + 0 * 8, vminq_f16(vmaxq_f16(c00, vmin), vmax));
        vst1q_f16(c0_ptr + 1 * 8, vminq_f16(vmaxq_f16(c01, vmin), vmax));
        vst1q_f16(c1_ptr + 0 * 8, vminq_f16(vmaxq_f16(c10, vmin), vmax));
        vst1q_f16(c1_ptr + 1 * 8, vminq_f16(vmaxq_f16(c11, vmin), vmax));
        vst1q_f16(c2_ptr + 0 * 8, vminq_f16(vmaxq_f16(c20, vmin), vmax));
        vst1q_f16(c2_ptr + 1 * 8, vminq_f16(vmaxq_f16(c21, vmin), vmax));
        vst1q_f16(c3_ptr + 0 * 8, vminq_f16(vmaxq_f16(c30, vmin), vmax));
        vst1q_f16(c3_ptr + 1 * 8, vminq_f16(vmaxq_f16(c31, vmin), vmax));

        c0_ptr += 16;
        c1_ptr += 16;
        c2_ptr += 16;
        c3_ptr += 16;

        B += ldb;
    }

    const int rem = nc - n;
    if (rem > 0) {
        float16x8_t c00 = vdupq_n_f16(0.0f16), c01 = c00;
        float16x8_t c10 = vdupq_n_f16(0.0f16), c11 = c10;
        float16x8_t c20 = vdupq_n_f16(0.0f16), c21 = c20;
        float16x8_t c30 = vdupq_n_f16(0.0f16), c31 = c30;

        const float16_t* NNOPS_RESTRICT a0 = A + 0 * lda;
        const float16_t* NNOPS_RESTRICT a1 = A + r1 * lda;
        const float16_t* NNOPS_RESTRICT a2 = A + r2 * lda;
        const float16_t* NNOPS_RESTRICT a3 = A + r3 * lda;
        const float16_t* NNOPS_RESTRICT b_ptr = B;
        if (rem > 8) {
            for (int k = 0; k < kc; ++k) {
                const float16x8_t b0 = vld1q_f16(b_ptr + 0 * 8);
                const float16x8_t b1 = vld1q_f16(b_ptr + 1 * 8);
                b_ptr += 16;

                const float16_t v0 = *a0++;
                const float16_t v1 = *a1++;
                const float16_t v2 = *a2++;
                const float16_t v3 = *a3++;

                c00 = vfmaq_n_f16(c00, b0, v0);
                c01 = vfmaq_n_f16(c01, b1, v0);
                c10 = vfmaq_n_f16(c10, b0, v1);
                c11 = vfmaq_n_f16(c11, b1, v1);
                c20 = vfmaq_n_f16(c20, b0, v2);
                c21 = vfmaq_n_f16(c21, b1, v2);
                c30 = vfmaq_n_f16(c30, b0, v3);
                c31 = vfmaq_n_f16(c31, b1, v3);
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const float16x8_t b0 = vld1q_f16(b_ptr);
                b_ptr += 16;

                const float16_t v0 = *a0++;
                const float16_t v1 = *a1++;
                const float16_t v2 = *a2++;
                const float16_t v3 = *a3++;

                c00 = vfmaq_n_f16(c00, b0, v0);
                c10 = vfmaq_n_f16(c10, b0, v1);
                c20 = vfmaq_n_f16(c20, b0, v2);
                c30 = vfmaq_n_f16(c30, b0, v3);
            }
        }

        const float16x8_t acc[4][2] = {
            {c00, c01}, {c10, c11}, {c20, c21}, {c30, c31},
        };
        float16_t* NNOPS_RESTRICT c_rows[4] = {c0_ptr, c1_ptr, c2_ptr, c3_ptr};
        for (int r = 0; r < mr; ++r) {
            float16_t* NNOPS_RESTRICT cp = c_rows[r];
            int rr = rem;
            int g = 0;
            for (; rr >= 8; rr -= 8, ++g) {
                float16x8_t v = acc[r][g];
                if constexpr (!zero_mode) {
                    v = vaddq_f16(v, vld1q_f16(cp));
                }
                vst1q_f16(cp, vminq_f16(vmaxq_f16(v, vmin), vmax));
                cp += 8;
            }
            if (rr > 0) {
                float16_t lane[8];
                vst1q_f16(lane, acc[r][g]);
                for (int i = 0; i < rr; ++i) {
                    float16_t v = lane[i];
                    if constexpr (!zero_mode) {
                        v = v + cp[i];
                    }
                    cp[i] = std::min(std::max(v, cmin), cmax);
                }
            }
        }
    }
}

// =========================================================================
//  mr=1  kernels
// =========================================================================

template <bool zero_mode = false>
inline void mma_packed_ab_mr1_nr16_mrxncxkc_f16(
    int mr, int nc, int kc, float16_t* NNOPS_RESTRICT C, int ldc,
    const float16_t* NNOPS_RESTRICT A,
    const float16_t* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max) noexcept {
    (void)mr;   // one row, so the row count never varies

    float16_t* NNOPS_RESTRICT c0_ptr = C;

    const int ldb = static_cast<int>(mma_pack_align(16 * kc * sizeof(float16_t)) / sizeof(float16_t));
    const float16_t cmin = static_cast<float16_t>(clamp_min);
    const float16_t cmax = static_cast<float16_t>(clamp_max);
    const float16x8_t vmin = vdupq_n_f16(cmin);
    const float16x8_t vmax = vdupq_n_f16(cmax);

    int n = 0;
    for (; n < nc - 15; n += 16) {
        float16x8_t c00 = vdupq_n_f16(0.0f16), c01 = c00;

        const float16_t* NNOPS_RESTRICT a_ptr = A;
        const float16_t* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kc; ++k) {
            const float16x8_t b0 = vld1q_f16(b_ptr + 0 * 8);
            const float16x8_t b1 = vld1q_f16(b_ptr + 1 * 8);
            b_ptr += 16;

            const float16_t a = *a_ptr++;

            c00 = vfmaq_n_f16(c00, b0, a);
            c01 = vfmaq_n_f16(c01, b1, a);
        }

        if constexpr (!zero_mode) {
            c00 = vaddq_f16(c00, vld1q_f16(c0_ptr + 0 * 8));
            c01 = vaddq_f16(c01, vld1q_f16(c0_ptr + 1 * 8));
        }

        vst1q_f16(c0_ptr + 0 * 8, vminq_f16(vmaxq_f16(c00, vmin), vmax));
        vst1q_f16(c0_ptr + 1 * 8, vminq_f16(vmaxq_f16(c01, vmin), vmax));

        c0_ptr += 16;
        B += ldb;
    }

    const int rem = nc - n;
    if (rem > 0) {
        float16x8_t c00 = vdupq_n_f16(0.0f16), c01 = c00;

        const float16_t* NNOPS_RESTRICT a_ptr = A;
        const float16_t* NNOPS_RESTRICT b_ptr = B;
        if (rem > 8) {
            for (int k = 0; k < kc; ++k) {
                const float16x8_t b0 = vld1q_f16(b_ptr + 0 * 8);
                const float16x8_t b1 = vld1q_f16(b_ptr + 1 * 8);
                b_ptr += 16;

                const float16_t a = *a_ptr++;

                c00 = vfmaq_n_f16(c00, b0, a);
                c01 = vfmaq_n_f16(c01, b1, a);
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const float16x8_t b0 = vld1q_f16(b_ptr);
                b_ptr += 16;

                const float16_t a = *a_ptr++;

                c00 = vfmaq_n_f16(c00, b0, a);
            }
        }

        const float16x8_t acc[2] = {c00, c01};
        float16_t* NNOPS_RESTRICT cp = c0_ptr;
        int rr = rem;
        int g = 0;
        for (; rr >= 8; rr -= 8, ++g) {
            float16x8_t v = acc[g];
            if constexpr (!zero_mode) {
                v = vaddq_f16(v, vld1q_f16(cp));
            }
            vst1q_f16(cp, vminq_f16(vmaxq_f16(v, vmin), vmax));
            cp += 8;
        }
        if (rr > 0) {
            float16_t lane[8];
            vst1q_f16(lane, acc[g]);
            for (int i = 0; i < rr; ++i) {
                float16_t v = lane[i];
                if constexpr (!zero_mode) {
                    v = v + cp[i];
                }
                cp[i] = std::min(std::max(v, cmin), cmax);
            }
        }
    }
}

template <bool zero_mode = false>
inline void mma_packed_b_mr1_nr16_mrxncxkc_f16(
    int mr, int nc, int kc, float16_t* NNOPS_RESTRICT C, int ldc,
    const float16_t* NNOPS_RESTRICT A, int lda,
    const float16_t* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max) noexcept {
    (void)mr;   // one row, so the row count never varies

    float16_t* NNOPS_RESTRICT c0_ptr = C;

    const int ldb = static_cast<int>(mma_pack_align(16 * kc * sizeof(float16_t)) / sizeof(float16_t));
    const float16_t cmin = static_cast<float16_t>(clamp_min);
    const float16_t cmax = static_cast<float16_t>(clamp_max);
    const float16x8_t vmin = vdupq_n_f16(cmin);
    const float16x8_t vmax = vdupq_n_f16(cmax);

    int n = 0;
    for (; n < nc - 15; n += 16) {
        float16x8_t c00 = vdupq_n_f16(0.0f16), c01 = c00;

        const float16_t* NNOPS_RESTRICT a_ptr = A;
        const float16_t* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kc; ++k) {
            const float16x8_t b0 = vld1q_f16(b_ptr + 0 * 8);
            const float16x8_t b1 = vld1q_f16(b_ptr + 1 * 8);
            b_ptr += 16;

            const float16_t a = *a_ptr++;
            a_ptr += lda - 1;

            c00 = vfmaq_n_f16(c00, b0, a);
            c01 = vfmaq_n_f16(c01, b1, a);
        }

        if constexpr (!zero_mode) {
            c00 = vaddq_f16(c00, vld1q_f16(c0_ptr + 0 * 8));
            c01 = vaddq_f16(c01, vld1q_f16(c0_ptr + 1 * 8));
        }

        vst1q_f16(c0_ptr + 0 * 8, vminq_f16(vmaxq_f16(c00, vmin), vmax));
        vst1q_f16(c0_ptr + 1 * 8, vminq_f16(vmaxq_f16(c01, vmin), vmax));

        c0_ptr += 16;
        B += ldb;
    }

    const int rem = nc - n;
    if (rem > 0) {
        float16x8_t c00 = vdupq_n_f16(0.0f16), c01 = c00;

        const float16_t* NNOPS_RESTRICT a_ptr = A;
        const float16_t* NNOPS_RESTRICT b_ptr = B;
        if (rem > 8) {
            for (int k = 0; k < kc; ++k) {
                const float16x8_t b0 = vld1q_f16(b_ptr + 0 * 8);
                const float16x8_t b1 = vld1q_f16(b_ptr + 1 * 8);
                b_ptr += 16;

                const float16_t a = *a_ptr++;
                a_ptr += lda - 1;

                c00 = vfmaq_n_f16(c00, b0, a);
                c01 = vfmaq_n_f16(c01, b1, a);
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const float16x8_t b0 = vld1q_f16(b_ptr);
                b_ptr += 16;

                const float16_t a = *a_ptr++;
                a_ptr += lda - 1;

                c00 = vfmaq_n_f16(c00, b0, a);
            }
        }

        const float16x8_t acc[2] = {c00, c01};
        float16_t* NNOPS_RESTRICT cp = c0_ptr;
        int rr = rem;
        int g = 0;
        for (; rr >= 8; rr -= 8, ++g) {
            float16x8_t v = acc[g];
            if constexpr (!zero_mode) {
                v = vaddq_f16(v, vld1q_f16(cp));
            }
            vst1q_f16(cp, vminq_f16(vmaxq_f16(v, vmin), vmax));
            cp += 8;
        }
        if (rr > 0) {
            float16_t lane[8];
            vst1q_f16(lane, acc[g]);
            for (int i = 0; i < rr; ++i) {
                float16_t v = lane[i];
                if constexpr (!zero_mode) {
                    v = v + cp[i];
                }
                cp[i] = std::min(std::max(v, cmin), cmax);
            }
        }
    }
}

// ---- dispatch tables, consumed by matmul_helper.cpp ---------------------
template <bool Z>
inline constexpr auto mma_pack_fns_f16 = std::array{
    &mma_packed_ab_mr8_nr16_mrxncxkc_f16<Z>,
    &mma_packed_ab_mr4_nr16_mrxncxkc_f16<Z>,
    &mma_packed_ab_mr1_nr16_mrxncxkc_f16<Z>};

template <bool Z>
inline constexpr auto mma_direct_fns_f16 = std::array{
    &mma_packed_b_mr8_nr16_mrxncxkc_f16<Z>,
    &mma_packed_b_mr4_nr16_mrxncxkc_f16<Z>,
    &mma_packed_b_mr1_nr16_mrxncxkc_f16<Z>};

}  // namespace nnops::backend::cpu::aarch64
