#pragma once
/// @file mma_f32.hpp
/// @brief AArch64 NEON float32 mrxncxkc MMA kernels over uniform 12-wide B.
///
/// The runtime-M kernel: the tile height comes in as `mr` (1..8) and the
/// column count as `nc`, so one kernel serves every panel the largest-first M
/// decomposition emits ({8, 4, 1}) instead of one per width.
///
/// A and B may both be packed (`mma_packed_ab_*`) or only B packed while A is
/// read row-major (`mma_packed_b_*`). Packed B is uniform: ceil(nc / 12)
/// panels, every one a full 12 columns wide with the columns past `nc`
/// zero-filled, each in a mma_pack_align(12*kc*sizeof(float))-byte slot
/// (12*kc floats). That lets the kernel read 12 floats per k at every column,
/// so a tail wider than 4 costs one k-loop instead of two.
///
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
inline void mma_packed_ab_mr8_nr12_mrxncxkc_f32(
    int mr, int nc, int kc, float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A,
    const float* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max) noexcept {

    float* NNOPS_RESTRICT c0_ptr = C + 0 * ldc;
    float* NNOPS_RESTRICT c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    float* NNOPS_RESTRICT c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    float* NNOPS_RESTRICT c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;
    float* NNOPS_RESTRICT c4_ptr = (mr > 4) ? C + 4 * ldc : c3_ptr;
    float* NNOPS_RESTRICT c5_ptr = (mr > 5) ? C + 5 * ldc : c4_ptr;
    float* NNOPS_RESTRICT c6_ptr = (mr > 6) ? C + 6 * ldc : c5_ptr;
    float* NNOPS_RESTRICT c7_ptr = (mr > 7) ? C + 7 * ldc : c6_ptr;

    const int ldb = static_cast<int>(mma_pack_align(12 * kc * sizeof(float)) / sizeof(float));
    const float32x4_t vmin = vdupq_n_f32(clamp_min);
    const float32x4_t vmax = vdupq_n_f32(clamp_max);

    int n = 0;
    for (; n < nc - 11; n += 12) {
        float32x4_t c00 = vdupq_n_f32(0.0f), c01 = c00, c02 = c00;
        float32x4_t c10 = vdupq_n_f32(0.0f), c11 = c10, c12 = c10;
        float32x4_t c20 = vdupq_n_f32(0.0f), c21 = c20, c22 = c20;
        float32x4_t c30 = vdupq_n_f32(0.0f), c31 = c30, c32 = c30;
        float32x4_t c40 = vdupq_n_f32(0.0f), c41 = c40, c42 = c40;
        float32x4_t c50 = vdupq_n_f32(0.0f), c51 = c50, c52 = c50;
        float32x4_t c60 = vdupq_n_f32(0.0f), c61 = c60, c62 = c60;
        float32x4_t c70 = vdupq_n_f32(0.0f), c71 = c70, c72 = c70;

        const float* NNOPS_RESTRICT a_ptr = A;
        const float* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kc; ++k) {
            const float32x4_t b0 = vld1q_f32(b_ptr + 0 * 4);
            const float32x4_t b1 = vld1q_f32(b_ptr + 1 * 4);
            const float32x4_t b2 = vld1q_f32(b_ptr + 2 * 4);
            b_ptr += 12;

            const float32x4_t a0 = vld1q_f32(a_ptr + 0 * 4);
            const float32x4_t a1 = vld1q_f32(a_ptr + 1 * 4);
            a_ptr += 8;

            c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
            c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
            c02 = vfmaq_laneq_f32(c02, b2, a0, 0);
            c10 = vfmaq_laneq_f32(c10, b0, a0, 1);
            c11 = vfmaq_laneq_f32(c11, b1, a0, 1);
            c12 = vfmaq_laneq_f32(c12, b2, a0, 1);
            c20 = vfmaq_laneq_f32(c20, b0, a0, 2);
            c21 = vfmaq_laneq_f32(c21, b1, a0, 2);
            c22 = vfmaq_laneq_f32(c22, b2, a0, 2);
            c30 = vfmaq_laneq_f32(c30, b0, a0, 3);
            c31 = vfmaq_laneq_f32(c31, b1, a0, 3);
            c32 = vfmaq_laneq_f32(c32, b2, a0, 3);
            c40 = vfmaq_laneq_f32(c40, b0, a1, 0);
            c41 = vfmaq_laneq_f32(c41, b1, a1, 0);
            c42 = vfmaq_laneq_f32(c42, b2, a1, 0);
            c50 = vfmaq_laneq_f32(c50, b0, a1, 1);
            c51 = vfmaq_laneq_f32(c51, b1, a1, 1);
            c52 = vfmaq_laneq_f32(c52, b2, a1, 1);
            c60 = vfmaq_laneq_f32(c60, b0, a1, 2);
            c61 = vfmaq_laneq_f32(c61, b1, a1, 2);
            c62 = vfmaq_laneq_f32(c62, b2, a1, 2);
            c70 = vfmaq_laneq_f32(c70, b0, a1, 3);
            c71 = vfmaq_laneq_f32(c71, b1, a1, 3);
            c72 = vfmaq_laneq_f32(c72, b2, a1, 3);
        }

        if constexpr (!zero_mode) {
            c00 = vaddq_f32(c00, vld1q_f32(c0_ptr + 0 * 4));
            c01 = vaddq_f32(c01, vld1q_f32(c0_ptr + 1 * 4));
            c02 = vaddq_f32(c02, vld1q_f32(c0_ptr + 2 * 4));
            c10 = vaddq_f32(c10, vld1q_f32(c1_ptr + 0 * 4));
            c11 = vaddq_f32(c11, vld1q_f32(c1_ptr + 1 * 4));
            c12 = vaddq_f32(c12, vld1q_f32(c1_ptr + 2 * 4));
            c20 = vaddq_f32(c20, vld1q_f32(c2_ptr + 0 * 4));
            c21 = vaddq_f32(c21, vld1q_f32(c2_ptr + 1 * 4));
            c22 = vaddq_f32(c22, vld1q_f32(c2_ptr + 2 * 4));
            c30 = vaddq_f32(c30, vld1q_f32(c3_ptr + 0 * 4));
            c31 = vaddq_f32(c31, vld1q_f32(c3_ptr + 1 * 4));
            c32 = vaddq_f32(c32, vld1q_f32(c3_ptr + 2 * 4));
            c40 = vaddq_f32(c40, vld1q_f32(c4_ptr + 0 * 4));
            c41 = vaddq_f32(c41, vld1q_f32(c4_ptr + 1 * 4));
            c42 = vaddq_f32(c42, vld1q_f32(c4_ptr + 2 * 4));
            c50 = vaddq_f32(c50, vld1q_f32(c5_ptr + 0 * 4));
            c51 = vaddq_f32(c51, vld1q_f32(c5_ptr + 1 * 4));
            c52 = vaddq_f32(c52, vld1q_f32(c5_ptr + 2 * 4));
            c60 = vaddq_f32(c60, vld1q_f32(c6_ptr + 0 * 4));
            c61 = vaddq_f32(c61, vld1q_f32(c6_ptr + 1 * 4));
            c62 = vaddq_f32(c62, vld1q_f32(c6_ptr + 2 * 4));
            c70 = vaddq_f32(c70, vld1q_f32(c7_ptr + 0 * 4));
            c71 = vaddq_f32(c71, vld1q_f32(c7_ptr + 1 * 4));
            c72 = vaddq_f32(c72, vld1q_f32(c7_ptr + 2 * 4));
        }

        vst1q_f32(c0_ptr + 0 * 4, vminq_f32(vmaxq_f32(c00, vmin), vmax));
        vst1q_f32(c0_ptr + 1 * 4, vminq_f32(vmaxq_f32(c01, vmin), vmax));
        vst1q_f32(c0_ptr + 2 * 4, vminq_f32(vmaxq_f32(c02, vmin), vmax));
        vst1q_f32(c1_ptr + 0 * 4, vminq_f32(vmaxq_f32(c10, vmin), vmax));
        vst1q_f32(c1_ptr + 1 * 4, vminq_f32(vmaxq_f32(c11, vmin), vmax));
        vst1q_f32(c1_ptr + 2 * 4, vminq_f32(vmaxq_f32(c12, vmin), vmax));
        vst1q_f32(c2_ptr + 0 * 4, vminq_f32(vmaxq_f32(c20, vmin), vmax));
        vst1q_f32(c2_ptr + 1 * 4, vminq_f32(vmaxq_f32(c21, vmin), vmax));
        vst1q_f32(c2_ptr + 2 * 4, vminq_f32(vmaxq_f32(c22, vmin), vmax));
        vst1q_f32(c3_ptr + 0 * 4, vminq_f32(vmaxq_f32(c30, vmin), vmax));
        vst1q_f32(c3_ptr + 1 * 4, vminq_f32(vmaxq_f32(c31, vmin), vmax));
        vst1q_f32(c3_ptr + 2 * 4, vminq_f32(vmaxq_f32(c32, vmin), vmax));
        vst1q_f32(c4_ptr + 0 * 4, vminq_f32(vmaxq_f32(c40, vmin), vmax));
        vst1q_f32(c4_ptr + 1 * 4, vminq_f32(vmaxq_f32(c41, vmin), vmax));
        vst1q_f32(c4_ptr + 2 * 4, vminq_f32(vmaxq_f32(c42, vmin), vmax));
        vst1q_f32(c5_ptr + 0 * 4, vminq_f32(vmaxq_f32(c50, vmin), vmax));
        vst1q_f32(c5_ptr + 1 * 4, vminq_f32(vmaxq_f32(c51, vmin), vmax));
        vst1q_f32(c5_ptr + 2 * 4, vminq_f32(vmaxq_f32(c52, vmin), vmax));
        vst1q_f32(c6_ptr + 0 * 4, vminq_f32(vmaxq_f32(c60, vmin), vmax));
        vst1q_f32(c6_ptr + 1 * 4, vminq_f32(vmaxq_f32(c61, vmin), vmax));
        vst1q_f32(c6_ptr + 2 * 4, vminq_f32(vmaxq_f32(c62, vmin), vmax));
        vst1q_f32(c7_ptr + 0 * 4, vminq_f32(vmaxq_f32(c70, vmin), vmax));
        vst1q_f32(c7_ptr + 1 * 4, vminq_f32(vmaxq_f32(c71, vmin), vmax));
        vst1q_f32(c7_ptr + 2 * 4, vminq_f32(vmaxq_f32(c72, vmin), vmax));

        c0_ptr += 12;
        c1_ptr += 12;
        c2_ptr += 12;
        c3_ptr += 12;
        c4_ptr += 12;
        c5_ptr += 12;
        c6_ptr += 12;
        c7_ptr += 12;

        B += ldb;
    }

    const int rem = nc - n;
    if (rem > 0) {
        // The tail is one full 12-wide panel in the packed layout -- the
        // packer zero-fills the columns past `nc`, so all three vectors can
        // always be read. A tail wider than 4 needs the second vector, so it
        // reuses the full-panel body; a narrower one reads the first only.
        float32x4_t c00 = vdupq_n_f32(0.0f), c01 = c00, c02 = c00;
        float32x4_t c10 = vdupq_n_f32(0.0f), c11 = c10, c12 = c10;
        float32x4_t c20 = vdupq_n_f32(0.0f), c21 = c20, c22 = c20;
        float32x4_t c30 = vdupq_n_f32(0.0f), c31 = c30, c32 = c30;
        float32x4_t c40 = vdupq_n_f32(0.0f), c41 = c40, c42 = c40;
        float32x4_t c50 = vdupq_n_f32(0.0f), c51 = c50, c52 = c50;
        float32x4_t c60 = vdupq_n_f32(0.0f), c61 = c60, c62 = c60;
        float32x4_t c70 = vdupq_n_f32(0.0f), c71 = c70, c72 = c70;

        const float* NNOPS_RESTRICT a_ptr = A;
        const float* NNOPS_RESTRICT b_ptr = B;
        if (rem > 4) {
            for (int k = 0; k < kc; ++k) {
                const float32x4_t b0 = vld1q_f32(b_ptr + 0 * 4);
                const float32x4_t b1 = vld1q_f32(b_ptr + 1 * 4);
                const float32x4_t b2 = vld1q_f32(b_ptr + 2 * 4);
                b_ptr += 12;

                const float32x4_t a0 = vld1q_f32(a_ptr + 0 * 4);
                const float32x4_t a1 = vld1q_f32(a_ptr + 1 * 4);
                a_ptr += 8;

                c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
                c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
                c02 = vfmaq_laneq_f32(c02, b2, a0, 0);
                c10 = vfmaq_laneq_f32(c10, b0, a0, 1);
                c11 = vfmaq_laneq_f32(c11, b1, a0, 1);
                c12 = vfmaq_laneq_f32(c12, b2, a0, 1);
                c20 = vfmaq_laneq_f32(c20, b0, a0, 2);
                c21 = vfmaq_laneq_f32(c21, b1, a0, 2);
                c22 = vfmaq_laneq_f32(c22, b2, a0, 2);
                c30 = vfmaq_laneq_f32(c30, b0, a0, 3);
                c31 = vfmaq_laneq_f32(c31, b1, a0, 3);
                c32 = vfmaq_laneq_f32(c32, b2, a0, 3);
                c40 = vfmaq_laneq_f32(c40, b0, a1, 0);
                c41 = vfmaq_laneq_f32(c41, b1, a1, 0);
                c42 = vfmaq_laneq_f32(c42, b2, a1, 0);
                c50 = vfmaq_laneq_f32(c50, b0, a1, 1);
                c51 = vfmaq_laneq_f32(c51, b1, a1, 1);
                c52 = vfmaq_laneq_f32(c52, b2, a1, 1);
                c60 = vfmaq_laneq_f32(c60, b0, a1, 2);
                c61 = vfmaq_laneq_f32(c61, b1, a1, 2);
                c62 = vfmaq_laneq_f32(c62, b2, a1, 2);
                c70 = vfmaq_laneq_f32(c70, b0, a1, 3);
                c71 = vfmaq_laneq_f32(c71, b1, a1, 3);
                c72 = vfmaq_laneq_f32(c72, b2, a1, 3);
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const float32x4_t b0 = vld1q_f32(b_ptr);
                b_ptr += 12;

                const float32x4_t a0 = vld1q_f32(a_ptr + 0 * 4);
                const float32x4_t a1 = vld1q_f32(a_ptr + 1 * 4);
                a_ptr += 8;

                c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
                c10 = vfmaq_laneq_f32(c10, b0, a0, 1);
                c20 = vfmaq_laneq_f32(c20, b0, a0, 2);
                c30 = vfmaq_laneq_f32(c30, b0, a0, 3);
                c40 = vfmaq_laneq_f32(c40, b0, a1, 0);
                c50 = vfmaq_laneq_f32(c50, b0, a1, 1);
                c60 = vfmaq_laneq_f32(c60, b0, a1, 2);
                c70 = vfmaq_laneq_f32(c70, b0, a1, 3);
            }
        }

        const float32x4_t acc[8][3] = {
            {c00, c01, c02}, {c10, c11, c12}, {c20, c21, c22}, {c30, c31, c32},
            {c40, c41, c42}, {c50, c51, c52}, {c60, c61, c62}, {c70, c71, c72},
        };
        float* NNOPS_RESTRICT c_rows[8] = {c0_ptr, c1_ptr, c2_ptr, c3_ptr,
                                           c4_ptr, c5_ptr, c6_ptr, c7_ptr};
        // Only the live rows are stored; the alias rows past `mr` would write
        // a duplicate of the last one back over itself.
        for (int r = 0; r < mr; ++r) {
            float* NNOPS_RESTRICT cp = c_rows[r];
            int rr = rem;
            int g = 0;
            for (; rr >= 4; rr -= 4, ++g) {
                float32x4_t v = acc[r][g];
                if constexpr (!zero_mode) {
                    v = vaddq_f32(v, vld1q_f32(cp));
                }
                vst1q_f32(cp, vminq_f32(vmaxq_f32(v, vmin), vmax));
                cp += 4;
            }
            if (rr > 0) {
                float lane[4];
                vst1q_f32(lane, acc[r][g]);
                for (int i = 0; i < rr; ++i) {
                    float v = lane[i];
                    if constexpr (!zero_mode) {
                        v += cp[i];
                    }
                    cp[i] = std::min(std::max(v, clamp_min), clamp_max);
                }
            }
        }
    }
}

template <bool zero_mode = false>
inline void mma_packed_ab_mr4_nr12_mrxncxkc_f32(
    int mr, int nc, int kc, float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A,
    const float* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max) noexcept {

    float* NNOPS_RESTRICT c0_ptr = C + 0 * ldc;
    float* NNOPS_RESTRICT c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    float* NNOPS_RESTRICT c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    float* NNOPS_RESTRICT c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;

    const int ldb = static_cast<int>(mma_pack_align(12 * kc * sizeof(float)) / sizeof(float));
    const float32x4_t vmin = vdupq_n_f32(clamp_min);
    const float32x4_t vmax = vdupq_n_f32(clamp_max);

    int n = 0;
    for (; n < nc - 11; n += 12) {
        float32x4_t c00 = vdupq_n_f32(0.0f), c01 = c00, c02 = c00;
        float32x4_t c10 = vdupq_n_f32(0.0f), c11 = c10, c12 = c10;
        float32x4_t c20 = vdupq_n_f32(0.0f), c21 = c20, c22 = c20;
        float32x4_t c30 = vdupq_n_f32(0.0f), c31 = c30, c32 = c30;

        const float* NNOPS_RESTRICT a_ptr = A;
        const float* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kc; ++k) {
            const float32x4_t b0 = vld1q_f32(b_ptr + 0 * 4);
            const float32x4_t b1 = vld1q_f32(b_ptr + 1 * 4);
            const float32x4_t b2 = vld1q_f32(b_ptr + 2 * 4);
            b_ptr += 12;

            const float32x4_t a0 = vld1q_f32(a_ptr);
            a_ptr += 4;

            c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
            c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
            c02 = vfmaq_laneq_f32(c02, b2, a0, 0);
            c10 = vfmaq_laneq_f32(c10, b0, a0, 1);
            c11 = vfmaq_laneq_f32(c11, b1, a0, 1);
            c12 = vfmaq_laneq_f32(c12, b2, a0, 1);
            c20 = vfmaq_laneq_f32(c20, b0, a0, 2);
            c21 = vfmaq_laneq_f32(c21, b1, a0, 2);
            c22 = vfmaq_laneq_f32(c22, b2, a0, 2);
            c30 = vfmaq_laneq_f32(c30, b0, a0, 3);
            c31 = vfmaq_laneq_f32(c31, b1, a0, 3);
            c32 = vfmaq_laneq_f32(c32, b2, a0, 3);
        }

        if constexpr (!zero_mode) {
            c00 = vaddq_f32(c00, vld1q_f32(c0_ptr + 0 * 4));
            c01 = vaddq_f32(c01, vld1q_f32(c0_ptr + 1 * 4));
            c02 = vaddq_f32(c02, vld1q_f32(c0_ptr + 2 * 4));
            c10 = vaddq_f32(c10, vld1q_f32(c1_ptr + 0 * 4));
            c11 = vaddq_f32(c11, vld1q_f32(c1_ptr + 1 * 4));
            c12 = vaddq_f32(c12, vld1q_f32(c1_ptr + 2 * 4));
            c20 = vaddq_f32(c20, vld1q_f32(c2_ptr + 0 * 4));
            c21 = vaddq_f32(c21, vld1q_f32(c2_ptr + 1 * 4));
            c22 = vaddq_f32(c22, vld1q_f32(c2_ptr + 2 * 4));
            c30 = vaddq_f32(c30, vld1q_f32(c3_ptr + 0 * 4));
            c31 = vaddq_f32(c31, vld1q_f32(c3_ptr + 1 * 4));
            c32 = vaddq_f32(c32, vld1q_f32(c3_ptr + 2 * 4));
        }

        vst1q_f32(c0_ptr + 0 * 4, vminq_f32(vmaxq_f32(c00, vmin), vmax));
        vst1q_f32(c0_ptr + 1 * 4, vminq_f32(vmaxq_f32(c01, vmin), vmax));
        vst1q_f32(c0_ptr + 2 * 4, vminq_f32(vmaxq_f32(c02, vmin), vmax));
        vst1q_f32(c1_ptr + 0 * 4, vminq_f32(vmaxq_f32(c10, vmin), vmax));
        vst1q_f32(c1_ptr + 1 * 4, vminq_f32(vmaxq_f32(c11, vmin), vmax));
        vst1q_f32(c1_ptr + 2 * 4, vminq_f32(vmaxq_f32(c12, vmin), vmax));
        vst1q_f32(c2_ptr + 0 * 4, vminq_f32(vmaxq_f32(c20, vmin), vmax));
        vst1q_f32(c2_ptr + 1 * 4, vminq_f32(vmaxq_f32(c21, vmin), vmax));
        vst1q_f32(c2_ptr + 2 * 4, vminq_f32(vmaxq_f32(c22, vmin), vmax));
        vst1q_f32(c3_ptr + 0 * 4, vminq_f32(vmaxq_f32(c30, vmin), vmax));
        vst1q_f32(c3_ptr + 1 * 4, vminq_f32(vmaxq_f32(c31, vmin), vmax));
        vst1q_f32(c3_ptr + 2 * 4, vminq_f32(vmaxq_f32(c32, vmin), vmax));

        c0_ptr += 12;
        c1_ptr += 12;
        c2_ptr += 12;
        c3_ptr += 12;

        B += ldb;
    }

    const int rem = nc - n;
    if (rem > 0) {
        float32x4_t c00 = vdupq_n_f32(0.0f), c01 = c00, c02 = c00;
        float32x4_t c10 = vdupq_n_f32(0.0f), c11 = c10, c12 = c10;
        float32x4_t c20 = vdupq_n_f32(0.0f), c21 = c20, c22 = c20;
        float32x4_t c30 = vdupq_n_f32(0.0f), c31 = c30, c32 = c30;

        const float* NNOPS_RESTRICT a_ptr = A;
        const float* NNOPS_RESTRICT b_ptr = B;
        if (rem > 4) {
            for (int k = 0; k < kc; ++k) {
                const float32x4_t b0 = vld1q_f32(b_ptr + 0 * 4);
                const float32x4_t b1 = vld1q_f32(b_ptr + 1 * 4);
                const float32x4_t b2 = vld1q_f32(b_ptr + 2 * 4);
                b_ptr += 12;

                const float32x4_t a0 = vld1q_f32(a_ptr);
                a_ptr += 4;

                c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
                c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
                c02 = vfmaq_laneq_f32(c02, b2, a0, 0);
                c10 = vfmaq_laneq_f32(c10, b0, a0, 1);
                c11 = vfmaq_laneq_f32(c11, b1, a0, 1);
                c12 = vfmaq_laneq_f32(c12, b2, a0, 1);
                c20 = vfmaq_laneq_f32(c20, b0, a0, 2);
                c21 = vfmaq_laneq_f32(c21, b1, a0, 2);
                c22 = vfmaq_laneq_f32(c22, b2, a0, 2);
                c30 = vfmaq_laneq_f32(c30, b0, a0, 3);
                c31 = vfmaq_laneq_f32(c31, b1, a0, 3);
                c32 = vfmaq_laneq_f32(c32, b2, a0, 3);
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const float32x4_t b0 = vld1q_f32(b_ptr);
                b_ptr += 12;

                const float32x4_t a0 = vld1q_f32(a_ptr);
                a_ptr += 4;

                c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
                c10 = vfmaq_laneq_f32(c10, b0, a0, 1);
                c20 = vfmaq_laneq_f32(c20, b0, a0, 2);
                c30 = vfmaq_laneq_f32(c30, b0, a0, 3);
            }
        }

        const float32x4_t acc[4][3] = {
            {c00, c01, c02}, {c10, c11, c12}, {c20, c21, c22}, {c30, c31, c32},
        };
        float* NNOPS_RESTRICT c_rows[4] = {c0_ptr, c1_ptr, c2_ptr, c3_ptr};
        for (int r = 0; r < mr; ++r) {
            float* NNOPS_RESTRICT cp = c_rows[r];
            int rr = rem;
            int g = 0;
            for (; rr >= 4; rr -= 4, ++g) {
                float32x4_t v = acc[r][g];
                if constexpr (!zero_mode) {
                    v = vaddq_f32(v, vld1q_f32(cp));
                }
                vst1q_f32(cp, vminq_f32(vmaxq_f32(v, vmin), vmax));
                cp += 4;
            }
            if (rr > 0) {
                float lane[4];
                vst1q_f32(lane, acc[r][g]);
                for (int i = 0; i < rr; ++i) {
                    float v = lane[i];
                    if constexpr (!zero_mode) {
                        v += cp[i];
                    }
                    cp[i] = std::min(std::max(v, clamp_min), clamp_max);
                }
            }
        }
    }
}

template <bool zero_mode = false>
inline void mma_packed_ab_mr1_nr12_mrxncxkc_f32(
    int mr, int nc, int kc, float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A,
    const float* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max) noexcept {

    float* NNOPS_RESTRICT c0_ptr = C;

    const int ldb = static_cast<int>(mma_pack_align(12 * kc * sizeof(float)) / sizeof(float));
    const float32x4_t vmin = vdupq_n_f32(clamp_min);
    const float32x4_t vmax = vdupq_n_f32(clamp_max);

    int n = 0;
    for (; n < nc - 11; n += 12) {
        float32x4_t c00 = vdupq_n_f32(0.0f), c01 = c00, c02 = c00;

        const float* NNOPS_RESTRICT a_ptr = A;
        const float* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kc; ++k) {
            const float32x4_t b0 = vld1q_f32(b_ptr + 0 * 4);
            const float32x4_t b1 = vld1q_f32(b_ptr + 1 * 4);
            const float32x4_t b2 = vld1q_f32(b_ptr + 2 * 4);
            b_ptr += 12;

            const float a = *a_ptr++;

            c00 = vfmaq_n_f32(c00, b0, a);
            c01 = vfmaq_n_f32(c01, b1, a);
            c02 = vfmaq_n_f32(c02, b2, a);
        }

        if constexpr (!zero_mode) {
            c00 = vaddq_f32(c00, vld1q_f32(c0_ptr + 0 * 4));
            c01 = vaddq_f32(c01, vld1q_f32(c0_ptr + 1 * 4));
            c02 = vaddq_f32(c02, vld1q_f32(c0_ptr + 2 * 4));
        }

        vst1q_f32(c0_ptr + 0 * 4, vminq_f32(vmaxq_f32(c00, vmin), vmax));
        vst1q_f32(c0_ptr + 1 * 4, vminq_f32(vmaxq_f32(c01, vmin), vmax));
        vst1q_f32(c0_ptr + 2 * 4, vminq_f32(vmaxq_f32(c02, vmin), vmax));

        c0_ptr += 12;
        B += ldb;
    }

    const int rem = nc - n;
    if (rem > 0) {
        float32x4_t c00 = vdupq_n_f32(0.0f), c01 = c00, c02 = c00;

        const float* NNOPS_RESTRICT a_ptr = A;
        const float* NNOPS_RESTRICT b_ptr = B;
        if (rem > 4) {
            for (int k = 0; k < kc; ++k) {
                const float32x4_t b0 = vld1q_f32(b_ptr + 0 * 4);
                const float32x4_t b1 = vld1q_f32(b_ptr + 1 * 4);
                const float32x4_t b2 = vld1q_f32(b_ptr + 2 * 4);
                b_ptr += 12;

                const float a = *a_ptr++;

                c00 = vfmaq_n_f32(c00, b0, a);
                c01 = vfmaq_n_f32(c01, b1, a);
                c02 = vfmaq_n_f32(c02, b2, a);
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const float32x4_t b0 = vld1q_f32(b_ptr);
                b_ptr += 12;

                const float a = *a_ptr++;

                c00 = vfmaq_n_f32(c00, b0, a);
            }
        }

        const float32x4_t acc[3] = {c00, c01, c02};
        float* NNOPS_RESTRICT cp = c0_ptr;
        int rr = rem;
        int g = 0;
        for (; rr >= 4; rr -= 4, ++g) {
            float32x4_t v = acc[g];
            if constexpr (!zero_mode) {
                v = vaddq_f32(v, vld1q_f32(cp));
            }
            vst1q_f32(cp, vminq_f32(vmaxq_f32(v, vmin), vmax));
            cp += 4;
        }
        if (rr > 0) {
            float lane[4];
            vst1q_f32(lane, acc[g]);
            for (int i = 0; i < rr; ++i) {
                float v = lane[i];
                if constexpr (!zero_mode) {
                    v += cp[i];
                }
                cp[i] = std::min(std::max(v, clamp_min), clamp_max);
            }
        }
    }
}

// =========================================================================
//  mr=8/4/1  kernels, raw A — only B packed
// =========================================================================

template <bool zero_mode = false>
inline void mma_packed_b_mr8_nr12_mrxncxkc_f32(
    int mr, int nc, int kc, float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A, int lda,
    const float* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max) noexcept {

    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;
    const int r4 = (mr > 4) ? 4 : r3;
    const int r5 = (mr > 5) ? 5 : r4;
    const int r6 = (mr > 6) ? 6 : r5;
    const int r7 = (mr > 7) ? 7 : r6;

    float* NNOPS_RESTRICT c0_ptr = C + 0 * ldc;
    float* NNOPS_RESTRICT c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    float* NNOPS_RESTRICT c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    float* NNOPS_RESTRICT c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;
    float* NNOPS_RESTRICT c4_ptr = (mr > 4) ? C + 4 * ldc : c3_ptr;
    float* NNOPS_RESTRICT c5_ptr = (mr > 5) ? C + 5 * ldc : c4_ptr;
    float* NNOPS_RESTRICT c6_ptr = (mr > 6) ? C + 6 * ldc : c5_ptr;
    float* NNOPS_RESTRICT c7_ptr = (mr > 7) ? C + 7 * ldc : c6_ptr;

    const int ldb = static_cast<int>(mma_pack_align(12 * kc * sizeof(float)) / sizeof(float));
    const float32x4_t vmin = vdupq_n_f32(clamp_min);
    const float32x4_t vmax = vdupq_n_f32(clamp_max);

    int n = 0;
    for (; n < nc - 11; n += 12) {
        float32x4_t c00 = vdupq_n_f32(0.0f), c01 = c00, c02 = c00;
        float32x4_t c10 = vdupq_n_f32(0.0f), c11 = c10, c12 = c10;
        float32x4_t c20 = vdupq_n_f32(0.0f), c21 = c20, c22 = c20;
        float32x4_t c30 = vdupq_n_f32(0.0f), c31 = c30, c32 = c30;
        float32x4_t c40 = vdupq_n_f32(0.0f), c41 = c40, c42 = c40;
        float32x4_t c50 = vdupq_n_f32(0.0f), c51 = c50, c52 = c50;
        float32x4_t c60 = vdupq_n_f32(0.0f), c61 = c60, c62 = c60;
        float32x4_t c70 = vdupq_n_f32(0.0f), c71 = c70, c72 = c70;

        const float* NNOPS_RESTRICT a0 = A + 0 * lda;
        const float* NNOPS_RESTRICT a1 = A + r1 * lda;
        const float* NNOPS_RESTRICT a2 = A + r2 * lda;
        const float* NNOPS_RESTRICT a3 = A + r3 * lda;
        const float* NNOPS_RESTRICT a4 = A + r4 * lda;
        const float* NNOPS_RESTRICT a5 = A + r5 * lda;
        const float* NNOPS_RESTRICT a6 = A + r6 * lda;
        const float* NNOPS_RESTRICT a7 = A + r7 * lda;
        const float* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kc; ++k) {
            const float32x4_t b0 = vld1q_f32(b_ptr + 0 * 4);
            const float32x4_t b1 = vld1q_f32(b_ptr + 1 * 4);
            const float32x4_t b2 = vld1q_f32(b_ptr + 2 * 4);
            b_ptr += 12;

            const float v0 = *a0++;
            const float v1 = *a1++;
            const float v2 = *a2++;
            const float v3 = *a3++;
            const float v4 = *a4++;
            const float v5 = *a5++;
            const float v6 = *a6++;
            const float v7 = *a7++;

            c00 = vfmaq_n_f32(c00, b0, v0);
            c01 = vfmaq_n_f32(c01, b1, v0);
            c02 = vfmaq_n_f32(c02, b2, v0);
            c10 = vfmaq_n_f32(c10, b0, v1);
            c11 = vfmaq_n_f32(c11, b1, v1);
            c12 = vfmaq_n_f32(c12, b2, v1);
            c20 = vfmaq_n_f32(c20, b0, v2);
            c21 = vfmaq_n_f32(c21, b1, v2);
            c22 = vfmaq_n_f32(c22, b2, v2);
            c30 = vfmaq_n_f32(c30, b0, v3);
            c31 = vfmaq_n_f32(c31, b1, v3);
            c32 = vfmaq_n_f32(c32, b2, v3);
            c40 = vfmaq_n_f32(c40, b0, v4);
            c41 = vfmaq_n_f32(c41, b1, v4);
            c42 = vfmaq_n_f32(c42, b2, v4);
            c50 = vfmaq_n_f32(c50, b0, v5);
            c51 = vfmaq_n_f32(c51, b1, v5);
            c52 = vfmaq_n_f32(c52, b2, v5);
            c60 = vfmaq_n_f32(c60, b0, v6);
            c61 = vfmaq_n_f32(c61, b1, v6);
            c62 = vfmaq_n_f32(c62, b2, v6);
            c70 = vfmaq_n_f32(c70, b0, v7);
            c71 = vfmaq_n_f32(c71, b1, v7);
            c72 = vfmaq_n_f32(c72, b2, v7);
        }

        if constexpr (!zero_mode) {
            c00 = vaddq_f32(c00, vld1q_f32(c0_ptr + 0 * 4));
            c01 = vaddq_f32(c01, vld1q_f32(c0_ptr + 1 * 4));
            c02 = vaddq_f32(c02, vld1q_f32(c0_ptr + 2 * 4));
            c10 = vaddq_f32(c10, vld1q_f32(c1_ptr + 0 * 4));
            c11 = vaddq_f32(c11, vld1q_f32(c1_ptr + 1 * 4));
            c12 = vaddq_f32(c12, vld1q_f32(c1_ptr + 2 * 4));
            c20 = vaddq_f32(c20, vld1q_f32(c2_ptr + 0 * 4));
            c21 = vaddq_f32(c21, vld1q_f32(c2_ptr + 1 * 4));
            c22 = vaddq_f32(c22, vld1q_f32(c2_ptr + 2 * 4));
            c30 = vaddq_f32(c30, vld1q_f32(c3_ptr + 0 * 4));
            c31 = vaddq_f32(c31, vld1q_f32(c3_ptr + 1 * 4));
            c32 = vaddq_f32(c32, vld1q_f32(c3_ptr + 2 * 4));
            c40 = vaddq_f32(c40, vld1q_f32(c4_ptr + 0 * 4));
            c41 = vaddq_f32(c41, vld1q_f32(c4_ptr + 1 * 4));
            c42 = vaddq_f32(c42, vld1q_f32(c4_ptr + 2 * 4));
            c50 = vaddq_f32(c50, vld1q_f32(c5_ptr + 0 * 4));
            c51 = vaddq_f32(c51, vld1q_f32(c5_ptr + 1 * 4));
            c52 = vaddq_f32(c52, vld1q_f32(c5_ptr + 2 * 4));
            c60 = vaddq_f32(c60, vld1q_f32(c6_ptr + 0 * 4));
            c61 = vaddq_f32(c61, vld1q_f32(c6_ptr + 1 * 4));
            c62 = vaddq_f32(c62, vld1q_f32(c6_ptr + 2 * 4));
            c70 = vaddq_f32(c70, vld1q_f32(c7_ptr + 0 * 4));
            c71 = vaddq_f32(c71, vld1q_f32(c7_ptr + 1 * 4));
            c72 = vaddq_f32(c72, vld1q_f32(c7_ptr + 2 * 4));
        }

        vst1q_f32(c0_ptr + 0 * 4, vminq_f32(vmaxq_f32(c00, vmin), vmax));
        vst1q_f32(c0_ptr + 1 * 4, vminq_f32(vmaxq_f32(c01, vmin), vmax));
        vst1q_f32(c0_ptr + 2 * 4, vminq_f32(vmaxq_f32(c02, vmin), vmax));
        vst1q_f32(c1_ptr + 0 * 4, vminq_f32(vmaxq_f32(c10, vmin), vmax));
        vst1q_f32(c1_ptr + 1 * 4, vminq_f32(vmaxq_f32(c11, vmin), vmax));
        vst1q_f32(c1_ptr + 2 * 4, vminq_f32(vmaxq_f32(c12, vmin), vmax));
        vst1q_f32(c2_ptr + 0 * 4, vminq_f32(vmaxq_f32(c20, vmin), vmax));
        vst1q_f32(c2_ptr + 1 * 4, vminq_f32(vmaxq_f32(c21, vmin), vmax));
        vst1q_f32(c2_ptr + 2 * 4, vminq_f32(vmaxq_f32(c22, vmin), vmax));
        vst1q_f32(c3_ptr + 0 * 4, vminq_f32(vmaxq_f32(c30, vmin), vmax));
        vst1q_f32(c3_ptr + 1 * 4, vminq_f32(vmaxq_f32(c31, vmin), vmax));
        vst1q_f32(c3_ptr + 2 * 4, vminq_f32(vmaxq_f32(c32, vmin), vmax));
        vst1q_f32(c4_ptr + 0 * 4, vminq_f32(vmaxq_f32(c40, vmin), vmax));
        vst1q_f32(c4_ptr + 1 * 4, vminq_f32(vmaxq_f32(c41, vmin), vmax));
        vst1q_f32(c4_ptr + 2 * 4, vminq_f32(vmaxq_f32(c42, vmin), vmax));
        vst1q_f32(c5_ptr + 0 * 4, vminq_f32(vmaxq_f32(c50, vmin), vmax));
        vst1q_f32(c5_ptr + 1 * 4, vminq_f32(vmaxq_f32(c51, vmin), vmax));
        vst1q_f32(c5_ptr + 2 * 4, vminq_f32(vmaxq_f32(c52, vmin), vmax));
        vst1q_f32(c6_ptr + 0 * 4, vminq_f32(vmaxq_f32(c60, vmin), vmax));
        vst1q_f32(c6_ptr + 1 * 4, vminq_f32(vmaxq_f32(c61, vmin), vmax));
        vst1q_f32(c6_ptr + 2 * 4, vminq_f32(vmaxq_f32(c62, vmin), vmax));
        vst1q_f32(c7_ptr + 0 * 4, vminq_f32(vmaxq_f32(c70, vmin), vmax));
        vst1q_f32(c7_ptr + 1 * 4, vminq_f32(vmaxq_f32(c71, vmin), vmax));
        vst1q_f32(c7_ptr + 2 * 4, vminq_f32(vmaxq_f32(c72, vmin), vmax));

        c0_ptr += 12;
        c1_ptr += 12;
        c2_ptr += 12;
        c3_ptr += 12;
        c4_ptr += 12;
        c5_ptr += 12;
        c6_ptr += 12;
        c7_ptr += 12;

        B += ldb;
    }

    const int rem = nc - n;
    if (rem > 0) {
        float32x4_t c00 = vdupq_n_f32(0.0f), c01 = c00, c02 = c00;
        float32x4_t c10 = vdupq_n_f32(0.0f), c11 = c10, c12 = c10;
        float32x4_t c20 = vdupq_n_f32(0.0f), c21 = c20, c22 = c20;
        float32x4_t c30 = vdupq_n_f32(0.0f), c31 = c30, c32 = c30;
        float32x4_t c40 = vdupq_n_f32(0.0f), c41 = c40, c42 = c40;
        float32x4_t c50 = vdupq_n_f32(0.0f), c51 = c50, c52 = c50;
        float32x4_t c60 = vdupq_n_f32(0.0f), c61 = c60, c62 = c60;
        float32x4_t c70 = vdupq_n_f32(0.0f), c71 = c70, c72 = c70;

        const float* NNOPS_RESTRICT a0 = A + 0 * lda;
        const float* NNOPS_RESTRICT a1 = A + r1 * lda;
        const float* NNOPS_RESTRICT a2 = A + r2 * lda;
        const float* NNOPS_RESTRICT a3 = A + r3 * lda;
        const float* NNOPS_RESTRICT a4 = A + r4 * lda;
        const float* NNOPS_RESTRICT a5 = A + r5 * lda;
        const float* NNOPS_RESTRICT a6 = A + r6 * lda;
        const float* NNOPS_RESTRICT a7 = A + r7 * lda;
        const float* NNOPS_RESTRICT b_ptr = B;
        if (rem > 4) {
            for (int k = 0; k < kc; ++k) {
                const float32x4_t b0 = vld1q_f32(b_ptr + 0 * 4);
                const float32x4_t b1 = vld1q_f32(b_ptr + 1 * 4);
                const float32x4_t b2 = vld1q_f32(b_ptr + 2 * 4);
                b_ptr += 12;

                const float v0 = *a0++;
                const float v1 = *a1++;
                const float v2 = *a2++;
                const float v3 = *a3++;
                const float v4 = *a4++;
                const float v5 = *a5++;
                const float v6 = *a6++;
                const float v7 = *a7++;

                c00 = vfmaq_n_f32(c00, b0, v0);
                c01 = vfmaq_n_f32(c01, b1, v0);
                c02 = vfmaq_n_f32(c02, b2, v0);
                c10 = vfmaq_n_f32(c10, b0, v1);
                c11 = vfmaq_n_f32(c11, b1, v1);
                c12 = vfmaq_n_f32(c12, b2, v1);
                c20 = vfmaq_n_f32(c20, b0, v2);
                c21 = vfmaq_n_f32(c21, b1, v2);
                c22 = vfmaq_n_f32(c22, b2, v2);
                c30 = vfmaq_n_f32(c30, b0, v3);
                c31 = vfmaq_n_f32(c31, b1, v3);
                c32 = vfmaq_n_f32(c32, b2, v3);
                c40 = vfmaq_n_f32(c40, b0, v4);
                c41 = vfmaq_n_f32(c41, b1, v4);
                c42 = vfmaq_n_f32(c42, b2, v4);
                c50 = vfmaq_n_f32(c50, b0, v5);
                c51 = vfmaq_n_f32(c51, b1, v5);
                c52 = vfmaq_n_f32(c52, b2, v5);
                c60 = vfmaq_n_f32(c60, b0, v6);
                c61 = vfmaq_n_f32(c61, b1, v6);
                c62 = vfmaq_n_f32(c62, b2, v6);
                c70 = vfmaq_n_f32(c70, b0, v7);
                c71 = vfmaq_n_f32(c71, b1, v7);
                c72 = vfmaq_n_f32(c72, b2, v7);
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const float32x4_t b0 = vld1q_f32(b_ptr);
                b_ptr += 12;

                const float v0 = *a0++;
                const float v1 = *a1++;
                const float v2 = *a2++;
                const float v3 = *a3++;
                const float v4 = *a4++;
                const float v5 = *a5++;
                const float v6 = *a6++;
                const float v7 = *a7++;

                c00 = vfmaq_n_f32(c00, b0, v0);
                c10 = vfmaq_n_f32(c10, b0, v1);
                c20 = vfmaq_n_f32(c20, b0, v2);
                c30 = vfmaq_n_f32(c30, b0, v3);
                c40 = vfmaq_n_f32(c40, b0, v4);
                c50 = vfmaq_n_f32(c50, b0, v5);
                c60 = vfmaq_n_f32(c60, b0, v6);
                c70 = vfmaq_n_f32(c70, b0, v7);
            }
        }

        const float32x4_t acc[8][3] = {
            {c00, c01, c02}, {c10, c11, c12}, {c20, c21, c22}, {c30, c31, c32},
            {c40, c41, c42}, {c50, c51, c52}, {c60, c61, c62}, {c70, c71, c72},
        };
        float* NNOPS_RESTRICT c_rows[8] = {c0_ptr, c1_ptr, c2_ptr, c3_ptr,
                                           c4_ptr, c5_ptr, c6_ptr, c7_ptr};
        for (int r = 0; r < mr; ++r) {
            float* NNOPS_RESTRICT cp = c_rows[r];
            int rr = rem;
            int g = 0;
            for (; rr >= 4; rr -= 4, ++g) {
                float32x4_t v = acc[r][g];
                if constexpr (!zero_mode) {
                    v = vaddq_f32(v, vld1q_f32(cp));
                }
                vst1q_f32(cp, vminq_f32(vmaxq_f32(v, vmin), vmax));
                cp += 4;
            }
            if (rr > 0) {
                float lane[4];
                vst1q_f32(lane, acc[r][g]);
                for (int i = 0; i < rr; ++i) {
                    float v = lane[i];
                    if constexpr (!zero_mode) {
                        v += cp[i];
                    }
                    cp[i] = std::min(std::max(v, clamp_min), clamp_max);
                }
            }
        }
    }
}

template <bool zero_mode = false>
inline void mma_packed_b_mr4_nr12_mrxncxkc_f32(
    int mr, int nc, int kc, float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A, int lda,
    const float* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max) noexcept {

    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;

    float* NNOPS_RESTRICT c0_ptr = C + 0 * ldc;
    float* NNOPS_RESTRICT c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    float* NNOPS_RESTRICT c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    float* NNOPS_RESTRICT c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;

    const int ldb = static_cast<int>(mma_pack_align(12 * kc * sizeof(float)) / sizeof(float));
    const float32x4_t vmin = vdupq_n_f32(clamp_min);
    const float32x4_t vmax = vdupq_n_f32(clamp_max);

    int n = 0;
    for (; n < nc - 11; n += 12) {
        float32x4_t c00 = vdupq_n_f32(0.0f), c01 = c00, c02 = c00;
        float32x4_t c10 = vdupq_n_f32(0.0f), c11 = c10, c12 = c10;
        float32x4_t c20 = vdupq_n_f32(0.0f), c21 = c20, c22 = c20;
        float32x4_t c30 = vdupq_n_f32(0.0f), c31 = c30, c32 = c30;

        const float* NNOPS_RESTRICT a0 = A + 0 * lda;
        const float* NNOPS_RESTRICT a1 = A + r1 * lda;
        const float* NNOPS_RESTRICT a2 = A + r2 * lda;
        const float* NNOPS_RESTRICT a3 = A + r3 * lda;
        const float* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kc; ++k) {
            const float32x4_t b0 = vld1q_f32(b_ptr + 0 * 4);
            const float32x4_t b1 = vld1q_f32(b_ptr + 1 * 4);
            const float32x4_t b2 = vld1q_f32(b_ptr + 2 * 4);
            b_ptr += 12;

            const float v0 = *a0++;
            const float v1 = *a1++;
            const float v2 = *a2++;
            const float v3 = *a3++;

            c00 = vfmaq_n_f32(c00, b0, v0);
            c01 = vfmaq_n_f32(c01, b1, v0);
            c02 = vfmaq_n_f32(c02, b2, v0);
            c10 = vfmaq_n_f32(c10, b0, v1);
            c11 = vfmaq_n_f32(c11, b1, v1);
            c12 = vfmaq_n_f32(c12, b2, v1);
            c20 = vfmaq_n_f32(c20, b0, v2);
            c21 = vfmaq_n_f32(c21, b1, v2);
            c22 = vfmaq_n_f32(c22, b2, v2);
            c30 = vfmaq_n_f32(c30, b0, v3);
            c31 = vfmaq_n_f32(c31, b1, v3);
            c32 = vfmaq_n_f32(c32, b2, v3);
        }

        if constexpr (!zero_mode) {
            c00 = vaddq_f32(c00, vld1q_f32(c0_ptr + 0 * 4));
            c01 = vaddq_f32(c01, vld1q_f32(c0_ptr + 1 * 4));
            c02 = vaddq_f32(c02, vld1q_f32(c0_ptr + 2 * 4));
            c10 = vaddq_f32(c10, vld1q_f32(c1_ptr + 0 * 4));
            c11 = vaddq_f32(c11, vld1q_f32(c1_ptr + 1 * 4));
            c12 = vaddq_f32(c12, vld1q_f32(c1_ptr + 2 * 4));
            c20 = vaddq_f32(c20, vld1q_f32(c2_ptr + 0 * 4));
            c21 = vaddq_f32(c21, vld1q_f32(c2_ptr + 1 * 4));
            c22 = vaddq_f32(c22, vld1q_f32(c2_ptr + 2 * 4));
            c30 = vaddq_f32(c30, vld1q_f32(c3_ptr + 0 * 4));
            c31 = vaddq_f32(c31, vld1q_f32(c3_ptr + 1 * 4));
            c32 = vaddq_f32(c32, vld1q_f32(c3_ptr + 2 * 4));
        }

        vst1q_f32(c0_ptr + 0 * 4, vminq_f32(vmaxq_f32(c00, vmin), vmax));
        vst1q_f32(c0_ptr + 1 * 4, vminq_f32(vmaxq_f32(c01, vmin), vmax));
        vst1q_f32(c0_ptr + 2 * 4, vminq_f32(vmaxq_f32(c02, vmin), vmax));
        vst1q_f32(c1_ptr + 0 * 4, vminq_f32(vmaxq_f32(c10, vmin), vmax));
        vst1q_f32(c1_ptr + 1 * 4, vminq_f32(vmaxq_f32(c11, vmin), vmax));
        vst1q_f32(c1_ptr + 2 * 4, vminq_f32(vmaxq_f32(c12, vmin), vmax));
        vst1q_f32(c2_ptr + 0 * 4, vminq_f32(vmaxq_f32(c20, vmin), vmax));
        vst1q_f32(c2_ptr + 1 * 4, vminq_f32(vmaxq_f32(c21, vmin), vmax));
        vst1q_f32(c2_ptr + 2 * 4, vminq_f32(vmaxq_f32(c22, vmin), vmax));
        vst1q_f32(c3_ptr + 0 * 4, vminq_f32(vmaxq_f32(c30, vmin), vmax));
        vst1q_f32(c3_ptr + 1 * 4, vminq_f32(vmaxq_f32(c31, vmin), vmax));
        vst1q_f32(c3_ptr + 2 * 4, vminq_f32(vmaxq_f32(c32, vmin), vmax));

        c0_ptr += 12;
        c1_ptr += 12;
        c2_ptr += 12;
        c3_ptr += 12;

        B += ldb;
    }

    const int rem = nc - n;
    if (rem > 0) {
        float32x4_t c00 = vdupq_n_f32(0.0f), c01 = c00, c02 = c00;
        float32x4_t c10 = vdupq_n_f32(0.0f), c11 = c10, c12 = c10;
        float32x4_t c20 = vdupq_n_f32(0.0f), c21 = c20, c22 = c20;
        float32x4_t c30 = vdupq_n_f32(0.0f), c31 = c30, c32 = c30;

        const float* NNOPS_RESTRICT a0 = A + 0 * lda;
        const float* NNOPS_RESTRICT a1 = A + r1 * lda;
        const float* NNOPS_RESTRICT a2 = A + r2 * lda;
        const float* NNOPS_RESTRICT a3 = A + r3 * lda;
        const float* NNOPS_RESTRICT b_ptr = B;
        if (rem > 4) {
            for (int k = 0; k < kc; ++k) {
                const float32x4_t b0 = vld1q_f32(b_ptr + 0 * 4);
                const float32x4_t b1 = vld1q_f32(b_ptr + 1 * 4);
                const float32x4_t b2 = vld1q_f32(b_ptr + 2 * 4);
                b_ptr += 12;

                const float v0 = *a0++;
                const float v1 = *a1++;
                const float v2 = *a2++;
                const float v3 = *a3++;

                c00 = vfmaq_n_f32(c00, b0, v0);
                c01 = vfmaq_n_f32(c01, b1, v0);
                c02 = vfmaq_n_f32(c02, b2, v0);
                c10 = vfmaq_n_f32(c10, b0, v1);
                c11 = vfmaq_n_f32(c11, b1, v1);
                c12 = vfmaq_n_f32(c12, b2, v1);
                c20 = vfmaq_n_f32(c20, b0, v2);
                c21 = vfmaq_n_f32(c21, b1, v2);
                c22 = vfmaq_n_f32(c22, b2, v2);
                c30 = vfmaq_n_f32(c30, b0, v3);
                c31 = vfmaq_n_f32(c31, b1, v3);
                c32 = vfmaq_n_f32(c32, b2, v3);
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const float32x4_t b0 = vld1q_f32(b_ptr);
                b_ptr += 12;

                const float v0 = *a0++;
                const float v1 = *a1++;
                const float v2 = *a2++;
                const float v3 = *a3++;

                c00 = vfmaq_n_f32(c00, b0, v0);
                c10 = vfmaq_n_f32(c10, b0, v1);
                c20 = vfmaq_n_f32(c20, b0, v2);
                c30 = vfmaq_n_f32(c30, b0, v3);
            }
        }

        const float32x4_t acc[4][3] = {
            {c00, c01, c02}, {c10, c11, c12}, {c20, c21, c22}, {c30, c31, c32},
        };
        float* NNOPS_RESTRICT c_rows[4] = {c0_ptr, c1_ptr, c2_ptr, c3_ptr};
        for (int r = 0; r < mr; ++r) {
            float* NNOPS_RESTRICT cp = c_rows[r];
            int rr = rem;
            int g = 0;
            for (; rr >= 4; rr -= 4, ++g) {
                float32x4_t v = acc[r][g];
                if constexpr (!zero_mode) {
                    v = vaddq_f32(v, vld1q_f32(cp));
                }
                vst1q_f32(cp, vminq_f32(vmaxq_f32(v, vmin), vmax));
                cp += 4;
            }
            if (rr > 0) {
                float lane[4];
                vst1q_f32(lane, acc[r][g]);
                for (int i = 0; i < rr; ++i) {
                    float v = lane[i];
                    if constexpr (!zero_mode) {
                        v += cp[i];
                    }
                    cp[i] = std::min(std::max(v, clamp_min), clamp_max);
                }
            }
        }
    }
}

template <bool zero_mode = false>
inline void mma_packed_b_mr1_nr12_mrxncxkc_f32(
    int mr, int nc, int kc, float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A, int lda,
    const float* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max) noexcept {

    float* NNOPS_RESTRICT c0_ptr = C;

    const int ldb = static_cast<int>(mma_pack_align(12 * kc * sizeof(float)) / sizeof(float));
    const float32x4_t vmin = vdupq_n_f32(clamp_min);
    const float32x4_t vmax = vdupq_n_f32(clamp_max);

    int n = 0;
    for (; n < nc - 11; n += 12) {
        float32x4_t c00 = vdupq_n_f32(0.0f), c01 = c00, c02 = c00;

        const float* NNOPS_RESTRICT a_ptr = A;
        const float* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kc; ++k) {
            const float32x4_t b0 = vld1q_f32(b_ptr + 0 * 4);
            const float32x4_t b1 = vld1q_f32(b_ptr + 1 * 4);
            const float32x4_t b2 = vld1q_f32(b_ptr + 2 * 4);
            b_ptr += 12;

            const float a = *a_ptr++;
            a_ptr += lda - 1;

            c00 = vfmaq_n_f32(c00, b0, a);
            c01 = vfmaq_n_f32(c01, b1, a);
            c02 = vfmaq_n_f32(c02, b2, a);
        }

        if constexpr (!zero_mode) {
            c00 = vaddq_f32(c00, vld1q_f32(c0_ptr + 0 * 4));
            c01 = vaddq_f32(c01, vld1q_f32(c0_ptr + 1 * 4));
            c02 = vaddq_f32(c02, vld1q_f32(c0_ptr + 2 * 4));
        }

        vst1q_f32(c0_ptr + 0 * 4, vminq_f32(vmaxq_f32(c00, vmin), vmax));
        vst1q_f32(c0_ptr + 1 * 4, vminq_f32(vmaxq_f32(c01, vmin), vmax));
        vst1q_f32(c0_ptr + 2 * 4, vminq_f32(vmaxq_f32(c02, vmin), vmax));

        c0_ptr += 12;
        B += ldb;
    }

    const int rem = nc - n;
    if (rem > 0) {
        float32x4_t c00 = vdupq_n_f32(0.0f), c01 = c00, c02 = c00;

        const float* NNOPS_RESTRICT a_ptr = A;
        const float* NNOPS_RESTRICT b_ptr = B;
        if (rem > 4) {
            for (int k = 0; k < kc; ++k) {
                const float32x4_t b0 = vld1q_f32(b_ptr + 0 * 4);
                const float32x4_t b1 = vld1q_f32(b_ptr + 1 * 4);
                const float32x4_t b2 = vld1q_f32(b_ptr + 2 * 4);
                b_ptr += 12;

                const float a = *a_ptr++;
                a_ptr += lda - 1;

                c00 = vfmaq_n_f32(c00, b0, a);
                c01 = vfmaq_n_f32(c01, b1, a);
                c02 = vfmaq_n_f32(c02, b2, a);
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const float32x4_t b0 = vld1q_f32(b_ptr);
                b_ptr += 12;

                const float a = *a_ptr++;
                a_ptr += lda - 1;

                c00 = vfmaq_n_f32(c00, b0, a);
            }
        }

        const float32x4_t acc[3] = {c00, c01, c02};
        float* NNOPS_RESTRICT cp = c0_ptr;
        int rr = rem;
        int g = 0;
        for (; rr >= 4; rr -= 4, ++g) {
            float32x4_t v = acc[g];
            if constexpr (!zero_mode) {
                v = vaddq_f32(v, vld1q_f32(cp));
            }
            vst1q_f32(cp, vminq_f32(vmaxq_f32(v, vmin), vmax));
            cp += 4;
        }
        if (rr > 0) {
            float lane[4];
            vst1q_f32(lane, acc[g]);
            for (int i = 0; i < rr; ++i) {
                float v = lane[i];
                if constexpr (!zero_mode) {
                    v += cp[i];
                }
                cp[i] = std::min(std::max(v, clamp_min), clamp_max);
            }
        }
    }
}

// ---- dispatch tables, consumed by matmul_helper.cpp ---------------------
// Rows are the MR levels {8,4,1}; every kernel takes the runtime panel height,
// so a remainder panel below its own MR runs through the level above it.
template <bool Z>
inline constexpr auto mma_pack_fns_f32 = std::array{
    &mma_packed_ab_mr8_nr12_mrxncxkc_f32<Z>,
    &mma_packed_ab_mr4_nr12_mrxncxkc_f32<Z>,
    &mma_packed_ab_mr1_nr12_mrxncxkc_f32<Z>};

template <bool Z>
inline constexpr auto mma_direct_fns_f32 = std::array{
    &mma_packed_b_mr8_nr12_mrxncxkc_f32<Z>,
    &mma_packed_b_mr4_nr12_mrxncxkc_f32<Z>,
    &mma_packed_b_mr1_nr12_mrxncxkc_f32<Z>};

}  // namespace nnops::backend::cpu::aarch64
