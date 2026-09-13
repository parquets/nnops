#pragma once
/// @file mma_pack_f16.hpp
/// @brief AArch64 NEON float16 MMA (matrix micro-accumulate) packed-B kernels.
///
/// These use native NEON fp16 arithmetic (ARMv8.2-A+).
/// Tile sizes (AArch64-optimised):
///   M ∈ {8, 4, 1}    N ∈ {16, 8, 1}
///
/// mr=1 kernels accumulate with a residue split: the dot product is spread
/// over 8 independent fp16 chains (k mod 8) plus a tail chain, balanced-tree
/// reduced at the end. Same reassociation trick as MLAS — the fp16
/// accumulation error drops ~sqrt(8) with zero extra inner-loop work.
///
/// Reference: nn_compute/src/cpu/kernel/mma/aarch64/mma_pack_f16.hpp

#include <arm_neon.h>
#include <algorithm>
#include "backend/cpu/common/restrict.hpp"

namespace nnops::backend::cpu::aarch64 {

// =========================================================================
//  mr=1  kernels
// =========================================================================

inline void mma_pack_1x1_f16(float16_t* NNOPS_RESTRICT C, int ldc,
                             const float16_t* NNOPS_RESTRICT A,
                             const float16_t* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    float16_t c0 = 0.0f16;
    for (int k = 0; k < K; ++k) {
        c0 += *A++ * B[0];
        B += ldb;
    }
    c0 = c0 + C[0 * ldc];
    C[0 * ldc] = std::min(std::max(c0, static_cast<float16_t>(clamp_min)), static_cast<float16_t>(clamp_max));
}

inline void mma_pack_1x8_f16(float16_t* NNOPS_RESTRICT C, int ldc,
                             const float16_t* NNOPS_RESTRICT A,
                             const float16_t* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    // c{r}: fp16 accumulator for residue class r = k mod 8; ct: tail chain.
    float16x8_t c0 = vdupq_n_f16(0.0f16);
    float16x8_t c1 = vdupq_n_f16(0.0f16);
    float16x8_t c2 = vdupq_n_f16(0.0f16);
    float16x8_t c3 = vdupq_n_f16(0.0f16);
    float16x8_t c4 = vdupq_n_f16(0.0f16);
    float16x8_t c5 = vdupq_n_f16(0.0f16);
    float16x8_t c6 = vdupq_n_f16(0.0f16);
    float16x8_t c7 = vdupq_n_f16(0.0f16);
    float16x8_t ct = vdupq_n_f16(0.0f16);

    int k = 0;
    for (; k <= K - 8; k += 8) {
        const float16x8_t a = vld1q_f16(A + k);  // packed A is K-contiguous
        c0 = vfmaq_laneq_f16(c0, vld1q_f16(B + 0 * ldb), a, 0);
        c1 = vfmaq_laneq_f16(c1, vld1q_f16(B + 1 * ldb), a, 1);
        c2 = vfmaq_laneq_f16(c2, vld1q_f16(B + 2 * ldb), a, 2);
        c3 = vfmaq_laneq_f16(c3, vld1q_f16(B + 3 * ldb), a, 3);
        c4 = vfmaq_laneq_f16(c4, vld1q_f16(B + 4 * ldb), a, 4);
        c5 = vfmaq_laneq_f16(c5, vld1q_f16(B + 5 * ldb), a, 5);
        c6 = vfmaq_laneq_f16(c6, vld1q_f16(B + 6 * ldb), a, 6);
        c7 = vfmaq_laneq_f16(c7, vld1q_f16(B + 7 * ldb), a, 7);
        B += 8 * ldb;
    }
    for (; k < K; ++k) {  // tail: short chain of K mod 8 terms
        ct = vfmaq_n_f16(ct, vld1q_f16(B), A[k]);
        B += ldb;
    }

    // Balanced tree reduce of the 8 residue chains + tail.
    c0 = vaddq_f16(vaddq_f16(c0, c1), vaddq_f16(c2, c3));
    c4 = vaddq_f16(vaddq_f16(c4, c5), vaddq_f16(c6, c7));
    c0 = vaddq_f16(c0, vaddq_f16(c4, ct));

    c0 = vaddq_f16(c0, vld1q_f16(C));
    const float16x8_t vmin = vdupq_n_f16(static_cast<float16_t>(clamp_min));
    const float16x8_t vmax = vdupq_n_f16(static_cast<float16_t>(clamp_max));
    vst1q_f16(C, vminq_f16(vmaxq_f16(c0, vmin), vmax));
}

inline void mma_pack_1x16_f16(float16_t* NNOPS_RESTRICT C, int ldc,
                              const float16_t* NNOPS_RESTRICT A,
                              const float16_t* NNOPS_RESTRICT B,
                              int ldb, int K,
                              float clamp_min, float clamp_max) noexcept {
    // c{r}{h}: fp16 accumulator for residue class r = k mod 8, N-half h.
    float16x8_t c00 = vdupq_n_f16(0.0f16), c01 = vdupq_n_f16(0.0f16);
    float16x8_t c10 = vdupq_n_f16(0.0f16), c11 = vdupq_n_f16(0.0f16);
    float16x8_t c20 = vdupq_n_f16(0.0f16), c21 = vdupq_n_f16(0.0f16);
    float16x8_t c30 = vdupq_n_f16(0.0f16), c31 = vdupq_n_f16(0.0f16);
    float16x8_t c40 = vdupq_n_f16(0.0f16), c41 = vdupq_n_f16(0.0f16);
    float16x8_t c50 = vdupq_n_f16(0.0f16), c51 = vdupq_n_f16(0.0f16);
    float16x8_t c60 = vdupq_n_f16(0.0f16), c61 = vdupq_n_f16(0.0f16);
    float16x8_t c70 = vdupq_n_f16(0.0f16), c71 = vdupq_n_f16(0.0f16);
    float16x8_t ct0 = vdupq_n_f16(0.0f16), ct1 = vdupq_n_f16(0.0f16);

    int k = 0;
    for (; k <= K - 8; k += 8) {
        const float16x8_t a = vld1q_f16(A + k);  // packed A is K-contiguous

        c00 = vfmaq_laneq_f16(c00, vld1q_f16(B + 0 * ldb + 0 * 8), a, 0);
        c01 = vfmaq_laneq_f16(c01, vld1q_f16(B + 0 * ldb + 1 * 8), a, 0);
        c10 = vfmaq_laneq_f16(c10, vld1q_f16(B + 1 * ldb + 0 * 8), a, 1);
        c11 = vfmaq_laneq_f16(c11, vld1q_f16(B + 1 * ldb + 1 * 8), a, 1);
        c20 = vfmaq_laneq_f16(c20, vld1q_f16(B + 2 * ldb + 0 * 8), a, 2);
        c21 = vfmaq_laneq_f16(c21, vld1q_f16(B + 2 * ldb + 1 * 8), a, 2);
        c30 = vfmaq_laneq_f16(c30, vld1q_f16(B + 3 * ldb + 0 * 8), a, 3);
        c31 = vfmaq_laneq_f16(c31, vld1q_f16(B + 3 * ldb + 1 * 8), a, 3);
        c40 = vfmaq_laneq_f16(c40, vld1q_f16(B + 4 * ldb + 0 * 8), a, 4);
        c41 = vfmaq_laneq_f16(c41, vld1q_f16(B + 4 * ldb + 1 * 8), a, 4);
        c50 = vfmaq_laneq_f16(c50, vld1q_f16(B + 5 * ldb + 0 * 8), a, 5);
        c51 = vfmaq_laneq_f16(c51, vld1q_f16(B + 5 * ldb + 1 * 8), a, 5);
        c60 = vfmaq_laneq_f16(c60, vld1q_f16(B + 6 * ldb + 0 * 8), a, 6);
        c61 = vfmaq_laneq_f16(c61, vld1q_f16(B + 6 * ldb + 1 * 8), a, 6);
        c70 = vfmaq_laneq_f16(c70, vld1q_f16(B + 7 * ldb + 0 * 8), a, 7);
        c71 = vfmaq_laneq_f16(c71, vld1q_f16(B + 7 * ldb + 1 * 8), a, 7);

        B += 8 * ldb;
    }
    for (; k < K; ++k) {  // tail: short chain of K mod 8 terms
        const float16_t a = A[k];
        ct0 = vfmaq_n_f16(ct0, vld1q_f16(B + 0 * 8), a);
        ct1 = vfmaq_n_f16(ct1, vld1q_f16(B + 1 * 8), a);
        B += ldb;
    }

    // Balanced tree reduce of the 8 residue chains + tail, per N-half.
    c00 = vaddq_f16(vaddq_f16(c00, c10), vaddq_f16(c20, c30));
    c40 = vaddq_f16(vaddq_f16(c40, c50), vaddq_f16(c60, c70));
    c00 = vaddq_f16(c00, vaddq_f16(c40, ct0));

    c01 = vaddq_f16(vaddq_f16(c01, c11), vaddq_f16(c21, c31));
    c41 = vaddq_f16(vaddq_f16(c41, c51), vaddq_f16(c61, c71));
    c01 = vaddq_f16(c01, vaddq_f16(c41, ct1));

    const float16x8_t vmin = vdupq_n_f16(static_cast<float16_t>(clamp_min));
    const float16x8_t vmax = vdupq_n_f16(static_cast<float16_t>(clamp_max));

    c00 = vaddq_f16(c00, vld1q_f16(C + 0 * 8));
    c01 = vaddq_f16(c01, vld1q_f16(C + 1 * 8));

    vst1q_f16(C + 0 * 8, vminq_f16(vmaxq_f16(c00, vmin), vmax));
    vst1q_f16(C + 1 * 8, vminq_f16(vmaxq_f16(c01, vmin), vmax));
}

// =========================================================================
//  mr=4  kernels
// =========================================================================

inline void mma_pack_4x1_f16(float16_t* NNOPS_RESTRICT C, int ldc,
                             const float16_t* NNOPS_RESTRICT A,
                             const float16_t* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    float16_t c0 = 0.0f16, c1 = 0.0f16, c2 = 0.0f16, c3 = 0.0f16;
    for (int k = 0; k < K; ++k) {
        const float16_t b = B[0]; B += ldb;
        c0 += A[0] * b;
        c1 += A[1] * b;
        c2 += A[2] * b;
        c3 += A[3] * b;
        A += 4;
    }
    c0 += C[0 * ldc]; c0 = std::min(std::max(c0, static_cast<float16_t>(clamp_min)), static_cast<float16_t>(clamp_max)); C[0 * ldc] = c0;
    c1 += C[1 * ldc]; c1 = std::min(std::max(c1, static_cast<float16_t>(clamp_min)), static_cast<float16_t>(clamp_max)); C[1 * ldc] = c1;
    c2 += C[2 * ldc]; c2 = std::min(std::max(c2, static_cast<float16_t>(clamp_min)), static_cast<float16_t>(clamp_max)); C[2 * ldc] = c2;
    c3 += C[3 * ldc]; c3 = std::min(std::max(c3, static_cast<float16_t>(clamp_min)), static_cast<float16_t>(clamp_max)); C[3 * ldc] = c3;
}

inline void mma_pack_4x8_f16(float16_t* NNOPS_RESTRICT C, int ldc,
                             const float16_t* NNOPS_RESTRICT A,
                             const float16_t* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    float16x8_t c0 = vdupq_n_f16(0.0f16);
    float16x8_t c1 = vdupq_n_f16(0.0f16);
    float16x8_t c2 = vdupq_n_f16(0.0f16);
    float16x8_t c3 = vdupq_n_f16(0.0f16);

    for (int k = 0; k < K; ++k) {
        const float16x4_t a = vld1_f16(A); A += 4;
        const float16x8_t b = vld1q_f16(B); B += ldb;

        c0 = vfmaq_lane_f16(c0, b, a, 0);
        c1 = vfmaq_lane_f16(c1, b, a, 1);
        c2 = vfmaq_lane_f16(c2, b, a, 2);
        c3 = vfmaq_lane_f16(c3, b, a, 3);
    }

    const float16x8_t vmin = vdupq_n_f16(static_cast<float16_t>(clamp_min));
    const float16x8_t vmax = vdupq_n_f16(static_cast<float16_t>(clamp_max));

    c0 = vaddq_f16(c0, vld1q_f16(C + 0 * ldc));
    c1 = vaddq_f16(c1, vld1q_f16(C + 1 * ldc));
    c2 = vaddq_f16(c2, vld1q_f16(C + 2 * ldc));
    c3 = vaddq_f16(c3, vld1q_f16(C + 3 * ldc));

    vst1q_f16(C + 0 * ldc, vminq_f16(vmaxq_f16(c0, vmin), vmax));
    vst1q_f16(C + 1 * ldc, vminq_f16(vmaxq_f16(c1, vmin), vmax));
    vst1q_f16(C + 2 * ldc, vminq_f16(vmaxq_f16(c2, vmin), vmax));
    vst1q_f16(C + 3 * ldc, vminq_f16(vmaxq_f16(c3, vmin), vmax));
}

inline void mma_pack_4x16_f16(float16_t* NNOPS_RESTRICT C, int ldc,
                              const float16_t* NNOPS_RESTRICT A,
                              const float16_t* NNOPS_RESTRICT B,
                              int ldb, int K,
                              float clamp_min, float clamp_max) noexcept {
    float16x8_t c00 = vdupq_n_f16(0.0f16), c01 = vdupq_n_f16(0.0f16);
    float16x8_t c10 = vdupq_n_f16(0.0f16), c11 = vdupq_n_f16(0.0f16);
    float16x8_t c20 = vdupq_n_f16(0.0f16), c21 = vdupq_n_f16(0.0f16);
    float16x8_t c30 = vdupq_n_f16(0.0f16), c31 = vdupq_n_f16(0.0f16);

    for (int k = 0; k < K; ++k) {
        const float16x4_t a = vld1_f16(A); A += 4;
        const float16x8_t b0 = vld1q_f16(B + 0 * 8);
        const float16x8_t b1 = vld1q_f16(B + 1 * 8);
        B += ldb;

        c00 = vfmaq_lane_f16(c00, b0, a, 0);
        c01 = vfmaq_lane_f16(c01, b1, a, 0);
        c10 = vfmaq_lane_f16(c10, b0, a, 1);
        c11 = vfmaq_lane_f16(c11, b1, a, 1);
        c20 = vfmaq_lane_f16(c20, b0, a, 2);
        c21 = vfmaq_lane_f16(c21, b1, a, 2);
        c30 = vfmaq_lane_f16(c30, b0, a, 3);
        c31 = vfmaq_lane_f16(c31, b1, a, 3);
    }

    const float16x8_t vmin = vdupq_n_f16(static_cast<float16_t>(clamp_min));
    const float16x8_t vmax = vdupq_n_f16(static_cast<float16_t>(clamp_max));

    c00 = vaddq_f16(c00, vld1q_f16(C + 0 * ldc + 0 * 8));
    c01 = vaddq_f16(c01, vld1q_f16(C + 0 * ldc + 1 * 8));
    vst1q_f16(C + 0 * ldc + 0 * 8, vminq_f16(vmaxq_f16(c00, vmin), vmax));
    vst1q_f16(C + 0 * ldc + 1 * 8, vminq_f16(vmaxq_f16(c01, vmin), vmax));

    c10 = vaddq_f16(c10, vld1q_f16(C + 1 * ldc + 0 * 8));
    c11 = vaddq_f16(c11, vld1q_f16(C + 1 * ldc + 1 * 8));
    vst1q_f16(C + 1 * ldc + 0 * 8, vminq_f16(vmaxq_f16(c10, vmin), vmax));
    vst1q_f16(C + 1 * ldc + 1 * 8, vminq_f16(vmaxq_f16(c11, vmin), vmax));

    c20 = vaddq_f16(c20, vld1q_f16(C + 2 * ldc + 0 * 8));
    c21 = vaddq_f16(c21, vld1q_f16(C + 2 * ldc + 1 * 8));
    vst1q_f16(C + 2 * ldc + 0 * 8, vminq_f16(vmaxq_f16(c20, vmin), vmax));
    vst1q_f16(C + 2 * ldc + 1 * 8, vminq_f16(vmaxq_f16(c21, vmin), vmax));

    c30 = vaddq_f16(c30, vld1q_f16(C + 3 * ldc + 0 * 8));
    c31 = vaddq_f16(c31, vld1q_f16(C + 3 * ldc + 1 * 8));
    vst1q_f16(C + 3 * ldc + 0 * 8, vminq_f16(vmaxq_f16(c30, vmin), vmax));
    vst1q_f16(C + 3 * ldc + 1 * 8, vminq_f16(vmaxq_f16(c31, vmin), vmax));
}

// =========================================================================
//  mr=8  kernels
// =========================================================================

inline void mma_pack_8x1_f16(float16_t* NNOPS_RESTRICT C, int ldc,
                             const float16_t* NNOPS_RESTRICT A,
                             const float16_t* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    float16_t c0 = 0.0f16, c1 = 0.0f16, c2 = 0.0f16, c3 = 0.0f16;
    float16_t c4 = 0.0f16, c5 = 0.0f16, c6 = 0.0f16, c7 = 0.0f16;
    for (int k = 0; k < K; ++k) {
        const float16_t b = B[0]; B += ldb;
        c0 += A[0] * b;
        c1 += A[1] * b;
        c2 += A[2] * b;
        c3 += A[3] * b;
        c4 += A[4] * b;
        c5 += A[5] * b;
        c6 += A[6] * b;
        c7 += A[7] * b;
        A += 8;
    }
    c0 += C[0 * ldc]; c0 = std::min(std::max(c0, static_cast<float16_t>(clamp_min)), static_cast<float16_t>(clamp_max)); C[0 * ldc] = c0;
    c1 += C[1 * ldc]; c1 = std::min(std::max(c1, static_cast<float16_t>(clamp_min)), static_cast<float16_t>(clamp_max)); C[1 * ldc] = c1;
    c2 += C[2 * ldc]; c2 = std::min(std::max(c2, static_cast<float16_t>(clamp_min)), static_cast<float16_t>(clamp_max)); C[2 * ldc] = c2;
    c3 += C[3 * ldc]; c3 = std::min(std::max(c3, static_cast<float16_t>(clamp_min)), static_cast<float16_t>(clamp_max)); C[3 * ldc] = c3;
    c4 += C[4 * ldc]; c4 = std::min(std::max(c4, static_cast<float16_t>(clamp_min)), static_cast<float16_t>(clamp_max)); C[4 * ldc] = c4;
    c5 += C[5 * ldc]; c5 = std::min(std::max(c5, static_cast<float16_t>(clamp_min)), static_cast<float16_t>(clamp_max)); C[5 * ldc] = c5;
    c6 += C[6 * ldc]; c6 = std::min(std::max(c6, static_cast<float16_t>(clamp_min)), static_cast<float16_t>(clamp_max)); C[6 * ldc] = c6;
    c7 += C[7 * ldc]; c7 = std::min(std::max(c7, static_cast<float16_t>(clamp_min)), static_cast<float16_t>(clamp_max)); C[7 * ldc] = c7;
}

inline void mma_pack_8x8_f16(float16_t* NNOPS_RESTRICT C, int ldc,
                             const float16_t* NNOPS_RESTRICT A,
                             const float16_t* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    float16x8_t c0 = vdupq_n_f16(0.0f16);
    float16x8_t c1 = vdupq_n_f16(0.0f16);
    float16x8_t c2 = vdupq_n_f16(0.0f16);
    float16x8_t c3 = vdupq_n_f16(0.0f16);
    float16x8_t c4 = vdupq_n_f16(0.0f16);
    float16x8_t c5 = vdupq_n_f16(0.0f16);
    float16x8_t c6 = vdupq_n_f16(0.0f16);
    float16x8_t c7 = vdupq_n_f16(0.0f16);

    for (int k = 0; k < K; ++k) {
        const float16x8_t a = vld1q_f16(A); A += 8;
        const float16x8_t b = vld1q_f16(B); B += ldb;

        c0 = vfmaq_laneq_f16(c0, b, a, 0);
        c1 = vfmaq_laneq_f16(c1, b, a, 1);
        c2 = vfmaq_laneq_f16(c2, b, a, 2);
        c3 = vfmaq_laneq_f16(c3, b, a, 3);
        c4 = vfmaq_laneq_f16(c4, b, a, 4);
        c5 = vfmaq_laneq_f16(c5, b, a, 5);
        c6 = vfmaq_laneq_f16(c6, b, a, 6);
        c7 = vfmaq_laneq_f16(c7, b, a, 7);
    }

    const float16x8_t vmin = vdupq_n_f16(static_cast<float16_t>(clamp_min));
    const float16x8_t vmax = vdupq_n_f16(static_cast<float16_t>(clamp_max));

    c0 = vaddq_f16(c0, vld1q_f16(C + 0 * ldc));
    c1 = vaddq_f16(c1, vld1q_f16(C + 1 * ldc));
    c2 = vaddq_f16(c2, vld1q_f16(C + 2 * ldc));
    c3 = vaddq_f16(c3, vld1q_f16(C + 3 * ldc));
    c4 = vaddq_f16(c4, vld1q_f16(C + 4 * ldc));
    c5 = vaddq_f16(c5, vld1q_f16(C + 5 * ldc));
    c6 = vaddq_f16(c6, vld1q_f16(C + 6 * ldc));
    c7 = vaddq_f16(c7, vld1q_f16(C + 7 * ldc));

    vst1q_f16(C + 0 * ldc, vminq_f16(vmaxq_f16(c0, vmin), vmax));
    vst1q_f16(C + 1 * ldc, vminq_f16(vmaxq_f16(c1, vmin), vmax));
    vst1q_f16(C + 2 * ldc, vminq_f16(vmaxq_f16(c2, vmin), vmax));
    vst1q_f16(C + 3 * ldc, vminq_f16(vmaxq_f16(c3, vmin), vmax));
    vst1q_f16(C + 4 * ldc, vminq_f16(vmaxq_f16(c4, vmin), vmax));
    vst1q_f16(C + 5 * ldc, vminq_f16(vmaxq_f16(c5, vmin), vmax));
    vst1q_f16(C + 6 * ldc, vminq_f16(vmaxq_f16(c6, vmin), vmax));
    vst1q_f16(C + 7 * ldc, vminq_f16(vmaxq_f16(c7, vmin), vmax));
}

inline void mma_pack_8x16_f16(float16_t* NNOPS_RESTRICT C, int ldc,
                              const float16_t* NNOPS_RESTRICT A,
                              const float16_t* NNOPS_RESTRICT B,
                              int ldb, int K,
                              float clamp_min, float clamp_max) noexcept {
    float16x8_t c00 = vdupq_n_f16(0.0f16), c01 = vdupq_n_f16(0.0f16);
    float16x8_t c10 = vdupq_n_f16(0.0f16), c11 = vdupq_n_f16(0.0f16);
    float16x8_t c20 = vdupq_n_f16(0.0f16), c21 = vdupq_n_f16(0.0f16);
    float16x8_t c30 = vdupq_n_f16(0.0f16), c31 = vdupq_n_f16(0.0f16);
    float16x8_t c40 = vdupq_n_f16(0.0f16), c41 = vdupq_n_f16(0.0f16);
    float16x8_t c50 = vdupq_n_f16(0.0f16), c51 = vdupq_n_f16(0.0f16);
    float16x8_t c60 = vdupq_n_f16(0.0f16), c61 = vdupq_n_f16(0.0f16);
    float16x8_t c70 = vdupq_n_f16(0.0f16), c71 = vdupq_n_f16(0.0f16);

    for (int k = 0; k < K; ++k) {
        const float16x8_t a = vld1q_f16(A); A += 8;
        const float16x8_t b0 = vld1q_f16(B + 0 * 8);
        const float16x8_t b1 = vld1q_f16(B + 1 * 8);
        B += ldb;

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

    const float16x8_t vmin = vdupq_n_f16(static_cast<float16_t>(clamp_min));
    const float16x8_t vmax = vdupq_n_f16(static_cast<float16_t>(clamp_max));

    c00 = vaddq_f16(c00, vld1q_f16(C + 0 * ldc + 0 * 8));
    c01 = vaddq_f16(c01, vld1q_f16(C + 0 * ldc + 1 * 8));
    vst1q_f16(C + 0 * ldc + 0 * 8, vminq_f16(vmaxq_f16(c00, vmin), vmax));
    vst1q_f16(C + 0 * ldc + 1 * 8, vminq_f16(vmaxq_f16(c01, vmin), vmax));

    c10 = vaddq_f16(c10, vld1q_f16(C + 1 * ldc + 0 * 8));
    c11 = vaddq_f16(c11, vld1q_f16(C + 1 * ldc + 1 * 8));
    vst1q_f16(C + 1 * ldc + 0 * 8, vminq_f16(vmaxq_f16(c10, vmin), vmax));
    vst1q_f16(C + 1 * ldc + 1 * 8, vminq_f16(vmaxq_f16(c11, vmin), vmax));

    c20 = vaddq_f16(c20, vld1q_f16(C + 2 * ldc + 0 * 8));
    c21 = vaddq_f16(c21, vld1q_f16(C + 2 * ldc + 1 * 8));
    vst1q_f16(C + 2 * ldc + 0 * 8, vminq_f16(vmaxq_f16(c20, vmin), vmax));
    vst1q_f16(C + 2 * ldc + 1 * 8, vminq_f16(vmaxq_f16(c21, vmin), vmax));

    c30 = vaddq_f16(c30, vld1q_f16(C + 3 * ldc + 0 * 8));
    c31 = vaddq_f16(c31, vld1q_f16(C + 3 * ldc + 1 * 8));
    vst1q_f16(C + 3 * ldc + 0 * 8, vminq_f16(vmaxq_f16(c30, vmin), vmax));
    vst1q_f16(C + 3 * ldc + 1 * 8, vminq_f16(vmaxq_f16(c31, vmin), vmax));

    c40 = vaddq_f16(c40, vld1q_f16(C + 4 * ldc + 0 * 8));
    c41 = vaddq_f16(c41, vld1q_f16(C + 4 * ldc + 1 * 8));
    vst1q_f16(C + 4 * ldc + 0 * 8, vminq_f16(vmaxq_f16(c40, vmin), vmax));
    vst1q_f16(C + 4 * ldc + 1 * 8, vminq_f16(vmaxq_f16(c41, vmin), vmax));

    c50 = vaddq_f16(c50, vld1q_f16(C + 5 * ldc + 0 * 8));
    c51 = vaddq_f16(c51, vld1q_f16(C + 5 * ldc + 1 * 8));
    vst1q_f16(C + 5 * ldc + 0 * 8, vminq_f16(vmaxq_f16(c50, vmin), vmax));
    vst1q_f16(C + 5 * ldc + 1 * 8, vminq_f16(vmaxq_f16(c51, vmin), vmax));

    c60 = vaddq_f16(c60, vld1q_f16(C + 6 * ldc + 0 * 8));
    c61 = vaddq_f16(c61, vld1q_f16(C + 6 * ldc + 1 * 8));
    vst1q_f16(C + 6 * ldc + 0 * 8, vminq_f16(vmaxq_f16(c60, vmin), vmax));
    vst1q_f16(C + 6 * ldc + 1 * 8, vminq_f16(vmaxq_f16(c61, vmin), vmax));

    c70 = vaddq_f16(c70, vld1q_f16(C + 7 * ldc + 0 * 8));
    c71 = vaddq_f16(c71, vld1q_f16(C + 7 * ldc + 1 * 8));
    vst1q_f16(C + 7 * ldc + 0 * 8, vminq_f16(vmaxq_f16(c70, vmin), vmax));
    vst1q_f16(C + 7 * ldc + 1 * 8, vminq_f16(vmaxq_f16(c71, vmin), vmax));
}

}  // namespace nnops::backend::cpu::aarch64
