#pragma once
/// @file mma_direct_f16.hpp
/// @brief AArch64 NEON float16 MMA direct (unpacked) micro-kernels.
///
/// These kernels compute C[mr][nr] += A[mr][K] × B[K][nr] directly
/// from row-major A (stride lda) and B (stride ldb), without packing.
/// Uses native NEON fp16 arithmetic (ARMv8.2-A+ required).
///
/// Tile sizes (AArch64-optimised):
///   M ∈ {8, 4, 1}    N ∈ {16, 8, 1}
///
/// mr=4/8 kernels unroll K by 4: four f16 A values are loaded as a
/// float16x4_t and broadcast lane-by-lane with vfmaq_lane_f16.
/// mr=1 kernels unroll K by 8 across 8 residue-split fp16 accumulators
/// (k mod 8, balanced-tree reduced at the end) — the same reassociation
/// MLAS uses to bound fp16 accumulation error. Tails handle remaining K.
///
/// Reference: nn_compute/src/cpu/kernel/mma/aarch64/mma_direct_f16.hpp

#include <arm_neon.h>
#include <algorithm>
#include "backend/cpu/common/restrict.hpp"

namespace nnops::backend::cpu::aarch64 {

// =========================================================================
//  mr=1  kernels
// =========================================================================

inline void mma_direct_1x1_f16(
    float16_t* NNOPS_RESTRICT C, int ldc,
    const float16_t* NNOPS_RESTRICT A, int lda,
    const float16_t* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const float16_t* NNOPS_RESTRICT A_ptr0 = A;

    float16_t c0 = 0.0f16;

    for (int k = 0; k < K; ++k) {
        c0 += *A_ptr0++ * B[0];
        B += ldb;
    }
    c0 = c0 + C[0];
    C[0] = std::min(std::max(c0, static_cast<float16_t>(clamp_min)), static_cast<float16_t>(clamp_max));
}

inline void mma_direct_1x8_f16(
    float16_t* NNOPS_RESTRICT C, int ldc,
    const float16_t* NNOPS_RESTRICT A, int lda,
    const float16_t* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const float16_t* NNOPS_RESTRICT A_ptr0 = A;

    // v_c{r}: fp16 accumulator for residue class r = k mod 8; v_ct: tail.
    float16x8_t v_c0 = vdupq_n_f16(0.0f16);
    float16x8_t v_c1 = vdupq_n_f16(0.0f16);
    float16x8_t v_c2 = vdupq_n_f16(0.0f16);
    float16x8_t v_c3 = vdupq_n_f16(0.0f16);
    float16x8_t v_c4 = vdupq_n_f16(0.0f16);
    float16x8_t v_c5 = vdupq_n_f16(0.0f16);
    float16x8_t v_c6 = vdupq_n_f16(0.0f16);
    float16x8_t v_c7 = vdupq_n_f16(0.0f16);
    float16x8_t v_ct = vdupq_n_f16(0.0f16);

    int k = 0;
    for (; k <= K - 8; k += 8) {
        const float16x8_t v_a0 = vld1q_f16(A_ptr0); A_ptr0 += 8;

        v_c0 = vfmaq_laneq_f16(v_c0, vld1q_f16(B + 0 * ldb), v_a0, 0);
        v_c1 = vfmaq_laneq_f16(v_c1, vld1q_f16(B + 1 * ldb), v_a0, 1);
        v_c2 = vfmaq_laneq_f16(v_c2, vld1q_f16(B + 2 * ldb), v_a0, 2);
        v_c3 = vfmaq_laneq_f16(v_c3, vld1q_f16(B + 3 * ldb), v_a0, 3);
        v_c4 = vfmaq_laneq_f16(v_c4, vld1q_f16(B + 4 * ldb), v_a0, 4);
        v_c5 = vfmaq_laneq_f16(v_c5, vld1q_f16(B + 5 * ldb), v_a0, 5);
        v_c6 = vfmaq_laneq_f16(v_c6, vld1q_f16(B + 6 * ldb), v_a0, 6);
        v_c7 = vfmaq_laneq_f16(v_c7, vld1q_f16(B + 7 * ldb), v_a0, 7);

        B += 8 * ldb;
    }
    for (; k < K; ++k) {  // tail: short chain of K mod 8 terms
        v_ct = vfmaq_n_f16(v_ct, vld1q_f16(B), *A_ptr0++);
        B += ldb;
    }

    // Balanced tree reduce of the 8 residue chains + tail.
    v_c0 = vaddq_f16(vaddq_f16(v_c0, v_c1), vaddq_f16(v_c2, v_c3));
    v_c4 = vaddq_f16(vaddq_f16(v_c4, v_c5), vaddq_f16(v_c6, v_c7));
    v_c0 = vaddq_f16(v_c0, vaddq_f16(v_c4, v_ct));

    v_c0 = vaddq_f16(v_c0, vld1q_f16(C));

    const float16x8_t v_min = vdupq_n_f16(static_cast<float16_t>(clamp_min));
    const float16x8_t v_max = vdupq_n_f16(static_cast<float16_t>(clamp_max));
    vst1q_f16(C, vminq_f16(vmaxq_f16(v_c0, v_min), v_max));
}

inline void mma_direct_1x16_f16(
    float16_t* NNOPS_RESTRICT C, int ldc,
    const float16_t* NNOPS_RESTRICT A, int lda,
    const float16_t* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const float16_t* NNOPS_RESTRICT A_ptr0 = A;

    // v_c{r}{h}: fp16 accumulator for residue class r = k mod 8, N-half h.
    float16x8_t v_c00 = vdupq_n_f16(0.0f16), v_c01 = vdupq_n_f16(0.0f16);
    float16x8_t v_c10 = vdupq_n_f16(0.0f16), v_c11 = vdupq_n_f16(0.0f16);
    float16x8_t v_c20 = vdupq_n_f16(0.0f16), v_c21 = vdupq_n_f16(0.0f16);
    float16x8_t v_c30 = vdupq_n_f16(0.0f16), v_c31 = vdupq_n_f16(0.0f16);
    float16x8_t v_c40 = vdupq_n_f16(0.0f16), v_c41 = vdupq_n_f16(0.0f16);
    float16x8_t v_c50 = vdupq_n_f16(0.0f16), v_c51 = vdupq_n_f16(0.0f16);
    float16x8_t v_c60 = vdupq_n_f16(0.0f16), v_c61 = vdupq_n_f16(0.0f16);
    float16x8_t v_c70 = vdupq_n_f16(0.0f16), v_c71 = vdupq_n_f16(0.0f16);
    float16x8_t v_ct0 = vdupq_n_f16(0.0f16), v_ct1 = vdupq_n_f16(0.0f16);

    int k = 0;
    for (; k <= K - 8; k += 8) {
        const float16x8_t v_a0 = vld1q_f16(A_ptr0); A_ptr0 += 8;

        v_c00 = vfmaq_laneq_f16(v_c00, vld1q_f16(B + 0 * ldb + 0 * 8), v_a0, 0);
        v_c01 = vfmaq_laneq_f16(v_c01, vld1q_f16(B + 0 * ldb + 1 * 8), v_a0, 0);
        v_c10 = vfmaq_laneq_f16(v_c10, vld1q_f16(B + 1 * ldb + 0 * 8), v_a0, 1);
        v_c11 = vfmaq_laneq_f16(v_c11, vld1q_f16(B + 1 * ldb + 1 * 8), v_a0, 1);
        v_c20 = vfmaq_laneq_f16(v_c20, vld1q_f16(B + 2 * ldb + 0 * 8), v_a0, 2);
        v_c21 = vfmaq_laneq_f16(v_c21, vld1q_f16(B + 2 * ldb + 1 * 8), v_a0, 2);
        v_c30 = vfmaq_laneq_f16(v_c30, vld1q_f16(B + 3 * ldb + 0 * 8), v_a0, 3);
        v_c31 = vfmaq_laneq_f16(v_c31, vld1q_f16(B + 3 * ldb + 1 * 8), v_a0, 3);
        v_c40 = vfmaq_laneq_f16(v_c40, vld1q_f16(B + 4 * ldb + 0 * 8), v_a0, 4);
        v_c41 = vfmaq_laneq_f16(v_c41, vld1q_f16(B + 4 * ldb + 1 * 8), v_a0, 4);
        v_c50 = vfmaq_laneq_f16(v_c50, vld1q_f16(B + 5 * ldb + 0 * 8), v_a0, 5);
        v_c51 = vfmaq_laneq_f16(v_c51, vld1q_f16(B + 5 * ldb + 1 * 8), v_a0, 5);
        v_c60 = vfmaq_laneq_f16(v_c60, vld1q_f16(B + 6 * ldb + 0 * 8), v_a0, 6);
        v_c61 = vfmaq_laneq_f16(v_c61, vld1q_f16(B + 6 * ldb + 1 * 8), v_a0, 6);
        v_c70 = vfmaq_laneq_f16(v_c70, vld1q_f16(B + 7 * ldb + 0 * 8), v_a0, 7);
        v_c71 = vfmaq_laneq_f16(v_c71, vld1q_f16(B + 7 * ldb + 1 * 8), v_a0, 7);

        B += 8 * ldb;
    }
    for (; k < K; ++k) {  // tail: short chain of K mod 8 terms
        const float16x8_t v_b0 = vld1q_f16(B + 0 * 8);
        const float16x8_t v_b1 = vld1q_f16(B + 1 * 8);

        v_ct0 = vfmaq_n_f16(v_ct0, v_b0, A_ptr0[0]);
        v_ct1 = vfmaq_n_f16(v_ct1, v_b1, A_ptr0[0]);
        A_ptr0 += 1;

        B += ldb;
    }

    // Balanced tree reduce of the 8 residue chains + tail, per N-half.
    v_c00 = vaddq_f16(vaddq_f16(v_c00, v_c10), vaddq_f16(v_c20, v_c30));
    v_c40 = vaddq_f16(vaddq_f16(v_c40, v_c50), vaddq_f16(v_c60, v_c70));
    v_c00 = vaddq_f16(v_c00, vaddq_f16(v_c40, v_ct0));

    v_c01 = vaddq_f16(vaddq_f16(v_c01, v_c11), vaddq_f16(v_c21, v_c31));
    v_c41 = vaddq_f16(vaddq_f16(v_c41, v_c51), vaddq_f16(v_c61, v_c71));
    v_c01 = vaddq_f16(v_c01, vaddq_f16(v_c41, v_ct1));

    const float16x8_t v_min = vdupq_n_f16(static_cast<float16_t>(clamp_min));
    const float16x8_t v_max = vdupq_n_f16(static_cast<float16_t>(clamp_max));

    v_c00 = vaddq_f16(v_c00, vld1q_f16(C + 0 * 8));
    v_c01 = vaddq_f16(v_c01, vld1q_f16(C + 1 * 8));

    vst1q_f16(C + 0 * 8, vminq_f16(vmaxq_f16(v_c00, v_min), v_max));
    vst1q_f16(C + 1 * 8, vminq_f16(vmaxq_f16(v_c01, v_min), v_max));
}

// =========================================================================
//  mr=4  kernels
// =========================================================================

inline void mma_direct_4x1_f16(
    float16_t* NNOPS_RESTRICT C, int ldc,
    const float16_t* NNOPS_RESTRICT A, int lda,
    const float16_t* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const float16_t* NNOPS_RESTRICT A_ptr0 = A + 0 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr1 = A + 1 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr2 = A + 2 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr3 = A + 3 * lda;

    float16_t c0 = 0.0f16, c1 = 0.0f16, c2 = 0.0f16, c3 = 0.0f16;

    for (int k = 0; k < K; ++k) {
        const float16_t b0 = B[0];
        c0 += *A_ptr0++ * b0;
        c1 += *A_ptr1++ * b0;
        c2 += *A_ptr2++ * b0;
        c3 += *A_ptr3++ * b0;

        B += ldb;
    }

    auto write = [&](float16_t* dst, float16_t acc) {
        float16_t v = *dst + acc;
        *dst = std::min(std::max(v, static_cast<float16_t>(clamp_min)), static_cast<float16_t>(clamp_max));
    };
    write(C + 0 * ldc, c0);
    write(C + 1 * ldc, c1);
    write(C + 2 * ldc, c2);
    write(C + 3 * ldc, c3);
}

inline void mma_direct_4x8_f16(
    float16_t* NNOPS_RESTRICT C, int ldc,
    const float16_t* NNOPS_RESTRICT A, int lda,
    const float16_t* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const float16_t* NNOPS_RESTRICT A_ptr0 = A + 0 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr1 = A + 1 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr2 = A + 2 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr3 = A + 3 * lda;

    float16x8_t v_c00 = vdupq_n_f16(0.0f16);
    float16x8_t v_c10 = v_c00;
    float16x8_t v_c20 = v_c00;
    float16x8_t v_c30 = v_c00;

    int k = 0;
    for (; k < K - 3; k += 4) {
        const float16x4_t v_a0 = vld1_f16(A_ptr0); A_ptr0 += 4;
        const float16x4_t v_a1 = vld1_f16(A_ptr1); A_ptr1 += 4;
        const float16x4_t v_a2 = vld1_f16(A_ptr2); A_ptr2 += 4;
        const float16x4_t v_a3 = vld1_f16(A_ptr3); A_ptr3 += 4;

        float16x8_t v_b0 = vld1q_f16(B + 0 * ldb);
        v_c00 = vfmaq_lane_f16(v_c00, v_b0, v_a0, 0);
        v_c10 = vfmaq_lane_f16(v_c10, v_b0, v_a1, 0);
        v_c20 = vfmaq_lane_f16(v_c20, v_b0, v_a2, 0);
        v_c30 = vfmaq_lane_f16(v_c30, v_b0, v_a3, 0);

        v_b0 = vld1q_f16(B + 1 * ldb);
        v_c00 = vfmaq_lane_f16(v_c00, v_b0, v_a0, 1);
        v_c10 = vfmaq_lane_f16(v_c10, v_b0, v_a1, 1);
        v_c20 = vfmaq_lane_f16(v_c20, v_b0, v_a2, 1);
        v_c30 = vfmaq_lane_f16(v_c30, v_b0, v_a3, 1);

        v_b0 = vld1q_f16(B + 2 * ldb);
        v_c00 = vfmaq_lane_f16(v_c00, v_b0, v_a0, 2);
        v_c10 = vfmaq_lane_f16(v_c10, v_b0, v_a1, 2);
        v_c20 = vfmaq_lane_f16(v_c20, v_b0, v_a2, 2);
        v_c30 = vfmaq_lane_f16(v_c30, v_b0, v_a3, 2);

        v_b0 = vld1q_f16(B + 3 * ldb);
        v_c00 = vfmaq_lane_f16(v_c00, v_b0, v_a0, 3);
        v_c10 = vfmaq_lane_f16(v_c10, v_b0, v_a1, 3);
        v_c20 = vfmaq_lane_f16(v_c20, v_b0, v_a2, 3);
        v_c30 = vfmaq_lane_f16(v_c30, v_b0, v_a3, 3);

        B += 4 * ldb;
    }
    for (; k < K; ++k) {
        const float16x8_t v_b0 = vld1q_f16(B); B += ldb;

        v_c00 = vfmaq_n_f16(v_c00, v_b0, *A_ptr0++);
        v_c10 = vfmaq_n_f16(v_c10, v_b0, *A_ptr1++);
        v_c20 = vfmaq_n_f16(v_c20, v_b0, *A_ptr2++);
        v_c30 = vfmaq_n_f16(v_c30, v_b0, *A_ptr3++);
    }

    const float16x8_t v_min = vdupq_n_f16(static_cast<float16_t>(clamp_min));
    const float16x8_t v_max = vdupq_n_f16(static_cast<float16_t>(clamp_max));

    v_c00 = vaddq_f16(v_c00, vld1q_f16(C + 0 * ldc));
    v_c10 = vaddq_f16(v_c10, vld1q_f16(C + 1 * ldc));
    v_c20 = vaddq_f16(v_c20, vld1q_f16(C + 2 * ldc));
    v_c30 = vaddq_f16(v_c30, vld1q_f16(C + 3 * ldc));

    vst1q_f16(C + 0 * ldc, vminq_f16(vmaxq_f16(v_c00, v_min), v_max));
    vst1q_f16(C + 1 * ldc, vminq_f16(vmaxq_f16(v_c10, v_min), v_max));
    vst1q_f16(C + 2 * ldc, vminq_f16(vmaxq_f16(v_c20, v_min), v_max));
    vst1q_f16(C + 3 * ldc, vminq_f16(vmaxq_f16(v_c30, v_min), v_max));
}

inline void mma_direct_4x16_f16(
    float16_t* NNOPS_RESTRICT C, int ldc,
    const float16_t* NNOPS_RESTRICT A, int lda,
    const float16_t* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const float16_t* NNOPS_RESTRICT A_ptr0 = A + 0 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr1 = A + 1 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr2 = A + 2 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr3 = A + 3 * lda;

    float16x8_t c00 = vdupq_n_f16(0.0f16), c01 = vdupq_n_f16(0.0f16);
    float16x8_t c10 = vdupq_n_f16(0.0f16), c11 = vdupq_n_f16(0.0f16);
    float16x8_t c20 = vdupq_n_f16(0.0f16), c21 = vdupq_n_f16(0.0f16);
    float16x8_t c30 = vdupq_n_f16(0.0f16), c31 = vdupq_n_f16(0.0f16);

    int k = 0;
    for (; k < K - 3; k += 4) {
        const float16x4_t v_a0 = vld1_f16(A_ptr0); A_ptr0 += 4;
        const float16x4_t v_a1 = vld1_f16(A_ptr1); A_ptr1 += 4;
        const float16x4_t v_a2 = vld1_f16(A_ptr2); A_ptr2 += 4;
        const float16x4_t v_a3 = vld1_f16(A_ptr3); A_ptr3 += 4;

        // kk=0
        float16x8_t v_b0 = vld1q_f16(B + 0 * ldb + 0 * 8);
        float16x8_t v_b1 = vld1q_f16(B + 0 * ldb + 1 * 8);
        c00 = vfmaq_lane_f16(c00, v_b0, v_a0, 0);
        c01 = vfmaq_lane_f16(c01, v_b1, v_a0, 0);
        c10 = vfmaq_lane_f16(c10, v_b0, v_a1, 0);
        c11 = vfmaq_lane_f16(c11, v_b1, v_a1, 0);
        c20 = vfmaq_lane_f16(c20, v_b0, v_a2, 0);
        c21 = vfmaq_lane_f16(c21, v_b1, v_a2, 0);
        c30 = vfmaq_lane_f16(c30, v_b0, v_a3, 0);
        c31 = vfmaq_lane_f16(c31, v_b1, v_a3, 0);

        // kk=1
        v_b0 = vld1q_f16(B + 1 * ldb + 0 * 8);
        v_b1 = vld1q_f16(B + 1 * ldb + 1 * 8);
        c00 = vfmaq_lane_f16(c00, v_b0, v_a0, 1);
        c01 = vfmaq_lane_f16(c01, v_b1, v_a0, 1);
        c10 = vfmaq_lane_f16(c10, v_b0, v_a1, 1);
        c11 = vfmaq_lane_f16(c11, v_b1, v_a1, 1);
        c20 = vfmaq_lane_f16(c20, v_b0, v_a2, 1);
        c21 = vfmaq_lane_f16(c21, v_b1, v_a2, 1);
        c30 = vfmaq_lane_f16(c30, v_b0, v_a3, 1);
        c31 = vfmaq_lane_f16(c31, v_b1, v_a3, 1);

        // kk=2
        v_b0 = vld1q_f16(B + 2 * ldb + 0 * 8);
        v_b1 = vld1q_f16(B + 2 * ldb + 1 * 8);
        c00 = vfmaq_lane_f16(c00, v_b0, v_a0, 2);
        c01 = vfmaq_lane_f16(c01, v_b1, v_a0, 2);
        c10 = vfmaq_lane_f16(c10, v_b0, v_a1, 2);
        c11 = vfmaq_lane_f16(c11, v_b1, v_a1, 2);
        c20 = vfmaq_lane_f16(c20, v_b0, v_a2, 2);
        c21 = vfmaq_lane_f16(c21, v_b1, v_a2, 2);
        c30 = vfmaq_lane_f16(c30, v_b0, v_a3, 2);
        c31 = vfmaq_lane_f16(c31, v_b1, v_a3, 2);

        // kk=3
        v_b0 = vld1q_f16(B + 3 * ldb + 0 * 8);
        v_b1 = vld1q_f16(B + 3 * ldb + 1 * 8);
        c00 = vfmaq_lane_f16(c00, v_b0, v_a0, 3);
        c01 = vfmaq_lane_f16(c01, v_b1, v_a0, 3);
        c10 = vfmaq_lane_f16(c10, v_b0, v_a1, 3);
        c11 = vfmaq_lane_f16(c11, v_b1, v_a1, 3);
        c20 = vfmaq_lane_f16(c20, v_b0, v_a2, 3);
        c21 = vfmaq_lane_f16(c21, v_b1, v_a2, 3);
        c30 = vfmaq_lane_f16(c30, v_b0, v_a3, 3);
        c31 = vfmaq_lane_f16(c31, v_b1, v_a3, 3);

        B += 4 * ldb;
    }
    for (; k < K; ++k) {
        const float16x8_t v_b0 = vld1q_f16(B + 0 * 8);
        const float16x8_t v_b1 = vld1q_f16(B + 1 * 8);

        c00 = vfmaq_n_f16(c00, v_b0, A_ptr0[0]);
        c01 = vfmaq_n_f16(c01, v_b1, A_ptr0[0]);
        A_ptr0 += 1;
        c10 = vfmaq_n_f16(c10, v_b0, A_ptr1[0]);
        c11 = vfmaq_n_f16(c11, v_b1, A_ptr1[0]);
        A_ptr1 += 1;
        c20 = vfmaq_n_f16(c20, v_b0, A_ptr2[0]);
        c21 = vfmaq_n_f16(c21, v_b1, A_ptr2[0]);
        A_ptr2 += 1;
        c30 = vfmaq_n_f16(c30, v_b0, A_ptr3[0]);
        c31 = vfmaq_n_f16(c31, v_b1, A_ptr3[0]);
        A_ptr3 += 1;

        B += ldb;
    }

    const float16x8_t v_min = vdupq_n_f16(static_cast<float16_t>(clamp_min));
    const float16x8_t v_max = vdupq_n_f16(static_cast<float16_t>(clamp_max));

    c00 = vaddq_f16(c00, vld1q_f16(C + 0 * ldc + 0 * 8));
    c01 = vaddq_f16(c01, vld1q_f16(C + 0 * ldc + 1 * 8));
    vst1q_f16(C + 0 * ldc + 0 * 8, vminq_f16(vmaxq_f16(c00, v_min), v_max));
    vst1q_f16(C + 0 * ldc + 1 * 8, vminq_f16(vmaxq_f16(c01, v_min), v_max));

    c10 = vaddq_f16(c10, vld1q_f16(C + 1 * ldc + 0 * 8));
    c11 = vaddq_f16(c11, vld1q_f16(C + 1 * ldc + 1 * 8));
    vst1q_f16(C + 1 * ldc + 0 * 8, vminq_f16(vmaxq_f16(c10, v_min), v_max));
    vst1q_f16(C + 1 * ldc + 1 * 8, vminq_f16(vmaxq_f16(c11, v_min), v_max));

    c20 = vaddq_f16(c20, vld1q_f16(C + 2 * ldc + 0 * 8));
    c21 = vaddq_f16(c21, vld1q_f16(C + 2 * ldc + 1 * 8));
    vst1q_f16(C + 2 * ldc + 0 * 8, vminq_f16(vmaxq_f16(c20, v_min), v_max));
    vst1q_f16(C + 2 * ldc + 1 * 8, vminq_f16(vmaxq_f16(c21, v_min), v_max));

    c30 = vaddq_f16(c30, vld1q_f16(C + 3 * ldc + 0 * 8));
    c31 = vaddq_f16(c31, vld1q_f16(C + 3 * ldc + 1 * 8));
    vst1q_f16(C + 3 * ldc + 0 * 8, vminq_f16(vmaxq_f16(c30, v_min), v_max));
    vst1q_f16(C + 3 * ldc + 1 * 8, vminq_f16(vmaxq_f16(c31, v_min), v_max));
}

// =========================================================================
//  mr=8  kernels
// =========================================================================

inline void mma_direct_8x1_f16(
    float16_t* NNOPS_RESTRICT C, int ldc,
    const float16_t* NNOPS_RESTRICT A, int lda,
    const float16_t* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const float16_t* NNOPS_RESTRICT A_ptr0 = A + 0 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr1 = A + 1 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr2 = A + 2 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr3 = A + 3 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr4 = A + 4 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr5 = A + 5 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr6 = A + 6 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr7 = A + 7 * lda;

    float16_t c0 = 0.0f16, c1 = 0.0f16, c2 = 0.0f16, c3 = 0.0f16;
    float16_t c4 = 0.0f16, c5 = 0.0f16, c6 = 0.0f16, c7 = 0.0f16;

    for (int k = 0; k < K; ++k) {
        const float16_t b0 = B[0];
        c0 += *A_ptr0++ * b0;
        c1 += *A_ptr1++ * b0;
        c2 += *A_ptr2++ * b0;
        c3 += *A_ptr3++ * b0;
        c4 += *A_ptr4++ * b0;
        c5 += *A_ptr5++ * b0;
        c6 += *A_ptr6++ * b0;
        c7 += *A_ptr7++ * b0;

        B += ldb;
    }

    auto write = [&](float16_t* dst, float16_t acc) {
        float16_t v = *dst + acc;
        *dst = std::min(std::max(v, static_cast<float16_t>(clamp_min)), static_cast<float16_t>(clamp_max));
    };
    write(C + 0 * ldc, c0);
    write(C + 1 * ldc, c1);
    write(C + 2 * ldc, c2);
    write(C + 3 * ldc, c3);
    write(C + 4 * ldc, c4);
    write(C + 5 * ldc, c5);
    write(C + 6 * ldc, c6);
    write(C + 7 * ldc, c7);
}

inline void mma_direct_8x8_f16(
    float16_t* NNOPS_RESTRICT C, int ldc,
    const float16_t* NNOPS_RESTRICT A, int lda,
    const float16_t* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const float16_t* NNOPS_RESTRICT A_ptr0 = A + 0 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr1 = A + 1 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr2 = A + 2 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr3 = A + 3 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr4 = A + 4 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr5 = A + 5 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr6 = A + 6 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr7 = A + 7 * lda;

    float16x8_t v_c0 = vdupq_n_f16(0.0f16);
    float16x8_t v_c1 = vdupq_n_f16(0.0f16);
    float16x8_t v_c2 = vdupq_n_f16(0.0f16);
    float16x8_t v_c3 = vdupq_n_f16(0.0f16);
    float16x8_t v_c4 = vdupq_n_f16(0.0f16);
    float16x8_t v_c5 = vdupq_n_f16(0.0f16);
    float16x8_t v_c6 = vdupq_n_f16(0.0f16);
    float16x8_t v_c7 = vdupq_n_f16(0.0f16);

    int k = 0;
    for (; k < K - 3; k += 4) {
        const float16x4_t v_a0 = vld1_f16(A_ptr0); A_ptr0 += 4;
        const float16x4_t v_a1 = vld1_f16(A_ptr1); A_ptr1 += 4;
        const float16x4_t v_a2 = vld1_f16(A_ptr2); A_ptr2 += 4;
        const float16x4_t v_a3 = vld1_f16(A_ptr3); A_ptr3 += 4;
        const float16x4_t v_a4 = vld1_f16(A_ptr4); A_ptr4 += 4;
        const float16x4_t v_a5 = vld1_f16(A_ptr5); A_ptr5 += 4;
        const float16x4_t v_a6 = vld1_f16(A_ptr6); A_ptr6 += 4;
        const float16x4_t v_a7 = vld1_f16(A_ptr7); A_ptr7 += 4;

        // kk=0
        float16x8_t v_b0 = vld1q_f16(B + 0 * ldb);
        v_c0 = vfmaq_lane_f16(v_c0, v_b0, v_a0, 0);
        v_c1 = vfmaq_lane_f16(v_c1, v_b0, v_a1, 0);
        v_c2 = vfmaq_lane_f16(v_c2, v_b0, v_a2, 0);
        v_c3 = vfmaq_lane_f16(v_c3, v_b0, v_a3, 0);
        v_c4 = vfmaq_lane_f16(v_c4, v_b0, v_a4, 0);
        v_c5 = vfmaq_lane_f16(v_c5, v_b0, v_a5, 0);
        v_c6 = vfmaq_lane_f16(v_c6, v_b0, v_a6, 0);
        v_c7 = vfmaq_lane_f16(v_c7, v_b0, v_a7, 0);

        // kk=1
        v_b0 = vld1q_f16(B + 1 * ldb);
        v_c0 = vfmaq_lane_f16(v_c0, v_b0, v_a0, 1);
        v_c1 = vfmaq_lane_f16(v_c1, v_b0, v_a1, 1);
        v_c2 = vfmaq_lane_f16(v_c2, v_b0, v_a2, 1);
        v_c3 = vfmaq_lane_f16(v_c3, v_b0, v_a3, 1);
        v_c4 = vfmaq_lane_f16(v_c4, v_b0, v_a4, 1);
        v_c5 = vfmaq_lane_f16(v_c5, v_b0, v_a5, 1);
        v_c6 = vfmaq_lane_f16(v_c6, v_b0, v_a6, 1);
        v_c7 = vfmaq_lane_f16(v_c7, v_b0, v_a7, 1);

        // kk=2
        v_b0 = vld1q_f16(B + 2 * ldb);
        v_c0 = vfmaq_lane_f16(v_c0, v_b0, v_a0, 2);
        v_c1 = vfmaq_lane_f16(v_c1, v_b0, v_a1, 2);
        v_c2 = vfmaq_lane_f16(v_c2, v_b0, v_a2, 2);
        v_c3 = vfmaq_lane_f16(v_c3, v_b0, v_a3, 2);
        v_c4 = vfmaq_lane_f16(v_c4, v_b0, v_a4, 2);
        v_c5 = vfmaq_lane_f16(v_c5, v_b0, v_a5, 2);
        v_c6 = vfmaq_lane_f16(v_c6, v_b0, v_a6, 2);
        v_c7 = vfmaq_lane_f16(v_c7, v_b0, v_a7, 2);

        // kk=3
        v_b0 = vld1q_f16(B + 3 * ldb);
        v_c0 = vfmaq_lane_f16(v_c0, v_b0, v_a0, 3);
        v_c1 = vfmaq_lane_f16(v_c1, v_b0, v_a1, 3);
        v_c2 = vfmaq_lane_f16(v_c2, v_b0, v_a2, 3);
        v_c3 = vfmaq_lane_f16(v_c3, v_b0, v_a3, 3);
        v_c4 = vfmaq_lane_f16(v_c4, v_b0, v_a4, 3);
        v_c5 = vfmaq_lane_f16(v_c5, v_b0, v_a5, 3);
        v_c6 = vfmaq_lane_f16(v_c6, v_b0, v_a6, 3);
        v_c7 = vfmaq_lane_f16(v_c7, v_b0, v_a7, 3);

        B += 4 * ldb;
    }
    for (; k < K; ++k) {
        const float16x8_t v_b0 = vld1q_f16(B); B += ldb;

        v_c0 = vfmaq_n_f16(v_c0, v_b0, *A_ptr0++);
        v_c1 = vfmaq_n_f16(v_c1, v_b0, *A_ptr1++);
        v_c2 = vfmaq_n_f16(v_c2, v_b0, *A_ptr2++);
        v_c3 = vfmaq_n_f16(v_c3, v_b0, *A_ptr3++);
        v_c4 = vfmaq_n_f16(v_c4, v_b0, *A_ptr4++);
        v_c5 = vfmaq_n_f16(v_c5, v_b0, *A_ptr5++);
        v_c6 = vfmaq_n_f16(v_c6, v_b0, *A_ptr6++);
        v_c7 = vfmaq_n_f16(v_c7, v_b0, *A_ptr7++);
    }

    const float16x8_t v_min = vdupq_n_f16(static_cast<float16_t>(clamp_min));
    const float16x8_t v_max = vdupq_n_f16(static_cast<float16_t>(clamp_max));

    v_c0 = vaddq_f16(v_c0, vld1q_f16(C + 0 * ldc));
    v_c1 = vaddq_f16(v_c1, vld1q_f16(C + 1 * ldc));
    v_c2 = vaddq_f16(v_c2, vld1q_f16(C + 2 * ldc));
    v_c3 = vaddq_f16(v_c3, vld1q_f16(C + 3 * ldc));
    v_c4 = vaddq_f16(v_c4, vld1q_f16(C + 4 * ldc));
    v_c5 = vaddq_f16(v_c5, vld1q_f16(C + 5 * ldc));
    v_c6 = vaddq_f16(v_c6, vld1q_f16(C + 6 * ldc));
    v_c7 = vaddq_f16(v_c7, vld1q_f16(C + 7 * ldc));

    vst1q_f16(C + 0 * ldc, vminq_f16(vmaxq_f16(v_c0, v_min), v_max));
    vst1q_f16(C + 1 * ldc, vminq_f16(vmaxq_f16(v_c1, v_min), v_max));
    vst1q_f16(C + 2 * ldc, vminq_f16(vmaxq_f16(v_c2, v_min), v_max));
    vst1q_f16(C + 3 * ldc, vminq_f16(vmaxq_f16(v_c3, v_min), v_max));
    vst1q_f16(C + 4 * ldc, vminq_f16(vmaxq_f16(v_c4, v_min), v_max));
    vst1q_f16(C + 5 * ldc, vminq_f16(vmaxq_f16(v_c5, v_min), v_max));
    vst1q_f16(C + 6 * ldc, vminq_f16(vmaxq_f16(v_c6, v_min), v_max));
    vst1q_f16(C + 7 * ldc, vminq_f16(vmaxq_f16(v_c7, v_min), v_max));
}

inline void mma_direct_8x16_f16(
    float16_t* NNOPS_RESTRICT C, int ldc,
    const float16_t* NNOPS_RESTRICT A, int lda,
    const float16_t* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const float16_t* NNOPS_RESTRICT A_ptr0 = A + 0 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr1 = A + 1 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr2 = A + 2 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr3 = A + 3 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr4 = A + 4 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr5 = A + 5 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr6 = A + 6 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr7 = A + 7 * lda;

    float16x8_t c00 = vdupq_n_f16(0.0f16), c01 = vdupq_n_f16(0.0f16);
    float16x8_t c10 = vdupq_n_f16(0.0f16), c11 = vdupq_n_f16(0.0f16);
    float16x8_t c20 = vdupq_n_f16(0.0f16), c21 = vdupq_n_f16(0.0f16);
    float16x8_t c30 = vdupq_n_f16(0.0f16), c31 = vdupq_n_f16(0.0f16);
    float16x8_t c40 = vdupq_n_f16(0.0f16), c41 = vdupq_n_f16(0.0f16);
    float16x8_t c50 = vdupq_n_f16(0.0f16), c51 = vdupq_n_f16(0.0f16);
    float16x8_t c60 = vdupq_n_f16(0.0f16), c61 = vdupq_n_f16(0.0f16);
    float16x8_t c70 = vdupq_n_f16(0.0f16), c71 = vdupq_n_f16(0.0f16);

    int k = 0;
    for (; k < K - 3; k += 4) {
        const float16x4_t v_a0 = vld1_f16(A_ptr0); A_ptr0 += 4;
        const float16x4_t v_a1 = vld1_f16(A_ptr1); A_ptr1 += 4;
        const float16x4_t v_a2 = vld1_f16(A_ptr2); A_ptr2 += 4;
        const float16x4_t v_a3 = vld1_f16(A_ptr3); A_ptr3 += 4;
        const float16x4_t v_a4 = vld1_f16(A_ptr4); A_ptr4 += 4;
        const float16x4_t v_a5 = vld1_f16(A_ptr5); A_ptr5 += 4;
        const float16x4_t v_a6 = vld1_f16(A_ptr6); A_ptr6 += 4;
        const float16x4_t v_a7 = vld1_f16(A_ptr7); A_ptr7 += 4;

        // kk=0
        float16x8_t v_b0 = vld1q_f16(B + 0 * ldb + 0 * 8);
        float16x8_t v_b1 = vld1q_f16(B + 0 * ldb + 1 * 8);
        c00 = vfmaq_lane_f16(c00, v_b0, v_a0, 0); c01 = vfmaq_lane_f16(c01, v_b1, v_a0, 0);
        c10 = vfmaq_lane_f16(c10, v_b0, v_a1, 0); c11 = vfmaq_lane_f16(c11, v_b1, v_a1, 0);
        c20 = vfmaq_lane_f16(c20, v_b0, v_a2, 0); c21 = vfmaq_lane_f16(c21, v_b1, v_a2, 0);
        c30 = vfmaq_lane_f16(c30, v_b0, v_a3, 0); c31 = vfmaq_lane_f16(c31, v_b1, v_a3, 0);
        c40 = vfmaq_lane_f16(c40, v_b0, v_a4, 0); c41 = vfmaq_lane_f16(c41, v_b1, v_a4, 0);
        c50 = vfmaq_lane_f16(c50, v_b0, v_a5, 0); c51 = vfmaq_lane_f16(c51, v_b1, v_a5, 0);
        c60 = vfmaq_lane_f16(c60, v_b0, v_a6, 0); c61 = vfmaq_lane_f16(c61, v_b1, v_a6, 0);
        c70 = vfmaq_lane_f16(c70, v_b0, v_a7, 0); c71 = vfmaq_lane_f16(c71, v_b1, v_a7, 0);

        // kk=1
        v_b0 = vld1q_f16(B + 1 * ldb + 0 * 8);
        v_b1 = vld1q_f16(B + 1 * ldb + 1 * 8);
        c00 = vfmaq_lane_f16(c00, v_b0, v_a0, 1); c01 = vfmaq_lane_f16(c01, v_b1, v_a0, 1);
        c10 = vfmaq_lane_f16(c10, v_b0, v_a1, 1); c11 = vfmaq_lane_f16(c11, v_b1, v_a1, 1);
        c20 = vfmaq_lane_f16(c20, v_b0, v_a2, 1); c21 = vfmaq_lane_f16(c21, v_b1, v_a2, 1);
        c30 = vfmaq_lane_f16(c30, v_b0, v_a3, 1); c31 = vfmaq_lane_f16(c31, v_b1, v_a3, 1);
        c40 = vfmaq_lane_f16(c40, v_b0, v_a4, 1); c41 = vfmaq_lane_f16(c41, v_b1, v_a4, 1);
        c50 = vfmaq_lane_f16(c50, v_b0, v_a5, 1); c51 = vfmaq_lane_f16(c51, v_b1, v_a5, 1);
        c60 = vfmaq_lane_f16(c60, v_b0, v_a6, 1); c61 = vfmaq_lane_f16(c61, v_b1, v_a6, 1);
        c70 = vfmaq_lane_f16(c70, v_b0, v_a7, 1); c71 = vfmaq_lane_f16(c71, v_b1, v_a7, 1);

        // kk=2
        v_b0 = vld1q_f16(B + 2 * ldb + 0 * 8);
        v_b1 = vld1q_f16(B + 2 * ldb + 1 * 8);
        c00 = vfmaq_lane_f16(c00, v_b0, v_a0, 2); c01 = vfmaq_lane_f16(c01, v_b1, v_a0, 2);
        c10 = vfmaq_lane_f16(c10, v_b0, v_a1, 2); c11 = vfmaq_lane_f16(c11, v_b1, v_a1, 2);
        c20 = vfmaq_lane_f16(c20, v_b0, v_a2, 2); c21 = vfmaq_lane_f16(c21, v_b1, v_a2, 2);
        c30 = vfmaq_lane_f16(c30, v_b0, v_a3, 2); c31 = vfmaq_lane_f16(c31, v_b1, v_a3, 2);
        c40 = vfmaq_lane_f16(c40, v_b0, v_a4, 2); c41 = vfmaq_lane_f16(c41, v_b1, v_a4, 2);
        c50 = vfmaq_lane_f16(c50, v_b0, v_a5, 2); c51 = vfmaq_lane_f16(c51, v_b1, v_a5, 2);
        c60 = vfmaq_lane_f16(c60, v_b0, v_a6, 2); c61 = vfmaq_lane_f16(c61, v_b1, v_a6, 2);
        c70 = vfmaq_lane_f16(c70, v_b0, v_a7, 2); c71 = vfmaq_lane_f16(c71, v_b1, v_a7, 2);

        // kk=3
        v_b0 = vld1q_f16(B + 3 * ldb + 0 * 8);
        v_b1 = vld1q_f16(B + 3 * ldb + 1 * 8);
        c00 = vfmaq_lane_f16(c00, v_b0, v_a0, 3); c01 = vfmaq_lane_f16(c01, v_b1, v_a0, 3);
        c10 = vfmaq_lane_f16(c10, v_b0, v_a1, 3); c11 = vfmaq_lane_f16(c11, v_b1, v_a1, 3);
        c20 = vfmaq_lane_f16(c20, v_b0, v_a2, 3); c21 = vfmaq_lane_f16(c21, v_b1, v_a2, 3);
        c30 = vfmaq_lane_f16(c30, v_b0, v_a3, 3); c31 = vfmaq_lane_f16(c31, v_b1, v_a3, 3);
        c40 = vfmaq_lane_f16(c40, v_b0, v_a4, 3); c41 = vfmaq_lane_f16(c41, v_b1, v_a4, 3);
        c50 = vfmaq_lane_f16(c50, v_b0, v_a5, 3); c51 = vfmaq_lane_f16(c51, v_b1, v_a5, 3);
        c60 = vfmaq_lane_f16(c60, v_b0, v_a6, 3); c61 = vfmaq_lane_f16(c61, v_b1, v_a6, 3);
        c70 = vfmaq_lane_f16(c70, v_b0, v_a7, 3); c71 = vfmaq_lane_f16(c71, v_b1, v_a7, 3);

        B += 4 * ldb;
    }
    for (; k < K; ++k) {
        const float16x8_t v_b0 = vld1q_f16(B + 0 * 8);
        const float16x8_t v_b1 = vld1q_f16(B + 1 * 8);

        c00 = vfmaq_n_f16(c00, v_b0, A_ptr0[0]); c01 = vfmaq_n_f16(c01, v_b1, A_ptr0[0]);
        A_ptr0 += 1;
        c10 = vfmaq_n_f16(c10, v_b0, A_ptr1[0]); c11 = vfmaq_n_f16(c11, v_b1, A_ptr1[0]);
        A_ptr1 += 1;
        c20 = vfmaq_n_f16(c20, v_b0, A_ptr2[0]); c21 = vfmaq_n_f16(c21, v_b1, A_ptr2[0]);
        A_ptr2 += 1;
        c30 = vfmaq_n_f16(c30, v_b0, A_ptr3[0]); c31 = vfmaq_n_f16(c31, v_b1, A_ptr3[0]);
        A_ptr3 += 1;
        c40 = vfmaq_n_f16(c40, v_b0, A_ptr4[0]); c41 = vfmaq_n_f16(c41, v_b1, A_ptr4[0]);
        A_ptr4 += 1;
        c50 = vfmaq_n_f16(c50, v_b0, A_ptr5[0]); c51 = vfmaq_n_f16(c51, v_b1, A_ptr5[0]);
        A_ptr5 += 1;
        c60 = vfmaq_n_f16(c60, v_b0, A_ptr6[0]); c61 = vfmaq_n_f16(c61, v_b1, A_ptr6[0]);
        A_ptr6 += 1;
        c70 = vfmaq_n_f16(c70, v_b0, A_ptr7[0]); c71 = vfmaq_n_f16(c71, v_b1, A_ptr7[0]);
        A_ptr7 += 1;

        B += ldb;
    }

    const float16x8_t v_min = vdupq_n_f16(static_cast<float16_t>(clamp_min));
    const float16x8_t v_max = vdupq_n_f16(static_cast<float16_t>(clamp_max));

    c00 = vaddq_f16(c00, vld1q_f16(C + 0 * ldc + 0 * 8));
    c01 = vaddq_f16(c01, vld1q_f16(C + 0 * ldc + 1 * 8));
    vst1q_f16(C + 0 * ldc + 0 * 8, vminq_f16(vmaxq_f16(c00, v_min), v_max));
    vst1q_f16(C + 0 * ldc + 1 * 8, vminq_f16(vmaxq_f16(c01, v_min), v_max));

    c10 = vaddq_f16(c10, vld1q_f16(C + 1 * ldc + 0 * 8));
    c11 = vaddq_f16(c11, vld1q_f16(C + 1 * ldc + 1 * 8));
    vst1q_f16(C + 1 * ldc + 0 * 8, vminq_f16(vmaxq_f16(c10, v_min), v_max));
    vst1q_f16(C + 1 * ldc + 1 * 8, vminq_f16(vmaxq_f16(c11, v_min), v_max));

    c20 = vaddq_f16(c20, vld1q_f16(C + 2 * ldc + 0 * 8));
    c21 = vaddq_f16(c21, vld1q_f16(C + 2 * ldc + 1 * 8));
    vst1q_f16(C + 2 * ldc + 0 * 8, vminq_f16(vmaxq_f16(c20, v_min), v_max));
    vst1q_f16(C + 2 * ldc + 1 * 8, vminq_f16(vmaxq_f16(c21, v_min), v_max));

    c30 = vaddq_f16(c30, vld1q_f16(C + 3 * ldc + 0 * 8));
    c31 = vaddq_f16(c31, vld1q_f16(C + 3 * ldc + 1 * 8));
    vst1q_f16(C + 3 * ldc + 0 * 8, vminq_f16(vmaxq_f16(c30, v_min), v_max));
    vst1q_f16(C + 3 * ldc + 1 * 8, vminq_f16(vmaxq_f16(c31, v_min), v_max));

    c40 = vaddq_f16(c40, vld1q_f16(C + 4 * ldc + 0 * 8));
    c41 = vaddq_f16(c41, vld1q_f16(C + 4 * ldc + 1 * 8));
    vst1q_f16(C + 4 * ldc + 0 * 8, vminq_f16(vmaxq_f16(c40, v_min), v_max));
    vst1q_f16(C + 4 * ldc + 1 * 8, vminq_f16(vmaxq_f16(c41, v_min), v_max));

    c50 = vaddq_f16(c50, vld1q_f16(C + 5 * ldc + 0 * 8));
    c51 = vaddq_f16(c51, vld1q_f16(C + 5 * ldc + 1 * 8));
    vst1q_f16(C + 5 * ldc + 0 * 8, vminq_f16(vmaxq_f16(c50, v_min), v_max));
    vst1q_f16(C + 5 * ldc + 1 * 8, vminq_f16(vmaxq_f16(c51, v_min), v_max));

    c60 = vaddq_f16(c60, vld1q_f16(C + 6 * ldc + 0 * 8));
    c61 = vaddq_f16(c61, vld1q_f16(C + 6 * ldc + 1 * 8));
    vst1q_f16(C + 6 * ldc + 0 * 8, vminq_f16(vmaxq_f16(c60, v_min), v_max));
    vst1q_f16(C + 6 * ldc + 1 * 8, vminq_f16(vmaxq_f16(c61, v_min), v_max));

    c70 = vaddq_f16(c70, vld1q_f16(C + 7 * ldc + 0 * 8));
    c71 = vaddq_f16(c71, vld1q_f16(C + 7 * ldc + 1 * 8));
    vst1q_f16(C + 7 * ldc + 0 * 8, vminq_f16(vmaxq_f16(c70, v_min), v_max));
    vst1q_f16(C + 7 * ldc + 1 * 8, vminq_f16(vmaxq_f16(c71, v_min), v_max));
}

}  // namespace nnops::backend::cpu::aarch64
