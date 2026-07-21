#pragma once
/// @file mma_direct_f32.hpp
/// @brief AArch64 NEON float32 MMA direct (unpacked) micro-kernels.
///
/// Unlike mma_pack, these kernels read A and B directly from their
/// original row-major layouts (A with stride lda, B with stride ldb).
/// No pre-packing step is needed — the inner loop loads strided B rows
/// and broadcasts A elements via lane-indexed FMA.
///
/// Each kernel computes  C[mr][nr] += A[mr][K] × B[K][nr]
/// then clamps the result to [clamp_min, clamp_max].
///
/// Tile sizes (AArch64-optimised):
///   M ∈ {8, 4, 1}    N ∈ {12, 4, 1}
///
/// K is unrolled by 4: four A elements are loaded as a float32x4_t
/// and broadcast lane-by-lane with vfmaq_laneq_f32 against four
/// successive B rows. A scalar tail handles any remaining K.
///
/// Reference: nn_compute/src/cpu/kernel/mma/aarch64/mma_direct_f32.hpp

#include <arm_neon.h>
#include <algorithm>
#include "backend/cpu/common/restrict.hpp"

namespace nnops::backend::cpu::aarch64 {

// =========================================================================
//  mr=1  kernels
// =========================================================================

inline void mma_direct_1x1_f32(
    float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A, int lda,
    const float* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const float* NNOPS_RESTRICT A_ptr0 = A;

    float c0 = 0.0f;
    for (int k = 0; k < K; ++k) {
        c0 += A_ptr0[0] * B[0];
        A_ptr0 += 1;
        B += ldb;
    }
    c0 += C[0];
    C[0] = std::min(std::max(c0, clamp_min), clamp_max);
}

inline void mma_direct_1x4_f32(
    float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A, int lda,
    const float* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const float* NNOPS_RESTRICT A_ptr0 = A;

    float32x4_t v_c0 = vdupq_n_f32(0.0f);

    int k = 0;
    for (; k < K - 3; k += 4) {
        const float32x4_t v_a0 = vld1q_f32(A_ptr0);

        float32x4_t v_b0 = vld1q_f32(B + 0 * ldb);
        v_c0 = vfmaq_laneq_f32(v_c0, v_b0, v_a0, 0);
        v_b0 = vld1q_f32(B + 1 * ldb);
        v_c0 = vfmaq_laneq_f32(v_c0, v_b0, v_a0, 1);
        v_b0 = vld1q_f32(B + 2 * ldb);
        v_c0 = vfmaq_laneq_f32(v_c0, v_b0, v_a0, 2);
        v_b0 = vld1q_f32(B + 3 * ldb);
        v_c0 = vfmaq_laneq_f32(v_c0, v_b0, v_a0, 3);

        A_ptr0 += 4;
        B += 4 * ldb;
    }
    for (; k < K; ++k) {
        const float32x4_t v_b0 = vld1q_f32(B);
        v_c0 = vfmaq_n_f32(v_c0, v_b0, A_ptr0[0]);
        A_ptr0 += 1;
        B += ldb;
    }

    v_c0 = vaddq_f32(v_c0, vld1q_f32(C));
    const float32x4_t v_min = vdupq_n_f32(clamp_min);
    const float32x4_t v_max = vdupq_n_f32(clamp_max);
    vst1q_f32(C, vminq_f32(vmaxq_f32(v_c0, v_min), v_max));
}

inline void mma_direct_1x12_f32(
    float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A, int lda,
    const float* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const float* NNOPS_RESTRICT A_ptr0 = A;

    float32x4_t v_c00 = vdupq_n_f32(0.0f);
    float32x4_t v_c01 = v_c00;
    float32x4_t v_c02 = v_c00;

    int k = 0;
    for (; k < K - 3; k += 4) {
        const float32x4_t v_a0 = vld1q_f32(A_ptr0);

        float32x4_t v_b0 = vld1q_f32(B + 0 * ldb + 0 * 4);
        float32x4_t v_b1 = vld1q_f32(B + 0 * ldb + 1 * 4);
        float32x4_t v_b2 = vld1q_f32(B + 0 * ldb + 2 * 4);

        v_c00 = vfmaq_laneq_f32(v_c00, v_b0, v_a0, 0);
        v_c01 = vfmaq_laneq_f32(v_c01, v_b1, v_a0, 0);
        v_c02 = vfmaq_laneq_f32(v_c02, v_b2, v_a0, 0);

        v_b0 = vld1q_f32(B + 1 * ldb + 0 * 4);
        v_b1 = vld1q_f32(B + 1 * ldb + 1 * 4);
        v_b2 = vld1q_f32(B + 1 * ldb + 2 * 4);

        v_c00 = vfmaq_laneq_f32(v_c00, v_b0, v_a0, 1);
        v_c01 = vfmaq_laneq_f32(v_c01, v_b1, v_a0, 1);
        v_c02 = vfmaq_laneq_f32(v_c02, v_b2, v_a0, 1);

        v_b0 = vld1q_f32(B + 2 * ldb + 0 * 4);
        v_b1 = vld1q_f32(B + 2 * ldb + 1 * 4);
        v_b2 = vld1q_f32(B + 2 * ldb + 2 * 4);

        v_c00 = vfmaq_laneq_f32(v_c00, v_b0, v_a0, 2);
        v_c01 = vfmaq_laneq_f32(v_c01, v_b1, v_a0, 2);
        v_c02 = vfmaq_laneq_f32(v_c02, v_b2, v_a0, 2);

        v_b0 = vld1q_f32(B + 3 * ldb + 0 * 4);
        v_b1 = vld1q_f32(B + 3 * ldb + 1 * 4);
        v_b2 = vld1q_f32(B + 3 * ldb + 2 * 4);

        v_c00 = vfmaq_laneq_f32(v_c00, v_b0, v_a0, 3);
        v_c01 = vfmaq_laneq_f32(v_c01, v_b1, v_a0, 3);
        v_c02 = vfmaq_laneq_f32(v_c02, v_b2, v_a0, 3);

        A_ptr0 += 4;
        B += 4 * ldb;
    }
    for (; k < K; ++k) {
        const float32x4_t v_b0 = vld1q_f32(B + 0 * 4);
        const float32x4_t v_b1 = vld1q_f32(B + 1 * 4);
        const float32x4_t v_b2 = vld1q_f32(B + 2 * 4);

        v_c00 = vfmaq_n_f32(v_c00, v_b0, A_ptr0[0]);
        v_c01 = vfmaq_n_f32(v_c01, v_b1, A_ptr0[0]);
        v_c02 = vfmaq_n_f32(v_c02, v_b2, A_ptr0[0]);

        A_ptr0 += 1;
        B += ldb;
    }

    const float32x4_t v_min = vdupq_n_f32(clamp_min);
    const float32x4_t v_max = vdupq_n_f32(clamp_max);

    v_c00 = vaddq_f32(v_c00, vld1q_f32(C + 0 * 4));
    v_c01 = vaddq_f32(v_c01, vld1q_f32(C + 1 * 4));
    v_c02 = vaddq_f32(v_c02, vld1q_f32(C + 2 * 4));

    vst1q_f32(C + 0 * 4, vminq_f32(vmaxq_f32(v_c00, v_min), v_max));
    vst1q_f32(C + 1 * 4, vminq_f32(vmaxq_f32(v_c01, v_min), v_max));
    vst1q_f32(C + 2 * 4, vminq_f32(vmaxq_f32(v_c02, v_min), v_max));
}

// =========================================================================
//  mr=4  kernels
// =========================================================================

inline void mma_direct_4x1_f32(
    float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A, int lda,
    const float* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const float* NNOPS_RESTRICT A_ptr0 = A + 0 * lda;
    const float* NNOPS_RESTRICT A_ptr1 = A + 1 * lda;
    const float* NNOPS_RESTRICT A_ptr2 = A + 2 * lda;
    const float* NNOPS_RESTRICT A_ptr3 = A + 3 * lda;

    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    for (int k = 0; k < K; ++k) {
        const float b0 = B[0];
        c0 += A_ptr0[0] * b0;
        c1 += A_ptr1[0] * b0;
        c2 += A_ptr2[0] * b0;
        c3 += A_ptr3[0] * b0;

        A_ptr0 += 1; A_ptr1 += 1; A_ptr2 += 1; A_ptr3 += 1;
        B += ldb;
    }

    auto write = [&](float* dst, float acc) {
        float v = *dst + acc;
        *dst = std::min(std::max(v, clamp_min), clamp_max);
    };
    write(C + 0 * ldc, c0);
    write(C + 1 * ldc, c1);
    write(C + 2 * ldc, c2);
    write(C + 3 * ldc, c3);
}

inline void mma_direct_4x4_f32(
    float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A, int lda,
    const float* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const float* NNOPS_RESTRICT A_ptr0 = A + 0 * lda;
    const float* NNOPS_RESTRICT A_ptr1 = A + 1 * lda;
    const float* NNOPS_RESTRICT A_ptr2 = A + 2 * lda;
    const float* NNOPS_RESTRICT A_ptr3 = A + 3 * lda;

    float32x4_t v_c0 = vdupq_n_f32(0.0f);
    float32x4_t v_c1 = v_c0;
    float32x4_t v_c2 = v_c0;
    float32x4_t v_c3 = v_c0;

    int k = 0;
    for (; k < K - 3; k += 4) {
        const float32x4_t v_a0 = vld1q_f32(A_ptr0);
        const float32x4_t v_a1 = vld1q_f32(A_ptr1);
        const float32x4_t v_a2 = vld1q_f32(A_ptr2);
        const float32x4_t v_a3 = vld1q_f32(A_ptr3);

        float32x4_t v_b0 = vld1q_f32(B + 0 * ldb);
        v_c0 = vfmaq_laneq_f32(v_c0, v_b0, v_a0, 0);
        v_c1 = vfmaq_laneq_f32(v_c1, v_b0, v_a1, 0);
        v_c2 = vfmaq_laneq_f32(v_c2, v_b0, v_a2, 0);
        v_c3 = vfmaq_laneq_f32(v_c3, v_b0, v_a3, 0);

        v_b0 = vld1q_f32(B + 1 * ldb);
        v_c0 = vfmaq_laneq_f32(v_c0, v_b0, v_a0, 1);
        v_c1 = vfmaq_laneq_f32(v_c1, v_b0, v_a1, 1);
        v_c2 = vfmaq_laneq_f32(v_c2, v_b0, v_a2, 1);
        v_c3 = vfmaq_laneq_f32(v_c3, v_b0, v_a3, 1);

        v_b0 = vld1q_f32(B + 2 * ldb);
        v_c0 = vfmaq_laneq_f32(v_c0, v_b0, v_a0, 2);
        v_c1 = vfmaq_laneq_f32(v_c1, v_b0, v_a1, 2);
        v_c2 = vfmaq_laneq_f32(v_c2, v_b0, v_a2, 2);
        v_c3 = vfmaq_laneq_f32(v_c3, v_b0, v_a3, 2);

        v_b0 = vld1q_f32(B + 3 * ldb);
        v_c0 = vfmaq_laneq_f32(v_c0, v_b0, v_a0, 3);
        v_c1 = vfmaq_laneq_f32(v_c1, v_b0, v_a1, 3);
        v_c2 = vfmaq_laneq_f32(v_c2, v_b0, v_a2, 3);
        v_c3 = vfmaq_laneq_f32(v_c3, v_b0, v_a3, 3);

        A_ptr0 += 4; A_ptr1 += 4; A_ptr2 += 4; A_ptr3 += 4;
        B += 4 * ldb;
    }
    for (; k < K; ++k) {
        const float32x4_t v_b0 = vld1q_f32(B);
        v_c0 = vfmaq_n_f32(v_c0, v_b0, A_ptr0[0]);
        v_c1 = vfmaq_n_f32(v_c1, v_b0, A_ptr1[0]);
        v_c2 = vfmaq_n_f32(v_c2, v_b0, A_ptr2[0]);
        v_c3 = vfmaq_n_f32(v_c3, v_b0, A_ptr3[0]);

        A_ptr0 += 1; A_ptr1 += 1; A_ptr2 += 1; A_ptr3 += 1;
        B += ldb;
    }

    const float32x4_t v_min = vdupq_n_f32(clamp_min);
    const float32x4_t v_max = vdupq_n_f32(clamp_max);

    v_c0 = vaddq_f32(v_c0, vld1q_f32(C + 0 * ldc));
    v_c1 = vaddq_f32(v_c1, vld1q_f32(C + 1 * ldc));
    v_c2 = vaddq_f32(v_c2, vld1q_f32(C + 2 * ldc));
    v_c3 = vaddq_f32(v_c3, vld1q_f32(C + 3 * ldc));

    vst1q_f32(C + 0 * ldc, vminq_f32(vmaxq_f32(v_c0, v_min), v_max));
    vst1q_f32(C + 1 * ldc, vminq_f32(vmaxq_f32(v_c1, v_min), v_max));
    vst1q_f32(C + 2 * ldc, vminq_f32(vmaxq_f32(v_c2, v_min), v_max));
    vst1q_f32(C + 3 * ldc, vminq_f32(vmaxq_f32(v_c3, v_min), v_max));
}

inline void mma_direct_4x12_f32(
    float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A, int lda,
    const float* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const float* NNOPS_RESTRICT A_ptr0 = A + 0 * lda;
    const float* NNOPS_RESTRICT A_ptr1 = A + 1 * lda;
    const float* NNOPS_RESTRICT A_ptr2 = A + 2 * lda;
    const float* NNOPS_RESTRICT A_ptr3 = A + 3 * lda;

    float32x4_t v_c00 = vdupq_n_f32(0.0f);
    float32x4_t v_c01 = v_c00;
    float32x4_t v_c02 = v_c00;

    float32x4_t v_c10 = v_c00;
    float32x4_t v_c11 = v_c00;
    float32x4_t v_c12 = v_c00;

    float32x4_t v_c20 = v_c00;
    float32x4_t v_c21 = v_c00;
    float32x4_t v_c22 = v_c00;

    float32x4_t v_c30 = v_c00;
    float32x4_t v_c31 = v_c00;
    float32x4_t v_c32 = v_c00;

    int k = 0;
    for (; k < K - 3; k += 4) {
        const float32x4_t v_a0 = vld1q_f32(A_ptr0);
        const float32x4_t v_a1 = vld1q_f32(A_ptr1);
        const float32x4_t v_a2 = vld1q_f32(A_ptr2);
        const float32x4_t v_a3 = vld1q_f32(A_ptr3);

        float32x4_t v_b0 = vld1q_f32(B + 0 * ldb + 0 * 4);
        float32x4_t v_b1 = vld1q_f32(B + 0 * ldb + 1 * 4);
        float32x4_t v_b2 = vld1q_f32(B + 0 * ldb + 2 * 4);

        v_c00 = vfmaq_laneq_f32(v_c00, v_b0, v_a0, 0);
        v_c01 = vfmaq_laneq_f32(v_c01, v_b1, v_a0, 0);
        v_c02 = vfmaq_laneq_f32(v_c02, v_b2, v_a0, 0);
        v_c10 = vfmaq_laneq_f32(v_c10, v_b0, v_a1, 0);
        v_c11 = vfmaq_laneq_f32(v_c11, v_b1, v_a1, 0);
        v_c12 = vfmaq_laneq_f32(v_c12, v_b2, v_a1, 0);
        v_c20 = vfmaq_laneq_f32(v_c20, v_b0, v_a2, 0);
        v_c21 = vfmaq_laneq_f32(v_c21, v_b1, v_a2, 0);
        v_c22 = vfmaq_laneq_f32(v_c22, v_b2, v_a2, 0);
        v_c30 = vfmaq_laneq_f32(v_c30, v_b0, v_a3, 0);
        v_c31 = vfmaq_laneq_f32(v_c31, v_b1, v_a3, 0);
        v_c32 = vfmaq_laneq_f32(v_c32, v_b2, v_a3, 0);

        v_b0 = vld1q_f32(B + 1 * ldb + 0 * 4);
        v_b1 = vld1q_f32(B + 1 * ldb + 1 * 4);
        v_b2 = vld1q_f32(B + 1 * ldb + 2 * 4);

        v_c00 = vfmaq_laneq_f32(v_c00, v_b0, v_a0, 1);
        v_c01 = vfmaq_laneq_f32(v_c01, v_b1, v_a0, 1);
        v_c02 = vfmaq_laneq_f32(v_c02, v_b2, v_a0, 1);
        v_c10 = vfmaq_laneq_f32(v_c10, v_b0, v_a1, 1);
        v_c11 = vfmaq_laneq_f32(v_c11, v_b1, v_a1, 1);
        v_c12 = vfmaq_laneq_f32(v_c12, v_b2, v_a1, 1);
        v_c20 = vfmaq_laneq_f32(v_c20, v_b0, v_a2, 1);
        v_c21 = vfmaq_laneq_f32(v_c21, v_b1, v_a2, 1);
        v_c22 = vfmaq_laneq_f32(v_c22, v_b2, v_a2, 1);
        v_c30 = vfmaq_laneq_f32(v_c30, v_b0, v_a3, 1);
        v_c31 = vfmaq_laneq_f32(v_c31, v_b1, v_a3, 1);
        v_c32 = vfmaq_laneq_f32(v_c32, v_b2, v_a3, 1);

        v_b0 = vld1q_f32(B + 2 * ldb + 0 * 4);
        v_b1 = vld1q_f32(B + 2 * ldb + 1 * 4);
        v_b2 = vld1q_f32(B + 2 * ldb + 2 * 4);

        v_c00 = vfmaq_laneq_f32(v_c00, v_b0, v_a0, 2);
        v_c01 = vfmaq_laneq_f32(v_c01, v_b1, v_a0, 2);
        v_c02 = vfmaq_laneq_f32(v_c02, v_b2, v_a0, 2);
        v_c10 = vfmaq_laneq_f32(v_c10, v_b0, v_a1, 2);
        v_c11 = vfmaq_laneq_f32(v_c11, v_b1, v_a1, 2);
        v_c12 = vfmaq_laneq_f32(v_c12, v_b2, v_a1, 2);
        v_c20 = vfmaq_laneq_f32(v_c20, v_b0, v_a2, 2);
        v_c21 = vfmaq_laneq_f32(v_c21, v_b1, v_a2, 2);
        v_c22 = vfmaq_laneq_f32(v_c22, v_b2, v_a2, 2);
        v_c30 = vfmaq_laneq_f32(v_c30, v_b0, v_a3, 2);
        v_c31 = vfmaq_laneq_f32(v_c31, v_b1, v_a3, 2);
        v_c32 = vfmaq_laneq_f32(v_c32, v_b2, v_a3, 2);

        v_b0 = vld1q_f32(B + 3 * ldb + 0 * 4);
        v_b1 = vld1q_f32(B + 3 * ldb + 1 * 4);
        v_b2 = vld1q_f32(B + 3 * ldb + 2 * 4);

        v_c00 = vfmaq_laneq_f32(v_c00, v_b0, v_a0, 3);
        v_c01 = vfmaq_laneq_f32(v_c01, v_b1, v_a0, 3);
        v_c02 = vfmaq_laneq_f32(v_c02, v_b2, v_a0, 3);
        v_c10 = vfmaq_laneq_f32(v_c10, v_b0, v_a1, 3);
        v_c11 = vfmaq_laneq_f32(v_c11, v_b1, v_a1, 3);
        v_c12 = vfmaq_laneq_f32(v_c12, v_b2, v_a1, 3);
        v_c20 = vfmaq_laneq_f32(v_c20, v_b0, v_a2, 3);
        v_c21 = vfmaq_laneq_f32(v_c21, v_b1, v_a2, 3);
        v_c22 = vfmaq_laneq_f32(v_c22, v_b2, v_a2, 3);
        v_c30 = vfmaq_laneq_f32(v_c30, v_b0, v_a3, 3);
        v_c31 = vfmaq_laneq_f32(v_c31, v_b1, v_a3, 3);
        v_c32 = vfmaq_laneq_f32(v_c32, v_b2, v_a3, 3);

        A_ptr0 += 4; A_ptr1 += 4; A_ptr2 += 4; A_ptr3 += 4;
        B += 4 * ldb;
    }
    for (; k < K; ++k) {
        const float32x4_t v_b0 = vld1q_f32(B + 0 * 4);
        const float32x4_t v_b1 = vld1q_f32(B + 1 * 4);
        const float32x4_t v_b2 = vld1q_f32(B + 2 * 4);

        v_c00 = vfmaq_n_f32(v_c00, v_b0, A_ptr0[0]);
        v_c01 = vfmaq_n_f32(v_c01, v_b1, A_ptr0[0]);
        v_c02 = vfmaq_n_f32(v_c02, v_b2, A_ptr0[0]);
        v_c10 = vfmaq_n_f32(v_c10, v_b0, A_ptr1[0]);
        v_c11 = vfmaq_n_f32(v_c11, v_b1, A_ptr1[0]);
        v_c12 = vfmaq_n_f32(v_c12, v_b2, A_ptr1[0]);
        v_c20 = vfmaq_n_f32(v_c20, v_b0, A_ptr2[0]);
        v_c21 = vfmaq_n_f32(v_c21, v_b1, A_ptr2[0]);
        v_c22 = vfmaq_n_f32(v_c22, v_b2, A_ptr2[0]);
        v_c30 = vfmaq_n_f32(v_c30, v_b0, A_ptr3[0]);
        v_c31 = vfmaq_n_f32(v_c31, v_b1, A_ptr3[0]);
        v_c32 = vfmaq_n_f32(v_c32, v_b2, A_ptr3[0]);

        A_ptr0 += 1; A_ptr1 += 1; A_ptr2 += 1; A_ptr3 += 1;
        B += ldb;
    }

    const float32x4_t v_min = vdupq_n_f32(clamp_min);
    const float32x4_t v_max = vdupq_n_f32(clamp_max);

    auto clamp_store = [&](float32x4_t& v, float* dst) {
        v = vaddq_f32(v, vld1q_f32(dst));
        vst1q_f32(dst, vminq_f32(vmaxq_f32(v, v_min), v_max));
    };

    clamp_store(v_c00, C + 0 * ldc + 0 * 4);
    clamp_store(v_c01, C + 0 * ldc + 1 * 4);
    clamp_store(v_c02, C + 0 * ldc + 2 * 4);
    clamp_store(v_c10, C + 1 * ldc + 0 * 4);
    clamp_store(v_c11, C + 1 * ldc + 1 * 4);
    clamp_store(v_c12, C + 1 * ldc + 2 * 4);
    clamp_store(v_c20, C + 2 * ldc + 0 * 4);
    clamp_store(v_c21, C + 2 * ldc + 1 * 4);
    clamp_store(v_c22, C + 2 * ldc + 2 * 4);
    clamp_store(v_c30, C + 3 * ldc + 0 * 4);
    clamp_store(v_c31, C + 3 * ldc + 1 * 4);
    clamp_store(v_c32, C + 3 * ldc + 2 * 4);
}

// =========================================================================
//  mr=8  kernels
// =========================================================================

inline void mma_direct_8x1_f32(
    float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A, int lda,
    const float* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const float* NNOPS_RESTRICT A_ptr0 = A + 0 * lda;
    const float* NNOPS_RESTRICT A_ptr1 = A + 1 * lda;
    const float* NNOPS_RESTRICT A_ptr2 = A + 2 * lda;
    const float* NNOPS_RESTRICT A_ptr3 = A + 3 * lda;
    const float* NNOPS_RESTRICT A_ptr4 = A + 4 * lda;
    const float* NNOPS_RESTRICT A_ptr5 = A + 5 * lda;
    const float* NNOPS_RESTRICT A_ptr6 = A + 6 * lda;
    const float* NNOPS_RESTRICT A_ptr7 = A + 7 * lda;

    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    float c4 = 0.0f, c5 = 0.0f, c6 = 0.0f, c7 = 0.0f;

    for (int k = 0; k < K; ++k) {
        const float b0 = B[0];
        c0 += A_ptr0[0] * b0;
        c1 += A_ptr1[0] * b0;
        c2 += A_ptr2[0] * b0;
        c3 += A_ptr3[0] * b0;
        c4 += A_ptr4[0] * b0;
        c5 += A_ptr5[0] * b0;
        c6 += A_ptr6[0] * b0;
        c7 += A_ptr7[0] * b0;

        A_ptr0 += 1; A_ptr1 += 1; A_ptr2 += 1; A_ptr3 += 1;
        A_ptr4 += 1; A_ptr5 += 1; A_ptr6 += 1; A_ptr7 += 1;
        B += ldb;
    }

    auto write = [&](float* dst, float acc) {
        float v = *dst + acc;
        *dst = std::min(std::max(v, clamp_min), clamp_max);
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

inline void mma_direct_8x4_f32(
    float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A, int lda,
    const float* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const float* NNOPS_RESTRICT A_ptr0 = A + 0 * lda;
    const float* NNOPS_RESTRICT A_ptr1 = A + 1 * lda;
    const float* NNOPS_RESTRICT A_ptr2 = A + 2 * lda;
    const float* NNOPS_RESTRICT A_ptr3 = A + 3 * lda;
    const float* NNOPS_RESTRICT A_ptr4 = A + 4 * lda;
    const float* NNOPS_RESTRICT A_ptr5 = A + 5 * lda;
    const float* NNOPS_RESTRICT A_ptr6 = A + 6 * lda;
    const float* NNOPS_RESTRICT A_ptr7 = A + 7 * lda;

    float32x4_t v_c0 = vdupq_n_f32(0.0f);
    float32x4_t v_c1 = v_c0;
    float32x4_t v_c2 = v_c0;
    float32x4_t v_c3 = v_c0;
    float32x4_t v_c4 = v_c0;
    float32x4_t v_c5 = v_c0;
    float32x4_t v_c6 = v_c0;
    float32x4_t v_c7 = v_c0;

    int k = 0;
    for (; k < K - 3; k += 4) {
        const float32x4_t v_a0 = vld1q_f32(A_ptr0);
        const float32x4_t v_a1 = vld1q_f32(A_ptr1);
        const float32x4_t v_a2 = vld1q_f32(A_ptr2);
        const float32x4_t v_a3 = vld1q_f32(A_ptr3);
        const float32x4_t v_a4 = vld1q_f32(A_ptr4);
        const float32x4_t v_a5 = vld1q_f32(A_ptr5);
        const float32x4_t v_a6 = vld1q_f32(A_ptr6);
        const float32x4_t v_a7 = vld1q_f32(A_ptr7);

        float32x4_t v_b0 = vld1q_f32(B + 0 * ldb);
        v_c0 = vfmaq_laneq_f32(v_c0, v_b0, v_a0, 0);
        v_c1 = vfmaq_laneq_f32(v_c1, v_b0, v_a1, 0);
        v_c2 = vfmaq_laneq_f32(v_c2, v_b0, v_a2, 0);
        v_c3 = vfmaq_laneq_f32(v_c3, v_b0, v_a3, 0);
        v_c4 = vfmaq_laneq_f32(v_c4, v_b0, v_a4, 0);
        v_c5 = vfmaq_laneq_f32(v_c5, v_b0, v_a5, 0);
        v_c6 = vfmaq_laneq_f32(v_c6, v_b0, v_a6, 0);
        v_c7 = vfmaq_laneq_f32(v_c7, v_b0, v_a7, 0);

        v_b0 = vld1q_f32(B + 1 * ldb);
        v_c0 = vfmaq_laneq_f32(v_c0, v_b0, v_a0, 1);
        v_c1 = vfmaq_laneq_f32(v_c1, v_b0, v_a1, 1);
        v_c2 = vfmaq_laneq_f32(v_c2, v_b0, v_a2, 1);
        v_c3 = vfmaq_laneq_f32(v_c3, v_b0, v_a3, 1);
        v_c4 = vfmaq_laneq_f32(v_c4, v_b0, v_a4, 1);
        v_c5 = vfmaq_laneq_f32(v_c5, v_b0, v_a5, 1);
        v_c6 = vfmaq_laneq_f32(v_c6, v_b0, v_a6, 1);
        v_c7 = vfmaq_laneq_f32(v_c7, v_b0, v_a7, 1);

        v_b0 = vld1q_f32(B + 2 * ldb);
        v_c0 = vfmaq_laneq_f32(v_c0, v_b0, v_a0, 2);
        v_c1 = vfmaq_laneq_f32(v_c1, v_b0, v_a1, 2);
        v_c2 = vfmaq_laneq_f32(v_c2, v_b0, v_a2, 2);
        v_c3 = vfmaq_laneq_f32(v_c3, v_b0, v_a3, 2);
        v_c4 = vfmaq_laneq_f32(v_c4, v_b0, v_a4, 2);
        v_c5 = vfmaq_laneq_f32(v_c5, v_b0, v_a5, 2);
        v_c6 = vfmaq_laneq_f32(v_c6, v_b0, v_a6, 2);
        v_c7 = vfmaq_laneq_f32(v_c7, v_b0, v_a7, 2);

        v_b0 = vld1q_f32(B + 3 * ldb);
        v_c0 = vfmaq_laneq_f32(v_c0, v_b0, v_a0, 3);
        v_c1 = vfmaq_laneq_f32(v_c1, v_b0, v_a1, 3);
        v_c2 = vfmaq_laneq_f32(v_c2, v_b0, v_a2, 3);
        v_c3 = vfmaq_laneq_f32(v_c3, v_b0, v_a3, 3);
        v_c4 = vfmaq_laneq_f32(v_c4, v_b0, v_a4, 3);
        v_c5 = vfmaq_laneq_f32(v_c5, v_b0, v_a5, 3);
        v_c6 = vfmaq_laneq_f32(v_c6, v_b0, v_a6, 3);
        v_c7 = vfmaq_laneq_f32(v_c7, v_b0, v_a7, 3);

        A_ptr0 += 4; A_ptr1 += 4; A_ptr2 += 4; A_ptr3 += 4;
        A_ptr4 += 4; A_ptr5 += 4; A_ptr6 += 4; A_ptr7 += 4;
        B += 4 * ldb;
    }
    for (; k < K; ++k) {
        const float32x4_t v_b0 = vld1q_f32(B);
        v_c0 = vfmaq_n_f32(v_c0, v_b0, A_ptr0[0]);
        v_c1 = vfmaq_n_f32(v_c1, v_b0, A_ptr1[0]);
        v_c2 = vfmaq_n_f32(v_c2, v_b0, A_ptr2[0]);
        v_c3 = vfmaq_n_f32(v_c3, v_b0, A_ptr3[0]);
        v_c4 = vfmaq_n_f32(v_c4, v_b0, A_ptr4[0]);
        v_c5 = vfmaq_n_f32(v_c5, v_b0, A_ptr5[0]);
        v_c6 = vfmaq_n_f32(v_c6, v_b0, A_ptr6[0]);
        v_c7 = vfmaq_n_f32(v_c7, v_b0, A_ptr7[0]);

        A_ptr0 += 1; A_ptr1 += 1; A_ptr2 += 1; A_ptr3 += 1;
        A_ptr4 += 1; A_ptr5 += 1; A_ptr6 += 1; A_ptr7 += 1;
        B += ldb;
    }

    const float32x4_t v_min = vdupq_n_f32(clamp_min);
    const float32x4_t v_max = vdupq_n_f32(clamp_max);

    auto clamp_store = [&](float32x4_t& v, float* dst) {
        v = vaddq_f32(v, vld1q_f32(dst));
        vst1q_f32(dst, vminq_f32(vmaxq_f32(v, v_min), v_max));
    };

    clamp_store(v_c0, C + 0 * ldc);
    clamp_store(v_c1, C + 1 * ldc);
    clamp_store(v_c2, C + 2 * ldc);
    clamp_store(v_c3, C + 3 * ldc);
    clamp_store(v_c4, C + 4 * ldc);
    clamp_store(v_c5, C + 5 * ldc);
    clamp_store(v_c6, C + 6 * ldc);
    clamp_store(v_c7, C + 7 * ldc);
}

inline void mma_direct_8x12_f32(
    float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A, int lda,
    const float* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const float* NNOPS_RESTRICT A_ptr0 = A + 0 * lda;
    const float* NNOPS_RESTRICT A_ptr1 = A + 1 * lda;
    const float* NNOPS_RESTRICT A_ptr2 = A + 2 * lda;
    const float* NNOPS_RESTRICT A_ptr3 = A + 3 * lda;
    const float* NNOPS_RESTRICT A_ptr4 = A + 4 * lda;
    const float* NNOPS_RESTRICT A_ptr5 = A + 5 * lda;
    const float* NNOPS_RESTRICT A_ptr6 = A + 6 * lda;
    const float* NNOPS_RESTRICT A_ptr7 = A + 7 * lda;

    // Use arrays of float32x4x3_t to reduce boilerplate
    float32x4_t v_c[8][3];
    for (int r = 0; r < 8; ++r) {
        v_c[r][0] = v_c[r][1] = v_c[r][2] = vdupq_n_f32(0.0f);
    }

    const float* NNOPS_RESTRICT A_ptrs[8] = {
        A_ptr0, A_ptr1, A_ptr2, A_ptr3,
        A_ptr4, A_ptr5, A_ptr6, A_ptr7
    };

    int k = 0;
    for (; k < K - 3; k += 4) {
        float32x4_t v_a[8];
        for (int r = 0; r < 8; ++r) {
            v_a[r] = vld1q_f32(A_ptrs[r]);
        }

        for (int kk = 0; kk < 4; ++kk) {
            const float32x4_t v_b0 = vld1q_f32(B + kk * ldb + 0 * 4);
            const float32x4_t v_b1 = vld1q_f32(B + kk * ldb + 1 * 4);
            const float32x4_t v_b2 = vld1q_f32(B + kk * ldb + 2 * 4);

            for (int r = 0; r < 8; ++r) {
                v_c[r][0] = vfmaq_laneq_f32(v_c[r][0], v_b0, v_a[r], kk);
                v_c[r][1] = vfmaq_laneq_f32(v_c[r][1], v_b1, v_a[r], kk);
                v_c[r][2] = vfmaq_laneq_f32(v_c[r][2], v_b2, v_a[r], kk);
            }
        }

        for (int r = 0; r < 8; ++r) {
            A_ptrs[r] += 4;
        }
        B += 4 * ldb;
    }
    for (; k < K; ++k) {
        const float32x4_t v_b0 = vld1q_f32(B + 0 * 4);
        const float32x4_t v_b1 = vld1q_f32(B + 1 * 4);
        const float32x4_t v_b2 = vld1q_f32(B + 2 * 4);

        for (int r = 0; r < 8; ++r) {
            v_c[r][0] = vfmaq_n_f32(v_c[r][0], v_b0, A_ptrs[r][0]);
            v_c[r][1] = vfmaq_n_f32(v_c[r][1], v_b1, A_ptrs[r][0]);
            v_c[r][2] = vfmaq_n_f32(v_c[r][2], v_b2, A_ptrs[r][0]);
            A_ptrs[r] += 1;
        }
        B += ldb;
    }

    const float32x4_t v_min = vdupq_n_f32(clamp_min);
    const float32x4_t v_max = vdupq_n_f32(clamp_max);

    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 3; ++c) {
            v_c[r][c] = vaddq_f32(v_c[r][c], vld1q_f32(C + r * ldc + c * 4));
            vst1q_f32(C + r * ldc + c * 4, vminq_f32(vmaxq_f32(v_c[r][c], v_min), v_max));
        }
    }
}

}  // namespace nnops::backend::cpu::aarch64
