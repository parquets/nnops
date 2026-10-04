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

template <bool zero_mode = false>
inline void mma_direct_1x1_f32(
    float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A, int lda,
    const float* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const float* NNOPS_RESTRICT A_ptr0 = A;

    float c0 = 0.0f;
    for (int k = 0; k < K; ++k) {
        c0 += *A_ptr0++ * B[0];
        B += ldb;
    }
    if constexpr (!zero_mode) {
        c0 += C[0];
    }
    C[0] = std::min(std::max(c0, clamp_min), clamp_max);
}

template <bool zero_mode = false>
inline void mma_direct_1x4_f32(
    float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A, int lda,
    const float* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const float* NNOPS_RESTRICT A_ptr0 = A;

    float32x4_t v_c0 = vdupq_n_f32(0.0f);

    int k = 0;
    for (; k < K - 3; k += 4) {
        const float32x4_t v_a0 = vld1q_f32(A_ptr0); A_ptr0 += 4;

        float32x4_t v_b0 = vld1q_f32(B + 0 * ldb);
        v_c0 = vfmaq_laneq_f32(v_c0, v_b0, v_a0, 0);
        v_b0 = vld1q_f32(B + 1 * ldb);
        v_c0 = vfmaq_laneq_f32(v_c0, v_b0, v_a0, 1);
        v_b0 = vld1q_f32(B + 2 * ldb);
        v_c0 = vfmaq_laneq_f32(v_c0, v_b0, v_a0, 2);
        v_b0 = vld1q_f32(B + 3 * ldb);
        v_c0 = vfmaq_laneq_f32(v_c0, v_b0, v_a0, 3);

        B += 4 * ldb;
    }
    for (; k < K; ++k) {
        const float32x4_t v_b0 = vld1q_f32(B); B += ldb;
        v_c0 = vfmaq_n_f32(v_c0, v_b0, *A_ptr0++);
    }

    if constexpr (!zero_mode) {
        v_c0 = vaddq_f32(v_c0, vld1q_f32(C));
    }
    const float32x4_t v_min = vdupq_n_f32(clamp_min);
    const float32x4_t v_max = vdupq_n_f32(clamp_max);
    vst1q_f32(C, vminq_f32(vmaxq_f32(v_c0, v_min), v_max));
}

template <bool zero_mode = false>
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
        const float32x4_t v_a0 = vld1q_f32(A_ptr0); A_ptr0 += 4;

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

    if constexpr (!zero_mode) {
        v_c00 = vaddq_f32(v_c00, vld1q_f32(C + 0 * 4));
        v_c01 = vaddq_f32(v_c01, vld1q_f32(C + 1 * 4));
        v_c02 = vaddq_f32(v_c02, vld1q_f32(C + 2 * 4));
    }

    vst1q_f32(C + 0 * 4, vminq_f32(vmaxq_f32(v_c00, v_min), v_max));
    vst1q_f32(C + 1 * 4, vminq_f32(vmaxq_f32(v_c01, v_min), v_max));
    vst1q_f32(C + 2 * 4, vminq_f32(vmaxq_f32(v_c02, v_min), v_max));
}

// =========================================================================
//  mr=4  kernels
// =========================================================================

template <bool zero_mode = false>
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
        c0 += *A_ptr0++ * b0;
        c1 += *A_ptr1++ * b0;
        c2 += *A_ptr2++ * b0;
        c3 += *A_ptr3++ * b0;

        B += ldb;
    }

    auto write = [&](float* dst, float acc) {
        float v = acc;
        if constexpr (!zero_mode) {
            v = *dst + acc;
        }
        *dst = std::min(std::max(v, clamp_min), clamp_max);
    };
    write(C + 0 * ldc, c0);
    write(C + 1 * ldc, c1);
    write(C + 2 * ldc, c2);
    write(C + 3 * ldc, c3);
}

template <bool zero_mode = false>
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
        const float32x4_t v_a0 = vld1q_f32(A_ptr0); A_ptr0 += 4;
        const float32x4_t v_a1 = vld1q_f32(A_ptr1); A_ptr1 += 4;
        const float32x4_t v_a2 = vld1q_f32(A_ptr2); A_ptr2 += 4;
        const float32x4_t v_a3 = vld1q_f32(A_ptr3); A_ptr3 += 4;

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

        B += 4 * ldb;
    }
    for (; k < K; ++k) {
        const float32x4_t v_b0 = vld1q_f32(B); B += ldb;
        v_c0 = vfmaq_n_f32(v_c0, v_b0, *A_ptr0++);
        v_c1 = vfmaq_n_f32(v_c1, v_b0, *A_ptr1++);
        v_c2 = vfmaq_n_f32(v_c2, v_b0, *A_ptr2++);
        v_c3 = vfmaq_n_f32(v_c3, v_b0, *A_ptr3++);
    }

    const float32x4_t v_min = vdupq_n_f32(clamp_min);
    const float32x4_t v_max = vdupq_n_f32(clamp_max);

    if constexpr (!zero_mode) {
        v_c0 = vaddq_f32(v_c0, vld1q_f32(C + 0 * ldc));
        v_c1 = vaddq_f32(v_c1, vld1q_f32(C + 1 * ldc));
        v_c2 = vaddq_f32(v_c2, vld1q_f32(C + 2 * ldc));
        v_c3 = vaddq_f32(v_c3, vld1q_f32(C + 3 * ldc));
    }

    vst1q_f32(C + 0 * ldc, vminq_f32(vmaxq_f32(v_c0, v_min), v_max));
    vst1q_f32(C + 1 * ldc, vminq_f32(vmaxq_f32(v_c1, v_min), v_max));
    vst1q_f32(C + 2 * ldc, vminq_f32(vmaxq_f32(v_c2, v_min), v_max));
    vst1q_f32(C + 3 * ldc, vminq_f32(vmaxq_f32(v_c3, v_min), v_max));
}

template <bool zero_mode = false>
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
        const float32x4_t v_a0 = vld1q_f32(A_ptr0); A_ptr0 += 4;
        const float32x4_t v_a1 = vld1q_f32(A_ptr1); A_ptr1 += 4;
        const float32x4_t v_a2 = vld1q_f32(A_ptr2); A_ptr2 += 4;
        const float32x4_t v_a3 = vld1q_f32(A_ptr3); A_ptr3 += 4;

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
        v_c10 = vfmaq_n_f32(v_c10, v_b0, A_ptr1[0]);
        v_c11 = vfmaq_n_f32(v_c11, v_b1, A_ptr1[0]);
        v_c12 = vfmaq_n_f32(v_c12, v_b2, A_ptr1[0]);
        A_ptr1 += 1;
        v_c20 = vfmaq_n_f32(v_c20, v_b0, A_ptr2[0]);
        v_c21 = vfmaq_n_f32(v_c21, v_b1, A_ptr2[0]);
        v_c22 = vfmaq_n_f32(v_c22, v_b2, A_ptr2[0]);
        A_ptr2 += 1;
        v_c30 = vfmaq_n_f32(v_c30, v_b0, A_ptr3[0]);
        v_c31 = vfmaq_n_f32(v_c31, v_b1, A_ptr3[0]);
        v_c32 = vfmaq_n_f32(v_c32, v_b2, A_ptr3[0]);
        A_ptr3 += 1;

        B += ldb;
    }

    const float32x4_t v_min = vdupq_n_f32(clamp_min);
    const float32x4_t v_max = vdupq_n_f32(clamp_max);

    if constexpr (!zero_mode) {
        v_c00 = vaddq_f32(v_c00, vld1q_f32(C + 0 * ldc + 0 * 4));
        v_c01 = vaddq_f32(v_c01, vld1q_f32(C + 0 * ldc + 1 * 4));
        v_c02 = vaddq_f32(v_c02, vld1q_f32(C + 0 * ldc + 2 * 4));
    }
    vst1q_f32(C + 0 * ldc + 0 * 4, vminq_f32(vmaxq_f32(v_c00, v_min), v_max));
    vst1q_f32(C + 0 * ldc + 1 * 4, vminq_f32(vmaxq_f32(v_c01, v_min), v_max));
    vst1q_f32(C + 0 * ldc + 2 * 4, vminq_f32(vmaxq_f32(v_c02, v_min), v_max));

    if constexpr (!zero_mode) {
        v_c10 = vaddq_f32(v_c10, vld1q_f32(C + 1 * ldc + 0 * 4));
        v_c11 = vaddq_f32(v_c11, vld1q_f32(C + 1 * ldc + 1 * 4));
        v_c12 = vaddq_f32(v_c12, vld1q_f32(C + 1 * ldc + 2 * 4));
    }
    vst1q_f32(C + 1 * ldc + 0 * 4, vminq_f32(vmaxq_f32(v_c10, v_min), v_max));
    vst1q_f32(C + 1 * ldc + 1 * 4, vminq_f32(vmaxq_f32(v_c11, v_min), v_max));
    vst1q_f32(C + 1 * ldc + 2 * 4, vminq_f32(vmaxq_f32(v_c12, v_min), v_max));

    if constexpr (!zero_mode) {
        v_c20 = vaddq_f32(v_c20, vld1q_f32(C + 2 * ldc + 0 * 4));
        v_c21 = vaddq_f32(v_c21, vld1q_f32(C + 2 * ldc + 1 * 4));
        v_c22 = vaddq_f32(v_c22, vld1q_f32(C + 2 * ldc + 2 * 4));
    }
    vst1q_f32(C + 2 * ldc + 0 * 4, vminq_f32(vmaxq_f32(v_c20, v_min), v_max));
    vst1q_f32(C + 2 * ldc + 1 * 4, vminq_f32(vmaxq_f32(v_c21, v_min), v_max));
    vst1q_f32(C + 2 * ldc + 2 * 4, vminq_f32(vmaxq_f32(v_c22, v_min), v_max));

    if constexpr (!zero_mode) {
        v_c30 = vaddq_f32(v_c30, vld1q_f32(C + 3 * ldc + 0 * 4));
        v_c31 = vaddq_f32(v_c31, vld1q_f32(C + 3 * ldc + 1 * 4));
        v_c32 = vaddq_f32(v_c32, vld1q_f32(C + 3 * ldc + 2 * 4));
    }
    vst1q_f32(C + 3 * ldc + 0 * 4, vminq_f32(vmaxq_f32(v_c30, v_min), v_max));
    vst1q_f32(C + 3 * ldc + 1 * 4, vminq_f32(vmaxq_f32(v_c31, v_min), v_max));
    vst1q_f32(C + 3 * ldc + 2 * 4, vminq_f32(vmaxq_f32(v_c32, v_min), v_max));
}

// =========================================================================
//  mr=6  kernels
//
// The 4-k unroll keeps one A vector per row live across all four kk steps, so
// the register budget is  mr*nr/4 (accumulators) + mr (A) + nr/4 (B). At nr=12
// that is 18+3+6 = 27 vectors for mr=6, versus 24+3+8 = 35 for mr=8 — over the
// 32 NEON registers, which makes an *unrolled* mr=8 kernel spill on every k. So
// mr=6 is the largest height that can unroll at nr=12; mr=8 at nr=12 is
// spill-free only if it drops the unroll, which is what the mr=8 kernels below
// do. Either way the accumulators are enough independent FMA chains to cover
// the pipeline latency.
// =========================================================================

template <bool zero_mode = false>
inline void mma_direct_6x1_f32(
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

    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f;
    float c3 = 0.0f, c4 = 0.0f, c5 = 0.0f;

    for (int k = 0; k < K; ++k) {
        const float b0 = B[0];
        c0 += *A_ptr0++ * b0;
        c1 += *A_ptr1++ * b0;
        c2 += *A_ptr2++ * b0;
        c3 += *A_ptr3++ * b0;
        c4 += *A_ptr4++ * b0;
        c5 += *A_ptr5++ * b0;

        B += ldb;
    }

    auto write = [&](float* dst, float acc) {
        float v = acc;
        if constexpr (!zero_mode) {
            v = *dst + acc;
        }
        *dst = std::min(std::max(v, clamp_min), clamp_max);
    };
    write(C + 0 * ldc, c0);
    write(C + 1 * ldc, c1);
    write(C + 2 * ldc, c2);
    write(C + 3 * ldc, c3);
    write(C + 4 * ldc, c4);
    write(C + 5 * ldc, c5);
}

template <bool zero_mode = false>
inline void mma_direct_6x4_f32(
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

    float32x4_t v_c0 = vdupq_n_f32(0.0f);
    float32x4_t v_c1 = v_c0;
    float32x4_t v_c2 = v_c0;
    float32x4_t v_c3 = v_c0;
    float32x4_t v_c4 = v_c0;
    float32x4_t v_c5 = v_c0;

    int k = 0;
    for (; k < K - 3; k += 4) {
        const float32x4_t v_a0 = vld1q_f32(A_ptr0); A_ptr0 += 4;
        const float32x4_t v_a1 = vld1q_f32(A_ptr1); A_ptr1 += 4;
        const float32x4_t v_a2 = vld1q_f32(A_ptr2); A_ptr2 += 4;
        const float32x4_t v_a3 = vld1q_f32(A_ptr3); A_ptr3 += 4;
        const float32x4_t v_a4 = vld1q_f32(A_ptr4); A_ptr4 += 4;
        const float32x4_t v_a5 = vld1q_f32(A_ptr5); A_ptr5 += 4;

        float32x4_t v_b0 = vld1q_f32(B + 0 * ldb);
        v_c0 = vfmaq_laneq_f32(v_c0, v_b0, v_a0, 0);
        v_c1 = vfmaq_laneq_f32(v_c1, v_b0, v_a1, 0);
        v_c2 = vfmaq_laneq_f32(v_c2, v_b0, v_a2, 0);
        v_c3 = vfmaq_laneq_f32(v_c3, v_b0, v_a3, 0);
        v_c4 = vfmaq_laneq_f32(v_c4, v_b0, v_a4, 0);
        v_c5 = vfmaq_laneq_f32(v_c5, v_b0, v_a5, 0);

        v_b0 = vld1q_f32(B + 1 * ldb);
        v_c0 = vfmaq_laneq_f32(v_c0, v_b0, v_a0, 1);
        v_c1 = vfmaq_laneq_f32(v_c1, v_b0, v_a1, 1);
        v_c2 = vfmaq_laneq_f32(v_c2, v_b0, v_a2, 1);
        v_c3 = vfmaq_laneq_f32(v_c3, v_b0, v_a3, 1);
        v_c4 = vfmaq_laneq_f32(v_c4, v_b0, v_a4, 1);
        v_c5 = vfmaq_laneq_f32(v_c5, v_b0, v_a5, 1);

        v_b0 = vld1q_f32(B + 2 * ldb);
        v_c0 = vfmaq_laneq_f32(v_c0, v_b0, v_a0, 2);
        v_c1 = vfmaq_laneq_f32(v_c1, v_b0, v_a1, 2);
        v_c2 = vfmaq_laneq_f32(v_c2, v_b0, v_a2, 2);
        v_c3 = vfmaq_laneq_f32(v_c3, v_b0, v_a3, 2);
        v_c4 = vfmaq_laneq_f32(v_c4, v_b0, v_a4, 2);
        v_c5 = vfmaq_laneq_f32(v_c5, v_b0, v_a5, 2);

        v_b0 = vld1q_f32(B + 3 * ldb);
        v_c0 = vfmaq_laneq_f32(v_c0, v_b0, v_a0, 3);
        v_c1 = vfmaq_laneq_f32(v_c1, v_b0, v_a1, 3);
        v_c2 = vfmaq_laneq_f32(v_c2, v_b0, v_a2, 3);
        v_c3 = vfmaq_laneq_f32(v_c3, v_b0, v_a3, 3);
        v_c4 = vfmaq_laneq_f32(v_c4, v_b0, v_a4, 3);
        v_c5 = vfmaq_laneq_f32(v_c5, v_b0, v_a5, 3);

        B += 4 * ldb;
    }
    for (; k < K; ++k) {
        const float32x4_t v_b0 = vld1q_f32(B); B += ldb;
        v_c0 = vfmaq_n_f32(v_c0, v_b0, *A_ptr0++);
        v_c1 = vfmaq_n_f32(v_c1, v_b0, *A_ptr1++);
        v_c2 = vfmaq_n_f32(v_c2, v_b0, *A_ptr2++);
        v_c3 = vfmaq_n_f32(v_c3, v_b0, *A_ptr3++);
        v_c4 = vfmaq_n_f32(v_c4, v_b0, *A_ptr4++);
        v_c5 = vfmaq_n_f32(v_c5, v_b0, *A_ptr5++);
    }

    const float32x4_t v_min = vdupq_n_f32(clamp_min);
    const float32x4_t v_max = vdupq_n_f32(clamp_max);

    if constexpr (!zero_mode) {
        v_c0 = vaddq_f32(v_c0, vld1q_f32(C + 0 * ldc));
        v_c1 = vaddq_f32(v_c1, vld1q_f32(C + 1 * ldc));
        v_c2 = vaddq_f32(v_c2, vld1q_f32(C + 2 * ldc));
        v_c3 = vaddq_f32(v_c3, vld1q_f32(C + 3 * ldc));
        v_c4 = vaddq_f32(v_c4, vld1q_f32(C + 4 * ldc));
        v_c5 = vaddq_f32(v_c5, vld1q_f32(C + 5 * ldc));
    }

    vst1q_f32(C + 0 * ldc, vminq_f32(vmaxq_f32(v_c0, v_min), v_max));
    vst1q_f32(C + 1 * ldc, vminq_f32(vmaxq_f32(v_c1, v_min), v_max));
    vst1q_f32(C + 2 * ldc, vminq_f32(vmaxq_f32(v_c2, v_min), v_max));
    vst1q_f32(C + 3 * ldc, vminq_f32(vmaxq_f32(v_c3, v_min), v_max));
    vst1q_f32(C + 4 * ldc, vminq_f32(vmaxq_f32(v_c4, v_min), v_max));
    vst1q_f32(C + 5 * ldc, vminq_f32(vmaxq_f32(v_c5, v_min), v_max));
}

template <bool zero_mode = false>
inline void mma_direct_6x12_f32(
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

    float32x4_t c00 = vdupq_n_f32(0.0f), c01 = vdupq_n_f32(0.0f), c02 = vdupq_n_f32(0.0f);
    float32x4_t c10 = vdupq_n_f32(0.0f), c11 = vdupq_n_f32(0.0f), c12 = vdupq_n_f32(0.0f);
    float32x4_t c20 = vdupq_n_f32(0.0f), c21 = vdupq_n_f32(0.0f), c22 = vdupq_n_f32(0.0f);
    float32x4_t c30 = vdupq_n_f32(0.0f), c31 = vdupq_n_f32(0.0f), c32 = vdupq_n_f32(0.0f);
    float32x4_t c40 = vdupq_n_f32(0.0f), c41 = vdupq_n_f32(0.0f), c42 = vdupq_n_f32(0.0f);
    float32x4_t c50 = vdupq_n_f32(0.0f), c51 = vdupq_n_f32(0.0f), c52 = vdupq_n_f32(0.0f);

    int k = 0;
    for (; k < K - 3; k += 4) {
        const float32x4_t v_a0 = vld1q_f32(A_ptr0); A_ptr0 += 4;
        const float32x4_t v_a1 = vld1q_f32(A_ptr1); A_ptr1 += 4;
        const float32x4_t v_a2 = vld1q_f32(A_ptr2); A_ptr2 += 4;
        const float32x4_t v_a3 = vld1q_f32(A_ptr3); A_ptr3 += 4;
        const float32x4_t v_a4 = vld1q_f32(A_ptr4); A_ptr4 += 4;
        const float32x4_t v_a5 = vld1q_f32(A_ptr5); A_ptr5 += 4;

        // kk=0
        float32x4_t v_b0 = vld1q_f32(B + 0 * ldb + 0 * 4);
        float32x4_t v_b1 = vld1q_f32(B + 0 * ldb + 1 * 4);
        float32x4_t v_b2 = vld1q_f32(B + 0 * ldb + 2 * 4);
        c00 = vfmaq_laneq_f32(c00, v_b0, v_a0, 0); c01 = vfmaq_laneq_f32(c01, v_b1, v_a0, 0); c02 = vfmaq_laneq_f32(c02, v_b2, v_a0, 0);
        c10 = vfmaq_laneq_f32(c10, v_b0, v_a1, 0); c11 = vfmaq_laneq_f32(c11, v_b1, v_a1, 0); c12 = vfmaq_laneq_f32(c12, v_b2, v_a1, 0);
        c20 = vfmaq_laneq_f32(c20, v_b0, v_a2, 0); c21 = vfmaq_laneq_f32(c21, v_b1, v_a2, 0); c22 = vfmaq_laneq_f32(c22, v_b2, v_a2, 0);
        c30 = vfmaq_laneq_f32(c30, v_b0, v_a3, 0); c31 = vfmaq_laneq_f32(c31, v_b1, v_a3, 0); c32 = vfmaq_laneq_f32(c32, v_b2, v_a3, 0);
        c40 = vfmaq_laneq_f32(c40, v_b0, v_a4, 0); c41 = vfmaq_laneq_f32(c41, v_b1, v_a4, 0); c42 = vfmaq_laneq_f32(c42, v_b2, v_a4, 0);
        c50 = vfmaq_laneq_f32(c50, v_b0, v_a5, 0); c51 = vfmaq_laneq_f32(c51, v_b1, v_a5, 0); c52 = vfmaq_laneq_f32(c52, v_b2, v_a5, 0);

        // kk=1
        v_b0 = vld1q_f32(B + 1 * ldb + 0 * 4);
        v_b1 = vld1q_f32(B + 1 * ldb + 1 * 4);
        v_b2 = vld1q_f32(B + 1 * ldb + 2 * 4);
        c00 = vfmaq_laneq_f32(c00, v_b0, v_a0, 1); c01 = vfmaq_laneq_f32(c01, v_b1, v_a0, 1); c02 = vfmaq_laneq_f32(c02, v_b2, v_a0, 1);
        c10 = vfmaq_laneq_f32(c10, v_b0, v_a1, 1); c11 = vfmaq_laneq_f32(c11, v_b1, v_a1, 1); c12 = vfmaq_laneq_f32(c12, v_b2, v_a1, 1);
        c20 = vfmaq_laneq_f32(c20, v_b0, v_a2, 1); c21 = vfmaq_laneq_f32(c21, v_b1, v_a2, 1); c22 = vfmaq_laneq_f32(c22, v_b2, v_a2, 1);
        c30 = vfmaq_laneq_f32(c30, v_b0, v_a3, 1); c31 = vfmaq_laneq_f32(c31, v_b1, v_a3, 1); c32 = vfmaq_laneq_f32(c32, v_b2, v_a3, 1);
        c40 = vfmaq_laneq_f32(c40, v_b0, v_a4, 1); c41 = vfmaq_laneq_f32(c41, v_b1, v_a4, 1); c42 = vfmaq_laneq_f32(c42, v_b2, v_a4, 1);
        c50 = vfmaq_laneq_f32(c50, v_b0, v_a5, 1); c51 = vfmaq_laneq_f32(c51, v_b1, v_a5, 1); c52 = vfmaq_laneq_f32(c52, v_b2, v_a5, 1);

        // kk=2
        v_b0 = vld1q_f32(B + 2 * ldb + 0 * 4);
        v_b1 = vld1q_f32(B + 2 * ldb + 1 * 4);
        v_b2 = vld1q_f32(B + 2 * ldb + 2 * 4);
        c00 = vfmaq_laneq_f32(c00, v_b0, v_a0, 2); c01 = vfmaq_laneq_f32(c01, v_b1, v_a0, 2); c02 = vfmaq_laneq_f32(c02, v_b2, v_a0, 2);
        c10 = vfmaq_laneq_f32(c10, v_b0, v_a1, 2); c11 = vfmaq_laneq_f32(c11, v_b1, v_a1, 2); c12 = vfmaq_laneq_f32(c12, v_b2, v_a1, 2);
        c20 = vfmaq_laneq_f32(c20, v_b0, v_a2, 2); c21 = vfmaq_laneq_f32(c21, v_b1, v_a2, 2); c22 = vfmaq_laneq_f32(c22, v_b2, v_a2, 2);
        c30 = vfmaq_laneq_f32(c30, v_b0, v_a3, 2); c31 = vfmaq_laneq_f32(c31, v_b1, v_a3, 2); c32 = vfmaq_laneq_f32(c32, v_b2, v_a3, 2);
        c40 = vfmaq_laneq_f32(c40, v_b0, v_a4, 2); c41 = vfmaq_laneq_f32(c41, v_b1, v_a4, 2); c42 = vfmaq_laneq_f32(c42, v_b2, v_a4, 2);
        c50 = vfmaq_laneq_f32(c50, v_b0, v_a5, 2); c51 = vfmaq_laneq_f32(c51, v_b1, v_a5, 2); c52 = vfmaq_laneq_f32(c52, v_b2, v_a5, 2);

        // kk=3
        v_b0 = vld1q_f32(B + 3 * ldb + 0 * 4);
        v_b1 = vld1q_f32(B + 3 * ldb + 1 * 4);
        v_b2 = vld1q_f32(B + 3 * ldb + 2 * 4);
        c00 = vfmaq_laneq_f32(c00, v_b0, v_a0, 3); c01 = vfmaq_laneq_f32(c01, v_b1, v_a0, 3); c02 = vfmaq_laneq_f32(c02, v_b2, v_a0, 3);
        c10 = vfmaq_laneq_f32(c10, v_b0, v_a1, 3); c11 = vfmaq_laneq_f32(c11, v_b1, v_a1, 3); c12 = vfmaq_laneq_f32(c12, v_b2, v_a1, 3);
        c20 = vfmaq_laneq_f32(c20, v_b0, v_a2, 3); c21 = vfmaq_laneq_f32(c21, v_b1, v_a2, 3); c22 = vfmaq_laneq_f32(c22, v_b2, v_a2, 3);
        c30 = vfmaq_laneq_f32(c30, v_b0, v_a3, 3); c31 = vfmaq_laneq_f32(c31, v_b1, v_a3, 3); c32 = vfmaq_laneq_f32(c32, v_b2, v_a3, 3);
        c40 = vfmaq_laneq_f32(c40, v_b0, v_a4, 3); c41 = vfmaq_laneq_f32(c41, v_b1, v_a4, 3); c42 = vfmaq_laneq_f32(c42, v_b2, v_a4, 3);
        c50 = vfmaq_laneq_f32(c50, v_b0, v_a5, 3); c51 = vfmaq_laneq_f32(c51, v_b1, v_a5, 3); c52 = vfmaq_laneq_f32(c52, v_b2, v_a5, 3);

        B += 4 * ldb;
    }
    for (; k < K; ++k) {
        const float32x4_t v_b0 = vld1q_f32(B + 0 * 4);
        const float32x4_t v_b1 = vld1q_f32(B + 1 * 4);
        const float32x4_t v_b2 = vld1q_f32(B + 2 * 4);

        c00 = vfmaq_n_f32(c00, v_b0, A_ptr0[0]); c01 = vfmaq_n_f32(c01, v_b1, A_ptr0[0]); c02 = vfmaq_n_f32(c02, v_b2, A_ptr0[0]);
        A_ptr0 += 1;
        c10 = vfmaq_n_f32(c10, v_b0, A_ptr1[0]); c11 = vfmaq_n_f32(c11, v_b1, A_ptr1[0]); c12 = vfmaq_n_f32(c12, v_b2, A_ptr1[0]);
        A_ptr1 += 1;
        c20 = vfmaq_n_f32(c20, v_b0, A_ptr2[0]); c21 = vfmaq_n_f32(c21, v_b1, A_ptr2[0]); c22 = vfmaq_n_f32(c22, v_b2, A_ptr2[0]);
        A_ptr2 += 1;
        c30 = vfmaq_n_f32(c30, v_b0, A_ptr3[0]); c31 = vfmaq_n_f32(c31, v_b1, A_ptr3[0]); c32 = vfmaq_n_f32(c32, v_b2, A_ptr3[0]);
        A_ptr3 += 1;
        c40 = vfmaq_n_f32(c40, v_b0, A_ptr4[0]); c41 = vfmaq_n_f32(c41, v_b1, A_ptr4[0]); c42 = vfmaq_n_f32(c42, v_b2, A_ptr4[0]);
        A_ptr4 += 1;
        c50 = vfmaq_n_f32(c50, v_b0, A_ptr5[0]); c51 = vfmaq_n_f32(c51, v_b1, A_ptr5[0]); c52 = vfmaq_n_f32(c52, v_b2, A_ptr5[0]);
        A_ptr5 += 1;

        B += ldb;
    }

    const float32x4_t v_min = vdupq_n_f32(clamp_min);
    const float32x4_t v_max = vdupq_n_f32(clamp_max);

    if constexpr (!zero_mode) {
        c00 = vaddq_f32(c00, vld1q_f32(C + 0 * ldc + 0 * 4));
        c01 = vaddq_f32(c01, vld1q_f32(C + 0 * ldc + 1 * 4));
        c02 = vaddq_f32(c02, vld1q_f32(C + 0 * ldc + 2 * 4));
    }
    vst1q_f32(C + 0 * ldc + 0 * 4, vminq_f32(vmaxq_f32(c00, v_min), v_max));
    vst1q_f32(C + 0 * ldc + 1 * 4, vminq_f32(vmaxq_f32(c01, v_min), v_max));
    vst1q_f32(C + 0 * ldc + 2 * 4, vminq_f32(vmaxq_f32(c02, v_min), v_max));

    if constexpr (!zero_mode) {
        c10 = vaddq_f32(c10, vld1q_f32(C + 1 * ldc + 0 * 4));
        c11 = vaddq_f32(c11, vld1q_f32(C + 1 * ldc + 1 * 4));
        c12 = vaddq_f32(c12, vld1q_f32(C + 1 * ldc + 2 * 4));
    }
    vst1q_f32(C + 1 * ldc + 0 * 4, vminq_f32(vmaxq_f32(c10, v_min), v_max));
    vst1q_f32(C + 1 * ldc + 1 * 4, vminq_f32(vmaxq_f32(c11, v_min), v_max));
    vst1q_f32(C + 1 * ldc + 2 * 4, vminq_f32(vmaxq_f32(c12, v_min), v_max));

    if constexpr (!zero_mode) {
        c20 = vaddq_f32(c20, vld1q_f32(C + 2 * ldc + 0 * 4));
        c21 = vaddq_f32(c21, vld1q_f32(C + 2 * ldc + 1 * 4));
        c22 = vaddq_f32(c22, vld1q_f32(C + 2 * ldc + 2 * 4));
    }
    vst1q_f32(C + 2 * ldc + 0 * 4, vminq_f32(vmaxq_f32(c20, v_min), v_max));
    vst1q_f32(C + 2 * ldc + 1 * 4, vminq_f32(vmaxq_f32(c21, v_min), v_max));
    vst1q_f32(C + 2 * ldc + 2 * 4, vminq_f32(vmaxq_f32(c22, v_min), v_max));

    if constexpr (!zero_mode) {
        c30 = vaddq_f32(c30, vld1q_f32(C + 3 * ldc + 0 * 4));
        c31 = vaddq_f32(c31, vld1q_f32(C + 3 * ldc + 1 * 4));
        c32 = vaddq_f32(c32, vld1q_f32(C + 3 * ldc + 2 * 4));
    }
    vst1q_f32(C + 3 * ldc + 0 * 4, vminq_f32(vmaxq_f32(c30, v_min), v_max));
    vst1q_f32(C + 3 * ldc + 1 * 4, vminq_f32(vmaxq_f32(c31, v_min), v_max));
    vst1q_f32(C + 3 * ldc + 2 * 4, vminq_f32(vmaxq_f32(c32, v_min), v_max));

    if constexpr (!zero_mode) {
        c40 = vaddq_f32(c40, vld1q_f32(C + 4 * ldc + 0 * 4));
        c41 = vaddq_f32(c41, vld1q_f32(C + 4 * ldc + 1 * 4));
        c42 = vaddq_f32(c42, vld1q_f32(C + 4 * ldc + 2 * 4));
    }
    vst1q_f32(C + 4 * ldc + 0 * 4, vminq_f32(vmaxq_f32(c40, v_min), v_max));
    vst1q_f32(C + 4 * ldc + 1 * 4, vminq_f32(vmaxq_f32(c41, v_min), v_max));
    vst1q_f32(C + 4 * ldc + 2 * 4, vminq_f32(vmaxq_f32(c42, v_min), v_max));

    if constexpr (!zero_mode) {
        c50 = vaddq_f32(c50, vld1q_f32(C + 5 * ldc + 0 * 4));
        c51 = vaddq_f32(c51, vld1q_f32(C + 5 * ldc + 1 * 4));
        c52 = vaddq_f32(c52, vld1q_f32(C + 5 * ldc + 2 * 4));
    }
    vst1q_f32(C + 5 * ldc + 0 * 4, vminq_f32(vmaxq_f32(c50, v_min), v_max));
    vst1q_f32(C + 5 * ldc + 1 * 4, vminq_f32(vmaxq_f32(c51, v_min), v_max));
    vst1q_f32(C + 5 * ldc + 2 * 4, vminq_f32(vmaxq_f32(c52, v_min), v_max));
}

// =========================================================================
//  mr=8  kernels
//
// 8x1 and 8x4 keep the 4-k unroll (their register budget is well inside 32).
// 8x12 does not — see the comment in its body.
// =========================================================================

template <bool zero_mode = false>
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

    auto write = [&](float* dst, float acc) {
        float v = acc;
        if constexpr (!zero_mode) {
            v = *dst + acc;
        }
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

template <bool zero_mode = false>
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
        const float32x4_t v_a0 = vld1q_f32(A_ptr0); A_ptr0 += 4;
        const float32x4_t v_a1 = vld1q_f32(A_ptr1); A_ptr1 += 4;
        const float32x4_t v_a2 = vld1q_f32(A_ptr2); A_ptr2 += 4;
        const float32x4_t v_a3 = vld1q_f32(A_ptr3); A_ptr3 += 4;
        const float32x4_t v_a4 = vld1q_f32(A_ptr4); A_ptr4 += 4;
        const float32x4_t v_a5 = vld1q_f32(A_ptr5); A_ptr5 += 4;
        const float32x4_t v_a6 = vld1q_f32(A_ptr6); A_ptr6 += 4;
        const float32x4_t v_a7 = vld1q_f32(A_ptr7); A_ptr7 += 4;

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

        B += 4 * ldb;
    }
    for (; k < K; ++k) {
        const float32x4_t v_b0 = vld1q_f32(B); B += ldb;
        v_c0 = vfmaq_n_f32(v_c0, v_b0, *A_ptr0++);
        v_c1 = vfmaq_n_f32(v_c1, v_b0, *A_ptr1++);
        v_c2 = vfmaq_n_f32(v_c2, v_b0, *A_ptr2++);
        v_c3 = vfmaq_n_f32(v_c3, v_b0, *A_ptr3++);
        v_c4 = vfmaq_n_f32(v_c4, v_b0, *A_ptr4++);
        v_c5 = vfmaq_n_f32(v_c5, v_b0, *A_ptr5++);
        v_c6 = vfmaq_n_f32(v_c6, v_b0, *A_ptr6++);
        v_c7 = vfmaq_n_f32(v_c7, v_b0, *A_ptr7++);
    }

    const float32x4_t v_min = vdupq_n_f32(clamp_min);
    const float32x4_t v_max = vdupq_n_f32(clamp_max);

    if constexpr (!zero_mode) {
        v_c0 = vaddq_f32(v_c0, vld1q_f32(C + 0 * ldc));
        v_c1 = vaddq_f32(v_c1, vld1q_f32(C + 1 * ldc));
        v_c2 = vaddq_f32(v_c2, vld1q_f32(C + 2 * ldc));
        v_c3 = vaddq_f32(v_c3, vld1q_f32(C + 3 * ldc));
        v_c4 = vaddq_f32(v_c4, vld1q_f32(C + 4 * ldc));
        v_c5 = vaddq_f32(v_c5, vld1q_f32(C + 5 * ldc));
        v_c6 = vaddq_f32(v_c6, vld1q_f32(C + 6 * ldc));
        v_c7 = vaddq_f32(v_c7, vld1q_f32(C + 7 * ldc));
    }

    vst1q_f32(C + 0 * ldc, vminq_f32(vmaxq_f32(v_c0, v_min), v_max));
    vst1q_f32(C + 1 * ldc, vminq_f32(vmaxq_f32(v_c1, v_min), v_max));
    vst1q_f32(C + 2 * ldc, vminq_f32(vmaxq_f32(v_c2, v_min), v_max));
    vst1q_f32(C + 3 * ldc, vminq_f32(vmaxq_f32(v_c3, v_min), v_max));
    vst1q_f32(C + 4 * ldc, vminq_f32(vmaxq_f32(v_c4, v_min), v_max));
    vst1q_f32(C + 5 * ldc, vminq_f32(vmaxq_f32(v_c5, v_min), v_max));
    vst1q_f32(C + 6 * ldc, vminq_f32(vmaxq_f32(v_c6, v_min), v_max));
    vst1q_f32(C + 7 * ldc, vminq_f32(vmaxq_f32(v_c7, v_min), v_max));
}

template <bool zero_mode = false>
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

    float32x4_t c00 = vdupq_n_f32(0.0f), c01 = vdupq_n_f32(0.0f), c02 = vdupq_n_f32(0.0f);
    float32x4_t c10 = vdupq_n_f32(0.0f), c11 = vdupq_n_f32(0.0f), c12 = vdupq_n_f32(0.0f);
    float32x4_t c20 = vdupq_n_f32(0.0f), c21 = vdupq_n_f32(0.0f), c22 = vdupq_n_f32(0.0f);
    float32x4_t c30 = vdupq_n_f32(0.0f), c31 = vdupq_n_f32(0.0f), c32 = vdupq_n_f32(0.0f);
    float32x4_t c40 = vdupq_n_f32(0.0f), c41 = vdupq_n_f32(0.0f), c42 = vdupq_n_f32(0.0f);
    float32x4_t c50 = vdupq_n_f32(0.0f), c51 = vdupq_n_f32(0.0f), c52 = vdupq_n_f32(0.0f);
    float32x4_t c60 = vdupq_n_f32(0.0f), c61 = vdupq_n_f32(0.0f), c62 = vdupq_n_f32(0.0f);
    float32x4_t c70 = vdupq_n_f32(0.0f), c71 = vdupq_n_f32(0.0f), c72 = vdupq_n_f32(0.0f);

    // K is *not* unrolled by 4 here, unlike the other direct kernels. The unroll
    // holds one A vector per row live across the whole k-group, and at mr=8 that
    // plus the 24 accumulators and 3 B vectors is 35 vectors — three past the 32
    // NEON registers, so the compiler spills in the hot loop. Measured at K=512
    // with everything resident in L1: 22.2 ps/FMA unrolled (spilling) versus 16.0
    // ps/FMA for the scalar form below, and 16.6 for mr=6 *with* the unroll. The
    // unroll only pays while it still fits; once it spills it is strictly worse,
    // and scalar also issues fewer memory ops per k (3 B vectors + 8 A lane loads
    // = 11, against 5 loads + ~7.5 spill ops = 12.5 for the unrolled form).
    for (int k = 0; k < K; ++k) {
        const float32x4_t v_b0 = vld1q_f32(B + 0 * 4);
        const float32x4_t v_b1 = vld1q_f32(B + 1 * 4);
        const float32x4_t v_b2 = vld1q_f32(B + 2 * 4);

        c00 = vfmaq_n_f32(c00, v_b0, A_ptr0[0]); c01 = vfmaq_n_f32(c01, v_b1, A_ptr0[0]); c02 = vfmaq_n_f32(c02, v_b2, A_ptr0[0]);
        A_ptr0 += 1;
        c10 = vfmaq_n_f32(c10, v_b0, A_ptr1[0]); c11 = vfmaq_n_f32(c11, v_b1, A_ptr1[0]); c12 = vfmaq_n_f32(c12, v_b2, A_ptr1[0]);
        A_ptr1 += 1;
        c20 = vfmaq_n_f32(c20, v_b0, A_ptr2[0]); c21 = vfmaq_n_f32(c21, v_b1, A_ptr2[0]); c22 = vfmaq_n_f32(c22, v_b2, A_ptr2[0]);
        A_ptr2 += 1;
        c30 = vfmaq_n_f32(c30, v_b0, A_ptr3[0]); c31 = vfmaq_n_f32(c31, v_b1, A_ptr3[0]); c32 = vfmaq_n_f32(c32, v_b2, A_ptr3[0]);
        A_ptr3 += 1;
        c40 = vfmaq_n_f32(c40, v_b0, A_ptr4[0]); c41 = vfmaq_n_f32(c41, v_b1, A_ptr4[0]); c42 = vfmaq_n_f32(c42, v_b2, A_ptr4[0]);
        A_ptr4 += 1;
        c50 = vfmaq_n_f32(c50, v_b0, A_ptr5[0]); c51 = vfmaq_n_f32(c51, v_b1, A_ptr5[0]); c52 = vfmaq_n_f32(c52, v_b2, A_ptr5[0]);
        A_ptr5 += 1;
        c60 = vfmaq_n_f32(c60, v_b0, A_ptr6[0]); c61 = vfmaq_n_f32(c61, v_b1, A_ptr6[0]); c62 = vfmaq_n_f32(c62, v_b2, A_ptr6[0]);
        A_ptr6 += 1;
        c70 = vfmaq_n_f32(c70, v_b0, A_ptr7[0]); c71 = vfmaq_n_f32(c71, v_b1, A_ptr7[0]); c72 = vfmaq_n_f32(c72, v_b2, A_ptr7[0]);
        A_ptr7 += 1;

        B += ldb;
    }

    const float32x4_t v_min = vdupq_n_f32(clamp_min);
    const float32x4_t v_max = vdupq_n_f32(clamp_max);

    if constexpr (!zero_mode) {
        c00 = vaddq_f32(c00, vld1q_f32(C + 0 * ldc + 0 * 4));
        c01 = vaddq_f32(c01, vld1q_f32(C + 0 * ldc + 1 * 4));
        c02 = vaddq_f32(c02, vld1q_f32(C + 0 * ldc + 2 * 4));
    }
    vst1q_f32(C + 0 * ldc + 0 * 4, vminq_f32(vmaxq_f32(c00, v_min), v_max));
    vst1q_f32(C + 0 * ldc + 1 * 4, vminq_f32(vmaxq_f32(c01, v_min), v_max));
    vst1q_f32(C + 0 * ldc + 2 * 4, vminq_f32(vmaxq_f32(c02, v_min), v_max));

    if constexpr (!zero_mode) {
        c10 = vaddq_f32(c10, vld1q_f32(C + 1 * ldc + 0 * 4));
        c11 = vaddq_f32(c11, vld1q_f32(C + 1 * ldc + 1 * 4));
        c12 = vaddq_f32(c12, vld1q_f32(C + 1 * ldc + 2 * 4));
    }
    vst1q_f32(C + 1 * ldc + 0 * 4, vminq_f32(vmaxq_f32(c10, v_min), v_max));
    vst1q_f32(C + 1 * ldc + 1 * 4, vminq_f32(vmaxq_f32(c11, v_min), v_max));
    vst1q_f32(C + 1 * ldc + 2 * 4, vminq_f32(vmaxq_f32(c12, v_min), v_max));

    if constexpr (!zero_mode) {
        c20 = vaddq_f32(c20, vld1q_f32(C + 2 * ldc + 0 * 4));
        c21 = vaddq_f32(c21, vld1q_f32(C + 2 * ldc + 1 * 4));
        c22 = vaddq_f32(c22, vld1q_f32(C + 2 * ldc + 2 * 4));
    }
    vst1q_f32(C + 2 * ldc + 0 * 4, vminq_f32(vmaxq_f32(c20, v_min), v_max));
    vst1q_f32(C + 2 * ldc + 1 * 4, vminq_f32(vmaxq_f32(c21, v_min), v_max));
    vst1q_f32(C + 2 * ldc + 2 * 4, vminq_f32(vmaxq_f32(c22, v_min), v_max));

    if constexpr (!zero_mode) {
        c30 = vaddq_f32(c30, vld1q_f32(C + 3 * ldc + 0 * 4));
        c31 = vaddq_f32(c31, vld1q_f32(C + 3 * ldc + 1 * 4));
        c32 = vaddq_f32(c32, vld1q_f32(C + 3 * ldc + 2 * 4));
    }
    vst1q_f32(C + 3 * ldc + 0 * 4, vminq_f32(vmaxq_f32(c30, v_min), v_max));
    vst1q_f32(C + 3 * ldc + 1 * 4, vminq_f32(vmaxq_f32(c31, v_min), v_max));
    vst1q_f32(C + 3 * ldc + 2 * 4, vminq_f32(vmaxq_f32(c32, v_min), v_max));

    if constexpr (!zero_mode) {
        c40 = vaddq_f32(c40, vld1q_f32(C + 4 * ldc + 0 * 4));
        c41 = vaddq_f32(c41, vld1q_f32(C + 4 * ldc + 1 * 4));
        c42 = vaddq_f32(c42, vld1q_f32(C + 4 * ldc + 2 * 4));
    }
    vst1q_f32(C + 4 * ldc + 0 * 4, vminq_f32(vmaxq_f32(c40, v_min), v_max));
    vst1q_f32(C + 4 * ldc + 1 * 4, vminq_f32(vmaxq_f32(c41, v_min), v_max));
    vst1q_f32(C + 4 * ldc + 2 * 4, vminq_f32(vmaxq_f32(c42, v_min), v_max));

    if constexpr (!zero_mode) {
        c50 = vaddq_f32(c50, vld1q_f32(C + 5 * ldc + 0 * 4));
        c51 = vaddq_f32(c51, vld1q_f32(C + 5 * ldc + 1 * 4));
        c52 = vaddq_f32(c52, vld1q_f32(C + 5 * ldc + 2 * 4));
    }
    vst1q_f32(C + 5 * ldc + 0 * 4, vminq_f32(vmaxq_f32(c50, v_min), v_max));
    vst1q_f32(C + 5 * ldc + 1 * 4, vminq_f32(vmaxq_f32(c51, v_min), v_max));
    vst1q_f32(C + 5 * ldc + 2 * 4, vminq_f32(vmaxq_f32(c52, v_min), v_max));

    if constexpr (!zero_mode) {
        c60 = vaddq_f32(c60, vld1q_f32(C + 6 * ldc + 0 * 4));
        c61 = vaddq_f32(c61, vld1q_f32(C + 6 * ldc + 1 * 4));
        c62 = vaddq_f32(c62, vld1q_f32(C + 6 * ldc + 2 * 4));
    }
    vst1q_f32(C + 6 * ldc + 0 * 4, vminq_f32(vmaxq_f32(c60, v_min), v_max));
    vst1q_f32(C + 6 * ldc + 1 * 4, vminq_f32(vmaxq_f32(c61, v_min), v_max));
    vst1q_f32(C + 6 * ldc + 2 * 4, vminq_f32(vmaxq_f32(c62, v_min), v_max));

    if constexpr (!zero_mode) {
        c70 = vaddq_f32(c70, vld1q_f32(C + 7 * ldc + 0 * 4));
        c71 = vaddq_f32(c71, vld1q_f32(C + 7 * ldc + 1 * 4));
        c72 = vaddq_f32(c72, vld1q_f32(C + 7 * ldc + 2 * 4));
    }
    vst1q_f32(C + 7 * ldc + 0 * 4, vminq_f32(vmaxq_f32(c70, v_min), v_max));
    vst1q_f32(C + 7 * ldc + 1 * 4, vminq_f32(vmaxq_f32(c71, v_min), v_max));
    vst1q_f32(C + 7 * ldc + 2 * 4, vminq_f32(vmaxq_f32(c72, v_min), v_max));
}

}  // namespace nnops::backend::cpu::aarch64
