#pragma once
/// @file mma_pack_f32.hpp
/// @brief AArch64 NEON float32 MMA (matrix micro-accumulate) packed-B kernels.
///
/// Each kernel computes  C[mr][nr] += A[mr][K] × B_packed[K][nr]
/// where B is pre-packed (row-major in panels of nr columns) and A is
/// accessed element-by-element with broadcast. The result is clamped
/// to [clamp_min, clamp_max] after accumulation.
///
/// Tile sizes (AArch64-optimised):
///   M ∈ {8, 4, 1}    N ∈ {12, 4, 1}
///
/// Reference: nn_compute/src/cpu/kernel/mma/aarch64/mma_pack_f32.hpp

#include <arm_neon.h>
#include <algorithm>
#include "backend/cpu/common/restrict.hpp"

namespace nnops::backend::cpu::aarch64 {

// =========================================================================
//  mr=1  kernels
// =========================================================================

inline void mma_pack_1x1_f32(float* NNOPS_RESTRICT C, int ldc,
                             const float* NNOPS_RESTRICT A,
                             const float* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    float c0 = 0.0f;
    for (int k = 0; k < K; ++k) {
        c0 += A[0] * B[0];
        A += 1;
        B += ldb;
    }
    c0 += C[0 * ldc];
    C[0 * ldc] = std::min(std::max(c0, clamp_min), clamp_max);
}

inline void mma_pack_1x4_f32(float* NNOPS_RESTRICT C, int ldc,
                             const float* NNOPS_RESTRICT A,
                             const float* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    float32x4_t c0 = vdupq_n_f32(0.0f);
    for (int k = 0; k < K; ++k) {
        c0 = vfmaq_n_f32(c0, vld1q_f32(B), A[0]);
        A += 1;
        B += ldb;
    }
    c0 = vaddq_f32(c0, vld1q_f32(C));
    const float32x4_t vmin = vdupq_n_f32(clamp_min);
    const float32x4_t vmax = vdupq_n_f32(clamp_max);
    vst1q_f32(C, vminq_f32(vmaxq_f32(c0, vmin), vmax));
}

inline void mma_pack_1x12_f32(float* NNOPS_RESTRICT C, int ldc,
                              const float* NNOPS_RESTRICT A,
                              const float* NNOPS_RESTRICT B,
                              int ldb, int K,
                              float clamp_min, float clamp_max) noexcept {
    float32x4_t c0 = vdupq_n_f32(0.0f);
    float32x4_t c1 = vdupq_n_f32(0.0f);
    float32x4_t c2 = vdupq_n_f32(0.0f);

    for (int k = 0; k < K; ++k) {
        const float a = A[0];
        c0 = vfmaq_n_f32(c0, vld1q_f32(B + 0 * 4), a);
        c1 = vfmaq_n_f32(c1, vld1q_f32(B + 1 * 4), a);
        c2 = vfmaq_n_f32(c2, vld1q_f32(B + 2 * 4), a);
        A += 1;
        B += ldb;
    }

    const float32x4_t vmin = vdupq_n_f32(clamp_min);
    const float32x4_t vmax = vdupq_n_f32(clamp_max);

    c0 = vaddq_f32(c0, vld1q_f32(C + 0 * 4));
    c1 = vaddq_f32(c1, vld1q_f32(C + 1 * 4));
    c2 = vaddq_f32(c2, vld1q_f32(C + 2 * 4));

    vst1q_f32(C + 0 * 4, vminq_f32(vmaxq_f32(c0, vmin), vmax));
    vst1q_f32(C + 1 * 4, vminq_f32(vmaxq_f32(c1, vmin), vmax));
    vst1q_f32(C + 2 * 4, vminq_f32(vmaxq_f32(c2, vmin), vmax));
}

// =========================================================================
//  mr=4  kernels
// =========================================================================

inline void mma_pack_4x1_f32(float* NNOPS_RESTRICT C, int ldc,
                             const float* NNOPS_RESTRICT A,
                             const float* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float b = B[0];
        c0 += A[0] * b;
        c1 += A[1] * b;
        c2 += A[2] * b;
        c3 += A[3] * b;
        A += 4;
        B += ldb;
    }
    c0 = std::min(std::max(c0 + C[0 * ldc], clamp_min), clamp_max);
    c1 = std::min(std::max(c1 + C[1 * ldc], clamp_min), clamp_max);
    c2 = std::min(std::max(c2 + C[2 * ldc], clamp_min), clamp_max);
    c3 = std::min(std::max(c3 + C[3 * ldc], clamp_min), clamp_max);
    C[0 * ldc] = c0;
    C[1 * ldc] = c1;
    C[2 * ldc] = c2;
    C[3 * ldc] = c3;
}

inline void mma_pack_4x4_f32(float* NNOPS_RESTRICT C, int ldc,
                             const float* NNOPS_RESTRICT A,
                             const float* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    float32x4_t c0 = vdupq_n_f32(0.0f);
    float32x4_t c1 = vdupq_n_f32(0.0f);
    float32x4_t c2 = vdupq_n_f32(0.0f);
    float32x4_t c3 = vdupq_n_f32(0.0f);

    for (int k = 0; k < K; ++k) {
        const float32x4_t a = vld1q_f32(A);
        const float32x4_t b = vld1q_f32(B);

        c0 = vfmaq_laneq_f32(c0, b, a, 0);
        c1 = vfmaq_laneq_f32(c1, b, a, 1);
        c2 = vfmaq_laneq_f32(c2, b, a, 2);
        c3 = vfmaq_laneq_f32(c3, b, a, 3);

        A += 4;
        B += ldb;
    }

    const float32x4_t vmin = vdupq_n_f32(clamp_min);
    const float32x4_t vmax = vdupq_n_f32(clamp_max);

    c0 = vaddq_f32(c0, vld1q_f32(C + 0 * ldc));
    c1 = vaddq_f32(c1, vld1q_f32(C + 1 * ldc));
    c2 = vaddq_f32(c2, vld1q_f32(C + 2 * ldc));
    c3 = vaddq_f32(c3, vld1q_f32(C + 3 * ldc));

    vst1q_f32(C + 0 * ldc, vminq_f32(vmaxq_f32(c0, vmin), vmax));
    vst1q_f32(C + 1 * ldc, vminq_f32(vmaxq_f32(c1, vmin), vmax));
    vst1q_f32(C + 2 * ldc, vminq_f32(vmaxq_f32(c2, vmin), vmax));
    vst1q_f32(C + 3 * ldc, vminq_f32(vmaxq_f32(c3, vmin), vmax));
}

inline void mma_pack_4x12_f32(float* NNOPS_RESTRICT C, int ldc,
                              const float* NNOPS_RESTRICT A,
                              const float* NNOPS_RESTRICT B,
                              int ldb, int K,
                              float clamp_min, float clamp_max) noexcept {
    float32x4_t c00 = vdupq_n_f32(0.0f), c01 = c00, c02 = c00;
    float32x4_t c10 = vdupq_n_f32(0.0f), c11 = c10, c12 = c10;
    float32x4_t c20 = vdupq_n_f32(0.0f), c21 = c20, c22 = c20;
    float32x4_t c30 = vdupq_n_f32(0.0f), c31 = c30, c32 = c30;

    for (int k = 0; k < K; ++k) {
        const float32x4_t a = vld1q_f32(A);
        const float32x4_t b0 = vld1q_f32(B + 0 * 4);
        const float32x4_t b1 = vld1q_f32(B + 1 * 4);
        const float32x4_t b2 = vld1q_f32(B + 2 * 4);

        c00 = vfmaq_laneq_f32(c00, b0, a, 0);
        c01 = vfmaq_laneq_f32(c01, b1, a, 0);
        c02 = vfmaq_laneq_f32(c02, b2, a, 0);

        c10 = vfmaq_laneq_f32(c10, b0, a, 1);
        c11 = vfmaq_laneq_f32(c11, b1, a, 1);
        c12 = vfmaq_laneq_f32(c12, b2, a, 1);

        c20 = vfmaq_laneq_f32(c20, b0, a, 2);
        c21 = vfmaq_laneq_f32(c21, b1, a, 2);
        c22 = vfmaq_laneq_f32(c22, b2, a, 2);

        c30 = vfmaq_laneq_f32(c30, b0, a, 3);
        c31 = vfmaq_laneq_f32(c31, b1, a, 3);
        c32 = vfmaq_laneq_f32(c32, b2, a, 3);

        A += 4;
        B += ldb;
    }

    const float32x4_t vmin = vdupq_n_f32(clamp_min);
    const float32x4_t vmax = vdupq_n_f32(clamp_max);

    auto clamp = [&](float32x4_t& v) { v = vminq_f32(vmaxq_f32(v, vmin), vmax); };

    c00 = vaddq_f32(c00, vld1q_f32(C + 0 * 4)); c01 = vaddq_f32(c01, vld1q_f32(C + 1 * 4)); c02 = vaddq_f32(c02, vld1q_f32(C + 2 * 4));
    c10 = vaddq_f32(c10, vld1q_f32(C + 1 * ldc + 0 * 4)); c11 = vaddq_f32(c11, vld1q_f32(C + 1 * ldc + 1 * 4)); c12 = vaddq_f32(c12, vld1q_f32(C + 1 * ldc + 2 * 4));
    c20 = vaddq_f32(c20, vld1q_f32(C + 2 * ldc + 0 * 4)); c21 = vaddq_f32(c21, vld1q_f32(C + 2 * ldc + 1 * 4)); c22 = vaddq_f32(c22, vld1q_f32(C + 2 * ldc + 2 * 4));
    c30 = vaddq_f32(c30, vld1q_f32(C + 3 * ldc + 0 * 4)); c31 = vaddq_f32(c31, vld1q_f32(C + 3 * ldc + 1 * 4)); c32 = vaddq_f32(c32, vld1q_f32(C + 3 * ldc + 2 * 4));

    clamp(c00); clamp(c01); clamp(c02);
    clamp(c10); clamp(c11); clamp(c12);
    clamp(c20); clamp(c21); clamp(c22);
    clamp(c30); clamp(c31); clamp(c32);

    vst1q_f32(C + 0 * 4, c00); vst1q_f32(C + 1 * 4, c01); vst1q_f32(C + 2 * 4, c02);
    vst1q_f32(C + 1 * ldc + 0 * 4, c10); vst1q_f32(C + 1 * ldc + 1 * 4, c11); vst1q_f32(C + 1 * ldc + 2 * 4, c12);
    vst1q_f32(C + 2 * ldc + 0 * 4, c20); vst1q_f32(C + 2 * ldc + 1 * 4, c21); vst1q_f32(C + 2 * ldc + 2 * 4, c22);
    vst1q_f32(C + 3 * ldc + 0 * 4, c30); vst1q_f32(C + 3 * ldc + 1 * 4, c31); vst1q_f32(C + 3 * ldc + 2 * 4, c32);
}

// =========================================================================
//  mr=8  kernels
// =========================================================================

inline void mma_pack_8x1_f32(float* NNOPS_RESTRICT C, int ldc,
                             const float* NNOPS_RESTRICT A,
                             const float* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    float c[8] = {};
    for (int k = 0; k < K; ++k) {
        const float b = B[0];
        for (int i = 0; i < 8; ++i) { c[i] += A[i] * b; }
        A += 8;
        B += ldb;
    }
    for (int i = 0; i < 8; ++i) {
        c[i] = std::min(std::max(c[i] + C[i * ldc], clamp_min), clamp_max);
        C[i * ldc] = c[i];
    }
}

inline void mma_pack_8x4_f32(float* NNOPS_RESTRICT C, int ldc,
                             const float* NNOPS_RESTRICT A,
                             const float* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    float32x4_t c0 = vdupq_n_f32(0.0f);
    float32x4_t c1 = vdupq_n_f32(0.0f);
    float32x4_t c2 = vdupq_n_f32(0.0f);
    float32x4_t c3 = vdupq_n_f32(0.0f);
    float32x4_t c4 = vdupq_n_f32(0.0f);
    float32x4_t c5 = vdupq_n_f32(0.0f);
    float32x4_t c6 = vdupq_n_f32(0.0f);
    float32x4_t c7 = vdupq_n_f32(0.0f);

    for (int k = 0; k < K; ++k) {
        const float32x4_t a0 = vld1q_f32(A + 0 * 4);
        const float32x4_t b  = vld1q_f32(B);

        c0 = vfmaq_laneq_f32(c0, b, a0, 0);
        c1 = vfmaq_laneq_f32(c1, b, a0, 1);
        c2 = vfmaq_laneq_f32(c2, b, a0, 2);
        c3 = vfmaq_laneq_f32(c3, b, a0, 3);

        const float32x4_t a1 = vld1q_f32(A + 1 * 4);

        c4 = vfmaq_laneq_f32(c4, b, a1, 0);
        c5 = vfmaq_laneq_f32(c5, b, a1, 1);
        c6 = vfmaq_laneq_f32(c6, b, a1, 2);
        c7 = vfmaq_laneq_f32(c7, b, a1, 3);

        A += 8;
        B += ldb;
    }

    const float32x4_t vmin = vdupq_n_f32(clamp_min);
    const float32x4_t vmax = vdupq_n_f32(clamp_max);

    c0 = vaddq_f32(c0, vld1q_f32(C + 0 * ldc));
    c1 = vaddq_f32(c1, vld1q_f32(C + 1 * ldc));
    c2 = vaddq_f32(c2, vld1q_f32(C + 2 * ldc));
    c3 = vaddq_f32(c3, vld1q_f32(C + 3 * ldc));
    c4 = vaddq_f32(c4, vld1q_f32(C + 4 * ldc));
    c5 = vaddq_f32(c5, vld1q_f32(C + 5 * ldc));
    c6 = vaddq_f32(c6, vld1q_f32(C + 6 * ldc));
    c7 = vaddq_f32(c7, vld1q_f32(C + 7 * ldc));

    vst1q_f32(C + 0 * ldc, vminq_f32(vmaxq_f32(c0, vmin), vmax));
    vst1q_f32(C + 1 * ldc, vminq_f32(vmaxq_f32(c1, vmin), vmax));
    vst1q_f32(C + 2 * ldc, vminq_f32(vmaxq_f32(c2, vmin), vmax));
    vst1q_f32(C + 3 * ldc, vminq_f32(vmaxq_f32(c3, vmin), vmax));
    vst1q_f32(C + 4 * ldc, vminq_f32(vmaxq_f32(c4, vmin), vmax));
    vst1q_f32(C + 5 * ldc, vminq_f32(vmaxq_f32(c5, vmin), vmax));
    vst1q_f32(C + 6 * ldc, vminq_f32(vmaxq_f32(c6, vmin), vmax));
    vst1q_f32(C + 7 * ldc, vminq_f32(vmaxq_f32(c7, vmin), vmax));
}

inline void mma_pack_8x12_f32(float* NNOPS_RESTRICT C, int ldc,
                              const float* NNOPS_RESTRICT A,
                              const float* NNOPS_RESTRICT B,
                              int ldb, int K,
                              float clamp_min, float clamp_max) noexcept {
    float32x4_t c[8][3];
    for (int i = 0; i < 8; ++i) {
        c[i][0] = vdupq_n_f32(0.0f);
        c[i][1] = vdupq_n_f32(0.0f);
        c[i][2] = vdupq_n_f32(0.0f);
    }

    for (int k = 0; k < K; ++k) {
        const float32x4_t a0 = vld1q_f32(A + 0 * 4);
        const float32x4_t b0 = vld1q_f32(B + 0 * 4);
        const float32x4_t b1 = vld1q_f32(B + 1 * 4);
        const float32x4_t b2 = vld1q_f32(B + 2 * 4);

        for (int i = 0; i < 4; ++i) {
            c[i][0] = vfmaq_laneq_f32(c[i][0], b0, a0, i);
            c[i][1] = vfmaq_laneq_f32(c[i][1], b1, a0, i);
            c[i][2] = vfmaq_laneq_f32(c[i][2], b2, a0, i);
        }

        const float32x4_t a1 = vld1q_f32(A + 1 * 4);
        for (int i = 0; i < 4; ++i) {
            c[4 + i][0] = vfmaq_laneq_f32(c[4 + i][0], b0, a1, i);
            c[4 + i][1] = vfmaq_laneq_f32(c[4 + i][1], b1, a1, i);
            c[4 + i][2] = vfmaq_laneq_f32(c[4 + i][2], b2, a1, i);
        }

        A += 8;
        B += ldb;
    }

    const float32x4_t vmin = vdupq_n_f32(clamp_min);
    const float32x4_t vmax = vdupq_n_f32(clamp_max);

    for (int i = 0; i < 8; ++i) {
        c[i][0] = vaddq_f32(c[i][0], vld1q_f32(C + i * ldc + 0 * 4));
        c[i][1] = vaddq_f32(c[i][1], vld1q_f32(C + i * ldc + 1 * 4));
        c[i][2] = vaddq_f32(c[i][2], vld1q_f32(C + i * ldc + 2 * 4));

        vst1q_f32(C + i * ldc + 0 * 4, vminq_f32(vmaxq_f32(c[i][0], vmin), vmax));
        vst1q_f32(C + i * ldc + 1 * 4, vminq_f32(vmaxq_f32(c[i][1], vmin), vmax));
        vst1q_f32(C + i * ldc + 2 * 4, vminq_f32(vmaxq_f32(c[i][2], vmin), vmax));
    }
}

}  // namespace nnops::backend::cpu::aarch64
