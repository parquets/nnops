#pragma once
/// @file mma_pack_f16.hpp
/// @brief AArch64 NEON float16 MMA (matrix micro-accumulate) packed-B kernels.
///
/// These use native NEON fp16 arithmetic (ARMv8.2-A+).
/// Tile sizes (AArch64-optimised):
///   M ∈ {8, 4, 1}    N ∈ {24, 8, 1}
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
                             float16_t clamp_min, float16_t clamp_max) noexcept {
    float16_t c0 = 0.0f16;
    for (int k = 0; k < K; ++k) {
        c0 = vaddh_f16(c0, vmulh_f16(A[0], B[0]));
        A += 1;
        B += ldb;
    }
    c0 = vaddh_f16(c0, C[0 * ldc]);
    C[0 * ldc] = std::min(std::max(c0, clamp_min), clamp_max);
}

inline void mma_pack_1x8_f16(float16_t* NNOPS_RESTRICT C, int ldc,
                             const float16_t* NNOPS_RESTRICT A,
                             const float16_t* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float16_t clamp_min, float16_t clamp_max) noexcept {
    float16x8_t c0 = vdupq_n_f16(0.0f16);
    for (int k = 0; k < K; ++k) {
        c0 = vfmaq_n_f16(c0, vld1q_f16(B), A[0]);
        A += 1;
        B += ldb;
    }
    c0 = vaddq_f16(c0, vld1q_f16(C));
    const float16x8_t vmin = vdupq_n_f16(clamp_min);
    const float16x8_t vmax = vdupq_n_f16(clamp_max);
    vst1q_f16(C, vminq_f16(vmaxq_f16(c0, vmin), vmax));
}

inline void mma_pack_1x24_f16(float16_t* NNOPS_RESTRICT C, int ldc,
                              const float16_t* NNOPS_RESTRICT A,
                              const float16_t* NNOPS_RESTRICT B,
                              int ldb, int K,
                              float16_t clamp_min, float16_t clamp_max) noexcept {
    float16x8_t c0 = vdupq_n_f16(0.0f16);
    float16x8_t c1 = vdupq_n_f16(0.0f16);
    float16x8_t c2 = vdupq_n_f16(0.0f16);

    for (int k = 0; k < K; ++k) {
        const float16_t a = A[0];
        c0 = vfmaq_n_f16(c0, vld1q_f16(B + 0 * 8), a);
        c1 = vfmaq_n_f16(c1, vld1q_f16(B + 1 * 8), a);
        c2 = vfmaq_n_f16(c2, vld1q_f16(B + 2 * 8), a);
        A += 1;
        B += ldb;
    }

    const float16x8_t vmin = vdupq_n_f16(clamp_min);
    const float16x8_t vmax = vdupq_n_f16(clamp_max);

    c0 = vaddq_f16(c0, vld1q_f16(C + 0 * 8));
    c1 = vaddq_f16(c1, vld1q_f16(C + 1 * 8));
    c2 = vaddq_f16(c2, vld1q_f16(C + 2 * 8));

    vst1q_f16(C + 0 * 8, vminq_f16(vmaxq_f16(c0, vmin), vmax));
    vst1q_f16(C + 1 * 8, vminq_f16(vmaxq_f16(c1, vmin), vmax));
    vst1q_f16(C + 2 * 8, vminq_f16(vmaxq_f16(c2, vmin), vmax));
}

// =========================================================================
//  mr=4  kernels
// =========================================================================

inline void mma_pack_4x1_f16(float16_t* NNOPS_RESTRICT C, int ldc,
                             const float16_t* NNOPS_RESTRICT A,
                             const float16_t* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float16_t clamp_min, float16_t clamp_max) noexcept {
    float16_t c0 = 0.0f16, c1 = 0.0f16, c2 = 0.0f16, c3 = 0.0f16;
    for (int k = 0; k < K; ++k) {
        const float16_t b = B[0];
        c0 = vaddh_f16(c0, vmulh_f16(A[0], b));
        c1 = vaddh_f16(c1, vmulh_f16(A[1], b));
        c2 = vaddh_f16(c2, vmulh_f16(A[2], b));
        c3 = vaddh_f16(c3, vmulh_f16(A[3], b));
        A += 4;
        B += ldb;
    }
    c0 = std::min(std::max(vaddh_f16(c0, C[0 * ldc]), clamp_min), clamp_max); C[0 * ldc] = c0;
    c1 = std::min(std::max(vaddh_f16(c1, C[1 * ldc]), clamp_min), clamp_max); C[1 * ldc] = c1;
    c2 = std::min(std::max(vaddh_f16(c2, C[2 * ldc]), clamp_min), clamp_max); C[2 * ldc] = c2;
    c3 = std::min(std::max(vaddh_f16(c3, C[3 * ldc]), clamp_min), clamp_max); C[3 * ldc] = c3;
}

inline void mma_pack_4x8_f16(float16_t* NNOPS_RESTRICT C, int ldc,
                             const float16_t* NNOPS_RESTRICT A,
                             const float16_t* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float16_t clamp_min, float16_t clamp_max) noexcept {
    float16x8_t c0 = vdupq_n_f16(0.0f16);
    float16x8_t c1 = vdupq_n_f16(0.0f16);
    float16x8_t c2 = vdupq_n_f16(0.0f16);
    float16x8_t c3 = vdupq_n_f16(0.0f16);

    for (int k = 0; k < K; ++k) {
        const float16x4_t a = vld1_f16(A);
        const float16x8_t b = vld1q_f16(B);

        c0 = vfmaq_lane_f16(c0, b, a, 0);
        c1 = vfmaq_lane_f16(c1, b, a, 1);
        c2 = vfmaq_lane_f16(c2, b, a, 2);
        c3 = vfmaq_lane_f16(c3, b, a, 3);

        A += 4;
        B += ldb;
    }

    const float16x8_t vmin = vdupq_n_f16(clamp_min);
    const float16x8_t vmax = vdupq_n_f16(clamp_max);

    c0 = vaddq_f16(c0, vld1q_f16(C + 0 * ldc));
    c1 = vaddq_f16(c1, vld1q_f16(C + 1 * ldc));
    c2 = vaddq_f16(c2, vld1q_f16(C + 2 * ldc));
    c3 = vaddq_f16(c3, vld1q_f16(C + 3 * ldc));

    vst1q_f16(C + 0 * ldc, vminq_f16(vmaxq_f16(c0, vmin), vmax));
    vst1q_f16(C + 1 * ldc, vminq_f16(vmaxq_f16(c1, vmin), vmax));
    vst1q_f16(C + 2 * ldc, vminq_f16(vmaxq_f16(c2, vmin), vmax));
    vst1q_f16(C + 3 * ldc, vminq_f16(vmaxq_f16(c3, vmin), vmax));
}

inline void mma_pack_4x24_f16(float16_t* NNOPS_RESTRICT C, int ldc,
                              const float16_t* NNOPS_RESTRICT A,
                              const float16_t* NNOPS_RESTRICT B,
                              int ldb, int K,
                              float16_t clamp_min, float16_t clamp_max) noexcept {
    float16x8_t c[4][3];
    for (int i = 0; i < 4; ++i) {
        c[i][0] = vdupq_n_f16(0.0f16);
        c[i][1] = vdupq_n_f16(0.0f16);
        c[i][2] = vdupq_n_f16(0.0f16);
    }

    for (int k = 0; k < K; ++k) {
        const float16x4_t a = vld1_f16(A);
        const float16x8_t b0 = vld1q_f16(B + 0 * 8);
        const float16x8_t b1 = vld1q_f16(B + 1 * 8);
        const float16x8_t b2 = vld1q_f16(B + 2 * 8);

        for (int i = 0; i < 4; ++i) {
            c[i][0] = vfmaq_lane_f16(c[i][0], b0, a, i);
            c[i][1] = vfmaq_lane_f16(c[i][1], b1, a, i);
            c[i][2] = vfmaq_lane_f16(c[i][2], b2, a, i);
        }

        A += 4;
        B += ldb;
    }

    const float16x8_t vmin = vdupq_n_f16(clamp_min);
    const float16x8_t vmax = vdupq_n_f16(clamp_max);

    for (int i = 0; i < 4; ++i) {
        c[i][0] = vaddq_f16(c[i][0], vld1q_f16(C + i * ldc + 0 * 8));
        c[i][1] = vaddq_f16(c[i][1], vld1q_f16(C + i * ldc + 1 * 8));
        c[i][2] = vaddq_f16(c[i][2], vld1q_f16(C + i * ldc + 2 * 8));

        vst1q_f16(C + i * ldc + 0 * 8, vminq_f16(vmaxq_f16(c[i][0], vmin), vmax));
        vst1q_f16(C + i * ldc + 1 * 8, vminq_f16(vmaxq_f16(c[i][1], vmin), vmax));
        vst1q_f16(C + i * ldc + 2 * 8, vminq_f16(vmaxq_f16(c[i][2], vmin), vmax));
    }
}

// =========================================================================
//  mr=8  kernels
// =========================================================================

inline void mma_pack_8x1_f16(float16_t* NNOPS_RESTRICT C, int ldc,
                             const float16_t* NNOPS_RESTRICT A,
                             const float16_t* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float16_t clamp_min, float16_t clamp_max) noexcept {
    float16_t c[8] = {};
    for (int k = 0; k < K; ++k) {
        const float16_t b = B[0];
        for (int i = 0; i < 8; ++i) {
            c[i] = vaddh_f16(c[i], vmulh_f16(A[i], b));
        }
        A += 8;
        B += ldb;
    }
    for (int i = 0; i < 8; ++i) {
        c[i] = std::min(std::max(vaddh_f16(c[i], C[i * ldc]), clamp_min), clamp_max);
        C[i * ldc] = c[i];
    }
}

inline void mma_pack_8x8_f16(float16_t* NNOPS_RESTRICT C, int ldc,
                             const float16_t* NNOPS_RESTRICT A,
                             const float16_t* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float16_t clamp_min, float16_t clamp_max) noexcept {
    float16x8_t c[8];
    for (int i = 0; i < 8; ++i) { c[i] = vdupq_n_f16(0.0f16); }

    for (int k = 0; k < K; ++k) {
        const float16x8_t a = vld1q_f16(A);
        const float16x8_t b = vld1q_f16(B);

        for (int i = 0; i < 8; ++i) {
            c[i] = vfmaq_laneq_f16(c[i], b, a, i);
        }

        A += 8;
        B += ldb;
    }

    const float16x8_t vmin = vdupq_n_f16(clamp_min);
    const float16x8_t vmax = vdupq_n_f16(clamp_max);

    for (int i = 0; i < 8; ++i) {
        c[i] = vaddq_f16(c[i], vld1q_f16(C + i * ldc));
        vst1q_f16(C + i * ldc, vminq_f16(vmaxq_f16(c[i], vmin), vmax));
    }
}

inline void mma_pack_8x24_f16(float16_t* NNOPS_RESTRICT C, int ldc,
                              const float16_t* NNOPS_RESTRICT A,
                              const float16_t* NNOPS_RESTRICT B,
                              int ldb, int K,
                              float16_t clamp_min, float16_t clamp_max) noexcept {
    float16x8_t c[8][3];
    for (int i = 0; i < 8; ++i) {
        c[i][0] = vdupq_n_f16(0.0f16);
        c[i][1] = vdupq_n_f16(0.0f16);
        c[i][2] = vdupq_n_f16(0.0f16);
    }

    for (int k = 0; k < K; ++k) {
        const float16x8_t a = vld1q_f16(A);
        const float16x8_t b0 = vld1q_f16(B + 0 * 8);
        const float16x8_t b1 = vld1q_f16(B + 1 * 8);
        const float16x8_t b2 = vld1q_f16(B + 2 * 8);

        for (int i = 0; i < 8; ++i) {
            c[i][0] = vfmaq_laneq_f16(c[i][0], b0, a, i);
            c[i][1] = vfmaq_laneq_f16(c[i][1], b1, a, i);
            c[i][2] = vfmaq_laneq_f16(c[i][2], b2, a, i);
        }

        A += 8;
        B += ldb;
    }

    const float16x8_t vmin = vdupq_n_f16(clamp_min);
    const float16x8_t vmax = vdupq_n_f16(clamp_max);

    for (int i = 0; i < 8; ++i) {
        c[i][0] = vaddq_f16(c[i][0], vld1q_f16(C + i * ldc + 0 * 8));
        c[i][1] = vaddq_f16(c[i][1], vld1q_f16(C + i * ldc + 1 * 8));
        c[i][2] = vaddq_f16(c[i][2], vld1q_f16(C + i * ldc + 2 * 8));

        vst1q_f16(C + i * ldc + 0 * 8, vminq_f16(vmaxq_f16(c[i][0], vmin), vmax));
        vst1q_f16(C + i * ldc + 1 * 8, vminq_f16(vmaxq_f16(c[i][1], vmin), vmax));
        vst1q_f16(C + i * ldc + 2 * 8, vminq_f16(vmaxq_f16(c[i][2], vmin), vmax));
    }
}

}  // namespace nnops::backend::cpu::aarch64
