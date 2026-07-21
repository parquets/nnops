#pragma once
/// @file mma_direct_f16.hpp
/// @brief AArch64 NEON float16 MMA direct (unpacked) micro-kernels.
///
/// These kernels compute C[mr][nr] += A[mr][K] × B[K][nr] directly
/// from row-major A (stride lda) and B (stride ldb), without packing.
/// Uses native NEON fp16 arithmetic (ARMv8.2-A+ required).
///
/// Tile sizes (AArch64-optimised):
///   M ∈ {8, 4, 1}    N ∈ {24, 8, 1}
///
/// K is unrolled by 4: four f16 A values are loaded as a float16x4_t
/// and broadcast lane-by-lane with vfmaq_lane_f16. A scalar tail
/// handles remaining K.
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
    int K, float16_t clamp_min, float16_t clamp_max) noexcept {

    const float16_t* NNOPS_RESTRICT A_ptr0 = A;

    float16_t c0 = 0.0f16;

    for (int k = 0; k < K; ++k) {
        c0 = vaddh_f16(c0, vmulh_f16(A_ptr0[0], B[0]));
        A_ptr0 += 1;
        B += ldb;
    }
    c0 = vaddh_f16(c0, C[0]);
    C[0] = std::min(std::max(c0, clamp_min), clamp_max);
}

inline void mma_direct_1x8_f16(
    float16_t* NNOPS_RESTRICT C, int ldc,
    const float16_t* NNOPS_RESTRICT A, int lda,
    const float16_t* NNOPS_RESTRICT B, int ldb,
    int K, float16_t clamp_min, float16_t clamp_max) noexcept {

    const float16_t* NNOPS_RESTRICT A_ptr0 = A;

    float16x8_t v_c00 = vdupq_n_f16(0.0f16);

    int k = 0;
    for (; k < K - 3; k += 4) {
        const float16x4_t v_a0 = vld1_f16(A_ptr0);

        float16x8_t v_b0 = vld1q_f16(B + 0 * ldb);
        v_c00 = vfmaq_lane_f16(v_c00, v_b0, v_a0, 0);
        v_b0 = vld1q_f16(B + 1 * ldb);
        v_c00 = vfmaq_lane_f16(v_c00, v_b0, v_a0, 1);
        v_b0 = vld1q_f16(B + 2 * ldb);
        v_c00 = vfmaq_lane_f16(v_c00, v_b0, v_a0, 2);
        v_b0 = vld1q_f16(B + 3 * ldb);
        v_c00 = vfmaq_lane_f16(v_c00, v_b0, v_a0, 3);

        A_ptr0 += 4;
        B += 4 * ldb;
    }
    for (; k < K; ++k) {
        const float16x8_t v_b0 = vld1q_f16(B);
        v_c00 = vfmaq_n_f16(v_c00, v_b0, A_ptr0[0]);
        A_ptr0 += 1;
        B += ldb;
    }

    v_c00 = vaddq_f16(v_c00, vld1q_f16(C));

    const float16x8_t v_min = vdupq_n_f16(clamp_min);
    const float16x8_t v_max = vdupq_n_f16(clamp_max);
    vst1q_f16(C, vminq_f16(vmaxq_f16(v_c00, v_min), v_max));
}

inline void mma_direct_1x24_f16(
    float16_t* NNOPS_RESTRICT C, int ldc,
    const float16_t* NNOPS_RESTRICT A, int lda,
    const float16_t* NNOPS_RESTRICT B, int ldb,
    int K, float16_t clamp_min, float16_t clamp_max) noexcept {

    const float16_t* NNOPS_RESTRICT A_ptr0 = A;

    float16x8_t v_c00 = vdupq_n_f16(0.0f16);
    float16x8_t v_c01 = v_c00;
    float16x8_t v_c02 = v_c00;

    int k = 0;
    for (; k < K - 3; k += 4) {
        const float16x4_t v_a0 = vld1_f16(A_ptr0);

        float16x8_t v_b0 = vld1q_f16(B + 0 * ldb + 0 * 8);
        float16x8_t v_b1 = vld1q_f16(B + 0 * ldb + 1 * 8);
        float16x8_t v_b2 = vld1q_f16(B + 0 * ldb + 2 * 8);

        v_c00 = vfmaq_lane_f16(v_c00, v_b0, v_a0, 0);
        v_c01 = vfmaq_lane_f16(v_c01, v_b1, v_a0, 0);
        v_c02 = vfmaq_lane_f16(v_c02, v_b2, v_a0, 0);

        v_b0 = vld1q_f16(B + 1 * ldb + 0 * 8);
        v_b1 = vld1q_f16(B + 1 * ldb + 1 * 8);
        v_b2 = vld1q_f16(B + 1 * ldb + 2 * 8);

        v_c00 = vfmaq_lane_f16(v_c00, v_b0, v_a0, 1);
        v_c01 = vfmaq_lane_f16(v_c01, v_b1, v_a0, 1);
        v_c02 = vfmaq_lane_f16(v_c02, v_b2, v_a0, 1);

        v_b0 = vld1q_f16(B + 2 * ldb + 0 * 8);
        v_b1 = vld1q_f16(B + 2 * ldb + 1 * 8);
        v_b2 = vld1q_f16(B + 2 * ldb + 2 * 8);

        v_c00 = vfmaq_lane_f16(v_c00, v_b0, v_a0, 2);
        v_c01 = vfmaq_lane_f16(v_c01, v_b1, v_a0, 2);
        v_c02 = vfmaq_lane_f16(v_c02, v_b2, v_a0, 2);

        v_b0 = vld1q_f16(B + 3 * ldb + 0 * 8);
        v_b1 = vld1q_f16(B + 3 * ldb + 1 * 8);
        v_b2 = vld1q_f16(B + 3 * ldb + 2 * 8);

        v_c00 = vfmaq_lane_f16(v_c00, v_b0, v_a0, 3);
        v_c01 = vfmaq_lane_f16(v_c01, v_b1, v_a0, 3);
        v_c02 = vfmaq_lane_f16(v_c02, v_b2, v_a0, 3);

        A_ptr0 += 4;
        B += 4 * ldb;
    }
    for (; k < K; ++k) {
        const float16x8_t v_b0 = vld1q_f16(B + 0 * 8);
        const float16x8_t v_b1 = vld1q_f16(B + 1 * 8);
        const float16x8_t v_b2 = vld1q_f16(B + 2 * 8);

        v_c00 = vfmaq_n_f16(v_c00, v_b0, A_ptr0[0]);
        v_c01 = vfmaq_n_f16(v_c01, v_b1, A_ptr0[0]);
        v_c02 = vfmaq_n_f16(v_c02, v_b2, A_ptr0[0]);

        A_ptr0 += 1;
        B += ldb;
    }

    const float16x8_t v_min = vdupq_n_f16(clamp_min);
    const float16x8_t v_max = vdupq_n_f16(clamp_max);

    v_c00 = vaddq_f16(v_c00, vld1q_f16(C + 0 * 8));
    v_c01 = vaddq_f16(v_c01, vld1q_f16(C + 1 * 8));
    v_c02 = vaddq_f16(v_c02, vld1q_f16(C + 2 * 8));

    vst1q_f16(C + 0 * 8, vminq_f16(vmaxq_f16(v_c00, v_min), v_max));
    vst1q_f16(C + 1 * 8, vminq_f16(vmaxq_f16(v_c01, v_min), v_max));
    vst1q_f16(C + 2 * 8, vminq_f16(vmaxq_f16(v_c02, v_min), v_max));
}

// =========================================================================
//  mr=4  kernels
// =========================================================================

inline void mma_direct_4x1_f16(
    float16_t* NNOPS_RESTRICT C, int ldc,
    const float16_t* NNOPS_RESTRICT A, int lda,
    const float16_t* NNOPS_RESTRICT B, int ldb,
    int K, float16_t clamp_min, float16_t clamp_max) noexcept {

    const float16_t* NNOPS_RESTRICT A_ptr0 = A + 0 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr1 = A + 1 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr2 = A + 2 * lda;
    const float16_t* NNOPS_RESTRICT A_ptr3 = A + 3 * lda;

    float16_t c0 = 0.0f16, c1 = 0.0f16, c2 = 0.0f16, c3 = 0.0f16;

    for (int k = 0; k < K; ++k) {
        const float16_t b0 = B[0];
        c0 = vaddh_f16(c0, vmulh_f16(A_ptr0[0], b0));
        c1 = vaddh_f16(c1, vmulh_f16(A_ptr1[0], b0));
        c2 = vaddh_f16(c2, vmulh_f16(A_ptr2[0], b0));
        c3 = vaddh_f16(c3, vmulh_f16(A_ptr3[0], b0));

        A_ptr0 += 1; A_ptr1 += 1; A_ptr2 += 1; A_ptr3 += 1;
        B += ldb;
    }

    auto write = [&](float16_t* dst, float16_t acc) {
        float16_t v = vaddh_f16(*dst, acc);
        *dst = std::min(std::max(v, clamp_min), clamp_max);
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
    int K, float16_t clamp_min, float16_t clamp_max) noexcept {

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
        const float16x4_t v_a0 = vld1_f16(A_ptr0);
        const float16x4_t v_a1 = vld1_f16(A_ptr1);
        const float16x4_t v_a2 = vld1_f16(A_ptr2);
        const float16x4_t v_a3 = vld1_f16(A_ptr3);

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

        A_ptr0 += 4; A_ptr1 += 4; A_ptr2 += 4; A_ptr3 += 4;
        B += 4 * ldb;
    }
    for (; k < K; ++k) {
        const float16x8_t v_b0 = vld1q_f16(B);

        v_c00 = vfmaq_n_f16(v_c00, v_b0, A_ptr0[0]);
        v_c10 = vfmaq_n_f16(v_c10, v_b0, A_ptr1[0]);
        v_c20 = vfmaq_n_f16(v_c20, v_b0, A_ptr2[0]);
        v_c30 = vfmaq_n_f16(v_c30, v_b0, A_ptr3[0]);

        A_ptr0 += 1; A_ptr1 += 1; A_ptr2 += 1; A_ptr3 += 1;
        B += ldb;
    }

    const float16x8_t v_min = vdupq_n_f16(clamp_min);
    const float16x8_t v_max = vdupq_n_f16(clamp_max);

    v_c00 = vaddq_f16(v_c00, vld1q_f16(C + 0 * ldc));
    v_c10 = vaddq_f16(v_c10, vld1q_f16(C + 1 * ldc));
    v_c20 = vaddq_f16(v_c20, vld1q_f16(C + 2 * ldc));
    v_c30 = vaddq_f16(v_c30, vld1q_f16(C + 3 * ldc));

    vst1q_f16(C + 0 * ldc, vminq_f16(vmaxq_f16(v_c00, v_min), v_max));
    vst1q_f16(C + 1 * ldc, vminq_f16(vmaxq_f16(v_c10, v_min), v_max));
    vst1q_f16(C + 2 * ldc, vminq_f16(vmaxq_f16(v_c20, v_min), v_max));
    vst1q_f16(C + 3 * ldc, vminq_f16(vmaxq_f16(v_c30, v_min), v_max));
}

inline void mma_direct_4x24_f16(
    float16_t* NNOPS_RESTRICT C, int ldc,
    const float16_t* NNOPS_RESTRICT A, int lda,
    const float16_t* NNOPS_RESTRICT B, int ldb,
    int K, float16_t clamp_min, float16_t clamp_max) noexcept {

    const float16_t* NNOPS_RESTRICT A_ptrs[4];
    for (int r = 0; r < 4; ++r) {
        A_ptrs[r] = A + r * lda;
    }

    // 4 rows × 3 columns of float16x8_t = 12 accumulators
    float16x8_t v_c[4][3];
    for (int r = 0; r < 4; ++r) {
        v_c[r][0] = v_c[r][1] = v_c[r][2] = vdupq_n_f16(0.0f16);
    }

    int k = 0;
    for (; k < K - 3; k += 4) {
        float16x4_t v_a[4];
        for (int r = 0; r < 4; ++r) {
            v_a[r] = vld1_f16(A_ptrs[r]);
        }

        for (int kk = 0; kk < 4; ++kk) {
            const float16x8_t v_b0 = vld1q_f16(B + kk * ldb + 0 * 8);
            const float16x8_t v_b1 = vld1q_f16(B + kk * ldb + 1 * 8);
            const float16x8_t v_b2 = vld1q_f16(B + kk * ldb + 2 * 8);

            for (int r = 0; r < 4; ++r) {
                v_c[r][0] = vfmaq_lane_f16(v_c[r][0], v_b0, v_a[r], kk);
                v_c[r][1] = vfmaq_lane_f16(v_c[r][1], v_b1, v_a[r], kk);
                v_c[r][2] = vfmaq_lane_f16(v_c[r][2], v_b2, v_a[r], kk);
            }
        }

        for (int r = 0; r < 4; ++r) {
            A_ptrs[r] += 4;
        }
        B += 4 * ldb;
    }
    for (; k < K; ++k) {
        const float16x8_t v_b0 = vld1q_f16(B + 0 * 8);
        const float16x8_t v_b1 = vld1q_f16(B + 1 * 8);
        const float16x8_t v_b2 = vld1q_f16(B + 2 * 8);

        for (int r = 0; r < 4; ++r) {
            v_c[r][0] = vfmaq_n_f16(v_c[r][0], v_b0, A_ptrs[r][0]);
            v_c[r][1] = vfmaq_n_f16(v_c[r][1], v_b1, A_ptrs[r][0]);
            v_c[r][2] = vfmaq_n_f16(v_c[r][2], v_b2, A_ptrs[r][0]);
            A_ptrs[r] += 1;
        }
        B += ldb;
    }

    const float16x8_t v_min = vdupq_n_f16(clamp_min);
    const float16x8_t v_max = vdupq_n_f16(clamp_max);

    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 3; ++c) {
            v_c[r][c] = vaddq_f16(v_c[r][c], vld1q_f16(C + r * ldc + c * 8));
            vst1q_f16(C + r * ldc + c * 8, vminq_f16(vmaxq_f16(v_c[r][c], v_min), v_max));
        }
    }
}

// =========================================================================
//  mr=8  kernels
// =========================================================================

inline void mma_direct_8x1_f16(
    float16_t* NNOPS_RESTRICT C, int ldc,
    const float16_t* NNOPS_RESTRICT A, int lda,
    const float16_t* NNOPS_RESTRICT B, int ldb,
    int K, float16_t clamp_min, float16_t clamp_max) noexcept {

    const float16_t* NNOPS_RESTRICT A_ptrs[8];
    for (int r = 0; r < 8; ++r) {
        A_ptrs[r] = A + r * lda;
    }

    float16_t c[8] = { 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16 };

    for (int k = 0; k < K; ++k) {
        const float16_t b0 = B[0];
        for (int r = 0; r < 8; ++r) {
            c[r] = vaddh_f16(c[r], vmulh_f16(A_ptrs[r][0], b0));
            A_ptrs[r] += 1;
        }
        B += ldb;
    }

    for (int r = 0; r < 8; ++r) {
        float16_t v = vaddh_f16(C[r * ldc], c[r]);
        C[r * ldc] = std::min(std::max(v, clamp_min), clamp_max);
    }
}

inline void mma_direct_8x8_f16(
    float16_t* NNOPS_RESTRICT C, int ldc,
    const float16_t* NNOPS_RESTRICT A, int lda,
    const float16_t* NNOPS_RESTRICT B, int ldb,
    int K, float16_t clamp_min, float16_t clamp_max) noexcept {

    const float16_t* NNOPS_RESTRICT A_ptrs[8];
    for (int r = 0; r < 8; ++r) {
        A_ptrs[r] = A + r * lda;
    }

    float16x8_t v_c[8];
    for (int r = 0; r < 8; ++r) {
        v_c[r] = vdupq_n_f16(0.0f16);
    }

    int k = 0;
    for (; k < K - 3; k += 4) {
        float16x4_t v_a[8];
        for (int r = 0; r < 8; ++r) {
            v_a[r] = vld1_f16(A_ptrs[r]);
        }

        for (int kk = 0; kk < 4; ++kk) {
            const float16x8_t v_b0 = vld1q_f16(B + kk * ldb);
            for (int r = 0; r < 8; ++r) {
                v_c[r] = vfmaq_lane_f16(v_c[r], v_b0, v_a[r], kk);
            }
        }

        for (int r = 0; r < 8; ++r) {
            A_ptrs[r] += 4;
        }
        B += 4 * ldb;
    }
    for (; k < K; ++k) {
        const float16x8_t v_b0 = vld1q_f16(B);
        for (int r = 0; r < 8; ++r) {
            v_c[r] = vfmaq_n_f16(v_c[r], v_b0, A_ptrs[r][0]);
            A_ptrs[r] += 1;
        }
        B += ldb;
    }

    const float16x8_t v_min = vdupq_n_f16(clamp_min);
    const float16x8_t v_max = vdupq_n_f16(clamp_max);

    for (int r = 0; r < 8; ++r) {
        v_c[r] = vaddq_f16(v_c[r], vld1q_f16(C + r * ldc));
        vst1q_f16(C + r * ldc, vminq_f16(vmaxq_f16(v_c[r], v_min), v_max));
    }
}

inline void mma_direct_8x24_f16(
    float16_t* NNOPS_RESTRICT C, int ldc,
    const float16_t* NNOPS_RESTRICT A, int lda,
    const float16_t* NNOPS_RESTRICT B, int ldb,
    int K, float16_t clamp_min, float16_t clamp_max) noexcept {

    const float16_t* NNOPS_RESTRICT A_ptrs[8];
    for (int r = 0; r < 8; ++r) {
        A_ptrs[r] = A + r * lda;
    }

    float16x8_t v_c[8][3];
    for (int r = 0; r < 8; ++r) {
        v_c[r][0] = v_c[r][1] = v_c[r][2] = vdupq_n_f16(0.0f16);
    }

    int k = 0;
    for (; k < K - 3; k += 4) {
        float16x4_t v_a[8];
        for (int r = 0; r < 8; ++r) {
            v_a[r] = vld1_f16(A_ptrs[r]);
        }

        for (int kk = 0; kk < 4; ++kk) {
            const float16x8_t v_b0 = vld1q_f16(B + kk * ldb + 0 * 8);
            const float16x8_t v_b1 = vld1q_f16(B + kk * ldb + 1 * 8);
            const float16x8_t v_b2 = vld1q_f16(B + kk * ldb + 2 * 8);

            for (int r = 0; r < 8; ++r) {
                v_c[r][0] = vfmaq_lane_f16(v_c[r][0], v_b0, v_a[r], kk);
                v_c[r][1] = vfmaq_lane_f16(v_c[r][1], v_b1, v_a[r], kk);
                v_c[r][2] = vfmaq_lane_f16(v_c[r][2], v_b2, v_a[r], kk);
            }
        }

        for (int r = 0; r < 8; ++r) {
            A_ptrs[r] += 4;
        }
        B += 4 * ldb;
    }
    for (; k < K; ++k) {
        const float16x8_t v_b0 = vld1q_f16(B + 0 * 8);
        const float16x8_t v_b1 = vld1q_f16(B + 1 * 8);
        const float16x8_t v_b2 = vld1q_f16(B + 2 * 8);

        for (int r = 0; r < 8; ++r) {
            v_c[r][0] = vfmaq_n_f16(v_c[r][0], v_b0, A_ptrs[r][0]);
            v_c[r][1] = vfmaq_n_f16(v_c[r][1], v_b1, A_ptrs[r][0]);
            v_c[r][2] = vfmaq_n_f16(v_c[r][2], v_b2, A_ptrs[r][0]);
            A_ptrs[r] += 1;
        }
        B += ldb;
    }

    const float16x8_t v_min = vdupq_n_f16(clamp_min);
    const float16x8_t v_max = vdupq_n_f16(clamp_max);

    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 3; ++c) {
            v_c[r][c] = vaddq_f16(v_c[r][c], vld1q_f16(C + r * ldc + c * 8));
            vst1q_f16(C + r * ldc + c * 8, vminq_f16(vmaxq_f16(v_c[r][c], v_min), v_max));
        }
    }
}

}  // namespace nnops::backend::cpu::aarch64
