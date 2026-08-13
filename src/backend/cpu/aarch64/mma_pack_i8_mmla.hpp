#pragma once
/// @file mma_pack_i8_mmla.hpp
/// @brief AArch64 int8 → int32 MMA (matrix micro-accumulate) kernels (I8MM / MMLA).
///
/// Each kernel computes  C[mr][nr] += A[mr][K] × B_packed[K][nr]  using the
/// vmmlaq_s32 instruction (2×8 × 8×2 → 2×2 int32). A and B are pre-packed by the
/// caller; `K` is the number of 8-element int8 groups. The accumulate operands are
/// held in the transposed 2×2 layout that vmmlaq produces, then recombined into
/// row-major rows before clamping (where applicable) and storing.
///
/// Tile sizes (AArch64-optimised):
///   M ∈ {4, 2}    N ∈ {16, 8, 4}
///
/// Reference: nn_compute/src/cpu/kernel/mma/aarch64/mma_pack_i8_mmla.hpp

#include <arm_neon.h>
#include <cstdint>
#include "backend/cpu/common/restrict.hpp"

namespace nnops::backend::cpu::aarch64 {

inline void mmma_2x4_s8s8(int32_t* NNOPS_RESTRICT C, int ldc,
                          const int8_t* NNOPS_RESTRICT A,
                          const int8_t* NNOPS_RESTRICT B, int K) noexcept {
    int32x4_t v_c00_t = vdupq_n_s32(0);
    int32x4_t v_c01_t = vdupq_n_s32(0);

    for (int k = 0; k < K; ++k) {
        int8x16_t v_a0 = vld1q_s8(A);             // 2x8
        int8x16_t v_b0 = vld1q_s8(B + 0 * 16);    // 8x2
        int8x16_t v_b1 = vld1q_s8(B + 1 * 16);    // 8x2

        v_c00_t = vmmlaq_s32(v_c00_t, v_a0, v_b0);
        v_c01_t = vmmlaq_s32(v_c01_t, v_a0, v_b1);

        A += 16;
        B += 32;
    }

    int32x4_t v_c0 = vcombine_s32(vget_low_s32(v_c00_t), vget_low_s32(v_c01_t));
    int32x4_t v_c1 = vcombine_s32(vget_high_s32(v_c00_t), vget_high_s32(v_c01_t));

    v_c0 = vaddq_s32(v_c0, vld1q_s32(C + 0 * ldc));
    v_c1 = vaddq_s32(v_c1, vld1q_s32(C + 1 * ldc));

    vst1q_s32(C + 0 * ldc, v_c0);
    vst1q_s32(C + 1 * ldc, v_c1);
}

inline void mmma_4x4_s8s8(int32_t* NNOPS_RESTRICT C, int ldc,
                          const int8_t* NNOPS_RESTRICT A,
                          const int8_t* NNOPS_RESTRICT B, int K) noexcept {
    int32x4_t v_c0_t = vdupq_n_s32(0);
    int32x4_t v_c1_t = vdupq_n_s32(0);
    int32x4_t v_c2_t = vdupq_n_s32(0);
    int32x4_t v_c3_t = vdupq_n_s32(0);

    for (int k = 0; k < K; ++k) {
        int8x16_t v_a0 = vld1q_s8(A + 0 * 16);    // 2x8
        int8x16_t v_a1 = vld1q_s8(A + 1 * 16);    // 2x8
        int8x16_t v_b0 = vld1q_s8(B + 0 * 16);    // 8x2
        int8x16_t v_b1 = vld1q_s8(B + 1 * 16);    // 8x2

        v_c0_t = vmmlaq_s32(v_c0_t, v_a0, v_b0);
        v_c1_t = vmmlaq_s32(v_c1_t, v_a0, v_b1);
        v_c2_t = vmmlaq_s32(v_c2_t, v_a1, v_b0);
        v_c3_t = vmmlaq_s32(v_c3_t, v_a1, v_b1);

        A += 32;
        B += 32;
    }

    int32x4_t v_c0 = vcombine_s32(vget_low_s32(v_c0_t), vget_low_s32(v_c1_t));
    int32x4_t v_c1 = vcombine_s32(vget_high_s32(v_c0_t), vget_high_s32(v_c1_t));

    int32x4_t v_c2 = vcombine_s32(vget_low_s32(v_c2_t), vget_low_s32(v_c3_t));
    int32x4_t v_c3 = vcombine_s32(vget_high_s32(v_c2_t), vget_high_s32(v_c3_t));

    v_c0 = vaddq_s32(v_c0, vld1q_s32(C + 0 * ldc));
    v_c1 = vaddq_s32(v_c1, vld1q_s32(C + 1 * ldc));
    v_c2 = vaddq_s32(v_c2, vld1q_s32(C + 2 * ldc));
    v_c3 = vaddq_s32(v_c3, vld1q_s32(C + 3 * ldc));

    vst1q_s32(C + 0 * ldc, v_c0);
    vst1q_s32(C + 1 * ldc, v_c1);
    vst1q_s32(C + 2 * ldc, v_c2);
    vst1q_s32(C + 3 * ldc, v_c3);
}

inline void mmma_4x8_s8s8_mmla(int32_t* NNOPS_RESTRICT C, int ldc,
                               const int8_t* NNOPS_RESTRICT A,
                               const int8_t* NNOPS_RESTRICT B, int ldb, int K,
                               int32_t clamp_min, int32_t clamp_max) noexcept {
    int32x4_t v_c00 = vcombine_s32(vld1_s32(C + 0 * ldc + 0), vld1_s32(C + 1 * ldc + 0));
    int32x4_t v_c01 = vcombine_s32(vld1_s32(C + 0 * ldc + 2), vld1_s32(C + 1 * ldc + 2));
    int32x4_t v_c02 = vcombine_s32(vld1_s32(C + 0 * ldc + 4), vld1_s32(C + 1 * ldc + 4));
    int32x4_t v_c03 = vcombine_s32(vld1_s32(C + 0 * ldc + 6), vld1_s32(C + 1 * ldc + 6));

    int32x4_t v_c10 = vcombine_s32(vld1_s32(C + 2 * ldc + 0), vld1_s32(C + 3 * ldc + 0));
    int32x4_t v_c11 = vcombine_s32(vld1_s32(C + 2 * ldc + 2), vld1_s32(C + 3 * ldc + 2));
    int32x4_t v_c12 = vcombine_s32(vld1_s32(C + 2 * ldc + 4), vld1_s32(C + 3 * ldc + 4));
    int32x4_t v_c13 = vcombine_s32(vld1_s32(C + 2 * ldc + 6), vld1_s32(C + 3 * ldc + 6));

    for (int k = 0; k < K; ++k) {
        int8x16_t v_a0 = vld1q_s8(A + 0 * 16);    // 2x8
        int8x16_t v_a1 = vld1q_s8(A + 1 * 16);    // 2x8

        int8x16_t v_b0 = vld1q_s8(B + 0 * 16);    // 8x2
        int8x16_t v_b1 = vld1q_s8(B + 1 * 16);    // 8x2
        int8x16_t v_b2 = vld1q_s8(B + 2 * 16);    // 8x2
        int8x16_t v_b3 = vld1q_s8(B + 3 * 16);    // 8x2

        v_c00 = vmmlaq_s32(v_c00, v_a0, v_b0);
        v_c01 = vmmlaq_s32(v_c01, v_a0, v_b1);
        v_c02 = vmmlaq_s32(v_c02, v_a0, v_b2);
        v_c03 = vmmlaq_s32(v_c03, v_a0, v_b3);

        v_c10 = vmmlaq_s32(v_c10, v_a1, v_b0);
        v_c11 = vmmlaq_s32(v_c11, v_a1, v_b1);
        v_c12 = vmmlaq_s32(v_c12, v_a1, v_b2);
        v_c13 = vmmlaq_s32(v_c13, v_a1, v_b3);

        A += 2 * 16;
        B += 4 * 16;
    }

    int32x4_t v_w_c00 = vcombine_s32(vget_low_s32(v_c00), vget_low_s32(v_c01));
    int32x4_t v_w_c01 = vcombine_s32(vget_low_s32(v_c02), vget_low_s32(v_c03));
    int32x4_t v_w_c10 = vcombine_s32(vget_high_s32(v_c00), vget_high_s32(v_c01));
    int32x4_t v_w_c11 = vcombine_s32(vget_high_s32(v_c02), vget_high_s32(v_c03));
    int32x4_t v_w_c20 = vcombine_s32(vget_low_s32(v_c10), vget_low_s32(v_c11));
    int32x4_t v_w_c21 = vcombine_s32(vget_low_s32(v_c12), vget_low_s32(v_c13));
    int32x4_t v_w_c30 = vcombine_s32(vget_high_s32(v_c10), vget_high_s32(v_c11));
    int32x4_t v_w_c31 = vcombine_s32(vget_high_s32(v_c12), vget_high_s32(v_c13));

    int32x4_t v_min = vdupq_n_s32(clamp_min);
    int32x4_t v_max = vdupq_n_s32(clamp_max);

    v_w_c00 = vminq_s32(vmaxq_s32(v_w_c00, v_min), v_max);
    v_w_c01 = vminq_s32(vmaxq_s32(v_w_c01, v_min), v_max);
    v_w_c10 = vminq_s32(vmaxq_s32(v_w_c10, v_min), v_max);
    v_w_c11 = vminq_s32(vmaxq_s32(v_w_c11, v_min), v_max);
    v_w_c20 = vminq_s32(vmaxq_s32(v_w_c20, v_min), v_max);
    v_w_c21 = vminq_s32(vmaxq_s32(v_w_c21, v_min), v_max);
    v_w_c30 = vminq_s32(vmaxq_s32(v_w_c30, v_min), v_max);
    v_w_c31 = vminq_s32(vmaxq_s32(v_w_c31, v_min), v_max);

    vst1q_s32(C + 0 * ldc + 0, v_w_c00);
    vst1q_s32(C + 0 * ldc + 4, v_w_c01);
    vst1q_s32(C + 1 * ldc + 0, v_w_c10);
    vst1q_s32(C + 1 * ldc + 4, v_w_c11);
    vst1q_s32(C + 2 * ldc + 0, v_w_c20);
    vst1q_s32(C + 2 * ldc + 4, v_w_c21);
    vst1q_s32(C + 3 * ldc + 0, v_w_c30);
    vst1q_s32(C + 3 * ldc + 4, v_w_c31);
}

inline void mmma_4x16_s8s8_mmla(int32_t* NNOPS_RESTRICT C, int ldc,
                                const int8_t* NNOPS_RESTRICT A,
                                const int8_t* NNOPS_RESTRICT B, int ldb, int K,
                                int32_t clamp_min, int32_t clamp_max) noexcept {
    int32x4_t v_c00 = vcombine_s32(vld1_s32(C + 0 * ldc + 0), vld1_s32(C + 1 * ldc + 0));
    int32x4_t v_c01 = vcombine_s32(vld1_s32(C + 0 * ldc + 2), vld1_s32(C + 1 * ldc + 2));
    int32x4_t v_c02 = vcombine_s32(vld1_s32(C + 0 * ldc + 4), vld1_s32(C + 1 * ldc + 4));
    int32x4_t v_c03 = vcombine_s32(vld1_s32(C + 0 * ldc + 6), vld1_s32(C + 1 * ldc + 6));
    int32x4_t v_c04 = vcombine_s32(vld1_s32(C + 0 * ldc + 8), vld1_s32(C + 1 * ldc + 8));
    int32x4_t v_c05 = vcombine_s32(vld1_s32(C + 0 * ldc + 10), vld1_s32(C + 1 * ldc + 10));
    int32x4_t v_c06 = vcombine_s32(vld1_s32(C + 0 * ldc + 12), vld1_s32(C + 1 * ldc + 12));
    int32x4_t v_c07 = vcombine_s32(vld1_s32(C + 0 * ldc + 14), vld1_s32(C + 1 * ldc + 14));

    int32x4_t v_c10 = vcombine_s32(vld1_s32(C + 2 * ldc + 0), vld1_s32(C + 3 * ldc + 0));
    int32x4_t v_c11 = vcombine_s32(vld1_s32(C + 2 * ldc + 2), vld1_s32(C + 3 * ldc + 2));
    int32x4_t v_c12 = vcombine_s32(vld1_s32(C + 2 * ldc + 4), vld1_s32(C + 3 * ldc + 4));
    int32x4_t v_c13 = vcombine_s32(vld1_s32(C + 2 * ldc + 6), vld1_s32(C + 3 * ldc + 6));
    int32x4_t v_c14 = vcombine_s32(vld1_s32(C + 2 * ldc + 8), vld1_s32(C + 3 * ldc + 8));
    int32x4_t v_c15 = vcombine_s32(vld1_s32(C + 2 * ldc + 10), vld1_s32(C + 3 * ldc + 10));
    int32x4_t v_c16 = vcombine_s32(vld1_s32(C + 2 * ldc + 12), vld1_s32(C + 3 * ldc + 12));
    int32x4_t v_c17 = vcombine_s32(vld1_s32(C + 2 * ldc + 14), vld1_s32(C + 3 * ldc + 14));

    for (int k = 0; k < K; ++k) {
        int8x16_t v_a0 = vld1q_s8(A + 0 * 16);    // 2x8
        int8x16_t v_a1 = vld1q_s8(A + 1 * 16);    // 2x8

        int8x16_t v_b0 = vld1q_s8(B + 0 * 16);    // 8x2
        int8x16_t v_b1 = vld1q_s8(B + 1 * 16);    // 8x2
        int8x16_t v_b2 = vld1q_s8(B + 2 * 16);    // 8x2
        int8x16_t v_b3 = vld1q_s8(B + 3 * 16);    // 8x2
        int8x16_t v_b4 = vld1q_s8(B + 4 * 16);    // 8x2
        int8x16_t v_b5 = vld1q_s8(B + 5 * 16);    // 8x2
        int8x16_t v_b6 = vld1q_s8(B + 6 * 16);    // 8x2
        int8x16_t v_b7 = vld1q_s8(B + 7 * 16);    // 8x2

        v_c00 = vmmlaq_s32(v_c00, v_a0, v_b0);
        v_c01 = vmmlaq_s32(v_c01, v_a0, v_b1);
        v_c02 = vmmlaq_s32(v_c02, v_a0, v_b2);
        v_c03 = vmmlaq_s32(v_c03, v_a0, v_b3);
        v_c04 = vmmlaq_s32(v_c04, v_a0, v_b4);
        v_c05 = vmmlaq_s32(v_c05, v_a0, v_b5);
        v_c06 = vmmlaq_s32(v_c06, v_a0, v_b6);
        v_c07 = vmmlaq_s32(v_c07, v_a0, v_b7);

        v_c10 = vmmlaq_s32(v_c10, v_a1, v_b0);
        v_c11 = vmmlaq_s32(v_c11, v_a1, v_b1);
        v_c12 = vmmlaq_s32(v_c12, v_a1, v_b2);
        v_c13 = vmmlaq_s32(v_c13, v_a1, v_b3);
        v_c14 = vmmlaq_s32(v_c14, v_a1, v_b4);
        v_c15 = vmmlaq_s32(v_c15, v_a1, v_b5);
        v_c16 = vmmlaq_s32(v_c16, v_a1, v_b6);
        v_c17 = vmmlaq_s32(v_c17, v_a1, v_b7);

        A += 2 * 16;
        B += 8 * 16;
    }

    int32x4_t v_w_c00 = vcombine_s32(vget_low_s32(v_c00), vget_low_s32(v_c01));
    int32x4_t v_w_c01 = vcombine_s32(vget_low_s32(v_c02), vget_low_s32(v_c03));
    int32x4_t v_w_c02 = vcombine_s32(vget_low_s32(v_c04), vget_low_s32(v_c05));
    int32x4_t v_w_c03 = vcombine_s32(vget_low_s32(v_c06), vget_low_s32(v_c07));

    int32x4_t v_w_c10 = vcombine_s32(vget_high_s32(v_c00), vget_high_s32(v_c01));
    int32x4_t v_w_c11 = vcombine_s32(vget_high_s32(v_c02), vget_high_s32(v_c03));
    int32x4_t v_w_c12 = vcombine_s32(vget_high_s32(v_c04), vget_high_s32(v_c05));
    int32x4_t v_w_c13 = vcombine_s32(vget_high_s32(v_c06), vget_high_s32(v_c07));

    int32x4_t v_w_c20 = vcombine_s32(vget_low_s32(v_c10), vget_low_s32(v_c11));
    int32x4_t v_w_c21 = vcombine_s32(vget_low_s32(v_c12), vget_low_s32(v_c13));
    int32x4_t v_w_c22 = vcombine_s32(vget_low_s32(v_c14), vget_low_s32(v_c15));
    int32x4_t v_w_c23 = vcombine_s32(vget_low_s32(v_c16), vget_low_s32(v_c17));

    int32x4_t v_w_c30 = vcombine_s32(vget_high_s32(v_c10), vget_high_s32(v_c11));
    int32x4_t v_w_c31 = vcombine_s32(vget_high_s32(v_c12), vget_high_s32(v_c13));
    int32x4_t v_w_c32 = vcombine_s32(vget_high_s32(v_c14), vget_high_s32(v_c15));
    int32x4_t v_w_c33 = vcombine_s32(vget_high_s32(v_c16), vget_high_s32(v_c17));

    int32x4_t v_min = vdupq_n_s32(clamp_min);
    int32x4_t v_max = vdupq_n_s32(clamp_max);

    v_w_c00 = vminq_s32(vmaxq_s32(v_w_c00, v_min), v_max);
    v_w_c01 = vminq_s32(vmaxq_s32(v_w_c01, v_min), v_max);
    v_w_c02 = vminq_s32(vmaxq_s32(v_w_c02, v_min), v_max);
    v_w_c03 = vminq_s32(vmaxq_s32(v_w_c03, v_min), v_max);
    v_w_c10 = vminq_s32(vmaxq_s32(v_w_c10, v_min), v_max);
    v_w_c11 = vminq_s32(vmaxq_s32(v_w_c11, v_min), v_max);
    v_w_c12 = vminq_s32(vmaxq_s32(v_w_c12, v_min), v_max);
    v_w_c13 = vminq_s32(vmaxq_s32(v_w_c13, v_min), v_max);
    v_w_c20 = vminq_s32(vmaxq_s32(v_w_c20, v_min), v_max);
    v_w_c21 = vminq_s32(vmaxq_s32(v_w_c21, v_min), v_max);
    v_w_c22 = vminq_s32(vmaxq_s32(v_w_c22, v_min), v_max);
    v_w_c23 = vminq_s32(vmaxq_s32(v_w_c23, v_min), v_max);
    v_w_c30 = vminq_s32(vmaxq_s32(v_w_c30, v_min), v_max);
    v_w_c31 = vminq_s32(vmaxq_s32(v_w_c31, v_min), v_max);
    v_w_c32 = vminq_s32(vmaxq_s32(v_w_c32, v_min), v_max);
    v_w_c33 = vminq_s32(vmaxq_s32(v_w_c33, v_min), v_max);

    vst1q_s32(C + 0 * ldc + 0, v_w_c00);
    vst1q_s32(C + 0 * ldc + 4, v_w_c01);
    vst1q_s32(C + 0 * ldc + 8, v_w_c02);
    vst1q_s32(C + 0 * ldc + 12, v_w_c03);
    vst1q_s32(C + 1 * ldc + 0, v_w_c10);
    vst1q_s32(C + 1 * ldc + 4, v_w_c11);
    vst1q_s32(C + 1 * ldc + 8, v_w_c12);
    vst1q_s32(C + 1 * ldc + 12, v_w_c13);
    vst1q_s32(C + 2 * ldc + 0, v_w_c20);
    vst1q_s32(C + 2 * ldc + 4, v_w_c21);
    vst1q_s32(C + 2 * ldc + 8, v_w_c22);
    vst1q_s32(C + 2 * ldc + 12, v_w_c23);
    vst1q_s32(C + 3 * ldc + 0, v_w_c30);
    vst1q_s32(C + 3 * ldc + 4, v_w_c31);
    vst1q_s32(C + 3 * ldc + 8, v_w_c32);
    vst1q_s32(C + 3 * ldc + 12, v_w_c33);
}

}  // namespace nnops::backend::cpu::aarch64
