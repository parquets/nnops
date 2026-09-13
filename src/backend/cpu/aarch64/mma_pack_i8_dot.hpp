#pragma once
/// @file mma_pack_i8_dot.hpp
/// @brief AArch64 NEON int8 → int32 MMA (matrix micro-accumulate) packed-B kernels (DOT).
///
/// Each kernel computes  C[mr][nr] += A[mr][K] × B_packed[K][nr]  then clamps to
/// [clamp_min, clamp_max]. A and B are pre-packed by the caller; `K` is the number
/// of 4-element int8 groups. Two operand flavours are provided:
///   - s8s8: signed int8 × signed int8   (vdotq_s32 / vdotq_laneq_s32)
///   - u8s8: unsigned int8 × signed int8 (vusdotq_s32 / vsudotq_laneq_s32)
///
/// Tile sizes (AArch64-optimised):
///   M ∈ {8, 4, 1}    N ∈ {12, 8, 4, 1}
///
/// Reference: nn_compute/src/cpu/kernel/mma/aarch64/mma_pack_i8_dot.hpp

#include <arm_neon.h>
#include <algorithm>
#include <cstdint>
#include "backend/cpu/common/restrict.hpp"

namespace nnops::backend::cpu::aarch64 {

// =========================================================================
//  s8s8 kernels
// =========================================================================

inline void mma_pack_1x1_s8s8_dot(int32_t* NNOPS_RESTRICT C, int ldc,
                                  const int8_t* NNOPS_RESTRICT A,
                                  const int8_t* NNOPS_RESTRICT B, int K,
                                  int32_t clamp_min, int32_t clamp_max) noexcept {
    int32_t c = C[0 * ldc];

    for (int k = 0; k < K; ++k) {
        c += A[0] * B[0];
        c += A[1] * B[1];
        c += A[2] * B[2];
        c += A[3] * B[3];

        A += 4;
        B += 4;
    }

    C[0 * ldc] = std::min(std::max(c, clamp_min), clamp_max);
}

inline void mma_pack_1x4_s8s8_dot(int32_t* NNOPS_RESTRICT C, int ldc,
                                  const int8_t* NNOPS_RESTRICT A,
                                  const int8_t* NNOPS_RESTRICT B, int K,
                                  int32_t clamp_min, int32_t clamp_max) noexcept {
    int32x4_t v_c00 = vld1q_s32(C + 0 * ldc);

    for (int k = 0; k < K; ++k) {
        int8x16_t v_a = vreinterpretq_s8_s32(vdupq_n_s32(*reinterpret_cast<const int32_t*>(A)));

        int8x16_t v_b0 = vld1q_s8(B);

        v_c00 = vdotq_s32(v_c00, v_a, v_b0);

        A += 4;
        B += 16;
    }

    int32x4_t v_min = vdupq_n_s32(clamp_min);
    v_c00 = vmaxq_s32(v_c00, v_min);

    int32x4_t v_max = vdupq_n_s32(clamp_max);
    v_c00 = vminq_s32(v_c00, v_max);

    vst1q_s32(C + 0 * ldc, v_c00);
}

inline void mma_pack_1x8_s8s8_dot(int32_t* NNOPS_RESTRICT C, int ldc,
                                  const int8_t* NNOPS_RESTRICT A,
                                  const int8_t* NNOPS_RESTRICT B, int K,
                                  int32_t clamp_min, int32_t clamp_max) noexcept {
    int32x4_t v_c00 = vld1q_s32(C + 0 * ldc + 0);
    int32x4_t v_c01 = vld1q_s32(C + 0 * ldc + 4);

    for (int k = 0; k < K; ++k) {
        int8x16_t v_a = vreinterpretq_s8_s32(vdupq_n_s32(*reinterpret_cast<const int32_t*>(A)));

        int8x16_t v_b0 = vld1q_s8(B + 0 * 16);
        int8x16_t v_b1 = vld1q_s8(B + 1 * 16);

        v_c00 = vdotq_s32(v_c00, v_a, v_b0);
        v_c01 = vdotq_s32(v_c01, v_a, v_b1);

        A += 4;
        B += 32;
    }

    int32x4_t v_min = vdupq_n_s32(clamp_min);
    v_c00 = vmaxq_s32(v_c00, v_min);
    v_c01 = vmaxq_s32(v_c01, v_min);

    int32x4_t v_max = vdupq_n_s32(clamp_max);
    v_c00 = vminq_s32(v_c00, v_max);
    v_c01 = vminq_s32(v_c01, v_max);

    vst1q_s32(C + 0 * ldc + 0, v_c00);
    vst1q_s32(C + 0 * ldc + 4, v_c01);
}

inline void mma_pack_1x12_s8s8_dot(int32_t* NNOPS_RESTRICT C, int ldc,
                                   const int8_t* NNOPS_RESTRICT A,
                                   const int8_t* NNOPS_RESTRICT B, int K,
                                   int32_t clamp_min, int32_t clamp_max) noexcept {
    int32x4_t v_c00 = vld1q_s32(C + 0 * ldc + 0);
    int32x4_t v_c01 = vld1q_s32(C + 0 * ldc + 4);
    int32x4_t v_c02 = vld1q_s32(C + 0 * ldc + 8);

    for (int k = 0; k < K; ++k) {
        int8x16_t v_a = vreinterpretq_s8_s32(vdupq_n_s32(*reinterpret_cast<const int32_t*>(A)));

        int8x16_t v_b0 = vld1q_s8(B + 0 * 16);
        int8x16_t v_b1 = vld1q_s8(B + 1 * 16);
        int8x16_t v_b2 = vld1q_s8(B + 2 * 16);

        v_c00 = vdotq_s32(v_c00, v_a, v_b0);
        v_c01 = vdotq_s32(v_c01, v_a, v_b1);
        v_c02 = vdotq_s32(v_c02, v_a, v_b2);

        A += 4;
        B += 48;
    }

    int32x4_t v_min = vdupq_n_s32(clamp_min);
    v_c00 = vmaxq_s32(v_c00, v_min);
    v_c01 = vmaxq_s32(v_c01, v_min);
    v_c02 = vmaxq_s32(v_c02, v_min);

    int32x4_t v_max = vdupq_n_s32(clamp_max);
    v_c00 = vminq_s32(v_c00, v_max);
    v_c01 = vminq_s32(v_c01, v_max);
    v_c02 = vminq_s32(v_c02, v_max);

    vst1q_s32(C + 0 * ldc + 0, v_c00);
    vst1q_s32(C + 0 * ldc + 4, v_c01);
    vst1q_s32(C + 0 * ldc + 8, v_c02);
}

inline void mma_pack_4x1_s8s8_dot(int32_t* NNOPS_RESTRICT C, int ldc,
                                  const int8_t* NNOPS_RESTRICT A,
                                  const int8_t* NNOPS_RESTRICT B, int K,
                                  int32_t clamp_min, int32_t clamp_max) noexcept {
    int32_t c0 = C[0 * ldc];
    int32_t c1 = C[1 * ldc];
    int32_t c2 = C[2 * ldc];
    int32_t c3 = C[3 * ldc];

    for (int k = 0; k < K; ++k) {
        c0 += (A[4 * 0 + 0] * B[0] + A[4 * 0 + 1] * B[1] + A[4 * 0 + 2] * B[2] + A[4 * 0 + 3] * B[3]);
        c1 += (A[4 * 1 + 0] * B[0] + A[4 * 1 + 1] * B[1] + A[4 * 1 + 2] * B[2] + A[4 * 1 + 3] * B[3]);
        c2 += (A[4 * 2 + 0] * B[0] + A[4 * 2 + 1] * B[1] + A[4 * 2 + 2] * B[2] + A[4 * 2 + 3] * B[3]);
        c3 += (A[4 * 3 + 0] * B[0] + A[4 * 3 + 1] * B[1] + A[4 * 3 + 2] * B[2] + A[4 * 3 + 3] * B[3]);

        A += 16;
        B += 4;
    }

    C[0 * ldc] = std::min(std::max(c0, clamp_min), clamp_max);
    C[1 * ldc] = std::min(std::max(c1, clamp_min), clamp_max);
    C[2 * ldc] = std::min(std::max(c2, clamp_min), clamp_max);
    C[3 * ldc] = std::min(std::max(c3, clamp_min), clamp_max);
}

inline void mma_pack_4x4_s8s8_dot(int32_t* NNOPS_RESTRICT C, int ldc,
                                  const int8_t* NNOPS_RESTRICT A,
                                  const int8_t* NNOPS_RESTRICT B, int K,
                                  int32_t clamp_min, int32_t clamp_max) noexcept {
    int32x4_t v_c0 = vld1q_s32(C + 0 * ldc);
    int32x4_t v_c1 = vld1q_s32(C + 1 * ldc);
    int32x4_t v_c2 = vld1q_s32(C + 2 * ldc);
    int32x4_t v_c3 = vld1q_s32(C + 3 * ldc);

    for (int k = 0; k < K; ++k) {
        int8x16_t v_a = vld1q_s8(A);
        int8x16_t v_b = vld1q_s8(B);

        v_c0 = vdotq_laneq_s32(v_c0, v_b, v_a, 0);
        v_c1 = vdotq_laneq_s32(v_c1, v_b, v_a, 1);
        v_c2 = vdotq_laneq_s32(v_c2, v_b, v_a, 2);
        v_c3 = vdotq_laneq_s32(v_c3, v_b, v_a, 3);

        A += 16;
        B += 16;
    }

    int32x4_t v_min = vdupq_n_s32(clamp_min);
    v_c0 = vmaxq_s32(v_c0, v_min);
    v_c1 = vmaxq_s32(v_c1, v_min);
    v_c2 = vmaxq_s32(v_c2, v_min);
    v_c3 = vmaxq_s32(v_c3, v_min);

    int32x4_t v_max = vdupq_n_s32(clamp_max);
    v_c0 = vminq_s32(v_c0, v_max);
    v_c1 = vminq_s32(v_c1, v_max);
    v_c2 = vminq_s32(v_c2, v_max);
    v_c3 = vminq_s32(v_c3, v_max);

    vst1q_s32(C + 0 * ldc, v_c0);
    vst1q_s32(C + 1 * ldc, v_c1);
    vst1q_s32(C + 2 * ldc, v_c2);
    vst1q_s32(C + 3 * ldc, v_c3);
}

inline void mma_pack_4x8_s8s8_dot(int32_t* NNOPS_RESTRICT C, int ldc,
                                  const int8_t* NNOPS_RESTRICT A,
                                  const int8_t* NNOPS_RESTRICT B, int K,
                                  int32_t clamp_min, int32_t clamp_max) noexcept {
    int32x4_t v_c00 = vld1q_s32(C + 0 * ldc + 0);
    int32x4_t v_c01 = vld1q_s32(C + 0 * ldc + 4);
    int32x4_t v_c10 = vld1q_s32(C + 1 * ldc + 0);
    int32x4_t v_c11 = vld1q_s32(C + 1 * ldc + 4);
    int32x4_t v_c20 = vld1q_s32(C + 2 * ldc + 0);
    int32x4_t v_c21 = vld1q_s32(C + 2 * ldc + 4);
    int32x4_t v_c30 = vld1q_s32(C + 3 * ldc + 0);
    int32x4_t v_c31 = vld1q_s32(C + 3 * ldc + 4);

    for (int k = 0; k < K; ++k) {
        int8x16_t v_a0 = vld1q_s8(A + 0 * 16);

        int8x16_t v_b0 = vld1q_s8(B + 0 * 16);
        int8x16_t v_b1 = vld1q_s8(B + 1 * 16);

        v_c00 = vdotq_laneq_s32(v_c00, v_b0, v_a0, 0);
        v_c10 = vdotq_laneq_s32(v_c10, v_b0, v_a0, 1);
        v_c20 = vdotq_laneq_s32(v_c20, v_b0, v_a0, 2);
        v_c30 = vdotq_laneq_s32(v_c30, v_b0, v_a0, 3);

        v_c01 = vdotq_laneq_s32(v_c01, v_b1, v_a0, 0);
        v_c11 = vdotq_laneq_s32(v_c11, v_b1, v_a0, 1);
        v_c21 = vdotq_laneq_s32(v_c21, v_b1, v_a0, 2);
        v_c31 = vdotq_laneq_s32(v_c31, v_b1, v_a0, 3);

        A += 16;
        B += 32;
    }

    int32x4_t v_min = vdupq_n_s32(clamp_min);
    int32x4_t v_max = vdupq_n_s32(clamp_max);

    v_c00 = vminq_s32(vmaxq_s32(v_c00, v_min), v_max);
    v_c01 = vminq_s32(vmaxq_s32(v_c01, v_min), v_max);
    v_c10 = vminq_s32(vmaxq_s32(v_c10, v_min), v_max);
    v_c11 = vminq_s32(vmaxq_s32(v_c11, v_min), v_max);
    v_c20 = vminq_s32(vmaxq_s32(v_c20, v_min), v_max);
    v_c21 = vminq_s32(vmaxq_s32(v_c21, v_min), v_max);
    v_c30 = vminq_s32(vmaxq_s32(v_c30, v_min), v_max);
    v_c31 = vminq_s32(vmaxq_s32(v_c31, v_min), v_max);

    vst1q_s32(C + 0 * ldc + 0, v_c00);
    vst1q_s32(C + 0 * ldc + 4, v_c01);
    vst1q_s32(C + 1 * ldc + 0, v_c10);
    vst1q_s32(C + 1 * ldc + 4, v_c11);
    vst1q_s32(C + 2 * ldc + 0, v_c20);
    vst1q_s32(C + 2 * ldc + 4, v_c21);
    vst1q_s32(C + 3 * ldc + 0, v_c30);
    vst1q_s32(C + 3 * ldc + 4, v_c31);
}

inline void mma_pack_4x12_s8s8_dot(int32_t* NNOPS_RESTRICT C, int ldc,
                                   const int8_t* NNOPS_RESTRICT A,
                                   const int8_t* NNOPS_RESTRICT B,
                                   int K, int32_t clamp_min, int32_t clamp_max) noexcept {
    int32x4_t v_c00 = vld1q_s32(C + 0 * ldc + 0);
    int32x4_t v_c01 = vld1q_s32(C + 0 * ldc + 4);
    int32x4_t v_c02 = vld1q_s32(C + 0 * ldc + 8);
    int32x4_t v_c10 = vld1q_s32(C + 1 * ldc + 0);
    int32x4_t v_c11 = vld1q_s32(C + 1 * ldc + 4);
    int32x4_t v_c12 = vld1q_s32(C + 1 * ldc + 8);
    int32x4_t v_c20 = vld1q_s32(C + 2 * ldc + 0);
    int32x4_t v_c21 = vld1q_s32(C + 2 * ldc + 4);
    int32x4_t v_c22 = vld1q_s32(C + 2 * ldc + 8);
    int32x4_t v_c30 = vld1q_s32(C + 3 * ldc + 0);
    int32x4_t v_c31 = vld1q_s32(C + 3 * ldc + 4);
    int32x4_t v_c32 = vld1q_s32(C + 3 * ldc + 8);

    for (int k = 0; k < K; ++k) {
        int8x16_t v_a = vld1q_s8(A);

        int8x16_t v_b0 = vld1q_s8(B + 0 * 16);
        int8x16_t v_b1 = vld1q_s8(B + 1 * 16);
        int8x16_t v_b2 = vld1q_s8(B + 2 * 16);

        v_c00 = vdotq_laneq_s32(v_c00, v_b0, v_a, 0);
        v_c10 = vdotq_laneq_s32(v_c10, v_b0, v_a, 1);
        v_c20 = vdotq_laneq_s32(v_c20, v_b0, v_a, 2);
        v_c30 = vdotq_laneq_s32(v_c30, v_b0, v_a, 3);

        v_c01 = vdotq_laneq_s32(v_c01, v_b1, v_a, 0);
        v_c11 = vdotq_laneq_s32(v_c11, v_b1, v_a, 1);
        v_c21 = vdotq_laneq_s32(v_c21, v_b1, v_a, 2);
        v_c31 = vdotq_laneq_s32(v_c31, v_b1, v_a, 3);

        v_c02 = vdotq_laneq_s32(v_c02, v_b2, v_a, 0);
        v_c12 = vdotq_laneq_s32(v_c12, v_b2, v_a, 1);
        v_c22 = vdotq_laneq_s32(v_c22, v_b2, v_a, 2);
        v_c32 = vdotq_laneq_s32(v_c32, v_b2, v_a, 3);

        A += 16;
        B += 48;
    }

    int32x4_t v_min = vdupq_n_s32(clamp_min);
    v_c00 = vmaxq_s32(v_c00, v_min);
    v_c01 = vmaxq_s32(v_c01, v_min);
    v_c02 = vmaxq_s32(v_c02, v_min);
    v_c10 = vmaxq_s32(v_c10, v_min);
    v_c11 = vmaxq_s32(v_c11, v_min);
    v_c12 = vmaxq_s32(v_c12, v_min);
    v_c20 = vmaxq_s32(v_c20, v_min);
    v_c21 = vmaxq_s32(v_c21, v_min);
    v_c22 = vmaxq_s32(v_c22, v_min);
    v_c30 = vmaxq_s32(v_c30, v_min);
    v_c31 = vmaxq_s32(v_c31, v_min);
    v_c32 = vmaxq_s32(v_c32, v_min);

    int32x4_t v_max = vdupq_n_s32(clamp_max);
    v_c00 = vminq_s32(v_c00, v_max);
    v_c01 = vminq_s32(v_c01, v_max);
    v_c02 = vminq_s32(v_c02, v_max);
    v_c10 = vminq_s32(v_c10, v_max);
    v_c11 = vminq_s32(v_c11, v_max);
    v_c12 = vminq_s32(v_c12, v_max);
    v_c20 = vminq_s32(v_c20, v_max);
    v_c21 = vminq_s32(v_c21, v_max);
    v_c22 = vminq_s32(v_c22, v_max);
    v_c30 = vminq_s32(v_c30, v_max);
    v_c31 = vminq_s32(v_c31, v_max);
    v_c32 = vminq_s32(v_c32, v_max);

    vst1q_s32(C + 0 * ldc + 0, v_c00);
    vst1q_s32(C + 0 * ldc + 4, v_c01);
    vst1q_s32(C + 0 * ldc + 8, v_c02);
    vst1q_s32(C + 1 * ldc + 0, v_c10);
    vst1q_s32(C + 1 * ldc + 4, v_c11);
    vst1q_s32(C + 1 * ldc + 8, v_c12);
    vst1q_s32(C + 2 * ldc + 0, v_c20);
    vst1q_s32(C + 2 * ldc + 4, v_c21);
    vst1q_s32(C + 2 * ldc + 8, v_c22);
    vst1q_s32(C + 3 * ldc + 0, v_c30);
    vst1q_s32(C + 3 * ldc + 4, v_c31);
    vst1q_s32(C + 3 * ldc + 8, v_c32);
}

inline void mma_pack_8x1_s8s8_dot(int32_t* NNOPS_RESTRICT C, int ldc,
                                  const int8_t* NNOPS_RESTRICT A,
                                  const int8_t* NNOPS_RESTRICT B,
                                  int K,
                                  int32_t clamp_min, int32_t clamp_max) noexcept {
    int32_t c0 = C[0 * ldc];
    int32_t c1 = C[1 * ldc];
    int32_t c2 = C[2 * ldc];
    int32_t c3 = C[3 * ldc];
    int32_t c4 = C[4 * ldc];
    int32_t c5 = C[5 * ldc];
    int32_t c6 = C[6 * ldc];
    int32_t c7 = C[7 * ldc];

    for (int k = 0; k < K; ++k) {
        c0 += (A[4 * 0 + 0] * B[0] + A[4 * 0 + 1] * B[1] + A[4 * 0 + 2] * B[2] + A[4 * 0 + 3] * B[3]);
        c1 += (A[4 * 1 + 0] * B[0] + A[4 * 1 + 1] * B[1] + A[4 * 1 + 2] * B[2] + A[4 * 1 + 3] * B[3]);
        c2 += (A[4 * 2 + 0] * B[0] + A[4 * 2 + 1] * B[1] + A[4 * 2 + 2] * B[2] + A[4 * 2 + 3] * B[3]);
        c3 += (A[4 * 3 + 0] * B[0] + A[4 * 3 + 1] * B[1] + A[4 * 3 + 2] * B[2] + A[4 * 3 + 3] * B[3]);
        c4 += (A[4 * 4 + 0] * B[0] + A[4 * 4 + 1] * B[1] + A[4 * 4 + 2] * B[2] + A[4 * 4 + 3] * B[3]);
        c5 += (A[4 * 5 + 0] * B[0] + A[4 * 5 + 1] * B[1] + A[4 * 5 + 2] * B[2] + A[4 * 5 + 3] * B[3]);
        c6 += (A[4 * 6 + 0] * B[0] + A[4 * 6 + 1] * B[1] + A[4 * 6 + 2] * B[2] + A[4 * 6 + 3] * B[3]);
        c7 += (A[4 * 7 + 0] * B[0] + A[4 * 7 + 1] * B[1] + A[4 * 7 + 2] * B[2] + A[4 * 7 + 3] * B[3]);

        A += 32;
        B += 4;
    }

    C[0 * ldc] = std::min(std::max(c0, clamp_min), clamp_max);
    C[1 * ldc] = std::min(std::max(c1, clamp_min), clamp_max);
    C[2 * ldc] = std::min(std::max(c2, clamp_min), clamp_max);
    C[3 * ldc] = std::min(std::max(c3, clamp_min), clamp_max);
    C[4 * ldc] = std::min(std::max(c4, clamp_min), clamp_max);
    C[5 * ldc] = std::min(std::max(c5, clamp_min), clamp_max);
    C[6 * ldc] = std::min(std::max(c6, clamp_min), clamp_max);
    C[7 * ldc] = std::min(std::max(c7, clamp_min), clamp_max);
}

inline void mma_pack_8x4_s8s8_dot(int32_t* NNOPS_RESTRICT C, int ldc,
                                  const int8_t* NNOPS_RESTRICT A,
                                  const int8_t* NNOPS_RESTRICT B,
                                  int K,
                                  int32_t clamp_min, int32_t clamp_max) noexcept {
    int32x4_t v_c0 = vld1q_s32(C + 0 * ldc);
    int32x4_t v_c1 = vld1q_s32(C + 1 * ldc);
    int32x4_t v_c2 = vld1q_s32(C + 2 * ldc);
    int32x4_t v_c3 = vld1q_s32(C + 3 * ldc);
    int32x4_t v_c4 = vld1q_s32(C + 4 * ldc);
    int32x4_t v_c5 = vld1q_s32(C + 5 * ldc);
    int32x4_t v_c6 = vld1q_s32(C + 6 * ldc);
    int32x4_t v_c7 = vld1q_s32(C + 7 * ldc);

    for (int k = 0; k < K; ++k) {
        int8x16_t v_a0 = vld1q_s8(A + 0 * 16);
        int8x16_t v_a1 = vld1q_s8(A + 1 * 16);
        int8x16_t v_b = vld1q_s8(B);

        v_c0 = vdotq_laneq_s32(v_c0, v_b, v_a0, 0);
        v_c1 = vdotq_laneq_s32(v_c1, v_b, v_a0, 1);
        v_c2 = vdotq_laneq_s32(v_c2, v_b, v_a0, 2);
        v_c3 = vdotq_laneq_s32(v_c3, v_b, v_a0, 3);

        v_c4 = vdotq_laneq_s32(v_c4, v_b, v_a1, 0);
        v_c5 = vdotq_laneq_s32(v_c5, v_b, v_a1, 1);
        v_c6 = vdotq_laneq_s32(v_c6, v_b, v_a1, 2);
        v_c7 = vdotq_laneq_s32(v_c7, v_b, v_a1, 3);

        A += 32;
        B += 16;
    }

    int32x4_t v_min = vdupq_n_s32(clamp_min);
    v_c0 = vmaxq_s32(v_c0, v_min);
    v_c1 = vmaxq_s32(v_c1, v_min);
    v_c2 = vmaxq_s32(v_c2, v_min);
    v_c3 = vmaxq_s32(v_c3, v_min);
    v_c4 = vmaxq_s32(v_c4, v_min);
    v_c5 = vmaxq_s32(v_c5, v_min);
    v_c6 = vmaxq_s32(v_c6, v_min);
    v_c7 = vmaxq_s32(v_c7, v_min);

    int32x4_t v_max = vdupq_n_s32(clamp_max);
    v_c0 = vminq_s32(v_c0, v_max);
    v_c1 = vminq_s32(v_c1, v_max);
    v_c2 = vminq_s32(v_c2, v_max);
    v_c3 = vminq_s32(v_c3, v_max);
    v_c4 = vminq_s32(v_c4, v_max);
    v_c5 = vminq_s32(v_c5, v_max);
    v_c6 = vminq_s32(v_c6, v_max);
    v_c7 = vminq_s32(v_c7, v_max);

    vst1q_s32(C + 0 * ldc, v_c0);
    vst1q_s32(C + 1 * ldc, v_c1);
    vst1q_s32(C + 2 * ldc, v_c2);
    vst1q_s32(C + 3 * ldc, v_c3);
    vst1q_s32(C + 4 * ldc, v_c4);
    vst1q_s32(C + 5 * ldc, v_c5);
    vst1q_s32(C + 6 * ldc, v_c6);
    vst1q_s32(C + 7 * ldc, v_c7);
}

inline void mma_pack_8x8_s8s8_dot(int32_t* NNOPS_RESTRICT C, int ldc,
                                  const int8_t* NNOPS_RESTRICT A,
                                  const int8_t* NNOPS_RESTRICT B,
                                  int K,
                                  int32_t clamp_min, int32_t clamp_max) noexcept {
    int32x4_t v_c00 = vld1q_s32(C + 0 * ldc + 0);
    int32x4_t v_c01 = vld1q_s32(C + 0 * ldc + 4);
    int32x4_t v_c10 = vld1q_s32(C + 1 * ldc + 0);
    int32x4_t v_c11 = vld1q_s32(C + 1 * ldc + 4);
    int32x4_t v_c20 = vld1q_s32(C + 2 * ldc + 0);
    int32x4_t v_c21 = vld1q_s32(C + 2 * ldc + 4);
    int32x4_t v_c30 = vld1q_s32(C + 3 * ldc + 0);
    int32x4_t v_c31 = vld1q_s32(C + 3 * ldc + 4);
    int32x4_t v_c40 = vld1q_s32(C + 4 * ldc + 0);
    int32x4_t v_c41 = vld1q_s32(C + 4 * ldc + 4);
    int32x4_t v_c50 = vld1q_s32(C + 5 * ldc + 0);
    int32x4_t v_c51 = vld1q_s32(C + 5 * ldc + 4);
    int32x4_t v_c60 = vld1q_s32(C + 6 * ldc + 0);
    int32x4_t v_c61 = vld1q_s32(C + 6 * ldc + 4);
    int32x4_t v_c70 = vld1q_s32(C + 7 * ldc + 0);
    int32x4_t v_c71 = vld1q_s32(C + 7 * ldc + 4);

    for (int k = 0; k < K; ++k) {
        int8x16_t v_a0 = vld1q_s8(A + 0 * 16);
        int8x16_t v_a1 = vld1q_s8(A + 1 * 16);

        int8x16_t v_b0 = vld1q_s8(B + 0 * 16);
        int8x16_t v_b1 = vld1q_s8(B + 1 * 16);

        v_c00 = vdotq_laneq_s32(v_c00, v_b0, v_a0, 0);
        v_c10 = vdotq_laneq_s32(v_c10, v_b0, v_a0, 1);
        v_c20 = vdotq_laneq_s32(v_c20, v_b0, v_a0, 2);
        v_c30 = vdotq_laneq_s32(v_c30, v_b0, v_a0, 3);

        v_c01 = vdotq_laneq_s32(v_c01, v_b1, v_a0, 0);
        v_c11 = vdotq_laneq_s32(v_c11, v_b1, v_a0, 1);
        v_c21 = vdotq_laneq_s32(v_c21, v_b1, v_a0, 2);
        v_c31 = vdotq_laneq_s32(v_c31, v_b1, v_a0, 3);

        v_c40 = vdotq_laneq_s32(v_c40, v_b0, v_a1, 0);
        v_c50 = vdotq_laneq_s32(v_c50, v_b0, v_a1, 1);
        v_c60 = vdotq_laneq_s32(v_c60, v_b0, v_a1, 2);
        v_c70 = vdotq_laneq_s32(v_c70, v_b0, v_a1, 3);

        v_c41 = vdotq_laneq_s32(v_c41, v_b1, v_a1, 0);
        v_c51 = vdotq_laneq_s32(v_c51, v_b1, v_a1, 1);
        v_c61 = vdotq_laneq_s32(v_c61, v_b1, v_a1, 2);
        v_c71 = vdotq_laneq_s32(v_c71, v_b1, v_a1, 3);

        A += 32;
        B += 32;
    }

    int32x4_t v_min = vdupq_n_s32(clamp_min);
    int32x4_t v_max = vdupq_n_s32(clamp_max);

    v_c00 = vminq_s32(vmaxq_s32(v_c00, v_min), v_max);
    v_c01 = vminq_s32(vmaxq_s32(v_c01, v_min), v_max);
    v_c10 = vminq_s32(vmaxq_s32(v_c10, v_min), v_max);
    v_c11 = vminq_s32(vmaxq_s32(v_c11, v_min), v_max);
    v_c20 = vminq_s32(vmaxq_s32(v_c20, v_min), v_max);
    v_c21 = vminq_s32(vmaxq_s32(v_c21, v_min), v_max);
    v_c30 = vminq_s32(vmaxq_s32(v_c30, v_min), v_max);
    v_c31 = vminq_s32(vmaxq_s32(v_c31, v_min), v_max);

    vst1q_s32(C + 0 * ldc + 0, v_c00);
    vst1q_s32(C + 0 * ldc + 4, v_c01);
    vst1q_s32(C + 1 * ldc + 0, v_c10);
    vst1q_s32(C + 1 * ldc + 4, v_c11);
    vst1q_s32(C + 2 * ldc + 0, v_c20);
    vst1q_s32(C + 2 * ldc + 4, v_c21);
    vst1q_s32(C + 3 * ldc + 0, v_c30);
    vst1q_s32(C + 3 * ldc + 4, v_c31);

    v_c40 = vminq_s32(vmaxq_s32(v_c40, v_min), v_max);
    v_c41 = vminq_s32(vmaxq_s32(v_c41, v_min), v_max);
    v_c50 = vminq_s32(vmaxq_s32(v_c50, v_min), v_max);
    v_c51 = vminq_s32(vmaxq_s32(v_c51, v_min), v_max);
    v_c60 = vminq_s32(vmaxq_s32(v_c60, v_min), v_max);
    v_c61 = vminq_s32(vmaxq_s32(v_c61, v_min), v_max);
    v_c70 = vminq_s32(vmaxq_s32(v_c70, v_min), v_max);
    v_c71 = vminq_s32(vmaxq_s32(v_c71, v_min), v_max);

    vst1q_s32(C + 4 * ldc + 0, v_c40);
    vst1q_s32(C + 4 * ldc + 4, v_c41);
    vst1q_s32(C + 5 * ldc + 0, v_c50);
    vst1q_s32(C + 5 * ldc + 4, v_c51);
    vst1q_s32(C + 6 * ldc + 0, v_c60);
    vst1q_s32(C + 6 * ldc + 4, v_c61);
    vst1q_s32(C + 7 * ldc + 0, v_c70);
    vst1q_s32(C + 7 * ldc + 4, v_c71);
}

inline void mma_pack_8x12_s8s8_dot(int32_t* NNOPS_RESTRICT C, int ldc,
                                   const int8_t* NNOPS_RESTRICT A,
                                   const int8_t* NNOPS_RESTRICT B,
                                   int K,
                                   int32_t clamp_min, int32_t clamp_max) noexcept {
    int32x4_t v_c00 = vld1q_s32(C + 0 * ldc + 0);
    int32x4_t v_c01 = vld1q_s32(C + 0 * ldc + 4);
    int32x4_t v_c02 = vld1q_s32(C + 0 * ldc + 8);
    int32x4_t v_c10 = vld1q_s32(C + 1 * ldc + 0);
    int32x4_t v_c11 = vld1q_s32(C + 1 * ldc + 4);
    int32x4_t v_c12 = vld1q_s32(C + 1 * ldc + 8);
    int32x4_t v_c20 = vld1q_s32(C + 2 * ldc + 0);
    int32x4_t v_c21 = vld1q_s32(C + 2 * ldc + 4);
    int32x4_t v_c22 = vld1q_s32(C + 2 * ldc + 8);
    int32x4_t v_c30 = vld1q_s32(C + 3 * ldc + 0);
    int32x4_t v_c31 = vld1q_s32(C + 3 * ldc + 4);
    int32x4_t v_c32 = vld1q_s32(C + 3 * ldc + 8);
    int32x4_t v_c40 = vld1q_s32(C + 4 * ldc + 0);
    int32x4_t v_c41 = vld1q_s32(C + 4 * ldc + 4);
    int32x4_t v_c42 = vld1q_s32(C + 4 * ldc + 8);
    int32x4_t v_c50 = vld1q_s32(C + 5 * ldc + 0);
    int32x4_t v_c51 = vld1q_s32(C + 5 * ldc + 4);
    int32x4_t v_c52 = vld1q_s32(C + 5 * ldc + 8);
    int32x4_t v_c60 = vld1q_s32(C + 6 * ldc + 0);
    int32x4_t v_c61 = vld1q_s32(C + 6 * ldc + 4);
    int32x4_t v_c62 = vld1q_s32(C + 6 * ldc + 8);
    int32x4_t v_c70 = vld1q_s32(C + 7 * ldc + 0);
    int32x4_t v_c71 = vld1q_s32(C + 7 * ldc + 4);
    int32x4_t v_c72 = vld1q_s32(C + 7 * ldc + 8);

    for (int k = 0; k < K; ++k) {
        int8x16_t v_a0 = vld1q_s8(A + 0 * 16);
        int8x16_t v_a1 = vld1q_s8(A + 1 * 16);

        int8x16_t v_b0 = vld1q_s8(B + 0 * 16);

        v_c00 = vdotq_laneq_s32(v_c00, v_b0, v_a0, 0);
        v_c10 = vdotq_laneq_s32(v_c10, v_b0, v_a0, 1);
        v_c20 = vdotq_laneq_s32(v_c20, v_b0, v_a0, 2);
        v_c30 = vdotq_laneq_s32(v_c30, v_b0, v_a0, 3);

        v_c40 = vdotq_laneq_s32(v_c40, v_b0, v_a1, 0);
        v_c50 = vdotq_laneq_s32(v_c50, v_b0, v_a1, 1);
        v_c60 = vdotq_laneq_s32(v_c60, v_b0, v_a1, 2);
        v_c70 = vdotq_laneq_s32(v_c70, v_b0, v_a1, 3);

        int8x16_t v_b1 = vld1q_s8(B + 1 * 16);

        v_c01 = vdotq_laneq_s32(v_c01, v_b1, v_a0, 0);
        v_c11 = vdotq_laneq_s32(v_c11, v_b1, v_a0, 1);
        v_c21 = vdotq_laneq_s32(v_c21, v_b1, v_a0, 2);
        v_c31 = vdotq_laneq_s32(v_c31, v_b1, v_a0, 3);

        v_c41 = vdotq_laneq_s32(v_c41, v_b1, v_a1, 0);
        v_c51 = vdotq_laneq_s32(v_c51, v_b1, v_a1, 1);
        v_c61 = vdotq_laneq_s32(v_c61, v_b1, v_a1, 2);
        v_c71 = vdotq_laneq_s32(v_c71, v_b1, v_a1, 3);

        int8x16_t v_b2 = vld1q_s8(B + 2 * 16);

        v_c02 = vdotq_laneq_s32(v_c02, v_b2, v_a0, 0);
        v_c12 = vdotq_laneq_s32(v_c12, v_b2, v_a0, 1);
        v_c22 = vdotq_laneq_s32(v_c22, v_b2, v_a0, 2);
        v_c32 = vdotq_laneq_s32(v_c32, v_b2, v_a0, 3);

        v_c42 = vdotq_laneq_s32(v_c42, v_b2, v_a1, 0);
        v_c52 = vdotq_laneq_s32(v_c52, v_b2, v_a1, 1);
        v_c62 = vdotq_laneq_s32(v_c62, v_b2, v_a1, 2);
        v_c72 = vdotq_laneq_s32(v_c72, v_b2, v_a1, 3);

        A += 32;
        B += 48;
    }

    int32x4_t v_min = vdupq_n_s32(clamp_min);
    int32x4_t v_max = vdupq_n_s32(clamp_max);

    v_c00 = vminq_s32(vmaxq_s32(v_c00, v_min), v_max);
    v_c01 = vminq_s32(vmaxq_s32(v_c01, v_min), v_max);
    v_c02 = vminq_s32(vmaxq_s32(v_c02, v_min), v_max);
    v_c10 = vminq_s32(vmaxq_s32(v_c10, v_min), v_max);
    v_c11 = vminq_s32(vmaxq_s32(v_c11, v_min), v_max);
    v_c12 = vminq_s32(vmaxq_s32(v_c12, v_min), v_max);
    v_c20 = vminq_s32(vmaxq_s32(v_c20, v_min), v_max);
    v_c21 = vminq_s32(vmaxq_s32(v_c21, v_min), v_max);
    v_c22 = vminq_s32(vmaxq_s32(v_c22, v_min), v_max);
    v_c30 = vminq_s32(vmaxq_s32(v_c30, v_min), v_max);
    v_c31 = vminq_s32(vmaxq_s32(v_c31, v_min), v_max);
    v_c32 = vminq_s32(vmaxq_s32(v_c32, v_min), v_max);

    vst1q_s32(C + 0 * ldc + 0, v_c00);
    vst1q_s32(C + 0 * ldc + 4, v_c01);
    vst1q_s32(C + 0 * ldc + 8, v_c02);
    vst1q_s32(C + 1 * ldc + 0, v_c10);
    vst1q_s32(C + 1 * ldc + 4, v_c11);
    vst1q_s32(C + 1 * ldc + 8, v_c12);
    vst1q_s32(C + 2 * ldc + 0, v_c20);
    vst1q_s32(C + 2 * ldc + 4, v_c21);
    vst1q_s32(C + 2 * ldc + 8, v_c22);
    vst1q_s32(C + 3 * ldc + 0, v_c30);
    vst1q_s32(C + 3 * ldc + 4, v_c31);
    vst1q_s32(C + 3 * ldc + 8, v_c32);

    v_c40 = vminq_s32(vmaxq_s32(v_c40, v_min), v_max);
    v_c41 = vminq_s32(vmaxq_s32(v_c41, v_min), v_max);
    v_c42 = vminq_s32(vmaxq_s32(v_c42, v_min), v_max);
    v_c50 = vminq_s32(vmaxq_s32(v_c50, v_min), v_max);
    v_c51 = vminq_s32(vmaxq_s32(v_c51, v_min), v_max);
    v_c52 = vminq_s32(vmaxq_s32(v_c52, v_min), v_max);
    v_c60 = vminq_s32(vmaxq_s32(v_c60, v_min), v_max);
    v_c61 = vminq_s32(vmaxq_s32(v_c61, v_min), v_max);
    v_c62 = vminq_s32(vmaxq_s32(v_c62, v_min), v_max);
    v_c70 = vminq_s32(vmaxq_s32(v_c70, v_min), v_max);
    v_c71 = vminq_s32(vmaxq_s32(v_c71, v_min), v_max);
    v_c72 = vminq_s32(vmaxq_s32(v_c72, v_min), v_max);

    vst1q_s32(C + 4 * ldc + 0, v_c40);
    vst1q_s32(C + 4 * ldc + 4, v_c41);
    vst1q_s32(C + 4 * ldc + 8, v_c42);
    vst1q_s32(C + 5 * ldc + 0, v_c50);
    vst1q_s32(C + 5 * ldc + 4, v_c51);
    vst1q_s32(C + 5 * ldc + 8, v_c52);
    vst1q_s32(C + 6 * ldc + 0, v_c60);
    vst1q_s32(C + 6 * ldc + 4, v_c61);
    vst1q_s32(C + 6 * ldc + 8, v_c62);
    vst1q_s32(C + 7 * ldc + 0, v_c70);
    vst1q_s32(C + 7 * ldc + 4, v_c71);
    vst1q_s32(C + 7 * ldc + 8, v_c72);
}

// =========================================================================
//  u8s8 kernels
// =========================================================================

inline void mma_pack_1x1_u8s8_dot(int32_t* NNOPS_RESTRICT C, int ldc,
                                  const uint8_t* NNOPS_RESTRICT A,
                                  const int8_t* NNOPS_RESTRICT B, int K,
                                  int32_t clamp_min, int32_t clamp_max) noexcept {
    int32_t c = C[0 * ldc];

    for (int k = 0; k < K; ++k) {
        c += A[0] * B[0];
        c += A[1] * B[1];
        c += A[2] * B[2];
        c += A[3] * B[3];

        A += 4;
        B += 4;
    }

    C[0 * ldc] = std::min(std::max(c, clamp_min), clamp_max);
}

inline void mma_pack_1x8_u8s8_dot(int32_t* NNOPS_RESTRICT C, int ldc,
                                  const uint8_t* NNOPS_RESTRICT A,
                                  const int8_t* NNOPS_RESTRICT B, int K,
                                  int32_t clamp_min, int32_t clamp_max) noexcept {
    int32x4_t v_c00 = vld1q_s32(C + 0 * ldc + 0);
    int32x4_t v_c01 = vld1q_s32(C + 0 * ldc + 4);

    for (int k = 0; k < K; ++k) {
        uint8x16_t v_a = vreinterpretq_u8_s32(vdupq_n_s32(*reinterpret_cast<const int32_t*>(A)));

        int8x16_t v_b0 = vld1q_s8(B + 0 * 16);
        int8x16_t v_b1 = vld1q_s8(B + 1 * 16);

        v_c00 = vusdotq_s32(v_c00, v_a, v_b0);
        v_c01 = vusdotq_s32(v_c01, v_a, v_b1);

        A += 4;
        B += 32;
    }

    int32x4_t v_min = vdupq_n_s32(clamp_min);
    v_c00 = vmaxq_s32(v_c00, v_min);
    v_c01 = vmaxq_s32(v_c01, v_min);

    int32x4_t v_max = vdupq_n_s32(clamp_max);
    v_c00 = vminq_s32(v_c00, v_max);
    v_c01 = vminq_s32(v_c01, v_max);

    vst1q_s32(C + 0 * ldc + 0, v_c00);
    vst1q_s32(C + 0 * ldc + 4, v_c01);
}

inline void mma_pack_1x12_u8s8_dot(int32_t* NNOPS_RESTRICT C, int ldc,
                                   const uint8_t* NNOPS_RESTRICT A,
                                   const int8_t* NNOPS_RESTRICT B, int K,
                                   int32_t clamp_min, int32_t clamp_max) noexcept {
    int32x4_t v_c00 = vld1q_s32(C + 0 * ldc + 0);
    int32x4_t v_c01 = vld1q_s32(C + 0 * ldc + 4);
    int32x4_t v_c02 = vld1q_s32(C + 0 * ldc + 8);

    for (int k = 0; k < K; ++k) {
        uint8x16_t v_a = vreinterpretq_u8_s32(vdupq_n_s32(*reinterpret_cast<const int32_t*>(A)));

        int8x16_t v_b0 = vld1q_s8(B + 0 * 16);
        int8x16_t v_b1 = vld1q_s8(B + 1 * 16);
        int8x16_t v_b2 = vld1q_s8(B + 2 * 16);

        v_c00 = vusdotq_s32(v_c00, v_a, v_b0);
        v_c01 = vusdotq_s32(v_c01, v_a, v_b1);
        v_c02 = vusdotq_s32(v_c02, v_a, v_b2);

        A += 4;
        B += 48;
    }

    int32x4_t v_min = vdupq_n_s32(clamp_min);
    v_c00 = vmaxq_s32(v_c00, v_min);
    v_c01 = vmaxq_s32(v_c01, v_min);
    v_c02 = vmaxq_s32(v_c02, v_min);

    int32x4_t v_max = vdupq_n_s32(clamp_max);
    v_c00 = vminq_s32(v_c00, v_max);
    v_c01 = vminq_s32(v_c01, v_max);
    v_c02 = vminq_s32(v_c02, v_max);

    vst1q_s32(C + 0 * ldc + 0, v_c00);
    vst1q_s32(C + 0 * ldc + 4, v_c01);
    vst1q_s32(C + 0 * ldc + 8, v_c02);
}

inline void mma_pack_4x1_u8s8_dot(int32_t* NNOPS_RESTRICT C, int ldc,
                                  const uint8_t* NNOPS_RESTRICT A,
                                  const int8_t* NNOPS_RESTRICT B, int K,
                                  int32_t clamp_min, int32_t clamp_max) noexcept {
    int32_t c0 = C[0 * ldc];
    int32_t c1 = C[1 * ldc];
    int32_t c2 = C[2 * ldc];
    int32_t c3 = C[3 * ldc];

    for (int k = 0; k < K; ++k) {
        c0 += (A[4 * 0 + 0] * B[0] + A[4 * 0 + 1] * B[1] + A[4 * 0 + 2] * B[2] + A[4 * 0 + 3] * B[3]);
        c1 += (A[4 * 1 + 0] * B[0] + A[4 * 1 + 1] * B[1] + A[4 * 1 + 2] * B[2] + A[4 * 1 + 3] * B[3]);
        c2 += (A[4 * 2 + 0] * B[0] + A[4 * 2 + 1] * B[1] + A[4 * 2 + 2] * B[2] + A[4 * 2 + 3] * B[3]);
        c3 += (A[4 * 3 + 0] * B[0] + A[4 * 3 + 1] * B[1] + A[4 * 3 + 2] * B[2] + A[4 * 3 + 3] * B[3]);

        A += 16;
        B += 4;
    }

    C[0 * ldc] = std::min(std::max(c0, clamp_min), clamp_max);
    C[1 * ldc] = std::min(std::max(c1, clamp_min), clamp_max);
    C[2 * ldc] = std::min(std::max(c2, clamp_min), clamp_max);
    C[3 * ldc] = std::min(std::max(c3, clamp_min), clamp_max);
}

inline void mma_pack_4x4_u8s8_dot(int32_t* NNOPS_RESTRICT C, int ldc,
                                  const uint8_t* NNOPS_RESTRICT A,
                                  const int8_t* NNOPS_RESTRICT B, int K,
                                  int32_t clamp_min, int32_t clamp_max) noexcept {
    int32x4_t v_c0 = vld1q_s32(C + 0 * ldc);
    int32x4_t v_c1 = vld1q_s32(C + 1 * ldc);
    int32x4_t v_c2 = vld1q_s32(C + 2 * ldc);
    int32x4_t v_c3 = vld1q_s32(C + 3 * ldc);

    for (int k = 0; k < K; ++k) {
        uint8x16_t v_a = vld1q_u8(A);
        int8x16_t v_b = vld1q_s8(B);

        v_c0 = vsudotq_laneq_s32(v_c0, v_b, v_a, 0);
        v_c1 = vsudotq_laneq_s32(v_c1, v_b, v_a, 1);
        v_c2 = vsudotq_laneq_s32(v_c2, v_b, v_a, 2);
        v_c3 = vsudotq_laneq_s32(v_c3, v_b, v_a, 3);

        A += 16;
        B += 16;
    }

    int32x4_t v_min = vdupq_n_s32(clamp_min);
    v_c0 = vmaxq_s32(v_c0, v_min);
    v_c1 = vmaxq_s32(v_c1, v_min);
    v_c2 = vmaxq_s32(v_c2, v_min);
    v_c3 = vmaxq_s32(v_c3, v_min);

    int32x4_t v_max = vdupq_n_s32(clamp_max);
    v_c0 = vminq_s32(v_c0, v_max);
    v_c1 = vminq_s32(v_c1, v_max);
    v_c2 = vminq_s32(v_c2, v_max);
    v_c3 = vminq_s32(v_c3, v_max);

    vst1q_s32(C + 0 * ldc, v_c0);
    vst1q_s32(C + 1 * ldc, v_c1);
    vst1q_s32(C + 2 * ldc, v_c2);
    vst1q_s32(C + 3 * ldc, v_c3);
}

inline void mma_pack_4x8_u8s8_dot(int32_t* NNOPS_RESTRICT C, int ldc,
                                  const uint8_t* NNOPS_RESTRICT A,
                                  const int8_t* NNOPS_RESTRICT B, int K,
                                  int32_t clamp_min, int32_t clamp_max) noexcept {
    int32x4_t v_c00 = vld1q_s32(C + 0 * ldc + 0);
    int32x4_t v_c01 = vld1q_s32(C + 0 * ldc + 4);
    int32x4_t v_c10 = vld1q_s32(C + 1 * ldc + 0);
    int32x4_t v_c11 = vld1q_s32(C + 1 * ldc + 4);
    int32x4_t v_c20 = vld1q_s32(C + 2 * ldc + 0);
    int32x4_t v_c21 = vld1q_s32(C + 2 * ldc + 4);
    int32x4_t v_c30 = vld1q_s32(C + 3 * ldc + 0);
    int32x4_t v_c31 = vld1q_s32(C + 3 * ldc + 4);

    for (int k = 0; k < K; ++k) {
        uint8x16_t v_a0 = vld1q_u8(A + 0 * 16);

        int8x16_t v_b0 = vld1q_s8(B + 0 * 16);
        int8x16_t v_b1 = vld1q_s8(B + 1 * 16);

        v_c00 = vsudotq_laneq_s32(v_c00, v_b0, v_a0, 0);
        v_c10 = vsudotq_laneq_s32(v_c10, v_b0, v_a0, 1);
        v_c20 = vsudotq_laneq_s32(v_c20, v_b0, v_a0, 2);
        v_c30 = vsudotq_laneq_s32(v_c30, v_b0, v_a0, 3);

        v_c01 = vsudotq_laneq_s32(v_c01, v_b1, v_a0, 0);
        v_c11 = vsudotq_laneq_s32(v_c11, v_b1, v_a0, 1);
        v_c21 = vsudotq_laneq_s32(v_c21, v_b1, v_a0, 2);
        v_c31 = vsudotq_laneq_s32(v_c31, v_b1, v_a0, 3);

        A += 16;
        B += 32;
    }

    int32x4_t v_min = vdupq_n_s32(clamp_min);
    int32x4_t v_max = vdupq_n_s32(clamp_max);

    v_c00 = vminq_s32(vmaxq_s32(v_c00, v_min), v_max);
    v_c01 = vminq_s32(vmaxq_s32(v_c01, v_min), v_max);
    v_c10 = vminq_s32(vmaxq_s32(v_c10, v_min), v_max);
    v_c11 = vminq_s32(vmaxq_s32(v_c11, v_min), v_max);
    v_c20 = vminq_s32(vmaxq_s32(v_c20, v_min), v_max);
    v_c21 = vminq_s32(vmaxq_s32(v_c21, v_min), v_max);
    v_c30 = vminq_s32(vmaxq_s32(v_c30, v_min), v_max);
    v_c31 = vminq_s32(vmaxq_s32(v_c31, v_min), v_max);

    vst1q_s32(C + 0 * ldc + 0, v_c00);
    vst1q_s32(C + 0 * ldc + 4, v_c01);
    vst1q_s32(C + 1 * ldc + 0, v_c10);
    vst1q_s32(C + 1 * ldc + 4, v_c11);
    vst1q_s32(C + 2 * ldc + 0, v_c20);
    vst1q_s32(C + 2 * ldc + 4, v_c21);
    vst1q_s32(C + 3 * ldc + 0, v_c30);
    vst1q_s32(C + 3 * ldc + 4, v_c31);
}

inline void mma_pack_4x12_u8s8_dot(int32_t* NNOPS_RESTRICT C, int ldc,
                                   const uint8_t* NNOPS_RESTRICT A,
                                   const int8_t* NNOPS_RESTRICT B,
                                   int K, int32_t clamp_min, int32_t clamp_max) noexcept {
    int32x4_t v_c00 = vld1q_s32(C + 0 * ldc + 0);
    int32x4_t v_c01 = vld1q_s32(C + 0 * ldc + 4);
    int32x4_t v_c02 = vld1q_s32(C + 0 * ldc + 8);
    int32x4_t v_c10 = vld1q_s32(C + 1 * ldc + 0);
    int32x4_t v_c11 = vld1q_s32(C + 1 * ldc + 4);
    int32x4_t v_c12 = vld1q_s32(C + 1 * ldc + 8);
    int32x4_t v_c20 = vld1q_s32(C + 2 * ldc + 0);
    int32x4_t v_c21 = vld1q_s32(C + 2 * ldc + 4);
    int32x4_t v_c22 = vld1q_s32(C + 2 * ldc + 8);
    int32x4_t v_c30 = vld1q_s32(C + 3 * ldc + 0);
    int32x4_t v_c31 = vld1q_s32(C + 3 * ldc + 4);
    int32x4_t v_c32 = vld1q_s32(C + 3 * ldc + 8);

    for (int k = 0; k < K; ++k) {
        uint8x16_t v_a = vld1q_u8(A);

        int8x16_t v_b0 = vld1q_s8(B + 0 * 16);
        int8x16_t v_b1 = vld1q_s8(B + 1 * 16);
        int8x16_t v_b2 = vld1q_s8(B + 2 * 16);

        v_c00 = vsudotq_laneq_s32(v_c00, v_b0, v_a, 0);
        v_c10 = vsudotq_laneq_s32(v_c10, v_b0, v_a, 1);
        v_c20 = vsudotq_laneq_s32(v_c20, v_b0, v_a, 2);
        v_c30 = vsudotq_laneq_s32(v_c30, v_b0, v_a, 3);

        v_c01 = vsudotq_laneq_s32(v_c01, v_b1, v_a, 0);
        v_c11 = vsudotq_laneq_s32(v_c11, v_b1, v_a, 1);
        v_c21 = vsudotq_laneq_s32(v_c21, v_b1, v_a, 2);
        v_c31 = vsudotq_laneq_s32(v_c31, v_b1, v_a, 3);

        v_c02 = vsudotq_laneq_s32(v_c02, v_b2, v_a, 0);
        v_c12 = vsudotq_laneq_s32(v_c12, v_b2, v_a, 1);
        v_c22 = vsudotq_laneq_s32(v_c22, v_b2, v_a, 2);
        v_c32 = vsudotq_laneq_s32(v_c32, v_b2, v_a, 3);

        A += 16;
        B += 48;
    }

    int32x4_t v_min = vdupq_n_s32(clamp_min);
    v_c00 = vmaxq_s32(v_c00, v_min);
    v_c01 = vmaxq_s32(v_c01, v_min);
    v_c02 = vmaxq_s32(v_c02, v_min);
    v_c10 = vmaxq_s32(v_c10, v_min);
    v_c11 = vmaxq_s32(v_c11, v_min);
    v_c12 = vmaxq_s32(v_c12, v_min);
    v_c20 = vmaxq_s32(v_c20, v_min);
    v_c21 = vmaxq_s32(v_c21, v_min);
    v_c22 = vmaxq_s32(v_c22, v_min);
    v_c30 = vmaxq_s32(v_c30, v_min);
    v_c31 = vmaxq_s32(v_c31, v_min);
    v_c32 = vmaxq_s32(v_c32, v_min);

    int32x4_t v_max = vdupq_n_s32(clamp_max);
    v_c00 = vminq_s32(v_c00, v_max);
    v_c01 = vminq_s32(v_c01, v_max);
    v_c02 = vminq_s32(v_c02, v_max);
    v_c10 = vminq_s32(v_c10, v_max);
    v_c11 = vminq_s32(v_c11, v_max);
    v_c12 = vminq_s32(v_c12, v_max);
    v_c20 = vminq_s32(v_c20, v_max);
    v_c21 = vminq_s32(v_c21, v_max);
    v_c22 = vminq_s32(v_c22, v_max);
    v_c30 = vminq_s32(v_c30, v_max);
    v_c31 = vminq_s32(v_c31, v_max);
    v_c32 = vminq_s32(v_c32, v_max);

    vst1q_s32(C + 0 * ldc + 0, v_c00);
    vst1q_s32(C + 0 * ldc + 4, v_c01);
    vst1q_s32(C + 0 * ldc + 8, v_c02);
    vst1q_s32(C + 1 * ldc + 0, v_c10);
    vst1q_s32(C + 1 * ldc + 4, v_c11);
    vst1q_s32(C + 1 * ldc + 8, v_c12);
    vst1q_s32(C + 2 * ldc + 0, v_c20);
    vst1q_s32(C + 2 * ldc + 4, v_c21);
    vst1q_s32(C + 2 * ldc + 8, v_c22);
    vst1q_s32(C + 3 * ldc + 0, v_c30);
    vst1q_s32(C + 3 * ldc + 4, v_c31);
    vst1q_s32(C + 3 * ldc + 8, v_c32);
}

inline void mma_pack_8x1_u8s8_dot(int32_t* NNOPS_RESTRICT C, int ldc,
                                  const uint8_t* NNOPS_RESTRICT A,
                                  const int8_t* NNOPS_RESTRICT B,
                                  int K,
                                  int32_t clamp_min, int32_t clamp_max) noexcept {
    int32_t c0 = C[0 * ldc];
    int32_t c1 = C[1 * ldc];
    int32_t c2 = C[2 * ldc];
    int32_t c3 = C[3 * ldc];
    int32_t c4 = C[4 * ldc];
    int32_t c5 = C[5 * ldc];
    int32_t c6 = C[6 * ldc];
    int32_t c7 = C[7 * ldc];

    for (int k = 0; k < K; ++k) {
        c0 += (A[4 * 0 + 0] * B[0] + A[4 * 0 + 1] * B[1] + A[4 * 0 + 2] * B[2] + A[4 * 0 + 3] * B[3]);
        c1 += (A[4 * 1 + 0] * B[0] + A[4 * 1 + 1] * B[1] + A[4 * 1 + 2] * B[2] + A[4 * 1 + 3] * B[3]);
        c2 += (A[4 * 2 + 0] * B[0] + A[4 * 2 + 1] * B[1] + A[4 * 2 + 2] * B[2] + A[4 * 2 + 3] * B[3]);
        c3 += (A[4 * 3 + 0] * B[0] + A[4 * 3 + 1] * B[1] + A[4 * 3 + 2] * B[2] + A[4 * 3 + 3] * B[3]);
        c4 += (A[4 * 4 + 0] * B[0] + A[4 * 4 + 1] * B[1] + A[4 * 4 + 2] * B[2] + A[4 * 4 + 3] * B[3]);
        c5 += (A[4 * 5 + 0] * B[0] + A[4 * 5 + 1] * B[1] + A[4 * 5 + 2] * B[2] + A[4 * 5 + 3] * B[3]);
        c6 += (A[4 * 6 + 0] * B[0] + A[4 * 6 + 1] * B[1] + A[4 * 6 + 2] * B[2] + A[4 * 6 + 3] * B[3]);
        c7 += (A[4 * 7 + 0] * B[0] + A[4 * 7 + 1] * B[1] + A[4 * 7 + 2] * B[2] + A[4 * 7 + 3] * B[3]);

        A += 32;
        B += 4;
    }

    C[0 * ldc] = std::min(std::max(c0, clamp_min), clamp_max);
    C[1 * ldc] = std::min(std::max(c1, clamp_min), clamp_max);
    C[2 * ldc] = std::min(std::max(c2, clamp_min), clamp_max);
    C[3 * ldc] = std::min(std::max(c3, clamp_min), clamp_max);
    C[4 * ldc] = std::min(std::max(c4, clamp_min), clamp_max);
    C[5 * ldc] = std::min(std::max(c5, clamp_min), clamp_max);
    C[6 * ldc] = std::min(std::max(c6, clamp_min), clamp_max);
    C[7 * ldc] = std::min(std::max(c7, clamp_min), clamp_max);
}

inline void mma_pack_8x4_u8s8_dot(int32_t* NNOPS_RESTRICT C, int ldc,
                                  const uint8_t* NNOPS_RESTRICT A,
                                  const int8_t* NNOPS_RESTRICT B,
                                  int K,
                                  int32_t clamp_min, int32_t clamp_max) noexcept {
    int32x4_t v_c0 = vld1q_s32(C + 0 * ldc);
    int32x4_t v_c1 = vld1q_s32(C + 1 * ldc);
    int32x4_t v_c2 = vld1q_s32(C + 2 * ldc);
    int32x4_t v_c3 = vld1q_s32(C + 3 * ldc);
    int32x4_t v_c4 = vld1q_s32(C + 4 * ldc);
    int32x4_t v_c5 = vld1q_s32(C + 5 * ldc);
    int32x4_t v_c6 = vld1q_s32(C + 6 * ldc);
    int32x4_t v_c7 = vld1q_s32(C + 7 * ldc);

    for (int k = 0; k < K; ++k) {
        uint8x16_t v_a0 = vld1q_u8(A + 0 * 16);
        uint8x16_t v_a1 = vld1q_u8(A + 1 * 16);
        int8x16_t v_b = vld1q_s8(B);

        v_c0 = vsudotq_laneq_s32(v_c0, v_b, v_a0, 0);
        v_c1 = vsudotq_laneq_s32(v_c1, v_b, v_a0, 1);
        v_c2 = vsudotq_laneq_s32(v_c2, v_b, v_a0, 2);
        v_c3 = vsudotq_laneq_s32(v_c3, v_b, v_a0, 3);

        v_c4 = vsudotq_laneq_s32(v_c4, v_b, v_a1, 0);
        v_c5 = vsudotq_laneq_s32(v_c5, v_b, v_a1, 1);
        v_c6 = vsudotq_laneq_s32(v_c6, v_b, v_a1, 2);
        v_c7 = vsudotq_laneq_s32(v_c7, v_b, v_a1, 3);

        A += 32;
        B += 16;
    }

    int32x4_t v_min = vdupq_n_s32(clamp_min);
    v_c0 = vmaxq_s32(v_c0, v_min);
    v_c1 = vmaxq_s32(v_c1, v_min);
    v_c2 = vmaxq_s32(v_c2, v_min);
    v_c3 = vmaxq_s32(v_c3, v_min);
    v_c4 = vmaxq_s32(v_c4, v_min);
    v_c5 = vmaxq_s32(v_c5, v_min);
    v_c6 = vmaxq_s32(v_c6, v_min);
    v_c7 = vmaxq_s32(v_c7, v_min);

    int32x4_t v_max = vdupq_n_s32(clamp_max);
    v_c0 = vminq_s32(v_c0, v_max);
    v_c1 = vminq_s32(v_c1, v_max);
    v_c2 = vminq_s32(v_c2, v_max);
    v_c3 = vminq_s32(v_c3, v_max);
    v_c4 = vminq_s32(v_c4, v_max);
    v_c5 = vminq_s32(v_c5, v_max);
    v_c6 = vminq_s32(v_c6, v_max);
    v_c7 = vminq_s32(v_c7, v_max);

    vst1q_s32(C + 0 * ldc, v_c0);
    vst1q_s32(C + 1 * ldc, v_c1);
    vst1q_s32(C + 2 * ldc, v_c2);
    vst1q_s32(C + 3 * ldc, v_c3);
    vst1q_s32(C + 4 * ldc, v_c4);
    vst1q_s32(C + 5 * ldc, v_c5);
    vst1q_s32(C + 6 * ldc, v_c6);
    vst1q_s32(C + 7 * ldc, v_c7);
}

inline void mma_pack_8x8_u8s8_dot(int32_t* NNOPS_RESTRICT C, int ldc,
                                  const uint8_t* NNOPS_RESTRICT A,
                                  const int8_t* NNOPS_RESTRICT B,
                                  int K,
                                  int32_t clamp_min, int32_t clamp_max) noexcept {
    int32x4_t v_c00 = vld1q_s32(C + 0 * ldc + 0);
    int32x4_t v_c01 = vld1q_s32(C + 0 * ldc + 4);
    int32x4_t v_c10 = vld1q_s32(C + 1 * ldc + 0);
    int32x4_t v_c11 = vld1q_s32(C + 1 * ldc + 4);
    int32x4_t v_c20 = vld1q_s32(C + 2 * ldc + 0);
    int32x4_t v_c21 = vld1q_s32(C + 2 * ldc + 4);
    int32x4_t v_c30 = vld1q_s32(C + 3 * ldc + 0);
    int32x4_t v_c31 = vld1q_s32(C + 3 * ldc + 4);
    int32x4_t v_c40 = vld1q_s32(C + 4 * ldc + 0);
    int32x4_t v_c41 = vld1q_s32(C + 4 * ldc + 4);
    int32x4_t v_c50 = vld1q_s32(C + 5 * ldc + 0);
    int32x4_t v_c51 = vld1q_s32(C + 5 * ldc + 4);
    int32x4_t v_c60 = vld1q_s32(C + 6 * ldc + 0);
    int32x4_t v_c61 = vld1q_s32(C + 6 * ldc + 4);
    int32x4_t v_c70 = vld1q_s32(C + 7 * ldc + 0);
    int32x4_t v_c71 = vld1q_s32(C + 7 * ldc + 4);

    for (int k = 0; k < K; ++k) {
        uint8x16_t v_a0 = vld1q_u8(A + 0 * 16);
        uint8x16_t v_a1 = vld1q_u8(A + 1 * 16);

        int8x16_t v_b0 = vld1q_s8(B + 0 * 16);
        int8x16_t v_b1 = vld1q_s8(B + 1 * 16);

        v_c00 = vsudotq_laneq_s32(v_c00, v_b0, v_a0, 0);
        v_c10 = vsudotq_laneq_s32(v_c10, v_b0, v_a0, 1);
        v_c20 = vsudotq_laneq_s32(v_c20, v_b0, v_a0, 2);
        v_c30 = vsudotq_laneq_s32(v_c30, v_b0, v_a0, 3);

        v_c01 = vsudotq_laneq_s32(v_c01, v_b1, v_a0, 0);
        v_c11 = vsudotq_laneq_s32(v_c11, v_b1, v_a0, 1);
        v_c21 = vsudotq_laneq_s32(v_c21, v_b1, v_a0, 2);
        v_c31 = vsudotq_laneq_s32(v_c31, v_b1, v_a0, 3);

        v_c40 = vsudotq_laneq_s32(v_c40, v_b0, v_a1, 0);
        v_c50 = vsudotq_laneq_s32(v_c50, v_b0, v_a1, 1);
        v_c60 = vsudotq_laneq_s32(v_c60, v_b0, v_a1, 2);
        v_c70 = vsudotq_laneq_s32(v_c70, v_b0, v_a1, 3);

        v_c41 = vsudotq_laneq_s32(v_c41, v_b1, v_a1, 0);
        v_c51 = vsudotq_laneq_s32(v_c51, v_b1, v_a1, 1);
        v_c61 = vsudotq_laneq_s32(v_c61, v_b1, v_a1, 2);
        v_c71 = vsudotq_laneq_s32(v_c71, v_b1, v_a1, 3);

        A += 32;
        B += 32;
    }

    int32x4_t v_min = vdupq_n_s32(clamp_min);
    int32x4_t v_max = vdupq_n_s32(clamp_max);

    v_c00 = vminq_s32(vmaxq_s32(v_c00, v_min), v_max);
    v_c01 = vminq_s32(vmaxq_s32(v_c01, v_min), v_max);
    v_c10 = vminq_s32(vmaxq_s32(v_c10, v_min), v_max);
    v_c11 = vminq_s32(vmaxq_s32(v_c11, v_min), v_max);
    v_c20 = vminq_s32(vmaxq_s32(v_c20, v_min), v_max);
    v_c21 = vminq_s32(vmaxq_s32(v_c21, v_min), v_max);
    v_c30 = vminq_s32(vmaxq_s32(v_c30, v_min), v_max);
    v_c31 = vminq_s32(vmaxq_s32(v_c31, v_min), v_max);

    vst1q_s32(C + 0 * ldc + 0, v_c00);
    vst1q_s32(C + 0 * ldc + 4, v_c01);
    vst1q_s32(C + 1 * ldc + 0, v_c10);
    vst1q_s32(C + 1 * ldc + 4, v_c11);
    vst1q_s32(C + 2 * ldc + 0, v_c20);
    vst1q_s32(C + 2 * ldc + 4, v_c21);
    vst1q_s32(C + 3 * ldc + 0, v_c30);
    vst1q_s32(C + 3 * ldc + 4, v_c31);

    v_c40 = vminq_s32(vmaxq_s32(v_c40, v_min), v_max);
    v_c41 = vminq_s32(vmaxq_s32(v_c41, v_min), v_max);
    v_c50 = vminq_s32(vmaxq_s32(v_c50, v_min), v_max);
    v_c51 = vminq_s32(vmaxq_s32(v_c51, v_min), v_max);
    v_c60 = vminq_s32(vmaxq_s32(v_c60, v_min), v_max);
    v_c61 = vminq_s32(vmaxq_s32(v_c61, v_min), v_max);
    v_c70 = vminq_s32(vmaxq_s32(v_c70, v_min), v_max);
    v_c71 = vminq_s32(vmaxq_s32(v_c71, v_min), v_max);

    vst1q_s32(C + 4 * ldc + 0, v_c40);
    vst1q_s32(C + 4 * ldc + 4, v_c41);
    vst1q_s32(C + 5 * ldc + 0, v_c50);
    vst1q_s32(C + 5 * ldc + 4, v_c51);
    vst1q_s32(C + 6 * ldc + 0, v_c60);
    vst1q_s32(C + 6 * ldc + 4, v_c61);
    vst1q_s32(C + 7 * ldc + 0, v_c70);
    vst1q_s32(C + 7 * ldc + 4, v_c71);
}

inline void mma_pack_8x12_u8s8_dot(int32_t* NNOPS_RESTRICT C, int ldc,
                                   const uint8_t* NNOPS_RESTRICT A,
                                   const int8_t* NNOPS_RESTRICT B,
                                   int K,
                                   int32_t clamp_min, int32_t clamp_max) noexcept {
    int32x4_t v_c00 = vld1q_s32(C + 0 * ldc + 0);
    int32x4_t v_c01 = vld1q_s32(C + 0 * ldc + 4);
    int32x4_t v_c02 = vld1q_s32(C + 0 * ldc + 8);
    int32x4_t v_c10 = vld1q_s32(C + 1 * ldc + 0);
    int32x4_t v_c11 = vld1q_s32(C + 1 * ldc + 4);
    int32x4_t v_c12 = vld1q_s32(C + 1 * ldc + 8);
    int32x4_t v_c20 = vld1q_s32(C + 2 * ldc + 0);
    int32x4_t v_c21 = vld1q_s32(C + 2 * ldc + 4);
    int32x4_t v_c22 = vld1q_s32(C + 2 * ldc + 8);
    int32x4_t v_c30 = vld1q_s32(C + 3 * ldc + 0);
    int32x4_t v_c31 = vld1q_s32(C + 3 * ldc + 4);
    int32x4_t v_c32 = vld1q_s32(C + 3 * ldc + 8);
    int32x4_t v_c40 = vld1q_s32(C + 4 * ldc + 0);
    int32x4_t v_c41 = vld1q_s32(C + 4 * ldc + 4);
    int32x4_t v_c42 = vld1q_s32(C + 4 * ldc + 8);
    int32x4_t v_c50 = vld1q_s32(C + 5 * ldc + 0);
    int32x4_t v_c51 = vld1q_s32(C + 5 * ldc + 4);
    int32x4_t v_c52 = vld1q_s32(C + 5 * ldc + 8);
    int32x4_t v_c60 = vld1q_s32(C + 6 * ldc + 0);
    int32x4_t v_c61 = vld1q_s32(C + 6 * ldc + 4);
    int32x4_t v_c62 = vld1q_s32(C + 6 * ldc + 8);
    int32x4_t v_c70 = vld1q_s32(C + 7 * ldc + 0);
    int32x4_t v_c71 = vld1q_s32(C + 7 * ldc + 4);
    int32x4_t v_c72 = vld1q_s32(C + 7 * ldc + 8);

    for (int k = 0; k < K; ++k) {
        uint8x16_t v_a0 = vld1q_u8(A + 0 * 16);
        uint8x16_t v_a1 = vld1q_u8(A + 1 * 16);

        int8x16_t v_b0 = vld1q_s8(B + 0 * 16);

        v_c00 = vsudotq_laneq_s32(v_c00, v_b0, v_a0, 0);
        v_c10 = vsudotq_laneq_s32(v_c10, v_b0, v_a0, 1);
        v_c20 = vsudotq_laneq_s32(v_c20, v_b0, v_a0, 2);
        v_c30 = vsudotq_laneq_s32(v_c30, v_b0, v_a0, 3);

        v_c40 = vsudotq_laneq_s32(v_c40, v_b0, v_a1, 0);
        v_c50 = vsudotq_laneq_s32(v_c50, v_b0, v_a1, 1);
        v_c60 = vsudotq_laneq_s32(v_c60, v_b0, v_a1, 2);
        v_c70 = vsudotq_laneq_s32(v_c70, v_b0, v_a1, 3);

        int8x16_t v_b1 = vld1q_s8(B + 1 * 16);

        v_c01 = vsudotq_laneq_s32(v_c01, v_b1, v_a0, 0);
        v_c11 = vsudotq_laneq_s32(v_c11, v_b1, v_a0, 1);
        v_c21 = vsudotq_laneq_s32(v_c21, v_b1, v_a0, 2);
        v_c31 = vsudotq_laneq_s32(v_c31, v_b1, v_a0, 3);

        v_c41 = vsudotq_laneq_s32(v_c41, v_b1, v_a1, 0);
        v_c51 = vsudotq_laneq_s32(v_c51, v_b1, v_a1, 1);
        v_c61 = vsudotq_laneq_s32(v_c61, v_b1, v_a1, 2);
        v_c71 = vsudotq_laneq_s32(v_c71, v_b1, v_a1, 3);

        int8x16_t v_b2 = vld1q_s8(B + 2 * 16);

        v_c02 = vsudotq_laneq_s32(v_c02, v_b2, v_a0, 0);
        v_c12 = vsudotq_laneq_s32(v_c12, v_b2, v_a0, 1);
        v_c22 = vsudotq_laneq_s32(v_c22, v_b2, v_a0, 2);
        v_c32 = vsudotq_laneq_s32(v_c32, v_b2, v_a0, 3);

        v_c42 = vsudotq_laneq_s32(v_c42, v_b2, v_a1, 0);
        v_c52 = vsudotq_laneq_s32(v_c52, v_b2, v_a1, 1);
        v_c62 = vsudotq_laneq_s32(v_c62, v_b2, v_a1, 2);
        v_c72 = vsudotq_laneq_s32(v_c72, v_b2, v_a1, 3);

        A += 32;
        B += 48;
    }

    int32x4_t v_min = vdupq_n_s32(clamp_min);
    int32x4_t v_max = vdupq_n_s32(clamp_max);

    v_c00 = vminq_s32(vmaxq_s32(v_c00, v_min), v_max);
    v_c01 = vminq_s32(vmaxq_s32(v_c01, v_min), v_max);
    v_c02 = vminq_s32(vmaxq_s32(v_c02, v_min), v_max);
    v_c10 = vminq_s32(vmaxq_s32(v_c10, v_min), v_max);
    v_c11 = vminq_s32(vmaxq_s32(v_c11, v_min), v_max);
    v_c12 = vminq_s32(vmaxq_s32(v_c12, v_min), v_max);
    v_c20 = vminq_s32(vmaxq_s32(v_c20, v_min), v_max);
    v_c21 = vminq_s32(vmaxq_s32(v_c21, v_min), v_max);
    v_c22 = vminq_s32(vmaxq_s32(v_c22, v_min), v_max);
    v_c30 = vminq_s32(vmaxq_s32(v_c30, v_min), v_max);
    v_c31 = vminq_s32(vmaxq_s32(v_c31, v_min), v_max);
    v_c32 = vminq_s32(vmaxq_s32(v_c32, v_min), v_max);

    vst1q_s32(C + 0 * ldc + 0, v_c00);
    vst1q_s32(C + 0 * ldc + 4, v_c01);
    vst1q_s32(C + 0 * ldc + 8, v_c02);
    vst1q_s32(C + 1 * ldc + 0, v_c10);
    vst1q_s32(C + 1 * ldc + 4, v_c11);
    vst1q_s32(C + 1 * ldc + 8, v_c12);
    vst1q_s32(C + 2 * ldc + 0, v_c20);
    vst1q_s32(C + 2 * ldc + 4, v_c21);
    vst1q_s32(C + 2 * ldc + 8, v_c22);
    vst1q_s32(C + 3 * ldc + 0, v_c30);
    vst1q_s32(C + 3 * ldc + 4, v_c31);
    vst1q_s32(C + 3 * ldc + 8, v_c32);

    v_c40 = vminq_s32(vmaxq_s32(v_c40, v_min), v_max);
    v_c41 = vminq_s32(vmaxq_s32(v_c41, v_min), v_max);
    v_c42 = vminq_s32(vmaxq_s32(v_c42, v_min), v_max);
    v_c50 = vminq_s32(vmaxq_s32(v_c50, v_min), v_max);
    v_c51 = vminq_s32(vmaxq_s32(v_c51, v_min), v_max);
    v_c52 = vminq_s32(vmaxq_s32(v_c52, v_min), v_max);
    v_c60 = vminq_s32(vmaxq_s32(v_c60, v_min), v_max);
    v_c61 = vminq_s32(vmaxq_s32(v_c61, v_min), v_max);
    v_c62 = vminq_s32(vmaxq_s32(v_c62, v_min), v_max);
    v_c70 = vminq_s32(vmaxq_s32(v_c70, v_min), v_max);
    v_c71 = vminq_s32(vmaxq_s32(v_c71, v_min), v_max);
    v_c72 = vminq_s32(vmaxq_s32(v_c72, v_min), v_max);

    vst1q_s32(C + 4 * ldc + 0, v_c40);
    vst1q_s32(C + 4 * ldc + 4, v_c41);
    vst1q_s32(C + 4 * ldc + 8, v_c42);
    vst1q_s32(C + 5 * ldc + 0, v_c50);
    vst1q_s32(C + 5 * ldc + 4, v_c51);
    vst1q_s32(C + 5 * ldc + 8, v_c52);
    vst1q_s32(C + 6 * ldc + 0, v_c60);
    vst1q_s32(C + 6 * ldc + 4, v_c61);
    vst1q_s32(C + 6 * ldc + 8, v_c62);
    vst1q_s32(C + 7 * ldc + 0, v_c70);
    vst1q_s32(C + 7 * ldc + 4, v_c71);
    vst1q_s32(C + 7 * ldc + 8, v_c72);
}

}  // namespace nnops::backend::cpu::aarch64
