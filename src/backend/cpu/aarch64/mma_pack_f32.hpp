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

template <bool zero_mode = false>
inline void mma_pack_1x1_f32(float* NNOPS_RESTRICT C, int ldc,
                             const float* NNOPS_RESTRICT A,
                             const float* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    float c0 = 0.0f;
    for (int k = 0; k < K; ++k) {
        c0 += (*A++) * B[0];
        B += ldb;
    }
    if constexpr (!zero_mode) { c0 += C[0 * ldc]; }
    C[0 * ldc] = std::min(std::max(c0, clamp_min), clamp_max);
}

template <bool zero_mode = false>
inline void mma_pack_1x4_f32(float* NNOPS_RESTRICT C, int ldc,
                             const float* NNOPS_RESTRICT A,
                             const float* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    float32x4_t c0 = vdupq_n_f32(0.0f);
    for (int k = 0; k < K; ++k) {
        c0 = vfmaq_n_f32(c0, vld1q_f32(B), *A++);
        B += ldb;
    }
    if constexpr (!zero_mode) { c0 = vaddq_f32(c0, vld1q_f32(C)); }
    const float32x4_t vmin = vdupq_n_f32(clamp_min);
    const float32x4_t vmax = vdupq_n_f32(clamp_max);
    vst1q_f32(C, vminq_f32(vmaxq_f32(c0, vmin), vmax));
}

template <bool zero_mode = false>
inline void mma_pack_1x12_f32(float* NNOPS_RESTRICT C, int ldc,
                              const float* NNOPS_RESTRICT A,
                              const float* NNOPS_RESTRICT B,
                              int ldb, int K,
                              float clamp_min, float clamp_max) noexcept {
    float32x4_t c0 = vdupq_n_f32(0.0f);
    float32x4_t c1 = vdupq_n_f32(0.0f);
    float32x4_t c2 = vdupq_n_f32(0.0f);

    for (int k = 0; k < K; ++k) {
        const float a = *A++;
        c0 = vfmaq_n_f32(c0, vld1q_f32(B + 0 * 4), a);
        c1 = vfmaq_n_f32(c1, vld1q_f32(B + 1 * 4), a);
        c2 = vfmaq_n_f32(c2, vld1q_f32(B + 2 * 4), a);
        B += ldb;
    }

    const float32x4_t vmin = vdupq_n_f32(clamp_min);
    const float32x4_t vmax = vdupq_n_f32(clamp_max);

    if constexpr (!zero_mode) { c0 = vaddq_f32(c0, vld1q_f32(C + 0 * 4)); 
    c1 = vaddq_f32(c1, vld1q_f32(C + 1 * 4)); 
    c2 = vaddq_f32(c2, vld1q_f32(C + 2 * 4)); }

    vst1q_f32(C + 0 * 4, vminq_f32(vmaxq_f32(c0, vmin), vmax));
    vst1q_f32(C + 1 * 4, vminq_f32(vmaxq_f32(c1, vmin), vmax));
    vst1q_f32(C + 2 * 4, vminq_f32(vmaxq_f32(c2, vmin), vmax));
}

// =========================================================================
//  mr=4  kernels
// =========================================================================

template <bool zero_mode = false>
inline void mma_pack_4x1_f32(float* NNOPS_RESTRICT C, int ldc,
                             const float* NNOPS_RESTRICT A,
                             const float* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float b = B[0]; B += ldb;
        c0 += A[0] * b;
        c1 += A[1] * b;
        c2 += A[2] * b;
        c3 += A[3] * b;
        A += 4;
    }
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c0 += C[0 * ldc]; } } c0 = std::min(std::max(c0, clamp_min), clamp_max);
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c1 += C[1 * ldc]; } } c1 = std::min(std::max(c1, clamp_min), clamp_max);
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c2 += C[2 * ldc]; } } c2 = std::min(std::max(c2, clamp_min), clamp_max);
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c3 += C[3 * ldc]; } } c3 = std::min(std::max(c3, clamp_min), clamp_max);
    C[0 * ldc] = c0;
    C[1 * ldc] = c1;
    C[2 * ldc] = c2;
    C[3 * ldc] = c3;
}

template <bool zero_mode = false>
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
        const float32x4_t a = vld1q_f32(A); A += 4;
        const float32x4_t b = vld1q_f32(B); B += ldb;

        c0 = vfmaq_laneq_f32(c0, b, a, 0);
        c1 = vfmaq_laneq_f32(c1, b, a, 1);
        c2 = vfmaq_laneq_f32(c2, b, a, 2);
        c3 = vfmaq_laneq_f32(c3, b, a, 3);
    }

    const float32x4_t vmin = vdupq_n_f32(clamp_min);
    const float32x4_t vmax = vdupq_n_f32(clamp_max);

    if constexpr (!zero_mode) { c0 = vaddq_f32(c0, vld1q_f32(C + 0 * ldc)); 
    c1 = vaddq_f32(c1, vld1q_f32(C + 1 * ldc)); 
    c2 = vaddq_f32(c2, vld1q_f32(C + 2 * ldc)); 
    c3 = vaddq_f32(c3, vld1q_f32(C + 3 * ldc)); }

    vst1q_f32(C + 0 * ldc, vminq_f32(vmaxq_f32(c0, vmin), vmax));
    vst1q_f32(C + 1 * ldc, vminq_f32(vmaxq_f32(c1, vmin), vmax));
    vst1q_f32(C + 2 * ldc, vminq_f32(vmaxq_f32(c2, vmin), vmax));
    vst1q_f32(C + 3 * ldc, vminq_f32(vmaxq_f32(c3, vmin), vmax));
}

template <bool zero_mode = false>
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
        const float32x4_t a = vld1q_f32(A); A += 4;
        const float32x4_t b0 = vld1q_f32(B + 0 * 4);
        const float32x4_t b1 = vld1q_f32(B + 1 * 4);
        const float32x4_t b2 = vld1q_f32(B + 2 * 4);
        B += ldb;

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
    }

    const float32x4_t vmin = vdupq_n_f32(clamp_min);
    const float32x4_t vmax = vdupq_n_f32(clamp_max);

    if constexpr (!zero_mode) { c00 = vaddq_f32(c00, vld1q_f32(C + 0 * 4)); 
    c01 = vaddq_f32(c01, vld1q_f32(C + 1 * 4)); 
    c02 = vaddq_f32(c02, vld1q_f32(C + 2 * 4)); 
    c10 = vaddq_f32(c10, vld1q_f32(C + 1 * ldc + 0 * 4)); 
    c11 = vaddq_f32(c11, vld1q_f32(C + 1 * ldc + 1 * 4)); 
    c12 = vaddq_f32(c12, vld1q_f32(C + 1 * ldc + 2 * 4)); 
    c20 = vaddq_f32(c20, vld1q_f32(C + 2 * ldc + 0 * 4)); 
    c21 = vaddq_f32(c21, vld1q_f32(C + 2 * ldc + 1 * 4)); 
    c22 = vaddq_f32(c22, vld1q_f32(C + 2 * ldc + 2 * 4)); 
    c30 = vaddq_f32(c30, vld1q_f32(C + 3 * ldc + 0 * 4)); 
    c31 = vaddq_f32(c31, vld1q_f32(C + 3 * ldc + 1 * 4)); 
    c32 = vaddq_f32(c32, vld1q_f32(C + 3 * ldc + 2 * 4)); }

    c00 = vminq_f32(vmaxq_f32(c00, vmin), vmax); c01 = vminq_f32(vmaxq_f32(c01, vmin), vmax); c02 = vminq_f32(vmaxq_f32(c02, vmin), vmax);
    c10 = vminq_f32(vmaxq_f32(c10, vmin), vmax); c11 = vminq_f32(vmaxq_f32(c11, vmin), vmax); c12 = vminq_f32(vmaxq_f32(c12, vmin), vmax);
    c20 = vminq_f32(vmaxq_f32(c20, vmin), vmax); c21 = vminq_f32(vmaxq_f32(c21, vmin), vmax); c22 = vminq_f32(vmaxq_f32(c22, vmin), vmax);
    c30 = vminq_f32(vmaxq_f32(c30, vmin), vmax); c31 = vminq_f32(vmaxq_f32(c31, vmin), vmax); c32 = vminq_f32(vmaxq_f32(c32, vmin), vmax);

    vst1q_f32(C + 0 * 4, c00); vst1q_f32(C + 1 * 4, c01); vst1q_f32(C + 2 * 4, c02);
    vst1q_f32(C + 1 * ldc + 0 * 4, c10); vst1q_f32(C + 1 * ldc + 1 * 4, c11); vst1q_f32(C + 1 * ldc + 2 * 4, c12);
    vst1q_f32(C + 2 * ldc + 0 * 4, c20); vst1q_f32(C + 2 * ldc + 1 * 4, c21); vst1q_f32(C + 2 * ldc + 2 * 4, c22);
    vst1q_f32(C + 3 * ldc + 0 * 4, c30); vst1q_f32(C + 3 * ldc + 1 * 4, c31); vst1q_f32(C + 3 * ldc + 2 * 4, c32);
}

// =========================================================================
//  mr=8  kernels
// =========================================================================

template <bool zero_mode = false>
inline void mma_pack_8x1_f32(float* NNOPS_RESTRICT C, int ldc,
                             const float* NNOPS_RESTRICT A,
                             const float* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    float c4 = 0.0f, c5 = 0.0f, c6 = 0.0f, c7 = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float b = B[0]; B += ldb;
        c0 += A[0] * b; c1 += A[1] * b; c2 += A[2] * b; c3 += A[3] * b;
        c4 += A[4] * b; c5 += A[5] * b; c6 += A[6] * b; c7 += A[7] * b;
        A += 8;
    }
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c0 += C[0 * ldc]; } } c0 = std::min(std::max(c0, clamp_min), clamp_max); C[0 * ldc] = c0;
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c1 += C[1 * ldc]; } } c1 = std::min(std::max(c1, clamp_min), clamp_max); C[1 * ldc] = c1;
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c2 += C[2 * ldc]; } } c2 = std::min(std::max(c2, clamp_min), clamp_max); C[2 * ldc] = c2;
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c3 += C[3 * ldc]; } } c3 = std::min(std::max(c3, clamp_min), clamp_max); C[3 * ldc] = c3;
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c4 += C[4 * ldc]; } } c4 = std::min(std::max(c4, clamp_min), clamp_max); C[4 * ldc] = c4;
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c5 += C[5 * ldc]; } } c5 = std::min(std::max(c5, clamp_min), clamp_max); C[5 * ldc] = c5;
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c6 += C[6 * ldc]; } } c6 = std::min(std::max(c6, clamp_min), clamp_max); C[6 * ldc] = c6;
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c7 += C[7 * ldc]; } } c7 = std::min(std::max(c7, clamp_min), clamp_max); C[7 * ldc] = c7;
}

template <bool zero_mode = false>
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
        const float32x4_t b  = vld1q_f32(B); B += ldb;

        c0 = vfmaq_laneq_f32(c0, b, a0, 0);
        c1 = vfmaq_laneq_f32(c1, b, a0, 1);
        c2 = vfmaq_laneq_f32(c2, b, a0, 2);
        c3 = vfmaq_laneq_f32(c3, b, a0, 3);

        const float32x4_t a1 = vld1q_f32(A + 1 * 4);
        A += 8;

        c4 = vfmaq_laneq_f32(c4, b, a1, 0);
        c5 = vfmaq_laneq_f32(c5, b, a1, 1);
        c6 = vfmaq_laneq_f32(c6, b, a1, 2);
        c7 = vfmaq_laneq_f32(c7, b, a1, 3);
    }

    const float32x4_t vmin = vdupq_n_f32(clamp_min);
    const float32x4_t vmax = vdupq_n_f32(clamp_max);

    if constexpr (!zero_mode) { c0 = vaddq_f32(c0, vld1q_f32(C + 0 * ldc)); 
    c1 = vaddq_f32(c1, vld1q_f32(C + 1 * ldc)); 
    c2 = vaddq_f32(c2, vld1q_f32(C + 2 * ldc)); 
    c3 = vaddq_f32(c3, vld1q_f32(C + 3 * ldc)); 
    c4 = vaddq_f32(c4, vld1q_f32(C + 4 * ldc)); 
    c5 = vaddq_f32(c5, vld1q_f32(C + 5 * ldc)); 
    c6 = vaddq_f32(c6, vld1q_f32(C + 6 * ldc)); 
    c7 = vaddq_f32(c7, vld1q_f32(C + 7 * ldc)); }

    vst1q_f32(C + 0 * ldc, vminq_f32(vmaxq_f32(c0, vmin), vmax));
    vst1q_f32(C + 1 * ldc, vminq_f32(vmaxq_f32(c1, vmin), vmax));
    vst1q_f32(C + 2 * ldc, vminq_f32(vmaxq_f32(c2, vmin), vmax));
    vst1q_f32(C + 3 * ldc, vminq_f32(vmaxq_f32(c3, vmin), vmax));
    vst1q_f32(C + 4 * ldc, vminq_f32(vmaxq_f32(c4, vmin), vmax));
    vst1q_f32(C + 5 * ldc, vminq_f32(vmaxq_f32(c5, vmin), vmax));
    vst1q_f32(C + 6 * ldc, vminq_f32(vmaxq_f32(c6, vmin), vmax));
    vst1q_f32(C + 7 * ldc, vminq_f32(vmaxq_f32(c7, vmin), vmax));
}

template <bool zero_mode = false>
inline void mma_pack_8x12_f32(float* NNOPS_RESTRICT C, int ldc,
                              const float* NNOPS_RESTRICT A,
                              const float* NNOPS_RESTRICT B,
                              int ldb, int K,
                              float clamp_min, float clamp_max) noexcept {
    float32x4_t c00 = vdupq_n_f32(0.0f), c01 = c00, c02 = c00;
    float32x4_t c10 = vdupq_n_f32(0.0f), c11 = c10, c12 = c10;
    float32x4_t c20 = vdupq_n_f32(0.0f), c21 = c20, c22 = c20;
    float32x4_t c30 = vdupq_n_f32(0.0f), c31 = c30, c32 = c30;
    float32x4_t c40 = vdupq_n_f32(0.0f), c41 = c40, c42 = c40;
    float32x4_t c50 = vdupq_n_f32(0.0f), c51 = c50, c52 = c50;
    float32x4_t c60 = vdupq_n_f32(0.0f), c61 = c60, c62 = c60;
    float32x4_t c70 = vdupq_n_f32(0.0f), c71 = c70, c72 = c70;

    for (int k = 0; k < K; ++k) {
        const float32x4_t a0 = vld1q_f32(A + 0 * 4);
        const float32x4_t b0 = vld1q_f32(B + 0 * 4);
        const float32x4_t b1 = vld1q_f32(B + 1 * 4);
        const float32x4_t b2 = vld1q_f32(B + 2 * 4);
        B += ldb;

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

        const float32x4_t a1 = vld1q_f32(A + 1 * 4);
        A += 8;
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

    const float32x4_t vmin = vdupq_n_f32(clamp_min);
    const float32x4_t vmax = vdupq_n_f32(clamp_max);

    if constexpr (!zero_mode) { c00 = vaddq_f32(c00, vld1q_f32(C + 0 * 4)); 
    c01 = vaddq_f32(c01, vld1q_f32(C + 1 * 4)); 
    c02 = vaddq_f32(c02, vld1q_f32(C + 2 * 4)); 
    c10 = vaddq_f32(c10, vld1q_f32(C + 1 * ldc + 0 * 4)); 
    c11 = vaddq_f32(c11, vld1q_f32(C + 1 * ldc + 1 * 4)); 
    c12 = vaddq_f32(c12, vld1q_f32(C + 1 * ldc + 2 * 4)); 
    c20 = vaddq_f32(c20, vld1q_f32(C + 2 * ldc + 0 * 4)); 
    c21 = vaddq_f32(c21, vld1q_f32(C + 2 * ldc + 1 * 4)); 
    c22 = vaddq_f32(c22, vld1q_f32(C + 2 * ldc + 2 * 4)); 
    c30 = vaddq_f32(c30, vld1q_f32(C + 3 * ldc + 0 * 4)); 
    c31 = vaddq_f32(c31, vld1q_f32(C + 3 * ldc + 1 * 4)); 
    c32 = vaddq_f32(c32, vld1q_f32(C + 3 * ldc + 2 * 4)); 
    c40 = vaddq_f32(c40, vld1q_f32(C + 4 * ldc + 0 * 4)); 
    c41 = vaddq_f32(c41, vld1q_f32(C + 4 * ldc + 1 * 4)); 
    c42 = vaddq_f32(c42, vld1q_f32(C + 4 * ldc + 2 * 4)); 
    c50 = vaddq_f32(c50, vld1q_f32(C + 5 * ldc + 0 * 4)); 
    c51 = vaddq_f32(c51, vld1q_f32(C + 5 * ldc + 1 * 4)); 
    c52 = vaddq_f32(c52, vld1q_f32(C + 5 * ldc + 2 * 4)); 
    c60 = vaddq_f32(c60, vld1q_f32(C + 6 * ldc + 0 * 4)); 
    c61 = vaddq_f32(c61, vld1q_f32(C + 6 * ldc + 1 * 4)); 
    c62 = vaddq_f32(c62, vld1q_f32(C + 6 * ldc + 2 * 4)); 
    c70 = vaddq_f32(c70, vld1q_f32(C + 7 * ldc + 0 * 4)); 
    c71 = vaddq_f32(c71, vld1q_f32(C + 7 * ldc + 1 * 4)); 
    c72 = vaddq_f32(c72, vld1q_f32(C + 7 * ldc + 2 * 4)); }

    c00 = vminq_f32(vmaxq_f32(c00, vmin), vmax); c01 = vminq_f32(vmaxq_f32(c01, vmin), vmax); c02 = vminq_f32(vmaxq_f32(c02, vmin), vmax);
    c10 = vminq_f32(vmaxq_f32(c10, vmin), vmax); c11 = vminq_f32(vmaxq_f32(c11, vmin), vmax); c12 = vminq_f32(vmaxq_f32(c12, vmin), vmax);
    c20 = vminq_f32(vmaxq_f32(c20, vmin), vmax); c21 = vminq_f32(vmaxq_f32(c21, vmin), vmax); c22 = vminq_f32(vmaxq_f32(c22, vmin), vmax);
    c30 = vminq_f32(vmaxq_f32(c30, vmin), vmax); c31 = vminq_f32(vmaxq_f32(c31, vmin), vmax); c32 = vminq_f32(vmaxq_f32(c32, vmin), vmax);
    c40 = vminq_f32(vmaxq_f32(c40, vmin), vmax); c41 = vminq_f32(vmaxq_f32(c41, vmin), vmax); c42 = vminq_f32(vmaxq_f32(c42, vmin), vmax);
    c50 = vminq_f32(vmaxq_f32(c50, vmin), vmax); c51 = vminq_f32(vmaxq_f32(c51, vmin), vmax); c52 = vminq_f32(vmaxq_f32(c52, vmin), vmax);
    c60 = vminq_f32(vmaxq_f32(c60, vmin), vmax); c61 = vminq_f32(vmaxq_f32(c61, vmin), vmax); c62 = vminq_f32(vmaxq_f32(c62, vmin), vmax);
    c70 = vminq_f32(vmaxq_f32(c70, vmin), vmax); c71 = vminq_f32(vmaxq_f32(c71, vmin), vmax); c72 = vminq_f32(vmaxq_f32(c72, vmin), vmax);

    vst1q_f32(C + 0 * 4, c00); vst1q_f32(C + 1 * 4, c01); vst1q_f32(C + 2 * 4, c02);
    vst1q_f32(C + 1 * ldc + 0 * 4, c10); vst1q_f32(C + 1 * ldc + 1 * 4, c11); vst1q_f32(C + 1 * ldc + 2 * 4, c12);
    vst1q_f32(C + 2 * ldc + 0 * 4, c20); vst1q_f32(C + 2 * ldc + 1 * 4, c21); vst1q_f32(C + 2 * ldc + 2 * 4, c22);
    vst1q_f32(C + 3 * ldc + 0 * 4, c30); vst1q_f32(C + 3 * ldc + 1 * 4, c31); vst1q_f32(C + 3 * ldc + 2 * 4, c32);
    vst1q_f32(C + 4 * ldc + 0 * 4, c40); vst1q_f32(C + 4 * ldc + 1 * 4, c41); vst1q_f32(C + 4 * ldc + 2 * 4, c42);
    vst1q_f32(C + 5 * ldc + 0 * 4, c50); vst1q_f32(C + 5 * ldc + 1 * 4, c51); vst1q_f32(C + 5 * ldc + 2 * 4, c52);
    vst1q_f32(C + 6 * ldc + 0 * 4, c60); vst1q_f32(C + 6 * ldc + 1 * 4, c61); vst1q_f32(C + 6 * ldc + 2 * 4, c62);
    vst1q_f32(C + 7 * ldc + 0 * 4, c70); vst1q_f32(C + 7 * ldc + 1 * 4, c71); vst1q_f32(C + 7 * ldc + 2 * 4, c72);
}

}  // namespace nnops::backend::cpu::aarch64
