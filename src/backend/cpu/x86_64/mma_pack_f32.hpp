#pragma once
/// @file mma_pack_f32.hpp
/// @brief x86_64 AVX2/FMA float32 MMA (matrix micro-accumulate) packed-B kernels.
///
/// Each kernel computes C[mr][nr] += A[mr][K] × B_packed[nr][K] then clamps.
/// Tile sizes (x86_64-optimised):
///   M ∈ {6, 4, 1}    N ∈ {16, 8, 1}
/// Uses _mm256_fmadd_ps (FMA) and _mm256_broadcast_ss for A-element splat.
///
/// Reference: nn_compute/src/cpu/kernel/mma/x86_64/mma_pack_f32.hpp

#include <immintrin.h>
#include <algorithm>
#include "backend/cpu/common/restrict.hpp"

namespace nnops::backend::cpu::x86_64 {

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
inline void mma_pack_1x8_f32(float* NNOPS_RESTRICT C, int ldc,
                             const float* NNOPS_RESTRICT A,
                             const float* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    __m256 v_c0 = _mm256_setzero_ps();
    for (int k = 0; k < K; ++k) {
        const __m256 v_a0 = _mm256_broadcast_ss(A++);
        const __m256 v_b0 = _mm256_loadu_ps(B); B += ldb;
        v_c0 = _mm256_fmadd_ps(v_a0, v_b0, v_c0);
    }
    if constexpr (!zero_mode) { v_c0 = _mm256_add_ps(v_c0, _mm256_loadu_ps(C)); }

    const __m256 v_min = _mm256_set1_ps(clamp_min);
    const __m256 v_max = _mm256_set1_ps(clamp_max);
    v_c0 = _mm256_max_ps(v_c0, v_min);
    v_c0 = _mm256_min_ps(v_c0, v_max);

    _mm256_storeu_ps(C, v_c0);
}

template <bool zero_mode = false>
inline void mma_pack_1x16_f32(float* NNOPS_RESTRICT C, int ldc,
                              const float* NNOPS_RESTRICT A,
                              const float* NNOPS_RESTRICT B,
                              int ldb, int K,
                              float clamp_min, float clamp_max) noexcept {
    __m256 v_c0 = _mm256_setzero_ps();
    __m256 v_c1 = _mm256_setzero_ps();
    for (int k = 0; k < K; ++k) {
        const __m256 v_a0 = _mm256_broadcast_ss(A++);
        const __m256 v_b0 = _mm256_loadu_ps(B + 0 * 8);
        const __m256 v_b1 = _mm256_loadu_ps(B + 1 * 8);
        B += ldb;
        v_c0 = _mm256_fmadd_ps(v_a0, v_b0, v_c0);
        v_c1 = _mm256_fmadd_ps(v_a0, v_b1, v_c1);
    }
    if constexpr (!zero_mode) { v_c0 = _mm256_add_ps(v_c0, _mm256_loadu_ps(C + 0 * 8)); 
    v_c1 = _mm256_add_ps(v_c1, _mm256_loadu_ps(C + 1 * 8)); }

    const __m256 v_min = _mm256_set1_ps(clamp_min);
    const __m256 v_max = _mm256_set1_ps(clamp_max);
    v_c0 = _mm256_max_ps(_mm256_min_ps(v_c0, v_max), v_min);
    v_c1 = _mm256_max_ps(_mm256_min_ps(v_c1, v_max), v_min);

    _mm256_storeu_ps(C + 0 * 8, v_c0);
    _mm256_storeu_ps(C + 1 * 8, v_c1);
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
        const float b = *B; B += ldb;
        c0 += A[0] * b;
        c1 += A[1] * b;
        c2 += A[2] * b;
        c3 += A[3] * b;
        A += 4;
    }
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c0 += C[0 * ldc]; } } C[0 * ldc] = std::min(std::max(c0, clamp_min), clamp_max);
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c1 += C[1 * ldc]; } } C[1 * ldc] = std::min(std::max(c1, clamp_min), clamp_max);
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c2 += C[2 * ldc]; } } C[2 * ldc] = std::min(std::max(c2, clamp_min), clamp_max);
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c3 += C[3 * ldc]; } } C[3 * ldc] = std::min(std::max(c3, clamp_min), clamp_max);
}

template <bool zero_mode = false>
inline void mma_pack_4x8_f32(float* NNOPS_RESTRICT C, int ldc,
                             const float* NNOPS_RESTRICT A,
                             const float* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    __m256 v_c0 = _mm256_setzero_ps();
    __m256 v_c1 = _mm256_setzero_ps();
    __m256 v_c2 = _mm256_setzero_ps();
    __m256 v_c3 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        const __m256 v_b0 = _mm256_loadu_ps(B); B += ldb;
        v_c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 0), v_b0, v_c0);
        v_c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 1), v_b0, v_c1);
        v_c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 2), v_b0, v_c2);
        v_c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 3), v_b0, v_c3);
        A += 4;
    }

    if constexpr (!zero_mode) { v_c0 = _mm256_add_ps(v_c0, _mm256_loadu_ps(C + 0 * ldc)); 
    v_c1 = _mm256_add_ps(v_c1, _mm256_loadu_ps(C + 1 * ldc)); 
    v_c2 = _mm256_add_ps(v_c2, _mm256_loadu_ps(C + 2 * ldc)); 
    v_c3 = _mm256_add_ps(v_c3, _mm256_loadu_ps(C + 3 * ldc)); }

    const __m256 v_min = _mm256_set1_ps(clamp_min);
    const __m256 v_max = _mm256_set1_ps(clamp_max);
    v_c0 = _mm256_max_ps(_mm256_min_ps(v_c0, v_max), v_min);
    v_c1 = _mm256_max_ps(_mm256_min_ps(v_c1, v_max), v_min);
    v_c2 = _mm256_max_ps(_mm256_min_ps(v_c2, v_max), v_min);
    v_c3 = _mm256_max_ps(_mm256_min_ps(v_c3, v_max), v_min);

    _mm256_storeu_ps(C + 0 * ldc, v_c0);
    _mm256_storeu_ps(C + 1 * ldc, v_c1);
    _mm256_storeu_ps(C + 2 * ldc, v_c2);
    _mm256_storeu_ps(C + 3 * ldc, v_c3);
}

template <bool zero_mode = false>
inline void mma_pack_4x16_f32(float* NNOPS_RESTRICT C, int ldc,
                              const float* NNOPS_RESTRICT A,
                              const float* NNOPS_RESTRICT B,
                              int ldb, int K,
                              float clamp_min, float clamp_max) noexcept {
    __m256 v_c00 = _mm256_setzero_ps(), v_c01 = _mm256_setzero_ps();
    __m256 v_c10 = _mm256_setzero_ps(), v_c11 = _mm256_setzero_ps();
    __m256 v_c20 = _mm256_setzero_ps(), v_c21 = _mm256_setzero_ps();
    __m256 v_c30 = _mm256_setzero_ps(), v_c31 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        const __m256 v_b0 = _mm256_loadu_ps(B + 0 * 8);
        const __m256 v_b1 = _mm256_loadu_ps(B + 1 * 8);
        B += ldb;

        __m256 v_a0 = _mm256_broadcast_ss(A + 0);
        __m256 v_a1 = _mm256_broadcast_ss(A + 1);
        v_c00 = _mm256_fmadd_ps(v_a0, v_b0, v_c00);
        v_c01 = _mm256_fmadd_ps(v_a0, v_b1, v_c01);
        v_c10 = _mm256_fmadd_ps(v_a1, v_b0, v_c10);
        v_c11 = _mm256_fmadd_ps(v_a1, v_b1, v_c11);

        v_a0 = _mm256_broadcast_ss(A + 2);
        v_a1 = _mm256_broadcast_ss(A + 3);
        v_c20 = _mm256_fmadd_ps(v_a0, v_b0, v_c20);
        v_c21 = _mm256_fmadd_ps(v_a0, v_b1, v_c21);
        v_c30 = _mm256_fmadd_ps(v_a1, v_b0, v_c30);
        v_c31 = _mm256_fmadd_ps(v_a1, v_b1, v_c31);

        A += 4;
    }

    if constexpr (!zero_mode) { v_c00 = _mm256_add_ps(v_c00, _mm256_loadu_ps(C + 0 * ldc + 0)); 
    v_c01 = _mm256_add_ps(v_c01, _mm256_loadu_ps(C + 0 * ldc + 8)); 
    v_c10 = _mm256_add_ps(v_c10, _mm256_loadu_ps(C + 1 * ldc + 0)); 
    v_c11 = _mm256_add_ps(v_c11, _mm256_loadu_ps(C + 1 * ldc + 8)); 
    v_c20 = _mm256_add_ps(v_c20, _mm256_loadu_ps(C + 2 * ldc + 0)); 
    v_c21 = _mm256_add_ps(v_c21, _mm256_loadu_ps(C + 2 * ldc + 8)); 
    v_c30 = _mm256_add_ps(v_c30, _mm256_loadu_ps(C + 3 * ldc + 0)); 
    v_c31 = _mm256_add_ps(v_c31, _mm256_loadu_ps(C + 3 * ldc + 8)); }

    const __m256 v_min = _mm256_set1_ps(clamp_min);
    const __m256 v_max = _mm256_set1_ps(clamp_max);

    v_c00 = _mm256_max_ps(_mm256_min_ps(v_c00, v_max), v_min);
    v_c01 = _mm256_max_ps(_mm256_min_ps(v_c01, v_max), v_min);
    v_c10 = _mm256_max_ps(_mm256_min_ps(v_c10, v_max), v_min);
    v_c11 = _mm256_max_ps(_mm256_min_ps(v_c11, v_max), v_min);
    v_c20 = _mm256_max_ps(_mm256_min_ps(v_c20, v_max), v_min);
    v_c21 = _mm256_max_ps(_mm256_min_ps(v_c21, v_max), v_min);
    v_c30 = _mm256_max_ps(_mm256_min_ps(v_c30, v_max), v_min);
    v_c31 = _mm256_max_ps(_mm256_min_ps(v_c31, v_max), v_min);

    _mm256_storeu_ps(C + 0 * ldc + 0, v_c00);
    _mm256_storeu_ps(C + 0 * ldc + 8, v_c01);
    _mm256_storeu_ps(C + 1 * ldc + 0, v_c10);
    _mm256_storeu_ps(C + 1 * ldc + 8, v_c11);
    _mm256_storeu_ps(C + 2 * ldc + 0, v_c20);
    _mm256_storeu_ps(C + 2 * ldc + 8, v_c21);
    _mm256_storeu_ps(C + 3 * ldc + 0, v_c30);
    _mm256_storeu_ps(C + 3 * ldc + 8, v_c31);
}

// =========================================================================
//  mr=6  kernels
// =========================================================================

template <bool zero_mode = false>
inline void mma_pack_6x1_f32(float* NNOPS_RESTRICT C, int ldc,
                             const float* NNOPS_RESTRICT A,
                             const float* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f, c4 = 0.0f, c5 = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float b = *B; B += ldb;
        c0 += A[0] * b;
        c1 += A[1] * b;
        c2 += A[2] * b;
        c3 += A[3] * b;
        c4 += A[4] * b;
        c5 += A[5] * b;
        A += 6;
    }
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c0 += C[0 * ldc]; } } C[0 * ldc] = std::min(std::max(c0, clamp_min), clamp_max);
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c1 += C[1 * ldc]; } } C[1 * ldc] = std::min(std::max(c1, clamp_min), clamp_max);
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c2 += C[2 * ldc]; } } C[2 * ldc] = std::min(std::max(c2, clamp_min), clamp_max);
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c3 += C[3 * ldc]; } } C[3 * ldc] = std::min(std::max(c3, clamp_min), clamp_max);
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c4 += C[4 * ldc]; } } C[4 * ldc] = std::min(std::max(c4, clamp_min), clamp_max);
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c5 += C[5 * ldc]; } } C[5 * ldc] = std::min(std::max(c5, clamp_min), clamp_max);
}

template <bool zero_mode = false>
inline void mma_pack_6x8_f32(float* NNOPS_RESTRICT C, int ldc,
                             const float* NNOPS_RESTRICT A,
                             const float* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    __m256 v_c0 = _mm256_setzero_ps();
    __m256 v_c1 = _mm256_setzero_ps();
    __m256 v_c2 = _mm256_setzero_ps();
    __m256 v_c3 = _mm256_setzero_ps();
    __m256 v_c4 = _mm256_setzero_ps();
    __m256 v_c5 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        const __m256 v_b0 = _mm256_loadu_ps(B); B += ldb;
        v_c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 0), v_b0, v_c0);
        v_c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 1), v_b0, v_c1);
        v_c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 2), v_b0, v_c2);
        v_c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 3), v_b0, v_c3);
        v_c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 4), v_b0, v_c4);
        v_c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 5), v_b0, v_c5);
        A += 6;
    }

    if constexpr (!zero_mode) { v_c0 = _mm256_add_ps(v_c0, _mm256_loadu_ps(C + 0 * ldc)); 
    v_c1 = _mm256_add_ps(v_c1, _mm256_loadu_ps(C + 1 * ldc)); 
    v_c2 = _mm256_add_ps(v_c2, _mm256_loadu_ps(C + 2 * ldc)); 
    v_c3 = _mm256_add_ps(v_c3, _mm256_loadu_ps(C + 3 * ldc)); 
    v_c4 = _mm256_add_ps(v_c4, _mm256_loadu_ps(C + 4 * ldc)); 
    v_c5 = _mm256_add_ps(v_c5, _mm256_loadu_ps(C + 5 * ldc)); }

    const __m256 v_min = _mm256_set1_ps(clamp_min);
    const __m256 v_max = _mm256_set1_ps(clamp_max);

    v_c0 = _mm256_max_ps(_mm256_min_ps(v_c0, v_max), v_min);
    v_c1 = _mm256_max_ps(_mm256_min_ps(v_c1, v_max), v_min);
    v_c2 = _mm256_max_ps(_mm256_min_ps(v_c2, v_max), v_min);
    v_c3 = _mm256_max_ps(_mm256_min_ps(v_c3, v_max), v_min);
    v_c4 = _mm256_max_ps(_mm256_min_ps(v_c4, v_max), v_min);
    v_c5 = _mm256_max_ps(_mm256_min_ps(v_c5, v_max), v_min);

    _mm256_storeu_ps(C + 0 * ldc, v_c0);
    _mm256_storeu_ps(C + 1 * ldc, v_c1);
    _mm256_storeu_ps(C + 2 * ldc, v_c2);
    _mm256_storeu_ps(C + 3 * ldc, v_c3);
    _mm256_storeu_ps(C + 4 * ldc, v_c4);
    _mm256_storeu_ps(C + 5 * ldc, v_c5);
}

template <bool zero_mode = false>
inline void mma_pack_6x16_f32(float* NNOPS_RESTRICT C, int ldc,
                              const float* NNOPS_RESTRICT A,
                              const float* NNOPS_RESTRICT B,
                              int ldb, int K,
                              float clamp_min, float clamp_max) noexcept {
    __m256 v_c00 = _mm256_setzero_ps(), v_c01 = _mm256_setzero_ps();
    __m256 v_c10 = _mm256_setzero_ps(), v_c11 = _mm256_setzero_ps();
    __m256 v_c20 = _mm256_setzero_ps(), v_c21 = _mm256_setzero_ps();
    __m256 v_c30 = _mm256_setzero_ps(), v_c31 = _mm256_setzero_ps();
    __m256 v_c40 = _mm256_setzero_ps(), v_c41 = _mm256_setzero_ps();
    __m256 v_c50 = _mm256_setzero_ps(), v_c51 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        const __m256 v_b0 = _mm256_loadu_ps(B + 0 * 8);
        const __m256 v_b1 = _mm256_loadu_ps(B + 1 * 8);
        B += ldb;

        __m256 v_a0 = _mm256_broadcast_ss(A + 0);
        __m256 v_a1 = _mm256_broadcast_ss(A + 1);
        v_c00 = _mm256_fmadd_ps(v_a0, v_b0, v_c00);
        v_c01 = _mm256_fmadd_ps(v_a0, v_b1, v_c01);
        v_c10 = _mm256_fmadd_ps(v_a1, v_b0, v_c10);
        v_c11 = _mm256_fmadd_ps(v_a1, v_b1, v_c11);

        v_a0 = _mm256_broadcast_ss(A + 2);
        v_a1 = _mm256_broadcast_ss(A + 3);
        v_c20 = _mm256_fmadd_ps(v_a0, v_b0, v_c20);
        v_c21 = _mm256_fmadd_ps(v_a0, v_b1, v_c21);
        v_c30 = _mm256_fmadd_ps(v_a1, v_b0, v_c30);
        v_c31 = _mm256_fmadd_ps(v_a1, v_b1, v_c31);

        v_a0 = _mm256_broadcast_ss(A + 4);
        v_a1 = _mm256_broadcast_ss(A + 5);
        v_c40 = _mm256_fmadd_ps(v_a0, v_b0, v_c40);
        v_c41 = _mm256_fmadd_ps(v_a0, v_b1, v_c41);
        v_c50 = _mm256_fmadd_ps(v_a1, v_b0, v_c50);
        v_c51 = _mm256_fmadd_ps(v_a1, v_b1, v_c51);

        A += 6;
    }

    if constexpr (!zero_mode) { v_c00 = _mm256_add_ps(v_c00, _mm256_loadu_ps(C + 0 * ldc + 0)); 
    v_c01 = _mm256_add_ps(v_c01, _mm256_loadu_ps(C + 0 * ldc + 8)); 
    v_c10 = _mm256_add_ps(v_c10, _mm256_loadu_ps(C + 1 * ldc + 0)); 
    v_c11 = _mm256_add_ps(v_c11, _mm256_loadu_ps(C + 1 * ldc + 8)); 
    v_c20 = _mm256_add_ps(v_c20, _mm256_loadu_ps(C + 2 * ldc + 0)); 
    v_c21 = _mm256_add_ps(v_c21, _mm256_loadu_ps(C + 2 * ldc + 8)); 
    v_c30 = _mm256_add_ps(v_c30, _mm256_loadu_ps(C + 3 * ldc + 0)); 
    v_c31 = _mm256_add_ps(v_c31, _mm256_loadu_ps(C + 3 * ldc + 8)); 
    v_c40 = _mm256_add_ps(v_c40, _mm256_loadu_ps(C + 4 * ldc + 0)); 
    v_c41 = _mm256_add_ps(v_c41, _mm256_loadu_ps(C + 4 * ldc + 8)); 
    v_c50 = _mm256_add_ps(v_c50, _mm256_loadu_ps(C + 5 * ldc + 0)); 
    v_c51 = _mm256_add_ps(v_c51, _mm256_loadu_ps(C + 5 * ldc + 8)); }

    const __m256 v_min = _mm256_set1_ps(clamp_min);
    const __m256 v_max = _mm256_set1_ps(clamp_max);

    v_c00 = _mm256_max_ps(_mm256_min_ps(v_c00, v_max), v_min);
    v_c01 = _mm256_max_ps(_mm256_min_ps(v_c01, v_max), v_min);
    v_c10 = _mm256_max_ps(_mm256_min_ps(v_c10, v_max), v_min);
    v_c11 = _mm256_max_ps(_mm256_min_ps(v_c11, v_max), v_min);
    v_c20 = _mm256_max_ps(_mm256_min_ps(v_c20, v_max), v_min);
    v_c21 = _mm256_max_ps(_mm256_min_ps(v_c21, v_max), v_min);
    v_c30 = _mm256_max_ps(_mm256_min_ps(v_c30, v_max), v_min);
    v_c31 = _mm256_max_ps(_mm256_min_ps(v_c31, v_max), v_min);
    v_c40 = _mm256_max_ps(_mm256_min_ps(v_c40, v_max), v_min);
    v_c41 = _mm256_max_ps(_mm256_min_ps(v_c41, v_max), v_min);
    v_c50 = _mm256_max_ps(_mm256_min_ps(v_c50, v_max), v_min);
    v_c51 = _mm256_max_ps(_mm256_min_ps(v_c51, v_max), v_min);

    _mm256_storeu_ps(C + 0 * ldc + 0, v_c00);
    _mm256_storeu_ps(C + 0 * ldc + 8, v_c01);
    _mm256_storeu_ps(C + 1 * ldc + 0, v_c10);
    _mm256_storeu_ps(C + 1 * ldc + 8, v_c11);
    _mm256_storeu_ps(C + 2 * ldc + 0, v_c20);
    _mm256_storeu_ps(C + 2 * ldc + 8, v_c21);
    _mm256_storeu_ps(C + 3 * ldc + 0, v_c30);
    _mm256_storeu_ps(C + 3 * ldc + 8, v_c31);
    _mm256_storeu_ps(C + 4 * ldc + 0, v_c40);
    _mm256_storeu_ps(C + 4 * ldc + 8, v_c41);
    _mm256_storeu_ps(C + 5 * ldc + 0, v_c50);
    _mm256_storeu_ps(C + 5 * ldc + 8, v_c51);
}

}  // namespace nnops::backend::cpu::x86_64
