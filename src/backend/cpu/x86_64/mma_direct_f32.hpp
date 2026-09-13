#pragma once
/// @file mma_direct_f32.hpp
/// @brief x86_64 AVX2/FMA float32 MMA direct (unpacked) micro-kernels.
///
/// These kernels compute C[mr][nr] += A[mr][K] × B[K][nr] directly
/// from row-major A (stride lda) and B (stride ldb), without packing.
/// Uses _mm256_fmadd_ps (FMA) and _mm256_broadcast_ss for A-element splat.
///
/// Tile sizes (x86_64-optimised):
///   M ∈ {6, 4, 1}    N ∈ {16, 8, 1}
///
/// No K unrolling is performed (unlike AArch64) — each iteration
/// processes one K element. Strided memory access makes unrolling
/// less beneficial than in the packed case.
///
/// Reference: nn_compute/src/cpu/kernel/mma/x86_64/mma_direct_f32.hpp

#include <immintrin.h>
#include <algorithm>
#include "backend/cpu/common/restrict.hpp"

namespace nnops::backend::cpu::x86_64 {

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
        const float b0 = B[0];
        c0 += *A_ptr0++ * b0;
        B += ldb;
    }
    if constexpr (!zero_mode) { c0 += C[0]; }
    C[0] = std::min(std::max(c0, clamp_min), clamp_max);
}

template <bool zero_mode = false>
inline void mma_direct_1x8_f32(
    float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A, int lda,
    const float* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const float* NNOPS_RESTRICT A_ptr0 = A;

    __m256 v_c0 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        const __m256 v_b0 = _mm256_loadu_ps(B); B += ldb;
        __m256 v_a0 = _mm256_broadcast_ss(A_ptr0); A_ptr0 += 1;
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
inline void mma_direct_1x16_f32(
    float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A, int lda,
    const float* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const float* NNOPS_RESTRICT A_ptr0 = A;

    __m256 v_c00 = _mm256_setzero_ps();
    __m256 v_c01 = v_c00;

    for (int k = 0; k < K; ++k) {
        const __m256 v_b0 = _mm256_loadu_ps(B + 0 * 8);
        const __m256 v_b1 = _mm256_loadu_ps(B + 1 * 8);
        B += ldb;

        __m256 v_a0 = _mm256_broadcast_ss(A_ptr0); A_ptr0 += 1;
        v_c00 = _mm256_fmadd_ps(v_a0, v_b0, v_c00);
        v_c01 = _mm256_fmadd_ps(v_a0, v_b1, v_c01);
    }

    const __m256 v_min = _mm256_set1_ps(clamp_min);
    const __m256 v_max = _mm256_set1_ps(clamp_max);

    if constexpr (!zero_mode) { v_c00 = _mm256_add_ps(v_c00, _mm256_loadu_ps(C + 0 * 8)); 
    v_c01 = _mm256_add_ps(v_c01, _mm256_loadu_ps(C + 1 * 8)); }

    _mm256_storeu_ps(C + 0 * 8, _mm256_min_ps(_mm256_max_ps(v_c00, v_min), v_max));
    _mm256_storeu_ps(C + 1 * 8, _mm256_min_ps(_mm256_max_ps(v_c01, v_min), v_max));
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
        if constexpr (!zero_mode) { v = *dst + acc; }
        *dst = std::min(std::max(v, clamp_min), clamp_max);
    };
    write(C + 0 * ldc, c0);
    write(C + 1 * ldc, c1);
    write(C + 2 * ldc, c2);
    write(C + 3 * ldc, c3);
}

template <bool zero_mode = false>
inline void mma_direct_4x8_f32(
    float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A, int lda,
    const float* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const float* NNOPS_RESTRICT A_ptr0 = A + 0 * lda;
    const float* NNOPS_RESTRICT A_ptr1 = A + 1 * lda;
    const float* NNOPS_RESTRICT A_ptr2 = A + 2 * lda;
    const float* NNOPS_RESTRICT A_ptr3 = A + 3 * lda;

    __m256 v_c0 = _mm256_setzero_ps();
    __m256 v_c1 = v_c0;
    __m256 v_c2 = v_c0;
    __m256 v_c3 = v_c0;

    for (int k = 0; k < K; ++k) {
        const __m256 v_b0 = _mm256_loadu_ps(B); B += ldb;

        __m256 v_a0 = _mm256_broadcast_ss(A_ptr0); A_ptr0 += 1;
        v_c0 = _mm256_fmadd_ps(v_a0, v_b0, v_c0);
        __m256 v_a1 = _mm256_broadcast_ss(A_ptr1); A_ptr1 += 1;
        v_c1 = _mm256_fmadd_ps(v_a1, v_b0, v_c1);
        __m256 v_a2 = _mm256_broadcast_ss(A_ptr2); A_ptr2 += 1;
        v_c2 = _mm256_fmadd_ps(v_a2, v_b0, v_c2);
        __m256 v_a3 = _mm256_broadcast_ss(A_ptr3); A_ptr3 += 1;
        v_c3 = _mm256_fmadd_ps(v_a3, v_b0, v_c3);
    }

    const __m256 v_min = _mm256_set1_ps(clamp_min);
    const __m256 v_max = _mm256_set1_ps(clamp_max);

    if constexpr (!zero_mode) { v_c0 = _mm256_add_ps(v_c0, _mm256_loadu_ps(C + 0 * ldc)); 
    v_c1 = _mm256_add_ps(v_c1, _mm256_loadu_ps(C + 1 * ldc)); 
    v_c2 = _mm256_add_ps(v_c2, _mm256_loadu_ps(C + 2 * ldc)); 
    v_c3 = _mm256_add_ps(v_c3, _mm256_loadu_ps(C + 3 * ldc)); }

    _mm256_storeu_ps(C + 0 * ldc, _mm256_min_ps(_mm256_max_ps(v_c0, v_min), v_max));
    _mm256_storeu_ps(C + 1 * ldc, _mm256_min_ps(_mm256_max_ps(v_c1, v_min), v_max));
    _mm256_storeu_ps(C + 2 * ldc, _mm256_min_ps(_mm256_max_ps(v_c2, v_min), v_max));
    _mm256_storeu_ps(C + 3 * ldc, _mm256_min_ps(_mm256_max_ps(v_c3, v_min), v_max));
}

template <bool zero_mode = false>
inline void mma_direct_4x16_f32(
    float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A, int lda,
    const float* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const float* NNOPS_RESTRICT A_ptr0 = A + 0 * lda;
    const float* NNOPS_RESTRICT A_ptr1 = A + 1 * lda;
    const float* NNOPS_RESTRICT A_ptr2 = A + 2 * lda;
    const float* NNOPS_RESTRICT A_ptr3 = A + 3 * lda;

    __m256 v_c00 = _mm256_setzero_ps();
    __m256 v_c01 = v_c00;
    __m256 v_c10 = v_c00;
    __m256 v_c11 = v_c00;
    __m256 v_c20 = v_c00;
    __m256 v_c21 = v_c00;
    __m256 v_c30 = v_c00;
    __m256 v_c31 = v_c00;

    for (int k = 0; k < K; ++k) {
        const __m256 v_b0 = _mm256_loadu_ps(B + 0 * 8);
        const __m256 v_b1 = _mm256_loadu_ps(B + 1 * 8);
        B += ldb;

        __m256 v_a0 = _mm256_broadcast_ss(A_ptr0); A_ptr0 += 1;
        v_c00 = _mm256_fmadd_ps(v_a0, v_b0, v_c00);
        v_c01 = _mm256_fmadd_ps(v_a0, v_b1, v_c01);

        __m256 v_a1 = _mm256_broadcast_ss(A_ptr1); A_ptr1 += 1;
        v_c10 = _mm256_fmadd_ps(v_a1, v_b0, v_c10);
        v_c11 = _mm256_fmadd_ps(v_a1, v_b1, v_c11);

        __m256 v_a2 = _mm256_broadcast_ss(A_ptr2); A_ptr2 += 1;
        v_c20 = _mm256_fmadd_ps(v_a2, v_b0, v_c20);
        v_c21 = _mm256_fmadd_ps(v_a2, v_b1, v_c21);

        __m256 v_a3 = _mm256_broadcast_ss(A_ptr3); A_ptr3 += 1;
        v_c30 = _mm256_fmadd_ps(v_a3, v_b0, v_c30);
        v_c31 = _mm256_fmadd_ps(v_a3, v_b1, v_c31);
    }

    const __m256 v_min = _mm256_set1_ps(clamp_min);
    const __m256 v_max = _mm256_set1_ps(clamp_max);

    if constexpr (!zero_mode) { v_c00 = _mm256_add_ps(v_c00, _mm256_loadu_ps(C + 0 * ldc + 0 * 8)); 
    v_c01 = _mm256_add_ps(v_c01, _mm256_loadu_ps(C + 0 * ldc + 1 * 8)); 
    v_c10 = _mm256_add_ps(v_c10, _mm256_loadu_ps(C + 1 * ldc + 0 * 8)); 
    v_c11 = _mm256_add_ps(v_c11, _mm256_loadu_ps(C + 1 * ldc + 1 * 8)); 
    v_c20 = _mm256_add_ps(v_c20, _mm256_loadu_ps(C + 2 * ldc + 0 * 8)); 
    v_c21 = _mm256_add_ps(v_c21, _mm256_loadu_ps(C + 2 * ldc + 1 * 8)); 
    v_c30 = _mm256_add_ps(v_c30, _mm256_loadu_ps(C + 3 * ldc + 0 * 8)); 
    v_c31 = _mm256_add_ps(v_c31, _mm256_loadu_ps(C + 3 * ldc + 1 * 8)); }

    _mm256_storeu_ps(C + 0 * ldc + 0 * 8, _mm256_min_ps(_mm256_max_ps(v_c00, v_min), v_max));
    _mm256_storeu_ps(C + 0 * ldc + 1 * 8, _mm256_min_ps(_mm256_max_ps(v_c01, v_min), v_max));
    _mm256_storeu_ps(C + 1 * ldc + 0 * 8, _mm256_min_ps(_mm256_max_ps(v_c10, v_min), v_max));
    _mm256_storeu_ps(C + 1 * ldc + 1 * 8, _mm256_min_ps(_mm256_max_ps(v_c11, v_min), v_max));
    _mm256_storeu_ps(C + 2 * ldc + 0 * 8, _mm256_min_ps(_mm256_max_ps(v_c20, v_min), v_max));
    _mm256_storeu_ps(C + 2 * ldc + 1 * 8, _mm256_min_ps(_mm256_max_ps(v_c21, v_min), v_max));
    _mm256_storeu_ps(C + 3 * ldc + 0 * 8, _mm256_min_ps(_mm256_max_ps(v_c30, v_min), v_max));
    _mm256_storeu_ps(C + 3 * ldc + 1 * 8, _mm256_min_ps(_mm256_max_ps(v_c31, v_min), v_max));
}

// =========================================================================
//  mr=6  kernels
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

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
    float acc3 = 0.0f, acc4 = 0.0f, acc5 = 0.0f;

    for (int k = 0; k < K; ++k) {
        const float b0 = B[0];
        acc0 += *A_ptr0++ * b0;
        acc1 += *A_ptr1++ * b0;
        acc2 += *A_ptr2++ * b0;
        acc3 += *A_ptr3++ * b0;
        acc4 += *A_ptr4++ * b0;
        acc5 += *A_ptr5++ * b0;
        B += ldb;
    }

    auto write = [&](float* dst, float acc) {
        float v = acc;
        if constexpr (!zero_mode) { v = *dst + acc; }
        *dst = std::min(std::max(v, clamp_min), clamp_max);
    };
    write(C + 0 * ldc, acc0);
    write(C + 1 * ldc, acc1);
    write(C + 2 * ldc, acc2);
    write(C + 3 * ldc, acc3);
    write(C + 4 * ldc, acc4);
    write(C + 5 * ldc, acc5);
}

template <bool zero_mode = false>
inline void mma_direct_6x8_f32(
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

    __m256 v_c0 = _mm256_setzero_ps();
    __m256 v_c1 = v_c0;
    __m256 v_c2 = v_c0;
    __m256 v_c3 = v_c0;
    __m256 v_c4 = v_c0;
    __m256 v_c5 = v_c0;

    for (int k = 0; k < K; ++k) {
        const __m256 v_b0 = _mm256_loadu_ps(B); B += ldb;

        __m256 v_a0 = _mm256_broadcast_ss(A_ptr0); A_ptr0 += 1;
        v_c0 = _mm256_fmadd_ps(v_a0, v_b0, v_c0);
        __m256 v_a1 = _mm256_broadcast_ss(A_ptr1); A_ptr1 += 1;
        v_c1 = _mm256_fmadd_ps(v_a1, v_b0, v_c1);
        __m256 v_a2 = _mm256_broadcast_ss(A_ptr2); A_ptr2 += 1;
        v_c2 = _mm256_fmadd_ps(v_a2, v_b0, v_c2);
        __m256 v_a3 = _mm256_broadcast_ss(A_ptr3); A_ptr3 += 1;
        v_c3 = _mm256_fmadd_ps(v_a3, v_b0, v_c3);
        __m256 v_a4 = _mm256_broadcast_ss(A_ptr4); A_ptr4 += 1;
        v_c4 = _mm256_fmadd_ps(v_a4, v_b0, v_c4);
        __m256 v_a5 = _mm256_broadcast_ss(A_ptr5); A_ptr5 += 1;
        v_c5 = _mm256_fmadd_ps(v_a5, v_b0, v_c5);
    }

    const __m256 v_min = _mm256_set1_ps(clamp_min);
    const __m256 v_max = _mm256_set1_ps(clamp_max);

    if constexpr (!zero_mode) { v_c0 = _mm256_add_ps(v_c0, _mm256_loadu_ps(C + 0 * ldc)); 
    v_c1 = _mm256_add_ps(v_c1, _mm256_loadu_ps(C + 1 * ldc)); 
    v_c2 = _mm256_add_ps(v_c2, _mm256_loadu_ps(C + 2 * ldc)); 
    v_c3 = _mm256_add_ps(v_c3, _mm256_loadu_ps(C + 3 * ldc)); 
    v_c4 = _mm256_add_ps(v_c4, _mm256_loadu_ps(C + 4 * ldc)); 
    v_c5 = _mm256_add_ps(v_c5, _mm256_loadu_ps(C + 5 * ldc)); }

    _mm256_storeu_ps(C + 0 * ldc, _mm256_min_ps(_mm256_max_ps(v_c0, v_min), v_max));
    _mm256_storeu_ps(C + 1 * ldc, _mm256_min_ps(_mm256_max_ps(v_c1, v_min), v_max));
    _mm256_storeu_ps(C + 2 * ldc, _mm256_min_ps(_mm256_max_ps(v_c2, v_min), v_max));
    _mm256_storeu_ps(C + 3 * ldc, _mm256_min_ps(_mm256_max_ps(v_c3, v_min), v_max));
    _mm256_storeu_ps(C + 4 * ldc, _mm256_min_ps(_mm256_max_ps(v_c4, v_min), v_max));
    _mm256_storeu_ps(C + 5 * ldc, _mm256_min_ps(_mm256_max_ps(v_c5, v_min), v_max));
}

template <bool zero_mode = false>
inline void mma_direct_6x16_f32(
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

    __m256 v_c00 = _mm256_setzero_ps();
    __m256 v_c01 = v_c00;
    __m256 v_c10 = v_c00;
    __m256 v_c11 = v_c00;
    __m256 v_c20 = v_c00;
    __m256 v_c21 = v_c00;
    __m256 v_c30 = v_c00;
    __m256 v_c31 = v_c00;
    __m256 v_c40 = v_c00;
    __m256 v_c41 = v_c00;
    __m256 v_c50 = v_c00;
    __m256 v_c51 = v_c00;

    for (int k = 0; k < K; ++k) {
        const __m256 v_b0 = _mm256_loadu_ps(B + 0 * 8);
        const __m256 v_b1 = _mm256_loadu_ps(B + 1 * 8);
        B += ldb;

        __m256 v_a0 = _mm256_broadcast_ss(A_ptr0); A_ptr0 += 1;
        v_c00 = _mm256_fmadd_ps(v_a0, v_b0, v_c00);
        v_c01 = _mm256_fmadd_ps(v_a0, v_b1, v_c01);

        __m256 v_a1 = _mm256_broadcast_ss(A_ptr1); A_ptr1 += 1;
        v_c10 = _mm256_fmadd_ps(v_a1, v_b0, v_c10);
        v_c11 = _mm256_fmadd_ps(v_a1, v_b1, v_c11);

        __m256 v_a2 = _mm256_broadcast_ss(A_ptr2); A_ptr2 += 1;
        v_c20 = _mm256_fmadd_ps(v_a2, v_b0, v_c20);
        v_c21 = _mm256_fmadd_ps(v_a2, v_b1, v_c21);

        __m256 v_a3 = _mm256_broadcast_ss(A_ptr3); A_ptr3 += 1;
        v_c30 = _mm256_fmadd_ps(v_a3, v_b0, v_c30);
        v_c31 = _mm256_fmadd_ps(v_a3, v_b1, v_c31);

        __m256 v_a4 = _mm256_broadcast_ss(A_ptr4); A_ptr4 += 1;
        v_c40 = _mm256_fmadd_ps(v_a4, v_b0, v_c40);
        v_c41 = _mm256_fmadd_ps(v_a4, v_b1, v_c41);

        __m256 v_a5 = _mm256_broadcast_ss(A_ptr5); A_ptr5 += 1;
        v_c50 = _mm256_fmadd_ps(v_a5, v_b0, v_c50);
        v_c51 = _mm256_fmadd_ps(v_a5, v_b1, v_c51);
    }

    const __m256 v_min = _mm256_set1_ps(clamp_min);
    const __m256 v_max = _mm256_set1_ps(clamp_max);

    if constexpr (!zero_mode) { v_c00 = _mm256_add_ps(v_c00, _mm256_loadu_ps(C + 0 * ldc + 0 * 8)); 
    v_c01 = _mm256_add_ps(v_c01, _mm256_loadu_ps(C + 0 * ldc + 1 * 8)); 
    v_c10 = _mm256_add_ps(v_c10, _mm256_loadu_ps(C + 1 * ldc + 0 * 8)); 
    v_c11 = _mm256_add_ps(v_c11, _mm256_loadu_ps(C + 1 * ldc + 1 * 8)); 
    v_c20 = _mm256_add_ps(v_c20, _mm256_loadu_ps(C + 2 * ldc + 0 * 8)); 
    v_c21 = _mm256_add_ps(v_c21, _mm256_loadu_ps(C + 2 * ldc + 1 * 8)); 
    v_c30 = _mm256_add_ps(v_c30, _mm256_loadu_ps(C + 3 * ldc + 0 * 8)); 
    v_c31 = _mm256_add_ps(v_c31, _mm256_loadu_ps(C + 3 * ldc + 1 * 8)); 
    v_c40 = _mm256_add_ps(v_c40, _mm256_loadu_ps(C + 4 * ldc + 0 * 8)); 
    v_c41 = _mm256_add_ps(v_c41, _mm256_loadu_ps(C + 4 * ldc + 1 * 8)); 
    v_c50 = _mm256_add_ps(v_c50, _mm256_loadu_ps(C + 5 * ldc + 0 * 8)); 
    v_c51 = _mm256_add_ps(v_c51, _mm256_loadu_ps(C + 5 * ldc + 1 * 8)); }

    _mm256_storeu_ps(C + 0 * ldc + 0 * 8, _mm256_min_ps(_mm256_max_ps(v_c00, v_min), v_max));
    _mm256_storeu_ps(C + 0 * ldc + 1 * 8, _mm256_min_ps(_mm256_max_ps(v_c01, v_min), v_max));
    _mm256_storeu_ps(C + 1 * ldc + 0 * 8, _mm256_min_ps(_mm256_max_ps(v_c10, v_min), v_max));
    _mm256_storeu_ps(C + 1 * ldc + 1 * 8, _mm256_min_ps(_mm256_max_ps(v_c11, v_min), v_max));
    _mm256_storeu_ps(C + 2 * ldc + 0 * 8, _mm256_min_ps(_mm256_max_ps(v_c20, v_min), v_max));
    _mm256_storeu_ps(C + 2 * ldc + 1 * 8, _mm256_min_ps(_mm256_max_ps(v_c21, v_min), v_max));
    _mm256_storeu_ps(C + 3 * ldc + 0 * 8, _mm256_min_ps(_mm256_max_ps(v_c30, v_min), v_max));
    _mm256_storeu_ps(C + 3 * ldc + 1 * 8, _mm256_min_ps(_mm256_max_ps(v_c31, v_min), v_max));
    _mm256_storeu_ps(C + 4 * ldc + 0 * 8, _mm256_min_ps(_mm256_max_ps(v_c40, v_min), v_max));
    _mm256_storeu_ps(C + 4 * ldc + 1 * 8, _mm256_min_ps(_mm256_max_ps(v_c41, v_min), v_max));
    _mm256_storeu_ps(C + 5 * ldc + 0 * 8, _mm256_min_ps(_mm256_max_ps(v_c50, v_min), v_max));
    _mm256_storeu_ps(C + 5 * ldc + 1 * 8, _mm256_min_ps(_mm256_max_ps(v_c51, v_min), v_max));
}

}  // namespace nnops::backend::cpu::x86_64
