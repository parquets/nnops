#pragma once
/// @file mma_direct_f16.hpp
/// @brief x86_64 F16C/FMA float16 MMA direct (unpacked) micro-kernels.
///
/// These kernels compute C[mr][nr] += A[mr][K] × B[K][nr] directly
/// from row-major A (stride lda) and B (stride ldb), without packing.
/// Uses F16C for fp16↔fp32 conversion and FMA for the inner product.
/// Accumulators are fp32; results clamped in fp32 then rounded to fp16.
///
/// Tile sizes (x86_64-optimised):
///   M ∈ {6, 4, 1}    N ∈ {16, 8, 1}
///
/// No K unrolling — each iteration processes one K element.
///
/// Reference: nn_compute/src/cpu/kernel/mma/x86_64/mma_direct_f16.hpp

#include <immintrin.h>
#include <algorithm>
#include "nnops/detail/half.hpp"
#include "backend/cpu/common/restrict.hpp"
#include "pack_f16.hpp"  // for load_f16x8, store_f16x8

namespace nnops::backend::cpu::x86_64 {

using nnops::backend::cpu::half;
using nnops::backend::cpu::half_to_float;
using nnops::backend::cpu::float_to_half;

// =========================================================================
//  mr=1  kernels
// =========================================================================

template <bool zero_mode = false>
inline void mma_direct_1x1_f16(
    half* NNOPS_RESTRICT C, int ldc,
    const half* NNOPS_RESTRICT A, int lda,
    const half* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const half* NNOPS_RESTRICT A_ptr0 = A;

    float c0 = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float b0 = half_to_float(B[0]);
        c0 += half_to_float(*A_ptr0++) * b0;
        B += ldb;
    }
    if constexpr (!zero_mode) { c0 += half_to_float(C[0]); }
    C[0] = float_to_half(std::min(std::max(c0, clamp_min), clamp_max));
}

template <bool zero_mode = false>
inline void mma_direct_1x8_f16(
    half* NNOPS_RESTRICT C, int ldc,
    const half* NNOPS_RESTRICT A, int lda,
    const half* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const half* NNOPS_RESTRICT A_ptr0 = A;

    __m256 v_c0 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        const __m256 v_b0 = load_f16x8(B); B += ldb;
        __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr0[0].bits))); A_ptr0 += 1;
        v_c0 = _mm256_fmadd_ps(v_a0, v_b0, v_c0);
    }

    const __m256 v_min = _mm256_set1_ps(clamp_min);
    const __m256 v_max = _mm256_set1_ps(clamp_max);
    if constexpr (!zero_mode) { v_c0 = _mm256_add_ps(v_c0, load_f16x8(C)); }
    v_c0 = _mm256_max_ps(_mm256_min_ps(v_c0, v_max), v_min);
    store_f16x8(C, v_c0);
}

template <bool zero_mode = false>
inline void mma_direct_1x16_f16(
    half* NNOPS_RESTRICT C, int ldc,
    const half* NNOPS_RESTRICT A, int lda,
    const half* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const half* NNOPS_RESTRICT A_ptr0 = A;

    __m256 v_c00 = _mm256_setzero_ps();
    __m256 v_c01 = v_c00;

    for (int k = 0; k < K; ++k) {
        const __m256 v_b0 = load_f16x8(B + 0 * 8);
        const __m256 v_b1 = load_f16x8(B + 1 * 8);
        B += ldb;

        __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr0[0].bits))); A_ptr0 += 1;
        v_c00 = _mm256_fmadd_ps(v_a0, v_b0, v_c00);
        v_c01 = _mm256_fmadd_ps(v_a0, v_b1, v_c01);
    }

    const __m256 v_min = _mm256_set1_ps(clamp_min);
    const __m256 v_max = _mm256_set1_ps(clamp_max);
    if constexpr (!zero_mode) { v_c00 = _mm256_add_ps(v_c00, load_f16x8(C + 0 * 8)); }
    v_c00 = _mm256_max_ps(_mm256_min_ps(v_c00, v_max), v_min);
    store_f16x8(C + 0 * 8, v_c00);
    if constexpr (!zero_mode) { v_c01 = _mm256_add_ps(v_c01, load_f16x8(C + 1 * 8)); }
    v_c01 = _mm256_max_ps(_mm256_min_ps(v_c01, v_max), v_min);
    store_f16x8(C + 1 * 8, v_c01);
}

// =========================================================================
//  mr=4  kernels
// =========================================================================

template <bool zero_mode = false>
inline void mma_direct_4x1_f16(
    half* NNOPS_RESTRICT C, int ldc,
    const half* NNOPS_RESTRICT A, int lda,
    const half* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const half* NNOPS_RESTRICT A_ptr0 = A + 0 * lda;
    const half* NNOPS_RESTRICT A_ptr1 = A + 1 * lda;
    const half* NNOPS_RESTRICT A_ptr2 = A + 2 * lda;
    const half* NNOPS_RESTRICT A_ptr3 = A + 3 * lda;

    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    for (int k = 0; k < K; ++k) {
        const float b0 = half_to_float(B[0]);
        c0 += half_to_float(*A_ptr0++) * b0;
        c1 += half_to_float(*A_ptr1++) * b0;
        c2 += half_to_float(*A_ptr2++) * b0;
        c3 += half_to_float(*A_ptr3++) * b0;
        B += ldb;
    }

    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c0 += half_to_float(C[0 * ldc]); } } C[0 * ldc] = float_to_half(std::min(std::max(c0, clamp_min), clamp_max));
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c1 += half_to_float(C[1 * ldc]); } } C[1 * ldc] = float_to_half(std::min(std::max(c1, clamp_min), clamp_max));
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c2 += half_to_float(C[2 * ldc]); } } C[2 * ldc] = float_to_half(std::min(std::max(c2, clamp_min), clamp_max));
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { c3 += half_to_float(C[3 * ldc]); } } C[3 * ldc] = float_to_half(std::min(std::max(c3, clamp_min), clamp_max));
}

template <bool zero_mode = false>
inline void mma_direct_4x8_f16(
    half* NNOPS_RESTRICT C, int ldc,
    const half* NNOPS_RESTRICT A, int lda,
    const half* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const half* NNOPS_RESTRICT A_ptr0 = A + 0 * lda;
    const half* NNOPS_RESTRICT A_ptr1 = A + 1 * lda;
    const half* NNOPS_RESTRICT A_ptr2 = A + 2 * lda;
    const half* NNOPS_RESTRICT A_ptr3 = A + 3 * lda;

    __m256 v_c0 = _mm256_setzero_ps();
    __m256 v_c1 = v_c0;
    __m256 v_c2 = v_c0;
    __m256 v_c3 = v_c0;

    for (int k = 0; k < K; ++k) {
        const __m256 v_b0 = load_f16x8(B); B += ldb;

        __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr0[0].bits))); A_ptr0 += 1;
        v_c0 = _mm256_fmadd_ps(v_a0, v_b0, v_c0);
        __m256 v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr1[0].bits))); A_ptr1 += 1;
        v_c1 = _mm256_fmadd_ps(v_a1, v_b0, v_c1);
        __m256 v_a2 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr2[0].bits))); A_ptr2 += 1;
        v_c2 = _mm256_fmadd_ps(v_a2, v_b0, v_c2);
        __m256 v_a3 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr3[0].bits))); A_ptr3 += 1;
        v_c3 = _mm256_fmadd_ps(v_a3, v_b0, v_c3);
    }

    const __m256 v_min = _mm256_set1_ps(clamp_min);
    const __m256 v_max = _mm256_set1_ps(clamp_max);
    if constexpr (!zero_mode) { v_c0 = _mm256_add_ps(v_c0, load_f16x8(C + 0 * ldc)); }
    v_c0 = _mm256_max_ps(_mm256_min_ps(v_c0, v_max), v_min);
    store_f16x8(C + 0 * ldc, v_c0);
    if constexpr (!zero_mode) { v_c1 = _mm256_add_ps(v_c1, load_f16x8(C + 1 * ldc)); }
    v_c1 = _mm256_max_ps(_mm256_min_ps(v_c1, v_max), v_min);
    store_f16x8(C + 1 * ldc, v_c1);
    if constexpr (!zero_mode) { v_c2 = _mm256_add_ps(v_c2, load_f16x8(C + 2 * ldc)); }
    v_c2 = _mm256_max_ps(_mm256_min_ps(v_c2, v_max), v_min);
    store_f16x8(C + 2 * ldc, v_c2);
    if constexpr (!zero_mode) { v_c3 = _mm256_add_ps(v_c3, load_f16x8(C + 3 * ldc)); }
    v_c3 = _mm256_max_ps(_mm256_min_ps(v_c3, v_max), v_min);
    store_f16x8(C + 3 * ldc, v_c3);
}

template <bool zero_mode = false>
inline void mma_direct_4x16_f16(
    half* NNOPS_RESTRICT C, int ldc,
    const half* NNOPS_RESTRICT A, int lda,
    const half* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const half* NNOPS_RESTRICT A_ptr0 = A + 0 * lda;
    const half* NNOPS_RESTRICT A_ptr1 = A + 1 * lda;
    const half* NNOPS_RESTRICT A_ptr2 = A + 2 * lda;
    const half* NNOPS_RESTRICT A_ptr3 = A + 3 * lda;

    __m256 v_c00 = _mm256_setzero_ps();
    __m256 v_c01 = v_c00;
    __m256 v_c10 = v_c00;
    __m256 v_c11 = v_c00;
    __m256 v_c20 = v_c00;
    __m256 v_c21 = v_c00;
    __m256 v_c30 = v_c00;
    __m256 v_c31 = v_c00;

    for (int k = 0; k < K; ++k) {
        const __m256 v_b0 = load_f16x8(B + 0 * 8);
        const __m256 v_b1 = load_f16x8(B + 1 * 8);
        B += ldb;

        __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr0[0].bits))); A_ptr0 += 1;
        v_c00 = _mm256_fmadd_ps(v_a0, v_b0, v_c00);
        v_c01 = _mm256_fmadd_ps(v_a0, v_b1, v_c01);

        __m256 v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr1[0].bits))); A_ptr1 += 1;
        v_c10 = _mm256_fmadd_ps(v_a1, v_b0, v_c10);
        v_c11 = _mm256_fmadd_ps(v_a1, v_b1, v_c11);

        __m256 v_a2 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr2[0].bits))); A_ptr2 += 1;
        v_c20 = _mm256_fmadd_ps(v_a2, v_b0, v_c20);
        v_c21 = _mm256_fmadd_ps(v_a2, v_b1, v_c21);

        __m256 v_a3 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr3[0].bits))); A_ptr3 += 1;
        v_c30 = _mm256_fmadd_ps(v_a3, v_b0, v_c30);
        v_c31 = _mm256_fmadd_ps(v_a3, v_b1, v_c31);
    }

    const __m256 v_min = _mm256_set1_ps(clamp_min);
    const __m256 v_max = _mm256_set1_ps(clamp_max);
    if constexpr (!zero_mode) { v_c00 = _mm256_add_ps(v_c00, load_f16x8(C + 0 * ldc + 0 * 8)); }
    v_c00 = _mm256_max_ps(_mm256_min_ps(v_c00, v_max), v_min);
    store_f16x8(C + 0 * ldc + 0 * 8, v_c00);
    if constexpr (!zero_mode) { v_c01 = _mm256_add_ps(v_c01, load_f16x8(C + 0 * ldc + 1 * 8)); }
    v_c01 = _mm256_max_ps(_mm256_min_ps(v_c01, v_max), v_min);
    store_f16x8(C + 0 * ldc + 1 * 8, v_c01);
    if constexpr (!zero_mode) { v_c10 = _mm256_add_ps(v_c10, load_f16x8(C + 1 * ldc + 0 * 8)); }
    v_c10 = _mm256_max_ps(_mm256_min_ps(v_c10, v_max), v_min);
    store_f16x8(C + 1 * ldc + 0 * 8, v_c10);
    if constexpr (!zero_mode) { v_c11 = _mm256_add_ps(v_c11, load_f16x8(C + 1 * ldc + 1 * 8)); }
    v_c11 = _mm256_max_ps(_mm256_min_ps(v_c11, v_max), v_min);
    store_f16x8(C + 1 * ldc + 1 * 8, v_c11);
    if constexpr (!zero_mode) { v_c20 = _mm256_add_ps(v_c20, load_f16x8(C + 2 * ldc + 0 * 8)); }
    v_c20 = _mm256_max_ps(_mm256_min_ps(v_c20, v_max), v_min);
    store_f16x8(C + 2 * ldc + 0 * 8, v_c20);
    if constexpr (!zero_mode) { v_c21 = _mm256_add_ps(v_c21, load_f16x8(C + 2 * ldc + 1 * 8)); }
    v_c21 = _mm256_max_ps(_mm256_min_ps(v_c21, v_max), v_min);
    store_f16x8(C + 2 * ldc + 1 * 8, v_c21);
    if constexpr (!zero_mode) { v_c30 = _mm256_add_ps(v_c30, load_f16x8(C + 3 * ldc + 0 * 8)); }
    v_c30 = _mm256_max_ps(_mm256_min_ps(v_c30, v_max), v_min);
    store_f16x8(C + 3 * ldc + 0 * 8, v_c30);
    if constexpr (!zero_mode) { v_c31 = _mm256_add_ps(v_c31, load_f16x8(C + 3 * ldc + 1 * 8)); }
    v_c31 = _mm256_max_ps(_mm256_min_ps(v_c31, v_max), v_min);
    store_f16x8(C + 3 * ldc + 1 * 8, v_c31);
}

// =========================================================================
//  mr=6  kernels
// =========================================================================

template <bool zero_mode = false>
inline void mma_direct_6x1_f16(
    half* NNOPS_RESTRICT C, int ldc,
    const half* NNOPS_RESTRICT A, int lda,
    const half* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const half* NNOPS_RESTRICT A_ptr0 = A + 0 * lda;
    const half* NNOPS_RESTRICT A_ptr1 = A + 1 * lda;
    const half* NNOPS_RESTRICT A_ptr2 = A + 2 * lda;
    const half* NNOPS_RESTRICT A_ptr3 = A + 3 * lda;
    const half* NNOPS_RESTRICT A_ptr4 = A + 4 * lda;
    const half* NNOPS_RESTRICT A_ptr5 = A + 5 * lda;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
    float acc3 = 0.0f, acc4 = 0.0f, acc5 = 0.0f;

    for (int k = 0; k < K; ++k) {
        const float b0 = half_to_float(B[0]);
        acc0 += half_to_float(*A_ptr0++) * b0;
        acc1 += half_to_float(*A_ptr1++) * b0;
        acc2 += half_to_float(*A_ptr2++) * b0;
        acc3 += half_to_float(*A_ptr3++) * b0;
        acc4 += half_to_float(*A_ptr4++) * b0;
        acc5 += half_to_float(*A_ptr5++) * b0;
        B += ldb;
    }

    if constexpr (!zero_mode) { if constexpr (!zero_mode) { acc0 += half_to_float(C[0 * ldc]); } } C[0 * ldc] = float_to_half(std::min(std::max(acc0, clamp_min), clamp_max));
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { acc1 += half_to_float(C[1 * ldc]); } } C[1 * ldc] = float_to_half(std::min(std::max(acc1, clamp_min), clamp_max));
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { acc2 += half_to_float(C[2 * ldc]); } } C[2 * ldc] = float_to_half(std::min(std::max(acc2, clamp_min), clamp_max));
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { acc3 += half_to_float(C[3 * ldc]); } } C[3 * ldc] = float_to_half(std::min(std::max(acc3, clamp_min), clamp_max));
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { acc4 += half_to_float(C[4 * ldc]); } } C[4 * ldc] = float_to_half(std::min(std::max(acc4, clamp_min), clamp_max));
    if constexpr (!zero_mode) { if constexpr (!zero_mode) { acc5 += half_to_float(C[5 * ldc]); } } C[5 * ldc] = float_to_half(std::min(std::max(acc5, clamp_min), clamp_max));
}

template <bool zero_mode = false>
inline void mma_direct_6x8_f16(
    half* NNOPS_RESTRICT C, int ldc,
    const half* NNOPS_RESTRICT A, int lda,
    const half* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const half* NNOPS_RESTRICT A_ptr0 = A + 0 * lda;
    const half* NNOPS_RESTRICT A_ptr1 = A + 1 * lda;
    const half* NNOPS_RESTRICT A_ptr2 = A + 2 * lda;
    const half* NNOPS_RESTRICT A_ptr3 = A + 3 * lda;
    const half* NNOPS_RESTRICT A_ptr4 = A + 4 * lda;
    const half* NNOPS_RESTRICT A_ptr5 = A + 5 * lda;

    __m256 v_c0 = _mm256_setzero_ps();
    __m256 v_c1 = v_c0;
    __m256 v_c2 = v_c0;
    __m256 v_c3 = v_c0;
    __m256 v_c4 = v_c0;
    __m256 v_c5 = v_c0;

    for (int k = 0; k < K; ++k) {
        const __m256 v_b0 = load_f16x8(B); B += ldb;

        __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr0[0].bits))); A_ptr0 += 1;
        v_c0 = _mm256_fmadd_ps(v_a0, v_b0, v_c0);
        __m256 v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr1[0].bits))); A_ptr1 += 1;
        v_c1 = _mm256_fmadd_ps(v_a1, v_b0, v_c1);
        __m256 v_a2 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr2[0].bits))); A_ptr2 += 1;
        v_c2 = _mm256_fmadd_ps(v_a2, v_b0, v_c2);
        __m256 v_a3 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr3[0].bits))); A_ptr3 += 1;
        v_c3 = _mm256_fmadd_ps(v_a3, v_b0, v_c3);
        __m256 v_a4 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr4[0].bits))); A_ptr4 += 1;
        v_c4 = _mm256_fmadd_ps(v_a4, v_b0, v_c4);
        __m256 v_a5 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr5[0].bits))); A_ptr5 += 1;
        v_c5 = _mm256_fmadd_ps(v_a5, v_b0, v_c5);
    }

    const __m256 v_min = _mm256_set1_ps(clamp_min);
    const __m256 v_max = _mm256_set1_ps(clamp_max);
    if constexpr (!zero_mode) { v_c0 = _mm256_add_ps(v_c0, load_f16x8(C + 0 * ldc)); }
    v_c0 = _mm256_max_ps(_mm256_min_ps(v_c0, v_max), v_min);
    store_f16x8(C + 0 * ldc, v_c0);
    if constexpr (!zero_mode) { v_c1 = _mm256_add_ps(v_c1, load_f16x8(C + 1 * ldc)); }
    v_c1 = _mm256_max_ps(_mm256_min_ps(v_c1, v_max), v_min);
    store_f16x8(C + 1 * ldc, v_c1);
    if constexpr (!zero_mode) { v_c2 = _mm256_add_ps(v_c2, load_f16x8(C + 2 * ldc)); }
    v_c2 = _mm256_max_ps(_mm256_min_ps(v_c2, v_max), v_min);
    store_f16x8(C + 2 * ldc, v_c2);
    if constexpr (!zero_mode) { v_c3 = _mm256_add_ps(v_c3, load_f16x8(C + 3 * ldc)); }
    v_c3 = _mm256_max_ps(_mm256_min_ps(v_c3, v_max), v_min);
    store_f16x8(C + 3 * ldc, v_c3);
    if constexpr (!zero_mode) { v_c4 = _mm256_add_ps(v_c4, load_f16x8(C + 4 * ldc)); }
    v_c4 = _mm256_max_ps(_mm256_min_ps(v_c4, v_max), v_min);
    store_f16x8(C + 4 * ldc, v_c4);
    if constexpr (!zero_mode) { v_c5 = _mm256_add_ps(v_c5, load_f16x8(C + 5 * ldc)); }
    v_c5 = _mm256_max_ps(_mm256_min_ps(v_c5, v_max), v_min);
    store_f16x8(C + 5 * ldc, v_c5);
}

template <bool zero_mode = false>
inline void mma_direct_6x16_f16(
    half* NNOPS_RESTRICT C, int ldc,
    const half* NNOPS_RESTRICT A, int lda,
    const half* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const half* NNOPS_RESTRICT A_ptr0 = A + 0 * lda;
    const half* NNOPS_RESTRICT A_ptr1 = A + 1 * lda;
    const half* NNOPS_RESTRICT A_ptr2 = A + 2 * lda;
    const half* NNOPS_RESTRICT A_ptr3 = A + 3 * lda;
    const half* NNOPS_RESTRICT A_ptr4 = A + 4 * lda;
    const half* NNOPS_RESTRICT A_ptr5 = A + 5 * lda;

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
        const __m256 v_b0 = load_f16x8(B + 0 * 8);
        const __m256 v_b1 = load_f16x8(B + 1 * 8);
        B += ldb;

        __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr0[0].bits))); A_ptr0 += 1;
        v_c00 = _mm256_fmadd_ps(v_a0, v_b0, v_c00);
        v_c01 = _mm256_fmadd_ps(v_a0, v_b1, v_c01);

        __m256 v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr1[0].bits))); A_ptr1 += 1;
        v_c10 = _mm256_fmadd_ps(v_a1, v_b0, v_c10);
        v_c11 = _mm256_fmadd_ps(v_a1, v_b1, v_c11);

        __m256 v_a2 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr2[0].bits))); A_ptr2 += 1;
        v_c20 = _mm256_fmadd_ps(v_a2, v_b0, v_c20);
        v_c21 = _mm256_fmadd_ps(v_a2, v_b1, v_c21);

        __m256 v_a3 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr3[0].bits))); A_ptr3 += 1;
        v_c30 = _mm256_fmadd_ps(v_a3, v_b0, v_c30);
        v_c31 = _mm256_fmadd_ps(v_a3, v_b1, v_c31);

        __m256 v_a4 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr4[0].bits))); A_ptr4 += 1;
        v_c40 = _mm256_fmadd_ps(v_a4, v_b0, v_c40);
        v_c41 = _mm256_fmadd_ps(v_a4, v_b1, v_c41);

        __m256 v_a5 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr5[0].bits))); A_ptr5 += 1;
        v_c50 = _mm256_fmadd_ps(v_a5, v_b0, v_c50);
        v_c51 = _mm256_fmadd_ps(v_a5, v_b1, v_c51);
    }

    const __m256 v_min = _mm256_set1_ps(clamp_min);
    const __m256 v_max = _mm256_set1_ps(clamp_max);
    if constexpr (!zero_mode) { v_c00 = _mm256_add_ps(v_c00, load_f16x8(C + 0 * ldc + 0 * 8)); }
    v_c00 = _mm256_max_ps(_mm256_min_ps(v_c00, v_max), v_min);
    store_f16x8(C + 0 * ldc + 0 * 8, v_c00);
    if constexpr (!zero_mode) { v_c01 = _mm256_add_ps(v_c01, load_f16x8(C + 0 * ldc + 1 * 8)); }
    v_c01 = _mm256_max_ps(_mm256_min_ps(v_c01, v_max), v_min);
    store_f16x8(C + 0 * ldc + 1 * 8, v_c01);
    if constexpr (!zero_mode) { v_c10 = _mm256_add_ps(v_c10, load_f16x8(C + 1 * ldc + 0 * 8)); }
    v_c10 = _mm256_max_ps(_mm256_min_ps(v_c10, v_max), v_min);
    store_f16x8(C + 1 * ldc + 0 * 8, v_c10);
    if constexpr (!zero_mode) { v_c11 = _mm256_add_ps(v_c11, load_f16x8(C + 1 * ldc + 1 * 8)); }
    v_c11 = _mm256_max_ps(_mm256_min_ps(v_c11, v_max), v_min);
    store_f16x8(C + 1 * ldc + 1 * 8, v_c11);
    if constexpr (!zero_mode) { v_c20 = _mm256_add_ps(v_c20, load_f16x8(C + 2 * ldc + 0 * 8)); }
    v_c20 = _mm256_max_ps(_mm256_min_ps(v_c20, v_max), v_min);
    store_f16x8(C + 2 * ldc + 0 * 8, v_c20);
    if constexpr (!zero_mode) { v_c21 = _mm256_add_ps(v_c21, load_f16x8(C + 2 * ldc + 1 * 8)); }
    v_c21 = _mm256_max_ps(_mm256_min_ps(v_c21, v_max), v_min);
    store_f16x8(C + 2 * ldc + 1 * 8, v_c21);
    if constexpr (!zero_mode) { v_c30 = _mm256_add_ps(v_c30, load_f16x8(C + 3 * ldc + 0 * 8)); }
    v_c30 = _mm256_max_ps(_mm256_min_ps(v_c30, v_max), v_min);
    store_f16x8(C + 3 * ldc + 0 * 8, v_c30);
    if constexpr (!zero_mode) { v_c31 = _mm256_add_ps(v_c31, load_f16x8(C + 3 * ldc + 1 * 8)); }
    v_c31 = _mm256_max_ps(_mm256_min_ps(v_c31, v_max), v_min);
    store_f16x8(C + 3 * ldc + 1 * 8, v_c31);
    if constexpr (!zero_mode) { v_c40 = _mm256_add_ps(v_c40, load_f16x8(C + 4 * ldc + 0 * 8)); }
    v_c40 = _mm256_max_ps(_mm256_min_ps(v_c40, v_max), v_min);
    store_f16x8(C + 4 * ldc + 0 * 8, v_c40);
    if constexpr (!zero_mode) { v_c41 = _mm256_add_ps(v_c41, load_f16x8(C + 4 * ldc + 1 * 8)); }
    v_c41 = _mm256_max_ps(_mm256_min_ps(v_c41, v_max), v_min);
    store_f16x8(C + 4 * ldc + 1 * 8, v_c41);
    if constexpr (!zero_mode) { v_c50 = _mm256_add_ps(v_c50, load_f16x8(C + 5 * ldc + 0 * 8)); }
    v_c50 = _mm256_max_ps(_mm256_min_ps(v_c50, v_max), v_min);
    store_f16x8(C + 5 * ldc + 0 * 8, v_c50);
    if constexpr (!zero_mode) { v_c51 = _mm256_add_ps(v_c51, load_f16x8(C + 5 * ldc + 1 * 8)); }
    v_c51 = _mm256_max_ps(_mm256_min_ps(v_c51, v_max), v_min);
    store_f16x8(C + 5 * ldc + 1 * 8, v_c51);
}

}  // namespace nnops::backend::cpu::x86_64
