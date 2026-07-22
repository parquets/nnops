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

namespace nnops::backend::cpu::x86_64 {

using nnops::backend::cpu::half;
using nnops::backend::cpu::half_to_float;
using nnops::backend::cpu::float_to_half;

namespace {

// Load 8 fp16 values → __m256 (fp32)
inline __m256 load_f16x8(const half* NNOPS_RESTRICT p) noexcept {
    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

// Convert __m256 fp32 → 8 fp16 values and store
inline void store_f16x8(half* NNOPS_RESTRICT p, __m256 v) noexcept {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p),
                     _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
}

// Accumulate C, clamp, store (8 fp16 values)
inline void load_clamp_store_f16x8(half* NNOPS_RESTRICT p, __m256 v_acc,
                                    float clamp_min, float clamp_max) noexcept {
    v_acc = _mm256_add_ps(v_acc, load_f16x8(p));
    const __m256 v_min = _mm256_set1_ps(clamp_min);
    const __m256 v_max = _mm256_set1_ps(clamp_max);
    v_acc = _mm256_max_ps(_mm256_min_ps(v_acc, v_max), v_min);
    store_f16x8(p, v_acc);
}

}  // anonymous namespace

// =========================================================================
//  mr=1  kernels
// =========================================================================

inline void mma_direct_1x1_f16(
    half* NNOPS_RESTRICT C, int ldc,
    const half* NNOPS_RESTRICT A, int lda,
    const half* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const half* NNOPS_RESTRICT A_ptr0 = A;

    float c0 = 0.0f;
    for (int k = 0; k < K; ++k) {
        c0 += half_to_float(A_ptr0[0]) * half_to_float(B[0]);
        A_ptr0 += 1;
        B += ldb;
    }
    c0 += half_to_float(C[0]);
    C[0] = float_to_half(std::min(std::max(c0, clamp_min), clamp_max));
}

inline void mma_direct_1x8_f16(
    half* NNOPS_RESTRICT C, int ldc,
    const half* NNOPS_RESTRICT A, int lda,
    const half* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const half* NNOPS_RESTRICT A_ptr0 = A;

    __m256 v_c0 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        const __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr0[0].bits)));
        const __m256 v_b0 = load_f16x8(B);
        v_c0 = _mm256_fmadd_ps(v_a0, v_b0, v_c0);
        A_ptr0 += 1;
        B += ldb;
    }

    load_clamp_store_f16x8(C, v_c0, clamp_min, clamp_max);
}

inline void mma_direct_1x16_f16(
    half* NNOPS_RESTRICT C, int ldc,
    const half* NNOPS_RESTRICT A, int lda,
    const half* NNOPS_RESTRICT B, int ldb,
    int K, float clamp_min, float clamp_max) noexcept {

    const half* NNOPS_RESTRICT A_ptr0 = A;

    __m256 v_c00 = _mm256_setzero_ps();
    __m256 v_c01 = v_c00;

    for (int k = 0; k < K; ++k) {
        const __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr0[0].bits)));
        const __m256 v_b0 = load_f16x8(B + 0 * 8);
        const __m256 v_b1 = load_f16x8(B + 1 * 8);

        v_c00 = _mm256_fmadd_ps(v_a0, v_b0, v_c00);
        v_c01 = _mm256_fmadd_ps(v_a0, v_b1, v_c01);

        A_ptr0 += 1;
        B += ldb;
    }

    load_clamp_store_f16x8(C + 0 * 8, v_c00, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 1 * 8, v_c01, clamp_min, clamp_max);
}

// =========================================================================
//  mr=4  kernels
// =========================================================================

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
        c0 += half_to_float(A_ptr0[0]) * b0;
        c1 += half_to_float(A_ptr1[0]) * b0;
        c2 += half_to_float(A_ptr2[0]) * b0;
        c3 += half_to_float(A_ptr3[0]) * b0;

        A_ptr0 += 1; A_ptr1 += 1; A_ptr2 += 1; A_ptr3 += 1;
        B += ldb;
    }

    auto write = [&](half* dst, float acc) {
        float v = half_to_float(*dst) + acc;
        *dst = float_to_half(std::min(std::max(v, clamp_min), clamp_max));
    };
    write(C + 0 * ldc, c0);
    write(C + 1 * ldc, c1);
    write(C + 2 * ldc, c2);
    write(C + 3 * ldc, c3);
}

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
        const __m256 v_b0 = load_f16x8(B);

        v_c0 = _mm256_fmadd_ps(
            _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr0[0].bits))), v_b0, v_c0);
        v_c1 = _mm256_fmadd_ps(
            _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr1[0].bits))), v_b0, v_c1);
        v_c2 = _mm256_fmadd_ps(
            _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr2[0].bits))), v_b0, v_c2);
        v_c3 = _mm256_fmadd_ps(
            _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr3[0].bits))), v_b0, v_c3);

        A_ptr0 += 1; A_ptr1 += 1; A_ptr2 += 1; A_ptr3 += 1;
        B += ldb;
    }

    load_clamp_store_f16x8(C + 0 * ldc, v_c0, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 1 * ldc, v_c1, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 2 * ldc, v_c2, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 3 * ldc, v_c3, clamp_min, clamp_max);
}

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

        __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr0[0].bits)));
        v_c00 = _mm256_fmadd_ps(v_a0, v_b0, v_c00);
        v_c01 = _mm256_fmadd_ps(v_a0, v_b1, v_c01);

        __m256 v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr1[0].bits)));
        v_c10 = _mm256_fmadd_ps(v_a1, v_b0, v_c10);
        v_c11 = _mm256_fmadd_ps(v_a1, v_b1, v_c11);

        __m256 v_a2 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr2[0].bits)));
        v_c20 = _mm256_fmadd_ps(v_a2, v_b0, v_c20);
        v_c21 = _mm256_fmadd_ps(v_a2, v_b1, v_c21);

        __m256 v_a3 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr3[0].bits)));
        v_c30 = _mm256_fmadd_ps(v_a3, v_b0, v_c30);
        v_c31 = _mm256_fmadd_ps(v_a3, v_b1, v_c31);

        A_ptr0 += 1; A_ptr1 += 1; A_ptr2 += 1; A_ptr3 += 1;
        B += ldb;
    }

    load_clamp_store_f16x8(C + 0 * ldc + 0 * 8, v_c00, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 0 * ldc + 1 * 8, v_c01, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 1 * ldc + 0 * 8, v_c10, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 1 * ldc + 1 * 8, v_c11, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 2 * ldc + 0 * 8, v_c20, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 2 * ldc + 1 * 8, v_c21, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 3 * ldc + 0 * 8, v_c30, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 3 * ldc + 1 * 8, v_c31, clamp_min, clamp_max);
}

// =========================================================================
//  mr=6  kernels
// =========================================================================

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
        acc0 += half_to_float(A_ptr0[0]) * b0;
        acc1 += half_to_float(A_ptr1[0]) * b0;
        acc2 += half_to_float(A_ptr2[0]) * b0;
        acc3 += half_to_float(A_ptr3[0]) * b0;
        acc4 += half_to_float(A_ptr4[0]) * b0;
        acc5 += half_to_float(A_ptr5[0]) * b0;

        A_ptr0 += 1; A_ptr1 += 1; A_ptr2 += 1;
        A_ptr3 += 1; A_ptr4 += 1; A_ptr5 += 1;
        B += ldb;
    }

    auto write = [&](half* dst, float acc) {
        float v = half_to_float(*dst) + acc;
        *dst = float_to_half(std::min(std::max(v, clamp_min), clamp_max));
    };
    write(C + 0 * ldc, acc0);
    write(C + 1 * ldc, acc1);
    write(C + 2 * ldc, acc2);
    write(C + 3 * ldc, acc3);
    write(C + 4 * ldc, acc4);
    write(C + 5 * ldc, acc5);
}

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
        const __m256 v_b0 = load_f16x8(B);

        v_c0 = _mm256_fmadd_ps(
            _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr0[0].bits))), v_b0, v_c0);
        v_c1 = _mm256_fmadd_ps(
            _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr1[0].bits))), v_b0, v_c1);
        v_c2 = _mm256_fmadd_ps(
            _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr2[0].bits))), v_b0, v_c2);
        v_c3 = _mm256_fmadd_ps(
            _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr3[0].bits))), v_b0, v_c3);
        v_c4 = _mm256_fmadd_ps(
            _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr4[0].bits))), v_b0, v_c4);
        v_c5 = _mm256_fmadd_ps(
            _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr5[0].bits))), v_b0, v_c5);

        A_ptr0 += 1; A_ptr1 += 1; A_ptr2 += 1;
        A_ptr3 += 1; A_ptr4 += 1; A_ptr5 += 1;
        B += ldb;
    }

    load_clamp_store_f16x8(C + 0 * ldc, v_c0, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 1 * ldc, v_c1, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 2 * ldc, v_c2, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 3 * ldc, v_c3, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 4 * ldc, v_c4, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 5 * ldc, v_c5, clamp_min, clamp_max);
}

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

        __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr0[0].bits)));
        v_c00 = _mm256_fmadd_ps(v_a0, v_b0, v_c00);
        v_c01 = _mm256_fmadd_ps(v_a0, v_b1, v_c01);

        __m256 v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr1[0].bits)));
        v_c10 = _mm256_fmadd_ps(v_a1, v_b0, v_c10);
        v_c11 = _mm256_fmadd_ps(v_a1, v_b1, v_c11);

        __m256 v_a2 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr2[0].bits)));
        v_c20 = _mm256_fmadd_ps(v_a2, v_b0, v_c20);
        v_c21 = _mm256_fmadd_ps(v_a2, v_b1, v_c21);

        __m256 v_a3 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr3[0].bits)));
        v_c30 = _mm256_fmadd_ps(v_a3, v_b0, v_c30);
        v_c31 = _mm256_fmadd_ps(v_a3, v_b1, v_c31);

        __m256 v_a4 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr4[0].bits)));
        v_c40 = _mm256_fmadd_ps(v_a4, v_b0, v_c40);
        v_c41 = _mm256_fmadd_ps(v_a4, v_b1, v_c41);

        __m256 v_a5 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A_ptr5[0].bits)));
        v_c50 = _mm256_fmadd_ps(v_a5, v_b0, v_c50);
        v_c51 = _mm256_fmadd_ps(v_a5, v_b1, v_c51);

        A_ptr0 += 1; A_ptr1 += 1; A_ptr2 += 1;
        A_ptr3 += 1; A_ptr4 += 1; A_ptr5 += 1;
        B += ldb;
    }

    load_clamp_store_f16x8(C + 0 * ldc + 0 * 8, v_c00, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 0 * ldc + 1 * 8, v_c01, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 1 * ldc + 0 * 8, v_c10, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 1 * ldc + 1 * 8, v_c11, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 2 * ldc + 0 * 8, v_c20, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 2 * ldc + 1 * 8, v_c21, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 3 * ldc + 0 * 8, v_c30, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 3 * ldc + 1 * 8, v_c31, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 4 * ldc + 0 * 8, v_c40, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 4 * ldc + 1 * 8, v_c41, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 5 * ldc + 0 * 8, v_c50, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 5 * ldc + 1 * 8, v_c51, clamp_min, clamp_max);
}

}  // namespace nnops::backend::cpu::x86_64
