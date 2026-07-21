#pragma once
/// @file mma_pack_f16.hpp
/// @brief x86_64 F16C/FMA float16 MMA (matrix micro-accumulate) packed-B kernels.
///
/// These use F16C intrinsics for fp16 <-> fp32 conversion and FMA for the
/// inner product. Accumulators are fp32; results are clamped in fp32 and
/// rounded back to fp16 before store.
///
/// Tile sizes (x86_64-optimised):
///   M ∈ {6, 4, 1}    N ∈ {16, 8, 1}
///
/// Reference: nn_compute/src/cpu/kernel/mma/x86_64/mma_pack_f16.hpp

#include <immintrin.h>
#include <algorithm>
#include "backend/cpu/common/half.hpp"
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

// Load 8 fp16, convert to fp32, then clamp → fp16 and store
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

inline void mma_pack_1x1_f16(half* NNOPS_RESTRICT C, int ldc,
                             const half* NNOPS_RESTRICT A,
                             const half* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    float c0 = 0.0f;
    for (int k = 0; k < K; ++k) {
        c0 += half_to_float(A[0]) * half_to_float(B[0]);
        A += 1;
        B += ldb;
    }
    c0 += half_to_float(C[0 * ldc]);
    C[0 * ldc] = float_to_half(std::min(std::max(c0, clamp_min), clamp_max));
}

inline void mma_pack_1x8_f16(half* NNOPS_RESTRICT C, int ldc,
                             const half* NNOPS_RESTRICT A,
                             const half* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    __m256 v_c0 = _mm256_setzero_ps();
    for (int k = 0; k < K; ++k) {
        const __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A[0].bits)));
        const __m256 v_b0 = load_f16x8(B);
        v_c0 = _mm256_fmadd_ps(v_a0, v_b0, v_c0);
        A += 1;
        B += ldb;
    }
    load_clamp_store_f16x8(C, v_c0, clamp_min, clamp_max);
}

inline void mma_pack_1x16_f16(half* NNOPS_RESTRICT C, int ldc,
                              const half* NNOPS_RESTRICT A,
                              const half* NNOPS_RESTRICT B,
                              int ldb, int K,
                              float clamp_min, float clamp_max) noexcept {
    __m256 v_c0 = _mm256_setzero_ps();
    __m256 v_c1 = _mm256_setzero_ps();
    for (int k = 0; k < K; ++k) {
        const __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A[0].bits)));
        const __m256 v_b0 = load_f16x8(B + 0 * 8);
        const __m256 v_b1 = load_f16x8(B + 1 * 8);
        v_c0 = _mm256_fmadd_ps(v_a0, v_b0, v_c0);
        v_c1 = _mm256_fmadd_ps(v_a0, v_b1, v_c1);
        A += 1;
        B += ldb;
    }
    load_clamp_store_f16x8(C + 0 * 8, v_c0, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 1 * 8, v_c1, clamp_min, clamp_max);
}

// =========================================================================
//  mr=4  kernels
// =========================================================================

inline void mma_pack_4x1_f16(half* NNOPS_RESTRICT C, int ldc,
                             const half* NNOPS_RESTRICT A,
                             const half* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float b = half_to_float(B[0]);
        c0 += half_to_float(A[0]) * b;
        c1 += half_to_float(A[1]) * b;
        c2 += half_to_float(A[2]) * b;
        c3 += half_to_float(A[3]) * b;
        A += 4;
        B += ldb;
    }
    auto clamp = [=](float v) noexcept {
        return std::min(std::max(v, clamp_min), clamp_max);
    };
    C[0 * ldc] = float_to_half(clamp(c0 + half_to_float(C[0 * ldc])));
    C[1 * ldc] = float_to_half(clamp(c1 + half_to_float(C[1 * ldc])));
    C[2 * ldc] = float_to_half(clamp(c2 + half_to_float(C[2 * ldc])));
    C[3 * ldc] = float_to_half(clamp(c3 + half_to_float(C[3 * ldc])));
}

inline void mma_pack_4x8_f16(half* NNOPS_RESTRICT C, int ldc,
                             const half* NNOPS_RESTRICT A,
                             const half* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    __m256 v_c0 = _mm256_setzero_ps();
    __m256 v_c1 = _mm256_setzero_ps();
    __m256 v_c2 = _mm256_setzero_ps();
    __m256 v_c3 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        const __m256 v_b0 = load_f16x8(B);
        v_c0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A[0].bits))), v_b0, v_c0);
        v_c1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A[1].bits))), v_b0, v_c1);
        v_c2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A[2].bits))), v_b0, v_c2);
        v_c3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A[3].bits))), v_b0, v_c3);
        A += 4;
        B += ldb;
    }

    load_clamp_store_f16x8(C + 0 * ldc, v_c0, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 1 * ldc, v_c1, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 2 * ldc, v_c2, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 3 * ldc, v_c3, clamp_min, clamp_max);
}

inline void mma_pack_4x16_f16(half* NNOPS_RESTRICT C, int ldc,
                              const half* NNOPS_RESTRICT A,
                              const half* NNOPS_RESTRICT B,
                              int ldb, int K,
                              float clamp_min, float clamp_max) noexcept {
    __m256 v_c00 = _mm256_setzero_ps(), v_c01 = _mm256_setzero_ps();
    __m256 v_c10 = _mm256_setzero_ps(), v_c11 = _mm256_setzero_ps();
    __m256 v_c20 = _mm256_setzero_ps(), v_c21 = _mm256_setzero_ps();
    __m256 v_c30 = _mm256_setzero_ps(), v_c31 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        const __m256 v_b0 = load_f16x8(B + 0 * 8);
        const __m256 v_b1 = load_f16x8(B + 1 * 8);

        __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A[0].bits)));
        __m256 v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A[1].bits)));
        v_c00 = _mm256_fmadd_ps(v_a0, v_b0, v_c00);
        v_c01 = _mm256_fmadd_ps(v_a0, v_b1, v_c01);
        v_c10 = _mm256_fmadd_ps(v_a1, v_b0, v_c10);
        v_c11 = _mm256_fmadd_ps(v_a1, v_b1, v_c11);

        v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A[2].bits)));
        v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A[3].bits)));
        v_c20 = _mm256_fmadd_ps(v_a0, v_b0, v_c20);
        v_c21 = _mm256_fmadd_ps(v_a0, v_b1, v_c21);
        v_c30 = _mm256_fmadd_ps(v_a1, v_b0, v_c30);
        v_c31 = _mm256_fmadd_ps(v_a1, v_b1, v_c31);

        A += 4;
        B += ldb;
    }

    load_clamp_store_f16x8(C + 0 * ldc + 0, v_c00, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 0 * ldc + 8, v_c01, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 1 * ldc + 0, v_c10, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 1 * ldc + 8, v_c11, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 2 * ldc + 0, v_c20, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 2 * ldc + 8, v_c21, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 3 * ldc + 0, v_c30, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 3 * ldc + 8, v_c31, clamp_min, clamp_max);
}

// =========================================================================
//  mr=6  kernels
// =========================================================================

inline void mma_pack_6x1_f16(half* NNOPS_RESTRICT C, int ldc,
                             const half* NNOPS_RESTRICT A,
                             const half* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f, c4 = 0.0f, c5 = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float b = half_to_float(B[0]);
        c0 += half_to_float(A[0]) * b;
        c1 += half_to_float(A[1]) * b;
        c2 += half_to_float(A[2]) * b;
        c3 += half_to_float(A[3]) * b;
        c4 += half_to_float(A[4]) * b;
        c5 += half_to_float(A[5]) * b;
        A += 6;
        B += ldb;
    }
    auto clamp = [=](float v) noexcept {
        return std::min(std::max(v, clamp_min), clamp_max);
    };
    C[0 * ldc] = float_to_half(clamp(c0 + half_to_float(C[0 * ldc])));
    C[1 * ldc] = float_to_half(clamp(c1 + half_to_float(C[1 * ldc])));
    C[2 * ldc] = float_to_half(clamp(c2 + half_to_float(C[2 * ldc])));
    C[3 * ldc] = float_to_half(clamp(c3 + half_to_float(C[3 * ldc])));
    C[4 * ldc] = float_to_half(clamp(c4 + half_to_float(C[4 * ldc])));
    C[5 * ldc] = float_to_half(clamp(c5 + half_to_float(C[5 * ldc])));
}

inline void mma_pack_6x8_f16(half* NNOPS_RESTRICT C, int ldc,
                             const half* NNOPS_RESTRICT A,
                             const half* NNOPS_RESTRICT B,
                             int ldb, int K,
                             float clamp_min, float clamp_max) noexcept {
    __m256 v_c0 = _mm256_setzero_ps();
    __m256 v_c1 = _mm256_setzero_ps();
    __m256 v_c2 = _mm256_setzero_ps();
    __m256 v_c3 = _mm256_setzero_ps();
    __m256 v_c4 = _mm256_setzero_ps();
    __m256 v_c5 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        const __m256 v_b0 = load_f16x8(B);
        v_c0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A[0].bits))), v_b0, v_c0);
        v_c1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A[1].bits))), v_b0, v_c1);
        v_c2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A[2].bits))), v_b0, v_c2);
        v_c3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A[3].bits))), v_b0, v_c3);
        v_c4 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A[4].bits))), v_b0, v_c4);
        v_c5 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A[5].bits))), v_b0, v_c5);
        A += 6;
        B += ldb;
    }

    load_clamp_store_f16x8(C + 0 * ldc, v_c0, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 1 * ldc, v_c1, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 2 * ldc, v_c2, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 3 * ldc, v_c3, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 4 * ldc, v_c4, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 5 * ldc, v_c5, clamp_min, clamp_max);
}

inline void mma_pack_6x16_f16(half* NNOPS_RESTRICT C, int ldc,
                              const half* NNOPS_RESTRICT A,
                              const half* NNOPS_RESTRICT B,
                              int ldb, int K,
                              float clamp_min, float clamp_max) noexcept {
    __m256 v_c00 = _mm256_setzero_ps(), v_c01 = _mm256_setzero_ps();
    __m256 v_c10 = _mm256_setzero_ps(), v_c11 = _mm256_setzero_ps();
    __m256 v_c20 = _mm256_setzero_ps(), v_c21 = _mm256_setzero_ps();
    __m256 v_c30 = _mm256_setzero_ps(), v_c31 = _mm256_setzero_ps();
    __m256 v_c40 = _mm256_setzero_ps(), v_c41 = _mm256_setzero_ps();
    __m256 v_c50 = _mm256_setzero_ps(), v_c51 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        const __m256 v_b0 = load_f16x8(B + 0 * 8);
        const __m256 v_b1 = load_f16x8(B + 1 * 8);

        __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A[0].bits)));
        __m256 v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A[1].bits)));
        v_c00 = _mm256_fmadd_ps(v_a0, v_b0, v_c00);
        v_c01 = _mm256_fmadd_ps(v_a0, v_b1, v_c01);
        v_c10 = _mm256_fmadd_ps(v_a1, v_b0, v_c10);
        v_c11 = _mm256_fmadd_ps(v_a1, v_b1, v_c11);

        v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A[2].bits)));
        v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A[3].bits)));
        v_c20 = _mm256_fmadd_ps(v_a0, v_b0, v_c20);
        v_c21 = _mm256_fmadd_ps(v_a0, v_b1, v_c21);
        v_c30 = _mm256_fmadd_ps(v_a1, v_b0, v_c30);
        v_c31 = _mm256_fmadd_ps(v_a1, v_b1, v_c31);

        v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A[4].bits)));
        v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(A[5].bits)));
        v_c40 = _mm256_fmadd_ps(v_a0, v_b0, v_c40);
        v_c41 = _mm256_fmadd_ps(v_a0, v_b1, v_c41);
        v_c50 = _mm256_fmadd_ps(v_a1, v_b0, v_c50);
        v_c51 = _mm256_fmadd_ps(v_a1, v_b1, v_c51);

        A += 6;
        B += ldb;
    }

    load_clamp_store_f16x8(C + 0 * ldc + 0, v_c00, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 0 * ldc + 8, v_c01, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 1 * ldc + 0, v_c10, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 1 * ldc + 8, v_c11, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 2 * ldc + 0, v_c20, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 2 * ldc + 8, v_c21, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 3 * ldc + 0, v_c30, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 3 * ldc + 8, v_c31, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 4 * ldc + 0, v_c40, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 4 * ldc + 8, v_c41, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 5 * ldc + 0, v_c50, clamp_min, clamp_max);
    load_clamp_store_f16x8(C + 5 * ldc + 8, v_c51, clamp_min, clamp_max);
}

}  // namespace nnops::backend::cpu::x86_64
