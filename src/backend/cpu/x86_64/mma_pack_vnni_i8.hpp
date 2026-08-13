#pragma once
/// @file mma_pack_vnni_i8.hpp
/// @brief x86_64 VNNI uint8×int8 → int32 MMA (matrix micro-accumulate) packed-B kernels.
///
/// Each kernel computes  C[mr][nr] += A[mr][K] × B_packed[K][nr]  then clamps to
/// [clamp_min, clamp_max]. A is uint8 (packed 4 elements per int32 for VNNI) and B
/// is int8, both pre-packed into the VNNI layout by the caller. `K` is the number
/// of int8 elements (a multiple of 4); each iteration consumes 4.
///
/// Tile sizes (x86_64-optimised):
///   M ∈ {6, 4, 1}    N ∈ {16, 8, 4, 1}
/// Uses _mm256_dpbusd_epi32 (AVX512-VNNI) or _mm256_dpbusd_avx_epi32 (AVX-VNNI).
///
/// Reference: nn_compute/src/cpu/kernel/mma/x86_64/mma_pack_vnni_i8.hpp

#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include <nmmintrin.h>
#include <smmintrin.h>
#include "backend/cpu/common/restrict.hpp"

namespace nnops::backend::cpu::x86_64 {

// =========================================================================
//  mr=1  kernels
// =========================================================================

inline void mma_pack_1x1_u8s8_vnni(int32_t* NNOPS_RESTRICT C, int ldc,
                                   const uint8_t* NNOPS_RESTRICT A,
                                   const int8_t* NNOPS_RESTRICT B, int K,
                                   int32_t clamp_min, int32_t clamp_max) noexcept {
    int32_t c = C[0 * ldc];

    K /= 4;
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

inline void mma_pack_1x4_u8s8_vnni(int32_t* NNOPS_RESTRICT C, int ldc,
                                   const uint8_t* NNOPS_RESTRICT A,
                                   const int8_t* NNOPS_RESTRICT B, int K,
                                   int32_t clamp_min, int32_t clamp_max) noexcept {
    __m128i v_c0 = _mm_loadu_epi32(C + 0 * ldc);

    const int32_t* A_i32_ptr = reinterpret_cast<const int32_t*>(A);
    K /= 4;
    for (int k = 0; k < K; ++k) {
        __m128i v_a0 = _mm_set1_epi32(A_i32_ptr[0]);
        __m128i v_b = _mm_loadu_epi8(B);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        v_c0 = _mm_dpbusd_epi32(v_c0, v_a0, v_b);
#elif defined(__AVXVNNI__)
        v_c0 = _mm_dpbusd_avx_epi32(v_c0, v_a0, v_b);
#endif
        A_i32_ptr += 1;
        B += 16;
    }

    __m128i v_min = _mm_set1_epi32(clamp_min);
    __m128i v_max = _mm_set1_epi32(clamp_max);

    v_c0 = _mm_max_epi32(v_c0, v_min);
    v_c0 = _mm_min_epi32(v_c0, v_max);

    _mm_storeu_epi32(C + 0 * ldc, v_c0);
}

inline void mma_pack_1x8_u8s8_vnni(int32_t* NNOPS_RESTRICT C, int ldc,
                                   const uint8_t* NNOPS_RESTRICT A,
                                   const int8_t* NNOPS_RESTRICT B, int K,
                                   int32_t clamp_min, int32_t clamp_max) noexcept {
    __m256i v_c0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 0 * ldc));

    const int32_t* A_i32_ptr = reinterpret_cast<const int32_t*>(A);
    K /= 4;
    for (int k = 0; k < K; ++k) {
        __m256i v_a0 = _mm256_set1_epi32(A_i32_ptr[0]);
        __m256i v_b = _mm256_loadu_epi8(B);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        v_c0 = _mm256_dpbusd_epi32(v_c0, v_a0, v_b);
#elif defined(__AVXVNNI__)
        v_c0 = _mm256_dpbusd_avx_epi32(v_c0, v_a0, v_b);
#endif
        A_i32_ptr += 1;
        B += 32;
    }

    __m256i v_min = _mm256_set1_epi32(clamp_min);
    __m256i v_max = _mm256_set1_epi32(clamp_max);

    v_c0 = _mm256_max_epi32(v_c0, v_min);
    v_c0 = _mm256_min_epi32(v_c0, v_max);

    _mm256_storeu_epi32(C + 0 * ldc, v_c0);
}

inline void mma_pack_1x16_u8s8_vnni(int32_t* NNOPS_RESTRICT C, int ldc,
                                    const uint8_t* NNOPS_RESTRICT A,
                                    const int8_t* NNOPS_RESTRICT B, int K,
                                    int32_t clamp_min, int32_t clamp_max) noexcept {
    __m256i v_c00 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 0 * ldc + 0));
    __m256i v_c01 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 0 * ldc + 8));

    const int32_t* A_i32_ptr = reinterpret_cast<const int32_t*>(A);

    K /= 4;
    for (int k = 0; k < K; ++k) {
        __m256i v_a0 = _mm256_set1_epi32(A_i32_ptr[0]);

        __m256i v_b0 = _mm256_loadu_epi8(B + 0 * 32);
        __m256i v_b1 = _mm256_loadu_epi8(B + 1 * 32);

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        v_c00 = _mm256_dpbusd_epi32(v_c00, v_a0, v_b0);
        v_c01 = _mm256_dpbusd_epi32(v_c01, v_a0, v_b1);
#elif defined(__AVXVNNI__)
        v_c00 = _mm256_dpbusd_avx_epi32(v_c00, v_a0, v_b0);
        v_c01 = _mm256_dpbusd_avx_epi32(v_c01, v_a0, v_b1);
#endif

        A_i32_ptr += 1;
        B += 64;
    }

    __m256i v_min = _mm256_set1_epi32(clamp_min);
    __m256i v_max = _mm256_set1_epi32(clamp_max);

    v_c00 = _mm256_max_epi32(v_c00, v_min);
    v_c00 = _mm256_min_epi32(v_c00, v_max);
    v_c01 = _mm256_max_epi32(v_c01, v_min);
    v_c01 = _mm256_min_epi32(v_c01, v_max);

    _mm256_storeu_epi32(C + 0 * ldc + 0, v_c00);
    _mm256_storeu_epi32(C + 0 * ldc + 8, v_c01);
}

// =========================================================================
//  mr=4  kernels
// =========================================================================

inline void mma_pack_4x1_u8s8_vnni(int32_t* NNOPS_RESTRICT C, int ldc,
                                   const uint8_t* NNOPS_RESTRICT A,
                                   const int8_t* NNOPS_RESTRICT B, int K,
                                   int32_t clamp_min, int32_t clamp_max) noexcept {
    int32_t c0 = C[0 * ldc];
    int32_t c1 = C[1 * ldc];
    int32_t c2 = C[2 * ldc];
    int32_t c3 = C[3 * ldc];

    K /= 4;
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

inline void mma_pack_4x4_u8s8_vnni(int32_t* NNOPS_RESTRICT C, int ldc,
                                   const uint8_t* NNOPS_RESTRICT A,
                                   const int8_t* NNOPS_RESTRICT B, int K,
                                   int32_t clamp_min, int32_t clamp_max) noexcept {
    __m128i v_c0 = _mm_loadu_epi32(C + 0 * ldc);
    __m128i v_c1 = _mm_loadu_epi32(C + 1 * ldc);
    __m128i v_c2 = _mm_loadu_epi32(C + 2 * ldc);
    __m128i v_c3 = _mm_loadu_epi32(C + 3 * ldc);

    const int32_t* A_i32_ptr = reinterpret_cast<const int32_t*>(A);

    K /= 4;
    for (int k = 0; k < K; ++k) {
        __m128i v_a0 = _mm_set1_epi32(A_i32_ptr[0]);
        __m128i v_a1 = _mm_set1_epi32(A_i32_ptr[1]);
        __m128i v_a2 = _mm_set1_epi32(A_i32_ptr[2]);
        __m128i v_a3 = _mm_set1_epi32(A_i32_ptr[3]);

        __m128i v_b = _mm_loadu_epi8(B);

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        v_c0 = _mm_dpbusd_epi32(v_c0, v_a0, v_b);
        v_c1 = _mm_dpbusd_epi32(v_c1, v_a1, v_b);
        v_c2 = _mm_dpbusd_epi32(v_c2, v_a2, v_b);
        v_c3 = _mm_dpbusd_epi32(v_c3, v_a3, v_b);
#elif defined(__AVXVNNI__)
        v_c0 = _mm_dpbusd_avx_epi32(v_c0, v_a0, v_b);
        v_c1 = _mm_dpbusd_avx_epi32(v_c1, v_a1, v_b);
        v_c2 = _mm_dpbusd_avx_epi32(v_c2, v_a2, v_b);
        v_c3 = _mm_dpbusd_avx_epi32(v_c3, v_a3, v_b);
#endif

        A_i32_ptr += 4;
        B += 16;
    }

    __m128i v_min = _mm_set1_epi32(clamp_min);
    __m128i v_max = _mm_set1_epi32(clamp_max);

    v_c0 = _mm_max_epi32(v_c0, v_min);
    v_c1 = _mm_max_epi32(v_c1, v_min);
    v_c2 = _mm_max_epi32(v_c2, v_min);
    v_c3 = _mm_max_epi32(v_c3, v_min);

    v_c0 = _mm_min_epi32(v_c0, v_max);
    v_c1 = _mm_min_epi32(v_c1, v_max);
    v_c2 = _mm_min_epi32(v_c2, v_max);
    v_c3 = _mm_min_epi32(v_c3, v_max);

    _mm_storeu_epi32(C + 0 * ldc, v_c0);
    _mm_storeu_epi32(C + 1 * ldc, v_c1);
    _mm_storeu_epi32(C + 2 * ldc, v_c2);
    _mm_storeu_epi32(C + 3 * ldc, v_c3);
}

inline void mma_pack_4x8_u8s8_vnni(int32_t* NNOPS_RESTRICT C, int ldc,
                                   const uint8_t* NNOPS_RESTRICT A,
                                   const int8_t* NNOPS_RESTRICT B, int K,
                                   int32_t clamp_min, int32_t clamp_max) noexcept {
    __m256i v_c0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 0 * ldc));
    __m256i v_c1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 1 * ldc));
    __m256i v_c2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 2 * ldc));
    __m256i v_c3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 3 * ldc));

    const int32_t* A_i32_ptr = reinterpret_cast<const int32_t*>(A);

    K /= 4;
    for (int k = 0; k < K; ++k) {
        __m256i v_a0 = _mm256_set1_epi32(A_i32_ptr[0]);
        __m256i v_a1 = _mm256_set1_epi32(A_i32_ptr[1]);
        __m256i v_a2 = _mm256_set1_epi32(A_i32_ptr[2]);
        __m256i v_a3 = _mm256_set1_epi32(A_i32_ptr[3]);

        __m256i v_b = _mm256_loadu_epi8(B);

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        v_c0 = _mm256_dpbusd_epi32(v_c0, v_a0, v_b);
        v_c1 = _mm256_dpbusd_epi32(v_c1, v_a1, v_b);
        v_c2 = _mm256_dpbusd_epi32(v_c2, v_a2, v_b);
        v_c3 = _mm256_dpbusd_epi32(v_c3, v_a3, v_b);
#elif defined(__AVXVNNI__)
        v_c0 = _mm256_dpbusd_avx_epi32(v_c0, v_a0, v_b);
        v_c1 = _mm256_dpbusd_avx_epi32(v_c1, v_a1, v_b);
        v_c2 = _mm256_dpbusd_avx_epi32(v_c2, v_a2, v_b);
        v_c3 = _mm256_dpbusd_avx_epi32(v_c3, v_a3, v_b);
#endif

        A_i32_ptr += 4;
        B += 32;
    }

    __m256i v_min = _mm256_set1_epi32(clamp_min);
    __m256i v_max = _mm256_set1_epi32(clamp_max);

    v_c0 = _mm256_max_epi32(v_c0, v_min);
    v_c1 = _mm256_max_epi32(v_c1, v_min);
    v_c2 = _mm256_max_epi32(v_c2, v_min);
    v_c3 = _mm256_max_epi32(v_c3, v_min);

    v_c0 = _mm256_min_epi32(v_c0, v_max);
    v_c1 = _mm256_min_epi32(v_c1, v_max);
    v_c2 = _mm256_min_epi32(v_c2, v_max);
    v_c3 = _mm256_min_epi32(v_c3, v_max);

    _mm256_storeu_epi32(C + 0 * ldc, v_c0);
    _mm256_storeu_epi32(C + 1 * ldc, v_c1);
    _mm256_storeu_epi32(C + 2 * ldc, v_c2);
    _mm256_storeu_epi32(C + 3 * ldc, v_c3);
}

inline void mma_pack_4x16_u8s8_vnni(int32_t* NNOPS_RESTRICT C, int ldc,
                                    const uint8_t* NNOPS_RESTRICT A,
                                    const int8_t* NNOPS_RESTRICT B, int K,
                                    int32_t clamp_min, int32_t clamp_max) noexcept {
    __m256i v_c00 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 0 * ldc + 0));
    __m256i v_c01 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 0 * ldc + 8));
    __m256i v_c10 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 1 * ldc + 0));
    __m256i v_c11 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 1 * ldc + 8));
    __m256i v_c20 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 2 * ldc + 0));
    __m256i v_c21 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 2 * ldc + 8));
    __m256i v_c30 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 3 * ldc + 0));
    __m256i v_c31 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 3 * ldc + 8));

    const int32_t* A_i32_ptr = reinterpret_cast<const int32_t*>(A);

    K /= 4;
    for (int k = 0; k < K; ++k) {
        __m256i v_a0 = _mm256_set1_epi32(A_i32_ptr[0]);
        __m256i v_a1 = _mm256_set1_epi32(A_i32_ptr[1]);
        __m256i v_a2 = _mm256_set1_epi32(A_i32_ptr[2]);
        __m256i v_a3 = _mm256_set1_epi32(A_i32_ptr[3]);

        __m256i v_b0 = _mm256_loadu_epi8(B + 0 * 32);
        __m256i v_b1 = _mm256_loadu_epi8(B + 1 * 32);

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        v_c00 = _mm256_dpbusd_epi32(v_c00, v_a0, v_b0);
        v_c10 = _mm256_dpbusd_epi32(v_c10, v_a1, v_b0);
        v_c20 = _mm256_dpbusd_epi32(v_c20, v_a2, v_b0);
        v_c30 = _mm256_dpbusd_epi32(v_c30, v_a3, v_b0);

        v_c01 = _mm256_dpbusd_epi32(v_c01, v_a0, v_b1);
        v_c11 = _mm256_dpbusd_epi32(v_c11, v_a1, v_b1);
        v_c21 = _mm256_dpbusd_epi32(v_c21, v_a2, v_b1);
        v_c31 = _mm256_dpbusd_epi32(v_c31, v_a3, v_b1);
#elif defined(__AVXVNNI__)
        v_c00 = _mm256_dpbusd_avx_epi32(v_c00, v_a0, v_b0);
        v_c10 = _mm256_dpbusd_avx_epi32(v_c10, v_a1, v_b0);
        v_c20 = _mm256_dpbusd_avx_epi32(v_c20, v_a2, v_b0);
        v_c30 = _mm256_dpbusd_avx_epi32(v_c30, v_a3, v_b0);

        v_c01 = _mm256_dpbusd_avx_epi32(v_c01, v_a0, v_b1);
        v_c11 = _mm256_dpbusd_avx_epi32(v_c11, v_a1, v_b1);
        v_c21 = _mm256_dpbusd_avx_epi32(v_c21, v_a2, v_b1);
        v_c31 = _mm256_dpbusd_avx_epi32(v_c31, v_a3, v_b1);
#endif

        A_i32_ptr += 4;
        B += 64;
    }

    __m256i v_min = _mm256_set1_epi32(clamp_min);
    __m256i v_max = _mm256_set1_epi32(clamp_max);

    v_c00 = _mm256_max_epi32(v_c00, v_min);
    v_c10 = _mm256_max_epi32(v_c10, v_min);
    v_c20 = _mm256_max_epi32(v_c20, v_min);
    v_c30 = _mm256_max_epi32(v_c30, v_min);

    v_c00 = _mm256_min_epi32(v_c00, v_max);
    v_c10 = _mm256_min_epi32(v_c10, v_max);
    v_c20 = _mm256_min_epi32(v_c20, v_max);
    v_c30 = _mm256_min_epi32(v_c30, v_max);

    v_c01 = _mm256_max_epi32(v_c01, v_min);
    v_c11 = _mm256_max_epi32(v_c11, v_min);
    v_c21 = _mm256_max_epi32(v_c21, v_min);
    v_c31 = _mm256_max_epi32(v_c31, v_min);

    v_c01 = _mm256_min_epi32(v_c01, v_max);
    v_c11 = _mm256_min_epi32(v_c11, v_max);
    v_c21 = _mm256_min_epi32(v_c21, v_max);
    v_c31 = _mm256_min_epi32(v_c31, v_max);

    _mm256_storeu_epi32(C + 0 * ldc + 0, v_c00);
    _mm256_storeu_epi32(C + 0 * ldc + 8, v_c01);
    _mm256_storeu_epi32(C + 1 * ldc + 0, v_c10);
    _mm256_storeu_epi32(C + 1 * ldc + 8, v_c11);
    _mm256_storeu_epi32(C + 2 * ldc + 0, v_c20);
    _mm256_storeu_epi32(C + 2 * ldc + 8, v_c21);
    _mm256_storeu_epi32(C + 3 * ldc + 0, v_c30);
    _mm256_storeu_epi32(C + 3 * ldc + 8, v_c31);
}

// =========================================================================
//  mr=6  kernels
// =========================================================================

inline void mma_pack_6x1_u8s8_vnni(int32_t* NNOPS_RESTRICT C, int ldc,
                                   const uint8_t* NNOPS_RESTRICT A,
                                   const int8_t* NNOPS_RESTRICT B, int K,
                                   int32_t clamp_min, int32_t clamp_max) noexcept {
    int32_t c0 = C[0 * ldc];
    int32_t c1 = C[1 * ldc];
    int32_t c2 = C[2 * ldc];
    int32_t c3 = C[3 * ldc];
    int32_t c4 = C[4 * ldc];
    int32_t c5 = C[5 * ldc];

    K /= 4;
    for (int k = 0; k < K; ++k) {
        c0 += (A[4 * 0 + 0] * B[0] + A[4 * 0 + 1] * B[1] + A[4 * 0 + 2] * B[2] + A[4 * 0 + 3] * B[3]);
        c1 += (A[4 * 1 + 0] * B[0] + A[4 * 1 + 1] * B[1] + A[4 * 1 + 2] * B[2] + A[4 * 1 + 3] * B[3]);
        c2 += (A[4 * 2 + 0] * B[0] + A[4 * 2 + 1] * B[1] + A[4 * 2 + 2] * B[2] + A[4 * 2 + 3] * B[3]);
        c3 += (A[4 * 3 + 0] * B[0] + A[4 * 3 + 1] * B[1] + A[4 * 3 + 2] * B[2] + A[4 * 3 + 3] * B[3]);
        c4 += (A[4 * 4 + 0] * B[0] + A[4 * 4 + 1] * B[1] + A[4 * 4 + 2] * B[2] + A[4 * 4 + 3] * B[3]);
        c5 += (A[4 * 5 + 0] * B[0] + A[4 * 5 + 1] * B[1] + A[4 * 5 + 2] * B[2] + A[4 * 5 + 3] * B[3]);

        A += 24;
        B += 4;
    }

    C[0 * ldc] = std::min(std::max(c0, clamp_min), clamp_max);
    C[1 * ldc] = std::min(std::max(c1, clamp_min), clamp_max);
    C[2 * ldc] = std::min(std::max(c2, clamp_min), clamp_max);
    C[3 * ldc] = std::min(std::max(c3, clamp_min), clamp_max);
    C[4 * ldc] = std::min(std::max(c4, clamp_min), clamp_max);
    C[5 * ldc] = std::min(std::max(c5, clamp_min), clamp_max);
}

inline void mma_pack_6x4_u8s8_vnni(int32_t* NNOPS_RESTRICT C, int ldc,
                                   const uint8_t* NNOPS_RESTRICT A,
                                   const int8_t* NNOPS_RESTRICT B, int K,
                                   int32_t clamp_min, int32_t clamp_max) noexcept {
    __m128i v_c0 = _mm_loadu_epi32(C + 0 * ldc);
    __m128i v_c1 = _mm_loadu_epi32(C + 1 * ldc);
    __m128i v_c2 = _mm_loadu_epi32(C + 2 * ldc);
    __m128i v_c3 = _mm_loadu_epi32(C + 3 * ldc);
    __m128i v_c4 = _mm_loadu_epi32(C + 4 * ldc);
    __m128i v_c5 = _mm_loadu_epi32(C + 5 * ldc);

    const int32_t* A_i32_ptr = reinterpret_cast<const int32_t*>(A);

    K /= 4;
    for (int k = 0; k < K; ++k) {
        __m128i v_b0 = _mm_loadu_epi8(B);

        __m128i v_a0 = _mm_set1_epi32(A_i32_ptr[0]);
        __m128i v_a1 = _mm_set1_epi32(A_i32_ptr[1]);

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        v_c0 = _mm_dpbusd_epi32(v_c0, v_a0, v_b0);
        v_c1 = _mm_dpbusd_epi32(v_c1, v_a1, v_b0);
#elif defined(__AVXVNNI__)
        v_c0 = _mm_dpbusd_avx_epi32(v_c0, v_a0, v_b0);
        v_c1 = _mm_dpbusd_avx_epi32(v_c1, v_a1, v_b0);
#endif
        v_a0 = _mm_set1_epi32(A_i32_ptr[2]);
        v_a1 = _mm_set1_epi32(A_i32_ptr[3]);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        v_c2 = _mm_dpbusd_epi32(v_c2, v_a0, v_b0);
        v_c3 = _mm_dpbusd_epi32(v_c3, v_a1, v_b0);
#elif defined(__AVXVNNI__)
        v_c2 = _mm_dpbusd_avx_epi32(v_c2, v_a0, v_b0);
        v_c3 = _mm_dpbusd_avx_epi32(v_c3, v_a1, v_b0);
#endif

        v_a0 = _mm_set1_epi32(A_i32_ptr[4]);
        v_a1 = _mm_set1_epi32(A_i32_ptr[5]);

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        v_c4 = _mm_dpbusd_epi32(v_c4, v_a0, v_b0);
        v_c5 = _mm_dpbusd_epi32(v_c5, v_a1, v_b0);
#elif defined(__AVXVNNI__)
        v_c4 = _mm_dpbusd_avx_epi32(v_c4, v_a0, v_b0);
        v_c5 = _mm_dpbusd_avx_epi32(v_c5, v_a1, v_b0);
#endif
        A_i32_ptr += 6;
        B += 16;
    }

    __m128i v_min = _mm_set1_epi32(clamp_min);
    __m128i v_max = _mm_set1_epi32(clamp_max);

    v_c0 = _mm_max_epi32(v_c0, v_min);
    v_c1 = _mm_max_epi32(v_c1, v_min);
    v_c2 = _mm_max_epi32(v_c2, v_min);
    v_c3 = _mm_max_epi32(v_c3, v_min);
    v_c4 = _mm_max_epi32(v_c4, v_min);
    v_c5 = _mm_max_epi32(v_c5, v_min);

    v_c0 = _mm_min_epi32(v_c0, v_max);
    v_c1 = _mm_min_epi32(v_c1, v_max);
    v_c2 = _mm_min_epi32(v_c2, v_max);
    v_c3 = _mm_min_epi32(v_c3, v_max);
    v_c4 = _mm_min_epi32(v_c4, v_max);
    v_c5 = _mm_min_epi32(v_c5, v_max);

    _mm_storeu_epi32(C + 0 * ldc, v_c0);
    _mm_storeu_epi32(C + 1 * ldc, v_c1);
    _mm_storeu_epi32(C + 2 * ldc, v_c2);
    _mm_storeu_epi32(C + 3 * ldc, v_c3);
    _mm_storeu_epi32(C + 4 * ldc, v_c4);
    _mm_storeu_epi32(C + 5 * ldc, v_c5);
}

inline void mma_pack_6x8_u8s8_vnni(int32_t* NNOPS_RESTRICT C, int ldc,
                                   const uint8_t* NNOPS_RESTRICT A,
                                   const int8_t* NNOPS_RESTRICT B, int K,
                                   int32_t clamp_min, int32_t clamp_max) noexcept {
    __m256i v_c0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 0 * ldc));
    __m256i v_c1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 1 * ldc));
    __m256i v_c2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 2 * ldc));
    __m256i v_c3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 3 * ldc));
    __m256i v_c4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 4 * ldc));
    __m256i v_c5 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 5 * ldc));

    const int32_t* A_i32_ptr = reinterpret_cast<const int32_t*>(A);

    K /= 4;
    for (int k = 0; k < K; ++k) {
        __m256i v_b0 = _mm256_loadu_epi8(B);

        __m256i v_a0 = _mm256_set1_epi32(A_i32_ptr[0]);
        __m256i v_a1 = _mm256_set1_epi32(A_i32_ptr[1]);

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        v_c0 = _mm256_dpbusd_epi32(v_c0, v_a0, v_b0);
        v_c1 = _mm256_dpbusd_epi32(v_c1, v_a1, v_b0);
#elif defined(__AVXVNNI__)
        v_c0 = _mm256_dpbusd_avx_epi32(v_c0, v_a0, v_b0);
        v_c1 = _mm256_dpbusd_avx_epi32(v_c1, v_a1, v_b0);
#endif
        v_a0 = _mm256_set1_epi32(A_i32_ptr[2]);
        v_a1 = _mm256_set1_epi32(A_i32_ptr[3]);

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        v_c2 = _mm256_dpbusd_epi32(v_c2, v_a0, v_b0);
        v_c3 = _mm256_dpbusd_epi32(v_c3, v_a1, v_b0);
#elif defined(__AVXVNNI__)
        v_c2 = _mm256_dpbusd_avx_epi32(v_c2, v_a0, v_b0);
        v_c3 = _mm256_dpbusd_avx_epi32(v_c3, v_a1, v_b0);
#endif
        v_a0 = _mm256_set1_epi32(A_i32_ptr[4]);
        v_a1 = _mm256_set1_epi32(A_i32_ptr[5]);

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        v_c4 = _mm256_dpbusd_epi32(v_c4, v_a0, v_b0);
        v_c5 = _mm256_dpbusd_epi32(v_c5, v_a1, v_b0);
#elif defined(__AVXVNNI__)
        v_c4 = _mm256_dpbusd_avx_epi32(v_c4, v_a0, v_b0);
        v_c5 = _mm256_dpbusd_avx_epi32(v_c5, v_a1, v_b0);
#endif
        A_i32_ptr += 6;
        B += 32;
    }

    __m256i v_min = _mm256_set1_epi32(clamp_min);
    __m256i v_max = _mm256_set1_epi32(clamp_max);

    v_c0 = _mm256_max_epi32(v_c0, v_min);
    v_c1 = _mm256_max_epi32(v_c1, v_min);
    v_c2 = _mm256_max_epi32(v_c2, v_min);
    v_c3 = _mm256_max_epi32(v_c3, v_min);
    v_c4 = _mm256_max_epi32(v_c4, v_min);
    v_c5 = _mm256_max_epi32(v_c5, v_min);

    v_c0 = _mm256_min_epi32(v_c0, v_max);
    v_c1 = _mm256_min_epi32(v_c1, v_max);
    v_c2 = _mm256_min_epi32(v_c2, v_max);
    v_c3 = _mm256_min_epi32(v_c3, v_max);
    v_c4 = _mm256_min_epi32(v_c4, v_max);
    v_c5 = _mm256_min_epi32(v_c5, v_max);

    _mm256_storeu_epi32(C + 0 * ldc, v_c0);
    _mm256_storeu_epi32(C + 1 * ldc, v_c1);
    _mm256_storeu_epi32(C + 2 * ldc, v_c2);
    _mm256_storeu_epi32(C + 3 * ldc, v_c3);
    _mm256_storeu_epi32(C + 4 * ldc, v_c4);
    _mm256_storeu_epi32(C + 5 * ldc, v_c5);
}

inline void mma_pack_6x16_u8s8_vnni(int32_t* NNOPS_RESTRICT C, int ldc,
                                    const uint8_t* NNOPS_RESTRICT A,
                                    const int8_t* NNOPS_RESTRICT B, int K,
                                    int32_t clamp_min, int32_t clamp_max) noexcept {
    __m256i v_c00 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 0 * ldc + 0));
    __m256i v_c01 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 0 * ldc + 8));
    __m256i v_c10 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 1 * ldc + 0));
    __m256i v_c11 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 1 * ldc + 8));
    __m256i v_c20 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 2 * ldc + 0));
    __m256i v_c21 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 2 * ldc + 8));
    __m256i v_c30 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 3 * ldc + 0));
    __m256i v_c31 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 3 * ldc + 8));
    __m256i v_c40 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 4 * ldc + 0));
    __m256i v_c41 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 4 * ldc + 8));
    __m256i v_c50 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 5 * ldc + 0));
    __m256i v_c51 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + 5 * ldc + 8));

    const int32_t* A_i32_ptr = reinterpret_cast<const int32_t*>(A);

    K /= 4;
    for (int k = 0; k < K; ++k) {
        __m256i v_b0 = _mm256_loadu_epi8(B + 0 * 32);
        __m256i v_b1 = _mm256_loadu_epi8(B + 1 * 32);

        __m256i v_a0 = _mm256_set1_epi32(A_i32_ptr[0]);
        __m256i v_a1 = _mm256_set1_epi32(A_i32_ptr[1]);

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        v_c00 = _mm256_dpbusd_epi32(v_c00, v_a0, v_b0);
        v_c10 = _mm256_dpbusd_epi32(v_c10, v_a1, v_b0);
        v_c01 = _mm256_dpbusd_epi32(v_c01, v_a0, v_b1);
        v_c11 = _mm256_dpbusd_epi32(v_c11, v_a1, v_b1);
#elif defined(__AVXVNNI__)
        v_c00 = _mm256_dpbusd_avx_epi32(v_c00, v_a0, v_b0);
        v_c10 = _mm256_dpbusd_avx_epi32(v_c10, v_a1, v_b0);
        v_c01 = _mm256_dpbusd_avx_epi32(v_c01, v_a0, v_b1);
        v_c11 = _mm256_dpbusd_avx_epi32(v_c11, v_a1, v_b1);
#endif
        v_a0 = _mm256_set1_epi32(A_i32_ptr[2]);
        v_a1 = _mm256_set1_epi32(A_i32_ptr[3]);

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        v_c20 = _mm256_dpbusd_epi32(v_c20, v_a0, v_b0);
        v_c30 = _mm256_dpbusd_epi32(v_c30, v_a1, v_b0);
        v_c21 = _mm256_dpbusd_epi32(v_c21, v_a0, v_b1);
        v_c31 = _mm256_dpbusd_epi32(v_c31, v_a1, v_b1);
#elif defined(__AVXVNNI__)
        v_c20 = _mm256_dpbusd_avx_epi32(v_c20, v_a0, v_b0);
        v_c30 = _mm256_dpbusd_avx_epi32(v_c30, v_a1, v_b0);
        v_c21 = _mm256_dpbusd_avx_epi32(v_c21, v_a0, v_b1);
        v_c31 = _mm256_dpbusd_avx_epi32(v_c31, v_a1, v_b1);
#endif

        v_a0 = _mm256_set1_epi32(A_i32_ptr[4]);
        v_a1 = _mm256_set1_epi32(A_i32_ptr[5]);

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        v_c40 = _mm256_dpbusd_epi32(v_c40, v_a0, v_b0);
        v_c50 = _mm256_dpbusd_epi32(v_c50, v_a1, v_b0);
        v_c41 = _mm256_dpbusd_epi32(v_c41, v_a0, v_b1);
        v_c51 = _mm256_dpbusd_epi32(v_c51, v_a1, v_b1);
#elif defined(__AVXVNNI__)
        v_c40 = _mm256_dpbusd_avx_epi32(v_c40, v_a0, v_b0);
        v_c50 = _mm256_dpbusd_avx_epi32(v_c50, v_a1, v_b0);
        v_c41 = _mm256_dpbusd_avx_epi32(v_c41, v_a0, v_b1);
        v_c51 = _mm256_dpbusd_avx_epi32(v_c51, v_a1, v_b1);
#endif
        A_i32_ptr += 6;
        B += 64;
    }

    __m256i v_min = _mm256_set1_epi32(clamp_min);
    __m256i v_max = _mm256_set1_epi32(clamp_max);

    v_c00 = _mm256_max_epi32(v_c00, v_min);
    v_c10 = _mm256_max_epi32(v_c10, v_min);
    v_c20 = _mm256_max_epi32(v_c20, v_min);
    v_c30 = _mm256_max_epi32(v_c30, v_min);
    v_c40 = _mm256_max_epi32(v_c40, v_min);
    v_c50 = _mm256_max_epi32(v_c50, v_min);

    v_c00 = _mm256_min_epi32(v_c00, v_max);
    v_c10 = _mm256_min_epi32(v_c10, v_max);
    v_c20 = _mm256_min_epi32(v_c20, v_max);
    v_c30 = _mm256_min_epi32(v_c30, v_max);
    v_c40 = _mm256_min_epi32(v_c40, v_max);
    v_c50 = _mm256_min_epi32(v_c50, v_max);

    v_c01 = _mm256_max_epi32(v_c01, v_min);
    v_c11 = _mm256_max_epi32(v_c11, v_min);
    v_c21 = _mm256_max_epi32(v_c21, v_min);
    v_c31 = _mm256_max_epi32(v_c31, v_min);
    v_c41 = _mm256_max_epi32(v_c41, v_min);
    v_c51 = _mm256_max_epi32(v_c51, v_min);

    v_c01 = _mm256_min_epi32(v_c01, v_max);
    v_c11 = _mm256_min_epi32(v_c11, v_max);
    v_c21 = _mm256_min_epi32(v_c21, v_max);
    v_c31 = _mm256_min_epi32(v_c31, v_max);
    v_c41 = _mm256_min_epi32(v_c41, v_max);
    v_c51 = _mm256_min_epi32(v_c51, v_max);

    _mm256_storeu_epi32(C + 0 * ldc + 0, v_c00);
    _mm256_storeu_epi32(C + 0 * ldc + 8, v_c01);
    _mm256_storeu_epi32(C + 1 * ldc + 0, v_c10);
    _mm256_storeu_epi32(C + 1 * ldc + 8, v_c11);
    _mm256_storeu_epi32(C + 2 * ldc + 0, v_c20);
    _mm256_storeu_epi32(C + 2 * ldc + 8, v_c21);
    _mm256_storeu_epi32(C + 3 * ldc + 0, v_c30);
    _mm256_storeu_epi32(C + 3 * ldc + 8, v_c31);
    _mm256_storeu_epi32(C + 4 * ldc + 0, v_c40);
    _mm256_storeu_epi32(C + 4 * ldc + 8, v_c41);
    _mm256_storeu_epi32(C + 5 * ldc + 0, v_c50);
    _mm256_storeu_epi32(C + 5 * ldc + 8, v_c51);
}

}  // namespace nnops::backend::cpu::x86_64
