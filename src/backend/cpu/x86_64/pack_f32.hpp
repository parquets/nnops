#pragma once
/// @file pack_f32.hpp
/// @brief x86_64 SSE/AVX float32 pack kernels for GEMM LHS/RHS.
///
/// The AVX path operates on 8-element __m256 vectors, giving a naturally
/// unrolled inner loop with K step of 8.
///
/// Reference: nn_compute/src/cpu/kernel/pack/x86_64/pack_f32.hpp

#include <immintrin.h>
#include "backend/cpu/common/restrict.hpp"
#include "transpose.hpp"

namespace nnops::backend::cpu::x86_64 {

inline void pack_trans_n1_f32(float* NNOPS_RESTRICT output,
                              const float* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale) noexcept {
    for (int k = 0; k < K; ++k) {
        *output++ = input[0] * scale;
        input += 1;  // consecutive k within the M×K source row
    }
}

inline void pack_trans_n4_f32(float* NNOPS_RESTRICT output,
                              const float* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale) noexcept {
    const float* NNOPS_RESTRICT p0 = input + 0 * ir_step;
    const float* NNOPS_RESTRICT p1 = input + 1 * ir_step;
    const float* NNOPS_RESTRICT p2 = input + 2 * ir_step;
    const float* NNOPS_RESTRICT p3 = input + 3 * ir_step;

    const __m256 v_scale = _mm256_set1_ps(scale);
    int k = 0;
    for (; k <= K - 8; k += 8) {
        __m256 v0 = _mm256_loadu_ps(p0); p0 += 8;
        __m256 v1 = _mm256_loadu_ps(p1); p1 += 8;
        __m256 v2 = _mm256_loadu_ps(p2); p2 += 8;
        __m256 v3 = _mm256_loadu_ps(p3); p3 += 8;

        transpose_4x8_f32(v0, v1, v2, v3);

        v0 = _mm256_mul_ps(v0, v_scale);
        v1 = _mm256_mul_ps(v1, v_scale);
        v2 = _mm256_mul_ps(v2, v_scale);
        v3 = _mm256_mul_ps(v3, v_scale);

        _mm256_storeu_ps(output + 0 * 8, v0);
        _mm256_storeu_ps(output + 1 * 8, v1);
        _mm256_storeu_ps(output + 2 * 8, v2);
        _mm256_storeu_ps(output + 3 * 8, v3);

        output += 4 * 8;
    }
    // scalar tail
    for (; k < K; ++k) {
        *output++ = *(p0++) * scale;
        *output++ = *(p1++) * scale;
        *output++ = *(p2++) * scale;
        *output++ = *(p3++) * scale;
    }
}

inline void pack_trans_n6_f32(float* NNOPS_RESTRICT output,
                              const float* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale) noexcept {
    const float* NNOPS_RESTRICT p0 = input + 0 * ir_step;
    const float* NNOPS_RESTRICT p1 = input + 1 * ir_step;
    const float* NNOPS_RESTRICT p2 = input + 2 * ir_step;
    const float* NNOPS_RESTRICT p3 = input + 3 * ir_step;
    const float* NNOPS_RESTRICT p4 = input + 4 * ir_step;
    const float* NNOPS_RESTRICT p5 = input + 5 * ir_step;

    const __m256 v_scale = _mm256_set1_ps(scale);
    int k = 0;
    for (; k <= K - 8; k += 8) {
        __m256 v0 = _mm256_loadu_ps(p0); p0 += 8;
        __m256 v1 = _mm256_loadu_ps(p1); p1 += 8;
        __m256 v2 = _mm256_loadu_ps(p2); p2 += 8;
        __m256 v3 = _mm256_loadu_ps(p3); p3 += 8;
        __m256 v4 = _mm256_loadu_ps(p4); p4 += 8;
        __m256 v5 = _mm256_loadu_ps(p5); p5 += 8;

        transpose_6x8_f32(v0, v1, v2, v3, v4, v5);

        v0 = _mm256_mul_ps(v0, v_scale);
        v1 = _mm256_mul_ps(v1, v_scale);
        v2 = _mm256_mul_ps(v2, v_scale);
        v3 = _mm256_mul_ps(v3, v_scale);
        v4 = _mm256_mul_ps(v4, v_scale);
        v5 = _mm256_mul_ps(v5, v_scale);

        _mm256_storeu_ps(output + 0 * 8, v0);
        _mm256_storeu_ps(output + 1 * 8, v1);
        _mm256_storeu_ps(output + 2 * 8, v2);
        _mm256_storeu_ps(output + 3 * 8, v3);
        _mm256_storeu_ps(output + 4 * 8, v4);
        _mm256_storeu_ps(output + 5 * 8, v5);

        output += 6 * 8;
    }
    for (; k < K; ++k) {
        *output++ = *(p0++) * scale;
        *output++ = *(p1++) * scale;
        *output++ = *(p2++) * scale;
        *output++ = *(p3++) * scale;
        *output++ = *(p4++) * scale;
        *output++ = *(p5++) * scale;
    }
}

inline void pack_trans_n8_f32(float* NNOPS_RESTRICT output,
                              const float* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale) noexcept {
    const float* NNOPS_RESTRICT p0 = input + 0 * ir_step;
    const float* NNOPS_RESTRICT p1 = input + 1 * ir_step;
    const float* NNOPS_RESTRICT p2 = input + 2 * ir_step;
    const float* NNOPS_RESTRICT p3 = input + 3 * ir_step;
    const float* NNOPS_RESTRICT p4 = input + 4 * ir_step;
    const float* NNOPS_RESTRICT p5 = input + 5 * ir_step;
    const float* NNOPS_RESTRICT p6 = input + 6 * ir_step;
    const float* NNOPS_RESTRICT p7 = input + 7 * ir_step;

    const __m256 v_scale = _mm256_set1_ps(scale);
    int k = 0;
    for (; k <= K - 8; k += 8) {
        __m256 v0 = _mm256_loadu_ps(p0); p0 += 8;
        __m256 v1 = _mm256_loadu_ps(p1); p1 += 8;
        __m256 v2 = _mm256_loadu_ps(p2); p2 += 8;
        __m256 v3 = _mm256_loadu_ps(p3); p3 += 8;
        __m256 v4 = _mm256_loadu_ps(p4); p4 += 8;
        __m256 v5 = _mm256_loadu_ps(p5); p5 += 8;
        __m256 v6 = _mm256_loadu_ps(p6); p6 += 8;
        __m256 v7 = _mm256_loadu_ps(p7); p7 += 8;

        transpose_8x8_f32(v0, v1, v2, v3, v4, v5, v6, v7);

        v0 = _mm256_mul_ps(v0, v_scale);
        v1 = _mm256_mul_ps(v1, v_scale);
        v2 = _mm256_mul_ps(v2, v_scale);
        v3 = _mm256_mul_ps(v3, v_scale);
        v4 = _mm256_mul_ps(v4, v_scale);
        v5 = _mm256_mul_ps(v5, v_scale);
        v6 = _mm256_mul_ps(v6, v_scale);
        v7 = _mm256_mul_ps(v7, v_scale);

        _mm256_storeu_ps(output + 0 * 8, v0);
        _mm256_storeu_ps(output + 1 * 8, v1);
        _mm256_storeu_ps(output + 2 * 8, v2);
        _mm256_storeu_ps(output + 3 * 8, v3);
        _mm256_storeu_ps(output + 4 * 8, v4);
        _mm256_storeu_ps(output + 5 * 8, v5);
        _mm256_storeu_ps(output + 6 * 8, v6);
        _mm256_storeu_ps(output + 7 * 8, v7);

        output += 8 * 8;
    }
    for (; k < K; ++k) {
        *output++ = *(p0++) * scale;
        *output++ = *(p1++) * scale;
        *output++ = *(p2++) * scale;
        *output++ = *(p3++) * scale;
        *output++ = *(p4++) * scale;
        *output++ = *(p5++) * scale;
        *output++ = *(p6++) * scale;
        *output++ = *(p7++) * scale;
    }
}

inline void pack_trans_n16_f32(float* NNOPS_RESTRICT output,
                               const float* NNOPS_RESTRICT input,
                               int ir_step, int K, float scale) noexcept {
    const float* NNOPS_RESTRICT p[16];
    for (int i = 0; i < 16; ++i) {
        p[i] = input + i * ir_step;
    }

    const __m256 v_scale = _mm256_set1_ps(scale);
    int k = 0;
    for (; k <= K - 8; k += 8) {
        __m256 v[16];
        for (int i = 0; i < 16; ++i) {
            v[i] = _mm256_loadu_ps(p[i]); p[i] += 8;
        }

        transpose_16x8_f32(
            v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
            v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);

        for (int i = 0; i < 16; ++i) {
            v[i] = _mm256_mul_ps(v[i], v_scale);
            _mm256_storeu_ps(output + i * 8, v[i]);
        }

        output += 16 * 8;
    }
    for (; k < K; ++k) {
        for (int i = 0; i < 16; ++i) {
            *output++ = *(p[i]++) * scale;
        }
    }
}

inline void pack_copy_n1_f32(float* NNOPS_RESTRICT output,
                             const float* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale) noexcept {
    for (int k = 0; k < K; ++k) {
        *output++ = input[0] * scale;
        input += ir_step;
    }
}

inline void pack_copy_n4_f32(float* NNOPS_RESTRICT output,
                             const float* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale) noexcept {
    const __m128 v_scale = _mm_set1_ps(scale);
    int k = 0;
    for (; k <= K - 2; k += 2) {
        __m128 v0 = _mm_loadu_ps(input + 0 * ir_step);
        __m128 v1 = _mm_loadu_ps(input + 1 * ir_step);
        v0 = _mm_mul_ps(v0, v_scale);
        v1 = _mm_mul_ps(v1, v_scale);
        _mm_storeu_ps(output + 0 * 4, v0);
        _mm_storeu_ps(output + 1 * 4, v1);
        output += 2 * 4;
        input  += 2 * ir_step;
    }
    for (; k < K; ++k) {
        __m128 v0 = _mm_loadu_ps(input);
        v0 = _mm_mul_ps(v0, v_scale);
        _mm_storeu_ps(output, v0);
        output += 4;
        input  += ir_step;
    }
}

inline void pack_copy_n6_f32(float* NNOPS_RESTRICT output,
                             const float* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale) noexcept {
    const __m128 v_scale = _mm_set1_ps(scale);
    int k = 0;
    for (; k <= K - 2; k += 2) {
        __m128 v0 = _mm_loadu_ps(input + 0 * ir_step);
        __m128 v1 = _mm_loadu_ps(input + 1 * ir_step);
        v0 = _mm_mul_ps(v0, v_scale);
        v1 = _mm_mul_ps(v1, v_scale);
        _mm_storeu_ps(output, v0);
        output[4] = input[0 * ir_step + 4] * scale;
        output[5] = input[0 * ir_step + 5] * scale;
        _mm_storeu_ps(output + 6, v1);
        output[10] = input[1 * ir_step + 4] * scale;
        output[11] = input[1 * ir_step + 5] * scale;
        output += 2 * 6;
        input  += 2 * ir_step;
    }
    for (; k < K; ++k) {
        for (int i = 0; i < 6; ++i) {
            *output++ = input[i] * scale;
        }
        input += ir_step;
    }
}

inline void pack_copy_n8_f32(float* NNOPS_RESTRICT output,
                             const float* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale) noexcept {
    const __m256 v_scale = _mm256_set1_ps(scale);
    int k = 0;
    for (; k <= K - 2; k += 2) {
        __m256 v0 = _mm256_loadu_ps(input + 0 * ir_step);
        __m256 v1 = _mm256_loadu_ps(input + 1 * ir_step);
        v0 = _mm256_mul_ps(v0, v_scale);
        v1 = _mm256_mul_ps(v1, v_scale);
        _mm256_storeu_ps(output + 0 * 8, v0);
        _mm256_storeu_ps(output + 1 * 8, v1);
        output += 2 * 8;
        input  += 2 * ir_step;
    }
    for (; k < K; ++k) {
        __m256 v0 = _mm256_loadu_ps(input);
        v0 = _mm256_mul_ps(v0, v_scale);
        _mm256_storeu_ps(output, v0);
        output += 8;
        input  += ir_step;
    }
}

inline void pack_copy_n12_f32(float* NNOPS_RESTRICT output,
                              const float* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale) noexcept {
    const __m128 v_scale = _mm_set1_ps(scale);
    for (int k = 0; k < K; ++k) {
        __m128 v0 = _mm_loadu_ps(input + 0 * 4);
        __m128 v1 = _mm_loadu_ps(input + 1 * 4);
        __m128 v2 = _mm_loadu_ps(input + 2 * 4);

        v0 = _mm_mul_ps(v0, v_scale);
        v1 = _mm_mul_ps(v1, v_scale);
        v2 = _mm_mul_ps(v2, v_scale);

        _mm_storeu_ps(output + 0 * 4, v0);
        _mm_storeu_ps(output + 1 * 4, v1);
        _mm_storeu_ps(output + 2 * 4, v2);

        output += 12;
        input  += ir_step;
    }
}

inline void pack_copy_n16_f32(float* NNOPS_RESTRICT output,
                              const float* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale) noexcept {
    const __m256 v_scale = _mm256_set1_ps(scale);
    for (int k = 0; k < K; ++k) {
        __m256 v0 = _mm256_loadu_ps(input + 0 * 8);
        __m256 v1 = _mm256_loadu_ps(input + 1 * 8);

        v0 = _mm256_mul_ps(v0, v_scale);
        v1 = _mm256_mul_ps(v1, v_scale);

        _mm256_storeu_ps(output + 0 * 8, v0);
        _mm256_storeu_ps(output + 1 * 8, v1);

        output += 16;
        input  += ir_step;
    }
}

/// @brief f32 LHS micro-panel heights, shared by both routes.
/// The 6-row direct kernels already match the pack side, and AVX2's 16 ymm
/// registers leave no room for a taller tile anyway.
inline constexpr int mr_f32[3] = {6, 4, 1};

/// @brief f32 RHS micro-panel widths.
inline constexpr int nr_f32[3] = {16, 8, 1};

}  // namespace nnops::backend::cpu::x86_64
