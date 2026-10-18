#pragma once
/// @file pack_f32.hpp
/// @brief x86_64 SSE/AVX float32 pack kernels for GEMM LHS/RHS.
///
/// The AVX path operates on 8-element __m256 vectors, giving a naturally
/// unrolled inner loop with K step of 8.
///
/// Every kernel takes the live source width, defaulting to its own panel
/// width N: `mr` for the transposing (LHS) kernels, `nr` for the row-major
/// (RHS) copies. A partial panel needs no separate tail kernel — rows past
/// `mr` alias the last live row, the same broadcast the MMA kernels apply to
/// a short A panel, and columns past `nr` are zero-filled so a short B panel
/// costs the runtime-width MMA kernel nothing.
///
/// Reference: nn_compute/src/cpu/kernel/pack/x86_64/pack_f32.hpp

#include <immintrin.h>
#include "backend/cpu/common/common_utils.hpp"
#include "transpose.hpp"

namespace nnops::backend::cpu::x86_64 {

inline void pack_trans_n1_f32(float* NNOPS_RESTRICT output,
                              const float* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int mr = 1) noexcept {
    for (int k = 0; k < K; ++k) {
        *output++ = input[0] * scale;
        input += 1;  // consecutive k within the M×K source row
    }
}

inline void pack_trans_n4_f32(float* NNOPS_RESTRICT output,
                              const float* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int mr = 4) noexcept {
    // Rows past `mr` alias the last live row — the same convention the
    // MMA kernels use — so a partial-M panel needs no padding.
    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;
    const float* NNOPS_RESTRICT p0 = input + 0 * ir_step;
    const float* NNOPS_RESTRICT p1 = input + r1 * ir_step;
    const float* NNOPS_RESTRICT p2 = input + r2 * ir_step;
    const float* NNOPS_RESTRICT p3 = input + r3 * ir_step;

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
                              int ir_step, int K, float scale, int mr = 6) noexcept {
    // Rows past `mr` alias the last live row — the same convention the
    // MMA kernels use — so a partial-M panel needs no padding.
    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;
    const int r4 = (mr > 4) ? 4 : r3;
    const int r5 = (mr > 5) ? 5 : r4;
    const float* NNOPS_RESTRICT p0 = input + 0 * ir_step;
    const float* NNOPS_RESTRICT p1 = input + r1 * ir_step;
    const float* NNOPS_RESTRICT p2 = input + r2 * ir_step;
    const float* NNOPS_RESTRICT p3 = input + r3 * ir_step;
    const float* NNOPS_RESTRICT p4 = input + r4 * ir_step;
    const float* NNOPS_RESTRICT p5 = input + r5 * ir_step;

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
                              int ir_step, int K, float scale, int mr = 8) noexcept {
    // Rows past `mr` alias the last live row — the same convention the
    // MMA kernels use — so a partial-M panel needs no padding.
    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;
    const int r4 = (mr > 4) ? 4 : r3;
    const int r5 = (mr > 5) ? 5 : r4;
    const int r6 = (mr > 6) ? 6 : r5;
    const int r7 = (mr > 7) ? 7 : r6;
    const float* NNOPS_RESTRICT p0 = input + 0 * ir_step;
    const float* NNOPS_RESTRICT p1 = input + r1 * ir_step;
    const float* NNOPS_RESTRICT p2 = input + r2 * ir_step;
    const float* NNOPS_RESTRICT p3 = input + r3 * ir_step;
    const float* NNOPS_RESTRICT p4 = input + r4 * ir_step;
    const float* NNOPS_RESTRICT p5 = input + r5 * ir_step;
    const float* NNOPS_RESTRICT p6 = input + r6 * ir_step;
    const float* NNOPS_RESTRICT p7 = input + r7 * ir_step;

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

/// @brief 16-wide transposing panel. Rows past `mr` alias the last live row,
/// so a partial-M panel needs no padding and the kernel never reads past the
/// tile — only the aliased pointer list differs from the full-panel case.
inline void pack_trans_n16_f32(float* NNOPS_RESTRICT output,
                               const float* NNOPS_RESTRICT input,
                               int ir_step, int K, float scale, int mr = 16) noexcept {
    // Rows past `mr` alias the last live row — the same convention the
    // MMA kernels use — so a partial-M panel needs no padding.
    const int last = (mr > 1) ? mr - 1 : 0;
    const float* NNOPS_RESTRICT p[16];
    for (int i = 0; i < 16; ++i) {
        p[i] = input + ((i < mr) ? i : last) * ir_step;
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
                             int ir_step, int K, float scale, int nr = 1) noexcept {
    for (int k = 0; k < K; ++k) {
        *output++ = (nr > 0) ? input[0] * scale : 0.0f;
        input += ir_step;
    }
}

/// @brief 4-wide row-major panel. One k loop serves every width: a full panel
/// takes the plain load, a short one a lane-masked load, which reads only the
/// live columns and zero-fills the rest on the way in.
inline void pack_copy_n4_f32(float* NNOPS_RESTRICT output,
                             const float* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale, int nr = 4) noexcept {
    const __m128 v_scale = _mm_set1_ps(scale);
    const __m128i v_idx = _mm_set_epi32(3, 2, 1, 0);
    const __m128i v_read = _mm_cmpgt_epi32(_mm_set1_epi32(nr), v_idx);
    int k = 0;
    for (; k <= K - 2; k += 2) {
        const __m128 a0 = (nr >= 4) ? _mm_loadu_ps(input) : _mm_maskload_ps(input, v_read);
        const __m128 a1 = (nr >= 4) ? _mm_loadu_ps(input + ir_step)
                                    : _mm_maskload_ps(input + ir_step, v_read);
        _mm_storeu_ps(output + 0, _mm_mul_ps(a0, v_scale));
        _mm_storeu_ps(output + 4, _mm_mul_ps(a1, v_scale));
        output += 2 * 4;
        input  += 2 * ir_step;
    }
    for (; k < K; ++k) {
        const __m128 v = (nr >= 4) ? _mm_loadu_ps(input) : _mm_maskload_ps(input, v_read);
        _mm_storeu_ps(output, _mm_mul_ps(v, v_scale));
        output += 4;
        input  += ir_step;
    }
}

/// @brief 6-wide row-major panel. Lanes 4 and 5 travel as a second, 2-wide
/// masked load, so the same body covers every width down to zero.
inline void pack_copy_n6_f32(float* NNOPS_RESTRICT output,
                             const float* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale, int nr = 6) noexcept {
    const __m128 v_scale = _mm_set1_ps(scale);
    const __m128i v_idx4 = _mm_set_epi32(3, 2, 1, 0);
    const __m128i v_idx2 = _mm_set_epi32(0x7fffffff, 0x7fffffff, 5, 4);
    const __m128i v_read4 = _mm_cmpgt_epi32(_mm_set1_epi32(nr), v_idx4);
    const __m128i v_read2 = _mm_cmpgt_epi32(_mm_set1_epi32(nr), v_idx2);
    int k = 0;
    for (; k <= K - 2; k += 2) {
        const __m128 a0 = (nr >= 6) ? _mm_loadu_ps(input) : _mm_maskload_ps(input, v_read4);
        const __m128 a1 = (nr >= 6)
            ? _mm_castsi128_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(input + 4)))
            : _mm_maskload_ps(input + 4, v_read2);
        const __m128 b0 = (nr >= 6) ? _mm_loadu_ps(input + ir_step)
                                    : _mm_maskload_ps(input + ir_step, v_read4);
        const __m128 b1 = (nr >= 6)
            ? _mm_castsi128_ps(
                  _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input + ir_step + 4)))
            : _mm_maskload_ps(input + ir_step + 4, v_read2);
        _mm_storeu_ps(output + 0, _mm_mul_ps(a0, v_scale));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(output + 4),
                         _mm_castps_si128(_mm_mul_ps(a1, v_scale)));
        _mm_storeu_ps(output + 6, _mm_mul_ps(b0, v_scale));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(output + 10),
                         _mm_castps_si128(_mm_mul_ps(b1, v_scale)));
        output += 2 * 6;
        input  += 2 * ir_step;
    }
    for (; k < K; ++k) {
        const __m128 v0 = (nr >= 6) ? _mm_loadu_ps(input) : _mm_maskload_ps(input, v_read4);
        const __m128 v1 = (nr >= 6)
            ? _mm_castsi128_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(input + 4)))
            : _mm_maskload_ps(input + 4, v_read2);
        _mm_storeu_ps(output, _mm_mul_ps(v0, v_scale));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(output + 4),
                         _mm_castps_si128(_mm_mul_ps(v1, v_scale)));
        output += 6;
        input  += ir_step;
    }
}

inline void pack_copy_n8_f32(float* NNOPS_RESTRICT output,
                             const float* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale, int nr = 8) noexcept {
    const __m256 v_scale = _mm256_set1_ps(scale);
    const __m256i v_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i v_read = _mm256_cmpgt_epi32(_mm256_set1_epi32(nr), v_idx);
    int k = 0;
    for (; k <= K - 2; k += 2) {
        const __m256 a0 = (nr >= 8) ? _mm256_loadu_ps(input)
                                    : _mm256_maskload_ps(input, v_read);
        const __m256 a1 = (nr >= 8) ? _mm256_loadu_ps(input + ir_step)
                                    : _mm256_maskload_ps(input + ir_step, v_read);
        _mm256_storeu_ps(output + 0, _mm256_mul_ps(a0, v_scale));
        _mm256_storeu_ps(output + 8, _mm256_mul_ps(a1, v_scale));
        output += 2 * 8;
        input  += 2 * ir_step;
    }
    for (; k < K; ++k) {
        const __m256 v = (nr >= 8) ? _mm256_loadu_ps(input)
                                   : _mm256_maskload_ps(input, v_read);
        _mm256_storeu_ps(output, _mm256_mul_ps(v, v_scale));
        output += 8;
        input  += ir_step;
    }
}

inline void pack_copy_n12_f32(float* NNOPS_RESTRICT output,
                              const float* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int nr = 12) noexcept {
    const __m128 v_scale = _mm_set1_ps(scale);
    const __m128i v_idx = _mm_set_epi32(3, 2, 1, 0);
    const __m128i v_read0 = _mm_cmpgt_epi32(_mm_set1_epi32(nr), v_idx);
    const __m128i v_read1 = _mm_cmpgt_epi32(_mm_set1_epi32(nr - 4), v_idx);
    const __m128i v_read2 = _mm_cmpgt_epi32(_mm_set1_epi32(nr - 8), v_idx);
    for (int k = 0; k < K; ++k) {
        __m128 v0, v1, v2;
        if (nr >= 12) {
            v0 = _mm_loadu_ps(input + 0);
            v1 = _mm_loadu_ps(input + 4);
            v2 = _mm_loadu_ps(input + 8);
        } else {
            v0 = _mm_maskload_ps(input + 0, v_read0);
            v1 = _mm_maskload_ps(input + 4, v_read1);
            v2 = _mm_maskload_ps(input + 8, v_read2);
        }
        _mm_storeu_ps(output + 0, _mm_mul_ps(v0, v_scale));
        _mm_storeu_ps(output + 4, _mm_mul_ps(v1, v_scale));
        _mm_storeu_ps(output + 8, _mm_mul_ps(v2, v_scale));
        output += 12;
        input  += ir_step;
    }
}

/// @brief 16-wide row-major panel. Columns past `nr` are zero-filled so the
/// runtime-width MMA kernels can load all 16 lanes of every k; a short panel
/// reads through a lane mask, which touches only the live columns — the upper
/// half of a panel narrower than nine is all-masked and never reaches memory.
inline void pack_copy_n16_f32(float* NNOPS_RESTRICT output,
                              const float* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int nr = 16) noexcept {
    const __m256 v_scale = _mm256_set1_ps(scale);
    const __m256i v_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i v_read_lo = _mm256_cmpgt_epi32(_mm256_set1_epi32(nr), v_idx);
    const __m256i v_read_hi = _mm256_cmpgt_epi32(_mm256_set1_epi32(nr - 8), v_idx);
    for (int k = 0; k < K; ++k) {
        __m256 v0, v1;
        if (nr >= 16) {
            v0 = _mm256_loadu_ps(input + 0);
            v1 = _mm256_loadu_ps(input + 8);
        } else {
            v0 = _mm256_maskload_ps(input + 0, v_read_lo);
            v1 = _mm256_maskload_ps(input + 8, v_read_hi);
        }
        _mm256_storeu_ps(output + 0, _mm256_mul_ps(v0, v_scale));
        _mm256_storeu_ps(output + 8, _mm256_mul_ps(v1, v_scale));
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
