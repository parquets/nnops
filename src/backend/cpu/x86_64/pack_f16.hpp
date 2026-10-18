#pragma once
/// @file pack_f16.hpp
/// @brief x86_64 F16C pack kernels for GEMM LHS/RHS (float16).
///
/// Data is stored as uint16_t (half) in memory. F16C intrinsics
/// (_mm256_cvtph_ps / _mm256_cvtps_ph) convert on the fly between the
/// fp16 storage format and fp32 accumulators.
///
/// Every kernel takes the live source width, defaulting to its own panel
/// width N: `mr` for the transposing (LHS) kernels, `nr` for the row-major
/// (RHS) copies. A partial panel needs no separate tail kernel — rows past
/// `mr` alias the last live row, the same broadcast the MMA kernels apply to
/// a short A panel, and columns past `nr` are zero-filled so a short B panel
/// costs the runtime-width MMA kernel nothing.
///
/// Reference: nn_compute/src/cpu/kernel/pack/x86_64/pack_f16.hpp

#include <immintrin.h>
#include "nnops/detail/half.hpp"
#include "backend/cpu/common/common_utils.hpp"
#include "transpose.hpp"

namespace nnops::backend::cpu::x86_64 {

using nnops::backend::cpu::half;
using nnops::backend::cpu::half_to_float;
using nnops::backend::cpu::float_to_half;

// The vector loops convert to f32 and run the SAME transpose_Nx8_f32 the
// f32 packs use, then round back to f16 on store; no extra conversions,
// since the scale multiply already round-trips through f32. This matters
// because the pack stores its registers linearly and the mma kernels read
// the panel as [K][n] (n contiguous per k): only the f32 transposes
// produce that order. A 16-bit-lane butterfly cannot express it (mr=6
// panels interleave two k's per register).

inline void pack_trans_n1_f16(half* NNOPS_RESTRICT output,
                              const half* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int mr = 1) noexcept {
    for (int k = 0; k < K; ++k) {
        *output++ = float_to_half(half_to_float(input[0]) * scale);
        input += 1;  // consecutive k within the M×K source row
    }
}

inline void pack_trans_n4_f16(half* NNOPS_RESTRICT output,
                              const half* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int mr = 4) noexcept {
    // Rows past `mr` alias the last live row — the same convention the
    // MMA kernels use — so a partial-M panel needs no padding.
    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;
    const half* NNOPS_RESTRICT p0 = input + 0 * ir_step;
    const half* NNOPS_RESTRICT p1 = input + r1 * ir_step;
    const half* NNOPS_RESTRICT p2 = input + r2 * ir_step;
    const half* NNOPS_RESTRICT p3 = input + r3 * ir_step;

    const __m256 v_scale = _mm256_set1_ps(scale);
    int k = 0;
    for (; k <= K - 8; k += 8) {
        __m256 v0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p0)));
        __m256 v1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p1)));
        __m256 v2 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p2)));
        __m256 v3 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p3)));
        p0 += 8; p1 += 8; p2 += 8; p3 += 8;

        transpose_4x8_f32(v0, v1, v2, v3);

        v0 = _mm256_mul_ps(v0, v_scale);
        v1 = _mm256_mul_ps(v1, v_scale);
        v2 = _mm256_mul_ps(v2, v_scale);
        v3 = _mm256_mul_ps(v3, v_scale);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 0 * 8),
                         _mm256_cvtps_ph(v0, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 1 * 8),
                         _mm256_cvtps_ph(v1, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 2 * 8),
                         _mm256_cvtps_ph(v2, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 3 * 8),
                         _mm256_cvtps_ph(v3, _MM_FROUND_TO_NEAREST_INT));

        output += 4 * 8;
    }
    // scalar tail (p0..p3 already point at k = 8*nblocks in each row)
    for (; k < K; ++k) {
        *output++ = float_to_half(half_to_float(*p0++) * scale);
        *output++ = float_to_half(half_to_float(*p1++) * scale);
        *output++ = float_to_half(half_to_float(*p2++) * scale);
        *output++ = float_to_half(half_to_float(*p3++) * scale);
    }
}

inline void pack_trans_n6_f16(half* NNOPS_RESTRICT output,
                              const half* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int mr = 6) noexcept {
    // Rows past `mr` alias the last live row — the same convention the
    // MMA kernels use — so a partial-M panel needs no padding.
    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;
    const int r4 = (mr > 4) ? 4 : r3;
    const int r5 = (mr > 5) ? 5 : r4;
    const half* NNOPS_RESTRICT p0 = input + 0 * ir_step;
    const half* NNOPS_RESTRICT p1 = input + r1 * ir_step;
    const half* NNOPS_RESTRICT p2 = input + r2 * ir_step;
    const half* NNOPS_RESTRICT p3 = input + r3 * ir_step;
    const half* NNOPS_RESTRICT p4 = input + r4 * ir_step;
    const half* NNOPS_RESTRICT p5 = input + r5 * ir_step;

    const __m256 v_scale = _mm256_set1_ps(scale);
    int k = 0;
    for (; k <= K - 8; k += 8) {
        __m256 v0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p0)));
        __m256 v1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p1)));
        __m256 v2 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p2)));
        __m256 v3 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p3)));
        __m256 v4 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p4)));
        __m256 v5 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p5)));
        p0 += 8; p1 += 8; p2 += 8; p3 += 8; p4 += 8; p5 += 8;

        transpose_6x8_f32(v0, v1, v2, v3, v4, v5);

        v0 = _mm256_mul_ps(v0, v_scale);
        v1 = _mm256_mul_ps(v1, v_scale);
        v2 = _mm256_mul_ps(v2, v_scale);
        v3 = _mm256_mul_ps(v3, v_scale);
        v4 = _mm256_mul_ps(v4, v_scale);
        v5 = _mm256_mul_ps(v5, v_scale);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 0 * 8),
                         _mm256_cvtps_ph(v0, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 1 * 8),
                         _mm256_cvtps_ph(v1, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 2 * 8),
                         _mm256_cvtps_ph(v2, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 3 * 8),
                         _mm256_cvtps_ph(v3, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 4 * 8),
                         _mm256_cvtps_ph(v4, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 5 * 8),
                         _mm256_cvtps_ph(v5, _MM_FROUND_TO_NEAREST_INT));

        output += 6 * 8;
    }
    for (; k < K; ++k) {
        *output++ = float_to_half(half_to_float(*p0++) * scale);
        *output++ = float_to_half(half_to_float(*p1++) * scale);
        *output++ = float_to_half(half_to_float(*p2++) * scale);
        *output++ = float_to_half(half_to_float(*p3++) * scale);
        *output++ = float_to_half(half_to_float(*p4++) * scale);
        *output++ = float_to_half(half_to_float(*p5++) * scale);
    }
}

inline void pack_trans_n8_f16(half* NNOPS_RESTRICT output,
                              const half* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int mr = 8) noexcept {
    // Rows past `mr` alias the last live row — the same convention the
    // MMA kernels use — so a partial-M panel needs no padding.
    const int last = (mr > 1) ? mr - 1 : 0;
    const half* p[8];
    for (int i = 0; i < 8; ++i) {
        p[i] = input + ((i < mr) ? i : last) * ir_step;
    }

    const __m256 v_scale = _mm256_set1_ps(scale);
    int k = 0;
    for (; k <= K - 8; k += 8) {
        __m256 v[8];
        for (int i = 0; i < 8; ++i) {
            v[i] = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p[i])));
            p[i] += 8;
        }

        transpose_8x8_f32(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);

        for (int i = 0; i < 8; ++i) {
            v[i] = _mm256_mul_ps(v[i], v_scale);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output + i * 8),
                             _mm256_cvtps_ph(v[i], _MM_FROUND_TO_NEAREST_INT));
        }

        output += 8 * 8;
    }
    for (; k < K; ++k) {
        for (int i = 0; i < 8; ++i) {
            *output++ = float_to_half(half_to_float(*p[i]++) * scale);
        }
    }
}

/// @brief 16-wide transposing panel. Rows past `mr` alias the last live row,
/// so a partial-M panel needs no padding and the kernel never reads past the
/// tile — only the aliased pointer list differs from the full-panel case.
inline void pack_trans_n16_f16(half* NNOPS_RESTRICT output,
                               const half* NNOPS_RESTRICT input,
                               int ir_step, int K, float scale, int mr = 16) noexcept {
    // Rows past `mr` alias the last live row — the same convention the
    // MMA kernels use — so a partial-M panel needs no padding.
    const int last = (mr > 1) ? mr - 1 : 0;
    const half* p[16];
    for (int i = 0; i < 16; ++i) {
        p[i] = input + ((i < mr) ? i : last) * ir_step;
    }

    const __m256 v_scale = _mm256_set1_ps(scale);
    int k = 0;
    for (; k <= K - 8; k += 8) {
        __m256 v[16];
        for (int i = 0; i < 16; ++i) {
            v[i] = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p[i])));
            p[i] += 8;
        }

        transpose_16x8_f32(
            v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
            v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);

        for (int i = 0; i < 16; ++i) {
            v[i] = _mm256_mul_ps(v[i], v_scale);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output + i * 8),
                             _mm256_cvtps_ph(v[i], _MM_FROUND_TO_NEAREST_INT));
        }

        output += 16 * 8;
    }
    for (; k < K; ++k) {
        for (int i = 0; i < 16; ++i) {
            *output++ = float_to_half(half_to_float(*p[i]++) * scale);
        }
    }
}

inline void pack_trans_n24_f16(half* NNOPS_RESTRICT output,
                               const half* NNOPS_RESTRICT input,
                               int ir_step, int K, float scale, int mr = 24) noexcept {
    // Pure scalar path for wide N=24 panel — rarely called.
    const int last = (mr > 1) ? mr - 1 : 0;
    int row[24];
    for (int i = 0; i < 24; ++i) {
        row[i] = ((i < mr) ? i : last) * ir_step;
    }
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < 24; ++i) {
            *output++ = float_to_half(half_to_float(input[row[i]]) * scale);
        }
        input += 1;
    }
}

inline void pack_copy_n1_f16(half* NNOPS_RESTRICT output,
                             const half* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale, int nr = 1) noexcept {
    for (int k = 0; k < K; ++k) {
        *output++ = (nr > 0) ? float_to_half(half_to_float(input[0]) * scale) : half{};
        input += ir_step;
    }
}

/// @brief 4-wide row-major panel. One k loop covers every width: the load is
/// masked at 32-bit (half-pair) granularity, and the single half that can sit
/// in a masked-off pair is spliced back in, so a short panel never reads past
/// its live columns. Even at full width this beats the old scalar body, where
/// four software half<->float conversions ran per k.
inline void pack_copy_n4_f16(half* NNOPS_RESTRICT output,
                             const half* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale, int nr = 4) noexcept {
    const __m128i v_idx32 = _mm_set_epi32(3, 2, 1, 0);
    const __m128i v_pair = _mm_cmpgt_epi32(_mm_set1_epi32(nr >> 1), v_idx32);
    const __m128i v_idx16 = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
    const __m128i v_last = _mm_cmpeq_epi16(v_idx16, _mm_set1_epi16(static_cast<short>(nr - 1)));
    const __m256 v_scale = _mm256_set1_ps(scale);
    const half* fix = input + ((nr > 0) ? nr - 1 : 0);
    for (int k = 0; k < K; ++k) {
        __m128i v = _mm_maskload_epi32(reinterpret_cast<const int*>(input), v_pair);
        v = _mm_blendv_epi8(v, _mm_set1_epi16(static_cast<short>(half_to_bits(*fix))), v_last);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(output),
                         _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(v), v_scale),
                                         _MM_FROUND_TO_NEAREST_INT));
        output += 4;
        input  += ir_step;
        fix    += ir_step;
    }
}

/// @brief 6-wide row-major panel (see n4). One masked 8-half load covers all
/// six live lanes; the store is a 4-half word plus the trailing pair.
inline void pack_copy_n6_f16(half* NNOPS_RESTRICT output,
                             const half* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale, int nr = 6) noexcept {
    const __m128i v_idx32 = _mm_set_epi32(3, 2, 1, 0);
    const __m128i v_pair = _mm_cmpgt_epi32(_mm_set1_epi32(nr >> 1), v_idx32);
    const __m128i v_idx16 = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
    const __m128i v_last = _mm_cmpeq_epi16(v_idx16, _mm_set1_epi16(static_cast<short>(nr - 1)));
    const __m256 v_scale = _mm256_set1_ps(scale);
    const half* fix = input + ((nr > 0) ? nr - 1 : 0);
    for (int k = 0; k < K; ++k) {
        __m128i v = _mm_maskload_epi32(reinterpret_cast<const int*>(input), v_pair);
        v = _mm_blendv_epi8(v, _mm_set1_epi16(static_cast<short>(half_to_bits(*fix))), v_last);
        const __m128i h = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(v), v_scale),
                                          _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(output), h);
        _mm_storeu_si32(output + 4, _mm_srli_si128(h, 8));
        output += 6;
        input  += ir_step;
        fix    += ir_step;
    }
}

inline void pack_copy_n8_f16(half* NNOPS_RESTRICT output,
                             const half* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale, int nr = 8) noexcept {
    const __m128i v_idx32 = _mm_set_epi32(3, 2, 1, 0);
    const __m128i v_pair = _mm_cmpgt_epi32(_mm_set1_epi32(nr >> 1), v_idx32);
    const __m128i v_idx16 = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
    const __m128i v_last = _mm_cmpeq_epi16(v_idx16, _mm_set1_epi16(static_cast<short>(nr - 1)));
    const __m256 v_scale = _mm256_set1_ps(scale);
    const half* fix = input + ((nr > 0) ? nr - 1 : 0);
    int k = 0;
    for (; k <= K - 2; k += 2) {
        if (nr >= 8) {
            const __m256 a = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(input)));
            const __m256 b = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + ir_step)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 0),
                             _mm256_cvtps_ph(_mm256_mul_ps(a, v_scale),
                                             _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 8),
                             _mm256_cvtps_ph(_mm256_mul_ps(b, v_scale),
                                             _MM_FROUND_TO_NEAREST_INT));
        } else {
            __m128i a = _mm_maskload_epi32(reinterpret_cast<const int*>(input), v_pair);
            a = _mm_blendv_epi8(a,
                                _mm_set1_epi16(static_cast<short>(half_to_bits(*fix))),
                                v_last);
            __m128i b = _mm_maskload_epi32(
                reinterpret_cast<const int*>(input + ir_step), v_pair);
            b = _mm_blendv_epi8(b,
                                _mm_set1_epi16(static_cast<short>(half_to_bits(fix[ir_step]))),
                                v_last);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 0),
                             _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(a), v_scale),
                                             _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 8),
                             _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(b), v_scale),
                                             _MM_FROUND_TO_NEAREST_INT));
        }
        output += 2 * 8;
        input  += 2 * ir_step;
        fix    += 2 * ir_step;
    }
    for (; k < K; ++k) {
        if (nr >= 8) {
            const __m256 v = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(input)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output),
                             _mm256_cvtps_ph(_mm256_mul_ps(v, v_scale),
                                             _MM_FROUND_TO_NEAREST_INT));
        } else {
            __m128i v = _mm_maskload_epi32(reinterpret_cast<const int*>(input), v_pair);
            v = _mm_blendv_epi8(v,
                                _mm_set1_epi16(static_cast<short>(half_to_bits(*fix))),
                                v_last);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output),
                             _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(v), v_scale),
                                             _MM_FROUND_TO_NEAREST_INT));
        }
        output += 8;
        input  += ir_step;
        fix    += ir_step;
    }
}

/// @brief 16-wide row-major panel. Columns past `nr` are zero-filled so the
/// runtime-width MMA kernels can load all 16 lanes of every k; a short panel
/// reads through one pair-masked 32-byte load, which never touches a dead
/// column.
inline void pack_copy_n16_f16(half* NNOPS_RESTRICT output,
                              const half* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int nr = 16) noexcept {
    const __m256i v_idx32 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i v_pair = _mm256_cmpgt_epi32(_mm256_set1_epi32(nr >> 1), v_idx32);
    const __m256i v_idx16 = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7,
                                              8, 9, 10, 11, 12, 13, 14, 15);
    const __m256i v_last = _mm256_cmpeq_epi16(v_idx16,
                                              _mm256_set1_epi16(static_cast<short>(nr - 1)));
    const __m256 v_scale = _mm256_set1_ps(scale);
    const half* fix = input + ((nr > 0) ? nr - 1 : 0);
    for (int k = 0; k < K; ++k) {
        if (nr >= 16) {
            const __m256 a = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + 0)));
            const __m256 b = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + 8)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 0),
                             _mm256_cvtps_ph(_mm256_mul_ps(a, v_scale),
                                             _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 8),
                             _mm256_cvtps_ph(_mm256_mul_ps(b, v_scale),
                                             _MM_FROUND_TO_NEAREST_INT));
        } else {
            __m256i v = _mm256_maskload_epi32(reinterpret_cast<const int*>(input), v_pair);
            v = _mm256_blendv_epi8(v,
                                   _mm256_set1_epi16(static_cast<short>(half_to_bits(*fix))),
                                   v_last);
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(output + 0),
                _mm256_cvtps_ph(
                    _mm256_mul_ps(_mm256_cvtph_ps(_mm256_castsi256_si128(v)), v_scale),
                    _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(output + 8),
                _mm256_cvtps_ph(
                    _mm256_mul_ps(_mm256_cvtph_ps(_mm256_extracti128_si256(v, 1)), v_scale),
                    _MM_FROUND_TO_NEAREST_INT));
        }
        output += 16;
        input  += ir_step;
        fix    += ir_step;
    }
}

inline void pack_copy_n24_f16(half* NNOPS_RESTRICT output,
                              const half* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int nr = 24) noexcept {
    const __m128i v_idx32 = _mm_set_epi32(3, 2, 1, 0);
    const __m128i v_pair0 = _mm_cmpgt_epi32(_mm_set1_epi32(nr >> 1), v_idx32);
    const __m128i v_pair1 = _mm_cmpgt_epi32(_mm_set1_epi32((nr >> 1) - 4), v_idx32);
    const __m128i v_pair2 = _mm_cmpgt_epi32(_mm_set1_epi32((nr >> 1) - 8), v_idx32);
    const __m128i v_idx16 = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
    const __m128i v_lane = _mm_set1_epi16(static_cast<short>(nr - 1));
    const __m128i v_last0 = _mm_cmpeq_epi16(v_idx16, v_lane);
    const __m128i v_last1 = _mm_cmpeq_epi16(_mm_add_epi16(v_idx16, _mm_set1_epi16(8)), v_lane);
    const __m128i v_last2 = _mm_cmpeq_epi16(_mm_add_epi16(v_idx16, _mm_set1_epi16(16)), v_lane);
    const __m256 v_scale = _mm256_set1_ps(scale);
    const half* fix = input + ((nr > 0) ? nr - 1 : 0);
    for (int k = 0; k < K; ++k) {
        if (nr >= 24) {
            const __m256 a = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + 0)));
            const __m256 b = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + 8)));
            const __m256 c = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + 16)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 0),
                             _mm256_cvtps_ph(_mm256_mul_ps(a, v_scale),
                                             _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 8),
                             _mm256_cvtps_ph(_mm256_mul_ps(b, v_scale),
                                             _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 16),
                             _mm256_cvtps_ph(_mm256_mul_ps(c, v_scale),
                                             _MM_FROUND_TO_NEAREST_INT));
        } else {
            const __m128i b = _mm_set1_epi16(static_cast<short>(half_to_bits(*fix)));
            const __m128i v0 = _mm_blendv_epi8(
                _mm_maskload_epi32(reinterpret_cast<const int*>(input + 0), v_pair0), b, v_last0);
            const __m128i v1 = _mm_blendv_epi8(
                _mm_maskload_epi32(reinterpret_cast<const int*>(input + 8), v_pair1), b, v_last1);
            const __m128i v2 = _mm_blendv_epi8(
                _mm_maskload_epi32(reinterpret_cast<const int*>(input + 16), v_pair2), b, v_last2);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 0),
                             _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(v0), v_scale),
                                             _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 8),
                             _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(v1), v_scale),
                                             _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 16),
                             _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(v2), v_scale),
                                             _MM_FROUND_TO_NEAREST_INT));
        }
        output += 24;
        input  += ir_step;
        fix    += ir_step;
    }
}

/// @brief f16 LHS micro-panel heights.
inline constexpr int mr_f16[3] = {6, 4, 1};

/// @brief f16 RHS micro-panel widths.
inline constexpr int nr_f16[3] = {16, 8, 1};

}  // namespace nnops::backend::cpu::x86_64
