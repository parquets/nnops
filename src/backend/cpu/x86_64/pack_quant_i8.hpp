#pragma once
/// @file pack_quant_i8.hpp
/// @brief x86_64 AVX2+FMA fused pack + per-row quantize kernels for GEMM LHS/RHS.
///
/// These kernels fuse the f32→i8 quantization step into the GEMM packing step,
/// eliminating the intermediate quantized-but-unpacked buffer and halving the
/// memory traffic (one pass instead of quantize + pack).
///
/// Two transform families, mirroring pack_f32 / pack_dp4a_i8:
///   - pack_quant_trans_nN (LHS/A): reads N f32 rows (stride ir_step), quantizes
///     each row independently with its own scale/zero (broadcast across all K),
///     narrows to int8, then packs 4 consecutive K values from the SAME row into
///     each int32 to match the VNNI dp4a interleaved format.
///   - pack_quant_copy_nN (RHS/B): reads N-column f32 rows, quantizes with
///     K-shared scale/zero (broadcast across all K), groups each 4-k block into
///     contiguous 4-i8 runs per column.
///
/// Quantization: dst = clamp(round(src * inv_scale) + zero_point, qmin, qmax)
/// where inv_scale = 1.0f / scale, zero_point is int32_t.
/// Rounding is round-to-nearest-even.  zero_point is added in the integer
/// domain after rounding.
/// Templated on Q_U8 (false=s8/int8_t, true=u8/uint8_t).
///
/// Reference: nn_compute/src/cpu/kernel/pack/x86_64/pack_quant_i8.hpp

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <type_traits>

#include "backend/cpu/common/restrict.hpp"
#include "transpose.hpp"

namespace nnops::backend::cpu::x86_64 {

// =========================================================================
//  Per-lane quantization helpers
// =========================================================================

/// Quantize 8 floats (__m256) with per-lane scale/zero to 8 int32 (__m256i).
/// Used after 8×8 transpose where each lane comes from a different row.
inline __m256i quant_8f32_to_i32(__m256 d, __m256 inv_scale, __m256 zero) noexcept
{
    d = _mm256_add_ps(_mm256_mul_ps(d, inv_scale), zero);
    return _mm256_cvtps_epi32(d);
}

/// Quantize 4 floats (__m128) with per-lane scale/zero to 4 int32 (__m128i).
inline __m128i quant_4f32_to_i32(__m128 d, __m128 inv_scale, __m128 zero) noexcept
{
    d = _mm_add_ps(_mm_mul_ps(d, inv_scale), zero);
    return _mm_cvtps_epi32(d);
}

/// Narrow 2×__m256i (8 int32) to 8 int8 (lower 64 bits of __m128i).
/// The 256-bit packs produce per-lane ordering; a cross-lane permute
/// restores the natural [0..7] order.
inline __m128i narrow_8i32_to_i8(__m256i i0, __m256i i1) noexcept
{
    const __m256i i16 = _mm256_packs_epi32(i0, i1);
    // i16 = [i0_lo, i0_hi, i1_lo, i1_hi] per 128-bit lane
    // Cross-lane permute to restore [i0_lo, i0_hi, i1_lo, i1_hi]
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const __m256i ordered = _mm256_permutevar8x32_epi32(i16, perm);
    return _mm256_castsi256_si128(ordered);
}

/// Narrow 4×__m256i (32 int32) to 32 int8 (__m256i), natural order.
/// Uses the same cross-lane permute pattern as quant_block32_f32_i8.
template <bool Q_U8 = false>
inline __m256i narrow_32i32_to_i8(__m256i i0, __m256i i1, __m256i i2, __m256i i3) noexcept
{
    const __m256i d01_i16 = _mm256_packs_epi32(i0, i1);
    const __m256i d23_i16 = _mm256_packs_epi32(i2, i3);
    __m256i packed;
    if constexpr (Q_U8) {
        packed = _mm256_packus_epi16(d01_i16, d23_i16);
    } else {
        packed = _mm256_packs_epi16(d01_i16, d23_i16);
    }
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    return _mm256_permutevar8x32_epi32(packed, perm);
}

// =========================================================================
//  LHS pack + quantize  (f32 → i8, per-row scale/zero_point)
//
//  Each row is quantized independently with its own scale/zero_point
//  (broadcast across all K elements).  K step = 4 (SSE __m128) to keep
//  register pressure low — each row produces 4 int8 = 1 int32 per step.
//  Quantization: round(src * inv_scale) + zero_point.
//  4 consecutive K values from the SAME row are packed into each int32,
//  matching the VNNI dp4a interleaved format:
//    out_i32[0] = [q(r0_k0), q(r0_k1), q(r0_k2), q(r0_k3)]
//    out_i32[1] = [q(r1_k0), q(r1_k1), q(r1_k2), q(r1_k3)]
//    ...
// =========================================================================

template <bool Q_U8 = false>
inline void pack_quant_trans_n1_i8(void* NNOPS_RESTRICT output,
                                    const float* NNOPS_RESTRICT input,
                                    int ir_step, int K,
                                    const float* NNOPS_RESTRICT scale,
                                    const int32_t* NNOPS_RESTRICT zero) noexcept
{
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    int32_t* NNOPS_RESTRICT out_i32 = static_cast<int32_t*>(output);
    const float inv_scale = 1.0f / scale[0];
    const int32_t zp = (zero == nullptr) ? 0 : zero[0];
    const __m128i v_zp = _mm_set1_epi32(zp);

    int k = 0;
    for (; k <= K - 4; k += 4) {
        __m128 v = _mm_loadu_ps(input);
        __m128 v_s = _mm_set1_ps(inv_scale);
        __m128i qi = _mm_cvtps_epi32(_mm_mul_ps(v, v_s));
        qi = _mm_add_epi32(qi, v_zp);
        __m128i i16 = _mm_packs_epi32(qi, _mm_setzero_si128());
        __m128i i8;
        if constexpr (Q_U8) {
            i8 = _mm_packus_epi16(i16, _mm_setzero_si128());
        } else {
            i8 = _mm_packs_epi16(i16, _mm_setzero_si128());
        }
        out_i32[0] = _mm_cvtsi128_si32(i8);
        out_i32++;
        input += 4;
    }
    if (k < K) {
        int8_t* out_i8 = reinterpret_cast<int8_t*>(out_i32);
        for (int kk = 0; kk < 4 && k + kk < K; ++kk) {
            float q = input[kk * ir_step] * inv_scale;
            int32_t qi = static_cast<int32_t>(std::nearbyintf(q)) + zp;
            out_i8[kk] = static_cast<int8_t>(std::min(std::max(qi, qmin), qmax));
        }
        for (int kk = K - k; kk < 4; ++kk) {
            out_i8[kk] = 0;
        }
    }
}

template <bool Q_U8 = false>
inline void pack_quant_trans_n4_i8(void* NNOPS_RESTRICT output,
                                    const float* NNOPS_RESTRICT input,
                                    int ir_step, int K,
                                    const float* NNOPS_RESTRICT scale,
                                    const int32_t* NNOPS_RESTRICT zero) noexcept
{
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    int32_t* NNOPS_RESTRICT out_i32 = static_cast<int32_t*>(output);

    float inv_s[4];
    int32_t zp[4];
    for (int i = 0; i < 4; ++i) {
        inv_s[i] = 1.0f / scale[i];
        zp[i] = (zero == nullptr) ? 0 : zero[i];
    }

    int k = 0;
    for (; k <= K - 4; k += 4) {
        // Quantize 4 rows, each producing 4 int32 → 4 int8 = 1 int32
        __m128i qi[4];
        for (int i = 0; i < 4; ++i) {
            __m128 v = _mm_loadu_ps(input + i * ir_step);
            __m128 v_s = _mm_set1_ps(inv_s[i]);
            qi[i] = _mm_cvtps_epi32(_mm_mul_ps(v, v_s));
            qi[i] = _mm_add_epi32(qi[i], _mm_set1_epi32(zp[i]));
        }
        // Pack 4 rows: 4×4 int32 → 16 int8 → 4 int32
        __m128i i16_01 = _mm_packs_epi32(qi[0], qi[1]);
        __m128i i16_23 = _mm_packs_epi32(qi[2], qi[3]);
        __m128i i8;
        if constexpr (Q_U8) {
            i8 = _mm_packus_epi16(i16_01, i16_23);
        } else {
            i8 = _mm_packs_epi16(i16_01, i16_23);
        }
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out_i32), i8);
        out_i32 += 4;
        input += 4;
    }

    for (; k < K; ++k) {
        int8_t* out_i8 = reinterpret_cast<int8_t*>(out_i32);
        for (int i = 0; i < 4; ++i) {
            float qval = input[i * ir_step] * inv_s[i];
            int32_t qi = static_cast<int32_t>(std::nearbyintf(qval)) + zp[i];
            out_i8[i] = static_cast<int8_t>(std::min(std::max(qi, qmin), qmax));
        }
        out_i32++;
        input += 1;
    }
}

template <bool Q_U8 = false>
inline void pack_quant_trans_n6_i8(void* NNOPS_RESTRICT output,
                                    const float* NNOPS_RESTRICT input,
                                    int ir_step, int K,
                                    const float* NNOPS_RESTRICT scale,
                                    const int32_t* NNOPS_RESTRICT zero) noexcept
{
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    int32_t* NNOPS_RESTRICT out_i32 = static_cast<int32_t*>(output);

    float inv_s[6];
    int32_t zp[6];
    for (int i = 0; i < 6; ++i) {
        inv_s[i] = 1.0f / scale[i];
        zp[i] = (zero == nullptr) ? 0 : zero[i];
    }

    int k = 0;
    for (; k <= K - 4; k += 4) {
        // Quantize 6 rows
        __m128i qi[6];
        for (int i = 0; i < 6; ++i) {
            __m128 v = _mm_loadu_ps(input + i * ir_step);
            __m128 v_s = _mm_set1_ps(inv_s[i]);
            qi[i] = _mm_cvtps_epi32(_mm_mul_ps(v, v_s));
            qi[i] = _mm_add_epi32(qi[i], _mm_set1_epi32(zp[i]));
        }
        // Rows 0-3: pack to 4 int32
        __m128i i16_01 = _mm_packs_epi32(qi[0], qi[1]);
        __m128i i16_23 = _mm_packs_epi32(qi[2], qi[3]);
        __m128i i8_03;
        if constexpr (Q_U8) {
            i8_03 = _mm_packus_epi16(i16_01, i16_23);
        } else {
            i8_03 = _mm_packs_epi16(i16_01, i16_23);
        }
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out_i32), i8_03);
        // Rows 4-5: pack to 2 int32
        __m128i i16_45 = _mm_packs_epi32(qi[4], qi[5]);
        __m128i i8_45;
        if constexpr (Q_U8) {
            i8_45 = _mm_packus_epi16(i16_45, _mm_setzero_si128());
        } else {
            i8_45 = _mm_packs_epi16(i16_45, _mm_setzero_si128());
        }
        _mm_storel_epi64(reinterpret_cast<__m128i*>(out_i32 + 4), i8_45);
        out_i32 += 6;
        input += 4;
    }

    for (; k < K; ++k) {
        int8_t* out_i8 = reinterpret_cast<int8_t*>(out_i32);
        for (int i = 0; i < 6; ++i) {
            float qval = input[i * ir_step] * inv_s[i];
            int32_t qi = static_cast<int32_t>(std::nearbyintf(qval)) + zp[i];
            out_i8[i] = static_cast<int8_t>(std::min(std::max(qi, qmin), qmax));
        }
        out_i32++;
        input += 1;
    }
}

template <bool Q_U8 = false>
inline void pack_quant_trans_n8_i8(void* NNOPS_RESTRICT output,
                                    const float* NNOPS_RESTRICT input,
                                    int ir_step, int K,
                                    const float* NNOPS_RESTRICT scale,
                                    const int32_t* NNOPS_RESTRICT zero) noexcept
{
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    int32_t* NNOPS_RESTRICT out_i32 = static_cast<int32_t*>(output);

    float inv_s[8];
    int32_t zp[8];
    for (int i = 0; i < 8; ++i) {
        inv_s[i] = 1.0f / scale[i];
        zp[i] = (zero == nullptr) ? 0 : zero[i];
    }

    int k = 0;
    for (; k <= K - 4; k += 4) {
        // Process in 2 groups of 4 rows to keep register pressure low
        for (int g = 0; g < 2; ++g) {
            __m128i qi[4];
            for (int j = 0; j < 4; ++j) {
                int i = g * 4 + j;
                __m128 v = _mm_loadu_ps(input + i * ir_step);
                __m128 v_s = _mm_set1_ps(inv_s[i]);
                qi[j] = _mm_cvtps_epi32(_mm_mul_ps(v, v_s));
                qi[j] = _mm_add_epi32(qi[j], _mm_set1_epi32(zp[i]));
            }
            __m128i i16_01 = _mm_packs_epi32(qi[0], qi[1]);
            __m128i i16_23 = _mm_packs_epi32(qi[2], qi[3]);
            __m128i i8;
            if constexpr (Q_U8) {
                i8 = _mm_packus_epi16(i16_01, i16_23);
            } else {
                i8 = _mm_packs_epi16(i16_01, i16_23);
            }
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out_i32 + g * 4), i8);
        }
        out_i32 += 8;
        input += 4;
    }

    for (; k < K; ++k) {
        int8_t* out_i8 = reinterpret_cast<int8_t*>(out_i32);
        for (int i = 0; i < 8; ++i) {
            float qval = input[i * ir_step] * inv_s[i];
            int32_t qi = static_cast<int32_t>(std::nearbyintf(qval)) + zp[i];
            out_i8[i] = static_cast<int8_t>(std::min(std::max(qi, qmin), qmax));
        }
        out_i32++;
        input += 1;
    }
}

template <bool Q_U8 = false>
inline void pack_quant_trans_n12_i8(void* NNOPS_RESTRICT output,
                                     const float* NNOPS_RESTRICT input,
                                     int ir_step, int K,
                                     const float* NNOPS_RESTRICT scale,
                                     const int32_t* NNOPS_RESTRICT zero) noexcept
{
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    int32_t* NNOPS_RESTRICT out_i32 = static_cast<int32_t*>(output);

    float inv_s[12];
    int32_t zp[12];
    for (int i = 0; i < 12; ++i) {
        inv_s[i] = 1.0f / scale[i];
        zp[i] = (zero == nullptr) ? 0 : zero[i];
    }

    int k = 0;
    for (; k <= K - 4; k += 4) {
        // Process in 3 groups of 4 rows
        for (int g = 0; g < 3; ++g) {
            __m128i qi[4];
            for (int j = 0; j < 4; ++j) {
                int i = g * 4 + j;
                __m128 v = _mm_loadu_ps(input + i * ir_step);
                __m128 v_s = _mm_set1_ps(inv_s[i]);
                qi[j] = _mm_cvtps_epi32(_mm_mul_ps(v, v_s));
                qi[j] = _mm_add_epi32(qi[j], _mm_set1_epi32(zp[i]));
            }
            __m128i i16_01 = _mm_packs_epi32(qi[0], qi[1]);
            __m128i i16_23 = _mm_packs_epi32(qi[2], qi[3]);
            __m128i i8;
            if constexpr (Q_U8) {
                i8 = _mm_packus_epi16(i16_01, i16_23);
            } else {
                i8 = _mm_packs_epi16(i16_01, i16_23);
            }
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out_i32 + g * 4), i8);
        }
        out_i32 += 12;
        input += 4;
    }

    for (; k < K; ++k) {
        int8_t* out_i8 = reinterpret_cast<int8_t*>(out_i32);
        for (int i = 0; i < 12; ++i) {
            float qval = input[i * ir_step] * inv_s[i];
            int32_t qi = static_cast<int32_t>(std::nearbyintf(qval)) + zp[i];
            out_i8[i] = static_cast<int8_t>(std::min(std::max(qi, qmin), qmax));
        }
        out_i32++;
        input += 1;
    }
}

template <bool Q_U8 = false>
inline void pack_quant_trans_n16_i8(void* NNOPS_RESTRICT output,
                                     const float* NNOPS_RESTRICT input,
                                     int ir_step, int K,
                                     const float* NNOPS_RESTRICT scale,
                                     const int32_t* NNOPS_RESTRICT zero) noexcept
{
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    int32_t* NNOPS_RESTRICT out_i32 = static_cast<int32_t*>(output);

    float inv_s[16];
    int32_t zp[16];
    for (int i = 0; i < 16; ++i) {
        inv_s[i] = 1.0f / scale[i];
        zp[i] = (zero == nullptr) ? 0 : zero[i];
    }

    int k = 0;
    for (; k <= K - 4; k += 4) {
        // Process in 4 groups of 4 rows
        for (int g = 0; g < 4; ++g) {
            __m128i qi[4];
            for (int j = 0; j < 4; ++j) {
                int i = g * 4 + j;
                __m128 v = _mm_loadu_ps(input + i * ir_step);
                __m128 v_s = _mm_set1_ps(inv_s[i]);
                qi[j] = _mm_cvtps_epi32(_mm_mul_ps(v, v_s));
                qi[j] = _mm_add_epi32(qi[j], _mm_set1_epi32(zp[i]));
            }
            __m128i i16_01 = _mm_packs_epi32(qi[0], qi[1]);
            __m128i i16_23 = _mm_packs_epi32(qi[2], qi[3]);
            __m128i i8;
            if constexpr (Q_U8) {
                i8 = _mm_packus_epi16(i16_01, i16_23);
            } else {
                i8 = _mm_packs_epi16(i16_01, i16_23);
            }
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out_i32 + g * 4), i8);
        }
        out_i32 += 16;
        input += 4;
    }

    for (; k < K; ++k) {
        int8_t* out_i8 = reinterpret_cast<int8_t*>(out_i32);
        for (int i = 0; i < 16; ++i) {
            float qval = input[i * ir_step] * inv_s[i];
            int32_t qi = static_cast<int32_t>(std::nearbyintf(qval)) + zp[i];
            out_i8[i] = static_cast<int8_t>(std::min(std::max(qi, qmin), qmax));
        }
        out_i32++;
        input += 1;
    }
}

// =========================================================================
//  RHS Copy pack + quantize  (f32 → i8, K-shared scale/zero_point)
//
//  Scale/zero_point is broadcast across all K elements (single scale/zero_point
//  for the entire K dimension).  K step = 4 (group 4 consecutive K elements).
//  Quantization: round(src * inv_scale) + zero_point.
// =========================================================================

template <bool Q_U8 = false>
inline void pack_quant_copy_n1_i8(void* NNOPS_RESTRICT output,
                                   const float* NNOPS_RESTRICT input,
                                   int ir_step, int K,
                                   const float* NNOPS_RESTRICT scale,
                                   const int32_t* NNOPS_RESTRICT zero) noexcept
{
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    int8_t* NNOPS_RESTRICT out_i8 = static_cast<int8_t*>(output);
    const float inv_s = 1.0f / scale[0];
    const int32_t zp_val = (zero == nullptr) ? 0 : zero[0];

    int k = 0;
    for (; k <= K - 4; k += 4) {
        for (int kk = 0; kk < 4; ++kk) {
            float qval = input[kk * ir_step] * inv_s;
            int32_t qi = static_cast<int32_t>(std::nearbyintf(qval)) + zp_val;
            out_i8[kk] = static_cast<int8_t>(std::min(std::max(qi, qmin), qmax));
        }
        out_i8 += 4;
        input += 4 * ir_step;
    }
    if (k < K) {
        for (int kk = 0; kk < 4 && k + kk < K; ++kk) {
            float qval = input[kk * ir_step] * inv_s;
            int32_t qi = static_cast<int32_t>(std::nearbyintf(qval)) + zp_val;
            out_i8[kk] = static_cast<int8_t>(std::min(std::max(qi, qmin), qmax));
        }
        for (int kk = K - k; kk < 4; ++kk) {
            out_i8[kk] = 0;
        }
    }
}

template <bool Q_U8 = false>
inline void pack_quant_copy_n4_i8(void* NNOPS_RESTRICT output,
                                   const float* NNOPS_RESTRICT input,
                                   int ir_step, int K,
                                   const float* NNOPS_RESTRICT scale,
                                   const int32_t* NNOPS_RESTRICT zero) noexcept
{
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    int8_t* NNOPS_RESTRICT out_i8 = static_cast<int8_t*>(output);
    const float inv_s = 1.0f / scale[0];
    const int32_t zp_val = (zero == nullptr) ? 0 : zero[0];

    int k = 0;
    for (; k <= K - 4; k += 4) {
        for (int x = 0; x < 4; ++x) {
            out_i8[0] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[0 * ir_step + x] * inv_s)) + zp_val, qmin), qmax));
            out_i8[1] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[1 * ir_step + x] * inv_s)) + zp_val, qmin), qmax));
            out_i8[2] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[2 * ir_step + x] * inv_s)) + zp_val, qmin), qmax));
            out_i8[3] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[3 * ir_step + x] * inv_s)) + zp_val, qmin), qmax));
            out_i8 += 4;
        }
        input += 4 * ir_step;
    }

    if (k < K) {
        for (int x = 0; x < 4; ++x) {
            out_i8[0] = k + 0 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[0 * ir_step + x] * inv_s)) + zp_val, qmin), qmax)) : 0;
            out_i8[1] = k + 1 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[1 * ir_step + x] * inv_s)) + zp_val, qmin), qmax)) : 0;
            out_i8[2] = k + 2 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[2 * ir_step + x] * inv_s)) + zp_val, qmin), qmax)) : 0;
            out_i8[3] = k + 3 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[3 * ir_step + x] * inv_s)) + zp_val, qmin), qmax)) : 0;
            out_i8 += 4;
        }
    }
}

template <bool Q_U8 = false>
inline void pack_quant_copy_n6_i8(void* NNOPS_RESTRICT output,
                                   const float* NNOPS_RESTRICT input,
                                   int ir_step, int K,
                                   const float* NNOPS_RESTRICT scale,
                                   const int32_t* NNOPS_RESTRICT zero) noexcept
{
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    int8_t* NNOPS_RESTRICT out_i8 = static_cast<int8_t*>(output);
    const float inv_s = 1.0f / scale[0];
    const int32_t zp_val = (zero == nullptr) ? 0 : zero[0];

    int k = 0;
    for (; k <= K - 4; k += 4) {
        for (int x = 0; x < 6; ++x) {
            out_i8[0] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[0 * ir_step + x] * inv_s)) + zp_val, qmin), qmax));
            out_i8[1] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[1 * ir_step + x] * inv_s)) + zp_val, qmin), qmax));
            out_i8[2] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[2 * ir_step + x] * inv_s)) + zp_val, qmin), qmax));
            out_i8[3] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[3 * ir_step + x] * inv_s)) + zp_val, qmin), qmax));
            out_i8 += 4;
        }
        input += 4 * ir_step;
    }

    if (k < K) {
        for (int x = 0; x < 6; ++x) {
            out_i8[0] = k + 0 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[0 * ir_step + x] * inv_s)) + zp_val, qmin), qmax)) : 0;
            out_i8[1] = k + 1 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[1 * ir_step + x] * inv_s)) + zp_val, qmin), qmax)) : 0;
            out_i8[2] = k + 2 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[2 * ir_step + x] * inv_s)) + zp_val, qmin), qmax)) : 0;
            out_i8[3] = k + 3 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[3 * ir_step + x] * inv_s)) + zp_val, qmin), qmax)) : 0;
            out_i8 += 4;
        }
    }
}

template <bool Q_U8 = false>
inline void pack_quant_copy_n8_i8(void* NNOPS_RESTRICT output,
                                   const float* NNOPS_RESTRICT input,
                                   int ir_step, int K,
                                   const float* NNOPS_RESTRICT scale,
                                   const int32_t* NNOPS_RESTRICT zero) noexcept
{
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    int8_t* NNOPS_RESTRICT out_i8 = static_cast<int8_t*>(output);
    const float inv_s = 1.0f / scale[0];
    const int32_t zp_val = (zero == nullptr) ? 0 : zero[0];

    int k = 0;
    for (; k <= K - 4; k += 4) {
        for (int x = 0; x < 8; ++x) {
            out_i8[0] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[0 * ir_step + x] * inv_s)) + zp_val, qmin), qmax));
            out_i8[1] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[1 * ir_step + x] * inv_s)) + zp_val, qmin), qmax));
            out_i8[2] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[2 * ir_step + x] * inv_s)) + zp_val, qmin), qmax));
            out_i8[3] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[3 * ir_step + x] * inv_s)) + zp_val, qmin), qmax));
            out_i8 += 4;
        }
        input += 4 * ir_step;
    }

    if (k < K) {
        for (int x = 0; x < 8; ++x) {
            out_i8[0] = k + 0 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[0 * ir_step + x] * inv_s)) + zp_val, qmin), qmax)) : 0;
            out_i8[1] = k + 1 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[1 * ir_step + x] * inv_s)) + zp_val, qmin), qmax)) : 0;
            out_i8[2] = k + 2 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[2 * ir_step + x] * inv_s)) + zp_val, qmin), qmax)) : 0;
            out_i8[3] = k + 3 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[3 * ir_step + x] * inv_s)) + zp_val, qmin), qmax)) : 0;
            out_i8 += 4;
        }
    }
}

template <bool Q_U8 = false>
inline void pack_quant_copy_n12_i8(void* NNOPS_RESTRICT output,
                                    const float* NNOPS_RESTRICT input,
                                    int ir_step, int K,
                                    const float* NNOPS_RESTRICT scale,
                                    const int32_t* NNOPS_RESTRICT zero) noexcept
{
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    int8_t* NNOPS_RESTRICT out_i8 = static_cast<int8_t*>(output);
    const float inv_s = 1.0f / scale[0];
    const int32_t zp_val = (zero == nullptr) ? 0 : zero[0];

    int k = 0;
    for (; k <= K - 4; k += 4) {
        for (int x = 0; x < 12; ++x) {
            out_i8[0] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[0 * ir_step + x] * inv_s)) + zp_val, qmin), qmax));
            out_i8[1] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[1 * ir_step + x] * inv_s)) + zp_val, qmin), qmax));
            out_i8[2] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[2 * ir_step + x] * inv_s)) + zp_val, qmin), qmax));
            out_i8[3] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[3 * ir_step + x] * inv_s)) + zp_val, qmin), qmax));
            out_i8 += 4;
        }
        input += 4 * ir_step;
    }

    if (k < K) {
        for (int x = 0; x < 12; ++x) {
            out_i8[0] = k + 0 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[0 * ir_step + x] * inv_s)) + zp_val, qmin), qmax)) : 0;
            out_i8[1] = k + 1 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[1 * ir_step + x] * inv_s)) + zp_val, qmin), qmax)) : 0;
            out_i8[2] = k + 2 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[2 * ir_step + x] * inv_s)) + zp_val, qmin), qmax)) : 0;
            out_i8[3] = k + 3 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[3 * ir_step + x] * inv_s)) + zp_val, qmin), qmax)) : 0;
            out_i8 += 4;
        }
    }
}

template <bool Q_U8 = false>
inline void pack_quant_copy_n16_i8(void* NNOPS_RESTRICT output,
                                    const float* NNOPS_RESTRICT input,
                                    int ir_step, int K,
                                    const float* NNOPS_RESTRICT scale,
                                    const int32_t* NNOPS_RESTRICT zero) noexcept
{
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    int8_t* NNOPS_RESTRICT out_i8 = static_cast<int8_t*>(output);
    const float inv_s = 1.0f / scale[0];
    const int32_t zp_val = (zero == nullptr) ? 0 : zero[0];
    const __m128 v_inv_s = _mm_set1_ps(inv_s);
    const __m128i v_zp = _mm_set1_epi32(zp_val);

    int k = 0;
    for (; k <= K - 4; k += 4) {
        auto quant_row = [&](int row) {
            __m128 d0 = _mm_loadu_ps(input + row * ir_step + 0);
            __m128 d1 = _mm_loadu_ps(input + row * ir_step + 4);
            __m128 d2 = _mm_loadu_ps(input + row * ir_step + 8);
            __m128 d3 = _mm_loadu_ps(input + row * ir_step + 12);
            __m128i qi0 = _mm_cvtps_epi32(_mm_mul_ps(d0, v_inv_s));
            __m128i qi1 = _mm_cvtps_epi32(_mm_mul_ps(d1, v_inv_s));
            __m128i qi2 = _mm_cvtps_epi32(_mm_mul_ps(d2, v_inv_s));
            __m128i qi3 = _mm_cvtps_epi32(_mm_mul_ps(d3, v_inv_s));
            qi0 = _mm_add_epi32(qi0, v_zp);
            qi1 = _mm_add_epi32(qi1, v_zp);
            qi2 = _mm_add_epi32(qi2, v_zp);
            qi3 = _mm_add_epi32(qi3, v_zp);
            __m128i i16_01 = _mm_packs_epi32(qi0, qi1);
            __m128i i16_23 = _mm_packs_epi32(qi2, qi3);
            if constexpr (Q_U8) {
                return _mm_packus_epi16(i16_01, i16_23);
            } else {
                return _mm_packs_epi16(i16_01, i16_23);
            }
        };

        __m128i r0 = quant_row(0);
        __m128i r1 = quant_row(1);
        __m128i r2 = quant_row(2);
        __m128i r3 = quant_row(3);

        // Interleaved store matching dp4a layout
        transpose_4x16_i8(r0, r1, r2, r3);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out_i8 + 0 * 16), r0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out_i8 + 1 * 16), r1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out_i8 + 2 * 16), r2);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out_i8 + 3 * 16), r3);
        out_i8 += 4 * 16;
        input += 4 * ir_step;
    }

    if (k < K) {
        for (int x = 0; x < 16; ++x) {
            out_i8[0] = k + 0 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[0 * ir_step + x] * inv_s)) + zp_val, qmin), qmax)) : 0;
            out_i8[1] = k + 1 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[1 * ir_step + x] * inv_s)) + zp_val, qmin), qmax)) : 0;
            out_i8[2] = k + 2 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[2 * ir_step + x] * inv_s)) + zp_val, qmin), qmax)) : 0;
            out_i8[3] = k + 3 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[3 * ir_step + x] * inv_s)) + zp_val, qmin), qmax)) : 0;
            out_i8 += 4;
        }
    }
}

}  // namespace nnops::backend::cpu::x86_64