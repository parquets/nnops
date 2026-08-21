#pragma once
/// @file pack_quant_i8.hpp
/// @brief x86_64 AVX2+FMA fused pack + per-row quantize kernels for GEMM LHS/RHS.
///
/// These kernels fuse the f32→i8 quantization step into the GEMM packing step,
/// eliminating the intermediate quantized-but-unpacked buffer and halving the
/// memory traffic (one pass instead of quantize + pack).
///
/// Two transform families, mirroring pack_f32 / pack_dp4a_i8:
///   - pack_quant_trans_nN (LHS/A): reads N f32 rows (stride ir_step), transposes
///     N×K blocks, quantizes per-row (each row has its own scale/zero), packs
///     4 i8 per i32, writes contiguously.
///   - pack_quant_copy_nN (RHS/B): reads N-column f32 rows, quantizes with
///     K-shared scale/zero (broadcast across all K), groups each 4-k block into
///     contiguous 4-i8 runs per column.
///
/// Quantization: dst = clamp(round(src * inv_scale + zero), qmin, qmax)
/// where inv_scale = 1.0f / scale. Rounding is round-to-nearest-even.
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
//  LHS Transpose pack + quantize  (f32 → i8, per-row scale/zero)
//
//  K step = 8 (AVX2 __m256). After 8×N transpose, each __m256 lane
//  comes from a different input row, requiring per-lane scale/zero.
// =========================================================================

template <bool Q_U8 = false>
inline void pack_quant_trans_n1_i8(void* NNOPS_RESTRICT output,
                                    const float* NNOPS_RESTRICT input,
                                    int ir_step, int K,
                                    const float* NNOPS_RESTRICT scale,
                                    const float* NNOPS_RESTRICT zero) noexcept
{
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    int32_t* NNOPS_RESTRICT out_i32 = static_cast<int32_t*>(output);
    const float inv_scale = 1.0f / scale[0];
    const float zero_val = (zero == nullptr) ? 0.0f : zero[0];

    int k = 0;
    for (; k <= K - 4; k += 4) {
        int8_t* out_i8 = reinterpret_cast<int8_t*>(out_i32);
        for (int kk = 0; kk < 4; ++kk) {
            float q = input[kk * ir_step] * inv_scale + zero_val;
            int32_t qi = static_cast<int32_t>(std::nearbyintf(q));
            out_i8[kk] = static_cast<int8_t>(std::min(std::max(qi, qmin), qmax));
        }
        out_i32++;
        input += 4 * ir_step;
    }
    if (k < K) {
        int8_t* out_i8 = reinterpret_cast<int8_t*>(out_i32);
        for (int kk = 0; kk < 4 && k + kk < K; ++kk) {
            float q = input[kk * ir_step] * inv_scale + zero_val;
            int32_t qi = static_cast<int32_t>(std::nearbyintf(q));
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
                                    const float* NNOPS_RESTRICT zero) noexcept
{
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    int32_t* NNOPS_RESTRICT out_i32 = static_cast<int32_t*>(output);

    float inv_s[4], z[4];
    for (int i = 0; i < 4; ++i) {
        inv_s[i] = 1.0f / scale[i];
        z[i] = (zero == nullptr) ? 0.0f : zero[i];
    }
    const __m128 v_inv_s = _mm_loadu_ps(inv_s);
    const __m128 v_z = _mm_loadu_ps(z);

    int k = 0;
    for (; k <= K - 8; k += 8) {
        const float* NNOPS_RESTRICT p0 = input + 0 * ir_step;
        const float* NNOPS_RESTRICT p1 = input + 1 * ir_step;
        const float* NNOPS_RESTRICT p2 = input + 2 * ir_step;
        const float* NNOPS_RESTRICT p3 = input + 3 * ir_step;

        __m256 v0 = _mm256_loadu_ps(p0);
        __m256 v1 = _mm256_loadu_ps(p1);
        __m256 v2 = _mm256_loadu_ps(p2);
        __m256 v3 = _mm256_loadu_ps(p3);

        transpose_4x8_f32(v0, v1, v2, v3);

        // After 4×8 transpose: v0..v3 each hold 8 elements from all 4 rows.
        // Each lane position within v0 has elements from all 4 rows, so we
        // need per-lane scale/zero.  Process 2 groups of 4 floats per vector.
        for (int lane = 0; lane < 2; ++lane) {
            __m128 v0_lo = (lane == 0) ? _mm256_castps256_ps128(v0)
                                       : _mm256_extractf128_ps(v0, 1);
            __m128 v1_lo = (lane == 0) ? _mm256_castps256_ps128(v1)
                                       : _mm256_extractf128_ps(v1, 1);
            __m128 v2_lo = (lane == 0) ? _mm256_castps256_ps128(v2)
                                       : _mm256_extractf128_ps(v2, 1);
            __m128 v3_lo = (lane == 0) ? _mm256_castps256_ps128(v3)
                                       : _mm256_extractf128_ps(v3, 1);

            __m128i i0 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(v0_lo, v_inv_s), v_z));
            __m128i i1 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(v1_lo, v_inv_s), v_z));
            __m128i i2 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(v2_lo, v_inv_s), v_z));
            __m128i i3 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(v3_lo, v_inv_s), v_z));

            const __m128i i16_01 = _mm_packs_epi32(i0, i1);
            const __m128i i16_23 = _mm_packs_epi32(i2, i3);
            __m128i i8;
            if constexpr (Q_U8) {
                i8 = _mm_packus_epi16(i16_01, i16_23);
            } else {
                i8 = _mm_packs_epi16(i16_01, i16_23);
            }
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out_i32), i8);
            out_i32 += 4;
        }

        input += 8;
    }

    // 4-wide tail
    for (; k <= K - 4; k += 4) {
        __m128 v0 = _mm_loadu_ps(input + 0 * ir_step);
        __m128 v1 = _mm_loadu_ps(input + 1 * ir_step);
        __m128 v2 = _mm_loadu_ps(input + 2 * ir_step);
        __m128 v3 = _mm_loadu_ps(input + 3 * ir_step);

        __m128i i0 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(v0, v_inv_s), v_z));
        __m128i i1 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(v1, v_inv_s), v_z));
        __m128i i2 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(v2, v_inv_s), v_z));
        __m128i i3 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(v3, v_inv_s), v_z));

        const __m128i i16_01 = _mm_packs_epi32(i0, i1);
        const __m128i i16_23 = _mm_packs_epi32(i2, i3);
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
            float qval = input[i * ir_step] * inv_s[i] + z[i];
            int32_t qi = static_cast<int32_t>(std::nearbyintf(qval));
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
                                    const float* NNOPS_RESTRICT zero) noexcept
{
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    int32_t* NNOPS_RESTRICT out_i32 = static_cast<int32_t*>(output);

    float inv_s[6], z[6];
    for (int i = 0; i < 6; ++i) {
        inv_s[i] = 1.0f / scale[i];
        z[i] = (zero == nullptr) ? 0.0f : zero[i];
    }

    int k = 0;
    for (; k <= K - 8; k += 8) {
        const float* NNOPS_RESTRICT p[6];
        for (int i = 0; i < 6; ++i) p[i] = input + i * ir_step;

        __m256 v[6];
        for (int i = 0; i < 6; ++i) {
            v[i] = _mm256_loadu_ps(p[i]);
        }

        transpose_6x8_f32(v[0], v[1], v[2], v[3], v[4], v[5]);

        // After 6×8 transpose: each v[i] has 8 elements, each from 6 different rows.
        // Process 2 groups of 4 floats.
        for (int lane = 0; lane < 2; ++lane) {
            auto extract = [lane](__m256 x) {
                return (lane == 0) ? _mm256_castps256_ps128(x)
                                   : _mm256_extractf128_ps(x, 1);
            };

            // Rows 0-3
            __m128 v_inv_s0 = _mm_loadu_ps(inv_s + 0);
            __m128 v_z0 = _mm_loadu_ps(z + 0);
            __m128i q0 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(extract(v[0]), v_inv_s0), v_z0));
            __m128i q1 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(extract(v[1]), v_inv_s0), v_z0));
            __m128i q2 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(extract(v[2]), v_inv_s0), v_z0));
            __m128i q3 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(extract(v[3]), v_inv_s0), v_z0));

            const __m128i i16_01 = _mm_packs_epi32(q0, q1);
            const __m128i i16_23 = _mm_packs_epi32(q2, q3);
            __m128i i8;
            if constexpr (Q_U8) {
                i8 = _mm_packus_epi16(i16_01, i16_23);
            } else {
                i8 = _mm_packs_epi16(i16_01, i16_23);
            }
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out_i32), i8);
            out_i32 += 4;

            // Rows 4-5: only 2 rows, need to pack with zeros
            __m128 v_inv_s1 = _mm_set_ps(0.0f, 0.0f, inv_s[5], inv_s[4]);
            __m128 v_z1 = _mm_set_ps(0.0f, 0.0f, z[5], z[4]);
            __m128i q4 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(extract(v[4]), v_inv_s1), v_z1));
            __m128i q5 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(extract(v[5]), v_inv_s1), v_z1));

            const __m128i i16_45 = _mm_packs_epi32(q4, q5);
            __m128i i8_45;
            if constexpr (Q_U8) {
                i8_45 = _mm_packus_epi16(i16_45, _mm_setzero_si128());
            } else {
                i8_45 = _mm_packs_epi16(i16_45, _mm_setzero_si128());
            }
            _mm_storel_epi64(reinterpret_cast<__m128i*>(out_i32), i8_45);
            out_i32 += 2;
        }

        input += 8;
    }

    // 4-wide tail
    for (; k <= K - 4; k += 4) {
        int8_t* out_i8 = reinterpret_cast<int8_t*>(out_i32);
        for (int i = 0; i < 6; ++i) {
            float qval = input[i * ir_step] * inv_s[i] + z[i];
            int32_t qi = static_cast<int32_t>(std::nearbyintf(qval));
            out_i8[i] = static_cast<int8_t>(std::min(std::max(qi, qmin), qmax));
        }
        out_i32++;
        input += 1;
    }

    for (; k < K; ++k) {
        int8_t* out_i8 = reinterpret_cast<int8_t*>(out_i32);
        for (int i = 0; i < 6; ++i) {
            float qval = input[i * ir_step] * inv_s[i] + z[i];
            int32_t qi = static_cast<int32_t>(std::nearbyintf(qval));
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
                                    const float* NNOPS_RESTRICT zero) noexcept
{
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    int32_t* NNOPS_RESTRICT out_i32 = static_cast<int32_t*>(output);

    float inv_s[8], z[8];
    for (int i = 0; i < 8; ++i) {
        inv_s[i] = 1.0f / scale[i];
        z[i] = (zero == nullptr) ? 0.0f : zero[i];
    }

    int k = 0;
    for (; k <= K - 8; k += 8) {
        const float* NNOPS_RESTRICT p[8];
        for (int i = 0; i < 8; ++i) p[i] = input + i * ir_step;

        __m256 v[8];
        for (int i = 0; i < 8; ++i) {
            v[i] = _mm256_loadu_ps(p[i]);
        }

        transpose_8x8_f32(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);

        // After 8×8 transpose: each v[i] has 8 elements, each from all 8 rows.
        // Process 2 groups of 4 floats. Within each group of 4, the lanes
        // come from 4 different rows, needing per-lane scale/zero.
        for (int lane = 0; lane < 2; ++lane) {
            auto extract = [lane](__m256 x) {
                return (lane == 0) ? _mm256_castps256_ps128(x)
                                   : _mm256_extractf128_ps(x, 1);
            };

            // Rows 0-3
            __m128 v_s0 = _mm_loadu_ps(inv_s + 0);
            __m128 v_z0 = _mm_loadu_ps(z + 0);
            __m128i qi0 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(extract(v[0]), v_s0), v_z0));
            __m128i qi1 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(extract(v[1]), v_s0), v_z0));
            __m128i qi2 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(extract(v[2]), v_s0), v_z0));
            __m128i qi3 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(extract(v[3]), v_s0), v_z0));

            const __m128i i16_01 = _mm_packs_epi32(qi0, qi1);
            const __m128i i16_23 = _mm_packs_epi32(qi2, qi3);
            __m128i q0;
            if constexpr (Q_U8) {
                q0 = _mm_packus_epi16(i16_01, i16_23);
            } else {
                q0 = _mm_packs_epi16(i16_01, i16_23);
            }
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out_i32), q0);
            out_i32 += 4;

            // Rows 4-7
            __m128 v_s1 = _mm_loadu_ps(inv_s + 4);
            __m128 v_z1 = _mm_loadu_ps(z + 4);
            __m128i qi4 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(extract(v[4]), v_s1), v_z1));
            __m128i qi5 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(extract(v[5]), v_s1), v_z1));
            __m128i qi6 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(extract(v[6]), v_s1), v_z1));
            __m128i qi7 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(extract(v[7]), v_s1), v_z1));

            const __m128i i16_45 = _mm_packs_epi32(qi4, qi5);
            const __m128i i16_67 = _mm_packs_epi32(qi6, qi7);
            __m128i q1;
            if constexpr (Q_U8) {
                q1 = _mm_packus_epi16(i16_45, i16_67);
            } else {
                q1 = _mm_packs_epi16(i16_45, i16_67);
            }
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out_i32), q1);
            out_i32 += 4;
        }

        input += 8;
    }

    for (; k <= K - 4; k += 4) {
        int8_t* out_i8 = reinterpret_cast<int8_t*>(out_i32);
        for (int i = 0; i < 8; ++i) {
            float qval = input[i * ir_step] * inv_s[i] + z[i];
            int32_t qi = static_cast<int32_t>(std::nearbyintf(qval));
            out_i8[i] = static_cast<int8_t>(std::min(std::max(qi, qmin), qmax));
        }
        out_i32++;
        input += 1;
    }

    for (; k < K; ++k) {
        int8_t* out_i8 = reinterpret_cast<int8_t*>(out_i32);
        for (int i = 0; i < 8; ++i) {
            float qval = input[i * ir_step] * inv_s[i] + z[i];
            int32_t qi = static_cast<int32_t>(std::nearbyintf(qval));
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
                                     const float* NNOPS_RESTRICT zero) noexcept
{
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    int32_t* NNOPS_RESTRICT out_i32 = static_cast<int32_t*>(output);

    float inv_s[12], z[12];
    for (int i = 0; i < 12; ++i) {
        inv_s[i] = 1.0f / scale[i];
        z[i] = (zero == nullptr) ? 0.0f : zero[i];
    }

    int k = 0;
    for (; k <= K - 8; k += 8) {
        const float* NNOPS_RESTRICT p[12];
        for (int i = 0; i < 12; ++i) p[i] = input + i * ir_step;

        __m256 v[12];
        for (int i = 0; i < 12; ++i) {
            v[i] = _mm256_loadu_ps(p[i]);
        }

        transpose_12x8_f32(v[0], v[1], v[2], v[3], v[4], v[5],
                           v[6], v[7], v[8], v[9], v[10], v[11]);

        for (int lane = 0; lane < 2; ++lane) {
            auto extract = [lane](__m256 x) {
                return (lane == 0) ? _mm256_castps256_ps128(x)
                                   : _mm256_extractf128_ps(x, 1);
            };

            // Process 3 groups of 4 rows
            for (int g = 0; g < 3; ++g) {
                __m128 v_s = _mm_loadu_ps(inv_s + g * 4);
                __m128 v_z = _mm_loadu_ps(z + g * 4);
                __m128i qi0 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(extract(v[g * 4 + 0]), v_s), v_z));
                __m128i qi1 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(extract(v[g * 4 + 1]), v_s), v_z));
                __m128i qi2 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(extract(v[g * 4 + 2]), v_s), v_z));
                __m128i qi3 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(extract(v[g * 4 + 3]), v_s), v_z));

                const __m128i i16_01 = _mm_packs_epi32(qi0, qi1);
                const __m128i i16_23 = _mm_packs_epi32(qi2, qi3);
                __m128i q;
                if constexpr (Q_U8) {
                    q = _mm_packus_epi16(i16_01, i16_23);
                } else {
                    q = _mm_packs_epi16(i16_01, i16_23);
                }
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out_i32), q);
                out_i32 += 4;
            }
        }

        input += 8;
    }

    for (; k <= K - 4; k += 4) {
        int8_t* out_i8 = reinterpret_cast<int8_t*>(out_i32);
        for (int i = 0; i < 12; ++i) {
            float qval = input[i * ir_step] * inv_s[i] + z[i];
            int32_t qi = static_cast<int32_t>(std::nearbyintf(qval));
            out_i8[i] = static_cast<int8_t>(std::min(std::max(qi, qmin), qmax));
        }
        out_i32++;
        input += 1;
    }

    for (; k < K; ++k) {
        int8_t* out_i8 = reinterpret_cast<int8_t*>(out_i32);
        for (int i = 0; i < 12; ++i) {
            float qval = input[i * ir_step] * inv_s[i] + z[i];
            int32_t qi = static_cast<int32_t>(std::nearbyintf(qval));
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
                                     const float* NNOPS_RESTRICT zero) noexcept
{
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    int32_t* NNOPS_RESTRICT out_i32 = static_cast<int32_t*>(output);

    float inv_s[16], z[16];
    for (int i = 0; i < 16; ++i) {
        inv_s[i] = 1.0f / scale[i];
        z[i] = (zero == nullptr) ? 0.0f : zero[i];
    }

    int k = 0;
    for (; k <= K - 8; k += 8) {
        const float* NNOPS_RESTRICT p[16];
        for (int i = 0; i < 16; ++i) p[i] = input + i * ir_step;

        __m256 v[16];
        for (int i = 0; i < 16; ++i) {
            v[i] = _mm256_loadu_ps(p[i]);
        }

        transpose_16x8_f32(
            v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
            v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);

        for (int lane = 0; lane < 2; ++lane) {
            auto extract = [lane](__m256 x) {
                return (lane == 0) ? _mm256_castps256_ps128(x)
                                   : _mm256_extractf128_ps(x, 1);
            };

            // Process 4 groups of 4 rows
            for (int g = 0; g < 4; ++g) {
                __m128 v_s = _mm_loadu_ps(inv_s + g * 4);
                __m128 v_z = _mm_loadu_ps(z + g * 4);
                __m128i qi0 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(extract(v[g * 4 + 0]), v_s), v_z));
                __m128i qi1 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(extract(v[g * 4 + 1]), v_s), v_z));
                __m128i qi2 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(extract(v[g * 4 + 2]), v_s), v_z));
                __m128i qi3 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(extract(v[g * 4 + 3]), v_s), v_z));

                const __m128i i16_01 = _mm_packs_epi32(qi0, qi1);
                const __m128i i16_23 = _mm_packs_epi32(qi2, qi3);
                __m128i q;
                if constexpr (Q_U8) {
                    q = _mm_packus_epi16(i16_01, i16_23);
                } else {
                    q = _mm_packs_epi16(i16_01, i16_23);
                }
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out_i32), q);
                out_i32 += 4;
            }
        }

        input += 8;
    }

    for (; k <= K - 4; k += 4) {
        int8_t* out_i8 = reinterpret_cast<int8_t*>(out_i32);
        for (int i = 0; i < 16; ++i) {
            float qval = input[i * ir_step] * inv_s[i] + z[i];
            int32_t qi = static_cast<int32_t>(std::nearbyintf(qval));
            out_i8[i] = static_cast<int8_t>(std::min(std::max(qi, qmin), qmax));
        }
        out_i32++;
        input += 1;
    }

    for (; k < K; ++k) {
        int8_t* out_i8 = reinterpret_cast<int8_t*>(out_i32);
        for (int i = 0; i < 16; ++i) {
            float qval = input[i * ir_step] * inv_s[i] + z[i];
            int32_t qi = static_cast<int32_t>(std::nearbyintf(qval));
            out_i8[i] = static_cast<int8_t>(std::min(std::max(qi, qmin), qmax));
        }
        out_i32++;
        input += 1;
    }
}

// =========================================================================
//  RHS Copy pack + quantize  (f32 → i8, K-shared scale/zero)
//
//  Scale/zero is broadcast across all K elements (single scale/zero for the
//  entire K dimension).  K step = 4 (group 4 consecutive K elements).
// =========================================================================

template <bool Q_U8 = false>
inline void pack_quant_copy_n1_i8(void* NNOPS_RESTRICT output,
                                   const float* NNOPS_RESTRICT input,
                                   int ir_step, int K,
                                   const float* NNOPS_RESTRICT scale,
                                   const float* NNOPS_RESTRICT zero) noexcept
{
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    int8_t* NNOPS_RESTRICT out_i8 = static_cast<int8_t*>(output);
    const float inv_s = 1.0f / scale[0];
    const float zval = (zero == nullptr) ? 0.0f : zero[0];

    int k = 0;
    for (; k <= K - 4; k += 4) {
        for (int kk = 0; kk < 4; ++kk) {
            float qval = input[kk * ir_step] * inv_s + zval;
            int32_t qi = static_cast<int32_t>(std::nearbyintf(qval));
            out_i8[kk] = static_cast<int8_t>(std::min(std::max(qi, qmin), qmax));
        }
        out_i8 += 4;
        input += 4 * ir_step;
    }
    if (k < K) {
        for (int kk = 0; kk < 4 && k + kk < K; ++kk) {
            float qval = input[kk * ir_step] * inv_s + zval;
            int32_t qi = static_cast<int32_t>(std::nearbyintf(qval));
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
                                   const float* NNOPS_RESTRICT zero) noexcept
{
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    int8_t* NNOPS_RESTRICT out_i8 = static_cast<int8_t*>(output);
    const float inv_s = 1.0f / scale[0];
    const float zval = (zero == nullptr) ? 0.0f : zero[0];

    int k = 0;
    for (; k <= K - 4; k += 4) {
        for (int x = 0; x < 4; ++x) {
            float q0 = input[0 * ir_step + x] * inv_s + zval;
            float q1 = input[1 * ir_step + x] * inv_s + zval;
            float q2 = input[2 * ir_step + x] * inv_s + zval;
            float q3 = input[3 * ir_step + x] * inv_s + zval;
            out_i8[0] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(q0)), qmin), qmax));
            out_i8[1] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(q1)), qmin), qmax));
            out_i8[2] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(q2)), qmin), qmax));
            out_i8[3] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(q3)), qmin), qmax));
            out_i8 += 4;
        }
        input += 4 * ir_step;
    }

    if (k < K) {
        for (int x = 0; x < 4; ++x) {
            out_i8[0] = k + 0 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[0 * ir_step + x] * inv_s + zval)), qmin), qmax)) : 0;
            out_i8[1] = k + 1 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[1 * ir_step + x] * inv_s + zval)), qmin), qmax)) : 0;
            out_i8[2] = k + 2 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[2 * ir_step + x] * inv_s + zval)), qmin), qmax)) : 0;
            out_i8[3] = k + 3 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[3 * ir_step + x] * inv_s + zval)), qmin), qmax)) : 0;
            out_i8 += 4;
        }
    }
}

template <bool Q_U8 = false>
inline void pack_quant_copy_n6_i8(void* NNOPS_RESTRICT output,
                                   const float* NNOPS_RESTRICT input,
                                   int ir_step, int K,
                                   const float* NNOPS_RESTRICT scale,
                                   const float* NNOPS_RESTRICT zero) noexcept
{
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    int8_t* NNOPS_RESTRICT out_i8 = static_cast<int8_t*>(output);
    const float inv_s = 1.0f / scale[0];
    const float zval = (zero == nullptr) ? 0.0f : zero[0];

    int k = 0;
    for (; k <= K - 4; k += 4) {
        for (int x = 0; x < 6; ++x) {
            out_i8[0] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[0 * ir_step + x] * inv_s + zval)), qmin), qmax));
            out_i8[1] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[1 * ir_step + x] * inv_s + zval)), qmin), qmax));
            out_i8[2] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[2 * ir_step + x] * inv_s + zval)), qmin), qmax));
            out_i8[3] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[3 * ir_step + x] * inv_s + zval)), qmin), qmax));
            out_i8 += 4;
        }
        input += 4 * ir_step;
    }

    if (k < K) {
        for (int x = 0; x < 6; ++x) {
            out_i8[0] = k + 0 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[0 * ir_step + x] * inv_s + zval)), qmin), qmax)) : 0;
            out_i8[1] = k + 1 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[1 * ir_step + x] * inv_s + zval)), qmin), qmax)) : 0;
            out_i8[2] = k + 2 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[2 * ir_step + x] * inv_s + zval)), qmin), qmax)) : 0;
            out_i8[3] = k + 3 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[3 * ir_step + x] * inv_s + zval)), qmin), qmax)) : 0;
            out_i8 += 4;
        }
    }
}

template <bool Q_U8 = false>
inline void pack_quant_copy_n8_i8(void* NNOPS_RESTRICT output,
                                   const float* NNOPS_RESTRICT input,
                                   int ir_step, int K,
                                   const float* NNOPS_RESTRICT scale,
                                   const float* NNOPS_RESTRICT zero) noexcept
{
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    int8_t* NNOPS_RESTRICT out_i8 = static_cast<int8_t*>(output);
    const float inv_s = 1.0f / scale[0];
    const float zval = (zero == nullptr) ? 0.0f : zero[0];

    int k = 0;
    for (; k <= K - 4; k += 4) {
        for (int x = 0; x < 8; ++x) {
            out_i8[0] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[0 * ir_step + x] * inv_s + zval)), qmin), qmax));
            out_i8[1] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[1 * ir_step + x] * inv_s + zval)), qmin), qmax));
            out_i8[2] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[2 * ir_step + x] * inv_s + zval)), qmin), qmax));
            out_i8[3] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[3 * ir_step + x] * inv_s + zval)), qmin), qmax));
            out_i8 += 4;
        }
        input += 4 * ir_step;
    }

    if (k < K) {
        for (int x = 0; x < 8; ++x) {
            out_i8[0] = k + 0 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[0 * ir_step + x] * inv_s + zval)), qmin), qmax)) : 0;
            out_i8[1] = k + 1 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[1 * ir_step + x] * inv_s + zval)), qmin), qmax)) : 0;
            out_i8[2] = k + 2 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[2 * ir_step + x] * inv_s + zval)), qmin), qmax)) : 0;
            out_i8[3] = k + 3 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[3 * ir_step + x] * inv_s + zval)), qmin), qmax)) : 0;
            out_i8 += 4;
        }
    }
}

template <bool Q_U8 = false>
inline void pack_quant_copy_n12_i8(void* NNOPS_RESTRICT output,
                                    const float* NNOPS_RESTRICT input,
                                    int ir_step, int K,
                                    const float* NNOPS_RESTRICT scale,
                                    const float* NNOPS_RESTRICT zero) noexcept
{
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    int8_t* NNOPS_RESTRICT out_i8 = static_cast<int8_t*>(output);
    const float inv_s = 1.0f / scale[0];
    const float zval = (zero == nullptr) ? 0.0f : zero[0];

    int k = 0;
    for (; k <= K - 4; k += 4) {
        for (int x = 0; x < 12; ++x) {
            out_i8[0] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[0 * ir_step + x] * inv_s + zval)), qmin), qmax));
            out_i8[1] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[1 * ir_step + x] * inv_s + zval)), qmin), qmax));
            out_i8[2] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[2 * ir_step + x] * inv_s + zval)), qmin), qmax));
            out_i8[3] = static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[3 * ir_step + x] * inv_s + zval)), qmin), qmax));
            out_i8 += 4;
        }
        input += 4 * ir_step;
    }

    if (k < K) {
        for (int x = 0; x < 12; ++x) {
            out_i8[0] = k + 0 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[0 * ir_step + x] * inv_s + zval)), qmin), qmax)) : 0;
            out_i8[1] = k + 1 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[1 * ir_step + x] * inv_s + zval)), qmin), qmax)) : 0;
            out_i8[2] = k + 2 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[2 * ir_step + x] * inv_s + zval)), qmin), qmax)) : 0;
            out_i8[3] = k + 3 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[3 * ir_step + x] * inv_s + zval)), qmin), qmax)) : 0;
            out_i8 += 4;
        }
    }
}

template <bool Q_U8 = false>
inline void pack_quant_copy_n16_i8(void* NNOPS_RESTRICT output,
                                    const float* NNOPS_RESTRICT input,
                                    int ir_step, int K,
                                    const float* NNOPS_RESTRICT scale,
                                    const float* NNOPS_RESTRICT zero) noexcept
{
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    int8_t* NNOPS_RESTRICT out_i8 = static_cast<int8_t*>(output);
    const float inv_s = 1.0f / scale[0];
    const float zval = (zero == nullptr) ? 0.0f : zero[0];
    const __m128 v_inv_s = _mm_set1_ps(inv_s);
    const __m128 v_z = _mm_set1_ps(zval);

    int k = 0;
    for (; k <= K - 4; k += 4) {
        // Quantize 16 elements per row
        auto quant_row = [&](int row) {
            __m128 d0 = _mm_loadu_ps(input + row * ir_step + 0);
            __m128 d1 = _mm_loadu_ps(input + row * ir_step + 4);
            __m128 d2 = _mm_loadu_ps(input + row * ir_step + 8);
            __m128 d3 = _mm_loadu_ps(input + row * ir_step + 12);
            __m128i qi0 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(d0, v_inv_s), v_z));
            __m128i qi1 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(d1, v_inv_s), v_z));
            __m128i qi2 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(d2, v_inv_s), v_z));
            __m128i qi3 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(d3, v_inv_s), v_z));
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
                static_cast<int32_t>(std::nearbyintf(input[0 * ir_step + x] * inv_s + zval)), qmin), qmax)) : 0;
            out_i8[1] = k + 1 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[1 * ir_step + x] * inv_s + zval)), qmin), qmax)) : 0;
            out_i8[2] = k + 2 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[2 * ir_step + x] * inv_s + zval)), qmin), qmax)) : 0;
            out_i8[3] = k + 3 < K ? static_cast<int8_t>(std::min(std::max(
                static_cast<int32_t>(std::nearbyintf(input[3 * ir_step + x] * inv_s + zval)), qmin), qmax)) : 0;
            out_i8 += 4;
        }
    }
}

}  // namespace nnops::backend::cpu::x86_64