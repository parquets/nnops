#pragma once
/// @file pack_quant_i8.hpp
/// @brief AArch64 NEON fused pack + per-row quantize kernels for GEMM LHS/RHS.
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
/// Reference: nn_compute/src/cpu/kernel/pack/aarch64/pack_quant_i8.hpp

#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "backend/cpu/common/restrict.hpp"
#include "transpose.hpp"

namespace nnops::backend::cpu::aarch64 {

// =========================================================================
//  Per-lane quantization helper (for pack_trans after 4×4 transpose)
//
//  After transpose, each float32x4_t lane comes from a different input row,
//  so each lane needs its own scale/zero.  We use mul+add (not FMA) because
//  the scale and zero are per-lane, not broadcast.
// =========================================================================

template <bool Q_U8 = false>
inline int8x16_t quant_trans_block_4x4_f32_i8(
    float32x4_t d0, float32x4_t d1, float32x4_t d2, float32x4_t d3,
    const float32x4_t& v_inv_scale, const float32x4_t& v_zero) noexcept
{
    // Per-lane: dst = src * inv_scale + zero
    d0 = vaddq_f32(vmulq_f32(d0, v_inv_scale), v_zero);
    d1 = vaddq_f32(vmulq_f32(d1, v_inv_scale), v_zero);
    d2 = vaddq_f32(vmulq_f32(d2, v_inv_scale), v_zero);
    d3 = vaddq_f32(vmulq_f32(d3, v_inv_scale), v_zero);

    const int32x4_t i0 = vcvtnq_s32_f32(d0);
    const int32x4_t i1 = vcvtnq_s32_f32(d1);
    const int32x4_t i2 = vcvtnq_s32_f32(d2);
    const int32x4_t i3 = vcvtnq_s32_f32(d3);

    const int16x8_t s01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
    const int16x8_t s23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));

    if constexpr (Q_U8) {
        const uint8x8_t u0 = vqmovun_s16(s01);
        const uint8x8_t u1 = vqmovun_s16(s23);
        return vreinterpretq_s8_u8(vcombine_u8(u0, u1));
    } else {
        const int8x8_t q0 = vqmovn_s16(s01);
        const int8x8_t q1 = vqmovn_s16(s23);
        return vcombine_s8(q0, q1);
    }
}

/// Quantize 4 floats (one float32x4_t) with per-lane scale/zero into int32x4_t.
/// Used for the pack_copy path where each K position has its own scale.
template <bool Q_U8 = false>
inline int32x4_t quant_4f32_to_i32(float32x4_t d,
                                    float32x4_t v_inv_scale,
                                    float32x4_t v_zero) noexcept
{
    d = vaddq_f32(vmulq_f32(d, v_inv_scale), v_zero);
    return vcvtnq_s32_f32(d);
}

/// Narrow 4 int32x4_t to int8x16_t, saturating to the target range.
template <bool Q_U8 = false>
inline int8x16_t narrow_4i32_to_i8(int32x4_t i0, int32x4_t i1,
                                    int32x4_t i2, int32x4_t i3) noexcept
{
    const int16x8_t s01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
    const int16x8_t s23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
    if constexpr (Q_U8) {
        const uint8x8_t u0 = vqmovun_s16(s01);
        const uint8x8_t u1 = vqmovun_s16(s23);
        return vreinterpretq_s8_u8(vcombine_u8(u0, u1));
    } else {
        const int8x8_t q0 = vqmovn_s16(s01);
        const int8x8_t q1 = vqmovn_s16(s23);
        return vcombine_s8(q0, q1);
    }
}

// =========================================================================
//  LHS Transpose pack + quantize  (f32 → i8, per-row scale/zero)
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
        float vals[4];
        for (int kk = 0; kk < 4; ++kk) {
            float q = input[kk * ir_step] * inv_scale + zero_val;
            int32_t qi = static_cast<int32_t>(std::nearbyintf(q));
            vals[kk] = static_cast<float>(std::min(std::max(qi, qmin), qmax));
        }
        // Pack 4 i8 into one i32
        int8_t* out_i8 = reinterpret_cast<int8_t*>(out_i32);
        out_i8[0] = static_cast<int8_t>(static_cast<int32_t>(vals[0]));
        out_i8[1] = static_cast<int8_t>(static_cast<int32_t>(vals[1]));
        out_i8[2] = static_cast<int8_t>(static_cast<int32_t>(vals[2]));
        out_i8[3] = static_cast<int8_t>(static_cast<int32_t>(vals[3]));
        out_i32++;
        input += 4 * ir_step;
    }
    if (k < K) {
        int8_t* out_i8 = reinterpret_cast<int8_t*>(out_i32);
        out_i8[0] = k + 0 < K ? static_cast<int8_t>(std::min(std::max(
            static_cast<int32_t>(std::nearbyintf(input[0 * ir_step] * inv_scale + zero_val)), qmin), qmax)) : 0;
        out_i8[1] = k + 1 < K ? static_cast<int8_t>(std::min(std::max(
            static_cast<int32_t>(std::nearbyintf(input[1 * ir_step] * inv_scale + zero_val)), qmin), qmax)) : 0;
        out_i8[2] = k + 2 < K ? static_cast<int8_t>(std::min(std::max(
            static_cast<int32_t>(std::nearbyintf(input[2 * ir_step] * inv_scale + zero_val)), qmin), qmax)) : 0;
        out_i8[3] = k + 3 < K ? static_cast<int8_t>(std::min(std::max(
            static_cast<int32_t>(std::nearbyintf(input[3 * ir_step] * inv_scale + zero_val)), qmin), qmax)) : 0;
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

    // Per-row scale/zero
    float inv_s[4], z[4];
    for (int i = 0; i < 4; ++i) {
        inv_s[i] = 1.0f / scale[i];
        z[i] = (zero == nullptr) ? 0.0f : zero[i];
    }
    const float32x4_t v_inv_scale = vld1q_f32(inv_s);
    const float32x4_t v_zero = vld1q_f32(z);

    int k = 0;
    for (; k <= K - 4; k += 4) {
        float32x4_t v0 = vld1q_f32(input + 0 * ir_step);
        float32x4_t v1 = vld1q_f32(input + 1 * ir_step);
        float32x4_t v2 = vld1q_f32(input + 2 * ir_step);
        float32x4_t v3 = vld1q_f32(input + 3 * ir_step);

        // Transpose 4×4 f32
        transpose_4x4_f32(v0, v1, v2, v3);

        // Quantize with per-lane scale/zero, then narrow to i8
        const int8x16_t q = quant_trans_block_4x4_f32_i8<Q_U8>(
            v0, v1, v2, v3, v_inv_scale, v_zero);

        vst1q_s32(out_i32, vreinterpretq_s32_s8(q));
        out_i32 += 4;
        input += 4;
    }

    // Scalar tail
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
    for (; k <= K - 4; k += 4) {
        float32x4_t v0 = vld1q_f32(input + 0 * ir_step);
        float32x4_t v1 = vld1q_f32(input + 1 * ir_step);
        float32x4_t v2 = vld1q_f32(input + 2 * ir_step);
        float32x4_t v3 = vld1q_f32(input + 3 * ir_step);
        float32x4_t v4 = vld1q_f32(input + 4 * ir_step);
        float32x4_t v5 = vld1q_f32(input + 5 * ir_step);
        float32x4_t v6 = vld1q_f32(input + 6 * ir_step);
        float32x4_t v7 = vld1q_f32(input + 7 * ir_step);

        // Transpose 8×4 f32
        transpose_8x4_f32(v0, v1, v2, v3, v4, v5, v6, v7);

        // Quantize rows 0-3 with per-lane scale
        float32x4_t v_s03 = vld1q_f32(inv_s + 0);
        float32x4_t v_z03 = vld1q_f32(z + 0);
        const int8x16_t q0 = quant_trans_block_4x4_f32_i8<Q_U8>(
            v0, v1, v2, v3, v_s03, v_z03);

        // Quantize rows 4-7 with per-lane scale
        float32x4_t v_s47 = vld1q_f32(inv_s + 4);
        float32x4_t v_z47 = vld1q_f32(z + 4);
        const int8x16_t q1 = quant_trans_block_4x4_f32_i8<Q_U8>(
            v4, v5, v6, v7, v_s47, v_z47);

        vst1q_s32(out_i32 + 0, vreinterpretq_s32_s8(q0));
        vst1q_s32(out_i32 + 4, vreinterpretq_s32_s8(q1));
        out_i32 += 8;
        input += 4;
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
    for (; k <= K - 4; k += 4) {
        float32x4_t v[12];
        for (int i = 0; i < 12; ++i) {
            v[i] = vld1q_f32(input + i * ir_step);
        }

        transpose_12x4_f32(v[0], v[1], v[2], v[3], v[4], v[5],
                           v[6], v[7], v[8], v[9], v[10], v[11]);

        // Quantize in 3 groups of 4 rows
        for (int g = 0; g < 3; ++g) {
            float32x4_t v_s = vld1q_f32(inv_s + g * 4);
            float32x4_t v_z = vld1q_f32(z + g * 4);
            const int8x16_t q = quant_trans_block_4x4_f32_i8<Q_U8>(
                v[g * 4 + 0], v[g * 4 + 1], v[g * 4 + 2], v[g * 4 + 3],
                v_s, v_z);
            vst1q_s32(out_i32 + g * 4, vreinterpretq_s32_s8(q));
        }
        out_i32 += 12;
        input += 4;
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
    for (; k <= K - 4; k += 4) {
        float32x4_t v[16];
        for (int i = 0; i < 16; ++i) {
            v[i] = vld1q_f32(input + i * ir_step);
        }

        transpose_16x4_f32(
            v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
            v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);

        // Quantize in 4 groups of 4 rows
        for (int g = 0; g < 4; ++g) {
            float32x4_t v_s = vld1q_f32(inv_s + g * 4);
            float32x4_t v_z = vld1q_f32(z + g * 4);
            const int8x16_t q = quant_trans_block_4x4_f32_i8<Q_U8>(
                v[g * 4 + 0], v[g * 4 + 1], v[g * 4 + 2], v[g * 4 + 3],
                v_s, v_z);
            vst1q_s32(out_i32 + g * 4, vreinterpretq_s32_s8(q));
        }
        out_i32 += 16;
        input += 4;
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
    const float32x4_t v_inv_s = vdupq_n_f32(inv_s);
    const float32x4_t v_z = vdupq_n_f32(zval);

    int k = 0;
    for (; k <= K - 4; k += 4) {
        // Quantize 8 elements per row
        float32x4_t d0_lo = vld1q_f32(input + 0 * ir_step + 0);
        float32x4_t d0_hi = vld1q_f32(input + 0 * ir_step + 4);
        int16x8_t s0 = vcombine_s16(
            vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d0_lo, v_inv_s), v_z))),
            vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d0_hi, v_inv_s), v_z))));
        int8x8_t r0 = vqmovn_s16(s0);

        float32x4_t d1_lo = vld1q_f32(input + 1 * ir_step + 0);
        float32x4_t d1_hi = vld1q_f32(input + 1 * ir_step + 4);
        int16x8_t s1 = vcombine_s16(
            vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d1_lo, v_inv_s), v_z))),
            vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d1_hi, v_inv_s), v_z))));
        int8x8_t r1 = vqmovn_s16(s1);

        float32x4_t d2_lo = vld1q_f32(input + 2 * ir_step + 0);
        float32x4_t d2_hi = vld1q_f32(input + 2 * ir_step + 4);
        int16x8_t s2 = vcombine_s16(
            vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d2_lo, v_inv_s), v_z))),
            vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d2_hi, v_inv_s), v_z))));
        int8x8_t r2 = vqmovn_s16(s2);

        float32x4_t d3_lo = vld1q_f32(input + 3 * ir_step + 0);
        float32x4_t d3_hi = vld1q_f32(input + 3 * ir_step + 4);
        int16x8_t s3 = vcombine_s16(
            vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d3_lo, v_inv_s), v_z))),
            vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d3_hi, v_inv_s), v_z))));
        int8x8_t r3 = vqmovn_s16(s3);

        // Interleaved store matching dp4a layout
        int8x8x4_t v_out;
        v_out.val[0] = r0; v_out.val[1] = r1;
        v_out.val[2] = r2; v_out.val[3] = r3;
        vst4_s8(out_i8, v_out);
        out_i8 += 4 * 8;
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
    const float32x4_t v_inv_s = vdupq_n_f32(inv_s);
    const float32x4_t v_z = vdupq_n_f32(zval);

    int k = 0;
    for (; k <= K - 4; k += 4) {
        // First 8 columns: quantize per-row, interleave via vst4_s8
        {
            float32x4_t d0_lo = vld1q_f32(input + 0 * ir_step + 0);
            float32x4_t d0_hi = vld1q_f32(input + 0 * ir_step + 4);
            int16x8_t s0 = vcombine_s16(
                vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d0_lo, v_inv_s), v_z))),
                vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d0_hi, v_inv_s), v_z))));
            int8x8_t r0 = vqmovn_s16(s0);

            float32x4_t d1_lo = vld1q_f32(input + 1 * ir_step + 0);
            float32x4_t d1_hi = vld1q_f32(input + 1 * ir_step + 4);
            int16x8_t s1 = vcombine_s16(
                vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d1_lo, v_inv_s), v_z))),
                vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d1_hi, v_inv_s), v_z))));
            int8x8_t r1 = vqmovn_s16(s1);

            float32x4_t d2_lo = vld1q_f32(input + 2 * ir_step + 0);
            float32x4_t d2_hi = vld1q_f32(input + 2 * ir_step + 4);
            int16x8_t s2 = vcombine_s16(
                vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d2_lo, v_inv_s), v_z))),
                vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d2_hi, v_inv_s), v_z))));
            int8x8_t r2 = vqmovn_s16(s2);

            float32x4_t d3_lo = vld1q_f32(input + 3 * ir_step + 0);
            float32x4_t d3_hi = vld1q_f32(input + 3 * ir_step + 4);
            int16x8_t s3 = vcombine_s16(
                vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d3_lo, v_inv_s), v_z))),
                vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d3_hi, v_inv_s), v_z))));
            int8x8_t r3 = vqmovn_s16(s3);

            int8x8x4_t v_out;
            v_out.val[0] = r0; v_out.val[1] = r1;
            v_out.val[2] = r2; v_out.val[3] = r3;
            vst4_s8(out_i8, v_out);
            out_i8 += 4 * 8;
        }

        // Remaining 4 columns: scalar interleaved
        for (int x = 8; x < 12; ++x) {
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
    const float32x4_t v_inv_s = vdupq_n_f32(inv_s);
    const float32x4_t v_z = vdupq_n_f32(zval);

    int k = 0;
    for (; k <= K - 4; k += 4) {
        // Quantize 16 elements per row
        float32x4_t d0_0 = vld1q_f32(input + 0 * ir_step + 0);
        float32x4_t d0_1 = vld1q_f32(input + 0 * ir_step + 4);
        float32x4_t d0_2 = vld1q_f32(input + 0 * ir_step + 8);
        float32x4_t d0_3 = vld1q_f32(input + 0 * ir_step + 12);
        int16x8_t s0_lo = vcombine_s16(
            vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d0_0, v_inv_s), v_z))),
            vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d0_1, v_inv_s), v_z))));
        int16x8_t s0_hi = vcombine_s16(
            vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d0_2, v_inv_s), v_z))),
            vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d0_3, v_inv_s), v_z))));
        int8x16_t r0 = vcombine_s8(vqmovn_s16(s0_lo), vqmovn_s16(s0_hi));

        float32x4_t d1_0 = vld1q_f32(input + 1 * ir_step + 0);
        float32x4_t d1_1 = vld1q_f32(input + 1 * ir_step + 4);
        float32x4_t d1_2 = vld1q_f32(input + 1 * ir_step + 8);
        float32x4_t d1_3 = vld1q_f32(input + 1 * ir_step + 12);
        int16x8_t s1_lo = vcombine_s16(
            vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d1_0, v_inv_s), v_z))),
            vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d1_1, v_inv_s), v_z))));
        int16x8_t s1_hi = vcombine_s16(
            vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d1_2, v_inv_s), v_z))),
            vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d1_3, v_inv_s), v_z))));
        int8x16_t r1 = vcombine_s8(vqmovn_s16(s1_lo), vqmovn_s16(s1_hi));

        float32x4_t d2_0 = vld1q_f32(input + 2 * ir_step + 0);
        float32x4_t d2_1 = vld1q_f32(input + 2 * ir_step + 4);
        float32x4_t d2_2 = vld1q_f32(input + 2 * ir_step + 8);
        float32x4_t d2_3 = vld1q_f32(input + 2 * ir_step + 12);
        int16x8_t s2_lo = vcombine_s16(
            vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d2_0, v_inv_s), v_z))),
            vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d2_1, v_inv_s), v_z))));
        int16x8_t s2_hi = vcombine_s16(
            vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d2_2, v_inv_s), v_z))),
            vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d2_3, v_inv_s), v_z))));
        int8x16_t r2 = vcombine_s8(vqmovn_s16(s2_lo), vqmovn_s16(s2_hi));

        float32x4_t d3_0 = vld1q_f32(input + 3 * ir_step + 0);
        float32x4_t d3_1 = vld1q_f32(input + 3 * ir_step + 4);
        float32x4_t d3_2 = vld1q_f32(input + 3 * ir_step + 8);
        float32x4_t d3_3 = vld1q_f32(input + 3 * ir_step + 12);
        int16x8_t s3_lo = vcombine_s16(
            vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d3_0, v_inv_s), v_z))),
            vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d3_1, v_inv_s), v_z))));
        int16x8_t s3_hi = vcombine_s16(
            vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d3_2, v_inv_s), v_z))),
            vqmovn_s32(vcvtnq_s32_f32(vaddq_f32(vmulq_f32(d3_3, v_inv_s), v_z))));
        int8x16_t r3 = vcombine_s8(vqmovn_s16(s3_lo), vqmovn_s16(s3_hi));

        // Interleaved store matching dp4a layout
        int8x16x4_t v_out;
        v_out.val[0] = r0; v_out.val[1] = r1;
        v_out.val[2] = r2; v_out.val[3] = r3;
        vst4q_s8(out_i8, v_out);
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

}  // namespace nnops::backend::cpu::aarch64