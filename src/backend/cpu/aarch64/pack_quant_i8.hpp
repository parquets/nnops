#pragma once
/// @file pack_quant_i8.hpp
/// @brief AArch64 NEON fused pack + per-row quantize kernels for GEMM LHS/RHS.
///
/// These kernels fuse the f32→i8 quantization step into the GEMM packing step,
/// eliminating the intermediate quantized-but-unpacked buffer and halving the
/// memory traffic (one pass instead of quantize + pack).
///
/// Two transform families, mirroring pack_f32 / pack_dp4a_i8:
///   - pack_quant_trans_nN (LHS/A): reads N f32 rows (stride ir_step), quantizes
///     each row independently with its own scale/zero (broadcast across all K),
///     narrows to int8, then packs 4 consecutive K values from the SAME row into
///     each int32 to match the SDOT dp4a interleaved format.
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
/// Reference: nn_compute/src/cpu/kernel/pack/aarch64/pack_quant_i8.hpp

#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "backend/cpu/common/restrict.hpp"

namespace nnops::backend::cpu::aarch64 {

// =========================================================================
//  Per-lane quantization helpers.
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
//  LHS pack + quantize  (f32 → i8, per-row scale/zero_point)
//
//  Each row is quantized independently with its own scale/zero_point
//  (broadcast across all K elements).  K step = 4 (float32x4_t).
//  Quantization: round(src * inv_scale) + zero_point.
//  After narrowing, 4 consecutive K values from the SAME row are packed
//  into each int32, matching the SDOT dp4a interleaved format:
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
    const int32x4_t v_zp = vdupq_n_s32(zp);

    int k = 0;
    for (; k <= K - 4; k += 4) {
        float32x4_t v = vld1q_f32(input);
        float32x4_t v_s = vdupq_n_f32(inv_scale);
        int32x4_t qi = vcvtnq_s32_f32(vmulq_f32(v, v_s));
        qi = vaddq_s32(qi, v_zp);

        int16x4_t i16 = vqmovn_s32(qi);
        if constexpr (Q_U8) {
            uint8x8_t u8 = vqmovun_s16(vcombine_s16(i16, vdup_n_s16(0)));
            vst1_lane_s32(out_i32, vreinterpret_s32_u8(u8), 0);
        } else {
            int8x8_t i8 = vqmovn_s16(vcombine_s16(i16, vdup_n_s16(0)));
            vst1_lane_s32(out_i32, vreinterpret_s32_s8(i8), 0);
        }
        out_i32++;
        input += 4;
    }
    if (k < K) {
        int8_t* out_i8 = reinterpret_cast<int8_t*>(out_i32);
        out_i8[0] = k + 0 < K ? static_cast<int8_t>(std::min(std::max(
            static_cast<int32_t>(std::nearbyintf(input[0 * ir_step] * inv_scale)) + zp, qmin), qmax)) : 0;
        out_i8[1] = k + 1 < K ? static_cast<int8_t>(std::min(std::max(
            static_cast<int32_t>(std::nearbyintf(input[1 * ir_step] * inv_scale)) + zp, qmin), qmax)) : 0;
        out_i8[2] = k + 2 < K ? static_cast<int8_t>(std::min(std::max(
            static_cast<int32_t>(std::nearbyintf(input[2 * ir_step] * inv_scale)) + zp, qmin), qmax)) : 0;
        out_i8[3] = k + 3 < K ? static_cast<int8_t>(std::min(std::max(
            static_cast<int32_t>(std::nearbyintf(input[3 * ir_step] * inv_scale)) + zp, qmin), qmax)) : 0;
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
        for (int i = 0; i < 4; ++i) {
            float32x4_t v = vld1q_f32(input + i * ir_step);
            float32x4_t v_s = vdupq_n_f32(inv_s[i]);
            int32x4_t qi = vcvtnq_s32_f32(vmulq_f32(v, v_s));
            qi = vaddq_s32(qi, vdupq_n_s32(zp[i]));

            int16x4_t i16 = vqmovn_s32(qi);
            if constexpr (Q_U8) {
                uint8x8_t u8 = vqmovun_s16(vcombine_s16(i16, vdup_n_s16(0)));
                vst1_lane_s32(out_i32 + i, vreinterpret_s32_u8(u8), 0);
            } else {
                int8x8_t i8 = vqmovn_s16(vcombine_s16(i16, vdup_n_s16(0)));
                vst1_lane_s32(out_i32 + i, vreinterpret_s32_s8(i8), 0);
            }
        }
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
        for (int i = 0; i < 8; ++i) {
            float32x4_t v = vld1q_f32(input + i * ir_step);
            float32x4_t v_s = vdupq_n_f32(inv_s[i]);
            int32x4_t qi = vcvtnq_s32_f32(vmulq_f32(v, v_s));
            qi = vaddq_s32(qi, vdupq_n_s32(zp[i]));

            int16x4_t i16 = vqmovn_s32(qi);
            if constexpr (Q_U8) {
                uint8x8_t u8 = vqmovun_s16(vcombine_s16(i16, vdup_n_s16(0)));
                vst1_lane_s32(out_i32 + i, vreinterpret_s32_u8(u8), 0);
            } else {
                int8x8_t i8 = vqmovn_s16(vcombine_s16(i16, vdup_n_s16(0)));
                vst1_lane_s32(out_i32 + i, vreinterpret_s32_s8(i8), 0);
            }
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
        for (int i = 0; i < 12; ++i) {
            float32x4_t v = vld1q_f32(input + i * ir_step);
            float32x4_t v_s = vdupq_n_f32(inv_s[i]);
            int32x4_t qi = vcvtnq_s32_f32(vmulq_f32(v, v_s));
            qi = vaddq_s32(qi, vdupq_n_s32(zp[i]));

            int16x4_t i16 = vqmovn_s32(qi);
            if constexpr (Q_U8) {
                uint8x8_t u8 = vqmovun_s16(vcombine_s16(i16, vdup_n_s16(0)));
                vst1_lane_s32(out_i32 + i, vreinterpret_s32_u8(u8), 0);
            } else {
                int8x8_t i8 = vqmovn_s16(vcombine_s16(i16, vdup_n_s16(0)));
                vst1_lane_s32(out_i32 + i, vreinterpret_s32_s8(i8), 0);
            }
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
        for (int i = 0; i < 16; ++i) {
            float32x4_t v = vld1q_f32(input + i * ir_step);
            float32x4_t v_s = vdupq_n_f32(inv_s[i]);
            int32x4_t qi = vcvtnq_s32_f32(vmulq_f32(v, v_s));
            qi = vaddq_s32(qi, vdupq_n_s32(zp[i]));

            int16x4_t i16 = vqmovn_s32(qi);
            if constexpr (Q_U8) {
                uint8x8_t u8 = vqmovun_s16(vcombine_s16(i16, vdup_n_s16(0)));
                vst1_lane_s32(out_i32 + i, vreinterpret_s32_u8(u8), 0);
            } else {
                int8x8_t i8 = vqmovn_s16(vcombine_s16(i16, vdup_n_s16(0)));
                vst1_lane_s32(out_i32 + i, vreinterpret_s32_s8(i8), 0);
            }
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
    const float32x4_t v_inv_s = vdupq_n_f32(inv_s);
    const int32x4_t v_zp = vdupq_n_s32(zp_val);

    int k = 0;
    for (; k <= K - 4; k += 4) {
        auto quant_row8 = [&](int row) {
            float32x4_t d_lo = vld1q_f32(input + row * ir_step + 0);
            float32x4_t d_hi = vld1q_f32(input + row * ir_step + 4);
            int32x4_t qi_lo = vcvtnq_s32_f32(vmulq_f32(d_lo, v_inv_s));
            int32x4_t qi_hi = vcvtnq_s32_f32(vmulq_f32(d_hi, v_inv_s));
            qi_lo = vaddq_s32(qi_lo, v_zp);
            qi_hi = vaddq_s32(qi_hi, v_zp);
            int16x8_t s = vcombine_s16(vqmovn_s32(qi_lo), vqmovn_s32(qi_hi));
            return vqmovn_s16(s);
        };

        int8x8_t r0 = quant_row8(0);
        int8x8_t r1 = quant_row8(1);
        int8x8_t r2 = quant_row8(2);
        int8x8_t r3 = quant_row8(3);

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
    const float32x4_t v_inv_s = vdupq_n_f32(inv_s);
    const int32x4_t v_zp = vdupq_n_s32(zp_val);

    int k = 0;
    for (; k <= K - 4; k += 4) {
        auto quant_row8 = [&](int row) {
            float32x4_t d_lo = vld1q_f32(input + row * ir_step + 0);
            float32x4_t d_hi = vld1q_f32(input + row * ir_step + 4);
            int32x4_t qi_lo = vcvtnq_s32_f32(vmulq_f32(d_lo, v_inv_s));
            int32x4_t qi_hi = vcvtnq_s32_f32(vmulq_f32(d_hi, v_inv_s));
            qi_lo = vaddq_s32(qi_lo, v_zp);
            qi_hi = vaddq_s32(qi_hi, v_zp);
            int16x8_t s = vcombine_s16(vqmovn_s32(qi_lo), vqmovn_s32(qi_hi));
            return vqmovn_s16(s);
        };

        // First 8 columns: quantize per-row, interleave via vst4_s8
        {
            int8x8_t r0 = quant_row8(0);
            int8x8_t r1 = quant_row8(1);
            int8x8_t r2 = quant_row8(2);
            int8x8_t r3 = quant_row8(3);

            int8x8x4_t v_out;
            v_out.val[0] = r0; v_out.val[1] = r1;
            v_out.val[2] = r2; v_out.val[3] = r3;
            vst4_s8(out_i8, v_out);
            out_i8 += 4 * 8;
        }

        // Remaining 4 columns: scalar interleaved
        for (int x = 8; x < 12; ++x) {
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
    const float32x4_t v_inv_s = vdupq_n_f32(inv_s);
    const int32x4_t v_zp = vdupq_n_s32(zp_val);

    int k = 0;
    for (; k <= K - 4; k += 4) {
        auto quant_row16 = [&](int row) {
            float32x4_t d0 = vld1q_f32(input + row * ir_step + 0);
            float32x4_t d1 = vld1q_f32(input + row * ir_step + 4);
            float32x4_t d2 = vld1q_f32(input + row * ir_step + 8);
            float32x4_t d3 = vld1q_f32(input + row * ir_step + 12);
            int32x4_t qi0 = vcvtnq_s32_f32(vmulq_f32(d0, v_inv_s));
            int32x4_t qi1 = vcvtnq_s32_f32(vmulq_f32(d1, v_inv_s));
            int32x4_t qi2 = vcvtnq_s32_f32(vmulq_f32(d2, v_inv_s));
            int32x4_t qi3 = vcvtnq_s32_f32(vmulq_f32(d3, v_inv_s));
            qi0 = vaddq_s32(qi0, v_zp);
            qi1 = vaddq_s32(qi1, v_zp);
            qi2 = vaddq_s32(qi2, v_zp);
            qi3 = vaddq_s32(qi3, v_zp);
            int16x8_t s_lo = vcombine_s16(vqmovn_s32(qi0), vqmovn_s32(qi1));
            int16x8_t s_hi = vcombine_s16(vqmovn_s32(qi2), vqmovn_s32(qi3));
            return vcombine_s8(vqmovn_s16(s_lo), vqmovn_s16(s_hi));
        };

        int8x16_t r0 = quant_row16(0);
        int8x16_t r1 = quant_row16(1);
        int8x16_t r2 = quant_row16(2);
        int8x16_t r3 = quant_row16(3);

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

}  // namespace nnops::backend::cpu::aarch64