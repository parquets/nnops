#pragma once
/// @file quant.hpp
/// @brief AArch64 NEON int8 / uint8 quantization and dequantization kernels.
///
/// Raw-intrinsic per-token (per-row) quantization for GEMM input preparation,
/// the NEON counterpart to `x86_64/quant.hpp`. Each row carries its own `scale`
/// and optional `zero_point`; `zero == nullptr` means all-zero.
///
///   - quantization:   f32 → s8/u8,  `dst = clamp(round(src * (1/scale) + zero), min, max)`
///   - dequantization: s8/u8 → f32,  `dst = (src - zero) * scale`
///
/// Both functions are templated on the integer type `T` (int8_t for s8, uint8_t
/// for u8); the clamp range is [-128, 127] for s8 and [0, 255] for u8. The
/// `zero` parameter makes the kernel asymmetric-capable; symmetric quantization
/// is simply the `zero == nullptr` (all-zero) case.
///
/// Unlike the x86_64 version there is no separate 32-wide block: NEON registers
/// are 128-bit (4 f32 / 16 i8), so a single 16-wide block is used for the whole
/// vector loop. Rounding is round-to-nearest-even (`vcvtnq_s32_f32` /
/// `std::nearbyintf`), matching `_mm_cvtps_epi32`. Narrowing saturates in two
/// stages (i32 → i16 → i8 / u8) via `vqmovn`/`vqmovun`, which is correct because
/// the int8 range is a subset of the int16 range.
///
/// Reference (x86_64 semantics): nn_compute/src/cpu/kernel/quant/x86_64/quant_impl.hpp

#include <algorithm>
#include <arm_neon.h>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "backend/cpu/common/restrict.hpp"
#include "nnops/detail/half.hpp"

namespace nnops::backend::cpu::aarch64 {

using nnops::backend::cpu::half;
using nnops::backend::cpu::half_to_float;
using nnops::backend::cpu::float_to_half;

// =========================================================================
//  Block helpers — register-level quantize / dequantize primitives
// =========================================================================

/// Quantize 16 floats (4×float32x4_t) to 16 int8 (1×int8x16_t), natural order.
///
/// Each lane computes `zero + d * inv_scale` via fused multiply-add, rounds to
/// nearest even (`vcvtnq_s32_f32`), then narrows i32 → i16 → i8 with saturating
/// `vqmovn`. The u8 path uses `vqmovun_s16` for the final signed→unsigned
/// narrowing to [0, 255].
template <bool Q_U8>
inline int8x16_t quant_block16_f32_i8(float32x4_t d0, float32x4_t d1, float32x4_t d2, float32x4_t d3,
                                      float32x4_t inv_scale, float32x4_t zero) noexcept {
    d0 = vfmaq_f32(zero, d0, inv_scale);
    d1 = vfmaq_f32(zero, d1, inv_scale);
    d2 = vfmaq_f32(zero, d2, inv_scale);
    d3 = vfmaq_f32(zero, d3, inv_scale);
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

/// Widen 8 int8 (int8x8_t) to 8 int32 (2×int32x4_t).
///
/// The u8 path reinterprets the bytes as unsigned before widening, so values
/// ≥ 128 (which read as negative int8) recover their true unsigned magnitude.
template <bool Q_U8>
inline void widen8_i8_to_i32(int8x8_t v, int32x4_t& lo, int32x4_t& hi) noexcept {
    if constexpr (Q_U8) {
        const uint16x8_t s = vmovl_u8(vreinterpret_u8_s8(v));
        lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s)));
        hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s)));
    } else {
        const int16x8_t s = vmovl_s8(v);
        lo = vmovl_s16(vget_low_s16(s));
        hi = vmovl_s16(vget_high_s16(s));
    }
}

/// Dequantize 16 int8 (1×int8x16_t) to 16 floats (4×float32x4_t).
///
/// Uses FMA: `i_f32 * scale + neg_zero_scale` where `neg_zero_scale = -zero * scale`
/// is precomputed once per row. This replaces the two-instruction `sub + mul` with
/// a single fused multiply-add, saving one arithmetic instruction per 4-wide lane.
template <bool Q_U8>
inline void dequant_block16_i8_f32(int8x16_t v, float32x4_t scale, float32x4_t neg_zero_scale,
                                   float32x4_t& res0, float32x4_t& res1,
                                   float32x4_t& res2, float32x4_t& res3) noexcept {
    int32x4_t i0, i1, i2, i3;
    widen8_i8_to_i32<Q_U8>(vget_low_s8(v), i0, i1);
    widen8_i8_to_i32<Q_U8>(vget_high_s8(v), i2, i3);
    res0 = vfmaq_f32(neg_zero_scale, vcvtq_f32_s32(i0), scale);
    res1 = vfmaq_f32(neg_zero_scale, vcvtq_f32_s32(i1), scale);
    res2 = vfmaq_f32(neg_zero_scale, vcvtq_f32_s32(i2), scale);
    res3 = vfmaq_f32(neg_zero_scale, vcvtq_f32_s32(i3), scale);
}

// =========================================================================
//  Top-level per-token quantize / dequantize (s8 and u8)
// =========================================================================

/// Quantize an M×N f32 matrix to integer `T`, one (scale, zero_point) per row.
///
/// `T` is int8_t (s8, clamp [-128, 127]) or uint8_t (u8, clamp [0, 255]).
/// `scale` has M entries; `zero` has M entries or is nullptr (all-zero). `dst`
/// is written with row stride `dr_step` elements, `src` read with `sr_step`
/// elements. SIMD and scalar tails round to nearest-even (`vcvtnq_s32_f32` /
/// `std::nearbyintf`) and saturate to the type's range.
///
/// The inner loop is unrolled 2× (32 elements per iteration) to reduce loop
/// overhead on medium-to-large N.
template <typename T>
inline void quantization(int M, int N,
                         T* NNOPS_RESTRICT dst, int dr_step,
                         const float* NNOPS_RESTRICT src, int sr_step,
                         const float* NNOPS_RESTRICT scale,
                         const float* NNOPS_RESTRICT zero) noexcept {
    static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>,
                  "quantization: T must be int8_t or uint8_t");
    constexpr bool Q_U8 = std::is_same_v<T, uint8_t>;
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    for (int m = 0; m < M; ++m) {
        const float inv_scale = 1.0f / scale[m];
        const float zero_val = (zero == nullptr) ? 0.0f : zero[m];

        const float32x4_t v_inv_scale = vdupq_n_f32(inv_scale);
        const float32x4_t v_zero = vdupq_n_f32(zero_val);

        T* dst_ptr = dst + m * dr_step;
        const float* src_ptr = src + m * sr_step;

        int n = 0;
        // 32-wide unrolled path: two 16-wide blocks per iteration
        for (; n + 32 <= N; n += 32) {
            const float32x4_t d0 = vld1q_f32(src_ptr + n + 0);
            const float32x4_t d1 = vld1q_f32(src_ptr + n + 4);
            const float32x4_t d2 = vld1q_f32(src_ptr + n + 8);
            const float32x4_t d3 = vld1q_f32(src_ptr + n + 12);
            const float32x4_t d4 = vld1q_f32(src_ptr + n + 16);
            const float32x4_t d5 = vld1q_f32(src_ptr + n + 20);
            const float32x4_t d6 = vld1q_f32(src_ptr + n + 24);
            const float32x4_t d7 = vld1q_f32(src_ptr + n + 28);
            const int8x16_t q0 = quant_block16_f32_i8<Q_U8>(d0, d1, d2, d3, v_inv_scale, v_zero);
            const int8x16_t q1 = quant_block16_f32_i8<Q_U8>(d4, d5, d6, d7, v_inv_scale, v_zero);
            vst1q_s8(reinterpret_cast<int8_t*>(dst_ptr + n + 0), q0);
            vst1q_s8(reinterpret_cast<int8_t*>(dst_ptr + n + 16), q1);
        }
        // 16-wide path for the remaining ≥16 tail
        for (; n + 16 <= N; n += 16) {
            const float32x4_t d0 = vld1q_f32(src_ptr + n + 0);
            const float32x4_t d1 = vld1q_f32(src_ptr + n + 4);
            const float32x4_t d2 = vld1q_f32(src_ptr + n + 8);
            const float32x4_t d3 = vld1q_f32(src_ptr + n + 12);
            const int8x16_t q = quant_block16_f32_i8<Q_U8>(d0, d1, d2, d3, v_inv_scale, v_zero);
            vst1q_s8(reinterpret_cast<int8_t*>(dst_ptr + n), q);
        }
        for (; n < N; ++n) {
            const float q = src_ptr[n] * inv_scale + zero_val;
            int32_t qi = static_cast<int32_t>(std::nearbyintf(q));
            qi = std::min(std::max(qi, qmin), qmax);
            dst_ptr[n] = static_cast<T>(qi);
        }
    }
}

// =========================================================================
//  Top-level per-token quantize with half (f16) input
// =========================================================================

/// Quantize an M×N f16 matrix to integer `T`, one (scale, zero_point) per row.
///
/// The half input is widened to f32 with the NEON fp16→fp32 conversion
/// (`vcvt_f32_f16` / `vcvt_high_f32_f16`) and then run through the same
/// f32→int8 arithmetic as the f32 `quantization<T>` overload. Rounding and
/// clamping are identical (round-to-nearest-even, saturating to `T`'s range).
/// `T` is int8_t (s8) or uint8_t (u8).
///
/// The inner loop is unrolled 2× (32 elements per iteration) to reduce loop
/// overhead on medium-to-large N.
template <typename T>
inline void quantization(int M, int N,
                         T* NNOPS_RESTRICT dst, int dr_step,
                         const half* NNOPS_RESTRICT src, int sr_step,
                         const float* NNOPS_RESTRICT scale,
                         const float* NNOPS_RESTRICT zero) noexcept {
    static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>,
                  "quantization: T must be int8_t or uint8_t");
    constexpr bool Q_U8 = std::is_same_v<T, uint8_t>;
    constexpr int32_t qmin = Q_U8 ? 0 : -128;
    constexpr int32_t qmax = Q_U8 ? 255 : 127;

    for (int m = 0; m < M; ++m) {
        const float inv_scale = 1.0f / scale[m];
        const float zero_val = (zero == nullptr) ? 0.0f : zero[m];

        const float32x4_t v_inv_scale = vdupq_n_f32(inv_scale);
        const float32x4_t v_zero = vdupq_n_f32(zero_val);

        T* dst_ptr = dst + m * dr_step;
        const half* src_ptr = src + m * sr_step;

        int n = 0;
        // 32-wide unrolled path: load 32 halfs, convert to 32 floats, quantize to 32 int8
        for (; n + 32 <= N; n += 32) {
            const float16x8_t h0 = vld1q_f16(reinterpret_cast<const float16_t*>(src_ptr + n + 0));
            const float16x8_t h1 = vld1q_f16(reinterpret_cast<const float16_t*>(src_ptr + n + 8));
            const float16x8_t h2 = vld1q_f16(reinterpret_cast<const float16_t*>(src_ptr + n + 16));
            const float16x8_t h3 = vld1q_f16(reinterpret_cast<const float16_t*>(src_ptr + n + 24));
            const float32x4_t d0 = vcvt_f32_f16(vget_low_f16(h0));
            const float32x4_t d1 = vcvt_high_f32_f16(h0);
            const float32x4_t d2 = vcvt_f32_f16(vget_low_f16(h1));
            const float32x4_t d3 = vcvt_high_f32_f16(h1);
            const float32x4_t d4 = vcvt_f32_f16(vget_low_f16(h2));
            const float32x4_t d5 = vcvt_high_f32_f16(h2);
            const float32x4_t d6 = vcvt_f32_f16(vget_low_f16(h3));
            const float32x4_t d7 = vcvt_high_f32_f16(h3);
            const int8x16_t q0 = quant_block16_f32_i8<Q_U8>(d0, d1, d2, d3, v_inv_scale, v_zero);
            const int8x16_t q1 = quant_block16_f32_i8<Q_U8>(d4, d5, d6, d7, v_inv_scale, v_zero);
            vst1q_s8(reinterpret_cast<int8_t*>(dst_ptr + n + 0), q0);
            vst1q_s8(reinterpret_cast<int8_t*>(dst_ptr + n + 16), q1);
        }
        for (; n + 16 <= N; n += 16) {
            const float16x8_t h0 = vld1q_f16(reinterpret_cast<const float16_t*>(src_ptr + n + 0));
            const float16x8_t h1 = vld1q_f16(reinterpret_cast<const float16_t*>(src_ptr + n + 8));
            const float32x4_t d0 = vcvt_f32_f16(vget_low_f16(h0));
            const float32x4_t d1 = vcvt_high_f32_f16(h0);
            const float32x4_t d2 = vcvt_f32_f16(vget_low_f16(h1));
            const float32x4_t d3 = vcvt_high_f32_f16(h1);
            const int8x16_t q = quant_block16_f32_i8<Q_U8>(d0, d1, d2, d3, v_inv_scale, v_zero);
            vst1q_s8(reinterpret_cast<int8_t*>(dst_ptr + n), q);
        }
        for (; n < N; ++n) {
            const float q = half_to_float(src_ptr[n]) * inv_scale + zero_val;
            int32_t qi = static_cast<int32_t>(std::nearbyintf(q));
            qi = std::min(std::max(qi, qmin), qmax);
            dst_ptr[n] = static_cast<T>(qi);
        }
    }
}

// =========================================================================
//  NCHWC8 8-wide primitives (8 channels = 8 bytes = one 64-bit group)
//
// These operate on a single channel-packed lane group (8 int8/uint8 values) —
// the granularity used by the quantized depthwise-conv / pooling SIMD kernels.
// =========================================================================

/// Requantize 8 floats to 8 int8/uint8 with a broadcast (per-tensor) scale and
/// zero_point: `dst[i] = clamp(round(src[i] * inv_scale + zero), qmin, qmax)`.
/// Rounding is round-to-nearest-even (`vcvtnq_s32_f32`), matching the per-token
/// `quantization<T>` path and the x86_64 `requantize_8`. Used for output
/// requantization of the NCHWC8 quantized depthwise-conv / pooling kernels.
template <typename T>
inline void requantize_8(T* NNOPS_RESTRICT dst, const float* NNOPS_RESTRICT src,
                         float inv_scale, float zero) noexcept {
    static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>,
                  "requantize_8: T must be int8_t or uint8_t");
    constexpr bool Q_U8 = std::is_same_v<T, uint8_t>;
    const float32x4_t v_inv = vdupq_n_f32(inv_scale);
    const float32x4_t v_zero = vdupq_n_f32(zero);
    const float32x4_t d0 = vfmaq_f32(v_zero, vld1q_f32(src + 0), v_inv);
    const float32x4_t d1 = vfmaq_f32(v_zero, vld1q_f32(src + 4), v_inv);
    const int32x4_t i0 = vcvtnq_s32_f32(d0);
    const int32x4_t i1 = vcvtnq_s32_f32(d1);
    const int16x8_t i16 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
    int8x8_t i8;
    if constexpr (Q_U8) {
        i8 = vreinterpret_s8_u8(vqmovun_s16(i16));
    } else {
        i8 = vqmovn_s16(i16);
    }
    vst1_s8(reinterpret_cast<int8_t*>(dst), i8);
}

/// Per-lane maximum of two 8-wide int8/uint8 vectors (raw integer max, no
/// dequantization). Used by the MaxPooling fast path, which is valid when input
/// and output share the same scale and zero_point.
template <typename T>
inline void max_8(T* NNOPS_RESTRICT dst, const T* NNOPS_RESTRICT a,
                  const T* NNOPS_RESTRICT b) noexcept {
    static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>,
                  "max_8: T must be int8_t or uint8_t");
    constexpr bool Q_U8 = std::is_same_v<T, uint8_t>;
    const int8x8_t va = vld1_s8(reinterpret_cast<const int8_t*>(a));
    const int8x8_t vb = vld1_s8(reinterpret_cast<const int8_t*>(b));
    int8x8_t vmax;
    if constexpr (Q_U8) {
        vmax = vreinterpret_s8_u8(vmax_u8(vreinterpret_u8_s8(va),
                                          vreinterpret_u8_s8(vb)));
    } else {
        vmax = vmax_s8(va, vb);
    }
    vst1_s8(reinterpret_cast<int8_t*>(dst), vmax);
}

/// Dequantize an M×N integer matrix of type `T` to f32, one (scale, zero_point)
/// per row.
///
/// `T` is int8_t (s8) or uint8_t (u8). `scale` and `zero` each have M entries
/// (or `zero` is nullptr for all-zero).
///
/// The inner loop is unrolled 2× (32 elements per iteration) and uses a single
/// FMA per 4-wide lane (`neg_zero_scale + i_f32 * scale`) instead of the
/// two-instruction `sub + mul` sequence.
template <typename T>
inline void dequantization(int M, int N,
                           float* NNOPS_RESTRICT dst, int dr_step,
                           const T* NNOPS_RESTRICT src, int sr_step,
                           const float* NNOPS_RESTRICT scale,
                           const float* NNOPS_RESTRICT zero) noexcept {
    static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>,
                  "dequantization: T must be int8_t or uint8_t");
    constexpr bool Q_U8 = std::is_same_v<T, uint8_t>;

    for (int m = 0; m < M; ++m) {
        const float scale_val = scale[m];
        const float zero_val = (zero == nullptr) ? 0.0f : zero[m];
        // Precompute neg_zero_scale = -zero * scale so the inner loop can use a
        // single FMA: fma(neg_zero_scale, i_f32, scale) = i_f32*scale - zero*scale
        const float neg_zero_scale = -zero_val * scale_val;

        const float32x4_t v_scale = vdupq_n_f32(scale_val);
        const float32x4_t v_neg_zero_scale = vdupq_n_f32(neg_zero_scale);

        float* dst_ptr = dst + m * dr_step;
        const T* src_ptr = src + m * sr_step;

        int n = 0;
        // 32-wide unrolled path: two 16-wide blocks per iteration
        for (; n + 32 <= N; n += 32) {
            const int8x16_t v0 = vld1q_s8(reinterpret_cast<const int8_t*>(src_ptr + n + 0));
            const int8x16_t v1 = vld1q_s8(reinterpret_cast<const int8_t*>(src_ptr + n + 16));
            float32x4_t r0, r1, r2, r3, r4, r5, r6, r7;
            dequant_block16_i8_f32<Q_U8>(v0, v_scale, v_neg_zero_scale, r0, r1, r2, r3);
            dequant_block16_i8_f32<Q_U8>(v1, v_scale, v_neg_zero_scale, r4, r5, r6, r7);
            vst1q_f32(dst_ptr + n + 0, r0);
            vst1q_f32(dst_ptr + n + 4, r1);
            vst1q_f32(dst_ptr + n + 8, r2);
            vst1q_f32(dst_ptr + n + 12, r3);
            vst1q_f32(dst_ptr + n + 16, r4);
            vst1q_f32(dst_ptr + n + 20, r5);
            vst1q_f32(dst_ptr + n + 24, r6);
            vst1q_f32(dst_ptr + n + 28, r7);
        }
        for (; n + 16 <= N; n += 16) {
            const int8x16_t v = vld1q_s8(reinterpret_cast<const int8_t*>(src_ptr + n));
            float32x4_t res0, res1, res2, res3;
            dequant_block16_i8_f32<Q_U8>(v, v_scale, v_neg_zero_scale, res0, res1, res2, res3);
            vst1q_f32(dst_ptr + n + 0, res0);
            vst1q_f32(dst_ptr + n + 4, res1);
            vst1q_f32(dst_ptr + n + 8, res2);
            vst1q_f32(dst_ptr + n + 12, res3);
        }
        for (; n < N; ++n) {
            dst_ptr[n] = (static_cast<float>(src_ptr[n]) - zero_val) * scale_val;
        }
    }
}

// =========================================================================
//  Top-level per-token dequantize with half (f16) output
// =========================================================================

/// Dequantize an M×N integer matrix of type `T` to f16, one (scale, zero_point)
/// per row.
///
/// The integer input is widened and dequantized to f32 with the same arithmetic
/// as the f32 `dequantization<T>`, then narrowed to f16 with the NEON fp32→fp16
/// conversion (`vcvt_f16_f32`). `T` is int8_t (s8) or uint8_t (u8).
///
/// The inner loop is unrolled 2× (32 elements per iteration) and uses a single
/// FMA per 4-wide lane.
template <typename T>
inline void dequantization(int M, int N,
                           half* NNOPS_RESTRICT dst, int dr_step,
                           const T* NNOPS_RESTRICT src, int sr_step,
                           const float* NNOPS_RESTRICT scale,
                           const float* NNOPS_RESTRICT zero) noexcept {
    static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>,
                  "dequantization: T must be int8_t or uint8_t");
    constexpr bool Q_U8 = std::is_same_v<T, uint8_t>;

    for (int m = 0; m < M; ++m) {
        const float scale_val = scale[m];
        const float zero_val = (zero == nullptr) ? 0.0f : zero[m];
        const float neg_zero_scale = -zero_val * scale_val;

        const float32x4_t v_scale = vdupq_n_f32(scale_val);
        const float32x4_t v_neg_zero_scale = vdupq_n_f32(neg_zero_scale);

        half* dst_ptr = dst + m * dr_step;
        const T* src_ptr = src + m * sr_step;

        int n = 0;
        // 32-wide unrolled path
        for (; n + 32 <= N; n += 32) {
            const int8x16_t v0 = vld1q_s8(reinterpret_cast<const int8_t*>(src_ptr + n + 0));
            const int8x16_t v1 = vld1q_s8(reinterpret_cast<const int8_t*>(src_ptr + n + 16));
            float32x4_t r0, r1, r2, r3, r4, r5, r6, r7;
            dequant_block16_i8_f32<Q_U8>(v0, v_scale, v_neg_zero_scale, r0, r1, r2, r3);
            dequant_block16_i8_f32<Q_U8>(v1, v_scale, v_neg_zero_scale, r4, r5, r6, r7);
            const float16x8_t h0 = vcombine_f16(vcvt_f16_f32(r0), vcvt_f16_f32(r1));
            const float16x8_t h1 = vcombine_f16(vcvt_f16_f32(r2), vcvt_f16_f32(r3));
            const float16x8_t h2 = vcombine_f16(vcvt_f16_f32(r4), vcvt_f16_f32(r5));
            const float16x8_t h3 = vcombine_f16(vcvt_f16_f32(r6), vcvt_f16_f32(r7));
            vst1q_f16(reinterpret_cast<float16_t*>(dst_ptr + n + 0), h0);
            vst1q_f16(reinterpret_cast<float16_t*>(dst_ptr + n + 8), h1);
            vst1q_f16(reinterpret_cast<float16_t*>(dst_ptr + n + 16), h2);
            vst1q_f16(reinterpret_cast<float16_t*>(dst_ptr + n + 24), h3);
        }
        for (; n + 16 <= N; n += 16) {
            const int8x16_t v = vld1q_s8(reinterpret_cast<const int8_t*>(src_ptr + n));
            float32x4_t res0, res1, res2, res3;
            dequant_block16_i8_f32<Q_U8>(v, v_scale, v_neg_zero_scale, res0, res1, res2, res3);
            const float16x8_t h0 = vcombine_f16(vcvt_f16_f32(res0), vcvt_f16_f32(res1));
            const float16x8_t h1 = vcombine_f16(vcvt_f16_f32(res2), vcvt_f16_f32(res3));
            vst1q_f16(reinterpret_cast<float16_t*>(dst_ptr + n + 0), h0);
            vst1q_f16(reinterpret_cast<float16_t*>(dst_ptr + n + 8), h1);
        }
        for (; n < N; ++n) {
            dst_ptr[n] = float_to_half((static_cast<float>(src_ptr[n]) - zero_val) * scale_val);
        }
    }
}

}  // namespace nnops::backend::cpu::aarch64