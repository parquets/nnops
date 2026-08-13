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

namespace nnops::backend::cpu::aarch64 {

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
template <bool Q_U8>
inline void dequant_block16_i8_f32(int8x16_t v, float32x4_t scale, float32x4_t zero,
                                   float32x4_t& res0, float32x4_t& res1,
                                   float32x4_t& res2, float32x4_t& res3) noexcept {
    int32x4_t i0, i1, i2, i3;
    widen8_i8_to_i32<Q_U8>(vget_low_s8(v), i0, i1);
    widen8_i8_to_i32<Q_U8>(vget_high_s8(v), i2, i3);
    res0 = vmulq_f32(vsubq_f32(vcvtq_f32_s32(i0), zero), scale);
    res1 = vmulq_f32(vsubq_f32(vcvtq_f32_s32(i1), zero), scale);
    res2 = vmulq_f32(vsubq_f32(vcvtq_f32_s32(i2), zero), scale);
    res3 = vmulq_f32(vsubq_f32(vcvtq_f32_s32(i3), zero), scale);
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

/// Dequantize an M×N integer matrix of type `T` to f32, one (scale, zero_point)
/// per row.
///
/// `T` is int8_t (s8) or uint8_t (u8). `scale` and `zero` each have M entries
/// (or `zero` is nullptr for all-zero).
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

        const float32x4_t v_scale = vdupq_n_f32(scale_val);
        const float32x4_t v_zero = vdupq_n_f32(zero_val);

        float* dst_ptr = dst + m * dr_step;
        const T* src_ptr = src + m * sr_step;

        int n = 0;
        for (; n + 16 <= N; n += 16) {
            const int8x16_t v = vld1q_s8(reinterpret_cast<const int8_t*>(src_ptr + n));
            float32x4_t res0, res1, res2, res3;
            dequant_block16_i8_f32<Q_U8>(v, v_scale, v_zero, res0, res1, res2, res3);
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

}  // namespace nnops::backend::cpu::aarch64
