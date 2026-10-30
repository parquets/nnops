#pragma once
/// @file quant.hpp
/// @brief x86_64 AVX2 + FMA int8 / uint8 quantization and dequantization kernels.
///
/// Per-token (per-row) quantization for GEMM input preparation over raw
/// row-major matrices (the TensorView SIMD path is `simd_kernel/simd_quant.hpp`).
/// Each row has its own `scale` and optional `zero_point`; `zero == nullptr`
/// means all-zero.
///
///   - quantization:   f32/f16 → s8/u8, `dst = clamp(round(src * (1/scale) + zero), min, max)`
///   - dequantization: s8/u8 → f32/f16, `dst = (src - zero) * scale`
///
/// `T` is int8_t (clamp [-128, 127]) or uint8_t ([0, 255]). Loops run a 32-wide
/// AVX2 path, a 16-wide path, then a scalar tail. Quantization clamps through
/// the saturating packs (the scalar tail through std::min/max); rounding is
/// round-to-nearest-even (`_mm_cvtps_epi32` / `std::nearbyintf`).
///
/// Lane-ordering quirks: the 16-wide block packs with 128-bit `_mm_packs_epi16`
/// because `_mm256_packs_epi16` packs per 128-bit lane and would interleave the
/// four 8-element groups; the 32-wide block packs with 256-bit ops and restores
/// order with a cross-lane dword permute (0,4,1,5,2,6,3,7). Dequantization folds
/// `i*scale - zero*scale` into one FMA per lane via a per-row `neg_zero_scale`.
/// The f16 overloads widen with F16C (`_mm256_cvtph_ps`), reuse the f32
/// arithmetic, and narrow with `_mm256_cvtps_ph`.
///
/// `requantize_8` (broadcast scale/zero) backs `requantize_store` in
/// simd_quant.hpp; `max_8` backs the MaxPooling fast path when scale and
/// zero_point match.
///
/// Reference: nn_compute/src/cpu/kernel/quant/x86_64/quant_impl.hpp

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <type_traits>

#include "backend/cpu/common/common_utils.hpp"
#include "nnops/detail/half.hpp"

namespace nnops::backend::cpu::x86_64 {

using nnops::backend::cpu::half;
using nnops::backend::cpu::half_to_float;
using nnops::backend::cpu::float_to_half;

/// Quantize 16 floats (4×__m128) to 16 int8, natural order.
template <bool Q_U8>
inline __m128i quant_block16_f32_i8(__m128 d0, __m128 d1, __m128 d2, __m128 d3,
                                    __m128 inv_scale, __m128 zero) noexcept {
    d0 = _mm_fmadd_ps(d0, inv_scale, zero);
    d1 = _mm_fmadd_ps(d1, inv_scale, zero);
    d2 = _mm_fmadd_ps(d2, inv_scale, zero);
    d3 = _mm_fmadd_ps(d3, inv_scale, zero);
    const __m128i d0_i32 = _mm_cvtps_epi32(d0);
    const __m128i d1_i32 = _mm_cvtps_epi32(d1);
    const __m128i d2_i32 = _mm_cvtps_epi32(d2);
    const __m128i d3_i32 = _mm_cvtps_epi32(d3);
    const __m128i d01_i16 = _mm_packs_epi32(d0_i32, d1_i32);
    const __m128i d23_i16 = _mm_packs_epi32(d2_i32, d3_i32);
    if constexpr (Q_U8) {
        return _mm_packus_epi16(d01_i16, d23_i16);
    } else {
        return _mm_packs_epi16(d01_i16, d23_i16);
    }
}

/// Quantize 32 floats (4×__m256) to 32 int8, natural order.
template <bool Q_U8>
inline __m256i quant_block32_f32_i8(__m256 d0, __m256 d1, __m256 d2, __m256 d3,
                                    __m256 inv_scale, __m256 zero) noexcept {
    d0 = _mm256_fmadd_ps(d0, inv_scale, zero);
    d1 = _mm256_fmadd_ps(d1, inv_scale, zero);
    d2 = _mm256_fmadd_ps(d2, inv_scale, zero);
    d3 = _mm256_fmadd_ps(d3, inv_scale, zero);
    const __m256i d0_i32 = _mm256_cvtps_epi32(d0);
    const __m256i d1_i32 = _mm256_cvtps_epi32(d1);
    const __m256i d2_i32 = _mm256_cvtps_epi32(d2);
    const __m256i d3_i32 = _mm256_cvtps_epi32(d3);
    const __m256i d01_i16 = _mm256_packs_epi32(d0_i32, d1_i32);
    const __m256i d23_i16 = _mm256_packs_epi32(d2_i32, d3_i32);
    __m256i packed;
    if constexpr (Q_U8) {
        packed = _mm256_packus_epi16(d01_i16, d23_i16);
    } else {
        packed = _mm256_packs_epi16(d01_i16, d23_i16);
    }
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    return _mm256_permutevar8x32_epi32(packed, perm);
}

/// Dequantize 16 int8 (1×__m128i) to 16 floats (2×__m256) by FMA.
template <bool Q_U8>
inline void dequant_block16_i8_f32(__m128i v, __m256 scale, __m256 neg_zero_scale,
                                   __m256& res0, __m256& res1) noexcept {
    __m256i d0_i32;
    if constexpr (Q_U8) {
        d0_i32 = _mm256_cvtepu8_epi32(v);
    } else {
        d0_i32 = _mm256_cvtepi8_epi32(v);
    }
    res0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d0_i32), scale, neg_zero_scale);

    const __m128i v_hi = _mm_srli_si128(v, 8);
    __m256i d1_i32;
    if constexpr (Q_U8) {
        d1_i32 = _mm256_cvtepu8_epi32(v_hi);
    } else {
        d1_i32 = _mm256_cvtepi8_epi32(v_hi);
    }
    res1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d1_i32), scale, neg_zero_scale);
}

/// Quantize an M×N f32 matrix to `T` with one (scale, zero_point) per row.
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

        const __m256 v_inv_scale = _mm256_set1_ps(inv_scale);
        const __m256 v_zero = _mm256_set1_ps(zero_val);
        const __m128 v_inv_scale_lo = _mm256_castps256_ps128(v_inv_scale);
        const __m128 v_zero_lo = _mm256_castps256_ps128(v_zero);

        T* dst_ptr = dst + m * dr_step;
        const float* src_ptr = src + m * sr_step;

        int n = 0;
        for (; n + 32 <= N; n += 32) {
            const __m256 d0 = _mm256_loadu_ps(src_ptr + n + 0);
            const __m256 d1 = _mm256_loadu_ps(src_ptr + n + 8);
            const __m256 d2 = _mm256_loadu_ps(src_ptr + n + 16);
            const __m256 d3 = _mm256_loadu_ps(src_ptr + n + 24);
            const __m256i q = quant_block32_f32_i8<Q_U8>(d0, d1, d2, d3, v_inv_scale, v_zero);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_ptr + n), q);
        }
        for (; n + 16 <= N; n += 16) {
            const __m128 d0 = _mm_loadu_ps(src_ptr + n + 0);
            const __m128 d1 = _mm_loadu_ps(src_ptr + n + 4);
            const __m128 d2 = _mm_loadu_ps(src_ptr + n + 8);
            const __m128 d3 = _mm_loadu_ps(src_ptr + n + 12);
            const __m128i q = quant_block16_f32_i8<Q_U8>(d0, d1, d2, d3, v_inv_scale_lo, v_zero_lo);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_ptr + n), q);
        }
        for (; n < N; ++n) {
            const float q = src_ptr[n] * inv_scale + zero_val;
            int32_t qi = static_cast<int32_t>(std::nearbyintf(q));
            qi = std::min(std::max(qi, qmin), qmax);
            dst_ptr[n] = static_cast<T>(qi);
        }
    }
}

/// Quantize an M×N f16 matrix to `T`; widen with F16C, then the f32 path.
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

        const __m256 v_inv_scale = _mm256_set1_ps(inv_scale);
        const __m256 v_zero = _mm256_set1_ps(zero_val);
        const __m128 v_inv_scale_lo = _mm256_castps256_ps128(v_inv_scale);
        const __m128 v_zero_lo = _mm256_castps256_ps128(v_zero);

        T* dst_ptr = dst + m * dr_step;
        const half* src_ptr = src + m * sr_step;

        int n = 0;
        for (; n + 32 <= N; n += 32) {
            const __m128i h0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr + n + 0));
            const __m128i h1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr + n + 8));
            const __m128i h2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr + n + 16));
            const __m128i h3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr + n + 24));
            const __m256 d0 = _mm256_cvtph_ps(h0);
            const __m256 d1 = _mm256_cvtph_ps(h1);
            const __m256 d2 = _mm256_cvtph_ps(h2);
            const __m256 d3 = _mm256_cvtph_ps(h3);
            const __m256i q = quant_block32_f32_i8<Q_U8>(d0, d1, d2, d3, v_inv_scale, v_zero);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_ptr + n), q);
        }
        for (; n + 16 <= N; n += 16) {
            const __m128i h0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr + n + 0));
            const __m128i h1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr + n + 8));
            const __m128 d0 = _mm_cvtph_ps(h0);
            const __m128 d1 = _mm_cvtph_ps(_mm_srli_si128(h0, 8));
            const __m128 d2 = _mm_cvtph_ps(h1);
            const __m128 d3 = _mm_cvtph_ps(_mm_srli_si128(h1, 8));
            const __m128i q = quant_block16_f32_i8<Q_U8>(d0, d1, d2, d3, v_inv_scale_lo, v_zero_lo);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_ptr + n), q);
        }
        for (; n < N; ++n) {
            const float q = half_to_float(src_ptr[n]) * inv_scale + zero_val;
            int32_t qi = static_cast<int32_t>(std::nearbyintf(q));
            qi = std::min(std::max(qi, qmin), qmax);
            dst_ptr[n] = static_cast<T>(qi);
        }
    }
}

/// Requantize 8 floats to `T` with a broadcast (per-tensor) scale/zero_point.
template <typename T>
inline void requantize_8(T* NNOPS_RESTRICT dst, const float* NNOPS_RESTRICT src,
                         float inv_scale, float zero) noexcept {
    static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>,
                  "requantize_8: T must be int8_t or uint8_t");
    constexpr bool Q_U8 = std::is_same_v<T, uint8_t>;
    const __m128 v_inv = _mm_set1_ps(inv_scale);
    const __m128 v_zero = _mm_set1_ps(zero);
    const __m128 d0 = _mm_fmadd_ps(_mm_loadu_ps(src + 0), v_inv, v_zero);
    const __m128 d1 = _mm_fmadd_ps(_mm_loadu_ps(src + 4), v_inv, v_zero);
    const __m128i i0 = _mm_cvtps_epi32(d0);
    const __m128i i1 = _mm_cvtps_epi32(d1);
    const __m128i i16 = _mm_packs_epi32(i0, i1);
    __m128i i8;
    if constexpr (Q_U8) {
        i8 = _mm_packus_epi16(i16, i16);
    } else {
        i8 = _mm_packs_epi16(i16, i16);
    }
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dst), i8);
}

/// Per-lane integer max of two 8-wide int8/uint8 vectors — no dequantization.
template <typename T>
inline void max_8(T* NNOPS_RESTRICT dst, const T* NNOPS_RESTRICT a,
                  const T* NNOPS_RESTRICT b) noexcept {
    static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>,
                  "max_8: T must be int8_t or uint8_t");
    constexpr bool Q_U8 = std::is_same_v<T, uint8_t>;
    const __m128i va = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(a));
    const __m128i vb = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b));
    const __m128i vmax = Q_U8 ? _mm_max_epu8(va, vb) : _mm_max_epi8(va, vb);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dst), vmax);
}

/// Dequantize an M×N integer matrix `T` to f32, one (scale, zero_point) per row.
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
        const float neg_zero_scale = -zero_val * scale_val;

        const __m256 v_scale = _mm256_set1_ps(scale_val);
        const __m256 v_neg_zero_scale = _mm256_set1_ps(neg_zero_scale);

        float* dst_ptr = dst + m * dr_step;
        const T* src_ptr = src + m * sr_step;

        int n = 0;
        for (; n + 32 <= N; n += 32) {
            const __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr + n + 0));
            const __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr + n + 16));
            __m256 r0, r1, r2, r3;
            dequant_block16_i8_f32<Q_U8>(v0, v_scale, v_neg_zero_scale, r0, r1);
            dequant_block16_i8_f32<Q_U8>(v1, v_scale, v_neg_zero_scale, r2, r3);
            _mm256_storeu_ps(dst_ptr + n + 0, r0);
            _mm256_storeu_ps(dst_ptr + n + 8, r1);
            _mm256_storeu_ps(dst_ptr + n + 16, r2);
            _mm256_storeu_ps(dst_ptr + n + 24, r3);
        }
        for (; n + 16 <= N; n += 16) {
            const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr + n));
            __m256 res0, res1;
            dequant_block16_i8_f32<Q_U8>(v, v_scale, v_neg_zero_scale, res0, res1);
            _mm256_storeu_ps(dst_ptr + n + 0, res0);
            _mm256_storeu_ps(dst_ptr + n + 8, res1);
        }
        for (; n < N; ++n) {
            dst_ptr[n] = (static_cast<float>(src_ptr[n]) - zero_val) * scale_val;
        }
    }
}

/// Dequantize an M×N integer matrix `T` to f16; narrow with F16C `_mm256_cvtps_ph`.
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

        const __m256 v_scale = _mm256_set1_ps(scale_val);
        const __m256 v_neg_zero_scale = _mm256_set1_ps(neg_zero_scale);

        half* dst_ptr = dst + m * dr_step;
        const T* src_ptr = src + m * sr_step;

        int n = 0;
        for (; n + 32 <= N; n += 32) {
            const __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr + n + 0));
            const __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr + n + 16));
            __m256 r0, r1, r2, r3;
            dequant_block16_i8_f32<Q_U8>(v0, v_scale, v_neg_zero_scale, r0, r1);
            dequant_block16_i8_f32<Q_U8>(v1, v_scale, v_neg_zero_scale, r2, r3);
            const __m128i h0 = _mm256_cvtps_ph(r0, _MM_FROUND_TO_NEAREST_INT);
            const __m128i h1 = _mm256_cvtps_ph(r1, _MM_FROUND_TO_NEAREST_INT);
            const __m128i h2 = _mm256_cvtps_ph(r2, _MM_FROUND_TO_NEAREST_INT);
            const __m128i h3 = _mm256_cvtps_ph(r3, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_ptr + n + 0), h0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_ptr + n + 8), h1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_ptr + n + 16), h2);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_ptr + n + 24), h3);
        }
        for (; n + 16 <= N; n += 16) {
            const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr + n));
            __m256 res0, res1;
            dequant_block16_i8_f32<Q_U8>(v, v_scale, v_neg_zero_scale, res0, res1);
            const __m128i h0 = _mm256_cvtps_ph(res0, _MM_FROUND_TO_NEAREST_INT);
            const __m128i h1 = _mm256_cvtps_ph(res1, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_ptr + n + 0), h0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_ptr + n + 8), h1);
        }
        for (; n < N; ++n) {
            dst_ptr[n] = float_to_half((static_cast<float>(src_ptr[n]) - zero_val) * scale_val);
        }
    }
}

}  // namespace nnops::backend::cpu::x86_64