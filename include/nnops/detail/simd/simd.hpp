#pragma once
/// @file simd.hpp
/// @brief Entry point for the nnops SIMD abstraction layer.
///
/// Type-safe wrappers around x86 SSE/AVX2 and ARM NEON intrinsics, in the
/// spirit of OpenCV's Universal Intrinsics and onnxruntime's MLAS. Include
/// this file to get every vector type plus the generic helpers below.
///
/// Layout:
///   vec_f32x4.hpp  128-bit f32 (SSE / NEON / scalar)
///   vec_f32x8.hpp  256-bit f32 (AVX2 / NEON-emulated / scalar)
///   vec_f16x8.hpp  128-bit f16 (NEON FP16 / x86 F16C / scalar)
///   vec_s8x16.hpp  128-bit s8  (SSE / NEON / scalar)
///   vec_u8x16.hpp  128-bit u8  (SSE / NEON / scalar)
///   arch/{scalar, x86/sse, x86/avx2, arm/neon}.hpp   backend implementations
///   cpu_features.hpp  runtime CpuFeatures detection
///
/// The generic helpers are overloaded on the element pointer type so kernel
/// code stays type-generic: float* → v_f32x8, half* → v_f16x8, int8_t* →
/// v_s8x16, uint8_t* → v_u8x16. v_store is the backends' overload set plus a
/// half* overload here that reinterprets to uint16_t*; s_load / s_store are the
/// scalar fp16↔fp32 counterparts.

#include "nnops/detail/simd/cpu_features.hpp"
#include "nnops/detail/simd/vec_f32x4.hpp"
#include "nnops/detail/simd/vec_f32x8.hpp"
#include "nnops/detail/simd/vec_f16x8.hpp"
#include "nnops/detail/simd/vec_s8x16.hpp"
#include "nnops/detail/simd/vec_u8x16.hpp"
#include "nnops/detail/half.hpp"

#include <type_traits>

/// @brief Number of lanes (elements) in each vector register type.
namespace nnops {
namespace simd {
inline constexpr int simd_lane_f32x4 = 4;
inline constexpr int simd_lane_f32x8 = 8;
inline constexpr int simd_lane_f16x8 = 8;
inline constexpr int simd_lane_i8x16 = 16;
inline constexpr int simd_lane_u8x16 = 16;

/// @brief Default f32 lane count (8): AVX2 native, NEON as two float32x4_t.
inline constexpr int simd_default_lane_f32 = 8;

/// @brief Default f16 lane count (8): x86 F16C cvt→compute→cvt, NEON native float16x8_t.
inline constexpr int simd_default_lane_f16 = 8;

/// @brief Lane count for element type T (8 for both float and half).
template <typename T>
inline constexpr int simd_lane_for = std::is_same_v<T, float>
    ? simd_default_lane_f32
    : simd_default_lane_f16;
} // namespace simd
} // namespace nnops

// ---- Generic load / store / broadcast / zero helpers ----

namespace nnops {
namespace simd {

using ::nnops::backend::cpu::half;

inline v_f32x8 v_load(const float* p) { return v_load_f32x8(p); }
inline v_f16x8 v_load(const half* p) {
    return v_load_f16x8(reinterpret_cast<const uint16_t*>(p));
}
inline v_s8x16 v_load(const int8_t* p) { return v_load_s8x16(p); }
inline v_u8x16 v_load(const uint8_t* p) { return v_load_u8x16(p); }

// Generic broadcast (the first pointer arg only selects the overload)
inline v_f32x8 v_set1(const float*, float s) { return v_set1_f32x8(s); }
inline v_f16x8 v_set1(const half*, float s) { return v_set1_f16x8(s); }
inline v_s8x16 v_set1(const int8_t*, int8_t s) { return v_set1_s8x16(s); }
inline v_u8x16 v_set1(const uint8_t*, uint8_t s) { return v_set1_u8x16(s); }

// Generic zero (same overload-selection trick)
inline v_f32x8 v_zero(const float*) { return v_zero_f32x8(); }
inline v_f16x8 v_zero(const half*) { return v_zero_f16x8(); }
inline v_s8x16 v_zero(const int8_t*) { return v_zero_s8x16(); }
inline v_u8x16 v_zero(const uint8_t*) { return v_zero_u8x16(); }

// Generic vector store — backend provides float*; add half* overload
inline void v_store(half* p, const v_f16x8& v) {
    v_store(reinterpret_cast<uint16_t*>(p), v);
}

// Scalar load — always returns float
inline float s_load(const float* p) { return *p; }
inline float s_load(const half* p) {
    return ::nnops::backend::cpu::half_to_float(*p);
}

// Scalar store — converts float to native element type
inline void s_store(float* p, float v) { *p = v; }
inline void s_store(half* p, float v) {
    *p = ::nnops::backend::cpu::float_to_half(v);
}

// ---- Store-with-accumulate helpers (*ptr += val) ----
template <typename Ptr, typename Vec>
inline void v_store_add(Ptr ptr, const Vec& val, bool add_to) {
    if (add_to) {
        v_store(ptr, v_add(v_load(ptr), val));
    } else {
        v_store(ptr, val);
    }
}
template <typename Ptr>
inline void s_store_add(Ptr ptr, float val, bool add_to) {
    if (add_to) {
        s_store(ptr, s_load(ptr) + val);
    } else {
        s_store(ptr, val);
    }
}

template <typename V>
inline V v_pow(V a, V b) {
    return v_exp(v_mul(b, v_log(a)));
}

// ---- Deinterleave (stride-2 gather); v_deinterleave returns {even, odd} ----

inline v_f32x8x2_t v_deinterleave(const float* p) {
    return v_deinterleave_f32x8(p);
}
inline v_f16x8x2_t v_deinterleave(const half* p) {
    return v_deinterleave_f16x8(reinterpret_cast<const uint16_t*>(p));
}

inline v_f32x8 v_load_even(const float* p) {
    return v_load_even_f32x8(p);
}
inline v_f32x8 v_load_odd(const float* p) {
    return v_load_odd_f32x8(p);
}
inline v_f16x8 v_load_even(const half* p) {
    return v_load_even_f16x8(reinterpret_cast<const uint16_t*>(p));
}
inline v_f16x8 v_load_odd(const half* p) {
    return v_load_odd_f16x8(reinterpret_cast<const uint16_t*>(p));
}

inline v_f32x8 v_load_stride2_even(const float* p) {
    return v_load_stride2_even_f32x8(p);
}
inline v_f32x8 v_load_stride2_odd(const float* p) {
    return v_load_stride2_odd_f32x8(p);
}
inline v_f16x8 v_load_stride2_even(const half* p) {
    return v_load_stride2_even_f16x8(reinterpret_cast<const uint16_t*>(p));
}
inline v_f16x8 v_load_stride2_odd(const half* p) {
    return v_load_stride2_odd_f16x8(reinterpret_cast<const uint16_t*>(p));
}

// ---- int8 / uint8 → f32 (8 lanes) ----

#if defined(NNOPS_ARCH_X86_64)
inline v_f32x8 v_cvt_s8_to_f32(const int8_t* p) {
    __m128i i8  = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
    __m128i i16 = _mm_cvtepi8_epi16(i8);
    __m128i i32_lo = _mm_cvtepi16_epi32(i16);
    __m128i i32_hi = _mm_cvtepi16_epi32(_mm_unpackhi_epi64(i16, i16));
    __m256 f32 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_cvtepi32_ps(i32_lo)),
        _mm_cvtepi32_ps(i32_hi), 1);
    return v_f32x8(f32);
}
inline v_f32x8 v_cvt_u8_to_f32(const uint8_t* p) {
    __m128i u8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
    __m128i z  = _mm_setzero_si128();
    __m128i u16     = _mm_unpacklo_epi8(u8, z);
    __m128i u32_lo  = _mm_unpacklo_epi16(u16, z);
    __m128i u32_hi  = _mm_unpackhi_epi16(u16, z);
    __m256 f32 = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_cvtepi32_ps(u32_lo)),
        _mm_cvtepi32_ps(u32_hi), 1);
    return v_f32x8(f32);
}

#elif defined(NNOPS_ARCH_AARCH64)
inline v_f32x8 v_cvt_s8_to_f32(const int8_t* p) {
    int8x8_t   i8     = vld1_s8(p);
    int16x8_t  i16    = vmovl_s8(i8);
    int32x4_t  i32_lo = vmovl_s16(vget_low_s16(i16));
    int32x4_t  i32_hi = vmovl_s16(vget_high_s16(i16));
    return v_f32x8(vcvtq_f32_s32(i32_lo), vcvtq_f32_s32(i32_hi));
}
inline v_f32x8 v_cvt_u8_to_f32(const uint8_t* p) {
    uint8x8_t  u8     = vld1_u8(p);
    uint16x8_t u16    = vmovl_u8(u8);
    uint32x4_t u32_lo = vmovl_u16(vget_low_u16(u16));
    uint32x4_t u32_hi = vmovl_u16(vget_high_u16(u16));
    return v_f32x8(vcvtq_f32_u32(u32_lo), vcvtq_f32_u32(u32_hi));
}

#else
inline v_f32x8 v_cvt_s8_to_f32(const int8_t* p) {
    float buf[8];
    for (int i = 0; i < 8; ++i) {
        buf[i] = static_cast<float>(p[i]);
    }
    return v_load(buf);
}
inline v_f32x8 v_cvt_u8_to_f32(const uint8_t* p) {
    float buf[8];
    for (int i = 0; i < 8; ++i) {
        buf[i] = static_cast<float>(p[i]);
    }
    return v_load(buf);
}
#endif

// ---- f16 → f32 load; returns this build's own v_f32x8 (vec_f16x8.hpp) ----

inline v_f32x8 v_cvt_f16_to_f32(const half* p) {
    return v_cvt_f16_to_f32(v_load_f16x8(reinterpret_cast<const uint16_t*>(p)));
}

// int8/uint8 → f16: direct per-element conversion (8 lanes)
inline v_f16x8 v_cvt_s8_to_f16(const int8_t* p) {
    half buf[8];
    for (int i = 0; i < 8; ++i) {
        buf[i] = ::nnops::backend::cpu::float_to_half(static_cast<float>(p[i]));
    }
    return v_load(buf);
}
inline v_f16x8 v_cvt_u8_to_f16(const uint8_t* p) {
    half buf[8];
    for (int i = 0; i < 8; ++i) {
        buf[i] = ::nnops::backend::cpu::float_to_half(static_cast<float>(p[i]));
    }
    return v_load(buf);
}

} // namespace simd
} // namespace nnops
