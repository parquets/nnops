#pragma once
/// @file simd.hpp
/// @brief Main entry point for the nnops SIMD abstraction layer.
///
/// Provides type-safe wrappers around x86 SSE/AVX2 and ARM NEON intrinsics,
/// inspired by OpenCV's Universal Intrinsics (intrin.hpp) and onnxruntime's MLAS.
///
/// ## Architecture
///
///   simd.hpp  (this file)
///   ├── cpu_features.hpp   — runtime CPU detection (CpuFeatures singleton)
///   ├── vec_f32x4.hpp      — 128-bit float32 vector (SSE / NEON / scalar)
///   ├── vec_f32x8.hpp      — 256-bit float32 vector (AVX2 / emulated / scalar)
///   └── vec_f16x8.hpp      — 128-bit float16 vector (NEON FP16 / scalar)
///   └── arch/
///       ├── scalar.hpp     — scalar C++ fallback (f32 + f16)
///       ├── x86/
///       │   ├── sse.hpp    — SSE4.1 backend
///       │   └── avx2.hpp   — AVX2+FMA backend
///       └── arm/
///           └── neon.hpp   — ARM NEON backend (f32 + f16)
///
/// ## Usage
///
/// ```cpp
/// #include "nnops/detail/simd/simd.hpp"
/// using namespace nnops::simd;
///
/// // Compile-time optimal (SSE or NEON depending on target):
/// void relu_128(const float* in, float* out, int64_t n) {
///     v_f32x4 zero = v_zero_f32x4();
///     for (int64_t i = 0; i + 4 <= n; i += 4) {
///         v_f32x4 v = v_load_f32x4(in + i);
///         v_store(out + i, v_max(v, zero));
///     }
/// }
///
/// // Runtime-dispatch for AVX2 (compile with /arch:AVX2):
/// if (cpu_has_avx2()) {
///     v_f32x8 zero = v_zero_f32x8();
///     for (int64_t i = 0; i + 8 <= n; i += 8) {
///         v_f32x8 v = v_load_f32x8(in + i);
///         v_store(out + i, v_max(v, zero));
///     }
/// }
/// ```
///
/// Reference pattern from OpenCV (modules/core/include/opencv2/core/hal/intrin.hpp):
///   - v_float32x4, v_float32x8: platform-abstracted vector structs
///   - v_load, v_store, v_add, v_mul, v_fma: platform-abstracted operations
/// Reference pattern from onnxruntime (mlas/lib/mlasi.h):
///   - MLAS_FLOAT32X4, MLAS_INT32X4: cross-platform type aliases
///   - platform.cpp: runtime dispatch cascade (SSE→AVX→AVX2→AVX512)

#include "nnops/detail/simd/cpu_features.hpp"
#include "nnops/detail/simd/vec_f32x4.hpp"
#include "nnops/detail/simd/vec_f32x8.hpp"
#include "nnops/detail/simd/vec_f16x8.hpp"
#include "nnops/detail/simd/vec_i8x16.hpp"
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

/// @brief Recommended default SIMD lane count for f32 kernels.
/// On all hardware SIMD backends this is 8 (AVX2 __m256 native;
/// NEON two float32x4_t emulated; RISC-V V vsetivli LMUL=2).
inline constexpr int simd_default_lane_f32 = 8;

/// @brief Recommended default SIMD lane count for f16 kernels.
/// On x86 F16C+AVX2 this is 8 (cvt→compute→cvt); on ARM NEON native fp16
/// (float16x8_t) this is 8; on RISC-V V with LMUL=2 this is 8.
inline constexpr int simd_default_lane_f16 = 8;

/// @brief Compile-time SIMD lane count for a given data type T.
/// Usage: simd_lane_for<T> — yields 8 for both float and half on all backends.
template <typename T>
inline constexpr int simd_lane_for = std::is_same_v<T, float>
    ? simd_default_lane_f32
    : simd_default_lane_f16;
} // namespace simd
} // namespace nnops

// ============================================================
// Generic load / store / broadcast / zero helpers
//
// Overloaded on pointer type so kernel code is type-generic:
//   float*   → v_f32x8   (v_load_f32x8 / v_store / v_set1_f32x8 / v_zero_f32x8)
//   half*    → v_f16x8   (v_load_f16x8 / v_store / v_set1_f16x8 / v_zero_f16x8)
//   int8_t*  → v_i8x16   (v_load_i8x16 / v_store / v_set1_i8x16 / v_zero_i8x16)
//   uint8_t* → v_u8x16   (v_load_u8x16 / v_store / v_set1_u8x16 / v_zero_u8x16)
//
// v_store is overloaded by each backend for all pointer types;
// we add a half* overload that reinterpret_cast's to uint16_t* internally.
// s_load / s_store handle scalar access with fp16↔fp32 conversion.
//
// Usage:
//   using namespace nnops::simd;
//   auto vacc = v_set1(ptr, init_val);
//   auto vin  = v_load(ptr);
//   auto vz   = v_zero(ptr);
//   float s   = s_load(ptr);
//   s_store(ptr, s);
//   v_store(ptr, vacc);
// ============================================================

namespace nnops {
namespace simd {

using ::nnops::backend::cpu::half;

// Generic vector load
inline v_f32x8 v_load(const float* p) { return v_load_f32x8(p); }
inline v_f16x8 v_load(const half* p) {
    return v_load_f16x8(reinterpret_cast<const uint16_t*>(p));
}
inline v_i8x16 v_load(const int8_t* p) { return v_load_i8x16(p); }
inline v_u8x16 v_load(const uint8_t* p) { return v_load_u8x16(p); }

// Generic vector broadcast (first arg for overload resolution)
inline v_f32x8 v_set1(const float*, float s) { return v_set1_f32x8(s); }
inline v_f16x8 v_set1(const half*, float s) { return v_set1_f16x8(s); }
inline v_i8x16 v_set1(const int8_t*, int8_t s) { return v_set1_i8x16(s); }
inline v_u8x16 v_set1(const uint8_t*, uint8_t s) { return v_set1_u8x16(s); }

// Generic zero vector (first arg for overload resolution)
inline v_f32x8 v_zero(const float*) { return v_zero_f32x8(); }
inline v_f16x8 v_zero(const half*) { return v_zero_f16x8(); }
inline v_i8x16 v_zero(const int8_t*) { return v_zero_i8x16(); }
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

// ============================================================
// Store-with-accumulate helpers — SIMD/scalar equivalents of
// *ptr += val, combining load+add+store into one call.
// Eliminates the v_add(v_load(ptr), val) / v_store(ptr, ...)
// if/else boilerplate that was duplicated across 6+ operators.
// ============================================================
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

// ============================================================
// v_pow — composed from existing SIMD primitives:
//   pow(a, b) = exp(b * log(a))
// Works for all vector types (v_f32x4, v_f32x8, v_f16x8) and
// all backends — no arch-specific intrinsics needed.
// ============================================================

template <typename V>
inline V v_pow(V a, V b) {
    return v_exp(v_mul(b, v_log(a)));
}

// ============================================================
// Deinterleave (stride-2 gather) — generic overloads
//
// Three API styles (方案 A / B / C), type-dispatched on pointer type:
//   float* → v_f32x8 / v_f32x8x2_t
//   half*  → v_f16x8 / v_f16x8x2_t  (via uint16_t*)
//
// v_deinterleave loads 2×N elements and returns {even, odd}.
// v_load_even / v_load_odd return a single vector with every other element.
// v_load_stride2_even / v_load_stride2_odd are explicit-named aliases.
// ============================================================

// Pair types (v_f32x8x2_t, v_f16x8x2_t) are re-exported from the arch
// namespaces via vec_f32x8.hpp / vec_f16x8.hpp using-declarations.

// ---- 方案 A: v_deinterleave (full pair result) --------------------------

inline v_f32x8x2_t v_deinterleave(const float* p) {
    return v_deinterleave_f32x8(p);
}
inline v_f16x8x2_t v_deinterleave(const half* p) {
    return v_deinterleave_f16x8(reinterpret_cast<const uint16_t*>(p));
}

// ---- 方案 B: v_load_even / v_load_odd -----------------------------------

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

// ---- 方案 C: v_load_stride2_even / v_load_stride2_odd -------------------

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

// ============================================================
// int8 / uint8 → float32 conversion (8 lanes → v_f32x8)
// ============================================================

#if defined(NNOPS_ARCH_X86_64)
inline v_f32x8 v_cvt_i8_to_f32(const int8_t* p) {
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
inline v_f32x8 v_cvt_i8_to_f32(const int8_t* p) {
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
inline v_f32x8 v_cvt_i8_to_f32(const int8_t* p) {
    float buf[8];
    for (int i = 0; i < 8; ++i) buf[i] = static_cast<float>(p[i]);
    return v_load(buf);
}
inline v_f32x8 v_cvt_u8_to_f32(const uint8_t* p) {
    float buf[8];
    for (int i = 0; i < 8; ++i) buf[i] = static_cast<float>(p[i]);
    return v_load(buf);
}
#endif

// int8/uint8 → f16: direct per-element conversion (8 lanes)
inline v_f16x8 v_cvt_i8_to_f16(const int8_t* p) {
    half buf[8];
    for (int i = 0; i < 8; ++i) buf[i] = ::nnops::backend::cpu::float_to_half(static_cast<float>(p[i]));
    return v_load(buf);
}
inline v_f16x8 v_cvt_u8_to_f16(const uint8_t* p) {
    half buf[8];
    for (int i = 0; i < 8; ++i) buf[i] = ::nnops::backend::cpu::float_to_half(static_cast<float>(p[i]));
    return v_load(buf);
}

} // namespace simd
} // namespace nnops
