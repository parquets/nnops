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
#include "nnops/detail/half.hpp"

#include <type_traits>

/// @brief Number of lanes (elements) in each vector register type.
namespace nnops {
namespace simd {
inline constexpr int simd_lane_f32x4 = 4;
inline constexpr int simd_lane_f32x8 = 8;
inline constexpr int simd_lane_f16x8 = 8;

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
//   float*  → v_f32x8  (v_load_f32x8 / v_store / v_set1_f32x8 / v_zero_f32x8)
//   half*   → v_f16x8  (v_load_f16x8 / v_store / v_set1_f16x8 / v_zero_f16x8)
//
// v_store is provided by each backend (float* and uint16_t*); we add a
// half* overload that reinterpret_cast's to uint16_t* internally.
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

// Generic vector broadcast (dummy first arg for overload resolution)
inline v_f32x8 v_set1(const float*, float s) { return v_set1_f32x8(s); }
inline v_f16x8 v_set1(const half*, float s) { return v_set1_f16x8(s); }

// Generic zero vector (dummy first arg for overload resolution)
inline v_f32x8 v_zero(const float*) { return v_zero_f32x8(); }
inline v_f16x8 v_zero(const half*) { return v_zero_f16x8(); }

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

} // namespace simd
} // namespace nnops
