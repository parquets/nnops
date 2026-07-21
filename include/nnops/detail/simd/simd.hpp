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

/// @brief Convenience: number of elements in each vector register.
namespace nnops {
namespace simd {
inline constexpr int simd_len_f32x4 = 4;
inline constexpr int simd_len_f32x8 = 8;
inline constexpr int simd_len_f16x8 = 8;
} // namespace simd
} // namespace nnops
