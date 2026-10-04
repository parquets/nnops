#pragma once
/// @file simd_unary.hpp
/// @brief Tiled element-wise unary SIMD kernels for m×n sub-matrices.
///
/// Each function operates on an m-row × n-column tile with pitch (stride
/// in elements) for both input and output. The API is type-generic via
/// v_load/v_store_add/s_load/s_store_add dispatch — instantiate with
/// `float` or `half` (f16).
///
/// Only lane=8 SIMD types are used (v_f32x8 / v_f16x8), matching the
/// simd_lane_for<T> = 8 convention used throughout the CPU backend.
///
/// Usage:
///   #include "simd_kernel/simd_unary.hpp"
///   using namespace nnops::kernel;
///   exp(in_ptr, out_ptr, m, n, in_pitch, out_pitch, add_to);
///
/// All kernels support the `add_to` flag: when true, the result is
/// accumulated onto the output (out[i] += fn(in[i])); when false, the
/// output is overwritten (out[i] = fn(in[i])).

#include "nnops/detail/simd/simd.hpp"
#include "tiled_map.hpp"

#include <cmath>

namespace nnops::kernel {

using namespace simd;

// ============================================================
// Row-processing helpers — thin arity adapters over tiled_map
// ============================================================

/// Row-by-row SIMD + scalar tail loop for unary ops (Nin = 1).
template <typename T, typename SimdK, typename ScalarK>
inline void tiled_unary_simd(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to,
    SimdK&& simd_kernel,
    ScalarK&& scalar_kernel)
{
    tiled_map_simd<T, 1>({in}, {in_pitch}, out, out_pitch, m, n, add_to,
                         simd_kernel, scalar_kernel);
}

/// Row-by-row scalar-only loop for unary ops without SIMD intrinsics
/// (Erf, Round, Ceil, Floor, Sign).
template <typename T, typename ScalarK>
inline void tiled_unary_scalar(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to,
    ScalarK&& scalar_kernel)
{
    tiled_map_scalar<T, 1>({in}, {in_pitch}, out, out_pitch, m, n, add_to,
                           scalar_kernel);
}

// ============================================================
// Tiled SIMD kernels — one function per unary operation
// ============================================================

template <typename T>
inline void exp(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to = false)
{
    tiled_unary_simd(in, out, m, n, in_pitch, out_pitch, add_to,
        [](auto x) { return v_exp(x); },
        [](float v) { return std::exp(v); });
}

template <typename T>
inline void log(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to = false)
{
    tiled_unary_simd(in, out, m, n, in_pitch, out_pitch, add_to,
        [](auto x) { return v_log(x); },
        [](float v) { return std::log(v); });
}

template <typename T>
inline void sin(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to = false)
{
    tiled_unary_simd(in, out, m, n, in_pitch, out_pitch, add_to,
        [](auto x) { return v_sin(x); },
        [](float v) { return std::sin(v); });
}

template <typename T>
inline void cos(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to = false)
{
    tiled_unary_simd(in, out, m, n, in_pitch, out_pitch, add_to,
        [](auto x) { return v_cos(x); },
        [](float v) { return std::cos(v); });
}

template <typename T>
inline void tan(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to = false)
{
    tiled_unary_simd(in, out, m, n, in_pitch, out_pitch, add_to,
        [](auto x) { return v_tan(x); },
        [](float v) { return std::tan(v); });
}

template <typename T>
inline void tanh(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to = false)
{
    tiled_unary_simd(in, out, m, n, in_pitch, out_pitch, add_to,
        [](auto x) { return v_tanh(x); },
        [](float v) { return std::tanh(v); });
}

template <typename T>
inline void abs(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to = false)
{
    tiled_unary_simd(in, out, m, n, in_pitch, out_pitch, add_to,
        [](auto x) { return v_abs(x); },
        [](float v) { return v < 0.0f ? -v : v; });
}

template <typename T>
inline void neg(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to = false)
{
    tiled_unary_simd(in, out, m, n, in_pitch, out_pitch, add_to,
        [](auto x) { return v_neg(x); },
        [](float v) { return -v; });
}

template <typename T>
inline void sqrt(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to = false)
{
    tiled_unary_simd(in, out, m, n, in_pitch, out_pitch, add_to,
        [](auto x) { return v_sqrt(x); },
        [](float v) { return std::sqrt(v); });
}

template <typename T>
inline void recip(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to = false)
{
    tiled_unary_simd(in, out, m, n, in_pitch, out_pitch, add_to,
        [](auto x) { return v_rcp(x); },
        [](float v) { return 1.0f / v; });
}

// ============================================================
// Tiled scalar-only kernels — ops without SIMD intrinsics
// ============================================================

template <typename T>
inline void erf(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to = false)
{
    tiled_unary_scalar(in, out, m, n, in_pitch, out_pitch, add_to,
        [](float v) { return std::erf(v); });
}

template <typename T>
inline void round(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to = false)
{
    tiled_unary_scalar(in, out, m, n, in_pitch, out_pitch, add_to,
        [](float v) { return std::round(v); });
}

template <typename T>
inline void ceil(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to = false)
{
    tiled_unary_scalar(in, out, m, n, in_pitch, out_pitch, add_to,
        [](float v) { return std::ceil(v); });
}

template <typename T>
inline void floor(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to = false)
{
    tiled_unary_scalar(in, out, m, n, in_pitch, out_pitch, add_to,
        [](float v) { return std::floor(v); });
}

template <typename T>
inline void sign(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to = false)
{
    tiled_unary_scalar(in, out, m, n, in_pitch, out_pitch, add_to,
        [](float v) { return (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f); });
}

}  // namespace nnops::kernel
