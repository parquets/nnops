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

#include <cmath>

namespace nnops::kernel {

using namespace simd;

// ============================================================
// Internal row-processing helpers
// ============================================================

/// Row-by-row SIMD + scalar tail loop. Used for ops with SIMD intrinsics.
template <typename T, typename SimdK, typename ScalarK>
inline void tiled_unary_simd(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to,
    SimdK&& simd_kernel,
    ScalarK&& scalar_kernel)
{
    constexpr int L = simd_lane_for<T>;  // 8 for both f32 and f16
    for (int64_t r = 0; r < m; ++r) {
        const T* in_row = in + r * in_pitch;
        T* out_row = out + r * out_pitch;
        int64_t i = 0;
        for (; i + L <= n; i += L) {
            v_store_add(out_row + i, simd_kernel(v_load(in_row + i)), add_to);
        }
        for (; i < n; ++i) {
            s_store_add(&out_row[i], scalar_kernel(s_load(&in_row[i])), add_to);
        }
    }
}

/// Row-by-row scalar-only loop. Used for ops without SIMD intrinsics
/// (Erf, Round, Ceil, Floor, Recip, Sign).
template <typename T, typename ScalarK>
inline void tiled_unary_scalar(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to,
    ScalarK&& scalar_kernel)
{
    for (int64_t r = 0; r < m; ++r) {
        const T* in_row = in + r * in_pitch;
        T* out_row = out + r * out_pitch;
        for (int64_t i = 0; i < n; ++i) {
            s_store_add(&out_row[i], scalar_kernel(s_load(&in_row[i])), add_to);
        }
    }
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
