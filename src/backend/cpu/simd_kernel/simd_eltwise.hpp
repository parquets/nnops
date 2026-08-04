#pragma once
/// @file simd_eltwise.hpp
/// @brief Tiled element-wise binary SIMD kernels for m×n sub-matrices.
///
/// Each function operates on an m-row × n-column tile with pitch (stride
/// in elements) for both inputs and output. The API is type-generic via
/// v_load/v_store_add/s_load/s_store_add dispatch — instantiate with
/// `float` or `half` (f16).
///
/// Only lane=8 SIMD types are used (v_f32x8 / v_f16x8), matching the
/// simd_lane_for<T> = 8 convention used throughout the CPU backend.
///
/// Usage:
///   #include "nnops/backend/cpu/simd_kernel/simd_eltwise.hpp"
///   using namespace nnops::kernel;
///   add(a_ptr, b_ptr, out_ptr, m, n, a_pitch, b_pitch, out_pitch, add_to);

#include "nnops/detail/simd/simd.hpp"

#include <cmath>

namespace nnops::kernel {

using namespace simd;

// ============================================================
// Internal row-processing helpers
// ============================================================

/// Row-by-row SIMD + scalar tail loop for binary ops.
template <typename T, typename SimdK, typename ScalarK>
inline void tiled_eltwise_simd(
    const T* a, const T* b, T* out,
    int64_t m, int64_t n,
    int64_t a_pitch, int64_t b_pitch, int64_t out_pitch,
    bool add_to,
    SimdK&& simd_kernel,
    ScalarK&& scalar_kernel)
{
    constexpr int L = simd_lane_for<T>;  // 8 for both f32 and f16
    for (int64_t r = 0; r < m; ++r) {
        const T* a_row = a + r * a_pitch;
        const T* b_row = b + r * b_pitch;
        T* out_row = out + r * out_pitch;
        int64_t i = 0;
        for (; i + L <= n; i += L) {
            v_store_add(out_row + i,
                simd_kernel(v_load(a_row + i), v_load(b_row + i)), add_to);
        }
        for (; i < n; ++i) {
            s_store_add(&out_row[i],
                scalar_kernel(s_load(&a_row[i]), s_load(&b_row[i])), add_to);
        }
    }
}

/// Row-by-row scalar-only loop for binary ops without SIMD intrinsics.
template <typename T, typename ScalarK>
inline void tiled_eltwise_scalar(
    const T* a, const T* b, T* out,
    int64_t m, int64_t n,
    int64_t a_pitch, int64_t b_pitch, int64_t out_pitch,
    bool add_to,
    ScalarK&& scalar_kernel)
{
    for (int64_t r = 0; r < m; ++r) {
        const T* a_row = a + r * a_pitch;
        const T* b_row = b + r * b_pitch;
        T* out_row = out + r * out_pitch;
        for (int64_t i = 0; i < n; ++i) {
            s_store_add(&out_row[i],
                scalar_kernel(s_load(&a_row[i]), s_load(&b_row[i])), add_to);
        }
    }
}

// ============================================================
// Tiled SIMD kernels — one function per binary operation
// ============================================================

template <typename T>
inline void add(
    const T* a, const T* b, T* out,
    int64_t m, int64_t n,
    int64_t a_pitch, int64_t b_pitch, int64_t out_pitch,
    bool add_to = false)
{
    tiled_eltwise_simd(a, b, out, m, n, a_pitch, b_pitch, out_pitch, add_to,
        [](auto va, auto vb) { return v_add(va, vb); },
        [](float fa, float fb) { return fa + fb; });
}

template <typename T>
inline void sub(
    const T* a, const T* b, T* out,
    int64_t m, int64_t n,
    int64_t a_pitch, int64_t b_pitch, int64_t out_pitch,
    bool add_to = false)
{
    tiled_eltwise_simd(a, b, out, m, n, a_pitch, b_pitch, out_pitch, add_to,
        [](auto va, auto vb) { return v_sub(va, vb); },
        [](float fa, float fb) { return fa - fb; });
}

template <typename T>
inline void mul(
    const T* a, const T* b, T* out,
    int64_t m, int64_t n,
    int64_t a_pitch, int64_t b_pitch, int64_t out_pitch,
    bool add_to = false)
{
    tiled_eltwise_simd(a, b, out, m, n, a_pitch, b_pitch, out_pitch, add_to,
        [](auto va, auto vb) { return v_mul(va, vb); },
        [](float fa, float fb) { return fa * fb; });
}

template <typename T>
inline void div(
    const T* a, const T* b, T* out,
    int64_t m, int64_t n,
    int64_t a_pitch, int64_t b_pitch, int64_t out_pitch,
    bool add_to = false)
{
    tiled_eltwise_simd(a, b, out, m, n, a_pitch, b_pitch, out_pitch, add_to,
        [](auto va, auto vb) { return v_div(va, vb); },
        [](float fa, float fb) { return fa / fb; });
}

template <typename T>
inline void min(
    const T* a, const T* b, T* out,
    int64_t m, int64_t n,
    int64_t a_pitch, int64_t b_pitch, int64_t out_pitch,
    bool add_to = false)
{
    tiled_eltwise_simd(a, b, out, m, n, a_pitch, b_pitch, out_pitch, add_to,
        [](auto va, auto vb) { return v_min(va, vb); },
        [](float fa, float fb) { return fa < fb ? fa : fb; });
}

template <typename T>
inline void max(
    const T* a, const T* b, T* out,
    int64_t m, int64_t n,
    int64_t a_pitch, int64_t b_pitch, int64_t out_pitch,
    bool add_to = false)
{
    tiled_eltwise_simd(a, b, out, m, n, a_pitch, b_pitch, out_pitch, add_to,
        [](auto va, auto vb) { return v_max(va, vb); },
        [](float fa, float fb) { return fa > fb ? fa : fb; });
}

// ============================================================
// Tiled SIMD pow — v_pow(a, b) = exp(b * log(a))
// ============================================================

template <typename T>
inline void pow(
    const T* a, const T* b, T* out,
    int64_t m, int64_t n,
    int64_t a_pitch, int64_t b_pitch, int64_t out_pitch,
    bool add_to = false)
{
    tiled_eltwise_simd(a, b, out, m, n, a_pitch, b_pitch, out_pitch, add_to,
        [](auto va, auto vb) { return v_pow(va, vb); },
        [](float fa, float fb) { return std::pow(fa, fb); });
}

}  // namespace nnops::kernel
