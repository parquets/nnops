#pragma once
/// @file simd_rms_norm.hpp
/// @brief SIMD kernel functions for RMS normalization (fast path: contiguous tail).
///
/// Algorithm:
///   1. SIMD reduce sum_sq with multi-accumulator unrolling.
///   2. Scalar tail, compute rms = sqrt(sum_sq / N + eps), inv_rms = 1 / rms.
///   3. Normalize: y = x * inv_rms * scale.
///
/// Only lane=8 SIMD types (v_f32x8 / v_f16x8), matching simd_lane_for<T>.

#include "nnops/detail/simd/simd.hpp"

#include <cmath>

namespace nnops::kernel {

using namespace simd;

template <typename T>
inline void rms_norm_process_row(
    const T* x, T* y,
    const T* scale,
    int64_t n, float epsilon,
    bool scale_is_scalar, bool add_to)
{
    constexpr int L = simd_lane_for<T>;

    // Pass 1: 4-wide multi-accumulator SIMD reduction + scalar tail
    float sum_sq = 0.0f;
    int64_t i = 0;

    {
        auto v_sq0 = v_zero(x);
        auto v_sq1 = v_zero(x);
        auto v_sq2 = v_zero(x);
        auto v_sq3 = v_zero(x);

        for (; i + 4 * L <= n; i += 4 * L) {
            auto v0 = v_load(x + i);
            auto v1 = v_load(x + i + L);
            auto v2 = v_load(x + i + 2 * L);
            auto v3 = v_load(x + i + 3 * L);
            v_sq0 = v_fmadd(v0, v0, v_sq0);
            v_sq1 = v_fmadd(v1, v1, v_sq1);
            v_sq2 = v_fmadd(v2, v2, v_sq2);
            v_sq3 = v_fmadd(v3, v3, v_sq3);
        }
        auto v_sum_sq = v_add(v_add(v_sq0, v_sq1), v_add(v_sq2, v_sq3));

        auto v_sq4 = v_zero(x);
        auto v_sq5 = v_zero(x);
        for (; i + 2 * L <= n; i += 2 * L) {
            auto v4 = v_load(x + i);
            auto v5 = v_load(x + i + L);
            v_sq4 = v_fmadd(v4, v4, v_sq4);
            v_sq5 = v_fmadd(v5, v5, v_sq5);
        }
        v_sum_sq = v_add(v_sum_sq, v_add(v_sq4, v_sq5));

        for (; i + L <= n; i += L) {
            auto v = v_load(x + i);
            v_sum_sq = v_fmadd(v, v, v_sum_sq);
        }

        sum_sq = v_reduce_sum(v_sum_sq);
    }

    for (; i < n; ++i) {
        float xv = s_load(&x[i]);
        sum_sq += xv * xv;
    }

    const float rms = std::sqrt(sum_sq / static_cast<float>(n) + epsilon);
    const float inv_rms = 1.0f / rms;

    // Pass 2: SIMD normalize
    i = 0;
    const auto v_inv_rms = v_set1(x, inv_rms);

    for (; i + L <= n; i += L) {
        auto xv = v_load(x + i);
        auto vs = v_load(scale + (scale_is_scalar ? 0 : i));
        auto rv = v_mul(v_mul(xv, v_inv_rms), vs);
        v_store_add(y + i, rv, add_to);
    }
    for (; i < n; ++i) {
        int64_t s_idx = scale_is_scalar ? 0 : i;
        float xv = s_load(&x[i]);
        float s = s_load(&scale[s_idx]);
        float rv = xv * inv_rms * s;
        s_store_add(&y[i], rv, add_to);
    }
}

}  // namespace nnops::kernel
