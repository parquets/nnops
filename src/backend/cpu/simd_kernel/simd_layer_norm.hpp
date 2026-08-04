#pragma once
/// @file simd_layer_norm.hpp
/// @brief SIMD kernel functions for layer normalization (fast path: contiguous tail).
///
/// Algorithm:
///   1. SIMD reduce sum/sum_sq over 4-wide multi-accumulator blocks + scalar tail.
///   2. Compute mean = sum/N, var = sum_sq/N - mean^2, inv_std = 1/sqrt(var+eps).
///   3. SIMD normalize: y = (x - mean) * inv_std * scale + bias + scalar tail.
///
/// Only lane=8 SIMD types (v_f32x8 / v_f16x8), matching simd_lane_for<T>.

#include "nnops/detail/simd/simd.hpp"

#include <cmath>

namespace nnops::kernel {

using namespace simd;

/// Process one row of layer normalization using SIMD fast path.
///
/// Performs: sum/sum_sq SIMD reduction → compute stats → SIMD normalize.
/// The row is contiguous (axis == rank-1, pack == 1).
template <typename T>
inline void layer_norm_process_row(
    const T* x, T* y,
    const T* scale, const T* bias,
    int64_t n, float epsilon,
    bool scale_is_scalar, bool has_bias, bool add_to)
{
    constexpr int L = simd_lane_for<T>;
    int64_t i = 0;

    // Pass 1 — 4-wide multi-accumulator SIMD reduction + scalar tail
    float sum = 0.0f;
    float sum_sq = 0.0f;

    {
        auto v_sum0 = v_zero(x);
        auto v_sum_sq0 = v_zero(x);
        auto v_sum1 = v_zero(x);
        auto v_sum_sq1 = v_zero(x);
        auto v_sum2 = v_zero(x);
        auto v_sum_sq2 = v_zero(x);
        auto v_sum3 = v_zero(x);
        auto v_sum_sq3 = v_zero(x);

        for (; i + 4 * L <= n; i += 4 * L) {
            auto v0 = v_load(x + i);
            auto v1 = v_load(x + i + L);
            auto v2 = v_load(x + i + 2 * L);
            auto v3 = v_load(x + i + 3 * L);
            v_sum0 = v_add(v_sum0, v0);
            v_sum_sq0 = v_fmadd(v0, v0, v_sum_sq0);
            v_sum1 = v_add(v_sum1, v1);
            v_sum_sq1 = v_fmadd(v1, v1, v_sum_sq1);
            v_sum2 = v_add(v_sum2, v2);
            v_sum_sq2 = v_fmadd(v2, v2, v_sum_sq2);
            v_sum3 = v_add(v_sum3, v3);
            v_sum_sq3 = v_fmadd(v3, v3, v_sum_sq3);
        }
        auto v_sum = v_add(v_add(v_sum0, v_sum1), v_add(v_sum2, v_sum3));
        auto v_sum_sq = v_add(v_add(v_sum_sq0, v_sum_sq1),
                              v_add(v_sum_sq2, v_sum_sq3));

        auto v_sum4 = v_zero(x);
        auto v_sum_sq4 = v_zero(x);
        auto v_sum5 = v_zero(x);
        auto v_sum_sq5 = v_zero(x);
        for (; i + 2 * L <= n; i += 2 * L) {
            auto v4 = v_load(x + i);
            auto v5 = v_load(x + i + L);
            v_sum4 = v_add(v_sum4, v4);
            v_sum_sq4 = v_fmadd(v4, v4, v_sum_sq4);
            v_sum5 = v_add(v_sum5, v5);
            v_sum_sq5 = v_fmadd(v5, v5, v_sum_sq5);
        }
        v_sum = v_add(v_sum, v_add(v_sum4, v_sum5));
        v_sum_sq = v_add(v_sum_sq, v_add(v_sum_sq4, v_sum_sq5));

        for (; i + L <= n; i += L) {
            auto v = v_load(x + i);
            v_sum = v_add(v_sum, v);
            v_sum_sq = v_fmadd(v, v, v_sum_sq);
        }

        sum = v_reduce_sum(v_sum);
        sum_sq = v_reduce_sum(v_sum_sq);
    }

    for (; i < n; ++i) {
        float xv = s_load(&x[i]);
        sum += xv;
        sum_sq += xv * xv;
    }

    const float inv_n = 1.0f / static_cast<float>(n);
    const float mean_val = sum * inv_n;
    float var_val = sum_sq * inv_n - mean_val * mean_val;
    if (var_val < 0.0f) { var_val = 0.0f; }
    const float inv_std = 1.0f / std::sqrt(var_val + epsilon);

    // Pass 2 — SIMD normalize
    i = 0;

    const auto v_mean    = v_set1(x, mean_val);
    const auto v_inv_std = v_set1(x, inv_std);
    const auto v_zero_b  = v_zero(x);

    for (; i + L <= n; i += L) {
        auto xv = v_load(x + i);
        auto vs = v_load(scale + (scale_is_scalar ? 0 : i));
        auto vb = (has_bias && bias)
            ? v_load(bias + (scale_is_scalar ? 0 : i))
            : v_zero_b;
        auto rv = v_fmadd(v_mul(v_sub(xv, v_mean), v_inv_std), vs, vb);
        v_store_add(y + i, rv, add_to);
    }
    for (; i < n; ++i) {
        int64_t s_idx = scale_is_scalar ? 0 : i;
        int64_t b_idx = scale_is_scalar ? 0 : i;
        float xv = s_load(&x[i]);
        float s = s_load(&scale[s_idx]);
        float b = (has_bias && bias) ? s_load(&bias[b_idx]) : 0.0f;
        float rv = (xv - mean_val) * inv_std * s + b;
        s_store_add(&y[i], rv, add_to);
    }
}

}  // namespace nnops::kernel
