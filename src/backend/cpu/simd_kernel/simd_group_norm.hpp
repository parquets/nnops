#pragma once
/// @file simd_group_norm.hpp
/// @brief SIMD kernel functions for group normalization.
///
/// Provides two primitives used by the GroupNorm SIMD kernel:
///   1. reduce_sum_sq — SIMD reduction of sum + sum_sq over a contiguous row
///   2. apply_row     — SIMD normalization of a contiguous row
///
/// These operate at the innermost-row (W dimension) level, called iteratively
/// over all channel-rows within a normalization group.
///
/// The reduction uses 4-wide multi-accumulators for ILP, then falls back to
/// 2-wide → 1-wide → scalar tail. Both f32 (v_f32x8) and f16 (v_f16x8) are
/// supported through the typed SIMD API.

#include "nnops/detail/simd/simd.hpp"

#include <cmath>
#include <utility>

namespace nnops::kernel {

using namespace simd;

/// SIMD sum + sum-of-squares reduction over n contiguous elements.
///
/// Returns {sum, sum_sq} as float regardless of T (fp16 values are promoted).
template <typename T>
inline std::pair<float, float> reduce_sum_sq(const T* x, int64_t n)
{
    constexpr int L = simd_lane_for<T>;
    int64_t i = 0;
    float sum = 0.0f;
    float sum_sq = 0.0f;

    // ---- Stage 1: 4-wide SIMD reduction (ILP ×4) ----
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

        // ---- Stage 2: 2-wide SIMD ----
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

        // ---- Stage 3: 1-wide SIMD ----
        for (; i + L <= n; i += L) {
            auto v = v_load(x + i);
            v_sum = v_add(v_sum, v);
            v_sum_sq = v_fmadd(v, v, v_sum_sq);
        }

        sum = v_reduce_sum(v_sum);
        sum_sq = v_reduce_sum(v_sum_sq);
    }

    // ---- Stage 4: scalar tail ----
    for (; i < n; ++i) {
        float xv = s_load(&x[i]);
        sum += xv;
        sum_sq += xv * xv;
    }

    return {sum, sum_sq};
}

/// SIMD normalize: apply `(x - mean) * inv_std * scale_val + bias_val`
/// to n contiguous elements.
///
/// @param x         Input pointer
/// @param y         Output pointer
/// @param n         Number of elements
/// @param mean      Group mean (broadcast scalar)
/// @param inv_std   1/sqrt(var + epsilon) (broadcast scalar)
/// @param scale_val Per-channel scale (scalar for this row)
/// @param bias_val  Per-channel bias (scalar for this row, 0 if no bias)
/// @param add_to    Accumulate into output instead of overwriting
template <typename T>
inline void apply_row(const T* x, T* y, int64_t n,
                       float mean, float inv_std,
                       float scale_val, float bias_val,
                       bool add_to)
{
    constexpr int L = simd_lane_for<T>;
    int64_t i = 0;

    const auto v_mean    = v_set1(x, mean);
    const auto v_inv_std = v_set1(x, inv_std);
    const auto v_scale   = v_set1(x, scale_val);
    const auto v_bias    = v_set1(x, bias_val);

    for (; i + L <= n; i += L) {
        auto xv = v_load(x + i);
        // y = (x - mean) * inv_std * scale + bias
        auto rv = v_fmadd(v_mul(v_sub(xv, v_mean), v_inv_std), v_scale, v_bias);
        v_store_add(y + i, rv, add_to);
    }
    for (; i < n; ++i) {
        float xv = s_load(&x[i]);
        float rv = (xv - mean) * inv_std * scale_val + bias_val;
        s_store_add(&y[i], rv, add_to);
    }
}

}  // namespace nnops::kernel
