#pragma once
/// @file simd_softmax.hpp
/// @brief SIMD kernel functions for softmax / log-softmax.
///
/// Three paths:
///   - Packed SIMD (axis == rank-1, pack > 1): per-lane SIMD reduction
///     within each physical row.
///   - Standard SIMD fast path (axis == rank-1, pack == 1): contiguous
///     tail, SIMD max/sum reduction.
///
/// Only lane=8 SIMD types (v_f32x8 / v_f16x8), matching simd_lane_for<T>.

#include "nnops/detail/simd/simd.hpp"

#include <cmath>
#include <cfloat>

namespace nnops::kernel::softmax {

using namespace simd;

// ============================================================
// Packed SIMD path: per-lane reduction within each physical row
//
// For NCHWC8 [N,C8,H,W] with axis=W, each of the `pack` C lanes
// is an independent softmax over D=W spatial positions.
// ============================================================

template <typename T>
inline void process_packed_row(
    const T* x_row, T* y_row,
    int64_t D, int64_t pack, bool log_softmax)
{
    // ---- Pass 1: per-lane max ----
    auto v_max_vec = v_set1(x_row, -std::numeric_limits<float>::infinity());
    for (int64_t w = 0; w < D; ++w)
        v_max_vec = v_max(v_max_vec, v_load(x_row + w * pack));

    // ---- Pass 2: per-lane sum of exp(x - max) ----
    auto v_sum_vec = v_zero(x_row);
    auto v_neg_max = v_neg(v_max_vec);
    for (int64_t w = 0; w < D; ++w) {
        auto v = v_add(v_load(x_row + w * pack), v_neg_max);
        v = v_exp(v);
        if (!log_softmax) { v_store(y_row + w * pack, v); }
        v_sum_vec = v_add(v_sum_vec, v);
    }

    // ---- Pass 3: per-lane normalize ----
    if (log_softmax) {
        auto v_bias = v_sub(v_neg(v_max_vec), v_log(v_sum_vec));
        for (int64_t w = 0; w < D; ++w)
            v_store(y_row + w * pack, v_add(v_load(x_row + w * pack), v_bias));
    } else {
        auto v_inv = v_div(v_set1(x_row, 1.0f), v_sum_vec);
        for (int64_t w = 0; w < D; ++w)
            v_store(y_row + w * pack, v_mul(v_load(y_row + w * pack), v_inv));
    }
}

// ============================================================
// Standard SIMD fast path: contiguous tail, pack == 1
// ============================================================

template <typename T>
inline void process_standard_row(
    const T* x, T* y,
    int64_t D, bool log_softmax)
{
    constexpr int L = simd_lane_for<T>;
    int64_t i = 0;

    // ---- Pass 1: Max reduction ----
    float max_val = -std::numeric_limits<float>::infinity();
    {
        auto v_max_val = v_set1(x, max_val);
        for (; i + L <= D; i += L)
            v_max_val = v_max(v_max_val, v_load(x + i));
        max_val = v_reduce_max(v_max_val);
    }
    for (; i < D; ++i) {
        float xv = s_load(&x[i]);
        if (xv > max_val) { max_val = xv; }
    }

    const float neg_max = -max_val;

    // ---- Pass 2: Sum of exp(x - max) ----
    float sum_exp = 0.0f;
    i = 0;
    const bool store_exp = !log_softmax;

    {
        auto v_sum = v_zero(x);
        auto v_neg_max = v_set1(x, neg_max);
        for (; i + L <= D; i += L) {
            auto v = v_add(v_load(x + i), v_neg_max);
            v = v_exp(v);
            if (store_exp) { v_store(y + i, v); }
            v_sum = v_add(v_sum, v);
        }
        sum_exp = v_reduce_sum(v_sum);
    }
    for (; i < D; ++i) {
        float val = std::exp(s_load(&x[i]) - max_val);
        if (store_exp) { s_store(&y[i], val); }
        sum_exp += val;
    }

    // ---- Pass 3: Normalize ----
    i = 0;
    if (log_softmax) {
        const float bias = neg_max - std::log(sum_exp);
        auto v_bias = v_set1(x, bias);
        for (; i + L <= D; i += L)
            v_store(y + i, v_add(v_load(x + i), v_bias));
        for (; i < D; ++i)
            s_store(&y[i], s_load(&x[i]) + bias);
    } else {
        auto v_inv = v_set1(x, 1.0f / sum_exp);
        for (; i + L <= D; i += L)
            v_store(y + i, v_mul(v_load(y + i), v_inv));
        for (; i < D; ++i)
            s_store(&y[i], s_load(&y[i]) / sum_exp);
    }
}

}  // namespace nnops::kernel::softmax
