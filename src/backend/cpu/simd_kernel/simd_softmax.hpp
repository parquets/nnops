#pragma once
/// @file simd_softmax.hpp
/// @brief SIMD kernel functions for softmax / log-softmax.
///
/// Two SIMD paths:
///   - Packed: per-lane SIMD reduction within each physical row.
///   - Standard: contiguous tail, SIMD max/sum reduction.
///
/// Only lane=8 SIMD types (v_f32x8 / v_f16x8), matching simd_lane_for<T>.

#include "nnops/detail/simd/simd.hpp"

#include <cmath>
#include <cfloat>

namespace nnops::kernel {

using namespace simd;

/// Packed SIMD path: per-lane reduction within each physical row.
/// For NCHWC8 [N,C8,H,W] with axis=W, each of the `pack` C lanes
/// is an independent softmax over D=W spatial positions.
template <typename T>
inline void softmax_process_packed_row(
    const T* x_row, T* y_row,
    int64_t D, int64_t pack, bool log_softmax)
{
    auto v_max_vec = v_set1(x_row, -std::numeric_limits<float>::infinity());
    for (int64_t w = 0; w < D; ++w)
        v_max_vec = v_max(v_max_vec, v_load(x_row + w * pack));

    auto v_sum_vec = v_zero(x_row);
    auto v_neg_max = v_neg(v_max_vec);
    for (int64_t w = 0; w < D; ++w) {
        auto v = v_add(v_load(x_row + w * pack), v_neg_max);
        v = v_exp(v);
        if (!log_softmax) { v_store(y_row + w * pack, v); }
        v_sum_vec = v_add(v_sum_vec, v);
    }

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

template <typename T>
inline void softmax_process_packed_col(
    const T* x_col, T* y_col,
    int x_pitch, int y_pitch,
    int64_t D, // number of rows
    int64_t pack, // number of lanes
    bool log_softmax
)
{
    auto v_max_vec = v_set1(x_col, -std::numeric_limits<float>::infinity());
    for (int64_t h = 0; h < D; ++h) {
        v_max_vec = v_max(v_max_vec, v_load(x_col + h * x_pitch));
    }

    auto v_sum_vec = v_zero(x_col);
    auto v_neg_max = v_neg(v_max_vec);
    for (int64_t h = 0; h < D; ++h) {
        auto v = v_add(v_load(x_col + h * x_pitch), v_neg_max);
        v = v_exp(v);
        if (!log_softmax) { v_store(y_col + h * y_pitch, v); }
        v_sum_vec = v_add(v_sum_vec, v);
    }

    if (log_softmax) {
        auto v_bias = v_sub(v_neg(v_max_vec), v_log(v_sum_vec));
        for (int64_t h = 0; h < D; ++h) {
            v_store(y_col + h * y_pitch, v_add(v_load(x_col + h * x_pitch), v_bias));
        }
    } else {
        auto v_inv = v_div(v_set1(x_col, 1.0f), v_sum_vec);
        for (int64_t h = 0; h < D; ++h) {
            v_store(y_col + h * y_pitch, v_mul(v_load(y_col + h * y_pitch), v_inv));
        }
    }
}

/// Channel-wise softmax for NCHWC8 / NCDHWC8 packed layout.
///
/// At each spatial position (n, h, w) or (n, d, h, w), D = C8 channel
/// blocks are traversed via chan_stride.  Each of the `pack` SIMD lanes
/// is an independent softmax across the C8 dimension.
///
/// x_chan_stride / y_chan_stride: element stride between consecutive C8
/// blocks at the same spatial position, i.e. stride_elems(1) in NCHWC8.
template <typename T>
inline void softmax_process_packed_channel(
    const T* x_chan, T* y_chan,
    int64_t x_chan_stride, int64_t y_chan_stride,
    int64_t D, // number of C8 blocks (= C/8, rounded up)
    int64_t pack, // number of SIMD lanes (8)
    bool log_softmax
) {
    // Pass 1 — per-lane max across all C8 blocks
    auto v_max_vec = v_set1(x_chan, -std::numeric_limits<float>::infinity());
    for (int64_t c = 0; c < D; ++c)
        v_max_vec = v_max(v_max_vec, v_load(x_chan + c * x_chan_stride));

    // Pass 2 — exp(x - max) + sum
    auto v_sum_vec = v_zero(x_chan);
    auto v_neg_max = v_neg(v_max_vec);
    for (int64_t c = 0; c < D; ++c) {
        auto v = v_add(v_load(x_chan + c * x_chan_stride), v_neg_max);
        v = v_exp(v);
        if (!log_softmax) { v_store(y_chan + c * y_chan_stride, v); }
        v_sum_vec = v_add(v_sum_vec, v);
    }

    // Pass 3 — normalize
    if (log_softmax) {
        auto v_bias = v_sub(v_neg(v_max_vec), v_log(v_sum_vec));
        for (int64_t c = 0; c < D; ++c)
            v_store(y_chan + c * y_chan_stride,
                    v_add(v_load(x_chan + c * x_chan_stride), v_bias));
    } else {
        auto v_inv = v_div(v_set1(x_chan, 1.0f), v_sum_vec);
        for (int64_t c = 0; c < D; ++c)
            v_store(y_chan + c * y_chan_stride,
                    v_mul(v_load(y_chan + c * y_chan_stride), v_inv));
    }
}


/// Standard SIMD fast path: contiguous tail, pack == 1.
template <typename T>
inline void softmax_process_standard_row(
    const T* x, T* y,
    int64_t D, bool log_softmax)
{
    constexpr int L = simd_lane_for<T>;
    int64_t i = 0;

    // Pass 1: Max reduction
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

    // Pass 2: Sum of exp(x - max)
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

    // Pass 3: Normalize
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

}  // namespace nnops::kernel
