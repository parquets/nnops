#pragma once
/// @file simd_softmax.hpp
/// @brief SIMD kernel functions for softmax / log-softmax.
///
/// Three SIMD dispatch paths:
///   - Packed row:    per-lane SIMD, contiguous pack-wide strides (axis == rank-1)
///   - Packed channel: per-lane SIMD with pitch between C8 blocks (axis == 1)
///   - Standard:       contiguous tail, SIMD max/sum reduction (pack == 1)
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
/// At each spatial position, D = C8 channel blocks are traversed via
/// chan_stride.  Each of the `pack` SIMD lanes is an independent softmax
/// across the C8 dimension.  Per-lane SIMD processes full C8 blocks;
/// the partial last block (when valid_lanes < pack) is handled with
/// scalar per-lane correction so that padded zeros do not perturb the
/// per-lane max / sum / normalization.
///
/// valid_lanes: number of valid channels in the last C8 block
///   (= C % pack, or pack when C is a multiple of pack).
template <typename T>
inline void softmax_process_packed_channel(
    const T* x_chan, T* y_chan,
    int64_t x_chan_stride, int64_t y_chan_stride,
    int64_t D, int64_t pack, int64_t valid_lanes,
    bool log_softmax
) {
    int64_t full_blocks = (valid_lanes == pack || D == 0) ? D : D - 1;

    // ---- helper: read a single lane from a SIMD vector ----
    auto lane_val = [](auto vec, int64_t l) -> float {
        // store to a small temp buffer and read back the requested lane
        // (avoids needing a dedicated v_extract_lane in the SIMD abstraction)
        T buf[16];
        v_store(buf, vec);
        return s_load(&buf[l]);
    };

    // ============================================================
    // Pass 1 — per-lane max
    // ============================================================
    auto v_max_vec = v_set1(x_chan, -std::numeric_limits<float>::infinity());
    for (int64_t c = 0; c < full_blocks; ++c)
        v_max_vec = v_max(v_max_vec, v_load(x_chan + c * x_chan_stride));

    // Partial last block: only valid lanes contribute to max (pad=0 could
    // incorrectly win over negative valid maxima).
    if (valid_lanes < pack) {
        T tmp[16];
        v_store(tmp, v_max_vec);
        const T* last = x_chan + (D - 1) * x_chan_stride;
        for (int64_t l = 0; l < valid_lanes; ++l) {
            float v = s_load(&last[l]);
            if (v > s_load(&tmp[l])) s_store(&tmp[l], v);
        }
        v_max_vec = v_load(tmp);
    }

    // ============================================================
    // Pass 2 — exp(x - max) + per-lane sum
    // ============================================================
    auto v_sum_vec = v_zero(x_chan);
    auto v_neg_max = v_neg(v_max_vec);

    for (int64_t c = 0; c < full_blocks; ++c) {
        auto v = v_add(v_load(x_chan + c * x_chan_stride), v_neg_max);
        v = v_exp(v);
        if (!log_softmax) { v_store(y_chan + c * y_chan_stride, v); }
        v_sum_vec = v_add(v_sum_vec, v);
    }

    // Partial last block: scalar exp + sum for valid lanes
    if (valid_lanes < pack) {
        T tmp[16];
        v_store(tmp, v_sum_vec);
        const T* last_x = x_chan + (D - 1) * x_chan_stride;
        T*       last_y = y_chan + (D - 1) * y_chan_stride;

        for (int64_t l = 0; l < valid_lanes; ++l) {
            float xv  = s_load(&last_x[l]);
            float nm  = lane_val(v_neg_max, l);   // -max for this lane
            float ev  = std::exp(xv + nm);         // exp(x - max)
            if (!log_softmax) { s_store(&last_y[l], ev); }
            s_store(&tmp[l], s_load(&tmp[l]) + ev); // accumulate sum
        }
        v_sum_vec = v_load(tmp);
    }

    // ============================================================
    // Pass 3 — normalize
    // ============================================================
    if (log_softmax) {
        auto v_bias = v_sub(v_neg(v_max_vec), v_log(v_sum_vec));
        for (int64_t c = 0; c < full_blocks; ++c)
            v_store(y_chan + c * y_chan_stride,
                    v_add(v_load(x_chan + c * x_chan_stride), v_bias));

        // Partial last block
        if (valid_lanes < pack) {
            const T* last_x = x_chan + (D - 1) * x_chan_stride;
            T*       last_y = y_chan + (D - 1) * y_chan_stride;
            for (int64_t l = 0; l < valid_lanes; ++l) {
                float xv = s_load(&last_x[l]);
                float b  = lane_val(v_bias, l);
                s_store(&last_y[l], xv + b);
            }
        }
    } else {
        auto v_inv = v_div(v_set1(x_chan, 1.0f), v_sum_vec);
        for (int64_t c = 0; c < full_blocks; ++c)
            v_store(y_chan + c * y_chan_stride,
                    v_mul(v_load(y_chan + c * y_chan_stride), v_inv));

        // Partial last block: exp values were already stored in pass 2,
        // just multiply by inv.
        if (valid_lanes < pack) {
            T* last_y = y_chan + (D - 1) * y_chan_stride;
            for (int64_t l = 0; l < valid_lanes; ++l) {
                float ev = s_load(&last_y[l]);
                float inv = lane_val(v_inv, l);
                s_store(&last_y[l], ev * inv);
            }
        }
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
