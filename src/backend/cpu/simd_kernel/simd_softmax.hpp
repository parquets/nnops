#pragma once
/// @file simd_softmax.hpp
/// @brief SIMD kernel functions for softmax / log-softmax.
///
/// Kernel architecture:
///   - softmax_per_lane:        unified per-lane 3-pass kernel (max→exp+sum→norm)
///                              for all strided-access variants
///   - softmax_packed_channel:   per-lane + partial last-block scalar correction
///   - softmax_standard_row:     horizontal SIMD reduction for contiguous tail
///
/// Only lane=8 SIMD types (v_f32x8 / v_f16x8), matching simd_lane_for<T>.

#include "nnops/detail/simd/simd.hpp"

#include <cmath>
#include <cfloat>

namespace nnops::kernel {

using namespace simd;

/// Per-lane SIMD softmax: each SIMD lane is an independent 3-pass softmax
/// over D strided elements.
///
/// Algorithm: max → exp((x - max) / T) + sum → normalize.
/// All per-lane softmax variants (packed row, packed column, strided axis)
/// reduce to this single kernel — the only difference is the stride value.
template <typename T>
inline void softmax_per_lane(
    const T* x, T* y,
    int64_t D, int64_t stride,
    bool log_softmax, float inv_T)
{
    auto v_inv_T = v_set1(x, inv_T);

    // Pass 1: per-lane max
    auto v_max_vec = v_set1(x, -std::numeric_limits<float>::infinity());
    for (int64_t k = 0; k < D; ++k) {
        v_max_vec = v_max(v_max_vec, v_load(x + k * stride));
    }

    // Pass 2: exp((x - max) / T) + per-lane sum
    auto v_sum_vec = v_zero(x);
    auto v_neg_max = v_neg(v_max_vec);
    for (int64_t k = 0; k < D; ++k) {
        auto v = v_add(v_load(x + k * stride), v_neg_max);
        if (inv_T != 1.0f) {
            v = v_mul(v, v_inv_T);
        }
        v = v_exp(v);
        if (!log_softmax) { v_store(y + k * stride, v); }
        v_sum_vec = v_add(v_sum_vec, v);
    }

    // Pass 3: normalize
    if (log_softmax) {
        auto v_bias = v_sub(v_neg(v_mul(v_max_vec, v_inv_T)), v_log(v_sum_vec));
        for (int64_t k = 0; k < D; ++k) {
            v_store(y + k * stride,
                    v_add(v_mul(v_load(x + k * stride), v_inv_T), v_bias));
        }
    } else {
        auto v_inv = v_div(v_set1(x, 1.0f), v_sum_vec);
        for (int64_t k = 0; k < D; ++k) {
            v_store(y + k * stride, v_mul(v_load(y + k * stride), v_inv));
        }
    }
}

/// Channel-wise softmax for NCHWC8 / NCDHWC8 packed layout.
///
/// This is the only per-lane variant that needs special handling: when the
/// last C8 block is partial (valid_lanes < pack), padded zeros must not
/// perturb the per-lane max / sum / normalization. Full blocks are processed
/// via the same 3-pass algorithm as softmax_per_lane; the partial last block
/// uses scalar per-lane correction.
template <typename T>
inline void softmax_process_packed_channel(
    const T* x_chan, T* y_chan,
    int64_t stride,
    int64_t D, int64_t pack, int64_t valid_lanes,
    bool log_softmax,
    float inv_T = 1.0f
) {
    int64_t full_blocks = (valid_lanes == pack || D == 0) ? D : D - 1;
    auto v_inv_T = v_set1(x_chan, inv_T);

    // ---- helper: read a single lane from a SIMD vector ----
    auto lane_val = [](auto vec, int64_t l) -> float {
        T buf[16];
        v_store(buf, vec);
        return s_load(&buf[l]);
    };

    // ============================================================
    // Pass 1: per-lane max over full blocks + partial block
    // ============================================================
    auto v_max_vec = v_set1(x_chan, -std::numeric_limits<float>::infinity());
    for (int64_t c = 0; c < full_blocks; ++c) {
        v_max_vec = v_max(v_max_vec, v_load(x_chan + c * stride));
    }

    if (valid_lanes < pack) {
        T tmp[16];
        v_store(tmp, v_max_vec);
        const T* last = x_chan + (D - 1) * stride;
        for (int64_t l = 0; l < valid_lanes; ++l) {
            float v = s_load(&last[l]);
            if (v > s_load(&tmp[l])) { s_store(&tmp[l], v); }
        }
        v_max_vec = v_load(tmp);
    }

    // ============================================================
    // Pass 2: exp((x - max) / T) + per-lane sum
    // ============================================================
    auto v_sum_vec = v_zero(x_chan);
    auto v_neg_max = v_neg(v_max_vec);

    for (int64_t c = 0; c < full_blocks; ++c) {
        auto v = v_add(v_load(x_chan + c * stride), v_neg_max);
        if (inv_T != 1.0f) { v = v_mul(v, v_inv_T); }
        v = v_exp(v);
        if (!log_softmax) { v_store(y_chan + c * stride, v); }
        v_sum_vec = v_add(v_sum_vec, v);
    }

    if (valid_lanes < pack) {
        T tmp[16];
        v_store(tmp, v_sum_vec);
        const T* last_x = x_chan + (D - 1) * stride;
        T*       last_y = y_chan + (D - 1) * stride;

        for (int64_t l = 0; l < valid_lanes; ++l) {
            float xv = s_load(&last_x[l]);
            float nm = lane_val(v_neg_max, l);
            float ev = std::exp((xv + nm) * inv_T);
            if (!log_softmax) { s_store(&last_y[l], ev); }
            s_store(&tmp[l], s_load(&tmp[l]) + ev);
        }
        v_sum_vec = v_load(tmp);
    }

    // ============================================================
    // Pass 3: normalize
    // ============================================================
    if (log_softmax) {
        auto v_bias = v_sub(v_neg(v_mul(v_max_vec, v_inv_T)), v_log(v_sum_vec));
        for (int64_t c = 0; c < full_blocks; ++c) {
            v_store(y_chan + c * stride,
                    v_add(v_mul(v_load(x_chan + c * stride), v_inv_T), v_bias));
        }
        if (valid_lanes < pack) {
            const T* last_x = x_chan + (D - 1) * stride;
            T*       last_y = y_chan + (D - 1) * stride;
            for (int64_t l = 0; l < valid_lanes; ++l) {
                float xv = s_load(&last_x[l]);
                float b  = lane_val(v_bias, l);
                s_store(&last_y[l], xv * inv_T + b);
            }
        }
    } else {
        auto v_inv = v_div(v_set1(x_chan, 1.0f), v_sum_vec);
        for (int64_t c = 0; c < full_blocks; ++c) {
            v_store(y_chan + c * stride,
                    v_mul(v_load(y_chan + c * stride), v_inv));
        }
        if (valid_lanes < pack) {
            T* last_y = y_chan + (D - 1) * stride;
            for (int64_t l = 0; l < valid_lanes; ++l) {
                float ev = s_load(&last_y[l]);
                float inv = lane_val(v_inv, l);
                s_store(&last_y[l], ev * inv);
            }
        }
    }
}


/// Standard SIMD fast path: contiguous tail, pack == 1.
/// Uses horizontal SIMD reduction (v_reduce_max / v_reduce_sum),
/// not per-lane SIMD — structurally different from softmax_per_lane.
template <typename T>
inline void softmax_process_standard_row(
    const T* x, T* y,
    int64_t D, bool log_softmax,
    float inv_T = 1.0f)
{
    constexpr int L = simd_lane_for<T>;
    int64_t i = 0;

    // Pass 1: Max reduction (temperature doesn't affect max)
    float max_val = -std::numeric_limits<float>::infinity();
    {
        auto v_max_val = v_set1(x, max_val);
        for (; i + L <= D; i += L) {
            v_max_val = v_max(v_max_val, v_load(x + i));
        }
        max_val = v_reduce_max(v_max_val);
    }
    for (; i < D; ++i) {
        float xv = s_load(&x[i]);
        if (xv > max_val) { max_val = xv; }
    }

    const float neg_max = -max_val;

    // Pass 2: Sum of exp((x - max) / T)
    float sum_exp = 0.0f;
    i = 0;
    const bool store_exp = !log_softmax;

    {
        auto v_sum = v_zero(x);
        auto v_neg_max = v_set1(x, neg_max);
        auto v_inv_T = v_set1(x, inv_T);
        for (; i + L <= D; i += L) {
            auto v = v_add(v_load(x + i), v_neg_max);
            if (inv_T != 1.0f) {
                v = v_mul(v, v_inv_T);
            }
            v = v_exp(v);
            if (store_exp) { v_store(y + i, v); }
            v_sum = v_add(v_sum, v);
        }
        sum_exp = v_reduce_sum(v_sum);
    }
    for (; i < D; ++i) {
        float val = std::exp((s_load(&x[i]) - max_val) * inv_T);
        if (store_exp) { s_store(&y[i], val); }
        sum_exp += val;
    }

    // Pass 3: Normalize
    i = 0;
    if (log_softmax) {
        const float bias = neg_max * inv_T - std::log(sum_exp);
        auto v_bias = v_set1(x, bias);
        if (inv_T != 1.0f) {
            auto v_inv_T = v_set1(x, inv_T);
            for (; i + L <= D; i += L) {
                v_store(y + i, v_add(v_mul(v_load(x + i), v_inv_T), v_bias));
            }
            for (; i < D; ++i) {
                s_store(&y[i], s_load(&x[i]) * inv_T + bias);
            }
        } else {
            for (; i + L <= D; i += L) {
                v_store(y + i, v_add(v_load(x + i), v_bias));
            }
            for (; i < D; ++i) {
                s_store(&y[i], s_load(&x[i]) + bias);
            }
        }
    } else {
        auto v_inv = v_set1(x, 1.0f / sum_exp);
        for (; i + L <= D; i += L) {
            v_store(y + i, v_mul(v_load(y + i), v_inv));
        }
        for (; i < D; ++i) {
            s_store(&y[i], s_load(&y[i]) / sum_exp);
        }
    }
}

}  // namespace nnops::kernel
