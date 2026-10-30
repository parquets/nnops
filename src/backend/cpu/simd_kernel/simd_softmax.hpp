#pragma once
/// @file simd_softmax.hpp
/// @brief SIMD softmax / log-softmax kernels.
///
/// softmax_per_lane is the unified 3-pass kernel — max, then exp((x-max)/T) plus
/// its sum, then normalize — over D elements strided by `stride`. Every per-lane
/// variant (packed row, packed column, strided axis) is this kernel and differs
/// only in the stride. softmax_process_packed_channel adds a scalar per-lane
/// correction over a partial last C8 block, whose padded zeros must not perturb
/// the max, sum or normalization.
///
/// softmax_process_standard_row is structurally different: horizontal
/// v_reduce_max / v_reduce_sum over one contiguous row. HasMask fuses an
/// additive mask (softmax(x + mask)) without materializing x+mask, saving a full
/// buffer pass; `mask` is read only when HasMask is true, and y may alias x.
///
/// softmax_row_max and softmax_row_exp_sum are the per-tile steps of the tiled
/// (FlashAttention) online softmax: the caller keeps the running max/sum across
/// KV tiles and does the final normalization itself.
///
/// Only lane=8 SIMD types (v_f32x8 / v_f16x8), matching simd_lane_for<T>.

#include "nnops/detail/simd/simd.hpp"
#include "simd_reduce_primitive.hpp"

#include <cmath>
#include <cfloat>

namespace nnops::kernel {

using namespace simd;

template <typename T>
inline void softmax_per_lane(
    const T* x, T* y,
    int64_t D, int64_t stride,
    bool log_softmax, float inv_T)
{
    auto v_inv_T = v_set1(x, inv_T);

    auto v_max_vec = v_set1(x, -std::numeric_limits<float>::infinity());
    for (int64_t k = 0; k < D; ++k) {
        v_max_vec = v_max(v_max_vec, v_load(x + k * stride));
    }

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

    auto lane_val = [](auto vec, int64_t l) -> float {
        T buf[16];
        v_store(buf, vec);
        return s_load(&buf[l]);
    };

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


template <typename T, bool HasMask>
inline auto softmax_xm(const T* x, const T* mask, int64_t i) {
    if constexpr (HasMask) {
        return v_add(v_load(x + i), v_load(mask + i));
    } else {
        return v_load(x + i);
    }
}

template <typename T, bool HasMask>
inline float softmax_xm_scalar(const T* x, const T* mask, int64_t i) {
    if constexpr (HasMask) {
        return s_load(&x[i]) + s_load(&mask[i]);
    } else {
        return s_load(&x[i]);
    }
}

template <typename T, bool HasMask>
inline void softmax_process_standard_row_impl(
    const T* x, const T* mask, T* y,
    int64_t D, bool log_softmax,
    float inv_T)
{
    constexpr int L = simd_lane_for<T>;
    int64_t i = 0;

    float max_val = -std::numeric_limits<float>::infinity();
    {
        auto v_max_val = v_set1(x, max_val);
        for (; i + L <= D; i += L) {
            v_max_val = v_max(v_max_val, softmax_xm<T, HasMask>(x, mask, i));
        }
        max_val = v_reduce_max(v_max_val);
    }
    for (; i < D; ++i) {
        float xv = softmax_xm_scalar<T, HasMask>(x, mask, i);
        if (xv > max_val) { max_val = xv; }
    }

    const float neg_max = -max_val;

    float sum_exp = 0.0f;
    i = 0;
    const bool store_exp = !log_softmax;

    {
        auto v_sum = v_zero(x);
        auto v_neg_max = v_set1(x, neg_max);
        auto v_inv_T = v_set1(x, inv_T);
        for (; i + L <= D; i += L) {
            auto v = v_add(softmax_xm<T, HasMask>(x, mask, i), v_neg_max);
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
        float val = std::exp((softmax_xm_scalar<T, HasMask>(x, mask, i) - max_val) * inv_T);
        if (store_exp) { s_store(&y[i], val); }
        sum_exp += val;
    }

    i = 0;
    if (log_softmax) {
        const float bias = neg_max * inv_T - std::log(sum_exp);
        auto v_bias = v_set1(x, bias);
        if (inv_T != 1.0f) {
            auto v_inv_T = v_set1(x, inv_T);
            for (; i + L <= D; i += L) {
                v_store(y + i, v_add(v_mul(softmax_xm<T, HasMask>(x, mask, i), v_inv_T), v_bias));
            }
            for (; i < D; ++i) {
                s_store(&y[i], softmax_xm_scalar<T, HasMask>(x, mask, i) * inv_T + bias);
            }
        } else {
            for (; i + L <= D; i += L) {
                v_store(y + i, v_add(softmax_xm<T, HasMask>(x, mask, i), v_bias));
            }
            for (; i < D; ++i) {
                s_store(&y[i], softmax_xm_scalar<T, HasMask>(x, mask, i) + bias);
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

template <typename T>
inline void softmax_process_standard_row(
    const T* x, T* y,
    int64_t D, bool log_softmax,
    float inv_T = 1.0f)
{
    softmax_process_standard_row_impl<T, false>(x, nullptr, y, D, log_softmax, inv_T);
}

template <typename T>
inline void mask_softmax_process_standard_row(
    const T* x, const T* mask, T* y,
    int64_t D, bool log_softmax,
    float inv_T = 1.0f)
{
    softmax_process_standard_row_impl<T, true>(x, mask, y, D, log_softmax, inv_T);
}

template <typename T>
inline float softmax_row_max(const T* x, int64_t n)
{
    return row_reduce<T, ReduceMax>(x, n);
}

/// y[i] = exp((x[i] - ref_max) * inv_T); returns sum(y).
template <typename T>
inline float softmax_row_exp_sum(const T* x, T* y, int64_t n, float ref_max, float inv_T)
{
    constexpr int L = simd_lane_for<T>;
    int64_t i = 0;
    float sum = 0.0f;
    {
        auto v_sum = v_zero(x);
        auto v_neg_ref = v_set1(x, -ref_max);
        auto v_inv_T = v_set1(x, inv_T);
        for (; i + L <= n; i += L) {
            auto v = v_add(v_load(x + i), v_neg_ref);
            if (inv_T != 1.0f) { v = v_mul(v, v_inv_T); }
            v = v_exp(v);
            v_store(y + i, v);
            v_sum = v_add(v_sum, v);
        }
        sum = v_reduce_sum(v_sum);
    }
    for (; i < n; ++i) {
        float val = std::exp((s_load(&x[i]) - ref_max) * inv_T);
        s_store(&y[i], val);
        sum += val;
    }
    return sum;
}

}  // namespace nnops::kernel
