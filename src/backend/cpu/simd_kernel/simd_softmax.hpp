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
#include "simd_reduce_primitive.hpp"

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


/// Value of the softmax element at index i: x[i] (unmasked) or x[i] + mask[i]
/// (masked). `if constexpr` selects the form, so the masked path never reads
/// `mask` when HasMask is false.
template <typename T, bool HasMask>
inline auto softmax_xm(const T* x, const T* mask, int64_t i) {
    if constexpr (HasMask) {
        return v_add(v_load(x + i), v_load(mask + i));
    } else {
        return v_load(x + i);
    }
}

/// Scalar form of softmax_xm.
template <typename T, bool HasMask>
inline float softmax_xm_scalar(const T* x, const T* mask, int64_t i) {
    if constexpr (HasMask) {
        return s_load(&x[i]) + s_load(&mask[i]);
    } else {
        return s_load(&x[i]);
    }
}

/// Shared core of the standard-row softmax (contiguous tail, pack == 1).
///
/// Uses horizontal SIMD reduction (v_reduce_max / v_reduce_sum), not per-lane
/// SIMD — structurally different from softmax_per_lane. `HasMask` selects
/// whether an additive mask is fused in (softmax(x + mask)) without
/// materializing the sum, saving one full-buffer pass versus an explicit
/// `x += mask` before a plain softmax. `mask` is only read when HasMask is
/// true (may be null otherwise); y may alias x (in-place).
template <typename T, bool HasMask>
inline void softmax_process_standard_row_impl(
    const T* x, const T* mask, T* y,
    int64_t D, bool log_softmax,
    float inv_T)
{
    constexpr int L = simd_lane_for<T>;
    int64_t i = 0;

    // Pass 1: Max reduction (temperature doesn't affect max)
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

    // Pass 2: Sum of exp((x - max) / T)
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

    // Pass 3: Normalize
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

/// Standard-row softmax (no mask).
template <typename T>
inline void softmax_process_standard_row(
    const T* x, T* y,
    int64_t D, bool log_softmax,
    float inv_T = 1.0f)
{
    softmax_process_standard_row_impl<T, false>(x, nullptr, y, D, log_softmax, inv_T);
}

/// Masked standard-row softmax: softmax(x + mask) without materializing the sum.
template <typename T>
inline void mask_softmax_process_standard_row(
    const T* x, const T* mask, T* y,
    int64_t D, bool log_softmax,
    float inv_T = 1.0f)
{
    softmax_process_standard_row_impl<T, true>(x, mask, y, D, log_softmax, inv_T);
}

/// Row maximum of one contiguous row — building block of the tiled
/// (FlashAttention) online-softmax, where the caller tracks the running max
/// across KV tiles instead of materializing the full softmax row.
template <typename T>
inline float softmax_row_max(const T* x, int64_t n)
{
    return row_reduce<T, ReduceMax>(x, n);
}

/// Scaled exp + sum of one contiguous row referenced to an externally supplied
/// running max `ref_max` (the max of this row across all KV tiles processed so
/// far): writes y[i] = exp((x[i] - ref_max) * inv_T) and returns sum(y). The
/// caller owns the running max/sum and the final normalization; this is the
/// per-tile update step of the tiled (FlashAttention) online-softmax.
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
