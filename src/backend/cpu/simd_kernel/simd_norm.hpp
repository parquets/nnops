#pragma once
/// @file simd_norm.hpp
/// @brief Row kernels shared by the normalization operators (BatchNorm,
/// LayerNorm, GroupNorm, RMSNorm, L2Norm).
///
/// Two shapes of work:
///  - Normalize from statistics: norm_reduce_sum_sq (the shared reduction,
///    re-exported) and one element-wise pass, either with a broadcast
///    scale/bias (norm_apply_affine_row, the GroupNorm shape) or with
///    mean/scale/bias terms gated at compile time by HasMean/HasScale/HasBias
///    (norm_apply_row, L2/RMS/Layer; scale/bias broadcast or per-element).
///  - BatchNorm's fused fmadd, one ns*x + nb per block with ns/nb pre-folding
///    scale/bias/mean/var (the non-spatial kernel folds them itself), written
///    per storage: packed C8, channel-first, or per-element block.
///
/// Every writer goes through v_store_add/s_store_add, so a residual `add_to`
/// costs no extra pass.

#include "nnops/detail/simd/simd.hpp"
#include "simd_reduce_primitive.hpp"

#include <cmath>
#include <type_traits>
#include <utility>

namespace nnops::kernel {

using namespace simd;

/// Re-export of row_reduce_sum_sq (simd_reduce_primitive.hpp).
template <typename T>
inline std::pair<float, float> norm_reduce_sum_sq(const T* x, int64_t n)
{
    return row_reduce_sum_sq<T>(x, n);
}

/// The same reduction over a pre-added row: Σ(x+r) and Σ(x+r)². The residual
/// is added here rather than in a separate pass, so a fused `norm(x + r)` reads
/// both operands twice and writes one output instead of writing and re-reading
/// a sum tensor.
template <typename T>
inline std::pair<float, float> norm_reduce_sum_sq_add(const T* x, const T* r, int64_t n)
{
    constexpr int L = simd_lane_for<T>;
    int64_t i = 0;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    if constexpr (std::is_same_v<T, half>) {
        // f16 storage, f32 accumulation: on x86 every f16 vector op is a
        // convert round trip, so the pair is widened once here.
        v_f32x8 v_sum = v_set1_f32x8(0.0f);
        v_f32x8 v_sq = v_set1_f32x8(0.0f);
        for (; i + L <= n; i += L) {
            const v_f32x8 v = v_add(v_cvt_f16_to_f32(v_load(x + i)),
                                    v_cvt_f16_to_f32(v_load(r + i)));
            v_sum = v_add(v_sum, v);
            v_sq = v_fmadd(v, v, v_sq);
        }
        sum = v_reduce_sum(v_sum);
        sum_sq = v_reduce_sum(v_sq);
    } else {
        v_f32x8 v_sum = v_set1_f32x8(0.0f);
        v_f32x8 v_sq = v_set1_f32x8(0.0f);
        for (; i + L <= n; i += L) {
            const v_f32x8 v = v_add(v_load(x + i), v_load(r + i));
            v_sum = v_add(v_sum, v);
            v_sq = v_fmadd(v, v, v_sq);
        }
        sum = v_reduce_sum(v_sum);
        sum_sq = v_reduce_sum(v_sq);
    }
    for (; i < n; ++i) {
        const float v = s_load(&x[i]) + s_load(&r[i]);
        sum += v;
        sum_sq += v * v;
    }
    return {sum, sum_sq};
}

/// (x - mean) * inv_std * scale + bias with broadcast scale/bias — GroupNorm,
/// where a row within a channel shares one scale/bias pair.
template <typename T>
inline void norm_apply_affine_row(const T* x, T* y, int64_t n,
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
        auto rv = v_fmadd(v_mul(v_sub(xv, v_mean), v_inv_std), v_scale, v_bias);
        v_store_add(y + i, rv, add_to);
    }
    for (; i < n; ++i) {
        float xv = s_load(&x[i]);
        float rv = (xv - mean) * inv_std * scale_val + bias_val;
        s_store_add(&y[i], rv, add_to);
    }
}

/// norm_apply_row over the pre-added row (x + r) — the write side of the fused
/// `norm(x + r)`.
template <typename T, bool HasMean, bool HasScale, bool HasBias,
          bool ScaleOnePlus = false>
inline void norm_apply_row_add(const T* x, const T* r, T* y, int64_t n,
                               float mean_val, float inv_std,
                               const T* scale, const T* bias,
                               bool scale_is_scalar, bool add_to)
{
    constexpr int L = simd_lane_for<T>;
    int64_t i = 0;

    const auto v_mean    = v_set1(x, mean_val);
    const auto v_inv_std = v_set1(x, inv_std);
    const auto v_one     = v_set1(x, 1.0f);

    for (; i + L <= n; i += L) {
        auto rv = v_add(v_load(x + i), v_load(r + i));
        if constexpr (HasMean) {
            rv = v_sub(rv, v_mean);
        }
        rv = v_mul(rv, v_inv_std);
        if constexpr (HasScale) {
            // A scalar scale is one element of storage: broadcast it instead
            // of loading a full lane from past the end of the buffer.
            auto vs = scale_is_scalar ? v_set1(x, s_load(scale)) : v_load(scale + i);
            if constexpr (ScaleOnePlus) {
                vs = v_add(vs, v_one);
            }
            rv = v_mul(rv, vs);
        }
        if constexpr (HasBias) {
            auto vb = scale_is_scalar ? v_set1(x, s_load(bias)) : v_load(bias + i);
            rv = v_add(rv, vb);
        }
        v_store_add(y + i, rv, add_to);
    }
    for (; i < n; ++i) {
        float rv = s_load(&x[i]) + s_load(&r[i]);
        if constexpr (HasMean) {
            rv -= mean_val;
        }
        rv *= inv_std;
        if constexpr (HasScale) {
            int64_t s_idx = scale_is_scalar ? 0 : i;
            rv *= s_load(&scale[s_idx]) + (ScaleOnePlus ? 1.0f : 0.0f);
        }
        if constexpr (HasBias) {
            int64_t b_idx = scale_is_scalar ? 0 : i;
            rv += s_load(&bias[b_idx]);
        }
        s_store_add(&y[i], rv, add_to);
    }
}

/// ((x - mean?) * inv_std) * scale? + bias? — the HasMean/HasScale/HasBias
/// bools compile out the missing terms. ScaleOnePlus makes the stored weight
/// 1-centered: the effective scale is (1 + w), as Gemma's zero-centered norms
/// and Qwen3-Next/4's need. Folding the +1 in here keeps it out of the stored
/// weight, which in bf16 would otherwise lose the small-|w| correction.
template <typename T, bool HasMean, bool HasScale, bool HasBias,
          bool ScaleOnePlus = false>
inline void norm_apply_row(const T* x, T* y, int64_t n,
                            float mean_val, float inv_std,
                            const T* scale, const T* bias,
                            bool scale_is_scalar, bool add_to)
{
    constexpr int L = simd_lane_for<T>;
    int64_t i = 0;

    const auto v_mean    = v_set1(x, mean_val);
    const auto v_inv_std = v_set1(x, inv_std);
    const auto v_one     = v_set1(x, 1.0f);

    for (; i + L <= n; i += L) {
        auto xv = v_load(x + i);
        auto rv = xv;
        if constexpr (HasMean) {
            rv = v_sub(rv, v_mean);
        }
        rv = v_mul(rv, v_inv_std);
        if constexpr (HasScale) {
            // A scalar scale is one element of storage: broadcast it instead
            // of loading a full lane from past the end of the buffer.
            auto vs = scale_is_scalar ? v_set1(x, s_load(scale)) : v_load(scale + i);
            if constexpr (ScaleOnePlus) {
                vs = v_add(vs, v_one);
            }
            rv = v_mul(rv, vs);
        }
        if constexpr (HasBias) {
            auto vb = scale_is_scalar ? v_set1(x, s_load(bias)) : v_load(bias + i);
            rv = v_add(rv, vb);
        }
        v_store_add(y + i, rv, add_to);
    }
    for (; i < n; ++i) {
        float xv = s_load(&x[i]);
        float rv = xv;
        if constexpr (HasMean) {
            rv -= mean_val;
        }
        rv *= inv_std;
        if constexpr (HasScale) {
            int64_t s_idx = scale_is_scalar ? 0 : i;
            rv *= s_load(&scale[s_idx]) + (ScaleOnePlus ? 1.0f : 0.0f);
        }
        if constexpr (HasBias) {
            int64_t b_idx = scale_is_scalar ? 0 : i;
            rv += s_load(&bias[b_idx]);
        }
        s_store_add(&y[i], rv, add_to);
    }
}

/// Runtime dispatch over ScaleOnePlus for norm_apply_row, so the operators keep
/// one call per (HasMean, HasScale, HasBias) shape.
template <typename T, bool HasMean, bool HasScale, bool HasBias>
inline void norm_apply_row_dispatch(const T* x, T* y, int64_t n,
                                    float mean_val, float inv_std,
                                    const T* scale, const T* bias,
                                    bool scale_is_scalar, bool add_to,
                                    bool one_plus_weight)
{
    if constexpr (HasScale) {
        if (one_plus_weight) {
            norm_apply_row<T, HasMean, HasScale, HasBias, true>(
                x, y, n, mean_val, inv_std, scale, bias, scale_is_scalar, add_to);
            return;
        }
    }
    norm_apply_row<T, HasMean, HasScale, HasBias, false>(
        x, y, n, mean_val, inv_std, scale, bias, scale_is_scalar, add_to);
}

/// Runtime dispatch over ScaleOnePlus for norm_apply_row_add.
template <typename T, bool HasMean, bool HasScale, bool HasBias>
inline void norm_apply_row_add_dispatch(const T* x, const T* r, T* y, int64_t n,
                                        float mean_val, float inv_std,
                                        const T* scale, const T* bias,
                                        bool scale_is_scalar, bool add_to,
                                        bool one_plus_weight)
{
    if constexpr (HasScale) {
        if (one_plus_weight) {
            norm_apply_row_add<T, HasMean, HasScale, HasBias, true>(
                x, r, y, n, mean_val, inv_std, scale, bias, scale_is_scalar, add_to);
            return;
        }
    }
    norm_apply_row_add<T, HasMean, HasScale, HasBias, false>(
        x, r, y, n, mean_val, inv_std, scale, bias, scale_is_scalar, add_to);
}

/// Channel-first layout: 4-wide unrolled SIMD fmadd per row.
template <typename T>
inline void batch_norm_process_channel_first_row(
    const T* x_row, T* y_row,
    int64_t last_dim,
    float ns, float nb, bool add_to)
{
    constexpr int L = simd_lane_for<T>;
    int64_t i = 0;

    const auto scale8 = v_set1(x_row, ns);
    const auto bias8  = v_set1(x_row, nb);

    for (; i + 4 * L <= last_dim; i += 4 * L) {
        auto x0 = v_load(x_row + i);
        auto x1 = v_load(x_row + i + L);
        auto x2 = v_load(x_row + i + 2 * L);
        auto x3 = v_load(x_row + i + 3 * L);
        v_store_add(y_row + i,       v_fmadd(scale8, x0, bias8), add_to);
        v_store_add(y_row + i + L,     v_fmadd(scale8, x1, bias8), add_to);
        v_store_add(y_row + i + 2 * L, v_fmadd(scale8, x2, bias8), add_to);
        v_store_add(y_row + i + 3 * L, v_fmadd(scale8, x3, bias8), add_to);
    }
    for (; i + L <= last_dim; i += L) {
        auto rv = v_fmadd(scale8, v_load(x_row + i), bias8);
        v_store_add(y_row + i, rv, add_to);
    }
    for (; i < last_dim; ++i) {
        float rv = s_load(&x_row[i]) * ns + nb;
        s_store_add(&y_row[i], rv, add_to);
    }
}

/// Non-spatial mode: full per-element scale/bias/mean/var arrays.
template <typename T>
inline void batch_norm_process_nonspatial_block(
    const T* x, T* y,
    const T* scale, const T* bias,
    const T* mean, const T* var,
    int64_t i_begin, int64_t i_end,
    float epsilon, bool add_to)
{
    constexpr int L = simd_lane_for<T>;
    int64_t i = i_begin;
    const auto eps8 = v_set1(x, epsilon);
    const auto one8 = v_set1(x, 1.0f);

    for (; i + 4 * L <= i_end; i += 4 * L) {
        for (int k = 0; k < 4; ++k) {
            int64_t off = i + k * L;
            auto xv  = v_load(x + off);
            auto sv  = v_load(scale + off);
            auto bv  = v_load(bias + off);
            auto mv  = v_load(mean + off);
            auto vv  = v_load(var + off);
            auto inv = v_div(one8, v_sqrt(v_add(vv, eps8)));
            auto ns  = v_mul(inv, sv);
            auto nb  = v_sub(bv, v_mul(mv, ns));
            v_store_add(y + off, v_fmadd(ns, xv, nb), add_to);
        }
    }
    for (; i + L <= i_end; i += L) {
        const auto x8 = v_load(x + i);
        const auto s8 = v_load(scale + i);
        const auto b8 = v_load(bias + i);
        const auto m8 = v_load(mean + i);
        const auto v8 = v_load(var + i);
        const auto inv_std = v_div(one8, v_sqrt(v_add(v8, eps8)));
        const auto ns = v_mul(inv_std, s8);
        const auto nb = v_sub(b8, v_mul(m8, ns));
        auto rv = v_fmadd(ns, x8, nb);
        v_store_add(y + i, rv, add_to);
    }
    for (; i < i_end; ++i) {
        float inv_std_val = 1.0f / std::sqrt(s_load(&var[i]) + epsilon);
        float ns = inv_std_val * s_load(&scale[i]);
        float nb_val = s_load(&bias[i]) - s_load(&mean[i]) * ns;
        float rv = s_load(&x[i]) * ns + nb_val;
        s_store_add(&y[i], rv, add_to);
    }
}

}  // namespace nnops::kernel
