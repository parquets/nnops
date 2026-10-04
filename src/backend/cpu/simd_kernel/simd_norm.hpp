#pragma once
/// @file simd_norm.hpp
/// @brief SIMD kernel functions for normalization operators.
///
/// Covers BatchNorm, LayerNorm, GroupNorm, RMSNorm, and L2Norm.
///
/// Shared primitives:
///   - norm_reduce_sum_sq<T>: 4-wide SIMD reduction of sum + sum_sq over contiguous data,
///     with 4→2→1-wide multi-accumulator unrolling and scalar tail.
///   - norm_apply_affine_row<T>: SIMD normalize with broadcast scale/bias:
///     y = (x - mean) * inv_std * scale_val + bias_val
///   - norm_apply_row<T, HasMean, HasScale, HasBias>: unified SIMD normalize pass
///     for L2/RMS/Layer norm: y = ((x - mean?) * inv_std) * scale? + bias?
///
/// BatchNorm kernels (fused fmadd formula, structurally different):
///   - batch_norm_process_packed_row<T>
///   - batch_norm_process_planar_row<T>
///   - batch_norm_process_nonspatial_block<T>

#include "nnops/detail/simd/simd.hpp"
#include "simd_reduce_primitive.hpp"

#include <cmath>
#include <utility>

namespace nnops::kernel {

using namespace simd;

/// SIMD sum + sum-of-squares reduction over n contiguous elements — thin
/// forwarder to the shared reduction primitive in simd_reduce_primitive.hpp.
template <typename T>
inline std::pair<float, float> norm_reduce_sum_sq(const T* x, int64_t n)
{
    return row_reduce_sum_sq<T>(x, n);
}

/// SIMD normalize: apply `(x - mean) * inv_std * scale_val + bias_val`
/// to n contiguous elements with broadcast (per-row) scale/bias.
///
/// Used by GroupNorm where each row within a channel shares the same
/// scale/bias value.
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

/// Unified SIMD element-wise normalize pass.
///
/// Computes: y[i] = ((x[i] - mean?) * inv_std) * scale? + bias?
/// Template bools control which operations are compiled in — the compiler
/// eliminates dead branches for each instantiation via `if constexpr`.
///
/// Used as the second pass by L2Norm, RMSNorm, and LayerNorm.
template <typename T, bool HasMean, bool HasScale, bool HasBias>
inline void norm_apply_row(const T* x, T* y, int64_t n,
                            float mean_val, float inv_std,
                            const T* scale, const T* bias,
                            bool scale_is_scalar, bool add_to)
{
    constexpr int L = simd_lane_for<T>;
    int64_t i = 0;

    const auto v_mean    = v_set1(x, mean_val);
    const auto v_inv_std = v_set1(x, inv_std);

    for (; i + L <= n; i += L) {
        auto xv = v_load(x + i);
        auto rv = xv;
        if constexpr (HasMean) {
            rv = v_sub(rv, v_mean);
        }
        rv = v_mul(rv, v_inv_std);
        if constexpr (HasScale) {
            auto vs = v_load(scale + (scale_is_scalar ? 0 : i));
            rv = v_mul(rv, vs);
        }
        if constexpr (HasBias) {
            auto vb = v_load(bias + (scale_is_scalar ? 0 : i));
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
            rv *= s_load(&scale[s_idx]);
        }
        if constexpr (HasBias) {
            int64_t b_idx = scale_is_scalar ? 0 : i;
            rv += s_load(&bias[b_idx]);
        }
        s_store_add(&y[i], rv, add_to);
    }
}

/// Packed layout: per-row SIMD fmadd with per-C8-block scale/bias.
template <typename T>
inline void batch_norm_process_packed_row(
    const T* x_row, T* y_row,
    const T* ns_packed, const T* nb_packed,
    int64_t c8, int64_t pack,
    int64_t last_dim, bool add_to)
{
    constexpr int L = simd_lane_for<T>;
    const auto ns8 = v_load(&ns_packed[c8 * pack]);
    const auto nb8 = v_load(&nb_packed[c8 * pack]);
    for (int64_t i = 0; i < last_dim; i += L) {
        v_store_add(y_row + i, v_fmadd(ns8, v_load(x_row + i), nb8), add_to);
    }
}

/// Planar layout: 4-wide unrolled SIMD fmadd per row.
template <typename T>
inline void batch_norm_process_planar_row(
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

/// Non-spatial mode: per-element SIMD processing.
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
