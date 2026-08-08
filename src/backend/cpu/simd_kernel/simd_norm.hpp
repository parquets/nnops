#pragma once
/// @file simd_norm.hpp
/// @brief SIMD kernel functions for normalization operators.
///
/// Covers BatchNorm, LayerNorm, GroupNorm, and RMSNorm.
///
/// Shared primitives:
///   - norm_reduce_sum_sq<T>: 4-wide SIMD reduction of sum + sum_sq over contiguous data,
///     with 4→2→1-wide multi-accumulator unrolling and scalar tail.
///   - norm_apply_affine_row<T>: SIMD normalize with broadcast scale/bias:
///     y = (x - mean) * inv_std * scale_val + bias_val
///
/// High-level kernels (consume the shared primitives):
///   - layer_norm_process_row<T>: full LN: norm_reduce_sum_sq → stats → per-element normalize
///   - rms_norm_process_row<T>:   full RMS: norm_reduce_sum_sq → rms stats → per-element normalize
///
/// BatchNorm kernels (fused fmadd formula, structurally different):
///   - batch_norm_process_packed_row<T>
///   - batch_norm_process_planar_row<T>
///   - batch_norm_process_nonspatial_block<T>

#include "nnops/detail/simd/simd.hpp"

#include <cmath>
#include <utility>

namespace nnops::kernel {

using namespace simd;

// ============================================================
// Shared primitives
// ============================================================

/// SIMD sum + sum-of-squares reduction over n contiguous elements.
///
/// Uses 4-wide multi-accumulator unrolling for ILP, then falls back
/// through 2-wide → 1-wide SIMD → scalar tail.
///
/// Returns {sum, sum_sq} as float regardless of T (fp16 values are promoted).
template <typename T>
inline std::pair<float, float> norm_reduce_sum_sq(const T* x, int64_t n)
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

// ============================================================
// BatchNorm kernels (fused fmadd formula, structurally unique)
// ============================================================

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

// ============================================================
// LayerNorm kernel
// ============================================================

/// Process one row of layer normalization using SIMD fast path.
///
/// Uses norm_reduce_sum_sq for the reduction pass, then per-element
/// SIMD normalize with optional per-element or scalar scale/bias.
template <typename T>
inline void layer_norm_process_row(
    const T* x, T* y,
    const T* scale, const T* bias,
    int64_t n, float epsilon,
    bool scale_is_scalar, bool has_bias, bool add_to)
{
    constexpr int L = simd_lane_for<T>;

    // Pass 1 — SIMD reduction via shared primitive
    auto [sum, sum_sq] = norm_reduce_sum_sq<T>(x, n);

    const float inv_n = 1.0f / static_cast<float>(n);
    const float mean_val = sum * inv_n;
    float var_val = sum_sq * inv_n - mean_val * mean_val;
    if (var_val < 0.0f) { var_val = 0.0f; }
    const float inv_std = 1.0f / std::sqrt(var_val + epsilon);

    // Pass 2 — SIMD normalize with per-element or scalar scale/bias
    int64_t i = 0;

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

// ============================================================
// RMSNorm kernel
// ============================================================

/// Process one row of RMS normalization using SIMD fast path.
///
/// Uses norm_reduce_sum_sq for the sum-of-squares reduction (ignores sum),
/// then per-element SIMD normalize with optional per-element or scalar scale.
template <typename T>
inline void rms_norm_process_row(
    const T* x, T* y,
    const T* scale,
    int64_t n, float epsilon,
    bool scale_is_scalar, bool add_to)
{
    constexpr int L = simd_lane_for<T>;

    // Pass 1 — SIMD sum_sq reduction via shared primitive (ignore sum)
    auto sum_sq_pair = norm_reduce_sum_sq<T>(x, n);
    float sum_sq = sum_sq_pair.second;
    (void)sum_sq_pair.first;

    const float rms = std::sqrt(sum_sq / static_cast<float>(n) + epsilon);
    const float inv_rms = 1.0f / rms;

    // Pass 2 — SIMD normalize with per-element or scalar scale
    int64_t i = 0;
    const auto v_inv_rms = v_set1(x, inv_rms);

    for (; i + L <= n; i += L) {
        auto xv = v_load(x + i);
        auto vs = v_load(scale + (scale_is_scalar ? 0 : i));
        auto rv = v_mul(v_mul(xv, v_inv_rms), vs);
        v_store_add(y + i, rv, add_to);
    }
    for (; i < n; ++i) {
        int64_t s_idx = scale_is_scalar ? 0 : i;
        float xv = s_load(&x[i]);
        float s = s_load(&scale[s_idx]);
        float rv = xv * inv_rms * s;
        s_store_add(&y[i], rv, add_to);
    }
}

}  // namespace nnops::kernel
