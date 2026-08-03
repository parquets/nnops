#pragma once
/// @file simd_batch_norm.hpp
/// @brief SIMD kernel functions for batch normalization (inference only).
///
/// Fused formula: new_scale = inv_std * scale, new_bias = bias - mean * new_scale,
///                y = x * new_scale + new_bias
///
/// Supports planar (NCHW/NCDHW) and packed (NCHWC8/NCDHWC8) layouts.
///
/// Only lane=8 SIMD types (v_f32x8 / v_f16x8), matching simd_lane_for<T>.

#include "nnops/detail/simd/simd.hpp"

#include <cmath>

namespace nnops::kernel::batch_norm {

using namespace simd;

// ============================================================
// Packed layout: per-row SIMD fmadd with per-C8-block scale/bias
// ============================================================

template <typename T>
inline void process_packed_row(
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

// ============================================================
// Planar layout: 4-wide unrolled SIMD fmadd per row
// ============================================================

template <typename T>
inline void process_planar_row(
    const T* x_row, T* y_row,
    int64_t last_dim,
    float ns, float nb, bool add_to)
{
    constexpr int L = simd_lane_for<T>;
    int64_t i = 0;

    const auto scale8 = v_set1(x_row, ns);
    const auto bias8  = v_set1(x_row, nb);

    // 4-wide unrolling
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
    // Single remainder
    for (; i + L <= last_dim; i += L) {
        auto rv = v_fmadd(scale8, v_load(x_row + i), bias8);
        v_store_add(y_row + i, rv, add_to);
    }
    // Scalar tail
    for (; i < last_dim; ++i) {
        float rv = s_load(&x_row[i]) * ns + nb;
        s_store_add(&y_row[i], rv, add_to);
    }
}

// ============================================================
// Non-spatial mode: per-element SIMD processing
// ============================================================

template <typename T>
inline void process_nonspatial_block(
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

    // 4-wide unrolling
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
    // Single remainder
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
    // Scalar tail
    for (; i < i_end; ++i) {
        float inv_std_val = 1.0f / std::sqrt(s_load(&var[i]) + epsilon);
        float ns = inv_std_val * s_load(&scale[i]);
        float nb_val = s_load(&bias[i]) - s_load(&mean[i]) * ns;
        float rv = s_load(&x[i]) * ns + nb_val;
        s_store_add(&y[i], rv, add_to);
    }
}

}  // namespace nnops::kernel::batch_norm
