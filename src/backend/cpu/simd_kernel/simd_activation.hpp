#pragma once
/// @file simd_activation.hpp
/// @brief Tiled activation SIMD wrappers for m×n sub-matrices.
///
/// Each function wraps a pure SIMD kernel from activation_kernels.hpp,
/// pre-computes vector constants once per call, and delegates the
/// row-by-row SIMD+scalar loop to tiled_unary_simd.
///
/// Only lane=8 SIMD types (v_f32x8 / v_f16x8), matching simd_lane_for<T>.
///
/// Usage:
///   #include "simd_kernel/simd_activation.hpp"
///   using namespace nnops::kernel;
///   relu<float>(in, out, m, n, in_pitch, out_pitch, add_to);

#include "nnops/detail/simd/simd.hpp"
#include "activation_kernels.hpp"
#include "simd_unary.hpp"

#include <algorithm>
#include <cmath>

namespace nnops::kernel {

using namespace simd;

// ============================================================
// Tiled activation kernels — one per activation type
// ============================================================

template <typename T>
inline void relu(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to = false)
{
    auto vzero = v_zero(in);
    tiled_unary_simd(in, out, m, n, in_pitch, out_pitch, add_to,
        [vzero](auto x) { return v_relu(x, vzero); },
        [](float v) { return v > 0.0f ? v : 0.0f; });
}

template <typename T>
inline void leaky_relu(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to, float alpha)
{
    auto vzero  = v_zero(in);
    auto valpha = v_set1(in, alpha);
    tiled_unary_simd(in, out, m, n, in_pitch, out_pitch, add_to,
        [vzero, valpha](auto x) { return v_leaky_relu(x, vzero, valpha); },
        [alpha](float v) { return v > 0.0f ? v : alpha * v; });
}

template <typename T>
inline void sigmoid(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to = false)
{
    auto vone = v_set1(in, 1.0f);
    tiled_unary_simd(in, out, m, n, in_pitch, out_pitch, add_to,
        [vone](auto x) { return v_sigmoid(x, vone); },
        [](float v) { return 1.0f / (1.0f + std::exp(-v)); });
}

template <typename T>
inline void gelu(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to = false)
{
    auto vhalf  = v_set1(in, 0.5f);
    auto vone   = v_set1(in, 1.0f);
    auto vc     = v_set1(in, 0.7978845608028654f);
    auto vcoeff = v_set1(in, 0.044715f);
    tiled_unary_simd(in, out, m, n, in_pitch, out_pitch, add_to,
        [vhalf, vone, vc, vcoeff](auto x) { return v_gelu(x, vhalf, vone, vc, vcoeff); },
        [](float x) {
            return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        });
}

template <typename T>
inline void silu(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to = false)
{
    auto vone = v_set1(in, 1.0f);
    tiled_unary_simd(in, out, m, n, in_pitch, out_pitch, add_to,
        [vone](auto x) { return v_silu(x, vone); },
        [](float x) { return x / (1.0f + std::exp(-x)); });
}

template <typename T>
inline void hard_swish(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to, float beta)
{
    auto vzero  = v_zero(in);
    auto v3     = v_set1(in, 3.0f);
    auto v6     = v_set1(in, 6.0f);
    auto vscale = v_set1(in, beta / 6.0f);
    tiled_unary_simd(in, out, m, n, in_pitch, out_pitch, add_to,
        [vzero, v3, v6, vscale](auto x) { return v_hard_swish(x, vzero, v3, v6, vscale); },
        [beta](float x) { return x * std::min(std::max(x + 3.0f, 0.0f), 6.0f) * (beta / 6.0f); });
}

template <typename T>
inline void elu(
    const T* in, T* out,
    int64_t m, int64_t n,
    int64_t in_pitch, int64_t out_pitch,
    bool add_to, float alpha)
{
    auto vzero  = v_zero(in);
    auto valpha = v_set1(in, alpha);
    auto vone   = v_set1(in, 1.0f);
    tiled_unary_simd(in, out, m, n, in_pitch, out_pitch, add_to,
        [vzero, valpha, vone](auto x) { return v_elu(x, vzero, valpha, vone); },
        [alpha](float x) { return x > 0.0f ? x : alpha * (std::exp(x) - 1.0f); });
}

}  // namespace nnops::kernel
