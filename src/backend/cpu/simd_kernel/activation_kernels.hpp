#pragma once
/// @file activation_kernels.hpp
/// @brief Shared SIMD activation kernels — pure per-vector math, no I/O.
///
/// Used by the standalone Activation operator and the epilogue fusion path
/// (simd_epilogue.hpp). Each kernel takes an input vector and pre-computed
/// constant vectors and returns the result vector.
///
/// Formulas:
///   relu        max(x, 0)
///   leaky_relu  max(x, 0) + alpha * min(x, 0)
///   sigmoid     1 / (1 + exp(-x))
///   tanh        tanh(x)
///   gelu        0.5 * x * (1 + tanh(c * (x + 0.044715*x^3))), c = sqrt(2/pi)
///   silu        x / (1 + exp(-x))
///   hard_swish  x * relu6(x + 3) * (beta / 6)
///   elu         max(x, 0) + alpha * (exp(min(x, 0)) - 1)
///
/// LeakyRelu and Elu use min/max decomposition, not compare+blend, so the same
/// code works across v_f32x8, v_f16x8 and every backend (AVX2, SSE, NEON,
/// scalar). The v_ prefix marks pure vector kernels, vs the tiled I/O wrappers
/// (v_relu vs relu).

#include "nnops/detail/simd/simd.hpp"

namespace nnops::kernel {

using namespace simd;

template <typename V>
inline V v_relu(V x, V vzero) {
    return v_max(x, vzero);
}

template <typename V>
inline V v_leaky_relu(V x, V vzero, V valpha) {
    return v_add(v_max(x, vzero), v_mul(valpha, v_min(x, vzero)));
}

template <typename V>
inline V v_sigmoid(V x, V v1) {
    return v_div(v1, v_add(v1, v_exp(v_neg(x))));
}

template <typename V>
inline V v_tanh(V x) {
    return v_tanh(x);
}

template <typename V>
inline V v_gelu(V x, V v0_5, V v1, V vc, V vcoeff) {
    auto x3 = v_mul(v_mul(x, x), x);
    auto inner = v_mul(vc, v_add(x, v_mul(vcoeff, x3)));
    return v_mul(v_mul(v0_5, x), v_add(v1, v_tanh(inner)));
}

template <typename V>
inline V v_silu(V x, V v1) {
    return v_div(x, v_add(v1, v_exp(v_neg(x))));
}

template <typename V>
inline V v_hard_swish(V x, V vzero, V v3, V v6, V vscale) {
    auto relu6 = v_min(v_max(v_add(x, v3), vzero), v6);
    return v_mul(v_mul(x, relu6), vscale);
}

template <typename V>
inline V v_elu(V x, V vzero, V valpha, V v1) {
    auto neg_part = v_mul(valpha, v_sub(v_exp(v_min(x, vzero)), v1));
    return v_add(v_max(x, vzero), neg_part);
}

}  // namespace nnops::kernel
