#pragma once
/// @file activation_kernels.hpp
/// @brief Shared SIMD activation kernels — pure per-vector math, no I/O.
///
/// Used by both the standalone Activation operator (tiled_activation.hpp) and
/// the epilogue fusion path (epilogue_impl.hpp). Each kernel takes an input
/// vector and pre-computed constant vectors, returns the result vector.
///
/// All kernels use min/max decomposition for conditional activation types
/// (LeakyRelu, Elu) — no comparison+blend required, so they work uniformly
/// across v_f32x8, v_f16x8, and all backends (AVX2, SSE, NEON, scalar).
///
/// Formulas match activation_ref.cpp exactly.

#include "nnops/detail/simd/simd.hpp"

namespace nnops::kernel {

using namespace simd;

// ---- Relu: max(x, 0) ----
template <typename V>
inline V relu(V x, V vzero) {
    return v_max(x, vzero);
}

// ---- LeakyRelu: max(x, 0) + alpha * min(x, 0) ----
template <typename V>
inline V leaky_relu(V x, V vzero, V valpha) {
    return v_add(v_max(x, vzero), v_mul(valpha, v_min(x, vzero)));
}

// ---- Sigmoid: 1 / (1 + exp(-x)) ----
template <typename V>
inline V sigmoid(V x, V v1) {
    return v_div(v1, v_add(v1, v_exp(v_neg(x))));
}

// ---- Tanh ----
template <typename V>
inline V tanh(V x) {
    return v_tanh(x);
}

// ---- GELU: 0.5 * x * (1 + tanh(c * (x + 0.044715 * x^3)))
//           where c = sqrt(2/pi) ----
template <typename V>
inline V gelu(V x, V v0_5, V v1, V vc, V vcoeff) {
    auto x3 = v_mul(v_mul(x, x), x);
    auto inner = v_mul(vc, v_add(x, v_mul(vcoeff, x3)));
    return v_mul(v_mul(v0_5, x), v_add(v1, v_tanh(inner)));
}

// ---- SiLU (Swish): x / (1 + exp(-x)) ----
template <typename V>
inline V silu(V x, V v1) {
    return v_div(x, v_add(v1, v_exp(v_neg(x))));
}

// ---- HardSwish: x * relu6(x + 3) * (beta / 6) ----
template <typename V>
inline V hard_swish(V x, V vzero, V v3, V v6, V vscale) {
    auto relu6 = v_min(v_max(v_add(x, v3), vzero), v6);
    return v_mul(v_mul(x, relu6), vscale);
}

// ---- ELU: max(x, 0) + alpha * (exp(min(x, 0)) - 1) ----
template <typename V>
inline V elu(V x, V vzero, V valpha, V v1) {
    auto neg_part = v_mul(valpha, v_sub(v_exp(v_min(x, vzero)), v1));
    return v_add(v_max(x, vzero), neg_part);
}

}  // namespace nnops::kernel
