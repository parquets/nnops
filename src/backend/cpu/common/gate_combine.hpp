#pragma once
/// @file gate_combine.hpp
/// @brief The gated-FFN combine shared by Linear's gated form and MoE's experts.
///
///   out = act(gate) · (value + beta)      both halves clamped to ±limit
///
/// `act` is selected at compile time through EpilogueActivateType, so the hot
/// loop carries no branch: Silu is `gate · sigmoid(alpha·gate)` (the SwiGLU
/// form, alpha 1 for standard SwiGLU and 1.702 for GPT-OSS), Gelu is the tanh
/// approximation GeGLU uses, and Relu / Sigmoid / Tanh / None cover the
/// remaining gated variants. alpha enters only Silu, as it always has.
///
/// Two layouts: a tile over two separately stored halves (Separate, and
/// Linear's [gate | value] column split), and the Interleaved pairs form where
/// one row holds (gate_0, value_0, gate_1, value_1, ...). `out` may alias
/// `gate`: each vector is loaded before it is stored.
///
/// gate_scale_row is the scale-only half, `y *= act(g)`, for the fused forms
/// whose value is already written: RMSNormGated and the attention output gate.

#include "nnops/core/epilogue.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "backend/cpu/simd_kernel/activation_kernels.hpp"
#include "backend/cpu/simd_kernel/tiled_map.hpp"

namespace nnops::backend::cpu {

/// True for the gated activations this helper implements.
inline bool gate_activation_supported(EpilogueActivateType a) noexcept
{
    switch (a) {
    case EpilogueActivateType::Silu:
    case EpilogueActivateType::Gelu:
    case EpilogueActivateType::Relu:
    case EpilogueActivateType::Sigmoid:
    case EpilogueActivateType::Tanh:
    case EpilogueActivateType::None:
        return true;
    default:
        return false;
    }
}

template <EpilogueActivateType Act, class V, class T>
inline V gate_act_vec(V g, const T* tag, float alpha)
{
    using namespace simd;
    const V vone = v_set1(tag, 1.0f);
    if constexpr (Act == EpilogueActivateType::Silu) {
        // gate · sigmoid(alpha·gate) — note this is not silu(alpha·gate), whose
        // extra alpha factor would change every existing alpha != 1 call.
        return v_mul(g, kernel::v_sigmoid(v_mul(g, v_set1(tag, alpha)), vone));
    } else if constexpr (Act == EpilogueActivateType::Gelu) {
        return kernel::v_gelu(g, v_set1(tag, 0.5f), vone,
                              v_set1(tag, 0.7978845608028654f),
                              v_set1(tag, 0.044715f));
    } else if constexpr (Act == EpilogueActivateType::Relu) {
        return kernel::v_relu(g, v_zero(tag));
    } else if constexpr (Act == EpilogueActivateType::Sigmoid) {
        return kernel::v_sigmoid(g, vone);
    } else if constexpr (Act == EpilogueActivateType::Tanh) {
        return kernel::v_tanh(g);
    } else {
        return g;
    }
}

/// Scalar twin of gate_act_vec.
template <EpilogueActivateType Act>
inline float gate_act_scalar(float g, float alpha)
{
    (void)alpha;  // only Silu reads it
    if constexpr (Act == EpilogueActivateType::Silu) {
        return g / (1.0f + std::exp(-alpha * g));
    } else if constexpr (Act == EpilogueActivateType::Gelu) {
        return 0.5f * g * (1.0f + std::tanh(0.7978845608028654f
                                            * (g + 0.044715f * g * g * g)));
    } else if constexpr (Act == EpilogueActivateType::Relu) {
        return g > 0.0f ? g : 0.0f;
    } else if constexpr (Act == EpilogueActivateType::Sigmoid) {
        return 1.0f / (1.0f + std::exp(-g));
    } else if constexpr (Act == EpilogueActivateType::Tanh) {
        return std::tanh(g);
    } else {
        return g;
    }
}

/// y_i *= act(g_i) over one row — the scale-only half of the fused forms:
/// RMSNormGated (the norm kernel has just written the row, so it is still in
/// cache) and the attention output gate. Callers hand it one row at a time, so
/// the parallelism is theirs.
template <EpilogueActivateType Act, typename T>
inline void gate_scale_row(T* y, const T* g, int64_t n, float alpha = 1.0f)
{
    using namespace simd;
    constexpr int L = simd_lane_for<T>;
    int64_t i = 0;
    for (; i + L <= n; i += L) {
        const auto gv = gate_act_vec<Act>(v_load(g + i), g, alpha);
        v_store(y + i, v_mul(v_load(y + i), gv));
    }
    for (; i < n; ++i) {
        s_store(&y[i], s_load(&y[i]) * gate_act_scalar<Act>(s_load(&g[i]), alpha));
    }
}

template <typename T>
inline void gate_scale_row_dispatch(EpilogueActivateType act, T* y, const T* g,
                                    int64_t n, float alpha = 1.0f)
{
#define NNOPS_GATE_CASE(name)                                                  \
    case EpilogueActivateType::name:                                           \
        gate_scale_row<EpilogueActivateType::name>(y, g, n, alpha);            \
        return
    switch (act) {
    NNOPS_GATE_CASE(Silu);
    NNOPS_GATE_CASE(Gelu);
    NNOPS_GATE_CASE(Relu);
    NNOPS_GATE_CASE(Sigmoid);
    NNOPS_GATE_CASE(Tanh);
    default:
        gate_scale_row<EpilogueActivateType::None>(y, g, n, alpha);
        return;
    }
#undef NNOPS_GATE_CASE
}

template <EpilogueActivateType Act, typename T>
inline void gate_combine_tile(const T* gate, const T* value, T* out,
                              int64_t m, int64_t n,
                              int64_t gate_pitch, int64_t value_pitch,
                              int64_t out_pitch,
                              float alpha, float beta, float limit)
{
    using namespace simd;
    const auto vlim  = v_set1(gate, limit);
    const auto vnlim = v_neg(vlim);
    const auto vbeta = v_set1(gate, beta);
    const bool clamp = std::isfinite(limit);

    auto vec = [=](auto g, auto v) {
        if (clamp) {
            g = v_max(v_min(g, vlim), vnlim);
            v = v_max(v_min(v, vlim), vnlim);
        }
        return v_mul(gate_act_vec<Act>(g, gate, alpha), v_add(v, vbeta));
    };
    auto scalar = [=](float g, float v) {
        if (clamp) {
            g = g < -limit ? -limit : (g > limit ? limit : g);
            v = v < -limit ? -limit : (v > limit ? limit : v);
        }
        return gate_act_scalar<Act>(g, alpha) * (v + beta);
    };
    kernel::tiled_map_simd<T, 2>({gate, value}, {gate_pitch, value_pitch},
                                 out, out_pitch, m, n, false, vec, scalar);
}

/// Interleaved form: one row holds (gate_0, value_0, gate_1, value_1, ...), so
/// the two operands are the even and odd lanes of one row and the result is
/// written compact.
template <EpilogueActivateType Act, typename T>
inline void gate_combine_pairs(const T* pairs, T* out, int64_t m, int64_t n,
                               int64_t pairs_pitch, int64_t out_pitch,
                               float alpha, float beta, float limit)
{
    using V = kernel::simd_vec_t<T>;
    constexpr int64_t L = simd::simd_lane_for<T>;
    using namespace simd;
    const V vlim   = v_set1(out, limit);
    const V vnlim  = v_neg(vlim);
    const V vbeta  = v_set1(out, beta);
    const bool clamp = std::isfinite(limit);

    for (int64_t r = 0; r < m; ++r) {
        const T* pr = pairs + r * pairs_pitch;
        T* orow = out + r * out_pitch;
        int64_t c = 0;
        for (; c + L <= n; c += L) {
            auto g = v_load_even(pr + 2 * c);
            auto v = v_load_odd(pr + 2 * c);
            if (clamp) {
                g = v_max(v_min(g, vlim), vnlim);
                v = v_max(v_min(v, vlim), vnlim);
            }
            v_store(orow + c,
                    v_mul(gate_act_vec<Act>(g, out, alpha), v_add(v, vbeta)));
        }
        for (; c < n; ++c) {
            float g = s_load(pr + 2 * c);
            float v = s_load(pr + 2 * c + 1);
            if (clamp) {
                g = g < -limit ? -limit : (g > limit ? limit : g);
                v = v < -limit ? -limit : (v > limit ? limit : v);
            }
            s_store(orow + c, gate_act_scalar<Act>(g, alpha) * (v + beta));
        }
    }
}

/// Runtime activation → the compile-time kernels above.
template <typename T>
inline void gate_combine_tile_dispatch(EpilogueActivateType act,
                                       const T* gate, const T* value, T* out,
                                       int64_t m, int64_t n,
                                       int64_t gate_pitch, int64_t value_pitch,
                                       int64_t out_pitch,
                                       float alpha, float beta, float limit)
{
#define NNOPS_GATE_CASE(name)                                                  \
    case EpilogueActivateType::name:                                           \
        gate_combine_tile<EpilogueActivateType::name>(                          \
            gate, value, out, m, n, gate_pitch, value_pitch, out_pitch,         \
            alpha, beta, limit);                                                \
        return
    switch (act) {
    NNOPS_GATE_CASE(Silu);
    NNOPS_GATE_CASE(Gelu);
    NNOPS_GATE_CASE(Relu);
    NNOPS_GATE_CASE(Sigmoid);
    NNOPS_GATE_CASE(Tanh);
    default:
        gate_combine_tile<EpilogueActivateType::None>(
            gate, value, out, m, n, gate_pitch, value_pitch, out_pitch,
            alpha, beta, limit);
        return;
    }
#undef NNOPS_GATE_CASE
}

template <typename T>
inline void gate_combine_pairs_dispatch(EpilogueActivateType act,
                                        const T* pairs, T* out,
                                        int64_t m, int64_t n,
                                        int64_t pairs_pitch, int64_t out_pitch,
                                        float alpha, float beta, float limit)
{
#define NNOPS_GATE_CASE(name)                                                  \
    case EpilogueActivateType::name:                                           \
        gate_combine_pairs<EpilogueActivateType::name>(                         \
            pairs, out, m, n, pairs_pitch, out_pitch, alpha, beta, limit);      \
        return
    switch (act) {
    NNOPS_GATE_CASE(Silu);
    NNOPS_GATE_CASE(Gelu);
    NNOPS_GATE_CASE(Relu);
    NNOPS_GATE_CASE(Sigmoid);
    NNOPS_GATE_CASE(Tanh);
    default:
        gate_combine_pairs<EpilogueActivateType::None>(
            pairs, out, m, n, pairs_pitch, out_pitch, alpha, beta, limit);
        return;
    }
#undef NNOPS_GATE_CASE
}

}  // namespace nnops::backend::cpu
