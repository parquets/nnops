/// @file moe_ref.cpp
/// @brief Naive scalar CPU reference implementation of the MoE layer.
///
/// Mirrors onnxruntime's `com.microsoft::MoE` CPU semantics, and is the
/// correctness oracle for the optimized kernel in `backend/cpu/moe.cpp` — it
/// deliberately shares no code with it.
///
/// Per token: score the router logits over *all* experts with
/// `attrs.router_gating` (softmax by default), take the top-K (ties toward the
/// lower expert index), optionally renormalize and scale the selected weights,
/// then run the selected experts' FFNs and accumulate `w * expert_out` in f32.
///
/// Scope matches the kernel: f32/f16, Relu/Gelu/Silu/Identity, and SwiGLU with
/// the Separate (fc3) layout. Everything else is rejected.

#include "nnops/ops/moe.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <cmath>
#include <vector>

namespace nnops::backend::cpu::reference {
namespace {

using nnops::backend::cpu::half;

/// Absent optional inputs arrive either as an omitted argument or as an empty
/// TensorView (data == nullptr) — treat both as "not provided".
inline bool present(std::span<const TensorView> inputs, size_t i) noexcept
{
    return i < inputs.size() && !inputs[i].is_empty();
}

/// Scalar element read through the simd load helper, so `float` and `half` share
/// one code path (half is __fp16 on aarch64, a struct elsewhere).
template <typename T>
inline float load_at(const T* base, int64_t i) noexcept
{
    return simd::s_load(base + i);
}

// ---- scalar activations (must match the kernel's SIMD forms) ----

inline float apply_activation(MoEActivation act, float x, float alpha, float beta) noexcept
{
    switch (act) {
    case MoEActivation::Relu:
        return x > 0.0f ? x : 0.0f;
    case MoEActivation::Gelu:
        // tanh approximation, identical to kernel::v_gelu / simd_activation.
        return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f *
                                            (x + 0.044715f * x * x * x)));
    case MoEActivation::Silu:
        return x / (1.0f + std::exp(-x));
    case MoEActivation::Identity:
        (void)alpha; (void)beta;
        return x;
    default:
        break;
    }
    NNOPS_ASSERT(!"moe_ref: unsupported activation");
    return x;
}

/// Router score for a single logit, for the gating functions that are
/// pointwise. Softmax normalizes across the row, so the caller computes it in
/// full — deliberately the same split as the optimized kernel, but written
/// independently.
///
/// Softplus guards at 20 because log1p(exp(x)) would overflow just above it;
/// below the guard it loses nothing. This mirrors llama.cpp's
/// `ggml_compute_softplus_f32`.
inline float gating_sigmoid(float x) noexcept
{
    return 1.0f / (1.0f + std::exp(-x));
}

inline float gating_sqrt_softplus(float x) noexcept
{
    return std::sqrt(x > 20.0f ? x : std::log1p(std::exp(x)));
}

/// SwiGLU value/gate combine, matching moe.hpp's documented formula:
///   h = gate * sigmoid(alpha * gate) * (value + beta)
/// with both halves clamped to +/- limit first when limit is finite.
inline float swiglu_combine(float gate, float value, float alpha, float beta,
                            float limit) noexcept
{
    if (std::isfinite(limit)) {
        gate  = gate  < -limit ? -limit : (gate  > limit ? limit : gate);
        value = value < -limit ? -limit : (value > limit ? limit : value);
    }
    const float sig = 1.0f / (1.0f + std::exp(-alpha * gate));
    return gate * sig * (value + beta);
}

template <typename T>
void moe_ref_impl(const MoEAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs)
{
    const auto& input        = inputs[0];
    const auto& router_probs = inputs[1];
    const auto& fc1_w        = inputs[2];
    const auto& fc2_w        = inputs[4];

    const bool has_fc1_bias = present(inputs, 3);
    const bool has_fc2_bias = present(inputs, 5);
    const bool has_fc3_w    = present(inputs, 6);
    const bool has_fc3_bias = present(inputs, 7);
    const bool has_rweights = present(inputs, 8);

    const int64_t H = input.shape(input.rank() - 1);
    const int64_t num_tokens = input.numel() / H;
    const int64_t E = router_probs.shape(1);
    const int64_t I = fc2_w.shape(2);

    const bool is_swiglu = (attrs.activation == MoEActivation::SwiGLU);
    if (is_swiglu) {
        NNOPS_ASSERT_MSG(attrs.swiglu_layout == SwiGLULayout::Separate,
                         "moe_ref: only the Separate SwiGLU layout is supported");
        NNOPS_ASSERT(has_fc3_w);
    }
    NNOPS_ASSERT(!attrs.use_sparse_mixer);
    NNOPS_ASSERT(attrs.k >= 1 && attrs.k <= E);

    const int64_t k = attrs.k;
    const int64_t fc1_out = fc1_w.shape(1);

    const auto* in_ptr   = input.ptr<T>();
    const auto* rout_ptr = router_probs.ptr<T>();
    const T* fc1_ptr     = fc1_w.ptr<T>();
    const T* fc2_ptr     = fc2_w.ptr<T>();
    const T* fc1b_ptr    = has_fc1_bias ? inputs[3].ptr<T>() : nullptr;
    const T* fc2b_ptr    = has_fc2_bias ? inputs[5].ptr<T>() : nullptr;
    const T* fc3_ptr     = has_fc3_w ? inputs[6].ptr<T>() : nullptr;
    const T* fc3b_ptr    = has_fc3_bias ? inputs[7].ptr<T>() : nullptr;
    const T* rw_ptr      = has_rweights ? inputs[8].ptr<T>() : nullptr;

    const int64_t in_row_stride   = input.row_stride_elems();
    const int64_t rout_row_stride = router_probs.row_stride_elems();
    const int64_t rw_row_stride   = has_rweights ? inputs[8].row_stride_elems() : 0;
    const int64_t out_row_stride  = output.row_stride_elems();

    std::vector<float> probs(static_cast<size_t>(E));
    std::vector<int64_t> ids(static_cast<size_t>(k));
    std::vector<float> weights(static_cast<size_t>(k));
    std::vector<float> gate(static_cast<size_t>(fc1_out));
    std::vector<float> value(static_cast<size_t>(is_swiglu ? I : 0));
    std::vector<char> used(static_cast<size_t>(E));

    for (int64_t t = 0; t < num_tokens; ++t) {
        const T* logits = rout_ptr + t * rout_row_stride;

        // ---- score the logits over all experts (f32) ----
        switch (attrs.router_gating) {
        case MoERouterGating::Softmax: {
            // Max-subtracted for stability; the softmax normalizes the row.
            float max_logit = load_at(logits, 0);
            for (int64_t e = 1; e < E; ++e) {
                const float v = load_at(logits, e);
                max_logit = v > max_logit ? v : max_logit;
            }
            float sum_exp = 0.0f;
            for (int64_t e = 0; e < E; ++e) {
                probs[static_cast<size_t>(e)] = std::exp(load_at(logits, e) - max_logit);
                sum_exp += probs[static_cast<size_t>(e)];
            }
            const float inv_sum = 1.0f / sum_exp;
            for (int64_t e = 0; e < E; ++e) {
                probs[static_cast<size_t>(e)] *= inv_sum;
            }
        } break;
        case MoERouterGating::Sigmoid:
            // Pointwise, and deliberately not max-subtracted: shifting the
            // logits would change the scores rather than rescale them.
            for (int64_t e = 0; e < E; ++e) {
                probs[static_cast<size_t>(e)] = gating_sigmoid(load_at(logits, e));
            }
            break;
        case MoERouterGating::SqrtSoftplus:
            for (int64_t e = 0; e < E; ++e) {
                probs[static_cast<size_t>(e)] = gating_sqrt_softplus(load_at(logits, e));
            }
            break;
        }

        // ---- top-K selection: scan ascending with a strict '>', so an exact
        // tie keeps the lower expert index ----
        for (int64_t e = 0; e < E; ++e) { used[static_cast<size_t>(e)] = 0; }
        for (int64_t j = 0; j < k; ++j) {
            int64_t best = -1;
            for (int64_t e = 0; e < E; ++e) {
                if (used[static_cast<size_t>(e)]) { continue; }
                if (best < 0 || probs[static_cast<size_t>(e)] > probs[static_cast<size_t>(best)]) {
                    best = e;
                }
            }
            used[static_cast<size_t>(best)] = 1;
            ids[static_cast<size_t>(j)] = best;
            weights[static_cast<size_t>(j)] = has_rweights
                ? load_at(rw_ptr + t * rw_row_stride, best)
                : probs[static_cast<size_t>(best)];
        }

        if (attrs.normalize_routing_weights) {
            float wsum = 0.0f;
            for (int64_t j = 0; j < k; ++j) { wsum += weights[static_cast<size_t>(j)]; }
            const float inv_wsum = (wsum != 0.0f) ? 1.0f / wsum : 0.0f;
            for (int64_t j = 0; j < k; ++j) { weights[static_cast<size_t>(j)] *= inv_wsum; }
        }

        // Routed scaling, applied last so it composes with the renormalization.
        if (attrs.routed_scaling_factor != 1.0f) {
            for (int64_t j = 0; j < k; ++j) {
                weights[static_cast<size_t>(j)] *= attrs.routed_scaling_factor;
            }
        }

        const T* x = in_ptr + t * in_row_stride;

        // ---- run the selected experts, accumulating in f32 ----
        std::vector<float> acc(static_cast<size_t>(H), 0.0f);
        for (int64_t j = 0; j < k; ++j) {
            const int64_t e = ids[static_cast<size_t>(j)];
            const float w = weights[static_cast<size_t>(j)];

            const T* w1 = fc1_ptr + e * fc1_out * H;
            const T* b1 = has_fc1_bias ? fc1b_ptr + e * fc1_out : nullptr;
            const T* w3 = has_fc3_w ? fc3_ptr + e * I * H : nullptr;
            const T* b3 = has_fc3_bias ? fc3b_ptr + e * I : nullptr;

            // FC1 (gate), plus FC3 (value) for SwiGLU.
            for (int64_t i = 0; i < fc1_out; ++i) {
                float s = b1 ? load_at(b1, i) : 0.0f;
                const T* row = w1 + i * H;
                for (int64_t h = 0; h < H; ++h) {
                    s += load_at(x, h) * load_at(row, h);
                }
                gate[static_cast<size_t>(i)] = s;

                if (is_swiglu) {
                    float v = b3 ? load_at(b3, i) : 0.0f;
                    const T* row3 = w3 + i * H;
                    for (int64_t h = 0; h < H; ++h) {
                        v += load_at(x, h) * load_at(row3, h);
                    }
                    value[static_cast<size_t>(i)] = v;
                }
            }

            if (is_swiglu) {
                // Separate layout: fc1_out == I, so gate/value align elementwise.
                for (int64_t i = 0; i < I; ++i) {
                    gate[static_cast<size_t>(i)] = swiglu_combine(
                        gate[static_cast<size_t>(i)], value[static_cast<size_t>(i)],
                        attrs.activation_alpha, attrs.activation_beta, attrs.swiglu_limit);
                }
            } else {
                for (int64_t i = 0; i < fc1_out; ++i) {
                    gate[static_cast<size_t>(i)] = apply_activation(
                        attrs.activation, gate[static_cast<size_t>(i)],
                        attrs.activation_alpha, attrs.activation_beta);
                }
            }

            // FC2 (down projection) and the weighted accumulate.
            const T* w2 = fc2_ptr + e * H * I;
            const T* b2 = has_fc2_bias ? fc2b_ptr + e * H : nullptr;
            for (int64_t n = 0; n < H; ++n) {
                float s = b2 ? load_at(b2, n) : 0.0f;
                const T* row = w2 + n * I;
                for (int64_t i = 0; i < I; ++i) {
                    s += gate[static_cast<size_t>(i)] * load_at(row, i);
                }
                acc[static_cast<size_t>(n)] += w * s;
            }
        }

        T* out_row = output.ptr<T>() + t * out_row_stride;
        for (int64_t n = 0; n < H; ++n) {
            simd::s_store(out_row + n, acc[static_cast<size_t>(n)]);
        }
    }
}

}  // anonymous namespace

void moe_ref(const MoEAttributes& attrs,
             TensorView& output,
             std::span<const TensorView> inputs,
             const ComputeContext& /*ctx*/,
             void* /*workspace*/)
{
    switch (inputs[0].data_type()) {
    case DataType::f32: moe_ref_impl<float>(attrs, output, inputs); return;
    case DataType::f16: moe_ref_impl<half>(attrs, output, inputs);  return;
    default: break;
    }
    NNOPS_ASSERT(!"moe_ref: unsupported data type (only f32 and f16)");
}

}  // namespace nnops::backend::cpu::reference
