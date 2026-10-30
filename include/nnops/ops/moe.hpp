#pragma once
/// @file moe.hpp
/// @brief MoE (Mixture-of-Experts) operator — fused router + per-expert FFN.
///
/// Runs, for every token, the selected top-K experts end-to-end (Mixtral,
/// DeepSeek-MoE/V2/V3, Qwen-MoE, GPT-OSS, Switch Transformer):
///
///     p       = gating(router_probs[token, :])        over all E experts
///     ids, ws = top_k(p, k)                           select + gather
///     if normalize_routing_weights: ws = ws / sum(ws)
///     if routed_scaling_factor:     ws = ws * factor
///     for (id, w) in zip(ids, ws):
///         h  = act(  x @ fc1_w[id]^T + fc1_b[id]  )
///         y += w * ( h @ fc2_w[id]^T + fc2_b[id] )
///     output[token] = y
///
/// The gate projection is deliberately external: the caller computes the raw
/// logits (typically linear(gate)) and passes them as router_probs. The scoring
/// over all experts, the top-K select and the mixing weights all happen here,
/// matching onnxruntime's com.microsoft::MoE contrib op. Each expert's FFN is a
/// two-(or three-)layer MLP stacked along a leading expert axis, and the
/// per-expert GEMMs drive the library's MatMul (transpose_b).
///
/// Implemented today (CPU): f32/f16, all three router_gating functions,
/// routed_scaling_factor, the Relu/Gelu/Silu/Identity activations, and SwiGLU
/// with the `Separate` layout. The Interleaved/Block SwiGLU layouts and
/// use_sparse_mixer exist in the interface but are rejected by an assertion.
///
/// Inputs (an optional one is omitted by passing an empty TensorView):
///   [0] input                  [num_tokens, hidden_size]     f32/f16
///   [1] router_probs           [num_tokens, num_experts]     raw LOGITS
///   [2] fc1_experts_weights    [E, fc1_out, hidden_size]     gate (or gate+value)
///   [3] fc1_experts_bias       [E, fc1_out]                  optional
///   [4] fc2_experts_weights    [E, hidden_size, inter_size]  down-projection
///   [5] fc2_experts_bias       [E, hidden_size]              optional
///   [6] fc3_experts_weights    [E, inter_size, hidden_size]  optional; SwiGLU
///                                                            Separate value half
///   [7] fc3_experts_bias       [E, inter_size]               optional
///   [8] router_weights         [num_tokens, num_experts]     optional; separate
///                                                            mixing weights
///
/// E = num_experts; fc1_out = inter_size on Separate / non-SwiGLU, and
/// 2 * inter_size on Interleaved / Block SwiGLU. Output is
/// [num_tokens, hidden_size], PlanarOnly (GEMM weights are planar), consistent
/// with Linear/MatMul/Attention.
///
/// `router_gating(router_probs)` drives both the top-K *selection* and the
/// mixing weights; with `router_weights` present the weights are gathered from
/// it at the selected indices and `router_probs` only selects.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <limits>
#include <memory>
#include <span>

namespace nnops {

/// Expert-FFN activation (the non-linearity between GEMM 1 and GEMM 2), separate
/// from the element-wise `ActivationType` because it also covers the fused
/// gate×value `SwiGLU` and `Identity`.
enum class MoEActivation : uint8_t {
    Relu     = 0,  ///< h = max(0, x)
    Gelu     = 1,  ///< h ≈ x * Phi(x), tanh approximation
    Silu     = 2,  ///< h = x * sigmoid(x)  (Swish; Mixtral/Gemma default)
    Identity = 3,  ///< h = x (linear expert, no activation)
    SwiGLU   = 4,  ///< h = gate * sigmoid(alpha*gate) * (value + beta)  (see below)
};

/// Router scoring: how the gate logits become the scores top-K selects from.
///
/// Mirrors llama.cpp's `llama_expert_gating_func_type` (NONE and
/// SOFTMAX_WEIGHT are not carried over); onnxruntime always softmaxes. Only the
/// *selection* score changes — the tie-break (ascending scan, exact ties to the
/// lower expert index) is the same for all of them.
enum class MoERouterGating : uint8_t {
    Softmax      = 0,  ///< p = softmax(logits)          (Mixtral, Qwen-MoE)
    Sigmoid      = 1,  ///< p = sigmoid(logits)          (DeepSeek-V3, Llama-4)
    SqrtSoftplus = 2,  ///< p = sqrt(softplus(logits))   (llama.cpp)
};

/// Where the two `inter_size`-wide projections of a SwiGLU expert
/// (`SiLU(gate) * value`) come from in the weight tensors; only meaningful when
/// `activation == SwiGLU`.
enum class SwiGLULayout : uint8_t {
    /// Separate weight tensors: gate from fc1, value from fc3 (three GEMMs).
    /// fc1_out = fc3_out = inter_size.
    Separate = 0,

    /// Fused interleaved: fc1 holds [Gate_0, Value_0, Gate_1, Value_1, ...]
    /// along its output axis; fc1_out = 2 * inter_size (GPT-OSS layout).
    Interleaved = 1,

    /// Fused block: fc1 holds [Gate_0..Gate_{N-1} | Value_0..Value_{N-1}]
    /// (concatenated halves); fc1_out = 2 * inter_size (Llama/Gemma layout).
    Block = 2,
};

/// Only the fields relevant to the selected `activation` are used; the rest are
/// ignored.
struct MoEAttributes {
    /// Experts selected per token (top-K), 1 <= k <= num_experts.
    int64_t k = 1;

    /// Default `Silu` — the dominant Mixtral/Gemma choice (ORT's default is
    /// `relu`).
    MoEActivation activation = MoEActivation::Silu;

    /// Scoring applied to `router_probs` before top-K; `Softmax` matches ORT.
    /// `Sigmoid` scores lie in (0, 1) and do **not** sum to 1, so an
    /// unnormalized routed output is systematically scaled down — the definition
    /// of the function, and why sigmoid routing is nearly always paired with
    /// `normalize_routing_weights`. DeepSeek-V3 is exactly Sigmoid +
    /// normalize_routing_weights + routed_scaling_factor 2.5.
    ///
    /// A Debug build warns when a non-softmax gating is left unnormalized and
    /// no `router_weights` supplies the mixing weights.
    MoERouterGating router_gating = MoERouterGating::Softmax;

    /// Multiplier applied to the mixing weights after normalization — llama.cpp's
    /// `w_scale`, DeepSeek-V3's 2.5. Default 1.0 leaves them alone.
    float routed_scaling_factor = 1.0f;

    /// Re-normalize the selected weights to sum to 1. When `router_weights`
    /// (input 8) is present this renormalizes the gathered weights instead; the
    /// selection still comes from `router_probs`. Strongly recommended with
    /// `Sigmoid` / `SqrtSoftplus`, whose scores do not sum to 1.
    bool normalize_routing_weights = false;

    /// Sparse-mixer routing variant (requires k == 2). Not implemented.
    bool use_sparse_mixer = false;

    /// Gate/Value weight layout when `activation == SwiGLU`.
    SwiGLULayout swiglu_layout = SwiGLULayout::Separate;

    /// SwiGLU gate scaling; 1.0 is standard SwiGLU, GPT-OSS uses 1.702.
    float activation_alpha = 1.0f;

    /// SwiGLU value offset; 0.0 is standard SwiGLU, GPT-OSS uses 1.0.
    float activation_beta = 0.0f;

    /// SwiGLU clamp on gate/value; +inf (default) disables it, GPT-OSS uses 7.0.
    float swiglu_limit = std::numeric_limits<float>::infinity();
};

class MoE : public OpBase {
public:
    static std::unique_ptr<MoE> create(const MoEAttributes& attrs = {},
                                       Backend backend = Backend::CPU);

    static std::unique_ptr<MoE> create(Backend backend = Backend::CPU) {
        return create(MoEAttributes{}, backend);
    }

    ~MoE();

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    Status validate(std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs) const override;

    size_t getWorkspaceSize(std::span<const TensorDesc> inputs,
                            std::span<const TensorDesc> outputs) const override;

    using OpBase::compute;

    Status compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::MoE; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::PlanarOnly; }

    const MoEAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;

private:
    MoE(const MoEAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    MoEAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
