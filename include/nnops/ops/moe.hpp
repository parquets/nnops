#pragma once
/// @file moe.hpp
/// @brief MoE (Mixture-of-Experts) operator — fused router + per-expert FFN.
///
/// Implements the standard sparse Mixture-of-Experts layer used across LLM
/// serving (Mixtral, DeepSeek-MoE/V2/V3, Qwen-MoE, GPT-OSS, Switch Transformer,
/// etc.). It is a *fused* operator that runs, for every token, the selected
/// top-K experts end-to-end:
///
///     p       = gating(router_probs[token, :])          (over all E experts)
///     ids, ws = top_k(p, k)                             (select + gather)
///     if normalize_routing_weights: ws = ws / sum(ws)   (renormalize the k)
///     if routed_scaling_factor:     ws = ws * factor
///     for (id, w) in zip(ids, ws):
///         h  = act(  x @ fc1_w[id]^T + fc1_b[id]  )      (per-expert GEMM 1)
///         y += w * ( h @ fc2_w[id]^T + fc2_b[id] )       (per-expert GEMM 2)
///     output[token] = y
///
/// Routing (the gate projection) is deliberately *external*: the caller computes
/// `router_probs` (typically `linear(gate)` — raw logits) with the existing
/// Linear op and passes it in. The scoring over all experts and the top-K select
/// happen *inside* this op, matching onnxruntime's `com.microsoft::MoE` contrib
/// op, which is the primary integration target of this library.
///
/// The scoring function is selectable (`router_gating`). onnxruntime has no such
/// attribute — it always softmaxes — so this one follows llama.cpp's
/// `expert_gating_func`, which is what lets the op cover DeepSeek-V3 / Llama-4
/// style sigmoid routing. `Softmax` is the default and matches ORT.
///
/// Each expert's FFN is a two-(or three-)layer MLP whose weights are stacked
/// along a leading expert axis (shape[0] == num_experts). The inner per-expert
/// GEMMs reuse the library's existing GEMM micro-kernels (and, on CUDA, are
/// intended to map onto grouped-GEMM / CUTLASS grouped kernels like onnxruntime's
/// `CutlassMoeFCRunner`).
///
/// Quantized expert weights (int8/uint8, per-channel or per-token) are supported
/// through each weight tensor's `TensorView::quant_params()` — mirroring how
/// Embed/QuantLinear fold quantization into the tensor rather than into a
/// separate QMoE op. The kernel dequantizes on the fly (or dispatches to a
/// quantized grouped GEMM). 4-bit / fp8 expert weights are future extension
/// points reserved by the DataType (s4/u4/f8_e4m3/f8_e5m2) enum entries.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <limits>
#include <memory>
#include <span>

namespace nnops {

/// Expert-FFN activation (the non-linearity between GEMM 1 and GEMM 2).
///
/// Distinct from the element-wise `ActivationType` (activation.hpp) because it
/// also covers `SwiGLU` (a fused gate×value gated-linear unit, not element-wise)
/// and `Identity` (linear experts), which are MoE-specific.
enum class MoEActivation : uint8_t {
    Relu     = 0,  ///< h = max(0, x)
    Gelu     = 1,  ///< h = x * Phi(x) (Gaussian error linear unit)
    Silu     = 2,  ///< h = x * sigmoid(x)  (Swish; Mixtral/Gemma default)
    Identity = 3,  ///< h = x (linear expert, no activation)
    SwiGLU   = 4,  ///< h = gate * sigmoid(alpha*gate) * (value + beta)  (see below)
};

/// Router scoring function: how the gate logits become the probabilities that
/// top-K selects from. Mirrors llama.cpp's `llama_expert_gating_func_type`
/// (its NONE and SOFTMAX_WEIGHT variants are not carried over — see below).
///
/// Only the *selection* score changes; the top-K tie-break (ascending scan,
/// exact ties resolved toward the lower expert index) is the same for all of
/// them.
enum class MoERouterGating : uint8_t {
    Softmax      = 0,  ///< p = softmax(logits)          (Mixtral, Qwen-MoE)
    Sigmoid      = 1,  ///< p = sigmoid(logits)          (DeepSeek-V3, Llama-4)
    SqrtSoftplus = 2,  ///< p = sqrt(softplus(logits))   (llama.cpp)
};

/// How the Gate / Value halves of a SwiGLU expert are laid out in the expert
/// weight tensors (only meaningful when `activation == SwiGLU`).
///
/// Standard SwiGLU computes `SiLU(gate) * value` where `gate` and `value` are
/// two independent linear projections of the same input, each `inter_size` wide.
/// The three layouts differ only in where those two projections come from.
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

/// Attributes for the MoE operator.
///
/// Only the fields relevant to the selected `activation` are used; the rest are
/// ignored (e.g. `activation_alpha/beta/swiglu_limit` only apply to SwiGLU).
struct MoEAttributes {
    /// Number of experts selected per token (top-K). Must be >= 1 and
    /// <= num_experts (derived from the router_probs / weight shapes at
    /// compute time).
    int64_t k = 1;

    /// Expert-FFN activation. Default `Silu` — the dominant choice for
    /// Mixtral/Gemma-style LLM MoE (onnxruntime's default is `relu`).
    MoEActivation activation = MoEActivation::Silu;

    /// Router scoring function applied to `router_probs` *before* top-K. The
    /// default `Softmax` reproduces the historical behaviour exactly.
    ///
    /// With `Sigmoid` the scores are in (0, 1) and do **not** sum to 1, so the
    /// routed output is systematically smaller than under `Softmax` — by roughly
    /// a factor of `k/2`. That is the definition of the function, not a defect,
    /// but it is why sigmoid routing is almost always paired with
    /// `normalize_routing_weights`. The DeepSeek-V3 combination is exactly:
    ///
    ///     router_gating             = Sigmoid;
    ///     normalize_routing_weights = true;
    ///     routed_scaling_factor     = 2.5f;
    ///
    /// A Debug build warns when a non-softmax gating is left unnormalized (and
    /// no `router_weights` supplies the mixing weights): that is legal, but far
    /// more often a forgotten line than an intentional choice.
    MoERouterGating router_gating = MoERouterGating::Softmax;

    /// Multiplier applied to the routing weights after normalization —
    /// llama.cpp's `w_scale`, DeepSeek-V3's `routed_scaling_factor` (2.5).
    /// Default 1.0 leaves the weights alone. Applied to whichever weights the
    /// routing gathered, so it composes with `router_weights` too.
    float routed_scaling_factor = 1.0f;

    /// Re-normalize the selected routing weights to sum to 1 (divide the top-K
    /// by their own sum). When false, the selected scores are used as-is (the
    /// softmax or sigmoid values, or the gathered `router_weights`). When
    /// `router_weights` (input 8) is present this renormalizes those gathered
    /// weights instead — the selection still comes from `router_probs`.
    ///
    /// Strongly recommended with `Sigmoid` / `SqrtSoftplus` gating, whose
    /// unnormalized scores do not sum to 1 — see `router_gating`.
    bool normalize_routing_weights = false;

    /// Sparse-mixer routing variant (selects k=2 experts and applies the
    /// sparse-mixer mixing rule). Requires k == 2.
    bool use_sparse_mixer = false;

    /// Gate/Value weight layout when `activation == SwiGLU`.
    SwiGLULayout swiglu_layout = SwiGLULayout::Separate;

    /// SwiGLU gate scaling: SwiGLU(x) = gate * sigmoid(alpha * gate) * (value + beta).
    /// Default 1.0 is standard SwiGLU; GPT-OSS uses 1.702.
    float activation_alpha = 1.0f;

    /// SwiGLU value offset. Default 0.0 is standard SwiGLU; GPT-OSS uses 1.0.
    float activation_beta = 0.0f;

    /// SwiGLU clamp limit on gate/value. Default +inf disables clamping;
    /// GPT-OSS uses 7.0.
    float swiglu_limit = std::numeric_limits<float>::infinity();
};

/// Mixture-of-Experts operator (class-based API).
///
/// Inputs (optional inputs may be omitted by passing an empty TensorView,
/// i.e. `is_empty() == true`):
///
///   [0] input                  [num_tokens, hidden_size]     (f32/f16)
///   [1] router_probs           [num_tokens, num_experts]     (f32/f16)
///                              raw router LOGITS — scored inside this op
///                              (see `router_gating`)
///   [2] fc1_experts_weights    [E, fc1_out, hidden_size]     gate (or gate+value)
///   [3] fc1_experts_bias       [E, fc1_out]                  (optional)
///   [4] fc2_experts_weights    [E, hidden_size, inter_size]  down-projection
///   [5] fc2_experts_bias       [E, hidden_size]              (optional)
///   [6] fc3_experts_weights    [E, inter_size, hidden_size]  (optional; SwiGLU
///                                                             Separate value/up)
///   [7] fc3_experts_bias       [E, inter_size]               (optional)
///   [8] router_weights         [num_tokens, num_experts]     (optional; separate
///                                                             mixing weights —
///                                                             DeepSeek noaux_tc)
///
///   where E = num_experts, and:
///     fc1_out = inter_size            (Separate, or non-SwiGLU)
///     fc1_out = 2 * inter_size        (Interleaved / Block SwiGLU)
///
/// `router_gating(router_probs)` drives both the top-K *selection* and the mixing
/// weights, unless `router_weights` is provided, in which case the mixing weights
/// are gathered from it at the selected indices and `router_probs` only selects.
///
/// Implemented today (CPU): f32/f16, all three `router_gating` functions,
/// `routed_scaling_factor`, the Relu/Gelu/Silu/Identity activations, and SwiGLU
/// with the `Separate` layout — the one that takes its value half from fc3. The
/// `Interleaved`/`Block` SwiGLU layouts and `use_sparse_mixer` are part of the
/// interface but not yet implemented; they are rejected by an assertion.
///
/// Output (1):
///   [0] output                    [num_tokens, hidden_size]
///
/// The output layout is PlanarOnly (GEMM weights are planar), consistent with
/// Linear/MatMul/Attention.
class MoE : public OpBase {
public:
    /// Create a MoE operator for the specified backend.
    static std::unique_ptr<MoE> create(const MoEAttributes& attrs = {},
                                       Backend backend = Backend::CPU);

    /// Create with defaults (convenience).
    static std::unique_ptr<MoE> create(Backend backend = Backend::CPU) {
        return create(MoEAttributes{}, backend);
    }

    ~MoE();

    // ---- OpBase interface ----
    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    /// Workspace (bytes) required by MoE. Always 0: the op stages the expanded
    /// top-K tokens and per-expert intermediate activations in its own pooled
    /// scratch, like MatMul/Attention/Conv2D — the caller allocates nothing.
    size_t getWorkspaceSize(std::span<const TensorDesc> inputs,
                            std::span<const TensorDesc> outputs) const override;

    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::MoE; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::PlanarOnly; }

    /// Access the MoE attributes.
    const MoEAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in moe.cpp (Pimpl pattern)

private:
    MoE(const MoEAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    MoEAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
