#pragma once
/// @file moe.hpp
/// @brief MoE (Mixture-of-Experts) operator — fused router + per-expert FFN.
///
/// Implements the standard sparse Mixture-of-Experts layer used across LLM
/// serving (Mixtral, DeepSeek-MoE/V2/V3, Qwen-MoE, GPT-OSS, Switch Transformer,
/// etc.). It is a *fused* operator that runs, for every token, the selected
/// top-K experts end-to-end:
///
///     scores  = router_probs[token, :]                 (computed externally)
///     ids, ws = top_k(scores, k)                        (select + gather)
///     if normalize_routing_weights: ws = renorm(ws)     (softmax / sum-to-1)
///     for (id, w) in zip(ids, ws):
///         h  = act(  x @ fc1_w[id]^T + fc1_b[id]  )      (per-expert GEMM 1)
///         y += w * ( h @ fc2_w[id]^T + fc2_b[id] )       (per-expert GEMM 2)
///     output[token] = y
///
/// Routing (the gate) is deliberately *external*: the caller computes
/// `router_probs` (typically `linear(gate)` + softmax/sigmoid, or a learned
/// gate) with the existing Linear/Softmax/TopK ops and passes the result in.
/// This keeps MoE composable and mirrors onnxruntime's `com.microsoft::MoE`
/// contrib op, which is the primary integration target of this library.
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

    /// Re-normalize the top-K routing weights to sum to 1 (softmax over the
    /// selected experts). When false, the gathered router_probs values are used
    /// as-is (already-softmaxed logits).
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
///   [0] input                  [num_tokens, hidden_size]     (f32/f16/bf16)
///   [1] router_probs           [num_tokens, num_experts]     (f32/f16/bf16)
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
/// `router_probs` drives top-K *selection*; the gathered values become the
/// mixing weights unless `router_weights` is provided, in which case it supplies
/// the mixing weights at the selected indices (router_probs then only selects).
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

    // ---- OpBase interface ----
    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    /// Workspace (bytes) required by MoE. Non-zero: the op stages the expanded
    /// top-K tokens and per-expert intermediate activations. The exact size is
    /// derived from the shapes in `inputs`/`outputs`.
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
