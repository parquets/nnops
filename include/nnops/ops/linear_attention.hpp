#pragma once
/// @file linear_attention.hpp
/// @brief LinearAttention operator — gated linear attention with recurrent state.
///
/// Implements O(n) linear attention (Gated Delta Net / KDA) as an alternative
/// to O(n²) softmax attention. Instead of an explicit KV-cache, a fixed-size
/// recurrent hidden state matrix [head_dim, head_dim] per head is maintained
/// and updated at each step.
///
/// The operator is stateless: all recurrent state memory is owned by the caller
/// and passed as input/output tensors.
///
/// References: onnxruntime com.microsoft::LinearAttention (the `update_rule`
/// modes, of which this op implements `gated_delta` — ORT's default), and
/// llama.cpp ggml_gated_linear_attn (ggml.h:2523) / ggml_gated_delta_net.
///
/// @note The default `beta` (1.0) makes this a *gated delta* rule with a
/// `(V - K@S)` correction term. That is NOT the same as llama.cpp's
/// ggml_gated_linear_attn, which has no such term; the two differ by
/// `beta * K^T @ (K @ S)`.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes for the LinearAttention (gated linear attention) operator.
///
/// Static configuration set at operator creation time. Dynamic per-step
/// values (token count, sequence count) are derived from input shapes.
struct LinearAttentionAttributes {
    /// Per-head state dimension (S). Same dimension for Q, K, V, and Gate.
    /// This is the key dimension that determines the recurrent state size.
    int64_t head_dim = 64;

    /// Number of query attention heads (H).
    int64_t num_heads = 8;

    /// Number of key/value heads for GQA (Grouped Query Attention).
    /// When 0 (default), equals num_heads (standard, no grouping).
    /// When < num_heads, each KV head is shared by num_heads/num_kv_heads query heads.
    int64_t num_kv_heads = 0;

    /// Scaling factor. If 0, auto-computed as 1/sqrt(head_dim).
    float scale = 0.0f;

    /// Short causal convolution kernel size applied to Q, K, V before the
    /// recurrent update. Set to 0 to disable.
    ///
    /// @note This is only *checked* against the supplied convolution weights —
    /// it does not turn the convolution on. The causal conv runs iff the
    /// optional `conv_w` input is present, in which case its tap count must
    /// equal this attribute. Omitting `conv_w` skips the convolution entirely,
    /// whatever this value is (so the 4-input form of the op stays valid).
    int64_t conv_kernel_size = 4;
};

/// Gated Linear Attention with recurrent state (class-based API).
///
/// Per (batch, kv head), carrying a state matrix S[D, D] over the sequence,
/// every token t runs a single fused pass:
///
///   1. gate_t[i] = exp(Gate[t, i])            apply the decay to S row-wise
///      S[i][:]  *= gate_t[i]
///   2. r[j]      = sum_i S[i][j] * k_t[i]     retrieval, on the *decayed* S
///   3. d[j]      = beta_t * (v_t[j] - r[j])   delta correction
///   4. S[i][j]  += k_t[i] * d[j]              rank-1 update
///   5. o_t[j]    = scale * sum_i q_t[i] * S[i][j]   read out with Q
///
/// Q/K/V are the (optionally causal-conv'd) rows; steps 1-4 are shared by every
/// query head in a GQA group, step 5 runs once per query head. This is exactly
/// ORT's `update_rule = "gated_delta"`: substituting `S' = exp(g)·S` gives
/// `S_new = S*exp(gate) + K^T * (V - K*S) * beta`, i.e. the `K @ S` term is
/// evaluated on the decayed state.
///
/// `scale` defaults to `1/sqrt(head_dim)` when the attribute is 0.
///
/// Unlike CausalAttention which stores all past K/V in a growing cache,
/// LinearAttention maintains a fixed-size recurrent state matrix [D, D] per head.
/// State is placed in outputs (like K_cache / V_cache in CausalAttention):
/// the caller initializes it once (to zeros or a checkpoint), then passes the
/// same buffer every step. The kernel reads the old state and overwrites it
/// with the new state in-place.
///
/// The sequence is processed whole: with `S == 1` and a carried State, the same
/// formulation degenerates to single-token incremental decoding.
///
/// Inputs (4 required, 2 optional). The optional slots are *positional*: an
/// omitted input is marked by an empty TensorView, and everything after it
/// shifts up only when the omission is a suffix.
///   inputs[0] = Q      [B, H,    S, D]    query
///   inputs[1] = K      [B, H_kv, S, D]    key
///   inputs[2] = V      [B, H_kv, S, D]    value
///   inputs[3] = Gate   [B, H_kv, S, D]    decay/forget gate, in log space
///   inputs[4] = conv_w [C, 1, K]          optional; causal depthwise conv1d
///   inputs[5] = beta   [B, H_kv, S]       optional; update rate, default 1.0
///
/// So `{Q,K,V,Gate,beta}` is *not* a valid way to ask for beta without a
/// convolution: beta would land in the conv_w slot and be read as tap weights.
/// Supply beta as the sixth entry with an empty TensorView in the fifth:
/// `{Q,K,V,Gate,{},beta}`. Dropping conv_w is only unambiguous when beta is
/// dropped with it.
///
///   where H_kv = num_kv_heads (or num_heads when the attribute is 0) and
///   C = (H + 2*H_kv) * D. The convolution channels are packed Q | K | V, in
///   that order, so channels [0, H*D) are Q, [H*D, H*D + H_kv*D) are K, and the
///   rest are V. Each channel gets its own K taps (depthwise); the weights are
///   shared across the batch. The conv is causal, with zeros left of t = 0:
///   y[t] = sum_{k=0}^{K-1} conv_w[:, k] * x[t - (K-1) + k].
///
///   Note the Gate head count is H_kv, not H: the recurrent state belongs to
///   the kv head, so the decay must too — under GQA several query heads share
///   one state and therefore one decay.
///
/// Outputs (2):
///   outputs[0] = Output [B, H,    S, D]   attention result
///   outputs[1] = State  [B, H_kv, D, D]   recurrent state (read then overwritten)
class LinearAttention : public OpBase {
public:
    static std::unique_ptr<LinearAttention> create(
        const LinearAttentionAttributes& attrs,
        Backend backend = Backend::CPU);

    static std::unique_ptr<LinearAttention> create(Backend backend = Backend::CPU) {
        return create(LinearAttentionAttributes{}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::LinearAttention; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::PlanarOnly;
    }

    const LinearAttentionAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in linear_attention.cpp (Pimpl pattern)
    ~LinearAttention();

private:
    LinearAttention(const LinearAttentionAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    LinearAttentionAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
