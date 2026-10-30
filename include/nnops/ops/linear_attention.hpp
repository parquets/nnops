#pragma once
/// @file linear_attention.hpp
/// @brief LinearAttention operator — gated linear attention with a recurrent
///        state.
///
/// O(n) linear attention (Gated Delta Net / KDA) instead of O(n²) softmax: no
/// KV cache, one fixed [D, D] state per kv head. The op is stateless — the
/// caller owns that state as outputs[1], initializes it once and reuses it each
/// step; the kernel reads the old state and overwrites it in place (with S == 1
/// this is incremental decoding).
///
/// Per (batch, kv head) and token t:
///   1. gate_t[i] = exp(Gate[t, i]);  S[i][:] *= gate_t[i]   decay
///   2. r[j]      = sum_i S[i][j] * k_t[i]                   retrieve (decayed S)
///   3. d[j]      = beta_t * (v_t[j] - r[j])                 delta correction
///   4. S[i][j]  += k_t[i] * d[j]                            rank-1 update
///   5. o_t[j]    = scale * sum_i q_t[i] * S[i][j]           readout
/// Steps 1-4 are per kv head, step 5 per query head: ORT's `update_rule =
/// "gated_delta"`. llama.cpp's ggml_gated_linear_attn omits the delta term.
///
/// Inputs (4 required, 2 optional). The optional slots are *positional*: an
/// omitted input is an empty TensorView, so later entries shift up only when the
/// omission is a suffix.
///   [0] Q [B,H,S,D]   [1] K [B,H_kv,S,D]   [2] V [B,H_kv,S,D]
///   [3] Gate [B,H_kv,S,D] log-space decay  [4] conv_w [C,1,K] optional
///   [5] beta [B,H_kv,S] optional update rate, default 1.0
/// `{Q,K,V,Gate,beta}` does not ask for beta without a conv — beta lands in the
/// conv_w slot; ask with `{Q,K,V,Gate,{},beta}`.
///
/// H_kv = num_kv_heads (or num_heads when 0); C = (H + 2*H_kv) * D, with the
/// conv channels packed Q | K | V and one tap set each (depthwise), shared
/// across the batch. Causal, zeros left of t = 0:
/// y[t] = sum_k conv_w[:,k] * x[t-(K-1)+k]. The conv runs iff conv_w is present,
/// and its tap count must then equal conv_kernel_size (the attribute only
/// checks).
///
/// Gate has H_kv heads, not H: the state belongs to the kv head, so several
/// query heads under GQA share one state and one decay.
///
/// Outputs: [0] [B,H,S,D] result, [1] [B,H_kv,D,D] state.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

struct LinearAttentionAttributes {
    /// Per-head state dimension (S), shared by Q, K, V and Gate.
    int64_t head_dim = 64;

    /// Query heads (H).
    int64_t num_heads = 8;

    /// Key/value heads for GQA. 0 means num_heads (no grouping); otherwise
    /// each kv head is shared by num_heads / num_kv_heads query heads.
    int64_t num_kv_heads = 0;

    /// Attention scale; 0 means 1/sqrt(head_dim).
    float scale = 0.0f;

    /// Causal conv tap count, checked against conv_w — it does not enable the
    /// conv. See the file header: the conv runs iff conv_w is present.
    int64_t conv_kernel_size = 4;
};

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

    Status validate(std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs) const override;

    using OpBase::compute;

    Status compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::LinearAttention; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::PlanarOnly;
    }

    const LinearAttentionAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~LinearAttention();

private:
    LinearAttention(const LinearAttentionAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    LinearAttentionAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
