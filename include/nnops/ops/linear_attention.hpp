#pragma once
/// @file linear_attention.hpp
/// @brief LinearAttention operator — gated linear attention with a recurrent
///        state.
///
/// O(n) linear attention (Gated Delta Net / KDA) instead of O(n²) softmax: no
/// KV cache, one fixed [Dk, Dv] state per value head. The op is stateless — the
/// caller owns that state as outputs[1], initializes it once and reuses it each
/// step; the kernel reads the old state and overwrites it in place (with S == 1
/// this is incremental decoding).
///
/// Per (batch, value head) and token t:
///   1. gate_t[i] = exp(Gate[t, i]);  S[i][:] *= gate_t[i]   decay
///   2. r[j]      = sum_i S[i][j] * k_t[i]                   retrieve (decayed S)
///   3. d[j]      = beta_t * (v_t[j] - r[j])                 delta correction
///   4. S[i][j]  += k_t[i] * d[j]                            rank-1 update
///   5. o_t[j]    = scale * sum_i q_t[i] * S[i][j]           readout
/// Steps 1-4 are per value head, step 5 per query head: ORT's `update_rule =
/// "gated_delta"`. llama.cpp's ggml_gated_linear_attn omits the delta term.
///
/// The CPU kernel runs this two ways: the direct per-token loop, and above a
/// chunk length the equivalent chunked (WY) form, which folds the decay into a
/// cumulative product over the chunk and pays it in two state passes instead of
/// three. Both compute this recurrence — they differ only in how the sum over a
/// chunk is associated, so results agree to f32 rounding and never bitwise.
///
/// The key row (i) is Dk wide — one decayed row of the state per key channel —
/// and the value row (j) is Dv wide. Dk == Dv and H_v == H_k is the plain Gated
/// Delta Net; a narrower value is Kimi-style (D_v != D_k), and more value heads
/// than key heads is Qwen3-Next (H_v = 2·H_k).
///
/// Inputs (4 required, 2 optional). The optional slots are *positional*: an
/// omitted input is an empty TensorView, so later entries shift up only when the
/// omission is a suffix.
///   [0] Q [B,H_q,S,Dk]   [1] K [B,H_k,S,Dk]   [2] V [B,H_v,S,Dv]
///   [3] Gate [B,H_v,S,Dk] log-space decay  [4] conv_w [C,1,K] optional
///   [5] beta [B,H_v,S] optional update rate, default 1.0
/// `{Q,K,V,Gate,beta}` does not ask for beta without a conv — beta lands in the
/// conv_w slot; ask with `{Q,K,V,Gate,{},beta}`.
///
/// H_k = num_kv_heads (or num_heads when 0); H_v and Dv come from V, and must
/// satisfy H_v % H_k == 0. Q is indexed by the value head and so must satisfy
/// H_q % H_v == 0 (one or more query heads reading one state). The state is per
/// value head because that is what the readout and the delta correction update;
/// a query head shared by several states would have several outputs and no place
/// to put them, so the caller repeats Q from the key heads — the same repeat the
/// conv blob needs (FLA's `repeat_interleave`).
///
/// C = (H_q + H_k) * Dk + H_v * Dv, with the conv channels packed Q | K | V and
/// one tap set each (depthwise), shared across the batch. Causal, zeros left of
/// t = 0: y[t] = sum_k conv_w[:,k] * x[t-(K-1)+k]. The conv runs iff conv_w is
/// present, and its tap count must then equal conv_kernel_size (the attribute
/// only checks). The Q block is laid out over the *expanded* query heads, so a
/// blob produced before the key-head-to-value-head repeat must replicate its Q
/// taps.
///
/// Outputs: [0] [B,H_q,S,Dv] result, [1] [B,H_v,Dk,Dv] state.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

struct LinearAttentionAttributes {
    /// Key-query width (Dk): the state's rows and the Gate width. The value
    /// width Dv comes from V, so Dv != Dk needs no other attribute.
    int64_t head_dim = 64;

    /// Query heads (H_q).
    int64_t num_heads = 8;

    /// Key heads (H_k) for GQA. 0 means num_heads (no grouping); otherwise
    /// each key head is shared by H_v / num_kv_heads value heads, and each
    /// value head by num_heads / H_v query heads.
    int64_t num_kv_heads = 0;

    /// Attention scale; 0 means 1/sqrt(Dk).
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
        return LayoutSupport::ChannelFirstOnly;
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
