#pragma once
/// @file multi_head_attention.hpp
/// @brief Multi-head scaled dot-product Attention (every query head has its own
///        key/value head).
///
/// scores = Q·Kᵀ·scale (+ bias, masked keys hidden); output = softmax·V.
///
/// The `causal` attribute picks between two input contracts, exactly as in
/// GroupQueryAttention (see that header for the full semantics); only the head
/// geometry differs — here H_kv == H.
///
/// ---- causal = false: explicit attention (ViT / encoder) -------------------
///
///   [0] query  [B, Sq, H·D]  or the rank-4 form (see head_layout)
///   [1] key    [B, Sk, H·D]     same layout as query
///   [2] value  [B, Sk, H·D]
///   [3] attention_bias    float additive, [Sq, Sk] or [B|1, H|1, Sq, Sk]
///   [4] key_padding_mask  int32 [B] lengths, [B, Sk] keep-mask or
///                         [B, Sq, Sk] keep-mask; padded key takes
///                         mask_filter_value
///   [5] head_sink         float [H]
///   [6] cu_seqlens_q, [7] cu_seqlens_k
///   [8] output_gate
///
/// ---- causal = true: KV-cache attention (decoder) --------------------------
///
///   [0] query  [B, Sq, H·D]  or the rank-4 form (see head_layout)
///   [1] k_cache  [B, H, Skv, D]  contiguous BNSH, always
///   [2] v_cache  [B, H, Skv, D]
///   [3] kv_seqlens  int32 [B], the valid rows of each batch's cache
///   [4] position_ids  int64 [B], the absolute position of each batch's first
///                    query row (rows consecutive) — the same tensor RoPE
///                    takes; absent means bottom-right, kv_seqlens[b] - Sq
///   [5] k_scale, [6] v_scale  f32, present exactly when the cache is int8
///   [7] head_sink  [H]
///   [8] output_gate
///
/// The cache is written by the caller before the call: the operator never
/// writes K/V and has no present output. An int8 cache is symmetric with an
/// external scale — [B, H, Skv] per token (the default) or numel 1 per tensor.
/// See GroupQueryAttention's header for the masking, softcap, window, sink and
/// packed-document rules.
///
/// Grouped-query attention — several query heads per key/value head — is
/// GroupQueryAttention, and the absorbed DeepSeek-style latent form is
/// MultiLatentAttention; this operator is the H == Hkv case and does not
/// implement either.

#include "nnops/core/op_base.hpp"
#include "nnops/ops/attention_common.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/epilogue.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes for the multi-head attention operator.
struct MultiHeadAttentionAttributes {
    int64_t num_heads = 8;

    /// Attention scale; 0 means 1/sqrt(head_dim).
    float scale = 0.0f;

    /// Input contract selector. false is explicit key/value attention with an
    /// external mask (ViT/encoder); true is KV-cache attention whose causal
    /// range follows each batch's kv_seqlens (decoder). See the file header.
    bool causal = false;

    /// Logit softcap (Gemma2/3): s = softcap·tanh(s/softcap), applied to the
    /// scaled score and before the mask. 0 disables.
    float softcap = 0.0f;

    /// Sliding-window attention (Gemma2/3 local layers, Mistral): a causal
    /// query row i also skips keys older than pos_i - local_window_size + 1.
    /// -1 disables, onnxruntime's sentinel; ignored by the explicit form.
    int64_t local_window_size = -1;

    /// StreamingLLM-style sinks: the first `sink_tokens` key positions stay
    /// visible to every query row even outside the window, so a row at
    /// position i reads [0, sink_tokens) ∪ [i - window + 1, i]. Gemma-3-style
    /// local layers keep 4. 0 disables; only meaningful with a window.
    int64_t sink_tokens = 0;

    /// Value a padded key takes in an integer key_padding_mask
    /// (onnxruntime's mask_filter_value; -10000, not -inf).
    float mask_filter_value = -10000.0f;

    /// Storage order of the rank-4 per-head form; see AttentionHeadLayout.
    /// The KV cache keeps BNSH regardless.
    AttentionHeadLayout head_layout = AttentionHeadLayout::Auto;

    /// Output gate (Qwen3-Next / 3.5 / 4): multiply the attention output by
    /// `act(gate)`, a trailing input shaped like the output. Sigmoid is the
    /// Qwen form; None disables. Applied in place in one pass after the route
    /// wrote the output, replacing the caller's Activation + Eltwise pair.
    EpilogueActivateType output_gate = EpilogueActivateType::None;
};

class MultiHeadAttention : public OpBase {
public:
    static std::unique_ptr<MultiHeadAttention> create(
        const MultiHeadAttentionAttributes& attrs,
        Backend backend = Backend::CPU);

    static std::unique_ptr<MultiHeadAttention> create(Backend backend = Backend::CPU) {
        return create(MultiHeadAttentionAttributes{}, backend);
    }

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

    OpType  getOpType()  const override { return OpType::MultiHeadAttention; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::Any; }

    const MultiHeadAttentionAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in multi_head_attention.cpp (Pimpl pattern)
    ~MultiHeadAttention();

private:
    MultiHeadAttention(const MultiHeadAttentionAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    MultiHeadAttentionAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
