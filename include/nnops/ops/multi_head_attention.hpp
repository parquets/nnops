#pragma once
/// @file multi_head_attention.hpp
/// @brief Multi-head scaled dot-product Attention (every query head has its own
///        key/value head).
///
/// scores = Q·Kᵀ·scale (+ mask, if given); output = softmax(scores, last dim)·V.
/// Q/K/V arrive as [B, S, H·D] (reshaped internally to [B, H, S, D]) or already
/// as [B, H, S, D]; inputs[0..2] = Q, K, V and inputs[3] = optional additive
/// mask, flat [Sq, Sk] shared by every batch and head or a per-head bias
/// [B, H, Sq, Sk]. scale = 0 means 1/sqrt(head_dim). Channel-first layouts only.
///
/// Chunked input: inputs[4] and inputs[5] are an optional past K and past V —
/// the same layout as K/V, with Sk_past rows instead of Sk_new. The key run
/// the operator reads is then [past | new], streamed segment by segment, so
/// nothing is concatenated or copied. The mask stays [Sq, Sk_past + Sk_new].
/// The past is read-only: growing and writing the cache stays the caller's job.
/// Optional slots are positional, so a step call without a mask passes an empty
/// view in inputs[3].
///
/// `position_offset` is the absolute position of query row 0 and fixes each
/// row's causal range, j <= position_offset + i. Unset (-1) means bottom-right,
/// Sk - Sq; set it for a chunk that is not the tail of the run.
///
/// inputs[6] is an optional int32 [B] `seqlens_k`: each batch's valid key-run
/// length, so the run is the prefix [0, seqlens_k[b]) — right padding is
/// truncated and batches may sit at different positions. Causal masking then
/// takes the per-batch bottom-right q0 = seqlens_k[b] - Sq unless
/// position_offset (a scalar for every batch) is given. Absent, every batch
/// runs to the full Sk.
///
/// `softcap` caps the scaled logit as softcap·tanh(s/softcap) before the mask;
/// `window` raises each row's first visible key to pos_i - window + 1. Both are
/// off at 0.
///
/// inputs[7]/[8] are the optional int32 [segments + 1] `cu_seqlens_q` /
/// `cu_seqlens_k` of a packed varlen batch: Q is [1, total_q, H·D], K/V are
/// [1, total_kv, ...], and each segment attends only inside its own cu range,
/// bottom-right aligned. Exclusive with the mask, the past pair, seqlens_k and
/// position_offset.
///
/// inputs[9] is the optional output gate of `output_gate`: the output is
/// multiplied in place by `act(gate)`, the gate shaped like the output.
///
/// Grouped-query attention — several query heads per key/value head — is
/// GroupQueryAttention, and the absorbed DeepSeek-style latent form is
/// MultiLatentAttention; this operator is the H == Hkv case and does not
/// implement either.

#include "nnops/core/op_base.hpp"
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

    /// Causal masking: query row i attends to key positions j <= q0 + i, where
    /// q0 is position_offset or, by default, Sk - Sq (bottom-right).
    bool causal = false;

    /// Absolute position of query row 0; negative means bottom-right. Only
    /// consulted when causal. With a past segment the run is [past | new] and
    /// the chunked-prefill value is the past length.
    int64_t position_offset = -1;

    /// Logit softcap (Gemma2/3): s = softcap·tanh(s/softcap), applied to the
    /// scaled score and before the mask. 0 disables.
    float softcap = 0.0f;

    /// Sliding-window attention (Gemma2/3 local layers, Mistral): query row i
    /// also skips keys older than its position minus window + 1. 0 disables.
    int64_t window = 0;

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
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::ChannelFirstOnly; }

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
