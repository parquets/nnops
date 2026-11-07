#pragma once
/// @file multi_latent_attention.hpp
/// @brief Multi-head Latent Attention (DeepSeek-V2/V3), absorbed form.
///
/// MLA compresses K and V into one latent row per token, `[c_KV | k_pe]`:
/// `c_KV` has kv_lora_rank channels and `k_pe` has qk_rope_head_dim. In the
/// absorbed form the talker matrices are folded into the projections, so the
/// query is already `[q_nope·W_UK | q_pe]` — the same width as the latent row —
/// and the value is the leading kv_lora_rank channels of that same row. The
/// operator therefore takes two inputs, not three, and never materializes a
/// per-head K or V:
///
///   scores = Q·KVᵀ·scale (+ mask);  output = softmax(scores, -1)·KV[:, :v_head_dim]
///
/// Inputs, channel-first only (the layout onnxruntime's latent PagedAttention reads):
///   [0] Q  [B, Sq, H·Dkv]     or [B, H,    Sq, Dkv]
///   [1] KV [B, Sk, H_kv·Dkv]  or [B, H_kv, Sk, Dkv]
///   [2] optional additive mask, flat [Sq, Sk], shared by every batch and head.
/// with Dkv = kv_lora_rank + qk_rope_head_dim, H_kv = kv_num_heads (1 for the
/// DeepSeek models — one latent head shared by every query head).
///
/// Output: [B, Sq, H·v_head_dim] or [B, H, Sq, v_head_dim], v_head_dim
/// defaulting to kv_lora_rank. The per-head output projection (W_UV) stays
/// outside the operator: this returns attention over the latent, so the caller
/// applies the absorbed value side exactly as it applied the absorbed query
/// side.
///
/// `scale` is *not* inferred from the latent width: MLA keeps the non-absorbed
/// scale, 1/sqrt(qk_nope_head_dim + qk_rope_head_dim), so the default here uses
/// those two attributes. DeepSeek-V2 is 1/sqrt(128 + 64).
///
/// `causal` (default true) uses j <= q0 + i for query row i, q0 being
/// position_offset or, by default, Sk - Sq (bottom-right): a single-token
/// decode against a cache passes the whole cached run as KV.
///
/// Chunked input: inputs[3] is an optional past KV — the same layout, with
/// Sk_past latent rows. The operator reads [past | new] by streaming each
/// segment in place, so latents gathered out of a paged cache need no per-step
/// concatenation, and the mask stays [Sq, Sk_past + Sk_new]. The past is
/// read-only. Optional slots are positional, so a step call without a mask
/// passes an empty view in inputs[2].
///
/// `position_offset` is the absolute position of query row 0; set it when the
/// chunk is not the tail of the run.
///
/// inputs[4] is an optional int32 [B] `seqlens_k`: each batch's valid key-run
/// length, so the run is the prefix [0, seqlens_k[b]) — right padding is
/// truncated and batches may sit at different positions. Causal masking then
/// takes the per-batch bottom-right q0 = seqlens_k[b] - Sq unless
/// position_offset (a scalar for every batch) is given. Absent, every batch
/// runs to the full Sk.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

struct MultiLatentAttentionAttributes {
    /// Query heads (H). DeepSeek-V2: 128.
    int64_t num_heads = 16;

    /// Latent heads (H_kv); H % H_kv == 0. DeepSeek-V2: 1.
    int64_t kv_num_heads = 1;

    /// Compressed KV width, and the default value width. DeepSeek-V2: 512.
    int64_t kv_lora_rank = 512;

    /// Decoupled RoPE width carried alongside the latent. DeepSeek-V2: 64.
    int64_t qk_rope_head_dim = 64;

    /// Pre-absorption query/key width; used only for the default scale, since
    /// the absorbed query no longer has it. DeepSeek-V2: 128.
    int64_t qk_nope_head_dim = 128;

    /// Value width; 0 means kv_lora_rank. Must be <= kv_lora_rank and the
    /// latent row's width.
    int64_t v_head_dim = 0;

    /// Attention scale; 0 means 1/sqrt(qk_nope_head_dim + qk_rope_head_dim).
    float scale = 0.0f;

    /// Causal mask: j <= q0 + i, with q0 = position_offset or Sk - Sq.
    bool causal = true;

    /// Absolute position of query row 0; negative means bottom-right. Only
    /// consulted when causal. With a past segment, the chunked-prefill value is
    /// the past length.
    int64_t position_offset = -1;
};

class MultiLatentAttention : public OpBase {
public:
    static std::unique_ptr<MultiLatentAttention> create(
        const MultiLatentAttentionAttributes& attrs,
        Backend backend = Backend::CPU);

    static std::unique_ptr<MultiLatentAttention> create(Backend backend = Backend::CPU) {
        return create(MultiLatentAttentionAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::MultiLatentAttention; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::ChannelFirstOnly; }

    const MultiLatentAttentionAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~MultiLatentAttention();

private:
    MultiLatentAttention(const MultiLatentAttentionAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    MultiLatentAttentionAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
