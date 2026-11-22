#pragma once
/// @file group_query_attention.hpp
/// @brief Grouped-query scaled dot-product attention.
///
/// Multi-head attention with at most as many key/value heads as query heads:
/// query head h reads kv head h / (H / H_kv), so consecutive query heads share
/// one K/V pair. **H_kv == H is ordinary multi-head attention** — there is no
/// separate MHA operator; this class is the general case and `kv_num_heads`
/// selects the degenerate one. H_kv == 1 is multi-query attention.
///
///   scores = Q·Kᵀ·scale (+ bias);  output = softmax(scores, -1)·V
///
/// The `causal` attribute picks between two input contracts, not merely a
/// masking rule.
///
/// ---- causal = false: explicit attention (ViT / encoder) -------------------
///
/// The keys and values are the caller's own tensors. Masking is an external
/// input: whatever the model means to hide — padding, document isolation,
/// ALiBi-style relative bias, the fully-masked rows of masked-autoencoder
/// queries — it says so with the mask, and the operator never derives one from
/// a position.
///
///   [0] query  [B, Sq, H·D]  or the rank-4 form (see head_layout)
///   [1] key    [B, Sk, H_kv·D]  same layout as query; Sk may differ from Sq
///   [2] value  [B, Sk, H_kv·D]  same layout as key
///   [3] attention_bias    float additive, [Sq, Sk] shared by every batch and
///                         head or [B|1, H|1, Sq, Sk] (an extent-1 axis
///                         broadcasts and is never expanded)
///   [4] key_padding_mask  int32 keep-mask, nonzero = keep: a padded key takes
///                         mask_filter_value. [B, Sk] is one row per batch,
///                         [B, Sq, Sk] one row per query row. Rank 1 [B] is
///                         not a mask but the batch's valid key-run length
///                         (right padding is truncated, every batch may sit at
///                         its own position)
///   [5] head_sink         float [H]: one extra logit per query head in the
///                         softmax denominator (a virtual key whose value is
///                         zero); a row hidden everywhere then reads zeros
///                         rather than NaN
///   [6] cu_seqlens_q, [7] cu_seqlens_k   packed documents (see below)
///   [8] output_gate       shaped like the output (see output_gate)
///
/// Every query row sees the batch's whole valid run — the mask, and only the
/// mask, narrows it. Fully masked rows read zeros. `softcap` and
/// `local_window_size` are causal-run attributes and are ignored here.
///
/// ---- causal = true: KV-cache attention (decoder) --------------------------
///
/// The current step's K/V is already in the cache: the caller appends (and
/// quantizes) it before the call, and the operator only reads. There is no
/// mask input, no past, and no present output — causality is derived from the
/// batch's actual position.
///
///   [0] query  [B, Sq, H·D]  or the rank-4 form (see head_layout)
///   [1] k_cache  [B, H_kv, Skv, D]  contiguous, onnxruntime's BNSH order,
///               always this order whatever head_layout says for the query
///   [2] v_cache  [B, H_kv, Skv, D]  a separate tensor, same order
///   [3] kv_seqlens  int32 [B]: the number of valid rows in batch b's cache
///               (the run length). Absent, every batch runs to Skv.
///   [4] position_ids  int64 [B]: the absolute position of batch b's first
///               query row, rows consecutive (row i is at position_ids[b]+i).
///               This is the same tensor RoPE takes. Absent, the bottom-right
///               alignment derives it from kv_seqlens (kv_seqlens[b] - Sq).
///               Present, it alone fixes the causal diagonal and the window;
///               kv_seqlens still bounds which keys are visible, so the two
///               may differ (a draft row can sit past its visible prefix).
///   [5] k_scale, [6] v_scale  f32, present exactly when the cache is int8
///   [7] head_sink  [H]
///   [8] output_gate
///
/// The cache dtype is the query's (f32 / f16) or s8. An s8 cache is symmetric —
/// the stored integer times its scale is the value — and the scales come from
/// whoever wrote the cache (the append quantizes and records them), so they
/// stay fixed across steps:
///   - per token: [B, H_kv, Skv], one scale per cache row — the default;
///   - per tensor: numel 1, one scale for the whole cache.
/// The operator dequantizes on the fly (the K scale folds into the score, the
/// V scale into the probabilities); the query stays fp.
///
/// The output follows Q's shape in both forms. scale = 0 means 1/sqrt(D).
///
/// ---- shared semantics -----------------------------------------------------
///
/// The mask is added after the scale and the softcap and before the softmax, so
/// a mask value of -inf behaves as "hidden", while a finite large negative
/// (mask_filter_value) waits for the softmax numerator to underflow. A row
/// whose valid range is empty reads zeros in every route.
///
/// `local_window_size` (Gemma2/3 local layers, Mistral) lowers each causal
/// row's first visible key to pos_i - window + 1; `sink_tokens` then keeps the
/// cache's first N positions visible regardless, so a row reads
/// [0, N) ∪ [pos_i - window + 1, pos_i + 1) — the StreamingLLM/Gemma-3 local
/// pattern with its 4 sink tokens. `softcap` rewrites the scaled logit before
/// the mask as cap·tanh(s/cap). Both are off at -1 / 0.
///
/// Packed documents (inputs[6]/[7]): the int32 [segments + 1] prefix offsets of
/// a packed batch — Q is [1, total_q, H·D], K/V are [1, total_kv, H_kv·D], and
/// segment s owns query rows [cu_q[s], cu_q[s+1]) and key rows
/// [cu_k[s], cu_k[s+1]). No query attends across a segment boundary. The pair
/// is exclusive with the masks and the cache — one packed call carries many
/// independent documents.
///
/// Optional slots are positional: an omitted input is an empty view.

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

struct GroupQueryAttentionAttributes {
    /// Query heads (H).
    int64_t num_heads = 8;

    /// Key/value heads (H_kv). 0 means num_heads; otherwise H % H_kv == 0.
    int64_t kv_num_heads = 0;

    /// Attention scale; 0 means 1/sqrt(head_dim).
    float scale = 0.0f;

    /// Input contract selector. false is explicit key/value attention with an
    /// external mask (ViT/encoder); true is KV-cache attention whose causal
    /// range follows each batch's kv_seqlens (decoder). See the file header.
    bool causal = true;

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
    /// `act(gate)`, where the gate is a trailing input shaped exactly like Q
    /// (the output). Sigmoid is the Qwen form; None disables. One elementwise
    /// pass after the route wrote the output, in place — the two kernels
    /// (activation, multiply) the caller would otherwise run over the hidden,
    /// folded into one.
    EpilogueActivateType output_gate = EpilogueActivateType::None;

    /// Dynamic int8 quantization of the query for Q·Kᵀ. With an s8 cache the
    /// query still arrives fp; this quantizes each query row on the fly (its
    /// own symmetric scale) and runs Q·Kᵀ on the integer MMA, so the product
    /// costs the hardware's integer rate instead of its fp one. The score tile
    /// is dequantized before the mask and softmax, so every other attribute and
    /// input keeps its meaning — at int8 accuracy for the logits.
    ///
    /// It applies to the flash route only (long or split key runs), and only to
    /// a query-key width the integer pack's k group divides and fits in one k
    /// block; anything else — a short run on the materialized route, the
    /// Sq == 1 GEMV route, an fp cache — runs unchanged on the fp path, so the
    /// attribute is a hint, not a contract. Off by default.
    bool quantize_query = false;
};

class GroupQueryAttention : public OpBase {
public:
    static std::unique_ptr<GroupQueryAttention> create(
        const GroupQueryAttentionAttributes& attrs,
        Backend backend = Backend::CPU);

    static std::unique_ptr<GroupQueryAttention> create(Backend backend = Backend::CPU) {
        return create(GroupQueryAttentionAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::GroupQueryAttention; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::Any; }

    const GroupQueryAttentionAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~GroupQueryAttention();

private:
    GroupQueryAttention(const GroupQueryAttentionAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    GroupQueryAttentionAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
