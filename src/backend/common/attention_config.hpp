#pragma once
/// @file attention_config.hpp
/// @brief The head geometry and behaviour one attention call is run with —
///        the description the CPU core and the CUDA kernel both read.
///
/// GroupQueryAttention (which covers ordinary MHA as its H_kv == H case) and
/// MultiLatentAttention differ only in how their attributes and input slots
/// describe a head geometry; AttentionConfig is that description. Each
/// operator's front end maps its onnxruntime-aligned input list onto these
/// fields (see make_attention_config), and neither backend's kernel knows
/// which operator called it.
///
/// Input slots are positional and an omitted optional input is an empty view.
/// A -1 slot means the operator has no such slot. The two forms are:
///
///   explicit  Q, K, V, [mask bias], [key_padding_mask], ... — K and V are the
///             caller's tensors in Q's layout, and the whole valid range is
///             [0, run). The optional past pair keeps the [past | new] run for
///             MLA, which has no cache input.
///   cache     Q, k_cache, v_cache, [seqlens], [k_scale], [v_scale], ... —
///             k_cache / v_cache are the run itself in [B, H_kv, S, D] order
///             (cfg.kv_bshd distinguishes BSHD for the explicit rank-4 form),
///             the caller having appended the step's rows before the call.
///             Nothing writes them here.
///
/// The backend split owns only the *routes* (standard / flash / decode on the
/// CPU; the tiled online-softmax kernel on the CUDA device). The config, the
/// scale/softcap/window/sink semantics and the mapping below are shared, so a
/// new field is defined once and both backends see it.

#include "nnops/core/epilogue.hpp"
#include "nnops/ops/group_query_attention.hpp"
#include "nnops/ops/multi_latent_attention.hpp"

#include <cmath>
#include <cstdint>

namespace nnops::backend {

/// The head geometry and behaviour one attention call is run with.
struct AttentionConfig {
    int64_t num_heads = 1;    ///< query heads (H)
    int64_t num_kv_heads = 0; ///< key/value heads; 0 means num_heads
    int64_t v_head_dim = 0;   ///< value width; only read when v_is_k
    float scale = 0.0f;       ///< 0 means 1/sqrt(query-key width)
    bool causal = false;      ///< causal mask over the run
    bool v_is_k = false;      ///< value = leading v_head_dim of the key row
    /// Absolute position of query row 0; negative (the default) means
    /// bottom-right, q0 = run - Sq. Only consulted when causal.
    int64_t position_offset = -1;
    /// Logit cap (Gemma2/3 softcapping): s = cap·tanh(s/cap), applied after the
    /// scale and before the additive mask. 0 disables.
    float softcap = 0.0f;
    /// Sliding window: query row i also skips keys older than its position
    /// minus window + 1 (Gemma2/3 local layers). 0 disables.
    int64_t window = 0;
    /// StreamingLLM-style sinks: the first `sink_tokens` key positions stay
    /// visible to every query row even outside the window. Only meaningful
    /// with `window`; 0 disables. A row's visible keys are then
    /// [0, sink) ∪ [pos_i - window + 1, pos_i + 1).
    int64_t sink_tokens = 0;
    /// Output gate (Qwen3-Next/3.5): the finished output is multiplied in place
    /// by act(gate) of the trailing input slot. None disables.
    EpilogueActivateType output_gate = EpilogueActivateType::None;

    // ---- input slots (-1 means the operator has no such slot) ----
    int past_k_slot = -1;  ///< past K rows; past_v_slot is past K + 1
    int past_v_slot = -1;
    int seqlens_slot = -1; ///< int32 [B] per-batch valid run (see seqlens_is_length)
    /// seqlens entries are already lengths; false is onnxruntime's last valid
    /// key index (T-1). The cache form is a length, MLA's seqlens_k is not.
    bool seqlens_is_length = false;
    int kpm_slot = -1;     ///< int32 key_padding_mask: [B] lengths or a keep mask
    int sink_slot = -1;    ///< per-head attention sink (softmax denominator)
    int bias_slot = -1;    ///< float additive attention_bias
    int cu_q_slot = -1;    ///< varlen offsets; cu_k_slot is cu_q_slot + 1
    int cu_k_slot = -1;
    int gate_slot = -1;    ///< output gate, shaped like the output
    /// int64 [B]: the absolute position of query row 0 for each batch, rows
    /// consecutive. Overrides position_offset (a scalar for every batch) and
    /// the bottom-right default. The same input RoPE reads.
    int pos_ids_slot = -1;
    /// f32 dequantization scales for an int8 K / V cache; -1 (or an empty
    /// view) with an fp cache. numel 1 is per-tensor, H_kv·D per-channel.
    int k_scale_slot = -1;
    int v_scale_slot = -1;
    /// The rank-4 per-head form of Q is [B, S, H, D] instead of [B, H, S, D]
    /// (onnxruntime's BNSH). Both are the same rank, so this is explicit.
    bool bshd = false;
    /// The rank-4 K/V form is [B, S, H_kv, D] rather than BNSH. The explicit
    /// form follows Q's layout; the cache is always BNSH.
    bool kv_bshd = false;
    /// Value a key_padding_mask uses for a padded key (onnxruntime's
    /// mask_filter_value, default -10000 — not -inf).
    float mask_filter_value = -10000.0f;
    /// Dynamic int8 quantization of the query for Q·Kᵀ (the CPU flash route's
    /// integer product). No CUDA kernel implements it yet.
    bool quantize_q = false;
    /// int8 probabilities in P·V (the CPU flash route's second integer
    /// product). No CUDA kernel implements it yet.
    bool quantize_pv = false;

    int64_t kv_heads() const noexcept { return num_kv_heads > 0 ? num_kv_heads : num_heads; }
};

/// GroupQueryAttention's attribute → config mapping. `causal` selects the
/// input contract (explicit vs cache), not merely the mask.
inline AttentionConfig make_attention_config(const GroupQueryAttentionAttributes& attrs)
{
    AttentionConfig cfg;
    cfg.num_heads = attrs.num_heads;
    cfg.num_kv_heads = attrs.kv_num_heads;
    cfg.scale = attrs.scale;
    cfg.causal = attrs.causal;
    cfg.softcap = attrs.softcap;
    cfg.window = attrs.local_window_size > 0 ? attrs.local_window_size : 0;
    cfg.sink_tokens = attrs.sink_tokens;
    cfg.output_gate = attrs.output_gate;
    cfg.bshd = attrs.head_layout == AttentionHeadLayout::BSHD;
    cfg.mask_filter_value = attrs.mask_filter_value;
    cfg.quantize_q = attrs.quantize_query;
    cfg.quantize_pv = attrs.quantize_pv;

    if (attrs.causal) {
        // Cache form: Q, k_cache, v_cache, kv_seqlens, position_ids, k_scale,
        // v_scale, head_sink, output gate. The caches are BNSH whatever Q's
        // layout is.
        cfg.seqlens_slot = 3;
        cfg.seqlens_is_length = true;
        cfg.pos_ids_slot = 4;
        cfg.k_scale_slot = 5;
        cfg.v_scale_slot = 6;
        cfg.sink_slot = 7;
        cfg.gate_slot = 8;
        cfg.kv_bshd = false;
    } else {
        // Explicit form: Q, K, V, attention_bias, key_padding_mask, head_sink,
        // cu_seqlens_q/k, output gate. K/V follow Q's layout.
        cfg.bias_slot = 3;
        cfg.kpm_slot = 4;
        cfg.sink_slot = 5;
        cfg.cu_q_slot = 6;
        cfg.cu_k_slot = 7;
        cfg.gate_slot = 8;
        cfg.kv_bshd = attrs.head_layout == AttentionHeadLayout::BSHD;
    }
    return cfg;
}

/// MultiLatentAttention's attribute → config mapping.
inline AttentionConfig make_attention_config(const MultiLatentAttentionAttributes& attrs)
{
    AttentionConfig cfg;
    cfg.num_heads = attrs.num_heads;
    cfg.num_kv_heads = attrs.kv_num_heads;
    cfg.v_head_dim = attrs.v_head_dim > 0 ? attrs.v_head_dim : attrs.kv_lora_rank;
    cfg.scale = attrs.scale != 0.0f ?
                    attrs.scale :
                    1.0f / std::sqrt(static_cast<float>(attrs.qk_nope_head_dim + attrs.qk_rope_head_dim));
    cfg.causal = attrs.causal;
    cfg.position_offset = attrs.position_offset;
    // Absorbed: one latent row per token and the value is its prefix. The
    // non-absorbed form takes a separate V tensor, as GQA does.
    cfg.v_is_k = attrs.absorbed;
    // MLA keeps its own slot list (there is no onnxruntime operator to mirror):
    // Q, KV, [mask], [past KV], [seqlens_k], [position_ids], [cu_q], [cu_k] —
    // or Q, K, V, [mask], [past K], [past V], [seqlens_k], [position_ids],
    // [cu_q], [cu_k].
    if (attrs.absorbed) {
        cfg.bias_slot = 2;
        cfg.past_k_slot = 3;
        cfg.past_v_slot = 3;
        cfg.seqlens_slot = 4;
        cfg.pos_ids_slot = 5;
        cfg.cu_q_slot = 6;
        cfg.cu_k_slot = 7;
    } else {
        cfg.bias_slot = 3;
        cfg.past_k_slot = 4;
        cfg.past_v_slot = 5;
        cfg.seqlens_slot = 6;
        cfg.pos_ids_slot = 7;
        cfg.cu_q_slot = 8;
        cfg.cu_k_slot = 9;
    }
    return cfg;
}

} // namespace nnops::backend
