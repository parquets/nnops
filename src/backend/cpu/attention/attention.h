#pragma once
/// @file attention.h
/// @brief The shared CPU attention core — Q·Kᵀ + softmax + attn·V — behind both
///        attention operators.
///
/// GroupQueryAttention (which covers ordinary MHA as its H_kv == H case) and
/// MultiLatentAttention differ only in how their attributes and inputs describe
/// a head geometry; the fused kernel is one. AttentionConfig is that
/// description: query/kv head counts, the query-key width, the value width, the
/// scale, causal masking, and whether the value is the leading slice of the key
/// row (MLA's latent). Each operator builds a config and calls attention_cpu;
/// nothing below knows which one did.
///
/// Three routes, picked by the shape:
///   - decode (Sq == 1, no mask): both products are matrix-vector, so K and V
///     are streamed once each with no packing and no M == 1 MMA tiles. One
///     task serves up to eight query heads of the same KV head (GQA/MQA), so
///     the run is read once for the whole group, and the run is split into
///     independent slices when the batched grid would leave workers idle;
///     a slice's partial output, max and sum are merged afterwards.
///   - standard (Sq*Sk < kFlashMinScores): materialize scores = Q·K^T·scale,
///     softmax each row in place, then output = attn·V.
///   - flash (Sq*Sk >= kFlashMinScores): FlashAttention-1 online softmax,
///     streaming KV in Br×Bc tiles so scores is never materialized. The task
///     grid is batch × KV head × head-block × query-block × KV-slice, so
///     long-sequence prefill keeps its parallelism even when batch × head is
///     small. Up to four query heads of a KV head share a task (one K/V pack
///     per group), and an underfilled grid splits the run into slices whose
///     online-softmax states are merged. A causal query block stops its KV
///     stream at the block's causal limit, so prefill never walks the masked
///     upper triangle.
///
/// Query head h reads kv head h / (H / H_kv), so one K/V slice serves a whole
/// group under grouped-query attention, and H_kv == H is ordinary MHA.
///
/// Causal masking: query row i sees key positions j <= q0 + i, where q0 is the
/// absolute position of row 0. Without an explicit offset q0 = run - Sq, the
/// bottom-right alignment — the Sq query rows are the last Sq positions of the
/// key run — which is the KV-cache decode convention and degenerates to
/// j <= i when Sq == run. An explicit q0 (cfg.position_offset) decodes a chunk
/// that is not the tail of the run. The run is the whole [past | new] tensor
/// unless a per-batch seqlens_k vector shortens it (see below), in which case
/// both the causal diagonal and the non-causal range are clamped to the batch's
/// prefix.
///
/// Chunked input: the key run may arrive as two segments, an optional past
/// tensor followed by the step's new K/V. The kernels address the run by
/// logical index and stream each segment in place, so nothing is concatenated
/// or copied. The past is read-only; writing the cache stays the caller's job.
/// The standard route packs B from one base pointer and so is not split-aware
/// — a split run always routes through flash (or decode).
///
/// Scale is fused into the GEMM1 RHS pack. Both GEMMs only accumulate, so the
/// first k-block of each output tile runs with zero_mode (add_to=false):
/// nothing is pre-zeroed, and that block skips the C read.
///
/// Mask handling differs by route: the standard path folds it into the
/// mask+softmax kernel (one pass fewer than `scores += mask`), the flash path
/// adds it into the S tile before the online-softmax update. A causal row whose
/// whole S tile is out of range contributes nothing: the standard path softmaxes
/// only its valid prefix and zeroes the tail, the flash path skips the tile.
///
/// The valid key range of query row i is the intersection of three bounds: the
/// causal limit [0, hi), the sliding window [lo, run) with
/// lo = max(0, pos_i - window + 1), and the batch's run length. cfg.sink_tokens
/// additionally keeps [0, sink) visible, so the range becomes
/// [0, sink) ∪ [lo, hi) — the StreamingLLM / Gemma-3 local pattern. Both bound
/// functions come from attention_internal.h (causal_limit / key_lo,
/// sink_prefix) so the plan and every route agree on them.
///
/// Softcap, when set, rewrites every score as cap·tanh(s/cap) after the scale
/// and before the mask — the order Gemma2/3 use. It is a per-row pass on the
/// score tile in each route, and the decode route applies scale and cap
/// together before its online-softmax update.
///
/// Scratch sizing counts uniform nr_max-wide panels, not a width cascade: the
/// widest n-block in a loop bounds the count (a greedy per-width decomposition
/// is not monotonic in n and undersized the slot). The packed-B stride takes
/// min(Kc, k) of the longer GEMM, since a head dim below Kc would otherwise
/// over-allocate by up to 2x.
///
/// Q/K/V/O head slices are M×K / N×K / K×N / M×N submatrices of the dense
/// tensors, for both merged [B, S, H*D] and explicit [B, H, S, D] layouts (a
/// head's D columns are contiguous within each row). The optional mask arrives
/// in one of two shapes, told apart by rank: a flat [Sq, Sk_total] shared by
/// every batch and head, or a per-(batch, head) additive bias [B, H, Sq, Sk].
///
/// Input slots are positional and an omitted optional input is an empty view.
/// Two forms reach the core:
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
/// seqlens is an int32 [B] vector of per-batch valid key-run lengths (the run's
/// prefix), read by every route through run_length; cfg.seqlens_is_length says
/// whether the entry is already a length (the cache form) or onnxruntime's last
/// valid index (MLA's seqlens_k). The gate (cfg.output_gate) is shaped like the
/// output and multiplies it in place after the route finished; it is outside
/// every route, so decode, standard, flash and varlen all share the one pass.
///
/// An int8 cache carries symmetric integers: the dequantized value is the
/// stored byte times a scale supplied from outside (cfg.k_scale_slot /
/// cfg.v_scale_slot, f32). The scale is per token — [B, H_kv, Sk] — or a
/// single value for the whole tensor. The query stays fp; the K scale folds
/// into the score and the V scale into the probabilities, so no fp copy of the
/// cache is made.
///
/// cfg.quantize_q adds one route through that cache: the flash route quantizes
/// each query row on the fly (its own symmetric scale) and runs Q·Kᵀ on the
/// integer MMA (VNNI u8×s8 on x86, SDOT/i8mm on aarch64), dequantizing the
/// int32 scores by scale·s_q[row]·s_k[key] before the mask and softmax. The
/// cache's keys are used as stored — no widening pass. The plan reports the
/// choice as flash_qk_i8, and it holds only for a head dim the integer pack's
/// k group divides and packs in one k block (see qk_i8_enabled); every other
/// route and shape runs fp.
///
/// Varlen (cu_seqlens_q / cu_seqlens_k): the two int32 [segments + 1] offsets of
/// HuggingFace's packed batches. Q is [1, total_q, H*D] and K/V are
/// [1, total_kv, ...], segment s owning query rows [cu_q[s], cu_q[s+1]) and key
/// rows [cu_k[s], cu_k[s+1]); no position crosses a segment boundary. The pair
/// is exclusive with the masks, a past segment, seqlens and a position_offset —
/// each segment is bottom-right aligned on its own key run. The core runs one
/// sub-call per segment over the parent's storage.
///
/// Implementation is one file per route (attention_standard.cpp,
/// attention_flash.cpp, attention_decode.cpp) over the shared machinery in
/// attention_internal.h; attention_plan.cpp owns the tile and scratch sizing,
/// and attention.cpp holds the public entry and the operator mappings.

#include "nnops/core/tensor_view.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/status.hpp"
#include "nnops/core/epilogue.hpp"

#include <cstddef>
#include <span>

namespace nnops::backend::cpu {

/// The head geometry and behaviour one attention call is run with.
///
/// Input slots are fields, not positions: each operator's front-end maps its
/// onnxruntime-aligned input list onto them, and the routes never assume a
/// numbering. A -1 means the operator has no such slot; an empty view there
/// means the caller left it out.
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
    int past_k_slot = -1;    ///< past K rows; past_v_slot is past K + 1
    int past_v_slot = -1;
    int seqlens_slot = -1;   ///< int32 [B] per-batch valid run (see seqlens_is_length)
    /// seqlens entries are already lengths; false is onnxruntime's last valid
    /// key index (T-1). The cache form is a length, MLA's seqlens_k is not.
    bool seqlens_is_length = false;
    int kpm_slot = -1;       ///< int32 key_padding_mask: [B] lengths or a keep mask
    int sink_slot = -1;      ///< per-head attention sink (softmax denominator)
    int bias_slot = -1;      ///< float additive attention_bias
    int cu_q_slot = -1;      ///< varlen offsets; cu_k_slot is cu_q_slot + 1
    int cu_k_slot = -1;
    int gate_slot = -1;      ///< output gate, shaped like the output
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
    /// Dynamic int8 quantization of the query for Q·Kᵀ: every query row is
    /// quantized on the fly (its own symmetric scale) and the product runs on
    /// the integer MMA, dequantized into the score tile afterwards. Only the
    /// flash route has the path, and only over an s8 cache whose query-key
    /// width divides the integer pack's k group; everywhere else the fp route
    /// runs unchanged.
    bool quantize_q = false;

    int64_t kv_heads() const noexcept { return num_kv_heads > 0 ? num_kv_heads : num_heads; }
};

/// Tile sizes + scratch layout, resolved once and shared by the kernels and
/// the scratch sizing.
struct AttentionPlan {
    bool use_flash = false;  ///< route through the FlashAttention path
    bool use_decode = false; ///< Sq == 1: route through the GEMV decode path
    int mc1 = 0, nc1 = 0;    ///< standard-path GEMM1 (Sq × Sk) tiles
    int mc2 = 0, nc2 = 0;    ///< standard-path GEMM2 (Sq × Dv) tiles
    int Br = 0, Bc = 0;      ///< flash-path query/KV block tiles
    /// Flash head batching: query heads one flash task serves. They share a KV
    /// head, so the K/V tile is packed and read once for the whole group.
    int flash_heads = 1;
    /// Flash KV splitting: slices of the run handled by independent tasks and
    /// merged with the online-softmax weights. > 1 only when the query-block
    /// grid would leave workers idle.
    int flash_splits = 1;
    /// Flash route runs Q·Kᵀ on the integer MMA (dynamic per-row query
    /// quantization over an s8 cache), and its scratch carries the quantized
    /// query block and the int32 score tile. See cfg.quantize_q for when the
    /// path is eligible.
    bool flash_qk_i8 = false;
    /// Decode head batching: query heads one task serves. Under GQA/MQA the
    /// group's heads share a KV head, so one pass over K/V feeds all of them
    /// instead of each head re-reading the run.
    int decode_heads = 1;
    /// Decode KV splitting: slices of the run handled by independent tasks,
    /// merged with the online-softmax weights. > 1 only when the head-batched
    /// task count would leave workers idle.
    int decode_splits = 1;
    int64_t decode_elems = 0;  ///< decode-path f32 scratch elements per slot
    int64_t num_slots = 0;     ///< scratch slots (worker threads, else tasks)
    size_t workspace_size = 0; ///< pooled scratch bytes (0 → no tiled route)
};

/// Resolve tile sizes + scratch bytes for f32/f16 (tiles and element size
/// follow the dtype). Any other dtype, or a Q/K/V dtype mismatch, returns a
/// zeroed plan (workspace_size 0). @p num_threads and @p use_thread_slots
/// bound the per-thread scratch slot count; @p outputs is unused.
AttentionPlan get_attention_plan(const AttentionConfig& cfg, std::span<const TensorDesc> inputs,
                                 std::span<const TensorDesc> outputs, int num_threads = 1,
                                 bool use_thread_slots = false);

/// Tiled attention kernel (f32 / f16), routed standard, flash or decode by the
/// plan. Inputs are laid out by the operator front-end and addressed through
/// the cfg slots. outputs[0] is the attention output; K/V are read-only, so a
/// cache written by the caller is never modified here. Scratch comes from the
/// internal pool, sized by get_attention_plan; @p workspace is unused. Returns
/// Status::unsupported unless Q, the output, the optional mask and the gate
/// share one f32 or f16 dtype; a named cfg.output_gate whose slot is not
/// exactly the output's geometry returns Status::invalid_argument.
Status attention_cpu(const AttentionConfig& cfg, std::span<TensorView> outputs,
                     std::span<const TensorView> inputs, const ComputeContext& ctx,
                     void* workspace);

} // namespace nnops::backend::cpu
