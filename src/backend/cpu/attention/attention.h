#pragma once
/// @file attention.h
/// @brief The shared CPU attention core — Q·Kᵀ + softmax + attn·V — behind all
///        three attention operators.
///
/// MultiHeadAttention, GroupQueryAttention and MultiLatentAttention differ only
/// in how their attributes and inputs describe a head geometry; the fused
/// kernel is one. AttentionConfig is that description: query/kv head counts,
/// the query-key width, the value width, the scale, causal masking, and whether
/// the value is the leading slice of the key row (MLA's latent). Each operator
/// builds a config and calls attention_cpu; nothing below knows which one did.
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
///     grid is batch × query head × query-block, so long-sequence prefill keeps
///     its parallelism even when batch × head is small. A causal query block
///     stops its KV stream at the block's causal limit, so prefill never walks
///     the masked upper triangle.
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
/// Scratch sizing counts uniform nr_max-wide panels, not a width cascade: the
/// widest n-block in a loop bounds the count (a greedy per-width decomposition
/// is not monotonic in n and undersized the slot). The packed-B stride takes
/// min(Kc, k) of the longer GEMM, since a head dim below Kc would otherwise
/// over-allocate by up to 2x.
///
/// Q/K/V/O head slices are M×K / N×K / K×N / M×N submatrices of the channel-first
/// tensors, for both merged [B, S, H*D] and explicit [B, H, S, D] layouts (a
/// head's D columns are contiguous within each row). The mask is flat
/// [Sq, Sk_total] over the whole run; its index carries no batch or head.
///
/// Input slots are positional and an omitted optional input is an empty view:
/// Q, K, V, [mask], [past K], [past V], [seqlens_k] — or Q, KV, [mask],
/// [past KV], [seqlens_k] when v_is_k. The past tensors carry the same layout
/// as their new counterpart, with Sk_past rows instead of Sk_new; the run is
/// [past | new]. seqlens_k is an int32 [B] vector of per-batch valid key-run
/// lengths (the run's prefix), read by every route through run_length.
///
/// Implementation is one file per route (attention_standard.cpp,
/// attention_flash.cpp, attention_decode.cpp) over the shared machinery in
/// attention_internal.h; attention_plan.cpp owns the tile and scratch sizing,
/// and attention.cpp holds the public entry and the operator mappings.

#include "nnops/core/tensor_view.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/status.hpp"

#include <cstddef>
#include <span>

namespace nnops::backend::cpu {

/// The head geometry and behaviour one attention call is run with.
struct AttentionConfig {
    int64_t num_heads = 1;     ///< query heads (H)
    int64_t num_kv_heads = 0;  ///< key/value heads; 0 means num_heads
    int64_t v_head_dim = 0;    ///< value width; only read when v_is_k
    float   scale = 0.0f;      ///< 0 means 1/sqrt(query-key width)
    bool    causal = false;    ///< causal mask over the run
    bool    v_is_k = false;    ///< value = leading v_head_dim of the key row
    /// Absolute position of query row 0; negative (the default) means
    /// bottom-right, q0 = run - Sq. Only consulted when causal.
    int64_t position_offset = -1;

    int64_t kv_heads() const noexcept {
        return num_kv_heads > 0 ? num_kv_heads : num_heads;
    }
};

/// Tile sizes + scratch layout, resolved once and shared by the kernels and
/// the scratch sizing.
struct AttentionPlan {
    bool   use_flash = false;   ///< route through the FlashAttention path
    bool   use_decode = false;  ///< Sq == 1: route through the GEMV decode path
    int    mc1 = 0, nc1 = 0;    ///< standard-path GEMM1 (Sq × Sk) tiles
    int    mc2 = 0, nc2 = 0;    ///< standard-path GEMM2 (Sq × Dv) tiles
    int    Br = 0, Bc = 0;      ///< flash-path query/KV block tiles
    /// Decode head batching: query heads one task serves. Under GQA/MQA the
    /// group's heads share a KV head, so one pass over K/V feeds all of them
    /// instead of each head re-reading the run.
    int    decode_heads = 1;
    /// Decode KV splitting: slices of the run handled by independent tasks,
    /// merged with the online-softmax weights. > 1 only when the head-batched
    /// task count would leave workers idle.
    int    decode_splits = 1;
    int64_t decode_elems = 0;   ///< decode-path f32 scratch elements per slot
    int64_t num_slots = 0;      ///< scratch slots (worker threads, else tasks)
    size_t workspace_size = 0;  ///< pooled scratch bytes (0 → no tiled route)
};

/// Resolve tile sizes + scratch bytes for f32/f16 (tiles and element size
/// follow the dtype). Any other dtype, or a Q/K/V dtype mismatch, returns a
/// zeroed plan (workspace_size 0). @p num_threads and @p use_thread_slots
/// bound the per-thread scratch slot count; @p outputs is unused.
AttentionPlan get_attention_plan(const AttentionConfig& cfg,
                                 std::span<const TensorDesc> inputs,
                                 std::span<const TensorDesc> outputs,
                                 int num_threads = 1,
                                 bool use_thread_slots = false);

/// Tiled attention kernel (f32 / f16), routed standard, flash or decode by the
/// plan. Inputs are Q, K, V [, mask [, past K, past V]] — or Q, KV [, mask
/// [, past KV]] when cfg.v_is_k, in which case the value is read from the key
/// tensor's leading columns. Scratch comes
/// from the internal pool, sized by get_attention_plan; @p workspace is unused.
/// Returns Status::unsupported unless every input, the output and the optional
/// mask share one f32 or f16 dtype.
Status attention_cpu(const AttentionConfig& cfg,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);

}  // namespace nnops::backend::cpu
