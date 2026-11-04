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
/// Two routes, picked by the per-head scores area:
///   - standard (Sq*Sk < kFlashMinScores): materialize scores = Q·K^T·scale,
///     softmax each row in place, then output = attn·V.
///   - flash (Sq*Sk >= kFlashMinScores): FlashAttention-1 online softmax,
///     streaming KV in Br×Bc tiles so scores is never materialized. The task
///     grid is batch × query head × query-block, so long-sequence prefill keeps
///     its parallelism even when batch × head is small.
///
/// Query head h reads kv head h / (H / H_kv), so one K/V slice serves a whole
/// group under grouped-query attention, and H_kv == H is ordinary MHA.
///
/// Causal masking is bottom-right aligned: query row i sees key positions
/// j <= i + (Sk - Sq). That is the KV-cache decode convention — the Sq query
/// rows are the last Sq positions of the key run — and it degenerates to
/// j <= i when Sq == Sk.
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
/// Q/K/V/O head slices are M×K / N×K / K×N / M×N submatrices of the planar
/// tensors, for both merged [B, S, H*D] and explicit [B, H, S, D] layouts (a
/// head's D columns are contiguous within each row). The mask is flat [Sq, Sk];
/// its index carries no batch or head.

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
    bool    causal = false;    ///< bottom-right aligned causal mask
    bool    v_is_k = false;    ///< value = leading v_head_dim of the key row

    int64_t kv_heads() const noexcept {
        return num_kv_heads > 0 ? num_kv_heads : num_heads;
    }
};

/// Tile sizes + scratch layout, resolved once and shared by the kernels and
/// the scratch sizing.
struct AttentionPlan {
    bool   use_flash = false;   ///< route through the FlashAttention path
    int    mc1 = 0, nc1 = 0;    ///< standard-path GEMM1 (Sq × Sk) tiles
    int    mc2 = 0, nc2 = 0;    ///< standard-path GEMM2 (Sq × Dv) tiles
    int    Br = 0, Bc = 0;      ///< flash-path query/KV block tiles
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

/// Tiled attention kernel (f32 / f16), routed standard or flash by the plan.
/// Inputs are Q, K, V [, mask] — or Q, KV [, mask] when cfg.v_is_k, in which
/// case the value is read from the key tensor's leading columns. Scratch comes
/// from the internal pool, sized by get_attention_plan; @p workspace is unused.
/// Returns Status::unsupported unless every input, the output and the optional
/// mask share one f32 or f16 dtype.
Status attention_cpu(const AttentionConfig& cfg,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);

}  // namespace nnops::backend::cpu
