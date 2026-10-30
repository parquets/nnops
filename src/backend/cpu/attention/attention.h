#pragma once
/// @file attention.h
/// @brief Multi-head scaled dot-product attention — tiled GEMM + softmax kernel.
///
/// For each (batch, head) — parallel via ctx.cpu.run:
///   scores = Q @ K^T * scale      (transpose-B GEMM, K^T packed into pooled scratch)
///   attn   = softmax(scores, -1)  (per contiguous row; the mask, when present,
///                                  folds into the same 3 passes)
///   output = attn @ V             (B-packed GEMM, V packed into the same scratch)
///
/// Both GEMMs reuse the tile_pack_rhs / tile_mma_direct route from
/// matmul_helper.h. Q/K/V/O head slices are M×K / N×K / K×N / M×N submatrices
/// of the planar tensors, for both merged [B, S, H*D] and explicit [B, H, S, D]
/// layouts (a head's D columns are contiguous within each row). The mask is
/// flat [Sq, Sk]; its index carries no batch or head.
///
/// Tile sizes and the scratch layout come from a single plan
/// (get_attention_plan), so the kernels and the workspace sizing never drift.
/// On the standard route, per (batch, head) or per worker thread, a scratch
/// slot is: [scores : Sq*Sk][pack_b : panels_for_pack(nc) * ldd_b]
/// pack_b holds the panels of K^T for GEMM1 and then of V for GEMM2; one slot
/// serves both.
///
/// Grouped-query attention (attrs.num_group) is not handled — K and V are
/// addressed with Q's head count and head size, i.e. standard MHA.

#include "nnops/core/tensor_view.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/ops/attention.hpp"

#include <cstddef>
#include <span>

namespace nnops::backend::cpu {

/// Tile sizes + scratch layout, resolved once and shared by the kernels and
/// the scratch sizing.
struct AttentionPlan {
    bool   use_flash = false;   ///< route through the FlashAttention path
    int    mc1 = 0, nc1 = 0;    ///< standard-path GEMM1 (Sq × Sk) tiles
    int    mc2 = 0, nc2 = 0;    ///< standard-path GEMM2 (Sq × D) tiles
    int    Br = 0, Bc = 0;      ///< flash-path query/KV block tiles
    int64_t num_slots = 0;      ///< scratch slots (worker threads, else tasks)
    size_t workspace_size = 0;  ///< pooled scratch bytes (0 → no tiled route)
};

/// Resolve tile sizes + scratch bytes for f32/f16 (tiles and element size
/// follow the dtype). Any other dtype, or a Q/K/V dtype mismatch, returns a
/// zeroed plan (workspace_size 0). @p num_threads and @p use_thread_slots
/// bound the per-thread scratch slot count; @p outputs is unused.
AttentionPlan get_attention_plan(const AttentionAttributes& attrs,
                                 std::span<const TensorDesc> inputs,
                                 std::span<const TensorDesc> outputs,
                                 int num_threads = 1,
                                 bool use_thread_slots = false);

/// Tiled attention kernel (f32 / f16), routed standard or flash by the plan.
/// Returns Status::unsupported unless Q, K, V, the output and the optional mask
/// all share one f32 or f16 dtype. Scratch comes from the internal pool, sized
/// by get_attention_plan; @p workspace is unused.
Status attention_cpu(const AttentionAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace);

}  // namespace nnops::backend::cpu
