#pragma once
/// @file attention.h
/// @brief Multi-head scaled dot-product attention — tiled GEMM + softmax kernel.
///
/// For each (batch, head) — parallel via ctx.cpu.run:
///   scores = Q @ K^T * scale      (transpose-B GEMM, K^T packed into pooled scratch)
///   scores += mask                (optional, flat [Sq, Sk] as in the reference)
///   attn   = softmax(scores, -1)  (simd_softmax per contiguous row)
///   output = attn @ V             (direct GEMM, V raw K×N)
///
/// Both GEMMs reuse the tile_mma_direct / tile_pack_rhs micro-kernel dispatch
/// from matmul_helper.h. Q/K/V/O head slices are M×K / N×K / K×N / M×N
/// submatrices of the planar tensors, for both merged [B, S, H*D] and explicit
/// [B, H, S, D] layouts (a head's D columns are contiguous within each row).
///
/// Tile sizes and the workspace layout come from a single plan
/// (get_attention_plan), so the kernels and the workspace sizing never drift.
///
/// Grouped-query attention (attrs.num_group) is not yet handled; the kernel
/// runs standard MHA (one head block per (batch, head)).
///
/// Scratch layout (pooled internally; per (batch, head) slice, thread-count
/// invariant):
///   [scores : Sq*Sk]
///   [pack_b : num_panels_max(nc, NR) * ldd_b  (transposed-K^T panel, GEMM1)]

#include "nnops/core/tensor_view.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/ops/attention.hpp"

#include <cstddef>
#include <span>

namespace nnops::backend::cpu {

/// Single source of truth for the attention tile sizes and workspace layout,
/// shared by the kernels and the workspace sizing (get_attention_plan).
struct AttentionPlan {
    bool   use_flash = false;   ///< route through the FlashAttention path
    int    mc1 = 0, nc1 = 0;    ///< standard-path GEMM1 (Sq × Sk) tiles
    int    mc2 = 0, nc2 = 0;    ///< standard-path GEMM2 (Sq × D) tiles
    int    Br = 0, Bc = 0;      ///< flash-path query/KV block tiles
    int64_t num_slots = 0;      ///< scratch slots (worker threads, else tasks)
    size_t workspace_size = 0;  ///< pooled scratch bytes (0 → reference fallback)
};

/// Resolve the attention plan (tile sizes + workspace bytes) from the
/// descriptors, for f32 or f16 (the two dtypes with a tiled fast path; the
/// tile sizes and scratch element size follow the dtype). Returns a zeroed plan
/// (workspace_size 0) for any other dtype, or when Q/K/V dtypes disagree —
/// which routes the kernel to the reference. @p num_threads bounds the
/// per-thread scratch slot count when @p use_thread_slots is set.
AttentionPlan get_attention_plan(const AttentionAttributes& attrs,
                                 std::span<const TensorDesc> inputs,
                                 std::span<const TensorDesc> outputs,
                                 int num_threads = 1,
                                 bool use_thread_slots = false);

/// Tiled attention kernel (f32 / f16). Falls back to the reference kernel for
/// an f32 dtype mismatch; an f16 mismatch is a hard error (the reference is
/// f32-only, so it would silently read f16 as f32). Scratch is pooled
/// internally (sized by get_attention_plan); @p workspace is vestigial.
void attention_cpu(const AttentionAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace);

}  // namespace nnops::backend::cpu
