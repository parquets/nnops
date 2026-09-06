#pragma once
/// @file attention.h
/// @brief Multi-head scaled dot-product attention — tiled GEMM + softmax kernel.
///
/// For each (batch, head) — parallel via ctx.cpu_parallel_for:
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
/// Grouped-query attention (attrs.num_group) is not yet handled; the kernel
/// runs standard MHA (one head block per (batch, head)).
///
/// Scratch layout (pooled internally; per (batch, head) slice, thread-count
/// invariant):
///   [scores : Sq*Sk]
///   [pack_b : num_panels(nc, NR) * ldd_b   (transposed-K^T panel, GEMM1)]

#include "nnops/core/tensor_view.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/ops/attention.hpp"

#include <cstddef>
#include <span>

namespace nnops::backend::cpu {

/// Tiled attention kernel (f32). Falls back to the reference kernel when the
/// dtype is not f32 (the reference is f32-only). Scratch is pooled internally
/// (sized by attention_get_workspace_size); @p workspace is vestigial.
void attention_kernel(const AttentionAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace);

/// Scratch bytes pooled by attention_kernel for the given descriptors.
///
/// Returns 0 for non-f32 dtypes (reference fallback, no scratch).
size_t attention_get_workspace_size(const AttentionAttributes& attrs,
                                    std::span<const TensorDesc> inputs,
                                    std::span<const TensorDesc> outputs);

}  // namespace nnops::backend::cpu
