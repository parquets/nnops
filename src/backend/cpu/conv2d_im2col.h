#pragma once
/// @file conv2d_im2col.h
/// @brief Planar (NCHW) Conv2D — tiled im2col + direct GEMM kernel.
///
/// Ported from nn_compute (`src/cpu/im2col_conv2d_cpu.cpp` +
/// `kernel/conv2d/nchw/implicit_gemm_conv2d.hpp` +
/// `kernel/conv2d/im2col_2d.hpp`). The output spatial plane is split into
/// oh-blocks; for each block the input is materialized into a per-(batch,
/// group) im2col buffer via tiled_im2col_2d, then the output tile is
/// accumulated with the raw (unpacked) weight through the same `tile_mma_direct`
/// micro-kernel dispatch that MatMul uses (matmul_helper.h).
///
/// Work is split at (batch, group, oh-chunk) granularity: when (batch, group)
/// alone cannot fill the thread pool (the common batch=1 / groups=1 case), the
/// oh-blocks are partitioned into parallel chunks, each owning its own im2col
/// scratch. Output tiles are disjoint, so the result is bit-identical to serial
/// execution.
///
/// Bias is fused into the accumulator init; the epilogue (activation) is
/// applied in-place after the last ic-block; `add_to` follows the reference
/// semantics (epilogue applied to bias + Σ, then added to the pre-existing
/// output), preserving the original tile in a scratch buffer when needed.
///
/// Block sizes, the parallel decomposition, and the scratch layout all come
/// from one plan (get_conv2d_plan), so the kernel and its sizing never drift.
///
/// Scratch layout (pooled internally; per task, reused across the task's
/// oh-blocks):
///   [col_data : icn_block*karea*oh_block*out_w]
///   [orig     : ocn_block*oh_block*out_w   (add_to only)]
///
/// col_data is the im2col matrix [K=icn_block*karea][N=oh_block*out_w];
/// orig preserves the output tile for the exact add_to semantics.

#include "nnops/core/tensor_view.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/ops/conv2d.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace nnops::backend::cpu {

/// Single source of truth for the im2col block sizes, the oh-chunk parallel
/// decomposition, and the scratch layout (get_conv2d_plan).
struct Conv2DPlan {
    int64_t icn_block = 0;         ///< input-channel block
    int64_t ocn_block = 0;         ///< output-channel block
    int64_t oh_block  = 0;         ///< output-row block
    int64_t num_oh_blocks = 1;     ///< ceil(OH / oh_block)
    int64_t oh_chunks     = 1;     ///< parallel oh-chunks per (batch, group)
    int64_t col_size  = 0;         ///< im2col scratch elements per task
    int64_t orig_size = 0;         ///< add_to scratch elements per task
    size_t  workspace_size = 0;    ///< pooled scratch bytes (0 → reference fallback)
};

/// Resolve the conv2d plan (block sizes, oh-chunk split, workspace bytes) from
/// the descriptors. Returns a zeroed plan (workspace_size 0) for unsupported
/// dtypes (f32/f16 only, all tensors matching), which routes the kernel to the
/// reference.
Conv2DPlan get_conv2d_plan(const Conv2DAttributes& attrs,
                           std::span<const TensorDesc> inputs,
                           std::span<const TensorDesc> outputs,
                           int num_threads);

/// Tiled im2col + direct GEMM Conv2D kernel (f32 / f16, planar NCHW).
///
/// Falls back to the reference kernel for f32 when the dtypes do not match (the
/// reference is f32-only). Scratch is pooled internally (sized by
/// get_conv2d_plan); @p workspace is vestigial.
void conv2d_im2col_kernel(const Conv2DAttributes& attrs,
                          TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx,
                          void* workspace);

}  // namespace nnops::backend::cpu
