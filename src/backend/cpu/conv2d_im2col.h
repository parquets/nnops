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
/// Bias is fused into the accumulator init; the epilogue (activation) is
/// applied in-place after the last ic-block; `add_to` follows the reference
/// semantics (epilogue applied to bias + Σ, then added to the pre-existing
/// output), preserving the original tile in a scratch buffer when needed.
///
/// Workspace layout (per (batch, group) slice, reused across oh-blocks):
///   [col_data : icn_block*karea*oh_block*out_w]
///   [orig     : ocn_block*oh_block*out_w   (add_to only)]
///
/// col_data is the im2col matrix [K=icn_block*karea][N=oh_block*out_w];
/// orig preserves the output tile for the exact add_to semantics.

#include "nnops/core/tensor_view.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/ops/conv2d.hpp"

#include <cstddef>
#include <span>

namespace nnops::backend::cpu {

/// Tiled im2col + direct GEMM Conv2D kernel (f32 / f16, planar NCHW).
///
/// Falls back to the reference kernel for f32 when @p workspace is null or the
/// dtypes do not match (the reference is f32-only; f16 requires a workspace).
/// The caller must provide @p workspace of at least
/// conv2d_im2col_get_workspace_size() bytes.
void conv2d_im2col_kernel(const Conv2DAttributes& attrs,
                          TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx,
                          void* workspace);

/// Workspace bytes required by conv2d_im2col_kernel for the given descriptors.
///
/// Returns 0 for unsupported dtypes (f32/f16 only, all tensors matching).
size_t conv2d_im2col_get_workspace_size(const Conv2DAttributes& attrs,
                                        std::span<const TensorDesc> inputs,
                                        std::span<const TensorDesc> outputs);

}  // namespace nnops::backend::cpu
