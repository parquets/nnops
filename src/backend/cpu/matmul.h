#pragma once
/// @file matmul.h
/// @brief Tiled matrix multiplication kernel using the imatmul packing and MMA
///        interfaces.  Loop order: NKM (N outer, K middle, M inner).
///
/// Workspace (pack buffers for A and B) is allocated externally — use
/// matmul_get_workspace_size() to determine the required size.

#include <cstddef>
#include <span>
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/ops/matmul.hpp"

namespace nnops::backend::cpu {

/// Compute the workspace size (bytes) required by matmul_kernel for the given
/// tensor descriptors and operator attributes.
size_t matmul_get_workspace_size(const MatMulAttributes& attrs,
                                 const TensorDesc& a_desc,
                                 const TensorDesc& b_desc,
                                 const TensorDesc& c_desc);

/// Tiled matrix multiplication: C = A x B   (2D GEMM, NKM loop order).
///
/// Uses the packing and MMA micro-kernel dispatch from imatmul.
/// The caller must pre-allocate a workspace buffer of at least
/// matmul_get_workspace_size() bytes and pass it via @p workspace.
///
/// Falls back to reference matmul for batched (rank > 2) or non-f32/f16 dtypes.
void matmul_kernel(const MatMulAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx,
                   void* workspace);

}  // namespace nnops::backend::cpu
