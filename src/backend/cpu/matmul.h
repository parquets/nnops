#pragma once
/// @file matmul.h
/// @brief Tiled matrix multiplication kernel using the pack + MMA interfaces.
///        Loop order is plan-driven: NKM (N outer, K middle, M inner) or MKN
///        (k{m{n}}), selected by the static pack decision in matmul_helper.h.
///
/// Workspace (pack buffer for B) is allocated externally — use
/// matmul_get_workspace_size() (declared in matmul_helper.h) to determine the
/// required size (nonzero iff the global plan packs B; packed A lives on the
/// kernel stack).

#include <cstddef>
#include <span>
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/ops/matmul.hpp"
#include "matmul_helper.h"

namespace nnops::backend::cpu {

/// Tiled matrix multiplication: C = A x B   (2D GEMM, NKM loop order).
///
/// Uses the packing and MMA micro-kernel dispatch from imatmul.
/// The caller must pre-allocate a workspace buffer of at least
/// matmul_get_workspace_size() bytes (0 unless transpose_b) and pass it via
/// @p workspace.
///
/// Falls back to reference matmul for batched (rank > 2) or non-f32/f16 dtypes.
void matmul_kernel(const MatMulAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx,
                   void* workspace);

}  // namespace nnops::backend::cpu
