#pragma once
/// @file matmul.h
/// @brief Tiled matrix multiplication kernel using the pack + MMA interfaces.
///        Loop order is plan-driven: NKM (split on N) or MKN (split on M),
///        selected by get_matmul_plan() in matmul_helper.h.
///
/// Scratch (pack buffer for B) is pooled internally, sized by the plan;
/// packed A lives on the kernel stack.

#include <cstddef>
#include <span>
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/ops/matmul.hpp"
#include "matmul_helper.h"

namespace nnops::backend::cpu {

/// Tiled matrix multiplication: C = A × B.
///
/// Uses the MatMulPlan and pack/MMA dispatch from matmul_helper. Scratch is
/// pooled internally (sized by the plan); @p workspace is vestigial.
///
/// Falls back to the reference kernel for non-f32/f16 dtypes.
void matmul_kernel(const MatMulAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx,
                   void* workspace);

}  // namespace nnops::backend::cpu
