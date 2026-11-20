#pragma once
/// @file conv2d_i8.cuh
/// @brief The quantized Conv2D route's entry, called by conv2d_cuda.
///
/// The kernel pair, its split-K reduce and the packed-filter geometry live in
/// conv2d_i8.cu; conv2d.cu owns the float family and the operator entry. The
/// route admits a pre-quantized u8/s8 activation against an s8 HWIO filter (raw
/// or PackedHWIOCuda) and answers everything else with unsupported.

#include "nnops/core/compute_context.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/ops/conv2d.hpp"

#include <span>

namespace nnops::backend::cuda {

Status conv2d_i8_cuda(const Conv2DAttributes& attrs, TensorView& output,
                      std::span<const TensorView> inputs, const ComputeContext& ctx);

} // namespace nnops::backend::cuda
