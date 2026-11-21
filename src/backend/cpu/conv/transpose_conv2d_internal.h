#pragma once
/// @file transpose_conv2d_internal.h
/// @brief Internals shared by the two TransposeConv2D translation units.
///
/// The operator is split one file per arithmetic family, the split Conv2D and
/// MatMul use:
///
///   - transpose_conv2d.cpp    — the float routes (pixel gather, col-matrix
///                               GEMM, depthwise) and the entry point;
///   - transpose_conv2d_i8.cpp — the quantized routes (s8 weight, u8/s8/f32/f16
///                               activation, int32 accumulate) and their
///                               dequant/requant merges.
///
/// The geometry is one struct for both: the routes read the same numbers and
/// only disagree on what an element is.

#include "backend/cpu/conv/conv_util.h"
#include "backend/cpu/conv/transpose_conv_common.h"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/ops/transpose_conv2d.hpp"

#include <cstdint>
#include <span>

namespace nnops::backend::cpu {

/// The quantized family's entry (transpose_conv2d_i8.cpp): s8 weights only; the
/// activation ladder and the output ladder are the route's own.
Status transpose_conv2d_int8_cpu(const TransposeConv2DAttributes& attrs, const ConvChannels& gm,
                                 TensorView& output, std::span<const TensorView> inputs,
                                 const ComputeContext& ctx);

} // namespace nnops::backend::cpu
