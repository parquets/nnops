#pragma once
/// @file depthwise_conv_internal.h
/// @brief Internals shared by the two DepthwiseConv translation units.
///
/// The operator is split one file per arithmetic family:
///
///   - depthwise_conv.cpp    — the float route (lane on C, raw HWCM filter) and
///                             the entry point's dtype dispatch;
///   - i8depthwise_conv.cpp  — the quantized route (s8/u8 activation and weight,
///                             dequantized per row into a staging vector).
///
/// Both run the same task grid — a strip of rows, cut to fit L2, with the
/// filter axis splitting a strip into channel chunks when the grid cannot fill
/// the pool — so only the entry declaration is shared here.

#include "nnops/core/compute_context.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/ops/depthwise_conv.hpp"

#include <span>

namespace nnops::backend::cpu {

/// The quantized family's entry (i8depthwise_conv.cpp): s8/u8 activation and
/// weight, the destination picking the requantize.
Status dwconv_quant_run(const DepthwiseConvAttributes& attrs, TensorView& output,
                        std::span<const TensorView> inputs, const ComputeContext& ctx);

} // namespace nnops::backend::cpu
