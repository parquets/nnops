#pragma once
/// @file depthwise_conv_internal.h
/// @brief Internals shared by the two DepthwiseConv translation units.
///
/// The operator is split one file per arithmetic family:
///
///   - depthwise_conv.cpp    — the float route (lane on C, raw HWCM filter) and
///                             the entry point's dtype dispatch;
///   - depthwise_conv_i8.cpp — the quantized route (s8/u8 activation and weight,
///                             dequantized per row into a staging vector).
///
/// Both run the same task grid — a strip of rows, cut to fit L2, with the
/// filter axis splitting a strip into channel chunks when the grid cannot fill
/// the pool — and walk the same window, so the params live here and the grid
/// arithmetic is conv_util.h's. Only the channel tile differs (16 lanes fp, 8
/// quantized).

#include "backend/cpu/simd_kernel/simd_depthwise_conv.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/ops/depthwise_conv.hpp"

#include <span>

namespace nnops::backend::cpu {

/// The window and the filter's access strides, filled from the activation and
/// the destination. Channel-last: a pixel and a tap are both C elements wide,
/// and a 2-D call leaves the D axis at its defaults.
inline kernel::DwConvParams dwconv_params(const DepthwiseConvAttributes& attrs,
                                          const TensorView& in, const TensorView& out) noexcept
{
    kernel::DwConvParams p{};
    const int64_t srank = DepthwiseConvAttributes::spatial_rank(in.rank());
    const int64_t C = in.shape(in.rank() - 1);
    p.ID = (srank == 3) ? in.shape(1) : 1;
    p.IH = in.shape(srank - 1);
    p.IW = in.shape(srank);
    p.OH = out.shape(srank - 1);
    p.OW = out.shape(srank);
    p.KD = attrs.kernel_size[0];
    p.KH = attrs.kernel_size[1];
    p.KW = attrs.kernel_size[2];
    p.SD = attrs.stride[0];
    p.SH = attrs.stride[1];
    p.SW = attrs.stride[2];
    p.DD = attrs.dilation[0];
    p.DH = attrs.dilation[1];
    p.DW = attrs.dilation[2];
    p.PD = attrs.padding[0];
    p.PH = attrs.padding[1];
    p.PW = attrs.padding[2];
    p.out_row_stride = out.row_stride_elems();
    p.in_row_stride = in.row_stride_elems();
    p.in_d_stride = p.IH * p.in_row_stride;
    p.pix_stride = C;   // a pixel holds all C channels, and so does a tap
    p.w_tap_stride = C; // the raw HWCM filter is [tap][C]
    return p;
}

/// The quantized family's entry (depthwise_conv_i8.cpp): s8/u8 activation and
/// weight, the destination picking the requantize.
Status dwconv_quant_run(const DepthwiseConvAttributes& attrs, TensorView& output,
                        std::span<const TensorView> inputs, const ComputeContext& ctx);

} // namespace nnops::backend::cpu
