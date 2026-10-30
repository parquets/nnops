/// @file depthwise_conv.cpp
/// @brief DepthwiseConv operator dispatch — descriptor checks and backend binding.
///
/// Both routes read the raw filter: planar (NCHW/NCDHW) the [C, 1, (KD,) KH, KW]
/// form with a [C] bias, channel-last (NHWC/NDHWC) the HWCM [KH][KW][C] form, whose
/// every tap is a contiguous run of C channels — the pattern the channel-lane
/// kernel already reads a vector at a time, so no prepack is involved either way.
/// The channel lane also covers the quantized s8/u8 pairs; the planar route is
/// f32/f16 only. Spatial rank follows the input rank (4→2D, 5→3D).

#include "nnops/ops/depthwise_conv.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

#include <cstdint>

namespace nnops {

NNOPS_DECLARE_CPU(DepthwiseConvAttributes, depthwise_conv_cpu)
NNOPS_DECLARE_CUDA(DepthwiseConvAttributes, depthwise_conv_cuda)

NNOPS_OP_BOILERPLATE(DepthwiseConv, DepthwiseConvAttributes,
        backend::cpu::depthwise_conv_cpu, NNOPS_IF_CUDA(backend::cuda::depthwise_conv_cuda), nullptr)

std::vector<TensorDesc> DepthwiseConv::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return {depthwise_conv_output_shape(
        attrs_.kernel_size, attrs_.stride, attrs_.dilation, attrs_.padding,
        0 /* NOTSET */, inputs)};
}

Status DepthwiseConv::validate(std::span<const TensorDesc> inputs,
                               std::span<const TensorDesc> outputs) const
{
    // input, weight [, bias]
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 2, 3, 1));
    const auto& in = inputs[0];
    if (is_channel_last(in.layout)) {
        // The channel lane loads a tap's channels as one vector, which the planar
        // [C, 1, KH, KW] cannot give it (a tap's channels are C elements apart).
        if (inputs[1].layout != TensorLayout::HWCM) {
            return Status::unsupported;
        }
        return detail::require(
            detail::dtype_in<DataType::f32, DataType::f16,
                             DataType::s8, DataType::u8>(in.dtype));
    }
    // Planar reads the raw filter one channel plane at a time, f32/f16 only.
    // Nothing produces PackedWeight any more, so it is rejected here rather than
    // misread as a raw filter.
    if (!is_planar_spatial(in.layout)) {
        return Status::unsupported;
    }
    return detail::require(
        detail::dtype_in<DataType::f32, DataType::f16>(in.dtype) &&
            inputs[1].layout != TensorLayout::PackedWeight,
        Status::unsupported);
}

Status DepthwiseConv::compute(std::span<TensorView> outputs,
                              std::span<const TensorView> inputs,
                              const ComputeContext& ctx,
                              void* workspace)
{
    NNOPS_TRY(detail::check_views(outputs, inputs, 2, 3, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
