/// @file depthwise_conv.cpp
/// @brief DepthwiseConv operator dispatch — descriptor checks and backend binding.
///
/// The filter is the HWCM [KH][KW][C] form (3D: [KD][KH][KW][C]) with a [C]
/// bias, whose every tap is a contiguous run of C channels — the pattern the
/// channel-lane kernel already reads a vector at a time, so there is no
/// prepack. The channel lane also covers the quantized s8/u8 pairs. Spatial
/// rank follows the input rank (4→2D, 5→3D).

#include "nnops/ops/depthwise_conv.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

#include <cstdint>

namespace nnops {

NNOPS_DECLARE_CPU(DepthwiseConvAttributes, depthwise_conv_cpu)
NNOPS_DECLARE_CUDA(DepthwiseConvAttributes, depthwise_conv_cuda)

NNOPS_OP_BOILERPLATE(DepthwiseConv, DepthwiseConvAttributes, backend::cpu::depthwise_conv_cpu,
                     NNOPS_IF_CUDA(backend::cuda::depthwise_conv_cuda), nullptr)

std::vector<TensorDesc> DepthwiseConv::getOutputTensorDesc(std::span<const TensorDesc> inputs) const
{
    return {depthwise_conv_output_shape(attrs_.kernel_size, attrs_.stride, attrs_.dilation,
                                        attrs_.padding, 0 /* NOTSET */, inputs)};
}

Status DepthwiseConv::validate(std::span<const TensorDesc> inputs,
                               std::span<const TensorDesc> outputs) const
{
    // input, weight [, bias]
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 2, 3, 1));
    const auto& in = inputs[0];
    // The CPU kernels fold the residual into the accumulate before the epilogue;
    // the CUDA kernel adds it after, so refusing beats applying it on the wrong side.
    if (attrs_.add_to && attrs_.residual_pre_activation && backend_ != Backend::CPU) {
        return Status::unsupported;
    }
    // The channel lane loads a tap's channels as one vector, so the filter must
    // be the per-tap HWCM order.
    if (!is_channel_last(in.layout) || inputs[1].layout != TensorLayout::HWCM) {
        return Status::unsupported;
    }
    return detail::require(
        detail::dtype_in<DataType::f32, DataType::f16, DataType::s8, DataType::u8>(backend_,
                                                                                   in.dtype));
}

Status DepthwiseConv::compute(std::span<TensorView> outputs, std::span<const TensorView> inputs,
                              const ComputeContext& ctx, void* workspace)
{
    NNOPS_TRY(detail::check_views(backend_, outputs, inputs, 2, 3, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) {
        return Status::unsupported;
    }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

} // namespace nnops
