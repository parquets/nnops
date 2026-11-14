/// @file depthwise_transpose_conv.cpp
/// @brief DepthwiseTransposeConv2D operator dispatch — descriptor checks and
///        backend binding.
///
/// One filter per channel, read raw: a channel-first activation takes the
/// [C, 1, KH, KW] filter, a channel-last one the HWCM [KH][KW][C] form whose
/// every tap is a contiguous run of C channels. There is no prepack either way
/// — the storage order already is the access pattern the kernel wants. A
/// channel-last activation also takes the [C, 1, KH, KW] filter, because that
/// is what a grouped TransposeConv2D hands over when its group count meets its
/// channel count; the entry repacks it to the per-tap order.

#include "nnops/ops/depthwise_transpose_conv.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "nnops/detail/validate.hpp"

#include <cstdint>

namespace nnops {

NNOPS_DECLARE_CPU(DepthwiseTransposeConv2DAttributes, depthwise_transpose_conv_cpu)
NNOPS_DECLARE_CUDA(DepthwiseTransposeConv2DAttributes, depthwise_transpose_conv_cuda)

NNOPS_OP_BOILERPLATE(DepthwiseTransposeConv2D, DepthwiseTransposeConv2DAttributes,
        backend::cpu::depthwise_transpose_conv_cpu,
        NNOPS_IF_CUDA(backend::cuda::depthwise_transpose_conv_cuda), nullptr)

std::vector<TensorDesc> DepthwiseTransposeConv2D::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    std::vector<TensorDesc> outs = depthwise_transpose_conv_output_shape(
        attrs_.kernel_size, attrs_.stride, attrs_.dilation, attrs_.padding,
        attrs_.output_padding, inputs);

    // The quantized route writes s8/u8 when output_dtype says so, else the fp
    // activation's dtype (output_dtype for a pre-quantized one); an unsupported
    // request passes through for the kernel to refuse.
    if (inputs.size() > 1 &&
        is_quantized_depthwise_transpose_conv2d(inputs[1].dtype)) {
        const DataType act = inputs[0].dtype;
        const bool fp_act = (act == DataType::f32 || act == DataType::f16);
        const DataType odt = attrs_.output_dtype;
        const bool int_out = odt == DataType::s8 || odt == DataType::u8;
        outs[0].dtype = (int_out || !fp_act) ? odt : act;
    }
    return outs;
}

Status DepthwiseTransposeConv2D::validate(std::span<const TensorDesc> inputs,
                                          std::span<const TensorDesc> outputs) const
{
    // input, weight [, bias]
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 2, 3, 1));
    const auto& in = inputs[0];
    const auto& weight = inputs[1];

    NNOPS_TRY(detail::require(in.rank == 4));

    if (is_quantized_depthwise_transpose_conv2d(weight.dtype)) {
        // The fp kernels fold the residual in before the epilogue; the integer
        // accumulate has no fold for it (add_to is refused below the same way).
        if (attrs_.add_to && attrs_.residual_pre_activation) {
            return Status::unsupported;
        }
        // s8 weights select the quantized route: the activation arrives fp and
        // is quantized inside, or already quantized; the bias is int32 on an
        // integer destination and the output dtype otherwise.
        NNOPS_TRY(detail::require(
            detail::dtype_in<DataType::f32, DataType::f16, DataType::s8,
                             DataType::u8>(backend_, in.dtype),
            Status::unsupported));
        if (inputs.size() > 2) {
            const DataType odt = outputs[0].dtype;
            const bool int_out = odt == DataType::s8 || odt == DataType::u8;
            NNOPS_TRY(detail::require(
                inputs[2].dtype == (int_out ? DataType::s32 : odt),
                Status::unsupported));
        }
    } else {
        NNOPS_TRY(detail::require(
            detail::dtype_in<DataType::f32, DataType::f16>(backend_, in.dtype) &&
            weight.dtype == in.dtype,
            Status::unsupported));
        if (inputs.size() > 2 && inputs[2].dtype != in.dtype) {
            return Status::unsupported;
        }
    }

    if (is_channel_last(in.layout)) {
        // The channel lane loads a tap's channels as one vector. The per-tap
        // [KH][KW][C] filter gives it that directly; the transpose-conv
        // [C, 1, KH, KW] the routed grouped call carries has the channel axis
        // outermost, and the entry repacks it — the fp kernels only, since the
        // int8 route reads the per-channel scales off that filter in place.
        if (weight.layout == TensorLayout::HWCM && weight.rank == 3) {
            return detail::require(weight.dims[2] == in.dims[1]);
        }
        NNOPS_TRY(detail::require(
            !is_quantized_depthwise_transpose_conv2d(weight.dtype),
            Status::unsupported));
        NNOPS_TRY(detail::require(
            is_channel_first_spatial(weight.layout) && weight.rank == 4,
            Status::unsupported));
        return detail::require(weight.dims[1] == 1 && weight.dims[0] == in.dims[1]);
    }
    if (is_channel_first_spatial(in.layout)) {
        NNOPS_TRY(detail::require(
            is_channel_first_spatial(weight.layout) && weight.rank == 4,
            Status::unsupported));
        return detail::require(weight.dims[0] == in.dims[1]);
    }
    return Status::unsupported;
}

Status DepthwiseTransposeConv2D::compute(std::span<TensorView> outputs,
                                         std::span<const TensorView> inputs,
                                         const ComputeContext& ctx,
                                         void* workspace)
{
    NNOPS_TRY(detail::check_views(backend_, outputs, inputs, 2, 3, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
