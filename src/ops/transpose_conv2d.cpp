/// @file transpose_conv2d.cpp
/// @brief TransposeConv2D operator dispatch — connects class API to backend implementations.
///
/// One grouped geometry is not a GEMM either: a group of one input and one
/// output channel is a depthwise transposed convolution, and compute() hands it
/// to the depthwise kernels. The filter is [IC, OC/G, KH, KW] = [C, 1, KH, KW]
/// in both layouts — the channel-first depthwise form the kernels read in
/// place; on a channel-last activation its channel axis is the outer one, so
/// the depthwise entry repacks it to [KH][KW][C] once per call.

#include "nnops/ops/transpose_conv2d.hpp"
#include "nnops/ops/depthwise_transpose_conv.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(TransposeConv2DAttributes, transpose_conv2d_cpu)
NNOPS_DECLARE_CPU(DepthwiseTransposeConv2DAttributes, depthwise_transpose_conv_cpu)
NNOPS_DECLARE_CUDA(TransposeConv2DAttributes, transpose_conv2d_cuda)

namespace {

bool is_fp(DataType dtype) noexcept
{
    return dtype == DataType::f32 || dtype == DataType::f16;
}

/// One input and one output channel per group, with the filter in the
/// channel-first order the depthwise kernels take.
bool is_depthwise_geometry(const TransposeConv2DAttributes& attrs, const TensorView& weight)
{
    if (attrs.groups <= 1) { return false; }
    if (!is_channel_first_spatial(weight.layout()) || weight.rank() != 4) { return false; }
    return weight.shape(1) == 1 && weight.shape(0) == attrs.groups;
}

}  // namespace

NNOPS_OP_BOILERPLATE(TransposeConv2D, TransposeConv2DAttributes,
        backend::cpu::transpose_conv2d_cpu,
        NNOPS_IF_CUDA(backend::cuda::transpose_conv2d_cuda), nullptr)

std::vector<TensorDesc> TransposeConv2D::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    std::vector<TensorDesc> outs = transpose_conv2d_output_shape(
        attrs_.kernel_size, attrs_.stride, attrs_.dilation, attrs_.padding,
        attrs_.output_padding, attrs_.groups, inputs);

    // The quantized route writes s8/u8 when output_dtype says so, else the fp
    // activation's dtype (output_dtype for a pre-quantized one); an unsupported
    // request passes through for the kernel to refuse.
    if (inputs.size() > 1 && is_quantized_transpose_conv2d(inputs[1].dtype)) {
        const DataType act = inputs[0].dtype;
        const bool fp_act = (act == DataType::f32 || act == DataType::f16);
        const DataType odt = attrs_.output_dtype;
        const bool int_out = odt == DataType::s8 || odt == DataType::u8;
        outs[0].dtype = (int_out || !fp_act) ? odt : act;
    }
    return outs;
}

Status TransposeConv2D::validate(std::span<const TensorDesc> inputs,
                                 std::span<const TensorDesc> outputs) const
{
    // input, weight [, bias]
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 2, 3, 1));
    // The fp routes fold the residual in before the epilogue; the quantized
    // route refuses add_to outright, so it has no pre-activation order either.
    if (attrs_.add_to && attrs_.residual_pre_activation && inputs.size() > 1 &&
        is_quantized_transpose_conv2d(inputs[1].dtype)) {
        return Status::unsupported;
    }
    return detail::require(
        is_layout_supported(inputs[0].layout, LayoutSupport::ChannelFirstOrChannelLast));
}

Status TransposeConv2D::compute(std::span<TensorView> outputs,
                                std::span<const TensorView> inputs,
                                const ComputeContext& ctx,
                                void* workspace)
{
    NNOPS_TRY(detail::check_views(outputs, inputs, 2, 3, 1));
    NNOPS_TRY(validate_views(outputs, inputs));

    // A depthwise geometry reads the filter this operator was handed, so it
    // costs nothing to hand the call over. The col GEMM degenerates there — one
    // channel per group is a single-column product, re-dispatched per group —
    // while the depthwise kernels accumulate the pixels directly.
    if (backend_ == Backend::CPU && inputs.size() >= 2 &&
        is_fp(inputs[0].data_type()) &&
        inputs[1].data_type() == inputs[0].data_type() &&
        (is_channel_first_spatial(inputs[0].layout()) ||
         is_channel_last(inputs[0].layout())) &&
        is_depthwise_geometry(attrs_, inputs[1])) {
        DepthwiseTransposeConv2DAttributes dw;
        dw.kernel_size     = attrs_.kernel_size;
        dw.stride          = attrs_.stride;
        dw.dilation        = attrs_.dilation;
        dw.padding         = attrs_.padding;
        dw.output_padding  = attrs_.output_padding;
        dw.epilogue        = attrs_.epilogue;
        dw.add_to          = attrs_.add_to;
        dw.residual_pre_activation = attrs_.residual_pre_activation;
        dw.output_dtype    = attrs_.output_dtype;
        dw.act_quant_dtype = attrs_.act_quant_dtype;
        return backend::cpu::depthwise_transpose_conv_cpu(dw, outputs[0], inputs, ctx,
                                                          workspace);
    }

    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
