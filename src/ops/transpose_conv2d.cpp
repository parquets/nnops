/// @file transpose_conv2d.cpp
/// @brief TransposeConv2D operator dispatch — connects class API to backend implementations.

#include "nnops/ops/transpose_conv2d.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(TransposeConv2DAttributes, transpose_conv2d_cpu)

NNOPS_OP_BOILERPLATE(TransposeConv2D, TransposeConv2DAttributes,
        backend::cpu::transpose_conv2d_cpu, nullptr, nullptr)

std::vector<TensorDesc> TransposeConv2D::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return transpose_conv2d_output_shape(
        attrs_.kernel_size, attrs_.stride, attrs_.dilation, attrs_.padding,
        attrs_.output_padding, attrs_.groups, inputs);
}

Status TransposeConv2D::validate(std::span<const TensorDesc> inputs,
                                 std::span<const TensorDesc> outputs) const
{
    // input, weight [, bias]
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 2, 3, 1));
    return detail::require(
        is_layout_supported(inputs[0].layout, LayoutSupport::PackedOnly));
}

Status TransposeConv2D::compute(std::span<TensorView> outputs,
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
