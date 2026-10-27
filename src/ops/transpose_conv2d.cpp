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

void TransposeConv2D::compute(std::span<TensorView> outputs,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* workspace)
{
    NNOPS_ASSERT(inputs.size() >= 2);
    NNOPS_ASSERT(inputs.size() <= 3);  // input, weight [, bias]
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());
    NNOPS_ASSERT(!inputs[1].is_empty());
    NNOPS_ASSERT(is_layout_supported(inputs[0].layout(), LayoutSupport::PackedOnly));

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
