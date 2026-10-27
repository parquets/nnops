/// @file conv3d.cpp
/// @brief Conv3D operator dispatch — connects class API to backend implementations.

#include "nnops/ops/conv3d.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(Conv3DAttributes, conv3d_cpu)

NNOPS_OP_BOILERPLATE(Conv3D, Conv3DAttributes,
        backend::cpu::conv3d_cpu, nullptr, nullptr)

std::vector<TensorDesc> Conv3D::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return {conv3d_output_shape(
        attrs_.kernel_size, attrs_.stride, attrs_.dilation, attrs_.padding,
        static_cast<int>(attrs_.auto_pad), attrs_.groups, inputs)};
}

void Conv3D::compute(std::span<TensorView> outputs,
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
    NNOPS_ASSERT(is_layout_supported(inputs[0].layout(), LayoutSupport::PlanarOnly));

    // auto_pad lives in the descriptor; kernels only see explicit leading padding.
    Conv3DAttributes attrs = attrs_;
    if (attrs.auto_pad != Conv3DAttributes::AutoPad::NOTSET) {
        const int ap = static_cast<int>(attrs.auto_pad);
        for (int i = 0; i < 3; ++i) {
            attrs.padding[static_cast<size_t>(i)] = resolve_leading_padding(
                inputs[0].shape(2 + i), attrs.kernel_size[static_cast<size_t>(i)],
                attrs.stride[static_cast<size_t>(i)], attrs.dilation[static_cast<size_t>(i)],
                attrs.padding[static_cast<size_t>(i)], ap);
        }
        attrs.auto_pad = Conv3DAttributes::AutoPad::NOTSET;
    }

    impl_->kernel_fn(attrs, output, inputs, ctx, workspace);
}

}  // namespace nnops
