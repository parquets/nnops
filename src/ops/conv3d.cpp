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

Status Conv3D::validate(std::span<const TensorDesc> inputs,
                        std::span<const TensorDesc> outputs) const
{
    // input, weight [, bias]
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 2, 3, 1));
    return detail::require(
        is_layout_supported(inputs[0].layout, LayoutSupport::PlanarOnly));
}

Status Conv3D::compute(std::span<TensorView> outputs,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace)
{
    NNOPS_TRY(detail::check_views(outputs, inputs, 2, 3, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    auto& output = outputs[0];

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

    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs, output, inputs, ctx, workspace);
}

}  // namespace nnops
