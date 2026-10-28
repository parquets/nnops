/// @file resize.cpp
/// @brief Resize operator dispatch.

#include "nnops/ops/resize.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(ResizeAttributes, resize_cpu)

NNOPS_OP_BOILERPLATE(Resize, ResizeAttributes,
        backend::cpu::resize_cpu, nullptr, nullptr)

std::vector<TensorDesc> Resize::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    std::array<int64_t, 3> os = attrs_.output_size;
    return {resize_output_shape(
        std::span<const int64_t>(os.data(), os.size()), inputs)};
}

Status Resize::validate(std::span<const TensorDesc> inputs,
                        std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1));
    return detail::require(
        is_layout_supported(inputs[0].layout, LayoutSupport::PackedOnly));
}

Status Resize::compute(std::span<TensorView> outputs,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace)
{
    NNOPS_TRY(detail::check_views(outputs, inputs, 1, 1, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
