/// @file grid_sample.cpp
/// @brief GridSample operator dispatch.

#include "nnops/ops/grid_sample.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(GridSampleAttributes, grid_sample_cpu)

NNOPS_OP_BOILERPLATE(GridSample, GridSampleAttributes, backend::cpu::grid_sample_cpu, nullptr,
                     nullptr)

std::vector<TensorDesc> GridSample::getOutputTensorDesc(std::span<const TensorDesc> inputs) const
{
    return grid_sample_output_shape(inputs);
}

Status GridSample::validate(std::span<const TensorDesc> inputs,
                            std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 2, 2, 1));
    if (!detail::dtype_in<DataType::f32, DataType::f16>(backend_, inputs[0].dtype)) {
        return Status::unsupported;
    }
    if (outputs[0].dtype != inputs[0].dtype) {
        return Status::invalid_argument;
    }
    // The grid is always read as f32 coordinates.
    if (inputs[1].dtype != DataType::f32) {
        return Status::invalid_argument;
    }
    return detail::require(
        is_layout_supported(inputs[0].layout, LayoutSupport::ChannelLastOnly));
}

Status GridSample::compute(std::span<TensorView> outputs, std::span<const TensorView> inputs,
                           const ComputeContext& ctx, void* workspace)
{
    NNOPS_TRY(detail::check_views(backend_, outputs, inputs, 2, 2, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) {
        return Status::unsupported;
    }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

} // namespace nnops
