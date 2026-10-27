/// @file grid_sample.cpp
/// @brief GridSample operator dispatch.

#include "nnops/ops/grid_sample.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(GridSampleAttributes, grid_sample_cpu)

NNOPS_OP_BOILERPLATE(GridSample, GridSampleAttributes,
        backend::cpu::grid_sample_cpu, nullptr, nullptr)

std::vector<TensorDesc> GridSample::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return grid_sample_output_shape(inputs);
}

void GridSample::compute(std::span<TensorView> outputs,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 2);
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());
    NNOPS_ASSERT(!inputs[1].is_empty());
    NNOPS_ASSERT(is_layout_supported(inputs[0].layout(), LayoutSupport::PackedOnly));

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
