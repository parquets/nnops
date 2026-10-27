/// @file argminmax.cpp
/// @brief ArgMax / ArgMin operator dispatch.

#include "nnops/ops/argminmax.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(ArgMinMaxAttributes, argmax_cpu)
NNOPS_DECLARE_CPU(ArgMinMaxAttributes, argmin_cpu)

NNOPS_OP_BOILERPLATE(ArgMax, ArgMinMaxAttributes,
        backend::cpu::argmax_cpu, nullptr, nullptr)

NNOPS_OP_BOILERPLATE(ArgMin, ArgMinMaxAttributes,
        backend::cpu::argmin_cpu, nullptr, nullptr)

std::vector<TensorDesc> ArgMax::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return argminmax_output_shape(attrs_.axis, attrs_.keepdims, inputs);
}

void ArgMax::compute(std::span<TensorView> outputs,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 1);
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

std::vector<TensorDesc> ArgMin::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return argminmax_output_shape(attrs_.axis, attrs_.keepdims, inputs);
}

void ArgMin::compute(std::span<TensorView> outputs,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 1);
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
