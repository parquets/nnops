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

Status ArgMax::validate(std::span<const TensorDesc> inputs,
                        std::span<const TensorDesc> outputs) const
{
    return detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1);
}

Status ArgMax::compute(std::span<TensorView> outputs,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace)
{
    NNOPS_TRY(detail::check_views(outputs, inputs, 1, 1, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

std::vector<TensorDesc> ArgMin::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return argminmax_output_shape(attrs_.axis, attrs_.keepdims, inputs);
}

Status ArgMin::validate(std::span<const TensorDesc> inputs,
                        std::span<const TensorDesc> outputs) const
{
    return detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1);
}

Status ArgMin::compute(std::span<TensorView> outputs,
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
