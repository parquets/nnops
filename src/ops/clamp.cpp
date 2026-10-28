/// @file clamp.cpp
/// @brief Clamp operator dispatch.

#include "nnops/ops/clamp.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(ClampAttributes, clamp_cpu)

NNOPS_OP_BOILERPLATE(Clamp, ClampAttributes,
        backend::cpu::clamp_cpu, nullptr, nullptr)

std::vector<TensorDesc> Clamp::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return {identity_output_shape(inputs)};
}

Status Clamp::validate(std::span<const TensorDesc> inputs,
                       std::span<const TensorDesc> outputs) const
{
    return detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1);
}

Status Clamp::compute(std::span<TensorView> outputs,
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
