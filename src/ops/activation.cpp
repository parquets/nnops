/// @file activation.cpp
/// @brief Activation operator dispatch.

#include "nnops/ops/activation.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(ActivationAttributes, activation_cpu)
NNOPS_DECLARE_CUDA(ActivationAttributes, activation_cuda)

NNOPS_OP_BOILERPLATE(Activation, ActivationAttributes,
        backend::cpu::activation_cpu, NNOPS_IF_CUDA(backend::cuda::activation_cuda), nullptr)

std::vector<TensorDesc> Activation::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return {identity_output_shape(inputs)};
}

Status Activation::validate(std::span<const TensorDesc> inputs,
                            std::span<const TensorDesc> outputs) const
{
    return detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1);
}

Status Activation::compute(std::span<TensorView> outputs,
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
