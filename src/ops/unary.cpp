/// @file unary.cpp
/// @brief Unary operator dispatch.

#include "nnops/ops/unary.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(UnaryAttributes, unary_cpu)
NNOPS_DECLARE_CUDA(UnaryAttributes, unary_cuda)
NNOPS_DECLARE_VULKAN(UnaryAttributes, unary_vulkan)

NNOPS_OP_BOILERPLATE(Unary, UnaryAttributes,
        backend::cpu::unary_cpu, NNOPS_IF_CUDA(backend::cuda::unary_cuda), NNOPS_IF_VULKAN(backend::vulkan::unary_vulkan))

std::vector<TensorDesc> Unary::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return {identity_output_shape(inputs)};
}

Status Unary::validate(std::span<const TensorDesc> inputs,
                       std::span<const TensorDesc> outputs) const
{
    return detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1);
}

Status Unary::compute(std::span<TensorView> outputs,
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
