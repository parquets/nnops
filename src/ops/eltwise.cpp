/// @file eltwise.cpp
/// @brief Eltwise operator dispatch.

#include "nnops/ops/eltwise.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(EltwiseAttributes, eltwise_cpu)
NNOPS_DECLARE_CUDA(EltwiseAttributes, eltwise_cuda)
NNOPS_DECLARE_VULKAN(EltwiseAttributes, eltwise_vulkan)

NNOPS_OP_BOILERPLATE(Eltwise, EltwiseAttributes,
        backend::cpu::eltwise_cpu, NNOPS_IF_CUDA(backend::cuda::eltwise_cuda), NNOPS_IF_VULKAN(backend::vulkan::eltwise_vulkan))

std::vector<TensorDesc> Eltwise::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return {identity_output_shape(inputs)};
}

Status Eltwise::validate(std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs) const
{
    return detail::check_counts(inputs.size(), outputs.size(), 2, 2, 1);
}

Status Eltwise::compute(std::span<TensorView> outputs,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx,
                        void* workspace)
{
    NNOPS_TRY(detail::check_views(outputs, inputs, 2, 2, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
