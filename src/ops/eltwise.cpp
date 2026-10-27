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

void Eltwise::compute(std::span<TensorView> outputs,
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

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
