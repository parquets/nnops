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

void Unary::compute(std::span<TensorView> outputs,
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
