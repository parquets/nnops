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

void Activation::compute(std::span<TensorView> outputs,
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
