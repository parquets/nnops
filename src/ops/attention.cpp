/// @file attention.cpp
/// @brief Attention operator dispatch.

#include "nnops/ops/attention.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "backend/cpu/attention/attention.h"

namespace nnops {

NNOPS_DECLARE_CPU(AttentionAttributes, attention_cpu)

NNOPS_OP_BOILERPLATE(Attention, AttentionAttributes,
        backend::cpu::attention_cpu, nullptr, nullptr)

std::vector<TensorDesc> Attention::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return {attention_output_shape(inputs)};
}

size_t Attention::getWorkspaceSize(std::span<const TensorDesc> inputs,
                                   std::span<const TensorDesc> outputs) const
{
    (void)inputs;
    (void)outputs;
    // Scratch (scores + packed K^T) is pooled internally by the CPU kernel.
    return 0;
}

void Attention::compute(std::span<TensorView> outputs,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         void* workspace)
{
    NNOPS_ASSERT(inputs.size() >= 3);
    NNOPS_ASSERT(inputs.size() <= 4);
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());
    NNOPS_ASSERT(!inputs[1].is_empty());
    NNOPS_ASSERT(!inputs[2].is_empty());
    NNOPS_ASSERT(is_layout_supported(inputs[0].layout(), LayoutSupport::PlanarOnly));

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
