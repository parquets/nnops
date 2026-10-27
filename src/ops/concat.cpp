/// @file concat.cpp
/// @brief Concat operator dispatch.

#include "nnops/ops/concat.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(ConcatAttributes, concat_cpu)

NNOPS_OP_BOILERPLATE(Concat, ConcatAttributes,
        backend::cpu::concat_cpu, nullptr, nullptr)

std::vector<TensorDesc> Concat::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return concat_output_shape(attrs_.axis, inputs);
}

void Concat::compute(std::span<TensorView> outputs,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace)
{
    NNOPS_ASSERT(inputs.size() >= 2);
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
