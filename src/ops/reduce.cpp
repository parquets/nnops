/// @file reduce.cpp
/// @brief Reduce operator dispatch.

#include "nnops/ops/reduce.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(ReduceAttributes, reduce_cpu)
NNOPS_DECLARE_CUDA(ReduceAttributes, reduce_cuda)

NNOPS_OP_BOILERPLATE(Reduce, ReduceAttributes,
        backend::cpu::reduce_cpu, NNOPS_IF_CUDA(backend::cuda::reduce_cuda), nullptr)

std::vector<TensorDesc> Reduce::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return {reduce_output_shape(attrs_.axis, attrs_.keepdims, inputs)};
}

void Reduce::compute(std::span<TensorView> outputs,
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
