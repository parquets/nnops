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

Status Reduce::validate(std::span<const TensorDesc> inputs,
                        std::span<const TensorDesc> outputs) const
{
    return detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1);
}

Status Reduce::compute(std::span<TensorView> outputs,
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
