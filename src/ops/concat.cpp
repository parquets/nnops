/// @file concat.cpp
/// @brief Concat operator dispatch.

#include "nnops/ops/concat.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(ConcatAttributes, concat_cpu)
NNOPS_DECLARE_CUDA(ConcatAttributes, concat_cuda)

NNOPS_OP_BOILERPLATE(Concat, ConcatAttributes, backend::cpu::concat_cpu,
                     NNOPS_IF_CUDA(backend::cuda::concat_cuda), nullptr)

std::vector<TensorDesc> Concat::getOutputTensorDesc(std::span<const TensorDesc> inputs) const
{
    return concat_output_shape(attrs_.axis, inputs);
}

Status Concat::validate(std::span<const TensorDesc> inputs,
                        std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 2, size_t(-1), 1));
    // A line copy: any whole-element dtype, all inputs in the same one.
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!detail::is_movement_dtype(inputs[i].dtype)) {
            return Status::unsupported;
        }
        if (inputs[i].dtype != inputs[0].dtype) {
            return Status::invalid_argument;
        }
    }
    if (outputs[0].dtype != inputs[0].dtype) {
        return Status::invalid_argument;
    }
    return Status::ok;
}

Status Concat::compute(std::span<TensorView> outputs, std::span<const TensorView> inputs,
                       const ComputeContext& ctx, void* workspace)
{
    NNOPS_TRY(detail::check_views(backend_, outputs, inputs, 2, size_t(-1), 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) {
        return Status::unsupported;
    }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

} // namespace nnops
