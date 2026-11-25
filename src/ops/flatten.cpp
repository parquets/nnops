/// @file flatten.cpp
/// @brief Flatten operator dispatch.

#include "nnops/ops/flatten.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(FlattenAttributes, flatten_cpu)
NNOPS_DECLARE_CUDA(FlattenAttributes, flatten_cuda)

NNOPS_OP_BOILERPLATE(Flatten, FlattenAttributes, backend::cpu::flatten_cpu,
                     NNOPS_IF_CUDA(backend::cuda::flatten_cuda), nullptr)

std::vector<TensorDesc> Flatten::getOutputTensorDesc(std::span<const TensorDesc> inputs) const
{
    // Same logical shape, dense row-major (the layout carries no channel axis).
    const auto& in = inputs[0];
    const int64_t rank = in.rank;

    TensorDesc out;
    out.rank = rank;
    out.layout = TensorLayout::Dense;
    out.dtype = in.dtype;
    out.dims = in.dims;

    return {out};
}

Status Flatten::validate(std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1));
    if (!detail::is_movement_dtype(inputs[0].dtype)) {
        return Status::unsupported;
    }
    if (outputs[0].dtype != inputs[0].dtype) {
        return Status::invalid_argument;
    }
    return Status::ok;
}

Status Flatten::compute(std::span<TensorView> outputs, std::span<const TensorView> inputs,
                        const ComputeContext& ctx, void* workspace)
{
    NNOPS_TRY(detail::check_views(backend_, outputs, inputs, 1, 1, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) {
        return Status::unsupported;
    }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

} // namespace nnops
