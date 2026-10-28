/// @file flatten.cpp
/// @brief Flatten operator dispatch.

#include "nnops/ops/flatten.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(FlattenAttributes, flatten_cpu)

NNOPS_OP_BOILERPLATE(Flatten, FlattenAttributes,
        backend::cpu::flatten_cpu, nullptr, nullptr)

std::vector<TensorDesc> Flatten::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    // Same logical shape, always dense planar layout.
    const auto& in = inputs[0];
    const int64_t rank = in.rank;
    const int64_t srank = rank - 2;

    TensorDesc out;
    out.rank   = rank;
    out.layout = (srank == 3) ? TensorLayout::NCDHW : TensorLayout::NCHW;
    out.dtype  = in.dtype;
    out.dims   = in.dims;

    return {out};
}

Status Flatten::validate(std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs) const
{
    return detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1);
}

Status Flatten::compute(std::span<TensorView> outputs,
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
