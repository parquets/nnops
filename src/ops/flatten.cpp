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

void Flatten::compute(std::span<TensorView> outputs,
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
