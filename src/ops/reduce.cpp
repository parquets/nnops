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
    return reduce_output_shape(
        std::span<const int64_t>(attrs_.axes.data(), attrs_.axes.size()),
        attrs_.keepdims, attrs_.noop_with_empty_axes, inputs);
}

Status Reduce::validate(std::span<const TensorDesc> inputs,
                        std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1));

    const int64_t rank = inputs[0].rank;
    if (rank > TensorDesc::kMaxRank) { return Status::invalid_argument; }

    bool seen[TensorDesc::kMaxRank] = {};
    for (const int64_t a : attrs_.axes) {
        const int64_t ax = a < 0 ? a + rank : a;
        if (ax < 0 || ax >= rank || seen[ax]) { return Status::invalid_argument; }
        seen[ax] = true;
    }

    const TensorDesc expected = getOutputTensorDesc(inputs)[0];
    const TensorDesc& got = outputs[0];
    if (expected.rank != got.rank || expected.dtype != got.dtype) {
        return Status::invalid_argument;
    }
    for (int64_t i = 0; i < expected.rank; ++i) {
        if (expected.dims[static_cast<size_t>(i)] !=
            got.dims[static_cast<size_t>(i)]) {
            return Status::invalid_argument;
        }
    }
    return Status::ok;
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
