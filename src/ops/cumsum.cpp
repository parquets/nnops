/// @file cumsum.cpp
/// @brief CumSum operator dispatch.

#include "nnops/ops/cumsum.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(CumSumAttributes, cumsum_cpu)
NNOPS_DECLARE_CUDA(CumSumAttributes, cumsum_cuda)

NNOPS_OP_BOILERPLATE(CumSum, CumSumAttributes,
        backend::cpu::cumsum_cpu, NNOPS_IF_CUDA(backend::cuda::cumsum_cuda), nullptr)

std::vector<TensorDesc> CumSum::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return {identity_output_shape(inputs)};
}

Status CumSum::validate(std::span<const TensorDesc> inputs,
                        std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1));
    return detail::require(
        is_layout_supported(inputs[0].layout, LayoutSupport::PlanarOnly));
}

Status CumSum::compute(std::span<TensorView> outputs,
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
