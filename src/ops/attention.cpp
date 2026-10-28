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

Status Attention::validate(std::span<const TensorDesc> inputs,
                           std::span<const TensorDesc> outputs) const
{
    // Q, K, V [, mask]
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 3, 4, 1));
    return detail::require(
        is_layout_supported(inputs[0].layout, LayoutSupport::PlanarOnly));
}

Status Attention::compute(std::span<TensorView> outputs,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx,
                          void* workspace)
{
    // The optional mask is allowed to arrive as an empty view (meaning absent),
    // so only the three required inputs are checked for data.
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 3, 4, 1));
    for (size_t i = 0; i < 3; ++i) {
        if (inputs[i].is_empty()) { return Status::invalid_argument; }
    }
    if (outputs[0].is_empty()) { return Status::invalid_argument; }
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
