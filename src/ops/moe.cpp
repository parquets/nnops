/// @file moe.cpp
/// @brief MoE (Mixture-of-Experts) operator dispatch.

#include "nnops/ops/moe.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(MoEAttributes, moe_cpu)

NNOPS_OP_BOILERPLATE(MoE, MoEAttributes,
        backend::cpu::moe_cpu, nullptr, nullptr)

std::vector<TensorDesc> MoE::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return moe_output_shape(inputs);
}

size_t MoE::getWorkspaceSize(std::span<const TensorDesc> /*inputs*/,
                             std::span<const TensorDesc> /*outputs*/) const
{
    // Staging (gathered top-K tokens, per-expert activations) comes from the
    // backend's internal pool, like MatMul/Attention/Conv2D.
    return 0;
}

Status MoE::validate(std::span<const TensorDesc> inputs,
                     std::span<const TensorDesc> outputs) const
{
    // At least 5 inputs (input, router_probs, fc1_w, fc1_b, fc2_w); fc1_b and
    // the tail may be omitted or passed as an empty view, so only the count
    // bounds and the required head are checked here.
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 5, 9, 1));
    return detail::require(
        is_layout_supported(inputs[0].layout, LayoutSupport::PlanarOnly));
}

Status MoE::compute(std::span<TensorView> outputs,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace)
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 5, 9, 1));
    if (outputs[0].is_empty()) { return Status::invalid_argument; }

    // Required: input, router_probs, fc1_w, fc2_w. fc1_b sits at index 3 and
    // may be empty.
    for (size_t i : {size_t{0}, size_t{1}, size_t{2}, size_t{4}}) {
        if (inputs[i].is_empty()) { return Status::invalid_argument; }
    }
    const DataType dt0 = inputs[0].data_type();
    if (!detail::dtype_in<DataType::f32, DataType::f16>(dt0)) {
        return Status::unsupported;
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!inputs[i].is_empty() && inputs[i].data_type() != dt0) {
            return Status::invalid_argument;
        }
    }
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
