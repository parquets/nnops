/// @file norm.cpp
/// @brief Normalization operator dispatch (BatchNorm/LayerNorm/RMSNorm/GroupNorm/L2Norm).

#include "nnops/ops/norm.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(NormAttributes, norm_cpu)

NNOPS_OP_BOILERPLATE(Norm, NormAttributes,
        backend::cpu::norm_cpu, nullptr, nullptr)

namespace {

/// Expected input count range for each norm type, as {min_inputs, max_inputs}.
std::pair<int, int> norm_input_range(NormType type) {
    switch (type) {
    case NormType::BatchNorm: return {5, 5};
    case NormType::LayerNorm: return {2, 3};
    case NormType::RMSNorm:   return {2, 2};
    case NormType::GroupNorm: return {2, 3};
    case NormType::L2Norm:    return {1, 1};
    }
    return {0, 0};
}
}  // anonymous namespace

std::vector<TensorDesc> Norm::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    auto out = identity_output_shape(inputs);
    // Quantized X (s8/u8): the output storage type is chosen by the caller —
    // f32/f16 to read back dequantized values, s8/u8 to requantize.
    if (is_quantized_dtype(inputs[0].dtype)) {
        out[0].dtype = attrs_.output_dtype;
    }
    return out;
}

Status Norm::validate(std::span<const TensorDesc> inputs,
                      std::span<const TensorDesc> outputs) const
{
    const auto [min_n, max_n] = norm_input_range(attrs_.type);
    return detail::check_counts(inputs.size(), outputs.size(),
                                static_cast<size_t>(min_n),
                                static_cast<size_t>(max_n), 1);
}

Status Norm::compute(std::span<TensorView> outputs,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace)
{
    NNOPS_TRY(detail::check_views(outputs, inputs, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
