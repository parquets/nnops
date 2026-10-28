/// @file embed.cpp
/// @brief Embed operator dispatch.

#include "nnops/ops/embed.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(EmbedAttributes, embed_cpu)

NNOPS_OP_BOILERPLATE(Embed, EmbedAttributes,
        backend::cpu::embed_cpu, nullptr, nullptr)

std::vector<TensorDesc> Embed::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return embed_output_shape(inputs);
}

size_t Embed::getWorkspaceSize(std::span<const TensorDesc> /*inputs*/,
                               std::span<const TensorDesc> /*outputs*/) const
{
    // No workspace required: int8/uint8 weight is dequantized directly into the
    // f32 or f16 output (f16 via the fused half-output kernel — no staging).
    return 0;
}

Status Embed::validate(std::span<const TensorDesc> inputs,
                       std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 2, 2, 1));

    const auto& weight = inputs[0];
    const auto& indices = inputs[1];
    const auto& out = outputs[0];

    if (!detail::dtype_in<DataType::s64, DataType::s32>(indices.dtype)) {
        return Status::invalid_argument;
    }
    if (weight.rank < 2) { return Status::invalid_argument; }
    if (!is_layout_supported(weight.layout, LayoutSupport::PlanarOnly) ||
        !is_layout_supported(indices.layout, LayoutSupport::PlanarOnly)) {
        return Status::invalid_argument;
    }

    if (detail::dtype_in<DataType::f32, DataType::f16>(weight.dtype)) {
        return detail::require(out.dtype == weight.dtype);
    }
    if (detail::dtype_in<DataType::s8, DataType::u8>(weight.dtype)) {
        // The output of a quantized weight is always float; the dequant params
        // themselves live on the view and are checked in compute().
        return detail::require(
            detail::dtype_in<DataType::f32, DataType::f16>(out.dtype));
    }
    return Status::unsupported;
}

Status Embed::compute(std::span<TensorView> outputs,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace)
{
    NNOPS_TRY(detail::check_views(outputs, inputs, 2, 2, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    // An int8 weight must carry its dequantization parameters.
    if (detail::dtype_in<DataType::s8, DataType::u8>(inputs[0].data_type()) &&
        !inputs[0].is_quantized()) {
        return Status::invalid_argument;
    }
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
