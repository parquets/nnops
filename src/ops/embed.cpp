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

void Embed::compute(std::span<TensorView> outputs,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace)
{
    NNOPS_ASSERT(outputs.size() == 1);
    NNOPS_ASSERT(inputs.size() == 2);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());

    const auto& weight  = inputs[0];
    const auto& indices = inputs[1];

    NNOPS_ASSERT(!weight.is_empty());
    NNOPS_ASSERT(!indices.is_empty());

    NNOPS_ASSERT(indices.data_type() == DataType::s64 ||
                 indices.data_type() == DataType::s32);

    NNOPS_ASSERT(weight.rank() >= 2);

    NNOPS_ASSERT(is_layout_supported(weight.layout(), LayoutSupport::PlanarOnly));
    NNOPS_ASSERT(is_layout_supported(indices.layout(), LayoutSupport::PlanarOnly));

    const auto w_dtype = weight.data_type();

    if (w_dtype == DataType::f32 || w_dtype == DataType::f16) {
        NNOPS_ASSERT(output.data_type() == w_dtype);
    } else if (w_dtype == DataType::s8 || w_dtype == DataType::u8) {
        // Int8 weight: dequant params come from weight.quant_params()
        NNOPS_ASSERT(weight.is_quantized());
        NNOPS_ASSERT(output.data_type() == DataType::f32 ||
                     output.data_type() == DataType::f16);
    } else {
        NNOPS_ASSERT(!"embed: unsupported weight dtype");
    }

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
