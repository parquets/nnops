/// @file linear.cpp
/// @brief Linear operator dispatch.

#include "nnops/ops/linear.hpp"
#include "nnops/ops/matmul.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(LinearAttributes, linear_cpu)

NNOPS_OP_BOILERPLATE(Linear, LinearAttributes,
        backend::cpu::linear_cpu, nullptr, nullptr)

std::vector<TensorDesc> Linear::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    auto out = linear_output_shape(inputs);

    // The quantized path's output follows output_dtype: f32/f16 dequantizes,
    // s32/s8/u8 keeps the result integral; the fp path follows the input dtype
    // (set by the shape helper already).
    if (is_int8_matmul(inputs[0].dtype, inputs[1].dtype)) {
        const DataType odt = attrs_.output_dtype;
        const bool known = odt == DataType::f16 || odt == DataType::f32 ||
                           odt == DataType::s32 || odt == DataType::s8 ||
                           odt == DataType::u8;
        out[0].dtype = known ? odt : DataType::f32;
    }

    return out;
}

Status Linear::validate(std::span<const TensorDesc> inputs,
                        std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 2, 3, 1));
    return detail::require(
        is_layout_supported(inputs[0].layout, LayoutSupport::PlanarOnly));
}

Status Linear::compute(std::span<TensorView> outputs,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace)
{
    NNOPS_TRY(detail::check_views(outputs, inputs, 2, 3, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
