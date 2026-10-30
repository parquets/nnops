/// @file linear.cpp
/// @brief Linear operator dispatch.

#include "nnops/ops/linear.hpp"
#include "nnops/ops/matmul.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "backend/cpu/matmul/matmul_helper.h"

#include <vector>

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

void Linear::prepackWeights(std::span<const TensorView> inputs,
                            std::span<TensorView> outputs,
                            const ComputeContext& ctx)
{
    NNOPS_ASSERT(inputs.size() >= 1);
    NNOPS_ASSERT(outputs.size() >= 1);
    const auto& weight = inputs[0];
    NNOPS_ASSERT(weight.rank() == 2);

    // The weight is [N, K] and the product is X × Wᵀ, so this is MatMul's
    // transposed RHS — the blob is that matrix's panels, packed over the whole
    // K at load time (see MatMulPackedB).
    const TensorDesc d = backend::cpu::matmul_packed_b_desc(weight.desc(),
                                                            /*transpose_b=*/true);
    std::vector<int64_t> shape(d.dims.begin(), d.dims.begin() + d.rank);
    const int64_t esz = static_cast<int64_t>(data_type_size(d.dtype));

    if (outputs[0].is_empty()) {
        // ---- Query: [N, ⌈elems / N⌉], with the row length as the pitch (a
        // packed RHS is not channel-packed, so the default alignment would not
        // describe it). No packed form (a quantized weight) is an empty answer
        // rather than a failure: a graph engine asks unconditionally.
        if (d.rank <= 0) { return; }
        outputs[0] = TensorView(shape, d.dtype, nullptr, d.row_stride_elems * esz,
                                d.layout, weight.quant_params());
        return;
    }

    NNOPS_ASSERT(d.rank > 0 && outputs[0].layout() == d.layout &&
                 outputs[0].rank() == d.rank &&
                 outputs[0].shape(0) == shape[0] &&
                 outputs[0].shape(1) == shape[1] &&
                 "Linear::prepackWeights: the packed buffer does not match the "
                 "geometry");
    backend::cpu::matmul_packed_b_pack(weight, outputs[0], /*transpose_b=*/true, ctx);
}

}  // namespace nnops
