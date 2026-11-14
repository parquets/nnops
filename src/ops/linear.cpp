/// @file linear.cpp
/// @brief Linear operator dispatch — descriptor checks, the quantized output
///        dtype rule, and the packed-weight prepack.
///
/// Linear is MatMul with a transposed RHS: the weight is [N, K] and the product is
/// X × Wᵀ, so prepackWeights() packs MatMul's transposed RHS over the whole K at
/// load time. For an integer GEMM (s8/u8 A against s8 B) the output dtype follows
/// output_dtype when it names a known type — f32/f16 dequantize, s32/s8/u8 stay
/// integral — defaulting to f32; the fp path keeps the input dtype the shape
/// helper already set.

#include "nnops/ops/linear.hpp"
#include "nnops/ops/matmul.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "backend/cpu/matmul/matmul_helper.h"

#include <vector>

namespace nnops {

NNOPS_DECLARE_CPU(LinearAttributes, linear_cpu)
NNOPS_DECLARE_CUDA(LinearAttributes, linear_cuda)

NNOPS_OP_BOILERPLATE(Linear, LinearAttributes,
        backend::cpu::linear_cpu,
        NNOPS_IF_CUDA(backend::cuda::linear_cuda), nullptr)

std::vector<TensorDesc> Linear::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    auto out = linear_output_shape(inputs);

    // The gated form's weight is [2N, K]: the output is the combined [..., N].
    if (attrs_.gated && !out.empty() && out[0].rank >= 1 && inputs.size() >= 2 &&
        inputs[1].rank == 2) {
        out[0].dims[static_cast<size_t>(out[0].rank - 1)] = inputs[1].dims[0] / 2;
    }

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
    NNOPS_TRY(detail::require(
        is_layout_supported(inputs[0].layout, LayoutSupport::ChannelFirstOnly)));

    // The CPU routes implement the pre-activation residual — int8 included —
    // but the gated combine and the CUDA kernel apply the residual after their
    // own activation. Refusing beats applying it on the wrong side.
    if (attrs_.add_to && attrs_.residual_pre_activation) {
        if (attrs_.gated || backend_ != Backend::CPU) {
            return Status::unsupported;
        }
    }

    if (!attrs_.gated) { return Status::ok; }

    // The gated combine implements Silu (SwiGLU), Gelu (GeGLU) and the plain
    // activations; anything else has no kernel.
    switch (attrs_.gate_activation) {
    case EpilogueActivateType::Silu:
    case EpilogueActivateType::Gelu:
    case EpilogueActivateType::Relu:
    case EpilogueActivateType::Sigmoid:
    case EpilogueActivateType::Tanh:
    case EpilogueActivateType::None:
        break;
    default:
        return Status::unsupported;
    }

    const TensorDesc& x = inputs[0];
    const TensorDesc& w = inputs[1];
    if (!detail::dtype_in<DataType::f32, DataType::f16>(backend_, x.dtype) || w.dtype != x.dtype) {
        return Status::unsupported;
    }
    if (w.rank != 2 || w.dims[0] <= 0 || (w.dims[0] % 2) != 0) {
        return Status::invalid_argument;
    }
    const int64_t n = w.dims[0] / 2;
    if (inputs.size() > 2) {
        const TensorDesc& b = inputs[2];
        if (b.dtype != x.dtype || b.numel() != w.dims[0]) {
            return Status::invalid_argument;
        }
    }
    if (outputs.size() == 1 && outputs[0].rank >= 1 &&
        outputs[0].dims[static_cast<size_t>(outputs[0].rank - 1)] != n) {
        return Status::invalid_argument;
    }
    return Status::ok;
}

Status Linear::compute(std::span<TensorView> outputs,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace)
{
    NNOPS_TRY(detail::check_views(backend_, outputs, inputs, 2, 3, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

void Linear::prepackWeights(std::span<const TensorView> inputs,
                            std::span<TensorView> outputs,
                            const ComputeContext& ctx)
{
    // CUDA reads the raw [N, K] weight in place; the query answers empty.
    if (backend_ != Backend::CPU) { return; }
    NNOPS_ASSERT(inputs.size() >= 1);
    NNOPS_ASSERT(outputs.size() >= 1);
    const auto& weight = inputs[0];
    NNOPS_ASSERT(weight.rank() == 2);

    const TensorDesc d = backend::cpu::matmul_packed_b_desc(weight.desc(),
                                                            /*transpose_b=*/true);
    std::vector<int64_t> shape(d.dims.begin(), d.dims.begin() + d.rank);
    const int64_t esz = static_cast<int64_t>(data_type_size(d.dtype));

    if (outputs[0].is_empty()) {
        // Query: [N, ⌈elems / N⌉] pitched at the row length — not channel-packed,
        // so the default alignment would not describe it. A weight with no packed
        // form answers empty rather than failing, since a graph engine asks
        // unconditionally.
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
