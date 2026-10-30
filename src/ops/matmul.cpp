/// @file matmul.cpp
/// @brief MatMul operator dispatch.

#include "nnops/ops/matmul.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

#include "backend/cpu/matmul/matmul.h"
#include "backend/cpu/matmul/matmul_helper.h"

#include <vector>

namespace nnops {

NNOPS_DECLARE_CPU(MatMulAttributes, matmul_cpu)

NNOPS_OP_BOILERPLATE(MatMul, MatMulAttributes,
        backend::cpu::matmul_cpu, nullptr, nullptr)

size_t MatMul::getWorkspaceSize(std::span<const TensorDesc> inputs,
                                std::span<const TensorDesc> outputs) const
{
    (void)inputs;
    (void)outputs;
    // Packed-B (and int8 accumulator) scratch is pooled internally by the CPU kernel.
    return 0;
}

std::vector<TensorDesc> MatMul::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    auto out = matmul_output_shape(attrs_.transpose_a, attrs_.transpose_b, inputs);

    // Integer (s8|u8 × s8) matmul: the output follows output_dtype (s32
    // accumulate or requantized s8), not the input dtype.
    if (is_int8_matmul(inputs[0].dtype, inputs[1].dtype)) {
        out[0].dtype = attrs_.output_dtype;
    }

    return out;
}

Status MatMul::validate(std::span<const TensorDesc> inputs,
                        std::span<const TensorDesc> outputs) const
{
    // A, B, [bias]
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 2, 3, 1));
    if (!is_layout_supported(inputs[0].layout, LayoutSupport::PlanarOnly)) {
        return Status::invalid_argument;
    }

    const auto& a = inputs[0];
    const auto& b = inputs[1];
    NNOPS_TRY(detail::check_rank(a.rank, 2));
    NNOPS_TRY(detail::check_rank(b.rank, 2));

    const int64_t k_a = attrs_.transpose_a
        ? a.dims[static_cast<size_t>(a.rank - 2)]
        : a.dims[static_cast<size_t>(a.rank - 1)];
    // A prepacked B carries no K (the blob is addressed from it, not shaped by
    // it): the product's K is A's, and the blob's own coherence — was it packed
    // against this K — is the backend's check.
    if (is_packed_matmul_b(b.layout)) { return Status::ok; }

    const int64_t k_b = attrs_.transpose_b
        ? b.dims[static_cast<size_t>(b.rank - 1)]
        : b.dims[static_cast<size_t>(b.rank - 2)];
    return detail::require(k_a == k_b);
}

void MatMul::prepackWeights(std::span<const TensorView> inputs,
                            std::span<TensorView> outputs,
                            const ComputeContext& ctx)
{
    NNOPS_ASSERT(inputs.size() >= 1);
    NNOPS_ASSERT(outputs.size() >= 1);
    const auto& raw = inputs[0];
    NNOPS_ASSERT(raw.rank() >= 2);

    const TensorDesc d = backend::cpu::matmul_packed_b_desc(raw.desc(),
                                                            attrs_.transpose_b);
    std::vector<int64_t> shape(d.dims.begin(), d.dims.begin() + d.rank);
    const int64_t esz = static_cast<int64_t>(data_type_size(d.dtype));

    if (outputs[0].is_empty()) {
        // No packed form (neither f32 nor f16) is an empty answer, not a
        // failure: a graph engine asks unconditionally.
        if (d.rank <= 0) { return; }
        outputs[0] = TensorView(shape, d.dtype, nullptr, d.row_stride_elems * esz,
                                d.layout, raw.quant_params());
        return;
    }

    NNOPS_ASSERT(d.rank > 0 &&
                 "MatMul::prepackWeights: this B has no packed form (f32/f16 only)");
    NNOPS_ASSERT(attrs_.alpha == 1.0f &&
                 "MatMul::prepackWeights: a prepacked B cannot carry alpha "
                 "(it rode the per-call RHS pack)");
    NNOPS_ASSERT(outputs[0].layout() == d.layout &&
                 outputs[0].rank() == d.rank &&
                 outputs[0].shape(0) == shape[0] &&
                 outputs[0].shape(1) == shape[1] &&
                 "MatMul::prepackWeights: the packed buffer does not match the geometry");
    backend::cpu::matmul_packed_b_pack(raw, outputs[0], attrs_.transpose_b, ctx);
}

Status MatMul::compute(std::span<TensorView> outputs,
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
