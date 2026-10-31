/// @file matmul.cpp
/// @brief MatMul operator dispatch.
///
/// [..., M, K] x [..., K, N] -> [..., M, N], batch dims broadcast; transpose_a
/// and transpose_b reorder the extents. An integer pair (s8/u8 A x s8 B)
/// resolves its output from output_dtype -- s32 for the raw MatMulInteger
/// accumulate, s8/u8 requantized, f32/f16 dequantized -- not from A's dtype.
///
/// prepackWeights is a caller-driven two-step: an empty output buffer is the
/// query (the call writes back the descriptor that sizes the blob, and rank 0 --
/// not f32/f16, or an empty extent -- is an answer rather than a failure), then
/// the same call fills the caller's buffer. The blob holds the whole K, which is what makes it
/// independent of the per-call plan -- and why alpha, which otherwise rides the
/// RHS pack, must be 1. A packed B carries N but no K: validate() takes K from
/// A, and the backend's numel check catches a blob packed against another K.

#include "nnops/ops/matmul.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

#include "backend/cpu/matmul/matmul.h"
#include "backend/cpu/matmul/matmul_helper.h"

#include <vector>

namespace nnops {

NNOPS_DECLARE_CPU(MatMulAttributes, matmul_cpu)
NNOPS_DECLARE_CUDA(MatMulAttributes, matmul_cuda)

NNOPS_OP_BOILERPLATE(MatMul, MatMulAttributes,
        backend::cpu::matmul_cpu,
        NNOPS_IF_CUDA(backend::cuda::matmul_cuda), nullptr)

size_t MatMul::getWorkspaceSize(std::span<const TensorDesc> inputs,
                                std::span<const TensorDesc> outputs) const
{
    (void)inputs;
    (void)outputs;
    // Packed-A/packed-B scratch is pooled inside the CPU kernel.
    return 0;
}

std::vector<TensorDesc> MatMul::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    auto out = matmul_output_shape(attrs_.transpose_a, attrs_.transpose_b, inputs);

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
    // A packed B carries no K: its coherence is the backend's check.
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
    // CUDA reads B raw (no panel format of its own): an empty query stays
    // empty, the documented "this weight has no packed form" answer.
    if (backend_ != Backend::CPU) { return; }
    NNOPS_ASSERT(inputs.size() >= 1);
    NNOPS_ASSERT(outputs.size() >= 1);
    const auto& raw = inputs[0];
    NNOPS_ASSERT(raw.rank() >= 2);

    const TensorDesc d = backend::cpu::matmul_packed_b_desc(raw.desc(),
                                                            attrs_.transpose_b);
    std::vector<int64_t> shape(d.dims.begin(), d.dims.begin() + d.rank);
    const int64_t esz = static_cast<int64_t>(data_type_size(d.dtype));

    if (outputs[0].is_empty()) {
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
