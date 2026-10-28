/// @file matmul.cpp
/// @brief MatMul operator dispatch.

#include "nnops/ops/matmul.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

#include "backend/cpu/matmul/matmul.h"

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
    const int64_t k_b = attrs_.transpose_b
        ? b.dims[static_cast<size_t>(b.rank - 1)]
        : b.dims[static_cast<size_t>(b.rank - 2)];
    return detail::require(k_a == k_b);
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
