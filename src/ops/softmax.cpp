/// @file softmax.cpp
/// @brief Softmax operator dispatch.

#include "nnops/ops/softmax.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(SoftmaxAttributes, softmax_cpu)
NNOPS_DECLARE_CUDA(SoftmaxAttributes, softmax_cuda)

NNOPS_OP_BOILERPLATE(Softmax, SoftmaxAttributes,
        backend::cpu::softmax_cpu, NNOPS_IF_CUDA(backend::cuda::softmax_cuda), nullptr)

std::vector<TensorDesc> Softmax::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    auto out = identity_output_shape(inputs);
    // Quantized input (s8/u8): softmax output is dequantized float (default f32),
    // not re-quantized int.
    if (is_quantized_dtype(inputs[0].dtype)) {
        out[0].dtype = DataType::f32;
    }
    return out;
}

size_t Softmax::getWorkspaceSize(std::span<const TensorDesc> inputs,
                                 std::span<const TensorDesc> outputs) const
{
    (void)inputs;
    (void)outputs;
    // Quantized dequantization scratch is pooled internally by the CPU kernel.
    return 0;
}

Status Softmax::validate(std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1));
    NNOPS_TRY(detail::check_rank(inputs[0].rank, 1));
    if (!(attrs_.temperature > 0.0f)) { return Status::invalid_argument; }

    if (is_quantized_dtype(inputs[0].dtype)) {
        // A quantized input is dequantized first, so the output must be float.
        if (!detail::dtype_in<DataType::f32, DataType::f16>(outputs[0].dtype)) {
            return Status::invalid_argument;
        }
        // And the dequantizing walk is only defined over planar layouts.
        return detail::require(
            is_layout_supported(inputs[0].layout, LayoutSupport::PlanarOnly),
            Status::unsupported);
    }
    return Status::ok;
}

Status Softmax::compute(std::span<TensorView> outputs,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx,
                        void* workspace)
{
    NNOPS_TRY(detail::check_views(outputs, inputs, 1, 1, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
