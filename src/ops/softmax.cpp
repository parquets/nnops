/// @file softmax.cpp
/// @brief Softmax operator dispatch.

#include "nnops/ops/softmax.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(SoftmaxAttributes, softmax_cpu)
NNOPS_DECLARE_CUDA(SoftmaxAttributes, softmax_cuda)

NNOPS_OP_BOILERPLATE(Softmax, SoftmaxAttributes, backend::cpu::softmax_cpu,
                     NNOPS_IF_CUDA(backend::cuda::softmax_cuda), nullptr)

std::vector<TensorDesc> Softmax::getOutputTensorDesc(std::span<const TensorDesc> inputs) const
{
    return identity_output_shape(inputs);
}

Status Softmax::validate(std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1));
    NNOPS_TRY(detail::check_rank(inputs[0].rank, 1));
    if (!(attrs_.temperature > 0.0f)) {
        return Status::invalid_argument;
    }

    // The axis walk reads the tensor through stride_elems(), which describes the
    // regular image layouts; the opaque weight tags have no axis semantics here.
    if (!is_channel_last(inputs[0].layout)) {
        return Status::unsupported;
    }
    // A quantized input would need a dequantizing walk that no longer exists.
    if (is_quantized_dtype(inputs[0].dtype)) {
        return Status::unsupported;
    }
    return Status::ok;
}

Status Softmax::compute(std::span<TensorView> outputs, std::span<const TensorView> inputs,
                        const ComputeContext& ctx, void* workspace)
{
    NNOPS_TRY(detail::check_views(backend_, outputs, inputs, 1, 1, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) {
        return Status::unsupported;
    }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

} // namespace nnops
