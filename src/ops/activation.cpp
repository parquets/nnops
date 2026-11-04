/// @file activation.cpp
/// @brief Activation operator dispatch.

#include "nnops/ops/activation.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(ActivationAttributes, activation_cpu)
NNOPS_DECLARE_CUDA(ActivationAttributes, activation_cuda)

NNOPS_OP_BOILERPLATE(Activation, ActivationAttributes,
        backend::cpu::activation_cpu, NNOPS_IF_CUDA(backend::cuda::activation_cuda), nullptr)

std::vector<TensorDesc> Activation::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return {identity_output_shape(inputs)};
}

Status Activation::validate(std::span<const TensorDesc> inputs,
                            std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1,
                                   attrs_.type == ActivationType::PRelu ? 2 : 1, 1));
    if (attrs_.type != ActivationType::PRelu) { return Status::ok; }

    // The slope rides as a second input: [C] (channel = dim 1) or [1].
    const auto& x = inputs[0];
    const auto& slope = inputs[1];
    if (x.rank < 2 || slope.rank != 1) { return Status::invalid_argument; }
    const int64_t c = x.dims[1];
    if (slope.dims[0] != c && slope.dims[0] != 1) {
        return Status::invalid_argument;
    }
    if (!detail::dtype_in<DataType::f32, DataType::f16>(x.dtype) ||
        slope.dtype != x.dtype) {
        return Status::unsupported;
    }
    return Status::ok;
}

Status Activation::compute(std::span<TensorView> outputs,
                           std::span<const TensorView> inputs,
                           const ComputeContext& ctx,
                           void* workspace)
{
    const size_t n_in = attrs_.type == ActivationType::PRelu ? 2 : 1;
    NNOPS_TRY(detail::check_views(outputs, inputs, n_in, n_in, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
