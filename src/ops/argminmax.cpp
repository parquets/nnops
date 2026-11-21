/// @file argminmax.cpp
/// @brief ArgMax / ArgMin operator dispatch.

#include "nnops/ops/argminmax.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(ArgMinMaxAttributes, argmax_cpu)
NNOPS_DECLARE_CPU(ArgMinMaxAttributes, argmin_cpu)
NNOPS_DECLARE_CUDA(ArgMinMaxAttributes, argmax_cuda)
NNOPS_DECLARE_CUDA(ArgMinMaxAttributes, argmin_cuda)

NNOPS_OP_BOILERPLATE(ArgMax, ArgMinMaxAttributes, backend::cpu::argmax_cpu,
                     NNOPS_IF_CUDA(backend::cuda::argmax_cuda), nullptr)

NNOPS_OP_BOILERPLATE(ArgMin, ArgMinMaxAttributes, backend::cpu::argmin_cpu,
                     NNOPS_IF_CUDA(backend::cuda::argmin_cuda), nullptr)

std::vector<TensorDesc> ArgMax::getOutputTensorDesc(std::span<const TensorDesc> inputs) const
{
    return argminmax_output_shape(attrs_.axis, attrs_.keepdims, inputs);
}

namespace {

/// Both arg reductions read one float row and write an s64 index per position.
Status check_argminmax(const Backend be, std::span<const TensorDesc> inputs,
                       std::span<const TensorDesc> outputs)
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1));
    if (!detail::dtype_in<DataType::f32, DataType::f16>(be, inputs[0].dtype)) {
        return Status::unsupported;
    }
    if (outputs[0].dtype != DataType::s64) {
        return Status::invalid_argument;
    }
    return Status::ok;
}

} // namespace

Status ArgMax::validate(std::span<const TensorDesc> inputs,
                        std::span<const TensorDesc> outputs) const
{
    return check_argminmax(backend_, inputs, outputs);
}

Status ArgMax::compute(std::span<TensorView> outputs, std::span<const TensorView> inputs,
                       const ComputeContext& ctx, void* workspace)
{
    NNOPS_TRY(detail::check_views(backend_, outputs, inputs, 1, 1, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) {
        return Status::unsupported;
    }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

std::vector<TensorDesc> ArgMin::getOutputTensorDesc(std::span<const TensorDesc> inputs) const
{
    return argminmax_output_shape(attrs_.axis, attrs_.keepdims, inputs);
}

Status ArgMin::validate(std::span<const TensorDesc> inputs,
                        std::span<const TensorDesc> outputs) const
{
    return check_argminmax(backend_, inputs, outputs);
}

Status ArgMin::compute(std::span<TensorView> outputs, std::span<const TensorView> inputs,
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
