/// @file unary.cpp
/// @brief Unary operator dispatch.

#include "nnops/ops/unary.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(UnaryAttributes, unary_cpu)
NNOPS_DECLARE_CUDA(UnaryAttributes, unary_cuda)
NNOPS_DECLARE_VULKAN(UnaryAttributes, unary_vulkan)
NNOPS_DECLARE_METAL(UnaryAttributes, unary_metal)

NNOPS_OP_BOILERPLATE_METAL(Unary, UnaryAttributes, backend::cpu::unary_cpu,
                           NNOPS_IF_CUDA(backend::cuda::unary_cuda),
                           NNOPS_IF_VULKAN(backend::vulkan::unary_vulkan),
                           NNOPS_IF_METAL(backend::metal::unary_metal))

std::vector<TensorDesc> Unary::getOutputTensorDesc(std::span<const TensorDesc> inputs) const
{
    return {identity_output_shape(inputs)};
}

Status Unary::validate(std::span<const TensorDesc> inputs,
                       std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1));
    const DataType d = inputs[0].dtype;
    // Float, or the int8 pair the fused quantized path carries (it needs the
    // per-tensor scale/zero_point, which only compute() can see).
    if (!detail::dtype_in<DataType::f32, DataType::f16>(backend_, d) && d != DataType::s8 &&
        d != DataType::u8) {
        return Status::unsupported;
    }
    if (outputs[0].dtype != d) {
        return Status::invalid_argument;
    }
    return Status::ok;
}

Status Unary::compute(std::span<TensorView> outputs, std::span<const TensorView> inputs,
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
