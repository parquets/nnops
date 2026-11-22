/// @file eltwise.cpp
/// @brief Eltwise operator dispatch.

#include "nnops/ops/eltwise.hpp"
#include "nnops/detail/aliasing.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(EltwiseAttributes, eltwise_cpu)
NNOPS_DECLARE_CUDA(EltwiseAttributes, eltwise_cuda)
NNOPS_DECLARE_VULKAN(EltwiseAttributes, eltwise_vulkan)
NNOPS_DECLARE_METAL(EltwiseAttributes, eltwise_metal)

NNOPS_OP_BOILERPLATE_METAL(Eltwise, EltwiseAttributes, backend::cpu::eltwise_cpu,
                           NNOPS_IF_CUDA(backend::cuda::eltwise_cuda),
                           NNOPS_IF_VULKAN(backend::vulkan::eltwise_vulkan),
                           NNOPS_IF_METAL(backend::metal::eltwise_metal))

std::vector<TensorDesc> Eltwise::getOutputTensorDesc(std::span<const TensorDesc> inputs) const
{
    return broadcast_binary_output_shape(inputs);
}

Status Eltwise::validate(std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 2, 2, 1));
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    if (a.dtype != b.dtype || outputs[0].dtype != a.dtype) {
        return Status::invalid_argument;
    }
    // Float, or the int8 pair the fused quantized path carries (it needs the
    // per-tensor scale/zero_point, which only compute() can see).
    if (!detail::dtype_in<DataType::f32, DataType::f16>(backend_, a.dtype) &&
        a.dtype != DataType::s8 && a.dtype != DataType::u8) {
        return Status::unsupported;
    }
    // A packed or prepacked tag describes a filter, never an activation.
    const auto is_activation = [](TensorLayout l) {
        return l == TensorLayout::Dense || is_channel_last(l);
    };
    if (!is_activation(a.layout) || !is_activation(b.layout)) {
        return Status::invalid_argument;
    }
    return detail::require(
        broadcast_compatible(std::span<const int64_t>(a.dims.data(), static_cast<size_t>(a.rank)),
                             std::span<const int64_t>(b.dims.data(), static_cast<size_t>(b.rank))));
}

Status Eltwise::compute(std::span<TensorView> outputs, std::span<const TensorView> inputs,
                        const ComputeContext& ctx, void* workspace)
{
    NNOPS_TRY(detail::check_views(backend_, outputs, inputs, 2, 2, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) {
        return Status::unsupported;
    }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

NNOPS_INPLACE_ELEMENTWISE(Eltwise)

} // namespace nnops
