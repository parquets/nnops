/// @file eltwise.cpp
/// @brief Eltwise operator dispatch.

#include "nnops/ops/eltwise.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(EltwiseAttributes, eltwise_cpu)
NNOPS_DECLARE_CUDA(EltwiseAttributes, eltwise_cuda)
NNOPS_DECLARE_VULKAN(EltwiseAttributes, eltwise_vulkan)

NNOPS_OP_BOILERPLATE(Eltwise, EltwiseAttributes,
        backend::cpu::eltwise_cpu, NNOPS_IF_CUDA(backend::cuda::eltwise_cuda), NNOPS_IF_VULKAN(backend::vulkan::eltwise_vulkan))

std::vector<TensorDesc> Eltwise::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
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
    // A packed or prepacked tag describes a filter, never an activation.
    if (!(is_planar_spatial(a.layout) || is_channel_last(a.layout)) ||
        !(is_planar_spatial(b.layout) || is_channel_last(b.layout))) {
        return Status::invalid_argument;
    }
    return detail::require(broadcast_compatible(
        std::span<const int64_t>(a.dims.data(), static_cast<size_t>(a.rank)),
        std::span<const int64_t>(b.dims.data(), static_cast<size_t>(b.rank))));
}

Status Eltwise::compute(std::span<TensorView> outputs,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx,
                        void* workspace)
{
    NNOPS_TRY(detail::check_views(outputs, inputs, 2, 2, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
