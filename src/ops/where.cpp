/// @file where.cpp
/// @brief Where operator dispatch.

#include "nnops/ops/where.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(WhereAttributes, where_cpu)

NNOPS_OP_BOILERPLATE(Where, WhereAttributes,
        backend::cpu::where_cpu, nullptr, nullptr)

std::vector<TensorDesc> Where::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    const auto& cond = inputs[0];
    const auto& x = inputs[1];
    const auto& y = inputs[2];

    const auto bc = compute_broadcast_shape(
        compute_broadcast_shape(
            std::span<const int64_t>(cond.dims.data(), static_cast<size_t>(cond.rank)),
            std::span<const int64_t>(x.dims.data(), static_cast<size_t>(x.rank))),
        std::span<const int64_t>(y.dims.data(), static_cast<size_t>(y.rank)));

    TensorDesc out;
    out.rank   = static_cast<int64_t>(bc.size());
    out.dtype  = x.dtype;
    out.layout = (out.rank == x.rank) ? x.layout : TensorLayout::NCHW;
    out.dims   = bc;
    return {out};
}

Status Where::validate(std::span<const TensorDesc> inputs,
                       std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 3, 3, 1));
    const auto& cond = inputs[0];
    const auto& x = inputs[1];
    const auto& y = inputs[2];

    if (!detail::dtype_in<DataType::u8, DataType::s8>(backend_, cond.dtype)) {
        return Status::unsupported;
    }
    if (x.dtype != y.dtype || outputs[0].dtype != x.dtype) {
        return Status::invalid_argument;
    }
    if (!detail::dtype_in<DataType::f32, DataType::f16, DataType::s8,
                          DataType::u8, DataType::s32, DataType::s64>(backend_, x.dtype)) {
        return Status::unsupported;
    }
    if (!is_layout_supported(cond.layout, LayoutSupport::ChannelFirstOnly) ||
        !is_layout_supported(x.layout, LayoutSupport::ChannelFirstOnly) ||
        !is_layout_supported(y.layout, LayoutSupport::ChannelFirstOnly)) {
        return Status::unsupported;
    }

    return detail::require(broadcast_compatible(
        std::span<const int64_t>(cond.dims.data(), static_cast<size_t>(cond.rank)),
        std::span<const int64_t>(x.dims.data(), static_cast<size_t>(x.rank))) &&
        broadcast_compatible(
        std::span<const int64_t>(x.dims.data(), static_cast<size_t>(x.rank)),
        std::span<const int64_t>(y.dims.data(), static_cast<size_t>(y.rank))));
}

Status Where::compute(std::span<TensorView> outputs,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace)
{
    NNOPS_TRY(detail::check_views(backend_, outputs, inputs, 3, 3, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
