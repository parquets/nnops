/// @file compare.cpp
/// @brief Compare and Logical dispatch.

#include "nnops/ops/compare.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(CompareAttributes, compare_cpu)
NNOPS_DECLARE_CPU(LogicalAttributes, logical_cpu)

NNOPS_OP_BOILERPLATE(Compare, CompareAttributes,
        backend::cpu::compare_cpu, nullptr, nullptr)
NNOPS_OP_BOILERPLATE(Logical, LogicalAttributes,
        backend::cpu::logical_cpu, nullptr, nullptr)

namespace {

template <size_t N>
bool all_channel_first(const TensorDesc* d) {
    for (size_t i = 0; i < N; ++i) {
        if (!is_layout_supported(d[i].layout, LayoutSupport::ChannelFirstOnly)) {
            return false;
        }
    }
    return true;
}

std::vector<TensorDesc> broadcast_bool_shape(std::span<const TensorDesc> inputs)
{
    auto out = broadcast_binary_output_shape(inputs);
    out[0].dtype = DataType::u8;
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Compare
// ---------------------------------------------------------------------------

std::vector<TensorDesc> Compare::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return broadcast_bool_shape(inputs);
}

Status Compare::validate(std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 2, 2, 1));
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    if (a.dtype != b.dtype) { return Status::invalid_argument; }
    if (!detail::dtype_in<DataType::f32, DataType::f16,
                          DataType::s32, DataType::s64>(a.dtype)) {
        return Status::unsupported;
    }
    if (outputs[0].dtype != DataType::u8) { return Status::invalid_argument; }
    if (!all_channel_first<2>(inputs.data())) { return Status::unsupported; }

    return detail::require(broadcast_compatible(
        std::span<const int64_t>(a.dims.data(), static_cast<size_t>(a.rank)),
        std::span<const int64_t>(b.dims.data(), static_cast<size_t>(b.rank))));
}

Status Compare::compute(std::span<TensorView> outputs,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx,
                        void* workspace)
{
    NNOPS_TRY(detail::check_views(outputs, inputs, 2, 2, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

// ---------------------------------------------------------------------------
// Logical
// ---------------------------------------------------------------------------

std::vector<TensorDesc> Logical::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    if (attrs_.type == LogicalType::Not) {
        auto out = identity_output_shape(inputs);
        out[0].dtype = DataType::u8;
        return out;
    }
    return broadcast_bool_shape(inputs);
}

Status Logical::validate(std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs) const
{
    const bool unary = (attrs_.type == LogicalType::Not);
    const size_t n_in = unary ? 1 : 2;

    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), n_in, n_in, 1));

    for (size_t i = 0; i < n_in; ++i) {
        if (!detail::dtype_in<DataType::s8, DataType::u8,
                              DataType::s32, DataType::s64>(inputs[i].dtype)) {
            return Status::unsupported;
        }
        if (i > 0 && inputs[i].dtype != inputs[0].dtype) {
            return Status::invalid_argument;
        }
    }
    if (outputs[0].dtype != DataType::u8) { return Status::invalid_argument; }
    for (size_t i = 0; i < n_in; ++i) {
        if (!is_layout_supported(inputs[i].layout, LayoutSupport::ChannelFirstOnly)) {
            return Status::unsupported;
        }
    }

    if (unary) { return Status::ok; }

    return detail::require(broadcast_compatible(
        std::span<const int64_t>(inputs[0].dims.data(), static_cast<size_t>(inputs[0].rank)),
        std::span<const int64_t>(inputs[1].dims.data(), static_cast<size_t>(inputs[1].rank))));
}

Status Logical::compute(std::span<TensorView> outputs,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx,
                        void* workspace)
{
    const size_t n_in = (attrs_.type == LogicalType::Not) ? 1 : 2;
    NNOPS_TRY(detail::check_views(outputs, inputs, n_in, n_in, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
