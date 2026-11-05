/// @file scatter.cpp
/// @brief ScatterElements / ScatterND dispatch.

#include "nnops/ops/scatter.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(ScatterElementsAttributes, scatter_elements_cpu)
NNOPS_DECLARE_CPU(ScatterNDAttributes, scatter_nd_cpu)

NNOPS_OP_BOILERPLATE(ScatterElements, ScatterElementsAttributes,
        backend::cpu::scatter_elements_cpu, nullptr, nullptr)
NNOPS_OP_BOILERPLATE(ScatterND, ScatterNDAttributes,
        backend::cpu::scatter_nd_cpu, nullptr, nullptr)

namespace {

Status check_scatter_dtypes(const TensorDesc& data, const TensorDesc& indices,
                            const TensorDesc& updates)
{
    if (!detail::dtype_in<DataType::f32, DataType::f16, DataType::s8,
                          DataType::u8, DataType::s32, DataType::s64>(data.dtype)) {
        return Status::unsupported;
    }
    if (updates.dtype != data.dtype) { return Status::invalid_argument; }
    return detail::require(
        detail::dtype_in<DataType::s64, DataType::s32>(indices.dtype));
}

Status check_scatter_layouts(const TensorDesc& data, const TensorDesc& indices,
                             const TensorDesc& updates)
{
    return detail::require(
        is_layout_supported(data.layout, LayoutSupport::ChannelFirstOnly) &&
        is_layout_supported(indices.layout, LayoutSupport::ChannelFirstOnly) &&
        is_layout_supported(updates.layout, LayoutSupport::ChannelFirstOnly));
}

}  // anonymous namespace

// ============================================================
// ScatterElements
// ============================================================

std::vector<TensorDesc> ScatterElements::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return {inputs[0]};
}

Status ScatterElements::validate(std::span<const TensorDesc> inputs,
                                 std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 3, 3, 1));
    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    const auto& updates = inputs[2];
    NNOPS_TRY(check_scatter_dtypes(data, indices, updates));
    NNOPS_TRY(check_scatter_layouts(data, indices, updates));

    if (data.rank < 1 || indices.rank != data.rank ||
        updates.rank != data.rank) {
        return Status::invalid_argument;
    }
    const int64_t axis = resolve_axis(attrs_.axis, data.rank);
    if (axis < 0 || axis >= data.rank) { return Status::invalid_argument; }

    for (int64_t d = 0; d < data.rank; ++d) {
        if (d == axis) { continue; }
        if (data.dims[static_cast<size_t>(d)] !=
            indices.dims[static_cast<size_t>(d)]) {
            return Status::invalid_argument;
        }
    }
    if (updates.dims != indices.dims) { return Status::invalid_argument; }
    return detail::require(outputs[0].dtype == data.dtype);
}

Status ScatterElements::compute(std::span<TensorView> outputs,
                                std::span<const TensorView> inputs,
                                const ComputeContext& ctx,
                                void* workspace)
{
    NNOPS_TRY(detail::check_views(outputs, inputs, 3, 3, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

// ============================================================
// ScatterND
// ============================================================

std::vector<TensorDesc> ScatterND::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return {inputs[0]};
}

Status ScatterND::validate(std::span<const TensorDesc> inputs,
                           std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 3, 3, 1));
    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    const auto& updates = inputs[2];
    NNOPS_TRY(check_scatter_dtypes(data, indices, updates));
    NNOPS_TRY(check_scatter_layouts(data, indices, updates));

    if (indices.rank < 1) { return Status::invalid_argument; }
    const int64_t qr = indices.rank;
    const int64_t k  = indices.dims[static_cast<size_t>(qr - 1)];
    if (k < 1 || k > data.rank) { return Status::invalid_argument; }

    // updates = indices[:-1] + data[k:], on the nose.
    int64_t want_rank = (qr - 1) + (data.rank - k);
    if (updates.rank != want_rank) { return Status::invalid_argument; }
    for (int64_t d = 0; d + 1 < qr; ++d) {
        if (updates.dims[static_cast<size_t>(d)] !=
            indices.dims[static_cast<size_t>(d)]) {
            return Status::invalid_argument;
        }
    }
    for (int64_t d = 0; d < data.rank - k; ++d) {
        if (updates.dims[static_cast<size_t>(qr - 1 + d)] !=
            data.dims[static_cast<size_t>(k + d)]) {
            return Status::invalid_argument;
        }
    }
    return detail::require(outputs[0].dtype == data.dtype);
}

Status ScatterND::compute(std::span<TensorView> outputs,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx,
                          void* workspace)
{
    NNOPS_TRY(detail::check_views(outputs, inputs, 3, 3, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
