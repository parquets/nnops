/// @file gather.cpp
/// @brief Gather, GatherElements and GatherND dispatch.

#include "nnops/ops/gather.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(GatherAttributes, gather_cpu)
NNOPS_DECLARE_CPU(GatherElementsAttributes, gather_elements_cpu)
NNOPS_DECLARE_CPU(GatherNDAttributes, gather_nd_cpu)
NNOPS_DECLARE_CUDA(GatherAttributes, gather_cuda)
NNOPS_DECLARE_CUDA(GatherElementsAttributes, gather_elements_cuda)
NNOPS_DECLARE_CUDA(GatherNDAttributes, gather_nd_cuda)

NNOPS_OP_BOILERPLATE(Gather, GatherAttributes,
        backend::cpu::gather_cpu, NNOPS_IF_CUDA(backend::cuda::gather_cuda),
        nullptr)
NNOPS_OP_BOILERPLATE(GatherElements, GatherElementsAttributes,
        backend::cpu::gather_elements_cpu,
        NNOPS_IF_CUDA(backend::cuda::gather_elements_cuda), nullptr)
NNOPS_OP_BOILERPLATE(GatherND, GatherNDAttributes,
        backend::cpu::gather_nd_cpu, NNOPS_IF_CUDA(backend::cuda::gather_nd_cuda),
        nullptr)

namespace {

Status check_gather_dtypes(const TensorDesc& data, const TensorDesc& indices)
{
    if (!detail::dtype_in<DataType::f32, DataType::f16, DataType::s8,
                          DataType::u8, DataType::s32, DataType::s64>(data.dtype)) {
        return Status::unsupported;
    }
    return detail::require(
        detail::dtype_in<DataType::s64, DataType::s32>(indices.dtype));
}

Status check_gather_layouts(const TensorDesc& data, const TensorDesc& indices)
{
    return detail::require(
        is_layout_supported(data.layout, LayoutSupport::ChannelFirstOnly) &&
        is_layout_supported(indices.layout, LayoutSupport::ChannelFirstOnly));
}

}  // anonymous namespace

// ============================================================
// Gather
// ============================================================

std::vector<TensorDesc> Gather::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    const int64_t axis = resolve_axis(attrs_.axis, data.rank);

    TensorDesc out;
    out.dtype  = data.dtype;
    out.layout = data.layout;
    for (int64_t d = 0; d < axis; ++d) {
        out.dims.push_back(data.dims[static_cast<size_t>(d)]);
    }
    for (int64_t d = 0; d < indices.rank; ++d) {
        out.dims.push_back(indices.dims[static_cast<size_t>(d)]);
    }
    for (int64_t d = axis + 1; d < data.rank; ++d) {
        out.dims.push_back(data.dims[static_cast<size_t>(d)]);
    }
    out.rank = static_cast<int64_t>(out.dims.size());
    return {out};
}

Status Gather::validate(std::span<const TensorDesc> inputs,
                        std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 2, 2, 1));
    NNOPS_TRY(check_gather_dtypes(inputs[0], inputs[1]));
    NNOPS_TRY(check_gather_layouts(inputs[0], inputs[1]));
    const auto& data = inputs[0];
    if (data.rank < 1) { return Status::invalid_argument; }
    const int64_t axis = resolve_axis(attrs_.axis, data.rank);
    if (axis < 0 || axis >= data.rank) { return Status::invalid_argument; }
    return detail::require(outputs[0].dtype == data.dtype);
}

Status Gather::compute(std::span<TensorView> outputs,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace)
{
    NNOPS_TRY(detail::check_views(outputs, inputs, 2, 2, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

// ============================================================
// GatherElements
// ============================================================

std::vector<TensorDesc> GatherElements::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    TensorDesc out = inputs[1];
    out.dtype  = inputs[0].dtype;
    out.layout = inputs[0].layout;
    return {out};
}

Status GatherElements::validate(std::span<const TensorDesc> inputs,
                                std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 2, 2, 1));
    NNOPS_TRY(check_gather_dtypes(inputs[0], inputs[1]));
    NNOPS_TRY(check_gather_layouts(inputs[0], inputs[1]));
    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    if (data.rank < 1 || data.rank != indices.rank) {
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
    return detail::require(outputs[0].dtype == data.dtype);
}

Status GatherElements::compute(std::span<TensorView> outputs,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* workspace)
{
    NNOPS_TRY(detail::check_views(outputs, inputs, 2, 2, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

// ============================================================
// GatherND
// ============================================================

std::vector<TensorDesc> GatherND::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    const int64_t qr = indices.rank;
    const int64_t k = (qr >= 1) ? indices.dims[static_cast<size_t>(qr - 1)] : 0;
    const int64_t b = attrs_.batch_dims;

    TensorDesc out;
    out.dtype  = data.dtype;
    out.layout = data.layout;
    for (int64_t d = 0; d + 1 < qr; ++d) {
        out.dims.push_back(indices.dims[static_cast<size_t>(d)]);
    }
    for (int64_t d = b + k; d < data.rank; ++d) {
        out.dims.push_back(data.dims[static_cast<size_t>(d)]);
    }
    out.rank = static_cast<int64_t>(out.dims.size());
    return {out};
}

Status GatherND::validate(std::span<const TensorDesc> inputs,
                          std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 2, 2, 1));
    NNOPS_TRY(check_gather_dtypes(inputs[0], inputs[1]));
    NNOPS_TRY(check_gather_layouts(inputs[0], inputs[1]));
    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    if (indices.rank < 1) { return Status::invalid_argument; }
    const int64_t k = indices.dims[static_cast<size_t>(indices.rank - 1)];
    const int64_t b = attrs_.batch_dims;
    if (b < 0 || b >= data.rank || b >= indices.rank) {
        return Status::invalid_argument;
    }
    if (k < 1 || b + k > data.rank) { return Status::invalid_argument; }
    // The batch axes are shared, so their extents must agree.
    for (int64_t d = 0; d < b; ++d) {
        if (indices.dims[static_cast<size_t>(d)] !=
            data.dims[static_cast<size_t>(d)]) {
            return Status::invalid_argument;
        }
    }
    return detail::require(outputs[0].dtype == data.dtype);
}

Status GatherND::compute(std::span<TensorView> outputs,
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
