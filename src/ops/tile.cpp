/// @file tile.cpp
/// @brief Tile and Expand dispatch.

#include "nnops/ops/tile.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

#include <algorithm>

namespace nnops {

NNOPS_DECLARE_CPU(TileAttributes, tile_cpu)
NNOPS_DECLARE_CPU(ExpandAttributes, expand_cpu)

NNOPS_OP_BOILERPLATE(Tile, TileAttributes,
        backend::cpu::tile_cpu, nullptr, nullptr)
NNOPS_OP_BOILERPLATE(Expand, ExpandAttributes,
        backend::cpu::expand_cpu, nullptr, nullptr)

namespace {

Status check_repeat_dtype(const TensorDesc& in)
{
    if (!detail::dtype_in<DataType::f32, DataType::f16, DataType::s8,
                          DataType::u8, DataType::s32, DataType::s64>(in.dtype)) {
        return Status::unsupported;
    }
    return detail::require(
        is_layout_supported(in.layout, LayoutSupport::ChannelFirstOnly));
}

}  // anonymous namespace

// ============================================================
// Tile
// ============================================================

std::vector<TensorDesc> Tile::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    const auto& in = inputs[0];
    TensorDesc out = in;
    if (static_cast<int64_t>(attrs_.repeats.size()) == in.rank) {
        for (int64_t d = 0; d < in.rank; ++d) {
            out.dims[static_cast<size_t>(d)] =
                in.dims[static_cast<size_t>(d)] * attrs_.repeats[static_cast<size_t>(d)];
        }
    }
    return {out};
}

Status Tile::validate(std::span<const TensorDesc> inputs,
                      std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1));
    const auto& in = inputs[0];
    NNOPS_TRY(check_repeat_dtype(in));
    if (static_cast<int64_t>(attrs_.repeats.size()) != in.rank) {
        return Status::invalid_argument;
    }
    for (int64_t r : attrs_.repeats) {
        if (r < 0) { return Status::invalid_argument; }
    }
    return detail::require(outputs[0].dtype == in.dtype);
}

Status Tile::compute(std::span<TensorView> outputs,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace)
{
    NNOPS_TRY(detail::check_views(outputs, inputs, 1, 1, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

// ============================================================
// Expand
// ============================================================

std::vector<TensorDesc> Expand::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    const auto& in = inputs[0];
    const int64_t target = static_cast<int64_t>(attrs_.shape.size());
    const int64_t rank = std::max(target, in.rank);

    TensorDesc out = in;
    out.rank = rank;
    out.dims.resize(static_cast<size_t>(rank));
    for (int64_t d = 0; d < rank; ++d) {
        const int64_t ti = target - rank + d;      // target axis, right-aligned
        const int64_t ii = in.rank - rank + d;     // input axis, right-aligned
        const int64_t in_dim = (ii >= 0) ? in.dims[static_cast<size_t>(ii)] : 1;
        int64_t want;
        if (ti >= 0) {
            want = attrs_.shape[static_cast<size_t>(ti)];
            if (want < 0) { want = in_dim; }  // -1 = keep the input axis
        } else {
            want = in_dim;
        }
        out.dims[static_cast<size_t>(d)] = want;
    }
    return {out};
}

Status Expand::validate(std::span<const TensorDesc> inputs,
                        std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1));
    const auto& in = inputs[0];
    NNOPS_TRY(check_repeat_dtype(in));

    const auto desc = getOutputTensorDesc(inputs);
    const auto& want = desc[0];
    const int64_t rank = want.rank;
    for (int64_t d = 0; d < rank; ++d) {
        const int64_t ii = in.rank - rank + d;
        const int64_t in_dim = (ii >= 0) ? in.dims[static_cast<size_t>(ii)] : 1;
        const int64_t out_dim = want.dims[static_cast<size_t>(d)];
        // A 1 repeats; anything else must land on the same extent.
        if (in_dim != out_dim && in_dim != 1) {
            return Status::invalid_argument;
        }
    }
    return detail::require(outputs[0].dtype == in.dtype);
}

Status Expand::compute(std::span<TensorView> outputs,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace)
{
    NNOPS_TRY(detail::check_views(outputs, inputs, 1, 1, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
