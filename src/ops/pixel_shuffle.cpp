/// @file pixel_shuffle.cpp
/// @brief SpaceToDepth / PixelShuffle dispatch.

#include "nnops/ops/pixel_shuffle.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(SpaceToDepthAttributes, space_to_depth_cpu)
NNOPS_DECLARE_CPU(PixelShuffleAttributes, pixel_shuffle_cpu)

NNOPS_OP_BOILERPLATE(SpaceToDepth, SpaceToDepthAttributes, backend::cpu::space_to_depth_cpu,
                     nullptr, nullptr)
NNOPS_OP_BOILERPLATE(PixelShuffle, PixelShuffleAttributes, backend::cpu::pixel_shuffle_cpu,
                     nullptr, nullptr)

namespace {

/// The movement dtypes, as everywhere else that only copies elements.
Status check_permute_dtype(Backend be, const TensorDesc& in)
{
    if (!detail::dtype_in<DataType::f32, DataType::f16, DataType::s8, DataType::u8, DataType::s32,
                          DataType::s64>(be, in.dtype)) {
        return Status::unsupported;
    }
    return detail::require(is_layout_supported(in.layout, LayoutSupport::ChannelLastOnly),
                           Status::unsupported);
}

} // anonymous namespace

// ============================================================
// SpaceToDepth
// ============================================================

std::vector<TensorDesc> SpaceToDepth::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    const TensorDesc& in = inputs[0];
    TensorDesc out;
    out.rank = in.rank;
    out.layout = in.layout;
    out.dtype = in.dtype;
    out.dims = in.dims;
    const int64_t b = attrs_.block_size;
    if (in.rank == 4 && b > 0) {
        out.dims[1] = in.dims[1] / b;
        out.dims[2] = in.dims[2] / b;
        out.dims[3] = in.dims[3] * b * b;
    }
    return {out};
}

Status SpaceToDepth::validate(std::span<const TensorDesc> inputs,
                              std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1));
    const TensorDesc& in = inputs[0];
    NNOPS_TRY(check_permute_dtype(backend_, in));
    NNOPS_TRY(detail::check_rank(in.rank, 4, 4));

    const int64_t b = attrs_.block_size;
    if (b < 1) {
        return Status::invalid_argument;
    }
    if (in.dims[1] % b != 0 || in.dims[2] % b != 0) {
        return Status::invalid_argument;
    }

    const auto want = getOutputTensorDesc(inputs)[0];
    const TensorDesc& y = outputs[0];
    if (y.dtype != in.dtype || y.layout != in.layout || y.rank != 4) {
        return Status::invalid_argument;
    }
    for (int64_t d = 0; d < 4; ++d) {
        if (y.dims[static_cast<size_t>(d)] != want.dims[static_cast<size_t>(d)]) {
            return Status::invalid_argument;
        }
    }
    return Status::ok;
}

Status SpaceToDepth::compute(std::span<TensorView> outputs, std::span<const TensorView> inputs,
                             const ComputeContext& ctx, void* workspace)
{
    NNOPS_TRY(detail::check_views(backend_, outputs, inputs, 1, 1, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) {
        return Status::unsupported;
    }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

// ============================================================
// PixelShuffle
// ============================================================

std::vector<TensorDesc> PixelShuffle::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    const TensorDesc& in = inputs[0];
    TensorDesc out;
    out.rank = in.rank;
    out.layout = in.layout;
    out.dtype = in.dtype;
    out.dims = in.dims;
    const int64_t b = attrs_.block_size;
    if (in.rank == 4 && b > 0) {
        out.dims[1] = in.dims[1] * b;
        out.dims[2] = in.dims[2] * b;
        out.dims[3] = in.dims[3] / (b * b);
    }
    return {out};
}

Status PixelShuffle::validate(std::span<const TensorDesc> inputs,
                              std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1));
    const TensorDesc& in = inputs[0];
    NNOPS_TRY(check_permute_dtype(backend_, in));
    NNOPS_TRY(detail::check_rank(in.rank, 4, 4));

    const int64_t b = attrs_.block_size;
    if (b < 1) {
        return Status::invalid_argument;
    }
    if (in.dims[3] % (b * b) != 0) {
        return Status::invalid_argument;
    }

    const auto want = getOutputTensorDesc(inputs)[0];
    const TensorDesc& y = outputs[0];
    if (y.dtype != in.dtype || y.layout != in.layout || y.rank != 4) {
        return Status::invalid_argument;
    }
    for (int64_t d = 0; d < 4; ++d) {
        if (y.dims[static_cast<size_t>(d)] != want.dims[static_cast<size_t>(d)]) {
            return Status::invalid_argument;
        }
    }
    return Status::ok;
}

Status PixelShuffle::compute(std::span<TensorView> outputs, std::span<const TensorView> inputs,
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
