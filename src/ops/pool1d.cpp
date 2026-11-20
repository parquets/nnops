/// @file pool1d.cpp
/// @brief Pool1D operator dispatch — NHWC/NDHWC I/O, output layout follows the
///        input's.
///
/// The dtype rules are Pooling's, read off a rank-3 tensor: f32/f16 and the
/// quantized s8/u8 pairs, whose lane is the channel; adaptive has no quantized
/// form. auto_pad resolution and the adaptive-all-ones shortcut are the 2D
/// backend's (backend/cpu/conv/pooling.cpp), which also carries the rank-4 lift.

#include "nnops/ops/pool1d.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(Pool1DAttributes, pool1d_cpu)

NNOPS_OP_BOILERPLATE(Pool1D, Pool1DAttributes, backend::cpu::pool1d_cpu, nullptr, nullptr)

std::vector<TensorDesc> Pool1D::getOutputTensorDesc(std::span<const TensorDesc> inputs) const
{
    if (attrs_.global || attrs_.adaptive()) {
        const auto& in = inputs[0];
        TensorDesc out;
        out.rank = 3;
        out.layout = in.layout;
        out.dtype = in.dtype;
        out.dims.resize(3);
        // Channel-last [N, L, C]: the length is axis 1, the channel axis last.
        out.dims[0] = in.dims[0];
        out.dims[1] = attrs_.global ? 1 : attrs_.output_size;
        out.dims[2] = in.dims[2];
        return {out};
    }
    return {pool1d_output_shape(attrs_.kernel_shape, attrs_.stride, attrs_.dilation, attrs_.padding,
                                static_cast<int>(attrs_.auto_pad), inputs)};
}

Status Pool1D::validate(std::span<const TensorDesc> inputs,
                        std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1));
    NNOPS_TRY(detail::check_rank(inputs[0].rank, 3, 3));
    NNOPS_TRY(detail::check_rank(outputs[0].rank, 3, 3));
    const auto& in = inputs[0];

    if (!is_channel_last(in.layout)) {
        return Status::unsupported;
    }
    // Adaptive pooling walks the window per output position; the quantized
    // channel-lane kernel has no such form, so it stays float.
    if (attrs_.adaptive() && detail::dtype_in<DataType::s8, DataType::u8>(backend_, in.dtype)) {
        return Status::unsupported;
    }
    return detail::require(
        detail::dtype_in<DataType::f32, DataType::f16, DataType::s8, DataType::u8>(backend_,
                                                                                   in.dtype));
}

Status Pool1D::compute(std::span<TensorView> outputs, std::span<const TensorView> inputs,
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
