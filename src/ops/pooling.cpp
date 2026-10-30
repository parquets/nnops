/// @file pooling.cpp
/// @brief Pooling operator dispatch — planar or channel-last I/O, direct
///        pass-through (output layout follows the input's).

#include "nnops/ops/pooling.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(PoolingAttributes, pooling_cpu)
NNOPS_DECLARE_CUDA(PoolingAttributes, pooling_cuda)

NNOPS_OP_BOILERPLATE(Pooling, PoolingAttributes,
        backend::cpu::pooling_cpu, NNOPS_IF_CUDA(backend::cuda::pooling_cuda), nullptr)

std::vector<TensorDesc> Pooling::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    if (attrs_.global) {
        // Every spatial axis pools to one; the window fields are not read, so
        // the generic convolution is not run either.
        const auto& in = inputs[0];
        TensorDesc out;
        out.rank   = in.rank;
        out.layout = in.layout;
        out.dtype  = in.dtype;
        out.dims.resize(static_cast<size_t>(in.rank));
        out.dims[0] = in.dims[0];
        out.dims[1] = in.dims[1];
        for (int64_t d = 2; d < in.rank; ++d) { out.dims[static_cast<size_t>(d)] = 1; }
        return {out};
    }
    return {pooling_output_shape(
        attrs_.kernel_shape, attrs_.stride, attrs_.dilation, attrs_.padding,
        static_cast<int>(attrs_.auto_pad), inputs)};
}

Status Pooling::validate(std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1));
    const auto& in = inputs[0];
    if (is_planar_spatial(in.layout)) {
        // Planar pooling has f32/f16 kernels; quantized planes do not.
        return detail::require(
            detail::dtype_in<DataType::f32, DataType::f16>(in.dtype),
            Status::unsupported);
    }
    if (is_channel_last(in.layout)) {
        // The lane is the channel, so the channel-lane kernels cover the whole
        // type menu: f32/f16 and the quantized s8/u8 pairs.
        return detail::require(
            detail::dtype_in<DataType::f32, DataType::f16,
                             DataType::s8, DataType::u8>(in.dtype));
    }
    // Anything else — a channel-packed tensor, say — has no route at all.
    return Status::unsupported;
}

Status Pooling::compute(std::span<TensorView> outputs,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx,
                        void* workspace)
{
    (void)workspace;
    NNOPS_TRY(detail::check_views(outputs, inputs, 1, 1, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    auto& output = outputs[0];
    auto& input = inputs[0];

    // auto_pad lives in the descriptor; kernels only see explicit leading padding.
    PoolingAttributes attrs = attrs_;
    if (attrs.global) {
        // The window is the whole extent, so there is no padding to resolve.
        attrs.auto_pad = PoolingAttributes::AutoPad::NOTSET;
    } else if (attrs.auto_pad != PoolingAttributes::AutoPad::NOTSET) {
        const int ap = static_cast<int>(attrs.auto_pad);
        const int64_t srank = PoolingAttributes::spatial_rank(input.rank());
        for (int i = 0; i < srank; ++i) {
            // 2D uses [1],[2]; 3D uses [0],[1],[2].
            const size_t ai = static_cast<size_t>(3 - srank + i);
            attrs.padding[ai] = resolve_leading_padding(
                input.shape(2 + i), attrs.kernel_shape[ai], attrs.stride[ai],
                attrs.dilation[ai], attrs.padding[ai], ap);
        }
        attrs.auto_pad = PoolingAttributes::AutoPad::NOTSET;
    }

    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs, output, inputs, ctx, nullptr);
}

}  // namespace nnops
