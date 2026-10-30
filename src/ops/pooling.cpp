/// @file pooling.cpp
/// @brief Pooling operator dispatch — planar or channel-last I/O, direct
///        pass-through (output layout follows the input's).
///
/// Attribute semantics live in pooling.hpp. What this file adds is auto_pad:
/// SAME_UPPER / SAME_LOWER / VALID are resolved here into explicit leading
/// padding, because the kernels only see a resolved pad. Planar input is
/// f32/f16; quantized pooling is channel-last only (the lane is the channel).

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
        return detail::require(
            detail::dtype_in<DataType::f32, DataType::f16>(in.dtype),
            Status::unsupported);
    }
    if (is_channel_last(in.layout)) {
        return detail::require(
            detail::dtype_in<DataType::f32, DataType::f16,
                             DataType::s8, DataType::u8>(in.dtype));
    }
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

    PoolingAttributes attrs = attrs_;
    if (attrs.global) {
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
