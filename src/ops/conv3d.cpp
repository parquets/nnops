/// @file conv3d.cpp
/// @brief Conv3D operator dispatch — connects class API to backend implementations.
///
/// The weight is DHWIO [KD][KH][KW][IC/G][OC] — TensorFlow's conv3d filter — or
/// the blob prepackWeights() packed it into (PackedDHWIO, PackedDHWIOSme on an
/// SME machine). compute() takes either; a packed blob is addressed as the
/// GEMM's B and nothing else, so it is refused by every shape it was not
/// packed for.

#include "nnops/ops/conv3d.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "nnops/ops/depthwise_conv.hpp"
#include "backend/cpu/conv/conv3d_implicit_gemm_internal.h"

#include <span>
#include <vector>

namespace nnops {

namespace {

/// One input channel a group on both sides: the grouped GEMM degenerates there —
/// a single-row product re-dispatched per group — while the depthwise kernels
/// accumulate whole voxels a lane. The filter DHWIO [KD][KH][KW][1][C] is the
/// depthwise [KD][KH][KW][C] order with the singleton axis dropped, so the hand
/// over converts nothing.
bool is_depthwise_geometry(const Conv3DAttributes& attrs, const TensorView& weight)
{
    if (attrs.groups <= 1 || !is_channel_last_weight_3d(weight.layout()) || weight.rank() != 5) {
        return false;
    }
    return weight.shape(3) == 1 && weight.shape(4) == attrs.groups;
}

bool is_fp_depthwise(DataType dtype) noexcept
{
    return dtype == DataType::f32 || dtype == DataType::f16;
}

/// The depthwise kernels walk the filter as a dense [KD][KH][KW][C] run, so a
/// padded row is the one thing the hand-over cannot carry: it would read the pad
/// as channels. A pitched filter stays on the grouped route.
bool is_dense_run(const TensorView& w) noexcept
{
    const int64_t cols = w.shape(w.rank() - 1);
    return w.pitch() == static_cast<int64_t>(data_type_row_bytes(w.data_type(), cols));
}

} // anonymous namespace

NNOPS_DECLARE_CPU(Conv3DAttributes, conv3d_cpu)
NNOPS_DECLARE_CUDA(Conv3DAttributes, conv3d_cuda)
// The depthwise hand-off a groups == channels geometry takes.
NNOPS_DECLARE_CPU(DepthwiseConvAttributes, depthwise_conv_cpu)
NNOPS_DECLARE_CUDA(DepthwiseConvAttributes, depthwise_conv_cuda)

NNOPS_OP_BOILERPLATE(Conv3D, Conv3DAttributes, backend::cpu::conv3d_cpu,
                     NNOPS_IF_CUDA(backend::cuda::conv3d_cuda), nullptr)

std::vector<TensorDesc> Conv3D::getOutputTensorDesc(std::span<const TensorDesc> inputs) const
{
    // A raw filter is DHWIO, so OC is its last axis; a packed blob leads with
    // OC. Anything else is malformed and is caught by validate().
    const bool raw = inputs.size() > 1 && inputs[1].rank == 5 &&
                     is_channel_last_weight_3d(inputs[1].layout);
    std::vector<TensorDesc> outs =
        conv3d_output_shape(attrs_.kernel_size, attrs_.stride, attrs_.dilation, attrs_.padding,
                            static_cast<int>(attrs_.auto_pad), attrs_.groups, inputs, raw);

    // The quantized route writes s8/u8 when output_dtype says so, else the fp
    // activation's dtype (output_dtype for a pre-quantized one); an unsupported
    // request passes through for the kernel to refuse.
    if (inputs.size() > 1 && is_quantized_conv3d(inputs[1].dtype)) {
        const DataType act = inputs[0].dtype;
        const bool fp_act = (act == DataType::f32 || act == DataType::f16);
        const DataType odt = attrs_.output_dtype;
        const bool int_out = odt == DataType::s8 || odt == DataType::u8;
        outs[0].dtype = (int_out || !fp_act) ? odt : act;
    }
    return outs;
}

Status Conv3D::validate(std::span<const TensorDesc> inputs,
                        std::span<const TensorDesc> outputs) const
{
    // input, weight [, bias]
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 2, 3, 1));
    NNOPS_TRY(
        detail::require(is_layout_supported(inputs[0].layout, LayoutSupport::ChannelLastOnly)));
    NNOPS_TRY(detail::require(inputs[0].rank == 5));

    // The filter order must pair with the activation: a DHWIO filter (then its
    // channel counts must split into the attribute's groups) or the packed
    // form prepackWeights produced.
    const auto& wt = inputs[1];
    if (is_channel_last_weight_3d(wt.layout)) {
        NNOPS_TRY(detail::require(wt.rank == 5));
        NNOPS_TRY(detail::require(attrs_.groups > 0));
        NNOPS_TRY(detail::require(wt.dims[3] * attrs_.groups == inputs[0].dims[4]));
        NNOPS_TRY(detail::require(wt.dims[4] % attrs_.groups == 0));
    } else {
        NNOPS_TRY(detail::require(is_packed_conv3d_weight(wt.layout) && wt.rank == 2));
    }

    // A quantized filter quantizes its activation per call and requantizes into
    // the same float, so a floating-point activation and output are one type:
    // getOutputTensorDesc reports the activation's dtype for such a call, and
    // this refuses the view that asks for the other one instead of converting
    // it silently.
    if (inputs.size() > 1 && is_quantized_conv3d(inputs[1].dtype) && !outputs.empty()) {
        const DataType act = inputs[0].dtype;
        const DataType odt = outputs[0].dtype;
        const bool fp_act = act == DataType::f32 || act == DataType::f16;
        const bool fp_out = odt == DataType::f32 || odt == DataType::f16;
        if (fp_act && fp_out && odt != act) {
            return Status::unsupported;
        }
    }
    return Status::ok;
}

Status Conv3D::compute(std::span<TensorView> outputs, std::span<const TensorView> inputs,
                       const ComputeContext& ctx, void* workspace)
{
    NNOPS_TRY(detail::check_views(backend_, outputs, inputs, 2, 3, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    auto& output = outputs[0];

    // auto_pad lives in the descriptor; kernels only see explicit leading padding.
    Conv3DAttributes attrs = attrs_;
    if (attrs.auto_pad != Conv3DAttributes::AutoPad::NOTSET) {
        const int ap = static_cast<int>(attrs.auto_pad);
        for (int i = 0; i < 3; ++i) {
            attrs.padding[static_cast<size_t>(i)] = resolve_leading_padding(
                inputs[0].shape(1 + i), attrs.kernel_size[static_cast<size_t>(i)],
                attrs.stride[static_cast<size_t>(i)], attrs.dilation[static_cast<size_t>(i)],
                attrs.padding[static_cast<size_t>(i)], ap);
        }
        attrs.auto_pad = Conv3DAttributes::AutoPad::NOTSET;
    }

    // A depthwise geometry reads the filter Conv3D was handed, so it costs no
    // conversion in either layout: one channel a group on both sides makes the
    // DHWIO filter the depthwise [KD][KH][KW][C] one axis shorter. The hand-over
    // is worth roughly an order of magnitude — a depthwise row measured 18
    // GFLOPS on the device, where every group is its own block on a tile that
    // can only fill one of its 64 columns.
    if ((backend_ == Backend::CPU || backend_ == Backend::CUDA) && inputs.size() >= 2 &&
        is_fp_depthwise(inputs[0].data_type()) &&
        inputs[1].data_type() == inputs[0].data_type() && output.data_type() == inputs[0].data_type() &&
        is_depthwise_geometry(attrs, inputs[1]) && is_dense_run(inputs[1])) {
        const auto& wgt = inputs[1];
        DepthwiseConvAttributes dw;
        dw.kernel_size = attrs.kernel_size;
        dw.stride = attrs.stride;
        dw.dilation = attrs.dilation;
        dw.padding = attrs.padding;
        dw.epilogue = attrs.epilogue;
        dw.add_to = attrs.add_to;
        // The filter as [KD][KH][KW][C]: the innermost row is the channel run in
        // both orders, so the pitch and the layout are all that change.
        const int64_t wsh[4] = {wgt.shape(0), wgt.shape(1), wgt.shape(2), wgt.shape(4)};
        const TensorView w_view(std::span<const int64_t>(wsh, 4), wgt.data_type(),
                                const_cast<void*>(wgt.ptr<void>()), wgt.pitch(),
                                TensorLayout::HWCM);
        TensorView operands[3] = {inputs[0], w_view, TensorView{}};
        const size_t n_in = inputs.size();
        if (n_in > 2) {
            operands[2] = inputs[2];
        }
        const std::span<const TensorView> ins(operands, n_in);
        if (backend_ == Backend::CPU) {
            return backend::cpu::depthwise_conv_cpu(dw, output, ins, ctx, workspace);
        }
#if defined(NNOPS_HAS_CUDA)
        return backend::cuda::depthwise_conv_cuda(dw, output, ins, ctx, workspace);
#else
        return Status::unsupported;
#endif
    }

    if (!impl_->kernel_fn) {
        return Status::unsupported;
    }
    return impl_->kernel_fn(attrs, output, inputs, ctx, workspace);
}

void Conv3D::prepackWeights(std::span<const TensorView> inputs, std::span<TensorView> outputs,
                            const ComputeContext& ctx)
{
    // Only the CPU im2col route has a packed filter; anything else answers "no
    // packed form" with an empty view, as MatMul does for a dtype it cannot
    // pack.
    if (backend_ != Backend::CPU || inputs.empty() || outputs.empty()) {
        return;
    }
    const auto& weight = inputs[0];
    NNOPS_ASSERT(weight.rank() == 5 && is_channel_last_weight_3d(weight.layout()) &&
                 "Conv3D::prepackWeights: the filter must be a DHWIO [KD][KH][KW][IC/G][OC] "
                 "tensor");

    const TensorDesc d = backend::cpu::conv3d_packed_weight_desc(attrs_, weight.desc());
    if (d.rank == 0) {
        return;
    }
    std::vector<int64_t> shape(d.dims.begin(), d.dims.begin() + d.rank);
    const int64_t esz = static_cast<int64_t>(data_type_size(d.dtype));
    if (outputs[0].is_empty()) {
        // The pitch is the row length in bytes: a packed blob's rows are not
        // the default alignment, so the default would not describe it.
        outputs[0] = TensorView(shape, d.dtype, nullptr, d.row_stride_elems * esz, d.layout,
                                weight.quant_params());
        return;
    }

    NNOPS_ASSERT(outputs[0].layout() == d.layout && outputs[0].rank() == d.rank &&
                 outputs[0].shape(0) == shape[0] && outputs[0].shape(1) == shape[1] &&
                 "Conv3D::prepackWeights: the packed buffer does not match the geometry");
    backend::cpu::conv3d_packed_weight_pack(attrs_, weight, outputs[0], ctx);
}

} // namespace nnops
