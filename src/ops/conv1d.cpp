/// @file conv1d.cpp
/// @brief Conv1D operator dispatch — descriptor checks and the kernel table.
///
/// The filter order pairs with the activation exactly as it does on Conv2D: a
/// channel-last activation takes HWIO [K, IC/G, OC], a channel-first one any
/// other filter tag. Only the raw orders are accepted — a prepacked blob has no
/// 1D form, since the filter already is the [OC][IC/G·K] matrix the GEMM wants.
///
/// Everything else — the depthwise hand-off, the int8 route, the L-axis chunk
/// split — lives in the CPU entry (backend/cpu/conv/conv1d.cpp).

#include "nnops/ops/conv1d.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(Conv1DAttributes, conv1d_cpu)

NNOPS_OP_BOILERPLATE(Conv1D, Conv1DAttributes, backend::cpu::conv1d_cpu, nullptr, nullptr)

std::vector<TensorDesc> Conv1D::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    // HWIO's last axis is OC, so the channel count is the weight's last dim.
    const bool wt_hwio = inputs.size() > 1 &&
        is_channel_last_weight(inputs[1].layout);
    std::vector<TensorDesc> outs = conv1d_output_shape(
        attrs_.kernel_size, attrs_.stride, attrs_.dilation, attrs_.padding,
        static_cast<int>(attrs_.auto_pad), attrs_.groups, inputs, wt_hwio);

    // The quantized route writes s8/u8 when output_dtype says so, else the fp
    // activation's dtype (output_dtype for a pre-quantized one).
    if (inputs.size() > 1 && is_quantized_conv1d(inputs[1].dtype)) {
        const DataType act = inputs[0].dtype;
        const bool fp_act = (act == DataType::f32 || act == DataType::f16);
        const DataType odt = attrs_.output_dtype;
        const bool int_out = odt == DataType::s8 || odt == DataType::u8;
        outs[0].dtype = (int_out || !fp_act) ? odt : act;
    }
    return outs;
}

size_t Conv1D::getWorkspaceSize(std::span<const TensorDesc> inputs,
                                std::span<const TensorDesc> outputs) const
{
    (void)inputs;
    (void)outputs;
    // im2col scratch is pooled internally by the CPU kernel.
    return 0;
}

Status Conv1D::validate(std::span<const TensorDesc> inputs,
                        std::span<const TensorDesc> outputs) const
{
    // input, weight [, bias]
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 2, 3, 1));
    NNOPS_TRY(detail::require(
        is_layout_supported(inputs[0].layout, LayoutSupport::ChannelFirstOrChannelLast)));
    // Rank 3 exactly: the lift reads shape(2) as the signal and fabricates the
    // fourth axis, so anything else would be silently misdescribed.
    NNOPS_TRY(detail::check_rank(inputs[0].rank, 3, 3));
    NNOPS_TRY(detail::check_rank(outputs[0].rank, 3, 3));
    NNOPS_TRY(detail::check_rank(inputs[1].rank, 3, 3));

    // A raw filter only: the packed and Winograd forms describe a 2D kernel.
    const TensorLayout wl = inputs[1].layout;
    NNOPS_TRY(detail::require(is_channel_first_spatial(wl) ||
                              is_channel_last_weight(wl)));
    return detail::require(is_channel_last(inputs[0].layout) ==
                           is_channel_last_weight(wl));
}

Status Conv1D::compute(std::span<TensorView> outputs,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace)
{
    NNOPS_TRY(detail::check_views(backend_, outputs, inputs, 2, 3, 1));
    NNOPS_TRY(validate_views(outputs, inputs));

    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
