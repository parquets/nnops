/// @file conv3d.cpp
/// @brief Conv3D operator dispatch — connects class API to backend implementations.

#include "nnops/ops/conv3d.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(Conv3DAttributes, conv3d_cpu)

NNOPS_OP_BOILERPLATE(Conv3D, Conv3DAttributes, backend::cpu::conv3d_cpu, nullptr, nullptr)

std::vector<TensorDesc> Conv3D::getOutputTensorDesc(std::span<const TensorDesc> inputs) const
{
    std::vector<TensorDesc> outs =
        conv3d_output_shape(attrs_.kernel_size, attrs_.stride, attrs_.dilation, attrs_.padding,
                            static_cast<int>(attrs_.auto_pad), attrs_.groups, inputs);

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

    if (!impl_->kernel_fn) {
        return Status::unsupported;
    }
    return impl_->kernel_fn(attrs, output, inputs, ctx, workspace);
}

} // namespace nnops
