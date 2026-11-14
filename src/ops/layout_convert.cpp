/// @file layout_convert.cpp
/// @brief LayoutConvert operator dispatch.

#include "nnops/ops/layout_convert.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "backend/cpu/layout_convert.hpp"

namespace nnops {

namespace {
    Status layout_convert_cpu(TensorView& output,
                              std::span<const TensorView> inputs,
                              const LayoutConvertAttributes& attrs,
                              const ComputeContext& ctx)
    {
        auto& input = inputs[0];
        TensorLayout src_layout = input.layout();
        TensorLayout dst_layout = output.layout();

        if (src_layout == dst_layout) { return Status::invalid_argument; }

        const bool channel_first_to_cl =
            (src_layout == TensorLayout::NCHW && dst_layout == TensorLayout::NHWC) ||
            (src_layout == TensorLayout::NCDHW && dst_layout == TensorLayout::NDHWC);
        const bool cl_to_channel_first =
            (src_layout == TensorLayout::NHWC && dst_layout == TensorLayout::NCHW) ||
            (src_layout == TensorLayout::NDHWC && dst_layout == TensorLayout::NCDHW);

        if (channel_first_to_cl) {
            convert_channel_first_to_channel_last(input, output, ctx);
        } else if (cl_to_channel_first) {
            convert_channel_last_to_channel_first(input, output, ctx);
        } else {
            // A weight-prepack tag or a mismatched rank-3 layout.
            return Status::unsupported;
        }
        return Status::ok;
    }
}  // anonymous namespace

struct LayoutConvert::Impl {
    using KernelFn = Status (*)(TensorView&,
                                std::span<const TensorView>,
                                const LayoutConvertAttributes&,
                                const ComputeContext&);
    KernelFn kernel_fn = nullptr;
};
LayoutConvert::~LayoutConvert() = default;


namespace {
auto resolve_layout_convert_kernel(Backend backend)
    -> LayoutConvert::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return layout_convert_cpu;
    case Backend::CUDA:
        return nullptr;
    case Backend::Vulkan:
        return nullptr;
    }
    return nullptr;
}
}  // anonymous namespace

NNOPS_OP_CREATE(LayoutConvert, LayoutConvertAttributes)

LayoutConvert::LayoutConvert(const LayoutConvertAttributes& attrs,
                               Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_layout_convert_kernel(backend);
}

std::vector<TensorDesc> LayoutConvert::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    NNOPS_ASSERT(inputs.size() == 1);
    return {layout_convert_output_shape(inputs, attrs_.target_layout)};
}

Status LayoutConvert::validate(std::span<const TensorDesc> inputs,
                               std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1));
    // Converting a layout to itself is a caller mistake, not a no-op.
    return detail::require(inputs[0].layout != outputs[0].layout);
}

Status LayoutConvert::compute(std::span<TensorView> outputs,
                              std::span<const TensorView> inputs,
                              const ComputeContext& ctx,
                              void* workspace)
{
    (void) workspace;
    NNOPS_TRY(detail::check_views(backend_, outputs, inputs, 1, 1, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(outputs[0], inputs, attrs_, ctx);
}

}  // namespace nnops
