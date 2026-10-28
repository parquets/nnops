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

        // 2D: NCHW ↔ NCHWC8
        if (src_layout == TensorLayout::NCHW && dst_layout == TensorLayout::NCHWC8) {
            pack_nchw_to_nchwc8(input, output, ctx);
        } else if (src_layout == TensorLayout::NCHWC8 && dst_layout == TensorLayout::NCHW) {
            unpack_nchwc8_to_nchw(input, output, ctx);
        }
        // 3D: NCDHW ↔ NCDHWC8
        else if (src_layout == TensorLayout::NCDHW && dst_layout == TensorLayout::NCDHWC8) {
            pack_ncdhw_to_ncdhwc8(input, output, ctx);
        } else if (src_layout == TensorLayout::NCDHWC8 && dst_layout == TensorLayout::NCDHW) {
            unpack_ncdhwc8_to_ncdhw(input, output, ctx);
        }
        else {
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
    NNOPS_TRY(detail::check_views(outputs, inputs, 1, 1, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(outputs[0], inputs, attrs_, ctx);
}

}  // namespace nnops
