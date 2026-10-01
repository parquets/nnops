/// @file layout_convert.cpp
/// @brief LayoutConvert operator dispatch.

#include "nnops/ops/layout_convert.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "backend/cpu/layout_convert.hpp"

namespace nnops {

namespace {
    void layout_convert_cpu(TensorView& output,
                            std::span<const TensorView> inputs,
                            const LayoutConvertAttributes& attrs,
                            const ComputeContext& ctx)
    {
        auto& input = inputs[0];
        TensorLayout src_layout = input.layout();
        TensorLayout dst_layout = output.layout();

        NNOPS_ASSERT(src_layout != dst_layout);

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
            NNOPS_ASSERT(!"layout_convert_cpu: unsupported layout pair");
        }
    }
}  // anonymous namespace

// ============================================================
// Impl
// ============================================================
struct LayoutConvert::Impl {
    using KernelFn = void (*)(TensorView&,
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

std::unique_ptr<LayoutConvert> LayoutConvert::create(
    const LayoutConvertAttributes& attrs, Backend backend)
{
    return std::unique_ptr<LayoutConvert>(new LayoutConvert(attrs, backend));
}

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

void LayoutConvert::compute(std::span<TensorView> outputs,
                             std::span<const TensorView> inputs,
                             const ComputeContext& ctx,
                             void* workspace)
{
    (void) workspace;
    NNOPS_ASSERT(inputs.size() == 1);
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());

    impl_->kernel_fn(output, inputs, attrs_, ctx);
}

}  // namespace nnops
