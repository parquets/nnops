/// @file conv2d.cpp
/// @brief Conv2D operator dispatch — connects class API to backend implementations.

#include "nnops/ops/conv2d.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "backend/cpu/conv2d_im2col.h"

namespace nnops {

// ============================================================
// Impl — holds the backend-bound kernel function pointer
// ============================================================
struct Conv2D::Impl {
    using KernelFn = void (*)(const Conv2DAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};
Conv2D::~Conv2D() = default;


namespace {
auto resolve_conv2d_kernel(Backend backend) -> Conv2D::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::conv2d_im2col_kernel;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return nullptr;  // backend::cuda::conv2d_cuda
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return nullptr;  // backend::vulkan::conv2d_vulkan
#endif
    }
    return nullptr;
}
}  // anonymous namespace

// ============================================================
// Factory
// ============================================================
std::unique_ptr<Conv2D> Conv2D::create(const Conv2DAttributes& attrs,
                                        Backend backend)
{
    return std::unique_ptr<Conv2D>(new Conv2D(attrs, backend));
}

// ============================================================
// Constructor
// ============================================================
Conv2D::Conv2D(const Conv2DAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_conv2d_kernel(backend);
}

// ============================================================
// getOutputTensorDesc
// ============================================================
std::vector<TensorDesc> Conv2D::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return {conv2d_output_shape(
        attrs_.kernel_size, attrs_.stride, attrs_.dilation, attrs_.padding,
        static_cast<int>(attrs_.auto_pad), attrs_.groups, inputs)};
}

// ============================================================
// getWorkspaceSize
// ============================================================
size_t Conv2D::getWorkspaceSize(std::span<const TensorDesc> inputs,
                                std::span<const TensorDesc> outputs) const
{
    (void)inputs;
    (void)outputs;
    // im2col scratch is pooled internally by the CPU kernel.
    return 0;
}

// ============================================================
// compute
// ============================================================
void Conv2D::compute(std::span<TensorView> outputs,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace)
{
    NNOPS_ASSERT(inputs.size() >= 2);
    NNOPS_ASSERT(inputs.size() <= 3);  // input, weight [, bias]
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());
    NNOPS_ASSERT(!inputs[1].is_empty());
    NNOPS_ASSERT(is_layout_supported(inputs[0].layout(), LayoutSupport::PlanarOnly));

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
