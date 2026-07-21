/// @file depthwise_conv2d.cpp
/// @brief DepthwiseConv2D operator dispatch — connects class API to backend implementations.

#include "nnops/ops/depthwise_conv2d.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops {

// Forward declarations of backend kernel entry points
namespace backend::cpu::reference {
    void depthwise_conv2d_ref(const DepthwiseConv2DAttributes& attrs,
                               const TensorView& output,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* workspace);
}

namespace backend::cpu {
    void depthwise_conv2d_cpu(const DepthwiseConv2DAttributes& attrs,
                               const TensorView& output,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* workspace);
}

// ============================================================
// Impl — holds the backend-bound kernel function pointer
// ============================================================
struct DepthwiseConv2D::Impl {
    using KernelFn = void (*)(const DepthwiseConv2DAttributes&,
                               const TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_depthwise_conv2d_kernel(Backend backend) -> DepthwiseConv2D::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::depthwise_conv2d_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return nullptr;  // backend::cuda::depthwise_conv2d_cuda
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return nullptr;  // backend::vulkan::depthwise_conv2d_vulkan
#endif
    }
    return nullptr;
}
}  // anonymous namespace

// ============================================================
// Factory
// ============================================================
std::unique_ptr<DepthwiseConv2D> DepthwiseConv2D::create(
    const DepthwiseConv2DAttributes& attrs, Backend backend)
{
    return std::unique_ptr<DepthwiseConv2D>(new DepthwiseConv2D(attrs, backend));
}

// ============================================================
// Constructor
// ============================================================
DepthwiseConv2D::DepthwiseConv2D(const DepthwiseConv2DAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_depthwise_conv2d_kernel(backend);
}

// ============================================================
// getWorkspace
// ============================================================
size_t DepthwiseConv2D::getWorkspace() const
{
    return 0;
}

// ============================================================
// compute
// ============================================================
void DepthwiseConv2D::compute(const TensorView& output,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* workspace)
{
    NNOPS_ASSERT(inputs.size() >= 2);
    NNOPS_ASSERT(inputs.size() <= 3);  // input, weight [, bias]
    NNOPS_ASSERT(output.data() != nullptr);
    NNOPS_ASSERT(inputs[0].data() != nullptr);
    NNOPS_ASSERT(inputs[1].data() != nullptr);

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

// ============================================================
// Functional API
// ============================================================
void depthwise_conv2d(const TensorView& input,
                       const TensorView& weight,
                       const TensorView& output,
                       const DepthwiseConv2DAttributes& attrs,
                       const ComputeContext& ctx,
                       void* workspace)
{
    auto op = DepthwiseConv2D::create(attrs, Backend::CPU);
    const TensorView ins[] = {input, weight};
    op->compute(output, ins, ctx, workspace);
}

void depthwise_conv2d(const TensorView& input,
                       const TensorView& weight,
                       const TensorView& bias,
                       const TensorView& output,
                       const DepthwiseConv2DAttributes& attrs,
                       const ComputeContext& ctx,
                       void* workspace)
{
    auto op = DepthwiseConv2D::create(attrs, Backend::CPU);
    const TensorView ins[] = {input, weight, bias};
    op->compute(output, ins, ctx, workspace);
}

}  // namespace nnops
