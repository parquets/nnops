/// @file depthwise_conv2d.cpp
/// @brief DepthwiseConv2D operator dispatch — connects class API to backend implementations.

#include "nnops/ops/depthwise_conv2d.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

// Forward declarations of backend kernel entry points
namespace backend::cpu::reference {
    void depthwise_conv2d_ref(const DepthwiseConv2DAttributes& attrs,
                               TensorView& output,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* workspace);
}

namespace backend::cpu {
    void depthwise_conv2d_cpu(const DepthwiseConv2DAttributes& attrs,
                               TensorView& output,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void depthwise_conv2d_cuda(const DepthwiseConv2DAttributes& attrs,
                                TensorView& output,
                                std::span<const TensorView> inputs,
                                const ComputeContext& ctx,
                                void* workspace);
}
#endif

// ============================================================
// Impl — holds the backend-bound kernel function pointer
// ============================================================
struct DepthwiseConv2D::Impl {
    using KernelFn = void (*)(const DepthwiseConv2DAttributes&,
                               TensorView&,
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
        return backend::cuda::depthwise_conv2d_cuda;
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
// getOutputShapes
// ============================================================
std::vector<TensorDesc> DepthwiseConv2D::getOutputShapes(
    std::span<const TensorDesc> inputs) const
{
    return {depthwise_conv2d_output_shape(
        attrs_.kernel_size, attrs_.stride, attrs_.dilation, attrs_.padding,
        0 /* NOTSET */, inputs)};
}

// ============================================================
// compute
// ============================================================
void DepthwiseConv2D::compute(std::span<TensorView> outputs,
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

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

// ============================================================
// Functional API
// ============================================================
void depthwise_conv2d(const TensorView& input,
                       const TensorView& weight,
                       TensorView& output,
                       const DepthwiseConv2DAttributes& attrs,
                       const ComputeContext& ctx,
                       void* workspace)
{
    auto op = DepthwiseConv2D::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {input, weight};
    op->compute(output, ins, ctx, workspace);
}

void depthwise_conv2d(const TensorView& input,
                       const TensorView& weight,
                       const TensorView& bias,
                       TensorView& output,
                       const DepthwiseConv2DAttributes& attrs,
                       const ComputeContext& ctx,
                       void* workspace)
{
    auto op = DepthwiseConv2D::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {input, weight, bias};
    op->compute(output, ins, ctx, workspace);
}

}  // namespace nnops
