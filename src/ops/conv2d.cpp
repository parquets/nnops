/// @file conv2d.cpp
/// @brief Conv2D operator dispatch — connects class API to backend implementations.

#include "nnops/ops/conv2d.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

// Forward declarations of backend kernel entry points
namespace backend::cpu::reference {
    void conv2d_ref(const Conv2DAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

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

namespace {
auto resolve_conv2d_kernel(Backend backend) -> Conv2D::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::reference::conv2d_ref;
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

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

// ============================================================
// Functional API
// ============================================================
void conv2d(const TensorView& input,
            const TensorView& weight,
            TensorView& output,
            const Conv2DAttributes& attrs,
            const ComputeContext& ctx,
            void* workspace)
{
    auto op = Conv2D::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {input, weight};
    op->compute(output, ins, ctx, workspace);
}

void conv2d(const TensorView& input,
            const TensorView& weight,
            const TensorView& bias,
            TensorView& output,
            const Conv2DAttributes& attrs,
            const ComputeContext& ctx,
            void* workspace)
{
    auto op = Conv2D::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {input, weight, bias};
    op->compute(output, ins, ctx, workspace);
}

}  // namespace nnops
