/// @file transpose_conv2d.cpp
/// @brief TransposeConv2D operator dispatch — connects class API to backend implementations.

#include "nnops/ops/transpose_conv2d.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

// Forward declarations of backend kernel entry points
namespace backend::cpu {
    void transpose_conv2d_cpu(const TransposeConv2DAttributes& attrs,
                               TensorView& output,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* workspace);
}

// ============================================================
// Impl — holds the backend-bound kernel function pointer
// ============================================================
struct TransposeConv2D::Impl {
    using KernelFn = void (*)(const TransposeConv2DAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};
TransposeConv2D::~TransposeConv2D() = default;


namespace {
auto resolve_transpose_conv2d_kernel(Backend backend) -> TransposeConv2D::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::transpose_conv2d_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return nullptr;  // backend::cuda::transpose_conv2d_cuda
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return nullptr;  // backend::vulkan::transpose_conv2d_vulkan
#endif
    }
    return nullptr;
}
}  // anonymous namespace

// ============================================================
// Factory
// ============================================================
std::unique_ptr<TransposeConv2D> TransposeConv2D::create(
    const TransposeConv2DAttributes& attrs, Backend backend)
{
    return std::unique_ptr<TransposeConv2D>(new TransposeConv2D(attrs, backend));
}

// ============================================================
// Constructor
// ============================================================
TransposeConv2D::TransposeConv2D(const TransposeConv2DAttributes& attrs,
                                   Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_transpose_conv2d_kernel(backend);
}

// ============================================================
// getOutputTensorDesc
// ============================================================
std::vector<TensorDesc> TransposeConv2D::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return transpose_conv2d_output_shape(
        attrs_.kernel_size, attrs_.stride, attrs_.dilation, attrs_.padding,
        attrs_.output_padding, attrs_.groups, inputs);
}

// ============================================================
// compute
// ============================================================
void TransposeConv2D::compute(std::span<TensorView> outputs,
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
    NNOPS_ASSERT(is_layout_supported(inputs[0].layout(), LayoutSupport::PackedOnly));

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
