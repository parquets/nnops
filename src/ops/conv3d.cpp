/// @file conv3d.cpp
/// @brief Conv3D operator dispatch — connects class API to backend implementations.

#include "nnops/ops/conv3d.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops {

// Forward declarations of backend kernel entry points
namespace backend::cpu::reference {
    void conv3d_ref(const Conv3DAttributes& attrs,
                    const TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

// ============================================================
// Impl — holds the backend-bound kernel function pointer
// ============================================================
struct Conv3D::Impl {
    using KernelFn = void (*)(const Conv3DAttributes&,
                               const TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_conv3d_kernel(Backend backend) -> Conv3D::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::reference::conv3d_ref;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return nullptr;  // backend::cuda::conv3d_cuda
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return nullptr;  // backend::vulkan::conv3d_vulkan
#endif
    }
    return nullptr;
}
}  // anonymous namespace

// ============================================================
// Factory
// ============================================================
std::unique_ptr<Conv3D> Conv3D::create(const Conv3DAttributes& attrs,
                                        Backend backend)
{
    return std::unique_ptr<Conv3D>(new Conv3D(attrs, backend));
}

// ============================================================
// Constructor
// ============================================================
Conv3D::Conv3D(const Conv3DAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_conv3d_kernel(backend);
}

// ============================================================
// getWorkspace
// ============================================================
size_t Conv3D::getWorkspace() const
{
    // Reference CPU implementation uses no workspace.
    // Optimized GEMM-based implementations may need im2col buffer.
    return 0;
}

// ============================================================
// compute
// ============================================================
void Conv3D::compute(std::span<const TensorView> outputs,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace)
{
    NNOPS_ASSERT(inputs.size() >= 2);
    NNOPS_ASSERT(inputs.size() <= 3);  // input, weight [, bias]
    NNOPS_ASSERT(outputs.size() == 1);
    const auto& output = outputs[0];
    NNOPS_ASSERT(output.data() != nullptr);
    NNOPS_ASSERT(inputs[0].data() != nullptr);
    NNOPS_ASSERT(inputs[1].data() != nullptr);

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

// ============================================================
// Functional API
// ============================================================
void conv3d(const TensorView& input,
            const TensorView& weight,
            const TensorView& output,
            const Conv3DAttributes& attrs,
            const ComputeContext& ctx,
            void* workspace)
{
    auto op = Conv3D::create(attrs, Backend::CPU);
    const TensorView ins[] = {input, weight};
    op->compute(output, ins, ctx, workspace);
}

void conv3d(const TensorView& input,
            const TensorView& weight,
            const TensorView& bias,
            const TensorView& output,
            const Conv3DAttributes& attrs,
            const ComputeContext& ctx,
            void* workspace)
{
    auto op = Conv3D::create(attrs, Backend::CPU);
    const TensorView ins[] = {input, weight, bias};
    op->compute(output, ins, ctx, workspace);
}

}  // namespace nnops
