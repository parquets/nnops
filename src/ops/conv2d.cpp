/// @file conv2d.cpp
/// @brief Conv2D operator dispatch — connects class API to backend implementations.

#include "nnops/ops/conv2d.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops {

// Forward declarations of backend kernel entry points
namespace backend::cpu::reference {
    void conv2d_ref(const Conv2DAttributes& attrs,
                    const TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

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
    : attrs_(attrs), backend_(backend)
{
}

// ============================================================
// getWorkspace
// ============================================================
size_t Conv2D::getWorkspace() const
{
    // Reference CPU implementation uses no workspace.
    // Optimized GEMM-based implementations may need im2col buffer.
    return 0;
}

// ============================================================
// compute
// ============================================================
void Conv2D::compute(const TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace)
{
    NNOPS_ASSERT(inputs.size() >= 2);
    NNOPS_ASSERT(inputs.size() <= 3);  // input, weight [, bias]
    NNOPS_ASSERT(output.data() != nullptr);
    NNOPS_ASSERT(inputs[0].data() != nullptr);
    NNOPS_ASSERT(inputs[1].data() != nullptr);

    switch (backend_) {
    case Backend::CPU:
        backend::cpu::reference::conv2d_ref(attrs_, output, inputs, ctx, workspace);
        break;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        // backend::cuda::conv2d_cuda(attrs_, output, inputs, ctx, workspace);
        break;
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        // backend::vulkan::conv2d_vulkan(attrs_, output, inputs, ctx, workspace);
        break;
#endif
    }
}

// ============================================================
// Functional API
// ============================================================
void conv2d(const TensorView& input,
            const TensorView& weight,
            const TensorView& output,
            const Conv2DAttributes& attrs,
            const ComputeContext& ctx,
            void* workspace)
{
    auto op = Conv2D::create(attrs, Backend::CPU);
    const TensorView ins[] = {input, weight};
    op->compute(output, ins, ctx, workspace);
}

void conv2d(const TensorView& input,
            const TensorView& weight,
            const TensorView& bias,
            const TensorView& output,
            const Conv2DAttributes& attrs,
            const ComputeContext& ctx,
            void* workspace)
{
    auto op = Conv2D::create(attrs, Backend::CPU);
    const TensorView ins[] = {input, weight, bias};
    op->compute(output, ins, ctx, workspace);
}

}  // namespace nnops
