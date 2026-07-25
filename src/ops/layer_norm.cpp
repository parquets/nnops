/// @file layer_norm.cpp
/// @brief LayerNorm operator dispatch.

#include "nnops/ops/layer_norm.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops {

namespace backend::cpu::reference {
    void layer_norm_ref(const LayerNormAttributes& attrs,
                        TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx,
                        void* workspace);
}

namespace backend::cpu {
    void layer_norm_cpu(const LayerNormAttributes& attrs,
                        TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx,
                        void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void layer_norm_cuda(const LayerNormAttributes& attrs,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         void* workspace);
}
#endif

// ============================================================
// Impl
// ============================================================
struct LayerNorm::Impl {
    using KernelFn = void (*)(const LayerNormAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_layer_norm_kernel(Backend backend) -> LayerNorm::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::layer_norm_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::layer_norm_cuda;
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return nullptr;
#endif
    }
    return nullptr;
}
}  // anonymous namespace

std::unique_ptr<LayerNorm> LayerNorm::create(const LayerNormAttributes& attrs,
                                              Backend backend)
{
    return std::unique_ptr<LayerNorm>(new LayerNorm(attrs, backend));
}

LayerNorm::LayerNorm(const LayerNormAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_layer_norm_kernel(backend);
}

void LayerNorm::compute(std::span<TensorView> outputs,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         void* workspace)
{
    NNOPS_ASSERT(inputs.size() >= 2);
    NNOPS_ASSERT(inputs.size() <= 3);  // x, scale [, bias]
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());  // X
    NNOPS_ASSERT(!inputs[1].is_empty());  // scale

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

// Functional API (without bias)
void layer_norm(const TensorView& x,
                const TensorView& scale,
                TensorView& output,
                const LayerNormAttributes& attrs,
                const ComputeContext& ctx)
{
    auto op = LayerNorm::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {x, scale};
    op->compute(output, ins, ctx, nullptr);
}

// Functional API (with bias)
void layer_norm(const TensorView& x,
                const TensorView& scale,
                const TensorView& bias,
                TensorView& output,
                const LayerNormAttributes& attrs,
                const ComputeContext& ctx)
{
    auto op = LayerNorm::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {x, scale, bias};
    op->compute(output, ins, ctx, nullptr);
}

}  // namespace nnops
