/// @file batch_norm.cpp
/// @brief BatchNorm operator dispatch.

#include "nnops/ops/batch_norm.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops {

namespace backend::cpu::reference {
    void batch_norm_ref(const BatchNormAttributes& attrs,
                        const TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx,
                        void* workspace);
}

namespace backend::cpu {
    void batch_norm_cpu(const BatchNormAttributes& attrs,
                        const TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx,
                        void* workspace);
}

// ============================================================
// Impl
// ============================================================
struct BatchNorm::Impl {
    using KernelFn = void (*)(const BatchNormAttributes&,
                               const TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_batch_norm_kernel(Backend backend) -> BatchNorm::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::batch_norm_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return nullptr;
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return nullptr;
#endif
    }
    return nullptr;
}
}  // anonymous namespace

std::unique_ptr<BatchNorm> BatchNorm::create(const BatchNormAttributes& attrs,
                                              Backend backend)
{
    return std::unique_ptr<BatchNorm>(new BatchNorm(attrs, backend));
}

BatchNorm::BatchNorm(const BatchNormAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_batch_norm_kernel(backend);
}

void BatchNorm::compute(std::span<const TensorView> outputs,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 5);
    NNOPS_ASSERT(outputs.size() == 1);
    const auto& output = outputs[0];
    NNOPS_ASSERT(output.data() != nullptr);
    NNOPS_ASSERT(inputs[0].data() != nullptr);  // X
    NNOPS_ASSERT(inputs[1].data() != nullptr);  // scale
    NNOPS_ASSERT(inputs[2].data() != nullptr);  // bias
    NNOPS_ASSERT(inputs[3].data() != nullptr);  // mean
    NNOPS_ASSERT(inputs[4].data() != nullptr);  // var

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

// Functional API
void batch_norm(const TensorView& x,
                const TensorView& scale,
                const TensorView& bias,
                const TensorView& mean,
                const TensorView& var,
                const TensorView& output,
                const BatchNormAttributes& attrs,
                const ComputeContext& ctx)
{
    auto op = BatchNorm::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {x, scale, bias, mean, var};
    op->compute(output, ins, ctx, nullptr);
}

}  // namespace nnops
