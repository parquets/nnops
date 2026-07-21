/// @file activation.cpp
/// @brief Activation operator dispatch.

#include "nnops/ops/activation.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops {

namespace backend::cpu {
    void activation_cpu(const ActivationAttributes& attrs,
                         const TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         void* workspace);
}

namespace backend::cpu::reference {
    void activation_ref(const ActivationAttributes& attrs,
                        const TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx,
                        void* workspace);
}

// ============================================================
// Impl
// ============================================================
struct Activation::Impl {
    using KernelFn = void (*)(const ActivationAttributes&,
                               const TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_activation_kernel(Backend backend) -> Activation::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::activation_cpu;
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

std::unique_ptr<Activation> Activation::create(const ActivationAttributes& attrs,
                                                Backend backend)
{
    return std::unique_ptr<Activation>(new Activation(attrs, backend));
}

Activation::Activation(const ActivationAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_activation_kernel(backend);
}

void Activation::compute(std::span<const TensorView> outputs,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 1);
    NNOPS_ASSERT(outputs.size() == 1);
    const auto& output = outputs[0];
    NNOPS_ASSERT(output.data() != nullptr);
    NNOPS_ASSERT(inputs[0].data() != nullptr);

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

// Functional API
void activation(const TensorView& input,
                const TensorView& output,
                const ActivationAttributes& attrs,
                const ComputeContext& ctx)
{
    auto op = Activation::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {input};
    op->compute(output, ins, ctx, nullptr);
}

}  // namespace nnops
