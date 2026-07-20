/// @file linear.cpp
/// @brief Linear operator dispatch.

#include "nnops/ops/linear.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops {

namespace backend::cpu::reference {
    void linear_ref(const TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

// ============================================================
// Impl — Linear has no attributes, so KernelFn is simpler
// ============================================================
struct Linear::Impl {
    using KernelFn = void (*)(const TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_linear_kernel(Backend backend) -> Linear::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::reference::linear_ref;
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

std::unique_ptr<Linear> Linear::create(Backend backend)
{
    return std::unique_ptr<Linear>(new Linear(backend));
}

Linear::Linear(Backend backend)
    : impl_(std::make_unique<Impl>()), backend_(backend)
{
    impl_->kernel_fn = resolve_linear_kernel(backend);
}

void Linear::compute(const TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace)
{
    NNOPS_ASSERT(inputs.size() >= 2);
    NNOPS_ASSERT(inputs.size() <= 3);
    NNOPS_ASSERT(output.data() != nullptr);
    NNOPS_ASSERT(inputs[0].data() != nullptr);
    NNOPS_ASSERT(inputs[1].data() != nullptr);

    impl_->kernel_fn(output, inputs, ctx, workspace);
}

// Functional API
void linear(const TensorView& input,
            const TensorView& weight,
            const TensorView& output,
            const ComputeContext& ctx)
{
    auto op = Linear::create(Backend::CPU);
    const TensorView ins[] = {input, weight};
    op->compute(output, ins, ctx, nullptr);
}

void linear(const TensorView& input,
            const TensorView& weight,
            const TensorView& bias,
            const TensorView& output,
            const ComputeContext& ctx)
{
    auto op = Linear::create(Backend::CPU);
    const TensorView ins[] = {input, weight, bias};
    op->compute(output, ins, ctx, nullptr);
}

}  // namespace nnops
