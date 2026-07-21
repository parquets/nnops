/// @file softmax.cpp
/// @brief Softmax operator dispatch.

#include "nnops/ops/softmax.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops {

namespace backend::cpu::reference {
    void softmax_ref(const SoftmaxAttributes& attrs,
                     const TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);
}

// ============================================================
// Impl
// ============================================================
struct Softmax::Impl {
    using KernelFn = void (*)(const SoftmaxAttributes&,
                               const TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_softmax_kernel(Backend backend) -> Softmax::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::reference::softmax_ref;
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

std::unique_ptr<Softmax> Softmax::create(const SoftmaxAttributes& attrs,
                                          Backend backend)
{
    return std::unique_ptr<Softmax>(new Softmax(attrs, backend));
}

Softmax::Softmax(const SoftmaxAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_softmax_kernel(backend);
}

void Softmax::compute(std::span<const TensorView> outputs,
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
void softmax(const TensorView& input,
             const TensorView& output,
             const SoftmaxAttributes& attrs,
             const ComputeContext& ctx)
{
    auto op = Softmax::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {input};
    op->compute(output, ins, ctx, nullptr);
}

}  // namespace nnops
