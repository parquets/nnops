/// @file pooling.cpp
/// @brief Pooling operator dispatch.

#include "nnops/ops/pooling.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops {

namespace backend::cpu::reference {
    void pooling_ref(const PoolingAttributes& attrs,
                     const TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);
}

namespace backend::cpu {
    void pooling_cpu(const PoolingAttributes& attrs,
                      const TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace);
}

// ============================================================
// Impl
// ============================================================
struct Pooling::Impl {
    using KernelFn = void (*)(const PoolingAttributes&,
                               const TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_pooling_kernel(Backend backend) -> Pooling::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::pooling_cpu;
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

std::unique_ptr<Pooling> Pooling::create(const PoolingAttributes& attrs,
                                          Backend backend)
{
    return std::unique_ptr<Pooling>(new Pooling(attrs, backend));
}

Pooling::Pooling(const PoolingAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_pooling_kernel(backend);
}

void Pooling::compute(const TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 1);
    NNOPS_ASSERT(output.data() != nullptr);
    NNOPS_ASSERT(inputs[0].data() != nullptr);

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

// Functional API
void pooling(const TensorView& input,
             const TensorView& output,
             const PoolingAttributes& attrs,
             const ComputeContext& ctx)
{
    auto op = Pooling::create(attrs, Backend::CPU);
    const TensorView ins[] = {input};
    op->compute(output, ins, ctx, nullptr);
}

}  // namespace nnops
