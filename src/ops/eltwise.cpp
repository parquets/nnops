/// @file eltwise.cpp
/// @brief Eltwise operator dispatch.

#include "nnops/ops/eltwise.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops {

// Forward declarations of backend kernel entry points
namespace backend::cpu::reference {
    void eltwise_ref(const EltwiseAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace);
}

namespace backend::cpu {
    void eltwise_cpu(const EltwiseAttributes& attrs,
                       TensorView& output,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void eltwise_cuda(const EltwiseAttributes& attrs,
                        TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx,
                        void* workspace);
}
#endif

#ifdef NNOPS_HAS_VULKAN
namespace backend::vulkan {
    void eltwise_vulkan(const EltwiseAttributes& attrs,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         void* workspace);
}
#endif

// ============================================================
// Impl
// ============================================================
struct Eltwise::Impl {
    using KernelFn = void (*)(const EltwiseAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_eltwise_kernel(Backend backend) -> Eltwise::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::eltwise_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::eltwise_cuda;
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return backend::vulkan::eltwise_vulkan;
#endif
    }
    return nullptr;
}
}  // anonymous namespace

std::unique_ptr<Eltwise> Eltwise::create(const EltwiseAttributes& attrs,
                                           Backend backend)
{
    return std::unique_ptr<Eltwise>(new Eltwise(attrs, backend));
}

Eltwise::Eltwise(const EltwiseAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_eltwise_kernel(backend);
}

void Eltwise::compute(std::span<TensorView> outputs,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx,
                        void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 2);
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());
    NNOPS_ASSERT(!inputs[1].is_empty());

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

// Functional API
void eltwise(const TensorView& a,
              const TensorView& b,
              TensorView& output,
              const EltwiseAttributes& attrs,
              const ComputeContext& ctx)
{
    auto op = Eltwise::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {a, b};
    op->compute(output, ins, ctx, nullptr);
}

}  // namespace nnops
