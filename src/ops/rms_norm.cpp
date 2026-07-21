/// @file rms_norm.cpp
/// @brief RMSNorm operator dispatch.

#include "nnops/ops/rms_norm.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops {

namespace backend::cpu::reference {
    void rms_norm_ref(const RMSNormAttributes& attrs,
                      const TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace);
}

// ============================================================
// Impl
// ============================================================
struct RMSNorm::Impl {
    using KernelFn = void (*)(const RMSNormAttributes&,
                               const TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_rms_norm_kernel(Backend backend) -> RMSNorm::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::reference::rms_norm_ref;
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

std::unique_ptr<RMSNorm> RMSNorm::create(const RMSNormAttributes& attrs,
                                          Backend backend)
{
    return std::unique_ptr<RMSNorm>(new RMSNorm(attrs, backend));
}

RMSNorm::RMSNorm(const RMSNormAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_rms_norm_kernel(backend);
}

void RMSNorm::compute(std::span<const TensorView> outputs,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 2);  // x, scale
    NNOPS_ASSERT(outputs.size() == 1);
    const auto& output = outputs[0];
    NNOPS_ASSERT(output.data() != nullptr);
    NNOPS_ASSERT(inputs[0].data() != nullptr);  // X
    NNOPS_ASSERT(inputs[1].data() != nullptr);  // scale

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

// Functional API
void rms_norm(const TensorView& x,
              const TensorView& scale,
              const TensorView& output,
              const RMSNormAttributes& attrs,
              const ComputeContext& ctx)
{
    auto op = RMSNorm::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {x, scale};
    op->compute(output, ins, ctx, nullptr);
}

}  // namespace nnops
