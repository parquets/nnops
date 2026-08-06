/// @file clamp.cpp
/// @brief Clamp operator dispatch.

#include "nnops/ops/clamp.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

// Forward declarations of backend kernel entry points
namespace backend::cpu::reference {
    void clamp_ref(const ClampAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

namespace backend::cpu {
    void clamp_cpu(const ClampAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void clamp_cuda(const ClampAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace);
}
#endif

// ============================================================
// Impl
// ============================================================
struct Clamp::Impl {
    using KernelFn = void (*)(const ClampAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_clamp_kernel(Backend backend) -> Clamp::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::clamp_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::clamp_cuda;
#endif
    }
    return nullptr;
}
}  // anonymous namespace

std::unique_ptr<Clamp> Clamp::create(const ClampAttributes& attrs,
                                       Backend backend)
{
    return std::unique_ptr<Clamp>(new Clamp(attrs, backend));
}

Clamp::Clamp(const ClampAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_clamp_kernel(backend);
}

std::vector<TensorDesc> Clamp::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return {identity_output_shape(inputs)};
}

void Clamp::compute(std::span<TensorView> outputs,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 1);
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

// Functional API
void clamp_op(const TensorView& input,
              TensorView& output,
              const ClampAttributes& attrs,
              const ComputeContext& ctx)
{
    auto op = Clamp::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {input};
    op->compute(output, ins, ctx, nullptr);
}

}  // namespace nnops
