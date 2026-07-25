/// @file softmax.cpp
/// @brief Softmax operator dispatch.

#include "nnops/ops/softmax.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

namespace backend::cpu {
    void softmax_cpu(const SoftmaxAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);
}

namespace backend::cpu::reference {
    void softmax_ref(const SoftmaxAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void softmax_cuda(const SoftmaxAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace);
}
#endif

// ============================================================
// Impl
// ============================================================
struct Softmax::Impl {
    using KernelFn = void (*)(const SoftmaxAttributes&,
                               TensorView&,
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
        return backend::cpu::softmax_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::softmax_cuda;
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

std::vector<TensorDesc> Softmax::getOutputShapes(
    std::span<const TensorDesc> inputs) const
{
    return {identity_output_shape(inputs)};
}

void Softmax::compute(std::span<TensorView> outputs,
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
void softmax(const TensorView& input,
             TensorView& output,
             const SoftmaxAttributes& attrs,
             const ComputeContext& ctx)
{
    auto op = Softmax::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {input};
    op->compute(output, ins, ctx, nullptr);
}

}  // namespace nnops
