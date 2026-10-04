/// @file unary.cpp
/// @brief Unary operator dispatch.

#include "nnops/ops/unary.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

namespace backend::cpu::reference {
    void unary_ref(const UnaryAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

namespace backend::cpu {
    void unary_cpu(const UnaryAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void unary_cuda(const UnaryAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace);
}
#endif

#ifdef NNOPS_HAS_VULKAN
namespace backend::vulkan {
    void unary_vulkan(const UnaryAttributes& attrs,
                       TensorView& output,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace);
}
#endif

struct Unary::Impl {
    using KernelFn = void (*)(const UnaryAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};
Unary::~Unary() = default;


namespace {
auto resolve_unary_kernel(Backend backend) -> Unary::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::unary_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::unary_cuda;
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return backend::vulkan::unary_vulkan;
#endif
    }
    return nullptr;
}
}  // anonymous namespace

std::unique_ptr<Unary> Unary::create(const UnaryAttributes& attrs,
                                       Backend backend)
{
    return std::unique_ptr<Unary>(new Unary(attrs, backend));
}

Unary::Unary(const UnaryAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_unary_kernel(backend);
}

std::vector<TensorDesc> Unary::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return {identity_output_shape(inputs)};
}

void Unary::compute(std::span<TensorView> outputs,
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

}  // namespace nnops
