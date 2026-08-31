/// @file attention.cpp
/// @brief Attention operator dispatch.

#include "nnops/ops/attention.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "backend/cpu/attention.h"

namespace nnops {

namespace backend::cpu::reference {
    void attention_ref(const AttentionAttributes& attrs,
                       TensorView& output,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace);
}

// ============================================================
// Impl
// ============================================================
struct Attention::Impl {
    using KernelFn = void (*)(const AttentionAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_attention_kernel(Backend backend) -> Attention::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::attention_kernel;
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

std::unique_ptr<Attention> Attention::create(const AttentionAttributes& attrs,
                                              Backend backend)
{
    return std::unique_ptr<Attention>(new Attention(attrs, backend));
}

Attention::Attention(const AttentionAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_attention_kernel(backend);
}

std::vector<TensorDesc> Attention::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return {attention_output_shape(inputs)};
}

size_t Attention::getWorkspaceSize(std::span<const TensorDesc> inputs,
                                   std::span<const TensorDesc> outputs) const
{
    NNOPS_ASSERT(inputs.size() >= 3);
    NNOPS_ASSERT(inputs.size() <= 4);
    NNOPS_ASSERT(outputs.size() == 1);
    return backend::cpu::attention_get_workspace_size(attrs_, inputs, outputs);
}

void Attention::compute(std::span<TensorView> outputs,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         void* workspace)
{
    NNOPS_ASSERT(inputs.size() >= 3);
    NNOPS_ASSERT(inputs.size() <= 4);
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());
    NNOPS_ASSERT(!inputs[1].is_empty());
    NNOPS_ASSERT(!inputs[2].is_empty());
    NNOPS_ASSERT(is_layout_supported(inputs[0].layout(), LayoutSupport::PlanarOnly));

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
