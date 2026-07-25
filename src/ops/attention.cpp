/// @file attention.cpp
/// @brief Attention operator dispatch.

#include "nnops/ops/attention.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

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
        return backend::cpu::reference::attention_ref;
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

std::vector<TensorDesc> Attention::getOutputShapes(
    std::span<const TensorDesc> inputs) const
{
    return {attention_output_shape(inputs)};
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

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

// Functional API (without mask)
void attention(const TensorView& query,
               const TensorView& key,
               const TensorView& value,
               TensorView& output,
               const AttentionAttributes& attrs,
               const ComputeContext& ctx,
               void* workspace)
{
    auto op = Attention::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {query, key, value};
    op->compute(output, ins, ctx, workspace);
}

// Functional API (with mask)
void attention(const TensorView& query,
               const TensorView& key,
               const TensorView& value,
               const TensorView& mask,
               TensorView& output,
               const AttentionAttributes& attrs,
               const ComputeContext& ctx,
               void* workspace)
{
    auto op = Attention::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {query, key, value, mask};
    op->compute(output, ins, ctx, workspace);
}

}  // namespace nnops
