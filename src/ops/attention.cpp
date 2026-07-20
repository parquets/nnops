/// @file attention.cpp
/// @brief Attention operator dispatch.

#include "nnops/ops/attention.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops {

namespace backend::cpu::reference {
    void attention_ref(const AttentionAttributes& attrs,
                       const TensorView& output,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace);
}

std::unique_ptr<Attention> Attention::create(const AttentionAttributes& attrs,
                                              Backend backend)
{
    return std::unique_ptr<Attention>(new Attention(attrs, backend));
}

Attention::Attention(const AttentionAttributes& attrs, Backend backend)
    : attrs_(attrs), backend_(backend)
{
}

size_t Attention::getWorkspace() const
{
    // Reference implementation uses a per-head scratch buffer for scores:
    // max size = Sq * Sk * sizeof(float)
    // We don't know the shapes here, so return 0 — user must provide
    // sufficient workspace or the implementation allocates internally.
    // For CPU reference, we use std::vector internally.
    return 0;
}

void Attention::compute(const TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         void* workspace)
{
    NNOPS_ASSERT(inputs.size() >= 3);
    NNOPS_ASSERT(inputs.size() <= 4);
    NNOPS_ASSERT(output.data() != nullptr);
    NNOPS_ASSERT(inputs[0].data() != nullptr);
    NNOPS_ASSERT(inputs[1].data() != nullptr);
    NNOPS_ASSERT(inputs[2].data() != nullptr);

    switch (backend_) {
    case Backend::CPU:
        backend::cpu::reference::attention_ref(attrs_, output, inputs, ctx, workspace);
        break;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        break;
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        break;
#endif
    }
}

// Functional API (without mask)
void attention(const TensorView& query,
               const TensorView& key,
               const TensorView& value,
               const TensorView& output,
               const AttentionAttributes& attrs,
               const ComputeContext& ctx,
               void* workspace)
{
    auto op = Attention::create(attrs, Backend::CPU);
    const TensorView ins[] = {query, key, value};
    op->compute(output, ins, ctx, workspace);
}

// Functional API (with mask)
void attention(const TensorView& query,
               const TensorView& key,
               const TensorView& value,
               const TensorView& mask,
               const TensorView& output,
               const AttentionAttributes& attrs,
               const ComputeContext& ctx,
               void* workspace)
{
    auto op = Attention::create(attrs, Backend::CPU);
    const TensorView ins[] = {query, key, value, mask};
    op->compute(output, ins, ctx, workspace);
}

}  // namespace nnops
