/// @file activation.cpp
/// @brief Activation operator dispatch.

#include "nnops/ops/activation.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops {

namespace backend::cpu::reference {
    void activation_ref(const ActivationAttributes& attrs,
                        const TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx,
                        void* workspace);
}

std::unique_ptr<Activation> Activation::create(const ActivationAttributes& attrs,
                                                Backend backend)
{
    return std::unique_ptr<Activation>(new Activation(attrs, backend));
}

Activation::Activation(const ActivationAttributes& attrs, Backend backend)
    : attrs_(attrs), backend_(backend)
{
}

void Activation::compute(const TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 1);
    NNOPS_ASSERT(output.data() != nullptr);
    NNOPS_ASSERT(inputs[0].data() != nullptr);

    switch (backend_) {
    case Backend::CPU:
        backend::cpu::reference::activation_ref(attrs_, output, inputs, ctx, workspace);
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

// Functional API
void activation(const TensorView& input,
                const TensorView& output,
                const ActivationAttributes& attrs,
                const ComputeContext& ctx)
{
    auto op = Activation::create(attrs, Backend::CPU);
    const TensorView ins[] = {input};
    op->compute(output, ins, ctx, nullptr);
}

}  // namespace nnops
