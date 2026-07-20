/// @file linear.cpp
/// @brief Linear operator dispatch.

#include "nnops/ops/linear.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops {

namespace backend::cpu::reference {
    void linear_ref(const TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

std::unique_ptr<Linear> Linear::create(Backend backend)
{
    return std::unique_ptr<Linear>(new Linear(backend));
}

Linear::Linear(Backend backend)
    : backend_(backend)
{
}

void Linear::compute(const TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace)
{
    NNOPS_ASSERT(inputs.size() >= 2);
    NNOPS_ASSERT(inputs.size() <= 3);
    NNOPS_ASSERT(output.data() != nullptr);
    NNOPS_ASSERT(inputs[0].data() != nullptr);
    NNOPS_ASSERT(inputs[1].data() != nullptr);

    switch (backend_) {
    case Backend::CPU:
        backend::cpu::reference::linear_ref(output, inputs, ctx, workspace);
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
void linear(const TensorView& input,
            const TensorView& weight,
            const TensorView& output,
            const ComputeContext& ctx)
{
    auto op = Linear::create(Backend::CPU);
    const TensorView ins[] = {input, weight};
    op->compute(output, ins, ctx, nullptr);
}

void linear(const TensorView& input,
            const TensorView& weight,
            const TensorView& bias,
            const TensorView& output,
            const ComputeContext& ctx)
{
    auto op = Linear::create(Backend::CPU);
    const TensorView ins[] = {input, weight, bias};
    op->compute(output, ins, ctx, nullptr);
}

}  // namespace nnops
