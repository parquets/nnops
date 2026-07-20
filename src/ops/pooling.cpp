/// @file pooling.cpp
/// @brief Pooling operator dispatch.

#include "nnops/ops/pooling.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops {

namespace backend::cpu::reference {
    void pooling_ref(const PoolingAttributes& attrs,
                     const TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);
}

std::unique_ptr<Pooling> Pooling::create(const PoolingAttributes& attrs,
                                          Backend backend)
{
    return std::unique_ptr<Pooling>(new Pooling(attrs, backend));
}

Pooling::Pooling(const PoolingAttributes& attrs, Backend backend)
    : attrs_(attrs), backend_(backend)
{
}

void Pooling::compute(const TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 1);
    NNOPS_ASSERT(output.data() != nullptr);
    NNOPS_ASSERT(inputs[0].data() != nullptr);

    switch (backend_) {
    case Backend::CPU:
        backend::cpu::reference::pooling_ref(attrs_, output, inputs, ctx, workspace);
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
void pooling(const TensorView& input,
             const TensorView& output,
             const PoolingAttributes& attrs,
             const ComputeContext& ctx)
{
    auto op = Pooling::create(attrs, Backend::CPU);
    const TensorView ins[] = {input};
    op->compute(output, ins, ctx, nullptr);
}

}  // namespace nnops
