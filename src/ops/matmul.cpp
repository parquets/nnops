/// @file matmul.cpp
/// @brief MatMul operator dispatch.

#include "nnops/ops/matmul.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops {

namespace backend::cpu::reference {
    void matmul_ref(const MatMulAttributes& attrs,
                    const TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

std::unique_ptr<MatMul> MatMul::create(const MatMulAttributes& attrs,
                                        Backend backend)
{
    return std::unique_ptr<MatMul>(new MatMul(attrs, backend));
}

MatMul::MatMul(const MatMulAttributes& attrs, Backend backend)
    : attrs_(attrs), backend_(backend)
{
}

void MatMul::compute(const TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 2);
    NNOPS_ASSERT(output.data() != nullptr);
    NNOPS_ASSERT(inputs[0].data() != nullptr);
    NNOPS_ASSERT(inputs[1].data() != nullptr);

    switch (backend_) {
    case Backend::CPU:
        backend::cpu::reference::matmul_ref(attrs_, output, inputs, ctx, workspace);
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
void matmul(const TensorView& a,
            const TensorView& b,
            const TensorView& c,
            const MatMulAttributes& attrs,
            const ComputeContext& ctx)
{
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorView ins[] = {a, b};
    op->compute(c, ins, ctx, nullptr);
}

}  // namespace nnops
