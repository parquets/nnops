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

// ============================================================
// Impl
// ============================================================
struct MatMul::Impl {
    using KernelFn = void (*)(const MatMulAttributes&,
                               const TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_matmul_kernel(Backend backend) -> MatMul::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::reference::matmul_ref;
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

std::unique_ptr<MatMul> MatMul::create(const MatMulAttributes& attrs,
                                        Backend backend)
{
    return std::unique_ptr<MatMul>(new MatMul(attrs, backend));
}

MatMul::MatMul(const MatMulAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_matmul_kernel(backend);
}

void MatMul::compute(std::span<const TensorView> outputs,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 2);
    NNOPS_ASSERT(outputs.size() == 1);
    const auto& output = outputs[0];
    NNOPS_ASSERT(output.data() != nullptr);
    NNOPS_ASSERT(inputs[0].data() != nullptr);
    NNOPS_ASSERT(inputs[1].data() != nullptr);

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
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
