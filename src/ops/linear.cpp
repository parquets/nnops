/// @file linear.cpp
/// @brief Linear operator dispatch.

#include "nnops/ops/linear.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops {

namespace backend::cpu::reference {
    void linear_ref(const LinearAttributes& attrs,
                    const TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

// ============================================================
// Impl — holds the backend-bound kernel function pointer
// ============================================================
struct Linear::Impl {
    using KernelFn = void (*)(const LinearAttributes&,
                               const TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_linear_kernel(Backend backend) -> Linear::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::reference::linear_ref;
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

std::unique_ptr<Linear> Linear::create(const LinearAttributes& attrs,
                                        Backend backend)
{
    return std::unique_ptr<Linear>(new Linear(attrs, backend));
}

Linear::Linear(const LinearAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_linear_kernel(backend);
}

void Linear::compute(std::span<const TensorView> outputs,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace)
{
    NNOPS_ASSERT(inputs.size() >= 2);
    NNOPS_ASSERT(inputs.size() <= 3);
    NNOPS_ASSERT(outputs.size() == 1);
    const auto& output = outputs[0];
    NNOPS_ASSERT(output.data() != nullptr);
    NNOPS_ASSERT(inputs[0].data() != nullptr);
    NNOPS_ASSERT(inputs[1].data() != nullptr);

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

// Functional API — no attributes (backward-compatible)
void linear(const TensorView& input,
            const TensorView& weight,
            const TensorView& output,
            const ComputeContext& ctx)
{
    auto op = Linear::create({}, ctx.expected_backend);
    const TensorView ins[] = {input, weight};
    op->compute(output, ins, ctx, nullptr);
}

void linear(const TensorView& input,
            const TensorView& weight,
            const TensorView& bias,
            const TensorView& output,
            const ComputeContext& ctx)
{
    auto op = Linear::create({}, ctx.expected_backend);
    const TensorView ins[] = {input, weight, bias};
    op->compute(output, ins, ctx, nullptr);
}

// Functional API — with attributes (epilogue support)
void linear(const TensorView& input,
            const TensorView& weight,
            const TensorView& output,
            const LinearAttributes& attrs,
            const ComputeContext& ctx)
{
    auto op = Linear::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {input, weight};
    op->compute(output, ins, ctx, nullptr);
}

void linear(const TensorView& input,
            const TensorView& weight,
            const TensorView& bias,
            const TensorView& output,
            const LinearAttributes& attrs,
            const ComputeContext& ctx)
{
    auto op = Linear::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {input, weight, bias};
    op->compute(output, ins, ctx, nullptr);
}

}  // namespace nnops
