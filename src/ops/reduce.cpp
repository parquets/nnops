/// @file reduce.cpp
/// @brief Reduce operator dispatch.

#include "nnops/ops/reduce.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

namespace backend::cpu {
    void reduce_cpu(const ReduceAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

namespace backend::cpu::reference {
    void reduce_ref(const ReduceAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void reduce_cuda(const ReduceAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);
}
#endif

// ============================================================
// Impl
// ============================================================
struct Reduce::Impl {
    using KernelFn = void (*)(const ReduceAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_reduce_kernel(Backend backend) -> Reduce::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::reduce_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::reduce_cuda;
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return nullptr;
#endif
    }
    return nullptr;
}
}  // anonymous namespace

std::unique_ptr<Reduce> Reduce::create(const ReduceAttributes& attrs,
                                        Backend backend)
{
    return std::unique_ptr<Reduce>(new Reduce(attrs, backend));
}

Reduce::Reduce(const ReduceAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_reduce_kernel(backend);
}

std::vector<TensorDesc> Reduce::getOutputShapes(
    std::span<const TensorDesc> inputs) const
{
    return {reduce_output_shape(attrs_.axis, attrs_.keepdims, inputs)};
}

void Reduce::compute(std::span<TensorView> outputs,
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

// Functional API
void reduce(const TensorView& input,
            TensorView& output,
            const ReduceAttributes& attrs,
            const ComputeContext& ctx)
{
    auto op = Reduce::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {input};
    op->compute(output, ins, ctx, nullptr);
}

}  // namespace nnops
