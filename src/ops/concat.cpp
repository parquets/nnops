/// @file concat.cpp
/// @brief Concat operator dispatch.

#include "nnops/ops/concat.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

namespace backend::cpu::reference {
    void concat_ref(const ConcatAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

namespace backend::cpu {
    void concat_cpu(const ConcatAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

// ============================================================
// Impl
// ============================================================
struct Concat::Impl {
    using KernelFn = void (*)(const ConcatAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_concat_kernel(Backend backend) -> Concat::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::concat_cpu;
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

std::unique_ptr<Concat> Concat::create(const ConcatAttributes& attrs,
                                         Backend backend)
{
    return std::unique_ptr<Concat>(new Concat(attrs, backend));
}

Concat::Concat(const ConcatAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_concat_kernel(backend);
}

std::vector<TensorDesc> Concat::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return concat_output_shape(attrs_.axis, inputs);
}

void Concat::compute(std::span<TensorView> outputs,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace)
{
    NNOPS_ASSERT(inputs.size() >= 2);
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

// Functional API
void concat(std::span<const TensorView> inputs,
            TensorView& output,
            const ConcatAttributes& attrs,
            const ComputeContext& ctx)
{
    auto op = Concat::create(attrs, ctx.expected_backend);
    op->compute(output, inputs, ctx, nullptr);
}

}  // namespace nnops
