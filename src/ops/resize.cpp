/// @file resize.cpp
/// @brief Resize operator dispatch.

#include "nnops/ops/resize.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

namespace backend::cpu::reference {
    void resize_ref(const ResizeAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

namespace backend::cpu {
    void resize_cpu(const ResizeAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

// ============================================================
// Impl
// ============================================================
struct Resize::Impl {
    using KernelFn = void (*)(const ResizeAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_resize_kernel(Backend backend) -> Resize::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::resize_cpu;
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

std::unique_ptr<Resize> Resize::create(const ResizeAttributes& attrs,
                                        Backend backend)
{
    return std::unique_ptr<Resize>(new Resize(attrs, backend));
}

Resize::Resize(const ResizeAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_resize_kernel(backend);
}

std::vector<TensorDesc> Resize::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    std::array<int64_t, 3> os = attrs_.output_size;
    return {resize_output_shape(
        std::span<const int64_t>(os.data(), os.size()), inputs)};
}

void Resize::compute(std::span<TensorView> outputs,
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
void resize(const TensorView& input,
            TensorView& output,
            const ResizeAttributes& attrs,
            const ComputeContext& ctx)
{
    auto op = Resize::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {input};
    op->compute(output, ins, ctx, nullptr);
}

}  // namespace nnops
