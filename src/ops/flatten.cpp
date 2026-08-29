/// @file flatten.cpp
/// @brief Flatten operator dispatch.

#include "nnops/ops/flatten.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

namespace backend::cpu::reference {
    void flatten_ref(const FlattenAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);
}

namespace backend::cpu {
    void flatten_cpu(const FlattenAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);
}

// ============================================================
// Impl
// ============================================================
struct Flatten::Impl {
    using KernelFn = void (*)(const FlattenAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_flatten_kernel(Backend backend) -> Flatten::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::flatten_cpu;
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

std::unique_ptr<Flatten> Flatten::create(const FlattenAttributes& attrs,
                                           Backend backend)
{
    return std::unique_ptr<Flatten>(new Flatten(attrs, backend));
}

Flatten::Flatten(const FlattenAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_flatten_kernel(backend);
}

std::vector<TensorDesc> Flatten::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    // Same logical shape, always dense planar layout.
    // Determine planar layout from input rank.
    const auto& in = inputs[0];
    const int64_t rank = in.rank;
    const int64_t srank = rank - 2;

    TensorDesc out;
    out.rank   = rank;
    out.layout = (srank == 3) ? TensorLayout::NCDHW : TensorLayout::NCHW;
    out.dtype  = in.dtype;
    out.dims   = in.dims;

    return {out};
}

void Flatten::compute(std::span<TensorView> outputs,
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

}  // namespace nnops
