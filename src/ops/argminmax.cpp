/// @file argminmax.cpp
/// @brief ArgMax / ArgMin operator dispatch.

#include "nnops/ops/argminmax.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

namespace backend::cpu::reference {
    void argmax_ref(const ArgMinMaxAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
    void argmin_ref(const ArgMinMaxAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

namespace backend::cpu {
    void argmax_cpu(const ArgMinMaxAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
    void argmin_cpu(const ArgMinMaxAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

struct ArgMax::Impl {
    using KernelFn = void (*)(const ArgMinMaxAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};
ArgMax::~ArgMax() = default;


struct ArgMin::Impl {
    using KernelFn = void (*)(const ArgMinMaxAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};
ArgMin::~ArgMin() = default;


namespace {
auto resolve_argmax_kernel(Backend backend) -> ArgMax::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::argmax_cpu;
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

auto resolve_argmin_kernel(Backend backend) -> ArgMin::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::argmin_cpu;
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

std::unique_ptr<ArgMax> ArgMax::create(const ArgMinMaxAttributes& attrs,
                                         Backend backend)
{
    return std::unique_ptr<ArgMax>(new ArgMax(attrs, backend));
}

ArgMax::ArgMax(const ArgMinMaxAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_argmax_kernel(backend);
}

std::vector<TensorDesc> ArgMax::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return argminmax_output_shape(attrs_.axis, attrs_.keepdims, inputs);
}

void ArgMax::compute(std::span<TensorView> outputs,
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

std::unique_ptr<ArgMin> ArgMin::create(const ArgMinMaxAttributes& attrs,
                                         Backend backend)
{
    return std::unique_ptr<ArgMin>(new ArgMin(attrs, backend));
}

ArgMin::ArgMin(const ArgMinMaxAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_argmin_kernel(backend);
}

std::vector<TensorDesc> ArgMin::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return argminmax_output_shape(attrs_.axis, attrs_.keepdims, inputs);
}

void ArgMin::compute(std::span<TensorView> outputs,
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
