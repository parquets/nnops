/// @file pooling.cpp
/// @brief Pooling operator dispatch — input/output always NCHWC8/NCDHWC8, direct pass-through.

#include "nnops/ops/pooling.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

namespace backend::cpu {
    void pooling_cpu(const PoolingAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void pooling_cuda(const PoolingAttributes& attrs,
                       TensorView& output,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace);
}
#endif

struct Pooling::Impl {
    using KernelFn = void (*)(const PoolingAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};
Pooling::~Pooling() = default;


namespace {
auto resolve_pooling_kernel(Backend backend) -> Pooling::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::pooling_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::pooling_cuda;
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return nullptr;
#endif
    default:
        return nullptr;
    }
}
}  // anonymous namespace

std::unique_ptr<Pooling> Pooling::create(const PoolingAttributes& attrs,
                                          Backend backend)
{
    return std::unique_ptr<Pooling>(new Pooling(attrs, backend));
}

Pooling::Pooling(const PoolingAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_pooling_kernel(backend);
}

std::vector<TensorDesc> Pooling::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return {pooling_output_shape(
        attrs_.kernel_shape, attrs_.stride, attrs_.dilation, attrs_.padding,
        static_cast<int>(attrs_.auto_pad), inputs)};
}

void Pooling::compute(std::span<TensorView> outputs,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace)
{
    (void)workspace;
    NNOPS_ASSERT(inputs.size() == 1);
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    auto& input = inputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!input.is_empty());
    NNOPS_ASSERT(is_layout_supported(input.layout(), LayoutSupport::PackedOnly));

    impl_->kernel_fn(attrs_, output, inputs, ctx, nullptr);
}

}  // namespace nnops
