/// @file cumsum.cpp
/// @brief CumSum operator dispatch.

#include "nnops/ops/cumsum.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

namespace backend::cpu::reference {
    void cumsum_ref(const CumSumAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void cumsum_cuda(const CumSumAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);
}
#endif

// ============================================================
// Impl
// ============================================================
struct CumSum::Impl {
    using KernelFn = void (*)(const CumSumAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_cumsum_kernel(Backend backend) -> CumSum::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::reference::cumsum_ref;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::cumsum_cuda;
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return nullptr;
#endif
    }
    return nullptr;
}
}  // anonymous namespace

std::unique_ptr<CumSum> CumSum::create(const CumSumAttributes& attrs,
                                        Backend backend)
{
    return std::unique_ptr<CumSum>(new CumSum(attrs, backend));
}

CumSum::CumSum(const CumSumAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_cumsum_kernel(backend);
}

std::vector<TensorDesc> CumSum::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return {identity_output_shape(inputs)};
}

void CumSum::compute(std::span<TensorView> outputs,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 1);
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());
    NNOPS_ASSERT(is_layout_supported(inputs[0].layout(), LayoutSupport::PlanarOnly));

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
