/// @file topk.cpp
/// @brief TopK operator dispatch.

#include "nnops/ops/topk.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

// Forward declarations of backend kernel entry points
namespace backend::cpu::reference {
    void topk_ref(const TopKAttributes& attrs,
                  TensorView& values,
                  TensorView& indices,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx,
                  void* workspace);
}

namespace backend::cpu {
    void topk_cpu(const TopKAttributes& attrs,
                  TensorView& values,
                  TensorView& indices,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx,
                  void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void topk_cuda(const TopKAttributes& attrs,
                   TensorView& values,
                   TensorView& indices,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx,
                   void* workspace);
}
#endif

// ============================================================
// Impl
// ============================================================
struct TopK::Impl {
    using KernelFn = void (*)(const TopKAttributes&,
                               TensorView&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_topk_kernel(Backend backend) -> TopK::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::topk_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::topk_cuda;
#endif
    }
    return nullptr;
}
}  // anonymous namespace

std::unique_ptr<TopK> TopK::create(const TopKAttributes& attrs,
                                     Backend backend)
{
    return std::unique_ptr<TopK>(new TopK(attrs, backend));
}

TopK::TopK(const TopKAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_topk_kernel(backend);
}

std::vector<TensorDesc> TopK::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return topk_output_shape(attrs_.axis, attrs_.k, inputs);
}

void TopK::compute(std::span<TensorView> outputs,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 1);
    NNOPS_ASSERT(outputs.size() == 2);
    auto& values  = outputs[0];
    auto& indices = outputs[1];
    NNOPS_ASSERT(!values.is_empty());
    NNOPS_ASSERT(!indices.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());

    impl_->kernel_fn(attrs_, values, indices, inputs, ctx, workspace);
}

// Functional API
void topk(const TensorView& input,
           TensorView& values,
           TensorView& indices,
           const TopKAttributes& attrs,
           const ComputeContext& ctx)
{
    auto op = TopK::create(attrs, ctx.expected_backend);
    TensorView outs[] = {values, indices};
    const TensorView ins[] = {input};
    op->compute(outs, ins, ctx, nullptr);
}

}  // namespace nnops
