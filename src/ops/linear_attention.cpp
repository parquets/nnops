/// @file linear_attention.cpp
/// @brief LinearAttention operator dispatch.

#include "nnops/ops/linear_attention.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops {

namespace backend::cpu {
    void linear_attention_cpu(const LinearAttentionAttributes& attrs,
                              TensorView& output,
                              std::span<const TensorView> inputs,
                              const ComputeContext& ctx,
                              void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void linear_attention_cuda(const LinearAttentionAttributes& attrs,
                               TensorView& output,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* workspace);
}
#endif

// ============================================================
// Impl
// ============================================================
struct LinearAttention::Impl {
    using KernelFn = void (*)(const LinearAttentionAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};
LinearAttention::~LinearAttention() = default;

namespace {
auto resolve_linear_attention_kernel(Backend backend) -> LinearAttention::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::linear_attention_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::linear_attention_cuda;
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return nullptr;
#endif
    }
    return nullptr;
}
}  // anonymous namespace

std::unique_ptr<LinearAttention> LinearAttention::create(
    const LinearAttentionAttributes& attrs, Backend backend)
{
    return std::unique_ptr<LinearAttention>(new LinearAttention(attrs, backend));
}

LinearAttention::LinearAttention(const LinearAttentionAttributes& attrs,
                                   Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_linear_attention_kernel(backend);
}

std::vector<TensorDesc> LinearAttention::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    NNOPS_ASSERT(inputs.size() == 4);
    // outputs[0] = same shape as Q
    // outputs[1] = State shape: [B, H_kv, D, D]
    //   D = head_dim from attributes (not derivable from Q alone when H_kv != H)
    const auto& Q = inputs[0];
    int64_t B  = Q.dims[0];
    int64_t H_kv = attrs_.num_kv_heads > 0 ? attrs_.num_kv_heads : attrs_.num_heads;
    int64_t D  = attrs_.head_dim;

    TensorDesc out_desc  = Q;
    TensorDesc state_desc;
    state_desc.rank   = 4;
    state_desc.dims   = {B, H_kv, D, D};
    state_desc.dtype  = Q.dtype;
    state_desc.layout = Q.layout;

    return {out_desc, state_desc};
}

void LinearAttention::compute(std::span<TensorView> outputs,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 4);
    NNOPS_ASSERT(outputs.size() == 2);

    NNOPS_ASSERT(!inputs[0].is_empty());   // Q
    NNOPS_ASSERT(!inputs[1].is_empty());   // K
    NNOPS_ASSERT(!inputs[2].is_empty());   // V
    NNOPS_ASSERT(!inputs[3].is_empty());   // Gate
    NNOPS_ASSERT(!outputs[0].is_empty());  // Output
    NNOPS_ASSERT(!outputs[1].is_empty());  // State

    impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
