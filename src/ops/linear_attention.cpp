/// @file linear_attention.cpp
/// @brief LinearAttention operator dispatch.

#include "nnops/ops/linear_attention.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops {

namespace backend::cpu {
    void linear_attention_cpu(const LinearAttentionAttributes& attrs,
                              std::span<TensorView> outputs,
                              std::span<const TensorView> inputs,
                              const ComputeContext& ctx,
                              void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void linear_attention_cuda(const LinearAttentionAttributes& attrs,
                               std::span<TensorView> outputs,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* workspace);
}
#endif

struct LinearAttention::Impl {
    // `outputs` is a span, not a single view: the operator's recurrent State
    // lives in outputs[1], and it has to reach the kernel.
    using KernelFn = void (*)(const LinearAttentionAttributes&,
                              std::span<TensorView>,
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
    default:
        return nullptr;
    }
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
    NNOPS_ASSERT(inputs.size() >= 4);
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
    NNOPS_ASSERT(inputs.size() >= 4 && inputs.size() <= 6);
    NNOPS_ASSERT(outputs.size() == 2);

    NNOPS_ASSERT(!inputs[0].is_empty());   // Q
    NNOPS_ASSERT(!inputs[1].is_empty());   // K
    NNOPS_ASSERT(!inputs[2].is_empty());   // V
    NNOPS_ASSERT(!inputs[3].is_empty());   // Gate
    NNOPS_ASSERT(!outputs[0].is_empty());  // Output
    NNOPS_ASSERT(!outputs[1].is_empty());  // State

    // Structural validation, mirroring the kernel's own derivation. This guards
    // the *contract* — it catches a mis-wired graph in debug builds, it does not
    // validate data. Remember NNOPS_ASSERT compiles away under NDEBUG.
    const DataType dt = inputs[0].data_type();
    NNOPS_ASSERT(dt == DataType::f32 || dt == DataType::f16);
    NNOPS_ASSERT(inputs[1].data_type() == dt && inputs[2].data_type() == dt &&
                 inputs[3].data_type() == dt && outputs[0].data_type() == dt &&
                 outputs[1].data_type() == dt);

    NNOPS_ASSERT(inputs[0].rank() == 4);
    const int64_t B    = inputs[0].shape(0);
    const int64_t H    = inputs[0].shape(1);
    const int64_t S    = inputs[0].shape(2);
    const int64_t D    = inputs[0].shape(3);
    const int64_t H_kv = attrs_.num_kv_heads > 0 ? attrs_.num_kv_heads : H;
    NNOPS_ASSERT(H_kv > 0 && H % H_kv == 0);

    // K/V/Gate share the kv-head geometry: [B, H_kv, S, D].
    for (int i = 1; i <= 3; ++i) {
        NNOPS_ASSERT(inputs[i].shape(0) == B && inputs[i].shape(1) == H_kv &&
                     inputs[i].shape(2) == S && inputs[i].shape(3) == D);
    }
    NNOPS_ASSERT(outputs[0].shape(0) == B && outputs[0].shape(1) == H &&
                 outputs[0].shape(2) == S && outputs[0].shape(3) == D);
    NNOPS_ASSERT(outputs[1].shape(0) == B && outputs[1].shape(1) == H_kv &&
                 outputs[1].shape(2) == D && outputs[1].shape(3) == D);

    if (inputs.size() > 4 && !inputs[4].is_empty()) {   // conv weights
        NNOPS_ASSERT(inputs[4].data_type() == dt);
        NNOPS_ASSERT(inputs[4].shape(0) == (H + 2 * H_kv) * D &&
                     inputs[4].shape(1) == 1 &&
                     inputs[4].shape(2) == attrs_.conv_kernel_size);
    }
    if (inputs.size() > 5 && !inputs[5].is_empty()) {   // beta
        NNOPS_ASSERT(inputs[5].data_type() == dt);
        NNOPS_ASSERT(inputs[5].shape(0) == B && inputs[5].shape(1) == H_kv &&
                     inputs[5].shape(2) == S);
    }

    // Vulkan has no LinearAttention kernel and resolves to nullptr — the assert
    // documents that invariant (and, like every other guard here, compiles away
    // under NDEBUG). CPU and CUDA always resolve to a real kernel.
    NNOPS_ASSERT(impl_->kernel_fn != nullptr);

    impl_->kernel_fn(attrs_, outputs, inputs, ctx, workspace);
}

}  // namespace nnops
