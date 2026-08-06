/// @file causal_attention.cpp
/// @brief CausalAttention operator dispatch.

#include "nnops/ops/causal_attention.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

namespace backend::cpu::reference {
    void causal_attention_ref(const CausalAttentionAttributes& attrs,
                               std::span<TensorView> outputs,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* workspace);
}

namespace backend::cpu {
    void causal_attention_cpu(const CausalAttentionAttributes& attrs,
                               std::span<TensorView> outputs,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void causal_attention_cuda(const CausalAttentionAttributes& attrs,
                                std::span<TensorView> outputs,
                                std::span<const TensorView> inputs,
                                const ComputeContext& ctx,
                                void* workspace);
}
#endif

// ============================================================
// Impl
// ============================================================
struct CausalAttention::Impl {
    using KernelFn = void (*)(const CausalAttentionAttributes&,
                               std::span<TensorView>,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_causal_attention_kernel(Backend backend) -> CausalAttention::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::causal_attention_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::causal_attention_cuda;
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return nullptr;
#endif
    }
    return nullptr;
}
}  // anonymous namespace

std::unique_ptr<CausalAttention> CausalAttention::create(
    const CausalAttentionAttributes& attrs, Backend backend)
{
    return std::unique_ptr<CausalAttention>(new CausalAttention(attrs, backend));
}

CausalAttention::CausalAttention(const CausalAttentionAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_causal_attention_kernel(backend);
}

std::vector<TensorDesc> CausalAttention::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return causal_attention_output_shape(inputs);
}

size_t CausalAttention::getWorkspaceSize(
    std::span<const TensorDesc> inputs,
    std::span<const TensorDesc> outputs) const
{
    (void)outputs;
    const int64_t B = inputs[0].dims[0];
    const int64_t H = attrs_.num_heads;
    const int64_t Sq = attrs_.max_chunk_size;
    const int64_t max_Sk = attrs_.max_cache_seq_len + Sq;
    return static_cast<size_t>(B * H * Sq * max_Sk) * sizeof(float);
}

void CausalAttention::compute(std::span<TensorView> outputs,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* workspace)
{
    // Validate input count: 5 (contiguous) or 6 (blocked with block_table)
    const bool has_block_table = (attrs_.block_size > 0);
    NNOPS_ASSERT(inputs.size() == (has_block_table ? 6u : 5u));
    NNOPS_ASSERT(outputs.size() == 3);

    const auto& Q = inputs[0];
    const auto& K_new = inputs[1];
    const auto& V_new = inputs[2];
    const auto& cache_pos = inputs[3];
    const auto& cache_len = inputs[4];

    auto& output   = outputs[0];
    auto& K_cache  = outputs[1];
    auto& V_cache  = outputs[2];

    // Basic validation: non-empty
    NNOPS_ASSERT(!Q.is_empty() && !K_new.is_empty() && !V_new.is_empty());
    NNOPS_ASSERT(!output.is_empty() && !K_cache.is_empty() && !V_cache.is_empty());
    NNOPS_ASSERT(!cache_pos.is_empty() && !cache_len.is_empty());

    // Scalar inputs
    NNOPS_ASSERT(cache_pos.rank() == 1 && cache_pos.shape(0) == 1);
    NNOPS_ASSERT(cache_len.rank() == 1 && cache_len.shape(0) == 1);

    // Layout: PlanarOnly
    NNOPS_ASSERT(is_layout_supported(Q.layout(), LayoutSupport::PlanarOnly));

    // Must be rank-4: [B, H, Sq, D]
    NNOPS_ASSERT(Q.rank() == 4);
    NNOPS_ASSERT(K_new.rank() == 4 && V_new.rank() == 4);
    NNOPS_ASSERT(output.rank() == 4);
    NNOPS_ASSERT(K_cache.rank() == 4 && V_cache.rank() == 4);

    // Consistent shapes
    NNOPS_ASSERT(Q.shape(0) == K_new.shape(0) && Q.shape(0) == V_new.shape(0));  // B
    NNOPS_ASSERT(Q.shape(1) == attrs_.num_heads);                                 // H
    NNOPS_ASSERT(Q.shape(1) == K_new.shape(1) && Q.shape(1) == V_new.shape(1));
    NNOPS_ASSERT(Q.shape(2) == K_new.shape(2) && Q.shape(2) == V_new.shape(2));  // Sq
    NNOPS_ASSERT(Q.shape(3) == K_new.shape(3) && Q.shape(3) == V_new.shape(3));  // D

    // Output shape matches Q
    NNOPS_ASSERT(output.shape(0) == Q.shape(0) && output.shape(1) == Q.shape(1));
    NNOPS_ASSERT(output.shape(2) == Q.shape(2) && output.shape(3) == Q.shape(3));

    // Cache bounds
    if (attrs_.block_size > 0) {
        NNOPS_ASSERT(K_cache.shape(2) == attrs_.block_size);
    }

    // Dtype consistency
    NNOPS_ASSERT(Q.data_type() == K_new.data_type());
    NNOPS_ASSERT(Q.data_type() == V_new.data_type());
    NNOPS_ASSERT(output.data_type() == Q.data_type());
    NNOPS_ASSERT(K_cache.data_type() == V_cache.data_type());

    // Call kernel
    NNOPS_ASSERT(impl_->kernel_fn != nullptr);
    impl_->kernel_fn(attrs_, outputs, inputs, ctx, workspace);
}

}  // namespace nnops
