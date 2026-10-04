/// @file causal_attention.cpp
/// @brief CPU implementation of CausalAttention.
///
/// Currently delegates to the reference kernel for all paths; the entry point
/// provides dtype dispatch. SIMD fast-paths will be added incrementally.

#include "nnops/ops/causal_attention.hpp"
#include "common/memory_pool.hpp"           // internal scratch-memory pool

namespace nnops::backend::cpu {

namespace reference {
    void causal_attention_ref(const CausalAttentionAttributes& attrs,
                               std::span<TensorView> outputs,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* workspace);
}

void causal_attention_cpu(const CausalAttentionAttributes& attrs,
                           std::span<TensorView> outputs,
                           std::span<const TensorView> inputs,
                           const ComputeContext& ctx,
                           void* /*workspace*/)
{
    // The reference materializes a per-(batch, head) scores buffer, which is
    // pooled internally (getWorkspaceSize returns 0). Sized once, before the
    // parallel dispatch.
    const int64_t B      = inputs[0].shape(0);
    const int64_t H      = attrs.num_heads;
    const int64_t Sq     = attrs.max_chunk_size;
    const int64_t max_Sk = attrs.max_cache_seq_len + Sq;
    const size_t ws = static_cast<size_t>(B * H * Sq * max_Sk) * sizeof(float);
    PoolPtr scratch(ws);

    reference::causal_attention_ref(attrs, outputs, inputs, ctx, scratch.get());
}

}  // namespace nnops::backend::cpu
