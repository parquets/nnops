/// @file causal_attention.cpp
/// @brief SIMD-optimized CPU implementation of CausalAttention.
///
/// Currently delegates to the reference kernel for all paths.
/// The entry point provides dtype dispatch; SIMD fast-paths for f32/f32
/// and f32/f16 will be added incrementally.

#include "nnops/ops/causal_attention.hpp"
#include "common/memory_pool.hpp"           // internal scratch-memory pool

namespace nnops::backend::cpu {

// Forward-declare reference kernel
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
    // Delegate to reference for all combinations; SIMD optimizations will be
    // added per (ComputeT, CacheT) pair. The reference materializes a per-
    // (batch, head) scores buffer, which is pooled internally (getWorkspaceSize
    // returns 0). Sized once, before the parallel dispatch.
    const int64_t B      = inputs[0].shape(0);
    const int64_t H      = attrs.num_heads;
    const int64_t Sq     = attrs.max_chunk_size;
    const int64_t max_Sk = attrs.max_cache_seq_len + Sq;
    const size_t ws = static_cast<size_t>(B * H * Sq * max_Sk) * sizeof(float);
    PoolPtr scratch(ws);

    reference::causal_attention_ref(attrs, outputs, inputs, ctx, scratch.get());
}

}  // namespace nnops::backend::cpu
