/// @file causal_attention.cpp
/// @brief SIMD-optimized CPU implementation of CausalAttention.
///
/// Currently delegates to the reference kernel for all paths.
/// The entry point provides dtype dispatch; SIMD fast-paths for f32/f32
/// and f32/f16 will be added incrementally.

#include "nnops/ops/causal_attention.hpp"
#include "nnops/detail/assert.hpp"

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
                           void* workspace)
{
    // Delegate to reference for all combinations.
    // SIMD optimizations will be added per (ComputeT, CacheT) pair.
    reference::causal_attention_ref(attrs, outputs, inputs, ctx, workspace);
}

}  // namespace nnops::backend::cpu
