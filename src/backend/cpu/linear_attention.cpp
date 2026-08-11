/// @file linear_attention.cpp
/// @brief CPU implementation stub for LinearAttention (gated linear attention).
///
/// TODO: Implement SIMD-accelerated gated linear attention kernel.
/// Reference: llama.cpp ggml_compute_forward_gated_linear_attn (ggml.c:5806)

#include "nnops/ops/linear_attention.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"

namespace nnops::backend::cpu {

template <typename T>
void linear_attention_impl(const LinearAttentionAttributes& attrs,
                           TensorView& output,
                           std::span<const TensorView> inputs,
                           const ComputeContext& ctx)
{
    // TODO: Implement gated linear attention recurrence
    //
    // Algorithm (Gated Delta Net):
    //   S_new = S * exp(gate) + K^T @ (V - K @ S) * beta
    //   O = Q @ S_new
    //
    // Input shapes (4):
    //   Q:    [B, H,    Sq, D]
    //   K:    [B, H_kv, Sk, D]
    //   V:    [B, H_kv, Sk, D]
    //   Gate: [B, H,    Sq, D]
    //
    // Output shapes (2):
    //   Output: [B, H,    Sq, D]
    //   State:  [B, H_kv, D,  D]  — read old state, write new state in-place
    //
    // References:
    //   - ggml_gated_linear_attn() in llama.cpp ggml/src/ggml.c
    //   - llm_build_delta_net_base in llama.cpp src/models/delta-net-base.cpp

    (void)attrs;
    (void)output;
    (void)inputs;
    (void)ctx;
    NNOPS_ASSERT(!"linear_attention_cpu: not yet implemented");
}

// Entry point with dtype dispatch
void linear_attention_cpu(const LinearAttentionAttributes& attrs,
                          TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx,
                          void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        linear_attention_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        linear_attention_impl<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"linear_attention_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
