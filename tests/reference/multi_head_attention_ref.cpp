/// @file multi_head_attention_ref.cpp
/// @brief Naive CPU reference for multi-head scaled dot-product attention.
///
/// Thin wrapper over attention_ref_common.hpp: every query head owns its own
/// key/value head, and the value is a third input tensor.

#include "attention_ref_common.hpp"
#include "nnops/ops/multi_head_attention.hpp"

namespace nnops::backend::cpu::reference {

void multi_head_attention_ref(const MultiHeadAttentionAttributes& attrs,
                              TensorView& output,
                              std::span<const TensorView> inputs,
                              const ComputeContext& ctx,
                              void* /*workspace*/)
{
    detail::attention_ref_generic(attrs.num_heads, attrs.num_heads, 0,
                                  attrs.scale, attrs.causal, /*v_is_k=*/false,
                                  attrs.position_offset, output, inputs, ctx);
}

}  // namespace nnops::backend::cpu::reference
