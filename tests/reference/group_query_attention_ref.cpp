/// @file group_query_attention_ref.cpp
/// @brief Naive CPU reference for grouped-query attention.
///
/// Thin wrapper over attention_ref_common.hpp: query head h reads kv head
/// h / (H / H_kv), so consecutive query heads share one K/V pair.

#include "attention_ref_common.hpp"
#include "nnops/ops/group_query_attention.hpp"

namespace nnops::backend::cpu::reference {

void group_query_attention_ref(const GroupQueryAttentionAttributes& attrs,
                               TensorView& output,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* /*workspace*/)
{
    detail::attention_ref_generic(attrs.num_heads, attrs.kv_num_heads, 0,
                                  attrs.scale, attrs.causal, /*v_is_k=*/false,
                                  output, inputs, ctx);
}

}  // namespace nnops::backend::cpu::reference
