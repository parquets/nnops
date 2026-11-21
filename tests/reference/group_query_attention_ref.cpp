/// @file group_query_attention_ref.cpp
/// @brief Naive CPU reference for grouped-query attention.
///
/// Thin wrapper over attention_ref_common.hpp with the operator's two slot
/// lists: the explicit form (Q, K, V, mask, sink, packed offsets, gate) and
/// the KV-cache form (Q, k_cache, v_cache, seqlens, scales, sink, gate).

#include "attention_ref_common.hpp"
#include "nnops/ops/group_query_attention.hpp"

namespace nnops::backend::cpu::reference {

void group_query_attention_ref(const GroupQueryAttentionAttributes& attrs,
                               TensorView& output,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* /*workspace*/)
{
    detail::AttentionRefSlots slots;
    if (attrs.causal) {
        slots.seqlens = 3;
        slots.pos_ids = 4;
        slots.k_scale = 5;
        slots.v_scale = 6;
        slots.sink = 7;
        slots.gate = 8;
    } else {
        slots.bias = 3;
        slots.kpm = 4;
        slots.sink = 5;
        slots.cu_q = 6;
        slots.cu_k = 7;
        slots.gate = 8;
    }
    const bool bshd = attrs.head_layout == AttentionHeadLayout::BSHD;
    detail::attention_ref_generic(
        attrs.num_heads, attrs.kv_num_heads, 0, attrs.scale, attrs.causal, /*v_is_k=*/false,
        /*position_offset=*/-1, bshd, /*kv_bshd=*/attrs.causal ? false : bshd, slots,
        attrs.mask_filter_value, output, inputs, ctx, attrs.softcap,
        attrs.local_window_size > 0 ? attrs.local_window_size : 0, attrs.output_gate,
        /*seqlens_plus_one=*/!attrs.causal, attrs.sink_tokens);
}

}  // namespace nnops::backend::cpu::reference
