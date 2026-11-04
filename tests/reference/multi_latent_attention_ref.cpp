/// @file multi_latent_attention_ref.cpp
/// @brief Naive CPU reference for absorbed multi-head latent attention.
///
/// Thin wrapper over attention_ref_common.hpp with v_is_k set: the value is the
/// leading v_head_dim channels of the latent key row, so there is no third
/// input. The default scale is MLA's non-absorbed one,
/// 1/sqrt(qk_nope_head_dim + qk_rope_head_dim).

#include "attention_ref_common.hpp"
#include "nnops/ops/multi_latent_attention.hpp"

#include <cmath>

namespace nnops::backend::cpu::reference {

void multi_latent_attention_ref(const MultiLatentAttentionAttributes& attrs,
                                TensorView& output,
                                std::span<const TensorView> inputs,
                                const ComputeContext& ctx,
                                void* /*workspace*/)
{
    const int64_t vd = attrs.v_head_dim > 0 ? attrs.v_head_dim : attrs.kv_lora_rank;
    const float scale = attrs.scale != 0.0f
        ? attrs.scale
        : 1.0f / std::sqrt(static_cast<float>(attrs.qk_nope_head_dim +
                                               attrs.qk_rope_head_dim));
    detail::attention_ref_generic(attrs.num_heads, attrs.kv_num_heads, vd,
                                  scale, attrs.causal, /*v_is_k=*/true,
                                  output, inputs, ctx);
}

}  // namespace nnops::backend::cpu::reference
