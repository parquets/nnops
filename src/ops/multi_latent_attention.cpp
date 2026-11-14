/// @file multi_latent_attention.cpp
/// @brief MultiLatentAttention operator dispatch.
///
/// Absorbed: Q and the packed latent KV [, mask], f32 or f16, channel-first,
/// merged [B, S, H*D] or explicit [B, H, S, D]. D = kv_lora_rank +
/// qk_rope_head_dim on both inputs; the value is the leading v_head_dim
/// channels of each latent row, so there is no third input. Non-absorbed
/// (absorbed = false) takes Q, K, V at the full key width and v_head_dim, with
/// the GQA slot layout. Everything below the attribute copy is the shared CPU
/// attention core.

#include "nnops/ops/multi_latent_attention.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "backend/cpu/attention/attention.h"

namespace nnops {

namespace {

int64_t kv_heads_of(const MultiLatentAttentionAttributes& attrs) {
    return attrs.kv_num_heads > 0 ? attrs.kv_num_heads : attrs.num_heads;
}

int64_t value_dim_of(const MultiLatentAttentionAttributes& attrs) {
    return attrs.v_head_dim > 0 ? attrs.v_head_dim : attrs.kv_lora_rank;
}

int64_t latent_dim_of(const MultiLatentAttentionAttributes& attrs) {
    return attrs.kv_lora_rank + attrs.qk_rope_head_dim;
}

}  // anonymous namespace

NNOPS_DECLARE_CPU(MultiLatentAttentionAttributes, multi_latent_attention_cpu)

NNOPS_OP_BOILERPLATE(MultiLatentAttention, MultiLatentAttentionAttributes,
        backend::cpu::multi_latent_attention_cpu, nullptr, nullptr)

std::vector<TensorDesc> MultiLatentAttention::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    // Q's shape with the last dim swapped from H*Dkv to H*v_head_dim.
    TensorDesc out;
    if (inputs.empty()) { return {out}; }

    out = inputs[0];
    const int64_t vd = value_dim_of(attrs_);
    if (out.rank == 3) {
        out.dims[2] = attrs_.num_heads * vd;
    } else if (out.rank == 4) {
        out.dims[3] = vd;
    }
    return {out};
}

size_t MultiLatentAttention::getWorkspaceSize(std::span<const TensorDesc> inputs,
                                              std::span<const TensorDesc> outputs) const
{
    (void)inputs;
    (void)outputs;
    // Scratch (scores + packed K^T) is pooled internally by the CPU kernel.
    return 0;
}

Status MultiLatentAttention::validate(std::span<const TensorDesc> inputs,
                                      std::span<const TensorDesc> outputs) const
{
    // Absorbed: Q, KV [, mask [, past KV [, seqlens_k [, cu_q, cu_k]]]].
    // Non-absorbed takes the GQA slots: Q, K, V [, mask [, past K, past V [,
    // seqlens_k [, cu_q, cu_k]]]].
    const bool absorbed = attrs_.absorbed;
    const size_t n_req  = absorbed ? 2u : 3u;
    const size_t m_slot = absorbed ? 2u : 3u;
    const size_t p_slot = absorbed ? 3u : 4u;
    const size_t s_slot = absorbed ? 4u : 6u;
    const size_t cq_slot = s_slot + 1;
    const size_t ck_slot = s_slot + 2;

    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(),
                                   n_req, ck_slot + 1, 1));
    if (!is_layout_supported(inputs[0].layout, LayoutSupport::ChannelFirstOnly)) {
        return Status::unsupported;
    }

    const DataType dt = inputs[0].dtype;
    if (!detail::dtype_in<DataType::f32, DataType::f16>(backend_, dt)) {
        return Status::unsupported;
    }
    for (size_t i = 1; i < n_req; ++i) {
        if (inputs[i].dtype != dt) { return Status::invalid_argument; }
    }
    if (outputs[0].dtype != dt) { return Status::invalid_argument; }
    for (size_t i = n_req; i < inputs.size(); ++i) {
        // seqlens_k and the two cu vectors are int32 metadata, not activations.
        if (i >= s_slot || inputs[i].rank == 0) { continue; }
        if (inputs[i].dtype != dt) { return Status::invalid_argument; }
    }

    const auto& q = inputs[0];
    const auto& kv = inputs[1];
    if (q.rank != 3 && q.rank != 4) { return Status::invalid_argument; }
    if (kv.rank != q.rank) { return Status::invalid_argument; }
    const bool has_seqlens = inputs.size() > s_slot && inputs[s_slot].rank != 0;
    if (has_seqlens) {
        const auto& s = inputs[s_slot];
        if (s.dtype != DataType::s32 || s.rank != 1 || s.dims[0] != q.dims[0]) {
            return Status::invalid_argument;
        }
    }

    const int64_t h = attrs_.num_heads;
    const int64_t hkv = kv_heads_of(attrs_);
    const int64_t vd = value_dim_of(attrs_);
    const int64_t dqk = attrs_.qk_nope_head_dim + attrs_.qk_rope_head_dim;
    if (h <= 0 || hkv <= 0 || h % hkv != 0) { return Status::invalid_argument; }
    if (attrs_.qk_nope_head_dim <= 0 || attrs_.qk_rope_head_dim <= 0) {
        return Status::invalid_argument;
    }
    if (vd <= 0) { return Status::invalid_argument; }
    if (absorbed) {
        if (attrs_.kv_lora_rank <= 0 || vd > attrs_.kv_lora_rank) {
            return Status::invalid_argument;
        }
    }

    // The key row's per-head width: the latent in the absorbed form, the full
    // query/key width otherwise.
    const int64_t dk = absorbed ? latent_dim_of(attrs_) : dqk;

    int64_t sq, sk;
    if (q.rank == 3) {
        sq = q.dims[1];
        sk = kv.dims[1];
        if (q.dims[2] != h * dk) { return Status::invalid_argument; }
        if (kv.dims[0] != q.dims[0] || kv.dims[2] != hkv * dk) {
            return Status::invalid_argument;
        }
    } else {
        sq = q.dims[2];
        sk = kv.dims[2];
        if (q.dims[1] != h || q.dims[3] != dk) { return Status::invalid_argument; }
        if (kv.dims[0] != q.dims[0] || kv.dims[1] != hkv || kv.dims[3] != dk) {
            return Status::invalid_argument;
        }
    }
    if (sq <= 0 || sk <= 0) { return Status::invalid_argument; }

    // Non-absorbed: V is its own tensor, vd wide per kv head.
    if (!absorbed) {
        const auto& v = inputs[2];
        if (v.rank != q.rank) { return Status::invalid_argument; }
        if (q.rank == 3) {
            if (v.dims[0] != q.dims[0] || v.dims[1] != sk || v.dims[2] != hkv * vd) {
                return Status::invalid_argument;
            }
        } else {
            if (v.dims[0] != q.dims[0] || v.dims[1] != hkv || v.dims[2] != sk ||
                v.dims[3] != vd) {
                return Status::invalid_argument;
            }
        }
    }

    // The output keeps Q's batch/heads and the value width.
    const auto& out = outputs[0];
    if (out.rank != q.rank) { return Status::invalid_argument; }
    if (q.rank == 3) {
        if (out.dims[0] != q.dims[0] || out.dims[1] != sq ||
            out.dims[2] != h * vd) {
            return Status::invalid_argument;
        }
    } else {
        if (out.dims[0] != q.dims[0] || out.dims[1] != h ||
            out.dims[2] != sq || out.dims[3] != vd) {
            return Status::invalid_argument;
        }
    }

    // Optional past segment(s): the same layout as K/V, with Sk_past rows.
    // Both or neither in the non-absorbed form.
    const int64_t seq_dim = (q.rank == 3) ? 1 : 2;
    const bool has_past_k = inputs.size() > p_slot && inputs[p_slot].rank != 0;
    if (!absorbed) {
        const bool has_past_v = inputs.size() > p_slot + 1 &&
                                inputs[p_slot + 1].rank != 0;
        if (has_past_k != has_past_v) { return Status::invalid_argument; }
    }
    int64_t sk_past = 0;
    if (has_past_k) {
        const auto& pk = inputs[p_slot];
        if (pk.rank != kv.rank) { return Status::invalid_argument; }
        for (int64_t i = 0; i < pk.rank; ++i) {
            if (i != seq_dim && pk.dims[i] != kv.dims[i]) {
                return Status::invalid_argument;
            }
        }
        sk_past = pk.dims[seq_dim];
        if (sk_past <= 0) { return Status::invalid_argument; }
        if (!absorbed) {
            const auto& pv = inputs[p_slot + 1];
            const auto& v = inputs[2];
            if (pv.rank != v.rank) { return Status::invalid_argument; }
            for (int64_t i = 0; i < pv.rank; ++i) {
                if (i == seq_dim) { continue; }
                if (pv.dims[i] != v.dims[i]) { return Status::invalid_argument; }
            }
            if (pv.dims[seq_dim] != sk_past) { return Status::invalid_argument; }
        }
    }

    const int64_t sk_all = sk + sk_past;
    if (inputs.size() > m_slot && inputs[m_slot].rank != 0) {
        const auto& m = inputs[m_slot];
        // Rank 2 is the flat [Sq, Sk] every head shares; rank 4 is a per-head
        // additive bias [B, H, Sq, Sk].
        if (m.rank == 2) {
            if (m.dims[0] != sq || m.dims[1] != sk_all) {
                return Status::invalid_argument;
            }
        } else if (m.rank == 4) {
            if (m.dims[0] != q.dims[0] || m.dims[1] != h ||
                m.dims[2] != sq || m.dims[3] != sk_all) {
                return Status::invalid_argument;
            }
        } else {
            return Status::invalid_argument;
        }
    }

    // Varlen: int32 [segments + 1] prefix offsets of a packed batch. Exclusive
    // with the mask, a past segment, seqlens_k and a global position_offset.
    const bool has_cu_q = inputs.size() > cq_slot && inputs[cq_slot].rank != 0;
    const bool has_cu_k = inputs.size() > ck_slot && inputs[ck_slot].rank != 0;
    if (has_cu_q != has_cu_k) { return Status::invalid_argument; }
    if (has_cu_q) {
        const auto& cq = inputs[cq_slot];
        const auto& ck = inputs[ck_slot];
        if (cq.dtype != DataType::s32 || ck.dtype != DataType::s32 ||
            cq.rank != 1 || ck.rank != 1 || cq.dims[0] != ck.dims[0] ||
            cq.dims[0] < 2) {
            return Status::invalid_argument;
        }
        if (q.rank != 3 || q.dims[0] != 1) { return Status::invalid_argument; }
        if (has_past_k || has_seqlens || attrs_.position_offset >= 0) {
            return Status::invalid_argument;
        }
        if (inputs.size() > m_slot && inputs[m_slot].rank != 0) {
            return Status::invalid_argument;
        }
    }

    // Every query row must sit inside the run it attends to.
    if (attrs_.causal && attrs_.position_offset >= 0 &&
        attrs_.position_offset + sq > sk_all) {
        return Status::invalid_argument;
    }
    return Status::ok;
}

Status MultiLatentAttention::compute(std::span<TensorView> outputs,
                                     std::span<const TensorView> inputs,
                                     const ComputeContext& ctx,
                                     void* workspace)
{
    const size_t n_req = attrs_.absorbed ? 2u : 3u;
    const size_t max_in = attrs_.absorbed ? 7u : 9u;
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), n_req, max_in, 1));
    for (size_t i = 0; i < n_req; ++i) {
        if (inputs[i].is_empty()) { return Status::invalid_argument; }
    }
    if (outputs[0].is_empty()) { return Status::invalid_argument; }
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
