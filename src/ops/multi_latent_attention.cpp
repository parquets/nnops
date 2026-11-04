/// @file multi_latent_attention.cpp
/// @brief MultiLatentAttention operator dispatch.
///
/// Q and the packed latent KV [, mask], f32 or f16, planar, merged [B, S, H*D]
/// or explicit [B, H, S, D]. D = kv_lora_rank + qk_rope_head_dim on both
/// inputs; the value is the leading v_head_dim channels of each latent row, so
/// there is no third input. Everything below the attribute copy is the shared
/// CPU attention core.

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
    // Q, KV [, mask]
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 2, 3, 1));
    if (!is_layout_supported(inputs[0].layout, LayoutSupport::PlanarOnly)) {
        return Status::unsupported;
    }

    const DataType dt = inputs[0].dtype;
    if (!detail::dtype_in<DataType::f32, DataType::f16>(dt)) {
        return Status::unsupported;
    }
    if (inputs[1].dtype != dt || outputs[0].dtype != dt) {
        return Status::invalid_argument;
    }
    if (inputs.size() > 2 && inputs[2].rank != 0 && inputs[2].dtype != dt) {
        return Status::invalid_argument;
    }

    const auto& q = inputs[0];
    const auto& kv = inputs[1];
    if (q.rank != 3 && q.rank != 4) { return Status::invalid_argument; }
    if (kv.rank != q.rank) { return Status::invalid_argument; }

    const int64_t h = attrs_.num_heads;
    const int64_t hkv = kv_heads_of(attrs_);
    const int64_t dkv = latent_dim_of(attrs_);
    const int64_t vd = value_dim_of(attrs_);
    if (h <= 0 || hkv <= 0 || h % hkv != 0) { return Status::invalid_argument; }
    if (attrs_.kv_lora_rank <= 0 || attrs_.qk_rope_head_dim <= 0) {
        return Status::invalid_argument;
    }
    if (vd <= 0 || vd > attrs_.kv_lora_rank) { return Status::invalid_argument; }

    int64_t sq, sk;
    if (q.rank == 3) {
        sq = q.dims[1];
        sk = kv.dims[1];
        if (q.dims[2] != h * dkv) { return Status::invalid_argument; }
        if (kv.dims[0] != q.dims[0] || kv.dims[2] != hkv * dkv) {
            return Status::invalid_argument;
        }
    } else {
        sq = q.dims[2];
        sk = kv.dims[2];
        if (q.dims[1] != h || q.dims[3] != dkv) { return Status::invalid_argument; }
        if (kv.dims[0] != q.dims[0] || kv.dims[1] != hkv || kv.dims[3] != dkv) {
            return Status::invalid_argument;
        }
    }
    if (sq <= 0 || sk <= 0) { return Status::invalid_argument; }

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

    if (inputs.size() > 2 && inputs[2].rank != 0) {
        const auto& m = inputs[2];
        if (m.rank != 2 || m.dims[0] != sq || m.dims[1] != sk) {
            return Status::invalid_argument;
        }
    }
    return Status::ok;
}

Status MultiLatentAttention::compute(std::span<TensorView> outputs,
                                     std::span<const TensorView> inputs,
                                     const ComputeContext& ctx,
                                     void* workspace)
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 2, 3, 1));
    for (size_t i = 0; i < 2; ++i) {
        if (inputs[i].is_empty()) { return Status::invalid_argument; }
    }
    if (outputs[0].is_empty()) { return Status::invalid_argument; }
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
