/// @file group_query_attention.cpp
/// @brief GroupQueryAttention operator dispatch.
///
/// Q, K, V [, mask] in f32 or f16, channel-first, merged [B, S, H*D] or explicit
/// [B, H, S, D]. K and V carry kv_num_heads heads, so their rows are
/// H_kv*D wide; query head h reads kv head h / (H / H_kv). Everything below
/// the attribute copy is the shared CPU attention core.

#include "nnops/ops/group_query_attention.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "backend/cpu/attention/attention.h"

namespace nnops {

namespace {
int64_t kv_heads_of(const GroupQueryAttentionAttributes& attrs) {
    return attrs.kv_num_heads > 0 ? attrs.kv_num_heads : attrs.num_heads;
}
}  // anonymous namespace

NNOPS_DECLARE_CPU(GroupQueryAttentionAttributes, group_query_attention_cpu)

NNOPS_OP_BOILERPLATE(GroupQueryAttention, GroupQueryAttentionAttributes,
        backend::cpu::group_query_attention_cpu, nullptr, nullptr)

std::vector<TensorDesc> GroupQueryAttention::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return {attention_output_shape(inputs)};
}

size_t GroupQueryAttention::getWorkspaceSize(std::span<const TensorDesc> inputs,
                                             std::span<const TensorDesc> outputs) const
{
    (void)inputs;
    (void)outputs;
    // Scratch (scores + packed K^T) is pooled internally by the CPU kernel.
    return 0;
}

Status GroupQueryAttention::validate(std::span<const TensorDesc> inputs,
                                     std::span<const TensorDesc> outputs) const
{
    // Q, K, V [, mask [, past K, past V [, seqlens_k]]]
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 3, 7, 1));
    if (!is_layout_supported(inputs[0].layout, LayoutSupport::ChannelFirstOnly)) {
        return Status::unsupported;
    }

    const DataType dt = inputs[0].dtype;
    if (!detail::dtype_in<DataType::f32, DataType::f16>(backend_, dt)) {
        return Status::unsupported;
    }
    for (size_t i = 1; i <= 2; ++i) {
        if (inputs[i].dtype != dt) { return Status::invalid_argument; }
    }
    if (outputs[0].dtype != dt) { return Status::invalid_argument; }
    // Slot 6 is the int32 [B] run-length vector, not an activation.
    constexpr size_t kSeqlensSlot = 6;
    for (size_t i = 3; i < inputs.size(); ++i) {
        if (i == kSeqlensSlot || inputs[i].rank == 0) { continue; }
        if (inputs[i].dtype != dt) { return Status::invalid_argument; }
    }
    const auto& q = inputs[0];
    const auto& k = inputs[1];
    const auto& v = inputs[2];
    if (q.rank != 3 && q.rank != 4) { return Status::invalid_argument; }
    if (k.rank != q.rank || v.rank != q.rank) { return Status::invalid_argument; }
    if (inputs.size() > kSeqlensSlot && inputs[kSeqlensSlot].rank != 0) {
        const auto& s = inputs[kSeqlensSlot];
        if (s.dtype != DataType::s32 || s.rank != 1 || s.dims[0] != q.dims[0]) {
            return Status::invalid_argument;
        }
    }

    const int64_t h = attrs_.num_heads;
    const int64_t hkv = kv_heads_of(attrs_);
    if (h <= 0 || hkv <= 0 || h % hkv != 0) { return Status::invalid_argument; }

    int64_t d, sq, sk;
    if (q.rank == 3) {
        // [B, S, H*D] and [B, S, H_kv*D].
        if (q.dims[2] <= 0 || q.dims[2] % h != 0) { return Status::invalid_argument; }
        d = q.dims[2] / h;
        sq = q.dims[1];
        sk = k.dims[1];
        if (k.dims[0] != q.dims[0] || k.dims[2] != hkv * d) {
            return Status::invalid_argument;
        }
        if (v.dims[0] != q.dims[0] || v.dims[1] != sk || v.dims[2] != hkv * d) {
            return Status::invalid_argument;
        }
    } else {
        d = q.dims[3];
        sq = q.dims[2];
        sk = k.dims[2];
        if (d <= 0 || q.dims[1] != h) { return Status::invalid_argument; }
        if (k.dims[0] != q.dims[0] || k.dims[1] != hkv || k.dims[3] != d) {
            return Status::invalid_argument;
        }
        if (v.dims[0] != q.dims[0] || v.dims[1] != hkv ||
            v.dims[2] != sk || v.dims[3] != d) {
            return Status::invalid_argument;
        }
    }
    if (sq <= 0 || sk <= 0) { return Status::invalid_argument; }

    // Optional trailing segments: [4] past K, [5] past V, same layout as their
    // new counterpart with Sk_past rows. Both or neither. [6] is the int32 [B]
    // per-batch run length (seqlens_k), checked above.
    const int64_t seq_dim = (q.rank == 3) ? 1 : 2;
    const bool has_past_k = inputs.size() > 4 && inputs[4].rank != 0;
    const bool has_past_v = inputs.size() > 5 && inputs[5].rank != 0;
    if (has_past_k != has_past_v) { return Status::invalid_argument; }
    int64_t sk_past = 0;
    if (has_past_k) {
        const auto& pk = inputs[4];
        const auto& pv = inputs[5];
        if (pk.rank != k.rank || pv.rank != v.rank) {
            return Status::invalid_argument;
        }
        for (int64_t i = 0; i < pk.rank; ++i) {
            if (i == seq_dim) { continue; }
            if (pk.dims[i] != k.dims[i] || pv.dims[i] != v.dims[i]) {
                return Status::invalid_argument;
            }
        }
        sk_past = pk.dims[seq_dim];
        if (sk_past <= 0 || pv.dims[seq_dim] != sk_past) {
            return Status::invalid_argument;
        }
    }

    const int64_t sk_all = sk + sk_past;
    if (inputs.size() > 3 && inputs[3].rank != 0) {
        const auto& m = inputs[3];
        if (m.rank != 2 || m.dims[0] != sq || m.dims[1] != sk_all) {
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

Status GroupQueryAttention::compute(std::span<TensorView> outputs,
                                    std::span<const TensorView> inputs,
                                    const ComputeContext& ctx,
                                    void* workspace)
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 3, 7, 1));
    for (size_t i = 0; i < 3; ++i) {
        if (inputs[i].is_empty()) { return Status::invalid_argument; }
    }
    if (outputs[0].is_empty()) { return Status::invalid_argument; }
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
