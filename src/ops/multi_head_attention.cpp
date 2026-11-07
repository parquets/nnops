/// @file multi_head_attention.cpp
/// @brief MultiHeadAttention operator dispatch.
///
/// Q, K, V [, mask] in f32 or f16, channel-first, merged [B, S, H*D] or explicit
/// [B, H, S, D]. The optional mask is positional and may still arrive as an
/// empty view (absent, not malformed). Everything below the attribute copy is
/// the shared CPU attention core.

#include "nnops/ops/multi_head_attention.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "backend/cpu/attention/attention.h"

namespace nnops {

NNOPS_DECLARE_CPU(MultiHeadAttentionAttributes, multi_head_attention_cpu)

NNOPS_OP_BOILERPLATE(MultiHeadAttention, MultiHeadAttentionAttributes,
        backend::cpu::multi_head_attention_cpu, nullptr, nullptr)

std::vector<TensorDesc> MultiHeadAttention::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return {attention_output_shape(inputs)};
}

size_t MultiHeadAttention::getWorkspaceSize(std::span<const TensorDesc> inputs,
                                            std::span<const TensorDesc> outputs) const
{
    (void)inputs;
    (void)outputs;
    // Scratch (scores + packed K^T) is pooled internally by the CPU kernel.
    return 0;
}

Status MultiHeadAttention::validate(std::span<const TensorDesc> inputs,
                                    std::span<const TensorDesc> outputs) const
{
    // Q, K, V [, mask [, past K, past V [, seqlens_k]]]
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 3, 7, 1));
    if (!is_layout_supported(inputs[0].layout, LayoutSupport::ChannelFirstOnly)) {
        return Status::unsupported;
    }

    const DataType dt = inputs[0].dtype;
    if (!detail::dtype_in<DataType::f32, DataType::f16>(dt)) {
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
    if (attrs_.num_heads <= 0) { return Status::invalid_argument; }
    if (inputs.size() > kSeqlensSlot && inputs[kSeqlensSlot].rank != 0) {
        const auto& s = inputs[kSeqlensSlot];
        if (s.dtype != DataType::s32 || s.rank != 1 || s.dims[0] != q.dims[0]) {
            return Status::invalid_argument;
        }
    }

    if (q.rank == 3) {
        // [B, S, H*D]: every head shares the row, so K and V must be as wide.
        const int64_t hd = q.dims[2];
        if (hd <= 0 || hd % attrs_.num_heads != 0) { return Status::invalid_argument; }
        if (k.dims[0] != q.dims[0] || k.dims[2] != hd) { return Status::invalid_argument; }
        if (v.dims[0] != q.dims[0] || v.dims[2] != hd) { return Status::invalid_argument; }
        if (v.dims[1] != k.dims[1]) { return Status::invalid_argument; }
    } else {
        const int64_t d = q.dims[3];
        if (d <= 0) { return Status::invalid_argument; }
        if (k.dims[0] != q.dims[0] || k.dims[1] != attrs_.num_heads ||
            k.dims[3] != d) {
            return Status::invalid_argument;
        }
        if (v.dims[0] != q.dims[0] || v.dims[1] != attrs_.num_heads ||
            v.dims[2] != k.dims[2] || v.dims[3] != d) {
            return Status::invalid_argument;
        }
    }

    const int64_t seq_dim = (q.rank == 3) ? 1 : 2;
    const int64_t sq = q.dims[seq_dim];
    const int64_t sk = k.dims[seq_dim];

    // Optional trailing segments: [4] past K, [5] past V, same layout as their
    // new counterpart with Sk_past rows. Both or neither. [6] is the int32 [B]
    // per-batch run length (seqlens_k), checked above.
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

Status MultiHeadAttention::compute(std::span<TensorView> outputs,
                                   std::span<const TensorView> inputs,
                                   const ComputeContext& ctx,
                                   void* workspace)
{
    // The optional mask and past segments are allowed to arrive as empty views
    // (meaning absent), so only the three required inputs are checked for data.
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
