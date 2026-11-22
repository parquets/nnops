/// @file group_query_attention.cpp
/// @brief GroupQueryAttention operator dispatch — the one attention operator
///        with a fused softmax core (MHA is kv_num_heads = num_heads).
///
/// `causal` picks the input contract, not just the mask. causal = false is the
/// explicit ViT/encoder form — Q, K, V and an optional external mask; causal =
/// true is the KV-cache form — Q and a written cache, with the causal range
/// derived from each batch's kv_seqlens. See the header for the slot lists.
/// Everything below the attribute copy is the shared CPU attention core.

#include "nnops/ops/group_query_attention.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "backend/cpu/attention/attention.h"

#include <cmath>

namespace nnops {

namespace {

int64_t kv_heads_of(const GroupQueryAttentionAttributes& attrs)
{
    return attrs.kv_num_heads > 0 ? attrs.kv_num_heads : attrs.num_heads;
}

// ---- explicit form (causal = false) ----
constexpr size_t kAttnBiasSlot = 3;
constexpr size_t kKpmSlot = 4;
constexpr size_t kSinkSlot = 5;
constexpr size_t kCuQSlot = 6;
constexpr size_t kCuKSlot = 7;
constexpr size_t kGateSlot = 8;
// ---- cache form (causal = true) ----
constexpr size_t kSeqlensSlot = 3;
constexpr size_t kPosIdsSlot = 4;
constexpr size_t kKScaleSlot = 5;
constexpr size_t kVScaleSlot = 6;
constexpr size_t kSinkSlotCached = 7;
constexpr size_t kGateSlotCached = 8;
constexpr size_t kMaxInputs = 9;

bool present(const std::span<const TensorDesc> inputs, size_t slot)
{
    return inputs.size() > slot && inputs[slot].rank != 0;
}

} // anonymous namespace

namespace backend::cpu {
Status group_query_attention_cpu(const GroupQueryAttentionAttributes& attrs,
                                 std::span<TensorView> outputs,
                                 std::span<const TensorView> inputs, const ComputeContext& ctx,
                                 void* workspace);
} // namespace backend::cpu

struct GroupQueryAttention::Impl {
    using KernelFn = Status (*)(const GroupQueryAttentionAttributes&, std::span<TensorView>,
                                std::span<const TensorView>, const ComputeContext&, void*);
    KernelFn kernel_fn = nullptr;
};
GroupQueryAttention::~GroupQueryAttention() = default;

namespace {
auto resolve_gqa_kernel(Backend backend) -> GroupQueryAttention::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::group_query_attention_cpu;
    default:
        return nullptr;
    }
}
} // anonymous namespace

NNOPS_OP_CREATE(GroupQueryAttention, GroupQueryAttentionAttributes)

GroupQueryAttention::GroupQueryAttention(const GroupQueryAttentionAttributes& attrs,
                                         Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_gqa_kernel(backend);
}

std::vector<TensorDesc> GroupQueryAttention::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    // One output in both forms, following Q's geometry.
    return {inputs[0]};
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
    if (attrs_.output_gate != EpilogueActivateType::None &&
        !is_gate_activation(attrs_.output_gate)) {
        return Status::unsupported;
    }
    if (outputs.size() != 1) {
        return Status::invalid_argument;
    }
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 3, kMaxInputs, 1));
    if (!is_layout_supported(inputs[0].layout, LayoutSupport::Any)) {
        return Status::unsupported;
    }

    const bool cache = attrs_.causal;
    const DataType dt = inputs[0].dtype;
    if (!detail::dtype_in<DataType::f32, DataType::f16>(backend_, dt)) {
        return Status::unsupported;
    }
    const DataType kdt = inputs[1].dtype;
    if (cache) {
        // The cache is the query's dtype or a symmetric int8 one; K and V
        // share it.
        if ((kdt != dt && kdt != DataType::s8) || inputs[2].dtype != kdt) {
            return Status::invalid_argument;
        }
    } else if (kdt != dt || inputs[2].dtype != dt) {
        return Status::invalid_argument;
    }
    if (outputs[0].dtype != dt) {
        return Status::invalid_argument;
    }

    const int64_t h = attrs_.num_heads;
    const int64_t hkv = kv_heads_of(attrs_);
    if (h <= 0 || hkv <= 0 || h % hkv != 0) {
        return Status::invalid_argument;
    }
    if (!(attrs_.softcap >= 0.0f) || !std::isfinite(attrs_.softcap) ||
        attrs_.local_window_size < -1 || attrs_.sink_tokens < 0) {
        return Status::invalid_argument;
    }
    if (attrs_.head_layout != AttentionHeadLayout::Auto &&
        attrs_.head_layout != AttentionHeadLayout::BSHD) {
        return Status::invalid_argument;
    }

    const bool bshd = attrs_.head_layout == AttentionHeadLayout::BSHD;
    const auto& q = inputs[0];
    if (q.rank != 3 && q.rank != 4) {
        return Status::invalid_argument;
    }
    const int64_t q_seq = (q.rank == 3 || bshd) ? 1 : 2;
    const int64_t q_head = (q.rank == 3) ? 2 : (bshd ? 2 : 1);
    int64_t d = 0;
    if (q.rank == 3) {
        if (q.dims[2] <= 0 || q.dims[2] % h != 0) {
            return Status::invalid_argument;
        }
        d = q.dims[2] / h;
    } else {
        d = q.dims[3];
        if (d <= 0 || q.dims[static_cast<size_t>(q_head)] != h) {
            return Status::invalid_argument;
        }
    }
    const int64_t sq = q.dims[static_cast<size_t>(q_seq)];
    if (sq <= 0) {
        return Status::invalid_argument;
    }

    const auto& k = inputs[1];
    const auto& v = inputs[2];
    if (cache) {
        // k / v are [B, H_kv, Skv, D] BNSH, contiguous, written by the caller.
        if (k.rank != 4 || v.rank != 4) {
            return Status::invalid_argument;
        }
        if (k.dims[0] != q.dims[0] || k.dims[1] != hkv || k.dims[3] != d || k.dims[2] <= 0) {
            return Status::invalid_argument;
        }
        if (v.dims != k.dims) {
            return Status::invalid_argument;
        }
        if (sq > k.dims[2]) {
            return Status::invalid_argument;
        }
        // int32 metadata: the per-batch run lengths.
        if (present(inputs, kSeqlensSlot)) {
            const auto& s = inputs[kSeqlensSlot];
            if (s.dtype != DataType::s32 || s.rank != 1 || s.dims[0] != q.dims[0]) {
                return Status::invalid_argument;
            }
        }
        // int64 metadata: the per-batch first-query-row positions.
        if (present(inputs, kPosIdsSlot)) {
            const auto& p = inputs[kPosIdsSlot];
            if (p.dtype != DataType::s64 || p.rank != 1 || p.dims[0] != q.dims[0]) {
                return Status::invalid_argument;
            }
        }
        // The dequantization scales are present exactly for an int8 cache, and
        // are either one value per tensor or one per (batch, kv head, row).
        const bool q8 = kdt == DataType::s8;
        for (const size_t s : {kKScaleSlot, kVScaleSlot}) {
            const bool has = present(inputs, s);
            if (has != q8) {
                return Status::invalid_argument;
            }
            if (has) {
                const auto& sc = inputs[s];
                const bool per_tensor = sc.dtype == DataType::f32 && sc.numel() == 1;
                const bool per_token = sc.dtype == DataType::f32 && sc.rank == 3 &&
                                       sc.dims[0] == q.dims[0] && sc.dims[1] == hkv &&
                                       sc.dims[2] == k.dims[2];
                if (!per_tensor && !per_token) {
                    return Status::invalid_argument;
                }
            }
        }
    } else {
        // Explicit K/V in Q's layout; the key run may differ from the query.
        if (k.rank != q.rank || v.rank != q.rank) {
            return Status::invalid_argument;
        }
        int64_t sk = 0;
        if (q.rank == 3) {
            if (k.dims[0] != q.dims[0] || k.dims[2] != hkv * d) {
                return Status::invalid_argument;
            }
            if (v.dims[0] != q.dims[0] || v.dims[1] != k.dims[1] || v.dims[2] != hkv * d) {
                return Status::invalid_argument;
            }
            sk = k.dims[1];
        } else {
            sk = k.dims[static_cast<size_t>(q_seq)];
            if (k.dims[0] != q.dims[0] || k.dims[static_cast<size_t>(q_head)] != hkv ||
                k.dims[3] != d) {
                return Status::invalid_argument;
            }
            if (v.dims[0] != q.dims[0] || v.dims[static_cast<size_t>(q_head)] != hkv ||
                v.dims[static_cast<size_t>(q_seq)] != sk || v.dims[3] != d) {
                return Status::invalid_argument;
            }
        }
        if (sk <= 0) {
            return Status::invalid_argument;
        }

        // attention_bias: rank 2 [Sq, Sk] or rank 4 [B|1, H|1, Sq, Sk].
        if (present(inputs, kAttnBiasSlot)) {
            const auto& m = inputs[kAttnBiasSlot];
            if (m.dtype != dt) {
                return Status::invalid_argument;
            }
            if (m.rank == 2) {
                if (m.dims[0] != sq || m.dims[1] != sk) {
                    return Status::invalid_argument;
                }
            } else if (m.rank == 4) {
                const bool b_ok = m.dims[0] == q.dims[0] || m.dims[0] == 1;
                const bool h_ok = m.dims[1] == h || m.dims[1] == 1;
                if (!b_ok || !h_ok || m.dims[2] != sq || m.dims[3] != sk) {
                    return Status::invalid_argument;
                }
            } else {
                return Status::invalid_argument;
            }
        }
        // key_padding_mask: [B] lengths, [B, Sk] or [B, Sq, Sk] keep-mask.
        if (present(inputs, kKpmSlot)) {
            const auto& m = inputs[kKpmSlot];
            if (m.dtype != DataType::s32) {
                return Status::invalid_argument;
            }
            if (m.rank == 1) {
                if (m.dims[0] != q.dims[0]) {
                    return Status::invalid_argument;
                }
            } else if (m.rank == 2) {
                if (m.dims[0] != q.dims[0] || m.dims[1] != sk) {
                    return Status::invalid_argument;
                }
            } else if (m.rank == 3) {
                if (m.dims[0] != q.dims[0] || m.dims[1] != sq || m.dims[2] != sk) {
                    return Status::invalid_argument;
                }
            } else {
                return Status::invalid_argument;
            }
        }
        // Packed documents: the two cu vectors arrive together, are int32
        // [segments + 1], and rule out a shared mask and a mask bias.
        const bool has_cu_q = present(inputs, kCuQSlot);
        const bool has_cu_k = present(inputs, kCuKSlot);
        if (has_cu_q != has_cu_k) {
            return Status::invalid_argument;
        }
        if (has_cu_q) {
            const auto& cq = inputs[kCuQSlot];
            const auto& ck = inputs[kCuKSlot];
            if (cq.dtype != DataType::s32 || ck.dtype != DataType::s32 || cq.rank != 1 ||
                ck.rank != 1 || cq.dims[0] != ck.dims[0] || cq.dims[0] < 2) {
                return Status::invalid_argument;
            }
            if (q.rank != 3 || q.dims[0] != 1) {
                return Status::invalid_argument;
            }
            if (present(inputs, kKpmSlot) || present(inputs, kAttnBiasSlot)) {
                return Status::invalid_argument;
            }
        }
    }

    // head_sink is one logit per query head in both forms.
    const size_t sink_slot = cache ? kSinkSlotCached : kSinkSlot;
    if (present(inputs, sink_slot)) {
        const auto& sink = inputs[sink_slot];
        if (sink.dtype != dt || sink.rank != 1 || sink.dims[0] != h) {
            return Status::invalid_argument;
        }
    }

    const size_t gate_slot = cache ? kGateSlotCached : kGateSlot;
    const bool has_gate = present(inputs, gate_slot);
    if (attrs_.output_gate != EpilogueActivateType::None) {
        if (!has_gate) {
            return Status::invalid_argument;
        }
        const auto& g = inputs[gate_slot];
        if (g.dtype != dt || g.rank != q.rank || g.dims != q.dims) {
            return Status::invalid_argument;
        }
    } else if (has_gate) {
        return Status::invalid_argument;
    }

    return Status::ok;
}

Status GroupQueryAttention::compute(std::span<TensorView> outputs,
                                    std::span<const TensorView> inputs, const ComputeContext& ctx,
                                    void* workspace)
{
    if (outputs.size() != 1) {
        return Status::invalid_argument;
    }
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 3, kMaxInputs, 1));
    for (size_t i = 0; i < 3; ++i) {
        if (inputs[i].is_empty()) {
            return Status::invalid_argument;
        }
    }
    if (outputs[0].is_empty()) {
        return Status::invalid_argument;
    }
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) {
        return Status::unsupported;
    }
    return impl_->kernel_fn(attrs_, outputs, inputs, ctx, workspace);
}

} // namespace nnops
