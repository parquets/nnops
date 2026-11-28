#pragma once
/// @file attention_geometry.hpp
/// @brief The head geometry one attention call addresses — the storage-order
///        classification and the strides both backends derive from it.
///
/// Three storage orders describe the same head geometry: merged [B, S, H*D]
/// (rank 3, heads packed inside the row), onnxruntime's BNSH [B, H, S, D], and
/// HuggingFace's BSHD [B, S, H, D]. The last two share a rank, so the operator
/// front-end picks between them (cfg.bshd / cfg.kv_bshd); the routes only see
/// the layout and address through strides, never through a per-layout formula.
///
/// AttentionConfig (attention_config.hpp) says *what* the call is; this header
/// says *where* its heads sit. The CUDA launch code and the CPU routes resolve
/// the same shape, the same layout and the same strides through these
/// functions, so a layout interpretation cannot drift between the backends.

#include "backend/common/attention_config.hpp"
#include "nnops/core/tensor_view.hpp"

#include <cstdint>
#include <span>

namespace nnops::backend {

enum class HeadLayout {
    Merged, ///< rank 3: [B, S, H*D]
    BHSD,   ///< rank 4: [B, H, S, D], onnxruntime's cache-major form
    BSHD    ///< rank 4: [B, S, H, D]
};

/// A call's head geometry after the layouts are known. `Sk` counts the rows the
/// key input carries (the new segment when a past tensor is present);
/// `Sk_past` counts the past segment's rows, so the logical run is their sum.
struct HeadShape {
    int64_t B = 0, H = 0, Hkv = 0, Sq = 0, Sk = 0, Sv = 0, Dqk = 0, Dv = 0;
    int64_t Sk_past = 0;      ///< valid rows in the past segment
    int64_t Sk_past_rows = 0; ///< rows the past tensor carries (its capacity)
    HeadLayout q_layout = HeadLayout::Merged;  ///< Q and the output
    HeadLayout kv_layout = HeadLayout::Merged; ///< K, V and the past pair

    /// Q's heads live inside the row (merged / BSHD), not in their own axis.
    bool row_heads() const noexcept { return q_layout != HeadLayout::BHSD; }
};

/// A slot is present when the operator has one and its view carries data.
inline bool slot_present(const AttentionConfig& cfg, int slot,
                         std::span<const TensorView> inputs)
{
    return slot >= 0 && inputs.size() > static_cast<size_t>(slot) &&
           !inputs[static_cast<size_t>(slot)].is_empty();
}

inline bool has_past_view(const AttentionConfig& cfg, std::span<const TensorView> inputs)
{
    return slot_present(cfg, cfg.past_k_slot, inputs);
}

inline HeadLayout head_layout_of(const AttentionConfig& cfg, const TensorView& q)
{
    if (q.rank() == 3) {
        return HeadLayout::Merged;
    }
    return cfg.bshd ? HeadLayout::BSHD : HeadLayout::BHSD;
}

/// K/V carry their own layout: the cache is BNSH whatever Q's order is, while
/// the explicit form keeps K/V in Q's order.
inline HeadLayout kv_layout_of(const AttentionConfig& cfg, const TensorView& k)
{
    if (k.rank() == 3) {
        return HeadLayout::Merged;
    }
    return cfg.kv_bshd ? HeadLayout::BSHD : HeadLayout::BHSD;
}

/// The axis a Q/K/V-like view carries its sequence on.
inline int64_t head_seq_dim(HeadLayout layout) { return layout == HeadLayout::BHSD ? 2 : 1; }

/// Elements from one sequence position to the next *within one head*. That is
/// the row pitch for merged and [B, H, S, D] — there the row is exactly one
/// head's slice — but for [B, S, H, D] a row is the head dim alone, so the step
/// is the whole head block.
inline int64_t head_seq_step(const TensorView& v, HeadLayout layout)
{
    return layout == HeadLayout::BSHD ? v.stride_elems(1) : v.row_stride_elems();
}

/// One head forward. A merged row packs the heads, so the step is the head
/// width; the rank-4 forms carry their own axis.
inline int64_t head_step(const TensorView& v, HeadLayout layout, int64_t d)
{
    return layout == HeadLayout::Merged ? d
                                        : (layout == HeadLayout::BHSD ? v.stride_elems(1)
                                                                      : v.stride_elems(2));
}

/// Element strides from one batch and one head to the next. @p d is the head
/// width for the merged form, whose heads are packed inside the row.
inline void head_strides(const TensorView& v, HeadLayout layout, int64_t d, int64_t& batch_step,
                         int64_t& head_advance)
{
    batch_step = v.stride_elems(0);
    head_advance = head_step(v, layout, d);
}

/// Offset of the value head inside a merged row: a latent K row is Dqk wide
/// per head even though only Dv of those columns are values.
inline int64_t merged_v_head_stride(const AttentionConfig& cfg, int64_t Dqk, int64_t Dv)
{
    return cfg.v_is_k ? Dqk : Dv;
}

/// Resolve the geometry from the operator's views: Q is input 0, K input 1, V
/// input 2 unless the value is the key's prefix. The past pair's row count is
/// read when the config names the slot.
inline HeadShape resolve_head_shape(const AttentionConfig& cfg,
                                    std::span<const TensorView> inputs)
{
    const auto& Q = inputs[0];
    const auto& K = inputs[1];
    const auto& V = cfg.v_is_k ? inputs[1] : inputs[2];

    HeadShape s;
    s.q_layout = head_layout_of(cfg, Q);
    s.kv_layout = kv_layout_of(cfg, K);
    s.B = Q.shape(0);
    s.H = cfg.num_heads;
    s.Hkv = cfg.kv_heads();
    s.Sq = Q.shape(head_seq_dim(s.q_layout));
    s.Sk = K.shape(head_seq_dim(s.kv_layout));
    s.Sv = V.shape(head_seq_dim(s.kv_layout));
    s.Dqk = (s.q_layout == HeadLayout::Merged) ? Q.shape(2) / s.H : Q.shape(3);
    // The value head is Dqk wide unless the config names its own width — MLA's
    // latent, or the non-absorbed form's v_head_dim.
    s.Dv = cfg.v_head_dim > 0 ? cfg.v_head_dim : s.Dqk;
    if (has_past_view(cfg, inputs)) {
        const auto& P = inputs[static_cast<size_t>(cfg.past_k_slot)];
        s.Sk_past_rows = P.shape(head_seq_dim(s.kv_layout));
        s.Sk_past = s.Sk_past_rows;
    }
    return s;
}

} // namespace nnops::backend
