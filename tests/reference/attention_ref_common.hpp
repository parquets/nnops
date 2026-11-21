#pragma once
/// @file attention_ref_common.hpp
/// @brief Naive f32 attention, shared by the three operators' references.
///
/// Q/K/V arrive in one of three storage orders — merged [B, S, H*D] (rank 3),
/// onnxruntime's [B, H, S, D], or [B, S, H, D] when @p bshd is set — and K/V
/// carry their own flag (@p kv_bshd), so a merged query can pair with a BNSH
/// cache. Query head h reads kv head h / (H / H_kv), and when @p v_is_k the
/// value is the leading @p v_head_dim channels of the key row (MLA's latent)
/// rather than a third tensor.
///
/// Per head: the Q·K dot over the key width, scaled, capped when @p softcap is
/// set (cap·tanh(s/cap), before the mask, as Gemma2/3 do), plus the two masks —
/// a float additive attention_bias ([Sq, Sk] by row, or [B|1, H|1, Sq, Sk] with
/// the extent-1 axes broadcast) and an int32 key_padding_mask whose padded keys
/// take @p mask_filter_value — softmaxed per row over its valid range, then
/// multiplied by V into the output. A row whose valid range is empty is all
/// zeros.
///
/// The valid range of row i is [lo, hi): hi is the causal limit (the run, when
/// not causal) and lo is the sliding-window bound, pos_i - window + 1, on a
/// causal run. Both are clamped to the batch's run.
///
/// A seqlens entry is a length (the cache form) or, with
/// @p seqlens_plus_one, onnxruntime's last valid key index. A rank-1
/// key_padding_mask always carries lengths. The optional past segment is the
/// first Sk_past rows of the key run, so [past | new] behaves as one tensor
/// (MLA); @p position_offset is the absolute position of query row 0, negative
/// meaning bottom-right (run - Sq).
///
/// An int8 K/V cache stores symmetric integers: the fp value is the byte times
/// the external scale, one per tensor or one per (kv head, channel).
///
/// Varlen (the pair of int32 [segments + 1] cu_seqlens vectors) runs each packed
/// segment through a recursive call over views that keep the parent's row pitch,
/// with bottom-right alignment and no global position.
///
/// When @p gate is not None the finished output is multiplied element-wise by
/// act(gate), the gate tensor shaped like the output. It is applied once over
/// the whole output, after the varlen segments ran.
///
/// This is deliberately the slowest possible spelling — no blocking, no
/// packing — so it can serve as ground truth for the tiled kernels.

#include "nnops/core/tensor_view.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/epilogue.hpp"
#include "nnops/core/parallel_for.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace nnops::backend::cpu::reference::detail {

constexpr size_t kNoSlot = static_cast<size_t>(-1);

/// Where each optional input lives in the operator's input list.
struct AttentionRefSlots {
    size_t bias = kNoSlot;      ///< float additive attention_bias
    size_t sink = kNoSlot;      ///< per-head attention sink logit
    size_t kpm = kNoSlot;       ///< int32 key_padding_mask (lengths or keep mask)
    size_t past_k = kNoSlot;
    size_t past_v = kNoSlot;
    size_t past_len = kNoSlot;  ///< int32 scalar narrowing a past buffer
    size_t seqlens = kNoSlot;   ///< int32 [B], a run length or ORT's last index
    size_t pos_ids = kNoSlot;   ///< int64 [B], the first query row of each batch
    size_t k_scale = kNoSlot;   ///< f32 dequantization scale for an int8 K
    size_t v_scale = kNoSlot;   ///< f32 dequantization scale for an int8 V
    size_t cu_q = kNoSlot;
    size_t cu_k = kNoSlot;
    size_t gate = kNoSlot;
};

inline bool ref_slot_has(std::span<const TensorView> inputs, size_t s)
{
    return s != kNoSlot && inputs.size() > s && inputs[s].rank() != 0;
}

/// Scalar activation of the output gate: act(gate).
inline float attention_ref_gate_act(EpilogueActivateType a, float g) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    switch (a) {
    case EpilogueActivateType::Silu:    return g / (1.0f + std::exp(-g));
    case EpilogueActivateType::Gelu:    return 0.5f * g *
        (1.0f + std::tanh(kSqrt2OverPi * (g + 0.044715f * g * g * g)));
    case EpilogueActivateType::Relu:    return g > 0.0f ? g : 0.0f;
    case EpilogueActivateType::Sigmoid: return 1.0f / (1.0f + std::exp(-g));
    case EpilogueActivateType::Tanh:    return std::tanh(g);
    default:                            return g;
    }
}

/// out *= act(gate), one flat row walk over the output's pitch.
inline void attention_ref_apply_gate(EpilogueActivateType gate,
                                     TensorView& output,
                                     std::span<const TensorView> inputs,
                                     size_t gate_slot,
                                     const ComputeContext& ctx)
{
    const auto& G = inputs[gate_slot];
    const int64_t n = output.shape(output.rank() - 1);
    const int64_t rows = output.numel() / n;
    const int64_t o_rs = output.row_stride_elems();
    const int64_t g_rs = G.row_stride_elems();
    float* op = output.ptr<float>();
    const float* gp = G.ptr<float>();
    ctx.cpu.run(0, rows, [&](int64_t r) {
        float* y = op + r * o_rs;
        const float* g = gp + r * g_rs;
        for (int64_t i = 0; i < n; ++i) {
            y[i] *= attention_ref_gate_act(gate, g[i]);
        }
    });
}

inline int64_t ref_seq_dim(const TensorView& q, bool bshd)
{
    return (q.rank() == 3 || bshd) ? 1 : 2;
}

/// Elements from one sequence position to the next within one head. The flag
/// describes this tensor's own rank-4 order, so a merged Q pairs with a BNSH
/// cache just as well as with a matching K.
inline int64_t ref_seq_step(const TensorView& v, bool bshd)
{
    const bool merged = (v.rank() == 3);
    return (!merged && bshd) ? v.stride_elems(1) : v.stride_elems(v.rank() - 2);
}

/// Element steps from one batch and one head to the next; @p d is the head
/// width for the merged form, whose heads are packed inside the row.
inline void ref_head_strides(const TensorView& v, bool bshd, int64_t d, int64_t& batch,
                             int64_t& head)
{
    batch = v.stride_elems(0);
    if (v.rank() == 3) {
        head = d;
    } else {
        head = v.stride_elems(bshd ? 2 : 1);
    }
}

inline void attention_ref_generic(int64_t num_heads, int64_t num_kv_heads,
                                  int64_t v_head_dim,
                                  float scale, bool causal, bool v_is_k,
                                  int64_t position_offset, bool bshd, bool kv_bshd,
                                  const AttentionRefSlots& slots, float mask_filter_value,
                                  TensorView& output,
                                  std::span<const TensorView> inputs,
                                  const ComputeContext& ctx,
                                  float softcap = 0.0f, int64_t window = 0,
                                  EpilogueActivateType gate = EpilogueActivateType::None,
                                  bool seqlens_plus_one = true, int64_t sink_tokens = 0)
{
    const auto& Q = inputs[0];
    const auto& K = inputs[1];
    const auto& V = v_is_k ? inputs[1] : inputs[2];
    const bool has_gate = gate != EpilogueActivateType::None &&
                          ref_slot_has(inputs, slots.gate);
    const bool has_seqlens = ref_slot_has(inputs, slots.seqlens);
    const bool has_past = ref_slot_has(inputs, slots.past_k);
    const auto& Kp = has_past ? inputs[slots.past_k] : K;
    const auto& Vp = has_past ? inputs[slots.past_v] : V;

    const int64_t sd = ref_seq_dim(Q, bshd);
    const int64_t ksd = ref_seq_dim(K, kv_bshd);
    const int64_t B   = Q.shape(0);
    const int64_t H   = num_heads;
    const int64_t Hkv = num_kv_heads > 0 ? num_kv_heads : num_heads;

    // Varlen: each packed segment is its own [1, len, W] problem over a view of
    // the parent's storage, so nothing is copied.
    if (ref_slot_has(inputs, slots.cu_q)) {
        const auto& cq = inputs[slots.cu_q];
        const auto& ck = inputs[slots.cu_k];
        const int32_t* qo = cq.ptr<int32_t>();
        const int32_t* ko = ck.ptr<int32_t>();
        const auto sub = [](const TensorView& p, int64_t row_off, int64_t rows) {
            const int64_t dims[3] = {1, rows, p.shape(2)};
            char* base = const_cast<char*>(p.ptr<char>()) + row_off * p.pitch();
            return TensorView(std::span<const int64_t>(dims, 3), p.data_type(),
                              base, p.pitch(), p.layout());
        };
        for (int64_t seg = 0; seg + 1 < cq.shape(0); ++seg) {
            const int64_t len_q = qo[seg + 1] - qo[seg];
            const int64_t len_k = ko[seg + 1] - ko[seg];
            if (len_q == 0) { continue; }
            TensorView qs = sub(Q, qo[seg], len_q);
            TensorView ks = sub(K, ko[seg], len_k);
            TensorView vs = v_is_k ? ks : sub(V, ko[seg], len_k);
            TensorView os = sub(output, qo[seg], len_q);
            const TensorView in3[3] = {qs, ks, vs};
            attention_ref_generic(num_heads, num_kv_heads, v_head_dim, scale,
                                  causal, v_is_k, /*position_offset=*/-1, /*bshd=*/false,
                                  /*kv_bshd=*/false, AttentionRefSlots{}, mask_filter_value,
                                  os, in3, ctx, softcap, window);
        }
        // The gate belongs to the parent output, not to a segment.
        if (has_gate) {
            attention_ref_apply_gate(gate, output, inputs, slots.gate, ctx);
        }
        return;
    }

    const int64_t Sq = Q.shape(static_cast<size_t>(sd));
    const int64_t Sk_new = K.shape(static_cast<size_t>(ksd));
    const int64_t Sv = V.shape(static_cast<size_t>(ksd));
    const int64_t Dqk = (Q.rank() == 3) ? Q.shape(2) / H : Q.shape(3);
    const int64_t Sk_past_rows = has_past ? Kp.shape(static_cast<size_t>(ksd)) : 0;
    int64_t Sk_past = Sk_past_rows;
    if (has_past && ref_slot_has(inputs, slots.past_len)) {
        const int64_t v = inputs[slots.past_len].ptr<int32_t>()[0];
        Sk_past = std::max<int64_t>(0, std::min<int64_t>(v, Sk_past_rows));
    }
    const int64_t Sk = Sk_new + Sk_past;
    const int64_t Dv = v_head_dim > 0 ? v_head_dim : Dqk;

    const float s = (scale == 0.0f)
        ? (1.0f / std::sqrt(static_cast<float>(Dqk)))
        : scale;

    // An int8 cache stores symmetric integers; the fp value is the byte times
    // the external scale (per tensor, or per kv head and channel).
    const bool kv8 = K.data_type() == DataType::s8;
    const auto* q_ptr = Q.ptr<float>();
    const auto* k8_ptr = kv8 ? K.ptr<int8_t>() : nullptr;
    const auto* v8_ptr = kv8 ? V.ptr<int8_t>() : nullptr;
    const auto* k_ptr = kv8 ? nullptr : K.ptr<float>();
    const auto* v_ptr = kv8 ? nullptr : V.ptr<float>();
    const auto* kp_ptr = has_past ? Kp.ptr<float>() : nullptr;
    const auto* vp_ptr = has_past ? Vp.ptr<float>() : nullptr;
    const float* ksc = ref_slot_has(inputs, slots.k_scale) ?
                           inputs[slots.k_scale].ptr<float>() : nullptr;
    const float* vsc = ref_slot_has(inputs, slots.v_scale) ?
                           inputs[slots.v_scale].ptr<float>() : nullptr;
    const bool ksc_pc = ksc != nullptr && inputs[slots.k_scale].numel() > 1;
    const bool vsc_pc = vsc != nullptr && inputs[slots.v_scale].numel() > 1;
    const bool has_bias = ref_slot_has(inputs, slots.bias);
    const auto* sink_ptr = ref_slot_has(inputs, slots.sink) ? inputs[slots.sink].ptr<float>()
                                                            : nullptr;
    const auto* bias_ptr = has_bias ? inputs[slots.bias].ptr<float>() : nullptr;
    const bool has_kpm = ref_slot_has(inputs, slots.kpm) &&
                         inputs[slots.kpm].rank() >= 2;
    const auto* kpm_ptr = has_kpm ? inputs[slots.kpm].ptr<int32_t>() : nullptr;
    const int32_t* seqlens = has_seqlens ? inputs[slots.seqlens].ptr<int32_t>() : nullptr;
    // A rank-1 key_padding_mask carries lengths instead of a keep mask.
    if (!seqlens && ref_slot_has(inputs, slots.kpm) && inputs[slots.kpm].rank() == 1) {
        seqlens = inputs[slots.kpm].ptr<int32_t>();
    }
    const int64_t* pos_ids = ref_slot_has(inputs, slots.pos_ids)
        ? inputs[slots.pos_ids].ptr<int64_t>()
        : nullptr;
    auto* out_ptr = output.ptr<float>();

    const int64_t q_rs = ref_seq_step(Q, bshd);
    const int64_t k_rs = ref_seq_step(K, kv_bshd);
    const int64_t v_rs = ref_seq_step(V, kv_bshd);
    const int64_t kp_rs = has_past ? ref_seq_step(Kp, kv_bshd) : k_rs;
    const int64_t vp_rs = has_past ? ref_seq_step(Vp, kv_bshd) : v_rs;
    const int64_t o_rs = ref_seq_step(output, bshd);
    // A latent key row is Dqk wide per head; the value is its prefix.
    const int64_t v_head_stride = v_is_k ? Dqk : Dv;

    int64_t q_b, q_h, k_b, k_h, v_b, v_h, o_b, o_h;
    ref_head_strides(Q, bshd, Dqk, q_b, q_h);
    ref_head_strides(K, kv_bshd, Dqk, k_b, k_h);
    ref_head_strides(V, kv_bshd, v_head_stride, v_b, v_h);
    ref_head_strides(output, bshd, Dv, o_b, o_h);
    int64_t kp_b = 0, kp_h = 0, vp_b = 0, vp_h = 0;
    if (has_past) {
        ref_head_strides(Kp, kv_bshd, Dqk, kp_b, kp_h);
        ref_head_strides(Vp, kv_bshd, v_head_stride, vp_b, vp_h);
    }

    const auto bias_off = [&](int64_t b, int64_t h, int64_t i) {
        const auto& M = inputs[slots.bias];
        if (M.rank() == 2) {
            return i * M.stride_elems(0);
        }
        const int64_t bs = M.shape(0) == 1 ? 0 : M.stride_elems(0);
        const int64_t hs = M.shape(1) == 1 ? 0 : M.stride_elems(1);
        return b * bs + h * hs + i * M.stride_elems(2);
    };
    const auto kpm_off = [&](int64_t b, int64_t i) {
        const auto& M = inputs[slots.kpm];
        const int64_t bs = M.stride_elems(0);
        const int64_t rs = M.rank() == 3 ? M.stride_elems(1) : 0;
        return b * bs + i * rs;
    };

    const auto compute_head_batch = [&](int64_t idx) {
        const int64_t b   = idx / H;
        const int64_t h   = idx % H;
        const int64_t hkv = h / (H / Hkv);

        const float sink = (sink_ptr != nullptr) ? sink_ptr[h]
                                                 : -std::numeric_limits<float>::infinity();
        const float* q_head = q_ptr + b * q_b + h * q_h;
        const float* k_head = kv8 ? nullptr : k_ptr + b * k_b + hkv * k_h;
        const float* v_head = kv8 ? nullptr : v_ptr + b * v_b + hkv * v_h;
        const int8_t* k8_head = kv8 ? k8_ptr + b * k_b + hkv * k_h : nullptr;
        const int8_t* v8_head = kv8 ? v8_ptr + b * v_b + hkv * v_h : nullptr;
        // The int8 scales are per token ([B, Hkv, Sk], the row of this batch
        // and kv head) or per tensor (one value).
        const float k_scalar = (ksc != nullptr && !ksc_pc) ? ksc[0] : 1.0f;
        const float v_scalar = (vsc != nullptr && !vsc_pc) ? vsc[0] : 1.0f;
        const float* k_sc_row = (ksc != nullptr && ksc_pc) ? ksc + (b * Hkv + hkv) * Sk : nullptr;
        const float* v_sc_row = (vsc != nullptr && vsc_pc) ? vsc + (b * Hkv + hkv) * Sk : nullptr;
        float* o_head = out_ptr + b * o_b + h * o_h;
        const float* kp_head = has_past ? kp_ptr + b * kp_b + hkv * kp_h : nullptr;
        const float* vp_head = has_past ? vp_ptr + b * vp_b + hkv * vp_h : nullptr;

        const auto k_at = [&](int64_t j, int64_t d) -> float {
            if (j < Sk_past) {
                return kp_head[j * kp_rs + d];
            }
            if (kv8) {
                const float sc = (k_sc_row != nullptr) ? k_sc_row[j] : k_scalar;
                return static_cast<float>(k8_head[(j - Sk_past) * k_rs + d]) * sc;
            }
            return k_head[(j - Sk_past) * k_rs + d];
        };
        const auto v_at = [&](int64_t j, int64_t d) -> float {
            if (j < Sk_past) {
                return vp_head[j * vp_rs + d];
            }
            if (kv8) {
                const float sc = (v_sc_row != nullptr) ? v_sc_row[j] : v_scalar;
                return static_cast<float>(v8_head[(j - Sk_past) * v_rs + d]) * sc;
            }
            return v_head[(j - Sk_past) * v_rs + d];
        };

        std::vector<float> scores(static_cast<size_t>(Sq * Sk), 0.0f);

        for (int64_t i = 0; i < Sq; ++i) {
            for (int64_t j = 0; j < Sk; ++j) {
                float dot = 0.0f;
                for (int64_t d = 0; d < Dqk; ++d) {
                    dot += q_head[i * q_rs + d] * k_at(j, d);
                }
                scores[static_cast<size_t>(i * Sk + j)] = dot * s;
            }
        }

        // The softcap owns the scaled logit and runs before the masks.
        if (softcap > 0.0f) {
            for (auto& v : scores) { v = softcap * std::tanh(v / softcap); }
        }

        for (int64_t i = 0; i < Sq; ++i) {
            float* row = scores.data() + i * Sk;
            if (kpm_ptr) {
                const int64_t off = kpm_off(b, i);
                for (int64_t j = 0; j < Sk; ++j) {
                    if (kpm_ptr[off + j] == 0) { row[j] = mask_filter_value; }
                }
            }
            if (bias_ptr) {
                const int64_t off = bias_off(b, h, i);
                for (int64_t j = 0; j < Sk; ++j) { row[j] += bias_ptr[off + j]; }
            }
        }

        // The batch's valid key-run prefix. A seqlens entry is a length in the
        // cache form and onnxruntime's last valid index otherwise; a rank-1
        // mask always holds the length.
        int64_t run = Sk;
        if (seqlens) {
            int64_t v = static_cast<int64_t>(seqlens[b]);
            if (has_seqlens && seqlens_plus_one) {
                ++v;
            }
            run = std::max<int64_t>(0, std::min<int64_t>(v, Sk));
        }

        // Valid range from the row's absolute position; the rest is never read.
        // Sinks keep [0, prefix) visible, so the range is [0, prefix) ∪ [lo, n)
        // with the hole [prefix, lo) hidden.
        const int64_t prefix = (causal && window > 0 && sink_tokens > 0)
            ? std::min<int64_t>(sink_tokens, run)
            : 0;
        // The batch's first-row position: the explicit vector, else the scalar,
        // else bottom-right.
        const int64_t q0b = pos_ids != nullptr
            ? pos_ids[b]
            : (position_offset >= 0 ? position_offset : (run - Sq));
        for (int64_t i = 0; i < Sq; ++i) {
            int64_t n = run;
            if (causal) {
                n = q0b + i + 1;
                n = std::max<int64_t>(0, std::min<int64_t>(n, run));
            }
            int64_t lo = 0;
            if (causal && window > 0) {
                lo = q0b + i - window + 1;
                lo = std::max<int64_t>(0, std::min<int64_t>(lo, run));
            }
            float* row = scores.data() + i * Sk;

            const int64_t a0 = 0;
            const int64_t b0 = (lo > prefix) ? prefix : n;
            const int64_t a1 = (lo > prefix) ? lo : n;
            const int64_t b1 = n;
            if (b0 <= a0 && b1 <= a1) {
                for (int64_t j = 0; j < Sk; ++j) { row[j] = 0.0f; }
                continue;
            }

            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t j = a0; j < b0; ++j) { max_val = std::max(max_val, row[j]); }
            for (int64_t j = a1; j < b1; ++j) { max_val = std::max(max_val, row[j]); }
            max_val = std::max(max_val, sink);

            float sum = 0.0f;
            for (int64_t j = a0; j < b0; ++j) {
                row[j] = std::exp(row[j] - max_val);
                sum += row[j];
            }
            for (int64_t j = a1; j < b1; ++j) {
                row[j] = std::exp(row[j] - max_val);
                sum += row[j];
            }
            sum += std::exp(sink - max_val);   // the sink's weight, value zero
            const float inv_sum = 1.0f / sum;
            for (int64_t j = a0; j < b0; ++j) { row[j] *= inv_sum; }
            for (int64_t j = a1; j < b1; ++j) { row[j] *= inv_sum; }
            for (int64_t j = b0; j < a1; ++j) { row[j] = 0.0f; }
            for (int64_t j = b1; j < Sk; ++j) { row[j] = 0.0f; }
        }

        for (int64_t i = 0; i < Sq; ++i) {
            for (int64_t d = 0; d < Dv; ++d) {
                float sum = 0.0f;
                for (int64_t j = 0; j < Sk; ++j) {
                    const float attn = scores[static_cast<size_t>(i * Sk + j)];
                    if (std::isinf(attn) || std::isnan(attn)) { continue; }
                    sum += attn * v_at(j, d);
                }
                o_head[i * o_rs + d] = sum;
            }
        }
    };

    const int64_t total = B * H;
    ctx.cpu.run(0, total, [&](int64_t idx) { compute_head_batch(idx); });

    if (has_gate) {
        attention_ref_apply_gate(gate, output, inputs, slots.gate, ctx);
    }
}

}  // namespace nnops::backend::cpu::reference::detail
