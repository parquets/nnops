#pragma once
/// @file attention_internal.h
/// @brief Machinery the attention routes and their plan share.
///
/// attention.h describes the routes, the head mapping and the scratch contract;
/// this header is the implementation side of that contract. It is included only
/// by the attention translation units, which are split by route:
///
///   attention_standard.cpp  — materialized scores, Q·Kᵀ → softmax → ·V
///   attention_flash.cpp     — FlashAttention-1 online softmax over KV tiles
///   attention_decode.cpp    — the Sq == 1 GEMV route, head batching + KV split
///   attention_plan.cpp      — get_attention_plan: tile sizes and scratch bytes
///   attention.cpp           — attention_cpu and the three operator entries
///
/// What lives here is what those files must agree on: the dtype tiling
/// constants, head-shape extraction, the scratch-slot rule (a kernel's slot
/// index and the plan's slot count are one decision), the causal row limit, and
/// the route geometry both the plan and the kernels size themselves from.

#include "attention.h"
#include "backend/cpu/common/gate_combine.hpp" // gate_scale_row, the output gate
#include "backend/cpu/matmul/matmul_helper.h"  // tile_mma_direct / tile_pack_rhs + panel constants
#include "backend/cpu/simd_kernel/activation_kernels.hpp" // v_tanh, for softcapping
#include "backend/cpu/simd_kernel/simd_int8_lane.hpp"     // v_cvt_i32_to_f32 (int8 QK dequant)
#include "backend/cpu/simd_kernel/simd_softmax.hpp"       // softmax + FlashAttention row kernels
#include "backend/cpu/simd_kernel/tiled_map.hpp"          // simd_vec_t
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace nnops::backend::cpu {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr float kPosInf = std::numeric_limits<float>::infinity();

inline size_t align_up_bytes(size_t n, size_t a)
{
    return (n + a - 1) & ~(a - 1);
}

// ---- dtype-selected tiling constants -------------------------------------
// k-block length per dtype, mirroring matmul_helper.h so plan and kernels agree.

template <class T> constexpr int64_t kc_for() { return std::is_same_v<T, half> ? KC_F16 : KC_F32; }

/// Size bound for the packed-B scratch's k-run: min(Kc, k). The loops still
/// step by Kc; pass the longer of the two GEMMs' k.
template <class T> constexpr int64_t kc_pack_bound(int64_t k)
{
    constexpr int64_t kc = std::is_same_v<T, half> ? KC_F16 : KC_F32;
    return k < kc ? k : kc;
}

/// Panel count a uniform packed-B tile of any n <= n_max emits.
template <class T> constexpr int panels_for_pack(int64_t n_max)
{
    constexpr int nr_max = nr_max_flt<T>();
    return static_cast<int>((n_max + nr_max - 1) / nr_max);
}

// ---- head-shape extraction -------------------------------------------------
//
// Three storage orders describe the same head geometry: merged [B, S, H*D]
// (rank 3, heads packed inside the row), onnxruntime's BNSH [B, H, S, D], and
// HuggingFace's BSHD [B, S, H, D]. The last two share a rank, so the operator
// front-end picks between them; the routes only see the layout and address
// through strides, never through a per-layout formula.

enum class HeadLayout {
    Merged, ///< rank 3: [B, S, H*D]
    BHSD,   ///< rank 4: [B, H, S, D], onnxruntime's cache-major form
    BSHD    ///< rank 4: [B, S, H, D]
};

struct HeadShape {
    int64_t B, H, Hkv, Sq, Sk, Sv, Dqk, Dv;
    int64_t Sk_past = 0;      ///< valid rows in the past segment
    int64_t Sk_past_rows = 0; ///< rows the past tensor carries (its capacity)
    HeadLayout q_layout = HeadLayout::Merged;  ///< Q and the output
    HeadLayout kv_layout = HeadLayout::Merged; ///< K, V and the past pair

    /// Q's heads live inside the row (merged / BSHD), not in their own axis.
    bool row_heads() const noexcept { return q_layout != HeadLayout::BHSD; }
};

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

/// A slot carries an activation (Q/K/V-like dtype) rather than int32 metadata.
inline bool slot_is_activation(const AttentionConfig& cfg, int slot)
{
    return slot == 0 || slot == 1 || (slot == 2 && !cfg.v_is_k) || slot == cfg.past_k_slot ||
           slot == cfg.past_v_slot || slot == cfg.bias_slot || slot == cfg.gate_slot;
}

inline HeadShape resolve_head_shape(const AttentionConfig& cfg, std::span<const TensorView> inputs)
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

/// Element strides from one batch and one head to the next. @p d is the head
/// width for the merged form, whose heads are packed inside the row.
inline void head_strides(const TensorView& v, HeadLayout layout, int64_t d,
                         int64_t& batch_step, int64_t& head_step)
{
    batch_step = v.stride_elems(0);
    head_step = (layout == HeadLayout::Merged)    ? d :
                (layout == HeadLayout::BHSD)      ? v.stride_elems(1) :
                                                    v.stride_elems(2);
}


/// Elements from one sequence position to the next *within one head*. That is
/// the row pitch for merged and [B, H, S, D] — there the row is exactly one
/// head's slice — but for [B, S, H, D] a row is the head dim alone, so the step
/// is the whole head block.
inline int64_t head_seq_step(const TensorView& v, HeadLayout layout)
{
    return layout == HeadLayout::BSHD ? v.stride_elems(1) : v.row_stride_elems();
}

/// Offset of the value head inside a merged row: a latent K row is Dqk wide
/// per head even though only Dv of those columns are values.
inline int64_t merged_v_head_stride(const AttentionConfig& cfg, int64_t Dqk, int64_t Dv)
{
    return cfg.v_is_k ? Dqk : Dv;
}

// ---- per-batch key-run length ---------------------------------------------
//
// Two inputs can shorten a batch's run. `seqlens_k` (GQA/MLA) stores the last
// valid key index, onnxruntime's convention, so the run is entry + 1; a
// key_padding_mask that carries per-batch lengths stores the length itself.
// Either way the valid keys are the run's prefix, so right padding is truncated
// and every batch may sit at its own position.

inline int64_t run_length(const int32_t* seqlens, int64_t b, int64_t Sk, bool plus_one)
{
    if (seqlens == nullptr) {
        return Sk;
    }
    int64_t v = static_cast<int64_t>(seqlens[b]);
    if (plus_one) {
        ++v;
    }
    if (v <= 0) {
        return 0;
    }
    return v < Sk ? v : Sk;
}

/// The pointer the routes ask for batch lengths through: seqlens_k first, else
/// a rank-1 key_padding_mask (a [B] length vector in onnxruntime's MHA).
inline const int32_t* run_lengths_ptr(const AttentionConfig& cfg,
                                      std::span<const TensorView> inputs, bool& plus_one)
{
    if (slot_present(cfg, cfg.seqlens_slot, inputs)) {
        plus_one = !cfg.seqlens_is_length;
        return inputs[static_cast<size_t>(cfg.seqlens_slot)].ptr<int32_t>();
    }
    if (slot_present(cfg, cfg.kpm_slot, inputs) &&
        inputs[static_cast<size_t>(cfg.kpm_slot)].rank() == 1) {
        plus_one = false;
        return inputs[static_cast<size_t>(cfg.kpm_slot)].ptr<int32_t>();
    }
    plus_one = true;
    return nullptr;
}

// ---- varlen (packed segments) --------------------------------------------

/// cu_seqlens_q / cu_seqlens_k are int32 [segments + 1] prefix offsets into the
/// packed query and key rows. Both or neither, and exclusive with the masks,
/// the past pair, seqlens_k and position_offset.
inline bool has_varlen(const AttentionConfig& cfg, std::span<const TensorView> inputs)
{
    return slot_present(cfg, cfg.cu_q_slot, inputs);
}

// ---- masks -----------------------------------------------------------------
//
// Two mask inputs, both folded into the score row before the softmax:
//   attention_bias     float additive, rank 2 [Sq, Sk] shared by every batch
//                      and head, or rank 4 [B|1, H|1, Sq, Sk] (extent 1
//                      broadcasts), matching onnxruntime.
//   key_padding_mask   int32, nonzero = keep (onnxruntime's MHA form); a padded
//                      key takes cfg.mask_filter_value. Rank 1 [B] is a length
//                      vector instead, which run_lengths_ptr reads.

struct MaskView {
    const void* bias = nullptr; ///< T additive attention_bias
    int64_t bias_b = 0, bias_h = 0, bias_row = 0;
    const int32_t* keep = nullptr; ///< int32 key_padding_mask, nonzero = keep
    int64_t keep_b = 0, keep_h = 0, keep_row = 0;
    float filter = -10000.0f;

    bool any() const noexcept { return bias != nullptr || keep != nullptr; }
};

inline int64_t mask_offset(int64_t b, int64_t h, int64_t i, int64_t b_step, int64_t h_step,
                           int64_t row_step)
{
    return b * b_step + h * h_step + i * row_step;
}

inline MaskView resolve_mask(const AttentionConfig& cfg, std::span<const TensorView> inputs)
{
    MaskView m;
    m.filter = cfg.mask_filter_value;
    if (slot_present(cfg, cfg.bias_slot, inputs)) {
        const auto& v = inputs[static_cast<size_t>(cfg.bias_slot)];
        m.bias = v.ptr<char>();
        if (v.rank() == 2) {
            // One row per query row, shared by every batch and head.
            m.bias_row = v.stride_elems(0);
        } else {
            m.bias_b = v.shape(0) == 1 ? 0 : v.stride_elems(0);
            m.bias_h = v.shape(1) == 1 ? 0 : v.stride_elems(1);
            m.bias_row = v.stride_elems(2);
        }
    }
    if (slot_present(cfg, cfg.kpm_slot, inputs) &&
        inputs[static_cast<size_t>(cfg.kpm_slot)].rank() >= 2) {
        const auto& v = inputs[static_cast<size_t>(cfg.kpm_slot)];
        m.keep = v.ptr<int32_t>();
        m.keep_b = v.stride_elems(0);
        // A [B, Sk] mask is the same row for every query row of the batch.
        m.keep_row = v.rank() == 3 ? v.stride_elems(1) : 0;
    }
    return m;
}

/// A mask is present only when a view carries data: a [B] key_padding_mask is a
/// length vector, not a mask.
inline bool has_mask_input(const AttentionConfig& cfg, std::span<const TensorView> inputs)
{
    return resolve_mask(cfg, inputs).any();
}

/// Apply a key_padding_mask to one score row: a padded key takes the filter
/// value (onnxruntime's mask_filter_value, not -inf).
template <class T> inline void row_apply_keep(T* row, const int32_t* keep, int64_t n, float filter)
{
    for (int64_t i = 0; i < n; ++i) {
        if (keep[i] == 0) {
            simd::s_store(&row[i], filter);
        }
    }
}

// ---- int8 KV cache ------------------------------------------------------
//
// A quantized cache stores symmetric integers, and the value of cache row t is
// its stored byte times a scale. Two granularities are supported: per-token —
// one f32 per (batch, kv head, cache row), [B, H_kv, Sk] — and per-tensor
// (numel 1). The writer owns the scales (it appends and quantizes), and they
// stay fixed while the step reads. The query stays fp: the K scale multiplies
// into the score and the V scale into the probabilities, so the cache is never
// widened into an fp copy.

/// One (batch, kv head)'s scale row, indexed by the cache row.
struct KvScaleRow {
    const float* ptr = nullptr; ///< null: an fp cache, no scaling
    int64_t step = 1;           ///< 0 = per-tensor (one value for every token)

    float at(int64_t t) const noexcept
    {
        return ptr != nullptr ? ptr[t * step] : 1.0f;
    }
    /// The same row re-based at token @p t (a no-op for per-tensor).
    KvScaleRow shifted(int64_t t) const noexcept
    {
        return {step != 0 && ptr != nullptr ? ptr + t : ptr, step};
    }
};

struct KvScale {
    const float* ptr = nullptr; ///< null: an fp cache
    bool per_token = false;

    KvScaleRow row(int64_t b, int64_t hkv, int64_t Hkv, int64_t Sk) const noexcept
    {
        if (ptr == nullptr) {
            return {};
        }
        if (!per_token) {
            return {ptr, 0};
        }
        return {ptr + (b * Hkv + hkv) * Sk, 1};
    }
};

inline KvScale resolve_kv_scale(const AttentionConfig& cfg, int slot,
                                std::span<const TensorView> inputs)
{
    KvScale s;
    if (!slot_present(cfg, slot, inputs)) {
        return s;
    }
    const auto& v = inputs[static_cast<size_t>(slot)];
    s.ptr = v.ptr<float>();
    s.per_token = v.numel() > 1;
    return s;
}

/// The cache form stores int8 K/V while the query stays fp.
inline bool kv_is_int8(std::span<const TensorView> inputs)
{
    return inputs.size() > 1 && !inputs[1].is_empty() &&
           inputs[1].data_type() == DataType::s8;
}

/// Dequantization scratch one packed panel of an int8 pack needs.
template <class T> constexpr int64_t kv_i8_panel_elems(int64_t kc_bound)
{
    return static_cast<int64_t>(nr_max_flt<T>()) * kc_bound;
}

/// Pack an int8 K/V tile into the uniform nr_max-wide panels tile_mma_direct
/// reads, dequantizing on the way; byte-identical layout to tile_pack_rhs.
/// Every source row is one key, so @p row_scale is that key's scale from the
/// block's first key on: K's `trans` form reads the keys as n rows, V's copy
/// form as k rows. @p alpha rides along like the fp pack's.
template <class T>
void kv_pack_i8(bool trans, int nc, int kc, T* dst, int ldd, const int8_t* src, int lds,
                const KvScaleRow& row_scale, float alpha, T* temp)
{
    constexpr int nr = nr_max_flt<T>();
    constexpr int L = 8; // int8 -> f32 conversion width
    const int src_step = trans ? lds : 1;

    const auto scale_of = [&](int64_t t) { return row_scale.at(t) * alpha; };

    for (int n0 = 0; n0 < nc; n0 += nr) {
        const int w = std::min(nr, nc - n0);
        if (trans) {
            for (int n = 0; n < w; ++n) {
                const int8_t* s = src + static_cast<int64_t>(n) * lds;
                T* row = temp + static_cast<int64_t>(n) * kc;
                const float sc = scale_of(n0 + n);
                const simd::v_f32x8 sv = simd::v_set1_f32x8(sc);
                int k = 0;
                for (; k + L <= kc; k += L) {
                    simd::v_store_f32(row + k, simd::v_mul(simd::v_cvt_s8_to_f32(s + k), sv));
                }
                for (; k < kc; ++k) {
                    simd::s_store(&row[k], static_cast<float>(s[k]) * sc);
                }
            }
            tile_pack_rhs(true, w, kc, dst, ldd, temp, kc, 1.0f);
        } else {
            for (int k = 0; k < kc; ++k) {
                const int8_t* s = src + static_cast<int64_t>(k) * lds;
                T* row = temp + static_cast<int64_t>(k) * nr;
                const float sc = scale_of(k);
                const simd::v_f32x8 sv = simd::v_set1_f32x8(sc);
                int n = 0;
                for (; n + L <= w; n += L) {
                    simd::v_store_f32(row + n, simd::v_mul(simd::v_cvt_s8_to_f32(s + n), sv));
                }
                for (; n < w; ++n) {
                    simd::s_store(&row[n], static_cast<float>(s[n]) * sc);
                }
            }
            tile_pack_rhs(false, w, kc, dst, ldd, temp, nr, 1.0f);
        }
        dst += ldd;
        src += static_cast<int64_t>(w) * src_step;
    }
}

// ---- dynamic-query int8 Q·Kᵀ --------------------------------------------
//
// The query is quantized on the fly, one symmetric scale per row, and the
// product runs on the integer MMA (VNNI u8×s8 on x86, SDOT/i8mm on aarch64)
// with the cache's own s8 keys. int32 scores are then scaled back by
// `scale · s_q[row] · s_k[key]` — the k scale can only ride the score, because
// it varies along the reduction's *output* axis (per key), not along K.

/// Whether the int8 P·V path can run: asked for, over an s8 cache, and with a
/// key tile the probability pack takes as one k block.
inline bool pv_i8_enabled(const AttentionConfig& cfg, bool kv8, int Bc)
{
    return cfg.quantize_pv && kv8 && Bc > 0 && Bc <= KC_CAP_I8;
}

/// Whether the int8 Q·Kᵀ path can run: asked for, over an s8 cache, with a
/// query-key width that divides the integer pack's k group (4, eight on the
/// runtime i8mm route) and inside the pack's one-k-block cap. Everything else
/// runs the fp route, which needs none of those.
inline bool qk_i8_enabled(const AttentionConfig& cfg, bool kv8, int64_t Dqk)
{
    return cfg.quantize_q && kv8 && Dqk > 0 && Dqk <= KC_CAP_I8 &&
           i8_k_round64(Dqk) == Dqk;
}

/// M panels walk_levels_exact emits for @p rows on the int8 ladder.
inline int i8_m_panels(int rows)
{
    const int l0 = MR_I8[0], l1 = MR_I8[1];
    const int greedy = rows / l0;
    int best = rows + 1;
    for (int q0 = greedy; q0 >= 0 && q0 >= greedy - 2; --q0) {
        const int rest = rows - q0 * l0;
        best = std::min(best, q0 + rest / l1 + rest % l1);
    }
    return best;
}

/// The most panels any row count up to @p rows emits: the count is not
/// monotonic (63 rows is 12 panels, 64 is 11 — a three-row tail beats a
/// six-row panel), and a task's last query block is exactly such a tail. A
/// packed query block is sized by this, so no tail can overrun the slot.
inline int i8_m_panels_max(int rows)
{
    int best = 1;
    for (int r = 1; r <= rows; ++r) {
        best = std::max(best, i8_m_panels(r));
    }
    return best;
}

/// A's correction block: the s8 query pack biases its bytes on x86 and neither
/// pack nor kernel needs the column sums on aarch64 (see int8_a_pack_bias).
constexpr bool kQkI8ABiased = int8_a_pack_bias(DataType::s8) != 0;

/// max |x| over a row, the dynamic quantization scale's numerator.
template <class T> inline float row_absmax(const T* x, int64_t n)
{
    using V = kernel::simd_vec_t<T>;
    constexpr int64_t L = simd::simd_lane_for<T>;
    int64_t i = 0;
    V acc = simd::v_zero(x);
    for (; i + L <= n; i += L) {
        acc = simd::v_max(acc, simd::v_abs(simd::v_load(x + i)));
    }
    float best = simd::v_reduce_max(acc);
    for (; i < n; ++i) {
        best = std::max(best, std::abs(simd::s_load(&x[i])));
    }
    return best;
}

// ---- int8 P·V -----------------------------------------------------------
//
// After the online-softmax subtraction the tile holds p = exp(s − m) ∈ [0,1],
// so the grid is static: 255 levels, no per-row scale and no extra maximum
// pass. The cache's per-key V scale is the one thing that cannot wait for the
// output — it sits on the reduction axis — so it rides on P instead:
// p·s_v[t]/max_t(s_v), still inside [0,1], with the tile's factor applied once
// on the way out. The denominator l stays the fp row sum taken before the
// quantization, so only the numerator's weights are rounded.

/// P's grid. u8 on both arches: P's domain is [0,1], so the unsigned range is
/// exactly that (IntAttention measured 0.9991 cos against 0.9966 for a signed
/// grid, and half the RMSE). x86's vpdpbusd takes the byte as it is; SDOT is
/// s8×s8, so aarch64 biases the pack by 0x80 — p̂ → p̂−128, the shift
/// int8_a_pack_bias applies to any unsigned A — and the product comes back off
/// by exactly +128·Σ_t v[t][d], which the dequantization folds out.
constexpr int kPvI8Levels = 255;
constexpr bool kPvI8BiasA = int8_a_pack_bias(DataType::u8) != 0;
constexpr int kPvI8ColCorrection = -int8_a_pack_bias(DataType::u8); // 0 / +128
static_assert(kPvI8BiasA == (kPvI8ColCorrection == 128),
              "the biased-P correction is exactly the pack shift the MMA does not fold");

/// Per-column sums of a tile of signed byte rows (src[t*lds + d]) — the exact
/// correction a biased-A product needs on an s8-only MMA. f32 holds them
/// exactly: at most Bc ≤ 256 rows of |v| ≤ 127.
inline void i8_col_sums(const int8_t* src, int64_t lds, int rows, int cols, float* sums)
{
    constexpr int L = 8; // int8 -> f32 conversion width
    int d = 0;
    for (; d + L <= cols; d += L) {
        simd::v_f32x8 acc = simd::v_set1_f32x8(0.0f);
        for (int t = 0; t < rows; ++t) {
            acc = simd::v_add(acc, simd::v_cvt_s8_to_f32(src + static_cast<int64_t>(t) * lds + d));
        }
        simd::v_store(sums + d, acc);
    }
    for (; d < cols; ++d) {
        int32_t s = 0;
        for (int t = 0; t < rows; ++t) {
            s += src[static_cast<int64_t>(t) * lds + d];
        }
        sums[d] = static_cast<float>(s);
    }
}

/// o[d] = (acc[d] + Corr·col_sum[d]) · f, added onto @p o unless @p add_to is
/// false (the very first tile of a task writes it).
template <class T, int Corr>
inline void pv_i8_accum_row(T* o, const int32_t* acc, const float* col_sum, int64_t n, float f,
                            bool add_to)
{
    constexpr int64_t L = simd::simd_default_lane_f32;
    const simd::v_f32x8 vf = simd::v_set1_f32x8(f);
    int64_t d = 0;
    for (; d + L <= n; d += L) {
        simd::v_f32x8 v = kernel::v_cvt_i32_to_f32(acc + d);
        if constexpr (Corr != 0) {
            v = simd::v_add(v, simd::v_mul(simd::v_set1_f32x8(static_cast<float>(Corr)),
                                           simd::v_load(col_sum + d)));
        }
        v = simd::v_mul(v, vf);
        if (add_to) {
            v = simd::v_add(v, simd::v_load_f32(o + d));
        }
        simd::v_store_f32<T>(o + d, v);
    }
    for (; d < n; ++d) {
        float v = static_cast<float>(acc[d]);
        if constexpr (Corr != 0) {
            v += static_cast<float>(Corr) * col_sum[d];
        }
        v *= f;
        simd::s_store(&o[d], add_to ? simd::s_load(&o[d]) + v : v);
    }
}

/// True when this build carries the ZA kernels at all.
#if defined(NNOPS_ARCH_SME)
inline constexpr bool kSmeBuilt = true;
#else
inline constexpr bool kSmeBuilt = false;
#endif

/// The ZA fp route serves f32 and f16 only; bf16 keeps the NEON engine.
template <class T> constexpr bool fp_sme_capable = std::is_same_v<T, float> ||
                                                  std::is_same_v<T, half>;

/// Elements one SME packed A slab of @p mc rows × @p kc K occupies. The fp
/// block is 32 rows (the 2x2 ZA tile), so this is the ZA pack's own geometry.
inline int64_t fp_sme_a_elems(DataType dt, int64_t mc, int64_t kc)
{
    if (dt != DataType::f32 && dt != DataType::f16) {
        return 0;
    }
    constexpr int64_t kBlock = 32;
    return ((mc + kBlock - 1) / kBlock) * sme_a_panel_stride(dt, kc);
}

/// Elements one SME packed B slab of @p nc columns × @p kc K occupies.
inline int64_t fp_sme_b_elems(DataType dt, int64_t nc, int64_t kc)
{
    if (dt != DataType::f32 && dt != DataType::f16) {
        return 0;
    }
    constexpr int64_t kNr = 32;
    return ((nc + kNr - 1) / kNr) * sme_b_panel_stride(dt, kc);
}

/// Bytes one SME packed int8 A slab of @p mc rows × @p kc K occupies; the
/// SMOPA A block is 16 rows and groups K in fours.
inline int64_t sme_i8_a_bytes(int64_t mc, int64_t kc)
{
    constexpr int64_t kMr = 16;
    return ((mc + kMr - 1) / kMr) * sme_a_panel_stride(DataType::s8, kc);
}

/// Bytes one SME packed int8 B slab of @p nc columns × @p kc K occupies; the
/// SMOPA B block is 64 columns.
inline int64_t sme_i8_b_bytes(int64_t nc, int64_t kc)
{
    constexpr int64_t kNr = 64;
    return ((nc + kNr - 1) / kNr) * sme_b_panel_stride(DataType::s8, kc);
}

/// Element offset of row @p row inside a packed fp A slab. The ZA A layout is
/// block-major in 32-row groups, so a tile's start is addressable only on a
/// 32-row boundary; the standard route tiles its M loop that way.
inline int64_t fp_sme_a_row_offset(DataType dt, int64_t kc, int64_t row)
{
    return dt == DataType::f32 ? row * kc : row * (((kc + 1) / 2) * 2);
}

/// Multiply every element of a packed A slab by @p scale. The ZA pack has no
/// scale parameter (MatMul's SME route requires alpha 1), so the attention
/// scale rides GEMM1's A, packed once per task; the padding entries are zero.
template <class T> inline void scale_packed_a(T* p, int64_t elems, float scale)
{
    if (scale == 1.0f) {
        return;
    }
    for (int64_t i = 0; i < elems; ++i) {
        simd::s_store(&p[i], simd::s_load(&p[i]) * scale);
    }
}

/// The standard route's whole-head packed query block.
template <class T> inline int64_t standard_sme_q_elems(int64_t Sq, int64_t Dqk)
{
    if constexpr (kSmeBuilt && fp_sme_capable<T>) {
        constexpr DataType dt = std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
        return fp_sme_a_elems(dt, Sq, Dqk);
    } else {
        return 0;
    }
}

/// The standard route's one packed probability tile, at its GEMM2 k-block.
template <class T> inline int64_t standard_sme_p_elems(int mc2, int64_t Sk)
{
    if constexpr (kSmeBuilt && fp_sme_capable<T>) {
        constexpr DataType dt = std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
        const int64_t kc2 = std::min<int64_t>(sme_kc_for(dt), std::max<int64_t>(Sk, 1));
        return fp_sme_a_elems(dt, mc2, kc2);
    } else {
        return 0;
    }
}

/// The standard route's packed-B slot: the NEON fp panels, or the wider ZA
/// blocks. GEMM1's B is K^T (N <= nc1, K = Dqk); GEMM2's is V (N <= Dv,
/// K <= the ZA k-block the kernel loops on).
template <class T>
inline int64_t standard_b_pack_elems(int nc1, int nc2, int64_t Dqk, int64_t Dv, int64_t Sk)
{
    constexpr int nr_max = nr_max_flt<T>();
    const int ldd_b_full = static_cast<int>(
        mma_pack_align(nr_max * static_cast<int>(kc_pack_bound<T>(std::max(Dqk, Sk))) *
                       static_cast<int>(sizeof(T))) /
        static_cast<int>(sizeof(T)));
    const int np_full =
        std::max(panels_for_pack<T>(nc1), panels_for_pack<T>(std::min<int64_t>(Dv, nc2)));
    const int64_t fp_elems = static_cast<int64_t>(np_full) * ldd_b_full;
    if constexpr (kSmeBuilt && fp_sme_capable<T>) {
        constexpr DataType dt = std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
        const int64_t kc2 = std::min<int64_t>(sme_kc_for(dt), std::max<int64_t>(Sk, 1));
        const int64_t sme_elems =
            std::max(fp_sme_b_elems(dt, nc1, Dqk),
                     fp_sme_b_elems(dt, std::min<int64_t>(Dv, nc2), kc2));
        return std::max(fp_elems, sme_elems);
    } else {
        return fp_elems;
    }
}

/// Byte layout of a flash slot's int8 scratch, after the packed-B area:
///   [packed Q, heads × q_head]   dynamic-query Q·Kᵀ only
///   [Q row scales, heads × Br]   "
///   [packed P, one Br-row block] int8 P·V only
///   [int32 tile, Br × max(Bc, Dv)] either, sized for the wider side
/// Each region starts 16-byte aligned; a route that does not use one leaves it
/// empty. The plan sizes the slot from this and the kernel carves it the same
/// way, so the two can never disagree.
struct FlashI8Layout {
    size_t q_head = 0; ///< bytes one head's packed Br-row query block takes
    size_t qscale = 0; ///< the row scales, after every head's block
    size_t ppack = 0;  ///< one packed Br-row probability block
    size_t qraw = 0;   ///< SME only: one Br×Dqk row-major int8 query block
    size_t s32 = 0;    ///< the int32 accumulator tile
    size_t rowmax = 0; ///< per (head, row) tile maximum, Q·Kᵀ only
    size_t bytes = 0;  ///< the whole region
};

/// The packed Q/P blocks are sized for whichever engine packs them: the NEON
/// panel layout and the ZA block layout differ, and the kernel picks one at run
/// time. The raw query block is the SME path's quantize-then-pack staging area.
inline FlashI8Layout flash_i8_layout(int Br, int Bc, int64_t Dqk, int64_t Dv, int heads,
                                     bool qk8, bool pv8)
{
    const size_t br = static_cast<size_t>(Br);
    FlashI8Layout o;
    const size_t neon_q = qk8 ? static_cast<size_t>(i8_m_panels_max(Br)) *
                                    static_cast<size_t>(geom_ldd_i8_a(Dqk)) :
                                0;
    const size_t sme_q = (kSmeBuilt && qk8) ? static_cast<size_t>(sme_i8_a_bytes(Br, Dqk)) : 0;
    o.q_head = std::max(neon_q, sme_q);
    size_t at = static_cast<size_t>(heads) * o.q_head;
    o.qscale = align_up_bytes(at, 16);
    at = o.qscale + (qk8 ? align_up_bytes(static_cast<size_t>(heads) * br * sizeof(float), 16) : 0);
    const size_t neon_p = pv8 ? static_cast<size_t>(i8_m_panels_max(Br)) *
                                    static_cast<size_t>(geom_ldd_i8_a(Bc)) :
                                0;
    const size_t sme_p = (kSmeBuilt && pv8) ? static_cast<size_t>(sme_i8_a_bytes(Br, Bc)) : 0;
    o.ppack = align_up_bytes(at, 16);
    at = o.ppack + std::max(neon_p, sme_p);
    o.qraw = align_up_bytes(at, 16);
    at = o.qraw + ((kSmeBuilt && qk8) ? align_up_bytes(br * static_cast<size_t>(Dqk), 16) : 0);
    o.s32 = align_up_bytes(at, 16);
    const int64_t tile_n = (pv8 && Dv > Bc) ? Dv : Bc;
    at = o.s32 + ((qk8 || pv8) ?
                      align_up_bytes(br * static_cast<size_t>(tile_n) * sizeof(int32_t), 16) :
                      0);
    o.rowmax = align_up_bytes(at, 16);
    at = o.rowmax + (qk8 ? align_up_bytes(static_cast<size_t>(heads) * br * sizeof(float), 16) : 0);
    o.bytes = at;
    return o;
}

/// One int32 score row back into the route's score tile:
/// s[n] = acc[n] · row_mult · col_scale[n], where @p col_scale is the cache's
/// per-key scale (null for a per-tensor one, folded into row_mult).
template <class T>
inline void qk_i8_dequant_row(T* dst, const int32_t* acc, const float* col_scale, int64_t n,
                              float row_mult)
{
    constexpr int64_t L = simd::simd_default_lane_f32;
    const simd::v_f32x8 vm = simd::v_set1_f32x8(row_mult);
    int64_t i = 0;
    for (; i + L <= n; i += L) {
        simd::v_f32x8 v = simd::v_mul(kernel::v_cvt_i32_to_f32(acc + i), vm);
        if (col_scale != nullptr) {
            v = simd::v_mul(v, simd::v_load(col_scale + i));
        }
        simd::v_store_f32<T>(dst + i, v);
    }
    for (; i < n; ++i) {
        float v = static_cast<float>(acc[i]) * row_mult;
        if (col_scale != nullptr) {
            v *= col_scale[i];
        }
        simd::s_store(&dst[i], v);
    }
}

/// Keys [0, sink) are visible to every row regardless of the window.
inline int64_t sink_prefix(const AttentionConfig& cfg, int64_t run)
{
    const int64_t n = cfg.sink_tokens;
    if (n <= 0 || cfg.window <= 0) {
        return 0;
    }
    return n < run ? n : run;
}

/// One row element as f32, for a fp or int8 KV cache alike.
template <class KV> inline simd::v_f32x8 kv_load_f32(const KV* p)
{
    if constexpr (std::is_same_v<KV, int8_t>) {
        return simd::v_cvt_s8_to_f32(p);
    } else {
        return simd::v_load_f32(p);
    }
}

template <class KV> inline float kv_load_scalar(const KV* p)
{
    if constexpr (std::is_same_v<KV, int8_t>) {
        return static_cast<float>(*p);
    } else {
        return simd::s_load(p);
    }
}

// ---- output gate (Qwen3-Next/3.5) ---------------------------------------
//
// The gate trails every other slot. It is shaped like the output and applied to
// it in place after the route wrote it, which is the one place every route (and
// the varlen recursion) ends.

// ---- scratch slots -------------------------------------------------------
//
// A slot holds one task's scratch. How many are needed depends on the backend:
// one per worker when thread ids are reported (a worker reuses its slot), one
// for a sequential run (only one task is live), and otherwise one per task.
// The kernel's slot index and the plan's slot count are the same decision, so
// both go through these — sizing per task on a sequential run over-allocates
// the scratch by the task count, which at 20+ tasks and a large Br×Bc tile
// pushes it past the pool's cap and turns every call into a fresh malloc.

inline bool scratch_per_worker(const CpuBackend& cpu, int64_t tasks)
{
    const int slots = cpu.thread_slot_count();
    return slots > 0 && slots < tasks;
}

inline int64_t scratch_slot_of(const CpuBackend& cpu, int64_t tasks, int64_t idx)
{
    if (scratch_per_worker(cpu, tasks)) {
        return cpu.current_thread_id();
    }
    return cpu.worker_count() <= 1 ? 0 : idx;
}

// ---- row limit -----------------------------------------------------------

/// Exclusive end of query row @p i's valid key range: the batch's whole run
/// when not causal, else q0 + i + 1 with q0 the row-0 position —
/// cfg.position_offset when given, bottom-right (run - Sq) otherwise. @p run is
/// the batch's valid key-run length (run_length), so a shorter run truncates
/// both the padding and the causal diagonal.
/// The absolute position of query row 0 for one batch: the explicit per-batch
/// vector (int64 [B]), else the operator's scalar, else -1 — the bottom-right
/// default (run - Sq), resolved by the callers.
inline int64_t batch_pos0(const AttentionConfig& cfg, std::span<const TensorView> inputs,
                          int64_t b)
{
    if (slot_present(cfg, cfg.pos_ids_slot, inputs)) {
        return inputs[static_cast<size_t>(cfg.pos_ids_slot)].ptr<int64_t>()[b];
    }
    return cfg.position_offset;
}

inline int64_t causal_limit(const AttentionConfig& cfg, int64_t pos0, int64_t i, int64_t Sq,
                            int64_t run)
{
    if (!cfg.causal) {
        return run;
    }
    const int64_t q0 = pos0 >= 0 ? pos0 : (run - Sq);
    const int64_t lim = q0 + i + 1;
    if (lim < 0) {
        return 0;
    }
    return lim > run ? run : lim;
}

/// Inclusive start of query row @p i's window: the first key position it may
/// still read. 0 without a window, so the valid range stays [0, causal_limit).
/// The window counts back from the row's own absolute position, the same q0 the
/// causal limit uses, which makes it a band around the diagonal.
inline int64_t key_lo(const AttentionConfig& cfg, int64_t pos0, int64_t i, int64_t Sq,
                      int64_t run)
{
    // The window is a band around the causal diagonal: without a diagonal
    // there is nothing to band.
    if (cfg.window <= 0 || !cfg.causal) {
        return 0;
    }
    const int64_t q0 = pos0 >= 0 ? pos0 : (run - Sq);
    const int64_t lo = q0 + i - cfg.window + 1;
    if (lo <= 0) {
        return 0;
    }
    return lo > run ? run : lo;
}

template <class T> inline void row_fill(T* row, int64_t n, float val)
{
    for (int64_t i = 0; i < n; ++i) {
        simd::s_store(&row[i], val);
    }
}

/// s' = cap·tanh(s·scale/cap), in place over one row. The softcap rides with
/// the scale because the standard and flash routes fuse scale into the GEMM1
/// RHS pack, while the decode route applies both here.
template <class T> inline void row_softcap(T* row, int64_t n, float scale, float cap)
{
    using V = kernel::simd_vec_t<T>;
    constexpr int L = simd::simd_lane_for<T>;
    const V vinv = simd::v_set1(row, scale / cap);
    const V vcap = simd::v_set1(row, cap);
    int64_t i = 0;
    for (; i + L <= n; i += L) {
        const V x = simd::v_load(row + i);
        simd::v_store(row + i, simd::v_mul(vcap, kernel::v_tanh(simd::v_mul(x, vinv))));
    }
    for (; i < n; ++i) {
        simd::s_store(&row[i], cap * std::tanh(simd::s_load(&row[i]) * scale / cap));
    }
}

/// One int32 score lane: dequantize it, apply the softcap, then the keep and
/// additive masks — the same sequence the fp routes run as separate passes.
template <class T>
inline float qk_i8_lane(const int32_t* acc, const float* col_scale, int64_t i, float row_mult,
                        float softcap, const T* bias, const int32_t* keep, float filter)
{
    float v = static_cast<float>(acc[i]) * row_mult;
    if (col_scale != nullptr) {
        v *= col_scale[i];
    }
    if (softcap > 0.0f) {
        v = softcap * std::tanh(v / softcap);
    }
    if (bias != nullptr) {
        v += simd::s_load(&bias[i]);
    }
    if (keep != nullptr && keep[i] == 0) {
        v = filter;
    }
    return v;
}

/// One pass over an int32 score row's visible range [n0, n1). kWrite stores the
/// dequantized fp scores (the single-pass finalize); kWrite=false only returns
/// the row maximum (the two-pass max). Writing fills the out-of-range lanes
/// with -inf, which is what the separate mask pass leaves behind.
template <bool kWrite, class T>
inline float qk_i8_score_row(T* dst, const int32_t* acc, const float* col_scale, int64_t n,
                             float row_mult, float softcap, const T* bias, const int32_t* keep,
                             float filter, int64_t n0, int64_t n1)
{
    if constexpr (kWrite) {
        row_fill(dst, n0, kNegInf);
        row_fill(dst + n1, n - n1, kNegInf);
    }
    if (n1 <= n0) {
        return kNegInf;
    }
    float mx = kNegInf;
    if (keep != nullptr) {
        // The keep mask filters lanes individually; the rare path stays scalar.
        for (int64_t i = n0; i < n1; ++i) {
            const float v = qk_i8_lane(acc, col_scale, i, row_mult, softcap, bias, keep, filter);
            if constexpr (kWrite) {
                simd::s_store(&dst[i], v);
            }
            mx = std::max(mx, v);
        }
        return mx;
    }
    constexpr int64_t L = simd::simd_default_lane_f32;
    const simd::v_f32x8 vm = simd::v_set1_f32x8(row_mult);
    const simd::v_f32x8 vcap = simd::v_set1_f32x8(softcap > 0.0f ? softcap : 1.0f);
    const simd::v_f32x8 vinv = simd::v_set1_f32x8(softcap > 0.0f ? 1.0f / softcap : 1.0f);
    simd::v_f32x8 vmax = simd::v_set1_f32x8(kNegInf);
    int64_t i = n0;
    for (; i + L <= n1; i += L) {
        simd::v_f32x8 v = simd::v_mul(kernel::v_cvt_i32_to_f32(acc + i), vm);
        if (col_scale != nullptr) {
            v = simd::v_mul(v, simd::v_load(col_scale + i));
        }
        if (softcap > 0.0f) {
            v = simd::v_mul(vcap, kernel::v_tanh(simd::v_mul(v, vinv)));
        }
        if (bias != nullptr) {
            v = simd::v_add(v, simd::v_load_f32(bias + i));
        }
        if constexpr (kWrite) {
            simd::v_store_f32<T>(dst + i, v);
        }
        vmax = simd::v_max(vmax, v);
    }
    mx = simd::v_reduce_max(vmax);
    for (; i < n1; ++i) {
        const float v = qk_i8_lane(acc, col_scale, i, row_mult, softcap, bias, nullptr, filter);
        if constexpr (kWrite) {
            simd::s_store(&dst[i], v);
        }
        mx = std::max(mx, v);
    }
    return mx;
}

/// Dequantize one int32 row, apply softcap and the mask, and return its
/// maximum over the visible lanes (the single-pass finalize's max half).
template <class T>
inline float qk_i8_finish_row(T* dst, const int32_t* acc, const float* col_scale, int64_t n,
                              float row_mult, float softcap, const T* bias, const int32_t* keep,
                              float filter, int64_t n0, int64_t n1)
{
    return qk_i8_score_row<true, T>(dst, acc, col_scale, n, row_mult, softcap, bias, keep, filter,
                                    n0, n1);
}

// ---- flash route geometry ------------------------------------------------
//
// The flash path is chosen once the per-head scores matrix is large enough
// that materializing it — plus the fused mask/softmax passes — would thrash
// cache. Br / Bc follow the MLAS L2 heuristic: Bc = L2 / (sizeof(T)·4·(Dqk+Dv))
// and Br = min(Bc, Dqk+Dv), then both capped to Sk / Sq.

constexpr int64_t kFlashMinScores = 1 << 16; // Sq*Sk threshold (≈ 256 KiB f32)

// Tile caps measured across 1T..20T:
//  - Bc tops out at 256. The L2 formula asks for 512 at D=128 f32, but the
//    S tile (Br·Bc), the accumulator and the packed K no longer share the L2
//    with the K/V stream, and 20T lands 12% below the 256-wide tile.
//  - Causal runs want small tiles, because the work a query block wastes above
//    the diagonal is Br/Sq of the total — but only while the key run is
//    comparable to the query run: the fraction is (Sq-Br)/(2·Sk), so a chunk
//    against a long cache gains nothing and keeps the wide tiles. At least
//    four query blocks are needed for the saved triangle to pay for the tiling
//    overhead (a 128-row MLA block lost 5% at Br 64, its 1024-row prefill
//    gained 7%).
constexpr int kFlashBcMax = 256;
constexpr int kCausalBrMax = 64;
constexpr int kCausalBcMax = 128;

inline bool should_use_flash_attention(int64_t Sq, int64_t Sk)
{
    return Sq * Sk >= kFlashMinScores;
}

inline void resolve_flash_tile_sizes(size_t elem_size, int64_t Sq, int64_t Sk, int64_t Dqk,
                                     int64_t Dv, bool causal, int& Br, int& Bc)
{
    const size_t l2 = simd::CpuFeatures::get().l2_cache_size();
    const size_t l2_eff = l2 > 0 ? l2 : (1024u * 1024u); // 1 MiB fallback
    const int64_t head_pairs = Dqk + Dv;                 // qk_head_size + v_head_size
    Bc = static_cast<int>(
        std::min<int64_t>(l2_eff / (elem_size * 4 * static_cast<size_t>(head_pairs)), kFlashBcMax));
    if (Bc < 1) {
        Bc = 1;
    }
    Br = static_cast<int>(std::min<int64_t>(Bc, head_pairs));
    if (causal && Sq >= 4 * kCausalBrMax && Sk <= 2 * Sq) {
        Br = std::min(Br, kCausalBrMax);
        Bc = std::min(Bc, kCausalBcMax);
    }
    Bc = static_cast<int>(std::min<int64_t>(Bc, Sk));
    Br = static_cast<int>(std::min<int64_t>(Br, Sq));
    if (Br < 1) {
        Br = 1;
    }
    if (Bc < 1) {
        Bc = 1;
    }
}

// Per-task scratch for one Br-row query block, matching the layout inside
// attention_flash_impl: per head [m:Br][l:Br][s:Br*Bc][o_acc:Br*Dv], then one
// shared [pack_b][i8 temp], then — on the dynamic-query int8 path — the
// quantized query block, its row scales and the int32 score tile. The packed-B
// slot serves GEMM1 (K = Dqk, N <= Bc) and GEMM2 (K <= Bc, N = Dv) for every
// batched head, holding the wider of the fp panels and the int8 Q·Kᵀ ones; the
// trailing slot is the int8 dequantization panel, present only when the KV
// cache is quantized.

/// Panels a packed int8 B tile of any n <= n_max emits.
constexpr int panels_for_pack_i8(int64_t n_max)
{
    return static_cast<int>((n_max + NR_MAX_I8 - 1) / NR_MAX_I8);
}

/// The packed-B slot, in T elements: the fp panels of both GEMMs, or the int8
/// panels of either integer GEMM when those are wider (they never are today,
/// but the slot is one region and every route must fit it), or the ZA fp
/// blocks. The int8 K panels are Bc keys × Dqk, the int8 V panels Dv values ×
/// Bc keys; the ZA K block is Bc keys × Dqk and its V block Dv values × Bc
/// keys.
template <class T> inline int64_t flash_pack_elems(int Bc, int64_t Dqk, int64_t Dv)
{
    constexpr int nr_max = nr_max_flt<T>();
    const int ldd_b_full = static_cast<int>(
        mma_pack_align(nr_max * static_cast<int>(kc_pack_bound<T>(std::max<int64_t>(Dqk, Bc))) *
                       static_cast<int>(sizeof(T))) /
        static_cast<int>(sizeof(T)));
    const int np_full = std::max(panels_for_pack<T>(Bc), panels_for_pack<T>(Dv));
    const int64_t fp_elems = static_cast<int64_t>(np_full) * ldd_b_full;
    const int64_t kc8 = std::min<int64_t>(Bc, KC_CAP_I8);
    const int64_t i8_bytes =
        std::max(static_cast<int64_t>(panels_for_pack_i8(Bc)) *
                     int8_panel_stride(std::min<int64_t>(Dqk, KC_CAP_I8)),
                 static_cast<int64_t>(panels_for_pack_i8(Dv)) * int8_panel_stride(kc8));
    const int64_t i8_elems = (i8_bytes + static_cast<int64_t>(sizeof(T)) - 1) /
                             static_cast<int64_t>(sizeof(T));
    constexpr DataType dt = std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
    constexpr bool sme_ok = kSmeBuilt && fp_sme_capable<T>;
    const int64_t sme_elems =
        sme_ok ? std::max(fp_sme_b_elems(dt, Bc, Dqk), fp_sme_b_elems(dt, Dv, Bc)) : 0;
    const int64_t sme_i8_bytes =
        kSmeBuilt ? std::max(sme_i8_b_bytes(Bc, Dqk), sme_i8_b_bytes(Dv, Bc)) : 0;
    const int64_t sme_i8_elems = (sme_i8_bytes + static_cast<int64_t>(sizeof(T)) - 1) /
                                 static_cast<int64_t>(sizeof(T));
    return std::max({fp_elems, i8_elems, sme_elems, sme_i8_elems});
}

/// Byte layout of a flash slot's ZA fp scratch, after the packed-B area:
///   [packed Q, heads × q_head]   the scaled query block, packed once per task
///   [packed P, one Br-row block] the probability tile, repacked per KV tile
/// Only the fp flash route uses it; the int8 route has its own layout, and the
/// two never run together.
struct FlashSmeLayout {
    size_t q_head = 0; ///< bytes one head's packed Br-row query block takes
    size_t ppack = 0;  ///< one packed Br-row probability block
    size_t bytes = 0;  ///< the whole region
};

template <class T>
inline FlashSmeLayout flash_sme_layout(int Br, int Bc, int64_t Dqk, int heads)
{
    FlashSmeLayout o;
    if constexpr (kSmeBuilt && fp_sme_capable<T>) {
        constexpr DataType dt = std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
        o.q_head = static_cast<size_t>(fp_sme_a_elems(dt, Br, Dqk)) * sizeof(T);
        size_t at = static_cast<size_t>(heads) * o.q_head;
        o.ppack = align_up_bytes(at, 16);
        at = o.ppack + static_cast<size_t>(fp_sme_a_elems(dt, Br, Bc)) * sizeof(T);
        o.bytes = align_up_bytes(at, 16);
    }
    return o;
}

template <class T>
inline int64_t flash_per_task_elems(int Br, int Bc, int64_t Dqk, int64_t Dv, int heads,
                                    bool kv8 = false, bool qk8 = false, bool pv8 = false)
{
    const int64_t i8_elems =
        kv8 ? kv_i8_panel_elems<T>(kc_pack_bound<T>(std::max<int64_t>(Dqk, Bc))) : 0;
    // The packed query/probability blocks of the integer route and the ZA fp
    // route occupy the same trailing region; a slot's layout shares one base,
    // so it must hold the wider of the two.
    const int64_t i8_layout_bytes =
        (qk8 || pv8) ? static_cast<int64_t>(flash_i8_layout(Br, Bc, Dqk, Dv, heads, qk8, pv8).bytes) :
                       0;
    const int64_t sme_layout_bytes =
        static_cast<int64_t>(flash_sme_layout<T>(Br, Bc, Dqk, heads).bytes);
    const int64_t tail_elems = (std::max(i8_layout_bytes, sme_layout_bytes) + sizeof(T) - 1) /
                               sizeof(T);
    const int64_t per_head = static_cast<int64_t>(Br) * 2 + static_cast<int64_t>(Br) * Bc +
                             static_cast<int64_t>(Br) * Dv;
    return per_head * heads + flash_pack_elems<T>(Bc, Dqk, Dv) + i8_elems + tail_elems;
}

/// Flash GQA head batching: the most query heads one task serves. Past four
/// the per-head accumulators stop paying for the shared pack.
constexpr int kFlashHeadBatchMax = 4;
/// Flash KV splitting: slices of the run one underfilled grid is cut into, and
/// the shortest slice worth its partial-output merge.
constexpr int64_t kFlashMaxSplits = 8;
constexpr int64_t kFlashMinSlice = 128;
/// Scratch one attention plan may ask the pool for. Above the pool's 32 MiB
/// size class the request bypasses the pool, so the head batch and split count
/// are reduced to stay under this.
constexpr size_t kFlashScratchBudget = 16u << 20;

// ---- decode route geometry -----------------------------------------------
// Shared by attention_decode.cpp and the plan: the score block size the kernel
// walks the run in, the head batch one task serves, and the KV-split caps.

constexpr int64_t kDecodeChunk = 1024;   ///< keys per decode softmax block
constexpr int kDecodeHeadBatch = 8;      ///< query heads one decode task serves
constexpr int kDecodeMaxSplits = 16;     ///< KV slices per decode unit, at most
constexpr int64_t kDecodeMinSlice = 256; ///< shortest slice worth its merge pass

// ---- route entries -------------------------------------------------------
// Each route file defines one; the dispatch in attention.cpp calls it after
// validating the dtype. Scratch is the pooled workspace from the plan.

void attention_standard_cpu(const AttentionConfig& cfg, TensorView& output,
                            std::span<const TensorView> inputs, const ComputeContext& ctx,
                            const AttentionPlan& plan, void* scratch);

void attention_flash_cpu(const AttentionConfig& cfg, TensorView& output,
                         std::span<const TensorView> inputs, const ComputeContext& ctx,
                         const AttentionPlan& plan, void* scratch);

void attention_decode_cpu(const AttentionConfig& cfg, TensorView& output,
                          std::span<const TensorView> inputs, const ComputeContext& ctx,
                          const AttentionPlan& plan, void* scratch);

} // namespace nnops::backend::cpu
