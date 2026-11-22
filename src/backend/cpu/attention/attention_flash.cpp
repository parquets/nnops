/// @file attention_flash.cpp
/// @brief FlashAttention-1: stream KV in Br×Bc tiles with an online softmax,
///        so the Sq×Sk score matrix never exists.
///
/// One task is one Br-row query block of one (batch, KV head, head block) and,
/// when the grid would leave workers idle, one KV slice of the run. The grid is
/// batch × KV head × head-block × query-block × slice. Each task walks its span
/// in Bc-key tiles, carrying a running max (m), a running sum (l) and an output
/// accumulator (o) per batched head, and folds the mask and the causal bound
/// into the S tile before the update.
///
/// GQA head batching: up to four query heads of the same KV head share a task,
/// so each K/V tile is packed and read once for the whole group. KV splitting:
/// when the grid is smaller than the worker count, the run is cut into slices
/// and the per-slice online-softmax states are merged afterwards, so a short
/// chunk against a long cache keeps the workers busy.
///
/// The run may be two segments, an optional past tensor followed by the step's
/// new K/V; each is streamed in place, so nothing is concatenated. A causal
/// query block stops its KV stream at the block's causal limit, so prefill never
/// walks the masked upper triangle.
///
/// Both GEMMs drive tile_mma_direct with a packed B. The first k-block of the
/// whole task runs zero_mode, which is what lets o_acc skip a pre-zeroing pass.
///
/// On an SME machine a dense f32/f16 run takes the ZA engine instead: both
/// GEMMs go through tile_pack_*_sme + tile_mma_pack_sme. The query block is
/// packed once per task, with the softmax scale folded into the packed A (the
/// ZA pack has no alpha), and the probability tile is packed per KV tile. The
/// int8 switches and a bf16 run keep the NEON path.
///
/// Over an s8 cache with cfg.quantize_q set, GEMM1 instead runs on the integer
/// MMA: the task's query rows are quantized once (one symmetric scale per row)
/// into a signed byte block, the cache's keys are packed as stored, and the
/// int32 scores are dequantized into the S tile by scale·s_q[row]·s_k[key].
///
/// cfg.quantize_pv does the same for GEMM2, with a static grid: the tile is a
/// probability in [0,1] once the max is subtracted, so 255 levels cover it, and
/// the cache's per-key V scale is folded into it (that scale is on the
/// reduction axis, so it cannot be applied to the output). The int32 tile is
/// dequantized onto the fp accumulator, keeping the exp(Δm) rescaling exact.
/// Both switches are independent; either or both may be off.

#include "attention_internal.h"
#include "backend/cpu/simd_kernel/simd_quant.hpp" // quantization — the ZA query staging block
#if defined(NNOPS_ARCH_SME)
#include "backend/cpu/aarch64/sme/mma_sme.hpp" // available_sme — the ZA route gate
#endif

namespace nnops::backend::cpu {

namespace {

// ---- tiled FlashAttention kernel -----------------------------------------
//
// One task = one query block (Br rows) of one (batch, head): it streams KV in
// Bc-sized blocks, carrying a running max/sum and an output accumulator, so
// the Sq×Sk scores matrix never exists. Reuses the packed-B route above.

template <class T>
void attention_flash_impl(const AttentionConfig& cfg, TensorView& output,
                          std::span<const TensorView> inputs, const ComputeContext& ctx,
                          const AttentionPlan& plan, void* workspace)
{
    const auto& Q = inputs[0];
    const auto& K = inputs[1];
    const auto& V = cfg.v_is_k ? inputs[1] : inputs[2];
    const MaskView mask = resolve_mask(cfg, inputs);
    const bool split = has_past_view(cfg, inputs);
    const auto& Kp = split ? inputs[static_cast<size_t>(cfg.past_k_slot)] : K;
    const auto& Vp = split ? inputs[static_cast<size_t>(cfg.past_v_slot)] : V;
    bool seqlens_plus_one = true;
    const int32_t* seqlens = run_lengths_ptr(cfg, inputs, seqlens_plus_one);

    const HeadShape hs = resolve_head_shape(cfg, inputs);
    const int64_t B = hs.B, H = hs.H, Hkv = hs.Hkv;
    const int64_t Sq = hs.Sq, Dqk = hs.Dqk, Dv = hs.Dv;
    const int64_t Sk_new = hs.Sk, Sk_past = hs.Sk_past;
    const int64_t Sk = Sk_new + Sk_past; // the whole [past | new] run

    const float scale =
        (cfg.scale == 0.0f) ? (1.0f / std::sqrt(static_cast<float>(Dqk))) : cfg.scale;

    const bool kv8 = kv_is_int8(inputs);
    NNOPS_ASSERT(!(kv8 && split)); // a quantized cache has no past segment
    // Dynamic-query int8 Q·Kᵀ and int8 P·V: the plan resolved both (and sized
    // the slot for them).
    const bool qk8 = plan.flash_qk_i8;
    const bool pv8 = plan.flash_pv_i8;
    NNOPS_ASSERT(!qk8 || kv8);
    NNOPS_ASSERT(!pv8 || kv8);
    const int8_t* k8_ptr = kv8 ? K.ptr<int8_t>() : nullptr;
    const int8_t* v8_ptr = kv8 ? V.ptr<int8_t>() : nullptr;
    const T* q_ptr = Q.ptr<T>();
    const T* k_ptr = kv8 ? nullptr : K.ptr<T>();
    const T* v_ptr = kv8 ? nullptr : V.ptr<T>();
    const T* kp_ptr = split ? Kp.ptr<T>() : nullptr;
    const T* vp_ptr = split ? Vp.ptr<T>() : nullptr;
    const KvScale kscale = resolve_kv_scale(cfg, cfg.k_scale_slot, inputs);
    const KvScale vscale = resolve_kv_scale(cfg, cfg.v_scale_slot, inputs);
    const T* mask_ptr = static_cast<const T*>(mask.bias);
    // The sink is one extra logit in the denominator, per query head.
    const bool has_sink = slot_present(cfg, cfg.sink_slot, inputs);
    const T* sink_ptr = has_sink ? inputs[static_cast<size_t>(cfg.sink_slot)].ptr<T>() : nullptr;
    T* out_ptr = output.ptr<T>();

    const int64_t q_rs = head_seq_step(Q, hs.q_layout);
    const int64_t k_rs = head_seq_step(K, hs.kv_layout);
    const int64_t v_rs = head_seq_step(V, hs.kv_layout);
    const int64_t kp_rs = split ? head_seq_step(Kp, hs.kv_layout) : k_rs;
    const int64_t vp_rs = split ? head_seq_step(Vp, hs.kv_layout) : v_rs;
    const int64_t o_rs = head_seq_step(output, hs.q_layout);
    const int64_t v_head_stride = merged_v_head_stride(cfg, Dqk, Dv);

    int64_t q_b, q_h, k_b, k_h, v_b, v_h, o_b, o_h;
    int64_t kp_b = 0, kp_h = 0, vp_b = 0, vp_h = 0;
    head_strides(Q, hs.q_layout, Dqk, q_b, q_h);
    head_strides(K, hs.kv_layout, Dqk, k_b, k_h);
    head_strides(V, hs.kv_layout, v_head_stride, v_b, v_h);
    head_strides(output, hs.q_layout, Dv, o_b, o_h);
    if (split) {
        head_strides(Kp, hs.kv_layout, Dqk, kp_b, kp_h);
        head_strides(Vp, hs.kv_layout, v_head_stride, vp_b, vp_h);
    }

    constexpr int nr_max = nr_max_flt<T>();
    constexpr int64_t kc = kc_for<T>();
    constexpr int L = simd::simd_lane_for<T>;

    const int Br = plan.Br;
    const int Bc = plan.Bc;
    // GQA head batching: one task serves `hb` query heads of the same KV head,
    // so each K/V tile is packed and read once for the whole group.
    const int hb = plan.flash_heads;
    const int64_t G = H / Hkv;
    const int64_t nb = (G + hb - 1) / hb;
    const int64_t per_task = flash_per_task_elems<T>(Br, Bc, Dqk, Dv, hb, kv8, qk8, pv8);
    const int64_t pack_elems = flash_pack_elems<T>(Bc, Dqk, Dv);
    const FlashI8Layout i8_layout = flash_i8_layout(Br, Bc, Dqk, Dv, hb, qk8, pv8);
    // The ZA fp engine, when this build and CPU carry it: f32/f16 only, a
    // dense cache only (an s8 cache keeps the dequantizing NEON pack), and a
    // head dim the packed A slab can hold as one ZA k block.
    constexpr DataType sme_dt = std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
    bool sme = false;
    bool sme_i8 = false;
#if defined(NNOPS_ARCH_SME)
    if constexpr (kSmeBuilt && fp_sme_capable<T>) {
        sme = !kv8 && aarch64::available_sme() &&
              Dqk <= sme_kc_for(sme_dt);
    }
    if constexpr (kSmeBuilt && fp_sme_capable<T>) {
        sme_i8 = kv8 && aarch64::available_sme();
    }
#endif
    // The integer products each decide their engine; either may be off while
    // the other is on, and an s8 cache keeps the fp path out entirely.
    const bool sme_qk = sme_i8 && qk8;
    const bool sme_pv = sme_i8 && pv8;
    // A 16-row query chunk is one ZA block and the packs dominate; a couple of
    // row blocks is where the integer ZA route starts to pay.
    constexpr int kSmeI8MinRows = 32;
    const FlashSmeLayout sme_layout = flash_sme_layout<T>(Br, Bc, Dqk, hb);
    const int64_t sme_q_head = static_cast<int64_t>(sme_layout.q_head) /
                               static_cast<int64_t>(sizeof(T));

    const int64_t NQ = (Sq + Br - 1) / Br; // number of Br-row query blocks
    const int64_t units = B * Hkv * nb * NQ;
    // KV splitting: slice s of every unit is one task; the slice dispatches
    // leave unnormalized states that the merge pass combines.
    const int64_t splits = plan.flash_splits;
    const int64_t NG = units * splits;
    const int64_t nmerge = B * H * NQ;
    constexpr int64_t lanes = simd::simd_default_lane_f32;
    T* ws_base = static_cast<T*>(workspace);
    const int64_t partial_step = static_cast<int64_t>(hb) * Br * (Dv + 2);
    float* partials = nullptr;
    float* merge_acc = nullptr;
    if (splits > 1) {
        char* base = static_cast<char*>(workspace);
        const size_t slot_bytes = static_cast<size_t>(plan.num_slots) *
                                  static_cast<size_t>(per_task) * sizeof(T);
        partials = reinterpret_cast<float*>(base + align_up_bytes(slot_bytes, 16));
        merge_acc = partials + static_cast<size_t>(units * splits) *
                                   static_cast<size_t>(partial_step);
    }

    const auto run = [&](int64_t idx) {
        const int64_t si = idx % splits;
        const int64_t unit = idx / splits;
        const int64_t qi = unit % NQ;
        const int64_t c = (unit / NQ) % nb;
        const int64_t hkv = (unit / (NQ * nb)) % Hkv;
        const int64_t b = unit / (NQ * nb * Hkv);
        const int64_t h0 = hkv * G + c * hb; // first query head of this block
        const int nh = static_cast<int>(std::min<int64_t>(hb, G - c * hb));
        const int64_t i0 = qi * Br;
        const int actual_br = static_cast<int>(std::min<int64_t>(Br, Sq - i0));
        const bool use_sme_qk = sme_qk && actual_br >= kSmeI8MinRows;
        const bool use_sme_pv = sme_pv && actual_br >= kSmeI8MinRows;
        // The batch's valid key run: its prefix of the [past | new] rows, and
        // where this batch's query rows sit (vector > scalar > bottom-right).
        const int64_t run_len = run_length(seqlens, b, Sk, seqlens_plus_one);
        const int64_t pos0 = batch_pos0(cfg, inputs, b);

        const T* q_base = q_ptr + b * q_b + h0 * q_h;
        const T* k_head = kv8 ? nullptr : k_ptr + b * k_b + hkv * k_h;
        const T* v_head = kv8 ? nullptr : v_ptr + b * v_b + hkv * v_h;
        const int8_t* k8_head = kv8 ? k8_ptr + b * k_b + hkv * k_h : nullptr;
        const int8_t* v8_head = kv8 ? v8_ptr + b * v_b + hkv * v_h : nullptr;
        const KvScaleRow ks_head = kscale.row(b, hkv, Hkv, Sk);
        const KvScaleRow vs_head = vscale.row(b, hkv, Hkv, Sk);
        T* o_base = out_ptr + b * o_b + h0 * o_h;
        // The run is two contiguous segments, past first; without a past the
        // first is empty. Each is streamed in place, so the caller never has to
        // concatenate.
        const T* k_base[2] = {k_head, k_head};
        const T* v_base[2] = {v_head, v_head};
        int64_t k_str[2] = {k_rs, k_rs};
        int64_t v_str[2] = {v_rs, v_rs};
        int64_t seg_lo[2] = {0, 0};
        int64_t seg_hi[2] = {0, Sk};
        if (split) {
            k_base[0] = kp_ptr + b * kp_b + hkv * kp_h;
            v_base[0] = vp_ptr + b * vp_b + hkv * vp_h;
            k_str[0] = kp_rs;
            v_str[0] = vp_rs;
            seg_lo[1] = Sk_past;
            seg_hi[0] = Sk_past;
        }

        const int64_t slot = scratch_slot_of(ctx.cpu, NG, idx);
        T* m = ws_base + slot * per_task;
        T* l = m + static_cast<int64_t>(hb) * Br;
        T* s = l + static_cast<int64_t>(hb) * Br;
        T* o_acc = s + static_cast<int64_t>(hb) * Br * Bc;
        T* pack_b = o_acc + static_cast<int64_t>(hb) * Br * Dv;
        T* i8_temp = ws_base + slot * per_task +
                     (per_task - (kv8 ? kv_i8_panel_elems<T>(kc_pack_bound<T>(
                                            std::max<int64_t>(Dqk, Bc))) :
                                       0));
        // The int8 scratch sits between the packed-B area and i8_temp: the
        // packed query blocks and their row scales (Q·Kᵀ), one packed
        // probability block (P·V), and the int32 accumulator tile both use.
        int8_t* qpack = nullptr;
        float* q_scale = nullptr;
        int8_t* qraw = nullptr;
        int8_t* ppack = nullptr;
        int32_t* s32 = nullptr;
        float* rowmax_buf = nullptr;
        if (qk8 || pv8) {
            char* tail = reinterpret_cast<char*>(pack_b + pack_elems);
            if (qk8) {
                qpack = reinterpret_cast<int8_t*>(tail);
                q_scale = reinterpret_cast<float*>(tail + i8_layout.qscale);
                qraw = reinterpret_cast<int8_t*>(tail + i8_layout.qraw);
            }
            if (pv8) {
                ppack = reinterpret_cast<int8_t*>(tail + i8_layout.ppack);
            }
            s32 = reinterpret_cast<int32_t*>(tail + i8_layout.s32);
            if (qk8) {
                rowmax_buf = reinterpret_cast<float*>(tail + i8_layout.rowmax);
            }
        }
        // The ZA fp route's two packed A blocks share the same trailing
        // region (the integer route and this one never run together).
        T* sme_q = nullptr;
        T* sme_p = nullptr;
        if (sme) {
            char* tail = reinterpret_cast<char*>(pack_b + pack_elems);
            sme_q = reinterpret_cast<T*>(tail);
            sme_p = reinterpret_cast<T*>(tail + sme_layout.ppack);
        }

        // Reset the running online-softmax state through the scalar simd
        // bridge: std::numeric_limits<T> is not specialized for half (it
        // silently yields 0, not -inf), and static_cast<half>(float) truncates
        // to an integer and stores it as bits on the x86 half wrapper.
        for (int j = 0; j < nh; ++j) {
            for (int i = 0; i < actual_br; ++i) {
                simd::s_store(&m[j * Br + i], -std::numeric_limits<float>::infinity());
                simd::s_store(&l[j * Br + i], 0.0f);
            }
        }
        // Causal early exit: the causal limit grows with the query row, so the
        // last row of the block bounds every row in it. A KV block starting at
        // or past that bound is fully masked for all of them and contributes
        // nothing — for a causal prefill that is the whole upper triangle, and
        // skipping it halves the QK and AV work.
        const int64_t kv_end =
            cfg.causal ? causal_limit(cfg, pos0, i0 + actual_br - 1, Sq, run_len) : run_len;
        // Streaming sinks: the first `prefix` positions stay visible to every
        // row, so they stream as a band of their own ahead of the window.
        const int64_t prefix = sink_prefix(cfg, run_len);

        // o_acc is written by the first KV block (zero_mode) — no pre-zeroing
        // pass. The exp_diff rescale may touch it first on the first block;
        // that garbage is fully overwritten by the first MMA.
        bool first_tile = true;

        // One KV tile over [j0, j0+bc): lanes [max(lo_i, lo_floor), lim_i)
        // survive for row i — or [0, lim_i) on the sink band, where the window
        // is exempt — everything else in the tile is -inf. Returns the next
        // tile's start.
        const auto do_tile = [&](int64_t j0, int64_t end, int64_t lo_floor,
                                 bool sink_band) -> int64_t {
            const int seg = (j0 < seg_lo[1]) ? 0 : 1;
            const int64_t seg_end = std::min(seg_hi[seg], end);
            // The diagonal block is the last one and is rarely a full Bc wide:
            // clamping it bounds the tile to the block's causal rectangle
            // instead of a full-width strip that is mostly masked. A segment
            // boundary also ends a tile, so one tile never spans both.
            const int actual_bc = static_cast<int>(std::min<int64_t>(Bc, seg_end - j0));
            const int64_t koff = j0 - seg_lo[seg];
            const T* kb = k_base[seg];
            const T* vb = v_base[seg];
            const int64_t k_step = k_str[seg];
            const int64_t v_step = v_str[seg];

            // ---- online-softmax update + accumulate into o_acc -----
            // On the int8 P·V path the exp step writes the probabilities
            // straight onto the u8 grid — folded with the V scale and placed
            // over the scores they came from — so the fp probabilities never
            // reach memory and the pack that follows is a pure byte shuffle.
            float sc_tile = 1.0f;
            float col_fold[kFlashBcMax];
            const float* fold = nullptr;
            if (pv8) {
                NNOPS_ASSERT(actual_bc <= kFlashBcMax);
                const bool v_col = vs_head.ptr != nullptr && vs_head.step != 0;
                sc_tile = vs_head.at(0);
                if (v_col) {
                    sc_tile = 0.0f;
                    for (int t = 0; t < actual_bc; ++t) {
                        sc_tile = std::max(sc_tile, vs_head.at(j0 + t));
                    }
                    NNOPS_ASSERT(sc_tile > 0.0f);
                    const float inv = sc_tile > 0.0f ? 1.0f / sc_tile : 1.0f;
                    for (int t = 0; t < actual_bc; ++t) {
                        col_fold[t] = vs_head.at(j0 + t) * inv;
                    }
                    fold = col_fold;
                }
            }
            // ---- S = Q[i0:i0+Br] @ K[j0:j0+Bc]^T * scale ----------
            // k0 == 0 overwrites (zero_mode) — each KV block owns its S tile.
            if (qk8) {
                // The cache's s8 keys go into the int8 B panels as they are,
                // and the query was quantized once per task. int32 scores come
                // back scaled by scale · s_q[row] · s_k[key].
                const bool k_col = ks_head.ptr != nullptr && ks_head.step != 0;
                const float k_all = k_col ? 1.0f : ks_head.at(0);
                const float* kcol = k_col ? ks_head.ptr + j0 : nullptr;
                // One packed K tile serves every batched head (the gate keeps
                // the head dim inside one integer k block).
                if (use_sme_qk) {
#if defined(NNOPS_ARCH_SME)
                    if constexpr (fp_sme_capable<T>) {
                        tile_pack_rhs_i8_sme(true, actual_bc, static_cast<int>(Dqk),
                                             reinterpret_cast<int8_t*>(pack_b),
                                             k8_head + koff * k_step, static_cast<int>(k_step));
                    }
#endif
                } else {
                    tile_pack_rhs_i8(true, actual_bc, static_cast<int>(Dqk),
                                     reinterpret_cast<int8_t*>(pack_b),
                                     static_cast<int>(int8_panel_stride(Dqk)),
                                     k8_head + koff * k_step, static_cast<int>(k_step),
                                     kQkI8ABiased);
                }
                // s32 is one shared tile and the MMA overwrites it, so each
                // head's scores must be finalized before the next head's
                // product lands on it. Dequantization, softcap, the mask and
                // the row maximum are one pass: the scores are int32, so the fp
                // passes the other routes run would re-read this output.
                const int8_t* b8 = reinterpret_cast<const int8_t*>(pack_b);
                for (int j = 0; j < nh; ++j) {
                    int8_t* qp = qpack + static_cast<int64_t>(j) *
                                              static_cast<int64_t>(i8_layout.q_head);
                    if (use_sme_qk) {
#if defined(NNOPS_ARCH_SME)
                        if constexpr (fp_sme_capable<T>) {
                            tile_mma_pack_i8_sme(actual_br, actual_bc, static_cast<int>(Dqk),
                                                 s32, actual_bc, qp, b8,
                                                 /*block_stride=*/0, INT32_MIN, INT32_MAX,
                                                 /*zero_mode=*/true);
                        }
#endif
                    } else {
                        tile_mma_pack_i8(actual_br, actual_bc, static_cast<int>(Dqk), s32,
                                         actual_bc, qp, b8, /*ldb=*/0, INT32_MIN, INT32_MAX,
                                         /*zero_mode=*/true);
                    }
                    const int64_t h = h0 + j;
                    const float* qs_j = q_scale + static_cast<int64_t>(j) * Br;
                    for (int i = 0; i < actual_br; ++i) {
                        const int64_t row_i = i0 + i;
                        const int64_t lo_i =
                            sink_band ? 0
                                      : std::max<int64_t>(key_lo(cfg, pos0, row_i, Sq, run_len),
                                                          lo_floor);
                        const int64_t lim = causal_limit(cfg, pos0, row_i, Sq, run_len);
                        int64_t n0 = std::min<int64_t>(std::max<int64_t>(lo_i - j0, 0), actual_bc);
                        int64_t n1 = std::min<int64_t>(std::max<int64_t>(lim - j0, n0), actual_bc);
                        const T* bias_row = nullptr;
                        const int32_t* keep_row = nullptr;
                        if (mask.keep != nullptr) {
                            const int64_t off =
                                mask_offset(b, h, row_i, mask.keep_b, mask.keep_h, mask.keep_row);
                            keep_row = mask.keep + off + j0;
                        }
                        if (mask_ptr != nullptr) {
                            const int64_t off =
                                mask_offset(b, h, row_i, mask.bias_b, mask.bias_h, mask.bias_row);
                            bias_row = mask_ptr + off + j0;
                        }
                        T* srow = s + (static_cast<int64_t>(j) * Br + i) * actual_bc;
                        const float row_mult = scale * qs_j[i] * k_all;
                        rowmax_buf[static_cast<int64_t>(j) * Br + i] = qk_i8_finish_row<T>(
                            srow, s32 + static_cast<int64_t>(i) * actual_bc, kcol, actual_bc,
                            row_mult, cfg.softcap, bias_row, keep_row, mask.filter, n0, n1);
                    }
                }
            }
            if (!qk8 && sme) {
                if constexpr (kSmeBuilt && fp_sme_capable<T>) {
                    // K as one ZA B block sequence; the query block was packed
                    // once per task, with the scale already in it.
                    tile_pack_rhs_sme(true, actual_bc, static_cast<int>(Dqk), pack_b,
                                      kb + koff * k_step, static_cast<int>(k_step));
                    for (int j = 0; j < nh; ++j) {
                        tile_mma_pack_sme(
                            actual_br, actual_bc, static_cast<int>(Dqk),
                            s + static_cast<int64_t>(j) * Br * actual_bc, actual_bc,
                            sme_q + static_cast<int64_t>(j) * sme_q_head, pack_b,
                            /*block_stride=*/0, kNegInf, kPosInf, /*zero_mode=*/true);
                    }
                }
            }
            if (!qk8 && !sme) {
                for (int64_t k0 = 0; k0 < Dqk; k0 += kc) {
                    const int actual_kc = static_cast<int>(std::min<int64_t>(kc, Dqk - k0));
                    const int ldd_b = static_cast<int>(
                        mma_pack_align(nr_max * actual_kc * static_cast<int>(sizeof(T))) /
                        static_cast<int>(sizeof(T)));
                    if (kv8) {
                        kv_pack_i8<T>(true, actual_bc, actual_kc, pack_b, ldd_b,
                                      k8_head + koff * k_step + k0, static_cast<int>(k_step),
                                      ks_head.shifted(j0), scale, i8_temp);
                    } else {
                        tile_pack_rhs(true, actual_bc, actual_kc, pack_b, ldd_b,
                                      kb + koff * k_step + k0, static_cast<int>(k_step), scale);
                    }
                    // One packed K tile feeds every batched head of the group.
                    for (int j = 0; j < nh; ++j) {
                        T* s_j = s + static_cast<int64_t>(j) * Br * actual_bc;
                        tile_mma_direct(actual_br, actual_bc, actual_kc, s_j, actual_bc,
                                        q_base + j * q_h + i0 * q_rs + k0, static_cast<int>(q_rs),
                                        pack_b, /*ldb=*/-1, /*stage=*/nullptr, kNegInf, kPosInf,
                                        k0 == 0);
                    }
                }
            }

            // ---- softcap, additive mask and the out-of-range lanes ----
            // The cap is part of the logit, so it runs before the mask. The
            // mask, the causal tail and the window prefix are all per-row, so
            // one pass covers them: add the mask over the row's valid range
            // [n0, n1), then fill the rest of the tile with -inf.
            if (!qk8 && cfg.softcap > 0.0f) {
                for (int j = 0; j < nh; ++j) {
                    for (int i = 0; i < actual_br; ++i) {
                        row_softcap(s + (static_cast<int64_t>(j) * Br + i) * actual_bc, actual_bc,
                                    1.0f, cfg.softcap);
                    }
                }
            }
            if (!qk8 && (mask.any() || cfg.causal || cfg.window > 0)) {
                for (int j = 0; j < nh; ++j) {
                    const int64_t h = h0 + j;
                    for (int i = 0; i < actual_br; ++i) {
                        const int64_t row_i = i0 + i;
                        const int64_t lo_i =
                            sink_band ? 0
                                      : std::max<int64_t>(key_lo(cfg, pos0, row_i, Sq, run_len),
                                                          lo_floor);
                        const int64_t lim = causal_limit(cfg, pos0, row_i, Sq, run_len);
                        int64_t n0 = lo_i - j0;
                        int64_t n1 = lim - j0;
                        if (n0 < 0) {
                            n0 = 0;
                        }
                        if (n0 > actual_bc) {
                            n0 = actual_bc;
                        }
                        if (n1 < n0) {
                            n1 = n0;
                        }
                        if (n1 > actual_bc) {
                            n1 = actual_bc;
                        }
                        const int64_t n = n1 - n0;
                        T* srow = s + (static_cast<int64_t>(j) * Br + i) * actual_bc;
                        if (mask.keep != nullptr) {
                            const int64_t off =
                                mask_offset(b, h, row_i, mask.keep_b, mask.keep_h, mask.keep_row);
                            row_apply_keep(srow + n0, mask.keep + off + j0 + n0, n, mask.filter);
                        }
                        if (mask_ptr != nullptr) {
                            const int64_t off =
                                mask_offset(b, h, row_i, mask.bias_b, mask.bias_h, mask.bias_row);
                            const T* mrow = mask_ptr + off + j0 + n0;
                            int64_t k = 0;
                            for (; k + L <= n; k += L) {
                                simd::v_store(srow + n0 + k,
                                              simd::v_add(simd::v_load(srow + n0 + k),
                                                          simd::v_load(mrow + k)));
                            }
                            for (; k < n; ++k) {
                                simd::s_store(&srow[n0 + k], simd::s_load(&srow[n0 + k]) +
                                                                simd::s_load(&mrow[k]));
                            }
                        }
                        row_fill(srow, n0, -std::numeric_limits<float>::infinity());
                        row_fill(srow + n1, actual_bc - n1,
                                 -std::numeric_limits<float>::infinity());
                    }
                }
            }

            for (int j = 0; j < nh; ++j) {
                const int64_t head_row = static_cast<int64_t>(j) * Br;
                for (int i = 0; i < actual_br; ++i) {
                    T* srow = s + (head_row + i) * actual_bc;
                    const float rowmax = qk8 ?
                                             rowmax_buf[head_row + i] :
                                             kernel::softmax_row_max<T>(srow, actual_bc);
                    if (rowmax == kNegInf) {
                        // The whole tile is out of range for this row (a causal
                        // row above the diagonal): S must be zero or the AV GEMM
                        // would add -inf·v.
                        if (pv8) {
                            std::memset(srow, 0, static_cast<size_t>(actual_bc));
                        } else {
                            row_fill(srow, actual_bc, 0.0f);
                        }
                        continue;
                    }
                    const float old_m = simd::s_load(&m[head_row + i]);
                    const float new_m = std::max(old_m, rowmax);
                    const float exp_diff = std::exp(old_m - new_m); // 0 on the first KV block
                    simd::s_store(&m[head_row + i], new_m);
                    float rowsum;
                    if (pv8) {
                        rowsum = kernel::softmax_row_exp_quant<T>(
                            srow, reinterpret_cast<uint8_t*>(srow), actual_bc, new_m, 1.0f,
                            static_cast<float>(kPvI8Levels), fold);
                    } else {
                        rowsum = kernel::softmax_row_exp_sum<T>(srow, srow, actual_bc, new_m, 1.0f);
                    }
                    simd::s_store(&l[head_row + i],
                                  exp_diff * simd::s_load(&l[head_row + i]) + rowsum);

                    if (exp_diff != 1.0f) {
                        T* orow = o_acc + (head_row + i) * Dv;
                        auto v_ed = simd::v_set1(orow, exp_diff);
                        int64_t d = 0;
                        for (; d + L <= Dv; d += L) {
                            simd::v_store(orow + d, simd::v_mul(simd::v_load(orow + d), v_ed));
                        }
                        for (; d < Dv; ++d) {
                            simd::s_store(&orow[d], simd::s_load(&orow[d]) * exp_diff);
                        }
                    }
                }
            }

            // ---- o_acc += S @ V[j0:j0+Bc] --------------------------
            if (pv8) {
                // V as stored — its scale already rode on P through the exp
                // pass, and sc_tile (below the softmax update) brings the
                // tile's factor back.
                // The biased P (aarch64 only) is off by +128·Σv per column.
                float col_sum[kFlashBcMax];
                if constexpr (kPvI8ColCorrection != 0) {
                    i8_col_sums(v8_head + koff * v_step, v_step, actual_bc,
                                static_cast<int>(Dv), col_sum);
                }
                const float deq = sc_tile / static_cast<float>(kPvI8Levels);
                // The probabilities are already bytes, packed at the head of
                // the rows they replaced: only the layout is left.
                const int p_row_bytes = static_cast<int>(actual_bc * sizeof(T));
                const auto head_p8 = [&](int j) {
                    return reinterpret_cast<const int8_t*>(
                        s + static_cast<int64_t>(j) * Br * actual_bc);
                };
                const auto accum_tile = [&](int j) {
                    for (int i = 0; i < actual_br; ++i) {
                        pv_i8_accum_row<T, kPvI8ColCorrection>(
                            o_acc + (static_cast<int64_t>(j) * Br + i) * Dv,
                            s32 + static_cast<int64_t>(i) * Dv, col_sum, Dv, deq, !first_tile);
                    }
                };
                if (use_sme_pv) {
#if defined(NNOPS_ARCH_SME)
                    if constexpr (fp_sme_capable<T>) {
                        tile_pack_rhs_i8_sme(false, static_cast<int>(Dv), actual_bc,
                                             reinterpret_cast<int8_t*>(pack_b),
                                             v8_head + koff * v_step, static_cast<int>(v_step));
                        for (int j = 0; j < nh; ++j) {
                            tile_pack_lhs_i8_sme(
                                /*trans=*/false, actual_br, actual_bc, ppack,
                                head_p8(j), p_row_bytes, kPvI8BiasA);
                            tile_mma_pack_i8_sme(
                                actual_br, static_cast<int>(Dv), actual_bc, s32,
                                static_cast<int>(Dv), ppack,
                                reinterpret_cast<const int8_t*>(pack_b), /*block_stride=*/0,
                                INT32_MIN, INT32_MAX, /*zero_mode=*/true);
                            accum_tile(j);
                        }
                    }
#endif
                } else {
                    tile_pack_rhs_i8(false, static_cast<int>(Dv), actual_bc,
                                     reinterpret_cast<int8_t*>(pack_b),
                                     static_cast<int>(int8_panel_stride(actual_bc)),
                                     v8_head + koff * v_step, static_cast<int>(v_step),
                                     /*a_biased=*/false);
                    const int8_t* b8v = reinterpret_cast<const int8_t*>(pack_b);
                    for (int j = 0; j < nh; ++j) {
                        tile_pack_lhs_i8(
                            /*trans=*/false, actual_br, actual_bc, ppack,
                            static_cast<int>(geom_ldd_i8_a(actual_bc)),
                            head_p8(j), p_row_bytes, kPvI8BiasA);
                        tile_mma_pack_i8(actual_br, static_cast<int>(Dv), actual_bc, s32,
                                         static_cast<int>(Dv), ppack, b8v, /*ldb=*/0, INT32_MIN,
                                         INT32_MAX, /*zero_mode=*/true);
                        accum_tile(j);
                    }
                }
                first_tile = false;
                return j0 + actual_bc;
            }
            if (sme) {
                if constexpr (kSmeBuilt && fp_sme_capable<T>) {
                    // V is one ZA B block sequence, so the whole key tile is one
                    // k block. P is packed per tile per head; o_acc takes the
                    // result with zero_mode on the task's first tile.
                    tile_pack_rhs_sme(false, static_cast<int>(Dv), actual_bc, pack_b,
                                      vb + koff * v_step, static_cast<int>(v_step));
                    for (int j = 0; j < nh; ++j) {
                        tile_pack_lhs_sme(false, actual_br, actual_bc, sme_p,
                                          s + static_cast<int64_t>(j) * Br * actual_bc,
                                          actual_bc);
                        tile_mma_pack_sme(actual_br, static_cast<int>(Dv), actual_bc,
                                          o_acc + static_cast<int64_t>(j) * Br * Dv,
                                          static_cast<int>(Dv), sme_p, pack_b,
                                          /*block_stride=*/0, kNegInf, kPosInf, first_tile);
                    }
                    first_tile = false;
                    return j0 + actual_bc;
                }
            }
            for (int64_t k0 = 0; k0 < actual_bc; k0 += kc) {
                const int actual_kc = static_cast<int>(std::min<int64_t>(kc, actual_bc - k0));
                const int ldd_b = static_cast<int>(
                    mma_pack_align(nr_max * actual_kc * static_cast<int>(sizeof(T))) /
                    static_cast<int>(sizeof(T)));
                // V rows [j0+k0, +actual_kc) x Dv, packed once for the whole
                // Br-row block.
                if (kv8) {
                    kv_pack_i8<T>(false, static_cast<int>(Dv), actual_kc, pack_b, ldd_b,
                                  v8_head + (koff + k0) * v_step, static_cast<int>(v_step),
                                  vs_head.shifted(j0 + k0), 1.0f, i8_temp);
                } else {
                    tile_pack_rhs(false, static_cast<int>(Dv), actual_kc, pack_b, ldd_b,
                                  vb + (koff + k0) * v_step, static_cast<int>(v_step), 1.0f);
                }
                // One packed V tile feeds every batched head of the group.
                for (int j = 0; j < nh; ++j) {
                    const int64_t head_row = static_cast<int64_t>(j) * Br;
                    tile_mma_direct(actual_br, static_cast<int>(Dv), actual_kc,
                                    o_acc + head_row * Dv, static_cast<int>(Dv),
                                    s + head_row * actual_bc + k0, actual_bc, pack_b, /*ldb=*/-1,
                                    /*stage=*/nullptr, kNegInf, kPosInf, first_tile && k0 == 0);
                }
            }
            first_tile = false;
            return j0 + actual_bc;
        };

        // ---- dynamic int8 quantization of the task's query rows ----------
        // Once per task and head, before the KV stream: the same block is read
        // by every KV tile, so quantizing here is amortized over the run. The
        // scale is the row's own symmetric max — that is what keeps the int8
        // product's error small on rows with outliers — and the block is packed
        // as it is quantized, so the MMA reads the panel layout it wants.
        if (qk8) {
            for (int j = 0; j < nh; ++j) {
                const T* qsrc = q_base + j * q_h + i0 * q_rs;
                float* qs_j = q_scale + static_cast<int64_t>(j) * Br;
                for (int i = 0; i < actual_br; ++i) {
                    const float amax = row_absmax<T>(qsrc + static_cast<int64_t>(i) * q_rs, Dqk);
                    qs_j[i] = amax > 0.0f ? amax / 127.0f : 1.0f;
                }
                if (use_sme_qk) {
#if defined(NNOPS_ARCH_SME)
                    if constexpr (fp_sme_capable<T>) {
                        // The ZA packer owns its layout, so quantize into the
                        // row-major staging block and pack from it. This runs
                        // once per task and head, like the NEON fused pack.
                        for (int i = 0; i < actual_br; ++i) {
                            kernel::quant_kernel::quantization<int8_t>(
                                1, static_cast<int>(Dqk),
                                qraw + static_cast<int64_t>(i) * Dqk, static_cast<int>(Dqk),
                                qsrc + static_cast<int64_t>(i) * q_rs, static_cast<int>(q_rs),
                                &qs_j[i], nullptr);
                        }
                        tile_pack_lhs_i8_sme(
                            /*trans=*/false, actual_br, static_cast<int>(Dqk),
                            qpack + static_cast<int64_t>(j) *
                                        static_cast<int64_t>(i8_layout.q_head),
                            qraw, static_cast<int>(Dqk), kQkI8ABiased);
                    }
#endif
                } else {
                    tile_pack_quant_lhs_i8(
                        actual_br, static_cast<int>(Dqk),
                        qpack + static_cast<int64_t>(j) * static_cast<int64_t>(i8_layout.q_head),
                        static_cast<int>(geom_ldd_i8_a(Dqk)), qsrc, static_cast<int>(q_rs), qs_j,
                        nullptr, 1.0f, 0, /*u8=*/false, kQkI8ABiased);
                }
            }
        }

        // The ZA fp route's A: the query block, packed once per task and head
        // and reused by every KV tile. The softmax scale rides the pack — the
        // ZA pack has no alpha, and GEMM1's B is the cache row as stored.
        if (sme) {
            if constexpr (kSmeBuilt && fp_sme_capable<T>) {
                for (int j = 0; j < nh; ++j) {
                    T* qp = sme_q + static_cast<int64_t>(j) * sme_q_head;
                    const T* qsrc = q_base + j * q_h + i0 * q_rs;
                    tile_pack_lhs_sme(false, actual_br, static_cast<int>(Dqk), qp, qsrc,
                                      static_cast<int>(q_rs));
                    scale_packed_a(qp, fp_sme_a_elems(sme_dt, actual_br, Dqk), scale);
                }
            }
        }

        // The sink band is slice 0's; the window band is partitioned across the
        // slices, so each task streams one contiguous span of the run.
        const int64_t band_lo =
            std::max<int64_t>(key_lo(cfg, pos0, i0, Sq, run_len), prefix);
        const int64_t span = kv_end > band_lo ? kv_end - band_lo : 0;
        const int64_t slice_lo = band_lo + (span * si) / splits;
        const int64_t slice_hi = band_lo + (span * (si + 1)) / splits;
        if (si == 0) {
            int64_t j = 0;
            while (j < prefix) {
                j = do_tile(j, prefix, 0, /*sink_band=*/true);
            }
        }
        for (int64_t j = slice_lo; j < slice_hi;) {
            j = do_tile(j, slice_hi, prefix, /*sink_band=*/false);
        }

        // ---- final normalize, or the unnormalized state for the merge ----
        // The sink joins the denominator here, once: a key whose value is zero
        // leaves the accumulator untouched. A row every tile masked keeps its
        // zeros — the whole weight lands on that virtual key. A split task
        // leaves o/m/l for the merge pass instead.
        float* ptask = (splits > 1) ? partials + idx * partial_step : nullptr;
        for (int j = 0; j < nh; ++j) {
            const int64_t h = h0 + j;
            const float sink = (sink_ptr != nullptr) ? simd::s_load(&sink_ptr[h]) : kNegInf;
            const int64_t head_row = static_cast<int64_t>(j) * Br;
            for (int i = 0; i < actual_br; ++i) {
                const float l_i = simd::s_load(&l[head_row + i]);
                const float m_i = simd::s_load(&m[head_row + i]);
                const T* src = o_acc + (head_row + i) * Dv;
                if (splits > 1) {
                    float* p = ptask + (head_row + i) * (Dv + 2);
                    p[Dv] = m_i;
                    p[Dv + 1] = (m_i == kNegInf) ? 0.0f : l_i;
                    if (m_i == kNegInf) {
                        continue;
                    }
                    int64_t d = 0;
                    for (; d + lanes <= Dv; d += lanes) {
                        simd::v_store_f32(p + d, simd::v_load_f32(src + d));
                    }
                    for (; d < Dv; ++d) {
                        p[d] = simd::s_load(&src[d]);
                    }
                    continue;
                }
                float inv_l = 0.0f;
                if (m_i != kNegInf) {
                    const float denom = l_i + std::exp(sink - m_i);
                    inv_l = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
                }
                T* dst = o_base + j * o_h + (i0 + i) * o_rs;
                auto v_inv = simd::v_set1(src, inv_l);
                int64_t d = 0;
                for (; d + L <= Dv; d += L) {
                    simd::v_store(dst + d, simd::v_mul(simd::v_load(src + d), v_inv));
                }
                for (; d < Dv; ++d) {
                    simd::s_store(&dst[d], simd::s_load(&src[d]) * inv_l);
                }
            }
        }
    };

    ctx.cpu.run(0, NG, run);

    if (splits == 1) {
        return;
    }

    // ---- merge the KV slices --------------------------------------------
    // Each slice left o = Σ e·v, m = its maximum and l = Σ e at that max. Under
    // the global maximum M a slice is weighted by exp(m - M), so the
    // online-softmax identity holds across slices exactly as it does across
    // blocks within one. The sink logit joins the merged denominator once.
    const auto merge = [&](int64_t idx) {
        const int64_t qi = idx % NQ;
        const int64_t h = (idx / NQ) % H;
        const int64_t b = idx / (NQ * H);
        const int64_t hkv = h / G;
        const int64_t g = h - hkv * G;
        const int64_t c = g / hb;
        const int64_t j = g - c * hb;
        const int64_t unit = ((b * Hkv + hkv) * nb + c) * NQ + qi;
        const int64_t i0 = qi * Br;
        const int actual_br = static_cast<int>(std::min<int64_t>(Br, Sq - i0));
        const int64_t slot = scratch_slot_of(ctx.cpu, nmerge, idx);
        float* acc = merge_acc + slot * Dv;
        const float* pbase = partials + (unit * splits) * partial_step +
                             (j * Br) * (Dv + 2);
        const float sink = (sink_ptr != nullptr) ? simd::s_load(&sink_ptr[h]) : kNegInf;
        for (int i = 0; i < actual_br; ++i) {
            const float* p0 = pbase + i * (Dv + 2);
            float M = kNegInf;
            for (int64_t s = 0; s < splits; ++s) {
                M = std::max(M, p0[s * partial_step + Dv]);
            }
            T* dst = out_ptr + b * o_b + h * o_h + (i0 + i) * o_rs;
            if (M == kNegInf) {
                for (int64_t d = 0; d < Dv; ++d) {
                    simd::s_store(&dst[d], 0.0f);
                }
                continue;
            }
            for (int64_t d = 0; d < Dv; ++d) {
                acc[d] = 0.0f;
            }
            float denom = 0.0f;
            for (int64_t s = 0; s < splits; ++s) {
                const float* p = p0 + s * partial_step;
                const float ms = p[Dv];
                if (ms == kNegInf) {
                    continue;
                }
                const float w = std::exp(ms - M);
                denom += w * p[Dv + 1];
                auto vw = simd::v_set1(acc, w);
                int64_t d = 0;
                for (; d + lanes <= Dv; d += lanes) {
                    simd::v_store(acc + d,
                                  simd::v_fmadd(vw, simd::v_load(p + d), simd::v_load(acc + d)));
                }
                for (; d < Dv; ++d) {
                    acc[d] += w * p[d];
                }
            }
            if (sink_ptr != nullptr) {
                denom += std::exp(sink - M);
            }
            const float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
            auto v_inv = simd::v_set1_f32x8(inv);
            int64_t d = 0;
            for (; d + lanes <= Dv; d += lanes) {
                simd::v_store_f32(dst + d, simd::v_mul(simd::v_load(acc + d), v_inv));
            }
            for (; d < Dv; ++d) {
                simd::s_store(&dst[d], acc[d] * inv);
            }
        }
    };
    ctx.cpu.run(0, nmerge, merge);
}

} // anonymous namespace

void attention_flash_cpu(const AttentionConfig& cfg, TensorView& output,
                         std::span<const TensorView> inputs, const ComputeContext& ctx,
                         const AttentionPlan& plan, void* scratch)
{
    NNOPS_ASSERT(inputs[0].data_type() == DataType::f16 || inputs[0].data_type() == DataType::f32 ||
                 inputs[0].data_type() == DataType::bf16);
    if (inputs[0].data_type() == DataType::f16) {
        attention_flash_impl<half>(cfg, output, inputs, ctx, plan, scratch);
#ifdef NNOPS_ARCH_AARCH64
    } else if (inputs[0].data_type() == DataType::bf16) {
        attention_flash_impl<bfloat16>(cfg, output, inputs, ctx, plan, scratch);
#endif
    } else {
        attention_flash_impl<float>(cfg, output, inputs, ctx, plan, scratch);
    }
}

} // namespace nnops::backend::cpu
