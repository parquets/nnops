/// @file attention_flash.cpp
/// @brief FlashAttention-1: stream KV in Br×Bc tiles with an online softmax,
///        so the Sq×Sk score matrix never exists.
///
/// One task is one Br-row query block of one (batch, query head) — the grid is
/// batch × query head × query-block, so long-sequence prefill keeps its
/// parallelism even when batch × head is small. Each task walks the run in
/// Bc-key tiles, carrying a running max (m), a running sum (l) and an output
/// accumulator (o), and folds the mask and the causal bound into the S tile
/// before the update.
///
/// The run may be two segments, an optional past tensor followed by the step's
/// new K/V; each is streamed in place, so nothing is concatenated. A causal
/// query block stops its KV stream at the block's causal limit, so prefill never
/// walks the masked upper triangle.
///
/// Both GEMMs drive tile_mma_direct with a packed B. The first k-block of the
/// whole task runs zero_mode, which is what lets o_acc skip a pre-zeroing pass.

#include "attention_internal.h"

namespace nnops::backend::cpu {

namespace {

// ---- tiled FlashAttention kernel -----------------------------------------
//
// One task = one query block (Br rows) of one (batch, head): it streams KV in
// Bc-sized blocks, carrying a running max/sum and an output accumulator, so
// the Sq×Sk scores matrix never exists. Reuses the packed-B route above.

template <class T>
void attention_flash_impl(const AttentionConfig& cfg,
                          TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx,
                          const AttentionPlan& plan,
                          void* workspace)
{
    const auto& Q = inputs[0];
    const auto& K = inputs[1];
    const auto& V = cfg.v_is_k ? inputs[1] : inputs[2];
    const bool has_mask = has_mask_input(cfg, inputs);
    const bool has_bias = mask_is_head_bias(cfg, inputs);
    const int64_t bias_rs = has_mask
        ? inputs[mask_slot(cfg)].row_stride_elems() : 0;
    const bool split = has_past_view(cfg, inputs);
    const auto& Kp = split ? inputs[past_slot(cfg)] : K;
    const auto& Vp = split ? inputs[past_slot(cfg) + (cfg.v_is_k ? 0 : 1)] : V;
    const int32_t* seqlens = has_seqlens_input(cfg, inputs)
        ? inputs[seqlens_slot(cfg)].ptr<int32_t>() : nullptr;

    const HeadShape hs = resolve_head_shape(cfg, inputs);
    const int64_t B = hs.B, H = hs.H, Hkv = hs.Hkv;
    const int64_t Sq = hs.Sq, Dqk = hs.Dqk, Dv = hs.Dv;
    const int64_t Sk_new = hs.Sk, Sk_past = hs.Sk_past;
    const int64_t Sk = Sk_new + Sk_past;   // the whole [past | new] run
    const bool merged = hs.merged;

    const float scale = (cfg.scale == 0.0f)
        ? (1.0f / std::sqrt(static_cast<float>(Dqk)))
        : cfg.scale;

    const T* q_ptr = Q.ptr<T>();
    const T* k_ptr = K.ptr<T>();
    const T* v_ptr = V.ptr<T>();
    const T* kp_ptr = split ? Kp.ptr<T>() : nullptr;
    const T* vp_ptr = split ? Vp.ptr<T>() : nullptr;
    const T* mask_ptr = has_mask ? inputs[mask_slot(cfg)].ptr<T>() : nullptr;
    T* out_ptr = output.ptr<T>();

    const int64_t q_rs = Q.row_stride_elems();
    const int64_t k_rs = K.row_stride_elems();
    const int64_t v_rs = V.row_stride_elems();
    const int64_t kp_rs = split ? Kp.row_stride_elems() : k_rs;
    const int64_t vp_rs = split ? Vp.row_stride_elems() : v_rs;
    const int64_t o_rs = output.row_stride_elems();
    const int64_t v_head_stride = merged_v_head_stride(cfg, Dqk, Dv);

    constexpr int nr_max = nr_max_flt<T>();
    constexpr int64_t kc = kc_for<T>();
    constexpr int L = simd::simd_lane_for<T>;

    const int Br = plan.Br;
    const int Bc = plan.Bc;
    const int64_t per_task = flash_per_task_elems<T>(Br, Bc, Dqk, Dv);

    const int64_t NQ = (Sq + Br - 1) / Br;  // number of Br-row query blocks
    const int64_t NG = B * H * NQ;
    T* ws_base = static_cast<T*>(workspace);

    const auto run = [&](int64_t idx) {
        const int64_t qi = idx % NQ;
        const int64_t b  = (idx / NQ) / H;
        const int64_t h  = (idx / NQ) % H;
        const int64_t hkv = h / (H / Hkv);
        const int64_t i0 = qi * Br;
        const int actual_br = static_cast<int>(std::min<int64_t>(Br, Sq - i0));
        // The batch's valid key run: its prefix of the [past | new] rows.
        const int64_t run_len = run_length(seqlens, b, Sk);

        const T* q_head;
        const T* k_head;
        const T* v_head;
        T* o_head;
        if (merged) {
            q_head = q_ptr + b * Sq * q_rs + h * Dqk;
            k_head = k_ptr + b * Sk_new * k_rs + hkv * Dqk;
            v_head = v_ptr + b * hs.Sv * v_rs + hkv * v_head_stride;
            o_head = out_ptr + b * Sq * o_rs + h * Dv;
        } else {
            q_head = q_ptr + (b * H + h) * Sq * q_rs;
            k_head = k_ptr + (b * Hkv + hkv) * Sk_new * k_rs;
            v_head = v_ptr + (b * Hkv + hkv) * hs.Sv * v_rs;
            o_head = out_ptr + (b * H + h) * Sq * o_rs;
        }
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
            k_base[0] = merged ? kp_ptr + b * Sk_past * kp_rs + hkv * Dqk
                               : kp_ptr + (b * Hkv + hkv) * Sk_past * kp_rs;
            v_base[0] = merged ? vp_ptr + b * Sk_past * vp_rs + hkv * v_head_stride
                               : vp_ptr + (b * Hkv + hkv) * Sk_past * vp_rs;
            k_str[0] = kp_rs;
            v_str[0] = vp_rs;
            seg_lo[1] = Sk_past;
            seg_hi[0] = Sk_past;
        }

        const int64_t slot = scratch_slot_of(ctx.cpu, NG, idx);
        T* m      = ws_base + slot * per_task;
        T* l      = m + Br;
        T* s      = l + Br;
        T* o_acc  = s + static_cast<int64_t>(Br) * Bc;
        T* pack_b = o_acc + static_cast<int64_t>(Br) * Dv;

        // Reset the running online-softmax state through the scalar simd
        // bridge: std::numeric_limits<T> is not specialized for half (it
        // silently yields 0, not -inf), and static_cast<half>(float) truncates
        // to an integer and stores it as bits on the x86 half wrapper.
        for (int i = 0; i < actual_br; ++i) {
            simd::s_store(&m[i], -std::numeric_limits<float>::infinity());
            simd::s_store(&l[i], 0.0f);
        }
        // Causal early exit: the causal limit grows with the query row, so the
        // last row of the block bounds every row in it. A KV block starting at
        // or past that bound is fully masked for all of them and contributes
        // nothing — for a causal prefill that is the whole upper triangle, and
        // skipping it halves the QK and AV work.
        const int64_t kv_end = cfg.causal
            ? causal_limit(cfg, i0 + actual_br - 1, Sq, run_len)
            : run_len;
        // The window's lower bound grows with the query row, so the block's
        // first row has the smallest one: KV tiles below it are fully masked
        // for every row and the stream can start there.
        int64_t j0 = key_lo(cfg, i0, Sq, run_len);

        // o_acc is written by the first KV block (zero_mode) — no pre-zeroing
        // pass. The exp_diff rescale may touch it first on the first block;
        // that garbage is fully overwritten by the first MMA.
        bool first_tile = true;
        while (j0 < kv_end) {
            const int seg = (j0 < seg_lo[1]) ? 0 : 1;
            const int64_t seg_end = std::min(seg_hi[seg], kv_end);
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

            // ---- S = Q[i0:i0+Br] @ K[j0:j0+Bc]^T * scale ----------
            // k0 == 0 overwrites (zero_mode) — each KV block owns its S tile.
            for (int64_t k0 = 0; k0 < Dqk; k0 += kc) {
                const int actual_kc = static_cast<int>(std::min<int64_t>(kc, Dqk - k0));
                const int ldd_b = static_cast<int>(
                    mma_pack_align(nr_max * actual_kc * static_cast<int>(sizeof(T)))
                    / static_cast<int>(sizeof(T)));
                tile_pack_rhs(true, actual_bc, actual_kc,
                                      pack_b, ldd_b,
                                      kb + koff * k_step + k0, static_cast<int>(k_step), scale);
                tile_mma_direct(actual_br, actual_bc, actual_kc,
                                        s, actual_bc,
                                        q_head + i0 * q_rs + k0, static_cast<int>(q_rs),
                                        pack_b, /*ldb=*/-1, /*stage=*/nullptr, kNegInf, kPosInf, k0 == 0);
            }

            // ---- softcap, additive mask and the out-of-range lanes ----
            // The cap is part of the logit, so it runs before the mask. The
            // mask, the causal tail and the window prefix are all per-row, so
            // one pass covers them: add the mask over the row's valid range
            // [n0, n1), then fill the rest of the tile with -inf.
            if (cfg.softcap > 0.0f) {
                for (int i = 0; i < actual_br; ++i) {
                    row_softcap(s + i * actual_bc, actual_bc, 1.0f, cfg.softcap);
                }
            }
            if (mask_ptr || cfg.causal || cfg.window > 0) {
                for (int i = 0; i < actual_br; ++i) {
                    const int64_t row_i = i0 + i;
                    const int64_t lo_i = key_lo(cfg, row_i, Sq, run_len);
                    const int64_t lim = causal_limit(cfg, row_i, Sq, run_len);
                    int64_t n0 = lo_i - j0;
                    int64_t n1 = lim - j0;
                    if (n0 < 0) { n0 = 0; }
                    if (n0 > actual_bc) { n0 = actual_bc; }
                    if (n1 < n0) { n1 = n0; }
                    if (n1 > actual_bc) { n1 = actual_bc; }
                    const int64_t n = n1 - n0;
                    T* srow = s + i * actual_bc;
                    if (mask_ptr) {
                        const T* mrow = has_bias
                            ? mask_ptr + head_bias_row_offset(b, h, row_i, H, Sq, bias_rs)
                                       + j0 + n0
                            : mask_ptr + row_i * Sk + j0 + n0;
                        int64_t j = 0;
                        for (; j + L <= n; j += L) {
                            simd::v_store(srow + n0 + j,
                                simd::v_add(simd::v_load(srow + n0 + j), simd::v_load(mrow + j)));
                        }
                        for (; j < n; ++j) {
                            simd::s_store(&srow[n0 + j],
                                simd::s_load(&srow[n0 + j]) + simd::s_load(&mrow[j]));
                        }
                    }
                    row_fill(srow, n0, -std::numeric_limits<float>::infinity());
                    row_fill(srow + n1, actual_bc - n1,
                             -std::numeric_limits<float>::infinity());
                }
            }

            // ---- online-softmax update + accumulate into o_acc -----
            for (int i = 0; i < actual_br; ++i) {
                T* srow = s + i * actual_bc;
                const float rowmax = kernel::softmax_row_max<T>(srow, actual_bc);
                if (rowmax == kNegInf) {
                    // The whole tile is out of range for this row (a causal
                    // row above the diagonal): S must be zero or the AV GEMM
                    // would add -inf·v.
                    row_fill(srow, actual_bc, 0.0f);
                    continue;
                }
                const float old_m = simd::s_load(&m[i]);
                const float new_m = std::max(old_m, rowmax);
                const float exp_diff = std::exp(old_m - new_m);  // 0 on the first KV block
                simd::s_store(&m[i], new_m);
                const float rowsum = kernel::softmax_row_exp_sum<T>(srow, srow, actual_bc, new_m, 1.0f);
                simd::s_store(&l[i], exp_diff * simd::s_load(&l[i]) + rowsum);

                if (exp_diff != 1.0f) {
                    T* orow = o_acc + i * Dv;
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

            // ---- o_acc += S @ V[j0:j0+Bc] --------------------------
            for (int64_t k0 = 0; k0 < actual_bc; k0 += kc) {
                const int actual_kc = static_cast<int>(std::min<int64_t>(kc, actual_bc - k0));
                const int ldd_b = static_cast<int>(
                    mma_pack_align(nr_max * actual_kc * static_cast<int>(sizeof(T)))
                    / static_cast<int>(sizeof(T)));
                // V rows [j0+k0, +actual_kc) x Dv, packed once for the whole
                // Br-row block.
                tile_pack_rhs(false, static_cast<int>(Dv), actual_kc,
                                      pack_b, ldd_b,
                                      vb + (koff + k0) * v_step, static_cast<int>(v_step),
                                      1.0f);
                tile_mma_direct(actual_br, static_cast<int>(Dv), actual_kc,
                                        o_acc, static_cast<int>(Dv),
                                        s + k0, actual_bc,
                                        pack_b, /*ldb=*/-1, /*stage=*/nullptr, kNegInf, kPosInf, first_tile && k0 == 0);
            }
            first_tile = false;
            j0 += actual_bc;
        }

        // ---- final normalize: output = o_acc / l -------------------
        for (int i = 0; i < actual_br; ++i) {
            const float l_i = simd::s_load(&l[i]);
            // A row every tile masked has l == 0; its output is zeros.
            const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
            const T* src = o_acc + i * Dv;
            T* dst = o_head + (i0 + i) * o_rs;
            auto v_inv = simd::v_set1(src, inv_l);
            int64_t d = 0;
            for (; d + L <= Dv; d += L) {
                simd::v_store(dst + d, simd::v_mul(simd::v_load(src + d), v_inv));
            }
            for (; d < Dv; ++d) {
                simd::s_store(&dst[d], simd::s_load(&src[d]) * inv_l);
            }
        }
    };

    ctx.cpu.run(0, NG, run);
}

}  // anonymous namespace

void attention_flash_cpu(const AttentionConfig& cfg,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         const AttentionPlan& plan,
                         void* scratch)
{
    NNOPS_ASSERT(inputs[0].data_type() == DataType::f16 ||
                 inputs[0].data_type() == DataType::f32 ||
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

}  // namespace nnops::backend::cpu
