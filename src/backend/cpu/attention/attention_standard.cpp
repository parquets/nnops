/// @file attention_standard.cpp
/// @brief The materialized-scores route: scores = Q·Kᵀ·scale, softmax each
///        row in place, output = attn·V.
///
/// This is the small-shape route (Sq*Sk < kFlashMinScores). It is the only one
/// that builds the whole per-head score matrix, which is why it needs the
/// largest scratch slot and why very long prefills are sent to flash instead.
///
/// One task is one (batch, query head). Both GEMMs drive the packed-B route:
/// the first packs K with the scale folded in, the second packs V. Both run
/// their first k-block with zero_mode (add_to=false) — nothing is pre-zeroed,
/// and that block skips the C read.
///
/// The mask folds into the softmax 3-pass (max → exp+sum → normalize), so
/// scores is never rewritten before normalization — one full-buffer pass
/// cheaper than an explicit `scores += mask`. A head-bias mask is indexed per
/// (batch, head) row; the flat one per row. A row only softmaxes its valid
/// range — the causal limit and the sliding window — and the lanes outside it
/// are zeroed so GEMM2 reads nothing from them. The softcap, when set, is a
/// per-row pass before the mask.

#include "attention_internal.h"
#if defined(NNOPS_ARCH_SME)
#include "backend/cpu/aarch64/sme/mma_sme.hpp" // available_sme — the ZA fp route gate
#endif

namespace nnops::backend::cpu {

namespace {

template <class T>
void attention_standard_impl(const AttentionConfig& cfg, TensorView& output,
                             std::span<const TensorView> inputs, const ComputeContext& ctx,
                             const AttentionPlan& plan, void* workspace)
{
    const auto& Q = inputs[0];
    const auto& K = inputs[1];
    const auto& V = cfg.v_is_k ? inputs[1] : inputs[2];
    const MaskView mask = resolve_mask(cfg, inputs);
    bool seqlens_plus_one = true;
    const int32_t* seqlens = run_lengths_ptr(cfg, inputs, seqlens_plus_one);

    const HeadShape hs = resolve_head_shape(cfg, inputs);
    NNOPS_ASSERT(hs.Sk_past == 0); // the plan sends a split run to flash
    const int64_t B = hs.B, H = hs.H, Hkv = hs.Hkv;
    const int64_t Sq = hs.Sq, Sk = hs.Sk, Dqk = hs.Dqk, Dv = hs.Dv;

    const float scale =
        (cfg.scale == 0.0f) ? (1.0f / std::sqrt(static_cast<float>(Dqk))) : cfg.scale;

    const T* q_ptr = Q.ptr<T>();
    const bool kv8 = kv_is_int8(inputs);
    const int8_t* k8_ptr = kv8 ? K.ptr<int8_t>() : nullptr;
    const int8_t* v8_ptr = kv8 ? V.ptr<int8_t>() : nullptr;
    const T* k_ptr = kv8 ? nullptr : K.ptr<T>();
    const T* v_ptr = kv8 ? nullptr : V.ptr<T>();
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
    const int64_t o_rs = head_seq_step(output, hs.q_layout);
    const int64_t v_head_stride = merged_v_head_stride(cfg, Dqk, Dv);

    // Batch- and head-to-head element steps, for whichever of the three storage
    // orders this call uses.
    int64_t q_b, q_h, k_b, k_h, v_b, v_h, o_b, o_h;
    head_strides(Q, hs.q_layout, Dqk, q_b, q_h);
    head_strides(K, hs.kv_layout, Dqk, k_b, k_h);
    head_strides(V, hs.kv_layout, v_head_stride, v_b, v_h);
    head_strides(output, hs.q_layout, Dv, o_b, o_h);

    constexpr int nr_max = nr_max_flt<T>();
    constexpr int64_t kc = kc_for<T>();

    const int mc1 = plan.mc1;
    const int nc1 = plan.nc1;
    const int mc2 = plan.mc2;
    const int nc2 = plan.nc2;

    // Scratch bounds: the packed-B slot both GEMMs share, plus the ZA engine's
    // packed query block and probability tile. The plan sizes the slot with the
    // same helpers, so the two cannot disagree.
    const int64_t scores_elems = Sq * Sk;
    const int64_t pack_elems = standard_b_pack_elems<T>(nc1, nc2, Dqk, Dv, Sk);
    const int64_t i8_elems =
        kv8 ? kv_i8_panel_elems<T>(kc_pack_bound<T>(std::max(Dqk, Sk))) : 0;
    const int64_t sme_q_elems = standard_sme_q_elems<T>(Sq, Dqk);
    const int64_t sme_p_elems = standard_sme_p_elems<T>(mc2, Sk);
    const int64_t per_head = scores_elems + pack_elems + i8_elems + sme_q_elems + sme_p_elems;

    // The ZA fp engine: f32/f16, a dense cache, and a head dim that fits one ZA
    // k block (the packed query block is one slab). The prob tile's k blocks are
    // the engine's own, independent of the NEON plan's kc.
    constexpr DataType sme_dt = std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
    bool sme = false;
#if defined(NNOPS_ARCH_SME)
    if constexpr (kSmeBuilt && fp_sme_capable<T>) {
        sme = !kv8 && aarch64::available_sme() && Dqk <= sme_kc_for(sme_dt);
    }
#endif
    const int64_t sme_kc = sme_kc_for(sme_dt);

    const int64_t NG = B * H;
    T* ws_base = static_cast<T*>(workspace);

    const auto run = [&](int64_t idx) {
        const int64_t b = idx / H;
        const int64_t h = idx % H;
        const int64_t hkv = h / (H / Hkv);

        const T* q_head = q_ptr + b * q_b + h * q_h;
        const T* k_head = kv8 ? nullptr : k_ptr + b * k_b + hkv * k_h;
        const T* v_head = kv8 ? nullptr : v_ptr + b * v_b + hkv * v_h;
        const int8_t* k8_head = kv8 ? k8_ptr + b * k_b + hkv * k_h : nullptr;
        const int8_t* v8_head = kv8 ? v8_ptr + b * v_b + hkv * v_h : nullptr;
        const KvScaleRow ks_head = kscale.row(b, hkv, Hkv, Sk);
        const KvScaleRow vs_head = vscale.row(b, hkv, Hkv, Sk);
        T* o_head = out_ptr + b * o_b + h * o_h;

        const int64_t slot = scratch_slot_of(ctx.cpu, NG, idx);
        T* scores = ws_base + slot * per_head;
        T* pack_b = scores + scores_elems;
        T* i8_temp = pack_b + pack_elems;
        T* sme_q = i8_temp + i8_elems; // the ZA route's packed query block
        T* sme_p = sme_q + sme_q_elems;
        const float sink = sink_ptr != nullptr ? simd::s_load(&sink_ptr[h]) : kNegInf;

        // The batch's valid key run: its prefix of the [past | new] rows, and
        // where this batch's query rows sit (vector > scalar > bottom-right).
        const int64_t run = run_length(seqlens, b, Sk, seqlens_plus_one);
        const int64_t pos0 = batch_pos0(cfg, inputs, b);

        // ---- GEMM1: scores = Q @ K^T * scale --------------------------
        // k0 == 0 runs zero_mode (overwrite) — no pre-zeroing pass.
        if (sme) {
            if constexpr (kSmeBuilt && fp_sme_capable<T>) {
                // The query block is packed once per head, scale folded in; the
                // ZA pack has no alpha and GEMM1's B is K as stored. The A slab
                // is block-major, so the M loop stays on 32-row boundaries.
                tile_pack_lhs_sme(false, static_cast<int>(Sq), static_cast<int>(Dqk), sme_q,
                                  q_head, static_cast<int>(q_rs));
                scale_packed_a(sme_q, sme_q_elems, scale);
                constexpr int kSmeMc = 64;
                for (int64_t n = 0; n < Sk; n += nc1) {
                    const int actual_nc = static_cast<int>(std::min<int64_t>(nc1, Sk - n));
                    tile_pack_rhs_sme(true, actual_nc, static_cast<int>(Dqk), pack_b,
                                      k_head + n * k_rs, static_cast<int>(k_rs));
                    for (int64_t m = 0; m < Sq; m += kSmeMc) {
                        const int actual_mc =
                            static_cast<int>(std::min<int64_t>(kSmeMc, Sq - m));
                        tile_mma_pack_sme(
                            actual_mc, actual_nc, static_cast<int>(Dqk), scores + m * Sk + n,
                            static_cast<int>(Sk),
                            sme_q + fp_sme_a_row_offset(sme_dt, Dqk, m), pack_b,
                            /*block_stride=*/0, kNegInf, kPosInf, /*zero_mode=*/true);
                    }
                }
            }
        } else {
        for (int64_t k0 = 0; k0 < Dqk; k0 += kc) {
            const int actual_kc = static_cast<int>(std::min<int64_t>(kc, Dqk - k0));
            const int ldd_b =
                static_cast<int>(mma_pack_align(nr_max * actual_kc * static_cast<int>(sizeof(T))) /
                                 static_cast<int>(sizeof(T)));
            for (int64_t n = 0; n < Sk; n += nc1) {
                const int actual_nc = static_cast<int>(std::min<int64_t>(nc1, Sk - n));
                if (kv8) {
                    kv_pack_i8<T>(true, actual_nc, actual_kc, pack_b, ldd_b,
                                  k8_head + n * k_rs + k0, static_cast<int>(k_rs),
                                  ks_head.shifted(n), scale, i8_temp);
                } else {
                    tile_pack_rhs(true, actual_nc, actual_kc, pack_b, ldd_b,
                                  k_head + n * k_rs + k0, static_cast<int>(k_rs), scale);
                }
                for (int64_t m = 0; m < Sq; m += mc1) {
                    const int actual_mc = static_cast<int>(std::min<int64_t>(mc1, Sq - m));
                    tile_mma_direct(actual_mc, actual_nc, actual_kc, scores + m * Sk + n,
                                    static_cast<int>(Sk), q_head + m * q_rs + k0,
                                    static_cast<int>(q_rs), pack_b, /*ldb=*/-1, /*stage=*/nullptr,
                                    kNegInf, kPosInf, k0 == 0);
                }
            }
        }
        }

        // ---- softcap + mask + softmax over the last dim, per row ------
        // The visible keys are the batch run's prefix [0, hi) and, when a
        // sliding window applies, the band [lo, hi); sink_tokens keeps the
        // first `prefix` positions visible outside the window, so the hidden
        // part is the hole [prefix, lo). The softcap and the keep-mask run
        // first, the hole is hidden, and one softmax pass covers [from, hi)
        // (from is 0 with sinks, lo without). Lanes outside it stay zero so
        // GEMM2 reads nothing from them.
        const int64_t prefix = sink_prefix(cfg, run);
        for (int64_t i = 0; i < Sq; ++i) {
            T* row = scores + i * Sk;
            const int64_t hi = causal_limit(cfg, pos0, i, Sq, run);
            const int64_t lo = key_lo(cfg, pos0, i, Sq, run);
            const int64_t from = prefix > 0 ? 0 : lo;
            if (hi <= from) {
                row_fill(row, Sk, 0.0f);
                continue;
            }
            if (from > 0) {
                row_fill(row, from, 0.0f);
            }
            if (cfg.softcap > 0.0f) {
                row_softcap(row + from, hi - from, 1.0f, cfg.softcap);
            }
            if (mask.keep != nullptr) {
                const int64_t off = mask_offset(b, h, i, mask.keep_b, mask.keep_h, mask.keep_row);
                row_apply_keep(row + from, mask.keep + off + from, hi - from, mask.filter);
            }
            if (prefix > 0 && lo > prefix) {
                row_fill(row + prefix, lo - prefix, kNegInf);
            }
            if (mask_ptr != nullptr) {
                const int64_t off = mask_offset(b, h, i, mask.bias_b, mask.bias_h, mask.bias_row);
                kernel::mask_softmax_process_standard_row<T>(row + from, mask_ptr + off + from,
                                                             row + from, hi - from,
                                                             /*log_softmax=*/false, 1.0f, sink);
            } else {
                kernel::softmax_process_standard_row<T>(row + from, row + from, hi - from,
                                                        /*log_softmax=*/false, 1.0f, sink);
            }
            if (hi < Sk) {
                row_fill(row + hi, Sk - hi, 0.0f);
            }
        }

        // ---- GEMM2: output = attn @ V ---------------------------------
        if (sme) {
            if constexpr (kSmeBuilt && fp_sme_capable<T>) {
                // V packed per (k-block, n-block); the probability tile packed
                // per (m-block, k-block) and reused across the n blocks.
                for (int64_t k0 = 0; k0 < Sk; k0 += sme_kc) {
                    const int actual_kc = static_cast<int>(std::min(sme_kc, Sk - k0));
                    for (int64_t n = 0; n < Dv; n += nc2) {
                        const int actual_nc = static_cast<int>(std::min<int64_t>(nc2, Dv - n));
                        tile_pack_rhs_sme(false, actual_nc, actual_kc, pack_b,
                                          v_head + k0 * v_rs + n, static_cast<int>(v_rs));
                        for (int64_t m = 0; m < Sq; m += mc2) {
                            const int actual_mc =
                                static_cast<int>(std::min<int64_t>(mc2, Sq - m));
                            tile_pack_lhs_sme(false, actual_mc, actual_kc, sme_p,
                                              scores + m * Sk + k0, static_cast<int>(Sk));
                            tile_mma_pack_sme(actual_mc, actual_nc, actual_kc,
                                              o_head + m * o_rs + n, static_cast<int>(o_rs),
                                              sme_p, pack_b, /*block_stride=*/0, kNegInf, kPosInf,
                                              k0 == 0);
                        }
                    }
                }
            }
        } else {
        for (int64_t k0 = 0; k0 < Sk; k0 += kc) {
            const int actual_kc = static_cast<int>(std::min<int64_t>(kc, Sk - k0));
            const int ldd_b =
                static_cast<int>(mma_pack_align(nr_max * actual_kc * static_cast<int>(sizeof(T))) /
                                 static_cast<int>(sizeof(T)));
            for (int64_t n = 0; n < Dv; n += nc2) {
                const int actual_nc = static_cast<int>(std::min<int64_t>(nc2, Dv - n));
                // V is the B side: packed once per (k-block, n-block), then
                // every m-block reuses it.
                if (kv8) {
                    kv_pack_i8<T>(false, actual_nc, actual_kc, pack_b, ldd_b,
                                  v8_head + k0 * v_rs + n, static_cast<int>(v_rs),
                                  vs_head.shifted(k0), 1.0f, i8_temp);
                } else {
                    tile_pack_rhs(false, actual_nc, actual_kc, pack_b, ldd_b,
                                  v_head + k0 * v_rs + n, static_cast<int>(v_rs), 1.0f);
                }
                for (int64_t m = 0; m < Sq; m += mc2) {
                    const int actual_mc = static_cast<int>(std::min<int64_t>(mc2, Sq - m));
                    tile_mma_direct(actual_mc, actual_nc, actual_kc, o_head + m * o_rs + n,
                                    static_cast<int>(o_rs), scores + m * Sk + k0,
                                    static_cast<int>(Sk), pack_b, /*ldb=*/-1, /*stage=*/nullptr,
                                    kNegInf, kPosInf, k0 == 0);
                }
            }
        }
        }
    };

    ctx.cpu.run(0, NG, run);
}

} // anonymous namespace

void attention_standard_cpu(const AttentionConfig& cfg, TensorView& output,
                            std::span<const TensorView> inputs, const ComputeContext& ctx,
                            const AttentionPlan& plan, void* scratch)
{
    NNOPS_ASSERT(inputs[0].data_type() == DataType::f16 || inputs[0].data_type() == DataType::f32 ||
                 inputs[0].data_type() == DataType::bf16);
    if (inputs[0].data_type() == DataType::f16) {
        attention_standard_impl<half>(cfg, output, inputs, ctx, plan, scratch);
#ifdef NNOPS_ARCH_AARCH64
    } else if (inputs[0].data_type() == DataType::bf16) {
        attention_standard_impl<bfloat16>(cfg, output, inputs, ctx, plan, scratch);
#endif
    } else {
        attention_standard_impl<float>(cfg, output, inputs, ctx, plan, scratch);
    }
}

} // namespace nnops::backend::cpu
