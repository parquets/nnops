/// @file attention_decode.cpp
/// @brief The Sq == 1 route: both products collapse to GEMV, so K and V are
///        streamed once each, with no B packing and no M == 1 MMA tiles.
///
/// Scores and the output accumulator are f32 whatever the element type — f16
/// converts once per loaded vector instead of accumulating in f16, which also
/// keeps the deviation inside kF16AccumTol.
///
/// One task serves up to kDecodeHeadBatch query heads of the same KV head. The
/// key loop stays outermost, so the run is read from memory once for the whole
/// group instead of once per query head; the per-head dots against a key row
/// hit L1. That is the GQA/MQA traffic win — with H/H_kv = 4 each K/V byte was
/// otherwise fetched four times.
///
/// The run is walked in kDecodeChunk-key blocks with the online-softmax update
/// of the flash route, so the score scratch does not grow with the cache. When
/// the batched grid no longer covers the workers, KV splitting cuts the run
/// into independent slices; each task leaves its unnormalized output, max and
/// sum, and a merge pass combines them — the flash-decoding schedule.

#include "attention_internal.h"

namespace nnops::backend::cpu {

namespace {

/// Dot product of two n-wide rows, f32 accumulation. The key row may be stored
/// fp (widened on load) or as int8 bytes (expanded and scaled by the caller's
/// pre-scaled query row).
template <class Q, class KV> inline float dot_row(const Q* a, const KV* b, int64_t n)
{
    constexpr int L = simd::simd_lane_for<Q>;
    auto acc = simd::v_zero_f32x8();
    int64_t i = 0;
    for (; i + L <= n; i += L) {
        acc = simd::v_fmadd(simd::v_load_f32(a + i), kv_load_f32(b + i), acc);
    }
    float s = simd::v_reduce_sum(acc);
    for (; i < n; ++i) {
        s += simd::s_load(&a[i]) * kv_load_scalar(&b[i]);
    }
    return s;
}

/// Four n-wide dot products against one query row, written to @p out. A
/// horizontal reduce costs about as much as the 16 vector FMAs of a 128-wide
/// row, so four keys per reduce is what keeps the GEMV off the latency floor.
template <class Q, class KV>
inline void dot_row4(const Q* q, const KV* k0, const KV* k1, const KV* k2, const KV* k3,
                     int64_t n, float* out)
{
    constexpr int L = simd::simd_lane_for<Q>;
    auto a0 = simd::v_zero_f32x8();
    auto a1 = a0, a2 = a0, a3 = a0;
    int64_t i = 0;
    for (; i + L <= n; i += L) {
        auto qv = simd::v_load_f32(q + i);
        a0 = simd::v_fmadd(qv, kv_load_f32(k0 + i), a0);
        a1 = simd::v_fmadd(qv, kv_load_f32(k1 + i), a1);
        a2 = simd::v_fmadd(qv, kv_load_f32(k2 + i), a2);
        a3 = simd::v_fmadd(qv, kv_load_f32(k3 + i), a3);
    }
    float s0 = simd::v_reduce_sum(a0);
    float s1 = simd::v_reduce_sum(a1);
    float s2 = simd::v_reduce_sum(a2);
    float s3 = simd::v_reduce_sum(a3);
    for (; i < n; ++i) {
        const float qs = simd::s_load(&q[i]);
        s0 += qs * kv_load_scalar(&k0[i]);
        s1 += qs * kv_load_scalar(&k1[i]);
        s2 += qs * kv_load_scalar(&k2[i]);
        s3 += qs * kv_load_scalar(&k3[i]);
    }
    out[0] = s0;
    out[1] = s1;
    out[2] = s2;
    out[3] = s3;
}

/// The same four keys dotted against two query rows. The K vector load (and its
/// f16/int8 widen) feeds both FMAs, halving the K-side load slots of the
/// head-batched GQA loop — the phase's limit is loads, not FMAs. Eight
/// accumulators + two Q + one K stay inside the 16 YMM registers.
template <class Q, class KV>
inline void dot_row4x2(const Q* q0, const Q* q1, const KV* k0, const KV* k1, const KV* k2,
                       const KV* k3, int64_t n, float* out0, float* out1)
{
    constexpr int L = simd::simd_lane_for<Q>;
    auto a0 = simd::v_zero_f32x8();
    auto a1 = a0, a2 = a0, a3 = a0;
    auto b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    int64_t i = 0;
    for (; i + L <= n; i += L) {
        const auto qv0 = simd::v_load_f32(q0 + i);
        const auto qv1 = simd::v_load_f32(q1 + i);
        auto kv = kv_load_f32(k0 + i);
        a0 = simd::v_fmadd(qv0, kv, a0);
        b0 = simd::v_fmadd(qv1, kv, b0);
        kv = kv_load_f32(k1 + i);
        a1 = simd::v_fmadd(qv0, kv, a1);
        b1 = simd::v_fmadd(qv1, kv, b1);
        kv = kv_load_f32(k2 + i);
        a2 = simd::v_fmadd(qv0, kv, a2);
        b2 = simd::v_fmadd(qv1, kv, b2);
        kv = kv_load_f32(k3 + i);
        a3 = simd::v_fmadd(qv0, kv, a3);
        b3 = simd::v_fmadd(qv1, kv, b3);
    }
    float s0 = simd::v_reduce_sum(a0);
    float s1 = simd::v_reduce_sum(a1);
    float s2 = simd::v_reduce_sum(a2);
    float s3 = simd::v_reduce_sum(a3);
    float t0 = simd::v_reduce_sum(b0);
    float t1 = simd::v_reduce_sum(b1);
    float t2 = simd::v_reduce_sum(b2);
    float t3 = simd::v_reduce_sum(b3);
    for (; i < n; ++i) {
        const float qa = simd::s_load(&q0[i]);
        const float qb = simd::s_load(&q1[i]);
        const float kk0 = kv_load_scalar(&k0[i]);
        const float kk1 = kv_load_scalar(&k1[i]);
        const float kk2 = kv_load_scalar(&k2[i]);
        const float kk3 = kv_load_scalar(&k3[i]);
        s0 += qa * kk0;
        s1 += qa * kk1;
        s2 += qa * kk2;
        s3 += qa * kk3;
        t0 += qb * kk0;
        t1 += qb * kk1;
        t2 += qb * kk2;
        t3 += qb * kk3;
    }
    out0[0] = s0;
    out0[1] = s1;
    out0[2] = s2;
    out0[3] = s3;
    out1[0] = t0;
    out1[1] = t1;
    out1[2] = t2;
    out1[3] = t3;
}

template <class T, class KV>
void attention_decode_impl(const AttentionConfig& cfg, TensorView& output,
                           std::span<const TensorView> inputs, const ComputeContext& ctx,
                           const AttentionPlan& plan, void* workspace)
{
    constexpr bool kInt8 = std::is_same_v<KV, int8_t>;
    const auto& Q = inputs[0];
    const auto& K = inputs[1];
    const auto& V = cfg.v_is_k ? inputs[1] : inputs[2];
    const bool split = has_past_view(cfg, inputs);
    NNOPS_ASSERT(!(kInt8 && split)); // a quantized cache has no past segment
    const auto& Kp = split ? inputs[static_cast<size_t>(cfg.past_k_slot)] : K;
    const auto& Vp = split ? inputs[static_cast<size_t>(cfg.past_v_slot)] : V;
    bool seqlens_plus_one = true;
    const int32_t* seqlens = run_lengths_ptr(cfg, inputs, seqlens_plus_one);

    const HeadShape hs = resolve_head_shape(cfg, inputs);
    const int64_t B = hs.B, H = hs.H, Hkv = hs.Hkv;
    const int64_t Sq = hs.Sq, Dqk = hs.Dqk, Dv = hs.Dv;
    const int64_t Sk_new = hs.Sk, Sk_past = hs.Sk_past;
    const int64_t Sk = Sk_new + Sk_past; // the whole [past | new] run
    NNOPS_ASSERT(Sq == 1);

    const float scale =
        (cfg.scale == 0.0f) ? (1.0f / std::sqrt(static_cast<float>(Dqk))) : cfg.scale;
    // The scale normally rides through the softmax exponential. A softcap is
    // applied to the scaled logit itself, so in that case it is folded into the
    // cap pass and the exponential runs unscaled.
    const float exp_scale = cfg.softcap > 0.0f ? 1.0f : scale;

    const T* q_ptr = Q.ptr<T>();
    const KV* k_ptr = K.ptr<KV>();
    const KV* v_ptr = V.ptr<KV>();
    const KV* kp_ptr = split ? Kp.ptr<KV>() : nullptr;
    const KV* vp_ptr = split ? Vp.ptr<KV>() : nullptr;
    const KvScale kscale = resolve_kv_scale(cfg, cfg.k_scale_slot, inputs);
    const KvScale vscale = resolve_kv_scale(cfg, cfg.v_scale_slot, inputs);
    const bool has_sink = slot_present(cfg, cfg.sink_slot, inputs);
    const T* sink_ptr = has_sink ? inputs[static_cast<size_t>(cfg.sink_slot)].ptr<T>() : nullptr;
    T* out_ptr = output.ptr<T>();

    const int64_t k_rs = head_seq_step(K, hs.kv_layout);
    const int64_t v_rs = head_seq_step(V, hs.kv_layout);
    const int64_t kp_rs = split ? head_seq_step(Kp, hs.kv_layout) : k_rs;
    const int64_t vp_rs = split ? head_seq_step(Vp, hs.kv_layout) : v_rs;
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

    const int64_t G = H / Hkv;
    const int64_t hb = plan.decode_heads;
    const int64_t splits = plan.decode_splits;
    const int64_t nb = (G + hb - 1) / hb; // head blocks per KV head
    const int64_t units = B * Hkv * nb;
    const int64_t NG = units * splits;

    const int64_t per_slot = plan.decode_elems; // f32: [scores][o_acc][max][sum]
    float* ws = static_cast<float*>(workspace);
    float* partials = ws + plan.num_slots * per_slot;

    constexpr int64_t lanes = simd::simd_default_lane_f32;

    const auto run = [&](int64_t idx) {
        // Grid (batch, kv head, head block) × KV slice. A slice is a
        // contiguous span of the run, empty when there are more slices than
        // keys; its head block holds up to hb query heads of one KV head.
        const int64_t si = idx % splits;
        const int64_t unit = idx / splits;
        const int64_t c = unit % nb;
        const int64_t hkv = (unit / nb) % Hkv;
        const int64_t b = (unit / nb) / Hkv;
        const int64_t h0 = hkv * G + c * hb;
        const int64_t nh = std::min(hb, G - c * hb);

        // The batch's valid key run and this row's limit over it: bottom-right
        // alignment reads every valid key, a shorter run or an explicit
        // position_offset cuts it shorter, and the sliding window raises the
        // start. The sinks are the run's first positions, always visible; the
        // window band starts past them so no key is counted twice. A slice is
        // a sub-range of that band.
        const int64_t run_b = run_length(seqlens, b, Sk, seqlens_plus_one);
        const int64_t pos0 = batch_pos0(cfg, inputs, b);
        const int64_t n_valid = causal_limit(cfg, pos0, 0, Sq, run_b);
        const int64_t prefix = sink_prefix(cfg, run_b);
        const int64_t band_lo =
            std::max<int64_t>(std::min(key_lo(cfg, pos0, 0, Sq, run_b), n_valid), prefix);
        const int64_t span = n_valid - band_lo;
        const int64_t lo = band_lo + (span * si) / splits;
        const int64_t hi = band_lo + (span * (si + 1)) / splits;

        const KV* k_head = k_ptr + b * k_b + hkv * k_h;
        const KV* v_head = v_ptr + b * v_b + hkv * v_h;
        const T* q_base = q_ptr + b * q_b + h0 * q_h;
        T* o_base = out_ptr + b * o_b + h0 * o_h;
        // Successive query heads of the block: the head stride, whatever the
        // layout — row-head layouts put it inside the row, BHSD in its own axis.
        const int64_t q_step = q_h;
        const int64_t o_step = o_h;

        // [past | new], the past segment empty when no past is passed.
        const KV* k_base[2] = {k_head, k_head};
        const KV* v_base[2] = {v_head, v_head};
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
        float* scores = ws + slot * per_slot;      // [nh][kDecodeChunk]
        float* o_acc = scores + hb * kDecodeChunk; // [nh][Dv]
        float* mrow = o_acc + hb * Dv;             // [nh]
        float* lrow = mrow + hb;                   // [nh]

        // The cache's per-token scales, this (batch, kv head)'s rows.
        const KvScaleRow ks_row = kscale.row(b, hkv, Hkv, Sk);
        const KvScaleRow vs_row = vscale.row(b, hkv, Hkv, Sk);

        for (int64_t j = 0; j < nh; ++j) {
            mrow[j] = kNegInf;
            lrow[j] = 0.0f;
            float* oa = o_acc + j * Dv;
            for (int64_t d = 0; d < Dv; ++d) {
                oa[d] = 0.0f;
            }
        }

        // Walk a range in kDecodeChunk-key blocks, keeping the running max,
        // sum and unnormalized output per head: the score scratch is a fixed
        // block, not the whole run, so it stays flat as the cache grows.
        const auto process_range = [&](int64_t rlo, int64_t rhi) {
            for (int64_t t0 = rlo; t0 < rhi; t0 += kDecodeChunk) {
                const int64_t t1 = std::min(rhi, t0 + kDecodeChunk);
                const int64_t nc = t1 - t0;

                // ---- scores = q · K^T --------------------------------
                // The key block is streamed once and every head of the group
                // dots against it: what was one full pass over the run per
                // query head is now one per group. Without a softcap the scale
                // rides through the softmax exponential, so this stays a pure
                // dot product.
                // Two heads per pass: each K vector feeds both heads' FMAs, so
                // the K-side load slots (and the f16/int8 widen) are halved.
                for (int64_t j = 0; j + 2 <= nh; j += 2) {
                    const T* q0 = q_base + j * q_step;
                    const T* q1 = q0 + q_step;
                    float* s0 = scores + j * kDecodeChunk;
                    float* s1 = s0 + kDecodeChunk;
                    for (int s = 0; s < 2; ++s) {
                        const int64_t a = std::max(t0, seg_lo[s]);
                        const int64_t b1 = std::min(t1, seg_hi[s]);
                        const KV* kb = k_base[s];
                        const int64_t rs = k_str[s];
                        int64_t t = a;
                        for (; t + 4 <= b1; t += 4) {
                            const KV* kk = kb + (t - seg_lo[s]) * rs;
                            dot_row4x2(q0, q1, kk, kk + rs, kk + 2 * rs, kk + 3 * rs, Dqk,
                                       s0 + (t - t0), s1 + (t - t0));
                        }
                        for (; t < b1; ++t) {
                            const KV* krow = kb + (t - seg_lo[s]) * rs;
                            s0[t - t0] = dot_row(q0, krow, Dqk);
                            s1[t - t0] = dot_row(q1, krow, Dqk);
                        }
                    }
                    // The cache's per-token K scale multiplies the raw dot; it
                    // has to land before the cap, which owns the scalar scale.
                    if constexpr (kInt8) {
                        for (int64_t t = 0; t < nc; ++t) {
                            const float sc = ks_row.at(t0 + t);
                            s0[t] *= sc;
                            s1[t] *= sc;
                        }
                    }
                }
                for (int64_t j = (nh / 2) * 2; j < nh; ++j) {
                    const T* q_row = q_base + j * q_step;
                    float* sj = scores + j * kDecodeChunk;
                    for (int s = 0; s < 2; ++s) {
                        const int64_t a = std::max(t0, seg_lo[s]);
                        const int64_t b1 = std::min(t1, seg_hi[s]);
                        const KV* kb = k_base[s];
                        const int64_t rs = k_str[s];
                        int64_t t = a;
                        for (; t + 4 <= b1; t += 4) {
                            const KV* kk = kb + (t - seg_lo[s]) * rs;
                            dot_row4(q_row, kk, kk + rs, kk + 2 * rs, kk + 3 * rs, Dqk,
                                     sj + (t - t0));
                        }
                        for (; t < b1; ++t) {
                            const KV* krow = kb + (t - seg_lo[s]) * rs;
                            sj[t - t0] = dot_row(q_row, krow, Dqk);
                        }
                    }
                    if constexpr (kInt8) {
                        for (int64_t t = 0; t < nc; ++t) {
                            sj[t] *= ks_row.at(t0 + t);
                        }
                    }
                }

                // ---- softcap ------------------------------------------
                // The cap owns the scale, so it runs on the raw dots before
                // the online-softmax update reads the row.
                if (cfg.softcap > 0.0f) {
                    for (int64_t j = 0; j < nh; ++j) {
                        row_softcap(scores + j * kDecodeChunk, nc, scale, cfg.softcap);
                    }
                }

                // ---- online-softmax update ----------------------------
                for (int64_t j = 0; j < nh; ++j) {
                    float* sj = scores + j * kDecodeChunk;
                    const float rowmax = kernel::softmax_row_max<float>(sj, nc);
                    if (rowmax == kNegInf) {
                        continue;
                    }
                    const float old_m = mrow[j];
                    const float new_m = std::max(old_m, rowmax);
                    // The running max is a raw dot, so the rescale carries the
                    // same exponential scale the weights do. A first block
                    // starts from zero, so it is skipped rather than evaluated
                    // as -inf·0.
                    const float exp_diff =
                        (old_m == kNegInf) ? 0.0f : std::exp((old_m - new_m) * exp_scale);
                    mrow[j] = new_m;
                    lrow[j] = exp_diff * lrow[j] +
                              kernel::softmax_row_exp_sum<float>(sj, sj, nc, new_m, exp_scale);
                    if (exp_diff != 1.0f) {
                        float* oa = o_acc + j * Dv;
                        auto v_ed = simd::v_set1(oa, exp_diff);
                        int64_t d = 0;
                        for (; d + lanes <= Dv; d += lanes) {
                            simd::v_store(oa + d, simd::v_mul(simd::v_load(oa + d), v_ed));
                        }
                        for (; d < Dv; ++d) {
                            oa[d] *= exp_diff;
                        }
                    }
                }

                // The cache's per-token V scale rides the probabilities, not
                // the denominator: out = Σ p·vs·v / Σ p.
                if constexpr (kInt8) {
                    for (int64_t j = 0; j < nh; ++j) {
                        float* sj = scores + j * kDecodeChunk;
                        for (int64_t t = 0; t < nc; ++t) {
                            sj[t] *= vs_row.at(t0 + t);
                        }
                    }
                }

                // ---- o += P · V --------------------------------------
                // The probabilities are the unnormalized block exponentials;
                // the running sum normalizes once at the end. Four value rows
                // per accumulator pass, as in the unbatched kernel.
                for (int s = 0; s < 2; ++s) {
                    const int64_t a = std::max(t0, seg_lo[s]);
                    const int64_t b1 = std::min(t1, seg_hi[s]);
                    const KV* vb = v_base[s];
                    const int64_t rs = v_str[s];
                    int64_t t = a;
                    for (; t + 4 <= b1; t += 4) {
                        const KV* r0 = vb + (t - seg_lo[s]) * rs;
                        const KV* r1 = r0 + rs;
                        const KV* r2 = r1 + rs;
                        const KV* r3 = r2 + rs;
                        const int64_t tt = t - t0;
                        for (int64_t j = 0; j < nh; ++j) {
                            const float* sj = scores + j * kDecodeChunk;
                            const auto vp0 = simd::v_set1_f32x8(sj[tt + 0]);
                            const auto vp1 = simd::v_set1_f32x8(sj[tt + 1]);
                            const auto vp2 = simd::v_set1_f32x8(sj[tt + 2]);
                            const auto vp3 = simd::v_set1_f32x8(sj[tt + 3]);
                            float* oa = o_acc + j * Dv;
                            int64_t d = 0;
                            for (; d + lanes <= Dv; d += lanes) {
                                auto acc = simd::v_load(oa + d);
                                acc = simd::v_fmadd(vp0, kv_load_f32<KV>(r0 + d), acc);
                                acc = simd::v_fmadd(vp1, kv_load_f32<KV>(r1 + d), acc);
                                acc = simd::v_fmadd(vp2, kv_load_f32<KV>(r2 + d), acc);
                                acc = simd::v_fmadd(vp3, kv_load_f32<KV>(r3 + d), acc);
                                simd::v_store(oa + d, acc);
                            }
                            for (; d < Dv; ++d) {
                                oa[d] += sj[tt + 0] * kv_load_scalar<KV>(&r0[d]) +
                                         sj[tt + 1] * kv_load_scalar<KV>(&r1[d]) +
                                         sj[tt + 2] * kv_load_scalar<KV>(&r2[d]) +
                                         sj[tt + 3] * kv_load_scalar<KV>(&r3[d]);
                            }
                        }
                    }
                    for (; t < b1; ++t) {
                        const KV* vrow = vb + (t - seg_lo[s]) * rs;
                        const int64_t tt = t - t0;
                        for (int64_t j = 0; j < nh; ++j) {
                            const float p = scores[j * kDecodeChunk + tt];
                            auto vp = simd::v_set1_f32x8(p);
                            float* oa = o_acc + j * Dv;
                            int64_t d = 0;
                            for (; d + lanes <= Dv; d += lanes) {
                                simd::v_store(oa + d,
                                              simd::v_fmadd(vp, kv_load_f32<KV>(vrow + d),
                                                            simd::v_load(oa + d)));
                            }
                            for (; d < Dv; ++d) {
                                oa[d] += p * kv_load_scalar<KV>(&vrow[d]);
                            }
                        }
                    }
                }
            }
        };

        // The sink band is slice 0's; the window band is this slice's.
        if (si == 0 && prefix > 0) {
            process_range(0, prefix);
        }
        process_range(lo, hi);

        // ---- store ---------------------------------------------------
        if (splits == 1) {
            for (int64_t j = 0; j < nh; ++j) {
                const float l_j = lrow[j];
                const float m_j = mrow[j];
                // The sink is one extra logit in the denominator (stored scores
                // are raw, so the logit max is m * exp_scale). With no live key
                // at all the row stays zeros: all the weight is on the sink.
                float inv_l = 0.0f;
                if (m_j != kNegInf) {
                    const float sink = (sink_ptr != nullptr) ? simd::s_load(&sink_ptr[h0 + j])
                                                             : kNegInf;
                    const float denom = l_j + std::exp(sink - m_j * exp_scale);
                    inv_l = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
                }
                const float* oa = o_acc + j * Dv;
                T* o_row = o_base + j * o_step;
                auto v_inv = simd::v_set1_f32x8(inv_l);
                int64_t d = 0;
                for (; d + lanes <= Dv; d += lanes) {
                    simd::v_store_f32(o_row + d, simd::v_mul(simd::v_load(oa + d), v_inv));
                }
                for (; d < Dv; ++d) {
                    simd::s_store(&o_row[d], oa[d] * inv_l);
                }
            }
        } else {
            // Unnormalized output, max and sum; the merge pass combines them.
            for (int64_t j = 0; j < nh; ++j) {
                float* pj = partials + idx * hb * (Dv + 2) + j * (Dv + 2);
                const float* oa = o_acc + j * Dv;
                for (int64_t d = 0; d < Dv; ++d) {
                    pj[d] = oa[d];
                }
                pj[Dv] = mrow[j];
                pj[Dv + 1] = lrow[j];
            }
        }
    };

    ctx.cpu.run(0, NG, run);

    if (splits == 1) {
        return;
    }

    // ---- merge the KV slices -----------------------------------------------
    // Each slice left o = Σ e·v, m = its maximum raw score and l = Σ e at that
    // max. Under the global maximum M a slice is weighted by exp((m - M)·scale),
    // so the online-softmax identity holds across slices exactly as it does
    // across blocks within one.
    const int64_t nmerge = B * H;
    const auto merge = [&](int64_t idx) {
        const int64_t b = idx / H;
        const int64_t h = idx % H;
        const int64_t hkv = h / G;
        const int64_t g = h - hkv * G;
        const int64_t c = g / hb;
        const int64_t j = g - c * hb;
        const int64_t u = (b * Hkv + hkv) * nb + c;
        const int64_t pstep = hb * (Dv + 2);
        const float* pj = partials + (u * splits) * pstep + j * (Dv + 2);

        T* o_row = out_ptr + b * o_b + h * o_h;

        float M = kNegInf;
        for (int64_t s = 0; s < splits; ++s) {
            M = std::max(M, pj[s * pstep + Dv]);
        }
        if (M == kNegInf) {
            for (int64_t d = 0; d < Dv; ++d) {
                simd::s_store(&o_row[d], 0.0f);
            }
            return;
        }

        // The merge dispatch runs after the slice dispatch, so no slot is
        // live; reuse one as the f32 accumulator.
        const int64_t slot = scratch_slot_of(ctx.cpu, nmerge, idx);
        float* acc = ws + slot * per_slot + hb * kDecodeChunk;
        for (int64_t d = 0; d < Dv; ++d) {
            acc[d] = 0.0f;
        }
        float denom = 0.0f;
        for (int64_t s = 0; s < splits; ++s) {
            const float* p = pj + s * pstep;
            const float ms = p[Dv];
            if (ms == kNegInf) {
                continue;
            }
            const float w = std::exp((ms - M) * exp_scale);
            denom += w * p[Dv + 1];
            auto vw = simd::v_set1(p, w);
            int64_t d = 0;
            for (; d + lanes <= Dv; d += lanes) {
                simd::v_store(acc + d,
                              simd::v_fmadd(vw, simd::v_load(p + d), simd::v_load(acc + d)));
            }
            for (; d < Dv; ++d) {
                acc[d] += w * p[d];
            }
        }
        // The sink joins the merged denominator once (its value is zero, so
        // the numerator is untouched).
        if (sink_ptr != nullptr) {
            const float sink = simd::s_load(&sink_ptr[h]);
            denom += std::exp(sink - M * exp_scale);
        }
        const float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
        auto v_inv = simd::v_set1_f32x8(inv);
        int64_t d = 0;
        for (; d + lanes <= Dv; d += lanes) {
            simd::v_store_f32(o_row + d, simd::v_mul(simd::v_load(acc + d), v_inv));
        }
        for (; d < Dv; ++d) {
            simd::s_store(&o_row[d], acc[d] * inv);
        }
    };
    ctx.cpu.run(0, nmerge, merge);
}

} // anonymous namespace

void attention_decode_cpu(const AttentionConfig& cfg, TensorView& output,
                          std::span<const TensorView> inputs, const ComputeContext& ctx,
                          const AttentionPlan& plan, void* scratch)
{
    NNOPS_ASSERT(inputs[0].data_type() == DataType::f16 || inputs[0].data_type() == DataType::f32 ||
                 inputs[0].data_type() == DataType::bf16);
    const bool kv8 = kv_is_int8(inputs);
    if (inputs[0].data_type() == DataType::f16) {
        if (kv8) {
            attention_decode_impl<half, int8_t>(cfg, output, inputs, ctx, plan, scratch);
        } else {
            attention_decode_impl<half, half>(cfg, output, inputs, ctx, plan, scratch);
        }
#ifdef NNOPS_ARCH_AARCH64
    } else if (inputs[0].data_type() == DataType::bf16) {
        NNOPS_ASSERT(!kv8); // bf16 queries have no int8 cache route
        attention_decode_impl<bfloat16, bfloat16>(cfg, output, inputs, ctx, plan, scratch);
#endif
    } else {
        if (kv8) {
            attention_decode_impl<float, int8_t>(cfg, output, inputs, ctx, plan, scratch);
        } else {
            attention_decode_impl<float, float>(cfg, output, inputs, ctx, plan, scratch);
        }
    }
}

} // namespace nnops::backend::cpu
