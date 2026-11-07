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

/// Dot product of two n-wide rows, f32 accumulation.
template <class T>
inline float dot_row(const T* a, const T* b, int64_t n)
{
    constexpr int L = simd::simd_lane_for<T>;
    auto acc = simd::v_zero_f32x8();
    int64_t i = 0;
    for (; i + L <= n; i += L) {
        acc = simd::v_fmadd(simd::v_load_f32(a + i), simd::v_load_f32(b + i), acc);
    }
    float s = simd::v_reduce_sum(acc);
    for (; i < n; ++i) { s += simd::s_load(&a[i]) * simd::s_load(&b[i]); }
    return s;
}

/// Four n-wide dot products against one query row, written to @p out. A
/// horizontal reduce costs about as much as the 16 vector FMAs of a 128-wide
/// row, so four keys per reduce is what keeps the GEMV off the latency floor.
template <class T>
inline void dot_row4(const T* q, const T* k0, const T* k1, const T* k2, const T* k3,
                     int64_t n, float* out)
{
    constexpr int L = simd::simd_lane_for<T>;
    auto a0 = simd::v_zero_f32x8();
    auto a1 = a0, a2 = a0, a3 = a0;
    int64_t i = 0;
    for (; i + L <= n; i += L) {
        auto qv = simd::v_load_f32(q + i);
        a0 = simd::v_fmadd(qv, simd::v_load_f32(k0 + i), a0);
        a1 = simd::v_fmadd(qv, simd::v_load_f32(k1 + i), a1);
        a2 = simd::v_fmadd(qv, simd::v_load_f32(k2 + i), a2);
        a3 = simd::v_fmadd(qv, simd::v_load_f32(k3 + i), a3);
    }
    float s0 = simd::v_reduce_sum(a0);
    float s1 = simd::v_reduce_sum(a1);
    float s2 = simd::v_reduce_sum(a2);
    float s3 = simd::v_reduce_sum(a3);
    for (; i < n; ++i) {
        const float qs = simd::s_load(&q[i]);
        s0 += qs * simd::s_load(&k0[i]);
        s1 += qs * simd::s_load(&k1[i]);
        s2 += qs * simd::s_load(&k2[i]);
        s3 += qs * simd::s_load(&k3[i]);
    }
    out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
}

template <class T>
void attention_decode_impl(const AttentionConfig& cfg,
                           TensorView& output,
                           std::span<const TensorView> inputs,
                           const ComputeContext& ctx,
                           const AttentionPlan& plan,
                           void* workspace)
{
    const auto& Q = inputs[0];
    const auto& K = inputs[1];
    const auto& V = cfg.v_is_k ? inputs[1] : inputs[2];
    const bool split = has_past_view(cfg, inputs);
    const auto& Kp = split ? inputs[past_slot(cfg)] : K;
    const auto& Vp = split ? inputs[past_slot(cfg) + (cfg.v_is_k ? 0 : 1)] : V;

    const HeadShape hs = resolve_head_shape(cfg, inputs);
    const int64_t B = hs.B, H = hs.H, Hkv = hs.Hkv;
    const int64_t Sq = hs.Sq, Dqk = hs.Dqk, Dv = hs.Dv;
    const int64_t Sk_new = hs.Sk, Sk_past = hs.Sk_past;
    const int64_t Sk = Sk_new + Sk_past;   // the whole [past | new] run
    const bool merged = hs.merged;
    NNOPS_ASSERT(Sq == 1);

    const float scale = (cfg.scale == 0.0f)
        ? (1.0f / std::sqrt(static_cast<float>(Dqk)))
        : cfg.scale;

    const T* q_ptr = Q.ptr<T>();
    const T* k_ptr = K.ptr<T>();
    const T* v_ptr = V.ptr<T>();
    const T* kp_ptr = split ? Kp.ptr<T>() : nullptr;
    const T* vp_ptr = split ? Vp.ptr<T>() : nullptr;
    T* out_ptr = output.ptr<T>();

    const int64_t q_rs = Q.row_stride_elems();
    const int64_t k_rs = K.row_stride_elems();
    const int64_t v_rs = V.row_stride_elems();
    const int64_t kp_rs = split ? Kp.row_stride_elems() : k_rs;
    const int64_t vp_rs = split ? Vp.row_stride_elems() : v_rs;
    const int64_t o_rs = output.row_stride_elems();
    const int64_t v_head_stride = merged_v_head_stride(cfg, Dqk, Dv);

    const int64_t G = H / Hkv;
    const int64_t hb = plan.decode_heads;
    const int64_t splits = plan.decode_splits;
    const int64_t nb = (G + hb - 1) / hb;   // head blocks per KV head
    const int64_t units = B * Hkv * nb;
    const int64_t NG = units * splits;

    const int64_t per_slot = plan.decode_elems;   // f32: [scores][o_acc][max][sum]
    float* ws = static_cast<float*>(workspace);
    float* partials = ws + plan.num_slots * per_slot;

    // Bottom-right alignment makes the limit Sk — the row attends to every
    // cached key; an explicit position_offset can cut it shorter.
    const int64_t n_valid = causal_limit(cfg, 0, Sq, Sk);

    constexpr int64_t lanes = simd::simd_default_lane_f32;

    const auto run = [&](int64_t idx) {
        // Grid (batch, kv head, head block) × KV slice. A slice is a
        // contiguous span of the run, empty when there are more slices than
        // keys; its head block holds up to hb query heads of one KV head.
        const int64_t si   = idx % splits;
        const int64_t unit = idx / splits;
        const int64_t c    = unit % nb;
        const int64_t hkv  = (unit / nb) % Hkv;
        const int64_t b    = (unit / nb) / Hkv;
        const int64_t h0   = hkv * G + c * hb;
        const int64_t nh   = std::min(hb, G - c * hb);

        const int64_t lo = (n_valid * si) / splits;
        const int64_t hi = (n_valid * (si + 1)) / splits;

        const T* k_head;
        const T* v_head;
        const T* q_base;
        T* o_base;
        int64_t q_step, o_step;
        if (merged) {
            q_base = q_ptr + b * q_rs + h0 * Dqk;
            q_step = Dqk;
            k_head = k_ptr + b * Sk_new * k_rs + hkv * Dqk;
            v_head = v_ptr + b * hs.Sv * v_rs + hkv * v_head_stride;
            o_base = out_ptr + b * o_rs + h0 * Dv;
            o_step = Dv;
        } else {
            q_base = q_ptr + (b * H + h0) * q_rs;
            q_step = q_rs;
            k_head = k_ptr + (b * Hkv + hkv) * Sk_new * k_rs;
            v_head = v_ptr + (b * Hkv + hkv) * hs.Sv * v_rs;
            o_base = out_ptr + (b * H + h0) * o_rs;
            o_step = o_rs;
        }

        // [past | new], the past segment empty when no past is passed.
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
        float* scores = ws + slot * per_slot;        // [nh][kDecodeChunk]
        float* o_acc  = scores + hb * kDecodeChunk;  // [nh][Dv]
        float* mrow   = o_acc + hb * Dv;             // [nh]
        float* lrow   = mrow + hb;                   // [nh]

        for (int64_t j = 0; j < nh; ++j) {
            mrow[j] = kNegInf;
            lrow[j] = 0.0f;
            float* oa = o_acc + j * Dv;
            for (int64_t d = 0; d < Dv; ++d) { oa[d] = 0.0f; }
        }

        // Walk the slice in kDecodeChunk-key blocks, keeping the running max,
        // sum and unnormalized output per head: the score scratch is a fixed
        // block, not the whole run, so it stays flat as the cache grows.
        for (int64_t t0 = lo; t0 < hi; t0 += kDecodeChunk) {
            const int64_t t1 = std::min(hi, t0 + kDecodeChunk);
            const int64_t nc = t1 - t0;

            // ---- scores = q · K^T ------------------------------------
            // The key block is streamed once and every head of the group dots
            // against it: what was one full pass over the run per query head
            // is now one per group. The scale rides through the softmax
            // exponential, so this stays a pure dot product.
            for (int64_t j = 0; j < nh; ++j) {
                const T* q_row = q_base + j * q_step;
                float* sj = scores + j * kDecodeChunk;
                for (int s = 0; s < 2; ++s) {
                    const int64_t a  = std::max(t0, seg_lo[s]);
                    const int64_t b1 = std::min(t1, seg_hi[s]);
                    const T* kb = k_base[s];
                    const int64_t rs = k_str[s];
                    int64_t t = a;
                    for (; t + 4 <= b1; t += 4) {
                        const T* kk = kb + (t - seg_lo[s]) * rs;
                        dot_row4(q_row, kk, kk + rs, kk + 2 * rs, kk + 3 * rs,
                                 Dqk, sj + (t - t0));
                    }
                    for (; t < b1; ++t) {
                        sj[t - t0] = dot_row(q_row, kb + (t - seg_lo[s]) * rs, Dqk);
                    }
                }
            }

            // ---- online-softmax update --------------------------------
            for (int64_t j = 0; j < nh; ++j) {
                float* sj = scores + j * kDecodeChunk;
                const float rowmax = kernel::softmax_row_max<float>(sj, nc);
                if (rowmax == kNegInf) { continue; }
                const float old_m = mrow[j];
                const float new_m = std::max(old_m, rowmax);
                // A first block starts from zero, so the rescale is skipped
                // rather than evaluated as -inf·0.
                const float exp_diff = (old_m == kNegInf) ? 0.0f : std::exp(old_m - new_m);
                mrow[j] = new_m;
                lrow[j] = exp_diff * lrow[j]
                        + kernel::softmax_row_exp_sum<float>(sj, sj, nc, new_m, scale);
                if (exp_diff != 1.0f) {
                    float* oa = o_acc + j * Dv;
                    auto v_ed = simd::v_set1(oa, exp_diff);
                    int64_t d = 0;
                    for (; d + lanes <= Dv; d += lanes) {
                        simd::v_store(oa + d, simd::v_mul(simd::v_load(oa + d), v_ed));
                    }
                    for (; d < Dv; ++d) { oa[d] *= exp_diff; }
                }
            }

            // ---- o += P · V ------------------------------------------
            // The probabilities are the unnormalized block exponentials; the
            // running sum normalizes once at the end. Four value rows per
            // accumulator pass, as in the unbatched kernel.
            for (int s = 0; s < 2; ++s) {
                const int64_t a  = std::max(t0, seg_lo[s]);
                const int64_t b1 = std::min(t1, seg_hi[s]);
                const T* vb = v_base[s];
                const int64_t rs = v_str[s];
                int64_t t = a;
                for (; t + 4 <= b1; t += 4) {
                    const T* r0 = vb + (t - seg_lo[s]) * rs;
                    const T* r1 = r0 + rs;
                    const T* r2 = r1 + rs;
                    const T* r3 = r2 + rs;
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
                            acc = simd::v_fmadd(vp0, simd::v_load_f32(r0 + d), acc);
                            acc = simd::v_fmadd(vp1, simd::v_load_f32(r1 + d), acc);
                            acc = simd::v_fmadd(vp2, simd::v_load_f32(r2 + d), acc);
                            acc = simd::v_fmadd(vp3, simd::v_load_f32(r3 + d), acc);
                            simd::v_store(oa + d, acc);
                        }
                        for (; d < Dv; ++d) {
                            oa[d] += sj[tt + 0] * simd::s_load(&r0[d])
                                   + sj[tt + 1] * simd::s_load(&r1[d])
                                   + sj[tt + 2] * simd::s_load(&r2[d])
                                   + sj[tt + 3] * simd::s_load(&r3[d]);
                        }
                    }
                }
                for (; t < b1; ++t) {
                    const T* vrow = vb + (t - seg_lo[s]) * rs;
                    const int64_t tt = t - t0;
                    for (int64_t j = 0; j < nh; ++j) {
                        const float p = scores[j * kDecodeChunk + tt];
                        auto vp = simd::v_set1_f32x8(p);
                        float* oa = o_acc + j * Dv;
                        int64_t d = 0;
                        for (; d + lanes <= Dv; d += lanes) {
                            simd::v_store(oa + d,
                                simd::v_fmadd(vp, simd::v_load_f32(vrow + d), simd::v_load(oa + d)));
                        }
                        for (; d < Dv; ++d) { oa[d] += p * simd::s_load(&vrow[d]); }
                    }
                }
            }
        }

        // ---- store ---------------------------------------------------
        if (splits == 1) {
            for (int64_t j = 0; j < nh; ++j) {
                const float l_j = lrow[j];
                // A row every key was masked out of leaves l == 0: zeros.
                const float inv_l = (l_j > 0.0f) ? (1.0f / l_j) : 0.0f;
                const float* oa = o_acc + j * Dv;
                T* o_row = o_base + j * o_step;
                auto v_inv = simd::v_set1(oa, inv_l);
                int64_t d = 0;
                for (; d + lanes <= Dv; d += lanes) {
                    simd::v_store_f32(o_row + d, simd::v_mul(simd::v_load(oa + d), v_inv));
                }
                for (; d < Dv; ++d) { simd::s_store(&o_row[d], oa[d] * inv_l); }
            }
        } else {
            // Unnormalized output, max and sum; the merge pass combines them.
            for (int64_t j = 0; j < nh; ++j) {
                float* pj = partials + idx * hb * (Dv + 2) + j * (Dv + 2);
                const float* oa = o_acc + j * Dv;
                for (int64_t d = 0; d < Dv; ++d) { pj[d] = oa[d]; }
                pj[Dv] = mrow[j];
                pj[Dv + 1] = lrow[j];
            }
        }
    };

    ctx.cpu.run(0, NG, run);

    if (splits == 1) { return; }

    // ---- merge the KV slices -----------------------------------------------
    // Each slice left o = Σ e·v, m = its maximum raw score and l = Σ e at that
    // max. Under the global maximum M a slice is weighted by exp((m - M)·scale),
    // so the online-softmax identity holds across slices exactly as it does
    // across blocks within one.
    const int64_t nmerge = B * H;
    const auto merge = [&](int64_t idx) {
        const int64_t b   = idx / H;
        const int64_t h   = idx % H;
        const int64_t hkv = h / G;
        const int64_t g   = h - hkv * G;
        const int64_t c   = g / hb;
        const int64_t j   = g - c * hb;
        const int64_t u   = (b * Hkv + hkv) * nb + c;
        const int64_t pstep = hb * (Dv + 2);
        const float* pj = partials + (u * splits) * pstep + j * (Dv + 2);

        T* o_row = merged ? out_ptr + b * o_rs + h * Dv
                          : out_ptr + (b * H + h) * o_rs;

        float M = kNegInf;
        for (int64_t s = 0; s < splits; ++s) {
            M = std::max(M, pj[s * pstep + Dv]);
        }
        if (M == kNegInf) {
            for (int64_t d = 0; d < Dv; ++d) { simd::s_store(&o_row[d], 0.0f); }
            return;
        }

        // The merge dispatch runs after the slice dispatch, so no slot is
        // live; reuse one as the f32 accumulator.
        const int64_t slot = scratch_slot_of(ctx.cpu, nmerge, idx);
        float* acc = ws + slot * per_slot + hb * kDecodeChunk;
        for (int64_t d = 0; d < Dv; ++d) { acc[d] = 0.0f; }
        float denom = 0.0f;
        for (int64_t s = 0; s < splits; ++s) {
            const float* p = pj + s * pstep;
            const float ms = p[Dv];
            if (ms == kNegInf) { continue; }
            const float w = std::exp((ms - M) * scale);
            denom += w * p[Dv + 1];
            auto vw = simd::v_set1(p, w);
            int64_t d = 0;
            for (; d + lanes <= Dv; d += lanes) {
                simd::v_store(acc + d,
                    simd::v_fmadd(vw, simd::v_load(p + d), simd::v_load(acc + d)));
            }
            for (; d < Dv; ++d) { acc[d] += w * p[d]; }
        }
        const float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
        auto v_inv = simd::v_set1(acc, inv);
        int64_t d = 0;
        for (; d + lanes <= Dv; d += lanes) {
            simd::v_store_f32(o_row + d, simd::v_mul(simd::v_load(acc + d), v_inv));
        }
        for (; d < Dv; ++d) { simd::s_store(&o_row[d], acc[d] * inv); }
    };
    ctx.cpu.run(0, nmerge, merge);
}

}  // anonymous namespace

void attention_decode_cpu(const AttentionConfig& cfg,
                          TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx,
                          const AttentionPlan& plan,
                          void* scratch)
{
    NNOPS_ASSERT(inputs[0].data_type() == DataType::f16 ||
                 inputs[0].data_type() == DataType::f32);
    if (inputs[0].data_type() == DataType::f16) {
        attention_decode_impl<half>(cfg, output, inputs, ctx, plan, scratch);
    } else {
        attention_decode_impl<float>(cfg, output, inputs, ctx, plan, scratch);
    }
}

}  // namespace nnops::backend::cpu
