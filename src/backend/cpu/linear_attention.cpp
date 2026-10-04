/// @file linear_attention.cpp
/// @brief CPU LinearAttention kernel — gated delta recurrence, SIMD-accelerated.
///
/// Per (batch, kv head), a state matrix S[D, D] is carried over the sequence and
/// each token runs one fused pass (see reference/linear_attention_ref.cpp for
/// the scalar statement of the same math):
///
///   S[i][:] *= exp(Gate[t, i])                 1. decay the state
///   r[j]     = sum_i S[i][j] * k[i]            2. retrieval, on the decayed S
///   d[j]     = beta_t * (v[j] - r[j])          3. delta
///   S[i][j] += k[i] * d[j]                     4. rank-1 update
///   o[j]     = scale * sum_i q[i] * S[i][j]    5. readout, per query head
///
/// Scheduling: one task per (batch, kv head). The token loop is inherently
/// sequential — step t+1 needs the state step t produced — so it stays serial
/// inside a task, and the parallelism comes from the B * H_kv independent
/// state matrices. Under GQA the G query heads sharing a kv head are read out
/// *within* the same task: splitting them across tasks would advance one state
/// G times.
///
/// SIMD: every stage is vectorized across the value dimension (j), with the key
/// dimension (i) broadcast as a scalar. That is the shape ORT's MLAS NEON kernel
/// uses (FusedTokenSinglePassNeon). The state, the delta, and the readout
/// accumulators are all f32 even for f16 input — f16 only appears at the I/O
/// boundary, where rows are widened on load and narrowed on store.
///
/// The convolution, when weights are supplied, is a causal depthwise conv1d over
/// the packed Q | K | V channels. Those weights are transposed once per task into
/// a [local channel][tap][d] layout, which makes each tap a contiguous per-lane
/// weight vector — a plain vector FMA with no gather.

#include "nnops/ops/linear_attention.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/simd/simd.hpp"

#include "common/memory_pool.hpp"

#include <cmath>
#include <span>

// Forward-declare the reference kernel for the unsupported-dtype fallback.
namespace nnops::backend::cpu::reference {
extern void linear_attention_ref(const LinearAttentionAttributes& attrs,
                                 std::span<TensorView> outputs,
                                 std::span<const TensorView> inputs,
                                 const ComputeContext& ctx,
                                 void* workspace);
}

namespace nnops::backend::cpu {
namespace {

using nnops::backend::cpu::half;
using namespace nnops::simd;

/// Widen L consecutive T elements to an f32 vector (f16 widens, f32 is a load).
inline v_f32x8 vload_f32(const float* p) { return v_load(p); }
inline v_f32x8 vload_f32(const half* p)  { return v_cvt_f16_to_f32(v_load(p)); }

/// Narrow an f32 vector back to T and store it.
inline void vstore_f32(float* p, const v_f32x8& v) { v_store(p, v); }
inline void vstore_f32(half* p, const v_f32x8& v)  { v_store(p, v_cvt_f32_to_f16(v)); }

template <typename T>
void linear_attention_impl(const LinearAttentionAttributes& attrs,
                           std::span<TensorView> outputs,
                           std::span<const TensorView> inputs,
                           const ComputeContext& ctx)
{
    constexpr int L = simd_lane_for<T>;

    const TensorView& Q = inputs[0];
    const TensorView& K = inputs[1];
    const TensorView& V = inputs[2];
    const TensorView& G = inputs[3];
    TensorView& O  = outputs[0];
    TensorView& St = outputs[1];

    const int64_t B    = Q.shape(0);
    const int64_t H    = Q.shape(1);
    const int64_t S    = Q.shape(2);
    const int64_t D    = Q.shape(3);
    const int64_t H_kv = attrs.num_kv_heads > 0 ? attrs.num_kv_heads : H;
    const int64_t Gq   = H / H_kv;   // query heads sharing one kv head

    NNOPS_ASSERT(H_kv > 0 && Gq * H_kv == H);

    const bool has_conv = inputs.size() > 4 && !inputs[4].is_empty();
    const bool has_beta = inputs.size() > 5 && !inputs[5].is_empty();
    const int64_t CK    = has_conv ? inputs[4].shape(2) : 0;

    const float scale = attrs.scale != 0.0f
        ? attrs.scale
        : 1.0f / std::sqrt(static_cast<float>(D));

    // Element strides (planar; the trailing dim is contiguous).
    const int64_t qs0 = Q.stride_elems(0), qs1 = Q.stride_elems(1), qs2 = Q.stride_elems(2);
    const int64_t ks0 = K.stride_elems(0), ks1 = K.stride_elems(1), ks2 = K.stride_elems(2);
    const int64_t vs0 = V.stride_elems(0), vs1 = V.stride_elems(1), vs2 = V.stride_elems(2);
    const int64_t gs0 = G.stride_elems(0), gs1 = G.stride_elems(1), gs2 = G.stride_elems(2);
    const int64_t os0 = O.stride_elems(0), os1 = O.stride_elems(1), os2 = O.stride_elems(2);
    const int64_t ts0 = St.stride_elems(0), ts1 = St.stride_elems(1),
                  ts2 = St.stride_elems(2), ts3 = St.stride_elems(3);
    const int64_t cws0 = has_conv ? inputs[4].stride_elems(0) : 0;
    const int64_t cws2 = has_conv ? inputs[4].stride_elems(2) : 0;
    const int64_t bs2  = has_beta ? inputs[5].stride_elems(2) : 0;
    const int64_t bs0  = has_beta ? inputs[5].stride_elems(0) : 0;
    const int64_t bs1  = has_beta ? inputs[5].stride_elems(1) : 0;

    const T* qp = Q.ptr<T>();
    const T* kp = K.ptr<T>();
    const T* vp = V.ptr<T>();
    const T* gp = G.ptr<T>();
    T*       op = O.ptr<T>();
    T*       sp = St.ptr<T>();

    // Convolution channel bases: the packed weights lay Q out first, then K, then V.
    const int64_t k_ch_base = H * D;
    const int64_t v_ch_base = H * D + H_kv * D;
    const int64_t n_ch = Gq + 2;   // q heads 0..Gq-1, then K at Gq, V at Gq+1

    const int64_t NTasks = B * H_kv;
    ctx.cpu.run(0, NTasks, [&](int64_t task) {
        const int64_t b   = task / H_kv;
        const int64_t hkv = task % H_kv;
        const int64_t hq0 = hkv * Gq;

        // Per-task scratch, all f32: the state matrix, then kc / vc / qc / dv /
        // gv row buffers. Indexed per task (not per worker) — a task owns its
        // state exclusively.
        PoolPtr state_owner(sizeof(float) * static_cast<size_t>(D) * static_cast<size_t>(D));
        PoolPtr rows_owner(sizeof(float) * static_cast<size_t>(D) * 5);
        float* Sf = state_owner.as<float>();
        float* kc = rows_owner.as<float>();
        float* vc = kc + D;
        float* qc = vc + D;
        float* dv = qc + D;
        float* gv = dv + D;

        // Convolution weights, transposed to [local channel][tap][d] so each tap
        // is a contiguous, per-lane weight vector.
        PoolPtr wt_owner;
        float* wt = nullptr;
        if (has_conv) {
            wt_owner = PoolPtr(sizeof(float) * static_cast<size_t>(n_ch) *
                               static_cast<size_t>(CK) * static_cast<size_t>(D));
            wt = wt_owner.as<float>();
            const T* cwp = inputs[4].ptr<T>();
            for (int64_t c = 0; c < n_ch; ++c) {
                const int64_t ch_base = (c < Gq) ? (hq0 + c) * D
                                      : (c == Gq) ? k_ch_base + hkv * D
                                                  : v_ch_base + hkv * D;
                for (int64_t k = 0; k < CK; ++k) {
                    float* dst = wt + (c * CK + k) * D;
                    for (int64_t d = 0; d < D; ++d) {
                        dst[d] = s_load(cwp + (ch_base + d) * cws0 + k * cws2);
                    }
                }
            }
        }

        // Fill dst[0..D) with the row of `src` at time t (optionally convoluted).
        // `src` points at the (batch, head) origin of the sequence, `src_ts` is
        // the stride between consecutive tokens, `lc` is this row's local
        // convolution channel.
        auto fetch_row = [&](const T* src, int64_t src_ts, int64_t lc,
                             int64_t t, float* dst) {
            if (!has_conv) {
                const T* row = src + t * src_ts;
                int64_t d = 0;
                for (; d + L <= D; d += L) { vstore_f32(dst + d, vload_f32(row + d)); }
                for (; d < D; ++d) { dst[d] = s_load(row + d); }
                return;
            }
            // y[t] = sum_k w[:, k] * x[t - (K-1) + k], x = 0 for t < 0.
            int64_t d = 0;
            for (; d + L <= D; d += L) {
                v_f32x8 acc = v_zero_f32x8();
                for (int64_t k = 0; k < CK; ++k) {
                    const int64_t tt = t - (CK - 1) + k;
                    if (tt < 0) { continue; }   // causal left zero-pad
                    acc = v_fmadd(vload_f32(src + tt * src_ts + d),
                                  v_load(wt + (lc * CK + k) * D + d),
                                  acc);
                }
                v_store(dst + d, acc);
            }
            for (; d < D; ++d) {
                float acc = 0.0f;
                for (int64_t k = 0; k < CK; ++k) {
                    const int64_t tt = t - (CK - 1) + k;
                    if (tt < 0) { continue; }
                    acc += wt[(lc * CK + k) * D + d] * s_load(src + tt * src_ts + d);
                }
                dst[d] = acc;
            }
        };

        // Read the carried state in (f32), run the sequence, write it back.
        for (int64_t i = 0; i < D; ++i) {
            const T* srow = sp + b * ts0 + hkv * ts1 + i * ts2;
            float* drow = Sf + i * D;
            int64_t j = 0;
            for (; j + L <= D; j += L) { vstore_f32(drow + j, vload_f32(srow + j * ts3)); }
            for (; j < D; ++j) { drow[j] = s_load(srow + j * ts3); }
        }

        const T* k_base = kp + b * ks0 + hkv * ks1;
        const T* v_base = vp + b * vs0 + hkv * vs1;
        const T* g_base = gp + b * gs0 + hkv * gs1;
        const T* b_base = has_beta ? inputs[5].ptr<T>() + b * bs0 + hkv * bs1 : nullptr;

        for (int64_t t = 0; t < S; ++t) {
            fetch_row(k_base, ks2, Gq,     t, kc);
            fetch_row(v_base, vs2, Gq + 1, t, vc);

            // --- 1. Decay: S[i][:] *= exp(gate[t, i]) ---
            // exp is evaluated a vector at a time (v_exp, Cephes-style) rather
            // than per element: the decay touches D rows, so D scalar exps would
            // outweigh the D*D/L vector ops it feeds.
            {
                const T* g_row = g_base + t * gs2;
                int64_t i = 0;
                for (; i + L <= D; i += L) { v_store(gv + i, v_exp(vload_f32(g_row + i))); }
                for (; i < D; ++i) { gv[i] = std::exp(s_load(g_row + i)); }

                for (i = 0; i < D; ++i) {
                    float* row = Sf + i * D;
                    const v_f32x8 vg = v_set1_f32x8(gv[i]);
                    int64_t j = 0;
                    for (; j + L <= D; j += L) {
                        v_store(row + j, v_mul(v_load(row + j), vg));
                    }
                    for (; j < D; ++j) { row[j] *= gv[i]; }
                }
            }

            // --- 2. Retrieval: r[j] = sum_i S[i][j] * k[i] (on the decayed S) ---
            {
                int64_t j = 0;
                for (; j + L <= D; j += L) { v_store(dv + j, v_zero_f32x8()); }
                for (; j < D; ++j) { dv[j] = 0.0f; }

                for (int64_t i = 0; i < D; ++i) {
                    const float ki = kc[i];
                    const float* row = Sf + i * D;
                    const v_f32x8 vk = v_set1_f32x8(ki);
                    j = 0;
                    for (; j + L <= D; j += L) {
                        v_store(dv + j, v_fmadd(v_load(row + j), vk, v_load(dv + j)));
                    }
                    for (; j < D; ++j) { dv[j] += row[j] * ki; }
                }
            }

            // --- 3. Delta: d[j] = beta * (v[j] - r[j]) ---
            {
                const float beta = has_beta ? s_load(b_base + t * bs2) : 1.0f;
                const v_f32x8 vb = v_set1_f32x8(beta);
                int64_t j = 0;
                for (; j + L <= D; j += L) {
                    v_store(dv + j, v_mul(vb, v_sub(v_load(vc + j), v_load(dv + j))));
                }
                for (; j < D; ++j) { dv[j] = beta * (vc[j] - dv[j]); }
            }

            // --- 4. Rank-1 update: S[i][j] += k[i] * d[j] ---
            for (int64_t i = 0; i < D; ++i) {
                const float ki = kc[i];
                float* row = Sf + i * D;
                const v_f32x8 vk = v_set1_f32x8(ki);
                int64_t j = 0;
                for (; j + L <= D; j += L) {
                    v_store(row + j, v_fmadd(vk, v_load(dv + j), v_load(row + j)));
                }
                for (; j < D; ++j) { row[j] += ki * dv[j]; }
            }

            // --- 5. Readout: o[j] = scale * sum_i q[i] * S[i][j], per query head ---
            for (int64_t g = 0; g < Gq; ++g) {
                const int64_t hq = hq0 + g;
                fetch_row(qp + b * qs0 + hq * qs1, qs2, g, t, qc);

                T* o_row = op + b * os0 + hq * os1 + t * os2;
                const v_f32x8 vs = v_set1_f32x8(scale);
                int64_t j = 0;
                for (; j + L <= D; j += L) {
                    v_f32x8 acc = v_zero_f32x8();
                    for (int64_t i = 0; i < D; ++i) {
                        acc = v_fmadd(v_load(Sf + i * D + j), v_set1_f32x8(qc[i]), acc);
                    }
                    vstore_f32(o_row + j, v_mul(acc, vs));
                }
                for (; j < D; ++j) {
                    float acc = 0.0f;
                    for (int64_t i = 0; i < D; ++i) { acc += (Sf + i * D)[j] * qc[i]; }
                    s_store(o_row + j, scale * acc);
                }
            }
        }

        for (int64_t i = 0; i < D; ++i) {
            T* srow = sp + b * ts0 + hkv * ts1 + i * ts2;
            const float* drow = Sf + i * D;
            int64_t j = 0;
            for (; j + L <= D; j += L) { vstore_f32(srow + j * ts3, v_load(drow + j)); }
            for (; j < D; ++j) { s_store(srow + j * ts3, drow[j]); }
        }
    });
}

}  // anonymous namespace

void linear_attention_cpu(const LinearAttentionAttributes& attrs,
                          std::span<TensorView> outputs,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx,
                          void* workspace)
{
    switch (inputs[0].data_type()) {
    case DataType::f32: linear_attention_impl<float>(attrs, outputs, inputs, ctx); return;
    case DataType::f16: linear_attention_impl<half>(attrs, outputs, inputs, ctx);  return;
    default: break;
    }
    // Only f32/f16 have kernels; everything else falls back to the reference
    // (which asserts in debug and is the correctness baseline in release).
    reference::linear_attention_ref(attrs, outputs, inputs, ctx, workspace);
}

}  // namespace nnops::backend::cpu
