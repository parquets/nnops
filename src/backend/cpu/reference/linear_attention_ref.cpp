/// @file linear_attention_ref.cpp
/// @brief Scalar reference for the LinearAttention operator.
///
/// Deliberately independent of the SIMD kernel in backend/cpu/linear_attention.cpp
/// — no shared helpers, no shared loop structure — so that agreement between the
/// two is evidence about the *math*, not about a shared bug. Everything is
/// computed in f32 with naive triple-nested loops; only the I/O is rounded
/// through T.
///
/// Semantics (see linear_attention.hpp for the full contract). Per (batch, kv
/// head), carrying S[D, D] over the sequence, every token t runs:
///
///   k, v, q  = causal_conv(Q/K/V rows at t)     (when conv weights are given)
///   S[i][:] *= exp(Gate[t, i])                  decay the state
///   r[j]     = sum_i S[i][j] * k[i]             retrieval, on the decayed S
///   d[j]     = beta_t * (v[j] - r[j])           delta
///   S[i][j] += k[i] * d[j]                      rank-1 update
///   o[j]     = scale * sum_i q[i] * S[i][j]     readout, once per query head

#include "nnops/ops/linear_attention.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace nnops::backend::cpu::reference {
namespace {

using nnops::backend::cpu::half;

// Scalar I/O bridge: f32 in, f32 out, converting for f16.
inline float load_f(const float* p) noexcept { return *p; }
inline float load_f(const half* p) noexcept { return half_to_float(*p); }
inline void store_f(float* p, float v) noexcept { *p = v; }
inline void store_f(half* p, float v) noexcept { *p = float_to_half(v); }

template <typename T>
void linear_attention_ref_impl(const LinearAttentionAttributes& attrs,
                               std::span<TensorView> outputs,
                               std::span<const TensorView> inputs)
{
    const TensorView& Q  = inputs[0];
    const TensorView& K  = inputs[1];
    const TensorView& V  = inputs[2];
    const TensorView& G  = inputs[3];
    TensorView& O  = outputs[0];
    TensorView& St = outputs[1];

    const int64_t B    = Q.shape(0);
    const int64_t H    = Q.shape(1);
    const int64_t S    = Q.shape(2);
    const int64_t D    = Q.shape(3);
    const int64_t H_kv = attrs.num_kv_heads > 0 ? attrs.num_kv_heads : H;
    const int64_t Gq   = H / H_kv;   // query heads sharing one kv head

    const bool has_conv = inputs.size() > 4 && !inputs[4].is_empty();
    const bool has_beta = inputs.size() > 5 && !inputs[5].is_empty();
    const int64_t CK    = has_conv ? inputs[4].shape(2) : 0;

    const float scale = attrs.scale != 0.0f
        ? attrs.scale
        : 1.0f / std::sqrt(static_cast<float>(D));

    // Element strides (planar, row-major; the trailing dim is contiguous).
    const int64_t qs0 = Q.stride_elems(0), qs1 = Q.stride_elems(1), qs2 = Q.stride_elems(2);
    const int64_t ks0 = K.stride_elems(0), ks1 = K.stride_elems(1), ks2 = K.stride_elems(2);
    const int64_t vs0 = V.stride_elems(0), vs1 = V.stride_elems(1), vs2 = V.stride_elems(2);
    const int64_t gs0 = G.stride_elems(0), gs1 = G.stride_elems(1), gs2 = G.stride_elems(2);
    const int64_t os0 = O.stride_elems(0), os1 = O.stride_elems(1), os2 = O.stride_elems(2);
    const int64_t ts0 = St.stride_elems(0), ts1 = St.stride_elems(1),
                  ts2 = St.stride_elems(2), ts3 = St.stride_elems(3);
    const int64_t cws0 = has_conv ? inputs[4].stride_elems(0) : 0;
    const int64_t cws2 = has_conv ? inputs[4].stride_elems(2) : 0;
    const int64_t bs0  = has_beta ? inputs[5].stride_elems(0) : 0;
    const int64_t bs1  = has_beta ? inputs[5].stride_elems(1) : 0;
    const int64_t bs2  = has_beta ? inputs[5].stride_elems(2) : 0;

    // Convolution channel bases. The weights pack Q | K | V along channels:
    //   Q channels [0, H*D), K channels [H*D, H*D + H_kv*D), V after that.
    const int64_t k_ch_base = H * D;                 // K starts right after Q
    const int64_t v_ch_base = H * D + H_kv * D;      // V starts right after K

    std::vector<float> Sf(static_cast<size_t>(D) * static_cast<size_t>(D));
    std::vector<float> kc(static_cast<size_t>(D));
    std::vector<float> vc(static_cast<size_t>(D));
    std::vector<float> qc(static_cast<size_t>(D));
    std::vector<float> d(static_cast<size_t>(D));
    std::vector<float> acc(static_cast<size_t>(D));

    // Fill dst[0..D) with the row of `src` at time t, causally convoluted when
    // conv weights are present. `src` points at the (batch, head) origin of the
    // sequence and `src_ts` is the stride between consecutive tokens; `ch_base`
    // is this row's first convolution channel.
    const T* cw = has_conv ? inputs[4].ptr<T>() : nullptr;
    auto fetch_row = [&](const T* src, int64_t src_ts, int64_t ch_base,
                         int64_t t, float* dst) {
        if (!has_conv) {
            const T* row = src + t * src_ts;
            for (int64_t i = 0; i < D; ++i) { dst[i] = load_f(row + i); }
            return;
        }
        // y[t] = sum_k w[:, k] * x[t - (K-1) + k], with x = 0 for t < 0.
        for (int64_t i = 0; i < D; ++i) {
            const T* w = cw + (ch_base + i) * cws0;
            float acc_v = 0.0f;
            for (int64_t ck = 0; ck < CK; ++ck) {
                const int64_t tt = t - (CK - 1) + ck;
                if (tt < 0) { continue; }   // causal left zero-pad
                acc_v += load_f(w + ck * cws2) * load_f(src + tt * src_ts + i);
            }
            dst[i] = acc_v;
        }
    };

    const T* qp = Q.ptr<T>();
    const T* kp = K.ptr<T>();
    const T* vp = V.ptr<T>();
    const T* gp = G.ptr<T>();
    T*       op = O.ptr<T>();
    T*       sp = St.ptr<T>();

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t hkv = 0; hkv < H_kv; ++hkv) {
            const int64_t hq0 = hkv * Gq;

            // Read the carried state (f32).
            for (int64_t i = 0; i < D; ++i) {
                for (int64_t j = 0; j < D; ++j) {
                    Sf[static_cast<size_t>(i * D + j)] =
                        load_f(sp + b * ts0 + hkv * ts1 + i * ts2 + j * ts3);
                }
            }

            const T* k_base = kp + b * ks0 + hkv * ks1;
            const T* v_base = vp + b * vs0 + hkv * vs1;
            const T* g_base = gp + b * gs0 + hkv * gs1;

            for (int64_t t = 0; t < S; ++t) {
                fetch_row(k_base, ks2, k_ch_base + hkv * D, t, kc.data());
                fetch_row(v_base, vs2, v_ch_base + hkv * D, t, vc.data());

                // 1. Decay: S[i][:] *= exp(gate[t, i]).
                for (int64_t i = 0; i < D; ++i) {
                    const float g = std::exp(load_f(g_base + t * gs2 + i));
                    float* row = Sf.data() + i * D;
                    for (int64_t j = 0; j < D; ++j) { row[j] *= g; }
                }

                // 2. Retrieval r[j] = sum_i S[i][j] * k[i], on the decayed S.
                std::fill(d.begin(), d.end(), 0.0f);
                for (int64_t i = 0; i < D; ++i) {
                    const float ki = kc[static_cast<size_t>(i)];
                    const float* row = Sf.data() + i * D;
                    for (int64_t j = 0; j < D; ++j) { d[static_cast<size_t>(j)] += row[j] * ki; }
                }

                // 3. Delta d[j] = beta * (v[j] - r[j]).
                const float beta = has_beta
                    ? load_f(inputs[5].ptr<T>() + b * bs0 + hkv * bs1 + t * bs2)
                    : 1.0f;
                for (int64_t j = 0; j < D; ++j) {
                    d[static_cast<size_t>(j)] =
                        beta * (vc[static_cast<size_t>(j)] - d[static_cast<size_t>(j)]);
                }

                // 4. Rank-1 update S[i][j] += k[i] * d[j].
                for (int64_t i = 0; i < D; ++i) {
                    const float ki = kc[static_cast<size_t>(i)];
                    float* row = Sf.data() + i * D;
                    for (int64_t j = 0; j < D; ++j) { row[j] += ki * d[static_cast<size_t>(j)]; }
                }

                // 5. Readout, once per query head in this kv head's group.
                for (int64_t g = 0; g < Gq; ++g) {
                    const int64_t hq = hq0 + g;
                    fetch_row(qp + b * qs0 + hq * qs1, qs2, hq * D, t, qc.data());

                    std::fill(acc.begin(), acc.end(), 0.0f);
                    for (int64_t i = 0; i < D; ++i) {
                        const float qi = qc[static_cast<size_t>(i)];
                        const float* row = Sf.data() + i * D;
                        for (int64_t j = 0; j < D; ++j) {
                            acc[static_cast<size_t>(j)] += row[j] * qi;
                        }
                    }
                    T* o_row = op + b * os0 + hq * os1 + t * os2;
                    for (int64_t j = 0; j < D; ++j) {
                        store_f(o_row + j, scale * acc[static_cast<size_t>(j)]);
                    }
                }
            }

            // Write the final state back.
            for (int64_t i = 0; i < D; ++i) {
                for (int64_t j = 0; j < D; ++j) {
                    store_f(sp + b * ts0 + hkv * ts1 + i * ts2 + j * ts3,
                            Sf[static_cast<size_t>(i * D + j)]);
                }
            }
        }
    }
}

}  // anonymous namespace

void linear_attention_ref(const LinearAttentionAttributes& attrs,
                          std::span<TensorView> outputs,
                          std::span<const TensorView> inputs,
                          const ComputeContext& /*ctx*/,
                          void* /*workspace*/)
{
    switch (inputs[0].data_type()) {
    case DataType::f32: linear_attention_ref_impl<float>(attrs, outputs, inputs); return;
    case DataType::f16: linear_attention_ref_impl<half>(attrs, outputs, inputs);  return;
    default: break;
    }
    NNOPS_ASSERT(!"linear_attention_ref: unsupported dtype");
}

}  // namespace nnops::backend::cpu::reference
