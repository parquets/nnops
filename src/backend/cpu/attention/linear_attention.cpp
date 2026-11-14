/// @file linear_attention.cpp
/// @brief CPU LinearAttention kernel — gated delta recurrence, SIMD-accelerated.
///
/// Per (batch, value head), a state matrix S[Dk, Dv] is carried over the
/// sequence and each token runs one fused pass:
///
///   S[i][:] *= exp(Gate[t, i])                 1. decay the state
///   r[j]     = sum_i S[i][j] * k[i]            2. retrieval, on the decayed S
///   d[j]     = beta_t * (v[j] - r[j])          3. delta
///   S[i][j] += k[i] * d[j]                     4. rank-1 update
///   o[j]     = scale * sum_i q[i] * S[i][j]    5. readout, per query head
///
/// The key row (i) is Dk wide — the state's rows are decayed and updated one per
/// key channel — while the value row (j), the state's columns and the output are
/// Dv wide. Dk == Dv and H_v == H_k is the plain Gated Delta Net; a narrower
/// value is Kimi-style (D_v != D_k), and more value heads than key heads is
/// Qwen3-Next (H_v = 2·H_k).
///
/// Scheduling: one task per (batch, value head), because the value head owns the
/// state. The token loop is inherently sequential — step t+1 needs the state step
/// t produced — so it stays serial inside a task, and the parallelism comes from
/// the B * H_v independent state matrices. The query heads that read one state
/// are read out *within* the same task: splitting them across tasks would
/// advance that state once per head. A key head is shared by H_v / H_k value
/// heads; Q is indexed by the value head (the caller repeats it from the key
/// heads), because the state — and so the readout — belongs to the value head.
///
/// Two routes over that recurrence. Both are written as two state sweeps per
/// token, not the five steps above:
///
///   S'    = g ⊙ S                         g = exp(Gate[t]), per key channel
///   r     = k·S',   o0[g] = q_g·S'
///   d     = beta ⊙ (v - r)
///   S_new = S' + k⊗d
///   o[g]  = scale ⊙ (o0[g] + (q_g·k) d)
///
/// so the decay has no pass of its own (it folds into the products as a scaled
/// key/query and into the single update), and the readout reuses the product's
/// state load instead of sweeping the state a second time. All H_q/H_v query
/// heads under a value head share those loads.
///
/// The per-token route owns decode, short sequences, and any tail the chunked
/// route hands back. Long sequences run the chunked ("WY") form, which turns
/// the per-token state decay into a cumulative product over the chunk's rows:
///
///   L[t]     = exp(sum_{s<=t} Gate[s])            cumulative decay, per channel
///   Ke[t]    = k[t] ⊙ L[t],  Ka[t] = k[t] ⊙ 1/L[t]
///   A[r][s]  = Ke[r] · Ka[s]                      strict lower, r > s
///   (I + b⊙A) D = b⊙(V - Ke·S_in)                 forward substitution, the deltas
///   S_out    = L_T ⊙ (S_in + Kaᵀ·D)               one state pass, decay folded in
///   O[t]     = scale·(Qe[t]·S_in + sum_{s<=t} (Qe[t]·Ka[s]) D[s])
///
/// The Ke·S and Qe·S products block eight rows per state load (a row at a time
/// would read the whole state once per token in the chunk, which is the traffic
/// chunking exists to remove), and the state update carries four value vectors
/// per key broadcast. The triangular solve is the only serial part: C vector
/// AXPYs. A full 16-token chunk therefore reads the state 2·(1 + Gqq) times
/// (two blocked blocks per product) plus one update pass, where a row at a time
/// read it 16·(1 + Gqq) times — which is what makes the chunked route ~1.5x the
/// fused per-token route on the GDN shapes rather than slower than it.
///
/// L is a product of exps, so the chunk's ± scaling only holds while the
/// cumulative log decay stays inside f32 (|sum| < ~88, and 80 with margin).
/// A chunk that exceeds it hands itself and the rest of the sequence back to
/// the per-token loop, which has no such bound, and the state is continuous
/// across the switch. Chunking is skipped below kChunkMinSeq tokens, where the
/// per-chunk overheads would not be repaid.
///
/// The chunked route is not bit-identical to the per-token one: it reassociates
/// the state's contributions into a per-chunk sum. It agrees to f32 rounding,
/// which is what the tests' tolerances measure.
///
/// SIMD: every stage is vectorized across the value dimension (j), with the key
/// dimension (i) broadcast as a scalar — the shape ORT's MLAS NEON kernel
/// (FusedTokenSinglePassNeon) uses. The state, the delta and the readout
/// accumulators are f32 even for f16 input; f16 appears only at the I/O
/// boundary, where the T-typed load/store wrappers widen and narrow rows.
///
/// The convolution, when weights are supplied, is a causal depthwise conv1d over
/// the packed Q | K | V channels. The weights are transposed once per task into a
/// [local channel][tap][d] layout, so each tap is a contiguous per-lane weight
/// vector — a plain vector FMA with no gather.

#include "nnops/ops/linear_attention.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/simd/simd.hpp"

#include "backend/cpu/common/memory_pool.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <span>
#include <type_traits>

namespace nnops::backend::cpu {
namespace {

using nnops::backend::cpu::half;
using nnops::backend::cpu::bfloat16;
using namespace nnops::simd;

/// Tokens per chunk on the chunked route. Measured on the GDN shapes at 8, 16
/// and 32: the first two tie at ~1.4x the per-token route, and 32 gives a few
/// percent back to the intra-chunk products. 16 also leaves a realistic gate's
/// cumulative log decay far inside f32's exponent range, so the fallback below
/// only fires on decay far outside what a trained model produces.
constexpr int64_t kChunk = 16;
/// Rows of a chunk that share one state load in the Ke·S / Qe·S products.
/// A row-at-a-time product would re-read the whole state once per token, which
/// is the traffic the chunked route's two passes are supposed to remove.
constexpr int64_t kChunkRows = 8;
/// Below this the per-token loop wins: one chunk cannot amortize the per-chunk
/// setup, and decode must never pay for it.
constexpr int64_t kChunkMinSeq = 64;
/// |cumulative log decay| beyond which a chunk's ± scaling leaves f32
/// (exp(80) ≈ 5.5e34; exp(88.7) overflows). Such a chunk falls back.
constexpr float kChunkSafeLog = 80.0f;

/// Route override for A/B measurement and for tests that run one problem
/// through both routes: NNOPS_LINEAR_ATTENTION_ROUTE=token|chunk. Unset (or
/// any other value) is the automatic choice, chunked from kChunkMinSeq up.
inline int route_override()
{
    const char* e = std::getenv("NNOPS_LINEAR_ATTENTION_ROUTE");
    if (!e) { return 0; }
    if (std::strcmp(e, "token") == 0) { return -1; }
    if (std::strcmp(e, "chunk") == 0) { return 1; }
    return 0;
}

/// Widen L consecutive T elements to an f32 vector (f16 and bf16 widen, f32 is
/// a load — simd.hpp's own widening overloads cover the bf16 case).
inline v_f32x8 vload_f32(const float* p) { return v_load(p); }
inline v_f32x8 vload_f32(const half* p) { return v_cvt_f16_to_f32(p); }
#ifdef NNOPS_ARCH_AARCH64
inline v_f32x8 vload_f32(const bfloat16* p) { return v_load(p); }
#endif

/// Narrow an f32 vector back to T and store it.
inline void vstore_f32(float* p, const v_f32x8& v) { v_store(p, v); }
inline void vstore_f32(half* p, const v_f32x8& v)  { v_store(p, v_cvt_f32_to_f16(v)); }
#ifdef NNOPS_ARCH_AARCH64
inline void vstore_f32(bfloat16* p, const v_f32x8& v) { v_store(p, v); }
#endif

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

    const int64_t B  = Q.shape(0);
    const int64_t Hq = Q.shape(1);
    const int64_t S  = Q.shape(2);
    const int64_t Dk = Q.shape(3);
    const int64_t Hk = attrs.num_kv_heads > 0 ? attrs.num_kv_heads : Hq;
    const int64_t Hv = V.shape(1);   // the value head owns the state
    const int64_t Dv = V.shape(3);   // the state's column / output width

    // Q is per value head: the expansion from the key heads to the value heads
    // (FLA's repeat_interleave) happens on the caller's side, because the state
    // and therefore the readout belong to the value head — a query head shared
    // by several states would have several outputs and no place to put them.
    const int64_t Gqq = Hq / Hv;   // query heads per value head
    const int64_t Gvk = Hv / Hk;   // value heads per key head

    NNOPS_ASSERT(Gqq > 0 && Gqq * Hv == Hq);
    NNOPS_ASSERT(Hk > 0 && Gvk * Hk == Hv);

    const bool has_conv = inputs.size() > 4 && !inputs[4].is_empty();
    const bool has_beta = inputs.size() > 5 && !inputs[5].is_empty();
    const int64_t CK    = has_conv ? inputs[4].shape(2) : 0;

    const float scale = attrs.scale != 0.0f
        ? attrs.scale
        : 1.0f / std::sqrt(static_cast<float>(Dk));

    // Element strides (channel-first; the trailing dim is contiguous).
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

    // Convolution channel bases: the packed weights lay Q out first, then K, then
    // V — the query block over its expanded heads, each at its own row width.
    const int64_t k_ch_base = Hq * Dk;
    const int64_t v_ch_base = Hq * Dk + Hk * Dk;

    // Task grid: one task per (batch, value head) is the natural split, but a
    // layer with fewer state matrices than workers leaves the machine idle (an
    // 8-value-head layer is 8 tasks on 20 threads). The value columns are
    // independent — every stage works one column at a time — so they are cut
    // into whole-vector slabs until the grid fills the workers. Slabs recompute
    // the per-key-channel setup (gate, Ke/Ka, A) but duplicate no state
    // arithmetic, and each column's sums keep their order, so results are
    // bit-identical to the unsplit route.
    const int64_t n_state = B * Hv;
    int64_t slab = Dv;
    int64_t n_slabs = 1;
    if (const int64_t workers = ctx.cpu.worker_count(); workers > n_state) {
        int64_t want = (workers + n_state - 1) / n_state;
        const int64_t max_slabs = Dv / L > 0 ? Dv / L : 1;
        if (want > max_slabs) { want = max_slabs; }
        if (want > 1) {
            slab = ((Dv + want - 1) / want + L - 1) / L * L;
            n_slabs = (Dv + slab - 1) / slab;
        }
    }

    const int64_t NTasks = n_state * n_slabs;
    ctx.cpu.run(0, NTasks, [&](int64_t task) {
        const int64_t js  = task % n_slabs;
        const int64_t hv  = (task / n_slabs) % Hv;
        const int64_t b   = task / (n_slabs * Hv);
        const int64_t j0  = js * slab;
        const int64_t j1  = (j0 + slab < Dv) ? j0 + slab : Dv;
        const int64_t hk  = hv / Gvk;
        const int64_t hq0 = hv * Gqq;

        // Per-task scratch, all f32: the state matrix, the per-token key / value
        // / gate rows, the gate-folded key, the delta, and one query row plus
        // its q·k for each query head under this value head.
        PoolPtr state_owner(sizeof(float) * static_cast<size_t>(Dk) *
                            static_cast<size_t>(Dv));
        PoolPtr rows_owner(sizeof(float) * static_cast<size_t>(
            4 * Dk + 2 * Dv + Gqq * (Dk + 1)));
        float* Sf = state_owner.as<float>();
        float* kc = rows_owner.as<float>();
        float* vc = kc + Dk;
        float* gv = vc + Dv;
        float* kg = gv + Dk;
        float* dv = kg + Dk;
        float* qbuf = dv + Dv;
        float* qk_h = qbuf + Gqq * Dk;

        PoolPtr wt_owner;
        float* wq = nullptr;
        float* wk = nullptr;
        float* wv = nullptr;
        if (has_conv) {
            const int64_t cq = Gqq * CK * Dk;
            const int64_t ck = CK * Dk;
            const int64_t cv = CK * Dv;
            wt_owner = PoolPtr(sizeof(float) * static_cast<size_t>(cq + ck + cv));
            wq = wt_owner.as<float>();
            wk = wq + cq;
            wv = wk + ck;
            const T* cwp = inputs[4].ptr<T>();
            for (int64_t c = 0; c < Gqq; ++c) {
                const int64_t ch = (hq0 + c) * Dk;
                for (int64_t k = 0; k < CK; ++k) {
                    float* dst = wq + (c * CK + k) * Dk;
                    for (int64_t d = 0; d < Dk; ++d) {
                        dst[d] = s_load(cwp + (ch + d) * cws0 + k * cws2);
                    }
                }
            }
            for (int64_t k = 0; k < CK; ++k) {
                float* dst = wk + k * Dk;
                for (int64_t d = 0; d < Dk; ++d) {
                    dst[d] = s_load(cwp + (k_ch_base + hk * Dk + d) * cws0 + k * cws2);
                }
            }
            for (int64_t k = 0; k < CK; ++k) {
                float* dst = wv + k * Dv;
                for (int64_t d = 0; d < Dv; ++d) {
                    dst[d] = s_load(cwp + (v_ch_base + hv * Dv + d) * cws0 + k * cws2);
                }
            }
        }

        // Fill dst[0..len) with the columns [off, off+len) of `src`'s row at
        // time t, convolved against the [tap][W] weights `wch` when has_conv.
        // W is the source's full row width; a value-column slab passes off/len.
        auto fetch_row = [&](const T* src, int64_t src_ts, const float* wch,
                             int64_t W, int64_t off, int64_t len, int64_t t,
                             float* dst) {
            if (!has_conv) {
                const T* row = src + t * src_ts + off;
                int64_t d = 0;
                for (; d + L <= len; d += L) { vstore_f32(dst + d, vload_f32(row + d)); }
                for (; d < len; ++d) { dst[d] = s_load(row + d); }
                return;
            }
            // y[t] = sum_k w[:, k] * x[t - (K-1) + k], x = 0 for t < 0.
            int64_t d = 0;
            for (; d + L <= len; d += L) {
                v_f32x8 acc = v_zero_f32x8();
                for (int64_t k = 0; k < CK; ++k) {
                    const int64_t tt = t - (CK - 1) + k;
                    if (tt < 0) { continue; }   // causal left zero-pad
                    acc = v_fmadd(vload_f32(src + tt * src_ts + off + d),
                                  v_load(wch + k * W + off + d),
                                  acc);
                }
                v_store(dst + d, acc);
            }
            for (; d < len; ++d) {
                float acc = 0.0f;
                for (int64_t k = 0; k < CK; ++k) {
                    const int64_t tt = t - (CK - 1) + k;
                    if (tt < 0) { continue; }
                    acc += wch[k * W + off + d] * s_load(src + tt * src_ts + off + d);
                }
                dst[d] = acc;
            }
        };

        for (int64_t i = 0; i < Dk; ++i) {
            const T* srow = sp + b * ts0 + hv * ts1 + i * ts2;
            float* drow = Sf + i * Dv;
            int64_t j = j0;
            for (; j + L <= j1; j += L) { vstore_f32(drow + j, vload_f32(srow + j * ts3)); }
            for (; j < j1; ++j) { drow[j] = s_load(srow + j * ts3); }
        }

        const T* k_base = kp + b * ks0 + hk * ks1;
        const T* v_base = vp + b * vs0 + hv * vs1;
        const T* g_base = gp + b * gs0 + hv * gs1;
        const T* b_base = has_beta ? inputs[5].ptr<T>() + b * bs0 + hv * bs1 : nullptr;

        // One token, two state sweeps. The five steps in the file header
        // associate as (g = exp(Gate[t]))
        //
        //   S'    = g ⊙ S                          per key channel
        //   r     = k·S',   o0[g] = q_g·S'
        //   d     = beta ⊙ (v - r)
        //   S_new = S' + k⊗d
        //   o[g]  = scale ⊙ (o0[g] + (q_g·k) d)
        //
        // so the decay does not need a pass of its own (it rides the two
        // products as a folded k and the one update), and the readout needs no
        // state pass either: q·(k⊗d) is (q·k)d. A state element is loaded twice
        // instead of once per step, and all Gqq query heads share both loads.
        // The g fold into k and q costs Dk multiplications; the update multiplies
        // the row by g as it adds k⊗d.
        auto token_step = [&](int64_t t) {
            fetch_row(k_base, ks2, wk, Dk, 0, Dk, t, kc);
            fetch_row(v_base, vs2, wv, Dv, j0, j1 - j0, t, vc + j0);

            // v_exp is the Cephes-style vector exp; Dk scalar exps per token
            // would outweigh the Dk*Dv/L vector ops they feed.
            const T* g_row = g_base + t * gs2;
            int64_t i = 0;
            for (; i + L <= Dk; i += L) { v_store(gv + i, v_exp(vload_f32(g_row + i))); }
            for (; i < Dk; ++i) { gv[i] = std::exp(s_load(g_row + i)); }
            for (i = 0; i < Dk; ++i) { kg[i] = kc[i] * gv[i]; }

            // One query row per query head, its q·k, and the gate folded in so
            // the shared product below carries one broadcast per head.
            for (int64_t g = 0; g < Gqq; ++g) {
                float* qrow = qbuf + g * Dk;
                fetch_row(qp + b * qs0 + (hq0 + g) * qs1, qs2,
                          has_conv ? wq + g * CK * Dk : nullptr, Dk, 0, Dk, t, qrow);
                v_f32x8 acc = v_zero_f32x8();
                int64_t ii = 0;
                for (; ii + L <= Dk; ii += L) {
                    acc = v_fmadd(v_load(qrow + ii), v_load(kc + ii), acc);
                }
                float qk = v_reduce_sum(acc);
                for (; ii < Dk; ++ii) { qk += qrow[ii] * kc[ii]; }
                qk_h[g] = qk;
                for (ii = 0; ii < Dk; ++ii) { qrow[ii] *= gv[ii]; }
            }

            const float beta = has_beta ? s_load(b_base + t * bs2) : 1.0f;
            const v_f32x8 vbeta  = v_set1_f32x8(beta);
            const v_f32x8 vscale = v_set1_f32x8(scale);

            // One block of NJ vectors for a group of NH query heads. NJ trades
            // registers for state re-reads: each state line serves NJ vectors,
            // and NH heads share the loads within a vector. NH*NJ is kept at 8
            // so the accumulators fit AVX2's 16 registers.
            auto run_block = [&](int64_t g0, int64_t j, auto nh_tag, auto nj_tag) {
                constexpr int NH = decltype(nh_tag)::value;
                constexpr int NJ = decltype(nj_tag)::value;
                const bool first = (g0 == 0);
                v_f32x8 accK[NJ];
                v_f32x8 accQ[NH][NJ];
                for (int jj = 0; jj < NJ; ++jj) { accK[jj] = v_zero_f32x8(); }
                for (int h = 0; h < NH; ++h) {
                    for (int jj = 0; jj < NJ; ++jj) { accQ[h][jj] = v_zero_f32x8(); }
                }
                for (int64_t ii = 0; ii < Dk; ++ii) {
                    v_f32x8 s[NJ];
                    for (int jj = 0; jj < NJ; ++jj) {
                        s[jj] = v_load(Sf + ii * Dv + j + jj * L);
                    }
                    if (first) {
                        const v_f32x8 b = v_set1_f32x8(kg[ii]);
                        for (int jj = 0; jj < NJ; ++jj) {
                            accK[jj] = v_fmadd(b, s[jj], accK[jj]);
                        }
                    }
                    for (int h = 0; h < NH; ++h) {
                        const v_f32x8 b = v_set1_f32x8(qbuf[(g0 + h) * Dk + ii]);
                        for (int jj = 0; jj < NJ; ++jj) {
                            accQ[h][jj] = v_fmadd(b, s[jj], accQ[h][jj]);
                        }
                    }
                }
                for (int jj = 0; jj < NJ; ++jj) {
                    v_f32x8 dvec;
                    if (first) {
                        dvec = v_mul(vbeta,
                                     v_sub(v_load(vc + j + jj * L), accK[jj]));
                        v_store(dv + j + jj * L, dvec);
                    } else {
                        dvec = v_load(dv + j + jj * L);
                    }
                    for (int h = 0; h < NH; ++h) {
                        const v_f32x8 o = v_mul(
                            vscale,
                            v_fmadd(v_set1_f32x8(qk_h[g0 + h]), dvec, accQ[h][jj]));
                        vstore_f32(op + b * os0 + (hq0 + g0 + h) * os1 + t * os2 +
                                       j + jj * L,
                                   o);
                    }
                }
            };

            // The scalar tail of a group; only the first group forms the delta.
            auto run_tail = [&](int64_t g0, int64_t jstart, auto nh_tag) {
                constexpr int NH = decltype(nh_tag)::value;
                const bool first = (g0 == 0);
                for (int64_t j = jstart; j < j1; ++j) {
                    float accK = 0.0f;
                    float accQ[NH];
                    for (int h = 0; h < NH; ++h) { accQ[h] = 0.0f; }
                    for (int64_t ii = 0; ii < Dk; ++ii) {
                        const float s = Sf[ii * Dv + j];
                        if (first) { accK += kg[ii] * s; }
                        for (int h = 0; h < NH; ++h) {
                            accQ[h] += qbuf[(g0 + h) * Dk + ii] * s;
                        }
                    }
                    float d;
                    if (first) {
                        d = beta * (vc[j] - accK);
                        dv[j] = d;
                    } else {
                        d = dv[j];
                    }
                    for (int h = 0; h < NH; ++h) {
                        s_store(op + b * os0 + (hq0 + g0 + h) * os1 + t * os2 + j,
                                scale * (accQ[h] + qk_h[g0 + h] * d));
                    }
                }
            };

            // Gqq is decomposed into groups of 8/4/2/1 query heads so NH is a
            // compile-time bound; the group sizes also pick NJ.
            auto run_group = [&](int64_t g0, auto nh_tag) {
                constexpr int NH = decltype(nh_tag)::value;
                constexpr int NJ = (NH == 8) ? 1 : (NH == 1 ? 4 : 2);
                int64_t j = j0;
                for (; j + NJ * L <= j1; j += NJ * L) {
                    run_block(g0, j, nh_tag, std::integral_constant<int, NJ>{});
                }
                for (; j + L <= j1; j += L) {
                    run_block(g0, j, nh_tag, std::integral_constant<int, 1>{});
                }
                run_tail(g0, j, nh_tag);
            };

            int64_t g0 = 0;
            for (; g0 + 8 <= Gqq; g0 += 8) {
                run_group(g0, std::integral_constant<int, 8>{});
            }
            if (Gqq - g0 >= 4) {
                run_group(g0, std::integral_constant<int, 4>{});
                g0 += 4;
            }
            if (Gqq - g0 >= 2) {
                run_group(g0, std::integral_constant<int, 2>{});
                g0 += 2;
            }
            if (Gqq - g0 >= 1) {
                run_group(g0, std::integral_constant<int, 1>{});
            }

            // Rank-1 update: S[i][j] = g[i]·S[i][j] + k[i]·d[j], one row-major
            // pass (the decay folded in, so it is not a pass of its own).
            for (int64_t ii = 0; ii < Dk; ++ii) {
                const v_f32x8 vg = v_set1_f32x8(gv[ii]);
                const v_f32x8 vk = v_set1_f32x8(kc[ii]);
                float* row = Sf + ii * Dv;
                int64_t j = j0;
                for (; j + L <= j1; j += L) {
                    v_store(row + j, v_fmadd(vk, v_load(dv + j),
                                             v_mul(vg, v_load(row + j))));
                }
                for (; j < j1; ++j) { row[j] = gv[ii] * row[j] + kc[ii] * dv[j]; }
            }
        };

        // ---- chunked prefill -----------------------------------------------
        // See the file header for the derivation. Returns the token the
        // per-token loop resumes at: S when every chunk ran, or the start of
        // the first chunk whose cumulative decay left f32's exponent range.
        auto chunk_route = [&]() -> int64_t {
            constexpr int64_t C = kChunk;
            constexpr int NV = static_cast<int>((C + L - 1) / L);   // a C×C row
            const int64_t W = Dk > Dv ? Dk : Dv;
            PoolPtr ck_owner(sizeof(float) * static_cast<size_t>(
                C * Dk * 6 + C * C + C * Dv * 2 + C + 2 * W));
            float* cum = ck_owner.as<float>();   // [C][Dk] log decay, then Qe
            float* ep  = cum + C * Dk;           // [C][Dk] exp(+cum)
            float* em  = ep + C * Dk;            // [C][Dk] exp(-cum)
            float* ke  = em + C * Dk;            // [C][Dk] Ke = k ⊙ ep
            float* ka  = ke + C * Dk;            // [C][Dk] Ka = k ⊙ em
            float* kat = ka + C * Dk;            // [Dk][C] Ka transposed
            float* am  = kat + C * Dk;           // [C][C] A, then Aq
            float* dl  = am + C * C;             // [C][Dv] deltas
            float* vb  = dl + C * Dv;            // [C][Dv] beta ⊙ V
            float* bt  = vb + C * Dv;            // [C] beta
            float* ra  = bt + C;                 // [W] row buffer
            float* rb  = ra + W;                 // [W] row buffer
            const v_f32x8 vs = v_set1_f32x8(scale);

            int64_t t0 = 0;
            while (t0 < S) {
                const int64_t ct = (S - t0 < C) ? S - t0 : C;

                // Cumulative log decay, and the bound its exponentials need.
                float worst = 0.0f;
                for (int64_t r = 0; r < ct; ++r) {
                    const T* g_row = g_base + (t0 + r) * gs2;
                    float* c_row = cum + r * Dk;
                    const float* prev = r > 0 ? c_row - Dk : nullptr;
                    int64_t i = 0;
                    for (; i + L <= Dk; i += L) {
                        const v_f32x8 v = vload_f32(g_row + i);
                        v_store(c_row + i, prev ? v_add(v, v_load(prev + i)) : v);
                    }
                    for (; i < Dk; ++i) {
                        c_row[i] = s_load(g_row + i) + (prev ? prev[i] : 0.0f);
                    }
                    for (i = 0; i < Dk; ++i) {
                        const float a = std::fabs(c_row[i]);
                        worst = a > worst ? a : worst;
                    }
                }
                if (worst > kChunkSafeLog) { break; }

                for (int64_t r = 0; r < ct; ++r) {
                    const float* c_row = cum + r * Dk;
                    float* p_row = ep + r * Dk;
                    float* m_row = em + r * Dk;
                    int64_t i = 0;
                    for (; i + L <= Dk; i += L) {
                        const v_f32x8 v = v_load(c_row + i);
                        v_store(p_row + i, v_exp(v));
                        v_store(m_row + i, v_exp(v_sub(v_zero_f32x8(), v)));
                    }
                    for (; i < Dk; ++i) {
                        p_row[i] = std::exp(c_row[i]);
                        m_row[i] = std::exp(-c_row[i]);
                    }
                }

                // Scaled key rows, in both layouts the products below want.
                for (int64_t r = 0; r < ct; ++r) {
                    fetch_row(k_base, ks2, wk, Dk, 0, Dk, t0 + r, ra);
                    const float* p_row = ep + r * Dk;
                    const float* m_row = em + r * Dk;
                    float* e_row = ke + r * Dk;
                    float* a_row = ka + r * Dk;
                    for (int64_t i = 0; i < Dk; ++i) {
                        e_row[i] = ra[i] * p_row[i];
                        a_row[i] = ra[i] * m_row[i];
                        kat[i * C + r] = a_row[i];
                    }
                }
                // The tail chunk leaves the transposed columns past ct holding
                // whatever the pool had; every reader stays inside ct, but a
                // zero keeps the unused lanes from carrying a NaN.
                for (int64_t i = 0; i < Dk && ct < C; ++i) {
                    for (int64_t s = ct; s < C; ++s) { kat[i * C + s] = 0.0f; }
                }

                // A[r][s] = Ke[r] · Ka[s]; only r > s is read back. Four rows
                // per pass share the transposed Ka loads.
                {
                    constexpr int NA = 4;
                    int64_t ra = 0;
                    for (; ra + NA <= ct; ra += NA) {
                        v_f32x8 acc[NA][NV];
                        for (int rr = 0; rr < NA; ++rr) {
                            for (int nv = 0; nv < NV; ++nv) { acc[rr][nv] = v_zero_f32x8(); }
                        }
                        for (int64_t i = 0; i < Dk; ++i) {
                            v_f32x8 kv[NV];
                            for (int nv = 0; nv < NV; ++nv) {
                                kv[nv] = v_load(kat + i * C + nv * L);
                            }
                            for (int rr = 0; rr < NA; ++rr) {
                                const v_f32x8 b = v_set1_f32x8(ke[(ra + rr) * Dk + i]);
                                for (int nv = 0; nv < NV; ++nv) {
                                    acc[rr][nv] = v_fmadd(b, kv[nv], acc[rr][nv]);
                                }
                            }
                        }
                        for (int rr = 0; rr < NA; ++rr) {
                            for (int nv = 0; nv < NV; ++nv) {
                                v_store(am + (ra + rr) * C + nv * L, acc[rr][nv]);
                            }
                        }
                    }
                    for (; ra < ct; ++ra) {
                        const float* e_row = ke + ra * Dk;
                        v_f32x8 acc[NV];
                        for (int nv = 0; nv < NV; ++nv) { acc[nv] = v_zero_f32x8(); }
                        for (int64_t i = 0; i < Dk; ++i) {
                            const v_f32x8 b = v_set1_f32x8(e_row[i]);
                            for (int nv = 0; nv < NV; ++nv) {
                                acc[nv] = v_fmadd(b, v_load(kat + i * C + nv * L), acc[nv]);
                            }
                        }
                        for (int nv = 0; nv < NV; ++nv) {
                            v_store(am + ra * C + nv * L, acc[nv]);
                        }
                    }
                }

                for (int64_t r = 0; r < ct; ++r) {
                    bt[r] = has_beta ? s_load(b_base + (t0 + r) * bs2) : 1.0f;
                }

                // beta ⊙ V, so the product below only has to subtract and scale.
                for (int64_t r = 0; r < ct; ++r) {
                    fetch_row(v_base, vs2, wv, Dv, j0, j1 - j0, t0 + r, rb);
                    const float b = bt[r];
                    float* vrow = vb + r * Dv;
                    for (int64_t d = 0; d < j1 - j0; ++d) { vrow[j0 + d] = b * rb[d]; }
                }

                // X = beta ⊙ (V - Ke · S_in), into the delta block. Rows share
                // the state load: a row-at-a-time product would read the whole
                // state once per token, the traffic chunking exists to remove.
                int64_t r0 = 0;
                for (; r0 + kChunkRows <= ct; r0 += kChunkRows) {
                    constexpr int NR = static_cast<int>(kChunkRows);
                    int64_t j = j0;
                    for (; j + L <= j1; j += L) {
                        v_f32x8 acc[NR];
                        for (int rr = 0; rr < NR; ++rr) { acc[rr] = v_zero_f32x8(); }
                        for (int64_t i = 0; i < Dk; ++i) {
                            const v_f32x8 s = v_load(Sf + i * Dv + j);
                            for (int rr = 0; rr < NR; ++rr) {
                                acc[rr] = v_fmadd(
                                    v_set1_f32x8(ke[(r0 + rr) * Dk + i]), s, acc[rr]);
                            }
                        }
                        for (int rr = 0; rr < NR; ++rr) {
                            const int64_t r = r0 + rr;
                            v_store(dl + r * Dv + j,
                                    v_sub(v_load(vb + r * Dv + j),
                                          v_mul(v_set1_f32x8(bt[r]), acc[rr])));
                        }
                    }
                    for (; j < j1; ++j) {
                        float acc[NR] = {};
                        for (int64_t i = 0; i < Dk; ++i) {
                            const float s = Sf[i * Dv + j];
                            for (int rr = 0; rr < NR; ++rr) {
                                acc[rr] += ke[(r0 + rr) * Dk + i] * s;
                            }
                        }
                        for (int rr = 0; rr < NR; ++rr) {
                            const int64_t r = r0 + rr;
                            dl[r * Dv + j] = vb[r * Dv + j] - bt[r] * acc[rr];
                        }
                    }
                }
                // The last partial chunk (at most seven rows, once per sequence).
                for (; r0 < ct; ++r0) {
                    const float* e_row = ke + r0 * Dk;
                    float* x_row = dl + r0 * Dv;
                    const float* v_row = vb + r0 * Dv;
                    const float b = bt[r0];
                    int64_t j = j0;
                    for (; j + L <= j1; j += L) {
                        v_f32x8 acc = v_zero_f32x8();
                        for (int64_t i = 0; i < Dk; ++i) {
                            acc = v_fmadd(v_set1_f32x8(e_row[i]),
                                          v_load(Sf + i * Dv + j), acc);
                        }
                        v_store(x_row + j, v_sub(v_load(v_row + j), v_mul(v_set1_f32x8(b), acc)));
                    }
                    for (; j < j1; ++j) {
                        float acc = 0.0f;
                        for (int64_t i = 0; i < Dk; ++i) { acc += e_row[i] * Sf[i * Dv + j]; }
                        x_row[j] = v_row[j] - b * acc;
                    }
                }

                // (I + beta ⊙ A) D = X, forward substitution in place.
                for (int64_t r = 0; r < ct; ++r) {
                    float* x_row = dl + r * Dv;
                    const float* a_row = am + r * C;
                    const float beta = bt[r];
                    for (int64_t s = 0; s < r; ++s) {
                        const v_f32x8 c = v_set1_f32x8(-beta * a_row[s]);
                        const float* d_row = dl + s * Dv;
                        int64_t j = j0;
                        for (; j + L <= j1; j += L) {
                            v_store(x_row + j,
                                    v_fmadd(c, v_load(d_row + j), v_load(x_row + j)));
                        }
                        for (; j < j1; ++j) { x_row[j] -= beta * a_row[s] * d_row[j]; }
                    }
                }

                // O = scale·(Qe·S_in + tril(Qe·Kaᵀ)·D), one query head at a
                // time; the readout reads S_in, so it precedes the update.
                for (int64_t g = 0; g < Gqq; ++g) {
                    const int64_t hq = hq0 + g;
                    const T* q_base = qp + b * qs0 + hq * qs1;
                    for (int64_t r = 0; r < ct; ++r) {
                        fetch_row(q_base, qs2, has_conv ? wq + g * CK * Dk : nullptr,
                                  Dk, 0, Dk, t0 + r, ra);
                        const float* p_row = ep + r * Dk;
                        float* q_row = cum + r * Dk;   // the log decay is dead here
                        for (int64_t i = 0; i < Dk; ++i) { q_row[i] = ra[i] * p_row[i]; }
                    }

                    // Aq[r][s] = Qe[r] · Ka[s], the same four-row blocking.
                    {
                        constexpr int NA = 4;
                        int64_t ra = 0;
                        for (; ra + NA <= ct; ra += NA) {
                            v_f32x8 acc[NA][NV];
                            for (int rr = 0; rr < NA; ++rr) {
                                for (int nv = 0; nv < NV; ++nv) {
                                    acc[rr][nv] = v_zero_f32x8();
                                }
                            }
                            for (int64_t i = 0; i < Dk; ++i) {
                                v_f32x8 kv[NV];
                                for (int nv = 0; nv < NV; ++nv) {
                                    kv[nv] = v_load(kat + i * C + nv * L);
                                }
                                for (int rr = 0; rr < NA; ++rr) {
                                    const v_f32x8 bq =
                                        v_set1_f32x8(cum[(ra + rr) * Dk + i]);
                                    for (int nv = 0; nv < NV; ++nv) {
                                        acc[rr][nv] = v_fmadd(bq, kv[nv], acc[rr][nv]);
                                    }
                                }
                            }
                            for (int rr = 0; rr < NA; ++rr) {
                                for (int nv = 0; nv < NV; ++nv) {
                                    v_store(am + (ra + rr) * C + nv * L, acc[rr][nv]);
                                }
                            }
                        }
                        for (; ra < ct; ++ra) {
                            const float* q_row = cum + ra * Dk;
                            v_f32x8 acc[NV];
                            for (int nv = 0; nv < NV; ++nv) { acc[nv] = v_zero_f32x8(); }
                            for (int64_t i = 0; i < Dk; ++i) {
                                const v_f32x8 bq = v_set1_f32x8(q_row[i]);
                                for (int nv = 0; nv < NV; ++nv) {
                                    acc[nv] = v_fmadd(bq, v_load(kat + i * C + nv * L),
                                                      acc[nv]);
                                }
                            }
                            for (int nv = 0; nv < NV; ++nv) {
                                v_store(am + ra * C + nv * L, acc[nv]);
                            }
                        }
                    }

                    // The state product, eight rows per state load, then the
                    // strict-lower intra-chunk terms: rows before r0 take every
                    // s, the block's own rows only the diagonal-and-below part.
                    int64_t r0 = 0;
                    for (; r0 + kChunkRows <= ct; r0 += kChunkRows) {
                        constexpr int NR = static_cast<int>(kChunkRows);
                        int64_t j = j0;
                        for (; j + L <= j1; j += L) {
                            v_f32x8 acc[NR];
                            for (int rr = 0; rr < NR; ++rr) { acc[rr] = v_zero_f32x8(); }
                            for (int64_t i = 0; i < Dk; ++i) {
                                const v_f32x8 s = v_load(Sf + i * Dv + j);
                                for (int rr = 0; rr < NR; ++rr) {
                                    acc[rr] = v_fmadd(
                                        v_set1_f32x8(cum[(r0 + rr) * Dk + i]), s, acc[rr]);
                                }
                            }
                            for (int64_t s = 0; s < r0; ++s) {
                                const v_f32x8 d = v_load(dl + s * Dv + j);
                                for (int rr = 0; rr < NR; ++rr) {
                                    acc[rr] = v_fmadd(
                                        v_set1_f32x8(am[(r0 + rr) * C + s]), d, acc[rr]);
                                }
                            }
                            for (int64_t s = r0; s < r0 + NR; ++s) {
                                const v_f32x8 d = v_load(dl + s * Dv + j);
                                for (int64_t rr = s - r0; rr < NR; ++rr) {
                                    acc[rr] = v_fmadd(
                                        v_set1_f32x8(am[(r0 + rr) * C + s]), d, acc[rr]);
                                }
                            }
                            for (int rr = 0; rr < NR; ++rr) {
                                const int64_t r = r0 + rr;
                                vstore_f32(op + b * os0 + hq * os1 + (t0 + r) * os2 + j,
                                           v_mul(acc[rr], vs));
                            }
                        }
                        for (; j < j1; ++j) {
                            float acc[NR] = {};
                            for (int64_t i = 0; i < Dk; ++i) {
                                const float s = Sf[i * Dv + j];
                                for (int rr = 0; rr < NR; ++rr) {
                                    acc[rr] += cum[(r0 + rr) * Dk + i] * s;
                                }
                            }
                            for (int64_t s = 0; s < r0 + NR; ++s) {
                                const float d = dl[s * Dv + j];
                                for (int64_t rr = (s >= r0) ? s - r0 : 0; rr < NR; ++rr) {
                                    acc[rr] += am[(r0 + rr) * C + s] * d;
                                }
                            }
                            for (int rr = 0; rr < NR; ++rr) {
                                const int64_t r = r0 + rr;
                                s_store(op + b * os0 + hq * os1 + (t0 + r) * os2 + j,
                                        scale * acc[rr]);
                            }
                        }
                    }
                    // The last partial chunk, row at a time.
                    for (; r0 < ct; ++r0) {
                        const float* q_row = cum + r0 * Dk;
                        const float* a_row = am + r0 * C;
                        const int64_t ns = r0 + 1;
                        T* o_row = op + b * os0 + hq * os1 + (t0 + r0) * os2;
                        int64_t j = j0;
                        for (; j + L <= j1; j += L) {
                            v_f32x8 acc = v_zero_f32x8();
                            for (int64_t i = 0; i < Dk; ++i) {
                                acc = v_fmadd(v_set1_f32x8(q_row[i]),
                                              v_load(Sf + i * Dv + j), acc);
                            }
                            for (int64_t s = 0; s < ns; ++s) {
                                acc = v_fmadd(v_set1_f32x8(a_row[s]),
                                              v_load(dl + s * Dv + j), acc);
                            }
                            vstore_f32(o_row + j, v_mul(acc, vs));
                        }
                        for (; j < j1; ++j) {
                            float acc = 0.0f;
                            for (int64_t i = 0; i < Dk; ++i) { acc += q_row[i] * Sf[i * Dv + j]; }
                            for (int64_t s = 0; s < ns; ++s) { acc += a_row[s] * dl[s * Dv + j]; }
                            s_store(o_row + j, scale * acc);
                        }
                    }
                }

                // S_out = L_T ⊙ (S_in + Kaᵀ·D), one pass over the state. Four
                // vectors per k broadcast so the delta slab is loaded once per
                // four FMAs.
                {
                    constexpr int NJU = 4;
                    for (int64_t i = 0; i < Dk; ++i) {
                        const float* k_row = kat + i * C;
                        float* s_row = Sf + i * Dv;
                        const v_f32x8 lam = v_set1_f32x8(ep[(ct - 1) * Dk + i]);
                        int64_t j = j0;
                        for (; j + NJU * L <= j1; j += NJU * L) {
                            v_f32x8 acc[NJU];
                            for (int jj = 0; jj < NJU; ++jj) { acc[jj] = v_zero_f32x8(); }
                            for (int64_t s = 0; s < ct; ++s) {
                                const v_f32x8 b = v_set1_f32x8(k_row[s]);
                                for (int jj = 0; jj < NJU; ++jj) {
                                    acc[jj] = v_fmadd(
                                        b, v_load(dl + s * Dv + j + jj * L), acc[jj]);
                                }
                            }
                            for (int jj = 0; jj < NJU; ++jj) {
                                v_store(s_row + j + jj * L,
                                        v_mul(v_add(v_load(s_row + j + jj * L), acc[jj]),
                                              lam));
                            }
                        }
                        for (; j + L <= j1; j += L) {
                            v_f32x8 acc = v_zero_f32x8();
                            for (int64_t s = 0; s < ct; ++s) {
                                acc = v_fmadd(v_set1_f32x8(k_row[s]),
                                              v_load(dl + s * Dv + j), acc);
                            }
                            v_store(s_row + j, v_mul(v_add(v_load(s_row + j), acc), lam));
                        }
                        for (; j < j1; ++j) {
                            float acc = 0.0f;
                            for (int64_t s = 0; s < ct; ++s) { acc += k_row[s] * dl[s * Dv + j]; }
                            s_row[j] = (s_row[j] + acc) * ep[(ct - 1) * Dk + i];
                        }
                    }
                }
                t0 += ct;
            }
            return t0;
        };

        const int route = route_override();
        int64_t t_done = 0;
        if (route > 0 || (route == 0 && S >= kChunkMinSeq)) { t_done = chunk_route(); }
        for (int64_t t = t_done; t < S; ++t) { token_step(t); }

        for (int64_t i = 0; i < Dk; ++i) {
            T* srow = sp + b * ts0 + hv * ts1 + i * ts2;
            const float* drow = Sf + i * Dv;
            int64_t j = j0;
            for (; j + L <= j1; j += L) { vstore_f32(srow + j * ts3, v_load(drow + j)); }
            for (; j < j1; ++j) { s_store(srow + j * ts3, drow[j]); }
        }
    });
}

}  // anonymous namespace

Status linear_attention_cpu(const LinearAttentionAttributes& attrs,
                          std::span<TensorView> outputs,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx,
                          void* workspace)
{
    switch (inputs[0].data_type()) {
    case DataType::f32: linear_attention_impl<float>(attrs, outputs, inputs, ctx); return Status::ok;
    case DataType::f16: linear_attention_impl<half>(attrs, outputs, inputs, ctx);  return Status::ok;
#ifdef NNOPS_ARCH_AARCH64
    case DataType::bf16: linear_attention_impl<bfloat16>(attrs, outputs, inputs, ctx); return Status::ok;
#endif
    default: break;
    }
    return Status::unsupported;
}

}  // namespace nnops::backend::cpu
