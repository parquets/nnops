#pragma once
/// @file attention_decode.cuh
/// @brief Flash decoding — the single-query-row kernel, its split-KV form and
///        the merge pass that combines the partial softmaxes.
///
/// A decode step has one query row, so the tiled kernel would leave fifteen of
/// its sixteen row slots idle. Here the whole block works on that row, and it
/// reads K straight from global rather than staging it in shared.
///
/// Warp-local spans. Each warp owns its own interleaved slice of the block's
/// key span and runs the online softmax over it with no barrier at all: eight
/// keys a round, four lanes a key, and the value pass column-parallel over the
/// warp's own lanes. The block's only cross-warp step is the closing combine,
/// which folds `kWarps` partial states in shared memory. The earlier form ran
/// one block-wide round of sixty-four keys through shared P, and its recorded
/// cost is the reason for this one: a probe over the run length gave a slope of
/// 0.19 us a key against a 10 us launch intercept, and the value loop there
/// issued one dependent load per thread per key — about one load in flight per
/// warp, which leaves the part at ~170 GB/s whatever the cache hits. The
/// rewrite issues eight (K) and eight (V) per lane, vectorized, with every lane
/// of every warp busy.
///
/// One block per head *group*. The query heads that share a K/V head read the
/// same rows, so a block serves up to `kMaxHeadBatch` of them: the row reaches
/// `hb` dots before it is dropped, and the block's request count falls by `hb`.
/// That is the term the profile was left with — after the rewrite the GQA row
/// moves 719 GB/s of requests to an L2 that serves ~4x less of it — and the
/// measured sharing curve (H=8/Hkv=8 0.064 ms, H=32/Hkv=8 0.094, H=32/Hkv=1
/// 0.092 at the same run) is what says the redundancy, not the run, is the
/// remaining cost.
///
/// Split-KV. One block per (head, batch) is sixteen to thirty-two blocks on a
/// 38-SM part — a single wave whose per-block prologue and epilogue have
/// nothing to overlap. The run is cut into `splits` contiguous spans, one block
/// per (head, batch, split), each keeping its own online-softmax state:
///
///   m = max(max_s m_s, head sink)
///   l = Σ_s exp(m_s − m)  ·l_s + exp(sink − m)
///   o = (Σ_s exp(m_s − m)·o_s) / l · gate
///
/// A one-span call finishes in the kernel (the sink, the normalization and the
/// gate inline); more than one writes partials and a second block-per-(head,
/// batch) kernel merges them. The partials come from the GEMM's cudaMemPool
/// workspace (matmul/gemm_common.cuh), so a split decode allocates nothing.
///
/// The masks, additive bias, softcap, keep filter and int8 dequantization are
/// the ones attention_kernel.cuh applies.

#include "attention_common.cuh"
#include "detail/bench_knobs.hpp"

namespace nnops::backend::cuda::attn {

/// Lanes that share one key's dot, the keys one warp covers in a round, the
/// warps a block runs, and the query heads a block serves at most. Fixed
/// rather than chosen per shape: the register dot makes a warp's footprint
/// independent of the round width, and the head bound is what keeps the
/// per-lane accumulators (heads × output groups × four columns) in registers.
constexpr int kLanesPerKey = 4;
constexpr int kWarps = kThreads / 32;
constexpr int kKeysPerWarp = 32 / kLanesPerKey;
constexpr int kKeysPerRound = kWarps * kKeysPerWarp;
constexpr int kMaxHeadBatch = 4;
constexpr int kMaxDecodeSplits = 32;
/// The automatic split choice stops here: on the two bench rows the fine sweep
/// reads 0.083 ms (GQA) and 0.030 (MLA) at eight splits, and worse at ten,
/// twelve and sixteen, so an automatic choice past eight is only prologue.
constexpr int kMaxAutoDecodeSplits = 8;
/// Output columns one lane owns are four; a warp therefore covers 128 and the
/// bound below is what a block's per-lane accumulator array holds.
constexpr int kMaxOutGroups = 4;

/// The per-split partial state. `m`/`l` are one float per (batch·head, split),
/// `o` is Dv floats per (split, batch, head). All null when the call fits in
/// one span.
struct DecodeSplits {
    int splits = 1;
    float* m = nullptr;
    float* l = nullptr;
    float* o = nullptr;
};

/// The merge pass needs the store geometry and the two per-head extras, not the
/// K/V description.
template <class T> struct DecodeMergeParams {
    int64_t B = 0, H = 0, Dv = 0;
    int64_t o_b = 0, o_h = 0;
    int64_t g_b = 0, g_h = 0;
    const void* sink = nullptr;
    const T* gate = nullptr;
    T* o = nullptr;
    int gate_act = 0;
    bool has_gate = false;
};

inline int64_t split_scratch_bytes(int64_t rows, int splits, int64_t dv)
{
    return rows * splits * 2 * static_cast<int64_t>(sizeof(float)) +
           rows * splits * dv * static_cast<int64_t>(sizeof(float));
}

/// Shared-memory bytes the decode kernel needs: `hb` staged query rows, the
/// per-(warp, head) value partials and the same shape of softmax state.
inline int64_t decode_smem_bytes(int64_t dqk, int64_t dv, int hb)
{
    const int64_t qs_stride = (dqk + 3) & ~static_cast<int64_t>(3);
    return static_cast<int64_t>(sizeof(float)) *
           (hb * qs_stride + static_cast<int64_t>(kWarps) * hb * dv + 2 * kWarps * hb);
}

/// Query heads one block serves: the block's K/V reads are shared by all of
/// them, so this is the traffic divisor, and it has to divide `group` (every
/// head of a block must sit under the same K/V head) while leaving room for the
/// partials in shared memory.
inline int decode_head_batch(int64_t group, int64_t dv)
{
    const int forced = ::nnops::detail::cuda_attn_headbatch.load(std::memory_order_relaxed);
    int hb = static_cast<int>(std::min<int64_t>(group, kMaxHeadBatch));
    if (forced > 0) {
        hb = std::min(forced, static_cast<int>(std::min<int64_t>(group, kMaxHeadBatch)));
    }
    hb = std::max(hb, 1);
    const int64_t budget = 24 * 1024 / static_cast<int64_t>(sizeof(float)); // 24 KB
    // Both conditions on every step: a width that does not divide the group
    // would put two K/V heads in one block, and the budget step must not walk
    // off the divisors.
    while (hb > 1 &&
           ((group % hb) != 0 ||
            (static_cast<int64_t>(kWarps) * hb * dv + 2 * kWarps * hb) > budget)) {
        --hb;
    }
    return hb;
}

// ============================================================
// Four elements of a K/V row, whatever their stored width
// ============================================================

/// The 4-wide row load behind the vector path. A lane's slice of a row is four
/// elements: a 16-byte load for f32 and an 8-byte one for the 2-byte types,
/// both of which the host only selects after checking the strides and the row
/// bases (`decode_vec_ok`).
template <class T> __device__ __forceinline__ float4 load_kv4(const T* p)
{
    if constexpr (std::is_same_v<T, float>) {
        return v4_load(p);
    } else if constexpr (std::is_same_v<T, __half>) {
        const uint2 raw = *reinterpret_cast<const uint2*>(p);
        const __half2 lo = *reinterpret_cast<const __half2*>(&raw.x);
        const __half2 hi = *reinterpret_cast<const __half2*>(&raw.y);
        const float2 a = __half22float2(lo);
        const float2 b = __half22float2(hi);
        return make_float4(a.x, a.y, b.x, b.y);
    } else {
        const uint2 raw = *reinterpret_cast<const uint2*>(p);
        const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
        const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
        const float2 a = __bfloat1622float2(lo);
        const float2 b = __bfloat1622float2(hi);
        return make_float4(a.x, a.y, b.x, b.y);
    }
}

/// Whether a call may take the 4-wide loads: every row base has to be aligned
/// to the load's width and every sequence/head step has to keep it there. The
/// scalar path answers for the rest.
template <class P> inline bool decode_vec_ok(const P& p)
{
    if constexpr (P::kv8) {
        return false;
    } else {
        using KVT = typename P::KVT;
        constexpr int64_t units = 4; // elements per load, so strides in whole loads
        constexpr int64_t mask = static_cast<int64_t>(sizeof(KVT)) * 4 - 1;
        const auto ok = [](const void* ptr) {
            return (reinterpret_cast<uintptr_t>(ptr) & static_cast<uintptr_t>(mask)) == 0;
        };
        return p.k_s % units == 0 && p.k_h % units == 0 && p.k_b % units == 0 &&
               p.pk_s % units == 0 && p.pk_h % units == 0 && p.pk_b % units == 0 &&
               p.v_s % units == 0 && p.v_h % units == 0 && p.v_b % units == 0 &&
               p.pv_s % units == 0 && p.pv_h % units == 0 && p.pv_b % units == 0 &&
               (p.k == nullptr || ok(p.k)) && (p.v == nullptr || ok(p.v)) &&
               (p.pk == nullptr || ok(p.pk)) && (p.pv == nullptr || ok(p.pv));
    }
}

// ============================================================
// The kernel
// ============================================================

template <class T, bool KV8, bool VEC, int NQ>
__global__ void __launch_bounds__(kThreads) attention_decode_kernel(AttnParams<T, KV8> p,
                                                                  DecodeSplits s)
{
    using KVT = typename AttnParams<T, KV8>::KVT;
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane & (kLanesPerKey - 1); // this lane's quarter of a dot
    const int slot = lane / kLanesPerKey;    // key slot inside the warp
    const int64_t b = blockIdx.y;
    const int64_t sp = blockIdx.z;
    const int hb = p.head_batch;                  // query heads this block serves
    const int64_t h0 = blockIdx.x * hb;
    const int64_t hkv = h0 / p.group;

    const int qs_stride = (static_cast<int>(p.Dqk) + 3) & ~3; // whole 4-cell loads
    float* Qs = smem;                          // hb x qs_stride
    float* Wo = Qs + hb * qs_stride;           // kWarps x hb x Dv
    float* wm = Wo + kWarps * hb * p.Dv;       // 2 x hb x kWarps: maxima, sums

    int64_t run = p.Sk;
    if (p.run_ptr != nullptr) {
        int64_t v = p.run_ptr[b];
        if (p.run_plus_one) {
            ++v;
        }
        run = v <= 0 ? 0 : (v < p.Sk ? v : p.Sk);
    }
    const int64_t pos0 = p.pos_ids != nullptr ? p.pos_ids[b] : p.position_offset;
    const int64_t q0 = pos0 >= 0 ? pos0 : (run - 1); // Sq == 1
    const int64_t sink_n = (p.sink_tokens > 0 && p.window > 0)
                               ? (p.sink_tokens < run ? p.sink_tokens : run)
                               : 0;

    // The single query row is row 0 of the call; its absolute position fixes
    // the causal/window/sink bounds.
    const int64_t hi = p.causal ? std::max<int64_t>(0, std::min<int64_t>(run, q0 + 1)) : run;
    const int64_t lo = (p.window > 0 && p.causal) ? std::max<int64_t>(0, q0 - p.window + 1) : 0;

    // This block's span of the visible run. A batch whose run ends before the
    // span contributes nothing (its m stays -FLT_MAX and the merge weights it
    // out), so a run shorter than the capacity costs a launch, not an answer.
    const int64_t span = (hi + s.splits - 1) / s.splits;
    const int64_t j_lo = sp * span;
    const int64_t j_end = std::min(hi, j_lo + span);

    // This lane's quarter of the head width, rounded to whole 4-cell loads.
    const int dq = ((static_cast<int>(p.Dqk) + kLanesPerKey - 1) / kLanesPerKey + 3) & ~3;
    const int d0 = std::min(g * dq, static_cast<int>(p.Dqk));
    const int d1 = std::min(d0 + dq, static_cast<int>(p.Dqk));

    for (int hh = 0; hh < hb; ++hh) {
        for (int64_t d = tid; d < p.Dqk; d += kThreads) {
            Qs[hh * qs_stride + d] = s_load(p.q + b * p.q_b + (h0 + hh) * p.q_h + d);
        }
    }
    __syncthreads();

    constexpr int nq = NQ; // four output columns a lane
    const int dv = static_cast<int>(p.Dv);
    float o[kMaxHeadBatch][kMaxOutGroups][4];
#pragma unroll
    for (int hh = 0; hh < kMaxHeadBatch; ++hh) {
#pragma unroll
        for (int i = 0; i < kMaxOutGroups; ++i) {
            o[hh][i][0] = 0.0f;
            o[hh][i][1] = 0.0f;
            o[hh][i][2] = 0.0f;
            o[hh][i][3] = 0.0f;
        }
    }
    float m_w[kMaxHeadBatch];
    float l_w[kMaxHeadBatch];
    float pv[kMaxHeadBatch];
#pragma unroll
    for (int hh = 0; hh < kMaxHeadBatch; ++hh) {
        m_w[hh] = kNegInf;
        l_w[hh] = 0.0f;
        pv[hh] = 0.0f;
    }

    // ---- this warp's slice of the span, one round of eight keys at a time ----
    for (int64_t jb = j_lo + warp * kKeysPerWarp; jb < j_end; jb += kWarps * kKeysPerWarp) {
        const int64_t j = jb + slot;

        // One key per lane group, read straight from global and dotted against
        // every head of the block. A lane reads its quarter as 4-wide loads, so
        // the round leaves eight of them in flight.
        float a0[kMaxHeadBatch];
        float a1[kMaxHeadBatch];
        float a2[kMaxHeadBatch];
        float a3[kMaxHeadBatch];
#pragma unroll
        for (int hh = 0; hh < kMaxHeadBatch; ++hh) {
            a0[hh] = 0.0f;
            a1[hh] = 0.0f;
            a2[hh] = 0.0f;
            a3[hh] = 0.0f;
        }
        if (j < j_end) {
            const KVT* row = k_row(p, b, hkv, j);
            if (row != nullptr) {
                int d = d0;
                if constexpr (VEC) {
                    for (; d + 4 <= d1; d += 4) {
                        const float4 k4 = load_kv4<KVT>(row + d);
#pragma unroll
                        for (int hh = 0; hh < kMaxHeadBatch; ++hh) {
                            if (hh >= hb) {
                                continue;
                            }
                            const float4 q4 =
                                *reinterpret_cast<const float4*>(Qs + hh * qs_stride + d);
                            a0[hh] += q4.x * k4.x;
                            a1[hh] += q4.y * k4.y;
                            a2[hh] += q4.z * k4.z;
                            a3[hh] += q4.w * k4.w;
                        }
                    }
                }
                for (; d < d1; ++d) {
                    const float ke = kv_elem<AttnParams<T, KV8>>(row + d);
#pragma unroll
                    for (int hh = 0; hh < kMaxHeadBatch; ++hh) {
                        if (hh >= hb) {
                            continue;
                        }
                        a0[hh] += Qs[hh * qs_stride + d] * ke;
                    }
                }
            }
        }

        // Fold the group, adjust each head's score, and run the round's online
        // softmax per head. Every lane runs the shuffles, so a partial round is
        // still a legal warp-wide exchange.
        float sc[kMaxHeadBatch];
#pragma unroll
        for (int hh = 0; hh < kMaxHeadBatch; ++hh) {
            sc[hh] = kNegInf;
            if (hh >= hb) {
                continue;
            }
            float acc = ((a0[hh] + a1[hh]) + (a2[hh] + a3[hh]));
            acc += __shfl_down_sync(0xffffffffu, acc, 1);
            acc += __shfl_down_sync(0xffffffffu, acc, 2);
            // The group's lane 0 owns the key's score from here on.
            if (j < j_end && g == 0) {
                const int64_t h = h0 + hh;
                float sv = acc * p.scale;
                if constexpr (KV8) {
                    sv *= kv_scale_at(p.kscale, p.scale_per_token, b, hkv, j, p.Hkv, p.Sk);
                }
                if (p.softcap_on) {
                    sv = p.softcap * tanhf(sv / p.softcap);
                }
                if (j >= lo || j < sink_n) {
                    if (p.has_keep) {
                        const int64_t koff = b * p.keep_b + j; // query row 0
                        if (p.keep[koff] == 0) {
                            sv = p.filter;
                        }
                    }
                    if (p.has_bias) {
                        sv += s_load(p.bias + b * p.bias_b + h * p.bias_h + j);
                    }
                    sc[hh] = sv;
                }
            }
            sc[hh] = __shfl_sync(0xffffffffu, sc[hh], lane & ~(kLanesPerKey - 1));
        }

#pragma unroll
        for (int hh = 0; hh < kMaxHeadBatch; ++hh) {
            if (hh >= hb) {
                continue;
            }
            float mt = sc[hh];
            mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 4));
            mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 8));
            mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 16));

            const float m_new = fmaxf(m_w[hh], mt);
            const float rs = (m_w[hh] == kNegInf) ? 0.0f : __expf(m_w[hh] - m_new);
            const float p_own = (m_new != kNegInf) ? __expf(sc[hh] - m_new) : 0.0f;
            float sum = p_own;
            sum += __shfl_xor_sync(0xffffffffu, sum, 4);
            sum += __shfl_xor_sync(0xffffffffu, sum, 8);
            sum += __shfl_xor_sync(0xffffffffu, sum, 16);
            m_w[hh] = m_new;
            l_w[hh] = l_w[hh] * rs + sum;
            pv[hh] = p_own;
#pragma unroll
            for (int i = 0; i < kMaxOutGroups; ++i) {
                o[hh][i][0] *= rs;
                o[hh][i][1] *= rs;
                o[hh][i][2] *= rs;
                o[hh][i][3] *= rs;
            }
        }

        // ---- P·V: four output columns a lane, the round's V rows read 4-wide ----
        if constexpr (VEC) {
            if (jb + kKeysPerWarp <= j_end) {
                const KVT* vp[kKeysPerWarp];
#pragma unroll
                for (int t = 0; t < kKeysPerWarp; ++t) {
                    vp[t] = v_row(p, b, hkv, jb + t);
                }
#pragma unroll
                for (int i = 0; i < NQ; ++i) {
                    const int d = 4 * lane + 128 * i;
                    const bool wide = (d + 4 <= dv);
                    const bool tail = (!wide && d < dv);
                    float4 a[kMaxHeadBatch];
#pragma unroll
                    for (int hh = 0; hh < kMaxHeadBatch; ++hh) {
                        a[hh] = make_float4(o[hh][i][0], o[hh][i][1], o[hh][i][2], o[hh][i][3]);
                    }
                    // The probability broadcasts stay outside the column test:
                    // a full-mask shuffle inside a lane-divergent branch hangs
                    // the warp whenever Dv does not cover all 32 lanes.
#pragma unroll
                    for (int t = 0; t < kKeysPerWarp; ++t) {
                        const int src = t * kLanesPerKey;
                        float pt[kMaxHeadBatch];
#pragma unroll
                        for (int hh = 0; hh < kMaxHeadBatch; ++hh) {
                            pt[hh] = __shfl_sync(0xffffffffu, pv[hh], src);
                        }
                        if (wide) {
                            const float4 v4 = load_kv4<KVT>(vp[t] + d);
#pragma unroll
                            for (int hh = 0; hh < kMaxHeadBatch; ++hh) {
                                if (hh < hb) {
                                    a[hh].x += pt[hh] * v4.x;
                                    a[hh].y += pt[hh] * v4.y;
                                    a[hh].z += pt[hh] * v4.z;
                                    a[hh].w += pt[hh] * v4.w;
                                }
                            }
                        } else if (tail) {
                            for (int e = 0; d + e < dv && e < 4; ++e) {
                                const float ve = kv_elem<AttnParams<T, KV8>>(vp[t] + d + e);
#pragma unroll
                                for (int hh = 0; hh < kMaxHeadBatch; ++hh) {
                                    if (hh < hb) {
                                        o[hh][i][e] += pt[hh] * ve;
                                    }
                                }
                            }
                        }
                    }
#pragma unroll
                    for (int hh = 0; hh < kMaxHeadBatch; ++hh) {
                        if (!wide) {
                            continue;
                        }
                        o[hh][i][0] = a[hh].x;
                        o[hh][i][1] = a[hh].y;
                        o[hh][i][2] = a[hh].z;
                        o[hh][i][3] = a[hh].w;
                    }
                }
                continue;
            }
        }
        // The scalar path: a partial round, or a call whose rows are not
        // aligned for a 4-wide load.
        for (int t = 0; t < kKeysPerWarp; ++t) {
            const int64_t jj = jb + t;
            float pt[kMaxHeadBatch];
            const int src = t * kLanesPerKey;
#pragma unroll
            for (int hh = 0; hh < kMaxHeadBatch; ++hh) {
                pt[hh] = __shfl_sync(0xffffffffu, pv[hh], src);
            }
            if (jj >= j_end) {
                continue;
            }
            const KVT* vrow = v_row(p, b, hkv, jj);
            if (vrow == nullptr) {
                continue;
            }
            if constexpr (KV8) {
                const float vs = kv_scale_at(p.vscale, p.vscale_per_token, b, hkv, jj, p.Hkv, p.Sk);
#pragma unroll
                for (int hh = 0; hh < kMaxHeadBatch; ++hh) {
                    pt[hh] *= vs;
                }
            }
#pragma unroll
            for (int i = 0; i < NQ; ++i) {
                const int d = 4 * lane + 128 * i;
                for (int e = 0; d + e < dv && e < 4; ++e) {
                    const float ve = kv_elem<AttnParams<T, KV8>>(vrow + d + e);
#pragma unroll
                    for (int hh = 0; hh < kMaxHeadBatch; ++hh) {
                        if (hh < hb) {
                            o[hh][i][e] += pt[hh] * ve;
                        }
                    }
                }
            }
        }
    }

    // ---- publish this warp's partials, then fold the block's ----
#pragma unroll
    for (int hh = 0; hh < kMaxHeadBatch; ++hh) {
        if (hh >= hb) {
            continue;
        }
#pragma unroll
        for (int i = 0; i < NQ; ++i) {
            const int d = 4 * lane + 128 * i;
            for (int e = 0; d + e < dv && e < 4; ++e) {
                Wo[(warp * hb + hh) * p.Dv + d + e] = o[hh][i][e];
            }
        }
        if (lane == 0) {
            wm[hh * kWarps + warp] = m_w[hh];
            wm[hb * kWarps + hh * kWarps + warp] = l_w[hh];
        }
    }
    __syncthreads();

    const bool single = (s.splits == 1);
#pragma unroll
    for (int hh = 0; hh < kMaxHeadBatch; ++hh) {
        if (hh >= hb) {
            continue;
        }
        float mg = kNegInf;
#pragma unroll
        for (int u = 0; u < kWarps; ++u) {
            mg = fmaxf(mg, wm[hh * kWarps + u]);
        }
        if (single && p.sink != nullptr) {
            mg = fmaxf(mg, s_load(static_cast<const T*>(p.sink) + h0 + hh));
        }
        float w[kWarps];
        float lsum = 0.0f;
#pragma unroll
        for (int u = 0; u < kWarps; ++u) {
            w[u] = (mg == kNegInf) ? 0.0f : __expf(wm[hh * kWarps + u] - mg);
            lsum += w[u] * wm[hb * kWarps + hh * kWarps + u];
        }
        if (single && p.sink != nullptr && mg != kNegInf) {
            lsum += __expf(s_load(static_cast<const T*>(p.sink) + h0 + hh) - mg);
        }

        const int64_t h = h0 + hh;
        T* orow = p.o + b * p.o_b + h * p.o_h; // query row 0
        const T* grow = p.has_gate ? p.gate + b * p.g_b + h * p.g_h : nullptr;

        if (single) {
            // ---- one span: head sink, normalization and gate close the call ----
            const float norm = lsum > 0.0f ? (1.0f / lsum) : 0.0f;
#pragma unroll
            for (int i = 0; i < NQ; ++i) {
                const int d = 4 * lane + 128 * i;
                for (int e = 0; d + e < dv && e < 4; ++e) {
                    float val = 0.0f;
#pragma unroll
                    for (int u = 0; u < kWarps; ++u) {
                        val += w[u] * Wo[(u * hb + hh) * p.Dv + d + e];
                    }
                    val *= norm;
                    if (grow != nullptr) {
                        val *= gate_act_value(p.gate_act, s_load(grow + d + e));
                    }
                    s_store(orow + d + e, val);
                }
            }
            continue;
        }

        // ---- the partial state, in this split's own softmax scale ----
        if (tid == 0) {
            const int64_t idx = (b * p.H + h) * s.splits + sp;
            s.m[idx] = mg;
            s.l[idx] = lsum;
        }
        float* op = s.o + ((sp * p.B + b) * p.H + h) * p.Dv;
#pragma unroll
        for (int i = 0; i < NQ; ++i) {
            const int d = 4 * lane + 128 * i;
            for (int e = 0; d + e < dv && e < 4; ++e) {
                float val = 0.0f;
#pragma unroll
                for (int u = 0; u < kWarps; ++u) {
                    val += w[u] * Wo[(u * hb + hh) * p.Dv + d + e];
                }
                op[d + e] = val;
            }
        }
    }
}

// ============================================================
// The merge pass: one block per (head, batch)
// ============================================================

template <class T>
__global__ void __launch_bounds__(kThreads) attention_decode_merge_kernel(
    DecodeMergeParams<T> p, DecodeSplits s)
{
    extern __shared__ float smem[];
    float* w = smem; // kMaxDecodeSplits weights
    const int tid = threadIdx.x;
    const int64_t b = blockIdx.y;
    const int64_t h = blockIdx.x;
    const int ns = s.splits;
    const int64_t base = (b * p.H + h) * ns;

    if (tid < ns) {
        w[tid] = s.m[base + tid];
    }
    __syncthreads();

    // The global max: every thread recomputes it over the few splits rather
    // than round-tripping through shared.
    float mg = kNegInf;
    for (int i = 0; i < ns; ++i) {
        mg = fmaxf(mg, w[i]);
    }
    if (p.sink != nullptr) {
        mg = fmaxf(mg, s_load(static_cast<const T*>(p.sink) + h));
    }
    __syncthreads(); // every thread has read the raw maxima

    if (tid < ns) {
        w[tid] = (mg == kNegInf) ? 0.0f : __expf(w[tid] - mg);
    }
    __syncthreads();

    float l = 0.0f;
    for (int i = 0; i < ns; ++i) {
        l += w[i] * s.l[base + i];
    }
    if (p.sink != nullptr && mg != kNegInf) {
        l += __expf(s_load(static_cast<const T*>(p.sink) + h) - mg);
    }
    const float inv = l > 0.0f ? (1.0f / l) : 0.0f;

    T* orow = p.o + b * p.o_b + h * p.o_h;
    const T* grow = p.has_gate ? p.gate + b * p.g_b + h * p.g_h : nullptr;
    for (int64_t d = tid; d < p.Dv; d += kThreads) {
        float acc = 0.0f;
        for (int i = 0; i < ns; ++i) {
            acc += w[i] * s.o[((static_cast<int64_t>(i) * p.B + b) * p.H + h) * p.Dv + d];
        }
        float val = acc * inv;
        if (grow != nullptr) {
            val *= gate_act_value(p.gate_act, s_load(grow + d));
        }
        s_store(orow + d, val);
    }
}

} // namespace nnops::backend::cuda::attn
