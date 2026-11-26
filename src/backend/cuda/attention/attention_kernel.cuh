#pragma once
/// @file attention_kernel.cuh
/// @brief The tiled online-softmax kernel: a query block of kBr rows streamed
///        against KV tiles, one block per (batch, query head, query block).
///
/// FlashAttention-1's loop — stream KV in tiles, keep the running max and sum,
/// never materialize the score matrix — in plain float FMA rather than tensor
/// cores, so it covers every dtype and every head width the operators accept.
///
/// Geometry. The block's query head selects its KV head as h / (H / H_kv), so
/// GQA and MQA read one K/V run for the whole group. A thread is (row, lane
/// group):
///
///   Q·Kᵀ      thread (r, g) dots Q row r against keys g, g+16 of the tile;
///             the D-long dot is a private FMA loop, so no shuffle reduction.
///   softmax   warp w owns rows 2w, 2w+1; its 32 lanes carry the tile's keys,
///             so the row max and sum are one warp reduction each.
///   P·V       thread (r, g) owns row r and the Dv/16 columns g*Cw..; the
///             probability tile is read back from shared, so P needs no
///             cross-lane exchange.
///
/// The Dv/16-column ownership caps a thread at 32 accumulators, which is why a
/// value width above 512 is refused (a decode-shaped MLA run at DeepSeek's 512
/// is the largest the operators describe).
///
/// K/V tiles live in shared memory as float whatever the cache dtype, so the
/// score loop is the same for an fp cache and an s8 one; an s8 cache carries
/// its per-token scale into the score (K) or into the probability (V), never a
/// widened copy of the run. Q is staged once per block, and the Q/K/score row
/// strides are padded odd so a warp's sixteen rows hit distinct banks.
///
/// The one-row decode shape gets its own kernel (attention_decode.cuh) because
/// this tiling leaves fifteen of the sixteen row slots idle there.

#include "attention_common.cuh"

namespace nnops::backend::cuda::attn {

// ============================================================
// The kernel
// ============================================================

template <class T, bool KV8>
__global__ void __launch_bounds__(kThreads) attention_kernel(AttnParams<T, KV8> p)
{
    extern __shared__ float smem[];
    const int bcw = p.bc;
    const int tid = threadIdx.x;
    const int r = tid & (kBr - 1); // query row inside the block
    const int g = tid >> 4;        // lane group: keys g, g+16 / value columns g*Cw..

    const int64_t b = blockIdx.z;
    const int64_t h = blockIdx.y;
    const int64_t hkv = h / p.group;
    const int64_t i0 = static_cast<int64_t>(blockIdx.x) * kBr;
    const int rows = static_cast<int>(std::min<int64_t>(kBr, p.Sq - i0));

    // The batch's run and where its query rows sit (vector > scalar >
    // bottom-right). The run bounds both the causal diagonal and the visible
    // prefix.
    int64_t run = p.Sk;
    if (p.run_ptr != nullptr) {
        int64_t v = p.run_ptr[b];
        if (p.run_plus_one) {
            ++v;
        }
        run = v <= 0 ? 0 : (v < p.Sk ? v : p.Sk);
    }
    const int64_t pos0 = p.pos_ids != nullptr ? p.pos_ids[b] : p.position_offset;
    const int64_t q0 = pos0 >= 0 ? pos0 : (run - p.Sq);
    const int64_t sink_n = (p.sink_tokens > 0 && p.window > 0)
                               ? (p.sink_tokens < run ? p.sink_tokens : run)
                               : 0;

    const int cw = static_cast<int>((p.Dv + 15) >> 4); // value columns per thread

    // Padded row strides for the two shared tiles a warp walks by *row*: the
    // Q/K dot and the softmax/P·V score row. A head width that is a multiple of
    // 32 maps every row to the same bank, so sixteen rows in a warp serialize;
    // an odd stride scatters them. (LeetCUDA's kPad is the same fix.)
    const int64_t kstride = pad_stride(p.Dqk);
    const int64_t sstride = static_cast<int64_t>(bcw) + 1;

    float* Qs = smem;                                   // kBr × kstride
    float* Ks = Qs + static_cast<int64_t>(kBr) * kstride; // bcw × kstride
    float* Vs = Ks + static_cast<int64_t>(bcw) * kstride; // bcw × Dv
    float* Ss = Vs + static_cast<int64_t>(bcw) * p.Dv;   // kBr × sstride
    float* rowM = Ss + static_cast<int64_t>(kBr) * sstride;
    float* rowL = rowM + kBr;
    float* rowScale = rowL + kBr;

    if (tid < kBr) {
        rowM[tid] = kNegInf;
        rowL[tid] = 0.0f;
        rowScale[tid] = 1.0f;
    }

    // Stage the query block once; every KV tile reads it back from shared with
    // the padded stride instead of re-reading it from global.
    for (int64_t idx = tid; idx < static_cast<int64_t>(kBr) * kstride; idx += kThreads) {
        const int rr = static_cast<int>(idx / kstride);
        const int d = static_cast<int>(idx - static_cast<int64_t>(rr) * kstride);
        const bool live = rr < rows && d < p.Dqk;
        Qs[idx] = live ? s_load(p.q + b * p.q_b + h * p.q_h + (i0 + rr) * p.q_s + d) : 0.0f;
    }

    // Value accumulator: this thread's row, its cw columns from group g.
    float o[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        o[j] = 0.0f;
    }
    __syncthreads();

    const int64_t kv_end = p.causal
                               ? std::max<int64_t>(0, std::min<int64_t>(run, q0 + i0 + rows))
                               : run;

    for (int64_t j0 = 0; j0 < kv_end; j0 += bcw) {
        __syncthreads(); // the previous P·V has finished reading Ks/Vs/Ss

        // ---- stage K and V ----
        const int64_t ktot = static_cast<int64_t>(bcw) * p.Dqk;
        for (int64_t idx = tid; idx < ktot; idx += kThreads) {
            const int c = static_cast<int>(idx / p.Dqk);
            const int d = static_cast<int>(idx - static_cast<int64_t>(c) * p.Dqk);
            Ks[static_cast<int64_t>(c) * kstride + d] = read_k(p, b, hkv, j0 + c, d);
        }
        const int64_t vtot = static_cast<int64_t>(bcw) * p.Dv;
        for (int64_t idx = tid; idx < vtot; idx += kThreads) {
            const int c = static_cast<int>(idx / p.Dv);
            const int d = static_cast<int>(idx - static_cast<int64_t>(c) * p.Dv);
            Vs[idx] = read_v(p, b, hkv, j0 + c, d);
        }
        __syncthreads();

        // ---- Q·Kᵀ ----
        if (r < rows && g < bcw) {
            const int64_t qi = i0 + r;
            const int64_t hi =
                p.causal ? std::max<int64_t>(0, std::min<int64_t>(run, q0 + qi + 1)) : run;
            const int64_t lo =
                (p.window > 0 && p.causal)
                    ? std::max<int64_t>(0, q0 + qi - p.window + 1)
                    : 0;
            const float* qrow = Qs + static_cast<int64_t>(r) * kstride;

            const int c0 = g;
            const bool two = (g + 16) < bcw;
            const int c1 = g + 16;
            const float* k0 = Ks + static_cast<int64_t>(c0) * kstride;
            const float* k1 = Ks + static_cast<int64_t>(two ? c1 : c0) * kstride;
            // Four independent partial sums per key. The dot is a serial FMA
            // chain otherwise, and its latency — not the FMA throughput — is
            // what bounds the block: with two accumulators an in-flight dot
            // leaves the fp32 pipe mostly idle. Eight chains (two keys) hide it
            // at the tile widths this kernel uses.
            float a0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float a1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            int64_t d = 0;
            for (; d + 4 <= p.Dqk; d += 4) {
                const float q0 = s_load(qrow + d);
                const float q1 = s_load(qrow + d + 1);
                const float q2 = s_load(qrow + d + 2);
                const float q3 = s_load(qrow + d + 3);
                a0[0] += q0 * k0[d];
                a0[1] += q1 * k0[d + 1];
                a0[2] += q2 * k0[d + 2];
                a0[3] += q3 * k0[d + 3];
                if (two) {
                    a1[0] += q0 * k1[d];
                    a1[1] += q1 * k1[d + 1];
                    a1[2] += q2 * k1[d + 2];
                    a1[3] += q3 * k1[d + 3];
                }
            }
            float s0 = (a0[0] + a0[1]) + (a0[2] + a0[3]);
            float s1 = (a1[0] + a1[1]) + (a1[2] + a1[3]);
            for (; d < p.Dqk; ++d) {
                const float qv = s_load(qrow + d);
                s0 += qv * k0[d];
                if (two) {
                    s1 += qv * k1[d];
                }
            }
            const auto finish_score = [&](int c, float acc) -> float {
                const int64_t j = j0 + c;
                float s = acc * p.scale;
                if constexpr (KV8) {
                    s *= kv_scale_at(p.kscale, p.scale_per_token, b, hkv, j, p.Hkv, p.Sk);
                }
                if (p.softcap_on) {
                    s = p.softcap * tanhf(s / p.softcap);
                }
                const bool vis = (j < hi) && (j >= lo || j < sink_n);
                if (!vis) {
                    return kNegInf;
                }
                if (p.has_keep) {
                    const int64_t koff = b * p.keep_b + qi * p.keep_row + j;
                    if (p.keep[koff] == 0) {
                        s = p.filter;
                    }
                }
                if (p.has_bias) {
                    s += s_load(p.bias + b * p.bias_b + h * p.bias_h + qi * p.bias_row + j);
                }
                return s;
            };
            Ss[static_cast<int64_t>(r) * sstride + c0] = finish_score(c0, s0);
            if (two) {
                Ss[static_cast<int64_t>(r) * sstride + c1] = finish_score(c1, s1);
            }
        }
        __syncthreads();

        // ---- softmax over the tile ----
        // Warp w owns rows 2w and 2w+1; lanes carry the tile's keys.
        const int warp = tid >> 5;
        const int lane = tid & 31;
        if (warp < kBr / 2) {
            for (int rr = warp * 2; rr < warp * 2 + 2; ++rr) {
                if (rr >= rows) {
                    continue;
                }
                const float s = (lane < bcw) ? Ss[static_cast<int64_t>(rr) * sstride + lane] : kNegInf;
                // warp_reduce is a shfl-down tree: only lane 0 holds the total.
                // Every lane needs the row max to build its own probability.
                const float mt = __shfl_sync(0xffffffffu, warp_reduce_max(s), 0);
                const float m_old = rowM[rr];
                const float m_new = fmaxf(m_old, mt);
                const float scale_old = (m_old == kNegInf) ? 0.0f : __expf(m_old - m_new);
                float pv = 0.0f;
                if (m_new != kNegInf && lane < bcw) {
                    pv = __expf(s - m_new);
                }
                const float sum = warp_reduce_sum(pv);
                if (lane == 0) {
                    rowM[rr] = m_new;
                    rowL[rr] = rowL[rr] * scale_old + sum;
                    rowScale[rr] = scale_old;
                }
                if (lane < bcw) {
                    Ss[static_cast<int64_t>(rr) * sstride + lane] = pv;
                }
            }
        }
        __syncthreads();

        // ---- P·V ----
        {
            const float rs = rowScale[r];
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                if (j < cw) {
                    o[j] *= rs;
                }
            }
            if (r < rows) {
                for (int c = 0; c < bcw; ++c) {
                    const int64_t j = j0 + c;
                    float pv = Ss[static_cast<int64_t>(r) * sstride + c];
                    if (pv == 0.0f) {
                        continue;
                    }
                    if constexpr (KV8) {
                        pv *= kv_scale_at(p.vscale, p.scale_per_token, b, hkv, j, p.Hkv, p.Sk);
                    }
                    const float* vrow = Vs + static_cast<int64_t>(c) * p.Dv;
                    const int col0 = g * cw;
                    for (int jj = 0; jj < cw; ++jj) {
                        const int col = col0 + jj;
                        if (col < p.Dv) {
                            o[jj] += pv * vrow[col];
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

    // ---- head sink and the final normalization ----
    if (tid < kBr) {
        const int rr = tid;
        if (rr < rows) {
            if (p.sink != nullptr) {
                const float m_old = rowM[rr];
                const float sinkv = p.sink[h];
                const float m_new = fmaxf(m_old, sinkv);
                const float sc = (m_old == kNegInf) ? 0.0f : __expf(m_old - m_new);
                rowL[rr] = rowL[rr] * sc + __expf(sinkv - m_new);
                rowScale[rr] = sc;
                rowM[rr] = m_new;
            } else {
                rowScale[rr] = 1.0f;
            }
        }
    }
    __syncthreads();

    // ---- store, with the output gate folded in ----
    if (r < rows) {
        const float l = rowL[r];
        const float norm = l > 0.0f ? (1.0f / l) : 0.0f;
        const float rs = rowScale[r];
        const int64_t qi = i0 + r;
        T* orow = p.o + b * p.o_b + h * p.o_h + qi * p.o_s;
        const T* grow = p.has_gate ? p.gate + b * p.g_b + h * p.g_h + qi * p.g_s : nullptr;
        const int col0 = g * cw;
        for (int jj = 0; jj < cw; ++jj) {
            const int col = col0 + jj;
            if (col >= static_cast<int>(p.Dv)) {
                break;
            }
            float val = o[jj] * rs * norm;
            if (grow != nullptr) {
                val *= gate_act_value(p.gate_act, s_load(grow + col));
            }
            s_store(orow + col, val);
        }
    }
}

} // namespace nnops::backend::cuda::attn
