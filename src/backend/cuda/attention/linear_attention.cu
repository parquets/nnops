/// @file linear_attention.cu
/// @brief CUDA LinearAttention — the gated delta rule with the state in
///        registers.
///
/// The recurrence is sequential over the token axis (see
/// include/nnops/ops/linear_attention.hpp for the five steps and
/// backend/cpu/attention/linear_attention.cpp for the reference form), so the
/// kernel parallelises the only two axes that are independent — the
/// (batch, value head) state matrices and the value columns inside one — and
/// runs the whole sequence in a single launch. Per token and per value head,
/// with g = exp(Gate[t]):
///
///   r[j] = k·(g ⊙ S)                  pass A: one state read
///   d[j] = beta ⊙ (v[j] - r[j])
///   S    = g ⊙ S + k ⊗ d              pass B: the update ...
///   o[j] = scale · q·S                ... and the readout from the updated S
///
/// The last line is the CPU's `o0 + (q·k)·d` reassociated: q·(S' + k⊗d) is the
/// same number, and folding the readout into the update pass removes the q·k
/// dot product — and with it the only per-token block-wide reduction — at the
/// cost of one f32 rounding difference from the CPU's association.
///
/// Layout: one block owns a value-column slab of one (batch, value head).
/// Inside the block, threads form `rg` row groups × `cj` columns; thread (r, j)
/// owns the state rows i = 4·rg·gi + 4·r + v, i.e. four contiguous rows per
/// float4 group, 16 rows — 16 f32 registers — in total, for `kLaCols` adjacent
/// value columns. The whole state slab therefore lives in the register file for
/// the entire sequence instead of being re-read from shared or global memory
/// three times per token (a Dk×Dv state read once per token is the traffic the
/// CPU kernel pays too; chunking is what removes it there, and registers remove
/// it here).
///
/// Everything the token loop reads per state row — the gate-folded key, the
/// decay, the raw key and the query — is staged once per token into shared
/// memory by the whole block and then read back as float4 by the column
/// groups, which is the LeetCUDA ladder applied to this shape: one coalesced
/// global read per element for the fill, 128-bit shared reads in the inner
/// loop, and a shuffle butterfly for the cross-lane sums.
///
/// Three properties of that inner loop are deliberate:
///   - float4 alignment. A thread's rows are four *contiguous* rows per group,
///     so each shared read is one 16 B instruction. The alternative — rows
///     interleaved by rg — keeps the same total but pays four scalar LDS for
///     it.
///   - bank behaviour. At a fixed group the rg threads read 16 B blocks at
///     addresses 4·r, so the eight groups of a warp cover banks 0..31 exactly
///     once while the lanes sharing a row group hit the same address (a
///     broadcast). Padding a row, the other standard answer, is not needed.
///   - two columns a thread (the thread tile). Three shared reads feed three
///     FMAs on four rows, so the passes are shared-memory limited; a second
///     column spends the same reads on twice the state and halves the read
///     traffic per result. Measured on the prefill shapes at 1.23 vs 1.04
///     TFLOPS (f32) and 1.11 vs 1.05 (f16) against one column.
///
/// The reduction over the row groups of a column is one XOR butterfly per
/// column: all lanes of a group end with the total, so `d` is computed by all
/// of them from the same operands and no leader broadcast is needed.
///
/// Reach: the state's own width decides the block, not the model — Dk ≤ 512 (rg
/// ≤ 32 at 16 rows a thread) and Gqq ≤ 8. One query head sharing a state is the
/// Qwen3-Next / GDN case and takes its own instantiation; the generic one
/// predicates the other seven.
///
/// The convolution, when weights are supplied, is the CPU's causal depthwise
/// conv1d over the packed Q | K | V channels, applied while staging. The four
/// taps a gated delta net carries are unrolled: with the tap count behind a
/// kernel argument, the rolled loop turns them into four dependent global round
/// trips per element, and the staging barrier waits for the last one (worth 20%
/// of a prefill). f32, f16 and bf16 ride the same code (the 16-bit forms widen
/// at the I/O boundary).
///
/// The stage runs one token ahead of the passes. The next token's reads are
/// issued into registers before this token's passes and consumed after them, so
/// the global round trip lands under the arithmetic instead of in front of it;
/// the raw rows the conv folds go through a per-thread ring in shared memory,
/// because four raw taps a thread held live across the passes cost more in
/// spilling the passes than the pipeline saves. That is worth 1.17x on the
/// prefill and 1.3x on the small-head shapes, where a single block a
/// multiprocessor has nothing to hide the round trip with (measured).

#include "backend/cuda/cuda_common.cuh"
#include "nnops/ops/linear_attention.hpp"

#include "linear_attention_wy_host.hpp"

#include <cmath>
#include <span>
#include <type_traits>

namespace nnops::backend::cuda {
namespace {

/// Threads per block. Split into rg row groups and kLaThreads / rg columns, so
/// rg must divide it — every rg the launcher picks is a power of two ≤ 32.
constexpr int kLaThreads = 128;
/// float4 groups of state rows a thread owns. Four groups x four rows is the
/// 16-row slice that bounds Dk to 16·rg.
constexpr int kLaGroups = 4;
constexpr int kLaRows = kLaGroups * 4;
/// Value columns one thread owns. The token passes read three shared float4s
/// per four state rows and do three FMAs on each, so the loop is shared-memory
/// bound; a second column spends the same reads on twice the state and halves
/// the read traffic per result.
constexpr int kLaCols = 2;
/// Stage slots one thread fills. The widest row block is 32 row groups x 16
/// rows (Dk = 512), and the widest column slab is kLaCols blocks of threads.
/// Both bounds are checked by the launcher, and both arrays are unrolled with
/// constant indices so they stay in registers.
constexpr int kLaSlots = (16 * kCudaWarpSize) / kLaThreads;
constexpr int kLaVSlots = kLaCols;
/// Blocks per SM to ask ptxas for. Left to itself ptxas spends the register
/// budget on ILP; pinning the residency caps the count at 512/kLaMinBlocks and
/// buys the token loop more warps to cover its staging round trip with.
constexpr int kLaMinBlocks = 4;

struct LaParams {
    const void* q = nullptr;
    const void* k = nullptr;
    const void* v = nullptr;
    const void* g = nullptr;
    void* o = nullptr;
    void* st = nullptr;
    const void* cw = nullptr;
    const void* beta = nullptr;

    int64_t s = 0, dk = 0, dv = 0, gqq = 0, gvk = 0;
    int64_t qs0 = 0, qs1 = 0, qs2 = 0;
    int64_t ks0 = 0, ks1 = 0, ks2 = 0;
    int64_t vs0 = 0, vs1 = 0, vs2 = 0;
    int64_t gs0 = 0, gs1 = 0, gs2 = 0;
    int64_t os0 = 0, os1 = 0, os2 = 0;
    int64_t ts0 = 0, ts1 = 0, ts2 = 0, ts3 = 0;
    int64_t cws0 = 0, cws2 = 0;
    int64_t bs0 = 0, bs1 = 0, bs2 = 0;

    /// Conv channel bases of the packed Q | K | V blob: the Q block is laid out
    /// over the expanded query heads, then the key heads, then the value heads.
    int64_t k_ch = 0, v_ch = 0;

    int64_t ck = 0;      ///< conv taps; 0 means no conv
    int64_t cj = 0;      ///< value columns per block
    int64_t rg = 0;      ///< row groups per column
    int64_t dkpad = 0;   ///< row count the block addresses (16·rg)
    float scale = 1.0f;
    int has_beta = 0;
};

/// One channel of the causal conv at token t: y[t] = sum_k w[k]·x[t-(ck-1)+k]
/// with zeroes left of t = 0, exactly as the CPU's la_fetch_row walks it.
///
/// The four-tap case — the gated delta net's — gets an unrolled body: the tap
/// loop's bound is a kernel argument, so the rolled form the compiler would
/// otherwise emit turns four taps into four dependent global round trips, and
/// the staging barrier waits for the last. Unrolled, the four loads issue
/// together and the token's staging costs one round trip. The weighting is a
/// tree, not a chain, for the same reason.
template <bool CONV, typename T>
__device__ __forceinline__ float la_fetch(const T* base, int64_t t, int64_t i, int64_t ts,
                                          const T* w, int64_t wtap, int64_t ck) noexcept
{
    if constexpr (!CONV) {
        return s_load(base + t * ts + i);
    } else if (ck == 4 && t >= 3) {
        const T* p = base + (t - 3) * ts + i;
        const float x0 = s_load(p);
        const float x1 = s_load(p + ts);
        const float x2 = s_load(p + 2 * ts);
        const float x3 = s_load(p + 3 * ts);
        const float a0 = s_load(w) * x0;
        const float a1 = s_load(w + wtap) * x1;
        const float a2 = s_load(w + 2 * wtap) * x2;
        const float a3 = s_load(w + 3 * wtap) * x3;
        return (a0 + a1) + (a2 + a3);
    } else {
        float acc = 0.0f;
        const int64_t first = t - (ck - 1);
        for (int64_t k = 0; k < ck; ++k) {
            if (first + k >= 0) {
                acc = fmaf(s_load(w + k * wtap), s_load(base + (first + k) * ts + i), acc);
            }
        }
        return acc;
    }
}

/// Sum over the rg lanes of one column — the XOR butterfly with the segment
/// width pinned to the group (LeetCUDA's warp_reduce_sum<kWarpWidth>). Every
/// lane of the group ends with the total.
__device__ __forceinline__ float la_group_sum(float v, int rg) noexcept
{
    for (int off = rg >> 1; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off, rg);
    }
    return v;
}

__device__ __forceinline__ float4 la_ld4(const float* p) noexcept
{
    return *reinterpret_cast<const float4*>(p);
}

/// Raw conv rows a thread keeps: the newest in a register, the three before it
/// in a ring in shared memory. Four is the conv a gated delta layer carries; a
/// wider kernel fetches its leading taps from global in the fold.
constexpr int kLaRing = 4;

/// Issue the next token's reads, and nothing more.
///
/// The split is the whole point. With the fold here the loads gain a dependent
/// chain in front of the passes -- the staged value is wanted before the
/// arithmetic can start -- and a token-deep pipeline then buys nothing at all
/// (measured: 6.0 ms prefill against 4.6 with the fold deferred to the commit).
/// The ring is the second half of that lesson: four raw taps as registers are
/// four values per channel held live across the passes, and ptxas pays for them
/// out of the passes themselves (measured: 9.2 ms, 12.8 at three blocks a
/// multiprocessor). One register per channel and three rows in shared memory
/// keeps the token loop's occupancy instead.
///
/// Rows are clamped rather than skipped so every load issues unconditionally;
/// the commit masks what the clamps spoiled.
template <typename T, int GQ>
__device__ __forceinline__ void la_prefetch(const LaParams& p, int64_t t, int tid,
                                            const T* kbase, const T* vbase, const T* gbase,
                                            const T* qbase, float (&k_out)[kLaSlots],
                                            float (&g_out)[kLaSlots],
                                            float (&q_out)[kLaSlots][GQ],
                                            float (&v_out)[kLaVSlots]) noexcept
{
#pragma unroll
    for (int sl = 0; sl < kLaSlots; ++sl) {
        // A slot past the row block is skipped whole: the test is uniform, so
        // the loads and their registers vanish for the narrow row blocks that
        // need one slot.
        if (int64_t(sl) * kLaThreads < p.dkpad) {
            const int64_t j = int64_t(tid) + int64_t(sl) * kLaThreads;
            const int64_t i = j < p.dk ? j : p.dk - 1;
            k_out[sl] = s_load(kbase + t * p.ks2 + i);
            g_out[sl] = s_load(gbase + t * p.gs2 + i);
#pragma unroll
            for (int h = 0; h < GQ; ++h) {
                q_out[sl][h] = s_load(qbase + h * p.qs1 + t * p.qs2 + i);
            }
        }
    }
#pragma unroll
    for (int sl = 0; sl < kLaVSlots; ++sl) {
        if (int64_t(sl) * kLaThreads < p.cj) {
            const int64_t j = int64_t(tid) + int64_t(sl) * kLaThreads;
            const int64_t jc = static_cast<int64_t>(blockIdx.x) * p.cj + j;
            v_out[sl] = s_load(vbase + t * p.vs2 + (jc < p.dv ? jc : 0));
        }
    }
}

/// The four-tap conv the gated delta layers carry, folded from the ring with the
/// weights already in registers. Straight line on purpose: a general tap count
/// needs a guard around each weight load, those guards are branches, and the fold
/// sits at the end of the token step where a chain of branch-wrapped loads is
/// exposed whole (measured: 6.0 ms prefill with the guards, 4.8 without).
__device__ __forceinline__ float la_fold4(const float* r, int64_t stride, float newest, int64_t t,
                                          int64_t i, const float (&w)[kLaRing]) noexcept
{
    const float a0 = w[0] * r[((t - 3) & (kLaRing - 1)) * stride + i];
    const float a1 = w[1] * r[((t - 2) & (kLaRing - 1)) * stride + i];
    const float a2 = w[2] * r[((t - 1) & (kLaRing - 1)) * stride + i];
    const float a3 = w[3] * newest;
    return (a0 + a1) + (a2 + a3);
}

/// The same walk for any tap count: a kernel wider than the ring fetches its
/// leading taps from global here, which is the one exposed round trip left.
template <typename T>
__device__ __forceinline__ float la_fold_any(const float* r, int64_t stride, float newest,
                                             int64_t t, int64_t i, int64_t ck, const T* w,
                                             int64_t wtap, const T* base, int64_t ts) noexcept
{
    float acc = 0.0f;
    for (int64_t k = 0; k + kLaRing < ck; ++k) {
        const int64_t row = t - (ck - 1) + k;
        if (row >= 0) {
            acc = fmaf(s_load(w + k * wtap), s_load(base + row * ts + i), acc);
        }
    }
    // The last four taps, k = ck-4 .. ck-1, at rows t-3 .. t. A narrower kernel
    // leaves the oldest of them outside its tap range and out of the sum.
    const int64_t k0 = ck - kLaRing;
    if (k0 >= 0) {
        acc = fmaf(s_load(w + (k0 + 0) * wtap), r[((t - 3) & (kLaRing - 1)) * stride + i], acc);
    }
    if (k0 + 1 >= 0) {
        acc = fmaf(s_load(w + (k0 + 1) * wtap), r[((t - 2) & (kLaRing - 1)) * stride + i], acc);
    }
    if (k0 + 2 >= 0) {
        acc = fmaf(s_load(w + (k0 + 2) * wtap), r[((t - 1) & (kLaRing - 1)) * stride + i], acc);
    }
    return fmaf(s_load(w + (ck - 1) * wtap), newest, acc);
}

/// Fold a prefetched token and write it out: the raw rows into the ring for the
/// tokens after this one, the convolved ones into the buffer the passes read.
/// Ring slot and element index are the writing thread's own, so the ring needs
/// no barrier of its own; the folded buffer is one of two, so this never races
/// the token being computed.
template <typename T, int GQ, int CV>
__device__ __forceinline__ void la_commit(float* sb, float* ringK, float* ringQ, float* ringV,
                                          const LaParams& p, int64_t t, int tid, const T* cwk,
                                          const T* cwq, const T* cwv, const T* kbase,
                                          const T* qbase, const T* vbase,
                                          const float (&k_in)[kLaSlots],
                                          const float (&g_in)[kLaSlots],
                                          const float (&q_in)[kLaSlots][GQ],
                                          const float (&v_in)[kLaVSlots],
                                          const float (&wk)[kLaRing],
                                          const float (&wq)[GQ][kLaRing],
                                          const float (&wv)[kLaRing]) noexcept
{
    constexpr bool CONV = CV != 0;
    constexpr bool CK4 = CV == 1;
    float* sKG = sb;
    float* sG = sKG + p.dkpad;
    float* sK = sG + p.dkpad;
    float* sQ = sK + p.dkpad;
    float* sV = sQ + GQ * p.dkpad;
    const int slot = static_cast<int>(t & (kLaRing - 1));

#pragma unroll
    for (int sl = 0; sl < kLaSlots; ++sl) {
        const int64_t i = int64_t(tid) + int64_t(sl) * kLaThreads;
        if (i < p.dkpad) {
            if constexpr (CONV) {
                ringK[slot * p.dkpad + i] = k_in[sl];
            }
            const bool live = i < p.dk;
            const float gv = live ? __expf(g_in[sl]) : 0.0f;
            float kv = 0.0f;
            if (live) {
                if constexpr (CV == 0) {
                    kv = k_in[sl];
                } else if constexpr (CK4) {
                    // The register taps are the thread's own channel's; a slot
                    // past the first is another channel's and takes its own.
                    kv = i == tid ? la_fold4(ringK, p.dkpad, k_in[sl], t, i, wk)
                                  : la_fold_any<T>(ringK, p.dkpad, k_in[sl], t, i, p.ck,
                                                   cwk + i * p.cws0, p.cws2, kbase, p.ks2);
                } else {
                    kv = la_fold_any<T>(ringK, p.dkpad, k_in[sl], t, i, p.ck, cwk + i * p.cws0,
                                        p.cws2, kbase, p.ks2);
                }
            }
            sKG[i] = kv * gv;
            sG[i] = gv;
            sK[i] = kv;
        }
    }
#pragma unroll
    for (int h = 0; h < GQ; ++h) {
        float* ringH = ringQ + h * kLaRing * p.dkpad;
#pragma unroll
        for (int sl = 0; sl < kLaSlots; ++sl) {
            const int64_t i = int64_t(tid) + int64_t(sl) * kLaThreads;
            if (i < p.dkpad) {
                if constexpr (CONV) {
                    ringH[slot * p.dkpad + i] = q_in[sl][h];
                }
            }
            float qv = 0.0f;
            if (h < p.gqq && i < p.dk) {
                if constexpr (CV == 0) {
                    qv = q_in[sl][h];
                } else if constexpr (CK4) {
                    qv = i == tid ? la_fold4(ringH, p.dkpad, q_in[sl][h], t, i, wq[h])
                                  : la_fold_any<T>(ringH, p.dkpad, q_in[sl][h], t, i, p.ck,
                                                   cwq + (h * p.dk + i) * p.cws0, p.cws2,
                                                   qbase + h * p.qs1, p.qs2);
                } else {
                    qv = la_fold_any<T>(ringH, p.dkpad, q_in[sl][h], t, i, p.ck,
                                        cwq + (h * p.dk + i) * p.cws0, p.cws2,
                                        qbase + h * p.qs1, p.qs2);
                }
            }
            if (i < p.dkpad) {
                sQ[h * p.dkpad + i] = qv;
            }
        }
    }
#pragma unroll
    for (int sl = 0; sl < kLaVSlots; ++sl) {
        const int64_t c = int64_t(tid) + int64_t(sl) * kLaThreads;
        if (c < p.cj) {
            if constexpr (CONV) {
                ringV[slot * p.cj + c] = v_in[sl];
            }
            const int64_t jc = static_cast<int64_t>(blockIdx.x) * p.cj + c;
            float vv = 0.0f;
            if (jc < p.dv) {
                if constexpr (CV == 0) {
                    vv = v_in[sl];
                } else if constexpr (CK4) {
                    vv = c == tid ? la_fold4(ringV, p.cj, v_in[sl], t, c, wv)
                                  : la_fold_any<T>(ringV, p.cj, v_in[sl], t, c, p.ck,
                                                   cwv + jc * p.cws0, p.cws2, vbase, p.vs2);
                } else {
                    vv = la_fold_any<T>(ringV, p.cj, v_in[sl], t, c, p.ck, cwv + jc * p.cws0,
                                        p.cws2, vbase, p.vs2);
                }
            }
            sV[c] = vv;
        }
    }
}

/// Load this thread's conv taps once: they are per channel and the channel is
/// the thread's for the whole sequence, so they belong in registers rather than
/// in a per-token load inside the fold's dependent chain. Only the thread's own
/// slot is cached: a wider block walks channels past `tid`, and those take
/// their own taps from global in the fold (the price of Dk > 128).
template <typename T, int GQ>
__device__ __forceinline__ void la_weights(const LaParams& p, int tid, const T* cwk,
                                           const T* cwq, const T* cwv, float (&wk)[kLaRing],
                                           float (&wq)[GQ][kLaRing],
                                           float (&wv)[kLaRing]) noexcept
{
    const int64_t i = tid < p.dk ? tid : p.dk - 1;
    const int64_t jc = static_cast<int64_t>(blockIdx.x) * p.cj + tid;
    const int64_t j = jc < p.dv ? jc : 0;
    const int64_t si = i * p.cws0;
    const int64_t sj = j * p.cws0;
#pragma unroll
    for (int k = 0; k < kLaRing; ++k) {
        wk[k] = s_load(cwk + si + k * p.cws2);
        wv[k] = s_load(cwv + sj + k * p.cws2);
    }
#pragma unroll
    for (int h = 0; h < GQ; ++h) {
#pragma unroll
        for (int k = 0; k < kLaRing; ++k) {
            wq[h][k] = s_load(cwq + (h * p.dk + i) * p.cws0 + k * p.cws2);
        }
    }
}

/// The stage for kernels with more than one query head per state: folded here
/// and now, no pipeline. Those shapes spend GQ times the arithmetic per token
/// and are rare enough not to be worth GQ taps a thread.
template <typename T, int GQ, bool CONV>
__device__ __forceinline__ void la_stage(float* sb, const LaParams& p, int64_t t, int tid,
                                         const T* kbase, const T* vbase, const T* gbase,
                                         const T* qbase, const T* cwk, const T* cwq,
                                         const T* cwv) noexcept
{
    float* sKG = sb;
    float* sG = sKG + p.dkpad;
    float* sK = sG + p.dkpad;
    float* sQ = sK + p.dkpad;
    float* sV = sQ + GQ * p.dkpad;
    for (int64_t i = tid; i < p.dkpad; i += kLaThreads) {
        float gv = 0.0f;
        float kv = 0.0f;
        if (i < p.dk) {
            gv = __expf(s_load(gbase + t * p.gs2 + i));
            kv = la_fetch<CONV>(kbase, t, i, p.ks2, cwk + i * p.cws0, p.cws2, p.ck);
        }
        sG[i] = gv;
        sK[i] = kv;
        sKG[i] = kv * gv;
#pragma unroll
        for (int h = 0; h < GQ; ++h) {
            float qv = 0.0f;
            if (h < p.gqq) {
                if (i < p.dk) {
                    qv = la_fetch<CONV>(qbase + h * p.qs1, t, i, p.qs2,
                                        cwq + (h * p.dk + i) * p.cws0, p.cws2, p.ck);
                }
                sQ[h * p.dkpad + i] = qv;
            }
        }
    }
    for (int64_t c = tid; c < p.cj; c += kLaThreads) {
        const int64_t jc = static_cast<int64_t>(blockIdx.x) * p.cj + c;
        const T* w = nullptr;
        if constexpr (CONV) {
            const int64_t cc = jc < p.dv ? jc : 0;
            w = cwv + cc * p.cws0;
        }
        sV[c] = jc < p.dv ? la_fetch<CONV>(vbase, t, jc, p.vs2, w, p.cws2, p.ck) : 0.0f;
    }
}

template <typename T, int GQ, int CV>
__global__ void __launch_bounds__(kLaThreads, kLaMinBlocks)
linear_attention_kernel(LaParams p)
{
    constexpr bool CONV = CV != 0;
    constexpr bool CK4 = CV == 1;
    extern __shared__ float smem[];
    // Two token buffers, each laid out as
    //   [gate-folded key | decay | raw key | GQ queries | value columns].
    // The token being computed reads one while the next one fills the other.
    const int64_t bufstride = (3 + GQ) * p.dkpad + p.cj;

    const int tid = threadIdx.x;
    const int rg = static_cast<int>(p.rg);
    const int r = tid % rg;
    const int jl = tid / rg;
    const int64_t hv = blockIdx.y;
    const int64_t b = blockIdx.z;
    // Each thread owns kLaCols adjacent value columns: one staged row read then
    // feeds kLaCols state updates and readouts (the thread tile, along the
    // value axis — this loop is shared-memory limited, and the columns are
    // where the reuse is).
    const int64_t jbase = static_cast<int64_t>(blockIdx.x) * p.cj + int64_t(jl) * kLaCols;
    bool active[kLaCols];
#pragma unroll
    for (int c = 0; c < kLaCols; ++c) {
        active[c] = jbase + c < p.dv;
    }

    const int64_t hk = hv / p.gvk;
    const int64_t hq0 = hv * p.gqq;

    const T* kbase = static_cast<const T*>(p.k) + b * p.ks0 + hk * p.ks1;
    const T* vbase = static_cast<const T*>(p.v) + b * p.vs0 + hv * p.vs1;
    const T* gbase = static_cast<const T*>(p.g) + b * p.gs0 + hv * p.gs1;
    const T* qbase = static_cast<const T*>(p.q) + b * p.qs0 + hq0 * p.qs1;
    const T* bbase = static_cast<const T*>(p.beta) + b * p.bs0 + hv * p.bs1;
    T* obase = static_cast<T*>(p.o) + b * p.os0 + hq0 * p.os1;
    T* sbase = static_cast<T*>(p.st) + b * p.ts0 + hv * p.ts1;

    // Conv channel pointers. An inactive column parks on channel 0 so the
    // pointer is in range even though nothing reads it.
    const T* cwq = nullptr;
    const T* cwk = nullptr;
    const T* cwv = nullptr;
    if constexpr (CONV) {
        const T* cw = static_cast<const T*>(p.cw);
        cwq = cw + hq0 * p.dk * p.cws0;
        cwk = cw + (p.k_ch + hk * p.dk) * p.cws0;
        cwv = cw + (p.v_ch + hv * p.dv) * p.cws0;
    }

    // The state slab into registers; one strided read each, then never again.
    float S[kLaCols][kLaRows];
#pragma unroll
    for (int gi = 0; gi < kLaGroups; ++gi) {
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int64_t i = static_cast<int64_t>(gi * 4 * rg + r * 4 + v);
#pragma unroll
            for (int c = 0; c < kLaCols; ++c) {
                S[c][gi * 4 + v] = (active[c] && i < p.dk)
                                       ? s_load(sbase + i * p.ts2 + (jbase + c) * p.ts3)
                                       : 0.0f;
            }
        }
    }

    // Ring of raw conv rows, one slot a token, one element a thread: the newest
    // row rides in the registers below and the three before it land here.
    float* ringK = smem + 2 * bufstride;
    float* ringQ = ringK + kLaRing * p.dkpad;
    float* ringV = ringQ + GQ * kLaRing * p.dkpad;

    // One token of stage in registers, in flight over the previous token's
    // passes: `rk_` the key, `rg_` its gate, `rq_` the GQ queries and `rv_` this
    // block's value columns.
    float rk_[kLaSlots];
    float rg_[kLaSlots];
    float rq_[kLaSlots][GQ];
    float rv_[kLaVSlots];

    // This thread's conv taps, loaded once for the whole sequence.
    float wk[kLaRing] = {};
    float wq[GQ][kLaRing] = {};
    float wv[kLaRing] = {};
    if constexpr (CK4 && GQ == 1) {
        la_weights<T, GQ>(p, tid, cwk, cwq, cwv, wk, wq, wv);
    }

    if constexpr (GQ == 1) {
        // Zeroed slots are the rows left of t = 0, which is what lets the fold
        // skip bound checks. Without a conv there is no fold and no ring.
        if constexpr (CONV) {
            for (int sl = 0; sl < kLaRing; ++sl) {
#pragma unroll
                for (int s2 = 0; s2 < kLaSlots; ++s2) {
                    const int i = tid + s2 * kLaThreads;
                    if (i < p.dkpad) {
                        ringK[sl * p.dkpad + i] = 0.0f;
                        ringQ[sl * p.dkpad + i] = 0.0f;
                    }
                }
#pragma unroll
                for (int s2 = 0; s2 < kLaVSlots; ++s2) {
                    const int c = tid + s2 * kLaThreads;
                    if (c < p.cj) {
                        ringV[sl * p.cj + c] = 0.0f;
                    }
                }
            }
        }
        la_prefetch<T, GQ>(p, 0, tid, kbase, vbase, gbase, qbase, rk_, rg_, rq_, rv_);
    }

    for (int64_t t = 0; t < p.s; ++t) {
        // The token prefetched last iteration goes to shared, into the buffer
        // the current token's passes do not read.
        float* sb = smem + (t & 1) * bufstride;
        if constexpr (GQ == 1) {
            la_commit<T, GQ, CV>(sb, ringK, ringQ, ringV, p, t, tid, cwk, cwq, cwv, kbase,
                                 qbase, vbase, rk_, rg_, rq_, rv_, wk, wq, wv);
            __syncthreads();
            if (t + 1 < p.s) {
                la_prefetch<T, GQ>(p, t + 1, tid, kbase, vbase, gbase, qbase, rk_, rg_, rq_,
                                   rv_);
            }
        } else {
            la_stage<T, GQ, CONV>(sb, p, t, tid, kbase, vbase, gbase, qbase, cwk, cwq, cwv);
            __syncthreads();
        }

        const float* sKG = sb;                // gate-folded key, pass A
        const float* sG = sKG + p.dkpad;      // decay
        const float* sK = sG + p.dkpad;       // raw key, pass B
        const float* sQ = sK + p.dkpad;       // GQ query heads, raw
        const float* sV = sQ + GQ * p.dkpad;  // the block's value columns

        // Pass A: r[j] = k·(g ⊙ S), accumulated over this thread's rows.
        float accK[kLaCols];
#pragma unroll
        for (int c = 0; c < kLaCols; ++c) {
            accK[c] = 0.0f;
        }
#pragma unroll
        for (int gi = 0; gi < kLaGroups; ++gi) {
            const float4 kg = la_ld4(sKG + gi * 4 * rg + r * 4);
#pragma unroll
            for (int c = 0; c < kLaCols; ++c) {
                accK[c] = fmaf(kg.x, S[c][gi * 4 + 0],
                               fmaf(kg.y, S[c][gi * 4 + 1],
                                    fmaf(kg.z, S[c][gi * 4 + 2],
                                         fmaf(kg.w, S[c][gi * 4 + 3], accK[c]))));
            }
        }

        const float beta = p.has_beta ? s_load(bbase + t * p.bs2) : 1.0f;
        float d[kLaCols];
#pragma unroll
        for (int c = 0; c < kLaCols; ++c) {
            const float rsum = la_group_sum(accK[c], rg);
            const float vv = active[c] ? sV[jl * kLaCols + c] : 0.0f;
            d[c] = beta * (vv - rsum);
        }

        // Pass B: S = g ⊙ S + k ⊗ d, and the readout of the updated state.
        float accQ[kLaCols][GQ];
#pragma unroll
        for (int c = 0; c < kLaCols; ++c) {
#pragma unroll
            for (int h = 0; h < GQ; ++h) {
                accQ[c][h] = 0.0f;
            }
        }
#pragma unroll
        for (int gi = 0; gi < kLaGroups; ++gi) {
            const int64_t i0 = gi * 4 * rg + r * 4;
            const float4 gg = la_ld4(sG + i0);
            const float4 kk = la_ld4(sK + i0);
            float sv[kLaCols][4];
#pragma unroll
            for (int c = 0; c < kLaCols; ++c) {
                sv[c][0] = fmaf(gg.x, S[c][gi * 4 + 0], kk.x * d[c]);
                sv[c][1] = fmaf(gg.y, S[c][gi * 4 + 1], kk.y * d[c]);
                sv[c][2] = fmaf(gg.z, S[c][gi * 4 + 2], kk.z * d[c]);
                sv[c][3] = fmaf(gg.w, S[c][gi * 4 + 3], kk.w * d[c]);
                S[c][gi * 4 + 0] = sv[c][0];
                S[c][gi * 4 + 1] = sv[c][1];
                S[c][gi * 4 + 2] = sv[c][2];
                S[c][gi * 4 + 3] = sv[c][3];
            }
#pragma unroll
            for (int h = 0; h < GQ; ++h) {
                if (h < p.gqq) {
                    // One query row feeds every column of the tile.
                    const float4 qq = la_ld4(sQ + h * p.dkpad + i0);
#pragma unroll
                    for (int c = 0; c < kLaCols; ++c) {
                        accQ[c][h] = fmaf(qq.x, sv[c][0],
                                          fmaf(qq.y, sv[c][1],
                                               fmaf(qq.z, sv[c][2],
                                                    fmaf(qq.w, sv[c][3], accQ[c][h]))));
                    }
                }
            }
        }

#pragma unroll
        for (int h = 0; h < GQ; ++h) {
            if (h < p.gqq) {
#pragma unroll
                for (int c = 0; c < kLaCols; ++c) {
                    const float osum = la_group_sum(accQ[c][h], rg);
                    if (r == 0 && active[c]) {
                        s_store(obase + h * p.os1 + t * p.os2 + (jbase + c), p.scale * osum);
                    }
                }
            }
        }
    }

    // The state back out, in place: the caller owns it across calls.
#pragma unroll
    for (int gi = 0; gi < kLaGroups; ++gi) {
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int64_t i = static_cast<int64_t>(gi * 4 * rg + r * 4 + v);
#pragma unroll
            for (int c = 0; c < kLaCols; ++c) {
                if (active[c] && i < p.dk) {
                    s_store(sbase + i * p.ts2 + (jbase + c) * p.ts3, S[c][gi * 4 + v]);
                }
            }
        }
    }
}

template <typename T, int GQ, int CV>
Status la_launch(const LaParams& p, dim3 grid, int smem, cudaStream_t stream)
{
    linear_attention_kernel<T, GQ, CV><<<grid, kLaThreads, smem, stream>>>(p);
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

/// The conv pose: 0 no conv, 1 the four taps a gated delta layer carries (its
/// weights in registers, its fold a straight line), 2 any other tap count.
template <typename T, int GQ>
Status la_launch_pose(const LaParams& p, dim3 grid, int smem, cudaStream_t stream, int cv)
{
    switch (cv) {
    case 0:
        return la_launch<T, GQ, 0>(p, grid, smem, stream);
    case 1:
        return la_launch<T, GQ, 1>(p, grid, smem, stream);
    default:
        return la_launch<T, GQ, 2>(p, grid, smem, stream);
    }
}

template <typename T>
Status la_launch_dtype(const LaParams& p, dim3 grid, int smem, cudaStream_t stream, int cv)
{
    // One query head per state is the GDN / Qwen3-Next shape and takes the
    // exact instantiation; the generic one handles up to eight.
    return p.gqq == 1 ? la_launch_pose<T, 1>(p, grid, smem, stream, cv)
                      : la_launch_pose<T, 8>(p, grid, smem, stream, cv);
}

template <typename T>
Status linear_attention_impl(const LinearAttentionAttributes& attrs,
                             std::span<TensorView> outputs, std::span<const TensorView> inputs,
                             const ComputeContext& ctx)
{
    const TensorView& Q = inputs[0];
    const TensorView& K = inputs[1];
    const TensorView& V = inputs[2];
    const TensorView& G = inputs[3];
    TensorView& O = outputs[0];
    TensorView& St = outputs[1];

    const int64_t B = Q.shape(0);
    const int64_t Hq = Q.shape(1);
    const int64_t S = Q.shape(2);
    const int64_t Dk = Q.shape(3);
    const int64_t Hk = attrs.num_kv_heads > 0 ? attrs.num_kv_heads : Hq;
    const int64_t Hv = V.shape(1);
    const int64_t Dv = V.shape(3);
    const int64_t Gqq = Hq / Hv;

    if (Dk <= 0 || Dv <= 0 || S < 0 || B <= 0 || Hv <= 0 || Gqq <= 0 || Gqq > 8 ||
        Hv > 65535 || B > 65535) {
        return Status::unsupported;
    }

    const bool has_conv = inputs.size() > 4 && !inputs[4].is_empty();
    const bool has_beta = inputs.size() > 5 && !inputs[5].is_empty();

    // The chunked-WY route, when the call fits its pose: f16, the tensor-core
    // chunk geometry (Dk = Dv = 128, C = 64), one query head per state, and a
    // whole number of chunks. A long prefill is what it is for. The route
    // answers false when a chunk's decay range leaves the f16 panel format;
    // the per-token kernel below is then the answer, unchanged.
    if constexpr (std::is_same_v<T, __half>) {
        const int64_t ck = has_conv ? inputs[4].shape(2) : 0;
        if (Dk == 128 && Dv == 128 && Gqq == 1 && S >= 64 && S % 64 == 0 && Hv % Hk == 0 &&
            ck <= 8) {
            LaWyRoute wr{};
            wr.q = Q.ptr<T>();
            wr.k = K.ptr<T>();
            wr.v = V.ptr<T>();
            wr.g = G.ptr<T>();
            wr.cw = has_conv ? inputs[4].ptr<T>() : nullptr;
            wr.beta = has_beta ? inputs[5].ptr<T>() : nullptr;
            wr.o = O.ptr<T>();
            wr.st = St.ptr<T>();
            wr.b = B;
            wr.hv = Hv;
            wr.hk = Hk;
            wr.s = S;
            wr.dk = Dk;
            wr.dv = Dv;
            wr.gvk = Hv / Hk;
            wr.ck = ck;
            wr.qs0 = Q.stride_elems(0);
            wr.qs1 = Q.stride_elems(1);
            wr.qs2 = Q.stride_elems(2);
            wr.ks0 = K.stride_elems(0);
            wr.ks1 = K.stride_elems(1);
            wr.ks2 = K.stride_elems(2);
            wr.vs0 = V.stride_elems(0);
            wr.vs1 = V.stride_elems(1);
            wr.vs2 = V.stride_elems(2);
            wr.gs0 = G.stride_elems(0);
            wr.gs1 = G.stride_elems(1);
            wr.gs2 = G.stride_elems(2);
            wr.os0 = O.stride_elems(0);
            wr.os1 = O.stride_elems(1);
            wr.os2 = O.stride_elems(2);
            wr.ts0 = St.stride_elems(0);
            wr.ts1 = St.stride_elems(1);
            wr.ts2 = St.stride_elems(2);
            wr.ts3 = St.stride_elems(3);
            wr.cws0 = has_conv ? inputs[4].stride_elems(0) : 0;
            wr.cws2 = has_conv ? inputs[4].stride_elems(2) : 0;
            wr.bs0 = has_beta ? inputs[5].stride_elems(0) : 0;
            wr.bs1 = has_beta ? inputs[5].stride_elems(1) : 0;
            wr.bs2 = has_beta ? inputs[5].stride_elems(2) : 0;
            wr.k_ch = Hq * Dk;
            wr.v_ch = Hq * Dk + Hk * Dk;
            wr.scale = attrs.scale != 0.0f ? attrs.scale : 1.0f / std::sqrt(static_cast<float>(Dk));
            if (la_wy_route_cuda(wr, static_cast<cudaStream_t>(ctx.cuda_stream))) {
                return Status::ok;
            }
        }
    }

    // 16 rows a thread bounds the key width at 16·rg, and rg at one warp.
    int64_t rg = 1;
    while (rg * kLaRows < Dk) {
        rg <<= 1;
    }
    if (rg > kCudaWarpSize) {
        return Status::unsupported;
    }

    LaParams p{};
    p.q = Q.ptr<T>();
    p.k = K.ptr<T>();
    p.v = V.ptr<T>();
    p.g = G.ptr<T>();
    p.o = O.ptr<T>();
    p.st = St.ptr<T>();
    p.cw = has_conv ? inputs[4].ptr<T>() : nullptr;
    p.beta = has_beta ? inputs[5].ptr<T>() : nullptr;

    p.s = S;
    p.dk = Dk;
    p.dv = Dv;
    p.gqq = Gqq;
    p.gvk = Hv / Hk;

    p.qs0 = Q.stride_elems(0);
    p.qs1 = Q.stride_elems(1);
    p.qs2 = Q.stride_elems(2);
    p.ks0 = K.stride_elems(0);
    p.ks1 = K.stride_elems(1);
    p.ks2 = K.stride_elems(2);
    p.vs0 = V.stride_elems(0);
    p.vs1 = V.stride_elems(1);
    p.vs2 = V.stride_elems(2);
    p.gs0 = G.stride_elems(0);
    p.gs1 = G.stride_elems(1);
    p.gs2 = G.stride_elems(2);
    p.os0 = O.stride_elems(0);
    p.os1 = O.stride_elems(1);
    p.os2 = O.stride_elems(2);
    p.ts0 = St.stride_elems(0);
    p.ts1 = St.stride_elems(1);
    p.ts2 = St.stride_elems(2);
    p.ts3 = St.stride_elems(3);
    p.cws0 = has_conv ? inputs[4].stride_elems(0) : 0;
    p.cws2 = has_conv ? inputs[4].stride_elems(2) : 0;
    p.bs0 = has_beta ? inputs[5].stride_elems(0) : 0;
    p.bs1 = has_beta ? inputs[5].stride_elems(1) : 0;
    p.bs2 = has_beta ? inputs[5].stride_elems(2) : 0;

    p.k_ch = Hq * Dk;
    p.v_ch = Hq * Dk + Hk * Dk;
    p.ck = has_conv ? inputs[4].shape(2) : 0;
    p.cj = kLaCols * (kLaThreads / rg);
    p.rg = rg;
    p.dkpad = rg * kLaRows;
    p.scale = attrs.scale != 0.0f ? attrs.scale : 1.0f / std::sqrt(static_cast<float>(Dk));
    p.has_beta = has_beta ? 1 : 0;

    const int64_t nslab = (Dv + p.cj - 1) / p.cj;
    if (nslab <= 0) {
        return Status::unsupported;
    }

    const int gq = p.gqq == 1 ? 1 : 8;
    const int cv = !has_conv ? 0 : (p.ck == kLaRing ? 1 : 2);
    // Two folded token buffers, and for the pipelined conv poses the ring: four
    // raw rows of the key, the queries and the value slab. The widest is Dk = 512
    // with GQ = 8 at 45 KB, still inside the 48 KB a launch may ask for without
    // opting in (that pose takes the unfolded stage and has no ring).
    size_t smem_floats = 2 * static_cast<size_t>((3 + gq) * p.dkpad + p.cj);
    if (gq == 1 && cv != 0) {
        smem_floats += static_cast<size_t>(kLaRing) * (2 * p.dkpad + p.cj);
    }
    const int smem = static_cast<int>(sizeof(float) * smem_floats);
    const dim3 grid(static_cast<unsigned>(nslab), static_cast<unsigned>(Hv),
                    static_cast<unsigned>(B));
    const cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    return la_launch_dtype<T>(p, grid, smem, stream, cv);
}

} // anonymous namespace

Status linear_attention_cuda(const LinearAttentionAttributes& attrs, std::span<TensorView> outputs,
                             std::span<const TensorView> inputs, const ComputeContext& ctx,
                             void* /*workspace*/)
{
    switch (inputs[0].data_type()) {
    case DataType::f32:
        return linear_attention_impl<float>(attrs, outputs, inputs, ctx);
    case DataType::f16:
        return linear_attention_impl<__half>(attrs, outputs, inputs, ctx);
    case DataType::bf16:
        return linear_attention_impl<__nv_bfloat16>(attrs, outputs, inputs, ctx);
    default:
        return Status::unsupported;
    }
}

} // namespace nnops::backend::cuda
