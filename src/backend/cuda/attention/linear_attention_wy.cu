/// @file linear_attention_wy.cu
/// @brief The chunked-WY route: a chunk-local panel kernel and the recurrent
///        kernel that walks the panels (kernel 1 and kernel 2 of the CPU route's
///        split, see linear_attention_wy_host.hpp for the algebra).
///
/// Kernel 1 is parallel over every (batch, value head, chunk): it accumulates
/// the log decay, builds the gated Q and K panels, forms the intra-chunk product
/// A = Ke·Ka^T, solves (I + beta A)[W|U] = [beta Ke | beta V] by forward
/// substitution, and writes W, U, Qe, Ka and the chunk's total decay. Kernel 2
/// is parallel over the heads only and walks the chunks in order, holding the
/// state in shared memory.
///
/// Geometry (fixed for the route). One CTA per (batch, value head, chunk) for
/// kernel 1 and one per (batch, value head) for kernel 2, 128 threads as four
/// warps; a warp owns sixteen chunk rows for the C-paced products and sixteen
/// state rows in each of the two passes the state update needs. Dk = Dv = 128, so
/// a chunk is a 64x128 panel and the state is 128x128.
///
/// The panels are [rows][D + 8] halfs — the eight-half skew the attention tile
/// uses to keep the eight rows one ldmatrix reads on disjoint bank quads. Every
/// product is an m16n8k16 HMMA over three poses, each one lifted from
/// attention_f16.cuh: A out of a [m][k] panel, B out of a [n][k] panel (the
/// gated key and the transposed state), and B out of a [k][n] panel with
/// ldmatrix.trans (the key again, for the state update).
///
/// Layout choices that are not obvious:
///  - The state is kept *transposed*, S^T[j][i], so both products that read it
///    (W·S and Qe·S) take the cheap non-transposing B pose, and the state
///    update's A is the transposed X rather than a transposed key.
///  - X is written transposed too, for the same reason: the readout and the
///    state update both use X^T as a [n][k] / [m][k] panel. The price is a
///    2-byte scatter on X's store.
///  - Aq is rounded to f16 and held in registers as the A fragments of the
///    readout product — the score-accumulator slots are the A-fragment slots,
///    exactly as attention_f16.cuh packs a softmax tile.
///  - The key panel is staged twice: once for Aq, then again after the readout
///    (the buffer that held W, dead by then). That keeps the live panels under
///    the 100 KB the launch may ask for.
///
/// Kernel 1's solve is a per-column forward substitution: each thread owns one
/// column of the right-hand side, so no barrier sits inside the row sweep —
/// column j's rows only ever depend on column j. The two solves (W's 128 key
/// columns, U's 128 value columns) reuse one f32 block.
///
/// The route's guard is the representation's own: Ka holds 1/L_s in f16, so a
/// chunk whose decay range leaves the f16 exponent range would be garbage. The
/// panel kernel tracks the largest panel entry it built and kernel 1 publishes
/// it; the route reads it once and answers false — the caller's per-token
/// kernel runs instead. This is the f16 analogue of the CPU route's
/// kChunkSafeLog guard, and it is data-dependent, so the call blocks once.

#include "linear_attention_wy_host.hpp"

#include "backend/cuda/matmul/ptx.h"

#include <cuda_fp16.h>

#include <atomic>
#include <cstring>

namespace nnops::backend::cuda {
namespace {

constexpr int kC = 64;
constexpr int kDk = 128;
constexpr int kDv = 128;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kRows = kC / kWarps; ///< chunk rows a warp owns (16)
constexpr int kPad = 8;

constexpr int kSStride = kDk + kPad; ///< S^T is [kDv][kDk + 8]
constexpr int kXStride = kC + kPad;  ///< X^T is [kDv][kC + 8]
constexpr int kPStride = kDk + kPad; ///< a chunk panel is [kC][kDk + 8]

constexpr int kSmemHalfs = kDv * kSStride + 2 * kC * kPStride + kDv * kXStride;

/// Kernel 1's shared block: the two gated key panels, the solved matrix, one
/// f32 right-hand-side block and the chunk's betas.
constexpr int kPanelHalfs = 2 * kC * kPStride;
constexpr int kPanelFloats = kC * kC + kC * kDk + kC;

/// Conv taps the fused panel kernel carries in registers; a wider kernel takes
/// the per-token route.
constexpr int kTaps = 8;

/// The largest panel entry the f16 panels can hold. Above it the 1/L_s
/// representation has left the format, and the per-token kernel has to run.
constexpr float kPanelMax = 6.0e4f;

/// m16n8k16, fp16 inputs against fp32 accumulators.
__device__ __forceinline__ void mma_h(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                      uint32_t b1)
{
#if PTX_SM80_ENABLED
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

/// Two floats as one packed f16 pair — an A fragment slot.
__device__ __forceinline__ uint32_t pack_h(float lo, float hi)
{
    const __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

/// ldmatrix.x4 as the A operand of one 16-row tile, from a [row][k] panel.
__device__ __forceinline__ void ld_a(const __half* base, int stride, int row0, int ks,
                                     int lane, uint32_t (&a)[4])
{
    const int r = row0 + (lane & 7) + ((lane >> 3) & 1) * 8;
    const int c = (lane >> 4) * 8 + ks * 16;
    ldmatrix_x4_at(&a[0], &a[1], &a[2], &a[3], smem_addr_u32(base + r * stride + c));
}

/// The same load as a B operand: the panel is [n][k] and its rows are the n
/// axis, which is the non-transposing pose.
__device__ __forceinline__ void ld_b_n(const __half* base, int stride, int n0, int ks,
                                       int lane, uint32_t (&b)[4])
{
    ld_a(base, stride, n0, ks, lane, b);
}

/// B out of a [k][n] panel: the transposing pose, sixteen k rows by sixteen n
/// columns a load.
__device__ __forceinline__ void ld_b_t(const __half* base, int stride, int k0, int n0,
                                       int lane, uint32_t (&b)[4])
{
    const int r = k0 + (lane & 7) + ((lane >> 3) & 1) * 8;
    const int c = (lane >> 4) * 8 + n0;
    ldmatrix_x4_trans_at(&b[0], &b[1], &b[2], &b[3], smem_addr_u32(base + r * stride + c));
}

/// Stage one [kC][kDk] panel: sixteen 16 B copies a row.
__device__ __forceinline__ void stage_panel(const __half* src, __half* dst, int tid)
{
    for (int idx = tid; idx < kC * (kDk / 8); idx += kThreads) {
        const int r = idx / (kDk / 8);
        const int ch = idx - r * (kDk / 8);
        cp_async_ca_16B(dst + r * kPStride + ch * 8, src + static_cast<int64_t>(r) * kDk + ch * 8);
    }
}

/// One channel's causal conv at the next token: shift the tap window in and
/// return the weighted sum. The walk reads each raw row once, which is why the
/// window is per-thread rather than a staged copy of the chunk.
///
/// The window is always kTaps wide and the tap count only decides which slots
/// carry weight (tap_load lays the taps out right-aligned). Both loops are then
/// compile-time bounded, which is what keeps the window in registers: a window
/// indexed by a runtime tap count becomes local memory, and one panel CTA with
/// four warps cannot hide that round trip (measured on the 2048-token prefill:
/// the panels kernel 4.9 ms with the runtime index, 2.3 without).
template <int T>
__device__ __forceinline__ float tap_fold(float (&win)[T], float newest, const float* w)
{
    for (int k = 0; k + 1 < T; ++k) {
        win[k] = win[k + 1];
    }
    win[T - 1] = newest;
    float acc = 0.0f;
    for (int k = 0; k < T; ++k) {
        acc = fmaf(w[k], win[k], acc);
    }
    return acc;
}

/// Load one channel's conv taps once for the whole chunk, right-aligned in the
/// fixed window: slot k carries tap k - (T - ck), and the leading slots — the
/// taps a narrower kernel does not have — carry a zero weight.
template <int T>
__device__ __forceinline__ void tap_load(const __half* cw, int64_t ch, int64_t cws0,
                                         int64_t cws2, int64_t ck, float (&w)[T])
{
    for (int k = 0; k < T; ++k) {
        const int64_t tap = k - (T - ck);
        w[k] = tap >= 0 ? __half2float(cw[ch * cws0 + tap * cws2]) : 0.0f;
    }
}

/// The taps before a chunk's first row: after the first fold's shift the
/// window holds x[t0-7 .. t0], so it starts as x[t0-8 .. t0-1]. Rows left of
/// the sequence are the causal conv's zeroes.
template <int T>
__device__ __forceinline__ void tap_preload(float (&win)[T], const __half* base, int64_t t0,
                                            int64_t stride, int64_t col)
{
    for (int k = 0; k < T; ++k) {
        const int64_t row = t0 - T + k;
        win[k] = row >= 0 ? __half2float(base[row * stride + col]) : 0.0f;
    }
}

/// The forward substitution's body: D[r] = X[r] - sum_{s<r} M[r][s] D[s], one
/// thread to a column. Column j never reads another column, so the row sweep
/// needs no barrier.
__device__ __forceinline__ void solve_column(float* x, const float* m, int col)
{
    for (int r = 0; r < kC; ++r) {
        const float* mr = m + r * kC;
        float a[8] = {};
        int s = 0;
        for (; s + 8 <= r; s += 8) {
            float v[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                v[k] = x[(s + k) * kDk + col];
            }
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                a[k] = fmaf(mr[s + k], v[k], a[k]);
            }
        }
        for (; s < r; ++s) {
            a[0] = fmaf(mr[s], x[s * kDk + col], a[0]);
        }
        x[r * kDk + col] -= ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
    }
}

} // namespace

/// Kernel 1: the chunk-local half. See the file header.
namespace {

struct LaPanelsParams {
    LaWyRoute r;
    void* w = nullptr;
    void* u = nullptr;
    void* qe = nullptr;
    void* ka = nullptr;
    float* lc = nullptr;
    unsigned* flag = nullptr;
};

__global__ void __launch_bounds__(kThreads) la_wy_panels_kernel(LaPanelsParams p)
{
    extern __shared__ __half hsmem[];
    __half* const ke = hsmem;                              // [kC][kPStride]
    __half* const ka = ke + kC * kPStride;                 // [kC][kPStride]
    float* const mf = reinterpret_cast<float*>(ka + kC * kPStride); // [kC][kC]
    float* const xf = mf + kC * kC;                        // [kC][kDk]
    float* const bt = xf + kC * kDk;                       // [kC]

    const LaWyRoute& r = p.r;
    const int64_t l = r.s / kC;
    const int64_t c = blockIdx.x % l;
    const int64_t bh = blockIdx.x / l;
    const int64_t h = bh % r.hv;
    const int64_t b = bh / r.hv;
    const int64_t hk = h / r.gvk;
    const int64_t t0 = c * kC;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const __half* const qb = static_cast<const __half*>(r.q) + b * r.qs0 + h * r.qs1;
    const __half* const kb = static_cast<const __half*>(r.k) + b * r.ks0 + hk * r.ks1;
    const __half* const vb = static_cast<const __half*>(r.v) + b * r.vs0 + h * r.vs1;
    const __half* const gb = static_cast<const __half*>(r.g) + b * r.gs0 + h * r.gs1;
    const __half* const bb = r.beta != nullptr
                                 ? static_cast<const __half*>(r.beta) + b * r.bs0 + h * r.bs1
                                 : nullptr;
    const __half* const cw = static_cast<const __half*>(r.cw);

    const int64_t panel = int64_t(kC) * kDk;
    __half* const wg = static_cast<__half*>(p.w) + bh * l * panel + c * panel;
    __half* const ug = static_cast<__half*>(p.u) + bh * l * panel + c * panel;
    __half* const qg = static_cast<__half*>(p.qe) + bh * l * panel + c * panel;
    __half* const kg = static_cast<__half*>(p.ka) + bh * l * panel + c * panel;
    float* const lg = p.lc + (bh * l + c) * kDk;

    const int ck = static_cast<int>(r.ck);

    if (tid < kC) {
        bt[tid] = bb != nullptr ? __half2float(bb[(t0 + tid) * r.bs2]) : 1.0f;
    }

    // ---- the panels: cumulative decay, conv, and the gate fold ---------------
    float wq[kTaps] = {};
    float wk[kTaps] = {};
    float wv[kTaps] = {};
    float xq[kTaps] = {};
    float xk[kTaps] = {};
    if (ck > 0) {
        tap_load(cw, int64_t(h) * kDk + tid, r.cws0, r.cws2, ck, wq);
        tap_load(cw, r.k_ch + hk * kDk + tid, r.cws0, r.cws2, ck, wk);
        tap_load(cw, r.v_ch + h * kDv + tid, r.cws0, r.cws2, ck, wv);
        tap_preload<kTaps>(xq, qb, t0, r.qs2, tid);
        tap_preload<kTaps>(xk, kb, t0, r.ks2, tid);
    }

    // Four rows of raw inputs in flight: the row walk is a dependency (the
    // cumulative decay), the loads are not, and four warps on a multiprocessor
    // cannot cover a global round trip they issue one at a time. Same reason
    // the solve below batches eight.
    float cum = 0.0f;
    float peak = 0.0f;
    for (int r0 = 0; r0 < kC; r0 += 4) {
        float gv[4], qv[4], kv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t t = t0 + r0 + j;
            gv[j] = __half2float(gb[t * r.gs2 + tid]);
            qv[j] = __half2float(qb[t * r.qs2 + tid]);
            kv[j] = __half2float(kb[t * r.ks2 + tid]);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int row = r0 + j;
            cum += gv[j];
            const float e = __expf(cum);
            const float em = __expf(-cum);
            float qq = qv[j];
            float kk = kv[j];
            if (ck > 0) {
                qq = tap_fold<kTaps>(xq, qv[j], wq);
                kk = tap_fold<kTaps>(xk, kv[j], wk);
            }
            const float kev = kk * e;
            const float kav = kk * em;
            const float qev = qq * e;
            ke[row * kPStride + tid] = __float2half_rn(kev);
            ka[row * kPStride + tid] = __float2half_rn(kav);
            qg[row * kDk + tid] = __float2half_rn(qev);
            peak = fmaxf(peak, fmaxf(fabsf(kev), fmaxf(fabsf(kav), fabsf(qev))));
            if (row == kC - 1) {
                lg[tid] = e;
            }
        }
    }

    // Publish the largest panel entry: above kPanelMax the f16 panels cannot
    // hold 1/L_s and the per-token route has to run.
    for (int off = 16; off > 0; off >>= 1) {
        peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, off));
    }
    __shared__ float warp_peak[kWarps];
    if (lane == 0) {
        warp_peak[warp] = peak;
    }
    __syncthreads();
    if (tid == 0) {
        float m = warp_peak[0];
        for (int i = 1; i < kWarps; ++i) {
            m = fmaxf(m, warp_peak[i]);
        }
        atomicMax(p.flag, __float_as_uint(m));
    }

    // ---- A = Ke Ka^T, masked strict-lower and scaled by beta ----------------
    float acc[8][4];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            acc[t][e] = 0.0f;
        }
    }
#pragma unroll
    for (int ks = 0; ks < kDk / 16; ++ks) {
        uint32_t af[4];
        ld_a(ke, kPStride, warp * kRows, ks, lane, af);
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            uint32_t bf[4];
            ld_b_n(ka, kPStride, 16 * g, ks, lane, bf);
            mma_h(acc[2 * g], af, bf[0], bf[2]);
            mma_h(acc[2 * g + 1], af, bf[1], bf[3]);
        }
    }
    const int cpair = 2 * (lane & 3);
    const int row0 = warp * kRows + (lane >> 2);
    const int row1 = row0 + 8;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const int s = 8 * t + cpair;
        mf[row0 * kC + s] = s < row0 ? bt[row0] * acc[t][0] : 0.0f;
        mf[row0 * kC + s + 1] = s + 1 < row0 ? bt[row0] * acc[t][1] : 0.0f;
        mf[row1 * kC + s] = s < row1 ? bt[row1] * acc[t][2] : 0.0f;
        mf[row1 * kC + s + 1] = s + 1 < row1 ? bt[row1] * acc[t][3] : 0.0f;
    }
    __syncthreads(); ///< M is whole, and every warp has read Ke and Ka

    // ---- W: (I + beta A) W = beta Ke, then U the same way from beta V ------
    for (int row = 0; row < kC; ++row) {
        xf[row * kDk + tid] = bt[row] * __half2float(ke[row * kPStride + tid]);
    }
    solve_column(xf, mf, tid);
    for (int row = 0; row < kC; ++row) {
        wg[row * kDk + tid] = __float2half_rn(xf[row * kDk + tid]);
    }

    float xv[kTaps] = {};
    if (ck > 0) {
        tap_preload<kTaps>(xv, vb, t0, r.vs2, tid);
    }
    for (int r0 = 0; r0 < kC; r0 += 4) {
        float vv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t t = t0 + r0 + j;
            vv[j] = __half2float(vb[t * r.vs2 + tid]);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float f = ck > 0 ? tap_fold<kTaps>(xv, vv[j], wv) : vv[j];
            xf[(r0 + j) * kDk + tid] = bt[r0 + j] * f;
        }
    }
    solve_column(xf, mf, tid);
    for (int row = 0; row < kC; ++row) {
        ug[row * kDk + tid] = __float2half_rn(xf[row * kDk + tid]);
    }

    // Ka goes out last: it is what A needed.
    for (int row = 0; row < kC; ++row) {
        kg[row * kDk + tid] = ka[row * kPStride + tid];
    }
}

} // namespace

namespace {

__global__ void __launch_bounds__(kThreads) la_wy_kernel(LaWyParams p)
{
    extern __shared__ __half smem[];
    __half* const sT = smem;                        // S^T [kDv][kSStride]
    __half* const qe = sT + kDv * kSStride;         // Qe  [kC][kPStride]
    __half* const buf = qe + kC * kPStride;         // Ka, then W, then Ka again
    __half* const xT = buf + kC * kPStride;         // X^T [kDv][kXStride]

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int64_t bh = blockIdx.x;
    const int64_t h = bh % p.hv;
    const int64_t b = bh / p.hv;

    const int64_t os0 = p.os0 > 0 ? p.os0 : p.s * kDv;
    const int64_t os1 = p.os1 > 0 ? p.os1 : p.s * kDv;
    const int64_t os2 = p.os2 > 0 ? p.os2 : kDv;
    const int64_t ts0 = p.ts0 > 0 ? p.ts0 : int64_t(kDk) * kDv;
    const int64_t ts1 = p.ts1 > 0 ? p.ts1 : int64_t(kDk) * kDv;
    const int64_t ts2 = p.ts2 > 0 ? p.ts2 : kDv;

    const int64_t panel_elems = int64_t(kC) * kDk;
    const __half* const w0 = static_cast<const __half*>(p.w) + bh * p.l * panel_elems;
    const __half* const qe0 = static_cast<const __half*>(p.qe) + bh * p.l * panel_elems;
    const __half* const ka0 = static_cast<const __half*>(p.ka) + bh * p.l * panel_elems;
    const __half* const u0 = static_cast<const __half*>(p.u) + bh * p.l * int64_t(kC) * kDv;
    __half* const o0 = static_cast<__half*>(p.o) + b * os0 + h * os1;
    const float* const lc0 = static_cast<const float*>(p.lc) + bh * p.l * kDk;
    __half* const st0 = static_cast<__half*>(p.st) + b * ts0 + h * ts1;

    // The state arrives [Dk][Dv] and is kept transposed for the products.
    for (int idx = tid; idx < kDk * kDv; idx += kThreads) {
        const int i = idx / kDv;
        const int j = idx - i * kDv;
        sT[j * kSStride + i] = st0[i * ts2 + j];
    }
    __syncthreads();

    // This lane's two rows and two columns, in every accumulator layout below.
    const int r0 = lane >> 2;
    const int r1 = r0 + 8;
    const int cpair = 2 * (lane & 3);
    const int row0 = warp * kRows + r0;
    const int row1 = warp * kRows + r1;

    for (int64_t c = 0; c < p.l; ++c) {
        // ---- keys and gated queries: Aq = Qe Ka^T --------------------------
        stage_panel(qe0 + c * panel_elems, qe, tid);
        stage_panel(ka0 + c * panel_elems, buf, tid);
        cp_async_commit_group();
        cp_async_wait_group<0>();
        __syncthreads();

        float aq[8][4];
#pragma unroll
        for (int t = 0; t < 8; ++t) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                aq[t][e] = 0.0f;
            }
        }
#pragma unroll
        for (int ks = 0; ks < kDk / 16; ++ks) {
            uint32_t af[4];
            ld_a(qe, kPStride, warp * kRows, ks, lane, af);
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                uint32_t bf[4];
                ld_b_n(buf, kPStride, 16 * g, ks, lane, bf);
                mma_h(aq[2 * g], af, bf[0], bf[2]);
                mma_h(aq[2 * g + 1], af, bf[1], bf[3]);
            }
        }

        // The readout keeps s <= r; pack what survives into the A fragments of
        // the readout product (k = s), the score-slot-cast the attention tile
        // uses for P.
        uint32_t pa[4][4];
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int s = 8 * t + cpair;
            aq[t][0] = (s <= row0) ? aq[t][0] : 0.0f;
            aq[t][1] = (s + 1 <= row0) ? aq[t][1] : 0.0f;
            aq[t][2] = (s <= row1) ? aq[t][2] : 0.0f;
            aq[t][3] = (s + 1 <= row1) ? aq[t][3] : 0.0f;
        }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            pa[kk][0] = pack_h(aq[2 * kk][0], aq[2 * kk][1]);
            pa[kk][1] = pack_h(aq[2 * kk][2], aq[2 * kk][3]);
            pa[kk][2] = pack_h(aq[2 * kk + 1][0], aq[2 * kk + 1][1]);
            pa[kk][3] = pack_h(aq[2 * kk + 1][2], aq[2 * kk + 1][3]);
        }

        // ---- X = U - W S, written transposed -------------------------------
        __syncthreads(); ///< every warp has read Ka; W takes its buffer
        stage_panel(w0 + c * panel_elems, buf, tid);
        cp_async_commit_group();
        cp_async_wait_group<0>();
        __syncthreads();

        const __half* const uc = u0 + c * int64_t(kC) * kDv;
#pragma unroll 1
        for (int nh = 0; nh < 2; ++nh) {
            float xa[8][4];
#pragma unroll
            for (int t = 0; t < 8; ++t) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    xa[t][e] = 0.0f;
                }
            }
#pragma unroll
            for (int ks = 0; ks < kDk / 16; ++ks) {
                uint32_t af[4];
                ld_a(buf, kPStride, warp * kRows, ks, lane, af);
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    uint32_t bf[4];
                    ld_b_n(sT, kSStride, 64 * nh + 16 * g, ks, lane, bf);
                    mma_h(xa[2 * g], af, bf[0], bf[2]);
                    mma_h(xa[2 * g + 1], af, bf[1], bf[3]);
                }
            }
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const int j = 64 * nh + 8 * t + cpair;
                const float2 f0 =
                    __half22float2(*reinterpret_cast<const __half2*>(uc + row0 * kDv + j));
                const float2 f1 =
                    __half22float2(*reinterpret_cast<const __half2*>(uc + row1 * kDv + j));
                const float x00 = f0.x - xa[t][0];
                const float x01 = f0.y - xa[t][1];
                const float x10 = f1.x - xa[t][2];
                const float x11 = f1.y - xa[t][3];
                xT[j * kXStride + row0] = __float2half_rn(x00);
                xT[(j + 1) * kXStride + row0] = __float2half_rn(x01);
                xT[j * kXStride + row1] = __float2half_rn(x10);
                xT[(j + 1) * kXStride + row1] = __float2half_rn(x11);
            }
        }
        __syncthreads(); ///< X^T is whole

        // ---- readout: O = scale (Qe S + tril(Aq) X) ------------------------
        __half* const oc = o0 + c * int64_t(kC) * os2;
#pragma unroll 1
        for (int nh = 0; nh < 2; ++nh) {
            float oa[8][4];
#pragma unroll
            for (int t = 0; t < 8; ++t) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    oa[t][e] = 0.0f;
                }
            }
#pragma unroll
            for (int ks = 0; ks < kDk / 16; ++ks) {
                uint32_t af[4];
                ld_a(qe, kPStride, warp * kRows, ks, lane, af);
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    uint32_t bf[4];
                    ld_b_n(sT, kSStride, 64 * nh + 16 * g, ks, lane, bf);
                    mma_h(oa[2 * g], af, bf[0], bf[2]);
                    mma_h(oa[2 * g + 1], af, bf[1], bf[3]);
                }
            }
#pragma unroll
            for (int ks = 0; ks < kC / 16; ++ks) {
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    uint32_t bf[4];
                    ld_b_n(xT, kXStride, 64 * nh + 16 * g, ks, lane, bf);
                    mma_h(oa[2 * g], pa[ks], bf[0], bf[2]);
                    mma_h(oa[2 * g + 1], pa[ks], bf[1], bf[3]);
                }
            }
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const int j = 64 * nh + 8 * t + cpair;
                *reinterpret_cast<__half2*>(oc + row0 * os2 + j) =
                    __floats2half2_rn(p.scale * oa[t][0], p.scale * oa[t][1]);
                *reinterpret_cast<__half2*>(oc + row1 * os2 + j) =
                    __floats2half2_rn(p.scale * oa[t][2], p.scale * oa[t][3]);
            }
        }

        // ---- state: S^T = Lc . (S^T + Ka^T X), two passes over the rows ----
        __syncthreads(); ///< W's buffer is free; the key panel comes back
        stage_panel(ka0 + c * panel_elems, buf, tid);
        cp_async_commit_group();
        cp_async_wait_group<0>();
        __syncthreads();

        const float* const lcc = lc0 + c * kDk;
#pragma unroll 1
        for (int mp = 0; mp < 2; ++mp) {
            const int j0 = 64 * mp + warp * kRows;
            float sa[16][4];
#pragma unroll
            for (int t = 0; t < 16; ++t) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    sa[t][e] = 0.0f;
                }
            }
#pragma unroll
            for (int ks = 0; ks < kC / 16; ++ks) {
                uint32_t af[4];
                ld_a(xT, kXStride, j0, ks, lane, af);
#pragma unroll
                for (int g = 0; g < 8; ++g) {
                    uint32_t bf[4];
                    ld_b_t(buf, kPStride, 16 * ks, 16 * g, lane, bf);
                    mma_h(sa[2 * g], af, bf[0], bf[1]);
                    mma_h(sa[2 * g + 1], af, bf[2], bf[3]);
                }
            }
            const int jrow0 = j0 + r0;
            const int jrow1 = j0 + r1;
#pragma unroll
            for (int t = 0; t < 16; ++t) {
                const int i = 8 * t + cpair;
                const float2 lc = *reinterpret_cast<const float2*>(lcc + i);
                const float2 f0 =
                    __half22float2(*reinterpret_cast<const __half2*>(sT + jrow0 * kSStride + i));
                const float2 f1 =
                    __half22float2(*reinterpret_cast<const __half2*>(sT + jrow1 * kSStride + i));
                *reinterpret_cast<__half2*>(sT + jrow0 * kSStride + i) = __floats2half2_rn(
                    lc.x * (f0.x + sa[t][0]), lc.y * (f0.y + sa[t][1]));
                *reinterpret_cast<__half2*>(sT + jrow1 * kSStride + i) = __floats2half2_rn(
                    lc.x * (f1.x + sa[t][2]), lc.y * (f1.y + sa[t][3]));
            }
        }
        __syncthreads(); ///< the state is whole for the next chunk
    }

    // The state goes back in the caller's layout.
    for (int idx = tid; idx < kDk * kDv; idx += kThreads) {
        const int i = idx / kDv;
        const int j = idx - i * kDv;
        st0[i * ts2 + j] = sT[j * kSStride + i];
    }
}

/// Scratch for the panels, through a private pool: the default pool releases on
/// every free, which on WDDM costs milliseconds a pair, and the route asks for
/// one allocation of tens of megabytes.
cudaMemPool_t wy_pool()
{
    static const cudaMemPool_t pool = [] {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) {
            return cudaMemPool_t{};
        }
        cudaMemPoolProps props{};
        props.allocType = cudaMemAllocationTypePinned;
        props.handleTypes = cudaMemHandleTypeNone;
        props.location.type = cudaMemLocationTypeDevice;
        props.location.id = dev;
        cudaMemPool_t p = nullptr;
        if (cudaMemPoolCreate(&p, &props) != cudaSuccess) {
            cudaGetLastError();
            return cudaMemPool_t{};
        }
        unsigned long long keep = 128ull << 20;
        cudaMemPoolSetAttribute(p, cudaMemPoolAttrReleaseThreshold, &keep);
        return p;
    }();
    return pool;
}

void* wy_alloc(int64_t bytes, cudaStream_t stream)
{
    void* ptr = nullptr;
    const cudaMemPool_t pool = wy_pool();
    if (pool != nullptr &&
        cudaMallocFromPoolAsync(&ptr, size_t(bytes), pool, stream) == cudaSuccess) {
        return ptr;
    }
    cudaGetLastError();
    if (cudaMalloc(&ptr, size_t(bytes)) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return ptr;
}

void wy_free(void* ptr, cudaStream_t stream)
{
    if (ptr == nullptr) {
        return;
    }
    if (cudaFreeAsync(ptr, stream) == cudaSuccess) {
        return;
    }
    cudaGetLastError();
    cudaFree(ptr);
}

/// The recurrent kernel's launch, shared by the panel route and the panel-fed
/// entry the bench and tests drive.
bool wy_recurrent_launch(const LaWyParams& p, cudaStream_t stream)
{
    if (p.w == nullptr || p.u == nullptr || p.qe == nullptr || p.ka == nullptr ||
        p.lc == nullptr || p.o == nullptr || p.st == nullptr) {
        return false;
    }
    if (p.b <= 0 || p.hv <= 0 || p.l <= 0 || p.s != p.l * kC) {
        return false;
    }
    if (p.b * p.hv > 65535) {
        return false;
    }

    static const bool ready = [] {
        return cudaFuncSetAttribute(reinterpret_cast<const void*>(&la_wy_kernel),
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    int(sizeof(__half) * kSmemHalfs)) == cudaSuccess;
    }();
    if (!ready) {
        cudaGetLastError();
        return false;
    }
    const int smem = int(sizeof(__half) * kSmemHalfs);
    la_wy_kernel<<<unsigned(p.b * p.hv), kThreads, smem, stream>>>(p);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace

bool la_wy_recurrent_cuda(const LaWyParams& p, cudaStream_t stream)
{
    return wy_recurrent_launch(p, stream);
}

namespace {
std::atomic<int64_t> g_route_runs{0};
}

int64_t la_wy_route_runs()
{
    return g_route_runs.load(std::memory_order_relaxed);
}

bool la_wy_route_cuda(const LaWyRoute& r, cudaStream_t stream)
{
    if (r.q == nullptr || r.k == nullptr || r.v == nullptr || r.g == nullptr ||
        r.o == nullptr || r.st == nullptr) {
        return false;
    }
    if (r.dk != kDk || r.dv != kDv) {
        return false;
    }
    if (r.b <= 0 || r.hv <= 0 || r.s <= 0 || r.s % kC != 0 || r.hk <= 0) {
        return false;
    }
    if (r.gvk <= 0 || r.hv % r.gvk != 0 || r.ck < 0 || r.ck > kTaps) {
        return false;
    }
    // The output store is an f16 pair and the state's last axis is read
    // elementwise: both need an even row stride and a unit-stride column.
    if (r.ts3 != 1 || (r.os0 | r.os1 | r.os2) % 2 != 0 || r.ts2 % 2 != 0) {
        return false;
    }
    const int64_t l = r.s / kC;
    if (r.b * r.hv > 65535 || r.b * r.hv * l > 2147483647) {
        return false;
    }

    static const bool ready = [] {
        const int panels = int(sizeof(__half) * kPanelHalfs + sizeof(float) * kPanelFloats);
        return cudaFuncSetAttribute(reinterpret_cast<const void*>(&la_wy_panels_kernel),
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    panels) == cudaSuccess &&
               cudaFuncSetAttribute(reinterpret_cast<const void*>(&la_wy_kernel),
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    int(sizeof(__half) * kSmemHalfs)) == cudaSuccess;
    }();
    if (!ready) {
        cudaGetLastError();
        return false;
    }

    const int64_t n = r.b * r.hv * l;
    const int64_t panel_elems = int64_t(kC) * kDk;
    const int64_t bytes = n * (4 * panel_elems * 2 + int64_t(kDk) * 4) + 16;
    unsigned* const base = static_cast<unsigned*>(wy_alloc(bytes, stream));
    if (base == nullptr) {
        return false;
    }
    __half* const w = reinterpret_cast<__half*>(base);
    __half* const u = w + n * panel_elems;
    __half* const qe = u + n * panel_elems;
    __half* const ka = qe + n * panel_elems;
    float* const lc = reinterpret_cast<float*>(ka + n * panel_elems);
    unsigned* const flag = reinterpret_cast<unsigned*>(lc + n * kDk);

    LaPanelsParams pp{};
    pp.r = r;
    pp.w = w;
    pp.u = u;
    pp.qe = qe;
    pp.ka = ka;
    pp.lc = lc;
    pp.flag = flag;

    if (cudaMemsetAsync(flag, 0, sizeof(unsigned), stream) != cudaSuccess) {
        cudaGetLastError();
        wy_free(base, stream);
        return false;
    }
    const int panels_smem = int(sizeof(__half) * kPanelHalfs + sizeof(float) * kPanelFloats);
    la_wy_panels_kernel<<<unsigned(n), kThreads, panels_smem, stream>>>(pp);
    if (cudaGetLastError() != cudaSuccess) {
        wy_free(base, stream);
        return false;
    }

    // The guard. The copy to pageable memory blocks the host until the panel
    // kernel is done, which is exactly the decision point; the stream sync is
    // then free and only makes the ordering explicit.
    unsigned peak = 0;
    if (cudaMemcpyAsync(&peak, flag, sizeof(unsigned), cudaMemcpyDeviceToHost, stream) !=
        cudaSuccess) {
        cudaGetLastError();
        wy_free(base, stream);
        return false;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        cudaGetLastError();
        wy_free(base, stream);
        return false;
    }
    float peak_f = 0.0f;
    std::memcpy(&peak_f, &peak, sizeof(peak_f));
    if (!(peak_f <= kPanelMax)) { ///< also catches a NaN from a broken gate
        wy_free(base, stream);
        return false;
    }

    LaWyParams lp{};
    lp.w = w;
    lp.u = u;
    lp.qe = qe;
    lp.ka = ka;
    lp.lc = lc;
    lp.o = r.o;
    lp.st = r.st;
    lp.b = r.b;
    lp.hv = r.hv;
    lp.s = r.s;
    lp.l = l;
    lp.scale = r.scale;
    lp.os0 = r.os0;
    lp.os1 = r.os1;
    lp.os2 = r.os2;
    lp.ts0 = r.ts0;
    lp.ts1 = r.ts1;
    lp.ts2 = r.ts2;

    const bool ok = wy_recurrent_launch(lp, stream);
    if (ok) {
        g_route_runs.fetch_add(1, std::memory_order_relaxed);
    }
    wy_free(base, stream);
    return ok;
}

} // namespace nnops::backend::cuda
