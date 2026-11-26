#pragma once
/// @file attention_tc.cuh
/// @brief 16-bit Tensor Core prefill — m16n8k16 HMMA over a 64 x 32 tile.
///
/// Why. The FFMA tile (attention_blocked.cuh) is bound by the vector FP32 pipe:
/// 16.2 TFLOPS on this device, of which it reaches a third. A 16-bit cache can
/// instead use the tensor cores, whose fp16 x fp16 -> fp32 rate is twice the
/// vector pipe, and the only way past it. This kernel is the blocked tile's
/// geometry re-cut for `mma.sync.aligned.m16n8k16`: a warp owns sixteen query
/// rows for the whole tile, so the score tile of a row never leaves the warp and
/// the online softmax needs no shared reduction at all.
///
/// Geometry. One block is 64 query rows x 32 keys, 128 threads as 4 warps along
/// the query axis: warp `w` owns rows [16w, 16w + 16), each lane the two rows
/// `lane / 4` and `lane / 4 + 8`, each with the columns `2 * (lane % 4)` and
/// `+1` of every 8-key group — the accumulator layout of the mma itself. The
/// whole score tile therefore lives in the warps that consume it, and the row
/// max/sum are a three-step `fmaxf`/add plus two `shfl_xor` over the four lanes
/// that share a row.
///
/// The tile is small on purpose: 26.1 KB of shared and 168 registers put *three*
/// blocks on a multiprocessor, and occupancy is what this kernel was short of.
/// Measured against the 128 x 64 / one-block tile it replaced, a counter profile
/// read 2 warps a scheduler, 18% issue-active cycles and a tensor pipe busy a
/// third of the time — the arithmetic was never the limit, the dependency chain
/// of one warp was. Three blocks buy three independent chains per scheduler for
/// +45% (f16 1024: 15.0 -> 21.2 TFLOPS), and the finer key tile is closer to the
/// causal triangle besides (6% over it at 64x32 against 11%).
///
/// Q stays in the A fragments (`4 * Dqk / 16` registers) for the whole block,
/// staged through shared once and read back with `ldmatrix.x4`; the staging
/// buffer is the K panel itself, which is what keeps the panel small. K arrives
/// as rows of 8-key groups and feeds the B operand straight out of shared layout
/// `[key][d]` (`ldmatrix.x4`, no transpose: the key is the n axis); V needs the
/// transposing form (`ldmatrix.x4.trans`) because there the key is the k axis.
///
/// P never round-trips through shared: the score accumulator layout and the PV
/// A-fragment layout are the same slots (n-tile 2k is the low half of k-step k),
/// so P is two `cvt.rn.f16x2.f32` per fragment, straight from the exp result.
///
/// Pipeline. K and V are on opposite schedules, which is what lets one of them
/// be single buffered: V is read after the softmax, so its next chunk is issued
/// at the top of the chunk and has the whole chunk to arrive, while K is read
/// the moment the chunk starts and is refilled after the Q.K nest's barrier.
/// Two `cp.async` groups and two barriers a chunk.
///
/// Scope. The 16-bit fast path: no additive bias, no keep mask, no softcap (all
/// per-score gathers), and a head width and value width that are multiples of
/// 64 with the value width at most 128 — the widest the accumulator bank holds
/// at 64 registers a lane. Everything else stays on the FFMA tile or the tiled
/// kernel.

#include "attention_common.cuh"
#include "backend/cuda/matmul/ptx.h"

namespace nnops::backend::cuda::attn {
namespace tc {

constexpr int kBlockM = 64;                    ///< query rows one block owns
constexpr int kBlockN = 32;                    ///< keys one chunk carries
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kRowsPerWarp = kBlockM / kWarps;
constexpr int kScoreNt = kBlockN / 8;          ///< 8-key score groups per warp
constexpr int kQkGroups = kScoreNt / 2;        ///< 16-key groups one ldmatrix feeds
constexpr int kKeySteps = kBlockN / 16;        ///< 16-key PV steps
/// Key stages and value stages differ on purpose. K is read the moment a chunk
/// starts, so its copy has to be issued a chunk early and it keeps the ring;
/// V is only read after the softmax, which is enough window for a single buffer
/// to be refilled at the tail of the chunk. That is what brings a block down
/// from 34.8 KB to 26.1 KB, i.e. three blocks a multiprocessor instead of two.
constexpr int kKStages = 1;
constexpr int kVStages = 2;

static_assert(kRowsPerWarp == 16, "one m16n8k16 tile a warp");
static_assert(kBlockN % 4 == 0, "four threads stage a row");

/// Row stride of a 16-bit panel. Eight halfs of skew put the eight rows one
/// ldmatrix reads at 16 B offsets of 16 mod 128 bytes, so they land on disjoint
/// bank quads; without it a 128-wide head has every row on the same banks.
__host__ __device__ inline int panel_stride(int64_t d) { return static_cast<int>(d) + 8; }

inline int64_t smem_bytes(int64_t dqk, int64_t dv)
{
    return static_cast<int64_t>(sizeof(half)) * kBlockN *
           (kKStages * panel_stride(dqk) + kVStages * panel_stride(dv));
}

/// One mma into four fp32 accumulators, the accumulator as the C operand.
template <class T>
__device__ __forceinline__ void hmma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                     uint32_t b1)
{
#if PTX_SM80_ENABLED
    if constexpr (std::is_same_v<T, __half>) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
#endif
}

/// exp2, one MUFU. The softmax runs in log2 units — the scores are scaled by
/// `log2(e)` on the way in and the head sink converted on the way out — so the
/// exponential is this without the multiply `__expf` hides inside it.
__device__ __forceinline__ float ex2(float x)
{
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

/// Two floats as one packed 16-bit pair — the A fragment of a PV step.
template <class T> __device__ __forceinline__ uint32_t pack2(float lo, float hi)
{
    if constexpr (std::is_same_v<T, __half>) {
        const __half2 v = __floats2half2_rn(lo, hi);
        return *reinterpret_cast<const uint32_t*>(&v);
    } else {
        const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
        return *reinterpret_cast<const uint32_t*>(&v);
    }
}

/// Stage one panel of a chunk: kBlockN rows of K (or of V), D columns each.
/// Four threads copy a row, sixteen bytes at a time; a row past the run, or one
/// the global alignment does not allow an async copy of, is written zero-filled
/// (a zero times a masked probability keeps the output finite, garbage does
/// not).
template <class P, int D, bool KPanel>
__device__ __forceinline__ void stage_panel(const P& p, int64_t b, int64_t hkv, int64_t j0,
                                            half* dst0, int tid)
{
    using KVT = typename P::KVT;
    constexpr int kPer = (D / 8) / 4;
    const int r = tid >> 2;
    const int s0 = (tid & 3) * kPer;
    const KVT* const src = KPanel ? k_row(p, b, hkv, j0 + r) : v_row(p, b, hkv, j0 + r);
    half* const dst = dst0 + static_cast<int64_t>(r) * (D + 8) + s0 * 8;
    if (src != nullptr && (reinterpret_cast<uintptr_t>(src) & 15u) == 0) {
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            cp_async_ca_16B(dst + i * 8, src + s0 * 8 + i * 8);
        }
    } else {
        const unsigned short* const sp =
            src != nullptr ? reinterpret_cast<const unsigned short*>(src + s0 * 8) : nullptr;
        unsigned short* const dp = reinterpret_cast<unsigned short*>(dst);
#pragma unroll
        for (int i = 0; i < kPer * 8; ++i) {
            dp[i] = sp != nullptr ? sp[i] : 0;
        }
    }
}

/// The gate activation, with the choice a template argument: a constant `act`
/// folds the switch in `gate_act_value` away, so one arm is emitted instead of
/// five. That is not what makes the kernel small — the arms are still unrolled
/// once per stored element — it is what makes the *gated* instantiation cost
/// 4.7x the ungated one instead of dwarfing it.
template <int kAct> __device__ __forceinline__ float gate_value(float g)
{
    return gate_act_value(kAct, g);
}

/// The store pass. The four row pointers are formed once a lane instead of once
/// an element, and `kAct < 0` — the common case — is a compile-time form that
/// carries no activation at all.
template <class T, int NT, int kAct, class P>
__device__ __forceinline__ void store_output(const P& p, const float (&o)[NT][4], int64_t obase,
                                             int64_t gbase, int64_t i0, int c0, int row0,
                                             int row1, bool live0, bool live1, float scl0,
                                             float scl1, float norm0, float norm1)
{
    T* const dst0 = p.o + obase + (i0 + row0) * p.o_s;
    T* const dst1 = p.o + obase + (i0 + row1) * p.o_s;
    const T* const grow0 = p.gate + gbase + (i0 + row0) * p.g_s;
    const T* const grow1 = p.gate + gbase + (i0 + row1) * p.g_s;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const int col = 8 * t + c0;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            if (!(half == 0 ? live0 : live1)) {
                continue;
            }
            const float scl = half == 0 ? scl0 : scl1;
            const float norm = half == 0 ? norm0 : norm1;
            T* const dst = half == 0 ? dst0 : dst1;
            float v0 = (half == 0 ? o[t][0] : o[t][2]) * scl * norm;
            float v1 = (half == 0 ? o[t][1] : o[t][3]) * scl * norm;
            if constexpr (kAct >= 0) {
                const T* const grow = half == 0 ? grow0 : grow1;
                v0 *= gate_value<kAct>(s_load(grow + col));
                v1 *= gate_value<kAct>(s_load(grow + col + 1));
            }
            s_store(dst + col, v0);
            s_store(dst + col + 1, v1);
        }
    }
}

/// Pick the store pass's activation once for the block, so the element loop is
/// instantiated once an activation rather than once an element.
template <class T, int NT, class P>
__device__ __forceinline__ void store_output_dispatch(const P& p, const float (&o)[NT][4],
                                                      int64_t obase, int64_t gbase, int64_t i0,
                                                      int c0, int row0, int row1, bool live0,
                                                      bool live1, float scl0, float scl1,
                                                      float norm0, float norm1)
{
#define NNOPS_TC_STORE(ACT)                                                          \
    store_output<T, NT, (ACT)>(p, o, obase, gbase, i0, c0, row0, row1, live0, live1, \
                               scl0, scl1, norm0, norm1)
    if (!p.has_gate) {
        NNOPS_TC_STORE(-1);
        return;
    }
    switch (p.gate_act) {
    case int(EpilogueActivateType::Silu): NNOPS_TC_STORE(int(EpilogueActivateType::Silu)); return;
    case int(EpilogueActivateType::Gelu): NNOPS_TC_STORE(int(EpilogueActivateType::Gelu)); return;
    case int(EpilogueActivateType::Relu): NNOPS_TC_STORE(int(EpilogueActivateType::Relu)); return;
    case int(EpilogueActivateType::Sigmoid):
        NNOPS_TC_STORE(int(EpilogueActivateType::Sigmoid));
        return;
    case int(EpilogueActivateType::Tanh): NNOPS_TC_STORE(int(EpilogueActivateType::Tanh)); return;
    default: NNOPS_TC_STORE(-1); return;
    }
#undef NNOPS_TC_STORE
}

// ============================================================
// The kernel
// ============================================================

/// `kGate` is a contract property, not a per-element value, so it is a template
/// argument: without it the store pass carries no activation at all, which is
/// what keeps the common kernel small enough to stay out of the I-cache's way
/// (five activations x 32 elements x 2 values was three quarters of its code).
template <class T, int DQK, int DV, bool kGate>
__global__ void __launch_bounds__(kThreads, 3) attention_tc_kernel(AttnParams<T, false> p)
{
    using P = AttnParams<T, false>;
    using KVT = typename P::KVT;
    static_assert(std::is_same_v<KVT, T>, "the tensor-core tile reads a 16-bit cache");

    constexpr int kqs = DQK + 8;
    constexpr int kvs = DV + 8;
    constexpr int kNt = DV / 8;

    extern __shared__ half smem[];
    half* const Kb = smem;                                          // kKStages x kBlockN x kqs
    half* const Vb = smem + static_cast<int64_t>(kKStages) * kBlockN * kqs; // kBlockN x kvs
    half* const Qs = smem; ///< the query panel, over the K ring until it is read

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int64_t b = blockIdx.z;
    const int64_t h = blockIdx.x;
    const int64_t i0 = static_cast<int64_t>(gridDim.y - 1 - blockIdx.y) * kBlockM;
    const int64_t hkv = h / p.group;
    const int rows = static_cast<int>(std::min<int64_t>(kBlockM, p.Sq - i0));

    // ---- the run and this lane's two rows ----
    int64_t run = p.Sk;
    if (p.run_ptr != nullptr) {
        const int64_t v = p.run_ptr[b];
        run = (p.run_plus_one ? v + 1 : v);
        run = run <= 0 ? 0 : (run < p.Sk ? run : p.Sk);
    }
    const int64_t pos0 = p.pos_ids != nullptr ? p.pos_ids[b] : p.position_offset;
    const int64_t q0 = pos0 >= 0 ? pos0 : (run - p.Sq);
    const int sink_n =
        (p.sink_tokens > 0 && p.window > 0)
            ? static_cast<int>(p.sink_tokens < run ? p.sink_tokens : run)
            : 0;

    const int r0 = lane >> 2;                        ///< this lane's first row
    const int r1 = r0 + 8;
    const int row0 = warp * kRowsPerWarp + r0;
    const int row1 = warp * kRowsPerWarp + r1;
    const bool live0 = row0 < rows;
    const bool live1 = row1 < rows;
    const int pos_r0 = static_cast<int>(q0 + i0 + row0);
    const int pos_r1 = static_cast<int>(q0 + i0 + row1);
    const int hi0 = p.causal ? std::max(0, std::min(static_cast<int>(run), pos_r0 + 1))
                             : static_cast<int>(run);
    const int hi1 = p.causal ? std::max(0, std::min(static_cast<int>(run), pos_r1 + 1))
                             : static_cast<int>(run);
    const int lo0 = (p.window > 0 && p.causal) ? std::max(0, pos_r0 - static_cast<int>(p.window) + 1) : 0;
    const int lo1 = (p.window > 0 && p.causal) ? std::max(0, pos_r1 - static_cast<int>(p.window) + 1) : 0;

    // ---- the query panel: staged once, then held in the A fragments ----
    {
        constexpr int kPer = (DQK / 8) / 2;    // two threads a row
        const int qr = tid >> 1;
        const int s0 = (tid & 1) * kPer;
        const T* const src =
            qr < rows ? p.q + b * p.q_b + h * p.q_h + (i0 + qr) * p.q_s : nullptr;
        half* const dst = Qs + static_cast<int64_t>(qr) * kqs + s0 * 8;
        if (src != nullptr && (reinterpret_cast<uintptr_t>(src) & 15u) == 0) {
            const uint4* const s4 = reinterpret_cast<const uint4*>(src + s0 * 8);
            uint4* const d4 = reinterpret_cast<uint4*>(dst);
#pragma unroll
            for (int i = 0; i < kPer; ++i) {
                d4[i] = s4[i];
            }
        } else {
            const unsigned short* const sp =
                src != nullptr ? reinterpret_cast<const unsigned short*>(src + s0 * 8) : nullptr;
            unsigned short* const dp = reinterpret_cast<unsigned short*>(dst);
#pragma unroll
            for (int i = 0; i < kPer * 8; ++i) {
                dp[i] = sp != nullptr ? sp[i] : 0;
            }
        }
    }

    uint32_t qf[DQK / 16][4];
    {
        const uint32_t qb = smem_addr_u32(Qs) +
                            static_cast<uint32_t>(warp * kRowsPerWarp) * static_cast<uint32_t>(kqs) * 2u;
        const uint32_t loff = static_cast<uint32_t>((lane & 7) + ((lane >> 3) & 1) * 8) *
                                  static_cast<uint32_t>(kqs) * 2u +
                              static_cast<uint32_t>(lane >> 4) * 16u;
#pragma unroll
        for (int ks = 0; ks < DQK / 16; ++ks) {
            ldmatrix_x4_at(&qf[ks][0], &qf[ks][1], &qf[ks][2], &qf[ks][3],
                           qb + loff + static_cast<uint32_t>(ks * 16) * 2u);
        }
    }
    __syncthreads(); ///< the panel has been read; the K stages take the region

    float o[kNt][4];
#pragma unroll
    for (int t = 0; t < kNt; ++t) {
        o[t][0] = 0.0f;
        o[t][1] = 0.0f;
        o[t][2] = 0.0f;
        o[t][3] = 0.0f;
    }
    float m0 = kNegInf, m1 = kNegInf, l0 = 0.0f, l1 = 0.0f;

    // Where this block's keys end, and -- for a window with no sink -- where the
    // first chunk that can still see a key starts.
    const int64_t kv_end =
        p.causal ? std::max<int64_t>(0, std::min<int64_t>(run, q0 + i0 + rows)) : run;
    int64_t j_begin = 0;
    if (p.causal && p.window > 0 && sink_n == 0) {
        const int64_t lo_min = std::max<int64_t>(0, q0 + i0 - p.window + 1);
        if (lo_min > kBlockN) {
            j_begin = ((lo_min - 1) / kBlockN) * kBlockN;
        }
    }

    if (j_begin < kv_end) {
        stage_panel<P, DQK, true>(p, b, hkv, j_begin, Kb, tid);
        stage_panel<P, DV, false>(p, b, hkv, j_begin, Vb, tid);
    }
    cp_async_commit_group();

    // The lane's offset inside one 16-row (K) or 16-column (V) mma pair: eight
    // rows and the two 8-wide halves of the k step.
    const uint32_t k_loff = static_cast<uint32_t>((lane & 7) + ((lane >> 3) & 1) * 8) *
                                static_cast<uint32_t>(kqs) * 2u +
                            static_cast<uint32_t>(lane >> 4) * 16u;
    const uint32_t v_loff = static_cast<uint32_t>((lane & 7) + ((lane >> 3) & 1) * 8) *
                                static_cast<uint32_t>(kvs) * 2u +
                            static_cast<uint32_t>(lane >> 4) * 16u;

    const int64_t jhi_w =
        p.causal ? std::min<int64_t>(run, q0 + i0 + warp * kRowsPerWarp + kRowsPerWarp) : run;
    // The window every row of this warp shares: a chunk whose keys all fall
    // inside it needs no per-score mask at all, which is every chunk but the
    // diagonal one. `jfull_w` is the first key past the warp's *first* row (its
    // `hi` is the smallest of the sixteen), `lofull_w` the last key before it.
    const int64_t jfull_w =
        p.causal ? std::min<int64_t>(run, q0 + i0 + warp * kRowsPerWarp + 1) : run;
    const int64_t lofull_w =
        (p.window > 0 && p.causal)
            ? std::max<int64_t>(0, q0 + i0 + warp * kRowsPerWarp - p.window + 1)
            : 0;

    int stage = 0;
    for (int64_t j0 = j_begin; j0 < kv_end; j0 += kBlockN) {
        cp_async_wait_group<0>();
        __syncthreads();

        // The two panels are on opposite schedules. V is read late, so the next
        // one can be issued here into the alternate buffer and has the whole
        // chunk to arrive; K is read the moment the chunk starts, so its copy
        // waits for the barrier the Q.K nest leaves behind.
        const int64_t jn = j0 + kBlockN;
        const int ns = stage ^ 1;
        if (jn < kv_end) {
            stage_panel<P, DV, false>(p, b, hkv, jn, Vb + static_cast<int64_t>(ns) * kBlockN * kvs,
                                      tid);
        }
        cp_async_commit_group();

        const uint32_t kb = smem_addr_u32(Kb);
        const uint32_t vb = smem_addr_u32(Vb) +
                            static_cast<uint32_t>(stage) * static_cast<uint32_t>(kBlockN * kvs) * 2u;

        // ---- Q.K: 8x8 score groups, Dqk/16 mma steps over the head width ----
        float s[kScoreNt][4];
#pragma unroll
        for (int t = 0; t < kScoreNt; ++t) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                s[t][c] = 0.0f;
            }
        }
#pragma unroll
        for (int ks = 0; ks < DQK / 16; ++ks) {
            uint32_t kf[kQkGroups][4];
#pragma unroll
            for (int g = 0; g < kQkGroups; ++g) {
                ldmatrix_x4_at(&kf[g][0], &kf[g][1], &kf[g][2], &kf[g][3],
                               kb + k_loff + static_cast<uint32_t>(16 * g * kqs + ks * 16) * 2u);
            }
#pragma unroll
            for (int g = 0; g < kQkGroups; ++g) {
                // A group wholly past this warp's last visible key is masked
                // everywhere: skip its mma (half of a 32-key chunk, a quarter of
                // the 64-key one this tiling started as).
                if (j0 + 16 * g < jhi_w) {
                    hmma<T>(s[2 * g], qf[ks], kf[g][0], kf[g][2]);
                }
                if (j0 + 16 * g + 8 < jhi_w) {
                    hmma<T>(s[2 * g + 1], qf[ks], kf[g][1], kf[g][3]);
                }
            }
        }

        __syncthreads(); ///< every warp has read the K panel; refill it
        if (jn < kv_end) {
            stage_panel<P, DQK, true>(p, b, hkv, jn, Kb, tid);
        }
        cp_async_commit_group();

        // ---- scale, visibility, online softmax ----
        const float sc = p.scale * 1.4426950408889634f;  // log2(e): the exp is an ex2
        // A fully visible chunk masks nothing: its max is over the raw scores, and
        // since `sc > 0` the scaled max is the max scaled, one multiply a row
        // rather than one a score. The mask's compares and selects were the bulk
        // of this kernel's integer instructions (the ALU pipe is half rate here).
        const bool full = j0 + kBlockN <= jfull_w && j0 >= lofull_w;
        float mx0 = kNegInf, mx1 = kNegInf;
        const int jc0 = static_cast<int>(j0) + 2 * (lane & 3);
        if (full) {
#pragma unroll
            for (int t = 0; t < kScoreNt; ++t) {
                mx0 = fmaxf(mx0, fmaxf(s[t][0], s[t][1]));
                mx1 = fmaxf(mx1, fmaxf(s[t][2], s[t][3]));
            }
        } else {
#pragma unroll
            for (int t = 0; t < kScoreNt; ++t) {
                const int jc = jc0 + 8 * t;
                s[t][0] = (live0 && jc < hi0 && (jc >= lo0 || jc < sink_n)) ? s[t][0] : kNegInf;
                s[t][1] =
                    (live0 && jc + 1 < hi0 && (jc + 1 >= lo0 || jc + 1 < sink_n)) ? s[t][1] : kNegInf;
                s[t][2] = (live1 && jc < hi1 && (jc >= lo1 || jc < sink_n)) ? s[t][2] : kNegInf;
                s[t][3] = (live1 && jc + 1 < hi1 && (jc + 1 >= lo1 || jc + 1 < sink_n)) ? s[t][3]
                                                                                       : kNegInf;
                mx0 = fmaxf(mx0, fmaxf(s[t][0], s[t][1]));
                mx1 = fmaxf(mx1, fmaxf(s[t][2], s[t][3]));
            }
        }
        mx0 *= sc;
        mx1 *= sc;
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));

        const float mn0 = fmaxf(m0, mx0);
        const float mn1 = fmaxf(m1, mx1);
        const float rs0 = (m0 == kNegInf) ? 0.0f : ex2(m0 - mn0);
        const float rs1 = (m1 == kNegInf) ? 0.0f : ex2(m1 - mn1);
        m0 = mn0;
        m1 = mn1;
        const bool ok0 = mn0 != kNegInf;
        const bool ok1 = mn1 != kNegInf;

        float sum0 = 0.0f, sum1 = 0.0f;
        if (full) {
            // Every row of this warp has a finite max here, so the empty-row
            // guard has nothing to do.
#pragma unroll
            for (int t = 0; t < kScoreNt; ++t) {
                s[t][0] = ex2(fmaf(s[t][0], sc, -mn0));
                s[t][1] = ex2(fmaf(s[t][1], sc, -mn0));
                s[t][2] = ex2(fmaf(s[t][2], sc, -mn1));
                s[t][3] = ex2(fmaf(s[t][3], sc, -mn1));
                sum0 += s[t][0] + s[t][1];
                sum1 += s[t][2] + s[t][3];
            }
        } else {
#pragma unroll
            for (int t = 0; t < kScoreNt; ++t) {
                s[t][0] = ok0 ? ex2(fmaf(s[t][0], sc, -mn0)) : 0.0f;
                s[t][1] = ok0 ? ex2(fmaf(s[t][1], sc, -mn0)) : 0.0f;
                s[t][2] = ok1 ? ex2(fmaf(s[t][2], sc, -mn1)) : 0.0f;
                s[t][3] = ok1 ? ex2(fmaf(s[t][3], sc, -mn1)) : 0.0f;
                sum0 += s[t][0] + s[t][1];
                sum1 += s[t][2] + s[t][3];
            }
        }
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
        l0 = l0 * rs0 + sum0;
        l1 = l1 * rs1 + sum1;

#pragma unroll
        for (int t = 0; t < kNt; ++t) {
            o[t][0] *= rs0;
            o[t][1] *= rs0;
            o[t][2] *= rs1;
            o[t][3] *= rs1;
        }

        // P as the A fragments of P.V: n-tile 2k is the low half of k-step k.
        uint32_t pa[kKeySteps][4];
#pragma unroll
        for (int kk = 0; kk < kKeySteps; ++kk) {
            pa[kk][0] = pack2<T>(s[2 * kk][0], s[2 * kk][1]);
            pa[kk][1] = pack2<T>(s[2 * kk][2], s[2 * kk][3]);
            pa[kk][2] = pack2<T>(s[2 * kk + 1][0], s[2 * kk + 1][1]);
            pa[kk][3] = pack2<T>(s[2 * kk + 1][2], s[2 * kk + 1][3]);
        }

        // ---- P.V: value columns in 16-wide pairs, 16 keys a step ----
#pragma unroll
        for (int kk = 0; kk < kKeySteps; ++kk) {
            uint32_t vf[kNt / 2][4];
#pragma unroll
            for (int g = 0; g < kNt / 2; ++g) {
                ldmatrix_x4_trans_at(&vf[g][0], &vf[g][1], &vf[g][2], &vf[g][3],
                                     vb + v_loff +
                                         static_cast<uint32_t>(16 * kk * kvs + 16 * g) * 2u);
            }
#pragma unroll
            for (int g = 0; g < kNt / 2; ++g) {
                hmma<T>(o[2 * g], pa[kk], vf[g][0], vf[g][1]);
                hmma<T>(o[2 * g + 1], pa[kk], vf[g][2], vf[g][3]);
            }
        }

        stage ^= 1;
    }

    // ---- the head sink, normalization and the gate ----
    float scl0 = 1.0f, scl1 = 1.0f, norm0 = 0.0f, norm1 = 0.0f;
    if (p.sink != nullptr) {
        const float sv = p.sink[h] * 1.4426950408889634f;
        const float a0 = fmaxf(m0, sv);
        const float a1 = fmaxf(m1, sv);
        scl0 = (m0 == kNegInf) ? 0.0f : ex2(m0 - a0);
        scl1 = (m1 == kNegInf) ? 0.0f : ex2(m1 - a1);
        l0 = l0 * scl0 + ex2(sv - a0);
        l1 = l1 * scl1 + ex2(sv - a1);
    }
    norm0 = l0 > 0.0f ? 1.0f / l0 : 0.0f;
    norm1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;

    const int64_t obase = b * p.o_b + h * p.o_h;
    if constexpr (kGate) {
        store_output_dispatch<T, kNt>(p, o, obase, b * p.g_b + h * p.g_h, i0, 2 * (lane & 3),
                                      row0, row1, live0, live1, scl0, scl1, norm0, norm1);
    } else {
        store_output<T, kNt, -1>(p, o, obase, 0, i0, 2 * (lane & 3), row0, row1, live0, live1,
                                 scl0, scl1, norm0, norm1);
    }
}

} // namespace tc
} // namespace nnops::backend::cuda::attn
