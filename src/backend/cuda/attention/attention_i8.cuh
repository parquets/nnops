#pragma once
/// @file attention_i8.cuh
/// @brief The int8 tensor-core prefill tile — m16n8k32 s8 Q·Kᵀ (and u8×s8 P·V)
///        over the same 64 x 32 block the 16-bit tile uses.
///
/// Why. The f16 tile (attention_f16.cuh) is what a 16-bit cache gets; an s8
/// cache had no tile of its own and fell back to the tiled kernel, which widens
/// every element on read and multiplies in FFMA. The integer atom is twice the
/// 16-bit one on this device and the cache arrives as the operand already, so
/// this is where an int8 cache stops paying a widening tax.
///
/// Contract. Same as the CPU flash route's cfg.quantize_q / cfg.quantize_pv:
///
///   - Q·Kᵀ: each query row is quantized on the fly with its own symmetric
///     scale (the row's absmax / 127), the cache's keys are the s8 bytes as
///     stored, and the int32 score is dequantized by
///     scale · s_q[row] · s_k[key] — the k scale rides the output axis, so it
///     multiplies the score column by column. The rounding is the CPU's
///     (round-to-nearest-even on x·(1/scale), clamped to an s8).
///   - P·V, when asked for, puts the [0,1] probability on a 255-level unsigned
///     grid with the cache's per-key V scale folded in — normalized by the
///     tile's largest so the grid stays covered — and dequantizes the int32
///     result by max(scale)/255 onto the float accumulator. This is exactly
///     the CPU's softmax_row_exp_quant and its sc_tile, so the two routes
///     differ only in summation order.
///
/// Without cfg.quantize_pv the value product stays 16-bit: V is widened into
/// the shared panel with its per-key scale and P is packed from the same
/// registers, i.e. the f16 tile's P·V. That needs a 16-bit query dtype, which
/// is why an f32 query only takes this kernel when both products are integer.
///
/// Geometry. 64 query rows x 32 keys, 128 threads as 4 warps along the query
/// axis, one warp per 16 rows — the f16 tile's block, re-cut for an atom with
/// 32 k. Q is quantized once into shared and held in the A fragments
/// (`4·Dqk/32` registers, half the 16-bit tile's), K is copied as bytes with
/// `cp.async`, and every panel row is padded 16 B so the eight rows of an
/// ldmatrix land on disjoint bank quads.
///
/// Staging is shaped by what the counters say. The f32 accumulator (Dv/8 tiles
/// of four floats) is what bounds the warps this kernel can run: 64 registers of
/// accumulator leave 128 a thread, and that is four blocks of four warps — 16 of
/// the SM's 48 — which is why the panels are sized to let an SM hold four (see
/// `kVtRow` and `p_in_q`). Unlike the 16-bit tile, which is throttled by the
/// tensor pipe, this one is latency-bound: no pipe passes 30% of its peak, and
/// occupancy is the lever that moves it (12 warps gave 34.5 TFLOPS at a 1024-key
/// causal run, 16 gave 35.5, eight gave 27.9). So each of the four reads a chunk
/// needs is resolved once a chunk and stepped by a constant rather than per row:
/// the run is one or two contiguous segments, so `base + r * step` is a row, and
/// `rows == kBlockN` inside one segment takes the fill with no bounds test and
/// no zero path (that is every chunk but the last a block walks, and a
/// straddling or tail chunk keeps the general accessor below). Two things left
/// the chunk loop entirely: the value run's transpose — a per-call pass into a
/// key-minor scratch, `[column][key]`, so the integer B panel is fed by
/// `cp.async` like the 16-bit tile's instead of four dependent 8 B gathers and a
/// sixteen-`__byte_perm` turn a chunk — and the per-key V scale, staged into
/// shared with the 255 level and the chunk's max already folded in, where the P
/// quantization used to gather it from global a score.
///
/// P never takes the fragment shuffle: the score accumulator's four values a
/// lane holds are two keys of one row, while the integer A fragment wants four
/// consecutive keys, so the warp writes its 16x32 byte tile into a staging
/// panel and reads it back with one `ldmatrix.x4` — the same move the integer
/// GEMM makes, and it stays inside the warp (one `__syncwarp`, no barrier).
///
/// Scope mirrors the 16-bit tile: no additive bias, no keep mask, no softcap,
/// and the same (Dqk, Dv) shapes the launcher instantiates.

#include "attention_common.cuh"
#include "backend/cuda/matmul/ptx.h"

namespace nnops::backend::cuda::attn {
namespace i8 {

constexpr int kBlockM = 64;                     ///< query rows one block owns
constexpr int kBlockN = 32;                     ///< keys one chunk carries (one k step)
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kRowsPerWarp = kBlockM / kWarps;  ///< 16: one m16n8k32 tile a warp
constexpr int kScoreNt = kBlockN / 8;           ///< 8-key score groups per warp
constexpr int kKStages = 1;
constexpr int kVStages = 2;
/// The Q and K panel rows carry 16 B of pad, and so does the P staging row: a
/// panel row is `Dqk` bytes (a P row 32), so the eight rows one fragment read
/// touches would otherwise land on the same bank quads. The value panel needs
/// none of it — one `ldmatrix.x2` there reads eight 32-key rows as 256
/// contiguous bytes, which is already the conflict-free case — and the block
/// wants the four blocks an SM will give it, so it keeps the bare stride.
constexpr int kPad = 16;
constexpr int kVtRow = kBlockN;                 ///< Vᵀ panel row, bytes
constexpr int kPs = kBlockN + kPad;             ///< P staging row, bytes

static_assert(kRowsPerWarp == 16, "one m16n8k32 tile a warp");
static_assert(kBlockN == 32, "the integer atom consumes one chunk's keys in one k step");

/// Whether the P staging panel fits in the Q panel beside the K ring's second
/// stage. It does for every shape but the narrow 64-key one, whose Q panel is
/// only a quarter over a K stage — and that shape has shared memory to spare.
constexpr bool p_in_q(int64_t dqk)
{
    return static_cast<int64_t>(kBlockM) * (dqk + kPad) >=
           static_cast<int64_t>(kBlockN) * (dqk + kPad) + static_cast<int64_t>(kBlockM) * kPs;
}

constexpr int64_t smem_bytes(int64_t dqk, int64_t dv, bool pv8)
{
    const int64_t q = static_cast<int64_t>(kBlockM) * (dqk + kPad);
    const int64_t qsc = static_cast<int64_t>(kBlockM) * sizeof(float);
    const int64_t k = static_cast<int64_t>(kKStages) * kBlockN * (dqk + kPad);
    const int64_t v = pv8 ? static_cast<int64_t>(kVStages) * dv * kVtRow
                          : static_cast<int64_t>(kVStages) * kBlockN * (dv + 8) * sizeof(half);
    const int64_t p = (pv8 && !p_in_q(dqk)) ? static_cast<int64_t>(kBlockM) * kPs : 0;
    const int64_t vsc = pv8 ? static_cast<int64_t>(kVStages) * sizeof(float) : 0;
    const int64_t vf = pv8 ? static_cast<int64_t>(kVStages) * kBlockN * sizeof(float) : 0;
    const int64_t ksc = static_cast<int64_t>(kVStages) * kBlockN * sizeof(float);
    return (q + qsc + k + v + p + vsc + vf + ksc + 15) & ~static_cast<int64_t>(15);
}

/// Blocks an SM can hold at this shape, which is also what the launch bounds
/// ask the register allocator for: aiming at four where the panel only fits
/// three trades a clean 168-register kernel for a spilled 128-register one.
constexpr int blocks_per_sm(int64_t dqk, int64_t dv, bool pv8)
{
    const int64_t per_block = smem_bytes(dqk, dv, pv8) + 1024; ///< + driver reserve
    return per_block * 4 <= 102400 ? 4 : 3;
}

// ============================================================
// Staging — the row base is resolved a chunk, never a row
// ============================================================

/// How many of a chunk's rows lie inside the tensor.
__device__ __forceinline__ int chunk_rows(int64_t j0, int64_t limit)
{
    const int64_t n = limit - j0;
    return n <= 0 ? 0 : (n < kBlockN ? static_cast<int>(n) : kBlockN);
}

/// One segment of the run as the chunk sees it: the row `j0` base and the step
/// to the next row. A chunk inside a single segment needs nothing else.
struct Src {
    const int8_t* base;
    int64_t step;
};

/// Eight query elements out of two 128-bit loads — one for a 2-byte dtype,
/// two for f32 — widened to floats. The quantization pass reads the row twice
/// (once for absmax, once to scale), so the two loads the f32 case needs are
/// still half of the eight `s_load`s it used to issue.
template <class T>
__device__ __forceinline__ void load8(const T* src, float (&x)[8])
{
    if constexpr (std::is_same_v<T, float>) {
        const float4 a = v4_load(src);
        const float4 b = v4_load(src + 4);
        x[0] = a.x;
        x[1] = a.y;
        x[2] = a.z;
        x[3] = a.w;
        x[4] = b.x;
        x[5] = b.y;
        x[6] = b.z;
        x[7] = b.w;
    } else {
        const pack8<T> a = v4_load(src);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            x[i] = elem_to_float(a.data[i]);
        }
    }
}

__host__ __device__ __forceinline__ Src seg_src(const int8_t* past, int64_t past_step,
                                                const int8_t* main_, int64_t main_step,
                                                int64_t past_rows, int64_t j0)
{
    if (j0 < past_rows) {
        return {past != nullptr ? past + j0 * past_step : nullptr, past_step};
    }
    const int64_t j = j0 - past_rows;
    return {main_ != nullptr ? main_ + j * main_step : nullptr, main_step};
}

/// Whether the branch-free fill can cover a chunk exactly: the threads must
/// split into whole segments and whole rows. A 192-key head gives twelve
/// segments, which divides neither, and takes the row walk below instead.
template <int DQK>
constexpr bool k_fill_exact = (kThreads % (DQK / 16)) == 0 &&
                              (kBlockN % (kThreads / (DQK / 16))) == 0;

/// The interior K fill: one `cp.async` 16 B segment a thread, no row test.
template <int DQK>
__device__ __forceinline__ void stage_k_fast(const int8_t* src, int64_t step, int8_t* dst,
                                             int tid)
{
    static_assert(k_fill_exact<DQK>, "the branch-free fill must cover the chunk exactly");
    constexpr int kSegs = DQK / 16;
    constexpr int kRowsPerPass = kThreads / kSegs;
    constexpr int kStride = DQK + kPad;
    const int s = (tid % kSegs) * 16;
    const int r0 = tid / kSegs;
#pragma unroll
    for (int q = 0; q < kBlockN / kRowsPerPass; ++q) {
        const int r = r0 + q * kRowsPerPass;
        cp_async_ca_16B(dst + static_cast<int64_t>(r) * kStride + s,
                        src + static_cast<int64_t>(r) * step + s);
    }
}

/// K arrives as the bytes the cache holds: one `cp.async` 16 B segment a
/// thread, straight into the panel. A row past the run, or one whose global
/// address is not 16 B aligned, is written synchronously (zero-filled when
/// absent). This is the general accessor: a tail chunk and one that straddles
/// the past segment are the only callers.
template <class P, int DQK>
__device__ __forceinline__ void stage_k(const P& p, int64_t b, int64_t hkv, int64_t j0,
                                        int8_t* dst, int tid)
{
    constexpr int kSegs = DQK / 16;
    constexpr int kRowsPerPass = kThreads / kSegs;
    constexpr int kStride = DQK + kPad;
    const int s = (tid % kSegs) * 16;
    for (int r = tid / kSegs; r < kBlockN; r += kRowsPerPass) {
        int8_t* const d = dst + static_cast<int64_t>(r) * kStride + s;
        const int8_t* const src = k_row(p, b, hkv, j0 + r);
        if (src != nullptr && (reinterpret_cast<uintptr_t>(src + s) & 15u) == 0) {
            cp_async_ca_16B(d, src + s);
        } else {
            const int8_t* const sp = src != nullptr ? src + s : nullptr;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                d[i] = sp != nullptr ? sp[i] : int8_t(0);
            }
        }
    }
}

/// The integer B panel from the pre-transposed value run: column `d`'s 32 keys
/// are contiguous there, so a panel row is two 16 B segments and no transpose
/// runs in the tile. The scratch is zero-padded to a 32-key row stride, so a
/// chunk past the run reads zeros rather than faulting.
template <int DV>
__device__ __forceinline__ void stage_vt_prep(const int8_t* vt, int64_t j0, int64_t skt, int8_t* dst,
                                              int tid)
{
    constexpr int kSegs = kBlockN / 16;
    constexpr int kRowsPerPass = kThreads / kSegs;
    constexpr int kStride = kVtRow;
    static_assert(DV % kRowsPerPass == 0, "the fill must divide the value columns");
    const int s = (tid % kSegs) * 16;
    const int r0 = tid / kSegs;
#pragma unroll
    for (int q = 0; q < DV / kRowsPerPass; ++q) {
        const int r = r0 + q * kRowsPerPass;
        cp_async_ca_16B(dst + static_cast<int64_t>(r) * kStride + s,
                        vt + static_cast<int64_t>(r) * skt + j0 + s);
    }
}

/// The chunk's per-key P level factor — the cache's V scale over the chunk's
/// largest, times the 255 levels — folded once a chunk by one warp, which also
/// publishes the matching dequant scale. The P quantization then reads a
/// `float2` out of shared a score group where it used to gather the scale from
/// global a score.
template <class P>
__device__ __forceinline__ void stage_vfact(const P& p, int64_t vsc_off, int64_t j0,
                                            float* dst_deq, float* dst_f, int tid)
{
    if (tid >= 32) {
        return;
    }
    float v = 1.0f;
    if (p.vscale != nullptr) {
        // The head offset only addresses a per-token array; a per-tensor scale
        // is one float and every head reads it as-is.
        const int64_t j = j0 + tid;
        v = (p.scale_per_token && j < p.Sk) ? p.vscale[vsc_off + j] : p.vscale[0];
    }
    float mx = v;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    }
    mx = mx > 0.0f ? mx : 1.0f;
    dst_f[tid] = v * (255.0f / mx);
    if (tid == 0) {
        *dst_deq = mx / 255.0f;
    }
}

/// The chunk's cache scales, staged beside the panels: one thread a key. The
/// dequantization reads them out of shared instead of issuing a dependent
/// global load a score, and the double buffer costs 256 B.
template <class P>
__device__ __forceinline__ void stage_ksc(const P& p, const float* base, int64_t j0, float* dst,
                                          int tid)
{
    if (tid >= kBlockN) {
        return;
    }
    const int64_t j = j0 + tid;
    dst[tid] = p.scale_per_token && j < p.Sk ? base[j] : base[0];
}

/// The value run widened once per call, in the cache's own order. The tile's
/// fp P·V then reads a 16-bit panel with nothing left to convert or scale, which
/// is what the f16 tile gets from an fp cache. One block a run row, a thread a
/// value column (Dv <= 128 in this tile), so both sides are coalesced.
template <class T, class P>
__global__ void attention_v_prep_kernel(P p, T* out)
{
    const int64_t row = blockIdx.x;                 // (b·Hkv + hkv)·Sk + j
    const int64_t bh = row / p.Sk;
    const int64_t j = row - bh * p.Sk;
    const int64_t b = bh / p.Hkv;
    const int64_t hkv = bh - b * p.Hkv;
    const int8_t* const src = v_row(p, b, hkv, j);
    const float vs = kv_scale_at(p.vscale, p.scale_per_token, b, hkv, j, p.Hkv, p.Sk);
    T* const dst = out + row * p.Dv;
    for (int64_t d = threadIdx.x; d < p.Dv; d += blockDim.x) {
        dst[d] = float_to_elem<T>(src != nullptr ? static_cast<float>(src[d]) * vs : 0.0f);
    }
}

/// The value run turned over once per call into `[column][key]`, the pose the
/// integer B fragment reads, zero-padded to a 32-key row stride. One block a
/// (32-key, all-column) tile turns it through shared: 16 B read a row, 16 B
/// written a column, so both sides move a sector a lane.
template <class P>
__global__ void attention_vt_prep_kernel(P p, int8_t* out, int64_t skt)
{
    __shared__ int8_t tile[128][33]; ///< [value column][key], 33 so no two share a bank
    const int64_t b = blockIdx.z;
    const int64_t hkv = blockIdx.y;
    const int64_t j0 = static_cast<int64_t>(blockIdx.x) * kBlockN;
    const int ncol = static_cast<int>(p.Dv);
    const int nseg = ncol / 16;

    for (int idx = threadIdx.x; idx < kBlockN * nseg; idx += blockDim.x) {
        const int r = idx / nseg;
        const int c = (idx - r * nseg) * 16;
        uint4 x = make_uint4(0u, 0u, 0u, 0u);
        const int8_t* const src = v_row(p, b, hkv, j0 + r);
        if (src != nullptr) {
            x = *reinterpret_cast<const uint4*>(src + c);
        }
        const int8_t* const xb = reinterpret_cast<const int8_t*>(&x);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            tile[c + i][r] = xb[i];
        }
    }
    __syncthreads();

    int8_t* const base = out + (b * p.Hkv + hkv) * p.Dv * skt;
    constexpr int kWide = kBlockN / 16;
    for (int idx = threadIdx.x; idx < ncol * kWide; idx += blockDim.x) {
        const int c = idx / kWide;
        const int k16 = (idx - c * kWide) * 16;
        int8_t xb[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            xb[i] = tile[c][k16 + i];
        }
        *reinterpret_cast<uint4*>(base + static_cast<int64_t>(c) * skt + j0 + k16) =
            *reinterpret_cast<const uint4*>(xb);
    }
}

/// Stage a chunk of that buffer into the 16-bit panel.
template <class T, int DV>
__device__ __forceinline__ void stage_v_prep(const T* vbase, int64_t j0, int64_t sk, T* dst,
                                             int tid)
{
    constexpr int kSegs = DV / 8;                   // 16 B segments a row
    constexpr int kRowsPerPass = kThreads / kSegs;
    const int s = (tid % kSegs) * 8;
    const int r0 = tid / kSegs;
    const int rows = chunk_rows(j0, sk);
    if (rows == kBlockN) {
        const T* const src = vbase + j0 * DV;
#pragma unroll
        for (int q = 0; q < kBlockN / kRowsPerPass; ++q) {
            const int r = r0 + q * kRowsPerPass;
            cp_async_ca_16B(dst + static_cast<int64_t>(r) * (DV + 8) + s, src + r * DV + s);
        }
        return;
    }
    for (int r = r0; r < kBlockN; r += kRowsPerPass) {
        T* const d = dst + static_cast<int64_t>(r) * (DV + 8) + s;
        if (r < rows) {
            cp_async_ca_16B(d, vbase + (j0 + r) * DV + s);
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                d[i] = float_to_elem<T>(0.0f);
            }
        }
    }
}

// ============================================================
// The kernel
// ============================================================

template <class T, int DQK, int DV, bool kPv8>
__global__ void __launch_bounds__(kThreads, blocks_per_sm(DQK, DV, kPv8))
    attention_kernel(AttnParams<T, true> p)
{
    using P = AttnParams<T, true>;

    constexpr int kqs = DQK + kPad;             ///< Q / K panel row, bytes
    constexpr int kVh = DV + 8;                 ///< 16-bit V panel row, elements
    constexpr int kNt = DV / 8;
    constexpr int kPvSteps = kBlockN / 16;      ///< 16-bit P·V steps
    constexpr int kVbytes = kPv8 ? DV * kVtRow : kBlockN * kVh * static_cast<int>(sizeof(T));

    extern __shared__ char tc8_smem[];
    int8_t* const Qs = reinterpret_cast<int8_t*>(tc8_smem);
    float* const Qsc = reinterpret_cast<float*>(Qs + static_cast<int64_t>(kBlockM) * kqs);
    int8_t* const Kb = reinterpret_cast<int8_t*>(Qsc + kBlockM);
    /// The K ring's second buffer is the Q panel, which is dead once the A
    /// fragments are in registers (it is twice a K stage, since the block owns
    /// twice the query rows a chunk holds keys). That buys the double buffer
    /// without an extra byte of shared memory, which is what lets the refill
    /// ride the chunk's one barrier — and the P staging panel, a quarter of the
    /// rest, lives in the other half of the same dead panel.
    int8_t* const Kb2 = Qs;
    constexpr int kKStageBytes = kBlockN * kqs;
    constexpr bool kPinQ = p_in_q(DQK);
    int8_t* const Vb = reinterpret_cast<int8_t*>(Kb + static_cast<int64_t>(kKStages) * kKStageBytes);
    int8_t* const Pb = kPinQ ? Qs + kKStageBytes : Vb + static_cast<int64_t>(kVStages) * kVbytes;
    float* const Vsc = reinterpret_cast<float*>(
        Vb + static_cast<int64_t>(kVStages) * kVbytes + (kPv8 && !kPinQ ? kBlockM * kPs : 0));
    float* const Vf = Vsc + (kPv8 ? kVStages : 0);
    float* const Ksc = Vf + (kPv8 ? static_cast<int64_t>(kVStages) * kBlockN : 0);
    T* const Vh = reinterpret_cast<T*>(Vb);

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

    const int r0 = lane >> 2;
    const int r1 = r0 + 8;
    const int row0 = warp * kRowsPerWarp + r0;
    const int row1 = warp * kRowsPerWarp + r1;
    const bool live0 = row0 < rows;
    const bool live1 = row1 < rows;
    const int pos_r0 = static_cast<int>(q0 + i0 + row0);
    const int pos_r1 = static_cast<int>(q0 + i0 + row1);
    const int hi0 =
        p.causal ? std::max(0, std::min(static_cast<int>(run), pos_r0 + 1)) : static_cast<int>(run);
    const int hi1 =
        p.causal ? std::max(0, std::min(static_cast<int>(run), pos_r1 + 1)) : static_cast<int>(run);
    const int lo0 = (p.window > 0 && p.causal) ? std::max(0, pos_r0 - static_cast<int>(p.window) + 1) : 0;
    const int lo1 = (p.window > 0 && p.causal) ? std::max(0, pos_r1 - static_cast<int>(p.window) + 1) : 0;

    // ---- the query: one symmetric int8 scale per row, then the A fragments ----
    {
        const int qr = tid >> 1;                 ///< two threads a row
        const int h0 = (tid & 1) * (DQK / 2);
        const T* const src =
            qr < rows ? p.q + b * p.q_b + h * p.q_h + (i0 + qr) * p.q_s + h0 : nullptr;
        float amax = 0.0f;
        if (src != nullptr) {
#pragma unroll
            for (int d = 0; d < DQK / 2; d += 8) {
                float x[8];
                load8(src + d, x);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    amax = fmaxf(amax, fabsf(x[i]));
                }
            }
        }
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        const float sq = amax > 0.0f ? amax / 127.0f : 1.0f;
        const float inv = 1.0f / sq;
        int8_t* const dst = Qs + static_cast<int64_t>(qr) * kqs + h0;
        if (src != nullptr) {
#pragma unroll
            for (int d = 0; d < DQK / 2; d += 8) {
                float x[8];
                load8(src + d, x);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int q = __float2int_rn(x[i] * inv);
                    dst[d + i] = static_cast<int8_t>(max(-128, min(127, q)));
                }
            }
        } else {
#pragma unroll
            for (int d = 0; d < DQK / 2; ++d) {
                dst[d] = int8_t(0);
            }
        }
        if ((tid & 1) == 0) {
            Qsc[qr] = sq;
        }
    }
    __syncwarp(); ///< this warp staged exactly its own sixteen rows

    uint32_t qf[DQK / 32][4];
    {
        const uint32_t qb = smem_addr_u32(Qs) +
                            static_cast<uint32_t>(warp * kRowsPerWarp) *
                                static_cast<uint32_t>(kqs);
        const uint32_t loff = static_cast<uint32_t>(lane & 15) * static_cast<uint32_t>(kqs) +
                              static_cast<uint32_t>((lane >> 4) * 16);
#pragma unroll
        for (int ks = 0; ks < DQK / 32; ++ks) {
            ldmatrix_x4_at(&qf[ks][0], &qf[ks][1], &qf[ks][2], &qf[ks][3],
                           qb + loff + static_cast<uint32_t>(ks * 32));
        }
    }

    float o[kNt][4];
#pragma unroll
    for (int t = 0; t < kNt; ++t) {
        o[t][0] = 0.0f;
        o[t][1] = 0.0f;
        o[t][2] = 0.0f;
        o[t][3] = 0.0f;
    }
    float m0 = kNegInf, m1 = kNegInf, l0 = 0.0f, l1 = 0.0f;

    const bool kcol = p.kscale != nullptr && p.scale_per_token;
    const float kall = (p.kscale != nullptr && !p.scale_per_token) ? p.kscale[0] : 1.0f;
    const bool vcol = p.vscale != nullptr && p.scale_per_token;
    // The per-(batch, kv head) base of both scale arrays, resolved once.
    const float* const ksc_base = p.kscale != nullptr ? p.kscale + (b * p.Hkv + hkv) * p.Sk : nullptr;
    const int64_t vsc_off = (b * p.Hkv + hkv) * p.Sk;
    // The key run's segment bases, likewise. The value run is read through the
    // staged scratch, so only this one is addressed in the tile.
    const int8_t* const pk_b0 = p.pk != nullptr ? p.pk + b * p.pk_b + hkv * p.pk_h : nullptr;
    const int8_t* const k_b0 = p.k != nullptr ? p.k + b * p.k_b + hkv * p.k_h : nullptr;
    auto ksrc_at = [&](int64_t j) { return seg_src(pk_b0, p.pk_s, k_b0, p.k_s, p.Sk_past, j); };
    /// A chunk is one segment's when it does not straddle the past boundary.
    auto one_segment = [&](int64_t j) { return j + kBlockN <= p.Sk_past || j >= p.Sk_past; };

    const int64_t kv_end =
        p.causal ? std::max<int64_t>(0, std::min<int64_t>(run, q0 + i0 + rows)) : run;
    int64_t j_begin = 0;
    if (p.causal && p.window > 0 && sink_n == 0) {
        const int64_t lo_min = std::max<int64_t>(0, q0 + i0 - p.window + 1);
        if (lo_min > kBlockN) {
            j_begin = ((lo_min - 1) / kBlockN) * kBlockN;
        }
    }

    /// The launcher stages the value run before it launches this tile and
    /// declines it otherwise, so one of the two is always there.
    const T* const vprep = static_cast<const T*>(p.vprep);
    const T* const vprep_base = vprep + (b * p.Hkv + hkv) * p.Sk * p.Dv;
    const int8_t* const vt_base =
        static_cast<const int8_t*>(p.vt) + (b * p.Hkv + hkv) * p.Dv * p.vt_s;
    const int64_t kv_rows = std::min(kv_end, p.Sk);

    if (j_begin < kv_end) {
        const int nrows = chunk_rows(j_begin, kv_rows);
        if constexpr (k_fill_exact<DQK>) {
            if (nrows == kBlockN && one_segment(j_begin)) {
                const Src ks = ksrc_at(j_begin);
                stage_k_fast<DQK>(ks.base, ks.step, Kb, tid);
            } else {
                stage_k<P, DQK>(p, b, hkv, j_begin, Kb, tid);
            }
        } else {
            stage_k<P, DQK>(p, b, hkv, j_begin, Kb, tid);
        }
        if constexpr (kPv8) {
            stage_vt_prep<DV>(vt_base, j_begin, p.vt_s, Vb, tid);
            stage_vfact(p, vsc_off, j_begin, Vsc, Vf, tid);
        } else {
            stage_v_prep<T, DV>(vprep_base, j_begin, p.Sk, Vh, tid);
        }
        if (kcol) {
            stage_ksc(p, ksc_base, j_begin, Ksc, tid);
        }
    }
    cp_async_commit_group();

    /// K's ring base is chosen per chunk below (the Q panel aliases stage 1).
    const uint32_t k_off = static_cast<uint32_t>(lane & 7) * static_cast<uint32_t>(kqs) +
                           static_cast<uint32_t>((lane >> 3) & 1) * 16u;
    const uint32_t vh_off = static_cast<uint32_t>((lane & 7) + ((lane >> 3) & 1) * 8) *
                                static_cast<uint32_t>(kVh * 2) +
                            static_cast<uint32_t>((lane >> 4) * 16);
    const uint32_t vt_off = static_cast<uint32_t>(lane & 7) * static_cast<uint32_t>(kVtRow) +
                            static_cast<uint32_t>((lane >> 3) & 1) * 16u;

    const int64_t jhi_w =
        p.causal ? std::min<int64_t>(run, q0 + i0 + warp * kRowsPerWarp + kRowsPerWarp) : run;
    const int64_t jfull_w =
        p.causal ? std::min<int64_t>(run, q0 + i0 + warp * kRowsPerWarp + 1) : run;
    const int64_t lofull_w =
        (p.window > 0 && p.causal)
            ? std::max<int64_t>(0, q0 + i0 + warp * kRowsPerWarp - p.window)
            : 0;

    const float l2e = 1.4426950408889634f;
    const float rm0 = p.scale * Qsc[row0] * l2e;
    const float rm1 = p.scale * Qsc[row1] * l2e;
    const int jc0 = 2 * (lane & 3);

    /// A pointer-advance fill (each thread's sources one add a chunk instead of
    /// a 64-bit multiply a row) was measured here and dropped: it removes a
    /// fifth of the loop's instructions but costs eight live pointers, and at
    /// this tile's four blocks an SM — 128 registers — the spills it forces
    /// come to more than the arithmetic saves. The fill is on the uniform pipe
    /// and is not what the loop waits on.
    int stage = 0;
    for (int64_t j0 = j_begin; j0 < kv_end; j0 += kBlockN) {
        cp_async_wait_group<0>();
        __syncthreads();

        const int64_t jn = j0 + kBlockN;
        const int ns = stage ^ 1;
        /// The chunk the score will read, and the one being refilled. Both are
        /// published by the barrier above, so a chunk's refill rides the same
        /// barrier as the previous chunk's reads.
        const uint32_t kb = smem_addr_u32(stage == 0 ? Kb : Kb2);
        if (jn < kv_end) {
            const int nrows = chunk_rows(jn, kv_rows);
            int8_t* const dst = ns == 0 ? Kb : Kb2;
            if constexpr (k_fill_exact<DQK>) {
                if (nrows == kBlockN && one_segment(jn)) {
                    const Src ks = ksrc_at(jn);
                    stage_k_fast<DQK>(ks.base, ks.step, dst, tid);
                } else {
                    stage_k<P, DQK>(p, b, hkv, jn, dst, tid);
                }
            } else {
                stage_k<P, DQK>(p, b, hkv, jn, dst, tid);
            }
            if constexpr (kPv8) {
                stage_vt_prep<DV>(vt_base, jn, p.vt_s, Vb + static_cast<int64_t>(ns) * kVbytes,
                                  tid);
            } else {
                stage_v_prep<T, DV>(vprep_base, jn, p.Sk,
                                    Vh + static_cast<int64_t>(ns) * kBlockN * kVh, tid);
            }
            if constexpr (kPv8) {
                stage_vfact(p, vsc_off, jn, Vsc + ns, Vf + static_cast<int64_t>(ns) * kBlockN, tid);
            }
            if (kcol) {
                stage_ksc(p, ksc_base, jn, Ksc + static_cast<int64_t>(ns) * kBlockN, tid);
            }
        }

        // ---- S = Q @ Kᵀ on the integer atom: 4 k steps x 4 key groups ----
        int32_t a32[kScoreNt][4];
#pragma unroll
        for (int t = 0; t < kScoreNt; ++t) {
            a32[t][0] = 0;
            a32[t][1] = 0;
            a32[t][2] = 0;
            a32[t][3] = 0;
        }
#pragma unroll
        for (int ks = 0; ks < DQK / 32; ++ks) {
            uint32_t kf[kScoreNt][2];
#pragma unroll
            for (int g = 0; g < kScoreNt; ++g) {
                ldmatrix_x2_at(&kf[g][0], &kf[g][1],
                               kb + k_off +
                                   static_cast<uint32_t>(8 * g * kqs + ks * 32));
            }
#pragma unroll
            for (int g = 0; g < kScoreNt; ++g) {
                if (j0 + 8 * g < jhi_w) {
                    uint32_t* const d = reinterpret_cast<uint32_t*>(a32[g]);
                    mma_m16n8k32_s8s8(&d[0], &d[1], &d[2], &d[3], qf[ks][0], qf[ks][1],
                                      qf[ks][2], qf[ks][3], kf[g][0], kf[g][1], d[0], d[1], d[2],
                                      d[3]);
                }
            }
        }

        // One group a chunk — every panel staged at the top, awaited at the next
        // chunk's head — so the chunk carries a single barrier.
        cp_async_commit_group();

        // ---- dequantize, mask, online softmax ----
        const bool full = j0 + kBlockN <= jfull_w && j0 >= lofull_w;
        float s[kScoreNt][4];
        float mx0 = kNegInf, mx1 = kNegInf;
#pragma unroll
        for (int t = 0; t < kScoreNt; ++t) {
            const int jc = static_cast<int>(j0) + jc0 + 8 * t;
            float ks0 = kall, ks1 = kall;
            if (kcol) {
                const int off = jc - static_cast<int>(j0);
                ks0 = Ksc[static_cast<int64_t>(stage) * kBlockN + off];
                ks1 = Ksc[static_cast<int64_t>(stage) * kBlockN + (off < kBlockN - 1 ? off + 1 : off)];
            }
            s[t][0] = static_cast<float>(a32[t][0]) * (rm0 * ks0);
            s[t][1] = static_cast<float>(a32[t][1]) * (rm0 * ks1);
            s[t][2] = static_cast<float>(a32[t][2]) * (rm1 * ks0);
            s[t][3] = static_cast<float>(a32[t][3]) * (rm1 * ks1);
            if (!full) {
                s[t][0] = (live0 && jc < hi0 && (jc >= lo0 || jc < sink_n)) ? s[t][0] : kNegInf;
                s[t][1] =
                    (live0 && jc + 1 < hi0 && (jc + 1 >= lo0 || jc + 1 < sink_n)) ? s[t][1]
                                                                                 : kNegInf;
                s[t][2] = (live1 && jc < hi1 && (jc >= lo1 || jc < sink_n)) ? s[t][2] : kNegInf;
                s[t][3] =
                    (live1 && jc + 1 < hi1 && (jc + 1 >= lo1 || jc + 1 < sink_n)) ? s[t][3]
                                                                                  : kNegInf;
            }
            mx0 = fmaxf(mx0, fmaxf(s[t][0], s[t][1]));
            mx1 = fmaxf(mx1, fmaxf(s[t][2], s[t][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));

        const float mn0 = fmaxf(m0, mx0);
        const float mn1 = fmaxf(m1, mx1);
        const float rs0 = (m0 == kNegInf) ? 0.0f : ex2(m0 - mn0);
        const float rs1 = (m1 == kNegInf) ? 0.0f : ex2(m1 - mn1);
        const bool moved = mn0 != m0 || mn1 != m1;
        m0 = mn0;
        m1 = mn1;
        const bool ok0 = mn0 != kNegInf;
        const bool ok1 = mn1 != kNegInf;

        float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
        for (int t = 0; t < kScoreNt; ++t) {
            s[t][0] = ok0 ? ex2(s[t][0] - mn0) : 0.0f;
            s[t][1] = ok0 ? ex2(s[t][1] - mn0) : 0.0f;
            s[t][2] = ok1 ? ex2(s[t][2] - mn1) : 0.0f;
            s[t][3] = ok1 ? ex2(s[t][3] - mn1) : 0.0f;
            sum0 += s[t][0] + s[t][1];
            sum1 += s[t][2] + s[t][3];
        }
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);

        // The rescale is 64 multiplies a chunk and is a no-op whenever no row's
        // maximum moved — which, once the run is a few chunks in, is most
        // chunks. A warp vote turns that into one branch the whole warp agrees
        // on, so the steady state skips the multiplies and nothing diverges.
        if (__any_sync(0xffffffffu, moved)) {
            l0 = l0 * rs0 + sum0;
            l1 = l1 * rs1 + sum1;
#pragma unroll
            for (int t = 0; t < kNt; ++t) {
                o[t][0] *= rs0;
                o[t][1] *= rs0;
                o[t][2] *= rs1;
                o[t][3] *= rs1;
            }
        } else {
            l0 += sum0;
            l1 += sum1;
        }

        // ---- O += P @ V ----
        if constexpr (kPv8) {
            // P on the 255-level grid, the cache's per-key V scale folded in and
            // normalized by the chunk's largest (the CPU's sc_tile) — both were
            // folded into `Vf` when the scale was staged.
            const float deq = p.vscale != nullptr ? Vsc[stage] : 1.0f / 255.0f;
            const float* const vf = Vf + static_cast<int64_t>(stage) * kBlockN + jc0;
            int8_t* const p0 = Pb + static_cast<int64_t>(row0) * kPs + jc0;
            int8_t* const p1 = Pb + static_cast<int64_t>(row1) * kPs + jc0;
#pragma unroll
            for (int t = 0; t < kScoreNt; ++t) {
                const float2 f = *reinterpret_cast<const float2*>(vf + 8 * t);
                // The CPU's order — fold, then levels — so the rounding lands
                // on the same side of a boundary.
                const uint32_t q00 = __float2uint_rn(s[t][0] * f.x);
                const uint32_t q01 = __float2uint_rn(s[t][1] * f.y);
                const uint32_t q10 = __float2uint_rn(s[t][2] * f.x);
                const uint32_t q11 = __float2uint_rn(s[t][3] * f.y);
                *reinterpret_cast<unsigned short*>(p0 + 8 * t) =
                    static_cast<unsigned short>((q01 << 8) | q00);
                *reinterpret_cast<unsigned short*>(p1 + 8 * t) =
                    static_cast<unsigned short>((q11 << 8) | q10);
            }
            __syncwarp(); ///< the warp's own rows, read back by its own ldmatrix

            uint32_t pa[4];
            ldmatrix_x4_at(&pa[0], &pa[1], &pa[2], &pa[3],
                           smem_addr_u32(Pb) +
                               static_cast<uint32_t>(warp * kRowsPerWarp) *
                                   static_cast<uint32_t>(kPs) +
                               static_cast<uint32_t>(lane & 15) * static_cast<uint32_t>(kPs) +
                               static_cast<uint32_t>((lane >> 4) * 16));
            const uint32_t vb = smem_addr_u32(Vb) + static_cast<uint32_t>(stage * kVbytes);
#pragma unroll
            for (int g = 0; g < kNt; ++g) {
                uint32_t vfrag[2];
                ldmatrix_x2_at(&vfrag[0], &vfrag[1],
                               vb + vt_off + static_cast<uint32_t>(8 * g * kVtRow));
                int32_t acc[4] = {0, 0, 0, 0};
                uint32_t* const d = reinterpret_cast<uint32_t*>(acc);
                mma_m16n8k32_u8s8(&d[0], &d[1], &d[2], &d[3], pa[0], pa[1], pa[2], pa[3],
                                  vfrag[0], vfrag[1], 0u, 0u, 0u, 0u);
                o[g][0] = fmaf(static_cast<float>(acc[0]), deq, o[g][0]);
                o[g][1] = fmaf(static_cast<float>(acc[1]), deq, o[g][1]);
                o[g][2] = fmaf(static_cast<float>(acc[2]), deq, o[g][2]);
                o[g][3] = fmaf(static_cast<float>(acc[3]), deq, o[g][3]);
            }
        } else {
            // 16-bit P·V: P leaves the same registers the f16 tile packs from.
            uint32_t pa[kPvSteps][4];
#pragma unroll
            for (int kk = 0; kk < kPvSteps; ++kk) {
                pa[kk][0] = pack2<T>(s[2 * kk][0], s[2 * kk][1]);
                pa[kk][1] = pack2<T>(s[2 * kk][2], s[2 * kk][3]);
                pa[kk][2] = pack2<T>(s[2 * kk + 1][0], s[2 * kk + 1][1]);
                pa[kk][3] = pack2<T>(s[2 * kk + 1][2], s[2 * kk + 1][3]);
            }
            const uint32_t vb = smem_addr_u32(Vh) +
                                static_cast<uint32_t>(stage) *
                                    static_cast<uint32_t>(kBlockN * kVh * 2);
#pragma unroll
            for (int kk = 0; kk < kPvSteps; ++kk) {
                uint32_t vfr[kNt / 2][4];
#pragma unroll
                for (int g = 0; g < kNt / 2; ++g) {
                    ldmatrix_x4_trans_at(&vfr[g][0], &vfr[g][1], &vfr[g][2], &vfr[g][3],
                                         vb + vh_off +
                                             static_cast<uint32_t>(16 * kk * kVh + 16 * g) * 2u);
                }
#pragma unroll
                for (int g = 0; g < kNt / 2; ++g) {
                    hmma<T>(o[2 * g], pa[kk], vfr[g][0], vfr[g][1]);
                    hmma<T>(o[2 * g + 1], pa[kk], vfr[g][2], vfr[g][3]);
                }
            }
        }

        stage ^= 1;
    }

    // ---- the head sink, normalization and the gate ----
    float scl0 = 1.0f, scl1 = 1.0f;
    if (p.sink != nullptr) {
        const float sv = p.sink[h] * l2e;
        const float a0 = fmaxf(m0, sv);
        const float a1 = fmaxf(m1, sv);
        scl0 = (m0 == kNegInf) ? 0.0f : ex2(m0 - a0);
        scl1 = (m1 == kNegInf) ? 0.0f : ex2(m1 - a1);
        l0 = l0 * scl0 + ex2(sv - a0);
        l1 = l1 * scl1 + ex2(sv - a1);
    }
    const float norm0 = l0 > 0.0f ? 1.0f / l0 : 0.0f;
    const float norm1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;

    const int64_t obase = b * p.o_b + h * p.o_h;
    store_output_dispatch<T, kNt>(p, o, obase, b * p.g_b + h * p.g_h, i0, 2 * (lane & 3),
                                      row0, row1, live0, live1, scl0, scl1, norm0, norm1);
}

} // namespace i8
} // namespace nnops::backend::cuda::attn
