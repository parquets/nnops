/// @file gemm_narrow.cuh
/// @brief The hand-written narrow-m engines: a product of at most sixteen rows
///        is a bandwidth problem, and neither tensor-core tile serves it.
///
/// The `[k][n]` loop (a MatMul with both operands plain) runs one thread per
/// column group with B staged sixteen rows at a time, and B's row is the shared
/// operand. The `[n][k]` loop (a `Linear` weight) cannot read columns at all, so
/// it transposes in shared memory instead and gives each thread one column's
/// accumulator down its own row of the stage, with A staged k-major so those
/// reads broadcast.

#pragma once

#include "gemm_splitk.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>

namespace nnops::backend::cuda {

namespace detail {

// ---- narrow m -------------------------------------------------------------

/// A product of at most sixteen rows is a bandwidth problem, not a tensor-core
/// one: sixteen rows need sixteen FFMA per element of B, B is streamed exactly
/// once, and the 64-row tile computes four times the mma for it. At
/// 16x4096x4096 the tile path ran 1.68x cuBLAS's own decode kernel; this
/// register-tiled FFMA loop runs 1.20x in f16 and 0.88x in f32, at 480 GB/s of
/// the bus's measured 528. Sixteen rows of B per group is the f16 sweet spot
/// (eight measured 0.117 ms against 0.107).
///
/// Neither the n tail nor the k tail is a branch. A row of B starts at a
/// multiple of the row stride and the stride is a multiple of the vector width,
/// so clamping the column window to `(n - VEC) & ~(VEC - 1)` is still aligned:
/// the ragged CTA reads a valid window twice and stores only its own columns.
/// That is not only tidier than a second loop body — a per-load `n_ok` predicate
/// made ptxas settle all eight `LDG.E.64` of a group on one destination register
/// pair, so they issued as a dependency chain and the f16 loop measured 0.170 ms
/// against 0.130 without it.
constexpr int kSmallMRows = 16;
constexpr int kSmallMThreads = 128;
constexpr int kSmallMVec = 4;
constexpr int kSmallMBn = kSmallMThreads * kSmallMVec;
constexpr int kSmallMK = 32;       ///< k per staged A chunk
constexpr int kSmallMUnroll = 16;  ///< rows of B in flight before the converts

/// The transposed-operand loop keeps the thread count and the k chunk, but a
/// thread owns one column instead of four, so a block is 128 columns wide.
constexpr int kSmallNTCols = kSmallMThreads;
constexpr int kSmallNTK = kSmallMK;

template <class Elem>
__global__ __launch_bounds__(kSmallMThreads) void dense_gemm_small_m_kernel(GemmParams p)
{
    constexpr int BM = kSmallMRows;
    constexpr int VEC = kSmallMVec;
    constexpr int KC = kSmallMK;
    constexpr int UK = kSmallMUnroll;
    constexpr int BN = kSmallMBn;

    const Elem* __restrict__ A = static_cast<const Elem*>(p.a);
    const Elem* __restrict__ B = static_cast<const Elem*>(p.b);
    const int M = int(p.m);
    const int N = int(p.n);
    const int K = int(p.k);
    const int lda = int(p.lda);
    const int ldb = int(p.ldb);

    const int tid = int(threadIdx.x);
    const int ncol = int(blockIdx.x) * BN + tid * VEC;
    const int nrd = min(ncol, (N - VEC) & ~(VEC - 1));
    const int split = gemm_block_split(p);
    int kt_begin = 0;
    int kt_count = 0;
    gemm_split_range(p, KC, split, kt_begin, kt_count);
    if (kt_count <= 0) { return; }
    const int k_begin = kt_begin * KC;
    const int k_end = min(K, (kt_begin + kt_count) * KC);

    __shared__ __align__(16) float sA[BM][KC];

    float acc[BM][VEC];
#pragma unroll
    for (int m = 0; m < BM; ++m) {
#pragma unroll
        for (int v = 0; v < VEC; ++v) { acc[m][v] = 0.0f; }
    }

    for (int k0 = k_begin; k0 < k_end; k0 += KC) {
        const int kc = min(KC, k_end - k0);
        __syncthreads();
#pragma unroll
        for (int i = tid; i < BM * KC; i += kSmallMThreads) {
            const int m = i / KC;
            const int kk = i - m * KC;
            sA[m][kk] = (m < M && kk < kc) ? float(A[int64_t(m) * lda + k0 + kk]) : 0.0f;
        }
        __syncthreads();

        int kk = 0;
        for (; kk + UK <= kc; kk += UK) {
            float b[UK][VEC];
#pragma unroll
            for (int j = 0; j < UK; ++j) {
                const Elem* src = B + int64_t(k0 + kk + j) * ldb + nrd;
                if constexpr (sizeof(Elem) == 4) {
                    const float4 v = *reinterpret_cast<const float4*>(src);
                    b[j][0] = v.x;
                    b[j][1] = v.y;
                    b[j][2] = v.z;
                    b[j][3] = v.w;
                } else {
                    const uint2 raw = *reinterpret_cast<const uint2*>(src);
                    const float2 f0 = __half22float2(__halves2half2(
                        __ushort_as_half(uint16_t(raw.x)),
                        __ushort_as_half(uint16_t(raw.x >> 16))));
                    const float2 f1 = __half22float2(__halves2half2(
                        __ushort_as_half(uint16_t(raw.y)),
                        __ushort_as_half(uint16_t(raw.y >> 16))));
                    b[j][0] = f0.x;
                    b[j][1] = f0.y;
                    b[j][2] = f1.x;
                    b[j][3] = f1.y;
                }
            }
#pragma unroll
            for (int m = 0; m < BM; ++m) {
#pragma unroll
                for (int g = 0; g < UK / 4; ++g) {
                    const float4 a = *reinterpret_cast<const float4*>(&sA[m][kk + 4 * g]);
                    const float av[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
                    for (int v = 0; v < VEC; ++v) {
                        acc[m][v] += av[0] * b[4 * g + 0][v] + av[1] * b[4 * g + 1][v] +
                                     av[2] * b[4 * g + 2][v] + av[3] * b[4 * g + 3][v];
                    }
                }
            }
        }
        for (; kk < kc; ++kk) {
            float b[VEC];
            const Elem* src = B + int64_t(k0 + kk) * ldb + nrd;
            if constexpr (sizeof(Elem) == 4) {
                const float4 v = *reinterpret_cast<const float4*>(src);
                b[0] = v.x;
                b[1] = v.y;
                b[2] = v.z;
                b[3] = v.w;
            } else {
                const uint2 raw = *reinterpret_cast<const uint2*>(src);
                const float2 f0 = __half22float2(__halves2half2(
                    __ushort_as_half(uint16_t(raw.x)),
                    __ushort_as_half(uint16_t(raw.x >> 16))));
                const float2 f1 = __half22float2(__halves2half2(
                    __ushort_as_half(uint16_t(raw.y)),
                    __ushort_as_half(uint16_t(raw.y >> 16))));
                b[0] = f0.x;
                b[1] = f0.y;
                b[2] = f1.x;
                b[3] = f1.y;
            }
#pragma unroll
            for (int m = 0; m < BM; ++m) {
                const float a = sA[m][kk];
#pragma unroll
                for (int v = 0; v < VEC; ++v) { acc[m][v] += a * b[v]; }
            }
        }
    }

    float* part = p.partial +
                  (static_cast<int64_t>(gemm_block_batch(p)) * p.k_splits + split) *
                      static_cast<int64_t>(M) * N;
    if (ncol >= N) { return; }
    const int lim = min(VEC, N - ncol);
#pragma unroll
    for (int m = 0; m < BM; ++m) {
        if (m >= M) { continue; }
        const int64_t base = static_cast<int64_t>(m) * N + ncol;
        for (int v = 0; v < lim; ++v) { part[base + v] = acc[m][v]; }
    }
}

/// True when the narrow-m loop owns this product. It reads A as [m][k] and B as
/// [k][n], both row-major, one matrix at a time, and hands the epilogue to the
/// split reduction; every other layout and shape stays on the tiled path.
inline bool small_m_applies(const GemmParams& p, GemmElem elem, bool trans_a,
                            bool trans_b, bool vectorized)
{
    return vectorized && !trans_a && !trans_b && p.m >= 1 && p.m <= kSmallMRows &&
           p.n >= kSmallMVec && p.k >= 1024 && gemm_batch_count(p) == 1 &&
           (elem == GemmElem::F32 || elem == GemmElem::F16);
}

/// One and a half splits per 512 k, never more than sixteen. A split is the
/// only axis this grid can grow on — 128 threads cover 512 columns, so the
/// block count is `n / 512 * splits` — and at eight splits the decode product
/// left the machine under two and a half blocks wide. Twelve measured 0.141 ms
/// in f32 and 0.115 in f16 at 16x4096x4096 against 0.147 and 0.127 at eight,
/// and the two other decode geometries want the same 1.5x: 16x8192x2048 (twice
/// the n blocks, half the k) peaks at six, 16x2048x8192 at sixteen.
///
/// The cap is the cache, not a constant. Every split writes one f32 plane and
/// the reduce reads them all back, so past a point the planes cost more than
/// the extra blocks buy: f16 at 16x4096x4096 ran 0.1065 ms at fourteen splits
/// and 0.1270 at sixteen. The knee sits a little past the cache rather than at
/// it — this card's L2 is 3 MB, and 3.07 MB of planes is still the fastest
/// point while 4.2 MB is 19% back — so the budget is nine eighths of the
/// queried size. The dtype does not enter: the planes are f32 whichever the
/// operands are.
///
/// `NNOPS_CUDA_SPLITK` overrides it, on the same terms as the tiled path.
inline int small_m_splits(const GemmParams& p)
{
    const int forced = splitk_env();
    if (forced == 0) { return 1; }
    const int k_tiles = int((p.k + kSmallMK - 1) / kSmallMK);
    if (forced > 1) { return std::max(1, std::min({forced, k_tiles, 16})); }

    const int64_t plane = int64_t(p.m) * p.n * int64_t(sizeof(float));
    int splits = int(std::min<int64_t>(std::max<int64_t>(p.k * 3 / 1024, 1), 16));
    const int64_t l2 = device_l2_bytes();
    if (l2 > 0) {
        splits = int(std::min<int64_t>(
            splits, std::max<int64_t>(l2 * 9 / 8 / plane, 1)));
    }
    splits = std::min(splits, k_tiles);
    while (splits > 1 && int64_t(splits) * plane > kSplitKMaxBytes) { splits /= 2; }
    return splits;
}

template <class Elem>
inline bool launch_small_m(const GemmParams& p, cudaStream_t stream)
{
    const int splits = small_m_splits(p);
    const int64_t bytes = int64_t(splits) * p.m * p.n * int64_t(sizeof(float));
    const GemmWorkspace ws = gemm_workspace_alloc(bytes, stream);
    if (ws.ptr == nullptr) { return false; }

    GemmParams q = p;
    q.k_splits = splits;
    q.partial = static_cast<float*>(ws.ptr);
    const dim3 grid(static_cast<unsigned>((p.n + kSmallMBn - 1) / kSmallMBn), 1,
                    static_cast<unsigned>(splits));
    dense_gemm_small_m_kernel<Elem><<<grid, kSmallMThreads, 0, stream>>>(q);
    launch_splitk_reduce<Elem>(q, stream);
    gemm_workspace_free(ws, stream);
    return true;
}

/// The narrow-m loop for a transposed B: `Linear` stores its weight as `[n][k]`,
/// so the layout the [k][n] loop wants — each lane reading a run of columns —
/// no longer exists. Reading along k per lane instead makes every lane a stream
/// of its own, a warp issuing thirty-two requests an eighth of a line wide, and
/// that path measured 117 GB/s.
///
/// So the block transposes in shared memory rather than in the access pattern:
/// one lane reads sixteen bytes of one B row, which is k-contiguous and whole
/// lines once a warp's lanes land on neighbouring rows, and each thread then
/// owns one column's accumulator and walks k down its own row of the stage.
/// A is staged k-major, so all threads read the same `float4`s and the reads
/// broadcast instead of conflicting. A warp-per-row variant of this, with the
/// lane as the k slice and a shuffle reduce, was built and measured first: it
/// re-reads all of A per column from shared memory, sixteen loads per sixteen
/// FMA, and lost by 2.9x to 9.3x.
///
/// Sixteen rows of A per element of B is what keeps this bandwidth-bound; `BM`
/// is the number of rows kept live, so a GEMV does not run fifteen FMAs that
/// accumulate into nothing. The 16x4096x4096 decode product runs at 407 GB/s
/// f32, against a bus measured at 528, and 508 at 1x4096x4096; in f16 the same
/// column walk reads 280, because there the kernel is not the limit — the loop
/// issues under half the cycles the machine has (ncu: L1TEX 78%, compute 67%,
/// three resident blocks) and neither more registers nor less traffic moved it.
///
/// f16 halves the B traffic and is otherwise the same loop, but two things have
/// to move with it. The stage row is padded by two halves rather than one, so
/// that its base is four-byte aligned: the staged store then lands as four
/// 32-bit writes instead of eight 16-bit ones — the odd pad is what made f16
/// staging instruction-bound — and a pair of k can be read and converted as
/// one `half2`. The 34-half row is 17 words, still coprime with the 32 banks,
/// so the column walk stays conflict-free either way.
template <class Elem, int BM>
__global__ __launch_bounds__(kSmallNTCols) void dense_gemm_small_nt_kernel(
    GemmParams p)
{
    constexpr int THREADS = kSmallNTCols;
    constexpr int COLS = THREADS;
    constexpr int ES = int(sizeof(Elem));
    constexpr int KC = kSmallNTK;
    constexpr int VE = 16 / ES;  ///< elements per staged load
    constexpr int NG = KC / VE;  ///< staged loads per column
    constexpr int BPAD = ES == 2 ? 2 : 1;
    constexpr int APAD = (BM + 3) & ~3;

    const Elem* __restrict__ A = static_cast<const Elem*>(p.a);
    const Elem* __restrict__ B = static_cast<const Elem*>(p.b);
    const int M = int(p.m);
    const int N = int(p.n);
    const int K = int(p.k);
    const int lda = int(p.lda);
    const int ldb = int(p.ldb);

    const int tid = int(threadIdx.x);
    const int col = int(blockIdx.x) * COLS + tid;
    const int split = gemm_block_split(p);
    int kt_begin = 0;
    int kt_count = 0;
    gemm_split_range(p, KC, split, kt_begin, kt_count);
    if (kt_count <= 0) { return; }
    const int k_begin = kt_begin * KC;
    const int k_end = min(K, (kt_begin + kt_count) * KC);

    __shared__ __align__(16) Elem sB[COLS][KC + BPAD];
    __shared__ __align__(16) float sAt[KC][APAD];

    float acc[BM];
#pragma unroll
    for (int m = 0; m < BM; ++m) { acc[m] = 0.0f; }

    constexpr int AR = (BM * KC + THREADS - 1) / THREADS;  ///< A per thread
    constexpr int NA = BM * KC;

    uint4 rb[NG];
    Elem ra[AR];

    /// One k tile read into registers, guarded exactly as the staged store
    /// would have filled it. Reading here and storing after the compute is
    /// what puts the load latency behind a whole tile: the store is the only
    /// consumer, and it runs a compute phase after the loads were issued.
    /// f32 stages one tile ahead in registers, worth a measured 1.25x on every
    /// decode shape; f16 does not, because its staging is half the bytes and
    /// the registers come back as lost residents instead — the GEMV the gate
    /// leaves here ran 0.120 that way against 0.110 staged in place.
    constexpr bool kPrefetch = ES == 4;

    const auto fetch = [&](int k0) {
        const int kc = min(KC, k_end - k0);
#pragma unroll
        for (int j = 0; j < NG; ++j) {
            const int i = tid + j * THREADS;
            const int c = i / NG;
            const int kb = (i - c * NG) * VE;
            const int g = int(blockIdx.x) * THREADS + c;
            if (g < N && kb + VE <= kc) {
                const Elem* src = B + int64_t(g) * ldb + k0 + kb;
                rb[j] = *reinterpret_cast<const uint4*>(src);
            } else {
                Elem* e = reinterpret_cast<Elem*>(&rb[j]);
#pragma unroll
                for (int t = 0; t < VE; ++t) {
                    const int kk = kb + t;
                    e[t] = (g < N && kk < kc) ? B[int64_t(g) * ldb + k0 + kk]
                                              : Elem(0);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < AR; ++j) {
            const int i = tid + j * THREADS;
            const int m = i / KC;
            const int kk = i - m * KC;
            ra[j] = (i < NA && m < M && kk < kc)
                        ? A[int64_t(m) * lda + k0 + kk]
                        : Elem(0);
        }
    };

    const auto store = [&]() {
#pragma unroll
        for (int j = 0; j < NG; ++j) {
            const int i = tid + j * THREADS;
            const int c = i / NG;
            const int kb = (i - c * NG) * VE;
            if constexpr (ES == 2) {
                uint32_t* dst = reinterpret_cast<uint32_t*>(&sB[c][kb]);
                const uint32_t* w = reinterpret_cast<const uint32_t*>(&rb[j]);
                dst[0] = w[0];
                dst[1] = w[1];
                dst[2] = w[2];
                dst[3] = w[3];
            } else {
                const Elem* e = reinterpret_cast<const Elem*>(&rb[j]);
#pragma unroll
                for (int t = 0; t < VE; ++t) { sB[c][kb + t] = e[t]; }
            }
        }
#pragma unroll
        for (int j = 0; j < AR; ++j) {
            const int i = tid + j * THREADS;
            if (i < NA) {
                const int m = i / KC;
                sAt[i - m * KC][m] = float(ra[j]);
            }
        }
    };

    if constexpr (kPrefetch) { fetch(k_begin); }
    for (int k0 = k_begin; k0 < k_end; k0 += KC) {
        const int kc = min(KC, k_end - k0);
        __syncthreads();
        if constexpr (!kPrefetch) { fetch(k0); }
        store();
        __syncthreads();
        if constexpr (kPrefetch) {
            if (k0 + KC < k_end) { fetch(k0 + KC); }
        }

        const auto rank1 = [&](int kk, float b) {
            float av[APAD];
#pragma unroll
            for (int m = 0; m < BM; m += 4) {
                const float4 v = *reinterpret_cast<const float4*>(&sAt[kk][m]);
                av[m] = v.x;
                av[m + 1] = v.y;
                av[m + 2] = v.z;
                av[m + 3] = v.w;
            }
#pragma unroll
            for (int m = 0; m < BM; ++m) { acc[m] += av[m] * b; }
        };

        int kk = 0;
        if constexpr (ES == 2) {
            for (; kk + 1 < kc; kk += 2) {
                const uint32_t w =
                    *reinterpret_cast<const uint32_t*>(&sB[tid][kk]);
                const float2 b2 =
                    __half22float2(*reinterpret_cast<const __half2*>(&w));
                rank1(kk, b2.x);
                rank1(kk + 1, b2.y);
            }
        }
        for (; kk < kc; ++kk) { rank1(kk, float(sB[tid][kk])); }
    }

    float* part = p.partial +
                  (static_cast<int64_t>(gemm_block_batch(p)) * p.k_splits + split) *
                      static_cast<int64_t>(M) * N;
    if (col >= N) { return; }
#pragma unroll
    for (int m = 0; m < BM; ++m) {
        if (m < M) { part[static_cast<int64_t>(m) * N + col] = acc[m]; }
    }
}

/// The transposed twin of `small_m_applies`. `vectorized` already means both
/// row strides are whole sixteen-byte vectors, which is what the staged read
/// needs; everything else about the shape matches the [k][n] loop.
///
/// Every narrow-m product in f32, and the single row in f16. Sixteen rows of
/// f16 stay on the tensor core tiles: measured against them in one process,
/// this loop is 1.62 against 1.66 of cuBLAS at 4096x4096, 1.48 against 1.52 at
/// 8192x2048, and 1.89 against 1.65 at 2048x8192 — a wash at best and a loss at
/// 1024 columns, because a 64-row mma block's wasted rows cost nothing when the
/// B stream is the limit. At one row the tiles do pay for them, 0.136 against
/// this loop's 0.110, so the GEMV stays here.
inline bool small_nt_applies(const GemmParams& p, GemmElem elem, bool trans_a,
                             bool trans_b, bool vectorized)
{
    if (!vectorized || trans_a || !trans_b) { return false; }
    if (p.m < 1 || p.m > kSmallMRows || p.n < 1 || p.k < 1024) { return false; }
    if (gemm_batch_count(p) != 1) { return false; }
    return elem == GemmElem::F32 || (elem == GemmElem::F16 && p.m == 1);
}

/// A split is the only axis this grid grows on here too, but the block is four
/// times narrower than the [k][n] loop's — 128 columns — so the same k wants a
/// quarter of the splits. A half split per 512 k, capped at eight, then capped
/// so the f32 planes stay inside the cache: six and eight measured within 1% of
/// each other at 16x4096x4096, four is the best at 16x8192x2048, and doubling
/// past eight costs 8-16%. `NNOPS_CUDA_SPLITK` still overrides.
inline int small_nt_splits(const GemmParams& p)
{
    const int forced = splitk_env();
    if (forced == 0) { return 1; }
    const int k_tiles = int((p.k + kSmallNTK - 1) / kSmallNTK);
    if (forced > 1) { return std::max(1, std::min({forced, k_tiles, 16})); }

    const int64_t plane = int64_t(p.m) * p.n * int64_t(sizeof(float));
    int splits = int(std::min<int64_t>(std::max<int64_t>(p.k / 512, 1), 8));
    const int64_t l2 = device_l2_bytes();
    if (l2 > 0) {
        splits = int(std::min<int64_t>(
            splits, std::max<int64_t>(l2 * 9 / 8 / plane, 1)));
    }
    splits = std::min(splits, k_tiles);
    while (splits > 1 && int64_t(splits) * plane > kSplitKMaxBytes) { splits /= 2; }
    return splits;
}

template <class Elem>
inline bool launch_small_nt(const GemmParams& p, cudaStream_t stream)
{
    const int splits = small_nt_splits(p);
    const int64_t bytes = int64_t(splits) * p.m * p.n * int64_t(sizeof(float));
    const GemmWorkspace ws = gemm_workspace_alloc(bytes, stream);
    if (ws.ptr == nullptr) { return false; }

    GemmParams q = p;
    q.k_splits = splits;
    q.partial = static_cast<float*>(ws.ptr);
    const dim3 grid(static_cast<unsigned>((p.n + kSmallNTCols - 1) / kSmallNTCols), 1,
                    static_cast<unsigned>(splits));
    if (p.m == 1) {
        dense_gemm_small_nt_kernel<Elem, 1><<<grid, kSmallNTCols, 0, stream>>>(q);
    } else if (p.m <= 4) {
        dense_gemm_small_nt_kernel<Elem, 4><<<grid, kSmallNTCols, 0, stream>>>(q);
    } else {
        dense_gemm_small_nt_kernel<Elem, kSmallMRows>
            <<<grid, kSmallNTCols, 0, stream>>>(q);
    }
    launch_splitk_reduce<Elem>(q, stream);
    gemm_workspace_free(ws, stream);
    return true;
}

}  // namespace detail

}  // namespace nnops::backend::cuda
