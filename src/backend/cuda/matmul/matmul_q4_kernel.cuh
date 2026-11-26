/// @file matmul_q4_kernel.cuh
/// @brief The w4a16 GEMM: a 16-bit activation against a llama.cpp block-packed
///        weight (Q4_0, Q4_1 or Q8_0), decoded into the B panel as it is filled.
///
/// The tile, the A fill, the mainloop and the write-back are the 16-bit GEMM's
/// (matmul_f16.cuh / matmul_f16_kernel.cuh); this file supplies the one piece
/// they do not have — a panel fill that reads a block stream. That is the whole
/// design: the weight never becomes a dense buffer, and the decode replaces the
/// B copy, so the weight stream is 20 bytes per 32 values (Q4_1) against 64 for
/// a dense f16 one.
///
/// The decode is bit-for-bit q4.hpp's: `d·(q−8)` for Q4_0, `d·q + m` for Q4_1,
/// `d·q` for Q8_0, with `d`/`m` read as fp16 and the arithmetic in f32 before
/// one narrowing to the panel's element type. Each block covers 32 k values, so
/// with a 32-deep k tile one stage is exactly one block per B row, and every
/// row's blocks are at the same k index — the block fetch is a plain strided
/// walk, no comparison against the block grid per k step.
///
/// Both B poses decode one block per thread per vector: the eight values a
/// thread writes are always inside one block, because a block covers 32 k and a
/// vector is eight — eight consecutive k of one row for the k-major pose
/// (`[N, K]`, Linear's weight and MatMul's transposed B), eight consecutive n of
/// one row for the n-major one (`[K, N]`). So a thread reads one header and
/// eight nibble bytes per eight outputs in either pose; only which coordinate
/// picks the block changes.
///
/// Only the automatic tile is instantiated: one row of the menu is enough for
/// a decode-bound kernel, and the two element types share every instruction but
/// the narrowing.
///
/// Measured on this card (RTX 3060 Ti, f16 activation, `[N, K]` weight): the
/// decode is issue work — about four ALU instructions a value, 2048 values a
/// stage — and the tile is shared-memory-bound, so the route is *slower* than
/// the same product with a dense f16 weight: 512x4096x4096 runs 1.21 ms against
/// 0.69, and 1x32000x4096 1.46 against 0.84. What it buys is the resident
/// footprint — 20 bytes per 32 weights against 64 — and the smaller stream;
/// turning that into time needs a decode the ALU can issue in one or two
/// instructions a value, and a narrow-m route where every weight value feeds
/// more than one MAC. The tile stays because it is the general route (batch,
/// both B poses, the whole epilogue) and because a cheaper decode drops in
/// here.

#pragma once

#include "matmul_f16_kernel.cuh"
#include "matmul_narrow.cuh"
#include "matmul_q4_host.hpp"

#include "nnops/core/data_type.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace nnops::backend::cuda {

/// The production row of the dense menu (32x32 warp tiles, four blocks an SM,
/// eight-CTA raster strips). The decode is issued between the panel copies, so
/// the tile that fills the machine is still the right one.
using Q4Tile = Tile7;

/// The eight values at `j0` (a multiple of 8, and inside one 32-value block) of
/// a llama.cpp block. Values at `t >= n` are zero, which is how a row whose k
/// ends inside a block fills its tail.
__device__ __forceinline__ void q4_decode8_f(const uint8_t* blk, int j0, int n, DataType fmt,
                                             float* v)
{
    const float d = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(blk)));

    if (fmt == DataType::q8_0) {
        const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            v[t] = t < n ? d * static_cast<float>(qs[j0 + t]) : 0.0f;
        }
        return;
    }

    // The 16 nibble bytes pack value j in byte j's low nibble for j < 16 and
    // byte j − 16's high nibble past that, so one 8-value chunk is the low or
    // the high nibbles of the same eight bytes.
    const bool with_min = fmt == DataType::q4_1;
    const float m = with_min ?
                        __half2float(__ushort_as_half(
                            *reinterpret_cast<const uint16_t*>(blk + 2))) :
                        0.0f;
    const uint8_t* src = blk + (with_min ? 4 : 2) + ((j0 & 8) ? 8 : 0);
    const int shift = (j0 & 16) ? 4 : 0;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const int q = (src[t] >> shift) & 0x0F;
        v[t] = t < n ? static_cast<float>(with_min ? d * static_cast<float>(q) + m :
                                                      d * static_cast<float>(q - 8)) :
                       0.0f;
    }
}

/// The same eight values as the panel's packed 16-bit vector.
template <class Elem>
__device__ __forceinline__ uint4 q4_decode8(const uint8_t* blk, int j0, int n, DataType fmt)
{
    float v[8];
    q4_decode8_f(blk, j0, n, fmt, v);
    return make_uint4(gemm_pack2<Elem>(v[0], v[1]), gemm_pack2<Elem>(v[2], v[3]),
                      gemm_pack2<Elem>(v[4], v[5]), gemm_pack2<Elem>(v[6], v[7]));
}

/// One llama.cpp block read as words: the payload plus `d` and the offset the
/// decode adds (`q4_1`'s minimum, or −8·d so the nibble bias folds into the
/// same FMA, or zero for Q8_0).
///
/// This is the narrow-m loop's form, where one thread owns a whole block and
/// pays for it once: thirty-two values out of five loads for Q4_1. Reading the
/// payload a byte at a time there cost one load instruction per value, and each
/// of those is thirty-two scattered sector requests in a warp — the first
/// version ran 8x slower than the dense loop for exactly that reason.
struct Q4Block {
    uint32_t w[8]; ///< Q4: four 4-byte payload words; Q8_0: eight
    float d = 0.0f;
    float m = 0.0f;
};

/// The payload starts two bytes into a Q4_0/Q8_0 block (odd 4-byte alignment
/// for half the blocks), so those read 16-bit words. A Q4_1 block is a multiple
/// of four bytes and its nibbles start on one, so it reads 32-bit words.
__device__ __forceinline__ void q4_load_block(const uint8_t* blk, DataType fmt, Q4Block& q)
{
    if (fmt == DataType::q4_1) {
        const uint32_t dm = *reinterpret_cast<const uint32_t*>(blk);
        q.d = __half2float(__ushort_as_half(uint16_t(dm)));
        q.m = __half2float(__ushort_as_half(uint16_t(dm >> 16)));
        q.w[0] = *reinterpret_cast<const uint32_t*>(blk + 4);
        q.w[1] = *reinterpret_cast<const uint32_t*>(blk + 8);
        q.w[2] = *reinterpret_cast<const uint32_t*>(blk + 12);
        q.w[3] = *reinterpret_cast<const uint32_t*>(blk + 16);
        return;
    }
    const float d = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(blk)));
    q.d = d;
    q.m = fmt == DataType::q4_0 ? -8.0f * d : 0.0f;
    const uint8_t* p = blk + 2;
    const int words = fmt == DataType::q4_0 ? 4 : 8;
    for (int i = 0; i < words; ++i) {
        q.w[i] = uint32_t(*reinterpret_cast<const uint16_t*>(p + 4 * i)) |
                 (uint32_t(*reinterpret_cast<const uint16_t*>(p + 4 * i + 2)) << 16);
    }
}

/// The eight values at `j0` (a multiple of 8) of a loaded block.
__device__ __forceinline__ void q4_block8(const Q4Block& q, int j0, DataType fmt, float* v)
{
    if (fmt == DataType::q8_0) {
        const uint32_t a = q.w[j0 >> 2];
        const uint32_t b = q.w[(j0 >> 2) + 1];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            v[t] = q.d * static_cast<float>(int8_t(a >> (8 * t)));
            v[t + 4] = q.d * static_cast<float>(int8_t(b >> (8 * t)));
        }
        return;
    }
    // Value j is byte j's low nibble for j < 16 and byte j − 16's high nibble
    // past that, so one 8-value chunk is two payload words at one shift.
    const int wb = (j0 & 8) ? 2 : 0;
    const int sh = (j0 & 16) ? 4 : 0;
    const uint32_t a = q.w[wb];
    const uint32_t b = q.w[wb + 1];
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        v[t] = q.d * static_cast<float>((a >> (8 * t + sh)) & 0x0F) + q.m;
        v[t + 4] = q.d * static_cast<float>((b >> (8 * t + sh)) & 0x0F) + q.m;
    }
}

/// One value at index `j` of a block — the n-major fill's unit, where the eight
/// values of one vector come from eight different blocks.
__device__ __forceinline__ float q4_decode1(const uint8_t* blk, int j, DataType fmt)
{
    const float d = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(blk)));
    if (fmt == DataType::q8_0) {
        return d * static_cast<float>(reinterpret_cast<const int8_t*>(blk + 2)[j]);
    }
    const bool with_min = fmt == DataType::q4_1;
    const float m = with_min ?
                        __half2float(__ushort_as_half(
                            *reinterpret_cast<const uint16_t*>(blk + 2))) :
                        0.0f;
    const uint8_t byte = blk[(with_min ? 4 : 2) + (j & 15)];
    const int q = (byte >> (j >= 16 ? 4 : 0)) & 0x0F;
    return with_min ? d * static_cast<float>(q) + m : d * static_cast<float>(q - 8);
}

/// The w4a16 tile kernel: the 16-bit tile with the B panel decoded from blocks.
template <class S, class Elem, bool TransA, bool TransB>
__global__ __launch_bounds__(S::threads, S::MinBlocks)
void q4_gemm_kernel(F16Params p, int64_t b_row_blocks, int fmt_i)
{
    constexpr int BM = S::BM, BN = S::BN, BK = S::BK, ST = S::stages;
    constexpr int MA = S::MA, NA = S::NA;
    const DataType fmt = static_cast<DataType>(fmt_i);
    const int block_bytes = int(data_type_block_bytes(fmt));

    extern __shared__ __align__(16) unsigned char q4_smem_raw[];
    Elem* const smem = reinterpret_cast<Elem*>(q4_smem_raw);
    Elem* const sAbase = smem;
    Elem* const sBbase = smem + ST * S::kAMax;

    const int tid = int(threadIdx.x);
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / S::wn;
    const int warp_n = warp - warp_m * S::wn;
    // The same strip rasterization the dense kernel runs: the CTAs of one strip
    // walk n before m, so the strip's B blocks are the ones still in the cache.
    int blk_m = int(blockIdx.x), blk_n = int(blockIdx.y);
    if constexpr (S::Raster > 0) {
        const int bm = int(gridDim.x), bn = int(gridDim.y);
        const int gsize = S::Raster * bn;
        const int full = bm / S::Raster;
        const int bid = blk_n * bm + blk_m;
        if (bid < full * gsize) {
            const int gid = bid / gsize;
            const int idx = bid - gid * gsize;
            blk_m = gid * S::Raster + idx % S::Raster;
            blk_n = idx / S::Raster;
        } else {
            const int rem = bm - full * S::Raster;
            const int idx = bid - full * gsize;
            blk_m = full * S::Raster + idx % rem;
            blk_n = idx / rem;
        }
    }
    const int m_tile = blk_m * BM;
    const int n_tile = blk_n * BN;
    const int M = int(p.m), N = int(p.n), K = int(p.k);
    const int k_tiles = (K + BK - 1) / BK;
    const int bsz = int(p.batch > 0 ? p.batch : 1);
    const int splits = int(gridDim.z) / bsz;
    const int batch = int(blockIdx.z) / splits;
    const int split = int(blockIdx.z) - batch * splits;
    const int kt_begin = int(int64_t(k_tiles) * split / splits);
    const int kt_end = int(int64_t(k_tiles) * (split + 1) / splits);
    if (kt_begin >= kt_end) {
        return;
    }

    const Elem* const A =
        static_cast<const Elem*>(p.a) + f16_batch_offset(p.bsize, p.a_bstr, p.batch_rank, batch);
    // B is a block stream: the batch strides were resolved in blocks, so the
    // slice offset scales by the block size before it is a byte address.
    const uint8_t* const Bb = static_cast<const uint8_t*>(p.b) +
                              f16_batch_offset(p.bsize, p.b_bstr, p.batch_rank, batch) *
                                  block_bytes;
    Elem* const C =
        static_cast<Elem*>(p.c) + f16_batch_offset(p.bsize, p.c_bstr, p.batch_rank, batch);
    const GemmEpilogue epi =
        gemm_epilogue_for_batch<Elem>(p.epi, p.bsize, p.r_bstr, p.batch_rank, batch);

    constexpr int kVecPerRow = S::kVecPerRow;
    constexpr int kRowsPerPass = S::kRowsPerPass;
    const int vrow = tid / kVecPerRow;
    const int vcol = (tid % kVecPerRow) * 8;
    constexpr int kMVecPerRow = BM / 8;
    constexpr int mRowStep = S::threads / kMVecPerRow;
    const int mCol = (tid % kMVecPerRow) * 8;
    const int mRow0 = tid / kMVecPerRow;
    constexpr int kNVec = BN / 8;

    auto fill = [&](int kt, int stage) {
        if (kt >= kt_end) {
            cp_async_commit_group();
            return;
        }
        const int k0 = kt * BK;
        Elem* const sA = sAbase + stage * S::kAMax;
        Elem* const sB = sBbase + stage * S::kBMax;
        const bool kFull = k0 + BK <= K;
        const bool aFull = kFull && m_tile + BM <= M;

        if constexpr (!TransA) {
            if (aFull && p.vec) {
#pragma unroll
                for (int r = 0; r < BM / kRowsPerPass; ++r) {
                    const int row = vrow + r * kRowsPerPass;
                    cp_async_ca_16B(&sA[f16_swz<S>(row, vcol)],
                                    A + int64_t(m_tile + row) * p.lda + k0 + vcol);
                }
            } else {
#pragma unroll
                for (int r = 0; r < BM / kRowsPerPass; ++r) {
                    const int row = vrow + r * kRowsPerPass;
                    const int am = m_tile + row;
                    if (p.vec && am < M && k0 + vcol + 8 <= K) {
                        cp_async_ca_16B(&sA[f16_swz<S>(row, vcol)],
                                        A + int64_t(am) * p.lda + k0 + vcol);
                    } else {
                        for (int t = 0; t < 8; ++t) {
                            sA[f16_swz<S>(row, vcol + t)] =
                                (am < M && k0 + vcol + t < K) ?
                                    A[int64_t(am) * p.lda + k0 + vcol + t] :
                                    gemm_zero<Elem>();
                        }
                    }
                }
            }
        } else {
            for (int kr = mRow0; kr < BK; kr += mRowStep) {
                const int kk = k0 + kr;
                const int gm = m_tile + mCol;
                if (p.vec && (aFull || (kk < K && gm + 8 <= M))) {
                    cp_async_ca_16B(&sA[f16_swz_t<S>(kr, mCol)], A + int64_t(kk) * p.lda + gm);
                } else {
                    for (int t = 0; t < 8; ++t) {
                        sA[f16_swz_t<S>(kr, mCol + t)] = (kk < K && gm + t < M) ?
                                                             A[int64_t(kk) * p.lda + gm + t] :
                                                             gemm_zero<Elem>();
                    }
                }
            }
        }

        if constexpr (TransB) {
            // `[n][k]`: one thread owns eight consecutive k of one row, and
            // with a 32-deep k tile those eight are always inside one block.
#pragma unroll
            for (int r = 0; r < BN / kRowsPerPass; ++r) {
                const int row = vrow + r * kRowsPerPass;
                const int an = n_tile + row;
                const int kk = k0 + vcol;
                uint4 raw = make_uint4(0, 0, 0, 0);
                if (an < N && kk < K) {
                    const int rem = K - kk;
                    raw = q4_decode8<Elem>(
                        Bb + (int64_t(an) * b_row_blocks + (kk >> 5)) * block_bytes, vcol,
                        rem < 8 ? rem : 8, fmt);
                }
                *reinterpret_cast<uint4*>(&sB[f16_swz<S>(row, vcol)]) = raw;
            }
        } else {
            // `[k][n]`: a vector is eight consecutive n of one k row, so its
            // eight values are eight consecutive quants of the block that row
            // owns at that n — the same one-header decode as the k-major pose,
            // with the row and the block role swapped.
            for (int i = tid; i < BK * kNVec; i += S::threads) {
                const int kr = i / kNVec;
                const int nv = i - kr * kNVec;
                const int kk = k0 + kr;
                const int nn = n_tile + nv * 8;
                uint4 raw = make_uint4(0, 0, 0, 0);
                if (kk < K && nn < N) {
                    const uint8_t* blk =
                        Bb + (int64_t(kk) * b_row_blocks + (nn >> 5)) * block_bytes;
                    const int j0 = nn & 31;
                    if (nn + 8 <= N) {
                        raw = q4_decode8<Elem>(blk, j0, 8, fmt);
                    } else {
                        float v[8];
#pragma unroll
                        for (int t = 0; t < 8; ++t) {
                            v[t] = nn + t < N ? q4_decode1(blk, j0 + t, fmt) : 0.0f;
                        }
                        raw = make_uint4(gemm_pack2<Elem>(v[0], v[1]),
                                         gemm_pack2<Elem>(v[2], v[3]),
                                         gemm_pack2<Elem>(v[4], v[5]),
                                         gemm_pack2<Elem>(v[6], v[7]));
                    }
                }
                *reinterpret_cast<uint4*>(&sB[f16_swz_n<S>(kr, nv * 8)]) = raw;
            }
        }
        cp_async_commit_group();
    };

    float acc[MA][NA][4];
#pragma unroll
    for (int i = 0; i < MA; ++i)
#pragma unroll
        for (int j = 0; j < NA; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                acc[i][j][t] = 0.0f;
            }

#pragma unroll
    for (int t = 0; t < ST - 1; ++t) {
        fill(kt_begin + t, (kt_begin + t) % ST);
    }
    for (int kt = kt_begin; kt < kt_end; ++kt) {
        const int stage = kt % ST;
        cp_async_wait_group<ST - 2>();
        __syncthreads();
        fill(kt + ST - 1, (kt + ST - 1) % ST);
        f16_gemm_tile<S, Elem, TransA, TransB>(sAbase + stage * S::kAMax,
                                               sBbase + stage * S::kBMax, warp_m, warp_n, lane,
                                               acc);
    }

    f16_gemm_writeback<S, Elem>(p, smem, m_tile, n_tile, warp_m, warp_n, lane, C, epi, acc);
}

/// The narrow-m GEMV for a block weight: at most sixteen rows, one thread per
/// column, and the weight decoded straight out of its blocks into registers.
///
/// The tile above computes a 64-row mma for a product that has one to sixteen
/// rows, and at these shapes the dense engine leaves the tile entirely — its
/// `small_nt` loop is the one to beat. That loop stages B through shared memory
/// because a row-major f16 weight read along k by each lane is a 16-byte
/// segment per lane; a Q4 row read along k is 20 bytes for thirty-two values,
/// and the decode that produces them costs about what the staging did, so the
/// block is decoded in place and never enters shared memory. A stays k-major in
/// shared memory, as it does there: every thread reads the same `float4`s, so
/// the reads broadcast.
///
/// Against the dense route this wins to four rows and loses from eight (see
/// `q4_narrow_applies`): one weighted value costs four instructions to decode
/// and one per row to multiply, so the route pays only while a column's weight
/// is used a few times. Against the tile it wins to sixteen, which is why the
/// loop runs that far.
template <class Elem, int BM>
__global__ __launch_bounds__(detail::kSmallNTCols) void q4_small_nt_kernel(GemmParams p,
                                                                          int64_t b_row_blocks,
                                                                          int fmt_i,
                                                                          int block_bytes)
{
    constexpr int THREADS = detail::kSmallNTCols;
    constexpr int blocks_per_chunk = 4;
    constexpr int KC = 32 * blocks_per_chunk; ///< four llama.cpp blocks a chunk
    constexpr int APAD = (BM + 3) & ~3;
    const DataType fmt = static_cast<DataType>(fmt_i);

    const Elem* __restrict__ A = static_cast<const Elem*>(p.a);
    const uint8_t* B = static_cast<const uint8_t*>(p.b);
    const int M = int(p.m);
    const int N = int(p.n);
    const int K = int(p.k);
    const int lda = int(p.lda);

    const int tid = int(threadIdx.x);
    const int col = int(blockIdx.x) * THREADS + tid;
    const int batch = detail::gemm_block_batch(p);
    const int split = detail::gemm_block_split(p);
    int kt_begin = 0;
    int kt_count = 0;
    detail::gemm_split_range(p, KC, split, kt_begin, kt_count);
    if (kt_count <= 0) {
        return;
    }
    const int k_begin = kt_begin * KC;
    const int k_end = min(K, (kt_begin + kt_count) * KC);

    A += detail::gemm_batch_offset(p, p.a_batch, batch);
    // The weight's batch stride counts blocks, so the slice is scaled to bytes
    // before it is an address.
    B += detail::gemm_batch_offset(p, p.b_batch, batch) * block_bytes;

    __shared__ __align__(16) float sAt[KC][APAD];

    float acc[BM];
#pragma unroll
    for (int m = 0; m < BM; ++m) {
        acc[m] = 0.0f;
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
        for (int m = 0; m < BM; ++m) {
            acc[m] = fmaf(av[m], b, acc[m]);
        }
    };

    for (int k0 = k_begin; k0 < k_end; k0 += KC) {
        const int kc = min(KC, k_end - k0);
        __syncthreads();
        for (int i = tid; i < BM * KC; i += THREADS) {
            const int m = i / KC;
            const int kk = i - m * KC;
            sAt[kk][m] = (m < M && k0 + kk < K) ? float(A[int64_t(m) * lda + k0 + kk]) : 0.0f;
        }
        __syncthreads();

        if (col >= N) {
            continue;
        }
        // One chunk is four blocks, so a thread has four independent block
        // loads to overlap with the arithmetic and the block's decode is paid
        // once for thirty-two values.
        const uint8_t* const bbase =
            B + (int64_t(col) * b_row_blocks + (k0 >> 5)) * block_bytes;
        // Every block of the chunk is read before any of it is decoded: a lane's
        // block is one sector of its own row, so these are scattered requests,
        // and what carries them is having all of them outstanding at once.
        Q4Block qb[blocks_per_chunk];
        int nblk = 0;
        for (int b = 0; b < blocks_per_chunk && b * 32 < kc; ++b) {
            q4_load_block(bbase + int64_t(b) * block_bytes, fmt, qb[b]);
            nblk = b + 1;
        }
        for (int b = 0; b < nblk; ++b) {
            const int kb = b * 32;
            const uint8_t* blk = bbase + int64_t(b) * block_bytes;
            const Q4Block& cur = qb[b];
            const int lim = min(32, kc - kb);
            int kk = 0;
            for (; kk + 8 <= lim; kk += 8) {
                float v[8];
                q4_block8(cur, kk, fmt, v);
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    rank1(kb + kk + t, v[t]);
                }
            }
            for (; kk < lim; ++kk) {
                rank1(kb + kk, q4_decode1(blk, kk, fmt));
            }
        }
    }

    float* part = p.partial + (int64_t(batch) * p.k_splits + split) * int64_t(M) * N;
    if (col >= N) {
        return;
    }
#pragma unroll
    for (int m = 0; m < BM; ++m) {
        if (m < M) {
            part[int64_t(m) * N + col] = acc[m];
        }
    }
}

/// The dynamic shared memory opt-in, once per (element type, pose).
template <class S, class Elem> bool q4_prepare()
{
    static const bool ok = [] {
        const int bytes = S::kSmemBytes;
        const auto set = [bytes](const void* fn) {
            return cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) ==
                   cudaSuccess;
        };
        if (!set(reinterpret_cast<const void*>(&q4_gemm_kernel<S, Elem, true, true>)) ||
            !set(reinterpret_cast<const void*>(&q4_gemm_kernel<S, Elem, true, false>)) ||
            !set(reinterpret_cast<const void*>(&q4_gemm_kernel<S, Elem, false, true>)) ||
            !set(reinterpret_cast<const void*>(&q4_gemm_kernel<S, Elem, false, false>))) {
            cudaGetLastError();
            return false;
        }
        return true;
    }();
    return ok;
}

template <class S, class Elem> bool q4_run(const Q4Call& call, cudaStream_t stream)
{
    if (!q4_prepare<S, Elem>()) {
        return false;
    }

    F16Params q;
    q.a = call.gemm.a;
    q.b = call.gemm.b;
    q.c = call.gemm.c;
    q.m = call.gemm.m;
    q.n = call.gemm.n;
    q.k = call.gemm.k;
    q.lda = call.gemm.lda;
    q.ldb = call.gemm.ldb;
    q.ldc = call.gemm.ldc;
    q.alpha = call.gemm.alpha;
    q.epi.bias = call.gemm.bias;
    q.epi.bias_row = call.gemm.bias_row;
    q.epi.alpha = call.gemm.alpha;
    q.epi.beta = call.gemm.beta;
    q.epi.residual = call.gemm.residual;
    q.epi.ldr = call.gemm.ldr;
    q.epi.type = static_cast<EpilogueActivateType>(call.gemm.epilogue);
    q.epi.ep_alpha = call.gemm.ep_alpha;
    q.epi.ep_beta = call.gemm.ep_beta;
    q.epi.min_clip = call.gemm.min_clip;
    q.epi.max_clip = call.gemm.max_clip;
    q.batch_rank = call.gemm.batch_rank;
    q.batch = call.gemm.batch > 0 ? call.gemm.batch : 1;
    q.vec = call.gemm.vec;
    for (int d = 0; d < 4; ++d) {
        q.bsize[d] = call.gemm.bsize[d];
        q.a_bstr[d] = call.gemm.a_bstr[d];
        q.b_bstr[d] = call.gemm.b_bstr[d];
        q.c_bstr[d] = call.gemm.c_bstr[d];
        q.r_bstr[d] = call.gemm.r_bstr[d];
    }

    const dim3 grid(unsigned((call.gemm.m + S::BM - 1) / S::BM),
                    unsigned((call.gemm.n + S::BN - 1) / S::BN), 1);
    constexpr int kLaunchSmem = S::kSmemBytes;
    const TileDesc tile{S::BM, S::BN, S::BK, S::threads, S::kSmemBytes, S::MinBlocks};
    int splits = f16_splits(call.gemm, tile);
    float* partial = nullptr;
    if (splits > 1) {
        partial = static_cast<float*>(f16_alloc(
            int64_t(splits) * q.batch * call.gemm.m * call.gemm.n * int64_t(sizeof(float)),
            stream));
        if (partial == nullptr) {
            splits = 1;
        }
    }

    const auto launch = [&](const dim3& g) {
        if (call.gemm.trans_a && call.gemm.trans_b) {
            q4_gemm_kernel<S, Elem, true, true>
                <<<g, S::threads, kLaunchSmem, stream>>>(q, call.b_row_blocks, call.fmt);
        } else if (call.gemm.trans_a) {
            q4_gemm_kernel<S, Elem, true, false>
                <<<g, S::threads, kLaunchSmem, stream>>>(q, call.b_row_blocks, call.fmt);
        } else if (call.gemm.trans_b) {
            q4_gemm_kernel<S, Elem, false, true>
                <<<g, S::threads, kLaunchSmem, stream>>>(q, call.b_row_blocks, call.fmt);
        } else {
            q4_gemm_kernel<S, Elem, false, false>
                <<<g, S::threads, kLaunchSmem, stream>>>(q, call.b_row_blocks, call.fmt);
        }
    };
    if (splits > 1) {
        const dim3 sgrid(grid.x, grid.y, unsigned(splits * q.batch));
        q.partial = partial;
        launch(sgrid);
        constexpr int kThreads = 256;
        const int64_t plane = int64_t(call.gemm.m) * call.gemm.n;
        const dim3 rgrid(unsigned((plane + kThreads - 1) / kThreads), unsigned(splits), 1);
        f16_gemm_reduce_kernel<Elem><<<rgrid, kThreads, 0, stream>>>(q);
        f16_free(partial, stream);
        return cudaGetLastError() == cudaSuccess;
    }

    launch(dim3(grid.x, grid.y, unsigned(q.batch)));
    return cudaGetLastError() == cudaSuccess;
}

} // namespace nnops::backend::cuda
