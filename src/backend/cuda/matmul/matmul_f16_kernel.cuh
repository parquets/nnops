/// @file matmul_f16_kernel.cuh
/// @brief The 16-bit GEMM kernel and its launch machinery, shared by the two
///        translation units that instantiate it: one per element type.
///
/// Only those two include this file. The conv TUs want the tile shapes and the
/// device primitives, which stay in matmul_f16.cuh, and must not pay for the
/// kernel bodies or the 24 instantiations each element type expands to.

#pragma once

#include "gemm_common.cuh"
#include "matmul_f16.cuh"
#include "matmul_f16_host.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>

namespace nnops::backend::cuda {

/// The 16-bit GEMM: per-warp value tile from `S`, cp.async ring, swizzled panels.
template <class S, class Elem, bool TransA, bool TransB>
__global__ __launch_bounds__(S::threads, S::MinBlocks) void f16_gemm_kernel(F16Params p)
{
    constexpr int BM = S::BM, BN = S::BN, BK = S::BK, ST = S::stages;
    constexpr int WM = S::WM, WN = S::WN, MA = S::MA, NA = S::NA;
    constexpr int LDA = S::LDA, LDN = S::LDN;

    // One buffer for both phases: the panels until the k loop ends, the staging
    // buffer after it. The loop's closing barrier is what makes the hand-off
    // legal without a second block of shared memory. Dynamic rather than static,
    // so one launch path covers every tile: the host does the opt-in once and
    // passes the size it computed from the very same constant the kernel
    // indexes with.
    extern __shared__ __align__(16) unsigned char f16_smem_raw[];
    Elem* const smem = reinterpret_cast<Elem*>(f16_smem_raw);
    Elem* const sAbase = smem;
    Elem* const sBbase = smem + ST * BM * LDA;

    const int tid = int(threadIdx.x);
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / S::wn;
    const int warp_n = warp - warp_m * S::wn;
    // L2 rasterization: the CTAs of one strip walk n before m, so the strip's
    // B panels are the ones still in the cache (the f32 engine's scheme).
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
    const int m0 = m_tile + warp_m * WM;
    const int n0 = n_tile + warp_n * WN;
    const int M = int(p.m), N = int(p.n), K = int(p.k);
    const int k_tiles = (K + BK - 1) / BK;
    // Split-K: blockIdx.z picks a proportional k-tile range. The tile grid alone
    // leaves a shallow product a fraction of the machine, and there is no third
    // axis left to cut.
    const int bsz = int(p.batch > 0 ? p.batch : 1);
    const int splits = int(gridDim.z) / bsz;
    const int batch = int(blockIdx.z) / splits;
    const int split = int(blockIdx.z) - batch * splits;
    const int kt_begin = int(int64_t(k_tiles) * split / splits);
    const int kt_end = int(int64_t(k_tiles) * (split + 1) / splits);
    if (kt_begin >= kt_end) {
        return;
    }

    // Batch offsets resolved once; a broadcast axis carries stride 0.
    const Elem* const A =
        static_cast<const Elem*>(p.a) + f16_batch_offset(p.bsize, p.a_bstr, p.batch_rank, batch);
    const Elem* const B =
        static_cast<const Elem*>(p.b) + f16_batch_offset(p.bsize, p.b_bstr, p.batch_rank, batch);
    Elem* const C =
        static_cast<Elem*>(p.c) + f16_batch_offset(p.bsize, p.c_bstr, p.batch_rank, batch);

    // gmem -> smem: one 16 B vector per thread per pass. A k row is BK/8 vectors
    // wide, so a pass covers threads/(BK/8) rows and the panel takes BM divided
    // by that many passes. B arrives transposed, so both panels are [rows][k].
    constexpr int kVecPerRow = S::kVecPerRow;
    constexpr int kRowsPerPass = S::kRowsPerPass;
    const int vrow = tid / kVecPerRow;
    const int vcol = (tid % kVecPerRow) * 8;
    // m-major A: one 8-wide vector of every `mRowStep`-th k row.
    constexpr int kMVecPerRow = BM / 8;
    constexpr int mRowStep = S::threads / kMVecPerRow;
    const int mCol = (tid % kMVecPerRow) * 8;
    const int mRow0 = tid / kMVecPerRow;

    constexpr int kNVec = BN / 8; // 16 B vectors along one B row of the NN panel
    // One call is one committed group, whether or not it copies anything: the
    // wait's group count has to stay uniform when a split runs dry near its end.
    auto fill = [&](int kt, int stage) {
        if (kt >= kt_end) {
            cp_async_commit_group();
            return;
        }
        const int k0 = kt * BK;
        Elem* const sA = sAbase + stage * S::kAMax;
        Elem* const sB = sBbase + stage * S::kBMax;
        // Interior tiles need no bounds test at all; the per-vector tests were
        // what took 13% off the f32 sibling until they were split out here.
        const bool kFull = k0 + BK <= K;
        const bool aFull = kFull && m_tile + BM <= M;
        const bool bFull = kFull && n_tile + BN <= N;
        if constexpr (!TransB) {
            // B arrives `[k][n]`: a vector is eight consecutive n of one k row.
            if (bFull && p.vec) {
                for (int i = tid; i < BK * kNVec; i += S::threads) {
                    const int kr = i / kNVec;
                    const int nv = i - kr * kNVec;
                    cp_async_ca_16B(&sB[f16_swz_n<S>(kr, nv * 8)],
                                    B + int64_t(k0 + kr) * p.ldb + n_tile + nv * 8);
                }
            } else
                for (int i = tid; i < BK * kNVec; i += S::threads) {
                    const int kr = i / kNVec;
                    const int nv = i - kr * kNVec;
                    const int kk = k0 + kr;
                    const int nn = n_tile + nv * 8;
                    if (p.vec && kk < K && nn + 8 <= N) {
                        cp_async_ca_16B(&sB[f16_swz_n<S>(kr, nv * 8)],
                                        B + int64_t(kk) * p.ldb + nn);
                    } else {
                        for (int t = 0; t < 8; ++t) {
                            sB[f16_swz_n<S>(kr, nv * 8 + t)] =
                                (kk < K && nn + t < N) ? B[int64_t(kk) * p.ldb + nn + t] :
                                                         gemm_zero<Elem>();
                        }
                    }
                }
        }
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
            // A arrives `[k][m]`: a vector is eight consecutive m of one k row,
            // and the panel keeps that order, so the copy stays async.
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
        // The two panels are not the same height: a CTA wider than it is tall
        // (128x256) has twice the B rows of A, so each panel walks its own.
        if constexpr (TransB) {
            if (bFull && p.vec) {
#pragma unroll
                for (int r = 0; r < BN / kRowsPerPass; ++r) {
                    const int row = vrow + r * kRowsPerPass;
                    cp_async_ca_16B(&sB[f16_swz<S>(row, vcol)],
                                    B + int64_t(n_tile + row) * p.ldb + k0 + vcol);
                }
            } else
#pragma unroll
                for (int r = 0; r < BN / kRowsPerPass; ++r) {
                    const int row = vrow + r * kRowsPerPass;
                    const int an = n_tile + row;
                    if (p.vec && an < N && k0 + vcol + 8 <= K) {
                        cp_async_ca_16B(&sB[f16_swz<S>(row, vcol)],
                                        B + int64_t(an) * p.ldb + k0 + vcol);
                    } else {
                        for (int t = 0; t < 8; ++t) {
                            sB[f16_swz<S>(row, vcol + t)] =
                                (an < N && k0 + vcol + t < K) ?
                                    B[int64_t(an) * p.ldb + k0 + vcol + t] :
                                    gemm_zero<Elem>();
                        }
                    }
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

    // Prime every stage but one; the loop then keeps exactly ST groups in flight,
    // so `wait_group<ST-1>` retires precisely the tile about to be consumed.
#pragma unroll
    for (int t = 0; t < ST - 1; ++t) {
        fill(kt_begin + t, (kt_begin + t) % ST);
    }
    for (int kt = kt_begin; kt < kt_end; ++kt) {
        const int stage = kt % ST;
        // One barrier a tile: the wait retires this tile's group, the barrier
        // then publishes it and retires every warp's reads of the tile before
        // it -- whose stage is the one the fill now overwrites.
        cp_async_wait_group<ST - 2>();
        __syncthreads();
        fill(kt + ST - 1, (kt + ST - 1) % ST);

        f16_gemm_tile<S, Elem, TransA, TransB>(sAbase + stage * S::kAMax,
                                                    sBbase + stage * S::kBMax, warp_m, warp_n,
                                                    lane, acc);
    }

    // A split's block stops at its accumulator: the sum and the alpha belong to
    // the reduce pass, which sees every plane.
    if (p.partial != nullptr) {
        float* part = p.partial + (int64_t(batch) * splits + split) * M * N;
#pragma unroll
        for (int i = 0; i < MA; ++i) {
            const int gm = m0 + i * 16 + lane / 4;
#pragma unroll
            for (int j = 0; j < NA; ++j) {
                const int gn = n0 + j * 8 + 2 * (lane % 4);
                if (gm < M && gn + 1 < N) {
                    part[int64_t(gm) * N + gn] = acc[i][j][0];
                    part[int64_t(gm) * N + gn + 1] = acc[i][j][1];
                    if (gm + 8 < M) {
                        part[int64_t(gm + 8) * N + gn] = acc[i][j][2];
                        part[int64_t(gm + 8) * N + gn + 1] = acc[i][j][3];
                    }
                } else {
                    const int rows[2] = {gm, gm + 8};
                    for (int rr = 0; rr < 2; ++rr) {
                        for (int cc = 0; cc < 2; ++cc) {
                            if (rows[rr] < M && gn + cc < N) {
                                part[int64_t(rows[rr]) * N + gn + cc] = acc[i][j][rr * 2 + cc];
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    // The staging buffer is the panel buffer, and the loop no longer ends on a
    // barrier: this one retires every warp's last ldmatrix before the epilogue
    // writes over it.
    __syncthreads();

    // C: lane l holds (m = l/4, n = 2*(l%4)), plus the +1 column and +8 row.
    const int cm = lane / 4;
    const int cn = 2 * (lane % 4);
    if constexpr (!S::StagedEpi) {
        // Straight out: one 4 B store per atom row, masked at the tile edge.
        // Cheaper than staging (no barrier, no round trip) but the stores are
        // scattered, which is why this is the search's other arm, not the
        // default. A row pitch that is not a whole number of 4 B stores -- an
        // odd N on a dense destination -- walks the same two elements as 2 B
        // writes instead, which is what an unpadded N=4103 does.
        const bool pair4 = (p.ldc % 2) == 0;
#pragma unroll
        for (int i = 0; i < MA; ++i) {
            const int gm = m0 + i * 16 + cm;
#pragma unroll
            for (int j = 0; j < NA; ++j) {
                const int gn = n0 + j * 8 + cn;
                for (int rr = 0; rr < 2; ++rr) {
                    const int row = gm + rr * 8;
                    if (row >= M) {
                        continue;
                    }
                    Elem* const dst = C + int64_t(row) * p.ldc;
                    const float v0 = gemm_epilogue_element<Elem>(p.epi, acc[i][j][rr * 2], row, gn, dst);
                    const float v1 =
                        gn + 1 < N ?
                            gemm_epilogue_element<Elem>(p.epi, acc[i][j][rr * 2 + 1], row, gn + 1, dst) :
                            0.0f;
                    if (pair4 && gn + 2 <= N) {
                        *reinterpret_cast<uint32_t*>(dst + gn) = gemm_pack2<Elem>(v0, v1);
                    } else {
                        for (int cc = 0; cc < 2 && gn + cc < N; ++cc) {
                            dst[gn + cc] = gemm_from_float<Elem>(cc == 0 ? v0 : v1);
                        }
                    }
                }
            }
        }
        return;
    }

    // Every slot of the tile is written -- the tail is masked on the way out, not
    // on the way in -- so the staging pass needs no bounds test.
    Elem* const stage = smem;
#pragma unroll
    for (int i = 0; i < MA; ++i) {
        const int em = warp_m * WM + i * 16 + cm;
        const int gm = m0 + i * 16 + cm;
#pragma unroll
        for (int j = 0; j < NA; ++j) {
            const int en = warp_n * WN + j * 8 + cn;
            const int gn = n0 + j * 8 + cn;
            // Every slot is written and the tail is masked on the way out, so an
            // out-of-range slot stores zero: the epilogue's own loads (bias, the
            // beta term, the residual) must not run past the tensors.
            const bool in_hi = gm < M && gn < N;
            const bool in_lo = gm + 8 < M && gn < N;
            const Elem* const dhi = C + int64_t(gm) * p.ldc;
            const Elem* const dlo = C + int64_t(gm + 8) * p.ldc;
            const float v0 = in_hi ? gemm_epilogue_element<Elem>(p.epi, acc[i][j][0], gm, gn, dhi) : 0.0f;
            const float v1 = in_hi && gn + 1 < N ?
                                 gemm_epilogue_element<Elem>(p.epi, acc[i][j][1], gm, gn + 1, dhi) :
                                 0.0f;
            const float v2 =
                in_lo ? gemm_epilogue_element<Elem>(p.epi, acc[i][j][2], gm + 8, gn, dlo) : 0.0f;
            const float v3 = in_lo && gn + 1 < N ?
                                 gemm_epilogue_element<Elem>(p.epi, acc[i][j][3], gm + 8, gn + 1, dlo) :
                                 0.0f;
            *reinterpret_cast<uint32_t*>(&stage[f16_epi_offset<S>(em, en)]) =
                gemm_pack2<Elem>(v0, v1);
            *reinterpret_cast<uint32_t*>(&stage[f16_epi_offset<S>(em + 8, en)]) =
                gemm_pack2<Elem>(v2, v3);
        }
    }
    __syncthreads();

    // 16 B per thread per pass: eight consecutive columns of one row. The chunk
    // the swizzle moved them to is the chunk this thread reads, so the store
    // stays inside one 256 B span of the row and the banks stay clean.
    const bool vec = (p.ldc % 8) == 0;
    const int total = BM * S::kNChunks;
    for (int idx = tid; idx < total; idx += S::threads) {
        const int row = idx / S::kNChunks;
        const int c = idx - row * S::kNChunks;
        const int gm = m_tile + row;
        if (gm >= M) {
            continue;
        }
        const int gn = n_tile + c * 8;
        const Elem* src = &stage[f16_epi_offset<S>(row, c * 8)];
        Elem* dst = C + int64_t(gm) * p.ldc + gn;
        if (vec && gn + 8 <= N) {
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
        } else {
            for (int t = 0; t < 8 && gn + t < N; ++t) {
                dst[t] = src[t];
            }
        }
    }
}

/// Sum the split-K planes and write the destination, one thread per element.
template <class Elem> __global__ void f16_gemm_reduce_kernel(F16Params p)
{
    const int64_t plane = int64_t(p.m) * p.n;
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= plane) {
        return;
    }
    const int splits = int(gridDim.y);
    const int m = int(idx / p.n);
    const int n = int(idx - int64_t(m) * p.n);
    for (int64_t b = 0; b < p.batch; ++b) {
        float sum = 0.0f;
        for (int s = 0; s < splits; ++s) {
            sum += p.partial[(b * splits + s) * plane + idx];
        }
        Elem* const dst =
            static_cast<Elem*>(p.c) + f16_batch_offset(p.bsize, p.c_bstr, p.batch_rank, b);
        dst[int64_t(m) * p.ldc + n] = gemm_from_float<Elem>(p.alpha * sum);
    }
}


/// The candidate tiles, in the order a sweep reports them. The list is the
/// search's menu, not a ranking: the automatic pick is Tile7 below, and the
/// rows here are the arms that were measured against it. Tile0 is the 64x64
/// warp tile on a 2x2 grid -- doubling the warp's n span halves the ldmatrix
/// traffic per MAC, and on two CTAs of 128 threads that trade is a clear win at
/// every size past a few hundred square (4096^3: 4.39 ms against 5.38 for the
/// 64x32 tile and 7.07 for the tiled engine it replaced).
using Tile0 = F16TileShape<64, 64, 2, 2, 32, 3>;
using Tile1 = F16TileShape<64, 32, 2, 4, 32, 3>;
using Tile2 = F16TileShape<64, 32, 1, 4, 32, 3>;
/// The winning tile under the search's other two switches: a skewed panel
/// instead of the XOR swizzle, and an unstaged epilogue.
using Tile3 = F16TileShape<64, 64, 2, 2, 32, 3, 0, true>;
using Tile4 = F16TileShape<64, 64, 2, 2, 32, 3, 1, false>;
/// The shallow-grid tile, the same fix f32 needed: a 64x64 CTA of 128 threads
/// with a 32x32 warp tile, so the accumulators are 32 floats and four blocks
/// live on an SM (24.6 KB of panels each). A 2048^3 square then has 1024 blocks
/// instead of 256, which is what fills the machine there.
using Tile5 = F16TileShape<32, 32, 2, 2, 32, 3, 1, true, 4>;
/// The wide-CTA arm: the shallow row's 32x32 warp tile on a 4x4 warp grid, so
/// the CTA is 128x128 on 512 threads and A/B are re-read a quarter as often.
/// It loses on this card, everywhere -- 4096^3 TN 5.77 ms against 4.69 for
/// Tile7, 2048^3 0.761 against 0.664 -- because 49 KB of panels allow one CTA
/// an SM, and the block-wide barrier every k tile then stalls all 16 warps
/// with no second CTA to run. Kept as the sweep's arm, not as a candidate.
using Tile6 = F16TileShape<32, 32, 4, 4, 32, 3, 1, true, 1, 8>;
/// The production tile with the strip rasterization: same 64x64 CTA and same
/// four-resident footprint, only the CTA order changes, so the strip's B panels
/// are still in L2 when the next m tile asks for them.
using Tile7 = F16TileShape<32, 32, 2, 2, 32, 3, 1, true, 4, 8>;
struct TileDesc {
    int bm, bn, bk, threads, smem_bytes, min_blocks;
};

constexpr int kTileCount = 8;
constexpr TileDesc kTileTable[kTileCount] = {
    {Tile0::BM, Tile0::BN, Tile0::BK, Tile0::threads, Tile0::kSmemBytes, Tile0::MinBlocks},
    {Tile1::BM, Tile1::BN, Tile1::BK, Tile1::threads, Tile1::kSmemBytes, Tile1::MinBlocks},
    {Tile2::BM, Tile2::BN, Tile2::BK, Tile2::threads, Tile2::kSmemBytes, Tile2::MinBlocks},
    {Tile3::BM, Tile3::BN, Tile3::BK, Tile3::threads, Tile3::kSmemBytes, Tile3::MinBlocks},
    {Tile4::BM, Tile4::BN, Tile4::BK, Tile4::threads, Tile4::kSmemBytes, Tile4::MinBlocks},
    {Tile5::BM, Tile5::BN, Tile5::BK, Tile5::threads, Tile5::kSmemBytes, Tile5::MinBlocks},
    {Tile6::BM, Tile6::BN, Tile6::BK, Tile6::threads, Tile6::kSmemBytes, Tile6::MinBlocks},
    {Tile7::BM, Tile7::BN, Tile7::BK, Tile7::threads, Tile7::kSmemBytes, Tile7::MinBlocks},
};

/// The automatic row: the shallow-grid tile (a 32x32 warp tile on four warps,
/// four blocks to an SM) plus the strip rasterization. The raster is worth 8%
/// at 4096^3 and costs nothing at 1024^3, so it ships on the production row;
/// Tile5 stays in the menu as its no-raster twin for the sweep.
constexpr int kF16TileAuto = 7;

/// Queried once per process; 0 when unavailable.
inline int f16_sm_count()
{
    static const int count = [] {
        int dev = 0;
        int n = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) {
            return 0;
        }
        if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) {
            return 0;
        }
        return n;
    }();
    return count;
}

inline int64_t f16_l2_bytes()
{
    static const int64_t bytes = [] {
        int dev = 0;
        int n = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) {
            return int64_t{0};
        }
        if (cudaDeviceGetAttribute(&n, cudaDevAttrL2CacheSize, dev) != cudaSuccess) {
            return int64_t{0};
        }
        return int64_t(n);
    }();
    return bytes;
}

/// A private pool for the planes. The default pool releases on every free, which
/// on WDDM costs milliseconds per pair; this one keeps 128 MiB resident.
inline cudaMemPool_t f16_pool()
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

inline void* f16_alloc(int64_t bytes, cudaStream_t stream)
{
    if (bytes <= 0) {
        return nullptr;
    }
    void* ptr = nullptr;
    const cudaMemPool_t pool = f16_pool();
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

inline void f16_free(void* ptr, cudaStream_t stream)
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

/// How many k ranges to cut. The split exists to fill the machine, so the grid
/// is measured against the tile's *resident slots* (`SMs x blocks per SM`), not
/// against the SM count: a 64x64 CTA that lives four to an SM makes 152 slots
/// on this card, and a 64-block decode grid is 42% of one wave, not "full".
/// The planes are still the price -- every split writes an f32 plane and the
/// reduce reads it back -- so the cut is taken only when the grid is starved
/// (under an eighth of a wave) or the planes are cheap against the operands
/// (under a quarter of them). Both conditions are the f32 engine's.
inline int f16_splits(const F16Call& call, const TileDesc& tile)
{
    const int forced = detail::splitk_env();
    if (forced == 0) {
        return 1;
    }
    // The reduce pass sums the planes and applies alpha; the rest of the
    // epilogue (bias, beta·C, the residual, the activation) only runs on the
    // single-pass write-back, so a call that asks for any of it stays whole.
    const bool plain_epi = call.bias == nullptr && call.beta == 0.0f &&
                           call.residual == nullptr && call.epilogue == 0;
    if (!plain_epi) {
        return 1;
    }
    constexpr int64_t kMaxPlaneBytes = int64_t{64} << 20;
    const int64_t m_tiles = (call.m + tile.bm - 1) / tile.bm;
    const int64_t n_tiles = (call.n + tile.bn - 1) / tile.bn;
    const int64_t blocks = m_tiles * n_tiles;
    const int64_t k_tiles = (call.k + tile.bk - 1) / tile.bk;
    if (k_tiles < 2) {
        return 1;
    }
    if (forced > 1) {
        return int(std::max<int64_t>(1, std::min<int64_t>(forced, std::min<int64_t>(k_tiles, 16))));
    }
    const int sm = f16_sm_count();
    if (sm <= 0 || blocks <= 0) {
        return 1;
    }
    const int64_t slots = int64_t(sm) * std::max(tile.min_blocks, 1);
    if (blocks >= slots) {
        return 1;
    }

    int64_t splits = std::min<int64_t>((slots + blocks - 1) / blocks, 16);
    splits = std::min<int64_t>(splits, k_tiles / 2);
    const int64_t plane = int64_t(call.m) * call.n * int64_t(sizeof(float));
    const int64_t l2 = f16_l2_bytes();
    if (l2 > 0) {
        splits = std::min<int64_t>(splits, std::max<int64_t>(l2 * 9 / 8 / plane, 1));
    }
    while (splits > 1 && splits * plane > kMaxPlaneBytes) {
        splits /= 2;
    }
    if (splits < 2) {
        return 1;
    }
    // Half precision: two bytes an element. The split is worth its planes only
    // when the grid is starved or the planes are a small part of the stream.
    const int64_t partial_bytes = int64_t{8} * splits * plane; // f32 write + read
    const int64_t operand_bytes = int64_t{2} * call.k * (int64_t(call.m) + call.n);
    const bool starved = blocks * 8 <= slots;
    const bool cheap = partial_bytes * 4 <= operand_bytes;
    // A square product's planes are f32 against f16 operands, so the starved
    // branch needs its own bound: 128^3 and 256^3 (planes 16-32x the operands)
    // measured 1.97x and 1.63x slower cut than uncut. Decode shapes stay well
    // inside eight operands (1x1024 and 16x1024 Linear, 16x4096x4096 tb).
    const bool small_planes = partial_bytes <= operand_bytes * 8;
    return (cheap || (starved && small_planes)) ? int(splits) : 1;
}

/// The grid gate. Below a full wave this kernel's per-block cost -- the panel
/// fill and the staged epilogue -- is what dominates, and the narrow-m loop is
/// ahead there (256^2: 0.0205 ms against 0.0151).
/// Once the tile grid fills the machine the trade reverses hard (4096^3: 5.03 ms
/// against 8.36), so that is the line. An explicit tile skips the gate: it is
/// how a sweep reaches every shape.
inline bool f16_grid_fills(const F16Call& call, const TileDesc& tile)
{
    const int sm = f16_sm_count();
    if (sm <= 0) {
        return true;
    }
    const int64_t blocks = ((call.m + tile.bm - 1) / tile.bm) * ((call.n + tile.bn - 1) / tile.bn);
    return blocks >= int64_t(sm);
}

/// A dynamic shared memory request above 48 KB needs the opt-in, once per
/// instantiation; a static `__shared__` array could not carry these tiles.
template <class S, class Elem> bool f16_prepare()
{
    static const bool ok = [] {
        const int bytes = S::kSmemBytes;
        const auto set = [bytes](const void* fn) {
            return cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) ==
                   cudaSuccess;
        };
        if (!set(reinterpret_cast<const void*>(&f16_gemm_kernel<S, Elem, true, true>)) ||
            !set(reinterpret_cast<const void*>(&f16_gemm_kernel<S, Elem, true, false>)) ||
            !set(reinterpret_cast<const void*>(&f16_gemm_kernel<S, Elem, false, true>)) ||
            !set(reinterpret_cast<const void*>(&f16_gemm_kernel<S, Elem, false, false>))) {
            cudaGetLastError();
            return false;
        }
        return true;
    }();
    return ok;
}

template <class S, class Elem> bool f16_run(const F16Call& call, cudaStream_t stream)
{
    if (!f16_prepare<S, Elem>()) {
        return false;
    }

    F16Params q;
    q.a = call.a;
    q.b = call.b;
    q.c = call.c;
    q.m = call.m;
    q.n = call.n;
    q.k = call.k;
    q.lda = call.lda;
    q.ldb = call.ldb;
    q.ldc = call.ldc;
    q.alpha = call.alpha;
    q.epi.bias = call.bias;
    q.epi.bias_row = call.bias_row;
    q.epi.alpha = call.alpha;
    q.epi.beta = call.beta;
    q.epi.residual = call.residual;
    q.epi.ldr = call.ldr;
    q.epi.type = static_cast<EpilogueActivateType>(call.epilogue);
    q.epi.ep_alpha = call.ep_alpha;
    q.epi.ep_beta = call.ep_beta;
    q.epi.min_clip = call.min_clip;
    q.epi.max_clip = call.max_clip;
    q.batch_rank = call.batch_rank;
    q.batch = call.batch > 0 ? call.batch : 1;
    q.vec = call.vec;
    for (int d = 0; d < 4; ++d) {
        q.bsize[d] = call.bsize[d];
        q.a_bstr[d] = call.a_bstr[d];
        q.b_bstr[d] = call.b_bstr[d];
        q.c_bstr[d] = call.c_bstr[d];
    }

    const dim3 grid(unsigned((call.m + S::BM - 1) / S::BM), unsigned((call.n + S::BN - 1) / S::BN),
                    1);
    constexpr int kLaunchSmem = S::kSmemBytes;
    const TileDesc tile{S::BM, S::BN, S::BK, S::threads, S::kSmemBytes, S::MinBlocks};
    int splits = f16_splits(call, tile);
    float* partial = nullptr;
    if (splits > 1) {
        partial = static_cast<float*>(f16_alloc(
            int64_t(splits) * q.batch * call.m * call.n * int64_t(sizeof(float)), stream));
        if (partial == nullptr) {
            splits = 1;
        }
    }

    const auto launch = [&](const dim3& g) {
        if (call.trans_a && call.trans_b) {
            f16_gemm_kernel<S, Elem, true, true><<<g, S::threads, kLaunchSmem, stream>>>(q);
        } else if (call.trans_a) {
            f16_gemm_kernel<S, Elem, true, false><<<g, S::threads, kLaunchSmem, stream>>>(q);
        } else if (call.trans_b) {
            f16_gemm_kernel<S, Elem, false, true><<<g, S::threads, kLaunchSmem, stream>>>(q);
        } else {
            f16_gemm_kernel<S, Elem, false, false><<<g, S::threads, kLaunchSmem, stream>>>(q);
        }
    };
    if (splits > 1) {
        const dim3 sgrid(grid.x, grid.y, unsigned(splits * q.batch));
        q.partial = partial;
        launch(sgrid);
        constexpr int kThreads = 256;
        const int64_t plane = int64_t(call.m) * call.n;
        const dim3 rgrid(unsigned((plane + kThreads - 1) / kThreads), unsigned(splits), 1);
        f16_gemm_reduce_kernel<Elem><<<rgrid, kThreads, 0, stream>>>(q);
        f16_free(partial, stream);
        return cudaGetLastError() == cudaSuccess;
    }

    launch(dim3(grid.x, grid.y, unsigned(q.batch)));
    return cudaGetLastError() == cudaSuccess;
}

/// The tile menu, per element type: one instantiation per (tile, pose).
/// The menu in two halves. Each half is instantiated in its own translation
/// unit, so the eight shapes a half expands to (four poses, two transposes)
/// build beside the other halves instead of after them.
template <class Elem>
inline bool f16_dispatch_lo(const F16Call& call, int tile, cudaStream_t stream)
{
    switch (tile) {
    case 0:
        return f16_run<Tile0, Elem>(call, stream);
    case 1:
        return f16_run<Tile1, Elem>(call, stream);
    case 2:
        return f16_run<Tile2, Elem>(call, stream);
    default:
        return false;
    }
}

template <class Elem>
inline bool f16_dispatch_hi(const F16Call& call, int tile, cudaStream_t stream)
{
    switch (tile) {
    case 3:
        return f16_run<Tile3, Elem>(call, stream);
    case 4:
        return f16_run<Tile4, Elem>(call, stream);
    case 5:
        return f16_run<Tile5, Elem>(call, stream);
    case 6:
        return f16_run<Tile6, Elem>(call, stream);
    case 7:
        return f16_run<Tile7, Elem>(call, stream);
    default:
        return false;
    }
}

} // namespace nnops::backend::cuda
