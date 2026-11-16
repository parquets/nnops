/// @file matmul_f32.cuh
/// @brief The hand-written f32 GEMM: a SIMT FFMA tile in the same skeleton as
///        the 16-bit PTX kernel, so the CuTe engine has nothing left to do for
///        a dense product.
///
/// f32 has no tensor-core path on this target (TF32 was measured at 15.3 TFLOPS
/// against 16.2 for plain FMA, and at a precision the operator contract does not
/// allow), so this is the classic register-blocked kernel: a cp.async panel
/// ring, a per-thread 8x8 (or 8x4) accumulator, and the k loop unrolled four
/// deep so each panel read is one 16 B vector.
///
/// The layout is the one the FFMA ratio wants. Per four k values a lane reads
/// `LaneRow` A vectors and `LaneCol` B vectors and issues
/// `LaneRow * LaneCol * 4` FMAs, so the smem traffic per FLOP is
/// 1/(LaneCol) + 1/(LaneRow) vectors' worth -- 8x8 is the cheapest square the
/// register file can hold twice.
///
/// Both B layouts share one code path: the n-major pose transposes into the
/// panel on the way in (four scalar stores per 16 B read), so the inner loop
/// only ever sees a k-major panel.
///
/// Batch and split-K share blockIdx.z: `z = b * splits + split`, and the batch
/// strides are 0 for a broadcast operand -- the same numpy-style contract the
/// CuTe engine's `gemm_batch_layout` produces.

#pragma once

#include "matmul_epilogue.cuh"
#include "ptx.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace nnops::backend::cuda {

/// One instantiation of the f32 kernel.
template <int LaneRow_, int LaneCol_, int WarpsM_, int WarpsN_, int BK_, int ST_, int KU_ = 4,
          int Raster_ = 8, int Pad_ = 4, int MinBlocks_ = 1>
struct ManualFShape {
    static constexpr int LaneRow = LaneRow_; ///< rows one thread owns
    static constexpr int LaneCol = LaneCol_; ///< columns one thread owns
    /// A warp is 32 lanes as an 8x4 grid: 8 lanes down the rows, 4 across.
    static constexpr int warpRows = 8 * LaneRow;
    static constexpr int warpCols = 4 * LaneCol;
    static constexpr int wm = WarpsM_;
    static constexpr int wn = WarpsN_;
    static constexpr int threads = wm * wn * 32;
    static constexpr int BM = wm * warpRows;
    static constexpr int BN = wn * warpCols;
    static constexpr int BK = BK_;
    static constexpr int stages = ST_;
    static constexpr int KU = KU_; ///< k values per panel vector
    /// L2 rasterization strip: the grid is walked in groups of this many m tiles,
    /// stepping n inside a group. 0 disables it. Without it the block order is
    /// plain row-major, and at 4096^3 that alone cost 13% -- the tiles that
    /// follow a block along n re-read a B panel nothing has kept warm.
    static constexpr int Raster = Raster_;
    /// Panels are padded, not swizzled: sixteen bytes of row offset puts eight
    /// consecutive rows on eight bank quads, and f32 panels are small enough
    /// that the padding costs no residency.
    /// Every panel keeps its operand's arrival order, so the stride is whichever
    /// of the two shapes the pose makes it. A is `[m][k]` when it comes in
    /// row-major and `[k][m]` when it comes transposed; B likewise. `Pad` is the
    /// skew that spreads consecutive rows over the bank quads: four elements
    /// costs a fifth of the panel, and a tile that would rather have the
    /// residency than the conflict can ask for none.
    static constexpr int Pad = Pad_;
    /// `__launch_bounds__` minimum blocks: it caps the register count, which is
    /// what stands between the wide tiles and a fourth resident block.
    static constexpr int MinBlocks = MinBlocks_;
    static constexpr int LDA = BK + Pad;  ///< A as `[m][k]`
    static constexpr int LDAt = BM + Pad; ///< A as `[k][m]`
    static constexpr int LDN = BN + Pad;  ///< B as `[k][n]`
    static constexpr int LDk = BK + Pad;  ///< B as `[n][k]`
    static constexpr int kKSteps = BK / KU;
    static constexpr int kVecPerRow = BK / KU;
    static constexpr int kRowsPerPass = threads / kVecPerRow;
    /// One buffer serves every pose: each panel is sized to the larger of its
    /// two orders.
    static constexpr int kAMax = BM * LDA > BK* LDAt ? BM* LDA : BK* LDAt;
    static constexpr int kBMax = BN * LDk > BK* LDN ? BN* LDk : BK* LDN;
    static constexpr int kPanelFloats = stages * (kAMax + kBMax);
    static constexpr int kSmemBytes = kPanelFloats * int(sizeof(float));

    static_assert(BK % KU == 0, "the k unroll must divide the stage depth");
    static_assert(LaneCol % 4 == 0, "a lane's columns fill whole 16 B vectors in the n-major pose");
    static_assert(BK % 4 == 0, "a panel row must hold whole 16 B vectors");
    static_assert(BM % kRowsPerPass == 0 && BN % kRowsPerPass == 0,
                  "a fill pass must divide both panels into whole row blocks");
};

struct ManualFParams {
    const void* a = nullptr;
    const void* b = nullptr;
    void* c = nullptr;
    int64_t m = 0, n = 0, k = 0;
    int64_t lda = 0, ldb = 0, ldc = 0;
    float alpha = 1.0f;
    /// Batch: up to four leading axes, each a (size, stride) pair with stride 0
    /// for a broadcast operand -- the same numpy-style contract the CuTe engine's
    /// `gemm_batch_layout` produces, resolved here instead.
    int batch_rank = 0;
    int64_t batch = 1;
    int64_t bsize[4] = {1, 1, 1, 1};
    int64_t a_bstr[4] = {0, 0, 0, 0};
    int64_t b_bstr[4] = {0, 0, 0, 0};
    int64_t c_bstr[4] = {0, 0, 0, 0};
    /// The operator-level epilogue: alpha, bias, beta, clamp, activation,
    /// residual, in that order.
    ManualEpilogue epi;
    /// Split-K scratch: [splits][m][n] f32 planes, or null for the single pass.
    float* partial = nullptr;
};

/// The f32 tile: 8x8 per thread, cp.async ring, padded panels.
template <class S, bool TransA, bool TransB>
__global__ __launch_bounds__(S::threads, S::MinBlocks) void manual_gemm_f_kernel(ManualFParams p)
{
    constexpr int BM = S::BM, BN = S::BN, BK = S::BK, ST = S::stages;
    constexpr int KU = S::KU, LR = S::LaneRow, LC = S::LaneCol;
    constexpr int LDA = TransA ? S::LDAt : S::LDA;
    constexpr int LDK = TransB ? S::LDk : S::LDN;

    extern __shared__ __align__(16) unsigned char manual_f_smem[];
    float* const sAbase = reinterpret_cast<float*>(manual_f_smem);
    float* const sBbase = sAbase + ST * S::kAMax;

    const int tid = int(threadIdx.x);
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / S::wn;
    const int warp_n = warp - warp_m * S::wn;
    // L2 rasterization: the CTAs of one strip walk n before m, so the strip's
    // B panels are the ones still in the cache.
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

    const int splits = int(gridDim.z / unsigned(p.batch > 0 ? p.batch : 1));
    const int z = int(blockIdx.z);
    const int batch = splits > 0 ? z / splits : 0;
    const int split = splits > 0 ? z - batch * splits : 0;
    const int kt_begin = int(int64_t(k_tiles) * split / (splits > 0 ? splits : 1));
    const int kt_end = int(int64_t(k_tiles) * (split + 1) / (splits > 0 ? splits : 1));
    if (kt_begin >= kt_end) {
        return;
    }

    const float* const A = static_cast<const float*>(p.a) +
                           manual_batch_offset(p.bsize, p.a_bstr, p.batch_rank, batch);
    const float* const B = static_cast<const float*>(p.b) +
                           manual_batch_offset(p.bsize, p.b_bstr, p.batch_rank, batch);
    float* const C =
        static_cast<float*>(p.c) + manual_batch_offset(p.bsize, p.c_bstr, p.batch_rank, batch);

    const int vrow = tid / S::kVecPerRow;
    const int vcol = (tid % S::kVecPerRow) * KU;
    // The fill walks a fixed set of rows per thread, so each row's base pointer
    // is resolved once and every k tile only adds its offset. The host refuses a
    // call whose strides overflow 32 bits, which is what lets these be ints.
    const int lda_i = int(p.lda);
    const int ldb_i = int(p.ldb);
    // Column fills: one 16 B vector of every `kStep`-th k row, so the row index
    // is a base plus a constant step instead of a divide and a modulo per tile.
    constexpr int kAVecPerRow = BM / KU;
    constexpr int kAStep = S::threads / kAVecPerRow;
    const int aCol = (tid % kAVecPerRow) * KU;
    const int aRow0 = tid / kAVecPerRow;
    constexpr int kBSvecPerRow = BN / KU;
    constexpr int kBStep = S::threads / kBSvecPerRow;
    const int bColOff = (tid % kBSvecPerRow) * KU;
    const int bRow0 = tid / kBSvecPerRow;
    constexpr int kAPass = BM / S::kRowsPerPass;
    const float* aFill[kAPass];
    const float* bFill[kAPass];
#pragma unroll
    for (int r = 0; r < kAPass; ++r) {
        aFill[r] = A + (m_tile + vrow + r * S::kRowsPerPass) * lda_i;
        bFill[r] = B + (n_tile + vrow + r * S::kRowsPerPass) * ldb_i;
    }

    // One call is one committed group whether or not it copies, so the wait's
    // group count stays uniform when a split runs dry.
    auto fill = [&](int kt, int stage) {
        if (kt >= kt_end) {
            cp_async_commit_group();
            return;
        }
        const int k0 = kt * BK;
        float* const sA = sAbase + stage * S::kAMax;
        float* const sB = sBbase + stage * S::kBMax;
        // Block-level fullness, decided once per k tile: an interior tile fills
        // with no bounds test at all, which is the split CuTe's load_a/load_b
        // make. Per-vector tests were 41 ISETP per k tile in the SASS body and
        // the FFMA density sat at 72% against CuTe's 86%.
        const bool kFull = k0 + BK <= K;
        const bool aFull = kFull && m_tile + BM <= M;
        const bool bFull = kFull && n_tile + BN <= N;
        if constexpr (TransB) {
            if (bFull) {
                for (int i = tid; i < BN * S::kVecPerRow; i += S::threads) {
                    const int row = i / S::kVecPerRow;
                    const int kk = (i % S::kVecPerRow) * KU;
                    cp_async_ca_16B(&sB[row * LDK + kk], B + (n_tile + row) * ldb_i + k0 + kk);
                }
            } else {
                for (int i = tid; i < BN * S::kVecPerRow; i += S::threads) {
                    const int row = i / S::kVecPerRow;
                    const int kk = (i % S::kVecPerRow) * KU;
                    const int an = n_tile + row;
                    if (an < N && k0 + kk + KU <= K) {
                        cp_async_ca_16B(&sB[row * LDK + kk], B + an * ldb_i + k0 + kk);
                    } else {
                        for (int t = 0; t < KU; ++t) {
                            sB[row * LDK + kk + t] =
                                (an < N && k0 + kk + t < K) ? B[an * ldb_i + k0 + kk + t] : 0.0f;
                        }
                    }
                }
            }
        } else {
            // B arrives `[k][n]`, which is the order the panel keeps: a vector
            // is four consecutive n of one k row, so the fill is the same async
            // copy the k-major pose gets. Transposing into a k-major panel costs
            // the pipeline its overlap outright -- measured at 30% on 2048^3 --
            // and the inner loop reads this order column-wise instead.
            if (bFull) {
                for (int kr = bRow0; kr < BK; kr += kBStep) {
                    cp_async_ca_16B(&sB[kr * LDK + bColOff],
                                    B + (k0 + kr) * ldb_i + n_tile + bColOff);
                }
            } else
                for (int kr = bRow0; kr < BK; kr += kBStep) {
                    const int nn = bColOff;
                    const int kk = k0 + kr;
                    const int gn = n_tile + nn;
                    if (kk < K && gn + KU <= N) {
                        cp_async_ca_16B(&sB[kr * LDK + nn], B + kk * ldb_i + gn);
                    } else {
                        for (int t = 0; t < KU; ++t) {
                            sB[kr * LDK + nn + t] =
                                (kk < K && gn + t < N) ? B[kk * ldb_i + gn + t] : 0.0f;
                        }
                    }
                }
        }
        if constexpr (!TransA) {
            if (aFull) {
#pragma unroll
                for (int r = 0; r < kAPass; ++r) {
                    cp_async_ca_16B(&sA[(vrow + r * S::kRowsPerPass) * LDA + vcol],
                                    aFill[r] + k0 + vcol);
                }
            } else
#pragma unroll
                for (int r = 0; r < kAPass; ++r) {
                    const int row = vrow + r * S::kRowsPerPass;
                    const int am = m_tile + row;
                    if (am < M && k0 + vcol + KU <= K) {
                        cp_async_ca_16B(&sA[row * LDA + vcol], aFill[r] + k0 + vcol);
                    } else {
                        for (int t = 0; t < KU; ++t) {
                            sA[row * LDA + vcol + t] =
                                (am < M && k0 + vcol + t < K) ? aFill[r][k0 + vcol + t] : 0.0f;
                        }
                    }
                }
        } else {
            // A arrives `[k][m]`: a vector is four consecutive m of one k row,
            // and the panel keeps that order, so the copy is async here too.
            if (aFull) {
                for (int kr = aRow0; kr < BK; kr += kAStep) {
                    cp_async_ca_16B(&sA[kr * LDA + aCol], A + (k0 + kr) * lda_i + m_tile + aCol);
                }
            } else
                for (int kr = aRow0; kr < BK; kr += kAStep) {
                    const int mm = aCol;
                    const int kk = k0 + kr;
                    const int gm = m_tile + mm;
                    if (kk < K && gm + KU <= M) {
                        cp_async_ca_16B(&sA[kr * LDA + mm], A + kk * lda_i + gm);
                    } else {
                        for (int t = 0; t < KU; ++t) {
                            sA[kr * LDA + mm + t] =
                                (kk < K && gm + t < M) ? A[kk * lda_i + gm + t] : 0.0f;
                        }
                    }
                }
        }
        cp_async_commit_group();
    };

    float acc[LR][LC];
#pragma unroll
    for (int r = 0; r < LR; ++r)
#pragma unroll
        for (int c = 0; c < LC; ++c) {
            acc[r][c] = 0.0f;
        }

    // A warp is eight lane-rows by four lane-cols. A k-major A panel hands a
    // lane the rows eight apart (one per vector); a transposed one hands it four
    // consecutive rows per vector, and the groups step by the warp's 32 so the
    // eight lane reads stay on eight bank quads.
    const int lrow = lane % 8;
    const int lcol = lane / 8;
    const int row0 = warp_m * S::warpRows + (TransA ? 4 * lrow : lrow);
    const int col0 = warp_n * S::warpCols + lcol * LC;
    // The row a lane's accumulator index r owns.
    const auto acc_row = [&](int r) {
        return TransA ? row0 + (r / 4) * 32 + (r % 4) : row0 + r * 8;
    };

#pragma unroll
    for (int t = 0; t < ST - 1; ++t) {
        fill(kt_begin + t, (kt_begin + t) % ST);
    }
    for (int kt = kt_begin; kt < kt_end; ++kt) {
        const int stage = kt % ST;
        // One barrier a tile: the wait retires this tile's group, the barrier
        // then both publishes it and retires every warp's reads of the tile
        // before it -- whose stage is exactly the one the fill now overwrites.
        cp_async_wait_group<ST - 2>();
        __syncthreads();
        fill(kt + ST - 1, (kt + ST - 1) % ST);

        // Per-lane panel bases, hoisted out of the k groups. Measured honestly:
        // the SASS is byte-identical to spelling the full index per load, so
        // ptxas was already doing this -- the 544 IADD3 in this function are not
        // the load addressing. Kept because it reads as what the code means.
        const float* const aRow = sAbase + stage * S::kAMax + row0 * LDA;
        // The k-major B panel is indexed by row, so the column offset scales by
        // the row stride; the n-major one is indexed by column inside a row.
        const float* const bCol = sBbase + stage * S::kBMax + col0 * LDK;
        const float* const sA_m = sAbase + stage * S::kAMax + row0;
        const float* const sB_k = sBbase + stage * S::kBMax + col0;
#pragma unroll
        for (int ks = 0; ks < S::kKSteps; ++ks) {
            // One A-fragment array for both arrival orders: the k-major panel
            // fills it one entry per row, the m-major one one entry per (k, row
            // group). Declaring a second array for the other order left it
            // live -- the transposed pose used 177 registers against 152 and
            // lost its third resident block on that alone.
            constexpr int G = LR / 4;
            float4 af[LR];
            const int kv = ks * KU;
            if constexpr (TransA) {
#pragma unroll
                for (int t = 0; t < KU; ++t) {
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        af[t * G + g] =
                            *reinterpret_cast<const float4*>(&sA_m[(kv + t) * LDA + 32 * g]);
                    }
                }
            } else {
#pragma unroll
                for (int r = 0; r < LR; ++r) {
                    af[r] = *reinterpret_cast<const float4*>(&aRow[(r * 8) * LDA + kv]);
                }
            }
            if constexpr (TransB) {
                // `[n][k]`: one vector per column, holding four k values.
                float4 bv[LC];
#pragma unroll
                for (int c = 0; c < LC; ++c) {
                    bv[c] = *reinterpret_cast<const float4*>(&bCol[c * LDK + kv]);
                }
#pragma unroll
                for (int c = 0; c < LC; ++c) {
                    const float4 b = bv[c];
                    if constexpr (!TransA) {
#pragma unroll
                        for (int r = 0; r < LR; ++r) {
                            acc[r][c] = fmaf(af[r].x, b.x, acc[r][c]);
                            acc[r][c] = fmaf(af[r].y, b.y, acc[r][c]);
                            acc[r][c] = fmaf(af[r].z, b.z, acc[r][c]);
                            acc[r][c] = fmaf(af[r].w, b.w, acc[r][c]);
                        }
                    } else {
                        // The A vector holds four rows for one k, the B vector
                        // four k for one column: a 4-row by 4-k outer product.
                        static_assert(KU == 4, "the transposed-A nest is a four-deep unroll");
#pragma unroll
                        for (int g = 0; g < LR / 4; ++g) {
                            acc[4 * g + 0][c] = fmaf(af[0 * G + g].x, b.x, acc[4 * g + 0][c]);
                            acc[4 * g + 0][c] = fmaf(af[1 * G + g].x, b.y, acc[4 * g + 0][c]);
                            acc[4 * g + 0][c] = fmaf(af[2 * G + g].x, b.z, acc[4 * g + 0][c]);
                            acc[4 * g + 0][c] = fmaf(af[3 * G + g].x, b.w, acc[4 * g + 0][c]);
                            acc[4 * g + 1][c] = fmaf(af[0 * G + g].y, b.x, acc[4 * g + 1][c]);
                            acc[4 * g + 1][c] = fmaf(af[1 * G + g].y, b.y, acc[4 * g + 1][c]);
                            acc[4 * g + 1][c] = fmaf(af[2 * G + g].y, b.z, acc[4 * g + 1][c]);
                            acc[4 * g + 1][c] = fmaf(af[3 * G + g].y, b.w, acc[4 * g + 1][c]);
                            acc[4 * g + 2][c] = fmaf(af[0 * G + g].z, b.x, acc[4 * g + 2][c]);
                            acc[4 * g + 2][c] = fmaf(af[1 * G + g].z, b.y, acc[4 * g + 2][c]);
                            acc[4 * g + 2][c] = fmaf(af[2 * G + g].z, b.z, acc[4 * g + 2][c]);
                            acc[4 * g + 2][c] = fmaf(af[3 * G + g].z, b.w, acc[4 * g + 2][c]);
                            acc[4 * g + 3][c] = fmaf(af[0 * G + g].w, b.x, acc[4 * g + 3][c]);
                            acc[4 * g + 3][c] = fmaf(af[1 * G + g].w, b.y, acc[4 * g + 3][c]);
                            acc[4 * g + 3][c] = fmaf(af[2 * G + g].w, b.z, acc[4 * g + 3][c]);
                            acc[4 * g + 3][c] = fmaf(af[3 * G + g].w, b.w, acc[4 * g + 3][c]);
                        }
                    }
                }
            } else {
                // `[k][n]`: one vector per k of the unroll, holding four
                // adjacent columns. A lane's columns are contiguous, so a LC=8
                // tile reads two vectors per k and the accumulator's column is
                // the vector's component. Every lane in a lane-column reads the
                // same vector, which the hardware broadcasts.
                float4 bv[KU][LC / 4];
#pragma unroll
                for (int t = 0; t < KU; ++t) {
#pragma unroll
                    for (int g = 0; g < LC / 4; ++g) {
                        bv[t][g] = *reinterpret_cast<const float4*>(&sB_k[(kv + t) * LDK + 4 * g]);
                    }
                }
                static_assert(KU == 4, "the n-major inner loop is a four-deep unroll");
                if constexpr (!TransA) {
#pragma unroll
                    for (int r = 0; r < LR; ++r) {
                        const float4 v = af[r];
#pragma unroll
                        for (int g = 0; g < LC / 4; ++g) {
                            acc[r][4 * g + 0] = fmaf(v.x, bv[0][g].x, acc[r][4 * g + 0]);
                            acc[r][4 * g + 0] = fmaf(v.y, bv[1][g].x, acc[r][4 * g + 0]);
                            acc[r][4 * g + 0] = fmaf(v.z, bv[2][g].x, acc[r][4 * g + 0]);
                            acc[r][4 * g + 0] = fmaf(v.w, bv[3][g].x, acc[r][4 * g + 0]);
                            acc[r][4 * g + 1] = fmaf(v.x, bv[0][g].y, acc[r][4 * g + 1]);
                            acc[r][4 * g + 1] = fmaf(v.y, bv[1][g].y, acc[r][4 * g + 1]);
                            acc[r][4 * g + 1] = fmaf(v.z, bv[2][g].y, acc[r][4 * g + 1]);
                            acc[r][4 * g + 1] = fmaf(v.w, bv[3][g].y, acc[r][4 * g + 1]);
                            acc[r][4 * g + 2] = fmaf(v.x, bv[0][g].z, acc[r][4 * g + 2]);
                            acc[r][4 * g + 2] = fmaf(v.y, bv[1][g].z, acc[r][4 * g + 2]);
                            acc[r][4 * g + 2] = fmaf(v.z, bv[2][g].z, acc[r][4 * g + 2]);
                            acc[r][4 * g + 2] = fmaf(v.w, bv[3][g].z, acc[r][4 * g + 2]);
                            acc[r][4 * g + 3] = fmaf(v.x, bv[0][g].w, acc[r][4 * g + 3]);
                            acc[r][4 * g + 3] = fmaf(v.y, bv[1][g].w, acc[r][4 * g + 3]);
                            acc[r][4 * g + 3] = fmaf(v.z, bv[2][g].w, acc[r][4 * g + 3]);
                            acc[r][4 * g + 3] = fmaf(v.w, bv[3][g].w, acc[r][4 * g + 3]);
                        }
                    }
                } else {
                    // Both panels are in their n/m order: the A vector is four
                    // rows of one k, the B vector four columns of one k.
#pragma unroll
                    for (int g = 0; g < LR / 4; ++g) {
#pragma unroll
                        for (int h = 0; h < LC / 4; ++h) {
                            acc[4 * g + 0][4 * h + 0] =
                                fmaf(af[0 * G + g].x, bv[0][h].x, acc[4 * g + 0][4 * h + 0]);
                            acc[4 * g + 0][4 * h + 0] =
                                fmaf(af[1 * G + g].x, bv[1][h].x, acc[4 * g + 0][4 * h + 0]);
                            acc[4 * g + 0][4 * h + 0] =
                                fmaf(af[2 * G + g].x, bv[2][h].x, acc[4 * g + 0][4 * h + 0]);
                            acc[4 * g + 0][4 * h + 0] =
                                fmaf(af[3 * G + g].x, bv[3][h].x, acc[4 * g + 0][4 * h + 0]);
                            acc[4 * g + 0][4 * h + 1] =
                                fmaf(af[0 * G + g].x, bv[0][h].y, acc[4 * g + 0][4 * h + 1]);
                            acc[4 * g + 0][4 * h + 1] =
                                fmaf(af[1 * G + g].x, bv[1][h].y, acc[4 * g + 0][4 * h + 1]);
                            acc[4 * g + 0][4 * h + 1] =
                                fmaf(af[2 * G + g].x, bv[2][h].y, acc[4 * g + 0][4 * h + 1]);
                            acc[4 * g + 0][4 * h + 1] =
                                fmaf(af[3 * G + g].x, bv[3][h].y, acc[4 * g + 0][4 * h + 1]);
                            acc[4 * g + 0][4 * h + 2] =
                                fmaf(af[0 * G + g].x, bv[0][h].z, acc[4 * g + 0][4 * h + 2]);
                            acc[4 * g + 0][4 * h + 2] =
                                fmaf(af[1 * G + g].x, bv[1][h].z, acc[4 * g + 0][4 * h + 2]);
                            acc[4 * g + 0][4 * h + 2] =
                                fmaf(af[2 * G + g].x, bv[2][h].z, acc[4 * g + 0][4 * h + 2]);
                            acc[4 * g + 0][4 * h + 2] =
                                fmaf(af[3 * G + g].x, bv[3][h].z, acc[4 * g + 0][4 * h + 2]);
                            acc[4 * g + 0][4 * h + 3] =
                                fmaf(af[0 * G + g].x, bv[0][h].w, acc[4 * g + 0][4 * h + 3]);
                            acc[4 * g + 0][4 * h + 3] =
                                fmaf(af[1 * G + g].x, bv[1][h].w, acc[4 * g + 0][4 * h + 3]);
                            acc[4 * g + 0][4 * h + 3] =
                                fmaf(af[2 * G + g].x, bv[2][h].w, acc[4 * g + 0][4 * h + 3]);
                            acc[4 * g + 0][4 * h + 3] =
                                fmaf(af[3 * G + g].x, bv[3][h].w, acc[4 * g + 0][4 * h + 3]);
                            acc[4 * g + 1][4 * h + 0] =
                                fmaf(af[0 * G + g].y, bv[0][h].x, acc[4 * g + 1][4 * h + 0]);
                            acc[4 * g + 1][4 * h + 0] =
                                fmaf(af[1 * G + g].y, bv[1][h].x, acc[4 * g + 1][4 * h + 0]);
                            acc[4 * g + 1][4 * h + 0] =
                                fmaf(af[2 * G + g].y, bv[2][h].x, acc[4 * g + 1][4 * h + 0]);
                            acc[4 * g + 1][4 * h + 0] =
                                fmaf(af[3 * G + g].y, bv[3][h].x, acc[4 * g + 1][4 * h + 0]);
                            acc[4 * g + 1][4 * h + 1] =
                                fmaf(af[0 * G + g].y, bv[0][h].y, acc[4 * g + 1][4 * h + 1]);
                            acc[4 * g + 1][4 * h + 1] =
                                fmaf(af[1 * G + g].y, bv[1][h].y, acc[4 * g + 1][4 * h + 1]);
                            acc[4 * g + 1][4 * h + 1] =
                                fmaf(af[2 * G + g].y, bv[2][h].y, acc[4 * g + 1][4 * h + 1]);
                            acc[4 * g + 1][4 * h + 1] =
                                fmaf(af[3 * G + g].y, bv[3][h].y, acc[4 * g + 1][4 * h + 1]);
                            acc[4 * g + 1][4 * h + 2] =
                                fmaf(af[0 * G + g].y, bv[0][h].z, acc[4 * g + 1][4 * h + 2]);
                            acc[4 * g + 1][4 * h + 2] =
                                fmaf(af[1 * G + g].y, bv[1][h].z, acc[4 * g + 1][4 * h + 2]);
                            acc[4 * g + 1][4 * h + 2] =
                                fmaf(af[2 * G + g].y, bv[2][h].z, acc[4 * g + 1][4 * h + 2]);
                            acc[4 * g + 1][4 * h + 2] =
                                fmaf(af[3 * G + g].y, bv[3][h].z, acc[4 * g + 1][4 * h + 2]);
                            acc[4 * g + 1][4 * h + 3] =
                                fmaf(af[0 * G + g].y, bv[0][h].w, acc[4 * g + 1][4 * h + 3]);
                            acc[4 * g + 1][4 * h + 3] =
                                fmaf(af[1 * G + g].y, bv[1][h].w, acc[4 * g + 1][4 * h + 3]);
                            acc[4 * g + 1][4 * h + 3] =
                                fmaf(af[2 * G + g].y, bv[2][h].w, acc[4 * g + 1][4 * h + 3]);
                            acc[4 * g + 1][4 * h + 3] =
                                fmaf(af[3 * G + g].y, bv[3][h].w, acc[4 * g + 1][4 * h + 3]);
                            acc[4 * g + 2][4 * h + 0] =
                                fmaf(af[0 * G + g].z, bv[0][h].x, acc[4 * g + 2][4 * h + 0]);
                            acc[4 * g + 2][4 * h + 0] =
                                fmaf(af[1 * G + g].z, bv[1][h].x, acc[4 * g + 2][4 * h + 0]);
                            acc[4 * g + 2][4 * h + 0] =
                                fmaf(af[2 * G + g].z, bv[2][h].x, acc[4 * g + 2][4 * h + 0]);
                            acc[4 * g + 2][4 * h + 0] =
                                fmaf(af[3 * G + g].z, bv[3][h].x, acc[4 * g + 2][4 * h + 0]);
                            acc[4 * g + 2][4 * h + 1] =
                                fmaf(af[0 * G + g].z, bv[0][h].y, acc[4 * g + 2][4 * h + 1]);
                            acc[4 * g + 2][4 * h + 1] =
                                fmaf(af[1 * G + g].z, bv[1][h].y, acc[4 * g + 2][4 * h + 1]);
                            acc[4 * g + 2][4 * h + 1] =
                                fmaf(af[2 * G + g].z, bv[2][h].y, acc[4 * g + 2][4 * h + 1]);
                            acc[4 * g + 2][4 * h + 1] =
                                fmaf(af[3 * G + g].z, bv[3][h].y, acc[4 * g + 2][4 * h + 1]);
                            acc[4 * g + 2][4 * h + 2] =
                                fmaf(af[0 * G + g].z, bv[0][h].z, acc[4 * g + 2][4 * h + 2]);
                            acc[4 * g + 2][4 * h + 2] =
                                fmaf(af[1 * G + g].z, bv[1][h].z, acc[4 * g + 2][4 * h + 2]);
                            acc[4 * g + 2][4 * h + 2] =
                                fmaf(af[2 * G + g].z, bv[2][h].z, acc[4 * g + 2][4 * h + 2]);
                            acc[4 * g + 2][4 * h + 2] =
                                fmaf(af[3 * G + g].z, bv[3][h].z, acc[4 * g + 2][4 * h + 2]);
                            acc[4 * g + 2][4 * h + 3] =
                                fmaf(af[0 * G + g].z, bv[0][h].w, acc[4 * g + 2][4 * h + 3]);
                            acc[4 * g + 2][4 * h + 3] =
                                fmaf(af[1 * G + g].z, bv[1][h].w, acc[4 * g + 2][4 * h + 3]);
                            acc[4 * g + 2][4 * h + 3] =
                                fmaf(af[2 * G + g].z, bv[2][h].w, acc[4 * g + 2][4 * h + 3]);
                            acc[4 * g + 2][4 * h + 3] =
                                fmaf(af[3 * G + g].z, bv[3][h].w, acc[4 * g + 2][4 * h + 3]);
                            acc[4 * g + 3][4 * h + 0] =
                                fmaf(af[0 * G + g].w, bv[0][h].x, acc[4 * g + 3][4 * h + 0]);
                            acc[4 * g + 3][4 * h + 0] =
                                fmaf(af[1 * G + g].w, bv[1][h].x, acc[4 * g + 3][4 * h + 0]);
                            acc[4 * g + 3][4 * h + 0] =
                                fmaf(af[2 * G + g].w, bv[2][h].x, acc[4 * g + 3][4 * h + 0]);
                            acc[4 * g + 3][4 * h + 0] =
                                fmaf(af[3 * G + g].w, bv[3][h].x, acc[4 * g + 3][4 * h + 0]);
                            acc[4 * g + 3][4 * h + 1] =
                                fmaf(af[0 * G + g].w, bv[0][h].y, acc[4 * g + 3][4 * h + 1]);
                            acc[4 * g + 3][4 * h + 1] =
                                fmaf(af[1 * G + g].w, bv[1][h].y, acc[4 * g + 3][4 * h + 1]);
                            acc[4 * g + 3][4 * h + 1] =
                                fmaf(af[2 * G + g].w, bv[2][h].y, acc[4 * g + 3][4 * h + 1]);
                            acc[4 * g + 3][4 * h + 1] =
                                fmaf(af[3 * G + g].w, bv[3][h].y, acc[4 * g + 3][4 * h + 1]);
                            acc[4 * g + 3][4 * h + 2] =
                                fmaf(af[0 * G + g].w, bv[0][h].z, acc[4 * g + 3][4 * h + 2]);
                            acc[4 * g + 3][4 * h + 2] =
                                fmaf(af[1 * G + g].w, bv[1][h].z, acc[4 * g + 3][4 * h + 2]);
                            acc[4 * g + 3][4 * h + 2] =
                                fmaf(af[2 * G + g].w, bv[2][h].z, acc[4 * g + 3][4 * h + 2]);
                            acc[4 * g + 3][4 * h + 2] =
                                fmaf(af[3 * G + g].w, bv[3][h].z, acc[4 * g + 3][4 * h + 2]);
                            acc[4 * g + 3][4 * h + 3] =
                                fmaf(af[0 * G + g].w, bv[0][h].w, acc[4 * g + 3][4 * h + 3]);
                            acc[4 * g + 3][4 * h + 3] =
                                fmaf(af[1 * G + g].w, bv[1][h].w, acc[4 * g + 3][4 * h + 3]);
                            acc[4 * g + 3][4 * h + 3] =
                                fmaf(af[2 * G + g].w, bv[2][h].w, acc[4 * g + 3][4 * h + 3]);
                            acc[4 * g + 3][4 * h + 3] =
                                fmaf(af[3 * G + g].w, bv[3][h].w, acc[4 * g + 3][4 * h + 3]);
                        }
                    }
                }
            }
        }
    }

    // A split's block stops at its accumulator; the reduce pass owns the sum and
    // the alpha.
    if (p.partial != nullptr) {
        float* part = p.partial + (int64_t(batch) * splits + split) * M * N;
#pragma unroll
        for (int r = 0; r < LR; ++r) {
            const int gm = m_tile + acc_row(r);
            if (gm >= M) {
                continue;
            }
#pragma unroll
            for (int c = 0; c < LC; ++c) {
                const int gn = n_tile + col0 + c;
                if (gn < N) {
                    part[int64_t(gm) * N + gn] = acc[r][c];
                }
            }
        }
        return;
    }

    // Straight out, one 16 B store per four adjacent columns; a lane's columns
    // are contiguous, so the warp's stores cover whole rows.
    const bool vec4 = (p.ldc % 4) == 0;
#pragma unroll
    for (int r = 0; r < LR; ++r) {
        const int gm = m_tile + acc_row(r);
        if (gm >= M) {
            continue;
        }
        float* const crow = C + int64_t(gm) * p.ldc;
#pragma unroll
        for (int g = 0; g < LC / 4; ++g) {
            const int c = 4 * g;
            const int gn = n_tile + col0 + c;
            // The whole operator epilogue, per element; the vector store only
            // changes how the four results leave.
            float v[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                v[q] = manual_epilogue<float>(p.epi, acc[r][c + q], gm, gn + q, crow);
            }
            if (vec4 && gn + 4 <= N) {
                *reinterpret_cast<float4*>(crow + gn) = make_float4(v[0], v[1], v[2], v[3]);
            } else {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    if (gn + q < N) {
                        crow[gn + q] = v[q];
                    }
                }
            }
        }
    }
}

/// Sum the split-K planes and write the destination, one thread per element.
__global__ void manual_gemm_f_reduce_kernel(ManualFParams p)
{
    const int64_t plane = int64_t(p.m) * p.n;
    const int splits = int(gridDim.y);
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= plane) {
        return;
    }
    for (int64_t b = 0; b < p.batch; ++b) {
        const float* base = p.partial + (b * splits) * plane + idx;
        float sum = 0.0f;
        for (int s = 0; s < splits; ++s) {
            sum += base[s * plane];
        }
        const int m = int(idx / p.n);
        const int n = int(idx - int64_t(m) * p.n);
        static_cast<float*>(
            p.c)[manual_batch_offset(p.bsize, p.c_bstr, p.batch_rank, b) + int64_t(m) * p.ldc + n] =
            p.alpha * sum;
    }
}

} // namespace nnops::backend::cuda
