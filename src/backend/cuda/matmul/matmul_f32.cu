/// @file matmul_f32.cu
/// @brief The host-callable front of the hand-written f32 GEMM: the tile menu,
///        the split-K heuristic, the reduction planes, and the pose checks.
///
/// The shape of this file mirrors matmul_f16.cu deliberately -- the two
/// engines differ in the tile and the inner loop, not in how a call is resolved
/// -- so the reader who has one has the other.

#include "matmul_f32_host.hpp"
#include <nnops/core/epilogue.hpp>
#include "matmul_f32.cuh"
#include "detail/bench_knobs.hpp"

#include <algorithm>

namespace nnops::backend::cuda {

/// The f32 tile: 8x8 per thread, cp.async ring, padded panels.
template <class S, bool TransA, bool TransB>
__global__ __launch_bounds__(S::threads, S::MinBlocks) void f32_gemm_kernel(F32Params p)
{
    constexpr int BM = S::BM, BN = S::BN, BK = S::BK, ST = S::stages;
    constexpr int KU = S::KU, LR = S::LaneRow, LC = S::LaneCol;
    constexpr int LDA = TransA ? S::LDAt : S::LDA;
    constexpr int LDK = TransB ? S::LDk : S::LDN;
    // This pose's panel sizes: what the launch above asked for, and all this
    // kernel ever addresses.
    constexpr int kAPanel = S::a_panel(TransA);
    constexpr int kBPanel = S::b_panel(TransB);

    extern __shared__ __align__(16) unsigned char f32_smem[];
    float* const sAbase = reinterpret_cast<float*>(f32_smem);
    float* const sBbase = sAbase + ST * kAPanel;

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
                           f16_batch_offset(p.bsize, p.a_bstr, p.batch_rank, batch);
    const float* const B = static_cast<const float*>(p.b) +
                           f16_batch_offset(p.bsize, p.b_bstr, p.batch_rank, batch);
    float* const C =
        static_cast<float*>(p.c) + f16_batch_offset(p.bsize, p.c_bstr, p.batch_rank, batch);

    const int vrow = tid / S::kVecPerRow;
    const int vcol = (tid % S::kVecPerRow) * KU;
    // The fill walks a fixed set of rows per thread, so each row's base pointer
    // is resolved once and every k tile only adds its offset. The host refuses a
    // call whose strides overflow 32 bits, which is what lets these be ints.
    // 64-bit on purpose: the products below are per k tile per thread, so the
    // wider arithmetic costs nothing measurable, and an operand whose row pitch
    // times the tile index passes 2^31 stays addressable.
    const int64_t lda_i = p.lda;
    const int64_t ldb_i = p.ldb;
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
        float* const sA = sAbase + stage * kAPanel;
        float* const sB = sBbase + stage * kBPanel;
        // Block-level fullness, decided once per k tile: an interior tile fills
        // with no bounds test at all. Per-vector tests were 41 ISETP per k tile
        // in the SASS body and the FFMA density sat at 72%.
        const bool kFull = k0 + BK <= K;
        const bool aFull = kFull && m_tile + BM <= M;
        const bool bFull = kFull && n_tile + BN <= N;
        if constexpr (TransB) {
            if (bFull && p.vec) {
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
                    if (p.vec && an < N && k0 + kk + KU <= K) {
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
            if (bFull && p.vec) {
                for (int kr = bRow0; kr < BK; kr += kBStep) {
                    cp_async_ca_16B(&sB[kr * LDK + bColOff],
                                    B + (k0 + kr) * ldb_i + n_tile + bColOff);
                }
            } else
                for (int kr = bRow0; kr < BK; kr += kBStep) {
                    const int nn = bColOff;
                    const int kk = k0 + kr;
                    const int gn = n_tile + nn;
                    if (p.vec && kk < K && gn + KU <= N) {
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
            if (aFull && p.vec) {
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
                    if (p.vec && am < M && k0 + vcol + KU <= K) {
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
            if (aFull && p.vec) {
                for (int kr = aRow0; kr < BK; kr += kAStep) {
                    cp_async_ca_16B(&sA[kr * LDA + aCol], A + (k0 + kr) * lda_i + m_tile + aCol);
                }
            } else
                for (int kr = aRow0; kr < BK; kr += kAStep) {
                    const int mm = aCol;
                    const int kk = k0 + kr;
                    const int gm = m_tile + mm;
                    if (p.vec && kk < K && gm + KU <= M) {
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

        f32_gemm_tile<S, TransA, TransB>(sAbase + stage * kAPanel,
                                              sBbase + stage * kBPanel, warp_m, warp_n, lane,
                                              acc);
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
                v[q] = gemm_epilogue_element<float>(p.epi, acc[r][c + q], gm, gn + q, crow);
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
/// Internal linkage: this header is included by more than one translation unit,
/// and a kernel has no `inline`.
static __global__ void f32_gemm_reduce_kernel(F32Params p)
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
            p.c)[f16_batch_offset(p.bsize, p.c_bstr, p.batch_rank, b) + int64_t(m) * p.ldc + n] =
            p.alpha * sum;
    }
}

namespace {

/// The candidate tiles. Three is the whole menu: the sweep retired the rest
/// (a 128x256 CTA, a 64x64 CTA for shallow grids, a deeper ring), and every
/// tile here costs twelve kernels once the four poses are instantiated.
using Tile0 = F32TileShape<8, 8, 2, 4, 16, 2, 4, 8, 4, 2>; ///< 128x128, 8 warps
/// The 128-thread tiles declare three blocks of launch bounds: without it the
/// A-transposed instantiation lands at 177 registers and the driver gives it
/// two blocks a SM where the row-major one gets three (ncu: 16.4% against 23.7%
/// occupancy, and the whole 15% between the poses).
using Tile1 = F32TileShape<8, 8, 2, 2, 16, 2, 4, 8, 4, 3>; ///< 128x128, 4 warps
using Tile2 = F32TileShape<8, 8, 2, 2, 16, 2, 4, 0, 4, 3>; ///< the same, no raster
/// 64x128 on four warps: 31 KB of panels, so three blocks stay resident where
/// the 128x128 tile keeps two. The register block is unchanged; the cost is that
/// every CTA re-reads the A rows its neighbour also reads.
using Tile3 = F32TileShape<8, 8, 1, 4, 16, 2, 4, 8, 4, 3>;
/// Deeper rings: with one barrier a tile the ring depth is what stands in for
/// the fragment shadow a software-pipelined kernel would keep, and it costs
/// panels rather than registers.
using Tile4 = F32TileShape<8, 8, 2, 4, 16, 3>; ///< 128x128, three stages
/// 128x64 on eight warps with a 8x4 lane tile: 32 accumulators keep the block
/// at 115 registers, so two of them are resident and a 128-block grid is 1.68
/// waves instead of 1.12 -- the shape ncu showed cuBLAS running.
using Tile6 = F32TileShape<8, 4, 2, 4, 16, 2>;
/// The four-blocks-a-SM pair ncu pointed at: 128 threads, no panel skew (21-25 KB
/// of panels), and launch bounds that cap the block at 128 registers so the
/// fourth block fits. 64x64 doubles the block count again, which is what a
/// 1024^3 grid needs to become 1.68 waves instead of 1.12.
using Tile7 = F32TileShape<8, 4, 1, 4, 16, 2, 4, 8, 0, 4>; ///< 64x64, 128 thr
using Tile8 = F32TileShape<8, 8, 2, 2, 16, 2, 4, 8, 0, 4>; ///< 128x64, 128 thr
/// The 1024^3 fix: a 64x64 CTA on 128 threads with an 8x4 lane tile. Its 20.5 KB
/// of panels and ~110 registers let four blocks live on an SM (152 slots), so
/// the 256 blocks a 1024^3 grid makes are 1.68 waves instead of 1.12 -- the
/// wave fill cuBLAS's own 4-block tile gets, without the register cap that
/// killed the earlier attempt.
using Tile9 = F32TileShape<8, 4, 1, 4, 16, 2, 4, 8, 4, 4>;

struct TileDesc {
    int bm, bn, bk, threads;
};

constexpr int kTileCount = 10;
/// One row per index `f32_dense_call` dispatches: there is no Tile5, index 6
/// and index 9 both land on Tile9 (the switch's default), and the table has to
/// say so — a named row is the only thing that reads it, and a missing row is a
/// zero descriptor the grid gate would divide by.
constexpr TileDesc kTileTable[kTileCount] = {
    {Tile0::BM, Tile0::BN, Tile0::BK, Tile0::threads},
    {Tile1::BM, Tile1::BN, Tile1::BK, Tile1::threads},
    {Tile2::BM, Tile2::BN, Tile2::BK, Tile2::threads},
    {Tile3::BM, Tile3::BN, Tile3::BK, Tile3::threads},
    {Tile4::BM, Tile4::BN, Tile4::BK, Tile4::threads},
    {Tile6::BM, Tile6::BN, Tile6::BK, Tile6::threads},
    {Tile9::BM, Tile9::BN, Tile9::BK, Tile9::threads},
    {Tile7::BM, Tile7::BN, Tile7::BK, Tile7::threads},
    {Tile8::BM, Tile8::BN, Tile8::BK, Tile8::threads},
    {Tile9::BM, Tile9::BN, Tile9::BK, Tile9::threads},
};

constexpr int kDefaultTile = 0;
/// Past this many blocks the tile grid's B panels no longer stay resident and
/// the strip order starts winning; below it the plain order reuses them better.
constexpr int64_t kRasterBlocks = 512;

int f32_sm_count()
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

int64_t f32_l2_bytes()
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

cudaMemPool_t f32_pool()
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

void* f32_alloc(int64_t bytes, cudaStream_t stream)
{
    if (bytes <= 0) {
        return nullptr;
    }
    void* ptr = nullptr;
    const cudaMemPool_t pool = f32_pool();
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

void f32_free(void* ptr, cudaStream_t stream)
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

/// How many k ranges to cut, on the same terms as the 16-bit engine: fill the
/// machine to a quarter wave, keep the planes near the cache, cap the total.
int f32_splits(const F32Call& call, const TileDesc& tile)
{
    // The reduce pass sums the planes and applies alpha; the rest of the
    // epilogue (bias, beta·C, the residual, the activation) only runs on the
    // single-pass write-back, so a call that asks for any of it stays whole.
    const bool plain_epi = call.bias == nullptr && call.beta == 0.0f &&
                           call.residual == nullptr && call.epilogue == 0;
    if (!plain_epi) {
        return 1;
    }
    constexpr int64_t kMaxPlaneBytes = int64_t{64} << 20;
    // The bench knob forces or disables the cut: 0 means one pass, N > 1
    // means N splits (still capped below).
    const int forced = ::nnops::detail::cuda_splitk.load(std::memory_order_relaxed);
    if (forced == 0) {
        return 1;
    }
    const int64_t blocks = ((call.m + tile.bm - 1) / tile.bm) * ((call.n + tile.bn - 1) / tile.bn);
    const int64_t k_tiles = (call.k + tile.bk - 1) / tile.bk;
    const int sm = f32_sm_count();
    if (sm <= 0 || blocks <= 0 || k_tiles < 2) {
        return 1;
    }
    if (forced <= 1 && blocks * 4 >= int64_t(sm)) {
        return 1;
    }

    int64_t splits = forced > 1 ? int64_t(forced) :
                                  std::min<int64_t>((int64_t(sm) * 4 + blocks - 1) / blocks, 16);
    splits = std::min<int64_t>(splits, k_tiles / 2);
    const int64_t plane = int64_t(call.m) * call.n * int64_t(sizeof(float));
    const int64_t l2 = f32_l2_bytes();
    // A forced count is the bench's question, not the heuristic's, so the cache
    // cap does not second-guess it.
    if (l2 > 0 && forced <= 1) {
        splits = std::min<int64_t>(splits, std::max<int64_t>(l2 * 9 / 8 / plane, 1));
    }
    while (splits > 1 && splits * plane > kMaxPlaneBytes) {
        splits /= 2;
    }
    return int(std::max<int64_t>(splits, 1));
}

template <class S> bool f32_prepare()
{
    static const bool ok = [] {
        // Each pose asks for what its own panels need; the four differ.
        const auto set = [](const void* fn, int bytes) {
            return cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) ==
                   cudaSuccess;
        };
        if (!set(reinterpret_cast<const void*>(&f32_gemm_kernel<S, false, true>),
                 S::panel_bytes(false, true)) ||
            !set(reinterpret_cast<const void*>(&f32_gemm_kernel<S, false, false>),
                 S::panel_bytes(false, false)) ||
            !set(reinterpret_cast<const void*>(&f32_gemm_kernel<S, true, true>),
                 S::panel_bytes(true, true)) ||
            !set(reinterpret_cast<const void*>(&f32_gemm_kernel<S, true, false>),
                 S::panel_bytes(true, false))) {
            cudaGetLastError();
            return false;
        }
        return true;
    }();
    return ok;
}

template <class S> bool f32_run(const F32Call& call, cudaStream_t stream)
{
    if (!f32_prepare<S>()) {
        return false;
    }

    F32Params q;
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
    // The epilogue carries the same alpha; the kernel applies it once, in the
    // chain, so the two stay in step by construction.
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
    const TileDesc tile{S::BM, S::BN, S::BK, S::threads};
    int splits = f32_splits(call, tile);
    float* partial = nullptr;
    if (splits > 1) {
        partial = static_cast<float*>(f32_alloc(
            int64_t(splits) * q.batch * call.m * call.n * int64_t(sizeof(float)), stream));
        if (partial == nullptr) {
            splits = 1;
        }
    }

    const auto launch = [&](const dim3& g) {
        if (call.trans_a && call.trans_b) {
            f32_gemm_kernel<S, true, true>
                <<<g, S::threads, S::panel_bytes(true, true), stream>>>(q);
        } else if (call.trans_a) {
            f32_gemm_kernel<S, true, false>
                <<<g, S::threads, S::panel_bytes(true, false), stream>>>(q);
        } else if (call.trans_b) {
            f32_gemm_kernel<S, false, true>
                <<<g, S::threads, S::panel_bytes(false, true), stream>>>(q);
        } else {
            f32_gemm_kernel<S, false, false>
                <<<g, S::threads, S::panel_bytes(false, false), stream>>>(q);
        }
    };
    if (splits > 1) {
        q.partial = partial;
        launch(dim3(grid.x, grid.y, unsigned(splits * q.batch)));
        constexpr int kThreads = 256;
        const int64_t plane = int64_t(call.m) * call.n;
        const dim3 rgrid(unsigned((plane + kThreads - 1) / kThreads), unsigned(splits), 1);
        f32_gemm_reduce_kernel<<<rgrid, kThreads, 0, stream>>>(q);
        f32_free(partial, stream);
        return cudaGetLastError() == cudaSuccess;
    }

    launch(dim3(grid.x, grid.y, unsigned(q.batch)));
    return cudaGetLastError() == cudaSuccess;
}

/// The grid gate: below a full wave the per-block fill and the epilogue
/// dominate the tile, so the narrow-m loops are the better engine there and
/// the ladder hands them the call first.
bool f32_grid_fills(const F32Call& call, const TileDesc& tile)
{
    const int sm = f32_sm_count();
    if (sm <= 0) {
        return true;
    }
    const int64_t blocks =
        ((call.m + tile.bm - 1) / tile.bm) * ((call.n + tile.bn - 1) / tile.bn) * call.batch;
    return blocks >= int64_t(sm);
}

} // namespace

bool f32_dense_call(const F32Call& call_in, cudaStream_t stream, bool force)
{
    if (call_in.m <= 0 || call_in.n <= 0 || call_in.k <= 0) {
        return false;
    }
    // cp.async needs both row pitches and the base pointers 16 B aligned. A view
    // that starts or steps off that bound keeps the call and takes the fill's
    // scalar path -- the same code the tile's edge blocks run -- so the pose
    // stays covered rather than falling through to another engine.
    F32Call call = call_in;
    call.vec = ((reinterpret_cast<uintptr_t>(call.a) % 16) == 0 &&
                (reinterpret_cast<uintptr_t>(call.b) % 16) == 0 &&
                (reinterpret_cast<uintptr_t>(call.c) % 16) == 0 && (call.lda % 4) == 0 &&
                (call.ldb % 4) == 0)
                   ? 1
                   : 0;
    const bool named = call.tile >= 0 && call.tile < kTileCount;
    const bool forced = named || force;
    const int tile = named ? call.tile : kDefaultTile;
    if (!forced && !f32_grid_fills(call, kTileTable[tile])) {
        return false;
    }

    // The wide CTA is a documented loser here: 128x256 with 512 threads puts the
    // block at one per SM and costs 20% at 4096^3 (18.84 against 15.52 ms), so
    // both poses take the 128x128 tile.
    if (tile == kDefaultTile) {
        // Both 128x128 tiles keep the raster strip; what the block count picks
        // is the warp count. Eight warps share one A panel and win the small and
        // the large grids (1024^3 6.72 against 6.24 TFLOPS, 4096^3 8.85 against
        // 8.57); four warps pack better in between (2048^3 9.29 against 8.02).
        const int64_t blocks = ((call.m + Tile0::BM - 1) / Tile0::BM) *
                               ((call.n + Tile0::BN - 1) / Tile0::BN) * call.batch;
        // Three regimes, each measured in one warm state: under a wave (fewer
        // than 32 blocks) the small four-warp tile wins on footprint alone
        // (512^2: 0.062 against 0.084 ms); then the pose picks between the
        // three-block 64x128 tile and the four-warp 128x128 one. Deeper cp.async
        // rings (three stages on 128x128, four on 64x128) both lost -- the ring
        // is not what limits this kernel now.
        // A short reduction never amortises a 128-wide tile's fill and
        // epilogue, and the 64x64 four-block tile is the row that does: it wins
        // from 128^3 up through 1024^3 in both poses, where the last is a tie
        // (256^3 1.66 against 0.80 TFLOPS, 512^3 5.39 against 3.95, a short-K
        // 50176x256 product 7.95 against 6.54), so it takes the call before the
        // block-count regimes.
        if (call.k <= 1024) {
            return f32_run<Tile9>(call, stream);
        }
        if (blocks < 32) {
            return f32_run<Tile2>(call, stream);
        }
        // ncu on cuBLAS's 1024^3 launch is what settled this: it runs 256 blocks
        // of 128 threads and reaches 68% of the SM pipe, where our 128x128 grid
        // is 64 blocks on 1.68 waves and sits at 52%. The 64x128 tile doubles the
        // block count for the same bytes per block, which is the same wave fill
        // at half the tile (1024^3: 7.41 against 6.45 TFLOPS).
        // Below a full wave of the 128x128 tile (114 slots) the 64x64 tile is
        // the one that fills the machine: four blocks a SM, so 1024^3's 256
        // blocks are 1.68 waves against 1.12 (7.78 against 7.16 TFLOPS for the
        // n-major pose, 6.36 against 6.11 for the k-major one).
        if (blocks < 114) {
            return f32_run<Tile9>(call, stream);
        }
        // Past that the four-warp 128x64 tile leads both poses. Matching the
        // 64x128 orientation for the n-major pose was tried and measured 2%
        // worse (1.19 against 1.17 of cuBLAS), so the axis order is not what
        // separates the two kernels -- ncu puts ours 29% heavier in instructions
        // at equal occupancy, which is code generation inside the k loop.
        // Forcing a fourth block a SM -- launch bounds to 128 registers plus
        // dropping the panel skew -- was measured and lost badly (6.9 vs 8.85).
        return f32_run<Tile1>(call, stream);
    }
    switch (tile) {
    case 0:
        return f32_run<Tile0>(call, stream);
    case 1:
        return f32_run<Tile1>(call, stream);
    case 2:
        return f32_run<Tile2>(call, stream);
    case 3:
        return f32_run<Tile3>(call, stream);
    case 4:
        return f32_run<Tile4>(call, stream);
    case 5:
        return f32_run<Tile6>(call, stream);
    case 7:
        return f32_run<Tile7>(call, stream);
    case 8:
        return f32_run<Tile8>(call, stream);
    default:
        return f32_run<Tile9>(call, stream);
    }
}

} // namespace nnops::backend::cuda
