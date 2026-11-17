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

namespace {

/// The candidate tiles. Three is the whole menu: the sweep retired the rest
/// (a 128x256 CTA, a 64x64 CTA for shallow grids, a deeper ring), and every
/// tile here costs twelve kernels once the four poses are instantiated.
using Tile0 = ManualFShape<8, 8, 2, 4, 16, 2, 4, 8, 4, 2>; ///< 128x128, 8 warps
/// The 128-thread tiles declare three blocks of launch bounds: without it the
/// A-transposed instantiation lands at 177 registers and the driver gives it
/// two blocks a SM where the row-major one gets three (ncu: 16.4% against 23.7%
/// occupancy, and the whole 15% between the poses).
using Tile1 = ManualFShape<8, 8, 2, 2, 16, 2, 4, 8, 4, 3>; ///< 128x128, 4 warps
using Tile2 = ManualFShape<8, 8, 2, 2, 16, 2, 4, 0, 4, 3>; ///< the same, no raster
/// 64x128 on four warps: 31 KB of panels, so three blocks stay resident where
/// the 128x128 tile keeps two. The register block is unchanged; the cost is that
/// every CTA re-reads the A rows its neighbour also reads.
using Tile3 = ManualFShape<8, 8, 1, 4, 16, 2, 4, 8, 4, 3>;
/// Deeper rings: with one barrier a tile the ring depth is what stands in for
/// the fragment shadow a software-pipelined kernel would keep, and it costs
/// panels rather than registers.
using Tile4 = ManualFShape<8, 8, 2, 4, 16, 3>; ///< 128x128, three stages
/// 128x64 on eight warps with a 8x4 lane tile: 32 accumulators keep the block
/// at 115 registers, so two of them are resident and a 128-block grid is 1.68
/// waves instead of 1.12 -- the shape ncu showed cuBLAS running.
using Tile6 = ManualFShape<8, 4, 2, 4, 16, 2>;
/// The four-blocks-a-SM pair ncu pointed at: 128 threads, no panel skew (21-25 KB
/// of panels), and launch bounds that cap the block at 128 registers so the
/// fourth block fits. 64x64 doubles the block count again, which is what a
/// 1024^3 grid needs to become 1.68 waves instead of 1.12.
using Tile7 = ManualFShape<8, 4, 1, 4, 16, 2, 4, 8, 0, 4>; ///< 64x64, 128 thr
using Tile8 = ManualFShape<8, 8, 2, 2, 16, 2, 4, 8, 0, 4>; ///< 128x64, 128 thr
/// The 1024^3 fix: a 64x64 CTA on 128 threads with an 8x4 lane tile. Its 20.5 KB
/// of panels and ~110 registers let four blocks live on an SM (152 slots), so
/// the 256 blocks a 1024^3 grid makes are 1.68 waves instead of 1.12 -- the
/// wave fill cuBLAS's own 4-block tile gets, without the register cap that
/// killed the earlier attempt.
using Tile9 = ManualFShape<8, 4, 1, 4, 16, 2, 4, 8, 4, 4>;

struct TileDesc {
    int bm, bn, bk, threads, smem_bytes;
};

constexpr int kTileCount = 10;
constexpr TileDesc kTileTable[kTileCount] = {
    {Tile0::BM, Tile0::BN, Tile0::BK, Tile0::threads, Tile0::kSmemBytes},
    {Tile1::BM, Tile1::BN, Tile1::BK, Tile1::threads, Tile1::kSmemBytes},
    {Tile2::BM, Tile2::BN, Tile2::BK, Tile2::threads, Tile2::kSmemBytes},
    {Tile3::BM, Tile3::BN, Tile3::BK, Tile3::threads, Tile3::kSmemBytes},
    {Tile4::BM, Tile4::BN, Tile4::BK, Tile4::threads, Tile4::kSmemBytes},
    {Tile6::BM, Tile6::BN, Tile6::BK, Tile6::threads, Tile6::kSmemBytes},
    {Tile7::BM, Tile7::BN, Tile7::BK, Tile7::threads, Tile7::kSmemBytes},
    {Tile8::BM, Tile8::BN, Tile8::BK, Tile8::threads, Tile8::kSmemBytes},
    {Tile9::BM, Tile9::BN, Tile9::BK, Tile9::threads, Tile9::kSmemBytes},
};

constexpr int kDefaultTile = 0;
/// Past this many blocks the tile grid's B panels no longer stay resident and
/// the strip order starts winning; below it the plain order reuses them better.
constexpr int64_t kRasterBlocks = 512;

int manual_f_sm_count()
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

int64_t manual_f_l2_bytes()
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

cudaMemPool_t manual_f_pool()
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

void* manual_f_alloc(int64_t bytes, cudaStream_t stream)
{
    if (bytes <= 0) {
        return nullptr;
    }
    void* ptr = nullptr;
    const cudaMemPool_t pool = manual_f_pool();
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

void manual_f_free(void* ptr, cudaStream_t stream)
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
int manual_f_splits(const ManualFCall& call, const TileDesc& tile)
{
    constexpr int64_t kMaxPlaneBytes = int64_t{64} << 20;
    // The bench knob forces or disables the cut, on the same terms as the CuTe
    // engine's: 0 means one pass, N > 1 means N splits (still capped below).
    const int forced = ::nnops::detail::cuda_splitk.load(std::memory_order_relaxed);
    if (forced == 0) {
        return 1;
    }
    const int64_t blocks = ((call.m + tile.bm - 1) / tile.bm) * ((call.n + tile.bn - 1) / tile.bn);
    const int64_t k_tiles = (call.k + tile.bk - 1) / tile.bk;
    const int sm = manual_f_sm_count();
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
    const int64_t l2 = manual_f_l2_bytes();
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

template <class S> bool manual_f_prepare()
{
    static const bool ok = [] {
        const int bytes = S::kSmemBytes;
        const auto set = [bytes](const void* fn) {
            return cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) ==
                   cudaSuccess;
        };
        if (!set(reinterpret_cast<const void*>(&manual_gemm_f_kernel<S, false, true>)) ||
            !set(reinterpret_cast<const void*>(&manual_gemm_f_kernel<S, false, false>)) ||
            !set(reinterpret_cast<const void*>(&manual_gemm_f_kernel<S, true, true>)) ||
            !set(reinterpret_cast<const void*>(&manual_gemm_f_kernel<S, true, false>))) {
            cudaGetLastError();
            return false;
        }
        return true;
    }();
    return ok;
}

template <class S> bool manual_f_run(const ManualFCall& call, cudaStream_t stream)
{
    if (!manual_f_prepare<S>()) {
        return false;
    }

    ManualFParams q;
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
    for (int d = 0; d < 4; ++d) {
        q.bsize[d] = call.bsize[d];
        q.a_bstr[d] = call.a_bstr[d];
        q.b_bstr[d] = call.b_bstr[d];
        q.c_bstr[d] = call.c_bstr[d];
    }

    const dim3 grid(unsigned((call.m + S::BM - 1) / S::BM), unsigned((call.n + S::BN - 1) / S::BN),
                    1);
    const TileDesc tile{S::BM, S::BN, S::BK, S::threads, S::kSmemBytes};
    int splits = manual_f_splits(call, tile);
    float* partial = nullptr;
    if (splits > 1) {
        partial = static_cast<float*>(manual_f_alloc(
            int64_t(splits) * q.batch * call.m * call.n * int64_t(sizeof(float)), stream));
        if (partial == nullptr) {
            splits = 1;
        }
    }

    const auto launch = [&](const dim3& g) {
        const unsigned blocks = S::kSmemBytes;
        if (call.trans_a && call.trans_b) {
            manual_gemm_f_kernel<S, true, true><<<g, S::threads, blocks, stream>>>(q);
        } else if (call.trans_a) {
            manual_gemm_f_kernel<S, true, false><<<g, S::threads, blocks, stream>>>(q);
        } else if (call.trans_b) {
            manual_gemm_f_kernel<S, false, true><<<g, S::threads, blocks, stream>>>(q);
        } else {
            manual_gemm_f_kernel<S, false, false><<<g, S::threads, blocks, stream>>>(q);
        }
    };
    if (splits > 1) {
        q.partial = partial;
        launch(dim3(grid.x, grid.y, unsigned(splits * q.batch)));
        constexpr int kThreads = 256;
        const int64_t plane = int64_t(call.m) * call.n;
        const dim3 rgrid(unsigned((plane + kThreads - 1) / kThreads), unsigned(splits), 1);
        manual_gemm_f_reduce_kernel<<<rgrid, kThreads, 0, stream>>>(q);
        manual_f_free(partial, stream);
        return cudaGetLastError() == cudaSuccess;
    }

    launch(dim3(grid.x, grid.y, unsigned(q.batch)));
    return cudaGetLastError() == cudaSuccess;
}

/// The grid gate: below a full wave the per-block fill and epilogue dominate and
/// the CuTe engine's narrow-m arm is still ahead, so the caller keeps it.
bool manual_f_grid_fills(const ManualFCall& call, const TileDesc& tile)
{
    const int sm = manual_f_sm_count();
    if (sm <= 0) {
        return true;
    }
    const int64_t blocks =
        ((call.m + tile.bm - 1) / tile.bm) * ((call.n + tile.bn - 1) / tile.bn) * call.batch;
    return blocks >= int64_t(sm);
}

} // namespace

bool manual_dense_f_call(const ManualFCall& call, cudaStream_t stream)
{
    if (call.m <= 0 || call.n <= 0 || call.k <= 0) {
        return false;
    }
    if ((reinterpret_cast<uintptr_t>(call.a) % 16) != 0 ||
        (reinterpret_cast<uintptr_t>(call.b) % 16) != 0 ||
        (reinterpret_cast<uintptr_t>(call.c) % 16) != 0 || (call.lda % 4) != 0 ||
        (call.ldb % 4) != 0) {
        return false;
    }

    // The fill works in 32-bit offsets; a product past an int32 keeps the call
    // on the CuTe engine.
    constexpr int64_t kMaxOffset = 2147483647LL;
    if (int64_t(call.m) * call.k > kMaxOffset || int64_t(call.n) * call.k > kMaxOffset) {
        return false;
    }
    const bool named = call.tile >= 0 && call.tile < kTileCount;
    const bool forced = named || ::nnops::detail::cuda_ptx_f.load(std::memory_order_relaxed) == 1;
    const int tile = named ? call.tile : kDefaultTile;
    if (!forced && !manual_f_grid_fills(call, kTileTable[tile])) {
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
        if (blocks < 32) {
            return manual_f_run<Tile2>(call, stream);
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
            return manual_f_run<Tile9>(call, stream);
        }
        // Past that the four-warp 128x64 tile leads both poses. Matching CuTe's
        // 64x128 orientation for the n-major pose was tried and measured 2%
        // worse (1.19 against 1.17 of cuBLAS), so the axis order is not what
        // separates the two kernels -- ncu puts ours 29% heavier in instructions
        // at equal occupancy, which is code generation inside the k loop.
        // Forcing a fourth block a SM -- launch bounds to 128 registers plus
        // dropping the panel skew -- was measured and lost badly (6.9 vs 8.85).
        return manual_f_run<Tile1>(call, stream);
    }
    switch (tile) {
    case 0:
        return manual_f_run<Tile0>(call, stream);
    case 1:
        return manual_f_run<Tile1>(call, stream);
    case 2:
        return manual_f_run<Tile2>(call, stream);
    case 3:
        return manual_f_run<Tile3>(call, stream);
    case 4:
        return manual_f_run<Tile4>(call, stream);
    case 5:
        return manual_f_run<Tile6>(call, stream);
    case 7:
        return manual_f_run<Tile7>(call, stream);
    case 8:
        return manual_f_run<Tile8>(call, stream);
    default:
        return manual_f_run<Tile9>(call, stream);
    }
}

} // namespace nnops::backend::cuda
