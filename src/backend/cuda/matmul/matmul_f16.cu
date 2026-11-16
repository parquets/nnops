/// @file matmul_f16.cu
/// @brief The host-callable front of the hand-written PTX f16 GEMM: it owns the
///        tile menu, resolves the split-K heuristic, rents the reduction planes,
///        and refuses the poses the kernel does not cover.
///
/// The tile is a template parameter, so the menu is a switch over a handful of
/// instantiations: `ManualHCall::tile` picks one explicitly (a bench sweep) and
/// -1 takes the default row of the table.
///
/// Split-K follows the fp engine's rule: the tile grid alone leaves a shallow
/// product a fraction of the machine and there is no third axis left to cut, so
/// the reduction is cut instead -- gridDim.z carries the split count, each block
/// writes its f32 accumulator plane, and one reduce pass sums them and applies
/// alpha once.
///
/// See matmul_f16.cuh for the kernel and the fragment mapping.

#include "matmul_f16_host.hpp"
#include <nnops/core/epilogue.hpp>
#include "matmul_f16.cuh"
#include "nnops/detail/bench_knobs.hpp"

#include <algorithm>

namespace nnops::backend::cuda {

namespace {

/// The candidate tiles, in the order a sweep reports them. The 64x64 warp tile
/// on a 2x2 grid is the default: doubling the warp's n span halves the ldmatrix
/// traffic per MAC, and on two CTAs of 128 threads that trade is a clear win at
/// every size past a few hundred square (4096^3: 4.39 ms against 5.38 for the
/// 64x32 tile and 7.07 for the CuTe engine it replaced). The other two are the
/// edges the sweep leaves standing: the 16-warp CTA and the narrow 64-row CTA.
using Tile0 = ManualHShape<64, 64, 2, 2, 32, 3>;
using Tile1 = ManualHShape<64, 32, 2, 4, 32, 3>;
using Tile2 = ManualHShape<64, 32, 1, 4, 32, 3>;
/// The winning tile under the search's other two switches: a skewed panel
/// instead of the XOR swizzle, and an unstaged epilogue.
using Tile3 = ManualHShape<64, 64, 2, 2, 32, 3, 0, true>;
using Tile4 = ManualHShape<64, 64, 2, 2, 32, 3, 1, false>;
/// The shallow-grid tile, the same fix f32 needed: a 64x64 CTA of 128 threads
/// with a 32x32 warp tile, so the accumulators are 32 floats and four blocks
/// live on an SM (24.6 KB of panels each). A 2048^3 square then has 1024 blocks
/// instead of 256, which is what fills the machine there.
using Tile5 = ManualHShape<32, 32, 2, 2, 32, 3, 1, true, 4>;

struct TileDesc {
    int bm, bn, bk, threads, smem_bytes;
};

constexpr int kTileCount = 6;
constexpr TileDesc kTileTable[kTileCount] = {
    {Tile0::BM, Tile0::BN, Tile0::BK, Tile0::threads, Tile0::kSmemBytes},
    {Tile1::BM, Tile1::BN, Tile1::BK, Tile1::threads, Tile1::kSmemBytes},
    {Tile2::BM, Tile2::BN, Tile2::BK, Tile2::threads, Tile2::kSmemBytes},
    {Tile3::BM, Tile3::BN, Tile3::BK, Tile3::threads, Tile3::kSmemBytes},
    {Tile4::BM, Tile4::BN, Tile4::BK, Tile4::threads, Tile4::kSmemBytes},
    {Tile5::BM, Tile5::BN, Tile5::BK, Tile5::threads, Tile5::kSmemBytes},
};

constexpr int kDefaultTile = 0;

/// Queried once per process; 0 when unavailable.
int manual_sm_count()
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

int64_t manual_l2_bytes()
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
cudaMemPool_t manual_pool()
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

void* manual_alloc(int64_t bytes, cudaStream_t stream)
{
    if (bytes <= 0) {
        return nullptr;
    }
    void* ptr = nullptr;
    const cudaMemPool_t pool = manual_pool();
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

void manual_free(void* ptr, cudaStream_t stream)
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

/// How many k ranges to cut. The block grid has to fill the machine (a quarter of
/// a wave is the floor the fp engine uses) and the planes have to stay near the
/// cache; past that the extra blocks cost more than they buy.
int manual_splits(const ManualHCall& call, const TileDesc& tile)
{
    constexpr int64_t kMaxPlaneBytes = int64_t{64} << 20;
    const int64_t m_tiles = (call.m + tile.bm - 1) / tile.bm;
    const int64_t n_tiles = (call.n + tile.bn - 1) / tile.bn;
    const int64_t blocks = m_tiles * n_tiles;
    const int64_t k_tiles = (call.k + tile.bk - 1) / tile.bk;
    const int sm = manual_sm_count();
    if (sm <= 0 || blocks <= 0 || k_tiles < 2) {
        return 1;
    }
    if (blocks * 4 >= int64_t(sm)) {
        return 1;
    }

    int64_t splits = std::min<int64_t>((int64_t(sm) * 4 + blocks - 1) / blocks, 16);
    splits = std::min<int64_t>(splits, k_tiles / 2);
    const int64_t plane = int64_t(call.m) * call.n * int64_t(sizeof(float));
    const int64_t l2 = manual_l2_bytes();
    if (l2 > 0) {
        splits = std::min<int64_t>(splits, std::max<int64_t>(l2 * 9 / 8 / plane, 1));
    }
    while (splits > 1 && splits * plane > kMaxPlaneBytes) {
        splits /= 2;
    }
    return int(std::max<int64_t>(splits, 1));
}

/// The grid gate. Below a full wave this kernel's per-block cost -- the panel
/// fill and the staged epilogue -- is what dominates, and the CuTe family's
/// narrow-m and split-K arms are still ahead (256^2: 0.0205 ms against 0.0151).
/// Once the tile grid fills the machine the trade reverses hard (4096^3: 5.03 ms
/// against 8.36), so that is the line. An explicit tile skips the gate: it is
/// how a sweep reaches every shape.
bool manual_grid_fills(const ManualHCall& call, const TileDesc& tile)
{
    const int sm = manual_sm_count();
    if (sm <= 0) {
        return true;
    }
    const int64_t blocks = ((call.m + tile.bm - 1) / tile.bm) * ((call.n + tile.bn - 1) / tile.bn);
    return blocks >= int64_t(sm);
}

/// A dynamic shared memory request above 48 KB needs the opt-in, once per
/// instantiation; a static `__shared__` array could not carry these tiles.
template <class S, class Elem> bool manual_prepare()
{
    static const bool ok = [] {
        const int bytes = S::kSmemBytes;
        const auto set = [bytes](const void* fn) {
            return cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) ==
                   cudaSuccess;
        };
        if (!set(reinterpret_cast<const void*>(&manual_gemm_h_kernel<S, Elem, true, true>)) ||
            !set(reinterpret_cast<const void*>(&manual_gemm_h_kernel<S, Elem, true, false>)) ||
            !set(reinterpret_cast<const void*>(&manual_gemm_h_kernel<S, Elem, false, true>)) ||
            !set(reinterpret_cast<const void*>(&manual_gemm_h_kernel<S, Elem, false, false>))) {
            cudaGetLastError();
            return false;
        }
        return true;
    }();
    return ok;
}

template <class S, class Elem> bool manual_run(const ManualHCall& call, cudaStream_t stream)
{
    if (!manual_prepare<S, Elem>()) {
        return false;
    }

    ManualHParams q;
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
    for (int d = 0; d < 4; ++d) {
        q.bsize[d] = call.bsize[d];
        q.a_bstr[d] = call.a_bstr[d];
        q.b_bstr[d] = call.b_bstr[d];
        q.c_bstr[d] = call.c_bstr[d];
    }

    const dim3 grid(unsigned((call.m + S::BM - 1) / S::BM), unsigned((call.n + S::BN - 1) / S::BN),
                    1);
    constexpr int kLaunchSmem = S::kSmemBytes;
    const TileDesc tile{S::BM, S::BN, S::BK, S::threads, S::kSmemBytes};
    int splits = manual_splits(call, tile);
    float* partial = nullptr;
    if (splits > 1) {
        partial = static_cast<float*>(manual_alloc(
            int64_t(splits) * q.batch * call.m * call.n * int64_t(sizeof(float)), stream));
        if (partial == nullptr) {
            splits = 1;
        }
    }

    const auto launch = [&](const dim3& g) {
        if (call.trans_a && call.trans_b) {
            manual_gemm_h_kernel<S, Elem, true, true><<<g, S::threads, kLaunchSmem, stream>>>(q);
        } else if (call.trans_a) {
            manual_gemm_h_kernel<S, Elem, true, false><<<g, S::threads, kLaunchSmem, stream>>>(q);
        } else if (call.trans_b) {
            manual_gemm_h_kernel<S, Elem, false, true><<<g, S::threads, kLaunchSmem, stream>>>(q);
        } else {
            manual_gemm_h_kernel<S, Elem, false, false><<<g, S::threads, kLaunchSmem, stream>>>(q);
        }
    };
    if (splits > 1) {
        const dim3 sgrid(grid.x, grid.y, unsigned(splits * q.batch));
        q.partial = partial;
        launch(sgrid);
        constexpr int kThreads = 256;
        const int64_t plane = int64_t(call.m) * call.n;
        const dim3 rgrid(unsigned((plane + kThreads - 1) / kThreads), unsigned(splits), 1);
        manual_gemm_h_reduce_kernel<Elem><<<rgrid, kThreads, 0, stream>>>(q);
        manual_free(partial, stream);
        return cudaGetLastError() == cudaSuccess;
    }

    launch(dim3(grid.x, grid.y, unsigned(q.batch)));
    return cudaGetLastError() == cudaSuccess;
}

/// The tile menu, per element type: one instantiation per (tile, pose).
template <class Elem>
bool manual_dispatch_tile(const ManualHCall& call, int tile, cudaStream_t stream)
{
    switch (tile) {
    case 0:
        return manual_run<Tile0, Elem>(call, stream);
    case 1:
        return manual_run<Tile1, Elem>(call, stream);
    case 2:
        return manual_run<Tile2, Elem>(call, stream);
    case 3:
        return manual_run<Tile3, Elem>(call, stream);
    case 4:
        return manual_run<Tile4, Elem>(call, stream);
    default:
        return manual_run<Tile5, Elem>(call, stream);
    }
}

} // namespace

bool manual_dense_h_call(const ManualHCall& call, cudaStream_t stream)
{
    // A transposed is supported: the panel keeps the `[k][m]` arrival order and
    // the fragment loads go through ldmatrix.trans.
    if (call.m <= 0 || call.n <= 0 || call.k <= 0) {
        return false;
    }
    if (call.alpha != 1.0f) {
        return false;
    }
    // cp.async needs both row pitches and the base pointers 16 B aligned.
    if ((reinterpret_cast<uintptr_t>(call.a) % 16) != 0 ||
        (reinterpret_cast<uintptr_t>(call.b) % 16) != 0 ||
        (reinterpret_cast<uintptr_t>(call.c) % 16) != 0 || (call.lda % 8) != 0 ||
        (call.ldb % 8) != 0) {
        return false;
    }

    // The knob is the other way to skip the gate: forcing the route is what a
    // bench does when it wants the PTX arm on every shape. It does not pick a
    // tile -- an unnamed tile still means the default row.
    const bool named = call.tile >= 0 && call.tile < kTileCount;
    const bool forced = named || ::nnops::detail::cuda_ptx_h.load(std::memory_order_relaxed) == 1;
    // The k-major pose wants the epilogue that writes straight out of the
    // accumulators: 4096^3 4.38 against 5.00 ms for the staged one. The n-major
    // pose is the other way round (3.96 against 4.05), so the pose picks.
    const int pick = (!call.trans_b) ? kDefaultTile : 4;
    const int tile = named ? call.tile : pick;
    if (!forced && !manual_grid_fills(call, kTileTable[tile])) {
        return false;
    }
    if (call.bf16) {
        return manual_dispatch_tile<__nv_bfloat16>(call, tile, stream);
    }
    return manual_dispatch_tile<__half>(call, tile, stream);
}

} // namespace nnops::backend::cuda
