/// @file gemm_splitk.cuh
/// @brief The k-split path: cutting the reduction when the grid has no other
///        axis left to fill, and the scratch plane it lands in.
///
/// A grid shallower than the tile wants — the decode-shaped products, where M
/// fits one tile and N is a few dozen — leaves most of the machine idle,
/// because there is no third axis left to cut. The launcher then cuts the
/// reduction instead: blockIdx.z carries batch·k_splits + split, each block owns
/// a proportional range of the k tiles, and it writes its f32 accumulator tile
/// to a scratch plane instead of running the epilogue. A second kernel sums the
/// planes and applies the epilogue once, so alpha, bias, beta·C, the clamp, the
/// activation and the residual keep their single-pass order. Only the narrow
/// tiles split — the wide-tile gates already require a grid that fills the
/// machine — and the scratch is stream-ordered from a private, retained memory
/// pool (the default pool releases on every free, which measured at 1.45 ms a
/// pair on WDDM). NNOPS_CUDA_SPLITK forces or disables it for a bench; unset
/// means the heuristic.

#pragma once

#include "gemm_tile_h.cuh"
#include "gemm_tile_f.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace nnops::backend::cuda {

// ============================================================
// split-K reduction
// ============================================================

/// Sum the k-split planes and run the epilogue once, exactly as the single-pass
/// kernels would have. One thread per output element, planes summed in a fixed
/// order, so a given split count is deterministic. The block tile is not
/// involved: this pass is over the whole [M, N] plane.
template <class Elem>
__global__ void dense_gemm_splitk_reduce_kernel(GemmParams p)
{
    const int64_t plane = static_cast<int64_t>(p.m) * p.n;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= plane) { return; }

    const int m = int(idx / p.n);
    const int n = int(idx - static_cast<int64_t>(m) * p.n);
    const int batch = int(blockIdx.y);

    const float* src = p.partial +
                       static_cast<int64_t>(batch) * p.k_splits * plane + idx;
    float sum = 0.0f;
    for (int s = 0; s < p.k_splits; ++s) {
        sum += src[static_cast<int64_t>(s) * plane];
    }

    Elem* C = static_cast<Elem*>(p.c) +
              detail::gemm_batch_offset(p, p.c_batch, batch);
    const int64_t off = static_cast<int64_t>(m) * p.ldc + n;

    float v = p.alpha * sum;
    if (p.bias != nullptr) {
        const Elem* bias = static_cast<const Elem*>(p.bias);
        v += gemm_to_float(bias[p.bias_row ? m : n]);
    }
    if (p.beta != 0.0f) { v += p.beta * gemm_to_float(C[off]); }
    v = fminf(fmaxf(v, p.min_clip), p.max_clip);
    v = gemm_apply_epilogue(p.epilogue, p.ep_alpha, p.ep_beta, v);
    if (p.residual != nullptr) {
        const Elem* residual = static_cast<const Elem*>(p.residual) +
                               detail::gemm_batch_offset(p, p.r_batch, batch);
        v += gemm_to_float(residual[static_cast<int64_t>(m) * p.ldr + n]);
    }
    C[off] = Elem(v);
}

namespace detail {

// ---- split-K --------------------------------------------------------------

/// Bench knob: NNOPS_CUDA_SPLITK=0 disables split-K, N > 1 forces N splits
/// (clamped like the auto choice); unset keeps the heuristic.
inline int splitk_env()
{
    static const int v = [] {
        const char* s = std::getenv("NNOPS_CUDA_SPLITK");
        return s != nullptr ? std::atoi(s) : -1;
    }();
    return v;
}

/// A split needs enough k tiles for the pipeline prologue to disappear into
/// work, and every split writes one f32 plane of partial stores.
constexpr int kSplitKMinTiles = 4;
constexpr int kSplitKMaxSplits = 64;
constexpr int64_t kSplitKMaxBytes = int64_t{64} << 20;

/// How many k ranges the product should be cut into. `capacity` is the resident
/// block count the tile is built for (`blocks` per SM, three for the 32 KB f16
/// tile) and `elem_size` the operand width.
///
/// The split count is the largest that still fits the machine in one wave:
/// rounding up instead (ceil) was measured at -25% on 512³ and -57% on the f16
/// twin, because the extra blocks become a second, nearly empty wave and each
/// split is also priced in prologue k tiles plus one plane of partial stores.
///
/// One wave is necessary but not sufficient. Splitting multiplies the output
/// stream — every split writes an f32 plane and the reduce reads it back — so
/// against a short K the planes outweigh the product itself: 512³ still pays
/// +19% at two splits even though 64 blocks fill only 42% of a wave. So the cut
/// is taken only when one of two measured conditions holds:
///  - the grid is starved, under an eighth of a wave (128³ and 256³, which gain
///    +15% and +19% despite plane traffic larger than the operands), or
///  - the planes are cheap against the operands, under a quarter of them
///    (16x4096x4096: 1 MB of planes against 67 MB of operands, +11%).
inline int pick_split_k(const GemmParams& p, int bM, int bN, int bK,
                        int capacity, int elem_size)
{
    const int forced = splitk_env();
    if (forced == 0) { return 1; }
    const int k_tiles = (p.k + bK - 1) / bK;
    if (forced > 1) {
        return std::max(1, std::min({forced, k_tiles, kSplitKMaxSplits}));
    }
    if (capacity <= 0) { return 1; }

    const int64_t batch = gemm_batch_count(p);
    const int64_t tiles =
        int64_t((p.m + bM - 1) / bM) * ((p.n + bN - 1) / bN) * batch;

    int64_t splits = std::min<int64_t>(kSplitKMaxSplits, capacity / tiles);
    splits = std::min<int64_t>(splits, k_tiles / kSplitKMinTiles);
    if (splits < 2) { return 1; }

    const int64_t plane = int64_t(p.m) * p.n * batch;
    const int64_t partial_bytes = int64_t{8} * splits * plane;  // f32 write + read
    const int64_t operand_bytes = int64_t(elem_size) * p.k * (int64_t(p.m) + p.n) * batch;
    const bool starved = tiles * 8 <= capacity;
    const bool cheap = partial_bytes * 4 <= operand_bytes;
    return (starved || cheap) ? int(splits) : 1;
}

/// The reduce pass's scratch. cudaMallocAsync keeps the whole call stream
/// ordered; the blocking cudaMalloc is the fallback when the async allocator is
/// unavailable, and the flag says which one owns the pointer.
struct GemmWorkspace {
    void* ptr = nullptr;
    bool async = false;
};

/// A private memory pool for the scratch, created once per process. The default
/// pool's release threshold is 0, so on WDDM every free hands the pages back to
/// the OS and the next alloc costs milliseconds — the pool measured at 1.45 ms
/// per alloc/free pair with kernels interleaved, against 2 µs here. Keeping
/// up to 128 MiB resident bounds the pool without touching the process-wide
/// default pool.
inline cudaMemPool_t gemm_workspace_pool()
{
    static const cudaMemPool_t pool = [] {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) { return cudaMemPool_t{}; }
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

inline GemmWorkspace gemm_workspace_alloc(int64_t bytes, cudaStream_t stream)
{
    GemmWorkspace ws;
    if (bytes <= 0) { return ws; }
#if CUDART_VERSION >= 11070
    const cudaMemPool_t pool = gemm_workspace_pool();
    if (pool != nullptr &&
        cudaMallocFromPoolAsync(&ws.ptr, static_cast<size_t>(bytes), pool,
                                stream) == cudaSuccess) {
        ws.async = true;
        return ws;
    }
    cudaGetLastError();
#endif
    if (cudaMalloc(&ws.ptr, static_cast<size_t>(bytes)) != cudaSuccess) {
        ws.ptr = nullptr;
        cudaGetLastError();
    }
    return ws;
}

inline void gemm_workspace_free(GemmWorkspace ws, cudaStream_t stream)
{
    if (ws.ptr == nullptr) { return; }
#if CUDART_VERSION >= 11020
    if (ws.async && cudaFreeAsync(ws.ptr, stream) == cudaSuccess) { return; }
    cudaGetLastError();
#endif
    cudaFree(ws.ptr);
}

template <class Elem>
inline void launch_splitk_reduce(const GemmParams& p, cudaStream_t stream)
{
    constexpr int kThreads = 256;
    const int64_t plane = static_cast<int64_t>(p.m) * p.n;
    dim3 grid(static_cast<unsigned>((plane + kThreads - 1) / kThreads),
              static_cast<unsigned>(gemm_batch_count(p)), 1);
    dense_gemm_splitk_reduce_kernel<Elem><<<grid, kThreads, 0, stream>>>(p);
}

/// Launch a tile, cutting the reduction into k ranges when the grid is shallower
/// than the tile wants, and reducing the planes afterwards. Falls back to the
/// single-pass launch whenever a split cannot help or its scratch cannot be
/// carved.
template <class Elem, class Tile, bool TransA, bool TransB, bool Vectorized,
          int Groups = 1>
inline void launch_dense_split(const GemmParams& p, cudaStream_t stream)
{
    constexpr bool Half = !std::is_same_v<Elem, float>;
    auto launch_one = [&](const GemmParams& q) {
        if constexpr (Half) {
            launch_dense_h<Tile, Elem, TransA, TransB, Vectorized, Groups>(q, stream);
        } else {
            launch_dense_f<Tile, TransA, TransB, Vectorized>(q, stream);
        }
    };

    // A grouped CTA is wider and takes more smem, so it holds fewer residents:
    // two 39 KB blocks against three 27 KB ones.
    constexpr int blocks_per_sm = [] {
        if constexpr (Half) { return Groups == 1 ? 3 : 2; }
        else { return Tile::blocks; }
    }();
    const int elem_size = Half ? 2 : 4;
    int splits = pick_split_k(p, Tile::bM, Tile::bN * Groups, Tile::bK,
                              device_sm_count() * blocks_per_sm, elem_size);
    if (splits <= 1) { launch_one(p); return; }

    const int64_t batch = gemm_batch_count(p);
    const int64_t plane = static_cast<int64_t>(p.m) * p.n;
    // gridDim.z is 16-bit, and the partial planes are f32 whatever the operands.
    while (splits > 1 && batch * splits > 65535) { splits /= 2; }
    int64_t bytes = int64_t(splits) * batch * plane * int64_t(sizeof(float));
    while (splits > 1 && bytes > kSplitKMaxBytes) {
        splits /= 2;
        bytes = int64_t(splits) * batch * plane * int64_t(sizeof(float));
    }
    if (splits <= 1) { launch_one(p); return; }

    const GemmWorkspace ws = gemm_workspace_alloc(bytes, stream);
    if (ws.ptr == nullptr) { launch_one(p); return; }

    GemmParams q = p;
    q.k_splits = splits;
    q.partial = static_cast<float*>(ws.ptr);
    launch_one(q);
    if constexpr (Half) {
        launch_splitk_reduce<Elem>(q, stream);
    } else {
        launch_splitk_reduce<float>(q, stream);
    }
    gemm_workspace_free(ws, stream);
}

}  // namespace detail

}  // namespace nnops::backend::cuda
