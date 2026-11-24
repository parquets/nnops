/// @file gemm_common.cuh
/// @brief The shared half of the dense GEMM contract: the parameter struct,
///        the shared epilogue, the batch/block/split geometry, the device
///        queries, and the split-K scratch.
///
/// Everything but the tile algebra lives here, so a kernel that fills its own
/// panels -- the int8 engines, and the conv policies built on the hand-written
/// PTX mainloops -- can include it without the tile machinery.

#pragma once

#include <nnops/core/epilogue.hpp>
#include <detail/bench_knobs.hpp>

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>

namespace nnops::backend::cuda {

/// The batch axes a kernel carries: with the two matrix axes a CUDA MatMul or
/// Linear is rank six at most. `GemmParams::batch_size` is sized by this.
inline constexpr int kGemmMaxBatchRank = 4;

/// gridDim is 16-bit on every axis. Three counts have to stay under it: the
/// flattened batch on blockIdx.z (times the k splits), the same batch on
/// blockIdx.y in the split-K reduce pass, and the f32 tile's n axis, which walks
/// blockIdx.y. The front end refuses a call that would overflow any of them.
inline constexpr int64_t kGemmMaxGridAxis = 65535;

/// The dense GEMM's element formats: the f32 SIMT tile, or one of the two
/// 2-byte tensor-core formats (same MMA shape, different operand format).
enum class GemmElem {
    F32,
    F16,
    BF16
};

/// Everything a dense GEMM needs besides the element type and the two
/// transpose flags (which are template parameters, since they pick the copy
/// atoms). Strides are in elements; batch strides are 0 for a broadcast axis.
struct GemmParams {
    const void* a = nullptr;
    const void* b = nullptr;
    void* c = nullptr;
    const void* bias = nullptr;
    const void* residual = nullptr;

    int m = 0, n = 0, k = 0;
    int64_t lda = 0, ldb = 0, ldc = 0, ldr = 0;

    float alpha = 1.0f;
    float beta = 0.0f;

    EpilogueActivateType epilogue = EpilogueActivateType::None;
    float ep_alpha = 0.0f;
    float ep_beta = 1.0f;
    float min_clip = -std::numeric_limits<float>::infinity();
    float max_clip = std::numeric_limits<float>::infinity();

    bool bias_row = false;

    // ---- split-K ----------------------------------------------------------
    // `k_splits` cuts the reduction into that many ranges, one per blockIdx.z
    // slice (z = batch · k_splits + split); each block writes its f32
    // accumulator tile to `partial` instead of running the epilogue, and a
    // second kernel sums the planes and applies the epilogue once. 1 means the
    // plain single-pass kernel and `partial` unused.
    int k_splits = 1;
    float* partial = nullptr; ///< [batch][split][m][n], row-major

    /// A caller's split count for the narrow-m loop, 0 for the automatic
    /// choice. The loop's own rule sizes the split by K and the L2 planes and
    /// never looks at N, so a short gate row (a recurrent step's N = gates·H)
    /// leaves the machine half empty; a caller that knows its shape family can
    /// say so. It only opens the loop for a pose the automatic rule keeps on
    /// the tile (f16 with more than one row), so every existing caller is
    /// unchanged. Clamped by the k tiles and the plane budget like the auto.
    int splits_hint = 0;

    int batch_rank = 0;
    int64_t batch_size[4] = {1, 1, 1, 1};
    int64_t a_batch[4] = {0, 0, 0, 0};
    int64_t b_batch[4] = {0, 0, 0, 0};
    int64_t c_batch[4] = {0, 0, 0, 0};
    int64_t r_batch[4] = {0, 0, 0, 0};
};

// ============================================================
// Epilogue
// ============================================================

__device__ inline float gemm_apply_epilogue(EpilogueActivateType ep, float alpha, float beta,
                                            float x)
{
    switch (ep) {
    case EpilogueActivateType::None:
        return x;
    case EpilogueActivateType::Relu:
        return fmaxf(x, 0.0f);
    case EpilogueActivateType::Relu6:
        return fminf(fmaxf(x, 0.0f), 6.0f);
    case EpilogueActivateType::LeakyRelu:
        return (x > 0.0f) ? x : alpha * x;
    case EpilogueActivateType::Sigmoid:
        return 1.0f / (1.0f + expf(-x));
    case EpilogueActivateType::Tanh:
        return tanhf(x);
    case EpilogueActivateType::Gelu: {
        constexpr float c = 0.7978845608028654f;
        return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
    }
    case EpilogueActivateType::Silu:
        return x / (1.0f + expf(-x));
    case EpilogueActivateType::HardSwish: {
        const float r = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
        return x * r * (beta / 6.0f);
    }
    case EpilogueActivateType::Elu:
        return (x > 0.0f) ? x : alpha * (expf(x) - 1.0f);
    }
    return x;
}

template <class Elem> __device__ inline float gemm_to_float(Elem x) { return float(x); }

template <> __device__ inline float gemm_to_float<float>(float x) { return x; }

// ============================================================
// Geometry
// ============================================================

namespace detail {

constexpr int kGemmThreads = 128;

/// Shared memory a block can ask for on this class of part (GA10x: 128 KB minus
/// the L1 carveout). A tile's `blocks` target has to fit under it.
constexpr int kSmemPerSm = 100 * 1024;

/// The batch index a block owns. Without split-K it is blockIdx.z; with it,
/// blockIdx.z carries batch · k_splits + split, split varying fastest so the
/// slices of one matrix land on different waves of the scheduler.
__device__ inline int gemm_block_batch(const GemmParams& p)
{
    const int z = int(blockIdx.z);
    return p.k_splits > 1 ? z / p.k_splits : z;
}

__device__ inline int gemm_block_split(const GemmParams& p)
{
    return p.k_splits > 1 ? int(blockIdx.z) % p.k_splits : 0;
}

__device__ inline int64_t gemm_batch_offset(const GemmParams& p, const int64_t* strides,
                                            int64_t idx)
{
    int64_t off = 0;
    for (int d = 0; d < p.batch_rank; ++d) {
        const int64_t i = idx % p.batch_size[d];
        idx /= p.batch_size[d];
        off += i * strides[d];
    }
    return off;
}

/// The k-tile range a split owns, as [begin, begin + count). The bounds are
/// proportional cuts of the tile count, so with `k_splits <= k_tiles` every
/// split is non-empty and the ranges differ by at most one tile.
__device__ inline void gemm_split_range(const GemmParams& p, int bK, int split, int& kt_begin,
                                        int& kt_count)
{
    const int k_tiles = (p.k + bK - 1) / bK;
    if (p.k_splits <= 1) {
        kt_begin = 0;
        kt_count = k_tiles;
        return;
    }
    const int64_t tiles = k_tiles;
    kt_begin = int(tiles * split / p.k_splits);
    kt_count = int(tiles * (split + 1) / p.k_splits) - kt_begin;
}

inline int gemm_batch_count(const GemmParams& p)
{
    int count = 1;
    for (int d = 0; d < p.batch_rank; ++d) {
        count *= int(p.batch_size[d]);
    }
    return count;
}

/// Queried once per process; 0 when unavailable, which keeps the narrow tile.
inline int device_sm_count()
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

/// Queried once per process; 0 when unavailable, which drops the plane cap.
inline int64_t device_l2_bytes()
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

// ---- split-K --------------------------------------------------------------

/// Bench knob: `detail::cuda_splitk` is 0 to disable split-K, N > 1 to force N
/// splits (clamped like the auto choice), -1 to keep the heuristic. A test or
/// benchmark stores to it directly; no environment is read.
inline int splitk_env() { return ::nnops::detail::cuda_splitk.load(std::memory_order_relaxed); }

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
inline int pick_split_k(const GemmParams& p, int bM, int bN, int bK, int capacity, int elem_size)
{
    const int forced = splitk_env();
    if (forced == 0) {
        return 1;
    }
    const int k_tiles = (p.k + bK - 1) / bK;
    if (forced > 1) {
        return std::max(1, std::min({forced, k_tiles, kSplitKMaxSplits}));
    }
    if (capacity <= 0) {
        return 1;
    }

    const int64_t batch = gemm_batch_count(p);
    const int64_t tiles = int64_t((p.m + bM - 1) / bM) * ((p.n + bN - 1) / bN) * batch;

    int64_t splits = std::min<int64_t>(kSplitKMaxSplits, capacity / tiles);
    splits = std::min<int64_t>(splits, k_tiles / kSplitKMinTiles);
    if (splits < 2) {
        return 1;
    }

    const int64_t plane = int64_t(p.m) * p.n * batch;
    const int64_t partial_bytes = int64_t{8} * splits * plane; // f32 write + read
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

inline GemmWorkspace gemm_workspace_alloc(int64_t bytes, cudaStream_t stream)
{
    GemmWorkspace ws;
    if (bytes <= 0) {
        return ws;
    }
#if CUDART_VERSION >= 11070
    const cudaMemPool_t pool = gemm_workspace_pool();
    if (pool != nullptr &&
        cudaMallocFromPoolAsync(&ws.ptr, static_cast<size_t>(bytes), pool, stream) == cudaSuccess) {
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
    if (ws.ptr == nullptr) {
        return;
    }
#if CUDART_VERSION >= 11020
    if (ws.async && cudaFreeAsync(ws.ptr, stream) == cudaSuccess) {
        return;
    }
    cudaGetLastError();
#endif
    cudaFree(ws.ptr);
}

} // namespace detail

} // namespace nnops::backend::cuda
