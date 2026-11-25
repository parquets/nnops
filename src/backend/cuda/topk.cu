/// @file topk.cu
/// @brief CUDA TopK — the k largest (or smallest) values along one axis and
///        their indices.
///
/// Three routes, all built on the same selection primitive: fold the best
/// remaining element out of a candidate set (best by value, ties to the lower
/// index), then exclude it from the next pass. The primitive is a serial chain
/// of k block-wide maxima, so what differs is the candidate set it runs on:
///
///  - **Wide rows, k small**: each thread keeps its own top-k in registers
///    while it makes one pass over the axis; the block then merges
///    `blockDim.x * k` candidates, k passes over a few hundred entries. One
///    read of the row instead of k, which is what the old form cost: a k=10
///    selection over 4096x32000 was ten full passes and 11.6 ms.
///  - **Few rows** (decode-time sampling): the row is cut into chunks, one
///    block each, and a second launch selects over the chunks' top-k. A single
///    151936-row had one block on the whole device and read 50 times.
///  - **k past the register list**: the k-pass over the row itself, which
///    needs no per-thread state and so has no k ceiling.
///
/// Every route breaks ties by the lower index; the CPU settles them inside a
/// std::sort that does not pin an order, so the values agree exactly either
/// way. The three routes agree with each other because the selection order is a
/// total order over (value, index) and each chunk's top-k is a subset of the
/// whole that provably contains the global top-k.
///
/// The axis is read through the view's strides (see axis_walk.hpp), so a padded
/// pitch is the same kernel with a different offset group. The operator is
/// layout-agnostic: there is no channel-axis claim to honour.
///
/// f32, f16 and bf16 in; values out at X's dtype, indices s64, both written
/// densely. Workspace: the chunk route takes a stream-ordered scratch block.

#include "cuda_common.cuh"
#include "nnops/ops/topk.hpp"
#include "nnops/detail/axis_walk.hpp"

#include <algorithm>
#include <vector>

namespace nnops::backend::cuda {
namespace {

using detail::AxisGroup;
using detail::axis_group_offset;

// ============================================================
// Geometry
// ============================================================

struct TopKWalk {
    int64_t rows = 0; ///< outer × inner positions
    int64_t inner = 0;
    int64_t axis_dim = 0;
    int64_t axis_stride = 0;
    int64_t k = 0;
    AxisGroup pre;
    AxisGroup post;
};

/// The register list's capacity; above it the row goes to the k-pass route.
constexpr int kTopKReg = 16;

/// The widest block the register route may use: one row is one block, so a
/// single-row selection is otherwise one SM wide.
constexpr int kTopKMaxThreads = 1024;

// ============================================================
// The running best
// ============================================================

/// A candidate pair; an index below zero marks the empty slot, so the first
/// element the block finds always wins.
struct Cand {
    float val;
    int64_t idx;
};

/// `a` beats `b` under (value, then lower index). A NaN never beats anything,
/// which is how the value comparisons already behave; making it explicit keeps
/// two NaNs from trading places between passes.
template <bool IsMax> __device__ inline bool cand_beats(const Cand& a, const Cand& b)
{
    if (a.idx < 0) {
        return false;
    }
    if (b.idx < 0) {
        return true;
    }
    if (isnan(a.val)) {
        return false;
    }
    if (isnan(b.val)) {
        return true;
    }
    if (IsMax ? (a.val > b.val) : (a.val < b.val)) {
        return true;
    }
    return a.val == b.val && a.idx < b.idx;
}

/// Block-wide best, broadcast to every thread through `smem`.
template <bool IsMax> __device__ inline void block_best(Cand& c, Cand* smem) noexcept
{
    const int lane = int(threadIdx.x) % kCudaWarpSize;
    const int wid = int(threadIdx.x) / kCudaWarpSize;

#pragma unroll
    for (int off = kCudaWarpSize / 2; off > 0; off >>= 1) {
        const Cand o{__shfl_down_sync(0xffffffffu, c.val, off),
                     __shfl_down_sync(0xffffffffu, c.idx, off)};
        if (cand_beats<IsMax>(o, c)) {
            c = o;
        }
    }
    if (lane == 0) {
        smem[wid] = c;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const int warps = int(ceil_div(int(blockDim.x), kCudaWarpSize));
        Cand best{0.0f, -1};
        for (int i = 0; i < warps; ++i) {
            if (cand_beats<IsMax>(smem[i], best)) {
                best = smem[i];
            }
        }
        smem[0] = best;
    }
    __syncthreads();

    c = smem[0];
}

/// Where row `row`'s j-th result lands: the output is [outer, k, inner], so the
/// inner axis sits between the selection and the row index. `row * k` is only
/// that when inner == 1.
__device__ inline int64_t topk_out_base(int64_t row, int64_t inner, int64_t k) noexcept
{
    const int64_t o = row / inner;
    const int64_t i = row - o * inner;
    return o * (k * inner) + i;
}

/// The k-pass selection over a strided run: for each of k passes, fold the best
/// remaining candidate out of `[0, n)` and broadcast it. `Load(flat)` returns
/// the candidate at `flat`. Writes the k results through `emit(j, best)`.
template <bool IsMax, class Load, class Emit>
__device__ inline void select_k(int64_t n, int64_t k, Cand* smem, Load load, Emit emit)
{
    Cand last{0.0f, -1};
    for (int64_t j = 0; j < k; ++j) {
        Cand best{0.0f, -1};
        for (int64_t t = int64_t(threadIdx.x); t < n; t += int64_t(blockDim.x)) {
            const Cand c = load(t);
            if ((last.idx < 0 || cand_beats<IsMax>(last, c)) && cand_beats<IsMax>(c, best)) {
                best = c;
            }
        }
        block_best<IsMax>(best, smem);
        if (threadIdx.x == 0) {
            emit(j, best);
        }
        last = best;
    }
}

// ============================================================
// Route 1 — per-thread register lists, one pass over the row
// ============================================================

template <class T, bool IsMax, int K>
__global__ void topk_reg_kernel(const T* __restrict__ x, T* __restrict__ v_out,
                                int64_t* __restrict__ i_out, TopKWalk w)
{
    // One slot per thread per output, so the merge reads exactly what the row
    // can produce rather than the full register capacity.
    extern __shared__ __align__(16) unsigned char smem_raw[];
    float* s_val = reinterpret_cast<float*>(smem_raw);
    int32_t* s_idx = reinterpret_cast<int32_t*>(s_val + int64_t(blockDim.x) * K);
    __shared__ Cand smem[kTopKMaxThreads / kCudaWarpSize];

    for (int64_t row = blockIdx.x; row < w.rows; row += gridDim.x) {
        const int64_t o = row / w.inner;
        const int64_t i = row - o * w.inner;
        const T* p = x + axis_group_offset(o, w.pre) + axis_group_offset(i, w.post);

        // A sorted run of exactly k in registers. `K` is the operator's k, not
        // a capacity: the insert below costs K compares, and a fixed 16-slot
        // capacity measured 4x the running time at k = 1.
        float tv[K];
        int32_t ti[K];
#pragma unroll
        for (int q = 0; q < K; ++q) {
            tv[q] = 0.0f;
            ti[q] = -1;
        }

        Cand worst{0.0f, -1};
        int64_t count = 0;
        for (int64_t t = int64_t(threadIdx.x); t < w.axis_dim; t += int64_t(blockDim.x)) {
            Cand c{s_load(&p[t * w.axis_stride]), t};
            if (count == K && !cand_beats<IsMax>(c, worst)) {
                continue;
            }
            // Insert and push the displaced candidate down. Unrolled with
            // constant indices, so the run stays in registers.
#pragma unroll
            for (int q = 0; q < K; ++q) {
                if (cand_beats<IsMax>(c, Cand{tv[q], ti[q]})) {
                    const Cand displaced{tv[q], ti[q]};
                    tv[q] = c.val;
                    ti[q] = static_cast<int32_t>(c.idx);
                    c = displaced;
                }
            }
            worst = Cand{tv[K - 1], ti[K - 1]};
            if (count < K) {
                ++count;
            }
        }

        const int64_t base = int64_t(threadIdx.x) * K;
#pragma unroll
        for (int q = 0; q < K; ++q) {
            s_val[base + q] = tv[q];
            s_idx[base + q] = ti[q];
        }
        __syncthreads();

        const int64_t total = int64_t(blockDim.x) * K;
        const int64_t out_base = topk_out_base(row, w.inner, K);
        const int64_t out_step = w.inner;
        T* vrow = v_out + out_base;
        int64_t* irow = i_out + out_base;
        // Captured by value: a reference capture would materialize the walk
        // parameter in local memory for every thread.
        select_k<IsMax>(total, K, smem, [=](int64_t t) { return Cand{s_val[t], s_idx[t]}; },
                        [=](int64_t j, const Cand& best) {
                            s_store(&vrow[j * out_step], best.val);
                            irow[j * out_step] = best.idx;
                        });
        __syncthreads();
    }
}

// ============================================================
// Route 2 — chunks for the few-row (decode) shape
// ============================================================

/// One block per (row, chunk); the chunk's top-k lands in the scratch, in order.
template <class T, bool IsMax>
__global__ void topk_chunk_kernel(const T* __restrict__ x, float* __restrict__ cv,
                                  int32_t* __restrict__ ci, TopKWalk w, int64_t chunks,
                                  int64_t chunk_len, int64_t k)
{
    __shared__ Cand smem[kCudaBlockSize / kCudaWarpSize];

    const int64_t row = blockIdx.y;
    const int64_t chunk = blockIdx.x;
    const int64_t t0 = chunk * chunk_len;
    const int64_t t1 = min(t0 + chunk_len, w.axis_dim);
    if (t0 >= t1) {
        return;
    }

    const int64_t o = row / w.inner;
    const int64_t i = row - o * w.inner;
    const T* p = x + axis_group_offset(o, w.pre) + axis_group_offset(i, w.post);
    const int64_t base = (row * chunks + chunk) * k;

    const int64_t stride = w.axis_stride;
    float* cvd = cv + base;
    int32_t* cid = ci + base;
    select_k<IsMax>(t1 - t0, k, smem,
                    [=](int64_t t) {
                        return Cand{s_load(&p[(t0 + t) * stride]), t0 + t};
                    },
                    [=](int64_t j, const Cand& best) {
                        cvd[j] = best.val;
                        cid[j] = static_cast<int32_t>(best.idx);
                    });
}

/// One block per row; the k-pass over the chunks' candidates.
template <class T, bool IsMax>
__global__ void topk_merge_kernel(const float* __restrict__ cv, const int32_t* __restrict__ ci,
                                  T* __restrict__ v_out, int64_t* __restrict__ i_out,
                                  int64_t ncand, int64_t k, int64_t inner)
{
    __shared__ Cand smem[kCudaBlockSize / kCudaWarpSize];

    const int64_t row = blockIdx.x;
    const int64_t base = row * ncand;
    const int64_t out_base = topk_out_base(row, inner, k);
    const int64_t out_step = inner;
    T* vrow = v_out + out_base;
    int64_t* irow = i_out + out_base;
    select_k<IsMax>(ncand, k, smem,
                    [=](int64_t t) { return Cand{cv[base + t], ci[base + t]}; },
                    [=](int64_t j, const Cand& best) {
                        s_store(&vrow[j * out_step], best.val);
                        irow[j * out_step] = best.idx;
                    });
}

// ============================================================
// Route 3 — the k-pass over the row
// ============================================================

template <class T, bool IsMax>
__global__ void topk_kernel(const T* __restrict__ x, T* __restrict__ v_out,
                            int64_t* __restrict__ i_out, TopKWalk w)
{
    __shared__ Cand smem[kCudaBlockSize / kCudaWarpSize];

    for (int64_t row = blockIdx.x; row < w.rows; row += gridDim.x) {
        const int64_t o = row / w.inner;
        const int64_t i = row - o * w.inner;
        const T* p = x + axis_group_offset(o, w.pre) + axis_group_offset(i, w.post);
        const int64_t out_base = topk_out_base(row, w.inner, w.k);
        const int64_t out_step = w.inner;
        T* vrow = v_out + out_base;
        int64_t* irow = i_out + out_base;
        const int64_t stride = w.axis_stride;
        select_k<IsMax>(w.axis_dim, w.k, smem,
                        [=](int64_t t) { return Cand{s_load(&p[t * stride]), t}; },
                        [=](int64_t j, const Cand& best) {
                            s_store(&vrow[j * out_step], best.val);
                            irow[j * out_step] = best.idx;
                        });
    }
}

// ============================================================
// The chunk route's scratch
// ============================================================

/// The chunk route's candidate scratch: a per-thread arena keyed by stream.
///
/// A cudaMallocAsync/cudaFreeAsync pair per selection measured ~0.1 ms on the
/// default stream here, which is more than the route saves at small k. The
/// arena is reused instead: a stream serializes its own launches, so a buffer
/// that a later call on the same stream overwrites cannot be read early, and a
/// second stream from the same thread gets its own block.
struct Arena {
    void* ptr = nullptr;
    size_t bytes = 0;
    cudaStream_t stream = nullptr;
};

/// One cache entry per (thread, stream, size class); the block is never freed,
/// so the cache stays bounded by the distinct streams a thread uses.
void* arena_acquire(size_t bytes, cudaStream_t stream)
{
    constexpr size_t kMaxCached = size_t(4) << 20;
    thread_local std::vector<Arena> arenas;
    for (const Arena& a : arenas) {
        if (a.stream == stream && a.bytes >= bytes) {
            return a.ptr;
        }
    }
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    if (bytes <= kMaxCached) {
        arenas.push_back(Arena{p, bytes, stream});
    }
    return p;
}

// ============================================================
// Entry point
// ============================================================

template <class T>
Status topk_entry(const TopKAttributes& attrs, TensorView& values, TensorView& indices,
                  std::span<const TensorView> inputs, cudaStream_t stream)
{
    const auto& x = inputs[0];
    const int64_t rank = x.rank();
    if (rank == 0) {
        return Status::invalid_argument;
    }
    const int64_t ax = attrs.axis < 0 ? attrs.axis + rank : attrs.axis;
    if (ax < 0 || ax >= rank || attrs.k < 1) {
        return Status::invalid_argument;
    }
    if (attrs.k > x.shape(ax)) {
        return Status::invalid_argument;
    }

    TopKWalk w;
    w.axis_dim = x.shape(ax);
    w.axis_stride = x.stride_elems(ax);
    w.k = attrs.k;
    w.inner = 1;
    for (int64_t d = ax + 1; d < rank; ++d) {
        w.inner *= x.shape(d);
    }
    int64_t outer = 1;
    for (int64_t d = 0; d < ax; ++d) {
        outer *= x.shape(d);
    }
    w.rows = outer * w.inner;

    if (ax > 0) {
        detail::axis_fill_group(w.pre, x, 0, ax, x.stride_elems(ax - 1));
    } else {
        detail::axis_fill_group(w.pre, x, 0, 0, 0);
    }
    detail::axis_fill_group_unit(w.post, x, ax + 1, rank);

    if (w.rows == 0 || w.axis_dim == 0) {
        return Status::ok;
    }

    const bool want_max = attrs.type == TopKType::Max;
    const T* xp = x.ptr<T>();
    T* vp = values.ptr<T>();
    int64_t* ip = indices.ptr<int64_t>();

    // Few rows past the register list: the chunked route, so one row can fill
    // the device. Below that the register list is both faster and free of the
    // scratch block (whose stream-ordered allocation measured ~0.1 ms per call
    // on the default stream, more than this route saves at small k).
    constexpr int64_t kFewRows = 64;
    constexpr int64_t kChunkTarget = 2048;
    if (w.rows <= kFewRows && w.axis_dim >= 4096) {
        const int64_t chunks = std::min<int64_t>((w.axis_dim + kChunkTarget - 1) / kChunkTarget,
                                                 int64_t(1024));
        const int64_t chunk_len = (w.axis_dim + chunks - 1) / chunks;
        const size_t ncand = static_cast<size_t>(w.rows * chunks * w.k);
        void* scratch = arena_acquire(ncand * (sizeof(float) + sizeof(int32_t)), stream);
        if (scratch != nullptr) {
            float* cv = static_cast<float*>(scratch);
            int32_t* ci = reinterpret_cast<int32_t*>(cv + ncand);
            const dim3 grid(static_cast<unsigned>(chunks), static_cast<unsigned>(w.rows));
            if (want_max) {
                topk_chunk_kernel<T, true><<<grid, kCudaBlockSize, 0, stream>>>(xp, cv, ci, w,
                                                                               chunks, chunk_len,
                                                                               w.k);
            } else {
                topk_chunk_kernel<T, false><<<grid, kCudaBlockSize, 0, stream>>>(xp, cv, ci, w,
                                                                                chunks, chunk_len,
                                                                                w.k);
            }
            const unsigned merge_blocks = static_cast<unsigned>(w.rows);
            if (want_max) {
                topk_merge_kernel<T, true><<<merge_blocks, kCudaBlockSize, 0, stream>>>(
                    cv, ci, vp, ip, chunks * w.k, w.k, w.inner);
            } else {
                topk_merge_kernel<T, false><<<merge_blocks, kCudaBlockSize, 0, stream>>>(
                    cv, ci, vp, ip, chunks * w.k, w.k, w.inner);
            }
            return Status::ok;
        }
    }

    // Wide rows with a small k: one pass, then a merge inside the block.
    if (w.k <= kTopKReg && w.axis_dim >= 512) {
        // One row is one block, so a single-row selection is one SM wide: give
        // it the widest block whose shared candidate list still fits.
        int threads = kCudaBlockSize;
        if (w.rows * kCudaBlockSize <= kFewRows * kCudaBlockSize) {
            constexpr int kMaxShare = 48 * 1024;
            for (int cand : {kTopKMaxThreads, kTopKMaxThreads / 2}) {
                if (cand > threads &&
                    int64_t(cand) * w.k * (4 + 4) <= kMaxShare) {
                    threads = cand;
                }
            }
        }
        const size_t shared =
            static_cast<size_t>(threads * w.k) * (sizeof(float) + sizeof(int32_t));
        const unsigned grid = static_cast<unsigned>(std::min<int64_t>(w.rows, 65535));
#define NNOPS_TOPK_REG(K)                                                                     \
    do {                                                                                      \
        if (want_max) {                                                                       \
            topk_reg_kernel<T, true, K><<<grid, kCudaBlockSize, shared, stream>>>(xp, vp, ip, \
                                                                                  w);         \
        } else {                                                                              \
            topk_reg_kernel<T, false, K><<<grid, kCudaBlockSize, shared, stream>>>(xp, vp, ip, \
                                                                                   w);        \
        }                                                                                     \
        return Status::ok;                                                                    \
    } while (false)
        switch (w.k) {
        case 1: NNOPS_TOPK_REG(1);
        case 2: NNOPS_TOPK_REG(2);
        case 3: NNOPS_TOPK_REG(3);
        case 4: NNOPS_TOPK_REG(4);
        case 5: NNOPS_TOPK_REG(5);
        case 6: NNOPS_TOPK_REG(6);
        case 7: NNOPS_TOPK_REG(7);
        case 8: NNOPS_TOPK_REG(8);
        case 9: NNOPS_TOPK_REG(9);
        case 10: NNOPS_TOPK_REG(10);
        case 11: NNOPS_TOPK_REG(11);
        case 12: NNOPS_TOPK_REG(12);
        case 13: NNOPS_TOPK_REG(13);
        case 14: NNOPS_TOPK_REG(14);
        case 15: NNOPS_TOPK_REG(15);
        default: NNOPS_TOPK_REG(16);
        }
#undef NNOPS_TOPK_REG
    }

    const unsigned grid = static_cast<unsigned>(std::min<int64_t>(w.rows, 65535));
    if (want_max) {
        topk_kernel<T, true><<<grid, kCudaBlockSize, 0, stream>>>(xp, vp, ip, w);
    } else {
        topk_kernel<T, false><<<grid, kCudaBlockSize, 0, stream>>>(xp, vp, ip, w);
    }
    return Status::ok;
}

} // anonymous namespace

Status topk_cuda(const TopKAttributes& attrs, TensorView& values, TensorView& indices,
                 std::span<const TensorView> inputs, const ComputeContext& ctx, void* /*workspace*/)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    switch (inputs[0].data_type()) {
    case DataType::f32:
        return topk_entry<float>(attrs, values, indices, inputs, stream);
    case DataType::f16:
        return topk_entry<__half>(attrs, values, indices, inputs, stream);
    case DataType::bf16:
        return topk_entry<__nv_bfloat16>(attrs, values, indices, inputs, stream);
    default:
        return Status::unsupported;
    }
}

} // namespace nnops::backend::cuda
