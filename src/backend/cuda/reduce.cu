/// @file reduce.cu
/// @brief CUDA Reduce — one axis, and four ways to walk it.
///
/// The CPU describes a reduction as an outer group, the axis, and an inner
/// group; so does this file, and the split decides the kernel:
///
///   - inner == 1 (the axis is the last one): the outputs are rows. A row
///     that is at most one 16 B load wide gets one thread, a short one a warp,
///     and a long one a block — the fan-out a row can use before the block
///     reduction costs more than it saves.
///   - inner > 1: the inner run is contiguous and the axis stride is a
///     constant, so threads span the inner axis (coalesced on every step) and
///     each carries one accumulator down the axis. That is also the shape of
///     `Reduce(NHWC, axis=0)`, where the old offset table cost 195 ms on
///     4096x4096: the table was an int64 per reduced element plus a
///     `cudaMalloc`/`cudaFree` pair per call, and the strides alone describe
///     the walk.
///
/// The comparison rules are the CPU's: a strict compare, so a NaN never
/// replaces a finite value, and the seed is the op's identity (so an all-NaN
/// row still answers ±1e30 the way the CPU does). Sum and Mean accumulate in
/// float like the CPU's SIMD path.
///
/// f32, f16 and bf16. Workspace: not used.

#include "cuda_common.cuh"
#include "nnops/ops/reduce.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/detail/assert.hpp"

#include <algorithm>
#include <span>

namespace nnops::backend::cuda {

using namespace nnops;

namespace {

enum RedOp : int { kRedSum = 0, kRedMin = 1, kRedMax = 2, kRedMean = 3 };

template <int Op> __device__ __forceinline__ float red_init() noexcept
{
    if constexpr (Op == kRedMin) {
        return 1e30f;
    } else if constexpr (Op == kRedMax) {
        return -1e30f;
    }
    return 0.0f;
}

template <int Op> __device__ __forceinline__ void red_step(float& a, float v) noexcept
{
    if constexpr (Op == kRedSum || Op == kRedMean) {
        a += v;
    } else if constexpr (Op == kRedMin) {
        if (v < a) {
            a = v;
        }
    } else {
        if (v > a) {
            a = v;
        }
    }
}

template <int Op> __device__ __forceinline__ float red_combine(float a, float b) noexcept
{
    if constexpr (Op == kRedSum || Op == kRedMean) {
        return a + b;
    } else if constexpr (Op == kRedMin) {
        return (b < a) ? b : a;
    } else {
        return (b > a) ? b : a;
    }
}

template <int Op> __device__ __forceinline__ float red_finish(float a, int64_t n) noexcept
{
    if constexpr (Op == kRedMean) {
        return a / static_cast<float>(n);
    }
    return a;
}

template <int Op> __device__ __forceinline__ float red_warp(float v) noexcept
{
#pragma unroll
    for (int off = kCudaWarpSize / 2; off > 0; off >>= 1) {
        v = red_combine<Op>(v, __shfl_down_sync(0xffffffffu, v, off));
    }
    return v;
}

/// The per-row result is written by one lane, so a warp reduction has to be
/// broadcast back out of lane 0.
template <int Op> __device__ __forceinline__ float red_block(float v, float* smem) noexcept
{
    const int lane = int(threadIdx.x) & (kCudaWarpSize - 1);
    const int wid = int(threadIdx.x) >> 5;
    v = red_warp<Op>(v);
    if (lane == 0) {
        smem[wid] = v;
    }
    __syncthreads();
    const int warps = int((blockDim.x + kCudaWarpSize - 1) / kCudaWarpSize);
    float total = red_init<Op>();
    if (wid == 0) {
        total = (lane < warps) ? smem[lane] : red_init<Op>();
        total = red_warp<Op>(total);
    }
    return total;
}

// ============================================================
// inner == 1 — a row of outputs, three widths
// ============================================================

/// Rows of at most one 16 B load: one thread per row, so a 16-element f16 row
/// still fills a warp's worth of sectors instead of a block.
template <class T, int Op>
__global__ void reduce_row_thread_kernel(const T* __restrict__ x, T* __restrict__ y, int64_t rows,
                                         int64_t n, int64_t row_stride, int64_t out_stride)
{
    const int64_t stride = int64_t(gridDim.x) * blockDim.x;
    for (int64_t r = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; r < rows; r += stride) {
        const T* p = x + r * row_stride;
        float acc = red_init<Op>();
        for (int64_t k = 0; k < n; ++k) {
            red_step<Op>(acc, s_load(&p[k]));
        }
        s_store(&y[r * out_stride], red_finish<Op>(acc, n));
    }
}

/// A warp a row: the lanes stride the row, which keeps every step coalesced
/// even when the row is shorter than a block.
template <class T, int Op>
__global__ void reduce_row_warp_kernel(const T* __restrict__ x, T* __restrict__ y, int64_t rows,
                                       int64_t n, int64_t row_stride, int64_t out_stride)
{
    const int lane = int(threadIdx.x) & (kCudaWarpSize - 1);
    const int warps = int(blockDim.x) >> 5;
    const int warp = int(threadIdx.x) >> 5;
    const int64_t stride = int64_t(gridDim.x) * warps;

    for (int64_t r = int64_t(blockIdx.x) * warps + warp; r < rows; r += stride) {
        const T* p = x + r * row_stride;
        float acc = red_init<Op>();
        for (int64_t k = lane; k < n; k += kCudaWarpSize) {
            red_step<Op>(acc, s_load(&p[k]));
        }
        acc = red_warp<Op>(acc);
        if (lane == 0) {
            s_store(&y[r * out_stride], red_finish<Op>(acc, n));
        }
    }
}

/// A block a row, for rows too long for one warp to keep in flight.
template <class T, int Op>
__global__ void reduce_row_block_kernel(const T* __restrict__ x, T* __restrict__ y, int64_t rows,
                                        int64_t n, int64_t row_stride, int64_t out_stride)
{
    extern __shared__ float smem[];

    for (int64_t r = blockIdx.x; r < rows; r += gridDim.x) {
        const T* p = x + r * row_stride;
        float acc = red_init<Op>();
        for (int64_t k = threadIdx.x; k < n; k += blockDim.x) {
            red_step<Op>(acc, s_load(&p[k]));
        }
        acc = red_block<Op>(acc, smem);
        if (threadIdx.x == 0) {
            s_store(&y[r * out_stride], red_finish<Op>(acc, n));
        }
        __syncthreads();
    }
}

// ============================================================
// inner > 1 — a thread an inner position
// ============================================================

/// The axes after the reduced one, fastest last, with the strides of both
/// operands — the inner run is only a flat index when both are compact, and a
/// padded pitch or a channel-last row breaks that.
struct InnerWalk {
    int64_t rank = 0;
    int64_t dims[8] = {};
    int64_t in_stride[8] = {};
    int64_t out_stride[8] = {};
};

__device__ __forceinline__ void inner_offsets(const InnerWalk& w, int64_t flat, int64_t& in,
                                              int64_t& out) noexcept
{
    in = 0;
    out = 0;
    for (int64_t d = w.rank - 1; d >= 0; --d) {
        const int64_t n = w.dims[d];
        const int64_t c = flat % n;
        in += c * w.in_stride[d];
        out += c * w.out_stride[d];
        flat /= n;
    }
}

template <class T, int Op>
__global__ void reduce_inner_kernel(const T* __restrict__ x, T* __restrict__ y, int64_t outer,
                                    int64_t d, int64_t inner, int64_t axis_stride,
                                    int64_t in_outer_stride, int64_t out_outer_stride,
                                    InnerWalk w)
{
    const int64_t inner_stride = int64_t(gridDim.x) * blockDim.x;
    for (int64_t o = blockIdx.y; o < outer; o += gridDim.y) {
        const T* p = x + o * in_outer_stride;
        T* q = y + o * out_outer_stride;
        for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < inner;
             i += inner_stride) {
            int64_t io = 0;
            int64_t oo = 0;
            inner_offsets(w, i, io, oo);
            const T* c = p + io;
            float acc = red_init<Op>();
            for (int64_t k = 0; k < d; ++k) {
                red_step<Op>(acc, s_load(&c[k * axis_stride]));
            }
            s_store(&q[oo], red_finish<Op>(acc, d));
        }
    }
}

// ============================================================
// Host side
// ============================================================

int to_op(ReduceType t) noexcept
{
    switch (t) {
    case ReduceType::Sum:
        return kRedSum;
    case ReduceType::Min:
        return kRedMin;
    case ReduceType::Max:
        return kRedMax;
    case ReduceType::Mean:
        return kRedMean;
    }
    return kRedSum;
}

template <class T, int Op>
void launch_rows(const T* x, T* y, int64_t rows, int64_t n, int64_t row_stride,
                 int64_t out_stride, cudaStream_t stream)
{
    if (rows <= 0 || n <= 0) {
        return;
    }
    // Up to 64 B a thread reads the whole row itself: 16 f16 or 16 f32 is one
    // or two sectors, which a warp already gathers coalesced, while a warp per
    // row would leave most of its lanes past the end.
    const int64_t row_bytes = n * static_cast<int64_t>(sizeof(T));
    if (row_bytes <= 64) {
        const int64_t threads = std::min<int64_t>(rows, int64_t(1) << 20);
        const unsigned blocks = static_cast<unsigned>((threads + kCudaBlockSize - 1) / kCudaBlockSize);
        reduce_row_thread_kernel<T, Op><<<blocks, kCudaBlockSize, 0, stream>>>(
            x, y, rows, n, row_stride, out_stride);
        return;
    }
    if (n <= 4096) {
        constexpr int kWarps = 8;
        const int64_t warps = std::min<int64_t>(rows, int64_t(1) << 20);
        const unsigned blocks = static_cast<unsigned>((warps + kWarps - 1) / kWarps);
        reduce_row_warp_kernel<T, Op><<<blocks, kWarps * kCudaWarpSize, 0, stream>>>(
            x, y, rows, n, row_stride, out_stride);
        return;
    }
    const unsigned blocks = static_cast<unsigned>(std::min<int64_t>(rows, int64_t(1) << 20));
    const int smem = int((kCudaBlockSize / kCudaWarpSize) * sizeof(float));
    reduce_row_block_kernel<T, Op><<<blocks, kCudaBlockSize, smem, stream>>>(
        x, y, rows, n, row_stride, out_stride);
}

template <class T, int Op>
void launch_inner(const T* x, T* y, int64_t outer, int64_t d, int64_t inner, int64_t axis_stride,
                  int64_t in_outer_stride, int64_t out_outer_stride, const InnerWalk& w,
                  cudaStream_t stream)
{
    if (outer <= 0 || inner <= 0 || d <= 0) {
        return;
    }
    const int64_t x_blocks = (inner + kCudaBlockSize - 1) / kCudaBlockSize;
    const dim3 grid(static_cast<unsigned>(std::min<int64_t>(x_blocks, 65535)),
                    static_cast<unsigned>(std::min<int64_t>(outer, 65535)));
    reduce_inner_kernel<T, Op><<<grid, kCudaBlockSize, 0, stream>>>(
        x, y, outer, d, inner, axis_stride, in_outer_stride, out_outer_stride, w);
}

template <class T>
void reduce_cuda_impl(const ReduceAttributes& attrs, TensorView& output, const TensorView& input,
                      int64_t axis, cudaStream_t stream)
{
    const int64_t rank = input.rank();
    const int64_t n = input.shape(axis);

    int64_t outer = 1;
    for (int64_t d = 0; d < axis; ++d) {
        outer *= input.shape(d);
    }
    int64_t inner = 1;
    for (int64_t d = axis + 1; d < rank; ++d) {
        inner *= input.shape(d);
    }

    const T* x = input.ptr<T>();
    T* y = output.ptr<T>();
    const int64_t axis_stride = input.stride_elems(axis);

    if (inner == 1) {
        // Rows are the axis-1 pitch apart: one spatial line of a channel-last
        // view, or one row of a padded Dense one.
        const int64_t row_stride = axis > 0 ? input.stride_elems(axis - 1) : n;
        const int64_t out_stride = axis > 0 ? output.stride_elems(axis - 1) : 1;
        switch (to_op(attrs.type)) {
        case kRedSum:
            launch_rows<T, kRedSum>(x, y, outer, n, row_stride, out_stride, stream);
            break;
        case kRedMin:
            launch_rows<T, kRedMin>(x, y, outer, n, row_stride, out_stride, stream);
            break;
        case kRedMax:
            launch_rows<T, kRedMax>(x, y, outer, n, row_stride, out_stride, stream);
            break;
        default:
            launch_rows<T, kRedMean>(x, y, outer, n, row_stride, out_stride, stream);
            break;
        }
        return;
    }

    // The axes after the reduced one are one odometer on each side: the input's
    // outer stride past the axis is the run between consecutive inner blocks,
    // and the output's is the same axis minus the one the reduce removed.
    InnerWalk w;
    for (int64_t d = axis + 1; d < rank; ++d) {
        w.dims[w.rank] = input.shape(d);
        w.in_stride[w.rank] = input.stride_elems(d);
        w.out_stride[w.rank] =
            output.stride_elems(attrs.keepdims ? d : d - 1);
        ++w.rank;
    }
    const int64_t out_outer_stride = axis > 0 ? output.stride_elems(axis - 1) : 0;
    const int64_t in_outer_stride = axis > 0 ? input.stride_elems(axis - 1) : 0;

    switch (to_op(attrs.type)) {
    case kRedSum:
        launch_inner<T, kRedSum>(x, y, outer, n, inner, axis_stride, in_outer_stride,
                                 out_outer_stride, w, stream);
        break;
    case kRedMin:
        launch_inner<T, kRedMin>(x, y, outer, n, inner, axis_stride, in_outer_stride,
                                 out_outer_stride, w, stream);
        break;
    case kRedMax:
        launch_inner<T, kRedMax>(x, y, outer, n, inner, axis_stride, in_outer_stride,
                                 out_outer_stride, w, stream);
        break;
    default:
        launch_inner<T, kRedMean>(x, y, outer, n, inner, axis_stride, in_outer_stride,
                                  out_outer_stride, w, stream);
        break;
    }
}

} // anonymous namespace

Status reduce_cuda(const ReduceAttributes& attrs, TensorView& output,
                   std::span<const TensorView> inputs, const ComputeContext& ctx,
                   void* /*workspace*/)
{
    const auto& input = inputs[0];
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    // Single-axis only; multi-axis and the empty-means-all default fall back to
    // the caller.
    if (attrs.axes.size() != 1) {
        return Status::unsupported;
    }
    int64_t axis = attrs.axes[0];
    if (axis < 0) {
        axis += input.rank();
    }
    if (axis < 0 || axis >= input.rank()) {
        return Status::invalid_argument;
    }

    switch (input.data_type()) {
    case DataType::f32:
        reduce_cuda_impl<float>(attrs, output, input, axis, stream);
        return Status::ok;
    case DataType::f16:
        reduce_cuda_impl<__half>(attrs, output, input, axis, stream);
        return Status::ok;
    case DataType::bf16:
        reduce_cuda_impl<__nv_bfloat16>(attrs, output, input, axis, stream);
        return Status::ok;
    default:
        return Status::unsupported;
    }
}

} // namespace nnops::backend::cuda
