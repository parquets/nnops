/// @file topk.cu
/// @brief CUDA TopK — the k largest (or smallest) values along one axis and
///        their indices.
///
/// One block per output row, and k passes over it: each pass folds the best
/// remaining element — best by value, ties to the lower index — and the pass
/// after it skips the index it just took. That is the whole algorithm. It needs
/// no sorted copy of the row, so the axis may be as long as the selection needs
/// it to be, and it writes the result already in order, so
/// the `sorted` attribute changes nothing on this backend: the output is sorted
/// whether or not it was asked for, which the contract allows.
///
/// Ties are the one place this can differ from the CPU: it breaks them by the
/// lower index, while the CPU settles them inside a std::sort that does not pin
/// an order. The values agree exactly either way.
///
/// The axis is read through the view's strides (see axis_walk.hpp), so a padded
/// pitch is the same kernel with a different offset group. The operator itself
/// is channel-first only, which the op layer checks.
///
/// f32, f16 and bf16 in; values out at X's dtype, indices s64, both written densely —
/// which is what their descriptors specify. Workspace: not used.

#include "cuda_common.cuh"
#include "nnops/ops/topk.hpp"
#include "nnops/detail/axis_walk.hpp"

#include <algorithm>

namespace nnops::backend::cuda {
namespace {

using detail::AxisGroup;
using detail::axis_group_offset;

// ============================================================
// Geometry
// ============================================================

struct TopKWalk {
    int64_t rows = 0;      ///< outer × inner positions
    int64_t inner = 0;
    int64_t axis_dim = 0;
    int64_t axis_stride = 0;
    int64_t k = 0;
    AxisGroup pre;
    AxisGroup post;
};

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
template <bool IsMax>
__device__ inline bool cand_beats(const Cand& a, const Cand& b)
{
    if (a.idx < 0) { return false; }
    if (b.idx < 0) { return true; }
    if (isnan(a.val)) { return false; }
    if (isnan(b.val)) { return true; }
    if (IsMax ? (a.val > b.val) : (a.val < b.val)) { return true; }
    return a.val == b.val && a.idx < b.idx;
}

/// Block-wide best, broadcast to every thread through `smem`.
template <bool IsMax>
__device__ inline void block_best(Cand& c, Cand* smem) noexcept
{
    constexpr int kWarpCap = kCudaBlockSize / kCudaWarpSize;
    const int lane = int(threadIdx.x) % kCudaWarpSize;
    const int wid = int(threadIdx.x) / kCudaWarpSize;

#pragma unroll
    for (int off = kCudaWarpSize / 2; off > 0; off >>= 1) {
        const Cand o{__shfl_down_sync(0xffffffffu, c.val, off),
                     __shfl_down_sync(0xffffffffu, c.idx, off)};
        if (cand_beats<IsMax>(o, c)) { c = o; }
    }
    if (lane == 0) { smem[wid] = c; }
    __syncthreads();

    if (threadIdx.x == 0) {
        const int warps = int(ceil_div(int(blockDim.x), kCudaWarpSize));
        Cand best{0.0f, -1};
        for (int i = 0; i < warps; ++i) {
            if (cand_beats<IsMax>(smem[i], best)) { best = smem[i]; }
        }
        smem[0] = best;
    }
    __syncthreads();

    c = smem[0];
}

// ============================================================
// Kernel
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

        int64_t v_base = row * w.k;
        Cand last{0.0f, -1};

        for (int64_t j = 0; j < w.k; ++j) {
            Cand best{0.0f, -1};
            for (int64_t t = threadIdx.x; t < w.axis_dim; t += blockDim.x) {
                const Cand c{s_load(&p[t * w.axis_stride]), t};
                // What is left after a pass is everything the last pick beats,
                // so a candidate has to be strictly past it in the (value,
                // index) order — that alone excludes every earlier pick.
                if ((last.idx < 0 || cand_beats<IsMax>(last, c)) &&
                    cand_beats<IsMax>(c, best)) {
                    best = c;
                }
            }
            block_best<IsMax>(best, smem);

            if (threadIdx.x == 0) {
                s_store(&v_out[v_base + j], best.val);
                i_out[v_base + j] = best.idx;
            }
            last = best;
        }
    }
}

// ============================================================
// Entry point
// ============================================================

template <class T>
Status topk_entry(const TopKAttributes& attrs, TensorView& values,
                  TensorView& indices, std::span<const TensorView> inputs,
                  cudaStream_t stream)
{
    const auto& x = inputs[0];
    const int64_t rank = x.rank();
    if (rank == 0) { return Status::invalid_argument; }
    const int64_t ax = attrs.axis < 0 ? attrs.axis + rank : attrs.axis;
    if (ax < 0 || ax >= rank || attrs.k < 1) { return Status::invalid_argument; }
    if (attrs.k > x.shape(ax)) { return Status::invalid_argument; }

    TopKWalk w;
    w.axis_dim = x.shape(ax);
    w.axis_stride = x.stride_elems(ax);
    w.k = attrs.k;
    w.inner = 1;
    for (int64_t d = ax + 1; d < rank; ++d) { w.inner *= x.shape(d); }
    int64_t outer = 1;
    for (int64_t d = 0; d < ax; ++d) { outer *= x.shape(d); }
    w.rows = outer * w.inner;

    if (ax > 0) {
        detail::axis_fill_group(w.pre, x, 0, ax, x.stride_elems(ax - 1));
    } else {
        detail::axis_fill_group(w.pre, x, 0, 0, 0);
    }
    detail::axis_fill_group_unit(w.post, x, ax + 1, rank);

    if (w.rows == 0 || w.axis_dim == 0) { return Status::ok; }

    const unsigned grid = static_cast<unsigned>(
        std::min<int64_t>(w.rows, 65535));
    if (attrs.type == TopKType::Max) {
        topk_kernel<T, true><<<grid, kCudaBlockSize, 0, stream>>>(
            x.ptr<T>(), values.ptr<T>(), indices.ptr<int64_t>(), w);
    } else {
        topk_kernel<T, false><<<grid, kCudaBlockSize, 0, stream>>>(
            x.ptr<T>(), values.ptr<T>(), indices.ptr<int64_t>(), w);
    }
    return Status::ok;
}

}  // anonymous namespace

Status topk_cuda(const TopKAttributes& attrs, TensorView& values,
                 TensorView& indices, std::span<const TensorView> inputs,
                 const ComputeContext& ctx, void* /*workspace*/)
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

}  // namespace nnops::backend::cuda
