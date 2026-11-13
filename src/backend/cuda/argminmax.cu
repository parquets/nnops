/// @file argminmax.cu
/// @brief CUDA ArgMax / ArgMin.
///
/// The reduction is described the way the CPU describes it — an outer group,
/// the axis, an inner group — and both groups come from the view's strides
/// alone (see axis_walk.hpp), so a padded row and a channel-last view need no
/// case of their own: either can flatten or ask for one term per dim.
/// The axis walk itself is the CPU's: a strict compare, so the first of equal
/// values keeps the index, and a NaN never beats a number. The one place the CPU
/// is not simply "strict compare" is the starting point — it seeds the reduction
/// with element 0 and never replaces it with a NaN — so both kernels re-read
/// element 0 and answer 0 when it is a NaN, which is the same rule said once.
///
/// Two kernels, split on whether the axis is the last one:
///   - arg_axis_last: inner == 1, so the outputs are rows. A warp owns a row and
///     its lanes stride the axis, then a shuffle tree folds the (value, index)
///     pairs; the lanes past the axis end carry an empty pair that never wins.
///     This is the argmax-over-logits shape, where the row is contiguous.
///   - arg_strided: inner > 1, so every output is one thread walking the axis
///     itself. Adjacent threads take adjacent inner offsets, which is the
///     coalesced order for a planar tail, and no cross-thread combine is needed.
///
/// Any layout: the operator reads only the strides, so channel-last and a
/// padded pitch are the same walk with different groups.
///
/// f32 and f16 in, s64 out. Workspace: not used.

#include "cuda_common.cuh"
#include "nnops/ops/argminmax.hpp"
#include "nnops/detail/axis_walk.hpp"

#include <algorithm>

namespace nnops::backend::cuda {
namespace {

using detail::AxisWalk;
using detail::axis_group_offset;

// ============================================================
// The running answer
// ============================================================

/// A NaN is not an indexable value for this comparison, so it must not ride the
/// shuffle tree either: an empty pair (idx < 0) loses to anything, a NaN loses to
/// any number, and two NaNs resolve on the lower index so the result stays
/// deterministic.
struct ArgPair {
    float val;
    int64_t idx;
};

constexpr int64_t kNoIdx = -1;

template <bool IsMax>
__device__ inline bool arg_beats(const ArgPair& cand, const ArgPair& cur)
{
    if (cand.idx == kNoIdx) { return false; }
    if (cur.idx == kNoIdx) { return true; }
    if (isnan(cand.val)) { return isnan(cur.val) && cand.idx < cur.idx; }
    if (isnan(cur.val)) { return true; }
    if (IsMax ? (cand.val > cur.val) : (cand.val < cur.val)) { return true; }
    return cand.val == cur.val && cand.idx < cur.idx;
}

// ============================================================
// inner == 1 — a warp a row
// ============================================================

template <class T, bool IsMax>
__global__ void arg_axis_last_kernel(const T* __restrict__ in,
                                     int64_t* __restrict__ out, AxisWalk w)
{
    const int lane = int(threadIdx.x) & (kCudaWarpSize - 1);
    const int warps = int(blockDim.x) >> 5;
    const int warp = int(threadIdx.x) >> 5;
    const int64_t row_step = int64_t(gridDim.x) * warps;

    for (int64_t r = int64_t(blockIdx.x) * warps + warp; r < w.outer; r += row_step) {
        const T* p = in + axis_group_offset(r, w.pre);

        ArgPair best{0.0f, kNoIdx};
        if (lane < w.axis_dim) {
            best.val = s_load(&p[int64_t(lane) * w.axis_stride]);
            best.idx = lane;
            for (int64_t k = int64_t(lane) + kCudaWarpSize; k < w.axis_dim;
                 k += kCudaWarpSize) {
                const ArgPair cand{s_load(&p[k * w.axis_stride]), k};
                if (arg_beats<IsMax>(cand, best)) { best = cand; }
            }
        }

#pragma unroll
        for (int off = kCudaWarpSize / 2; off > 0; off >>= 1) {
            const ArgPair cand{__shfl_down_sync(0xffffffffu, best.val, off),
                               __shfl_down_sync(0xffffffffu, best.idx, off)};
            if (arg_beats<IsMax>(cand, best)) { best = cand; }
        }

        if (lane == 0) {
            out[axis_group_offset(r, w.out_pre)] =
                isnan(s_load(p)) ? int64_t(0) : best.idx;
        }
    }
}

// ============================================================
// inner > 1 — a thread an output
// ============================================================

template <class T, bool IsMax>
__global__ void arg_strided_kernel(const T* __restrict__ in,
                                   int64_t* __restrict__ out, AxisWalk w,
                                   int64_t count)
{
    const int64_t step = int64_t(gridDim.x) * blockDim.x;
    for (int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; t < count;
         t += step) {
        const int64_t o = t / w.inner;
        const int64_t i = t - o * w.inner;
        const T* p = in + axis_group_offset(o, w.pre) + axis_group_offset(i, w.post);

        const float first = s_load(p);
        ArgPair best{first, 0};
        for (int64_t k = 1; k < w.axis_dim; ++k) {
            const ArgPair cand{s_load(p + k * w.axis_stride), k};
            if (arg_beats<IsMax>(cand, best)) { best = cand; }
        }
        out[axis_group_offset(o, w.out_pre) + axis_group_offset(i, w.out_post)] =
            isnan(first) ? int64_t(0) : best.idx;
    }
}

// ============================================================
// Launch
// ============================================================

template <class T, bool IsMax>
void launch_arg(const AxisWalk& w, const T* in, int64_t* out, cudaStream_t stream)
{
    if (w.inner == 1) {
        constexpr int kWarps = kCudaBlockSize / kCudaWarpSize;
        const unsigned grid = static_cast<unsigned>(
            std::min<int64_t>(ceil_div(w.outer, kWarps), 65535));
        arg_axis_last_kernel<T, IsMax>
            <<<grid, kCudaBlockSize, 0, stream>>>(in, out, w);
        return;
    }
    const int64_t count = w.outer * w.inner;
    const unsigned grid = static_cast<unsigned>(
        std::min<int64_t>(ceil_div(count, kCudaBlockSize), 65535));
    arg_strided_kernel<T, IsMax>
        <<<grid, kCudaBlockSize, 0, stream>>>(in, out, w, count);
}

template <class T>
void argminmax_impl(const ArgMinMaxAttributes& attrs, TensorView& output,
                    const TensorView& input, cudaStream_t stream)
{
    const int64_t rank = input.rank();
    const int64_t ax = attrs.axis < 0 ? attrs.axis + rank : attrs.axis;
    const AxisWalk w = detail::make_axis_walk(input, output, ax, attrs.keepdims);

    if (w.axis_dim == 0 || w.outer == 0 || w.inner == 0) { return; }

    const T* in = input.ptr<T>();
    int64_t* out = output.ptr<int64_t>();
    if (attrs.type == ArgMinMaxType::Max) {
        launch_arg<T, true>(w, in, out, stream);
    } else {
        launch_arg<T, false>(w, in, out, stream);
    }
}

template <class T>
Status argminmax_entry(const ArgMinMaxAttributes& attrs, TensorView& output,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();
    const int64_t ax = attrs.axis < 0 ? attrs.axis + rank : attrs.axis;
    if (rank == 0 || ax < 0 || ax >= rank) { return Status::invalid_argument; }
    argminmax_impl<T>(attrs, output, input,
                      static_cast<cudaStream_t>(ctx.cuda_stream));
    return Status::ok;
}

}  // anonymous namespace

// ============================================================
// Entry points
// ============================================================

Status argmax_cuda(const ArgMinMaxAttributes& attrs, TensorView& output,
                   std::span<const TensorView> inputs, const ComputeContext& ctx,
                   void* /*workspace*/)
{
    switch (inputs[0].data_type()) {
    case DataType::f32:
        return argminmax_entry<float>(attrs, output, inputs, ctx);
    case DataType::f16:
        return argminmax_entry<__half>(attrs, output, inputs, ctx);
    default:
        return Status::unsupported;
    }
}

Status argmin_cuda(const ArgMinMaxAttributes& attrs, TensorView& output,
                   std::span<const TensorView> inputs, const ComputeContext& ctx,
                   void* workspace)
{
    ArgMinMaxAttributes a = attrs;
    a.type = ArgMinMaxType::Min;
    return argmax_cuda(a, output, inputs, ctx, workspace);
}

}  // namespace nnops::backend::cuda
