/// @file move_common.cuh
/// @brief The copy engine the pure-movement operators share.
///
/// Permute, Slice, Tile, Expand, Concat and Split all reduce to one loop: an
/// output row is a source run copied element by element, and the row's source
/// base is a mixed-radix walk over the output coordinate. That walk is a CuTe
/// layout — `make_move_layout` builds one from the output extents and a stride
/// per output axis — so each operator only states its mapping (which input axis
/// each output axis reads, with what offset and step) and the address algebra
/// stays here. The destination is described the same way, except that the
/// operators never ask for a pitched output, so it is usually compact and its
/// offset is just the row's flat index.
///
/// The layout is fixed at rank 8 with the unused modes padded to extent 1, so a
/// single type serves every operator and no rank has to be templated. `crd2idx`
/// then divides once per mode, which the kernel pays once per row, not once per
/// element: within a row the source is one constant stride, which is also what
/// decides whether the run vectorizes to 16 B (both sides unit-stride and the
/// row's two pointers aligned).
///
/// Non-affine movements — Tile's modulo, Pad's clamped axis maps, Gather's
/// index walk — do not fit a layout and have their own kernels next to the
/// operator.

#pragma once

#include "cuda_common.cuh"
#include "nnops/core/status.hpp"
#include "nnops/core/tensor_view.hpp"

#include <cute/tensor.hpp>

#include <algorithm>
#include <cstdint>

namespace nnops::backend::cuda {

// ============================================================
// The layout pair
// ============================================================

/// One runtime layout, rank 8 (unused modes: extent 1, stride 0).
using MoveShape  = cute::Shape<int64_t, int64_t, int64_t, int64_t,
                               int64_t, int64_t, int64_t, int64_t>;
using MoveStride = cute::Stride<int64_t, int64_t, int64_t, int64_t,
                                int64_t, int64_t, int64_t, int64_t>;
using MoveLayout = cute::Layout<MoveShape, MoveStride>;

/// CuTe's flat coordinate runs mode 0 fastest, which is the tensor's *last*
/// axis, so the modes are filled in reverse axis order; the extent-1 modes the
/// padding adds end up in front, where they contribute nothing.
inline MoveLayout make_move_layout(const int64_t (&dims)[8],
                                   const int64_t (&strides)[8])
{
    return cute::make_layout(
        cute::make_shape(dims[7], dims[6], dims[5], dims[4],
                         dims[3], dims[2], dims[1], dims[0]),
        cute::make_stride(strides[7], strides[6], strides[5], strides[4],
                          strides[3], strides[2], strides[1], strides[0]));
}

/// `view`'s logical strides as the layout's stride array, padded to rank 8.
inline void fill_view_strides(const TensorView& view, int64_t (&strides)[8])
{
    const int64_t rank = view.rank();
    for (int64_t d = 0; d < rank; ++d) { strides[d] = view.stride_elems(d); }
    for (int64_t d = rank; d < 8; ++d) { strides[d] = 0; }
}

/// Whether `view` is laid out densely (pitch padding absent), which is what
/// lets the transpose tile read and write whole rows.
inline bool view_is_compact(const TensorView& view)
{
    int64_t want = 1;
    for (int64_t d = view.rank() - 1; d >= 0; --d) {
        if (view.shape(d) != 1 && view.stride_elems(d) != want) { return false; }
        want *= view.shape(d);
    }
    return true;
}

/// `dims` from the output view, padded to rank 8 with ones.
inline void fill_output_dims(const TensorView& view, int64_t (&dims)[8])
{
    const int64_t rank = view.rank();
    for (int64_t d = 0; d < rank; ++d) { dims[d] = view.shape(d); }
    for (int64_t d = rank; d < 8; ++d) { dims[d] = 1; }
}

// ============================================================
// The plan
// ============================================================

struct MovePlan {
    MoveLayout src;              ///< output flat index -> source element offset
    MoveLayout dst;              ///< output flat index -> destination offset
    int64_t src_base = 0;
    int64_t dst_base = 0;
    int64_t rows = 0;            ///< output extent over axes 0..rank-2
    int64_t cols = 0;            ///< output extent over the last axis
    int64_t src_step = 1;        ///< source stride along the last axis
    int64_t dst_step = 1;        ///< destination stride along the last axis
    /// The destination strides are the compact ones, so `dst(flat) == flat`.
    bool dst_compact = true;
    int64_t rank = 0;
    int64_t dims[8] = {};
    int64_t src_strides[8] = {};
};

/// Whether the whole movement is one contiguous block in both operands, which
/// is what a single `cudaMemcpyAsync` does better than any kernel.
inline bool move_plan_is_linear(const MovePlan& p)
{
    if (!p.dst_compact || p.src_base != 0 || p.dst_base != 0) { return false; }
    int64_t want = 1;
    for (int64_t d = p.rank - 1; d >= 0; --d) {
        if (p.dims[d] != 1 && p.src_strides[d] != want) { return false; }
        want *= p.dims[d];
    }
    return true;
}

// ============================================================
// The row copy
// ============================================================

/// One output row: `n` elements from `src` at `src_step` to `dst` at
/// `dst_step`, split across the block's threads. Both sides unit-stride is the
/// common case and takes the 16 B path when the row happens to be aligned;
/// otherwise every thread loads and stores one element per iteration, which
/// still coalesces on whichever side is contiguous.
template <class T>
__device__ inline void copy_run(T* dst, const T* src, int64_t n,
                                int64_t src_step, int64_t dst_step)
{
    const int64_t tid = int64_t(threadIdx.x);
    const int64_t nth = int64_t(blockDim.x);

    if (src_step == 1 && dst_step == 1) {
        constexpr int64_t kVec = 16 / int64_t(sizeof(T));
        const bool aligned =
            ((reinterpret_cast<uintptr_t>(dst) |
              reinterpret_cast<uintptr_t>(src)) & 15u) == 0;
        if (aligned) {
            const int64_t nvec = (n / kVec) * kVec;
            for (int64_t i = tid * kVec; i < nvec; i += nth * kVec) {
                *reinterpret_cast<uint4*>(dst + i) =
                    *reinterpret_cast<const uint4*>(src + i);
            }
            for (int64_t i = nvec + tid; i < n; i += nth) { dst[i] = src[i]; }
            return;
        }
        for (int64_t i = tid; i < n; i += nth) { dst[i] = src[i]; }
        return;
    }

    for (int64_t i = tid; i < n; i += nth) {
        dst[i * dst_step] = src[i * src_step];
    }
}

// ============================================================
// Kernel and launch
// ============================================================

template <class T>
__global__ void move_rows_kernel(const T* __restrict__ src,
                                 T* __restrict__ dst, MovePlan p)
{
    // The row base is the only place the layouts are evaluated: inside a row
    // both operands advance by a constant stride.
    for (int64_t r = blockIdx.x; r < p.rows; r += gridDim.x) {
        const int64_t flat = r * p.cols;
        const int64_t so = p.src_base + p.src(flat);
        const int64_t dof = p.dst_base +
                            (p.dst_compact ? flat : p.dst(flat));
        copy_run(dst + dof, src + so, p.cols, p.src_step, p.dst_step);
    }
}

template <class T>
Status launch_move(const T* src, T* dst, const MovePlan& plan,
                   cudaStream_t stream)
{
    if (plan.rows <= 0 || plan.cols <= 0) { return Status::ok; }

    if (move_plan_is_linear(plan)) {
        const size_t bytes = static_cast<size_t>(plan.rows * plan.cols) * sizeof(T);
        CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice,
                                   stream));
        return Status::ok;
    }

    const unsigned grid =
        static_cast<unsigned>(std::min<int64_t>(plan.rows, int64_t(1) << 20));
    move_rows_kernel<T><<<grid, kCudaBlockSize, 0, stream>>>(src, dst, plan);
    return Status::ok;
}

}  // namespace nnops::backend::cuda
