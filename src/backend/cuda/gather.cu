/// @file gather.cu
/// @brief CUDA Gather / GatherElements / GatherND — one block per row, an index
///        lookup per row or per element.
///
/// All three map an output coordinate to a source offset, so there is no layout
/// to walk; what differs is how often the index has to be read.
///
///  - Gather keeps the data's tail when the gathered axis is not last, so the
///    index is constant over an output row and the row is one source run — the
///    embedding lookup, and the shape this op exists for. copy_run gives it the
///    vectorized path.
///  - GatherElements, and Gather with the index block last, read one index per
///    element; only the destination is coalesced, which is inherent to the op.
///  - GatherND reads one tuple per row and then copies the data's trailing
///    slice, which is contiguous on both sides.
///
/// The row coordinate is decoded once per block, not once per element, and the
/// output offset is a fixed stride per row.
///
/// Unlike the CPU, an out-of-range index is clamped rather than reported: a
/// device kernel cannot fail mid-stream, and a pre-pass would need a host sync
/// on every call. Indices inside the range — including the negative wrap — read
/// exactly what the CPU reads.
///
/// Data f32/f16/s8/u8/s32/s64, indices s64/s32.

#include "move_common.cuh"
#include "nnops/ops/gather.hpp"

namespace nnops::backend::cuda {
namespace {

/// Clamp after the negative wrap; see the file header.
__device__ inline int64_t gather_index(int64_t k, int64_t n)
{
    if (k < 0) { k += n; }
    if (n <= 0) { return 0; }
    if (k < 0) { return 0; }
    return k < n ? k : n - 1;
}

/// Decode `r` over `dims[0..outer_rank-1]` into `c`.
__device__ inline void decode_row(int64_t r, const int64_t* dims, int64_t outer_rank,
                                  int64_t* c)
{
    int64_t rem = r;
    for (int64_t d = outer_rank - 1; d >= 0; --d) {
        const int64_t n = dims[d];
        c[d] = (n > 0) ? rem % n : 0;
        rem = (n > 0) ? rem / n : 0;
    }
}

// ============================================================
// Gather / GatherElements
// ============================================================

struct GatherGeom {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t outer_rank = 0;     ///< output rank - 1
    int64_t odims[8] = {};
    int64_t dstep[8] = {};      ///< data offset per unit of output dim d
    int64_t istep[8] = {};      ///< index offset per unit of output dim d
    int64_t axis_stride = 0;    ///< data stride of the gathered axis
    int64_t axis_dim = 0;
    int64_t out_step = 1;
    int64_t idx_last_step = 0;
    int64_t elem_trail = 0;     ///< data step per element when the axis is not last
    int64_t src_step = 1;       ///< data stride of the data's last axis
    MoveLayout dst;
    bool dst_compact = true;
};

/// Gather with the data's tail after the gathered axis: the index is the same
/// for every element of the row.
template <class T, class I>
__global__ void gather_rows_kernel(const T* __restrict__ data,
                                   const I* __restrict__ idx,
                                   T* __restrict__ out, GatherGeom g)
{
    for (int64_t r = blockIdx.x; r < g.rows; r += gridDim.x) {
        int64_t c[8] = {};
        decode_row(r, g.odims, g.outer_rank, c);
        int64_t dofs = 0, iofs = 0;
        for (int64_t d = 0; d < g.outer_rank; ++d) {
            dofs += c[d] * g.dstep[d];
            iofs += c[d] * g.istep[d];
        }
        const int64_t k = gather_index(int64_t(idx[iofs]), g.axis_dim);
        const int64_t flat = r * g.cols;
        const int64_t oo = g.dst_compact ? flat : g.dst(flat);
        copy_run(out + oo, data + dofs + k * g.axis_stride, g.cols, g.src_step,
                 g.out_step);
    }
}

/// One index per element.
template <class T, class I>
__global__ void gather_elems_kernel(const T* __restrict__ data,
                                    const I* __restrict__ idx,
                                    T* __restrict__ out, GatherGeom g)
{
    for (int64_t r = blockIdx.x; r < g.rows; r += gridDim.x) {
        int64_t c[8] = {};
        decode_row(r, g.odims, g.outer_rank, c);
        int64_t dofs = 0, iofs = 0;
        for (int64_t d = 0; d < g.outer_rank; ++d) {
            dofs += c[d] * g.dstep[d];
            iofs += c[d] * g.istep[d];
        }
        for (int64_t i = int64_t(threadIdx.x); i < g.cols;
             i += int64_t(blockDim.x)) {
            const int64_t k =
                gather_index(int64_t(idx[iofs + i * g.idx_last_step]), g.axis_dim);
            const int64_t flat = r * g.cols + i;
            const int64_t oo = g.dst_compact ? flat : g.dst(flat);
            // With the gathered axis inside the row, the last axis still steps
            // the source once per element.
            out[oo] = data[dofs + i * g.elem_trail + k * g.axis_stride];
        }
    }
}

template <class T, class I>
Status launch_gather(const TensorView& data, const TensorView& indices,
                     TensorView& output, int64_t axis, bool elements_form,
                     cudaStream_t stream)
{
    const int64_t orank = output.rank();
    const int64_t irank = indices.rank();
    const int64_t total = output.numel();
    if (total == 0) { return Status::ok; }

    GatherGeom g;
    g.outer_rank = orank - 1;
    g.cols = output.shape(orank - 1);
    g.rows = (g.cols > 0) ? total / g.cols : 0;
    g.axis_stride = data.stride_elems(axis);
    g.axis_dim = data.shape(axis);
    g.out_step = output.stride_elems(orank - 1);
    g.src_step = data.stride_elems(data.rank() - 1);
    g.idx_last_step = indices.stride_elems(irank - 1);
    // Only GatherElements has the data's last axis inside a row it must step per
    // element; the plain form reaches this kernel with the index block last.
    g.elem_trail = (elements_form && axis != orank - 1)
                       ? data.stride_elems(orank - 1)
                       : 0;

    int64_t dims[8], dst_strides[8];
    fill_output_dims(output, dims);
    fill_view_strides(output, dst_strides);
    for (int64_t d = 0; d < 8; ++d) {
        g.odims[d] = dims[d];
        g.dstep[d] = 0;
        g.istep[d] = 0;
    }

    // Output axis d comes from either the index block or the data. The gathered
    // axis itself carries no dstep: its offset is the index's k.
    for (int64_t d = 0; d < orank; ++d) {
        if (elements_form) {
            if (d != axis) { g.dstep[d] = data.stride_elems(d); }
            g.istep[d] = indices.stride_elems(d);
        } else if (d < axis) {
            g.dstep[d] = data.stride_elems(d);
        } else if (d < axis + irank) {
            g.istep[d] = indices.stride_elems(d - axis);
        } else {
            g.dstep[d] = data.stride_elems(d - irank + 1);
        }
    }

    g.dst = make_move_layout(dims, dst_strides);
    g.dst_compact = view_is_compact(output);

    const unsigned grid =
        static_cast<unsigned>(std::min<int64_t>(g.rows, int64_t(1) << 20));
    const bool row_form = !elements_form && (axis + irank < orank);
    if (row_form) {
        gather_rows_kernel<T, I><<<grid, kCudaBlockSize, 0, stream>>>(
            data.ptr<T>(), indices.ptr<I>(), output.ptr<T>(), g);
    } else {
        gather_elems_kernel<T, I><<<grid, kCudaBlockSize, 0, stream>>>(
            data.ptr<T>(), indices.ptr<I>(), output.ptr<T>(), g);
    }
    return Status::ok;
}

// ============================================================
// GatherND
// ============================================================

struct GatherNDGeom {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t outer_rank = 0;     ///< output rank - 1
    int64_t k = 0;
    int64_t odims[8] = {};
    int64_t dstep[8] = {};      ///< data offset per unit of output dim d
    int64_t tstep[8] = {};      ///< index offset per unit of output dim d
    int64_t data_stride[8] = {};///< data strides of the k gathered axes
    int64_t data_dim[8] = {};   ///< data extents of the k gathered axes
    int64_t tuple_step = 1;     ///< index stride inside a tuple
    int64_t idx_last_step = 1;  ///< index stride along the last output dim
    int64_t src_step = 1;
    int64_t out_step = 1;
    MoveLayout dst;
    bool dst_compact = true;
};

template <class T, class I>
__global__ void gather_nd_rows_kernel(const T* __restrict__ data,
                                      const I* __restrict__ idx,
                                      T* __restrict__ out, GatherNDGeom g)
{
    for (int64_t r = blockIdx.x; r < g.rows; r += gridDim.x) {
        int64_t c[8] = {};
        decode_row(r, g.odims, g.outer_rank, c);
        int64_t iofs = 0, dofs = 0;
        for (int64_t d = 0; d < g.outer_rank; ++d) {
            iofs += c[d] * g.tstep[d];
            dofs += c[d] * g.dstep[d];
        }
        for (int64_t j = 0; j < g.k; ++j) {
            const int64_t v =
                gather_index(int64_t(idx[iofs + j * g.tuple_step]), g.data_dim[j]);
            dofs += v * g.data_stride[j];
        }
        const int64_t flat = r * g.cols;
        const int64_t oo = g.dst_compact ? flat : g.dst(flat);
        copy_run(out + oo, data + dofs, g.cols, g.src_step, g.out_step);
    }
}

/// No trailing data axes: one element per tuple, so the tuple varies along the
/// last output axis too.
template <class T, class I>
__global__ void gather_nd_elems_kernel(const T* __restrict__ data,
                                       const I* __restrict__ idx,
                                       T* __restrict__ out, GatherNDGeom g)
{
    for (int64_t r = blockIdx.x; r < g.rows; r += gridDim.x) {
        int64_t c[8] = {};
        decode_row(r, g.odims, g.outer_rank, c);
        int64_t iofs = 0, dofs = 0;
        for (int64_t d = 0; d < g.outer_rank; ++d) {
            iofs += c[d] * g.tstep[d];
            dofs += c[d] * g.dstep[d];
        }
        for (int64_t i = int64_t(threadIdx.x); i < g.cols;
             i += int64_t(blockDim.x)) {
            const int64_t io = iofs + i * g.idx_last_step;
            int64_t ds = dofs;
            for (int64_t j = 0; j < g.k; ++j) {
                const int64_t v = gather_index(
                    int64_t(idx[io + j * g.tuple_step]), g.data_dim[j]);
                ds += v * g.data_stride[j];
            }
            const int64_t flat = r * g.cols + i;
            const int64_t oo = g.dst_compact ? flat : g.dst(flat);
            out[oo] = data[ds];
        }
    }
}

template <class T, class I>
Status launch_gather_nd(const TensorView& data, const TensorView& indices,
                        TensorView& output, int64_t batch, cudaStream_t stream)
{
    const int64_t qr = indices.rank();
    const int64_t k = indices.shape(qr - 1);
    const int64_t orank = output.rank();
    const int64_t total = output.numel();
    if (total == 0) { return Status::ok; }

    const int64_t trailing = data.rank() - batch - k;

    GatherNDGeom g;
    g.k = k;
    g.outer_rank = orank - 1;
    g.cols = output.shape(orank - 1);
    g.rows = (g.cols > 0) ? total / g.cols : 0;
    g.tuple_step = indices.stride_elems(qr - 1);
    g.out_step = output.stride_elems(orank - 1);
    g.src_step = data.stride_elems(data.rank() - 1);

    int64_t dims[8], dst_strides[8];
    fill_output_dims(output, dims);
    fill_view_strides(output, dst_strides);
    for (int64_t d = 0; d < 8; ++d) {
        g.odims[d] = dims[d];
        g.dstep[d] = 0;
        g.tstep[d] = 0;
    }
    for (int64_t j = 0; j < k; ++j) {
        g.data_stride[j] = data.stride_elems(batch + j);
        g.data_dim[j] = data.shape(batch + j);
    }

    // Output axes: the batch data axes, then the index tuple axes, then the
    // data axes the tuples do not address.
    for (int64_t d = 0; d < orank; ++d) {
        if (d < batch) {
            // A batch axis advances the data and the index block together.
            g.dstep[d] = data.stride_elems(d);
            g.tstep[d] = indices.stride_elems(d);
        } else if (d < qr - 1) {
            g.tstep[d] = indices.stride_elems(d - batch);
        } else {
            g.dstep[d] = data.stride_elems(batch + k + (d - (qr - 1)));
        }
    }
    // The tuple index varies along the output's last axis, which is the index's
    // axis qr-2 — not the tuple's own stride.
    g.idx_last_step = (orank - 1 >= batch && orank - 1 < qr - 1)
                          ? indices.stride_elems(orank - 1 - batch)
                          : 0;

    g.dst = make_move_layout(dims, dst_strides);
    g.dst_compact = view_is_compact(output);

    const unsigned grid =
        static_cast<unsigned>(std::min<int64_t>(g.rows, int64_t(1) << 20));
    if (trailing > 0) {
        gather_nd_rows_kernel<T, I><<<grid, kCudaBlockSize, 0, stream>>>(
            data.ptr<T>(), indices.ptr<I>(), output.ptr<T>(), g);
    } else {
        gather_nd_elems_kernel<T, I><<<grid, kCudaBlockSize, 0, stream>>>(
            data.ptr<T>(), indices.ptr<I>(), output.ptr<T>(), g);
    }
    return Status::ok;
}

// ============================================================
// Entries
// ============================================================

template <class T, class I>
Status gather_pair(const TensorView& data, const TensorView& indices,
                   TensorView& output, int64_t axis, bool elements_form,
                   cudaStream_t stream)
{
    return launch_gather<T, I>(data, indices, output, axis, elements_form, stream);
}

template <class T>
Status gather_dispatch(const TensorView& data, const TensorView& indices,
                       TensorView& output, int64_t axis, bool elements_form,
                       cudaStream_t stream)
{
    if (indices.data_type() == DataType::s64) {
        return gather_pair<T, int64_t>(data, indices, output, axis, elements_form, stream);
    }
    if (indices.data_type() == DataType::s32) {
        return gather_pair<T, int32_t>(data, indices, output, axis, elements_form, stream);
    }
    return Status::unsupported;
}

template <class T>
Status gather_nd_dispatch(const TensorView& data, const TensorView& indices,
                          TensorView& output, int64_t batch, cudaStream_t stream)
{
    if (indices.data_type() == DataType::s64) {
        return launch_gather_nd<T, int64_t>(data, indices, output, batch, stream);
    }
    if (indices.data_type() == DataType::s32) {
        return launch_gather_nd<T, int32_t>(data, indices, output, batch, stream);
    }
    return Status::unsupported;
}

}  // anonymous namespace

Status gather_cuda(const GatherAttributes& attrs, TensorView& output,
                   std::span<const TensorView> inputs, const ComputeContext& ctx,
                   void* /*workspace*/)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    const int64_t axis = attrs.axis < 0 ? attrs.axis + data.rank() : attrs.axis;
    switch (data.data_type()) {
    case DataType::f32: return gather_dispatch<float>(data, indices, output, axis, false, stream);
    case DataType::f16: return gather_dispatch<__half>(data, indices, output, axis, false, stream);
    case DataType::s8:  return gather_dispatch<int8_t>(data, indices, output, axis, false, stream);
    case DataType::u8:  return gather_dispatch<uint8_t>(data, indices, output, axis, false, stream);
    case DataType::s32: return gather_dispatch<int32_t>(data, indices, output, axis, false, stream);
    case DataType::s64: return gather_dispatch<int64_t>(data, indices, output, axis, false, stream);
    default:            return Status::unsupported;
    }
}

Status gather_elements_cuda(const GatherElementsAttributes& attrs,
                            TensorView& output,
                            std::span<const TensorView> inputs,
                            const ComputeContext& ctx, void* /*workspace*/)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    const int64_t axis = attrs.axis < 0 ? attrs.axis + data.rank() : attrs.axis;
    switch (data.data_type()) {
    case DataType::f32: return gather_dispatch<float>(data, indices, output, axis, true, stream);
    case DataType::f16: return gather_dispatch<__half>(data, indices, output, axis, true, stream);
    case DataType::s8:  return gather_dispatch<int8_t>(data, indices, output, axis, true, stream);
    case DataType::u8:  return gather_dispatch<uint8_t>(data, indices, output, axis, true, stream);
    case DataType::s32: return gather_dispatch<int32_t>(data, indices, output, axis, true, stream);
    case DataType::s64: return gather_dispatch<int64_t>(data, indices, output, axis, true, stream);
    default:            return Status::unsupported;
    }
}

Status gather_nd_cuda(const GatherNDAttributes& attrs, TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx, void* /*workspace*/)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    const int64_t batch = attrs.batch_dims;
    switch (data.data_type()) {
    case DataType::f32: return gather_nd_dispatch<float>(data, indices, output, batch, stream);
    case DataType::f16: return gather_nd_dispatch<__half>(data, indices, output, batch, stream);
    case DataType::s8:  return gather_nd_dispatch<int8_t>(data, indices, output, batch, stream);
    case DataType::u8:  return gather_nd_dispatch<uint8_t>(data, indices, output, batch, stream);
    case DataType::s32: return gather_nd_dispatch<int32_t>(data, indices, output, batch, stream);
    case DataType::s64: return gather_nd_dispatch<int64_t>(data, indices, output, batch, stream);
    default:            return Status::unsupported;
    }
}

}  // namespace nnops::backend::cuda
