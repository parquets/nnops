/// @file scatter.cu
/// @brief CUDA ScatterElements / ScatterND — copy `data`, then write the
///        updates at their indices.
///
/// Both are the same two steps as the CPU's: the output starts as a copy of
/// data (the shared layout walk, so a pitched or transposed operand needs no
/// case of its own), and then the updates land. ScatterElements walks the index
/// tensor one row per block — the row base of all three operands is fixed, only
/// the gathered coordinate varies — and ScatterND walks tuples, each writing
/// its data slice as one run.
///
/// Duplicates. Under Reduction::None two writers to the same element race, and
/// the contract already calls that unordered. Add uses the ISA's atomics, so it
/// accumulates exactly like the CPU; Mul has no atomic form, so it is built from
/// a compare-and-swap loop on the 32-bit dtypes and reported unsupported on the
/// others — the CPU route keeps covering those.
///
/// An out-of-range index is clamped rather than reported, for the reason in
/// gather.cu.

#include "move_common.cuh"
#include "nnops/detail/validate.hpp"
#include "nnops/ops/scatter.hpp"

namespace nnops::backend::cuda {
namespace {

// ============================================================
// Reduction
// ============================================================

template <class T>
__device__ inline void scatter_none(T* dst, T v)
{
    *dst = v;
}

__device__ inline void scatter_add(float* d, float v) { atomicAdd(d, v); }
__device__ inline void scatter_add(__half* d, __half v) { atomicAdd(d, v); }
__device__ inline void scatter_add(int32_t* d, int32_t v) { atomicAdd(d, v); }
__device__ inline void scatter_add(int64_t* d, int64_t v)
{
    atomicAdd(reinterpret_cast<unsigned long long*>(d),
              static_cast<unsigned long long>(v));
}

__device__ inline void scatter_mul(float* d, float v)
{
    int* addr = reinterpret_cast<int*>(d);
    int old = *addr;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(addr, assumed,
                        __float_as_int(__int_as_float(assumed) * v));
    } while (assumed != old);
}

/// Whether the backend can do this (dtype, reduction) pair; the answer is a
/// property of the ISA, so the host decides it and the kernel keeps one path.
template <class T>
constexpr bool scatter_add_supported()
{
    return std::is_same_v<T, float> || std::is_same_v<T, __half> ||
           std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>;
}

template <class T>
constexpr bool scatter_mul_supported()
{
    return std::is_same_v<T, float>;
}

template <class T>
__device__ inline void scatter_apply(T* dst, const T* src, ScatterReduction red)
{
    // The host rejects the (dtype, reduction) pairs the ISA cannot do, so the
    // unreachable branches are there only to keep the template well-formed.
    if (red == ScatterReduction::None) {
        scatter_none(dst, *src);
    } else if (red == ScatterReduction::Add) {
        if constexpr (scatter_add_supported<T>()) {
            scatter_add(dst, *src);
        } else {
            scatter_none(dst, *src);
        }
    } else {
        if constexpr (scatter_mul_supported<T>()) {
            scatter_mul(dst, *src);
        } else {
            scatter_none(dst, *src);
        }
    }
}

/// Clamp after the negative wrap (see gather.cu).
__device__ inline int64_t scatter_index(int64_t k, int64_t n)
{
    if (k < 0) { k += n; }
    if (n <= 0) { return 0; }
    if (k < 0) { return 0; }
    return k < n ? k : n - 1;
}

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
// Copying data into the output
// ============================================================

template <class T>
Status copy_into(const TensorView& data, TensorView& output, cudaStream_t stream)
{
    const int64_t rank = data.rank();
    int64_t dims[8], src_strides[8], dst_strides[8];
    fill_output_dims(output, dims);
    fill_view_strides(data, src_strides);
    fill_view_strides(output, dst_strides);

    MovePlan p;
    p.rank = rank;
    p.src = make_move_layout(dims, src_strides);
    p.dst = make_move_layout(dims, dst_strides);
    p.cols = dims[rank - 1];
    p.rows = (p.cols > 0) ? output.numel() / p.cols : 0;
    p.src_step = src_strides[rank - 1];
    p.dst_step = dst_strides[rank - 1];
    p.dst_compact = view_is_compact(output);
    for (int d = 0; d < 8; ++d) {
        p.dims[d] = dims[d];
        p.src_strides[d] = src_strides[d];
    }
    return launch_move(data.ptr<T>(), output.ptr<T>(), p, stream);
}

// ============================================================
// ScatterElements
// ============================================================

struct ScatterGeom {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t outer_rank = 0;
    int64_t odims[8] = {};
    int64_t istep[8] = {};
    int64_t ustep[8] = {};
    int64_t ostep[8] = {};
    int64_t axis_ostep = 0;
    int64_t axis_dim = 0;
    int64_t otrail = 0;         ///< output step per element, axis not last
    int64_t idx_last = 1;
    int64_t upd_last = 1;
    ScatterReduction red = ScatterReduction::None;
};

template <class T, class I>
__global__ void scatter_elements_kernel(const I* __restrict__ idx,
                                        const T* __restrict__ updates,
                                        T* __restrict__ out, ScatterGeom g)
{
    for (int64_t r = blockIdx.x; r < g.rows; r += gridDim.x) {
        int64_t c[8] = {};
        decode_row(r, g.odims, g.outer_rank, c);
        int64_t iofs = 0, uofs = 0, oofs = 0;
        for (int64_t d = 0; d < g.outer_rank; ++d) {
            iofs += c[d] * g.istep[d];
            uofs += c[d] * g.ustep[d];
            oofs += c[d] * g.ostep[d];
        }
        for (int64_t i = int64_t(threadIdx.x); i < g.cols;
             i += int64_t(blockDim.x)) {
            const int64_t k =
                scatter_index(int64_t(idx[iofs + i * g.idx_last]), g.axis_dim);
            // The last axis still steps the destination once per element when
            // the scattered axis sits inside the row.
            scatter_apply(out + oofs + i * g.otrail + k * g.axis_ostep,
                          updates + uofs + i * g.upd_last, g.red);
        }
    }
}

template <class T, class I>
Status run_scatter_elements(const ScatterElementsAttributes& attrs,
                            TensorView& output, const TensorView& data,
                            const TensorView& indices,
                            const TensorView& updates, cudaStream_t stream)
{
    NNOPS_TRY(copy_into<T>(data, output, stream));

    const int64_t rank = indices.rank();
    const int64_t axis = attrs.axis < 0 ? attrs.axis + rank : attrs.axis;
    const int64_t total = indices.numel();
    if (total == 0) { return Status::ok; }

    ScatterGeom g;
    g.outer_rank = rank - 1;
    g.cols = indices.shape(rank - 1);
    g.rows = (g.cols > 0) ? total / g.cols : 0;
    g.axis_ostep = output.stride_elems(axis);
    g.axis_dim = data.shape(axis);
    g.otrail = (axis == rank - 1) ? 0 : output.stride_elems(rank - 1);
    g.idx_last = indices.stride_elems(rank - 1);
    g.upd_last = updates.stride_elems(rank - 1);
    g.red = attrs.reduction;

    int64_t dims[8];
    fill_output_dims(indices, dims);
    for (int64_t d = 0; d < 8; ++d) {
        g.odims[d] = dims[d];
        g.istep[d] = 0;
        g.ustep[d] = 0;
        g.ostep[d] = 0;
    }
    for (int64_t d = 0; d < rank; ++d) {
        g.istep[d] = indices.stride_elems(d);
        g.ustep[d] = updates.stride_elems(d);
        if (d != axis) { g.ostep[d] = output.stride_elems(d); }
    }

    const unsigned grid =
        static_cast<unsigned>(std::min<int64_t>(g.rows, int64_t(1) << 20));
    scatter_elements_kernel<T, I><<<grid, kCudaBlockSize, 0, stream>>>(
        indices.ptr<I>(), updates.ptr<T>(), output.ptr<T>(), g);
    return Status::ok;
}

// ============================================================
// ScatterND
// ============================================================

struct ScatterNDGeom {
    int64_t rows = 0;
    int64_t inner = 0;
    int64_t k = 0;
    int64_t row_rank = 0;       ///< index dims - 1
    int64_t odims[8] = {};
    int64_t tstep[8] = {};
    int64_t ustep[8] = {};
    int64_t data_stride[8] = {};
    int64_t data_dim[8] = {};
    int64_t tuple_step = 1;
    int64_t out_step = 1;
    int64_t upd_step = 1;
    ScatterReduction red = ScatterReduction::None;
};

template <class T, class I>
__global__ void scatter_nd_kernel(const I* __restrict__ idx,
                                  const T* __restrict__ updates,
                                  T* __restrict__ out, ScatterNDGeom g)
{
    for (int64_t t = blockIdx.x; t < g.rows; t += gridDim.x) {
        int64_t c[8] = {};
        decode_row(t, g.odims, g.row_rank, c);
        int64_t iofs = 0, uofs = 0;
        for (int64_t d = 0; d < g.row_rank; ++d) {
            iofs += c[d] * g.tstep[d];
            uofs += c[d] * g.ustep[d];
        }
        int64_t ds = 0;
        for (int64_t j = 0; j < g.k; ++j) {
            const int64_t v =
                scatter_index(int64_t(idx[iofs + j * g.tuple_step]), g.data_dim[j]);
            ds += v * g.data_stride[j];
        }
        for (int64_t i = int64_t(threadIdx.x); i < g.inner;
             i += int64_t(blockDim.x)) {
            scatter_apply(out + ds + i * g.out_step,
                          updates + uofs + i * g.upd_step, g.red);
        }
    }
}

template <class T, class I>
Status run_scatter_nd(TensorView& output, const TensorView& data,
                      const TensorView& indices, const TensorView& updates,
                      ScatterReduction red, cudaStream_t stream)
{
    NNOPS_TRY(copy_into<T>(data, output, stream));

    const int64_t qr = indices.rank();
    const int64_t k = indices.shape(qr - 1);
    const int64_t tuples = (k > 0) ? indices.numel() / k : 0;
    if (tuples == 0) { return Status::ok; }

    int64_t inner = 1;
    for (int64_t d = k; d < data.rank(); ++d) { inner *= data.shape(d); }

    ScatterNDGeom g;
    g.k = k;
    g.inner = inner;
    g.rows = tuples;
    g.row_rank = qr - 1;
    g.tuple_step = indices.stride_elems(qr - 1);
    g.out_step = data.stride_elems(data.rank() - 1);
    g.upd_step = updates.stride_elems(updates.rank() - 1);
    g.red = red;

    int64_t dims[8];
    fill_output_dims(indices, dims);
    for (int64_t d = 0; d < 8; ++d) {
        g.odims[d] = dims[d];
        g.tstep[d] = 0;
        g.ustep[d] = 0;
    }
    for (int64_t d = 0; d < g.row_rank; ++d) {
        g.tstep[d] = indices.stride_elems(d);
        g.ustep[d] = updates.stride_elems(d);
    }
    for (int64_t j = 0; j < k; ++j) {
        g.data_stride[j] = data.stride_elems(j);
        g.data_dim[j] = data.shape(j);
    }

    const unsigned grid =
        static_cast<unsigned>(std::min<int64_t>(g.rows, int64_t(1) << 20));
    scatter_nd_kernel<T, I><<<grid, kCudaBlockSize, 0, stream>>>(
        indices.ptr<I>(), updates.ptr<T>(), output.ptr<T>(), g);
    return Status::ok;
}

// ============================================================
// Dispatch
// ============================================================

template <class T>
bool reduction_supported(ScatterReduction red)
{
    switch (red) {
    case ScatterReduction::None: return true;
    case ScatterReduction::Add:  return scatter_add_supported<T>();
    default:                     return scatter_mul_supported<T>();
    }
}

template <class T>
Status scatter_elements_dispatch(const ScatterElementsAttributes& attrs,
                                 TensorView& output, const TensorView& data,
                                 const TensorView& indices,
                                 const TensorView& updates, cudaStream_t stream)
{
    if (indices.data_type() == DataType::s64) {
        return run_scatter_elements<T, int64_t>(attrs, output, data, indices,
                                                updates, stream);
    }
    if (indices.data_type() == DataType::s32) {
        return run_scatter_elements<T, int32_t>(attrs, output, data, indices,
                                                updates, stream);
    }
    return Status::unsupported;
}

template <class T>
Status scatter_nd_dispatch(TensorView& output, const TensorView& data,
                           const TensorView& indices, const TensorView& updates,
                           ScatterReduction red, cudaStream_t stream)
{
    if (indices.data_type() == DataType::s64) {
        return run_scatter_nd<T, int64_t>(output, data, indices, updates, red,
                                          stream);
    }
    if (indices.data_type() == DataType::s32) {
        return run_scatter_nd<T, int32_t>(output, data, indices, updates, red,
                                          stream);
    }
    return Status::unsupported;
}

#define NNOPS_SCATTER_TYPES(Macro, ...)                     \
    Macro(float, __VA_ARGS__)                               \
    Macro(__half, __VA_ARGS__)                              \
    Macro(int8_t, __VA_ARGS__)                              \
    Macro(uint8_t, __VA_ARGS__)                             \
    Macro(int32_t, __VA_ARGS__)                             \
    Macro(int64_t, __VA_ARGS__)

}  // anonymous namespace

Status scatter_elements_cuda(const ScatterElementsAttributes& attrs,
                             TensorView& output,
                             std::span<const TensorView> inputs,
                             const ComputeContext& ctx, void* /*workspace*/)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    const auto& updates = inputs[2];

#define NNOPS_TRY_ELEM(T, ...)                                             \
    if (data.data_type() == dtype_of<T>()) {                              \
        if (!reduction_supported<T>(attrs.reduction)) {                    \
            return Status::unsupported;                                    \
        }                                                                  \
        return scatter_elements_dispatch<T>(attrs, output, data, indices,  \
                                            updates, stream);          \
    }
    // (dispatch is a plain switch; the macro above is never expanded)
#undef NNOPS_TRY_ELEM

    switch (data.data_type()) {
    case DataType::f32:
        if (!reduction_supported<float>(attrs.reduction)) { return Status::unsupported; }
        return scatter_elements_dispatch<float>(attrs, output, data, indices, updates, stream);
    case DataType::f16:
        if (!reduction_supported<__half>(attrs.reduction)) { return Status::unsupported; }
        return scatter_elements_dispatch<__half>(attrs, output, data, indices, updates, stream);
    case DataType::s8:
        if (!reduction_supported<int8_t>(attrs.reduction)) { return Status::unsupported; }
        return scatter_elements_dispatch<int8_t>(attrs, output, data, indices, updates, stream);
    case DataType::u8:
        if (!reduction_supported<uint8_t>(attrs.reduction)) { return Status::unsupported; }
        return scatter_elements_dispatch<uint8_t>(attrs, output, data, indices, updates, stream);
    case DataType::s32:
        if (!reduction_supported<int32_t>(attrs.reduction)) { return Status::unsupported; }
        return scatter_elements_dispatch<int32_t>(attrs, output, data, indices, updates, stream);
    case DataType::s64:
        if (!reduction_supported<int64_t>(attrs.reduction)) { return Status::unsupported; }
        return scatter_elements_dispatch<int64_t>(attrs, output, data, indices, updates, stream);
    default:
        return Status::unsupported;
    }
}

Status scatter_nd_cuda(const ScatterNDAttributes& attrs, TensorView& output,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx, void* /*workspace*/)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    const auto& updates = inputs[2];
    switch (data.data_type()) {
    case DataType::f32:
        if (!reduction_supported<float>(attrs.reduction)) { return Status::unsupported; }
        return scatter_nd_dispatch<float>(output, data, indices, updates, attrs.reduction, stream);
    case DataType::f16:
        if (!reduction_supported<__half>(attrs.reduction)) { return Status::unsupported; }
        return scatter_nd_dispatch<__half>(output, data, indices, updates, attrs.reduction, stream);
    case DataType::s8:
        if (!reduction_supported<int8_t>(attrs.reduction)) { return Status::unsupported; }
        return scatter_nd_dispatch<int8_t>(output, data, indices, updates, attrs.reduction, stream);
    case DataType::u8:
        if (!reduction_supported<uint8_t>(attrs.reduction)) { return Status::unsupported; }
        return scatter_nd_dispatch<uint8_t>(output, data, indices, updates, attrs.reduction, stream);
    case DataType::s32:
        if (!reduction_supported<int32_t>(attrs.reduction)) { return Status::unsupported; }
        return scatter_nd_dispatch<int32_t>(output, data, indices, updates, attrs.reduction, stream);
    case DataType::s64:
        if (!reduction_supported<int64_t>(attrs.reduction)) { return Status::unsupported; }
        return scatter_nd_dispatch<int64_t>(output, data, indices, updates, attrs.reduction, stream);
    default:
        return Status::unsupported;
    }
}

}  // namespace nnops::backend::cuda
