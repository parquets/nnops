/// @file gather.cpp
/// @brief CPU Gather / GatherElements / GatherND — one address per element.
///
/// All three are the same shape of loop: walk the output in logical order, map
/// the tuple to a source offset, copy the element. The element copy is a plain
/// assignment, so f16 bits and 64-bit integers never round-trip through float.
///
/// Negative indices wrap from the end of the gathered axis. The range is checked
/// in a pre-pass (the walk itself has no way to fail mid-stream), so a bad index
/// reports invalid_argument instead of reading out of bounds.

#include "nnops/ops/gather.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "common/strided_walk.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <cstdint>
#include <type_traits>
#include <utility>

namespace nnops::backend::cpu {

namespace {

template <class F>
void for_each_data_type(DataType t, F&& f)
{
    switch (t) {
    case DataType::f32: f(std::type_identity<float>{}); break;
    case DataType::f16: f(std::type_identity<half>{}); break;
    case DataType::s8:  f(std::type_identity<int8_t>{}); break;
    case DataType::u8:  f(std::type_identity<uint8_t>{}); break;
    case DataType::s32: f(std::type_identity<int32_t>{}); break;
    case DataType::s64: f(std::type_identity<int64_t>{}); break;
    default: break;
    }
}

template <typename I>
bool index_in_range(const TensorView& indices, int64_t limit)
{
    const I* p = indices.ptr<I>();
    int64_t st[8];
    logical_strides(indices, st);
    int64_t dims[8];
    for (int64_t d = 0; d < indices.rank(); ++d) {
        dims[d] = indices.shape(d);
    }
    const int64_t n = indices.numel();
    int64_t idx[8] = {};
    int64_t off = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t k = static_cast<int64_t>(p[off]);
        if (k < -limit || k >= limit) { return false; }
        for (int64_t d = indices.rank() - 1; d >= 0; --d) {
            ++idx[d];
            off += st[d];
            if (idx[d] < dims[d]) { break; }
            idx[d] = 0;
            off -= st[d] * dims[d];
        }
    }
    return true;
}

template <typename I>
bool gather_nd_indices_valid(const TensorView& indices, const TensorView& data,
                             int64_t batch, int64_t k)
{
    const I* p = indices.ptr<I>();
    const int64_t qr = indices.rank();
    const int64_t tuples = (qr >= 1)
        ? indices.numel() / indices.shape(qr - 1) : 0;

    int64_t st[8];
    logical_strides(indices, st);
    const int64_t last_stride = st[qr - 1];

    int64_t dims[8];
    for (int64_t d = 0; d < qr - 1; ++d) { dims[d] = indices.shape(d); }
    int64_t idx[8] = {};
    int64_t base = 0;
    for (int64_t t = 0; t < tuples; ++t) {
        for (int64_t j = 0; j < k; ++j) {
            const int64_t v = static_cast<int64_t>(p[base + j * last_stride]);
            const int64_t limit = data.shape(batch + j);
            if (v < -limit || v >= limit) { return false; }
        }
        for (int64_t d = qr - 2; d >= 0; --d) {
            ++idx[d];
            base += st[d];
            if (idx[d] < dims[d]) { break; }
            idx[d] = 0;
            base -= st[d] * dims[d];
        }
    }
    return true;
}

template <typename T, typename I>
void gather_impl(const GatherAttributes& attrs, TensorView& output,
                 const TensorView& data, const TensorView& indices,
                 const ComputeContext& ctx)
{
    const int64_t rank = data.rank();
    const int64_t axis = resolve_axis(attrs.axis, rank);
    const int64_t ir   = indices.rank();
    const int64_t orank = output.rank();

    int64_t dstr[8], istr[8], ostr[8], odims[8];
    logical_strides(data, dstr);
    logical_strides(indices, istr);
    logical_strides(output, ostr);
    for (int64_t d = 0; d < orank; ++d) { odims[d] = output.shape(d); }

    const T* dptr = data.ptr<T>();
    const I* iptr = indices.ptr<I>();
    T* optr = output.ptr<T>();

    walk_logical(odims, orank, output.numel(), ctx, [&](const int64_t* o) {
        int64_t ds = 0;
        for (int64_t d = 0; d < axis; ++d) { ds += o[d] * dstr[d]; }

        int64_t io = 0;
        for (int64_t j = 0; j < ir; ++j) { io += o[axis + j] * istr[j]; }

        int64_t k = static_cast<int64_t>(iptr[io]);
        if (k < 0) { k += data.shape(axis); }
        ds += k * dstr[axis];

        for (int64_t m = 0; m < rank - axis - 1; ++m) {
            ds += o[axis + ir + m] * dstr[axis + 1 + m];
        }

        optr[flat_offset(o, ostr, orank)] = dptr[ds];
    });
}

template <typename T, typename I>
void gather_elements_impl(const GatherElementsAttributes& attrs,
                          TensorView& output, const TensorView& data,
                          const TensorView& indices, const ComputeContext& ctx)
{
    const int64_t rank = data.rank();
    const int64_t axis = resolve_axis(attrs.axis, rank);

    int64_t dstr[8], istr[8], ostr[8], odims[8];
    logical_strides(data, dstr);
    logical_strides(indices, istr);
    logical_strides(output, ostr);
    for (int64_t d = 0; d < rank; ++d) { odims[d] = output.shape(d); }

    const T* dptr = data.ptr<T>();
    const I* iptr = indices.ptr<I>();
    T* optr = output.ptr<T>();

    walk_logical(odims, rank, output.numel(), ctx, [&](const int64_t* o) {
        int64_t io = 0, ds = 0, os = 0;
        for (int64_t d = 0; d < rank; ++d) {
            io += o[d] * istr[d];
            os += o[d] * ostr[d];
            if (d != axis) { ds += o[d] * dstr[d]; }
        }
        int64_t k = static_cast<int64_t>(iptr[io]);
        if (k < 0) { k += data.shape(axis); }
        ds += k * dstr[axis];
        optr[os] = dptr[ds];
    });
}

template <typename T, typename I>
void gather_nd_impl(const GatherNDAttributes& attrs, TensorView& output,
                    const TensorView& data, const TensorView& indices,
                    const ComputeContext& ctx)
{
    const int64_t qr    = indices.rank();
    const int64_t k     = indices.shape(qr - 1);
    const int64_t batch = attrs.batch_dims;
    const int64_t orank = output.rank();

    int64_t dstr[8], istr[8], ostr[8], odims[8];
    logical_strides(data, dstr);
    logical_strides(indices, istr);
    logical_strides(output, ostr);
    for (int64_t d = 0; d < orank; ++d) { odims[d] = output.shape(d); }
    const int64_t last_stride = istr[qr - 1];

    const T* dptr = data.ptr<T>();
    const I* iptr = indices.ptr<I>();
    T* optr = output.ptr<T>();

    walk_logical(odims, orank, output.numel(), ctx, [&](const int64_t* o) {
        int64_t ds = 0;
        for (int64_t d = 0; d < batch; ++d) { ds += o[d] * dstr[d]; }

        int64_t io = 0;
        for (int64_t d = 0; d < qr - 1; ++d) { io += o[d] * istr[d]; }

        for (int64_t j = 0; j < k; ++j) {
            int64_t v = static_cast<int64_t>(iptr[io + j * last_stride]);
            const int64_t limit = data.shape(batch + j);
            if (v < 0) { v += limit; }
            ds += v * dstr[batch + j];
        }

        for (int64_t m = 0; m < data.rank() - batch - k; ++m) {
            ds += o[qr - 1 + m] * dstr[batch + k + m];
        }

        optr[flat_offset(o, ostr, orank)] = dptr[ds];
    });
}

}  // anonymous namespace

Status gather_cpu(const GatherAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx,
                  void* /*workspace*/)
{
    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    const int64_t axis = resolve_axis(attrs.axis, data.rank());
    if (indices.data_type() == DataType::s64) {
        if (!index_in_range<int64_t>(indices, data.shape(axis))) {
            return Status::invalid_argument;
        }
    } else {
        if (!index_in_range<int32_t>(indices, data.shape(axis))) {
            return Status::invalid_argument;
        }
    }

    if (indices.data_type() == DataType::s64) {
        for_each_data_type(data.data_type(), [&](auto tag) {
            using T = typename decltype(tag)::type;
            gather_impl<T, int64_t>(attrs, output, data, indices, ctx);
        });
    } else {
        for_each_data_type(data.data_type(), [&](auto tag) {
            using T = typename decltype(tag)::type;
            gather_impl<T, int32_t>(attrs, output, data, indices, ctx);
        });
    }
    return Status::ok;
}

Status gather_elements_cpu(const GatherElementsAttributes& attrs,
                           TensorView& output,
                           std::span<const TensorView> inputs,
                           const ComputeContext& ctx,
                           void* /*workspace*/)
{
    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    const int64_t axis = resolve_axis(attrs.axis, data.rank());
    if (indices.data_type() == DataType::s64) {
        if (!index_in_range<int64_t>(indices, data.shape(axis))) {
            return Status::invalid_argument;
        }
        for_each_data_type(data.data_type(), [&](auto tag) {
            using T = typename decltype(tag)::type;
            gather_elements_impl<T, int64_t>(attrs, output, data, indices, ctx);
        });
    } else {
        if (!index_in_range<int32_t>(indices, data.shape(axis))) {
            return Status::invalid_argument;
        }
        for_each_data_type(data.data_type(), [&](auto tag) {
            using T = typename decltype(tag)::type;
            gather_elements_impl<T, int32_t>(attrs, output, data, indices, ctx);
        });
    }
    return Status::ok;
}

Status gather_nd_cpu(const GatherNDAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* /*workspace*/)
{
    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    const int64_t k = indices.shape(indices.rank() - 1);
    if (indices.data_type() == DataType::s64) {
        if (!gather_nd_indices_valid<int64_t>(indices, data, attrs.batch_dims, k)) {
            return Status::invalid_argument;
        }
        for_each_data_type(data.data_type(), [&](auto tag) {
            using T = typename decltype(tag)::type;
            gather_nd_impl<T, int64_t>(attrs, output, data, indices, ctx);
        });
    } else {
        if (!gather_nd_indices_valid<int32_t>(indices, data, attrs.batch_dims, k)) {
            return Status::invalid_argument;
        }
        for_each_data_type(data.data_type(), [&](auto tag) {
            using T = typename decltype(tag)::type;
            gather_nd_impl<T, int32_t>(attrs, output, data, indices, ctx);
        });
    }
    return Status::ok;
}

}  // namespace nnops::backend::cpu
