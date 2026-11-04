/// @file scatter.cpp
/// @brief CPU ScatterElements / ScatterND — copy data, then write the updates.
///
/// The output starts as a copy of data (a logical walk, so pitch and layout are
/// carried by the strides), then every index tuple writes its element or slice.
/// The index range is checked before the walk for the same reason as Gather:
/// the parallel body cannot fail mid-stream.

#include "nnops/ops/scatter.hpp"
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

template <typename T>
inline void apply_reduction(T* dst, const T* src, ScatterReduction red)
{
    if (red == ScatterReduction::None) {
        *dst = *src;
        return;
    }
    if constexpr (std::is_same_v<T, half>) {
        const float d = simd::s_load(dst);
        const float s = simd::s_load(src);
        simd::s_store(dst, red == ScatterReduction::Add ? d + s : d * s);
    } else {
        *dst = (red == ScatterReduction::Add) ? static_cast<T>(*dst + *src)
                                              : static_cast<T>(*dst * *src);
    }
}

template <typename T>
void copy_logical(const TensorView& src, TensorView& dst,
                  const ComputeContext& ctx)
{
    int64_t dims[8], sstr[8], dstr[8];
    for (int64_t d = 0; d < src.rank(); ++d) {
        dims[d] = src.shape(d);
        sstr[d] = src.stride_elems(d);
        dstr[d] = dst.stride_elems(d);
    }
    const T* sp = src.ptr<T>();
    T* dp = dst.ptr<T>();
    walk_logical(dims, src.rank(), src.numel(), ctx, [&](const int64_t* o) {
        dp[flat_offset(o, dstr, src.rank())] =
            sp[flat_offset(o, sstr, src.rank())];
    });
}

template <typename I>
bool scatter_elements_indices_valid(const TensorView& indices, int64_t limit)
{
    const I* p = indices.ptr<I>();
    int64_t st[8];
    logical_strides(indices, st);
    int64_t dims[8];
    for (int64_t d = 0; d < indices.rank(); ++d) { dims[d] = indices.shape(d); }

    const int64_t n = indices.numel();
    int64_t idx[8] = {};
    int64_t off = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t v = static_cast<int64_t>(p[off]);
        if (v < -limit || v >= limit) { return false; }
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
bool scatter_nd_indices_valid(const TensorView& indices, const TensorView& data)
{
    const I* p = indices.ptr<I>();
    const int64_t qr = indices.rank();
    const int64_t k  = indices.shape(qr - 1);
    const int64_t tuples = indices.numel() / k;

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
            const int64_t limit = data.shape(j);
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
void scatter_elements_impl(const ScatterElementsAttributes& attrs,
                           TensorView& output, const TensorView& data,
                           const TensorView& indices, const TensorView& updates,
                           const ComputeContext& ctx)
{
    copy_logical<T>(data, output, ctx);

    const int64_t rank = data.rank();
    const int64_t axis = resolve_axis(attrs.axis, rank);

    int64_t istr[8], ustr[8], ostr[8], idims[8];
    logical_strides(indices, istr);
    logical_strides(updates, ustr);
    logical_strides(output, ostr);
    for (int64_t d = 0; d < rank; ++d) { idims[d] = indices.shape(d); }

    const I* iptr = indices.ptr<I>();
    const T* uptr = updates.ptr<T>();
    T* optr = output.ptr<T>();

    walk_logical(idims, rank, indices.numel(), ctx, [&](const int64_t* o) {
        int64_t io = 0, oo = 0;
        for (int64_t d = 0; d < rank; ++d) {
            io += o[d] * istr[d];
            if (d != axis) { oo += o[d] * ostr[d]; }
        }
        int64_t k = static_cast<int64_t>(iptr[io]);
        if (k < 0) { k += data.shape(axis); }
        oo += k * ostr[axis];

        apply_reduction(&optr[oo], &uptr[flat_offset(o, ustr, rank)],
                        attrs.reduction);
    });
}

template <typename T, typename I>
void scatter_nd_impl(const ScatterNDAttributes& attrs, TensorView& output,
                     const TensorView& data, const TensorView& indices,
                     const TensorView& updates, const ComputeContext& ctx)
{
    copy_logical<T>(data, output, ctx);

    const int64_t qr  = indices.rank();
    const int64_t k   = indices.shape(qr - 1);
    int64_t inner = 1;
    for (int64_t d = k; d < data.rank(); ++d) { inner *= data.shape(d); }
    const int64_t tuples = indices.numel() / k;

    int64_t istr[8], dstr[8], ostr[8];
    logical_strides(indices, istr);
    logical_strides(data, dstr);
    logical_strides(output, ostr);
    const int64_t last_stride = istr[qr - 1];

    // The inner slice of the data and of the updates share the same index math:
    // both have data[k:] trailing axes with the same extents.
    int64_t inner_dstr[8], inner_ustr[8];
    int64_t trailing_dims[8];
    const int64_t inner_rank = data.rank() - k;
    logical_strides(updates, inner_ustr);
    for (int64_t d = 0; d < inner_rank; ++d) {
        inner_dstr[d] = dstr[k + d];
        trailing_dims[d] = data.shape(k + d);
    }

    const I* iptr = indices.ptr<I>();
    const T* uptr = updates.ptr<T>();
    T* optr = output.ptr<T>();

    int64_t dims[8];
    for (int64_t d = 0; d < qr - 1; ++d) { dims[d] = indices.shape(d); }
    const int64_t tuple_rank = qr - 1;

    int64_t idx[8] = {};
    int64_t base = 0;
    for (int64_t t = 0; t < tuples; ++t) {
        int64_t ds = 0;
        for (int64_t j = 0; j < k; ++j) {
            int64_t v = static_cast<int64_t>(iptr[base + j * last_stride]);
            if (v < 0) { v += data.shape(j); }
            ds += v * dstr[j];
        }
        const int64_t us = flat_offset(idx, inner_ustr, tuple_rank);

        int64_t inner_idx[8] = {};
        int64_t io = 0, uo = 0;
        for (int64_t m = 0; m < inner; ++m) {
            apply_reduction(&optr[ds + io], &uptr[us + uo], attrs.reduction);
            for (int64_t d = inner_rank - 1; d >= 0; --d) {
                ++inner_idx[d];
                io += inner_dstr[d];
                uo += inner_ustr[tuple_rank + d];
                if (inner_idx[d] < trailing_dims[d]) { break; }
                inner_idx[d] = 0;
                io -= inner_dstr[d] * trailing_dims[d];
                uo -= inner_ustr[tuple_rank + d] * trailing_dims[d];
            }
        }

        for (int64_t d = tuple_rank - 1; d >= 0; --d) {
            ++idx[d];
            base += istr[d];
            if (idx[d] < dims[d]) { break; }
            idx[d] = 0;
            base -= istr[d] * dims[d];
        }
    }
}

}  // anonymous namespace

Status scatter_elements_cpu(const ScatterElementsAttributes& attrs,
                            TensorView& output,
                            std::span<const TensorView> inputs,
                            const ComputeContext& ctx,
                            void* /*workspace*/)
{
    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    const auto& updates = inputs[2];
    const int64_t axis = resolve_axis(attrs.axis, data.rank());
    const int64_t limit = data.shape(axis);

    if (indices.data_type() == DataType::s64) {
        if (!scatter_elements_indices_valid<int64_t>(indices, limit)) {
            return Status::invalid_argument;
        }
        for_each_data_type(data.data_type(), [&](auto tag) {
            using T = typename decltype(tag)::type;
            scatter_elements_impl<T, int64_t>(attrs, output, data, indices,
                                              updates, ctx);
        });
    } else {
        if (!scatter_elements_indices_valid<int32_t>(indices, limit)) {
            return Status::invalid_argument;
        }
        for_each_data_type(data.data_type(), [&](auto tag) {
            using T = typename decltype(tag)::type;
            scatter_elements_impl<T, int32_t>(attrs, output, data, indices,
                                              updates, ctx);
        });
    }
    return Status::ok;
}

Status scatter_nd_cpu(const ScatterNDAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* /*workspace*/)
{
    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    const auto& updates = inputs[2];

    if (indices.data_type() == DataType::s64) {
        if (!scatter_nd_indices_valid<int64_t>(indices, data)) {
            return Status::invalid_argument;
        }
        for_each_data_type(data.data_type(), [&](auto tag) {
            using T = typename decltype(tag)::type;
            scatter_nd_impl<T, int64_t>(attrs, output, data, indices, updates, ctx);
        });
    } else {
        if (!scatter_nd_indices_valid<int32_t>(indices, data)) {
            return Status::invalid_argument;
        }
        for_each_data_type(data.data_type(), [&](auto tag) {
            using T = typename decltype(tag)::type;
            scatter_nd_impl<T, int32_t>(attrs, output, data, indices, updates, ctx);
        });
    }
    return Status::ok;
}

}  // namespace nnops::backend::cpu
