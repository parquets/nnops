#pragma once
/// @file tensor_view.hpp
/// @brief TensorView — a non-owning view of tensor data with shape, stride, layout, and type metadata.
///
/// TensorView does not own the data buffer. The user is responsible for memory
/// management. Shape and stride are stored inline (no heap allocation for rank ≤ 8).

#include "nnops/core/data_type.hpp"
#include "nnops/core/tensor_layout.hpp"
#include "nnops/detail/small_vector.hpp"
#include "nnops/detail/assert.hpp"

#include <cstdint>
#include <cstddef>
#include <span>

namespace nnops {

class TensorView {
public:
    static constexpr int64_t kMaxRank = 8;

    // ---- Construction ----

    /// Default: empty view.
    TensorView() = default;

    /// Construct a dense (contiguous) TensorView.
    /// Strides are computed automatically in row-major / NCHW order.
    TensorView(std::span<const int64_t> shape, DataType dtype,
               void* data, TensorLayout layout = TensorLayout::NCHW)
        : dtype_(dtype), layout_(layout), data_(data)
    {
        NNOPS_ASSERT(shape.size() <= kMaxRank);
        rank_ = static_cast<int64_t>(shape.size());
        shape_ = shape;
        stride_.resize(shape.size());
        compute_dense_strides();
    }

    /// Construct with explicit strides.
    TensorView(std::span<const int64_t> shape,
               std::span<const int64_t> strides,
               DataType dtype,
               void* data,
               TensorLayout layout = TensorLayout::NCHW)
        : dtype_(dtype), layout_(layout), data_(data)
    {
        NNOPS_ASSERT(shape.size() <= kMaxRank);
        NNOPS_ASSERT(shape.size() == strides.size());
        rank_ = static_cast<int64_t>(shape.size());
        shape_ = shape;
        stride_ = strides;
    }

    // ---- Accessors ----

    /// Number of dimensions.
    int64_t rank() const noexcept { return rank_; }

    /// Pointer to shape array (length = rank()).
    const int64_t* shape() const noexcept { return shape_.data(); }

    /// Shape as a span.
    std::span<const int64_t> shape_span() const noexcept {
        return {shape_.data(), static_cast<size_t>(rank_)};
    }

    /// Pointer to stride array (length = rank()).
    const int64_t* stride() const noexcept { return stride_.data(); }

    /// Shape of a specific dimension.
    int64_t shape(int64_t dim) const noexcept {
        NNOPS_ASSERT(dim >= 0 && dim < rank_);
        return shape_[static_cast<size_t>(dim)];
    }

    /// Stride of a specific dimension.
    int64_t stride(int64_t dim) const noexcept {
        NNOPS_ASSERT(dim >= 0 && dim < rank_);
        return stride_[static_cast<size_t>(dim)];
    }

    /// Element data type.
    DataType data_type() const noexcept { return dtype_; }

    /// Memory layout.
    TensorLayout layout() const noexcept { return layout_; }

    /// Raw data pointer (mutable).
    void* data() noexcept { return data_; }

    /// Raw data pointer (const).
    const void* data() const noexcept { return data_; }

    /// Typed data pointer (mutable).
    template <typename T>
    T* data_as() noexcept { return static_cast<T*>(data_); }

    /// Typed data pointer (mutable, even when TensorView is const).
    template <typename T>
    T* data_as() const noexcept { return static_cast<T*>(data_); }

    /// Total number of elements (product of all dimensions).
    int64_t numel() const noexcept {
        if (rank_ == 0) return 0;
        int64_t n = 1;
        for (int64_t i = 0; i < rank_; ++i) {
            n *= shape_[static_cast<size_t>(i)];
        }
        return n;
    }

    /// Total size in bytes.
    size_t nbytes() const noexcept {
        return static_cast<size_t>(numel()) * data_type_size(dtype_);
    }

    /// Compute the flat byte offset for a given linear index.
    /// Only works correctly for dense or explicitly-strided tensors.
    int64_t offset(int64_t flat_idx) const noexcept {
        int64_t off = 0;
        int64_t remainder = flat_idx;
        for (int64_t i = rank_ - 1; i >= 0; --i) {
            int64_t dim = shape_[static_cast<size_t>(i)];
            off += (remainder % dim) * stride_[static_cast<size_t>(i)];
            remainder /= dim;
        }
        return off;
    }

    /// Whether the tensor is densely packed (row-major contiguous).
    bool is_contiguous() const noexcept {
        int64_t expected_stride = 1;
        for (int64_t i = rank_ - 1; i >= 0; --i) {
            if (stride_[static_cast<size_t>(i)] != expected_stride) {
                return false;
            }
            expected_stride *= shape_[static_cast<size_t>(i)];
        }
        return true;
    }

    /// Whether this is an empty view.
    bool is_empty() const noexcept { return data_ == nullptr || rank_ == 0; }

private:
    void compute_dense_strides() noexcept {
        int64_t st = 1;
        for (int64_t i = rank_ - 1; i >= 0; --i) {
            stride_[static_cast<size_t>(i)] = st;
            st *= shape_[static_cast<size_t>(i)];
        }
    }

    detail::SmallVector<int64_t, kMaxRank> shape_;
    detail::SmallVector<int64_t, kMaxRank> stride_;
    void* data_ = nullptr;
    DataType dtype_ = DataType::f32;
    TensorLayout layout_ = TensorLayout::NCHW;
    int64_t rank_ = 0;
};

}  // namespace nnops
