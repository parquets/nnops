#pragma once
/// @file tensor_view.hpp
/// @brief TensorView — a non-owning view of tensor data with shape, pitch, layout, and type metadata.
///
/// TensorView does not own the data buffer. The user is responsible for memory
/// management. Shape is stored inline (no heap allocation for rank ≤ 8).
///
/// Pitch model (like OpenCV cv::Mat::step[0]):
///   pitch is the byte distance between consecutive rows, where a "row" is the
///   innermost dimension. For NCHW, pitch = byte stride from (h, w) to (h+1, w).
///   Elements within a row are always contiguous (byte stride = elem_size).
///   Higher dimensions (C, N, D, etc.) are densely packed — their stride is
///   the product of the inner dimension size and its byte stride.
///
/// Examples (assuming 32-byte alignment for row padding):
///   NCHW uint8,  W=70: pitch = 72  (70 rounded up to next aligned value)
///   NCHWC8 uint8, W=7:  pitch = 72  (7*8=56, aligned → 72)

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
    /// The row pitch is computed as last_dim * elem_size (no alignment padding).
    TensorView(std::span<const int64_t> shape, DataType dtype,
               void* data, TensorLayout layout = TensorLayout::NCHW)
        : dtype_(dtype), layout_(layout), data_(data)
    {
        NNOPS_ASSERT(shape.size() <= kMaxRank);
        rank_ = static_cast<int64_t>(shape.size());
        shape_ = shape;
        if (rank_ >= 2) {
            // pitch = innermost dim * elem_size (bytes between consecutive rows)
            pitch_ = shape_[static_cast<size_t>(rank_ - 1)]
                   * static_cast<int64_t>(data_type_size(dtype));
        } else if (rank_ == 1) {
            // 1D tensor: pitch = elem_size (single-element rows)
            pitch_ = static_cast<int64_t>(data_type_size(dtype));
        } else {
            pitch_ = 0;
        }
    }

    /// Construct with explicit row pitch (in bytes).
    /// For aligned/padded data where the row stride is larger than
    /// last_dim * elem_size.
    TensorView(std::span<const int64_t> shape, DataType dtype,
               void* data, int64_t pitch,
               TensorLayout layout = TensorLayout::NCHW)
        : dtype_(dtype), layout_(layout), data_(data), pitch_(pitch)
    {
        NNOPS_ASSERT(shape.size() <= kMaxRank);
        rank_ = static_cast<int64_t>(shape.size());
        shape_ = shape;
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

    /// Shape of a specific dimension.
    int64_t shape(int64_t dim) const noexcept {
        NNOPS_ASSERT(dim >= 0 && dim < rank_);
        return shape_[static_cast<size_t>(dim)];
    }

    /// Row pitch in bytes.
    /// The byte distance from one row to the next, where a "row" is the
    /// innermost dimension. For rank < 2, pitch equals elem_size.
    int64_t pitch() const noexcept { return pitch_; }

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

    /// Total size in bytes of the logical tensor (NOT including pitch padding).
    /// This is numel() * elem_size. The actual buffer may be larger due to
    /// pitch padding at the end of each row.
    size_t nbytes() const noexcept {
        return static_cast<size_t>(numel()) * data_type_size(dtype_);
    }

    /// Whether this is an empty view.
    bool is_empty() const noexcept { return data_ == nullptr || rank_ == 0; }

    // ---- Derived stride helpers (element counts, not bytes) ----

    /// Compute the element stride for a given dimension.
    /// Stride is the number of elements to skip to advance by 1 in that dim.
    /// For rank-1 (innermost): always 1 (contiguous within a row).
    /// For rank-2: pitch / elem_size (may be > shape[rank-1] if padded).
    /// For higher dims: shape[dim+1] * stride(dim+1).
    int64_t stride_elems(int64_t dim) const noexcept {
        NNOPS_ASSERT(dim >= 0 && dim < rank_);
        // Innermost dimension: always contiguous (stride = 1 element)
        if (dim == rank_ - 1) return 1;
        // Second-innermost: stride comes from pitch (may be > shape[rank-1] if padded)
        const int64_t elem_size = static_cast<int64_t>(data_type_size(dtype_));
        int64_t s = pitch_ / elem_size;
        if (dim == rank_ - 2) return s;
        // Outer dimensions: accumulate via shape[i+1] * stride(i+1)
        for (int64_t i = rank_ - 3; i >= dim; --i) {
            s = shape_[static_cast<size_t>(i + 1)] * s;
        }
        return s;
    }

    /// Convenience: number of elements per row (stride of the 2nd-innermost dim).
    /// Equivalent to stride_elems(rank() - 2) for rank >= 2, or 1 for rank < 2.
    int64_t row_stride_elems() const noexcept {
        if (rank_ < 2) return 1;
        return pitch_ / static_cast<int64_t>(data_type_size(dtype_));
    }

private:
    detail::SmallVector<int64_t, kMaxRank> shape_;
    void* data_ = nullptr;
    int64_t pitch_ = 0;  // row pitch in bytes
    DataType dtype_ = DataType::f32;
    TensorLayout layout_ = TensorLayout::NCHW;
    int64_t rank_ = 0;
};

}  // namespace nnops
