#pragma once
/// @file tensor_view.hpp
/// @brief TensorView — a non-owning view of tensor data with shape, pitch, layout, and type metadata.
///
/// TensorView holds no memory; the caller owns the buffer. Shape is inline
/// (SmallVector, kMaxRank = 8), so construction never allocates.
///
/// Pitch model (like OpenCV cv::Mat::step[0]): pitch is the byte distance
/// between consecutive rows, a row being the innermost dimension; outer
/// dimensions are dense. Departures from "a row is elements × elem_size":
///  - block-packed dtypes (q4_1) have no per-element size — data_type_size()
///    is 0 and strides step in 20-byte blocks — so every row/stride computation
///    goes through data_type_row_bytes() / data_type_stride_unit();
///  - channel-last keeps C inside the row, so a row is one
///    spatial line: W pixels × C channels.
///
/// TensorDesc::row_pitch() aligns rows to 32 bytes by default; a TensorView
/// built with an explicit pitch accepts any value >= the minimum.
///   Channel-first uint8,  W=70:  pitch = 96  (70 rounded up to a multiple of 32)
///   Channel-last uint8, W*C=56: pitch = 64  (56 rounded up to a multiple of 32)

#include "nnops/core/data_type.hpp"
#include "nnops/core/tensor_layout.hpp"
#include "nnops/core/quant_params.hpp"
#include "nnops/detail/small_vector.hpp"
#include "nnops/detail/assert.hpp"

#include <cstdint>
#include <cstddef>
#include <span>

namespace nnops {

/// Lightweight tensor metadata (no data pointer, no ownership): the descriptor
/// getWorkspaceSize() receives, analogous to TensorRT's PluginTensorDesc.
struct TensorDesc {
    static constexpr int64_t kMaxRank = 8;

    detail::SmallVector<int64_t, kMaxRank> dims;
    int64_t rank = 0;
    DataType dtype = DataType::f32;
    TensorLayout layout = TensorLayout::NCHW;

    /// Row stride in units of data_type_stride_unit(); 0 = unknown, which
    /// consumers must treat as compact.
    int64_t row_stride_elems = 0;

    /// Logical element count (product of dims).
    int64_t numel() const noexcept {
        if (rank == 0) {
            return 1; // scalar = 1 element
        }
        int64_t n = 1;
        for (int64_t i = 0; i < rank; ++i) {
            n *= dims[static_cast<size_t>(i)];
        }
        return n;
    }

    /// Bytes for numel(), rounding the innermost dim up to whole blocks
    /// (q4_1, s4/u4); excludes per-row pitch padding — see storage_bytes().
    size_t nbytes() const noexcept {
        if (data_type_size(dtype) == 0) {
            if (rank <= 0) {
                return data_type_row_bytes(dtype, 1);
            }
            const int64_t cols = dims[static_cast<size_t>(rank - 1)];
            const int64_t rows = (cols > 0) ? numel() / cols : 0;
            return static_cast<size_t>(rows) * data_type_row_bytes(dtype, cols);
        }
        return static_cast<size_t>(numel()) * data_type_size(dtype);
    }

    /// Row pitch in bytes, aligned up to `alignment`.
    /// For rank < 2 this is the storage unit (element or block size).
    int64_t row_pitch(int64_t alignment = 32) const noexcept {
        if (rank < 2) {
            return static_cast<int64_t>(data_type_stride_unit(dtype));
        }
        const int64_t row_elems = is_channel_last(layout)
            ? dims[1] * dims[static_cast<size_t>(rank - 1)]
            : dims[static_cast<size_t>(rank - 1)];
        const int64_t row_bytes =
            static_cast<int64_t>(data_type_row_bytes(dtype, row_elems));
        return ((row_bytes + alignment - 1) / alignment) * alignment;
    }

    /// Actual storage size including default-aligned row pitch (32B).
    size_t storage_bytes(int64_t alignment = 32) const noexcept {
        if (rank < 2) {
            return nbytes();
        }
        const int64_t aligned_row = row_pitch(alignment);
        int64_t rows = dims[0];
        // dims[1] merges into the row pitch only where it is a channel axis of
        // its own; at rank 2 it *is* the last dim, which row_pitch already
        // covers, so counting it here sizes the tensor by another factor of W.
        if (!is_channel_last(layout) && rank > 2) {
            rows *= dims[1];
        }
        for (int64_t d = 2; d < rank - 1; ++d) {
            rows *= dims[static_cast<size_t>(d)];
        }
        return static_cast<size_t>(rows * aligned_row);
    }
};

class TensorView {
public:
    static constexpr int64_t kMaxRank = 8;

    // ---- Construction ----

    TensorView() = default;

    /// Dense (contiguous) view: pitch is one row's bytes, no padding.
    TensorView(std::span<const int64_t> shape, DataType dtype,
               void* data, TensorLayout layout = TensorLayout::NCHW,
               QuantParams qp = {})
        : dtype_(dtype), layout_(layout), data_(data), quant_params_(qp)
    {
        NNOPS_ASSERT(shape.size() <= kMaxRank);
        rank_ = static_cast<int64_t>(shape.size());
        shape_ = shape;
        if (rank_ >= 2) {
            const int64_t row_elems = is_channel_last(layout)
                ? shape_[1] * shape_[static_cast<size_t>(rank_ - 1)]
                : shape_[static_cast<size_t>(rank_ - 1)];
            pitch_ = static_cast<int64_t>(data_type_row_bytes(dtype, row_elems));
        } else if (rank_ == 1) {
            pitch_ = static_cast<int64_t>(data_type_stride_unit(dtype));
        } else {
            pitch_ = 0;
        }
    }

    /// View over padded/aligned data: `pitch` (bytes) may exceed the dense row size.
    TensorView(std::span<const int64_t> shape, DataType dtype,
               void* data, int64_t pitch,
               TensorLayout layout = TensorLayout::NCHW,
               QuantParams qp = {})
        : dtype_(dtype), layout_(layout), data_(data), pitch_(pitch), quant_params_(qp)
    {
        NNOPS_ASSERT(shape.size() <= kMaxRank);
        rank_ = static_cast<int64_t>(shape.size());
        shape_ = shape;
        if (rank_ >= 2) {
            const int64_t row_elems = is_channel_last(layout)
                ? shape_[1] * shape_[static_cast<size_t>(rank_ - 1)]
                : shape_[static_cast<size_t>(rank_ - 1)];
            const int64_t min_pitch =
                static_cast<int64_t>(data_type_row_bytes(dtype, row_elems));
            NNOPS_ASSERT(pitch_ >= min_pitch);
        }
    }

    // ---- Accessors ----

    int64_t rank() const noexcept { return rank_; }

    const int64_t* shape() const noexcept { return shape_.data(); }

    std::span<const int64_t> shape_span() const noexcept {
        return {shape_.data(), static_cast<size_t>(rank_)};
    }

    int64_t shape(int64_t dim) const noexcept {
        NNOPS_ASSERT(dim >= 0 && dim < rank_);
        return shape_[static_cast<size_t>(dim)];
    }

    /// Row pitch in bytes.
    int64_t pitch() const noexcept { return pitch_; }

    DataType data_type() const noexcept { return dtype_; }

    TensorLayout layout() const noexcept { return layout_; }

    const QuantParams& quant_params() const noexcept { return quant_params_; }

    /// s8/u8/s4/u4 with active scale/zp; f8 and q4_1 carry no QuantParams.
    bool is_quantized() const noexcept {
        return is_quantized_dtype(dtype_) && quant_params_.is_active();
    }

    template <typename T>
    T* ptr() noexcept { return static_cast<T*>(data_); }

    template <typename T>
    const T* ptr() const noexcept { return static_cast<const T*>(data_); }

    /// Pointer to the first element of row `row_id` (offset = row_id * pitch()).
    template <typename T>
    T* ptr(size_t row_id) noexcept {
        return reinterpret_cast<T*>(static_cast<char*>(data_) + row_id * static_cast<size_t>(pitch_));
    }

    template <typename T>
    const T* ptr(size_t row_id) const noexcept {
        return reinterpret_cast<const T*>(static_cast<const char*>(data_) + row_id * static_cast<size_t>(pitch_));
    }

    /// Logical element count (product of dims); storage_bytes() adds padding.
    int64_t numel() const noexcept {
        if (rank_ == 0) { return data_ != nullptr ? 1 : 0; }  // scalar=1, empty=0
        int64_t n = 1;
        for (int64_t i = 0; i < rank_; ++i) {
            n *= shape_[static_cast<size_t>(i)];
        }
        return n;
    }

    /// Buffer size in bytes including row pitch padding.
    size_t nbytes() const noexcept {
        return storage_bytes();
    }

    /// total_rows() × pitch; for rank < 2, one row's bytes.
    size_t storage_bytes() const noexcept {
        if (rank_ < 2) {
            return data_type_row_bytes(dtype_, numel());
        }
        return static_cast<size_t>(total_rows()) * static_cast<size_t>(pitch_);
    }

    bool is_empty() const noexcept { return data_ == nullptr; }

    /// Lightweight TensorDesc (no data pointer) for workspace queries.
    TensorDesc desc() const noexcept {
        TensorDesc d;
        d.rank = rank_;
        d.dims = shape_;
        d.dtype = dtype_;
        d.layout = layout_;
        if (rank_ >= 2) {
            const int64_t unit = static_cast<int64_t>(data_type_stride_unit(dtype_));
            d.row_stride_elems = (unit > 0) ? pitch_ / unit : 0;
        }
        return d;
    }

    // ---- Derived stride helpers (element counts, not bytes) ----

    /// Element stride for one step in `dim` (element counts, not bytes).
    /// The innermost dim steps one element; the second-innermost comes from
    /// pitch, so it may exceed shape[rank-1].
    int64_t stride_elems(int64_t dim) const noexcept {
        NNOPS_ASSERT(dim >= 0 && dim < rank_);
        // Channel-last stores [N][spatial…][C]: C (dim 1) is the innermost
        // axis, a spatial step moves a whole C block. A step in H is one row
        // pitch, so a pitched view's outer strides account for the padding.
        if (is_channel_last(layout_)) {
            if (dim == 1) { return 1; }
            if (dim == rank_ - 1) { return shape_[1]; }
            int64_t s = row_stride_elems();
            for (int64_t i = rank_ - 2; i > dim; --i) {
                // C is already inside the row; multiplying by it again would
                // size a batch step by another factor of C.
                if (i != 1) { s *= shape_[static_cast<size_t>(i)]; }
            }
            return s;
        }
        if (dim == rank_ - 1) {
            return 1;
        }
        const int64_t unit = static_cast<int64_t>(data_type_stride_unit(dtype_));
        if (unit <= 0) { return 0; }  // no element/block storage unit (s4/u4)
        int64_t s = pitch_ / unit;
        if (dim == rank_ - 2) { return s; }
        for (int64_t i = rank_ - 3; i >= dim; --i) {
            s = shape_[static_cast<size_t>(i + 1)] * s;
        }
        return s;
    }

    /// Elements per row: pitch / data_type_stride_unit(); 1 for rank < 2.
    int64_t row_stride_elems() const noexcept {
        if (rank_ < 2) { return 1; }
        const int64_t unit = static_cast<int64_t>(data_type_stride_unit(dtype_));
        return (unit > 0) ? pitch_ / unit : 0;
    }

    // ---- Layout-aware accessors ----

    /// Channels per contiguously-stored block: channel-first keeps each channel as
    /// its own block (C), channel-last packs all of C into every pixel (1
    /// channel block per spatial position).
    int64_t num_channel_blocks() const noexcept {
        if (rank_ < 2) {
            return 1;
        }
        return is_channel_last(layout_) ? 1 : shape_[1];
    }

    /// Element stride between channel blocks: row_stride_elems() × inner
    /// spatial dims.
    int64_t channel_block_stride_elems() const noexcept {
        if (rank_ < 3) {
            return row_stride_elems();
        }
        int64_t s = row_stride_elems();
        for (int64_t d = 2; d < rank_ - 1; ++d) {
            s *= shape_[static_cast<size_t>(d)];
        }
        return s;
    }

    /// Flat row count: N × channel blocks × inner spatial dims (channel-last
    /// has no channel axis in the count).
    int64_t total_rows() const noexcept {
        if (rank_ < 1) {
            return 0;
        }
        if (is_channel_last(layout_)) {
            int64_t rows = shape_[0];
            for (int64_t d = 2; d < rank_ - 1; ++d) {
                rows *= shape_[static_cast<size_t>(d)];
            }
            return rows;
        }
        int64_t rows = shape_[0];  // N
        // Channel blocks only apply with spatial dims (rank >= 3): a rank-2
        // [N, C] has C as the last dim, so a row is the whole C block.
        if (rank_ >= 3) {
            rows *= num_channel_blocks();
        }
        for (int64_t d = 2; d < rank_ - 1; ++d) {
            rows *= shape_[static_cast<size_t>(d)];
        }
        return rows;
    }

private:
    detail::SmallVector<int64_t, kMaxRank> shape_;
    void* data_ = nullptr;
    int64_t pitch_ = 0;  // row pitch in bytes
    DataType dtype_ = DataType::f32;
    TensorLayout layout_ = TensorLayout::NCHW;
    QuantParams quant_params_;  // scale / zero_point / granularity
    int64_t rank_ = 0;
};

}  // namespace nnops
