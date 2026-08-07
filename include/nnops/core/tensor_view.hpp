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
#include "nnops/core/quant_params.hpp"
#include "nnops/detail/small_vector.hpp"
#include "nnops/detail/assert.hpp"

#include <cstdint>
#include <cstddef>
#include <span>

namespace nnops {

/// Lightweight tensor metadata descriptor (no data pointer, no ownership).
/// Analogous to TensorRT's nvinfer1::PluginTensorDesc.
/// Used by getWorkspaceSize() to compute workspace requirements from shapes
/// and data types without exposing full TensorView data pointers.
struct TensorDesc {
    static constexpr int64_t kMaxRank = 8;

    detail::SmallVector<int64_t, kMaxRank> dims;
    int64_t rank = 0;
    DataType dtype = DataType::f32;
    TensorLayout layout = TensorLayout::NCHW;

    /// Total number of logical elements.
    /// For packed layouts (NCHWC8, etc.), counts storage elements
    /// including channel block rounding: ceil(C/pack) × pack instead of C.
    int64_t numel() const noexcept {
        if (rank == 0) return 1;  // scalar = 1 element
        int64_t n = 1;
        for (int64_t i = 0; i < rank; ++i) {
            int64_t d = dims[static_cast<size_t>(i)];
            if (i == 1) {
                int64_t pack = layout_channel_pack(layout);
                if (pack > 1) d = ((d + pack - 1) / pack) * pack;
            }
            n *= d;
        }
        return n;
    }

    /// Total storage size in bytes (numel × elem_size, with channel rounding).
    /// Does NOT include per-row pitch padding — call storage_bytes() for that.
    size_t nbytes() const noexcept {
        return static_cast<size_t>(numel()) * data_type_size(dtype);
    }

    /// Row pitch (byte stride between consecutive rows).
    /// For rank < 2, returns elem_size.
    /// For packed layouts, pitch = align_up(W × pack × elem_size, alignment).
    int64_t row_pitch(int64_t alignment = 32) const noexcept {
        if (rank < 2) return static_cast<int64_t>(data_type_size(dtype));
        int64_t pack = std::max<int64_t>(1, layout_channel_pack(layout));
        int64_t W = dims[static_cast<size_t>(rank - 1)];
        int64_t elem_size = static_cast<int64_t>(data_type_size(dtype));
        int64_t row_bytes = W * pack * elem_size;
        return ((row_bytes + alignment - 1) / alignment) * alignment;
    }

    /// Actual storage size including default-aligned row pitch (32B).
    size_t storage_bytes(int64_t alignment = 32) const noexcept {
        if (rank < 2) return nbytes();
        int64_t pack = std::max<int64_t>(1, layout_channel_pack(layout));
        int64_t W = dims[static_cast<size_t>(rank - 1)];
        int64_t elem_size = static_cast<int64_t>(data_type_size(dtype));
        int64_t row_bytes = W * pack * elem_size;
        int64_t aligned_row = ((row_bytes + alignment - 1) / alignment) * alignment;
        int64_t rows = dims[0];
        if (rank >= 2) rows *= ((dims[1] + pack - 1) / pack);
        for (int64_t d = 2; d < rank - 1; ++d)
            rows *= dims[static_cast<size_t>(d)];
        return static_cast<size_t>(rows * aligned_row);
    }
};

class TensorView {
public:
    static constexpr int64_t kMaxRank = 8;

    // ---- Construction ----

    /// Default: empty view.
    TensorView() = default;

    /// Construct a dense (contiguous) TensorView.
    /// The row pitch is computed as last_dim * elem_size (no alignment padding).
    TensorView(std::span<const int64_t> shape, DataType dtype,
               void* data, TensorLayout layout = TensorLayout::NCHW,
               QuantParams qp = {})
        : dtype_(dtype), layout_(layout), data_(data), quant_params_(qp)
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
               TensorLayout layout = TensorLayout::NCHW,
               QuantParams qp = {})
        : dtype_(dtype), layout_(layout), data_(data), pitch_(pitch), quant_params_(qp)
    {
        NNOPS_ASSERT(shape.size() <= kMaxRank);
        rank_ = static_cast<int64_t>(shape.size());
        shape_ = shape;
        // Validate pitch for packed layouts: must hold at least
        // last_dim * pack_size elements per row.
        const int64_t pack = layout_channel_pack(layout);
        if (pack > 1 && rank_ >= 2) {
            const int64_t elem_size = static_cast<int64_t>(data_type_size(dtype));
            const int64_t min_pitch = shape_[static_cast<size_t>(rank_ - 1)]
                                    * pack * elem_size;
            NNOPS_ASSERT(pitch_ >= min_pitch);
        }
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

    /// Quantization parameters (scale, zero_point, granularity).
    const QuantParams& quant_params() const noexcept { return quant_params_; }

    /// Whether the tensor holds quantized integer data that should be
    /// dequantized (s8/u8 with active scale/zp; extensible to s4/u4).
    bool is_quantized() const noexcept {
        return is_quantized_dtype(dtype_) && quant_params_.is_active();
    }

    /// Typed data pointer (mutable).
    template <typename T>
    T* ptr() noexcept { return static_cast<T*>(data_); }

    /// Typed data pointer (const).
    template <typename T>
    const T* ptr() const noexcept { return static_cast<const T*>(data_); }

    /// Typed data pointer for a specific row (mutable).
    /// Returns a pointer to the first element of the given row.
    /// row_id selects the row; pitch_ (in bytes) determines the row stride.
    template <typename T>
    T* ptr(size_t row_id) noexcept {
        return reinterpret_cast<T*>(static_cast<char*>(data_) + row_id * static_cast<size_t>(pitch_));
    }

    /// Typed data pointer for a specific row (const).
    template <typename T>
    const T* ptr(size_t row_id) const noexcept {
        return reinterpret_cast<const T*>(static_cast<const char*>(data_) + row_id * static_cast<size_t>(pitch_));
    }

    /// Total number of logical elements (product of all dimensions).
    /// For packed layouts this is the logical count (N×C×H×W), not the storage
    /// count (which includes channel block rounding and row padding).
    int64_t numel() const noexcept {
        if (rank_ == 0) { return data_ != nullptr ? 1 : 0; }  // scalar=1, empty=0
        int64_t n = 1;
        for (int64_t i = 0; i < rank_; ++i) {
            n *= shape_[static_cast<size_t>(i)];
        }
        return n;
    }

    /// Total buffer size in bytes including row pitch padding.
    /// For planar layouts this equals numel × elem_size.
    /// For packed layouts this accounts for channel block rounding
    /// and aligned row pitch padding.
    size_t nbytes() const noexcept {
        return storage_bytes();
    }

    /// Actual storage size in bytes: total_rows × pitch.
    /// For planar layouts equals numel × elem_size.
    /// For packed layouts accounts for channel block rounding and row alignment.
    size_t storage_bytes() const noexcept {
        if (rank_ < 2) return static_cast<size_t>(numel()) * data_type_size(dtype_);
        return static_cast<size_t>(total_rows()) * static_cast<size_t>(pitch_);
    }

    /// Whether this is an empty view.
    bool is_empty() const noexcept { return data_ == nullptr; }

    /// Return a lightweight TensorDesc (no data pointer) for workspace queries.
    TensorDesc desc() const noexcept {
        TensorDesc d;
        d.rank = rank_;
        d.dims = shape_;
        d.dtype = dtype_;
        d.layout = layout_;
        return d;
    }

    // ---- Derived stride helpers (element counts, not bytes) ----

    /// Compute the element stride for a given dimension.
    /// Stride is the number of elements to skip to advance by 1 in that dim.
    /// For rank-1 (innermost): always 1 (contiguous within a row).
    /// For rank-2: pitch / elem_size (may be > shape[rank-1] if padded).
    /// For higher dims: shape[dim+1] * stride(dim+1).
    int64_t stride_elems(int64_t dim) const noexcept {
        NNOPS_ASSERT(dim >= 0 && dim < rank_);
        // Innermost dimension: for packed layouts, each logical step spans
        // the pack lane (e.g., 8 for NCHWC8); for planar, stride is 1.
        if (dim == rank_ - 1) {
            return channel_pack_size();
        }
        // Second-innermost: stride comes from pitch (may be > shape[rank-1] if padded)
        const int64_t elem_size = static_cast<int64_t>(data_type_size(dtype_));
        int64_t s = pitch_ / elem_size;
        if (dim == rank_ - 2) { return s; }
        // Outer dimensions: accumulate via shape[i+1] * stride(i+1).
        // For channel-packed layouts, the physical size of the C dimension
        // is the number of channel blocks, not the logical channel count.
        for (int64_t i = rank_ - 3; i >= dim; --i) {
            int64_t inner_size = shape_[static_cast<size_t>(i + 1)];
            if (i + 1 == 1 && channel_pack_size() > 1) {
                inner_size = num_channel_blocks();
            }
            s = inner_size * s;
        }
        return s;
    }

    /// Convenience: number of elements per row (stride of the 2nd-innermost dim).
    /// Equivalent to stride_elems(rank() - 2) for rank >= 2, or 1 for rank < 2.
    int64_t row_stride_elems() const noexcept {
        if (rank_ < 2) { return 1; }
        return pitch_ / static_cast<int64_t>(data_type_size(dtype_));
    }

    // ---- Layout-aware accessors ----

    /// Channel pack size (1 for NCHW/NCDHW, 8 for NCHWC8/NCDHWC8, etc.).
    int64_t channel_pack_size() const noexcept {
        return layout_channel_pack(layout_);
    }

    /// Number of channel blocks: ceil(C / pack_size).
    /// For NCHWC8 with C=20: returns 3 (two full C8 blocks + one partial).
    int64_t num_channel_blocks() const noexcept {
        if (rank_ < 2) return 1;
        int64_t pack = channel_pack_size();
        if (pack <= 1) return shape_[1];  // planar: each channel is its own "block"
        return (shape_[1] + pack - 1) / pack;
    }

    /// Element stride between consecutive channel blocks.
    /// For NCHWC8 [N,C,H,W]: H * row_stride_elems() = H * W * 8.
    /// For NCDHWC8 [N,C,D,H,W]: D * H * row_stride_elems() = D * H * W * 8.
    int64_t channel_block_stride_elems() const noexcept {
        if (rank_ < 3) return row_stride_elems();
        int64_t s = row_stride_elems();
        // Multiply by all spatial dimensions between C and the row
        for (int64_t d = 2; d < rank_ - 1; ++d) {
            s *= shape_[static_cast<size_t>(d)];
        }
        return s;
    }

    /// Total number of rows for flat row-by-row processing.
    /// NCHW:  N * C * H.
    /// NCHWC8: N * ceil(C/8) * H.
    /// NCDHWC8: N * ceil(C/8) * D * H.
    int64_t total_rows() const noexcept {
        if (rank_ < 1) return 0;
        int64_t rows = shape_[0];  // N
        // Channel blocks only matter when there are spatial dims (rank >= 3).
        // For rank-2 [N, C], shape_[1] IS the last dim, not a separate channel
        // dimension — each "row" is the whole C-contiguous block.
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
