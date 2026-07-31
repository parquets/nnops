#pragma once
/// @file shape_inference.hpp
/// @brief Shape inference helper functions used by all operator types.
///
/// These are pure functions: given operator attributes and input shapes,
/// they compute output shapes. No data buffers are accessed.
///
/// Design: this header depends ONLY on tensor_view.hpp (TensorDesc) and
/// detail/small_vector.hpp — it does NOT include any op headers. Each op
/// header includes this file and uses the helpers in its inline override.
///
/// The auto_pad modes mirror the enum values in Conv2DAttributes::AutoPad,
/// Conv3DAttributes::AutoPad, and PoolingAttributes::AutoPad (0=NOTSET,
/// 1=SAME_UPPER, 2=SAME_LOWER, 3=VALID).

#include "nnops/core/tensor_view.hpp"
#include "nnops/detail/small_vector.hpp"

#include <cstdint>
#include <algorithm>
#include <span>

namespace nnops {

// ============================================================
// Utility helpers
// ============================================================

/// Resolve a potentially negative axis to [0, rank).
inline int64_t resolve_axis(int64_t axis, int64_t rank) noexcept {
    return axis < 0 ? axis + rank : axis;
}

/// Integer ceil division: ceil(a / b).
inline int64_t ceil_div(int64_t a, int64_t b) noexcept {
    return (a + b - 1) / b;
}

/// Compute a single spatial output dimension for convolution / pooling.
///
/// @param input_size   Spatial input size (IH, IW, ID).
/// @param kernel       Kernel size for this dimension.
/// @param stride       Stride for this dimension.
/// @param dilation     Dilation factor for this dimension.
/// @param padding      Explicit symmetric padding (only used when auto_pad == 0 / NOTSET).
/// @param auto_pad     0=NOTSET, 1=SAME_UPPER, 2=SAME_LOWER, 3=VALID.
/// @return             Output spatial size, or 0 if the configuration is invalid.
inline int64_t compute_spatial_output_size(
    int64_t input_size,
    int64_t kernel,
    int64_t stride,
    int64_t dilation,
    int64_t padding,
    int auto_pad) noexcept
{
    const int64_t effective_kernel = dilation * (kernel - 1) + 1;

    switch (auto_pad) {
    case 0: // NOTSET — explicit padding
        return (input_size + 2 * padding - effective_kernel) / stride + 1;

    case 1: // SAME_UPPER — ceil(input / stride)
        return ceil_div(input_size, stride);

    case 2: // SAME_LOWER — floor((input - 1) / stride) + 1
        return (input_size > 0)
            ? (input_size - 1) / stride + 1
            : 1;

    case 3: // VALID — no padding
        return (input_size - effective_kernel) / stride + 1;

    default:
        return 0;
    }
}

/// Compute the broadcast output shape for two tensors with numpy semantics.
///
/// Aligns shapes from the right; each dimension is the max of the two.
/// A dimension of 1 broadcasts to the other size.
inline detail::SmallVector<int64_t, TensorDesc::kMaxRank> compute_broadcast_shape(
    std::span<const int64_t> shape_a,
    std::span<const int64_t> shape_b) noexcept
{
    const int64_t ra = static_cast<int64_t>(shape_a.size());
    const int64_t rb = static_cast<int64_t>(shape_b.size());
    const int64_t rmax = std::max(ra, rb);

    detail::SmallVector<int64_t, TensorDesc::kMaxRank> result;
    result.resize(static_cast<size_t>(rmax));

    for (int64_t i = 0; i < rmax; ++i) {
        const int64_t dim_a = (i < ra) ? shape_a[static_cast<size_t>(ra - 1 - i)] : 1;
        const int64_t dim_b = (i < rb) ? shape_b[static_cast<size_t>(rb - 1 - i)] : 1;

        if (dim_a == dim_b) {
            result[static_cast<size_t>(rmax - 1 - i)] = dim_a;
        } else if (dim_a == 1) {
            result[static_cast<size_t>(rmax - 1 - i)] = dim_b;
        } else if (dim_b == 1) {
            result[static_cast<size_t>(rmax - 1 - i)] = dim_a;
        } else {
            // Incompatible broadcast — default to dim_a (caller should validate)
            result[static_cast<size_t>(rmax - 1 - i)] = dim_a;
        }
    }
    return result;
}

// ============================================================
// Per-operator shape inference functions
// ============================================================

/// Conv2D shape inference.
/// inputs[0] = input  [N, IC, IH, IW]
/// inputs[1] = weight [OC, IC/G, KH, KW]
/// inputs[2] = bias   [OC] (optional)
/// Returns: [N, OC, OH, OW]
inline std::vector<TensorDesc> conv2d_output_shape(
    std::span<const int64_t> kernel_size,
    std::span<const int64_t> stride,
    std::span<const int64_t> dilation,
    std::span<const int64_t> padding,
    int auto_pad,
    int64_t groups,
    std::span<const TensorDesc> inputs)
{
    const auto& in = inputs[0];  // [N, IC, IH, IW]
    const auto& wt = inputs[1];  // [OC, IC/G, KH, KW]

    TensorDesc out;
    out.rank   = 4;
    out.layout = in.layout;
    out.dtype  = in.dtype;

    out.dims.resize(4);
    out.dims[0] = in.dims[0];  // N
    out.dims[1] = wt.dims[0];  // OC

    out.dims[2] = compute_spatial_output_size(
        in.dims[2], kernel_size[0], stride[0], dilation[0], padding[0], auto_pad);
    out.dims[3] = compute_spatial_output_size(
        in.dims[3], kernel_size[1], stride[1], dilation[1], padding[1], auto_pad);

    return {out};
}

/// Conv3D shape inference.
/// inputs[0] = input  [N, IC, ID, IH, IW]
/// inputs[1] = weight [OC, IC/G, KD, KH, KW]
/// inputs[2] = bias   [OC] (optional)
/// Returns: [N, OC, OD, OH, OW]
inline std::vector<TensorDesc> conv3d_output_shape(
    std::span<const int64_t> kernel_size,
    std::span<const int64_t> stride,
    std::span<const int64_t> dilation,
    std::span<const int64_t> padding,
    int auto_pad,
    int64_t groups,
    std::span<const TensorDesc> inputs)
{
    const auto& in = inputs[0];  // [N, IC, ID, IH, IW]
    const auto& wt = inputs[1];  // [OC, IC/G, KD, KH, KW]

    TensorDesc out;
    out.rank   = 5;
    out.layout = in.layout;
    out.dtype  = in.dtype;

    out.dims.resize(5);
    out.dims[0] = in.dims[0];  // N
    out.dims[1] = wt.dims[0];  // OC

    out.dims[2] = compute_spatial_output_size(
        in.dims[2], kernel_size[0], stride[0], dilation[0], padding[0], auto_pad);
    out.dims[3] = compute_spatial_output_size(
        in.dims[3], kernel_size[1], stride[1], dilation[1], padding[1], auto_pad);
    out.dims[4] = compute_spatial_output_size(
        in.dims[4], kernel_size[2], stride[2], dilation[2], padding[2], auto_pad);

    return {out};
}

/// DepthwiseConv shape inference (2D / 3D auto-detected from input rank).
/// inputs[0] = input  [N, C, (D,) H, W]
/// inputs[1] = weight [C, 1, (KD,) KH, KW]
/// inputs[2] = bias   [C] (optional)
/// Returns: [N, C, (OD,) OH, OW]
inline std::vector<TensorDesc> depthwise_conv_output_shape(
    std::span<const int64_t> kernel_size,
    std::span<const int64_t> stride,
    std::span<const int64_t> dilation,
    std::span<const int64_t> padding,
    int auto_pad,
    std::span<const TensorDesc> inputs)
{
    const auto& in = inputs[0];
    const int64_t rank = in.rank;
    const int64_t srank = rank - 2;  // spatial rank: 2 or 3

    TensorDesc out;
    out.rank   = rank;
    out.layout = in.layout;
    out.dtype  = in.dtype;

    out.dims.resize(static_cast<size_t>(rank));
    out.dims[0] = in.dims[0];  // N
    out.dims[1] = in.dims[1];  // C (same as input channels)

    // Spatial dimensions start at index 2
    // kernel_size/stride/dilation/padding are [KD, KH, KW]; for 2D, skip KD
    const int64_t k_offset = (srank == 2) ? 1 : 0;

    for (int64_t d = 0; d < srank; ++d) {
        const int64_t in_dim   = in.dims[static_cast<size_t>(2 + d)];
        const int64_t k_idx    = static_cast<size_t>(k_offset + d);
        const int64_t k_size   = kernel_size[k_idx];
        const int64_t k_stride = stride[k_idx];
        const int64_t k_dil    = dilation[k_idx];
        const int64_t k_pad    = padding[k_idx];

        out.dims[static_cast<size_t>(2 + d)] = compute_spatial_output_size(
            in_dim, k_size, k_stride, k_dil, k_pad, auto_pad);
    }

    return {out};
}

/// Pooling shape inference (2D / 3D auto-detected from input rank).
/// inputs[0] = input [N, C, (D,) H, W]
/// Returns: [N, C, (OD,) OH, OW]
inline std::vector<TensorDesc> pooling_output_shape(
    std::span<const int64_t> kernel_shape,  // [KD, KH, KW]
    std::span<const int64_t> stride,        // [SD, SH, SW]
    std::span<const int64_t> dilation,      // [DD, DH, DW]
    std::span<const int64_t> padding,       // [PD, PH, PW]
    int auto_pad,
    std::span<const TensorDesc> inputs)
{
    const auto& in = inputs[0];
    const int64_t rank = in.rank;
    const int64_t srank = rank - 2;  // spatial rank: 2 or 3

    TensorDesc out;
    out.rank   = rank;
    out.layout = in.layout;
    out.dtype  = in.dtype;

    out.dims.resize(static_cast<size_t>(rank));
    out.dims[0] = in.dims[0];  // N
    out.dims[1] = in.dims[1];  // C

    // Spatial dimensions start at index 2
    const int64_t k_offset = (srank == 2) ? 1 : 0;  // shift kernel/stride/pad index for 2D

    for (int64_t d = 0; d < srank; ++d) {
        const int64_t in_dim   = in.dims[static_cast<size_t>(2 + d)];
        const int64_t k_idx    = static_cast<size_t>(k_offset + d);
        const int64_t k_size   = kernel_shape[k_idx];
        const int64_t k_stride = stride[k_idx];
        const int64_t k_dil    = dilation[k_idx];
        const int64_t k_pad    = padding[k_idx];

        out.dims[static_cast<size_t>(2 + d)] = compute_spatial_output_size(
            in_dim, k_size, k_stride, k_dil, k_pad, auto_pad);
    }

    return {out};
}

/// Linear shape inference.
/// inputs[0] = input  [M, K] (or [*, K] — batch dims broadcast)
/// inputs[1] = weight [N, K]
/// inputs[2] = bias   [N] (optional)
/// Returns: [M, N] (with batch dims from input)
inline std::vector<TensorDesc> linear_output_shape(
    std::span<const TensorDesc> inputs)
{
    const auto& in = inputs[0];   // [..., K]
    const auto& wt = inputs[1];   // [N, K]

    TensorDesc out;
    out.layout = in.layout;
    out.dtype  = in.dtype;

    const int64_t in_rank = in.rank;
    const int64_t K = in.dims[static_cast<size_t>(in_rank - 1)];
    const int64_t N = wt.dims[0];

    if (in_rank == 2) {
        // Simple 2D: [M, K] × [N, K]^T → [M, N]
        out.rank = 2;
        out.dims.resize(2);
        out.dims[0] = in.dims[0];  // M
        out.dims[1] = N;
    } else {
        // Higher-dimensional input: batch dims preserved, last dim becomes N
        out.rank = in_rank;
        out.dims.resize(static_cast<size_t>(in_rank));
        for (int64_t i = 0; i < in_rank - 1; ++i) {
            out.dims[static_cast<size_t>(i)] = in.dims[static_cast<size_t>(i)];
        }
        out.dims[static_cast<size_t>(in_rank - 1)] = N;
    }

    return {out};
}

/// MatMul shape inference.
/// A: [..., M, K] (transpose_a: [..., K, M])
/// B: [..., K, N] (transpose_b: [..., N, K])
/// Returns: [..., M, N] with broadcast batch dims
inline std::vector<TensorDesc> matmul_output_shape(
    bool transpose_a,
    bool transpose_b,
    std::span<const TensorDesc> inputs)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    TensorDesc out;
    out.layout = a.layout;
    out.dtype  = a.dtype;

    const int64_t ra = a.rank;
    const int64_t rb = b.rank;

    // Extract M, K_a from A
    const int64_t K_a = a.dims[static_cast<size_t>(ra - 1)];  // K (or M if transpose_a)
    const int64_t M   = transpose_a ? K_a : a.dims[static_cast<size_t>(ra - 2)];

    // Extract K_b, N from B
    const int64_t K_b = transpose_b ? b.dims[static_cast<size_t>(rb - 1)] : b.dims[static_cast<size_t>(rb - 2)];
    const int64_t N   = transpose_b ? b.dims[static_cast<size_t>(rb - 2)] : b.dims[static_cast<size_t>(rb - 1)];

    (void)K_a; (void)K_b; // Validation done by caller

    // Broadcast batch dimensions
    std::span<const int64_t> batch_a(a.dims.data(), static_cast<size_t>(ra - 2));
    std::span<const int64_t> batch_b(b.dims.data(), static_cast<size_t>(rb - 2));

    auto batch = compute_broadcast_shape(batch_a, batch_b);
    const int64_t batch_rank = static_cast<int64_t>(batch.size());

    out.rank = batch_rank + 2;
    out.dims.resize(static_cast<size_t>(out.rank));
    for (int64_t i = 0; i < batch_rank; ++i) {
        out.dims[static_cast<size_t>(i)] = batch[static_cast<size_t>(i)];
    }
    out.dims[static_cast<size_t>(batch_rank)]     = M;
    out.dims[static_cast<size_t>(batch_rank + 1)] = N;

    return {out};
}

/// Attention shape inference.
/// Q: [B, H, Sq, D] or [B, Sq, H*D]
/// K: [B, H, Sk, D] or [B, Sk, H*D]
/// V: [B, H, Sk, D] or [B, Sk, H*D]
/// Returns: same shape as Q
inline std::vector<TensorDesc> attention_output_shape(
    std::span<const TensorDesc> inputs)
{
    const auto& q = inputs[0];

    TensorDesc out;
    out.rank   = q.rank;
    out.layout = q.layout;
    out.dtype  = q.dtype;
    out.dims   = q.dims;  // Output shape == Q shape

    return {out};
}

/// Element-wise (activation, eltwise, unary, softmax, cumsum, batchnorm,
/// layernorm, rmsnorm) shape inference — output matches input[0].
/// These operators all preserve the input shape.
inline std::vector<TensorDesc> identity_output_shape(
    std::span<const TensorDesc> inputs)
{
    const auto& in = inputs[0];

    TensorDesc out;
    out.rank   = in.rank;
    out.layout = in.layout;
    out.dtype  = in.dtype;
    out.dims   = in.dims;

    return {out};
}

/// LayoutConvert shape inference.
/// Output has the same logical shape and dtype as input;
/// only the layout field changes to target_layout.
inline std::vector<TensorDesc> layout_convert_output_shape(
    std::span<const TensorDesc> inputs,
    TensorLayout target_layout)
{
    const auto& in = inputs[0];

    TensorDesc out;
    out.rank   = in.rank;
    out.layout = target_layout;
    out.dtype  = in.dtype;
    out.dims   = in.dims;

    return {out};
}

/// Reduce shape inference.
/// Axis removed (or kept as size-1 if keepdims).
inline std::vector<TensorDesc> reduce_output_shape(
    int64_t axis,
    bool keepdims,
    std::span<const TensorDesc> inputs)
{
    const auto& in = inputs[0];
    const int64_t rank = in.rank;
    const int64_t ax = resolve_axis(axis, rank);

    TensorDesc out;
    out.layout = in.layout;
    out.dtype  = in.dtype;

    if (keepdims) {
        out.rank = rank;
        out.dims.resize(static_cast<size_t>(rank));
        for (int64_t i = 0; i < rank; ++i) {
            out.dims[static_cast<size_t>(i)] = (i == ax)
                ? int64_t(1)
                : in.dims[static_cast<size_t>(i)];
        }
    } else {
        out.rank = rank - 1;
        out.dims.resize(static_cast<size_t>(rank - 1));
        for (int64_t i = 0, o = 0; i < rank; ++i) {
            if (i != ax) {
                out.dims[static_cast<size_t>(o++)] = in.dims[static_cast<size_t>(i)];
            }
        }
    }

    return {out};
}

/// Resize shape inference (2D / 3D auto-detected from input rank).
/// inputs[0] = input [N, C, (D,) H, W]
/// Returns: [N, C, (OD,) OH, OW]
/// output_size has 3 elements [OD, OH, OW]; for 2D (rank=4), only [OH, OW] are used.
inline std::vector<TensorDesc> resize_output_shape(
    std::span<const int64_t> output_size,
    std::span<const TensorDesc> inputs)
{
    const auto& in = inputs[0];
    const int64_t rank = in.rank;
    const int64_t srank = rank - 2;  // spatial rank: 2 or 3

    TensorDesc out;
    out.rank   = rank;
    out.layout = in.layout;
    out.dtype  = in.dtype;

    out.dims.resize(static_cast<size_t>(rank));
    out.dims[0] = in.dims[0];  // N
    out.dims[1] = in.dims[1];  // C

    // output_size is [OD, OH, OW]; for 2D, skip the first element
    const int64_t os_offset = (srank == 2) ? 1 : 0;
    for (int64_t d = 0; d < srank; ++d) {
        out.dims[static_cast<size_t>(2 + d)] = output_size[static_cast<size_t>(os_offset + d)];
    }

    return {out};
}

/// GridSample shape inference (2D / 3D auto-detected from input rank).
/// inputs[0] = input  [N, C, (D,) H, W]
/// inputs[1] = grid   [N, (OD,) OH, OW, 2|3]
/// Returns: [N, C, (OD,) OH, OW]
inline std::vector<TensorDesc> grid_sample_output_shape(
    std::span<const TensorDesc> inputs)
{
    const auto& in = inputs[0];    // [N, C, (D,) H, W]
    const auto& grid = inputs[1];  // [N, (D_out,) H_out, W_out, 2|3]
    const int64_t irank = in.rank;
    const int64_t srank = irank - 2;  // spatial rank: 2 or 3

    TensorDesc out;
    out.rank   = irank;
    out.layout = in.layout;
    out.dtype  = in.dtype;

    out.dims.resize(static_cast<size_t>(irank));
    out.dims[0] = in.dims[0];  // N
    out.dims[1] = in.dims[1];  // C

    // Grid spatial dims: grid rank = srank + 3 (N + spatial + coordinate)
    // For 2D: grid = [N, OH, OW, 2], spatial dims at indices 1,2
    // For 3D: grid = [N, OD, OH, OW, 3], spatial dims at indices 1,2,3
    for (int64_t d = 0; d < srank; ++d) {
        out.dims[static_cast<size_t>(2 + d)] =
            grid.dims[static_cast<size_t>(1 + d)];
    }

    return {out};
}

}  // namespace nnops
