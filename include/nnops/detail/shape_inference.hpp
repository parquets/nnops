#pragma once
/// @file shape_inference.hpp
/// @brief Pure shape-inference helpers used by most operators.
///
/// Given an operator's attributes and input descriptors, these compute output
/// descriptors; no data buffer is touched. The header depends only on
/// TensorDesc and detail/small_vector.hpp, never on an op header, so an op's
/// dispatch unit can call it from getOutputTensorDesc without an include cycle.
///
/// An output inherits `layout` and `dtype` from inputs[0] unless its helper
/// says otherwise. Validation belongs to the caller: the helpers assume the
/// ranks and axis indices they read exist.
///
/// Conventions the per-op helpers encode:
///  - `auto_pad` mirrors Conv2DAttributes::AutoPad, Conv3DAttributes::AutoPad
///    and PoolingAttributes::AutoPad: 0=NOTSET (explicit symmetric `padding`),
///    1=SAME_UPPER, 2=SAME_LOWER, 3=VALID.
///  - Conv2D weights are OIHW [OC, IC/G, KH, KW] (Conv3D adds KD). The
///    channel-last reading `weight_hwio` instead expects TensorLayout::HWIO,
///    TensorFlow's [KH, KW, IC/G, OC], so OC is the last axis. TransposeConv
///    mirrors it: weight [IC, OC/G, KH, KW] and OC = dims[1] * groups.
///  - DepthwiseConv, Pooling and Resize take their spatial rank from the input
///    (rank - 2 is 2 or 3) and, for a 2D input, skip the leading D entry of the
///    per-axis attribute span. GridSample reads the same spatial dims from the
///    grid instead.
///  - MatMul's B may be a prepacked RHS (TensorLayout::PackedB): it leads with
///    N whatever its source order and has no K axis, because the blob is
///    addressed by geometry derived from the activation's K.

#include "nnops/core/tensor_view.hpp"
#include "nnops/detail/small_vector.hpp"

#include <cstdint>
#include <algorithm>
#include <span>

namespace nnops {

// ============================================================
// Utility helpers
// ============================================================

inline int64_t resolve_axis(int64_t axis, int64_t rank) noexcept {
    return axis < 0 ? axis + rank : axis;
}

inline int64_t ceil_div(int64_t a, int64_t b) noexcept {
    return (a + b - 1) / b;
}

/// One spatial output dimension; `padding` applies only to NOTSET, and an
/// unrecognized `auto_pad` returns 0.
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
    case 0: // NOTSET
        return (input_size + 2 * padding - effective_kernel) / stride + 1;

    case 1: // SAME_UPPER
        return ceil_div(input_size, stride);

    case 2: // SAME_LOWER
        return (input_size > 0)
            ? (input_size - 1) / stride + 1
            : 1;

    case 3: // VALID
        return (input_size - effective_kernel) / stride + 1;

    default:
        return 0;
    }
}

/// Leading zero-pad for one dimension; the trailing half only affects the
/// output size, so it is never returned.
inline int64_t resolve_leading_padding(
    int64_t input_size,
    int64_t kernel,
    int64_t stride,
    int64_t dilation,
    int64_t padding,
    int auto_pad) noexcept
{
    switch (auto_pad) {
    case 1: // SAME_UPPER: extra padding goes after
    case 2: { // SAME_LOWER: extra padding goes before
        const int64_t eff = dilation * (kernel - 1) + 1;
        const int64_t total = std::max<int64_t>(
            0, (ceil_div(input_size, stride) - 1) * stride + eff - input_size);
        return (auto_pad == 1) ? total / 2 : total - total / 2;
    }
    case 3: // VALID
        return 0;
    default: // NOTSET
        return padding;
    }
}

/// NumPy broadcast: right-aligned, per-axis max, a 1 broadcasts to the other size.
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
            // Incompatible broadcast: keep dim_a; the caller validates.
            result[static_cast<size_t>(rmax - 1 - i)] = dim_a;
        }
    }
    return result;
}

// ============================================================
// Per-operator shape inference functions
// ============================================================

/// Returns [N, OC, OH, OW]; `weight_hwio` selects the HWIO reading of the weight.
inline std::vector<TensorDesc> conv2d_output_shape(
    std::span<const int64_t> kernel_size,
    std::span<const int64_t> stride,
    std::span<const int64_t> dilation,
    std::span<const int64_t> padding,
    int auto_pad,
    int64_t groups,
    std::span<const TensorDesc> inputs,
    bool weight_hwio = false)
{
    const auto& in = inputs[0];
    const auto& wt = inputs[1];

    TensorDesc out;
    out.rank   = 4;
    out.layout = in.layout;
    out.dtype  = in.dtype;

    out.dims.resize(4);
    out.dims[0] = in.dims[0];
    out.dims[1] = weight_hwio ? wt.dims[3] : wt.dims[0];

    out.dims[2] = compute_spatial_output_size(
        in.dims[2], kernel_size[0], stride[0], dilation[0], padding[0], auto_pad);
    out.dims[3] = compute_spatial_output_size(
        in.dims[3], kernel_size[1], stride[1], dilation[1], padding[1], auto_pad);

    return {out};
}

/// Returns [N, OC, OD, OH, OW] from an OIHW weight.
inline std::vector<TensorDesc> conv3d_output_shape(
    std::span<const int64_t> kernel_size,
    std::span<const int64_t> stride,
    std::span<const int64_t> dilation,
    std::span<const int64_t> padding,
    int auto_pad,
    int64_t groups,
    std::span<const TensorDesc> inputs)
{
    const auto& in = inputs[0];
    const auto& wt = inputs[1];

    TensorDesc out;
    out.rank   = 5;
    out.layout = in.layout;
    out.dtype  = in.dtype;

    out.dims.resize(5);
    out.dims[0] = in.dims[0];
    out.dims[1] = wt.dims[0];

    out.dims[2] = compute_spatial_output_size(
        in.dims[2], kernel_size[0], stride[0], dilation[0], padding[0], auto_pad);
    out.dims[3] = compute_spatial_output_size(
        in.dims[3], kernel_size[1], stride[1], dilation[1], padding[1], auto_pad);
    out.dims[4] = compute_spatial_output_size(
        in.dims[4], kernel_size[2], stride[2], dilation[2], padding[2], auto_pad);

    return {out};
}

/// Returns [N, OC, OH, OW] with OC = weight.dims[1] * groups.
inline std::vector<TensorDesc> transpose_conv2d_output_shape(
    std::span<const int64_t> kernel_size,
    std::span<const int64_t> stride,
    std::span<const int64_t> dilation,
    std::span<const int64_t> padding,
    std::span<const int64_t> output_padding,
    int64_t groups,
    std::span<const TensorDesc> inputs)
{
    const auto& in = inputs[0];
    const auto& wt = inputs[1];

    TensorDesc out;
    out.rank   = 4;
    out.layout = in.layout;
    out.dtype  = in.dtype;

    out.dims.resize(4);
    out.dims[0] = in.dims[0];
    out.dims[1] = wt.dims[1] * groups;

    auto compute_output = [](int64_t I, int64_t K, int64_t S,
                              int64_t D, int64_t P, int64_t OP) {
        return (I - 1) * S - 2 * P + D * (K - 1) + OP + 1;
    };

    out.dims[2] = compute_output(
        in.dims[2], kernel_size[0], stride[0], dilation[0], padding[0], output_padding[0]);
    out.dims[3] = compute_output(
        in.dims[3], kernel_size[1], stride[1], dilation[1], padding[1], output_padding[1]);

    return {out};
}

/// Returns [N, C, ...spatial]; rank 4 is 2D, rank 5 is 3D.
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
    const int64_t srank = rank - 2;

    TensorDesc out;
    out.rank   = rank;
    out.layout = in.layout;
    out.dtype  = in.dtype;

    out.dims.resize(static_cast<size_t>(rank));
    out.dims[0] = in.dims[0];
    out.dims[1] = in.dims[1];

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

/// Returns [N, C, ...spatial]; rank 4 is 2D, rank 5 is 3D.
inline std::vector<TensorDesc> pooling_output_shape(
    std::span<const int64_t> kernel_shape,
    std::span<const int64_t> stride,
    std::span<const int64_t> dilation,
    std::span<const int64_t> padding,
    int auto_pad,
    std::span<const TensorDesc> inputs)
{
    const auto& in = inputs[0];
    const int64_t rank = in.rank;
    const int64_t srank = rank - 2;

    TensorDesc out;
    out.rank   = rank;
    out.layout = in.layout;
    out.dtype  = in.dtype;

    out.dims.resize(static_cast<size_t>(rank));
    out.dims[0] = in.dims[0];
    out.dims[1] = in.dims[1];

    const int64_t k_offset = (srank == 2) ? 1 : 0;

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

/// [..., K] x [N, K] -> [..., N].
inline std::vector<TensorDesc> linear_output_shape(
    std::span<const TensorDesc> inputs)
{
    const auto& in = inputs[0];
    const auto& wt = inputs[1];

    TensorDesc out;
    out.layout = in.layout;
    out.dtype  = in.dtype;

    const int64_t in_rank = in.rank;
    const int64_t K = in.dims[static_cast<size_t>(in_rank - 1)];
    const int64_t N = wt.dims[0];

    if (in_rank == 2) {
        out.rank = 2;
        out.dims.resize(2);
        out.dims[0] = in.dims[0];
        out.dims[1] = N;
    } else {
        out.rank = in_rank;
        out.dims.resize(static_cast<size_t>(in_rank));
        for (int64_t i = 0; i < in_rank - 1; ++i) {
            out.dims[static_cast<size_t>(i)] = in.dims[static_cast<size_t>(i)];
        }
        out.dims[static_cast<size_t>(in_rank - 1)] = N;
    }

    return {out};
}

/// [..., M, K] x [..., K, N] -> [..., M, N], batch dims broadcast.
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

    const int64_t K_a = a.dims[static_cast<size_t>(ra - 1)];  // K (or M if transpose_a)
    const int64_t M   = transpose_a ? K_a : a.dims[static_cast<size_t>(ra - 2)];

    const bool packed_b = is_packed_matmul_b(b.layout);
    const int64_t K_b = packed_b ? K_a
        : transpose_b ? b.dims[static_cast<size_t>(rb - 1)] : b.dims[static_cast<size_t>(rb - 2)];
    const int64_t N   = packed_b ? b.dims[0]
        : transpose_b ? b.dims[static_cast<size_t>(rb - 2)] : b.dims[static_cast<size_t>(rb - 1)];

    (void)K_a; (void)K_b; // Validation done by caller

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

/// Output matches Q.
inline std::vector<TensorDesc> attention_output_shape(
    std::span<const TensorDesc> inputs)
{
    const auto& q = inputs[0];

    TensorDesc out;
    out.rank   = q.rank;
    out.layout = q.layout;
    out.dtype  = q.dtype;
    out.dims   = q.dims;

    return {out};
}

/// Shape-preserving ops (activation, eltwise, unary, softmax, cumsum, clamp,
/// rope, norm).
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

/// Same dims and dtype; only `layout` becomes target_layout.
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

/// One axis reduced, dropped or kept as 1.
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

/// Spatial dims come from output_size [OD, OH, OW]; a 2D input skips OD.
inline std::vector<TensorDesc> resize_output_shape(
    std::span<const int64_t> output_size,
    std::span<const TensorDesc> inputs)
{
    const auto& in = inputs[0];
    const int64_t rank = in.rank;
    const int64_t srank = rank - 2;

    TensorDesc out;
    out.rank   = rank;
    out.layout = in.layout;
    out.dtype  = in.dtype;

    out.dims.resize(static_cast<size_t>(rank));
    out.dims[0] = in.dims[0];
    out.dims[1] = in.dims[1];

    const int64_t os_offset = (srank == 2) ? 1 : 0;
    for (int64_t d = 0; d < srank; ++d) {
        out.dims[static_cast<size_t>(2 + d)] = output_size[static_cast<size_t>(os_offset + d)];
    }

    return {out};
}

/// Spatial dims come from grid [N, (OD,) OH, OW, 2|3].
inline std::vector<TensorDesc> grid_sample_output_shape(
    std::span<const TensorDesc> inputs)
{
    const auto& in = inputs[0];
    const auto& grid = inputs[1];
    const int64_t irank = in.rank;
    const int64_t srank = irank - 2;

    TensorDesc out;
    out.rank   = irank;
    out.layout = in.layout;
    out.dtype  = in.dtype;

    out.dims.resize(static_cast<size_t>(irank));
    out.dims[0] = in.dims[0];
    out.dims[1] = in.dims[1];

    for (int64_t d = 0; d < srank; ++d) {
        out.dims[static_cast<size_t>(2 + d)] =
            grid.dims[static_cast<size_t>(1 + d)];
    }

    return {out};
}

/// Shape of x, dtype output_dtype.
inline std::vector<TensorDesc> quantize_linear_output_shape(
    std::span<const TensorDesc> inputs,
    DataType output_dtype = DataType::s8)
{
    const auto& x = inputs[0];

    TensorDesc out;
    out.rank   = x.rank;
    out.layout = x.layout;
    out.dtype  = output_dtype;
    out.dims   = x.dims;

    return {out};
}

/// Shape of x, dtype output_dtype.
inline std::vector<TensorDesc> dequantize_linear_output_shape(
    std::span<const TensorDesc> inputs,
    DataType output_dtype = DataType::f32)
{
    const auto& x = inputs[0];

    TensorDesc out;
    out.rank   = x.rank;
    out.layout = x.layout;
    out.dtype  = output_dtype;
    out.dims   = x.dims;

    return {out};
}

/// Sums the axis dim across inputs; all inputs share rank, dtype and layout.
inline std::vector<TensorDesc> concat_output_shape(
    int64_t axis,
    std::span<const TensorDesc> inputs)
{
    const auto& in0 = inputs[0];
    const int64_t rank = in0.rank;
    const int64_t ax = resolve_axis(axis, rank);

    int64_t axis_sum = 0;
    for (const auto& in : inputs) {
        axis_sum += in.dims[static_cast<size_t>(ax)];
    }

    TensorDesc out;
    out.rank   = rank;
    out.layout = in0.layout;
    out.dtype  = in0.dtype;
    out.dims.resize(static_cast<size_t>(rank));
    for (int64_t i = 0; i < rank; ++i) {
        out.dims[static_cast<size_t>(i)] =
            (i == ax) ? axis_sum : in0.dims[static_cast<size_t>(i)];
    }

    return {out};
}

/// One axis reduced to int64 indices.
inline std::vector<TensorDesc> argminmax_output_shape(
    int64_t axis,
    bool keepdims,
    std::span<const TensorDesc> inputs)
{
    const auto& in = inputs[0];
    const int64_t rank = in.rank;
    const int64_t ax = resolve_axis(axis, rank);

    TensorDesc out;
    out.layout = in.layout;
    out.dtype  = DataType::s64;

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

/// Two outputs: values (input dtype) and indices (s64), axis dim = k.
inline std::vector<TensorDesc> topk_output_shape(
    int64_t axis,
    int64_t k,
    std::span<const TensorDesc> inputs)
{
    const auto& in = inputs[0];
    const int64_t rank = in.rank;
    const int64_t ax = resolve_axis(axis, rank);

    TensorDesc values;
    values.rank   = rank;
    values.layout = in.layout;
    values.dtype  = in.dtype;
    values.dims.resize(static_cast<size_t>(rank));
    for (int64_t i = 0; i < rank; ++i) {
        values.dims[static_cast<size_t>(i)] =
            (i == ax) ? k : in.dims[static_cast<size_t>(i)];
    }

    TensorDesc indices = values;
    indices.dtype = DataType::s64;

    return {values, indices};
}

/// [indices..., weight dims[1:]]; an s8/u8 weight yields f32.
inline std::vector<TensorDesc> embed_output_shape(
    std::span<const TensorDesc> inputs)
{
    const auto& weight  = inputs[0];
    const auto& indices = inputs[1];

    TensorDesc out;
    out.layout = weight.layout;

    if (weight.dtype == DataType::s8 || weight.dtype == DataType::u8) {
        out.dtype = DataType::f32;
    }
    else {
        out.dtype = weight.dtype;
    }

    const int64_t w_extra = weight.rank - 1;
    out.rank = indices.rank + w_extra;
    out.dims.resize(static_cast<size_t>(out.rank));

    for (int64_t i = 0; i < indices.rank; ++i) {
        out.dims[static_cast<size_t>(i)] = indices.dims[static_cast<size_t>(i)];
    }

    for (int64_t i = 0; i < w_extra; ++i) {
        out.dims[static_cast<size_t>(indices.rank + i)] =
            weight.dims[static_cast<size_t>(1 + i)];
    }
    return {out};
}

/// dims[i] = input.dims[perm[i]].
inline std::vector<TensorDesc> permute_output_shape(
    std::span<const int64_t> perm,
    std::span<const TensorDesc> inputs)
{
    const auto& in = inputs[0];
    const int64_t rank = in.rank;

    TensorDesc out;
    out.layout = in.layout;
    out.dtype  = in.dtype;
    out.rank   = rank;
    out.dims.resize(static_cast<size_t>(rank));

    for (int64_t i = 0; i < rank; ++i) {
        int64_t src = perm[static_cast<size_t>(i)];
        out.dims[static_cast<size_t>(i)] = in.dims[static_cast<size_t>(src)];
    }

    return {out};
}

/// Slices the listed axes; an unlisted axis passes through in full.
inline std::vector<TensorDesc> slice_output_shape(
    std::span<const int64_t> starts,
    std::span<const int64_t> ends,
    std::span<const int64_t> axes,
    std::span<const int64_t> steps,
    std::span<const TensorDesc> inputs)
{
    const auto& in = inputs[0];
    const int64_t rank = in.rank;

    TensorDesc out;
    out.layout = in.layout;
    out.dtype  = in.dtype;
    out.rank   = rank;
    out.dims.resize(static_cast<size_t>(rank));
    for (int64_t i = 0; i < rank; ++i) {
        out.dims[static_cast<size_t>(i)] = in.dims[static_cast<size_t>(i)];
    }

    for (size_t a = 0; a < axes.size(); ++a) {
        int64_t ax = axes[a];
        if (ax < 0) {
            ax += rank;
        }
        int64_t dim = in.dims[static_cast<size_t>(ax)];
        int64_t s = starts[a];
        int64_t e = ends[a];
        int64_t step = (a < steps.size()) ? steps[a] : int64_t(1);

        // ONNX-style: negative starts/ends wrap before clamping.
        if (s < 0) {
            s += dim;
        }
        if (e < 0) {
            e += dim;
        }
        s = std::max<int64_t>(0, std::min(s, dim));
        e = std::max<int64_t>(0, std::min(e, dim));

        int64_t out_dim = (e > s) ? (e - s + step - 1) / step : int64_t(0);
        out.dims[static_cast<size_t>(ax)] = out_dim;
    }

    return {out};
}

/// Shape-preserving; reads only inputs[0], so a partial input span is fine.
inline std::vector<TensorDesc> moe_output_shape(
    std::span<const TensorDesc> inputs)
{
    const auto& in = inputs[0];

    TensorDesc out;
    out.layout = in.layout;
    out.dtype  = in.dtype;
    out.rank   = in.rank;
    out.dims.resize(static_cast<size_t>(in.rank));
    for (int64_t i = 0; i < in.rank; ++i) {
        out.dims[static_cast<size_t>(i)] = in.dims[static_cast<size_t>(i)];
    }
    return {out};
}

}  // namespace nnops
