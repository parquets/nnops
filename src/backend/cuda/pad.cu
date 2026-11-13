/// @file pad.cu
/// @brief CUDA Pad — one block per output row, with each axis's map applied
///        where it is needed and skipped where it is the identity.
///
/// Padding is a gather whose index map is known in closed form: every mode but
/// Constant maps an out-of-range coordinate back inside the axis, and inside
/// the axis every mode *is* the identity. So a row is three pieces — the two
/// borders, whose indices go through the map, and the middle, which is the
/// source run copied as it stands and takes the vectorized path. A row whose
/// outer coordinate left its axis under Constant is a plain fill.
///
/// Maps are computed, not tabulated: each is a clamp, one fold, or a modulo, so
/// the kernel carries a few branches instead of a host-built table.
///
/// The output's extents come from the descriptor, so a crop (a negative pad) is
/// the same code with a narrower output and a positive source shift.

#include "move_common.cuh"
#include "nnops/ops/pad.hpp"

#include <cmath>
#include <limits>

namespace nnops::backend::cuda {
namespace {

using PadMode = PadAttributes::Mode;

struct PadGeom {
    int64_t rows = 0;        ///< output extents over axes 0..rank-2
    int64_t cols = 0;        ///< output extent of the last axis
    int64_t rank = 0;
    int64_t out_dims[8] = {};
    int64_t out_strides[8] = {};
    int64_t in_dims[8] = {};
    int64_t in_strides[8] = {};
    int64_t pad_begin[8] = {};   ///< begin pad per axis (negative crops)
    PadMode mode = PadMode::Constant;
};

__host__ __device__ inline int64_t pad_map(int64_t i, int64_t n, PadMode mode)
{
    if (n <= 0) { return -1; }
    switch (mode) {
    case PadMode::Edge:
        return i < 0 ? 0 : (i >= n ? n - 1 : i);
    case PadMode::Reflect:
        if (i < 0) { return -i; }
        if (i >= n) { return 2 * (n - 1) - i; }
        return i;
    case PadMode::Wrap:
        return ((i % n) + n) % n;
    case PadMode::Constant:
    default:
        return (i < 0 || i >= n) ? -1 : i;
    }
}

/// The fill value at the output's dtype; ints round it like the CPU does.
template <class T>
inline T pad_constant(float v)
{
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, __half>) {
        return __float2half(v);
    } else {
        const double r = std::nearbyint(static_cast<double>(v));
        return static_cast<T>(std::clamp<double>(
            r, static_cast<double>(std::numeric_limits<T>::lowest()),
            static_cast<double>(std::numeric_limits<T>::max())));
    }
}

template <class T>
__global__ void pad_rows_kernel(const T* __restrict__ src,
                                T* __restrict__ dst, PadGeom g, T cval)
{
    const int64_t tid = int64_t(threadIdx.x);
    const int64_t nth = int64_t(blockDim.x);
    const int64_t last = g.rank - 1;
    const int64_t n_in = g.in_dims[last];
    const int64_t begin = g.pad_begin[last];
    const int64_t in_step = g.in_strides[last];
    const int64_t out_step = g.out_strides[last];

    // Inside [lo, hi) the last axis maps to itself, so nothing there is padded.
    const int64_t mid_begin = begin < 0 ? 0 : (begin > g.cols ? g.cols : begin);
    int64_t mid_end = begin + n_in;
    if (mid_end > g.cols) { mid_end = g.cols; }
    if (mid_end < mid_begin) { mid_end = mid_begin; }

    for (int64_t r = blockIdx.x; r < g.rows; r += gridDim.x) {
        int64_t rem = r;
        int64_t so = 0;
        int64_t dof = 0;
        bool fill = false;
        for (int64_t d = g.rank - 2; d >= 0; --d) {
            const int64_t n = g.out_dims[d];
            const int64_t c = (n > 0) ? rem % n : 0;
            rem = (n > 0) ? rem / n : 0;
            dof += c * g.out_strides[d];
            const int64_t m = pad_map(c - g.pad_begin[d], g.in_dims[d], g.mode);
            if (m < 0) {
                fill = true;
            } else {
                so += m * g.in_strides[d];
            }
        }

        T* drow = dst + dof;
        if (fill) {
            for (int64_t i = tid; i < g.cols; i += nth) { drow[i * out_step] = cval; }
            continue;
        }

        for (int64_t i = tid; i < mid_begin; i += nth) {
            const int64_t m = pad_map(i - begin, n_in, g.mode);
            drow[i * out_step] = (m < 0) ? cval : src[so + m * in_step];
        }
        copy_run(drow + mid_begin * out_step,
                 src + so + (mid_begin - begin) * in_step,
                 mid_end - mid_begin, in_step, out_step);
        for (int64_t i = mid_end + tid; i < g.cols; i += nth) {
            const int64_t m = pad_map(i - begin, n_in, g.mode);
            drow[i * out_step] = (m < 0) ? cval : src[so + m * in_step];
        }
    }
}

template <class T>
Status pad_entry(const PadAttributes& attrs, TensorView& output,
                 const TensorView& input, cudaStream_t stream)
{
    const int64_t rank = input.rank();
    const int64_t total = output.numel();
    if (total == 0) { return Status::ok; }

    PadGeom g;
    g.rank = rank;
    fill_output_dims(output, g.out_dims);
    fill_view_strides(output, g.out_strides);
    fill_view_strides(input, g.in_strides);
    for (int64_t d = 0; d < rank; ++d) { g.in_dims[d] = input.shape(d); }
    for (int64_t d = 0; d < rank; ++d) {
        g.pad_begin[d] = attrs.pads[static_cast<size_t>(d)];
    }
    g.mode = attrs.mode;
    g.cols = g.out_dims[rank - 1];
    g.rows = (g.cols > 0) ? total / g.cols : 0;

    const unsigned grid =
        static_cast<unsigned>(std::min<int64_t>(g.rows, int64_t(1) << 20));
    pad_rows_kernel<T><<<grid, kCudaBlockSize, 0, stream>>>(
        input.ptr<T>(), output.ptr<T>(), g, pad_constant<T>(attrs.constant_value));
    return Status::ok;
}

}  // anonymous namespace

Status pad_cuda(const PadAttributes& attrs, TensorView& output,
                std::span<const TensorView> inputs, const ComputeContext& ctx,
                void* /*workspace*/)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    switch (inputs[0].data_type()) {
    case DataType::f32: return pad_entry<float>(attrs, output, inputs[0], stream);
    case DataType::f16: return pad_entry<__half>(attrs, output, inputs[0], stream);
    case DataType::s8:  return pad_entry<int8_t>(attrs, output, inputs[0], stream);
    case DataType::u8:  return pad_entry<uint8_t>(attrs, output, inputs[0], stream);
    case DataType::s32: return pad_entry<int32_t>(attrs, output, inputs[0], stream);
    case DataType::s64: return pad_entry<int64_t>(attrs, output, inputs[0], stream);
    default:            return Status::unsupported;
    }
}

}  // namespace nnops::backend::cuda
