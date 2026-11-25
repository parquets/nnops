/// @file quant_linear.cu
/// @brief CUDA QuantizeLinear / DequantizeLinear.
///
/// The tensor is a flat row-major array and the granularity picks how its rows
/// are grouped:
///
///   PerTensor  one scale for everything (M = 1, N = numel)
///   PerToken   one scale per innermost row (M = numel/last, N = last)
///   PerChannel one scale per index along `axis`, broadcast over the rest
///
/// which is the CPU route's decomposition, so the two agree on which element
/// reads which parameter. One block slice covers a row and `gridDim.y` walks
/// the rows; the scale and zero point are ordinary device inputs (ONNX passes
/// them as operands), so nothing is staged.
///
/// Rows past the 16-bit grid axis are looped inside the kernel. Rounding is
/// round-to-nearest-even (`rintf`), then the zero point, then a saturating
/// clamp to the integer type's range — ONNX's order and the CPU kernels'.

#include "cuda_common.cuh"
#include "nnops/ops/quant_linear.hpp"

#include <limits>

namespace nnops::backend::cuda {

namespace {

constexpr int kThreads = 256;

/// Which parameter a row reads: 0 a single one, 1 its own, 2 its channel slot
/// (`row % D` inside each outer group).
struct QGeom {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t channels = 1;
    int mode = 0;
};

inline bool q_geom(const QuantLinearAttributes& attrs, const TensorView& x,
                   const TensorView& scale, QGeom& g)
{
    const int64_t rank = x.rank();
    const int64_t numel = x.numel();
    if (numel <= 0) {
        return false;
    }
    const QuantGranularity gran = resolve_quant_granularity(scale, attrs.axis);
    switch (gran) {
    case QuantGranularity::PerTensor:
        g.rows = 1;
        g.cols = numel;
        g.mode = 0;
        break;
    case QuantGranularity::PerToken: {
        const int64_t last = x.shape(rank - 1);
        if (last <= 0) {
            return false;
        }
        g.rows = numel / last;
        g.cols = last;
        g.mode = 1;
        break;
    }
    case QuantGranularity::PerChannel: {
        int64_t axis = attrs.axis;
        if (axis < 0) {
            axis += rank;
        }
        g.channels = x.shape(axis);
        g.cols = 1;
        for (int64_t d = axis + 1; d < rank; ++d) {
            g.cols *= x.shape(d);
        }
        int64_t outer = 1;
        for (int64_t d = 0; d < axis; ++d) {
            outer *= x.shape(d);
        }
        g.rows = outer * g.channels;
        g.mode = 2;
        break;
    }
    default:
        return false;
    }
    return g.rows > 0 && g.cols > 0;
}

template <class S, class D, class Z>
__global__ void quantize_kernel(const S* __restrict__ x, D* __restrict__ y,
                                const float* __restrict__ scale, const Z* __restrict__ zp,
                                QGeom g, bool zp_present)
{
    constexpr int LO = int(std::numeric_limits<D>::lowest());
    constexpr int HI = int(std::numeric_limits<D>::max());
    const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t r = blockIdx.y; r < g.rows; r += gridDim.y) {
        const int64_t p = g.mode == 2 ? (r % g.channels) : (g.mode == 1 ? r : 0);
        const float s = scale[p];
        const float z = zp_present ? static_cast<float>(zp[p]) : 0.0f;
        const S* xrow = x + r * g.cols;
        D* yrow = y + r * g.cols;
        for (int64_t i = tid; i < g.cols; i += stride) {
            float q = rintf(elem_to_float(xrow[i]) / s) + z;
            q = fminf(fmaxf(q, float(LO)), float(HI));
            yrow[i] = static_cast<D>(q);
        }
    }
}

template <class S, class D, class Z>
__global__ void dequantize_kernel(const S* __restrict__ x, D* __restrict__ y,
                                  const float* __restrict__ scale, const Z* __restrict__ zp,
                                  QGeom g, bool zp_present)
{
    const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t r = blockIdx.y; r < g.rows; r += gridDim.y) {
        const int64_t p = g.mode == 2 ? (r % g.channels) : (g.mode == 1 ? r : 0);
        const float s = scale[p];
        const float z = zp_present ? static_cast<float>(zp[p]) : 0.0f;
        const S* xrow = x + r * g.cols;
        D* yrow = y + r * g.cols;
        for (int64_t i = tid; i < g.cols; i += stride) {
            yrow[i] = float_to_elem<D>((static_cast<float>(xrow[i]) - z) * s);
        }
    }
}

inline dim3 q_grid(const QGeom& g) noexcept
{
    return dim3(static_cast<unsigned>((g.cols + kThreads - 1) / kThreads),
                static_cast<unsigned>(std::min<int64_t>(g.rows, 65535)));
}

/// The zero point may be absent (an empty view) or one of the two integer
/// types, independently of the output's type.
template <class S, class D>
Status quantize_zp(const QuantLinearAttributes& attrs, const TensorView& x,
                   const TensorView& scale, const TensorView& zp, TensorView& output,
                   const QGeom& g, cudaStream_t stream)
{
    const bool has_zp = !zp.is_empty();
    const dim3 grid = q_grid(g);
    if (!has_zp || zp.data_type() == DataType::s8) {
        quantize_kernel<S, D, int8_t><<<grid, kThreads, 0, stream>>>(
            x.ptr<S>(), output.ptr<D>(), scale.ptr<float>(),
            has_zp ? zp.ptr<int8_t>() : nullptr, g, has_zp);
    } else {
        quantize_kernel<S, D, uint8_t><<<grid, kThreads, 0, stream>>>(
            x.ptr<S>(), output.ptr<D>(), scale.ptr<float>(),
            has_zp ? zp.ptr<uint8_t>() : nullptr, g, has_zp);
    }
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

template <class S, class D>
Status dequantize_zp(const QuantLinearAttributes& attrs, const TensorView& x,
                     const TensorView& scale, const TensorView& zp, TensorView& output,
                     const QGeom& g, cudaStream_t stream)
{
    (void)attrs;
    const bool has_zp = !zp.is_empty();
    const dim3 grid = q_grid(g);
    if (!has_zp || zp.data_type() == DataType::s8) {
        dequantize_kernel<S, D, int8_t><<<grid, kThreads, 0, stream>>>(
            x.ptr<S>(), output.ptr<D>(), scale.ptr<float>(),
            has_zp ? zp.ptr<int8_t>() : nullptr, g, has_zp);
    } else {
        dequantize_kernel<S, D, uint8_t><<<grid, kThreads, 0, stream>>>(
            x.ptr<S>(), output.ptr<D>(), scale.ptr<float>(),
            has_zp ? zp.ptr<uint8_t>() : nullptr, g, has_zp);
    }
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

template <class S>
Status quantize_type(const QuantLinearAttributes& attrs, std::span<const TensorView> inputs,
                     TensorView& output, const QGeom& g, cudaStream_t stream)
{
    const TensorView& zp = inputs.size() > 2 ? inputs[2] : TensorView{};
    if (output.data_type() == DataType::s8) {
        return quantize_zp<S, int8_t>(attrs, inputs[0], inputs[1], zp, output, g, stream);
    }
    if (output.data_type() == DataType::u8) {
        return quantize_zp<S, uint8_t>(attrs, inputs[0], inputs[1], zp, output, g, stream);
    }
    return Status::unsupported;
}

template <class S>
Status dequantize_type(const QuantLinearAttributes& attrs, std::span<const TensorView> inputs,
                       TensorView& output, const QGeom& g, cudaStream_t stream)
{
    const TensorView& zp = inputs.size() > 2 ? inputs[2] : TensorView{};
    switch (output.data_type()) {
    case DataType::f32:
        return dequantize_zp<S, float>(attrs, inputs[0], inputs[1], zp, output, g, stream);
    case DataType::f16:
        return dequantize_zp<S, __half>(attrs, inputs[0], inputs[1], zp, output, g, stream);
    case DataType::bf16:
        return dequantize_zp<S, __nv_bfloat16>(attrs, inputs[0], inputs[1], zp, output, g,
                                               stream);
    default:
        return Status::unsupported;
    }
}

} // anonymous namespace

Status quantize_linear_cuda(const QuantLinearAttributes& attrs, TensorView& output,
                            std::span<const TensorView> inputs, const ComputeContext& ctx, void*)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    QGeom g;
    if (!q_geom(attrs, inputs[0], inputs[1], g)) {
        return Status::ok;
    }
    switch (inputs[0].data_type()) {
    case DataType::f32:
        return quantize_type<float>(attrs, inputs, output, g, stream);
    case DataType::f16:
        return quantize_type<__half>(attrs, inputs, output, g, stream);
    case DataType::bf16:
        return quantize_type<__nv_bfloat16>(attrs, inputs, output, g, stream);
    default:
        return Status::unsupported;
    }
}

Status dequantize_linear_cuda(const QuantLinearAttributes& attrs, TensorView& output,
                              std::span<const TensorView> inputs, const ComputeContext& ctx,
                              void*)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    QGeom g;
    if (!q_geom(attrs, inputs[0], inputs[1], g)) {
        return Status::ok;
    }
    switch (inputs[0].data_type()) {
    case DataType::s8:
        return dequantize_type<int8_t>(attrs, inputs, output, g, stream);
    case DataType::u8:
        return dequantize_type<uint8_t>(attrs, inputs, output, g, stream);
    default:
        return Status::unsupported;
    }
}

} // namespace nnops::backend::cuda
