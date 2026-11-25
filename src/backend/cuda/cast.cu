/// @file cast.cu
/// @brief CUDA dtype conversion (ONNX Cast).
///
/// The conversion rules are the CPU route's, because a graph's output must not
/// depend on where it ran:
///
///   - float -> integer: truncate toward zero and saturate to the destination's
///     range; NaN maps to 0 (CUDA's `static_cast` is undefined out of range, so
///     the clamp is explicit);
///   - integer -> integer through int64, so an s64 index past f32's mantissa
///     survives;
///   - integer -> float widens through int64 then converts;
///   - float -> float rounds to nearest even in the destination format.
///
/// The two float pairs every graph repeats — f32 <-> f16 / bf16 — take an
/// eight-lane path (one 32 B read, one 16 B write per thread); the rest walk a
/// scalar row loop, which is still coalesced on whichever side is contiguous.
/// A pitched source or destination is walked by its own row stride.
///
/// f32, f16, bf16, s32, s64, s8 and u8, every pair.

#include "cuda_common.cuh"
#include "nnops/ops/cast.hpp"

#include <limits>
#include <type_traits>

namespace nnops::backend::cuda {

namespace {

constexpr int kThreads = 256;

/// The one conversion, exactly as the CPU rules state it.
template <class S, class D> __device__ inline D cast_one(S v) noexcept
{
    constexpr bool src_int = std::is_integral_v<S>;
    constexpr bool dst_int = std::is_integral_v<D>;
    if constexpr (src_int && dst_int) {
        const int64_t x = static_cast<int64_t>(v);
        constexpr int64_t lo = static_cast<int64_t>(std::numeric_limits<D>::lowest());
        constexpr int64_t hi = static_cast<int64_t>(std::numeric_limits<D>::max());
        return static_cast<D>(x < lo ? lo : (x > hi ? hi : x));
    } else if constexpr (src_int) {
        return float_to_elem<D>(static_cast<float>(static_cast<int64_t>(v)));
    } else if constexpr (dst_int) {
        const float f = elem_to_float(v);
        if (isnan(f)) {
            return D{0};
        }
        constexpr float lo = static_cast<float>(std::numeric_limits<D>::lowest());
        constexpr float hi = static_cast<float>(std::numeric_limits<D>::max());
        return f <= lo ? std::numeric_limits<D>::lowest() :
               f >= hi ? std::numeric_limits<D>::max() :
                         static_cast<D>(f);
    } else {
        return float_to_elem<D>(elem_to_float(v));
    }
}

template <class S, class D>
__global__ void cast_scalar_kernel(const S* __restrict__ in, D* __restrict__ out, int64_t rows,
                                   int64_t cols, int64_t in_ld, int64_t out_ld)
{
    for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
        const S* irow = in + r * in_ld;
        D* orow = out + r * out_ld;
        const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        const int64_t stride = int64_t(blockDim.x) * gridDim.x;
        for (int64_t i = tid; i < cols; i += stride) {
            orow[i] = cast_one<S, D>(irow[i]);
        }
    }
}

/// f32 -> 2-byte float (f16 or bf16), eight lanes a thread.
template <class D>
__global__ void cast_f32_narrow_kernel(const float* __restrict__ in, D* __restrict__ out,
                                       int64_t rows, int64_t cols, int64_t in_ld, int64_t out_ld)
{
    const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
        const float* irow = in + r * in_ld;
        D* orow = out + r * out_ld;
        const int64_t vec_end = aligned16(irow, orow) ? (cols / 8) * 8 : 0;
        for (int64_t i = tid * 8; i < vec_end; i += stride * 8) {
            const float4 a = v4_load(irow + i);
            const float4 b = v4_load(irow + i + 4);
            v4_store(orow + i, pack8_set<D>(a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w));
        }
        for (int64_t i = vec_end + tid; i < cols; i += stride) {
            orow[i] = float_to_elem<D>(irow[i]);
        }
    }
}

/// 2-byte float -> f32, eight lanes a thread.
template <class S>
__global__ void cast_f32_widen_kernel(const S* __restrict__ in, float* __restrict__ out,
                                      int64_t rows, int64_t cols, int64_t in_ld, int64_t out_ld)
{
    const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
        const S* irow = in + r * in_ld;
        float* orow = out + r * out_ld;
        const int64_t vec_end = aligned16(irow, orow) ? (cols / 8) * 8 : 0;
        for (int64_t i = tid * 8; i < vec_end; i += stride * 8) {
            const pack8<S> v = v4_load(irow + i);
            v4_store(orow + i,
                     make_float4(elem_to_float(v.data[0]), elem_to_float(v.data[1]),
                                 elem_to_float(v.data[2]), elem_to_float(v.data[3])));
            v4_store(orow + i + 4,
                     make_float4(elem_to_float(v.data[4]), elem_to_float(v.data[5]),
                                 elem_to_float(v.data[6]), elem_to_float(v.data[7])));
        }
        for (int64_t i = vec_end + tid; i < cols; i += stride) {
            orow[i] = elem_to_float(irow[i]);
        }
    }
}

/// The per-call geometry both the scalar and the vector kernels share.
struct CastGeom {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t in_ld = 0;
    int64_t out_ld = 0;
    dim3 grid;
};

inline bool cast_geom(const TensorView& in, const TensorView& out, CastGeom& g)
{
    const int64_t rank = in.rank();
    g.cols = rank > 0 ? in.shape(rank - 1) : 1;
    const int64_t total = in.numel();
    if (total <= 0 || g.cols <= 0) {
        return false;
    }
    g.rows = total / g.cols;
    if (g.rows <= 0) {
        return false;
    }
    g.in_ld = rank >= 2 ? in.row_stride_elems() : g.cols;
    g.out_ld = rank >= 2 ? out.row_stride_elems() : g.cols;
    g.grid = dim3(static_cast<unsigned>((g.cols + kThreads - 1) / kThreads),
                  static_cast<unsigned>(std::min<int64_t>(g.rows, 65535)));
    return true;
}

template <class S, class D>
Status cast_run(const TensorView& in, TensorView& out, const CastGeom& g, cudaStream_t stream)
{
    cast_scalar_kernel<S, D><<<g.grid, kThreads, 0, stream>>>(in.ptr<S>(), out.ptr<D>(), g.rows,
                                                              g.cols, g.in_ld, g.out_ld);
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

template <class S>
Status cast_from(DataType dst, const TensorView& in, TensorView& out, const CastGeom& g,
                 cudaStream_t stream)
{
    switch (dst) {
    case DataType::f32:
        if constexpr (!std::is_same_v<S, float>) {
            if constexpr (sizeof(S) == 2 && !std::is_same_v<S, int16_t>) {
                cast_f32_widen_kernel<S><<<g.grid, kThreads, 0, stream>>>(
                    in.ptr<S>(), out.ptr<float>(), g.rows, g.cols, g.in_ld, g.out_ld);
                return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
            }
        }
        return cast_run<S, float>(in, out, g, stream);
    case DataType::f16:
        if constexpr (std::is_same_v<S, float>) {
            cast_f32_narrow_kernel<__half><<<g.grid, kThreads, 0, stream>>>(
                in.ptr<float>(), out.ptr<__half>(), g.rows, g.cols, g.in_ld, g.out_ld);
            return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
        }
        return cast_run<S, __half>(in, out, g, stream);
    case DataType::bf16:
        if constexpr (std::is_same_v<S, float>) {
            cast_f32_narrow_kernel<__nv_bfloat16><<<g.grid, kThreads, 0, stream>>>(
                in.ptr<float>(), out.ptr<__nv_bfloat16>(), g.rows, g.cols, g.in_ld, g.out_ld);
            return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
        }
        return cast_run<S, __nv_bfloat16>(in, out, g, stream);
    case DataType::s32:
        return cast_run<S, int32_t>(in, out, g, stream);
    case DataType::s64:
        return cast_run<S, int64_t>(in, out, g, stream);
    case DataType::s8:
        return cast_run<S, int8_t>(in, out, g, stream);
    case DataType::u8:
        return cast_run<S, uint8_t>(in, out, g, stream);
    default:
        return Status::unsupported;
    }
}

} // anonymous namespace

Status cast_cuda(const CastAttributes& attrs, TensorView& output,
                 std::span<const TensorView> inputs, const ComputeContext& ctx, void*)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    CastGeom g;
    if (!cast_geom(inputs[0], output, g)) {
        return Status::ok;
    }
    switch (inputs[0].data_type()) {
    case DataType::f32:
        return cast_from<float>(attrs.to, inputs[0], output, g, stream);
    case DataType::f16:
        return cast_from<__half>(attrs.to, inputs[0], output, g, stream);
    case DataType::bf16:
        return cast_from<__nv_bfloat16>(attrs.to, inputs[0], output, g, stream);
    case DataType::s32:
        return cast_from<int32_t>(attrs.to, inputs[0], output, g, stream);
    case DataType::s64:
        return cast_from<int64_t>(attrs.to, inputs[0], output, g, stream);
    case DataType::s8:
        return cast_from<int8_t>(attrs.to, inputs[0], output, g, stream);
    case DataType::u8:
        return cast_from<uint8_t>(attrs.to, inputs[0], output, g, stream);
    default:
        return Status::unsupported;
    }
}

} // namespace nnops::backend::cuda
