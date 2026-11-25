/// @file clamp.cu
/// @brief CUDA Clamp — an elementwise min/max with a residual option.
///
/// One block slice per output row, `gridDim.y` over the rows, so both a wide
/// tile ([H, W], few rows) and a deep one (many rows) fill the machine. The
/// bounds are converted once on the host and travel as floats; a row on the
/// 16 B bound takes the vector load/store path, the rest walk scalars.
///
/// f32, f16 and bf16. add_to reads the old output and adds the clamped value,
/// the same order as the CPU route.

#include "cuda_common.cuh"
#include "nnops/ops/clamp.hpp"

namespace nnops::backend::cuda {

namespace {

constexpr int kThreads = 256;

template <class T>
__global__ void clamp_kernel(const T* __restrict__ in, T* __restrict__ out, int64_t rows,
                             int64_t cols, int64_t in_ld, int64_t out_ld, float lo, float hi,
                             bool add_to)
{
    constexpr int VEC = int(16 / sizeof(T));
    for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
        const T* irow = in + r * in_ld;
        T* orow = out + r * out_ld;
        const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        const int64_t stride = int64_t(blockDim.x) * gridDim.x;
        const int64_t vec_end = aligned16(irow, orow) ? (cols / VEC) * VEC : 0;

        for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
            if constexpr (std::is_same_v<T, float>) {
                const float4 v = v4_load(irow + i);
                const float4 o = add_to ? v4_load(orow + i) : make_float4(0, 0, 0, 0);
                float f[4] = {v.x, v.y, v.z, v.w};
                const float a[4] = {o.x, o.y, o.z, o.w};
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    f[k] = fminf(fmaxf(f[k], lo), hi);
                    if (add_to) {
                        f[k] += a[k];
                    }
                }
                v4_store(orow + i, make_float4(f[0], f[1], f[2], f[3]));
            } else {
                const pack8<T> v = v4_load(irow + i);
                const pack8<T> o = add_to ? v4_load(orow + i) : pack8<T>{};
                float f[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    f[k] = fminf(fmaxf(elem_to_float(v.data[k]), lo), hi);
                    if (add_to) {
                        f[k] += elem_to_float(o.data[k]);
                    }
                }
                v4_store(orow + i,
                         pack8_set<T>(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]));
            }
        }
        for (int64_t i = vec_end + tid; i < cols; i += stride) {
            float f = fminf(fmaxf(elem_to_float(irow[i]), lo), hi);
            if (add_to) {
                f += elem_to_float(orow[i]);
            }
            orow[i] = float_to_elem<T>(f);
        }
    }
}

template <class T>
Status clamp_entry(const ClampAttributes& attrs, TensorView& output,
                   std::span<const TensorView> inputs, cudaStream_t stream)
{
    const TensorView& in = inputs[0];
    const int64_t rank = in.rank();
    const int64_t cols = rank > 0 ? in.shape(rank - 1) : 1;
    const int64_t total = in.numel();
    if (total <= 0) {
        return Status::ok;
    }
    const int64_t rows = cols > 0 ? total / cols : 0;
    if (rows <= 0 || cols <= 0) {
        return Status::ok;
    }
    const int64_t in_ld = rank >= 2 ? in.row_stride_elems() : cols;
    const int64_t out_ld = rank >= 2 ? output.row_stride_elems() : cols;

    const dim3 grid(static_cast<unsigned>((cols + kThreads - 1) / kThreads),
                    static_cast<unsigned>(std::min<int64_t>(rows, 65535)));
    clamp_kernel<T><<<grid, kThreads, 0, stream>>>(in.ptr<T>(), output.ptr<T>(), rows, cols,
                                                        in_ld, out_ld, attrs.min_val,
                                                        attrs.max_val, attrs.add_to);
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

} // anonymous namespace

Status clamp_cuda(const ClampAttributes& attrs, TensorView& output,
                  std::span<const TensorView> inputs, const ComputeContext& ctx, void*)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    switch (inputs[0].data_type()) {
    case DataType::f32:
        return clamp_entry<float>(attrs, output, inputs, stream);
    case DataType::f16:
        return clamp_entry<__half>(attrs, output, inputs, stream);
    case DataType::bf16:
        return clamp_entry<__nv_bfloat16>(attrs, output, inputs, stream);
    default:
        return Status::unsupported;
    }
}

} // namespace nnops::backend::cuda
