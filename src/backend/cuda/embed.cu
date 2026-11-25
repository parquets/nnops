/// @file embed.cu
/// @brief CUDA Embed — a row lookup, copy for fp weights and per-row dequant
///        for s8/u8 ones.
///
/// One block per output row, threads over the embedding dimension, so each row
/// is a coalesced 16 B run; the index is clamped to [0, V-1] exactly as the CPU
/// route does. An s8/u8 table dequantizes with the row's own scale and zero
/// point — the same per-row contract, PerToken or PerChannel along axis 0 —
/// straight into the f32/f16 output, with no intermediate buffer. The row's
/// parameters live in the caller's host QuantParams, so they are staged once
/// per call into a stream-ordered device buffer.
///
/// A pitched table is walked by its row stride.

#include "cuda_common.cuh"
#include "nnops/ops/embed.hpp"

namespace nnops::backend::cuda {

namespace {

constexpr int kThreads = 256;

template <class T>
__global__ void embed_copy_kernel(const T* __restrict__ table, const void* __restrict__ indices,
                                  bool i64, T* __restrict__ out, int64_t n_rows, int64_t dim,
                                  int64_t w_ld, int64_t o_ld, int64_t vocab)
{
    const int64_t n = blockIdx.y;
    if (n >= n_rows) {
        return;
    }
    int64_t idx = i64 ? static_cast<const int64_t*>(indices)[n]
                      : static_cast<const int32_t*>(indices)[n];
    idx = idx < 0 ? 0 : (idx >= vocab ? vocab - 1 : idx);
    const T* src = table + idx * w_ld;
    T* dst = out + n * o_ld;

    const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = int64_t(blockDim.x) * gridDim.x;
    constexpr int VEC = int(16 / sizeof(T));
    const int64_t vec_end = aligned16(src, dst) ? (dim / VEC) * VEC : 0;
    for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
        *reinterpret_cast<uint4*>(dst + i) = *reinterpret_cast<const uint4*>(src + i);
    }
    for (int64_t i = vec_end + tid; i < dim; i += stride) {
        dst[i] = src[i];
    }
}

template <class W, class D>
__global__ void embed_dequant_kernel(const W* __restrict__ table, const void* __restrict__ indices,
                                     bool i64, D* __restrict__ out, int64_t n_rows, int64_t dim,
                                     int64_t o_ld, int64_t vocab, const float* __restrict__ scale,
                                     const int32_t* __restrict__ zp)
{
    const int64_t n = blockIdx.y;
    if (n >= n_rows) {
        return;
    }
    int64_t idx = i64 ? static_cast<const int64_t*>(indices)[n]
                      : static_cast<const int32_t*>(indices)[n];
    idx = idx < 0 ? 0 : (idx >= vocab ? vocab - 1 : idx);
    const W* src = table + idx * dim;
    D* dst = out + n * o_ld;
    const float s = scale[idx];
    const float z = zp != nullptr ? static_cast<float>(zp[idx]) : 0.0f;

    const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t i = tid; i < dim; i += stride) {
        dst[i] = float_to_elem<D>((static_cast<float>(src[i]) - z) * s);
    }
}

/// A stream-ordered staging buffer for the row parameters (host -> device).
struct Stage {
    void* ptr = nullptr;
    bool async = false;

    bool alloc(int64_t bytes, cudaStream_t stream)
    {
        if (bytes <= 0) {
            return true;
        }
        if (cudaMallocAsync(&ptr, size_t(bytes), stream) == cudaSuccess) {
            async = true;
            return true;
        }
        cudaGetLastError();
        if (cudaMalloc(&ptr, size_t(bytes)) == cudaSuccess) {
            return true;
        }
        cudaGetLastError();
        return false;
    }
    bool upload(const void* src, int64_t bytes, cudaStream_t stream)
    {
        if (bytes <= 0) {
            return true;
        }
        return cudaMemcpyAsync(ptr, src, size_t(bytes), cudaMemcpyHostToDevice, stream) ==
               cudaSuccess;
    }
    void release(cudaStream_t stream)
    {
        if (ptr == nullptr) {
            return;
        }
        if (async) {
            if (cudaFreeAsync(ptr, stream) == cudaSuccess) {
                ptr = nullptr;
                return;
            }
            cudaGetLastError();
        }
        cudaFree(ptr);
        ptr = nullptr;
    }
};

template <class T>
Status embed_fp(const TensorView& weight, const TensorView& indices, TensorView& output,
                cudaStream_t stream)
{
    const int64_t V = weight.shape(0), dim = weight.shape(1);
    const int64_t n_rows = indices.numel();
    if (n_rows <= 0 || dim <= 0 || V <= 0) {
        return Status::ok;
    }
    const int64_t o_ld = output.row_stride_elems();
    const bool i64 = indices.data_type() == DataType::s64;
    const dim3 grid(static_cast<unsigned>((dim + kThreads - 1) / kThreads),
                    static_cast<unsigned>(std::min<int64_t>(n_rows, 65535)));
    embed_copy_kernel<T><<<grid, kThreads, 0, stream>>>(
        weight.ptr<T>(), indices.ptr<void>(), i64, output.ptr<T>(), n_rows, dim,
        weight.row_stride_elems(), o_ld, V);
    // Rows past the 16-bit grid axis: one more pass over the remainder.
    for (int64_t r0 = 65535; r0 < n_rows; r0 += 65535) {
        const int64_t take = std::min<int64_t>(65535, n_rows - r0);
        const char* idx = static_cast<const char*>(indices.ptr<void>()) + r0 * (i64 ? 8 : 4);
        embed_copy_kernel<T><<<dim3(grid.x, unsigned(take)), kThreads, 0, stream>>>(
            weight.ptr<T>(), idx, i64, output.ptr<T>() + r0 * o_ld, take, dim,
            weight.row_stride_elems(), o_ld, V);
    }
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

template <class W, class D>
Status embed_int(const TensorView& weight, const TensorView& indices, TensorView& output,
                 cudaStream_t stream)
{
    const int64_t V = weight.shape(0), dim = weight.shape(1);
    const int64_t n_rows = indices.numel();
    if (n_rows <= 0 || dim <= 0 || V <= 0) {
        return Status::ok;
    }
    const QuantParams& qp = weight.quant_params();
    if (qp.scale_data == nullptr) {
        return Status::unsupported;
    }

    Stage scale;
    Stage zp;
    if (!scale.alloc(V * 4, stream) || !zp.alloc(V * 4, stream)) {
        return Status::out_of_memory;
    }
    const bool ok = scale.upload(qp.scale_data, V * 4, stream) &&
                    (qp.zero_point_data == nullptr ||
                     zp.upload(qp.zero_point_data, V * 4, stream));
    if (!ok) {
        scale.release(stream);
        zp.release(stream);
        return Status::internal;
    }

    const int64_t o_ld = output.row_stride_elems();
    const bool i64 = indices.data_type() == DataType::s64;
    const int64_t rows_per_launch = 65535;
    for (int64_t r0 = 0; r0 < n_rows; r0 += rows_per_launch) {
        const int64_t take = std::min<int64_t>(rows_per_launch, n_rows - r0);
        const void* idx = i64
                              ? static_cast<const void*>(static_cast<const int64_t*>(
                                    indices.ptr<void>()) + r0)
                              : static_cast<const void*>(static_cast<const int32_t*>(
                                    indices.ptr<void>()) + r0);
        const dim3 grid(static_cast<unsigned>((dim + kThreads - 1) / kThreads),
                        static_cast<unsigned>(take));
        embed_dequant_kernel<W, D><<<grid, kThreads, 0, stream>>>(
            weight.ptr<W>(), idx, i64, output.ptr<D>() + r0 * o_ld, take, dim, o_ld, V,
            static_cast<const float*>(scale.ptr),
            zp.ptr != nullptr ? static_cast<const int32_t*>(zp.ptr) : nullptr);
    }

    const Status st = cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
    scale.release(stream);
    zp.release(stream);
    return st;
}

} // anonymous namespace

Status embed_cuda(const EmbedAttributes&, TensorView& output,
                  std::span<const TensorView> inputs, const ComputeContext& ctx, void*)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    const TensorView& weight = inputs[0];
    const TensorView& indices = inputs[1];
    switch (weight.data_type()) {
    case DataType::f32:
        return embed_fp<float>(weight, indices, output, stream);
    case DataType::f16:
        return embed_fp<__half>(weight, indices, output, stream);
    case DataType::bf16:
        return embed_fp<__nv_bfloat16>(weight, indices, output, stream);
    case DataType::s8:
        return output.data_type() == DataType::f16
                   ? embed_int<int8_t, __half>(weight, indices, output, stream)
               : output.data_type() == DataType::bf16
                   ? embed_int<int8_t, __nv_bfloat16>(weight, indices, output, stream)
                   : embed_int<int8_t, float>(weight, indices, output, stream);
    case DataType::u8:
        return output.data_type() == DataType::f16
                   ? embed_int<uint8_t, __half>(weight, indices, output, stream)
               : output.data_type() == DataType::bf16
                   ? embed_int<uint8_t, __nv_bfloat16>(weight, indices, output, stream)
                   : embed_int<uint8_t, float>(weight, indices, output, stream);
    default:
        return Status::unsupported;
    }
}

} // namespace nnops::backend::cuda
