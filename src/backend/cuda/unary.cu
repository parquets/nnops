/// @file unary.cu
/// @brief CUDA implementation of element-wise unary math operations.
///
/// Element-wise operations — flat grid-stride loop over all elements,
/// plus a row-by-row kernel for multi-dimensional pitch-aware tensors.
/// All 9 operations: Exp, Log, Sin, Cos, Tan, Tanh, Abs, Neg, Sqrt.
///
/// f32, f16 and bf16 run the element functions above. The quantized route takes
/// an s8/u8 input and output and fuses dequantize → op → requantize around the
/// same device functions, walking dense rows of the last dim as the CPU's fused
/// transform does. add_to and the integer-valued Round/Ceil/Floor have no
/// quantized meaning and are refused, as on the CPU.
///
/// Workspace: not used.

#include "cuda_common.cuh"
#include "quant_elementwise.cuh"
#include "nnops/ops/unary.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops::backend::cuda {

// ============================================================
// Device unary functions (compute a single element)
// ============================================================

__device__ inline float unary_exp(float x) { return expf(x); }
__device__ inline float unary_log(float x) { return logf(x); }
__device__ inline float unary_sin(float x) { return sinf(x); }
__device__ inline float unary_cos(float x) { return cosf(x); }
__device__ inline float unary_tan(float x) { return tanf(x); }
__device__ inline float unary_tanh(float x) { return tanhf(x); }
__device__ inline float unary_abs(float x) { return fabsf(x); }
__device__ inline float unary_neg(float x) { return -x; }
__device__ inline float unary_sqrt(float x) { return sqrtf(x); }
__device__ inline float unary_erf(float x) { return erff(x); }
__device__ inline float unary_round(float x) { return rintf(x); }
__device__ inline float unary_ceil(float x) { return ceilf(x); }
__device__ inline float unary_floor(float x) { return floorf(x); }
__device__ inline float unary_recip(float x) { return 1.0f / x; }
__device__ inline float unary_sign(float x)
{
    return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
}
__device__ inline float unary_exp2(float x) { return exp2f(x); }
__device__ inline float unary_log2(float x) { return log2f(x); }
__device__ inline float unary_log10(float x) { return log10f(x); }
__device__ inline float unary_asin(float x) { return asinf(x); }
__device__ inline float unary_acos(float x) { return acosf(x); }
__device__ inline float unary_atan(float x) { return atanf(x); }
__device__ inline float unary_sinh(float x) { return sinhf(x); }
__device__ inline float unary_cosh(float x) { return coshf(x); }
__device__ inline float unary_asinh(float x) { return asinhf(x); }
__device__ inline float unary_acosh(float x) { return acoshf(x); }
__device__ inline float unary_atanh(float x) { return atanhf(x); }
__device__ inline float unary_erfc(float x) { return erfcf(x); }

// ============================================================
// Function pointer type for unary
// ============================================================

using UnaryFn = float (*)(float);

__device__ UnaryFn get_unary_fn(UnaryType type)
{
    switch (type) {
    case UnaryType::Exp:
        return unary_exp;
    case UnaryType::Log:
        return unary_log;
    case UnaryType::Sin:
        return unary_sin;
    case UnaryType::Cos:
        return unary_cos;
    case UnaryType::Tan:
        return unary_tan;
    case UnaryType::Tanh:
        return unary_tanh;
    case UnaryType::Abs:
        return unary_abs;
    case UnaryType::Neg:
        return unary_neg;
    case UnaryType::Sqrt:
        return unary_sqrt;
    case UnaryType::Erf:
        return unary_erf;
    case UnaryType::Round:
        return unary_round;
    case UnaryType::Ceil:
        return unary_ceil;
    case UnaryType::Floor:
        return unary_floor;
    case UnaryType::Recip:
        return unary_recip;
    case UnaryType::Sign:
        return unary_sign;
    case UnaryType::Exp2:
        return unary_exp2;
    case UnaryType::Log2:
        return unary_log2;
    case UnaryType::Log10:
        return unary_log10;
    case UnaryType::Asin:
        return unary_asin;
    case UnaryType::Acos:
        return unary_acos;
    case UnaryType::Atan:
        return unary_atan;
    case UnaryType::Sinh:
        return unary_sinh;
    case UnaryType::Cosh:
        return unary_cosh;
    case UnaryType::Asinh:
        return unary_asinh;
    case UnaryType::Acosh:
        return unary_acosh;
    case UnaryType::Atanh:
        return unary_atanh;
    case UnaryType::Erfc:
        return unary_erfc;
    }
    return unary_exp;
}

// ============================================================
// Main unary CUDA kernel — row-by-row, pitch-aware
// ============================================================

template <typename T>
__global__ void unary_kernel(const T* __restrict__ input, T* __restrict__ output, int64_t num_rows,
                             int64_t last_dim, int64_t in_row_stride, int64_t out_row_stride,
                             UnaryType type, bool add_to)
{
    const UnaryFn fn = get_unary_fn(type);
    constexpr int VEC = sizeof(T) == 2 ? 8 : 4; // 128-bit / sizeof(T)

    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    if (row >= num_rows)
        return;

    const T* in_row = input + row * in_row_stride;
    T* out_row = output + row * out_row_stride;

    // An unaligned row walks the scalar loop instead of the 16 B one.
    const int64_t vec_end = aligned16(in_row, out_row) ? (last_dim / VEC) * VEC : 0;

    // 128-bit vectorized loop: VEC elements per iteration
    if constexpr (sizeof(T) == 4) {
        // f32 path: float4 = 4 floats
        for (int64_t i = tid * VEC; i < vec_end; i += blockDim.x * VEC) {
            float4 vin = v4_load(&in_row[i]);
            float r0 = fn(vin.x), r1 = fn(vin.y), r2 = fn(vin.z), r3 = fn(vin.w);
            if (add_to) {
                float4 vout = v4_load(&out_row[i]);
                v4_store(&out_row[i],
                         make_float4(vout.x + r0, vout.y + r1, vout.z + r2, vout.w + r3));
            } else {
                v4_store(&out_row[i], make_float4(r0, r1, r2, r3));
            }
        }
    } else {
        // 2-byte path: pack8<T> = 8 elements
        for (int64_t i = tid * VEC; i < vec_end; i += blockDim.x * VEC) {
            pack8<T> vin = v4_load(&in_row[i]);
            float r0 = fn(elem_to_float(vin.data[0]));
            float r1 = fn(elem_to_float(vin.data[1]));
            float r2 = fn(elem_to_float(vin.data[2]));
            float r3 = fn(elem_to_float(vin.data[3]));
            float r4 = fn(elem_to_float(vin.data[4]));
            float r5 = fn(elem_to_float(vin.data[5]));
            float r6 = fn(elem_to_float(vin.data[6]));
            float r7 = fn(elem_to_float(vin.data[7]));
            if (add_to) {
                v4_store(&out_row[i],
                         pack8_add<T>(v4_load(&out_row[i]), r0, r1, r2, r3, r4, r5, r6, r7));
            } else {
                v4_store(&out_row[i], pack8_set<T>(r0, r1, r2, r3, r4, r5, r6, r7));
            }
        }
    }

    // Scalar tail for remaining elements
    for (int64_t i = vec_end + tid; i < last_dim; i += blockDim.x) {
        float x = s_load(&in_row[i]);
        float rv = fn(x);
        if (add_to) {
            s_store(&out_row[i], s_load(&out_row[i]) + rv);
        } else {
            s_store(&out_row[i], rv);
        }
    }
}

// ============================================================
// Flat kernel for 1D / scalar tensors
// ============================================================

template <typename T>
__global__ void unary_flat_kernel(const T* __restrict__ input, T* __restrict__ output,
                                  int64_t total, UnaryType type, bool add_to)
{
    const UnaryFn fn = get_unary_fn(type);
    constexpr int VEC = sizeof(T) == 2 ? 8 : 4; // 128-bit / sizeof(T)

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int64_t vec_end = aligned16(input, output) ? (total / VEC) * VEC : 0;

    // 128-bit vectorized grid-stride loop
    if constexpr (sizeof(T) == 4) {
        // f32 path: float4 = 4 floats
        for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
            float4 vin = v4_load(&input[i]);
            float r0 = fn(vin.x), r1 = fn(vin.y), r2 = fn(vin.z), r3 = fn(vin.w);
            if (add_to) {
                float4 vout = v4_load(&output[i]);
                v4_store(&output[i],
                         make_float4(vout.x + r0, vout.y + r1, vout.z + r2, vout.w + r3));
            } else {
                v4_store(&output[i], make_float4(r0, r1, r2, r3));
            }
        }
    } else {
        // 2-byte path: pack8<T> = 8 elements
        for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
            pack8<T> vin = v4_load(&input[i]);
            float r0 = fn(elem_to_float(vin.data[0]));
            float r1 = fn(elem_to_float(vin.data[1]));
            float r2 = fn(elem_to_float(vin.data[2]));
            float r3 = fn(elem_to_float(vin.data[3]));
            float r4 = fn(elem_to_float(vin.data[4]));
            float r5 = fn(elem_to_float(vin.data[5]));
            float r6 = fn(elem_to_float(vin.data[6]));
            float r7 = fn(elem_to_float(vin.data[7]));
            if (add_to) {
                v4_store(&output[i],
                         pack8_add<T>(v4_load(&output[i]), r0, r1, r2, r3, r4, r5, r6, r7));
            } else {
                v4_store(&output[i], pack8_set<T>(r0, r1, r2, r3, r4, r5, r6, r7));
            }
        }
    }

    // Scalar tail for remaining elements
    for (int64_t i = vec_end + tid; i < total; i += stride) {
        float x = s_load(&input[i]);
        float rv = fn(x);
        if (add_to) {
            s_store(&output[i], s_load(&output[i]) + rv);
        } else {
            s_store(&output[i], rv);
        }
    }
}

// ============================================================
// Host-side launcher
// ============================================================

template <typename T>
void unary_cuda_impl(const UnaryAttributes& attrs, TensorView& output,
                     std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t total = input.numel();
    if (total == 0)
        return;

    const int64_t rank = input.rank();
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    if (rank <= 1) {
        // Flat tensor — use flat kernel with grid-stride loop
        const int block_size = kCudaBlockSize;
        const int grid_size =
            std::min(static_cast<int>(ceil_div(total, static_cast<int64_t>(block_size))), 65535);
        unary_flat_kernel<<<grid_size, block_size, 0, stream>>>(input.ptr<T>(), output.ptr<T>(),
                                                                total, attrs.type, attrs.add_to);
    } else {
        // Row-by-row processing, respecting pitch. An NHWC row is one
        // spatial line — W pixels × C channels.
        const int64_t last_dim = (rank >= 3 && is_channel_last(input.layout())) ?
                                     input.shape(rank - 2) * input.shape(rank - 1) :
                                     input.shape(rank - 1);
        const int64_t num_rows = input.total_rows();
        const int64_t in_row_stride = input.row_stride_elems();
        const int64_t out_row_stride = output.row_stride_elems();

        const int block_size = std::min(static_cast<int>(last_dim), kCudaBlockSize);

        unary_kernel<<<num_rows, block_size, 0, stream>>>(input.ptr<T>(), output.ptr<T>(), num_rows,
                                                          last_dim, in_row_stride, out_row_stride,
                                                          attrs.type, attrs.add_to);
    }
}

// ============================================================
// Quantized route — dequantize, apply, requantize in one pass
// ============================================================

template <class InT, class OutT>
__global__ void unary_quant_kernel(const InT* __restrict__ input, OutT* __restrict__ output,
                                   int64_t rows, int64_t cols, qelem::ParamRef qin,
                                   qelem::ParamRef qout, UnaryType type)
{
    constexpr int VEC = 16;
    const UnaryFn fn = get_unary_fn(type);

    for (int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
        const qelem::RowParam pin = qelem::row_param(qin, row);
        const qelem::RowParam pout = qelem::row_param(qout, row);
        const InT* in_row = input + row * cols;
        OutT* out_row = output + row * cols;

        const int64_t vec_end = aligned16(in_row, out_row) ? (cols / VEC) * VEC : 0;
        const int64_t vstep = int64_t(gridDim.x) * blockDim.x * VEC;
        for (int64_t i = int64_t(blockIdx.x) * blockDim.x * VEC + threadIdx.x * VEC; i < vec_end;
             i += vstep) {
            const qelem::pack16<InT> v = *reinterpret_cast<const qelem::pack16<InT>*>(&in_row[i]);
            qelem::pack16<OutT> r;
#pragma unroll
            for (int k = 0; k < VEC; ++k) {
                r.data[k] = qelem::requant<OutT>(fn(qelem::dequant(v.data[k], pin)), pout);
            }
            *reinterpret_cast<qelem::pack16<OutT>*>(&out_row[i]) = r;
        }

        const int64_t sstep = int64_t(gridDim.x) * blockDim.x;
        for (int64_t i = vec_end + int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < cols;
             i += sstep) {
            out_row[i] = qelem::requant<OutT>(fn(qelem::dequant(in_row[i], pin)), pout);
        }
    }
}

template <class InT, class OutT>
Status unary_quant_cuda(const UnaryAttributes& attrs, TensorView& output, const TensorView& input,
                        cudaStream_t stream)
{
    if (attrs.add_to) {
        return Status::unsupported;
    }
    // Round/Ceil/Floor have no quantized meaning; refuse rather than fall through.
    if (attrs.type == UnaryType::Round || attrs.type == UnaryType::Ceil ||
        attrs.type == UnaryType::Floor) {
        return Status::unsupported;
    }
    const int64_t total = input.numel();
    if (total == 0) {
        return Status::ok;
    }
    const int64_t cols = input.shape(input.rank() - 1);
    if (cols <= 0) {
        return Status::invalid_argument;
    }
    const int64_t rows = total / cols;

    qelem::Stage in_scale_s, in_zp_s, out_scale_s, out_zp_s;
    qelem::ParamRef qin, qout;
    Status st = qelem::stage_qparams(input.quant_params(), rows, stream, in_scale_s, in_zp_s, qin);
    if (is_ok(st)) {
        st = qelem::stage_qparams(output.quant_params(), rows, stream, out_scale_s, out_zp_s, qout);
    }
    if (is_ok(st)) {
        constexpr int VEC = 16;
        const int block_size = kCudaBlockSize;
        const int64_t col_tiles = ceil_div(cols, int64_t(block_size) * VEC);
        const dim3 grid(static_cast<unsigned>(std::min<int64_t>(col_tiles, 65535)),
                        static_cast<unsigned>(std::min<int64_t>(rows, 65535)));
        unary_quant_kernel<InT, OutT><<<grid, block_size, 0, stream>>>(
            input.ptr<InT>(), output.ptr<OutT>(), rows, cols, qin, qout, attrs.type);
        st = cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
    }
    in_scale_s.release(stream);
    in_zp_s.release(stream);
    out_scale_s.release(stream);
    out_zp_s.release(stream);
    return st;
}

template <class InT>
Status unary_quant_pair(const UnaryAttributes& attrs, TensorView& output,
                        std::span<const TensorView> inputs, cudaStream_t stream)
{
    if (output.data_type() == DataType::s8) {
        return unary_quant_cuda<InT, int8_t>(attrs, output, inputs[0], stream);
    }
    return unary_quant_cuda<InT, uint8_t>(attrs, output, inputs[0], stream);
}

// ============================================================
// Entry point
// ============================================================

Status unary_cuda(const UnaryAttributes& attrs, TensorView& output,
                  std::span<const TensorView> inputs, const ComputeContext& ctx,
                  void* /*workspace*/)
{
    const DataType dtype = inputs[0].data_type();
    const DataType out_dtype = output.data_type();
    const bool in_int = dtype == DataType::s8 || dtype == DataType::u8;
    const bool out_int = out_dtype == DataType::s8 || out_dtype == DataType::u8;
    if (in_int || out_int) {
        if (!in_int || !out_int) {
            return Status::invalid_argument;
        }
        const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
        if (dtype == DataType::s8) {
            return unary_quant_pair<int8_t>(attrs, output, inputs, stream);
        }
        return unary_quant_pair<uint8_t>(attrs, output, inputs, stream);
    }
    switch (dtype) {
    case DataType::f32:
        unary_cuda_impl<float>(attrs, output, inputs, ctx);
        return Status::ok;
    case DataType::f16:
        unary_cuda_impl<__half>(attrs, output, inputs, ctx);
        return Status::ok;
    case DataType::bf16:
        unary_cuda_impl<__nv_bfloat16>(attrs, output, inputs, ctx);
        return Status::ok;
    default:
        return Status::unsupported;
    }
    return Status::ok;
}

} // namespace nnops::backend::cuda
