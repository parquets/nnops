/// @file activation.cu
/// @brief CUDA implementation of activation functions.
///
/// Element-wise operations — flat grid, each thread strides over the tensor.
/// All 8 activation types supported: Relu, LeakyRelu, Sigmoid, Tanh,
/// Gelu, Silu (Swish), HardSwish, Elu.
///
/// The op moves two elements per unit of work and does one scalar function on
/// each, so it is a bandwidth kernel and the loads are 128-bit lanes: a 2-byte
/// dtype fills a warp's 128 B transaction with 8 per thread, a 4-byte one with
/// 4. The row base has to sit on the 16 B bound for that, so a row that does not
/// — and the tail of one that does — walks the scalar loop.
///
/// f32, f16 and bf16 run the element functions above. The quantized route takes
/// an s8/u8 input and output and fuses the two conversions around the same
/// device functions: dequantize each element with its row's parameters, apply
/// the activation in f32, requantize with the output's. It walks the tensor as
/// dense rows of the last dim, the decomposition the CPU's fused transform
/// uses, and refuses add_to (there is no float destination to accumulate into).
///
/// Workspace: not used.

#include "cuda_common.cuh"
#include "quant_elementwise.cuh"
#include "nnops/ops/activation.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops::backend::cuda {

// ============================================================
// Device activation functions (compute a single element)
// ============================================================

__device__ inline float activation_relu(float x, float, float) { return fmaxf(x, 0.0f); }

__device__ inline float activation_leaky_relu(float x, float alpha, float)
{
    return (x > 0.0f) ? x : alpha * x;
}

__device__ inline float activation_sigmoid(float x, float, float)
{
    return 1.0f / (1.0f + expf(-x));
}

__device__ inline float activation_tanh(float x, float, float) { return tanhf(x); }

__device__ inline float activation_gelu(float x, float, float)
{
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

__device__ inline float activation_silu(float x, float, float) { return x / (1.0f + expf(-x)); }

__device__ inline float activation_hard_swish(float x, float, float beta)
{
    float relu6 = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * relu6 * (beta / 6.0f);
}

__device__ inline float activation_elu(float x, float alpha, float)
{
    return (x > 0.0f) ? x : alpha * (expf(x) - 1.0f);
}

__device__ inline float activation_hard_sigmoid(float x, float alpha, float beta)
{
    return fminf(fmaxf(alpha * x + beta, 0.0f), 1.0f);
}

__device__ inline float activation_softplus(float x, float, float beta)
{
    const float z = beta * x;
    return (fmaxf(z, 0.0f) + log1pf(expf(-fabsf(z)))) / beta;
}

__device__ inline float activation_selu(float x, float alpha, float scale)
{
    return scale * ((x > 0.0f) ? x : alpha * (expf(x) - 1.0f));
}

__device__ inline float activation_mish(float x, float, float)
{
    return x * tanhf(fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x))));
}

__device__ inline float activation_gelu_exact(float x, float, float)
{
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

__device__ inline float activation_quick_gelu(float x, float alpha, float)
{
    return x / (1.0f + expf(-alpha * x));
}

// ============================================================
// Function pointer type for activation
// ============================================================

using ActivationFn = float (*)(float, float, float);

__device__ ActivationFn get_activation_fn(ActivationType type)
{
    switch (type) {
    case ActivationType::Relu:
        return activation_relu;
    case ActivationType::LeakyRelu:
        return activation_leaky_relu;
    case ActivationType::Sigmoid:
        return activation_sigmoid;
    case ActivationType::Tanh:
        return activation_tanh;
    case ActivationType::Gelu:
        return activation_gelu;
    case ActivationType::Silu:
        return activation_silu;
    case ActivationType::HardSwish:
        return activation_hard_swish;
    case ActivationType::Elu:
        return activation_elu;
    case ActivationType::HardSigmoid:
        return activation_hard_sigmoid;
    case ActivationType::Softplus:
        return activation_softplus;
    case ActivationType::Selu:
        return activation_selu;
    case ActivationType::Mish:
        return activation_mish;
    case ActivationType::GeluExact:
        return activation_gelu_exact;
    case ActivationType::QuickGelu:
        return activation_quick_gelu;
    case ActivationType::PRelu:
        return activation_relu; // rejected on entry
    }
    return activation_relu;
}

// ============================================================
// Main activation CUDA kernel — row-by-row, pitch-aware
// ============================================================

/// One 128-bit lane per thread step. An unaligned row (a 2-byte dtype with an
/// odd last dim leaves alternate rows off the 16 B bound) drops to the scalar
/// loop instead, and so does the tail.
template <typename T>
__global__ void activation_kernel(const T* __restrict__ input, T* __restrict__ output,
                                  int64_t num_rows, int64_t last_dim, int64_t in_row_stride,
                                  int64_t out_row_stride, ActivationType type, float alpha,
                                  float beta, bool add_to)
{
    constexpr int VEC = sizeof(T) == 2 ? 8 : 4;
    const ActivationFn fn = get_activation_fn(type);

    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    if (row >= num_rows)
        return;

    const T* in_row = input + row * in_row_stride;
    T* out_row = output + row * out_row_stride;

    const int64_t vec_end = aligned16(in_row, out_row) ? (last_dim / VEC) * VEC : 0;
    for (int64_t i = tid * VEC; i < vec_end; i += int64_t(blockDim.x) * VEC) {
        float v[VEC];
        load_channels<T, VEC>(&in_row[i], v);
        float r[VEC];
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
            r[k] = fn(v[k], alpha, beta);
        }
        if (add_to) {
            float o[VEC];
            load_channels<T, VEC>(&out_row[i], o);
#pragma unroll
            for (int k = 0; k < VEC; ++k) {
                r[k] += o[k];
            }
        }
        store_channels<T, VEC>(&out_row[i], r);
    }

    for (int64_t i = vec_end + tid; i < last_dim; i += blockDim.x) {
        float x = s_load(&in_row[i]);
        float rv = fn(x, alpha, beta);
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
__global__ void activation_flat_kernel(const T* __restrict__ input, T* __restrict__ output,
                                       int64_t total, ActivationType type, float alpha, float beta,
                                       bool add_to)
{
    constexpr int VEC = sizeof(T) == 2 ? 8 : 4;
    const ActivationFn fn = get_activation_fn(type);

    const int64_t lane = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t step = int64_t(gridDim.x) * blockDim.x;

    const int64_t vec_end = aligned16(input, output) ? (total / VEC) * VEC : 0;
    for (int64_t i = lane * VEC; i < vec_end; i += step * VEC) {
        float v[VEC];
        load_channels<T, VEC>(&input[i], v);
        float r[VEC];
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
            r[k] = fn(v[k], alpha, beta);
        }
        if (add_to) {
            float o[VEC];
            load_channels<T, VEC>(&output[i], o);
#pragma unroll
            for (int k = 0; k < VEC; ++k) {
                r[k] += o[k];
            }
        }
        store_channels<T, VEC>(&output[i], r);
    }

    for (int64_t i = vec_end + lane; i < total; i += step) {
        float x = s_load(&input[i]);
        float rv = fn(x, alpha, beta);
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
void activation_cuda_impl(const ActivationAttributes& attrs, TensorView& output,
                          std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t total = input.numel();
    if (total == 0)
        return;

    const int64_t rank = input.rank();
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    if (rank <= 1) {
        // Flat tensor — use flat kernel with grid-stride loop. The grid is sized
        // in lanes, not elements: a block covers block_size × VEC of them, and
        // sizing it per element would launch 4× (f32) or 8× (2-byte) the blocks
        // the tensor needs and leave most of them empty.
        constexpr int VEC = sizeof(T) == 2 ? 8 : 4;
        const int block_size = kCudaBlockSize;
        const int64_t lane = static_cast<int64_t>(block_size) * VEC;
        const int grid_size = std::min(static_cast<int>(ceil_div(total, lane)), 65535);
        activation_flat_kernel<<<grid_size, block_size, 0, stream>>>(
            input.ptr<T>(), output.ptr<T>(), total, attrs.type, attrs.alpha, attrs.beta,
            attrs.add_to);
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

        activation_kernel<<<num_rows, block_size, 0, stream>>>(
            input.ptr<T>(), output.ptr<T>(), num_rows, last_dim, in_row_stride, out_row_stride,
            attrs.type, attrs.alpha, attrs.beta, attrs.add_to);
    }
}

// ============================================================
// Quantized route — dequantize, activate, requantize in one pass
// ============================================================

/// One 128-bit lane (16 elements) per thread step, with the row's parameters
/// decoded once per row. The grid covers the rows on y and column lanes on x;
/// a row longer than the grid's reach strides on.
template <class InT, class OutT>
__global__ void activation_quant_kernel(const InT* __restrict__ input, OutT* __restrict__ output,
                                        int64_t rows, int64_t cols, qelem::ParamRef qin,
                                        qelem::ParamRef qout, ActivationType type, float alpha,
                                        float beta)
{
    constexpr int VEC = 16;
    const ActivationFn fn = get_activation_fn(type);

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
                r.data[k] = qelem::requant<OutT>(
                    fn(qelem::dequant(v.data[k], pin), alpha, beta), pout);
            }
            *reinterpret_cast<qelem::pack16<OutT>*>(&out_row[i]) = r;
        }

        const int64_t sstep = int64_t(gridDim.x) * blockDim.x;
        for (int64_t i = vec_end + int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < cols;
             i += sstep) {
            const float x = qelem::dequant(in_row[i], pin);
            out_row[i] = qelem::requant<OutT>(fn(x, alpha, beta), pout);
        }
    }
}

template <class InT, class OutT>
Status activation_quant_cuda(const ActivationAttributes& attrs, TensorView& output,
                             const TensorView& input, cudaStream_t stream)
{
    if (attrs.add_to) {
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
        activation_quant_kernel<InT, OutT><<<grid, block_size, 0, stream>>>(
            input.ptr<InT>(), output.ptr<OutT>(), rows, cols, qin, qout, attrs.type, attrs.alpha,
            attrs.beta);
        st = cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
    }
    in_scale_s.release(stream);
    in_zp_s.release(stream);
    out_scale_s.release(stream);
    out_zp_s.release(stream);
    return st;
}

/// The four storage pairs the fused route carries. Both sides are int8-sized,
/// so one kernel serves each pair with the same lane layout.
template <class InT>
Status activation_quant_pair(const ActivationAttributes& attrs, TensorView& output,
                             std::span<const TensorView> inputs, cudaStream_t stream)
{
    if (output.data_type() == DataType::s8) {
        return activation_quant_cuda<InT, int8_t>(attrs, output, inputs[0], stream);
    }
    return activation_quant_cuda<InT, uint8_t>(attrs, output, inputs[0], stream);
}

// ============================================================
// Entry point
// ============================================================

Status activation_cuda(const ActivationAttributes& attrs, TensorView& output,
                       std::span<const TensorView> inputs, const ComputeContext& ctx,
                       void* /*workspace*/)
{
    // PRelu reads a per-channel slope tensor, which the scalar device-function
    // table has no room for.
    if (attrs.type == ActivationType::PRelu) {
        return Status::unsupported;
    }
    const DataType dtype = inputs[0].data_type();
    const DataType out_dtype = output.data_type();
    const bool in_int = dtype == DataType::s8 || dtype == DataType::u8;
    const bool out_int = out_dtype == DataType::s8 || out_dtype == DataType::u8;
    if (in_int || out_int) {
        // No float↔int mixing: the fused route reads and writes integers.
        if (!in_int || !out_int) {
            return Status::invalid_argument;
        }
        const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
        if (dtype == DataType::s8) {
            return activation_quant_pair<int8_t>(attrs, output, inputs, stream);
        }
        return activation_quant_pair<uint8_t>(attrs, output, inputs, stream);
    }
    switch (dtype) {
    case DataType::f32:
        activation_cuda_impl<float>(attrs, output, inputs, ctx);
        return Status::ok;
    case DataType::f16:
        activation_cuda_impl<__half>(attrs, output, inputs, ctx);
        return Status::ok;
    case DataType::bf16:
        activation_cuda_impl<__nv_bfloat16>(attrs, output, inputs, ctx);
        return Status::ok;
    default:
        return Status::unsupported;
    }
    return Status::ok;
}

} // namespace nnops::backend::cuda
