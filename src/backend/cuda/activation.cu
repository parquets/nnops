/// @file activation.cu
/// @brief CUDA implementation of activation functions.
///
/// Element-wise operations — flat grid, each thread strides over the tensor.
/// All 8 activation types supported: Relu, LeakyRelu, Sigmoid, Tanh,
/// Gelu, Silu (Swish), HardSwish, Elu.
///
/// Supports f32 and f16. Workspace: not used.

#include "cuda_common.cuh"
#include "nnops/ops/activation.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops::backend::cuda {

// ============================================================
// Device activation functions (compute a single element)
// ============================================================

__device__ inline float activation_relu(float x, float, float) {
    return fmaxf(x, 0.0f);
}

__device__ inline float activation_leaky_relu(float x, float alpha, float) {
    return (x > 0.0f) ? x : alpha * x;
}

__device__ inline float activation_sigmoid(float x, float, float) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ inline float activation_tanh(float x, float, float) {
    return tanhf(x);
}

__device__ inline float activation_gelu(float x, float, float) {
    const float c = 0.7978845608028654f;  // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

__device__ inline float activation_silu(float x, float, float) {
    return x / (1.0f + expf(-x));
}

__device__ inline float activation_hard_swish(float x, float, float beta) {
    float relu6 = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * relu6 * (beta / 6.0f);
}

__device__ inline float activation_elu(float x, float alpha, float) {
    return (x > 0.0f) ? x : alpha * (expf(x) - 1.0f);
}

// ============================================================
// Function pointer type for activation
// ============================================================

using ActivationFn = float (*)(float, float, float);

__device__ ActivationFn get_activation_fn(ActivationType type) {
    switch (type) {
    case ActivationType::Relu:       return activation_relu;
    case ActivationType::LeakyRelu:  return activation_leaky_relu;
    case ActivationType::Sigmoid:    return activation_sigmoid;
    case ActivationType::Tanh:       return activation_tanh;
    case ActivationType::Gelu:       return activation_gelu;
    case ActivationType::Silu:       return activation_silu;
    case ActivationType::HardSwish:  return activation_hard_swish;
    case ActivationType::Elu:        return activation_elu;
    }
    return activation_relu;
}

// ============================================================
// Main activation CUDA kernel — row-by-row, pitch-aware
// ============================================================

template <typename T>
__global__ void activation_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t num_rows,
    int64_t last_dim,
    int64_t in_row_stride,
    int64_t out_row_stride,
    ActivationType type,
    float alpha,
    float beta,
    bool add_to)
{
    const ActivationFn fn = get_activation_fn(type);

    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    if (row >= num_rows) return;

    const T* in_row  = input  + row * in_row_stride;
    T*       out_row = output + row * out_row_stride;

    for (int i = tid; i < last_dim; i += blockDim.x) {
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
__global__ void activation_flat_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t total,
    ActivationType type,
    float alpha,
    float beta,
    bool add_to)
{
    const ActivationFn fn = get_activation_fn(type);

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < total; i += stride) {
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
void activation_cuda_impl(
    const ActivationAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t total = input.numel();
    if (total == 0) return;

    const int64_t rank = input.rank();
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    if (rank <= 1) {
        // Flat tensor — use flat kernel with grid-stride loop
        const int block_size = kCudaBlockSize;
        const int grid_size = std::min(static_cast<int>(ceil_div(total, static_cast<int64_t>(block_size))),
                                       65535);
        activation_flat_kernel<<<grid_size, block_size, 0, stream>>>(
            input.ptr<T>(), output.ptr<T>(), total,
            attrs.type, attrs.alpha, attrs.beta, attrs.add_to);
    } else {
        // Row-by-row processing, respecting pitch.
        // For packed layouts, last_dim = W * pack covers all C8 lanes,
        // and num_rows = total_rows() accounts for channel block rounding.
        const int64_t last_dim = input.shape(rank - 1) * input.channel_pack_size();
        const int64_t num_rows = input.total_rows();
        const int64_t in_row_stride  = input.row_stride_elems();
        const int64_t out_row_stride = output.row_stride_elems();

        const int block_size = std::min(static_cast<int>(last_dim), kCudaBlockSize);

        activation_kernel<<<num_rows, block_size, 0, stream>>>(
            input.ptr<T>(), output.ptr<T>(),
            num_rows, last_dim, in_row_stride, out_row_stride,
            attrs.type, attrs.alpha, attrs.beta, attrs.add_to);
    }
}

// ============================================================
// Entry point
// ============================================================

void activation_cuda(
    const ActivationAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx,
    void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        activation_cuda_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        activation_cuda_impl<__half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"activation_cuda: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cuda
