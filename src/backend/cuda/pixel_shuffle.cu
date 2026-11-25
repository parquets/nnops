/// @file pixel_shuffle.cu
/// @brief CUDA SpaceToDepth and PixelShuffle — the block/channel transpose.
///
/// Both move a pixel's channel vector across a b×b pixel grid, so a direct
/// walk is coalesced on one side and strided on the other. Each block therefore
/// transposes through shared memory: the source pixel(s)' channels arrive as
/// one contiguous read and the destination row leaves as one contiguous write.
/// The block unit is one output pixel for SpaceToDepth and one source pixel for
/// PixelShuffle — exactly the unit whose channels stay together.
///
/// When a pixel's channel run does not fit the shared-memory budget the block
/// falls back to the direct index walk, which is strided but correct. A pitched
/// NHWC view is walked by its own strides either way; rank 4 only.
///
/// Every whole-element dtype the movement family carries.

#include "cuda_common.cuh"
#include "nnops/ops/pixel_shuffle.hpp"

namespace nnops::backend::cuda {

namespace {

constexpr int kThreads = 256;

// ============================================================
// SpaceToDepth
// ============================================================

/// One output pixel: b² source pixels in, one interleaved channel row out.
template <class T>
__global__ void space_to_depth_smem_kernel(const T* __restrict__ x, T* __restrict__ y,
                                           int64_t pixels, int64_t ho_n, int64_t wo_n,
                                           int64_t cb2, int64_t b, int64_t xs0, int64_t xs1,
                                           int64_t xs2, int64_t ys0, int64_t ys1, int64_t ys2)
{
    extern __shared__ __align__(16) unsigned char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);
    const int64_t C = cb2 / (b * b);

    for (int64_t p = blockIdx.x; p < pixels; p += gridDim.x) {
        const int64_t n = p / (ho_n * wo_n);
        const int64_t rem = p - n * (ho_n * wo_n);
        const int64_t ho = rem / wo_n;
        const int64_t wo = rem - ho * wo_n;
        const T* base = x + n * xs0 + (ho * b) * xs1 + (wo * b) * xs2;

        for (int64_t i = threadIdx.x; i < cb2; i += blockDim.x) {
            const int64_t q = i / C;
            const int64_t c = i - q * C;
            const int64_t bh = q / b;
            const int64_t bw = q - bh * b;
            smem[q * C + c] = base[bh * xs1 + bw * xs2 + c];
        }
        __syncthreads();

        T* dst = y + n * ys0 + ho * ys1 + wo * ys2;
        for (int64_t o = threadIdx.x; o < cb2; o += blockDim.x) {
            const int64_t c = o / (b * b);
            const int64_t q = o - c * (b * b);
            dst[o] = smem[q * C + c];
        }
        __syncthreads();
    }
}

template <class T>
__global__ void space_to_depth_direct_kernel(const T* __restrict__ x, T* __restrict__ y,
                                             int64_t total, int64_t ho_n, int64_t wo_n,
                                             int64_t cb2, int64_t b, int64_t xs0, int64_t xs1,
                                             int64_t xs2, int64_t ys0, int64_t ys1, int64_t ys2)
{
    const int64_t b2 = b * b;
    for (int64_t o = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; o < total;
         o += int64_t(blockDim.x) * gridDim.x) {
        const int64_t cq = o % cb2;
        const int64_t row = o / cb2;
        const int64_t wo = row % wo_n;
        const int64_t r2 = row / wo_n;
        const int64_t ho = r2 % ho_n;
        const int64_t n = r2 / ho_n;
        const int64_t c = cq / b2;
        const int64_t q = cq - c * b2;
        const int64_t bh = q / b;
        const int64_t bw = q - bh * b;
        const int64_t v = n * xs0 + (ho * b + bh) * xs1 + (wo * b + bw) * xs2 + c;
        y[n * ys0 + ho * ys1 + wo * ys2 + cq] = x[v];
    }
}

// ============================================================
// PixelShuffle
// ============================================================

/// One source pixel: C_in channels in, b² destination pixels out.
template <class T>
__global__ void pixel_shuffle_smem_kernel(const T* __restrict__ x, T* __restrict__ y,
                                          int64_t pixels, int64_t h_n, int64_t w_n,
                                          int64_t c_in, int64_t b, bool crd, int64_t xs0,
                                          int64_t xs1, int64_t xs2, int64_t ys0, int64_t ys1,
                                          int64_t ys2)
{
    extern __shared__ __align__(16) unsigned char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);
    const int64_t b2 = b * b;
    const int64_t c_out = c_in / b2;

    for (int64_t p = blockIdx.x; p < pixels; p += gridDim.x) {
        const int64_t n = p / (h_n * w_n);
        const int64_t rem = p - n * (h_n * w_n);
        const int64_t h = rem / w_n;
        const int64_t w = rem - h * w_n;
        const T* src = x + n * xs0 + h * xs1 + w * xs2;

        for (int64_t i = threadIdx.x; i < c_in; i += blockDim.x) {
            smem[i] = src[i];
        }
        __syncthreads();

        const int64_t row0 = (h * b) * ys1 + (w * b) * ys2;
        for (int64_t o = threadIdx.x; o < b2 * c_out; o += blockDim.x) {
            const int64_t q = o / c_out;
            const int64_t c = o - q * c_out;
            const int64_t bh = q / b;
            const int64_t bw = q - bh * b;
            const int64_t ch = crd ? q * c_out + c : c * b2 + q;
            y[n * ys0 + row0 + bh * ys1 + bw * ys2 + c] = smem[ch];
        }
        __syncthreads();
    }
}

template <class T>
__global__ void pixel_shuffle_direct_kernel(const T* __restrict__ x, T* __restrict__ y,
                                            int64_t total, int64_t ho_n, int64_t wo_n,
                                            int64_t c_out, int64_t b, bool crd, int64_t xs0,
                                            int64_t xs1, int64_t xs2, int64_t ys0, int64_t ys1,
                                            int64_t ys2)
{
    const int64_t b2 = b * b;
    for (int64_t o = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; o < total;
         o += int64_t(blockDim.x) * gridDim.x) {
        const int64_t c = o % c_out;
        const int64_t row = o / c_out;
        const int64_t wo = row % wo_n;
        const int64_t r2 = row / wo_n;
        const int64_t ho = r2 % ho_n;
        const int64_t n = r2 / ho_n;
        const int64_t bh = ho % b;
        const int64_t h = ho / b;
        const int64_t bw = wo % b;
        const int64_t w = wo / b;
        const int64_t ch = crd ? (bh * b + bw) * c_out + c : c * b2 + (bh * b + bw);
        const int64_t v = n * xs0 + h * xs1 + w * xs2 + ch;
        y[n * ys0 + ho * ys1 + wo * ys2 + c] = x[v];
    }
}

// ============================================================
// Host
// ============================================================

inline unsigned grid_for(int64_t work) noexcept
{
    return static_cast<unsigned>(std::min<int64_t>(work > 0 ? work : 1, int64_t{1} << 20));
}

template <class T>
Status space_to_depth_entry(const SpaceToDepthAttributes& attrs, TensorView& output,
                            const TensorView& input, cudaStream_t stream)
{
    if (input.rank() != 4 || output.rank() != 4) {
        return Status::invalid_argument;
    }
    const int64_t b = attrs.block_size;
    const int64_t N = input.shape(0), H = input.shape(1), W = input.shape(2), C = input.shape(3);
    const int64_t Ho = H / b, Wo = W / b, cb2 = C * b * b;
    if (N <= 0 || Ho <= 0 || Wo <= 0 || cb2 <= 0) {
        return Status::ok;
    }
    const int64_t xs0 = input.stride_elems(0), xs1 = input.stride_elems(1);
    const int64_t xs2 = input.stride_elems(2);
    const int64_t ys0 = output.stride_elems(0), ys1 = output.stride_elems(1);
    const int64_t ys2 = output.stride_elems(2);
    const int64_t pixels = N * Ho * Wo;
    const int64_t smem_bytes = cb2 * static_cast<int64_t>(sizeof(T));

    if (smem_bytes <= kCudaSmemBudget) {
        space_to_depth_smem_kernel<T><<<grid_for(pixels), kThreads, size_t(smem_bytes), stream>>>(
            input.ptr<T>(), output.ptr<T>(), pixels, Ho, Wo, cb2, b, xs0, xs1, xs2, ys0, ys1,
            ys2);
    } else {
        const int64_t total = pixels * cb2;
        space_to_depth_direct_kernel<T><<<grid_for((total + kThreads - 1) / kThreads), kThreads,
                                          0, stream>>>(
            input.ptr<T>(), output.ptr<T>(), total, Ho, Wo, cb2, b, xs0, xs1, xs2, ys0, ys1, ys2);
    }
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

template <class T>
Status pixel_shuffle_entry(const PixelShuffleAttributes& attrs, TensorView& output,
                           const TensorView& input, cudaStream_t stream)
{
    if (input.rank() != 4 || output.rank() != 4) {
        return Status::invalid_argument;
    }
    const int64_t b = attrs.block_size;
    const int64_t N = input.shape(0), H = input.shape(1), W = input.shape(2);
    const int64_t c_in = input.shape(3);
    const int64_t c_out = c_in / (b * b);
    const int64_t Ho = H * b, Wo = W * b;
    if (N <= 0 || H <= 0 || W <= 0 || c_out <= 0) {
        return Status::ok;
    }
    const bool crd = attrs.mode == PixelShuffleMode::Crd;
    const int64_t xs0 = input.stride_elems(0), xs1 = input.stride_elems(1);
    const int64_t xs2 = input.stride_elems(2);
    const int64_t ys0 = output.stride_elems(0), ys1 = output.stride_elems(1);
    const int64_t ys2 = output.stride_elems(2);
    const int64_t pixels = N * H * W;
    const int64_t smem_bytes = c_in * static_cast<int64_t>(sizeof(T));

    if (smem_bytes <= kCudaSmemBudget) {
        pixel_shuffle_smem_kernel<T><<<grid_for(pixels), kThreads, size_t(smem_bytes), stream>>>(
            input.ptr<T>(), output.ptr<T>(), pixels, H, W, c_in, b, crd, xs0, xs1, xs2, ys0, ys1,
            ys2);
    } else {
        const int64_t total = N * Ho * Wo * c_out;
        pixel_shuffle_direct_kernel<T><<<grid_for((total + kThreads - 1) / kThreads), kThreads, 0,
                                          stream>>>(
            input.ptr<T>(), output.ptr<T>(), total, Ho, Wo, c_out, b, crd, xs0, xs1, xs2, ys0,
            ys1, ys2);
    }
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

} // anonymous namespace

Status space_to_depth_cuda(const SpaceToDepthAttributes& attrs, TensorView& output,
                           std::span<const TensorView> inputs, const ComputeContext& ctx, void*)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    switch (inputs[0].data_type()) {
    case DataType::f32:
        return space_to_depth_entry<float>(attrs, output, inputs[0], stream);
    case DataType::f16:
        return space_to_depth_entry<__half>(attrs, output, inputs[0], stream);
    case DataType::bf16:
        return space_to_depth_entry<__nv_bfloat16>(attrs, output, inputs[0], stream);
    case DataType::s8:
        return space_to_depth_entry<int8_t>(attrs, output, inputs[0], stream);
    case DataType::u8:
        return space_to_depth_entry<uint8_t>(attrs, output, inputs[0], stream);
    case DataType::s32:
        return space_to_depth_entry<int32_t>(attrs, output, inputs[0], stream);
    case DataType::s64:
        return space_to_depth_entry<int64_t>(attrs, output, inputs[0], stream);
    default:
        return Status::unsupported;
    }
}

Status pixel_shuffle_cuda(const PixelShuffleAttributes& attrs, TensorView& output,
                          std::span<const TensorView> inputs, const ComputeContext& ctx, void*)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    switch (inputs[0].data_type()) {
    case DataType::f32:
        return pixel_shuffle_entry<float>(attrs, output, inputs[0], stream);
    case DataType::f16:
        return pixel_shuffle_entry<__half>(attrs, output, inputs[0], stream);
    case DataType::bf16:
        return pixel_shuffle_entry<__nv_bfloat16>(attrs, output, inputs[0], stream);
    case DataType::s8:
        return pixel_shuffle_entry<int8_t>(attrs, output, inputs[0], stream);
    case DataType::u8:
        return pixel_shuffle_entry<uint8_t>(attrs, output, inputs[0], stream);
    case DataType::s32:
        return pixel_shuffle_entry<int32_t>(attrs, output, inputs[0], stream);
    case DataType::s64:
        return pixel_shuffle_entry<int64_t>(attrs, output, inputs[0], stream);
    default:
        return Status::unsupported;
    }
}

} // namespace nnops::backend::cuda
