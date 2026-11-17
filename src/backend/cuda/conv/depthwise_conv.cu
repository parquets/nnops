/// @file depthwise_conv.cu
/// @brief CUDA implementation of 2D/3D depthwise convolution on NHWC/NDHWC
///        tensors.
///
/// Spatial rank comes from the input rank (4 -> 2D, 5 -> 3D).
///
/// The channel axis is dense, so the lane spans channels: a thread owns one
/// 16 B vector of channels (four f32 or eight f16) and every tap is then two
/// 128-bit loads, the activation vector and the filter vector. The filter is
/// the raw HWCM [KH][KW][C] form, whose taps are contiguous runs of C — the
/// kernel reads it as it lies, no prepack. A channel count that is not a vector
/// multiple is covered by a second, scalar launch over the remainder; an
/// unaligned pitch or pixel stride falls back to the scalar script entirely.
///
/// A square unit-stride kernel walks two output rows per thread.
/// Consecutive rows share K-1 of their input rows, so the walk fetches
/// ROWS+K-1 input vectors per tap column for ROWS rows, and each one feeds up
/// to K accumulators; the filter column stays in registers across the walk.
/// Two rows beat four and eight here (0.380 vs 0.383 vs 0.394 ms summed over
/// the layer set): deeper shares more (1.5 vs 2.0 input vectors per output),
/// but the shorter block keeps the grid wider, and on this part the grid wins.
///
/// Epilogue: activation fusion applied during output write-back, every route,
/// both accumulate-before-epilogue and the add_to residual ordering.
///
/// Supports f32, f16 and bf16. Workspace: not used.

#include "../cuda_common.cuh"
#include "nnops/ops/depthwise_conv.hpp"
#include "nnops/detail/assert.hpp"

#include <algorithm>

namespace nnops::backend::cuda {

// ============================================================
// Device-side epilogue apply (mirrors host apply_epilogue)
// ============================================================

__device__ inline float device_apply_epilogue(EpilogueActivateType ep_type, float alpha, float beta,
                                              float min_clip, float max_clip, float x)
{
    // activate(clamp(x)) — the same contract as the CPU's apply_epilogue.
    x = fminf(fmaxf(x, min_clip), max_clip);
    switch (ep_type) {
    case EpilogueActivateType::None:
        return x;
    case EpilogueActivateType::Relu:
        return fmaxf(x, 0.0f);
    case EpilogueActivateType::Relu6:
        return fminf(fmaxf(x, 0.0f), 6.0f);
    case EpilogueActivateType::LeakyRelu:
        return (x > 0.0f) ? x : alpha * x;
    case EpilogueActivateType::Sigmoid:
        return 1.0f / (1.0f + expf(-x));
    case EpilogueActivateType::Tanh:
        return tanhf(x);
    case EpilogueActivateType::Gelu: {
        constexpr float c = 0.7978845608028654f;
        return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
    }
    case EpilogueActivateType::Silu:
        return x / (1.0f + expf(-x));
    case EpilogueActivateType::HardSwish: {
        float relu6 = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
        return x * relu6 * (beta / 6.0f);
    }
    case EpilogueActivateType::Elu:
        return (x > 0.0f) ? x : alpha * (expf(x) - 1.0f);
    }
    return x;
}

// ============================================================
// The vector kernel — the lane spans channels
// ============================================================

/// One launch's geometry: the window, the strides of both operands,
/// and the channel span this launch covers. All scalars, passed by value.
struct DwGeom {
    int64_t c_begin;
    int64_t chunks;
    int64_t N;
    int64_t ID, IH, IW;
    int64_t OD, OH, OW;
    int64_t KD, KH, KW;
    int64_t SD, SH, SW;
    int64_t DD, DH, DW;
    int64_t PD, PH, PW;
    int64_t in_d_stride, in_row_stride, in_pix_stride;
    int64_t out_d_stride, out_row_stride, out_pix_stride;
    int64_t w_tap_stride;
    EpilogueActivateType ep_type;
    float ep_alpha;
    float ep_beta;
    float ep_min;
    float ep_max;
    bool has_bias;
    bool add_to;
};

/// The block both vector kernels share: x lanes walk channel vectors (a
/// warp covers consecutive chunks of one pixel, so each lane's 16 B abuts its
/// neighbour's), y lanes walk output columns. Every index a thread needs then
/// comes out of a block coordinate, which is what the first kernel spent four
/// 64-bit divisions on.
struct DwBlock {
    int bx, by;
};

inline DwBlock dw_block(int64_t chunks)
{
    const int bx = int(std::min<int64_t>(chunks, kCudaWarpSize));
    return {bx, kCudaBlockSize / bx};
}

/// One output vector per thread, flat over the plane. The fallback for anything
/// the row-walking kernel below does not cover.
template <typename T, int Vec>
__global__ void depthwise_conv_last_flat_kernel(const T* __restrict__ input,
                                                const T* __restrict__ weight,
                                                const T* __restrict__ bias, // may be nullptr
                                                T* __restrict__ output, DwGeom g)
{
    const int64_t chunk = int64_t(blockIdx.y) * blockDim.x + threadIdx.x;
    const int64_t ow = int64_t(blockIdx.z) * blockDim.y + threadIdx.y;
    if (chunk >= g.chunks || ow >= g.OW) {
        return;
    }

    const int64_t plane = int64_t(blockIdx.x) / g.OH;
    const int64_t oh = int64_t(blockIdx.x) % g.OH;
    const int64_t img = plane / g.OD;
    const int64_t od = plane % g.OD;
    const int64_t c0 = g.c_begin + chunk * Vec;

    const T* in_img = input + img * g.ID * g.in_d_stride + c0;
    T* out_img = output + img * g.OD * g.out_d_stride + c0;
    const T* w_base = weight + c0;

    float acc[Vec];
    if (g.has_bias) {
        load_channels<T, Vec>(bias + c0, acc);
    } else {
#pragma unroll
        for (int i = 0; i < Vec; ++i) {
            acc[i] = 0.0f;
        }
    }

    const int64_t iw_base = ow * g.SW - g.PW;
    for (int64_t kd = 0; kd < g.KD; ++kd) {
        const int64_t id = od * g.SD + kd * g.DD - g.PD;
        if (id < 0 || id >= g.ID)
            continue;
        const T* in_d = in_img + id * g.in_d_stride;
        const T* w_d = w_base + kd * g.KH * g.KW * g.w_tap_stride;
        for (int64_t kh = 0; kh < g.KH; ++kh) {
            const int64_t ih = oh * g.SH + kh * g.DH - g.PH;
            if (ih < 0 || ih >= g.IH)
                continue;
            const T* row = in_d + ih * g.in_row_stride;
            const T* w_row = w_d + kh * g.KW * g.w_tap_stride;
            for (int64_t kw = 0; kw < g.KW; ++kw) {
                const int64_t iw = iw_base + kw * g.DW;
                if (iw < 0 || iw >= g.IW)
                    continue;
                float v[Vec];
                float wv[Vec];
                load_channels<T, Vec>(row + iw * g.in_pix_stride, v);
                load_channels<T, Vec>(w_row + kw * g.w_tap_stride, wv);
#pragma unroll
                for (int i = 0; i < Vec; ++i) {
                    acc[i] = fmaf(wv[i], v[i], acc[i]);
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < Vec; ++i) {
        acc[i] =
            device_apply_epilogue(g.ep_type, g.ep_alpha, g.ep_beta, g.ep_min, g.ep_max, acc[i]);
    }

    T* out = out_img + od * g.out_d_stride + oh * g.out_row_stride + ow * g.out_pix_stride;
    if (g.add_to) {
        float prev[Vec];
        load_channels<T, Vec>(out, prev);
#pragma unroll
        for (int i = 0; i < Vec; ++i) {
            acc[i] += prev[i];
        }
    }
    store_channels<T, Vec>(out, acc);
}

/// Square stride-1 kernel, `kRows` output rows per thread. Consecutive output
/// rows share K-1 input rows, so a thread loads `kRows+K-1` input vectors per
/// tap column for `kRows` rows instead of `kRows*K`, and each one feeds up to K
/// accumulators. The filter column is loaded once per (kd, kw) and held in
/// registers across the walk.
constexpr int kDwRows = 2;

template <typename T, int Vec, int K, int ROWS>
__global__ void depthwise_conv_last_rows_kernel(const T* __restrict__ input,
                                                const T* __restrict__ weight,
                                                const T* __restrict__ bias, // may be nullptr
                                                T* __restrict__ output, DwGeom g)
{
    const int64_t chunk = int64_t(blockIdx.y) * blockDim.x + threadIdx.x;
    const int64_t ow = int64_t(blockIdx.z) * blockDim.y + threadIdx.y;
    if (chunk >= g.chunks || ow >= g.OW) {
        return;
    }

    const int64_t row_blocks = (g.OH + ROWS - 1) / ROWS;
    const int64_t plane = int64_t(blockIdx.x) / row_blocks;
    const int64_t oh0 = (int64_t(blockIdx.x) % row_blocks) * ROWS;
    const int64_t img = plane / g.OD;
    const int64_t od = plane % g.OD;
    const int64_t c0 = g.c_begin + chunk * Vec;

    const T* in_img = input + img * g.ID * g.in_d_stride + c0;
    T* out_img = output + img * g.OD * g.out_d_stride + c0;
    const T* w_base = weight + c0;

    float acc[ROWS][Vec];
    if (g.has_bias) {
        float b[Vec];
        load_channels<T, Vec>(bias + c0, b);
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
#pragma unroll
            for (int i = 0; i < Vec; ++i) {
                acc[r][i] = b[i];
            }
    } else {
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
#pragma unroll
            for (int i = 0; i < Vec; ++i) {
                acc[r][i] = 0.0f;
            }
    }

    for (int64_t kd = 0; kd < g.KD; ++kd) {
        const int64_t id = od * g.SD + kd * g.DD - g.PD;
        if (id < 0 || id >= g.ID)
            continue;
        const T* in_d = in_img + id * g.in_d_stride;
        const T* w_d = w_base + kd * K * K * g.w_tap_stride;
#pragma unroll
        for (int kw = 0; kw < K; ++kw) {
            const int64_t iw = ow * g.SW + kw * g.DW - g.PW;
            if (iw < 0 || iw >= g.IW)
                continue;
            const T* w_col = w_d + kw * g.w_tap_stride;
            float wv[K][Vec];
#pragma unroll
            for (int kh = 0; kh < K; ++kh) {
                load_channels<T, Vec>(w_col + kh * K * g.w_tap_stride, wv[kh]);
            }
#pragma unroll
            for (int j = 0; j < ROWS + K - 1; ++j) {
                const int64_t iy = oh0 + j - g.PH;
                if (iy < 0 || iy >= g.IH)
                    continue;
                float v[Vec];
                load_channels<T, Vec>(in_d + iy * g.in_row_stride + iw * g.in_pix_stride, v);
#pragma unroll
                for (int kh = 0; kh < K; ++kh) {
                    const int r = j - kh;
                    if (r < 0 || r >= ROWS || oh0 + r >= g.OH)
                        continue;
#pragma unroll
                    for (int i = 0; i < Vec; ++i) {
                        acc[r][i] = fmaf(wv[kh][i], v[i], acc[r][i]);
                    }
                }
            }
        }
    }

    T* out_plane = out_img + od * g.out_d_stride + ow * g.out_pix_stride;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        if (oh0 + r >= g.OH) {
            break;
        }
        float res[Vec];
#pragma unroll
        for (int i = 0; i < Vec; ++i) {
            res[i] = device_apply_epilogue(g.ep_type, g.ep_alpha, g.ep_beta, g.ep_min, g.ep_max,
                                           acc[r][i]);
        }
        T* out = out_plane + (oh0 + r) * g.out_row_stride;
        if (g.add_to) {
            float prev[Vec];
            load_channels<T, Vec>(out, prev);
#pragma unroll
            for (int i = 0; i < Vec; ++i) {
                res[i] += prev[i];
            }
        }
        store_channels<T, Vec>(out, res);
    }
}

// ============================================================
// Host-side launcher
// ============================================================

template <typename T, int Vec>
void depthwise_launch(DwGeom g, int64_t c_begin, int64_t c_count, const TensorView& input,
                           const TensorView& weight, const TensorView* bias, TensorView& output,
                           cudaStream_t stream)
{
    if (c_count <= 0) {
        return;
    }
    g.c_begin = c_begin;
    g.chunks = (c_count + Vec - 1) / Vec;
    g.has_bias = bias != nullptr;

    const DwBlock blk = dw_block(g.chunks);
    const dim3 block(unsigned(blk.bx), unsigned(blk.by), 1);
    const unsigned gy = unsigned((g.chunks + blk.bx - 1) / blk.bx);
    const unsigned gz = unsigned((g.OW + blk.by - 1) / blk.by);

    // The row walk needs a square, stride-1, undilated window (so that output
    // row r and tap kh share an input row), and four times its own depth in
    // rows: a 14x14 plane cut into four-row blocks leaves eight blocks for the
    // machine, and measured 1.2x *worse* than the flat kernel's 28.
    const bool rows = g.KH == g.KW && (g.KH == 1 || g.KH == 3 || g.KH == 5) && g.SH == 1 &&
                      g.DH == 1 && g.OH >= 2 * kDwRows;
    const int64_t row_blocks = rows ? (g.OH + kDwRows - 1) / kDwRows : g.OH;
    const dim3 grid(unsigned(g.N * g.OD * row_blocks), gy, gz);

    const T* in = input.ptr<T>();
    const T* w = weight.ptr<T>();
    const T* b = bias != nullptr ? bias->ptr<T>() : nullptr;
    T* out = output.ptr<T>();

    if (rows) {
        switch (int(g.KH)) {
        case 1:
            depthwise_conv_last_rows_kernel<T, Vec, 1, kDwRows>
                <<<grid, block, 0, stream>>>(in, w, b, out, g);
            return;
        case 3:
            depthwise_conv_last_rows_kernel<T, Vec, 3, kDwRows>
                <<<grid, block, 0, stream>>>(in, w, b, out, g);
            return;
        default:
            depthwise_conv_last_rows_kernel<T, Vec, 5, kDwRows>
                <<<grid, block, 0, stream>>>(in, w, b, out, g);
            return;
        }
    }
    depthwise_conv_last_flat_kernel<T, Vec><<<grid, block, 0, stream>>>(in, w, b, out, g);
}

template <typename T>
void depthwise_conv_impl(const DepthwiseConvAttributes& attrs, TensorView& output,
                              std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;
    const int64_t rank = input.rank();
    const bool three_d = rank == 5;
    const int64_t C = input.shape(1);

    DwGeom g{};
    g.N = input.shape(0);
    g.ID = three_d ? input.shape(2) : 1;
    g.IH = input.shape(rank - 2);
    g.IW = input.shape(rank - 1);
    g.OD = three_d ? output.shape(2) : 1;
    g.OH = output.shape(rank - 2);
    g.OW = output.shape(rank - 1);

    g.KD = three_d ? attrs.kernel_size[0] : 1;
    g.KH = attrs.kernel_size[1];
    g.KW = attrs.kernel_size[2];
    g.SD = three_d ? attrs.stride[0] : 1;
    g.SH = attrs.stride[1];
    g.SW = attrs.stride[2];
    g.DD = three_d ? attrs.dilation[0] : 1;
    g.DH = attrs.dilation[1];
    g.DW = attrs.dilation[2];
    g.PD = three_d ? attrs.padding[0] : 0;
    g.PH = attrs.padding[1];
    g.PW = attrs.padding[2];

    // The pixel stride: a pixel is C channels wide, a row IW of them, and the filter
    // is [tap][C].
    g.in_pix_stride = C;
    g.in_row_stride = input.row_stride_elems();
    g.in_d_stride = g.IH * g.in_row_stride;
    g.out_pix_stride = C;
    g.out_row_stride = output.row_stride_elems();
    g.out_d_stride = g.OH * g.out_row_stride;
    g.w_tap_stride = C;

    g.ep_type = attrs.epilogue.type;
    g.ep_alpha = attrs.epilogue.alpha;
    g.ep_beta = attrs.epilogue.beta;
    g.ep_min = attrs.epilogue.min_clip;
    g.ep_max = attrs.epilogue.max_clip;
    g.add_to = attrs.add_to;

    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    constexpr int kVec = static_cast<int>(16 / sizeof(T));
    const bool vec_ok = C % kVec == 0 && g.in_pix_stride % kVec == 0 &&
                        g.in_row_stride % kVec == 0 && g.out_pix_stride % kVec == 0 &&
                        g.out_row_stride % kVec == 0;

    const TensorView* bias = has_bias ? &inputs[2] : nullptr;
    const int64_t full = vec_ok ? (C / kVec) * kVec : 0;
    depthwise_launch<T, kVec>(g, 0, full, input, weight, bias, output, stream);
    depthwise_launch<T, 1>(g, full, C - full, input, weight, bias, output, stream);
}

// ============================================================
// Unified entry point
// ============================================================

Status depthwise_conv_cuda(const DepthwiseConvAttributes& attrs, TensorView& output,
                           std::span<const TensorView> inputs, const ComputeContext& ctx,
                           void* /*workspace*/)
{
    switch (inputs[0].data_type()) {
    case DataType::f32:
        depthwise_conv_impl<float>(attrs, output, inputs, ctx);
        return Status::ok;
    case DataType::f16:
        depthwise_conv_impl<__half>(attrs, output, inputs, ctx);
        return Status::ok;
    case DataType::bf16:
        depthwise_conv_impl<__nv_bfloat16>(attrs, output, inputs, ctx);
        return Status::ok;
    default:
        return Status::unsupported;
    }
}

} // namespace nnops::backend::cuda
