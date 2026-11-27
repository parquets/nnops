#pragma once
/// @file quant_rows.cuh
/// @brief Row-wise s8 quantization for the int8 routes whose activation is
///        floating — MatMul's offline-quantized A and the recurrent layer's
///        per-row dynamic A.
///
/// Two scale sources, one arithmetic. The offline form reads the caller's
/// parameters: a per-row scale/zero-point table or the per-tensor scalars, the
/// pair the QuantParams on the view carry. The dynamic form computes each row's
/// scale from the row itself — |max|/127, or 1 for an all-zero row, which is the
/// recurrent layer's rule — and writes the table out for the GEMM's dequantize.
///
/// Both store round-to-nearest-even after `v·(1/scale) + zp`, saturated to the
/// int8 range, which is the CPU arch kernel's formula (the multiply by the
/// reciprocal included, so the two agree on the boundary values).
///
/// One block owns one row and walks it twice in the dynamic form. The rows are
/// the tensor's physical rows in storage order — the same order the dense A_q
/// the offline route materializes comes out in.

#include "../cuda_common.cuh"
#include "nnops/core/status.hpp"

#include <cstdint>
#include <type_traits>

namespace nnops::backend::cuda {
namespace qrows {

constexpr int kThreads = 128;
constexpr int kWarps = kThreads / kCudaWarpSize;

/// The integer store, saturated to the destination's own range: u8 keeps
/// 0..255, s8 -128..127. The zero point is in that domain too.
template <class D> __device__ inline D quant_value(float v, float inv_scale, float zp) noexcept
{
    constexpr float kLo = std::is_same_v<D, uint8_t> ? 0.0f : -128.0f;
    constexpr float kHi = std::is_same_v<D, uint8_t> ? 255.0f : 127.0f;
    return static_cast<D>(fminf(fmaxf(rintf(fmaf(v, inv_scale, zp)), kLo), kHi));
}

/// The offline form: one row's parameters come from the staged tables (null for
/// the per-tensor scalar).
template <class T, class D>
__global__ void quant_fixed_kernel(const T* __restrict__ src, int64_t rows, int64_t k, int64_t ld,
                                   D* __restrict__ dst, const float* __restrict__ scales,
                                   const int32_t* __restrict__ zps, float scale_scalar,
                                   int32_t zp_scalar)
{
    const int64_t r = blockIdx.x;
    if (r >= rows) {
        return;
    }
    const float s = scales != nullptr ? scales[r] : scale_scalar;
    const float z = static_cast<float>(zps != nullptr ? zps[r] : zp_scalar);
    const float inv = 1.0f / s;
    const T* src_row = src + r * ld;
    D* dst_row = dst + r * k;
    for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {
        dst_row[i] = quant_value<D>(elem_to_float(src_row[i]), inv, z);
    }
}

/// The dynamic form: |max| over the row, then the same store. The scale is
/// published to `scales[r]` for the GEMM's dequantize.
template <class T>
__global__ void quant_dynamic_kernel(const T* __restrict__ src, int64_t rows, int64_t k,
                                     int64_t ld, int8_t* __restrict__ dst,
                                     float* __restrict__ scales)
{
    __shared__ float red[kWarps];
    const int64_t r = blockIdx.x;
    if (r >= rows) {
        return;
    }
    const T* src_row = src + r * ld;

    float mx = 0.0f;
    for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {
        mx = fmaxf(mx, fabsf(elem_to_float(src_row[i])));
    }
    mx = warp_reduce_max(mx);
    const int lane = int(threadIdx.x) % kCudaWarpSize;
    const int wid = int(threadIdx.x) / kCudaWarpSize;
    if (lane == 0) {
        red[wid] = mx;
    }
    __syncthreads();
    if (wid == 0) {
        mx = warp_reduce_max(lane < kWarps ? red[lane] : 0.0f);
        if (lane == 0) {
            red[0] = mx;
        }
    }
    __syncthreads();

    const float s = red[0] > 0.0f ? red[0] * (1.0f / 127.0f) : 1.0f;
    if (threadIdx.x == 0) {
        scales[r] = s;
    }
    const float inv = 1.0f / s;
    int8_t* dst_row = dst + r * k;
    for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {
        dst_row[i] = quant_value<int8_t>(elem_to_float(src_row[i]), inv, 0.0f);
    }
}

/// Dispatch the offline form over the activation's element type and the
/// destination's storage. `scales`/`zps` are device tables, or null for the
/// scalars.
inline Status quantize_fixed(DataType dt, DataType dst_dt, const void* src, int64_t rows,
                             int64_t k, int64_t ld, int8_t* dst, const float* scales,
                             const int32_t* zps, float scale_scalar, int32_t zp_scalar,
                             cudaStream_t stream)
{
    if (rows <= 0 || k <= 0) {
        return Status::ok;
    }
    const bool u8 = dst_dt == DataType::u8;
    if (!u8 && dst_dt != DataType::s8) {
        return Status::unsupported;
    }
    const unsigned grid = static_cast<unsigned>(rows);
    const auto launch = [&](const auto* src_p) {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(src_p)>>;
        if (u8) {
            quant_fixed_kernel<T, uint8_t><<<grid, kThreads, 0, stream>>>(
                src_p, rows, k, ld, reinterpret_cast<uint8_t*>(dst), scales, zps, scale_scalar,
                zp_scalar);
        } else {
            quant_fixed_kernel<T, int8_t><<<grid, kThreads, 0, stream>>>(
                src_p, rows, k, ld, dst, scales, zps, scale_scalar, zp_scalar);
        }
    };
    switch (dt) {
    case DataType::f32:
        launch(static_cast<const float*>(src));
        break;
    case DataType::f16:
        launch(static_cast<const __half*>(src));
        break;
    case DataType::bf16:
        launch(static_cast<const __nv_bfloat16*>(src));
        break;
    default:
        return Status::unsupported;
    }
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

/// The dynamic form: `scales` is written by the kernel (one float a row).
inline Status quantize_dynamic(DataType dt, const void* src, int64_t rows, int64_t k, int64_t ld,
                               int8_t* dst, float* scales, cudaStream_t stream)
{
    if (rows <= 0 || k <= 0) {
        return Status::ok;
    }
    const unsigned grid = static_cast<unsigned>(rows);
    switch (dt) {
    case DataType::f32:
        quant_dynamic_kernel<float><<<grid, kThreads, 0, stream>>>(
            static_cast<const float*>(src), rows, k, ld, dst, scales);
        break;
    case DataType::f16:
        quant_dynamic_kernel<__half><<<grid, kThreads, 0, stream>>>(
            static_cast<const __half*>(src), rows, k, ld, dst, scales);
        break;
    case DataType::bf16:
        quant_dynamic_kernel<__nv_bfloat16><<<grid, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(src), rows, k, ld, dst, scales);
        break;
    default:
        return Status::unsupported;
    }
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

} // namespace qrows
} // namespace nnops::backend::cuda
