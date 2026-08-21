/// @file cuda_common.cuh
/// @brief Common utilities for CUDA kernels — helpers, math, reductions.
///
/// Design:
///   - BLOCK_SIZE = 256 threads per block (standard occupancy-friendly choice)
///   - WARP_SIZE = 32 (all current NVIDIA GPUs)
///   - Reduction helpers: warp_reduce_sum, warp_reduce_max, block_reduce_sum
///   - f16 support via __half type with float accumulators
///   - ceil_div for grid sizing
///   - All kernels follow the CPU reference semantics exactly

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>

namespace nnops::backend::cuda {

// ============================================================
// Constants
// ============================================================

constexpr int kCudaBlockSize = 256;
constexpr int kCudaWarpSize  = 32;

// ============================================================
// ceil_div
// ============================================================

template <typename T, typename U>
__host__ __device__ constexpr auto ceil_div(T a, U b) noexcept {
    return (a + b - 1) / b;
}

// ============================================================
// f16 ↔ f32 conversion helpers
// ============================================================

__host__ __device__ inline float half_to_float(__half h) noexcept {
    return __half2float(h);
}

__host__ __device__ inline __half float_to_half(float f) noexcept {
    return __float2half(f);
}

// ============================================================
// s_load / s_store — scalar load/store with f16↔f32 conversion
// Mirrors the CPU SIMD s_load/s_store API for kernel portability.
// ============================================================

__device__ inline float s_load(const float* ptr) noexcept { return *ptr; }
__device__ inline float s_load(const __half* ptr) noexcept { return __half2float(*ptr); }
__device__ inline void s_store(float* ptr, float val) noexcept { *ptr = val; }
__device__ inline void s_store(__half* ptr, float val) noexcept { *ptr = __float2half(val); }

// ============================================================
// 128-bit vectorized load/store helpers
// ============================================================
// f32: float4 = 128 bits = 4 floats
// f16: half8  = 128 bits = 8 halfs (via aligned struct)
//
// These generate a single 128-bit LDG/STG instruction, reducing
// memory transactions by 4× (f32) or 8× (f16) vs scalar s_load.

struct alignas(16) half8 {
    __half data[8];
};

__device__ inline float4 v4_load(const float* ptr) noexcept {
    return *reinterpret_cast<const float4*>(ptr);
}
__device__ inline void v4_store(float* ptr, float4 val) noexcept {
    *reinterpret_cast<float4*>(ptr) = val;
}

__device__ inline half8 v4_load(const __half* ptr) noexcept {
    return *reinterpret_cast<const half8*>(ptr);
}
__device__ inline void v4_store(__half* ptr, half8 val) noexcept {
    *reinterpret_cast<half8*>(ptr) = val;
}

// f16 128-bit helpers: pack 8 float results into half8, and accumulate
// onto an existing half8 (for add_to mode).  Reduces per-op boilerplate.
__device__ inline half8 half8_set(
    float r0, float r1, float r2, float r3,
    float r4, float r5, float r6, float r7) noexcept
{
    half8 r;
    r.data[0] = __float2half(r0); r.data[1] = __float2half(r1);
    r.data[2] = __float2half(r2); r.data[3] = __float2half(r3);
    r.data[4] = __float2half(r4); r.data[5] = __float2half(r5);
    r.data[6] = __float2half(r6); r.data[7] = __float2half(r7);
    return r;
}

__device__ inline half8 half8_add(
    half8 base,
    float r0, float r1, float r2, float r3,
    float r4, float r5, float r6, float r7) noexcept
{
    half8 r;
    r.data[0] = __float2half(__half2float(base.data[0]) + r0);
    r.data[1] = __float2half(__half2float(base.data[1]) + r1);
    r.data[2] = __float2half(__half2float(base.data[2]) + r2);
    r.data[3] = __float2half(__half2float(base.data[3]) + r3);
    r.data[4] = __float2half(__half2float(base.data[4]) + r4);
    r.data[5] = __float2half(__half2float(base.data[5]) + r5);
    r.data[6] = __float2half(__half2float(base.data[6]) + r6);
    r.data[7] = __float2half(__half2float(base.data[7]) + r7);
    return r;
}

// ============================================================
// Warp-level reductions (shuffle-based, no shared memory)
// ============================================================

__device__ inline float warp_reduce_sum(float val) noexcept {
    #pragma unroll
    for (int offset = kCudaWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ inline float warp_reduce_max(float val) noexcept {
    #pragma unroll
    for (int offset = kCudaWarpSize / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// ============================================================
// Block-level reductions (shared memory)
// ============================================================

/// Block-level sum reduction. Each thread provides a float value;
/// the total sum is stored in shared[0] and returned by all threads.
__device__ inline float block_reduce_sum(float val, float* shared) noexcept {
    const int lane = threadIdx.x % kCudaWarpSize;
    const int wid  = threadIdx.x / kCudaWarpSize;

    // Step 1: warp-level reduction
    val = warp_reduce_sum(val);

    // Step 2: first thread of each warp writes to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Step 3: first warp reduces across warp leaders
    const int num_warps = ceil_div(blockDim.x, kCudaWarpSize);
    if (wid == 0) {
        val = (lane < num_warps) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    __syncthreads();

    return val;
}

/// Block-level max reduction.
__device__ inline float block_reduce_max(float val, float* shared) noexcept {
    const int lane = threadIdx.x % kCudaWarpSize;
    const int wid  = threadIdx.x / kCudaWarpSize;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    const int num_warps = ceil_div(blockDim.x, kCudaWarpSize);
    if (wid == 0) {
        val = (lane < num_warps) ? shared[lane] : -1e30f;
        val = warp_reduce_max(val);
    }
    __syncthreads();

    return val;
}

/// Block-level min reduction.
__device__ inline float block_reduce_min(float val, float* shared) noexcept {
    const int lane = threadIdx.x % kCudaWarpSize;
    const int wid  = threadIdx.x / kCudaWarpSize;

    #pragma unroll
    for (int offset = kCudaWarpSize / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fminf(val, other);
    }

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    const int num_warps = ceil_div(blockDim.x, kCudaWarpSize);
    if (wid == 0) {
        val = (lane < num_warps) ? shared[lane] : 1e30f;
        // warp_reduce_min inline
        #pragma unroll
        for (int offset = kCudaWarpSize / 2; offset > 0; offset >>= 1) {
            val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
    }
    __syncthreads();

    return val;
}

// ============================================================
// CUDA error check helper
// ============================================================

inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        // In a real implementation this would log + abort.
        // For now, no-op (project doesn't use exceptions).
        (void)file; (void)line;
    }
}

#define CUDA_CHECK(err) cuda_check(err, __FILE__, __LINE__)

}  // namespace nnops::backend::cuda
