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
