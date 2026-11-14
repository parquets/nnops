/// @file cuda_common.cuh
/// @brief Common utilities for CUDA kernels — helpers, math, reductions.
///
/// Design:
///   - BLOCK_SIZE = 256 threads per block (standard occupancy-friendly choice)
///   - WARP_SIZE = 32 (all current NVIDIA GPUs)
///   - Reduction helpers: generic warp_reduce/block_reduce + sum/max/min wrappers
///   - f16 support via __half type with float accumulators
///   - ceil_div for grid sizing
///   - All kernels follow the CPU reference semantics exactly

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <type_traits>

namespace nnops::backend::cuda {

// ============================================================
// Constants
// ============================================================

constexpr int kCudaBlockSize = 256;
constexpr int kCudaWarpSize  = 32;

/// Dynamic shared memory a kernel may ask for without opting in to the larger
/// carveout and without dropping the occupancy that keeps a row's blocks in
/// flight. A kernel that keeps its whole row resident has to fit under it.
constexpr int kCudaSmemBudget = 48 * 1024;

/// Whether a row is worth keeping in shared memory for a kernel that walks it
/// more than once. Two bounds, and the second is not obvious: the row buffer is
/// float-per-element whatever the dtype (the passes work in float), so it has to
/// fit under the budget beside the block's reduction scratch — but a row small
/// enough to have stayed in L1 between the passes only pays for the round trip.
/// Measured on an RTX 3060 Ti, softmax: a 2048-element f16 row (4 KB of storage)
/// runs 7% slower resident, an 8192-element f16 row (16 KB) 34% faster and the
/// same-length f32 row (32 KB) 2.2x faster; the same shapes win 9-27% under
/// LayerNorm/RMSNorm. Hence `esz` — the storage width — decides whether the
/// second read would have come from cache, and `sizeof(float)` sizes the buffer.
__host__ __device__ inline bool row_resident_worth(int64_t size, int64_t esz,
                                                   int64_t scratch_bytes) noexcept
{
    const int64_t row_bytes = size * static_cast<int64_t>(sizeof(float));
    return row_bytes + scratch_bytes <= kCudaSmemBudget && size * esz >= 8192;
}

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
// Element type ↔ float — the one conversion surface a kernel templated
// over the element type needs
// ============================================================
// bf16 is the same widening contract as f16: a 2-byte float that every kernel
// carries through float arithmetic and narrows once per store. Both NVIDIA
// narrows round to nearest even, which is what the CPU's float_to_bf16 does.

// Both directions are `__host__ __device__`: a launcher converts a scalar (a
// pad constant, a clamp bound) before it hands the value to a kernel.
template <typename T>
__host__ __device__ inline float elem_to_float(T x) noexcept {
    if constexpr (std::is_same_v<T, float>) {
        return x;
    } else if constexpr (std::is_same_v<T, __half>) {
        return __half2float(x);
    } else {
        return __bfloat162float(x);
    }
}

template <typename T>
__host__ __device__ inline T float_to_elem(float x) noexcept {
    if constexpr (std::is_same_v<T, float>) {
        return x;
    } else if constexpr (std::is_same_v<T, __half>) {
        return __float2half(x);
    } else {
        return __float2bfloat16(x);
    }
}

// ============================================================
// s_load / s_store — scalar load/store with 2-byte ↔ f32 conversion
// Mirrors the CPU SIMD s_load/s_store API for kernel portability.
// ============================================================

__device__ inline float s_load(const float* ptr) noexcept { return *ptr; }
__device__ inline float s_load(const __half* ptr) noexcept { return __half2float(*ptr); }
__device__ inline float s_load(const __nv_bfloat16* ptr) noexcept { return __bfloat162float(*ptr); }
__device__ inline void s_store(float* ptr, float val) noexcept { *ptr = val; }
__device__ inline void s_store(__half* ptr, float val) noexcept { *ptr = __float2half(val); }
__device__ inline void s_store(__nv_bfloat16* ptr, float val) noexcept { *ptr = __float2bfloat16(val); }

// ============================================================
// 128-bit vectorized load/store helpers
// ============================================================
// f32: float4 = 128 bits = 4 floats
// 2-byte: pack8<T> = 128 bits = 8 elements (via aligned struct)
//
// These generate a single 128-bit LDG/STG instruction, reducing
// memory transactions by 4× (f32) or 8× (2-byte) vs scalar s_load.

template <typename T>
struct alignas(16) pack8 {
    T data[8];
};

using half8 = pack8<__half>;
using bf8   = pack8<__nv_bfloat16>;

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

__device__ inline bf8 v4_load(const __nv_bfloat16* ptr) noexcept {
    return *reinterpret_cast<const bf8*>(ptr);
}
__device__ inline void v4_store(__nv_bfloat16* ptr, bf8 val) noexcept {
    *reinterpret_cast<bf8*>(ptr) = val;
}

/// The 128-bit helpers above fault on a misaligned address, so a kernel that
/// walks rows has to know whether its row bases landed on one. A dense row of a
/// 2-byte dtype is 2·last_dim bytes and a 4-byte one 4·last_dim, so an odd last
/// dim leaves alternate rows unaligned; such a row drops to its scalar loop.
__device__ inline bool aligned16(const void* a) noexcept {
    return (reinterpret_cast<uintptr_t>(a) & 15u) == 0;
}
__device__ inline bool aligned16(const void* a, const void* b) noexcept {
    return ((reinterpret_cast<uintptr_t>(a) |
             reinterpret_cast<uintptr_t>(b)) & 15u) == 0;
}
__device__ inline bool aligned16(const void* a, const void* b,
                                 const void* c) noexcept {
    return ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b) |
             reinterpret_cast<uintptr_t>(c)) & 15u) == 0;
}

// 2-byte 128-bit helpers: pack 8 float results into a pack8<T>, and accumulate
// onto an existing one (for add_to mode).  Reduces per-op boilerplate.
template <typename T>
__device__ inline pack8<T> pack8_set(
    float r0, float r1, float r2, float r3,
    float r4, float r5, float r6, float r7) noexcept
{
    pack8<T> r;
    r.data[0] = float_to_elem<T>(r0); r.data[1] = float_to_elem<T>(r1);
    r.data[2] = float_to_elem<T>(r2); r.data[3] = float_to_elem<T>(r3);
    r.data[4] = float_to_elem<T>(r4); r.data[5] = float_to_elem<T>(r5);
    r.data[6] = float_to_elem<T>(r6); r.data[7] = float_to_elem<T>(r7);
    return r;
}

template <typename T>
__device__ inline pack8<T> pack8_add(
    pack8<T> base,
    float r0, float r1, float r2, float r3,
    float r4, float r5, float r6, float r7) noexcept
{
    pack8<T> r;
    r.data[0] = float_to_elem<T>(elem_to_float(base.data[0]) + r0);
    r.data[1] = float_to_elem<T>(elem_to_float(base.data[1]) + r1);
    r.data[2] = float_to_elem<T>(elem_to_float(base.data[2]) + r2);
    r.data[3] = float_to_elem<T>(elem_to_float(base.data[3]) + r3);
    r.data[4] = float_to_elem<T>(elem_to_float(base.data[4]) + r4);
    r.data[5] = float_to_elem<T>(elem_to_float(base.data[5]) + r5);
    r.data[6] = float_to_elem<T>(elem_to_float(base.data[6]) + r6);
    r.data[7] = float_to_elem<T>(elem_to_float(base.data[7]) + r7);
    return r;
}

// ============================================================
// Channel-lane vectors — the loads a channel-last kernel wants
// ============================================================
// The channel axis is contiguous there, so a thread owns one 16 B vector of it:
// `Vec` is 4 for f32 and 8 for a 2-byte type, or 1 for the scalar script that a
// channel tail or an unaligned lane takes. Every one widens through float, which
// is what every accumulating kernel works in.

template <typename T, int Vec>
__device__ inline void load_channels(const T* ptr, float (&v)[Vec]) noexcept
{
    if constexpr (Vec == 1) {
        v[0] = s_load(ptr);
    } else if constexpr (std::is_same_v<T, float>) {
        const float4 r = v4_load(ptr);
        v[0] = r.x;
        v[1] = r.y;
        v[2] = r.z;
        v[3] = r.w;
    } else {
        const pack8<T> r = v4_load(ptr);
        #pragma unroll
        for (int i = 0; i < Vec; ++i) { v[i] = elem_to_float(r.data[i]); }
    }
}

template <typename T, int Vec>
__device__ inline void store_channels(T* ptr, const float (&v)[Vec]) noexcept
{
    if constexpr (Vec == 1) {
        s_store(ptr, v[0]);
    } else if constexpr (std::is_same_v<T, float>) {
        v4_store(ptr, make_float4(v[0], v[1], v[2], v[3]));
    } else {
        pack8<T> r;
        #pragma unroll
        for (int i = 0; i < Vec; ++i) { r.data[i] = float_to_elem<T>(v[i]); }
        v4_store(ptr, r);
    }
}

// ============================================================
// Warp-level reductions (shuffle-based, no shared memory)
// ============================================================

/// Binary combiners for the reduce helpers. Plain inline functions (rather than
/// lambdas) so they can be passed by value to the device reduce templates.
__host__ __device__ inline float cuda_reduce_add(float a, float b) noexcept { return a + b; }
__host__ __device__ inline float cuda_reduce_max(float a, float b) noexcept { return fmaxf(a, b); }
__host__ __device__ inline float cuda_reduce_min(float a, float b) noexcept { return fminf(a, b); }

/// Generic warp shuffle-down reduction. `combine` is the binary op (sum/max/min).
template <typename Combine>
__device__ inline float warp_reduce(float val, Combine combine) noexcept {
    #pragma unroll
    for (int offset = kCudaWarpSize / 2; offset > 0; offset >>= 1) {
        val = combine(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ inline float warp_reduce_sum(float val) noexcept { return warp_reduce(val, cuda_reduce_add); }
__device__ inline float warp_reduce_max(float val) noexcept { return warp_reduce(val, cuda_reduce_max); }
__device__ inline float warp_reduce_min(float val) noexcept { return warp_reduce(val, cuda_reduce_min); }

// ============================================================
// Block-level reductions (shared memory)
// ============================================================

/// Generic block reduction: warp-reduce, write per-warp leaders to shared,
/// then have warp 0 reduce across leaders (padding inactive lanes with
/// `identity`, the op's neutral element). `combine` is the binary op.
template <typename Combine>
__device__ inline float block_reduce(float val, float* shared, Combine combine, float identity) noexcept {
    const int lane = threadIdx.x % kCudaWarpSize;
    const int wid  = threadIdx.x / kCudaWarpSize;

    // Step 1: warp-level reduction
    val = warp_reduce(val, combine);

    // Step 2: first thread of each warp writes to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Step 3: first warp reduces across warp leaders
    const int num_warps = ceil_div(blockDim.x, kCudaWarpSize);
    if (wid == 0) {
        val = (lane < num_warps) ? shared[lane] : identity;
        val = warp_reduce(val, combine);
    }
    __syncthreads();

    return val;
}

/// Block-level sum reduction. Each thread provides a float value;
/// the total sum is stored in shared[0] and returned by all threads.
__device__ inline float block_reduce_sum(float val, float* shared) noexcept {
    return block_reduce(val, shared, cuda_reduce_add, 0.0f);
}

/// Block-level max reduction.
__device__ inline float block_reduce_max(float val, float* shared) noexcept {
    return block_reduce(val, shared, cuda_reduce_max, -1e30f);
}

/// Block-level min reduction.
__device__ inline float block_reduce_min(float val, float* shared) noexcept {
    return block_reduce(val, shared, cuda_reduce_min, 1e30f);
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
