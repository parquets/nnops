/// @file matmul_f16_host.hpp
/// @brief Host-callable entry of the hand-written PTX f16 GEMM (plain C++, so a
///        host translation unit such as a benchmark can reach it).
///
/// The kernel itself is nvcc-only (see matmul_f16.cuh); this interface is
/// what matmul_f16.cu implements.

#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace nnops::backend::cuda {

/// The poses the hand-written kernel covers today: A `[M, K]` row-major, B
/// k-major `[N, K]` or n-major `[K, N]`, alpha 1, no bias, no epilogue, f16 or
/// bf16 in and out.
struct F16Call {
    const void* a = nullptr;
    const void* b = nullptr;
    void* c = nullptr;
    int m = 0, n = 0, k = 0;
    int64_t lda = 0, ldb = 0, ldc = 0;
    float alpha = 1.0f;
    bool trans_a = false;
    bool trans_b = true;
    /// The element type of all three operands: f16 when false, bf16 when true.
    /// The two share every register path and differ only in the mma opcode.
    bool bf16 = false;
    /// The operator epilogue: bias and residual point at arrays of the operand
    /// element type; `epilogue` is EpilogueActivateType as an int.
    const void* bias = nullptr;
    bool bias_row = false;
    float beta = 0.0f;
    const void* residual = nullptr;
    int64_t ldr = 0;
    int epilogue = 0;
    float ep_alpha = 0.0f;
    float ep_beta = 1.0f;
    float min_clip = -3.402823466e+38f;
    float max_clip = 3.402823466e+38f;
    /// Up to four leading batch axes as (size, stride) pairs; a broadcast axis
    /// carries stride 0.
    int batch_rank = 0;
    int64_t batch = 1;
    int64_t bsize[4] = {1, 1, 1, 1};
    int64_t a_bstr[4] = {0, 0, 0, 0};
    int64_t b_bstr[4] = {0, 0, 0, 0};
    int64_t c_bstr[4] = {0, 0, 0, 0};
    /// Tile index into the compiled candidate list (-1 = the automatic pick).
    /// A bench sweeps it; every production caller leaves it alone.
    int tile = -1;
    /// 1 when a 16 B cp.async can address both operands; 0 takes the fill's
    /// scalar path. Resolved by the entry point from the alignment it checks.
    int vec = 1;
};

/// Launch one element type's tile menu: the f16 and bf16 kernels are compiled in
/// separate translation units so the instantiations build in parallel.
bool f16_tile_half_lo(const F16Call& call, int tile, cudaStream_t stream);
bool f16_tile_half_hi(const F16Call& call, int tile, cudaStream_t stream);
bool f16_tile_bf16(const F16Call& call, int tile, cudaStream_t stream);

/// Launch the hand-written kernel; false when the call is outside its pose.
/// `force` skips the grid-fill gate: a shallow product runs on this tile too,
/// at whatever rate it gets.
bool f16_dense_call(const F16Call& call, cudaStream_t stream, bool force = false);

} // namespace nnops::backend::cuda
