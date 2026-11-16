/// @file matmul_f32_host.hpp
/// @brief Host-callable entry of the hand-written f32 GEMM (plain C++, so a host
///        translation unit such as a benchmark can reach it).

#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace nnops::backend::cuda {

/// The pose the f32 kernel covers: A `[M, K]` row-major, B k-major `[N, K]` or
/// n-major `[K, N]`, an f32 destination, and a batch axis whose strides may be
/// zero for a broadcast operand.
struct ManualFCall {
    const void* a = nullptr;
    const void* b = nullptr;
    void* c = nullptr;
    int m = 0, n = 0, k = 0;
    int64_t lda = 0, ldb = 0, ldc = 0;
    float alpha = 1.0f;
    /// A arrives `[M, K]` row-major unless this is set, in which case it is
    /// `[K, M]` and `lda` is that storage's row stride.
    bool trans_a = false;
    bool trans_b = false;
    /// Up to four leading batch axes as (size, stride) pairs; a broadcast axis
    /// carries stride 0.
    int batch_rank = 0;
    int64_t batch = 1;
    int64_t bsize[4] = {1, 1, 1, 1};
    int64_t a_bstr[4] = {0, 0, 0, 0};
    int64_t b_bstr[4] = {0, 0, 0, 0};
    int64_t c_bstr[4] = {0, 0, 0, 0};
    /// The operator epilogue: bias and residual point at f32 arrays. `epilogue`
    /// is EpilogueActivateType as an int, so this header stays free of the
    /// operator's includes.
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
    /// Tile index into the compiled candidate list (-1 = the automatic pick).
    int tile = -1;
};

/// Launch the hand-written f32 kernel; false when the call is outside its pose.
/// The fill resolves every gmem offset in 32 bits, so an operand whose element
/// count overflows an int32 is refused here and left to the CuTe engine.
bool manual_dense_f_call(const ManualFCall& call, cudaStream_t stream);

}  // namespace nnops::backend::cuda
