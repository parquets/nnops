/// @file matmul_bf16.cu
/// @brief The bf16 half of the 16-bit GEMM.
///
/// The kernels are the same templates the f16 side instantiates -- same
/// registers, different mma opcode -- and the split exists only so nvcc's
/// per-instantiation work divides across two translation units.

#include "matmul_f16_kernel.cuh"

#include <cuda_bf16.h>

namespace nnops::backend::cuda {

bool f16_tile_bf16(const F16Call& call, int tile, cudaStream_t stream)
{
    return f16_dispatch_lo<__nv_bfloat16>(call, tile, stream) ||
           f16_dispatch_hi<__nv_bfloat16>(call, tile, stream);
}

} // namespace nnops::backend::cuda
