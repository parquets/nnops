/// @file matmul_f16_tail.cu
/// @brief The second half of the f16 tile menu: the two search switches and the
///        shallow-grid tile.
///
/// Same templates as matmul_f16.cu, kept apart so the instantiations spread
/// across the machine instead of queueing behind one another.

#include "matmul_f16_kernel.cuh"

#include <cuda_fp16.h>

namespace nnops::backend::cuda {

bool f16_tile_half_hi(const F16Call& call, int tile, cudaStream_t stream)
{
    return f16_dispatch_hi<__half>(call, tile, stream);
}

} // namespace nnops::backend::cuda
