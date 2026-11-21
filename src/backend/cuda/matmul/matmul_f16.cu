/// @file matmul_f16.cu
/// @brief The f16 GEMM's entry point, plus the first half of its tile menu.
///
/// One kernel instantiation costs nvcc a few seconds, so the two element types'
/// 48 kernels are spread over three translation units (this one,
/// matmul_f16_tail.cu and matmul_bf16.cu) and build in parallel.

#include "matmul_f16_kernel.cuh"

#include <cuda_fp16.h>

namespace nnops::backend::cuda {

bool f16_tile_half_lo(const F16Call& call, int tile, cudaStream_t stream)
{
    return f16_dispatch_lo<__half>(call, tile, stream);
}

bool f16_dense_call(const F16Call& call_in, cudaStream_t stream, bool force)
{
    // A transposed is supported: the panel keeps the `[k][m]` arrival order and
    // the fragment loads go through ldmatrix.trans.
    if (call_in.m <= 0 || call_in.n <= 0 || call_in.k <= 0) {
        return false;
    }
    // cp.async needs both row pitches and the base pointers 16 B aligned. A view
    // that starts or steps off that bound keeps the call and takes the fill's
    // scalar path -- the same code the tile's edge blocks run -- so the pose
    // stays covered rather than falling through to another engine.
    F16Call call = call_in;
    call.vec = ((reinterpret_cast<uintptr_t>(call.a) % 16) == 0 &&
                (reinterpret_cast<uintptr_t>(call.b) % 16) == 0 &&
                (reinterpret_cast<uintptr_t>(call.c) % 16) == 0 && (call.lda % 8) == 0 &&
                (call.ldb % 8) == 0)
                   ? 1
                   : 0;

    // An explicit tile skips the gate -- that is how a sweep reaches every
    // shape -- but it does not pick one: an unnamed tile still means the row
    // below.
    const bool named = call.tile >= 0 && call.tile < kTileCount;
    const bool forced = named || force;
    // The shallow 32x32 row, in both B poses. The rest of the table was ranked
    // on square products with a deep reduction, and on the shapes an operator
    // actually poses it loses from 512^3 down -- 512^3 4.2 against 12.6, 1024^3
    // 17.2 against 24.6 -- and by 3.5x on a short-K n-major product
    // (50176x256x256: 5.7 against 20.1 TFLOPS). Only 4096^3 with B k-major puts
    // another row first, by 1.2%, which is inside this host's run-to-run spread.
    const int tile = named ? call.tile : kF16TileShallow;
    if (!forced && !f16_grid_fills(call, kTileTable[tile])) {
        return false;
    }
    if (call.bf16) {
        return f16_tile_bf16(call, tile, stream);
    }
    return f16_tile_half_lo(call, tile, stream) || f16_tile_half_hi(call, tile, stream);
}

} // namespace nnops::backend::cuda
