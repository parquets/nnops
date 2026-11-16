/// @file matmul_i8_ptx_host.hpp
/// @brief The declaration of the hand-written int8 tile's dispatch, kept apart
///        from the kernel so the forty-eight instantiations live in one
///        translation unit instead of every one that includes matmul_i8.cuh.
///
/// The pose gate the callers rely on: an s32/s8/u8/f32/f16 destination, s8 or u8
/// A, and a word-addressable source for whichever fill has to transpose. A false
/// return means the caller keeps its own engine, so this never fails a call --
/// it only declines it.

#pragma once

#include "matmul_i8_common.cuh"

#include <cuda_runtime.h>

namespace nnops::backend::cuda {

/// `tile` picks a menu row explicitly (-1 takes the automatic one).
bool manual_dense_i8_tile(GemmI8Params q, DataType a_dt, bool trans_a, bool trans_b,
                          cudaStream_t stream, int tile = -1);

} // namespace nnops::backend::cuda
