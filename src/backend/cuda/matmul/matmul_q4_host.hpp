/// @file matmul_q4_host.hpp
/// @brief Host-callable entry of the w4a16 GEMM: a 16-bit activation against a
///        llama.cpp block-packed weight (plain C++, so a host translation unit
///        such as a benchmark can reach it).
///
/// The kernel itself is nvcc-only (see matmul_q4_kernel.cuh); this interface is
/// what matmul_q4.cu implements. Everything but the weight's own geometry is
/// the dense 16-bit call: the batch axes, the epilogue and the split-K planes
/// are the same fields, because the kernel is the same tile.

#pragma once

#include "matmul_f16_host.hpp"

#include <cuda_runtime_api.h>
#include <cstdint>

namespace nnops::backend::cuda {

struct Q4Call {
    /// The dense 16-bit call. `b` is the block stream; `ldb` is its row stride
    /// in blocks and its batch strides (`b_bstr`) are in blocks too.
    F16Call gemm;
    /// Blocks per B row — the pitch the kernel walks between rows.
    int64_t b_row_blocks = 0;
    /// DataType::q4_0, q4_1 or q8_0 as an int.
    int fmt = 0;
};

/// Launch the w4a16 kernel; false when the call is outside its pose (row or k
/// extent not positive, or an element type that is not 16-bit).
bool q4_dense_call(const Q4Call& call, cudaStream_t stream);

} // namespace nnops::backend::cuda
