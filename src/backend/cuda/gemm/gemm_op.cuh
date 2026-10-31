/// @file gemm_op.cuh
/// @brief Shared front end of the dense GEMM operators — batch broadcasting,
///        the 16 B vectorisable test, and the launch wrapper that turns a
///        configured GemmParams into a Status.
///
/// MatMul and Linear differ only in how they read their operands (Linear is
/// MatMul with transpose_b and a rank-2 weight), so both resolve their shapes
/// and strides here and nothing else.

#pragma once

#include "cute_gemm.cuh"
#include "nnops/core/data_type.hpp"
#include "nnops/core/status.hpp"
#include "nnops/core/tensor_view.hpp"

#include <algorithm>

namespace nnops::backend::cuda {

/// Fill the broadcast batch shape and the per-axis batch strides of every
/// operand. `c` (and the optional residual `r`) supply their own strides;
/// a broadcast axis carries stride 0, so every batch slice reads the same data.
inline bool gemm_batch_layout(const TensorView& a, const TensorView& b,
                              const TensorView& c, const TensorView* r,
                              GemmParams& p)
{
    const int64_t al = a.rank() - 2;
    const int64_t bl = b.rank() - 2;
    const int64_t rank = std::max<int64_t>(al, bl);
    if (rank > 4) { return false; }

    p.batch_rank = static_cast<int>(rank);
    for (int64_t d = 0; d < rank; ++d) {
        const int64_t ai = al - 1 - d;
        const int64_t bi = bl - 1 - d;
        const int64_t ad = ai >= 0 ? a.shape(ai) : 1;
        const int64_t bd = bi >= 0 ? b.shape(bi) : 1;
        if (ad != bd && ad != 1 && bd != 1) { return false; }

        const int64_t slot = rank - 1 - d;
        p.batch_size[slot] = std::max(ad, bd);
        p.a_batch[slot] = (ad == 1) ? 0 : a.stride_elems(ai);
        p.b_batch[slot] = (bd == 1) ? 0 : b.stride_elems(bi);

        const int64_t ci = c.rank() - 3 - d;
        p.c_batch[slot] = ci >= 0 ? c.stride_elems(ci) : 0;
        p.r_batch[slot] = (r != nullptr && ci >= 0) ? r->stride_elems(ci) : 0;

        if (p.batch_size[slot] > 65535) { return false; }
    }
    return true;
}

/// A 16 B cp.async needs every vector address aligned: the two base pointers
/// and each row stride (the k and batch offsets are multiples of the row
/// stride). Anything else takes the scalar-fill kernel.
inline bool gemm_vectorizable(const GemmParams& p, int64_t elem_size)
{
    const int64_t vec = 16 / elem_size;
    return (reinterpret_cast<uintptr_t>(p.a) % 16 == 0) &&
           (reinterpret_cast<uintptr_t>(p.b) % 16 == 0) &&
           (p.lda % vec == 0) && (p.ldb % vec == 0);
}

/// Launch a fully configured dense GEMM; f32/f16 only.
inline Status dense_gemm_status(const GemmParams& p, DataType dtype,
                                bool transpose_a, bool transpose_b,
                                cudaStream_t stream)
{
    if (dtype != DataType::f32 && dtype != DataType::f16) {
        return Status::unsupported;
    }
    if (p.m <= 0 || p.n <= 0 || p.k <= 0) { return Status::unsupported; }
    if (p.n > 65535) { return Status::unsupported; }

    const int64_t elem_size = static_cast<int64_t>(data_type_size(dtype));
    if (!dense_gemm_launch(p, static_cast<int>(elem_size), transpose_a,
                           transpose_b, gemm_vectorizable(p, elem_size),
                           stream)) {
        return Status::unsupported;
    }
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

}  // namespace nnops::backend::cuda
