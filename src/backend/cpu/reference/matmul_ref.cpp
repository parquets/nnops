/// @file matmul_ref.cpp
/// @brief Naive CPU reference implementation of matrix multiplication.
///
/// Computes: C = A × B  (optionally transposed)
///
/// Supports N-D batch matmul with numpy-style broadcasting:
///   A: [..., M, K]  ×  B: [..., K, N]  →  C: [..., M, N]
/// where leading batch dimensions are broadcast-compatible.

#include "nnops/ops/matmul.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/assert.hpp"

#include <vector>
#include <algorithm>

namespace nnops::backend::cpu::reference {

void matmul_ref(const MatMulAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* /*workspace*/)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    const int64_t a_rank = a.rank();
    const int64_t b_rank = b.rank();

    NNOPS_ASSERT(a_rank >= 2);
    NNOPS_ASSERT(b_rank >= 2);

    // ---- extract matrix dimensions ----
    const int64_t M  = attrs.transpose_a ? a.shape(a_rank - 1) : a.shape(a_rank - 2);
    const int64_t Ka = attrs.transpose_a ? a.shape(a_rank - 2) : a.shape(a_rank - 1);
    const int64_t Kb = attrs.transpose_b ? b.shape(b_rank - 1) : b.shape(b_rank - 2);
    const int64_t N  = attrs.transpose_b ? b.shape(b_rank - 2) : b.shape(b_rank - 1);
    NNOPS_ASSERT(Ka == Kb);
    const int64_t K = Ka;

    const auto* a_base = a.ptr<float>();
    const auto* b_base = b.ptr<float>();
    auto*       c_base = output.ptr<float>();

    const int64_t a_row_stride = a.row_stride_elems();
    const int64_t b_row_stride = b.row_stride_elems();
    const int64_t c_row_stride = output.row_stride_elems();

    // ---- compute broadcast batch shape ----
    const int64_t batch_a_dims = a_rank - 2;
    const int64_t batch_b_dims = b_rank - 2;
    const int64_t batch_ndim   = std::max(batch_a_dims, batch_b_dims);

    std::vector<int64_t> batch_a_shape(batch_ndim, 1);
    std::vector<int64_t> batch_b_shape(batch_ndim, 1);
    std::vector<int64_t> batch_out_shape(batch_ndim, 1);

    for (int64_t i = 0; i < batch_a_dims; ++i)
        batch_a_shape[batch_ndim - batch_a_dims + i] = a.shape(i);
    for (int64_t i = 0; i < batch_b_dims; ++i)
        batch_b_shape[batch_ndim - batch_b_dims + i] = b.shape(i);

    int64_t total_batch = 1;
    for (int64_t i = 0; i < batch_ndim; ++i) {
        const int64_t da = batch_a_shape[i];
        const int64_t db = batch_b_shape[i];
        if (da == db) {
            batch_out_shape[i] = da;
        } else if (da == 1) {
            batch_out_shape[i] = db;
        } else if (db == 1) {
            batch_out_shape[i] = da;
        } else {
            NNOPS_ASSERT(!"MatMul: incompatible batch dimensions for broadcast");
        }
        total_batch *= batch_out_shape[i];
    }

    // ---- 2D GEMM kernel (per sub-matrix) ----
    // For each output row m, compute dot(K) with each column n of B.
    const auto gemm_2d = [&](const float* a_ptr, const float* b_ptr, float* c_ptr,
                              int64_t lda, int64_t ldb, int64_t ldc) {
        const auto compute_row = [&](int64_t m) {
            for (int64_t n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    const float a_val = attrs.transpose_a
                        ? a_ptr[k * lda + m]
                        : a_ptr[m * lda + k];
                    const float b_val = attrs.transpose_b
                        ? b_ptr[n * ldb + k]
                        : b_ptr[k * ldb + n];
                    sum += a_val * b_val;
                }
                float val = apply_epilogue(attrs.epilogue, sum, n);
                float* dst = c_ptr + m * ldc + n;
                *dst = attrs.add_to ? *dst + val : val;
            }
        };

        if (ctx.cpu_parallel_for) {
            ctx.cpu_parallel_for(0, M,
                [&](int64_t m) { compute_row(m); });
        } else {
            for (int64_t m = 0; m < M; ++m) {
                compute_row(m);
            }
        }
    };

    // ---- batch iteration ----
    for (int64_t bi = 0; bi < total_batch; ++bi) {
        // Unflatten batch index → multi-dimensional coords
        int64_t rem = bi;
        int64_t a_offset = 0;
        int64_t b_offset = 0;
        int64_t c_offset = 0;

        for (int64_t d = batch_ndim - 1; d >= 0; --d) {
            const int64_t coord = rem % batch_out_shape[d];
            rem /= batch_out_shape[d];

            // A offset: use coord only if that batch dim is not broadcast (size != 1)
            {
                const int64_t a_dim = d - (batch_ndim - batch_a_dims);
                if (a_dim >= 0) {
                    const int64_t a_coord = (batch_a_shape[d] == 1) ? 0 : coord;
                    a_offset += a_coord * a.stride_elems(a_dim);
                }
            }

            // B offset
            {
                const int64_t b_dim = d - (batch_ndim - batch_b_dims);
                if (b_dim >= 0) {
                    const int64_t b_coord = (batch_b_shape[d] == 1) ? 0 : coord;
                    b_offset += b_coord * b.stride_elems(b_dim);
                }
            }

            // C offset: output always has batch_out_shape
            c_offset += coord * output.stride_elems(d);
        }

        gemm_2d(a_base + a_offset, b_base + b_offset, c_base + c_offset,
                a_row_stride, b_row_stride, c_row_stride);
    }
}

}  // namespace nnops::backend::cpu::reference
