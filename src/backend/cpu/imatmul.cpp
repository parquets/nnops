/// @file imatmul.cpp
/// @brief Platform-abstracted tiled MatMul — batch iteration implementation.
///
/// Handles N-D batch broadcasting for MatMul (f32 and f16), extracting 2D
/// sub-matrices and delegating to gemm_tiled_f32 / gemm_tiled_f16 from imatmul.h.

#include "imatmul.h"
#include "nnops/detail/assert.hpp"

#include <algorithm>
#include <vector>

namespace nnops::backend::cpu {

void matmul_tiled_kernel(
    const MatMulAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx,
    void* /*workspace*/)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    const int64_t a_rank = a.rank();
    const int64_t b_rank = b.rank();

    // ---- extract matrix dimensions ----
    const int64_t M  = attrs.transpose_a ? a.shape(a_rank - 1) : a.shape(a_rank - 2);
    const int64_t Ka = attrs.transpose_a ? a.shape(a_rank - 2) : a.shape(a_rank - 1);
    const int64_t Kb = attrs.transpose_b ? b.shape(b_rank - 1) : b.shape(b_rank - 2);
    const int64_t N  = attrs.transpose_b ? b.shape(b_rank - 2) : b.shape(b_rank - 1);
    const int64_t K  = Ka;  // validated equal in MatMul::compute()
    (void)Kb;

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

    // ---- batch iteration ----
    for (int64_t bi = 0; bi < total_batch; ++bi) {
        // Unflatten linear batch index → multi-dimensional coords
        int64_t rem = bi;
        int64_t a_offset = 0;
        int64_t b_offset = 0;
        int64_t c_offset = 0;

        for (int64_t d = batch_ndim - 1; d >= 0; --d) {
            const int64_t coord = rem % batch_out_shape[d];
            rem /= batch_out_shape[d];

            // A offset: use coord only if that batch dim is not broadcast
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

        gemm_tiled_f32(
            c_base + c_offset, c_row_stride,
            a_base + a_offset, a_row_stride,
            b_base + b_offset, b_row_stride,
            M, N, K,
            attrs.transpose_a, attrs.transpose_b,
            attrs.epilogue, attrs.add_to,
            ctx);
    }
}

void matmul_tiled_kernel_f16(
    const MatMulAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx,
    void* /*workspace*/)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    const int64_t a_rank = a.rank();
    const int64_t b_rank = b.rank();

    // ---- extract matrix dimensions ----
    const int64_t M  = attrs.transpose_a ? a.shape(a_rank - 1) : a.shape(a_rank - 2);
    const int64_t Ka = attrs.transpose_a ? a.shape(a_rank - 2) : a.shape(a_rank - 1);
    const int64_t Kb = attrs.transpose_b ? b.shape(b_rank - 1) : b.shape(b_rank - 2);
    const int64_t N  = attrs.transpose_b ? b.shape(b_rank - 2) : b.shape(b_rank - 1);
    const int64_t K  = Ka;
    (void)Kb;

    const auto* a_base = a.ptr<half_t>();
    const auto* b_base = b.ptr<half_t>();
    auto*       c_base = output.ptr<half_t>();

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

    // ---- batch iteration ----
    for (int64_t bi = 0; bi < total_batch; ++bi) {
        int64_t rem = bi;
        int64_t a_offset = 0;
        int64_t b_offset = 0;
        int64_t c_offset = 0;

        for (int64_t d = batch_ndim - 1; d >= 0; --d) {
            const int64_t coord = rem % batch_out_shape[d];
            rem /= batch_out_shape[d];

            {
                const int64_t a_dim = d - (batch_ndim - batch_a_dims);
                if (a_dim >= 0) {
                    const int64_t a_coord = (batch_a_shape[d] == 1) ? 0 : coord;
                    a_offset += a_coord * a.stride_elems(a_dim);
                }
            }

            {
                const int64_t b_dim = d - (batch_ndim - batch_b_dims);
                if (b_dim >= 0) {
                    const int64_t b_coord = (batch_b_shape[d] == 1) ? 0 : coord;
                    b_offset += b_coord * b.stride_elems(b_dim);
                }
            }

            c_offset += coord * output.stride_elems(d);
        }

        gemm_tiled_f16(
            c_base + c_offset, c_row_stride,
            a_base + a_offset, a_row_stride,
            b_base + b_offset, b_row_stride,
            M, N, K,
            attrs.transpose_a, attrs.transpose_b,
            attrs.epilogue, attrs.add_to,
            ctx);
    }
}

}  // namespace nnops::backend::cpu
