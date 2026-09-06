/// @file matmul_int8_ref.cpp
/// @brief Naive CPU reference for integer (s8×s8) matrix multiplication.
///
/// Semantics: C = A × B with per-token / per-channel (asymmetric) quantization.
/// The accumulate is the MatMulInteger value — the dot product of the
/// dezero-pointed operands — written directly to an s32 output, or requantized
/// to an s8 output with the output scale / zero-point.
///
/// Supports N-D batch matmul with numpy-style broadcasting (same as matmul_ref).

#include "nnops/ops/matmul.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/assert.hpp"
#include "../simd_kernel/simd_epilogue.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace nnops::backend::cpu::reference {

namespace {

inline int32_t zp_at(const QuantParams& q, int64_t idx) noexcept {
    return (q.zero_point_data != nullptr) ? q.zero_point_data[idx] : q.zero_point;
}
inline float scale_at(const QuantParams& q, int64_t idx) noexcept {
    return (q.scale_data != nullptr) ? q.scale_data[idx] : q.scale;
}

}  // anonymous namespace

void matmul_int8_ref(const MatMulAttributes& attrs,
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

    const int64_t M  = attrs.transpose_a ? a.shape(a_rank - 1) : a.shape(a_rank - 2);
    const int64_t Ka = attrs.transpose_a ? a.shape(a_rank - 2) : a.shape(a_rank - 1);
    const int64_t Kb = attrs.transpose_b ? b.shape(b_rank - 1) : b.shape(b_rank - 2);
    const int64_t N  = attrs.transpose_b ? b.shape(b_rank - 2) : b.shape(b_rank - 1);
    NNOPS_ASSERT(Ka == Kb);
    const int64_t K = Ka;

    const int8_t* a_base = a.ptr<int8_t>();
    const int8_t* b_base = b.ptr<int8_t>();

    const int64_t a_row_stride = a.row_stride_elems();
    const int64_t b_row_stride = b.row_stride_elems();
    const int64_t c_row_stride = output.row_stride_elems();

    const bool out_s8 = (output.data_type() == DataType::s8);

    const QuantParams& qa = a.quant_params();
    const QuantParams& qb = b.quant_params();
    const QuantParams& qc = output.quant_params();
    const double  scale_out = (qc.scale_data != nullptr) ? static_cast<double>(qc.scale_data[0]) : static_cast<double>(qc.scale);
    const int32_t zp_out    = (qc.zero_point_data != nullptr) ? qc.zero_point_data[0] : qc.zero_point;
    const bool relu = (attrs.epilogue.type == EpilogueActivateType::Relu);

    // ---- broadcast batch shape ----
    const int64_t batch_a_dims = a_rank - 2;
    const int64_t batch_b_dims = b_rank - 2;
    const int64_t batch_ndim   = std::max(batch_a_dims, batch_b_dims);

    std::vector<int64_t> batch_a_shape(batch_ndim, 1);
    std::vector<int64_t> batch_b_shape(batch_ndim, 1);
    std::vector<int64_t> batch_out_shape(batch_ndim, 1);

    for (int64_t i = 0; i < batch_a_dims; ++i) {
        batch_a_shape[batch_ndim - batch_a_dims + i] = a.shape(i);
    }
    for (int64_t i = 0; i < batch_b_dims; ++i) {
        batch_b_shape[batch_ndim - batch_b_dims + i] = b.shape(i);
    }

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

    // ---- per-batch 2D GEMM ----
    const auto gemm_2d = [&](const int8_t* a_ptr, const int8_t* b_ptr,
                             int32_t* c_int, int8_t* c_out) {
        // Raw reductions over K.
        std::vector<int32_t> r_a(static_cast<size_t>(M), 0);
        std::vector<int32_t> r_b(static_cast<size_t>(N), 0);
        for (int64_t m = 0; m < M; ++m) {
            int32_t s = 0;
            for (int64_t k = 0; k < K; ++k) {
                s += attrs.transpose_a ? a_ptr[k * a_row_stride + m] : a_ptr[m * a_row_stride + k];
            }
            r_a[static_cast<size_t>(m)] = s;
        }
        for (int64_t n = 0; n < N; ++n) {
            int32_t s = 0;
            for (int64_t k = 0; k < K; ++k) {
                s += attrs.transpose_b ? b_ptr[n * b_row_stride + k] : b_ptr[k * b_row_stride + n];
            }
            r_b[static_cast<size_t>(n)] = s;
        }

        const auto compute_row = [&](int64_t m) {
            const int32_t zp_a = zp_at(qa, m);
            const double  s_a  = static_cast<double>(scale_at(qa, m));
            for (int64_t n = 0; n < N; ++n) {
                // True s8×s8 dot product (no u8 offset in the reference).
                int32_t raw = 0;
                for (int64_t k = 0; k < K; ++k) {
                    const int32_t av = attrs.transpose_a ? a_ptr[k * a_row_stride + m] : a_ptr[m * a_row_stride + k];
                    const int32_t bv = attrs.transpose_b ? b_ptr[n * b_row_stride + k] : b_ptr[k * b_row_stride + n];
                    raw += av * bv;
                }

                const int32_t zp_b = zp_at(qb, n);
                int32_t v = matmul_int8_compensate(raw, 0, zp_a, zp_b,
                                                   r_a[static_cast<size_t>(m)],
                                                   r_b[static_cast<size_t>(n)],
                                                   static_cast<int32_t>(K));
                if (relu) {
                    v = std::max(v, 0);
                }

                if (out_s8) {
                    const double req = s_a * static_cast<double>(scale_at(qb, n)) / scale_out;
                    c_out[m * c_row_stride + n] = matmul_int8_requant(v, req, zp_out);
                } else {
                    c_int[m * c_row_stride + n] = v;
                }
            }
        };

        ctx.cpu.run(0, M, [&](int64_t m) { compute_row(m); });
    };

    // ---- batch iteration ----
    for (int64_t bi = 0; bi < total_batch; ++bi) {
        int64_t rem = bi;
        int64_t a_offset = 0;
        int64_t b_offset = 0;
        int64_t c_offset = 0;

        for (int64_t d = batch_ndim - 1; d >= 0; --d) {
            const int64_t coord = rem % batch_out_shape[d];
            rem /= batch_out_shape[d];

            const int64_t a_dim = d - (batch_ndim - batch_a_dims);
            if (a_dim >= 0) {
                const int64_t a_coord = (batch_a_shape[d] == 1) ? 0 : coord;
                a_offset += a_coord * a.stride_elems(a_dim);
            }

            const int64_t b_dim = d - (batch_ndim - batch_b_dims);
            if (b_dim >= 0) {
                const int64_t b_coord = (batch_b_shape[d] == 1) ? 0 : coord;
                b_offset += b_coord * b.stride_elems(b_dim);
            }

            c_offset += coord * output.stride_elems(d);
        }

        gemm_2d(a_base + a_offset, b_base + b_offset,
                output.ptr<int32_t>() + c_offset,
                output.ptr<int8_t>() + c_offset);
    }
}

}  // namespace nnops::backend::cpu::reference
