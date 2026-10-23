/// @file matmul_dq_ref.cpp
/// @brief Naive CPU reference for the offline-quantized-activation GEMM.
///
/// Semantics: a floating-point A with active (per-tensor or per-token)
/// quantization parameters against an int8 B. A is quantized in place of the
/// GEMM into `attrs.act_quant_dtype` (s8 or u8) with the caller's offline
/// parameters, the int8 GEMM produces the MatMulInteger accumulate, and the
/// accumulate is dequantized back to A's floating-point dtype:
///
///   A_q[m,k] = clamp(rne(A[m,k] * (1/s_a[m]) + zp_a[m]), qmin, qmax)
///   acc      = Σ_k (A_q[m,k] − zp_a[m]) · (B[k,n] − zp_b[n])
///   C[m,n]   = ep(clamp(acc · s_a[m] · s_b[n], min_clip, max_clip))
///
/// Every expression below is written to match the SIMD kernel bit for bit: the
/// same fused `x·(1/s) + zp`, the same f32 dequant multiply, the same float→half
/// narrowing. Supports N-D batch matmul with numpy-style broadcasting.
///
/// `transpose_a` is not part of the fast path, but it is well defined here, so
/// the backend routes that combination to this kernel.

#include "nnops/ops/matmul.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"
#include "../simd_kernel/simd_epilogue.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace nnops::backend::cpu::reference {

namespace {

using nnops::backend::cpu::float_to_half;
using nnops::backend::cpu::half;
using nnops::backend::cpu::half_to_float;

inline int32_t zp_at(const QuantParams& q, int64_t idx) noexcept {
    return (q.zero_point_data != nullptr) ? q.zero_point_data[idx] : q.zero_point;
}
inline float scale_at(const QuantParams& q, int64_t idx) noexcept {
    return (q.scale_data != nullptr) ? q.scale_data[idx] : q.scale;
}

// The quantization pass the tiled kernel runs, one element at a time.
template <class QInt>
inline QInt quantize_value(float x, float inv_scale, float zero) noexcept {
    constexpr int32_t qmin = std::is_same_v<QInt, uint8_t> ? 0 : -128;
    constexpr int32_t qmax = std::is_same_v<QInt, uint8_t> ? 255 : 127;
    // Same fused multiply-add as the arch's vfmaq_f32 / _mm256_fmadd_ps.
    const float v = std::fmaf(x, inv_scale, zero);
    int32_t q = static_cast<int32_t>(std::nearbyintf(v));  // round-to-nearest-even
    q = std::min(std::max(q, qmin), qmax);
    return static_cast<QInt>(q);
}

template <class AT, class QInt>
void matmul_dq_ref_impl(const MatMulAttributes& attrs,
                        TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx)
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

    const AT*     a_base = a.ptr<AT>();
    const int8_t* b_base = b.ptr<int8_t>();

    const int64_t a_row_stride = a.row_stride_elems();
    const int64_t b_row_stride = b.row_stride_elems();
    const int64_t c_row_stride = output.row_stride_elems();

    // The op resolves the output dtype from A on this path (fp in, fp out).
    NNOPS_ASSERT(output.data_type() == a.data_type());
    AT* c_base = output.ptr<AT>();

    const QuantParams& qa = a.quant_params();
    const QuantParams& qb = b.quant_params();
    const Epilogue&    ep = attrs.epilogue;

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

    // Read one A element as float — the arch quant kernels widen f16 with F16C,
    // which is the same exact conversion half_to_float performs.
    const auto a_load = [](const AT* a_ptr, int64_t elem) -> float {
        if constexpr (std::is_same_v<AT, half>) {
            return half_to_float(a_ptr[elem]);
        } else {
            return a_ptr[elem];
        }
    };

    const auto gemm_2d = [&](const AT* a_ptr, int64_t a_row_base,
                             const int8_t* b_ptr, int64_t c_offset) {
        // Raw reductions over B (A's must come from the quantized values).
        std::vector<int32_t> r_b(static_cast<size_t>(N), 0);
        for (int64_t n = 0; n < N; ++n) {
            int32_t s = 0;
            for (int64_t k = 0; k < K; ++k) {
                s += static_cast<int32_t>(attrs.transpose_b ? b_ptr[n * b_row_stride + k]
                                                            : b_ptr[k * b_row_stride + n]);
            }
            r_b[static_cast<size_t>(n)] = s;
        }

        std::vector<QInt> aq(static_cast<size_t>(K));

        const auto compute_row = [&](int64_t m) {
            const int32_t zp_a = zp_at(qa, a_row_base + m);
            const float   s_a  = scale_at(qa, a_row_base + m);
            const float   inv  = 1.0f / s_a;

            // 1. Quantize this A row.
            int32_t r_a = 0;
            for (int64_t k = 0; k < K; ++k) {
                const float x = a_load(a_ptr, attrs.transpose_a ? k * a_row_stride + m
                                                                : m * a_row_stride + k);
                aq[static_cast<size_t>(k)] = quantize_value<QInt>(x, inv, static_cast<float>(zp_a));
                r_a += static_cast<int32_t>(aq[static_cast<size_t>(k)]);
            }

            // 2. int8 dot product + zero-point compensation (no pack bias here:
            //    the reference multiplies the stored values directly).
            for (int64_t n = 0; n < N; ++n) {
                int32_t raw = 0;
                for (int64_t k = 0; k < K; ++k) {
                    const int32_t bv = attrs.transpose_b ? b_ptr[n * b_row_stride + k]
                                                         : b_ptr[k * b_row_stride + n];
                    raw += static_cast<int32_t>(aq[static_cast<size_t>(k)]) * bv;
                }

                const int32_t zp_b = zp_at(qb, n);
                const int32_t acc = matmul_int8_compensate(
                    raw, 0, zp_a, zp_b, r_a, r_b[static_cast<size_t>(n)],
                    static_cast<int32_t>(K));

                // 3. Dequantize, clip, activate — the SIMD kernel's exact order.
                const float scale = s_a * scale_at(qb, n);
                float f = static_cast<float>(acc) * scale;
                f = std::min(std::max(f, ep.min_clip), ep.max_clip);
                f = nnops::apply_epilogue(ep, f);

                const int64_t ci = c_offset + m * c_row_stride + n;
                if constexpr (std::is_same_v<AT, half>) {
                    c_base[ci] = float_to_half(f);
                } else {
                    c_base[ci] = f;
                }
            }
        };

        ctx.cpu.run(0, M, [&](int64_t m) { compute_row(m); });
    };

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

        // Per-token arrays are indexed by A's global row; every batch stride is
        // a whole number of rows, so the division is exact.
        gemm_2d(a_base + a_offset, a_offset / a_row_stride,
                b_base + b_offset, c_offset);
    }
}

template <class AT>
void matmul_dq_ref_dispatch(const MatMulAttributes& attrs,
                            TensorView& output,
                            std::span<const TensorView> inputs,
                            const ComputeContext& ctx)
{
    if (attrs.act_quant_dtype == DataType::u8) {
        matmul_dq_ref_impl<AT, uint8_t>(attrs, output, inputs, ctx);
    } else {
        matmul_dq_ref_impl<AT, int8_t>(attrs, output, inputs, ctx);
    }
}

}  // anonymous namespace

void matmul_dq_ref(const MatMulAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx,
                   void* /*workspace*/)
{
    if (inputs[0].data_type() == DataType::f16) {
        matmul_dq_ref_dispatch<half>(attrs, output, inputs, ctx);
    } else {
        matmul_dq_ref_dispatch<float>(attrs, output, inputs, ctx);
    }
}

}  // namespace nnops::backend::cpu::reference
