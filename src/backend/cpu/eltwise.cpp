/// @file eltwise.cpp
/// @brief SIMD-optimized CPU implementation of element-wise binary operations.
///
/// Supports both f32 and f16 via a single templated implementation.
/// Row-by-row pitch-aware processing with SIMD inner loop + scalar tail.
/// All 4 operations (Add/Sub/Mul/Div) are vectorized.
///
/// Design:
///   1. Row-by-row processing via row_stride_elems() respects pitch padding
///   2. simd_lane_for<T> selects lane count (8 for both f32 and f16)
///   3. Branch on op type is hoisted outside the row loop
///   4. add_to is checked inside the loop body (branch predictor handles it)

#include "nnops/ops/eltwise.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"

namespace nnops::backend::cpu {

using namespace nnops::simd;

// ============================================================
// Templated implementation (f32 and f16)
// ============================================================

template <typename T>
void eltwise_impl(const EltwiseAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& /*ctx*/)
{
    const auto& A = inputs[0];
    const auto& B = inputs[1];
    const int64_t total = A.numel();
    if (total == 0) return;

    const int64_t rank = A.rank();
    NNOPS_ASSERT(A.numel() == B.numel());
    NNOPS_ASSERT(output.numel() == total);

    // Row-by-row layout (pitch-aware)
    const int64_t last_dim = (rank >= 1) ? A.shape(rank - 1) : 1;
    const int64_t num_rows = total / last_dim;
    const int64_t a_row_stride = A.row_stride_elems();
    const int64_t b_row_stride = B.row_stride_elems();
    const int64_t o_row_stride = output.row_stride_elems();

    const auto* a_ptr = A.ptr<T>();
    const auto* b_ptr = B.ptr<T>();
    auto*       o_ptr = output.ptr<T>();
    const bool add_to = attrs.add_to;

    constexpr int L = simd_lane_for<T>;

    switch (attrs.type) {

    // ---- Add: C = A + B ----
    case EltwiseType::Add: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* a_row = a_ptr + r * a_row_stride;
            const T* b_row = b_ptr + r * b_row_stride;
            T* o_row = o_ptr + r * o_row_stride;
            int64_t i = 0;
            for (; i + L <= last_dim; i += L) {
                auto rv = v_add(v_load(a_row + i), v_load(b_row + i));
                if (add_to) {
                    v_store(o_row + i, v_add(v_load(o_row + i), rv));
                } else {
                    v_store(o_row + i, rv);
                }
            }
            for (; i < last_dim; ++i) {
                float rv = s_load(&a_row[i]) + s_load(&b_row[i]);
                if (add_to) {
                    s_store(&o_row[i], s_load(&o_row[i]) + rv);
                } else {
                    s_store(&o_row[i], rv);
                }
            }
        }
        break;
    }

    // ---- Sub: C = A - B ----
    case EltwiseType::Sub: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* a_row = a_ptr + r * a_row_stride;
            const T* b_row = b_ptr + r * b_row_stride;
            T* o_row = o_ptr + r * o_row_stride;
            int64_t i = 0;
            for (; i + L <= last_dim; i += L) {
                auto rv = v_sub(v_load(a_row + i), v_load(b_row + i));
                if (add_to) {
                    v_store(o_row + i, v_add(v_load(o_row + i), rv));
                } else {
                    v_store(o_row + i, rv);
                }
            }
            for (; i < last_dim; ++i) {
                float rv = s_load(&a_row[i]) - s_load(&b_row[i]);
                if (add_to) {
                    s_store(&o_row[i], s_load(&o_row[i]) + rv);
                } else {
                    s_store(&o_row[i], rv);
                }
            }
        }
        break;
    }

    // ---- Mul: C = A * B ----
    case EltwiseType::Mul: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* a_row = a_ptr + r * a_row_stride;
            const T* b_row = b_ptr + r * b_row_stride;
            T* o_row = o_ptr + r * o_row_stride;
            int64_t i = 0;
            for (; i + L <= last_dim; i += L) {
                auto rv = v_mul(v_load(a_row + i), v_load(b_row + i));
                if (add_to) {
                    v_store(o_row + i, v_add(v_load(o_row + i), rv));
                } else {
                    v_store(o_row + i, rv);
                }
            }
            for (; i < last_dim; ++i) {
                float rv = s_load(&a_row[i]) * s_load(&b_row[i]);
                if (add_to) {
                    s_store(&o_row[i], s_load(&o_row[i]) + rv);
                } else {
                    s_store(&o_row[i], rv);
                }
            }
        }
        break;
    }

    // ---- Div: C = A / B ----
    case EltwiseType::Div: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* a_row = a_ptr + r * a_row_stride;
            const T* b_row = b_ptr + r * b_row_stride;
            T* o_row = o_ptr + r * o_row_stride;
            int64_t i = 0;
            for (; i + L <= last_dim; i += L) {
                auto rv = v_div(v_load(a_row + i), v_load(b_row + i));
                if (add_to) {
                    v_store(o_row + i, v_add(v_load(o_row + i), rv));
                } else {
                    v_store(o_row + i, rv);
                }
            }
            for (; i < last_dim; ++i) {
                float rv = s_load(&a_row[i]) / s_load(&b_row[i]);
                if (add_to) {
                    s_store(&o_row[i], s_load(&o_row[i]) + rv);
                } else {
                    s_store(&o_row[i], rv);
                }
            }
        }
        break;
    }

    }  // switch
}

// ============================================================
// Entry point with dtype dispatch
// ============================================================

void eltwise_cpu(const EltwiseAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx,
                   void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        eltwise_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        eltwise_impl<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"eltwise_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
