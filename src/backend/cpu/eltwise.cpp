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

#include <cmath>

namespace nnops::backend::cpu {

using namespace nnops::simd;

// ============================================================
// Shared row-processing helper — eliminates the identical
// row-loop + SIMD/scalar + add_to boilerplate across all 4
// binary operations. Each case passes just its SIMD op and
// scalar formula as lambdas.
// ============================================================

template <typename T, typename SimdK, typename ScalarK>
inline void process_eltwise_rows(
    const T* a_ptr, const T* b_ptr, T* o_ptr,
    int64_t num_rows, int64_t last_dim,
    int64_t a_row_stride, int64_t b_row_stride, int64_t o_row_stride,
    bool add_to,
    SimdK&& simd_kernel,
    ScalarK&& scalar_kernel)
{
    constexpr int L = simd_lane_for<T>;
    for (int64_t r = 0; r < num_rows; ++r) {
        const T* a_row = a_ptr + r * a_row_stride;
        const T* b_row = b_ptr + r * b_row_stride;
        T* o_row = o_ptr + r * o_row_stride;
        int64_t i = 0;
        for (; i + L <= last_dim; i += L) {
            v_store_add(o_row + i, simd_kernel(v_load(a_row + i), v_load(b_row + i)), add_to);
        }
        for (; i < last_dim; ++i) {
            s_store_add(&o_row[i], scalar_kernel(s_load(&a_row[i]), s_load(&b_row[i])), add_to);
        }
    }
}

// ============================================================
// Scalar-only row-processing helper for ops without SIMD intrinsics
// (e.g. Pow). Same row-loop structure as process_eltwise_rows but
// scalar-only — no SIMD loop, no tail.
// ============================================================

template <typename T, typename ScalarK>
inline void process_eltwise_rows_scalar(
    const T* a_ptr, const T* b_ptr, T* o_ptr,
    int64_t num_rows, int64_t last_dim,
    int64_t a_row_stride, int64_t b_row_stride, int64_t o_row_stride,
    bool add_to,
    ScalarK&& scalar_kernel)
{
    for (int64_t r = 0; r < num_rows; ++r) {
        const T* a_row = a_ptr + r * a_row_stride;
        const T* b_row = b_ptr + r * b_row_stride;
        T* o_row = o_ptr + r * o_row_stride;
        for (int64_t i = 0; i < last_dim; ++i) {
            s_store_add(&o_row[i], scalar_kernel(s_load(&a_row[i]), s_load(&b_row[i])), add_to);
        }
    }
}

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
    if (total == 0) { return; }

    const int64_t rank = A.rank();
    NNOPS_ASSERT(A.numel() == B.numel());
    NNOPS_ASSERT(output.numel() == total);

    // Row-by-row layout (pitch-aware).
    // For packed layouts (NCHWC8 etc.), last_dim = W * pack covers all C8 lanes
    // and num_rows = total_rows() accounts for channel block rounding.
    const int64_t last_dim = (rank >= 1) ? A.shape(rank - 1) * A.channel_pack_size() : 1;
    const int64_t num_rows = (rank >= 2) ? A.total_rows() : total / last_dim;
    const int64_t a_row_stride = A.row_stride_elems();
    const int64_t b_row_stride = B.row_stride_elems();
    const int64_t o_row_stride = output.row_stride_elems();

    const auto* a_ptr = A.ptr<T>();
    const auto* b_ptr = B.ptr<T>();
    auto*       o_ptr = output.ptr<T>();
    const bool add_to = attrs.add_to;

    switch (attrs.type) {

    case EltwiseType::Add:
        process_eltwise_rows(a_ptr, b_ptr, o_ptr, num_rows, last_dim,
            a_row_stride, b_row_stride, o_row_stride, add_to,
            [](auto va, auto vb) { return v_add(va, vb); },
            [](float a, float b) { return a + b; });
        break;

    case EltwiseType::Sub:
        process_eltwise_rows(a_ptr, b_ptr, o_ptr, num_rows, last_dim,
            a_row_stride, b_row_stride, o_row_stride, add_to,
            [](auto va, auto vb) { return v_sub(va, vb); },
            [](float a, float b) { return a - b; });
        break;

    case EltwiseType::Mul:
        process_eltwise_rows(a_ptr, b_ptr, o_ptr, num_rows, last_dim,
            a_row_stride, b_row_stride, o_row_stride, add_to,
            [](auto va, auto vb) { return v_mul(va, vb); },
            [](float a, float b) { return a * b; });
        break;

    case EltwiseType::Div:
        process_eltwise_rows(a_ptr, b_ptr, o_ptr, num_rows, last_dim,
            a_row_stride, b_row_stride, o_row_stride, add_to,
            [](auto va, auto vb) { return v_div(va, vb); },
            [](float a, float b) { return a / b; });
        break;

    case EltwiseType::Min:
        process_eltwise_rows(a_ptr, b_ptr, o_ptr, num_rows, last_dim,
            a_row_stride, b_row_stride, o_row_stride, add_to,
            [](auto va, auto vb) { return v_min(va, vb); },
            [](float a, float b) { return a < b ? a : b; });
        break;

    case EltwiseType::Max:
        process_eltwise_rows(a_ptr, b_ptr, o_ptr, num_rows, last_dim,
            a_row_stride, b_row_stride, o_row_stride, add_to,
            [](auto va, auto vb) { return v_max(va, vb); },
            [](float a, float b) { return a > b ? a : b; });
        break;

    case EltwiseType::Pow:
        // No SIMD intrinsic for pow — scalar-only path
        process_eltwise_rows_scalar(a_ptr, b_ptr, o_ptr, num_rows, last_dim,
            a_row_stride, b_row_stride, o_row_stride, add_to,
            [](float a, float b) { return std::pow(a, b); });
        break;

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
