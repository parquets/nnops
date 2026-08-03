/// @file unary.cpp
/// @brief SIMD-optimized CPU implementation of element-wise unary operations.
///
/// Supports both f32 and f16 via a single templated implementation.
/// Row-by-row pitch-aware processing with SIMD inner loop + scalar tail.
/// All 9 operations (Exp/Log/Sin/Cos/Tan/Tanh/Abs/Neg/Sqrt) are vectorized.
///
/// Design:
///   1. Row-by-row processing via row_stride_elems() respects pitch padding
///   2. simd_lane_for<T> selects lane count (8 for both f32 and f16)
///   3. Branch on op type is hoisted outside the row loop
///   4. add_to is checked inside the loop body (branch predictor handles it)

#include "nnops/ops/unary.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <cmath>

namespace nnops::backend::cpu {

using namespace nnops::simd;

// ============================================================
// Shared row-processing helper — eliminates the identical
// row-loop + SIMD/scalar + add_to boilerplate across all 9
// unary operations. Each case passes just its SIMD intrinsic
// and scalar formula as lambdas.
// ============================================================

template <typename T, typename SimdK, typename ScalarK>
inline void process_unary_rows(
    const T* in_ptr, T* out_ptr,
    int64_t num_rows, int64_t last_dim,
    int64_t in_row_stride, int64_t out_row_stride,
    bool add_to,
    SimdK&& simd_kernel,
    ScalarK&& scalar_kernel)
{
    constexpr int L = simd_lane_for<T>;
    for (int64_t r = 0; r < num_rows; ++r) {
        const T* in_row = in_ptr + r * in_row_stride;
        T* out_row = out_ptr + r * out_row_stride;
        int64_t i = 0;
        for (; i + L <= last_dim; i += L) {
            v_store_add(out_row + i, simd_kernel(v_load(in_row + i)), add_to);
        }
        for (; i < last_dim; ++i) {
            s_store_add(&out_row[i], scalar_kernel(s_load(&in_row[i])), add_to);
        }
    }
}

// ============================================================
// Scalar-only row-processing helper for ops without SIMD intrinsics
// (e.g. Erf, Round, Sign). Same row-loop structure as process_unary_rows
// but scalar-only — no SIMD loop, no tail.
// ============================================================

template <typename T, typename ScalarK>
inline void process_unary_rows_scalar(
    const T* in_ptr, T* out_ptr,
    int64_t num_rows, int64_t last_dim,
    int64_t in_row_stride, int64_t out_row_stride,
    bool add_to,
    ScalarK&& scalar_kernel)
{
    for (int64_t r = 0; r < num_rows; ++r) {
        const T* in_row = in_ptr + r * in_row_stride;
        T* out_row = out_ptr + r * out_row_stride;
        for (int64_t i = 0; i < last_dim; ++i) {
            s_store_add(&out_row[i], scalar_kernel(s_load(&in_row[i])), add_to);
        }
    }
}

// ============================================================
// Templated implementation (f32 and f16)
// ============================================================

template <typename T>
void unary_impl(const UnaryAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& /*ctx*/)
{
    const auto& input = inputs[0];
    const int64_t total = input.numel();
    if (total == 0) { return; }

    const int64_t rank = input.rank();
    NNOPS_ASSERT(output.numel() == total);

    // Row-by-row layout (pitch-aware).
    // For packed layouts (NCHWC8 etc.), last_dim = W * pack covers all C8 lanes
    // and num_rows = total_rows() accounts for channel block rounding.
    const int64_t last_dim = (rank >= 1) ? input.shape(rank - 1) * input.channel_pack_size() : 1;
    const int64_t num_rows = (rank >= 2) ? input.total_rows() : total / last_dim;
    const int64_t in_row_stride = input.row_stride_elems();
    const int64_t out_row_stride = output.row_stride_elems();

    const auto* in_ptr  = input.ptr<T>();
    auto*       out_ptr = output.ptr<T>();
    const bool add_to = attrs.add_to;

    switch (attrs.type) {

    case UnaryType::Exp:
        process_unary_rows(in_ptr, out_ptr, num_rows, last_dim,
            in_row_stride, out_row_stride, add_to,
            [](auto x) { return v_exp(x); },
            [](float v) { return std::exp(v); });
        break;

    case UnaryType::Log:
        process_unary_rows(in_ptr, out_ptr, num_rows, last_dim,
            in_row_stride, out_row_stride, add_to,
            [](auto x) { return v_log(x); },
            [](float v) { return std::log(v); });
        break;

    case UnaryType::Sin:
        process_unary_rows(in_ptr, out_ptr, num_rows, last_dim,
            in_row_stride, out_row_stride, add_to,
            [](auto x) { return v_sin(x); },
            [](float v) { return std::sin(v); });
        break;

    case UnaryType::Cos:
        process_unary_rows(in_ptr, out_ptr, num_rows, last_dim,
            in_row_stride, out_row_stride, add_to,
            [](auto x) { return v_cos(x); },
            [](float v) { return std::cos(v); });
        break;

    case UnaryType::Tan:
        process_unary_rows(in_ptr, out_ptr, num_rows, last_dim,
            in_row_stride, out_row_stride, add_to,
            [](auto x) { return v_tan(x); },
            [](float v) { return std::tan(v); });
        break;

    case UnaryType::Tanh:
        process_unary_rows(in_ptr, out_ptr, num_rows, last_dim,
            in_row_stride, out_row_stride, add_to,
            [](auto x) { return v_tanh(x); },
            [](float v) { return std::tanh(v); });
        break;

    case UnaryType::Abs:
        process_unary_rows(in_ptr, out_ptr, num_rows, last_dim,
            in_row_stride, out_row_stride, add_to,
            [](auto x) { return v_abs(x); },
            [](float v) { return v < 0.0f ? -v : v; });
        break;

    case UnaryType::Neg:
        process_unary_rows(in_ptr, out_ptr, num_rows, last_dim,
            in_row_stride, out_row_stride, add_to,
            [](auto x) { return v_neg(x); },
            [](float v) { return -v; });
        break;

    case UnaryType::Sqrt:
        process_unary_rows(in_ptr, out_ptr, num_rows, last_dim,
            in_row_stride, out_row_stride, add_to,
            [](auto x) { return v_sqrt(x); },
            [](float v) { return std::sqrt(v); });
        break;

    case UnaryType::Erf:
        // No SIMD intrinsic — scalar-only path
        process_unary_rows_scalar(in_ptr, out_ptr, num_rows, last_dim,
            in_row_stride, out_row_stride, add_to,
            [](float v) { return std::erf(v); });
        break;

    case UnaryType::Round:
        // No SIMD intrinsic — scalar-only path
        process_unary_rows_scalar(in_ptr, out_ptr, num_rows, last_dim,
            in_row_stride, out_row_stride, add_to,
            [](float v) { return std::round(v); });
        break;

    case UnaryType::Ceil:
        // No SIMD intrinsic — scalar-only path
        process_unary_rows_scalar(in_ptr, out_ptr, num_rows, last_dim,
            in_row_stride, out_row_stride, add_to,
            [](float v) { return std::ceil(v); });
        break;

    case UnaryType::Floor:
        // No SIMD intrinsic — scalar-only path
        process_unary_rows_scalar(in_ptr, out_ptr, num_rows, last_dim,
            in_row_stride, out_row_stride, add_to,
            [](float v) { return std::floor(v); });
        break;

    case UnaryType::Recip:
        // No SIMD intrinsic — scalar-only path
        process_unary_rows_scalar(in_ptr, out_ptr, num_rows, last_dim,
            in_row_stride, out_row_stride, add_to,
            [](float v) { return 1.0f / v; });
        break;

    case UnaryType::Sign:
        // No SIMD intrinsic — scalar-only path
        process_unary_rows_scalar(in_ptr, out_ptr, num_rows, last_dim,
            in_row_stride, out_row_stride, add_to,
            [](float v) { return (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f); });
        break;

    }  // switch
}

// ============================================================
// Entry point with dtype dispatch
// ============================================================

void unary_cpu(const UnaryAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx,
                 void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        unary_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        unary_impl<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"unary_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
