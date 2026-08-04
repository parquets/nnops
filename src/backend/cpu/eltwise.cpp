/// @file eltwise.cpp
/// @brief SIMD-optimized CPU implementation of element-wise binary operations.
///
/// Supports both f32 and f16 via a single templated implementation.
/// Processing is tiled in groups of 8 rows and dispatched via
/// ComputeContext::cpu_parallel_for when available.
///
/// Design:
///   1. Rows are grouped into tiles of 8 for SIMD-friendly blocking
///   2. Each tile calls into tiled_eltwise kernel library (lane=8, v_f32x8/v_f16x8)
///   3. Tile iteration is parallelized via ctx.cpu_parallel_for; falls back to
///      sequential when no parallel hook is provided
///   4. Branch on op type is hoisted outside all loops

#include "nnops/ops/eltwise.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "simd_kernel/simd_eltwise.hpp"

#include <algorithm>
#include <cmath>

namespace nnops::backend::cpu {

// ============================================================
// Templated implementation (f32 and f16)
// ============================================================

template <typename T>
void eltwise_impl(const EltwiseAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx)
{
    const auto& A = inputs[0];
    const auto& B = inputs[1];
    const int64_t total = A.numel();
    if (total == 0) { return; }

    const int64_t rank = A.rank();
    NNOPS_ASSERT(A.numel() == B.numel());
    NNOPS_ASSERT(output.numel() == total);

    // Row-by-row layout (pitch-aware).
    const int64_t last_dim = (rank >= 1) ? A.shape(rank - 1) * A.channel_pack_size() : 1;
    const int64_t num_rows = (rank >= 2) ? A.total_rows() : total / last_dim;
    const int64_t a_row_stride = A.row_stride_elems();
    const int64_t b_row_stride = B.row_stride_elems();
    const int64_t o_row_stride = output.row_stride_elems();

    const auto* a_ptr = A.ptr<T>();
    const auto* b_ptr = B.ptr<T>();
    auto*       o_ptr = output.ptr<T>();
    const bool  add_to = attrs.add_to;

    constexpr int64_t TILE_M = 32;
    const int64_t num_tiles = (num_rows + TILE_M - 1) / TILE_M;

    // Shared tiled dispatch: splits rows into tiles of ≤8, then invokes
    // the given tiled kernel for each tile, parallelized when available.
    auto tiled_for_each = [&](auto&& kernel) {
        auto body = [&](int64_t ti) {
            int64_t r = ti * TILE_M;
            int64_t m = std::min(TILE_M, num_rows - r);
            kernel(a_ptr + r * a_row_stride,
                   b_ptr + r * b_row_stride,
                   o_ptr + r * o_row_stride,
                   m, last_dim, a_row_stride, b_row_stride, o_row_stride, add_to);
        };
        if (ctx.cpu_parallel_for)
            ctx.cpu_parallel_for(0, num_tiles, body);
        else
            for (int64_t t = 0; t < num_tiles; ++t) body(t);
    };

    // Dispatch to the appropriate tiled_eltwise kernel
    switch (attrs.type) {

    case EltwiseType::Add:
        tiled_for_each([](const T* a, const T* b, T* out,
                          int64_t m, int64_t n,
                          int64_t ap, int64_t bp, int64_t op, bool at) {
            kernel::add<T>(a, b, out, m, n, ap, bp, op, at);
        });
        break;

    case EltwiseType::Sub:
        tiled_for_each([](const T* a, const T* b, T* out,
                          int64_t m, int64_t n,
                          int64_t ap, int64_t bp, int64_t op, bool at) {
            kernel::sub<T>(a, b, out, m, n, ap, bp, op, at);
        });
        break;

    case EltwiseType::Mul:
        tiled_for_each([](const T* a, const T* b, T* out,
                          int64_t m, int64_t n,
                          int64_t ap, int64_t bp, int64_t op, bool at) {
            kernel::mul<T>(a, b, out, m, n, ap, bp, op, at);
        });
        break;

    case EltwiseType::Div:
        tiled_for_each([](const T* a, const T* b, T* out,
                          int64_t m, int64_t n,
                          int64_t ap, int64_t bp, int64_t op, bool at) {
            kernel::div<T>(a, b, out, m, n, ap, bp, op, at);
        });
        break;

    case EltwiseType::Min:
        tiled_for_each([](const T* a, const T* b, T* out,
                          int64_t m, int64_t n,
                          int64_t ap, int64_t bp, int64_t op, bool at) {
            kernel::min<T>(a, b, out, m, n, ap, bp, op, at);
        });
        break;

    case EltwiseType::Max:
        tiled_for_each([](const T* a, const T* b, T* out,
                          int64_t m, int64_t n,
                          int64_t ap, int64_t bp, int64_t op, bool at) {
            kernel::max<T>(a, b, out, m, n, ap, bp, op, at);
        });
        break;

    case EltwiseType::Pow:
        tiled_for_each([](const T* a, const T* b, T* out,
                          int64_t m, int64_t n,
                          int64_t ap, int64_t bp, int64_t op, bool at) {
            kernel::pow<T>(a, b, out, m, n, ap, bp, op, at);
        });
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
