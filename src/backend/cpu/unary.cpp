/// @file unary.cpp
/// @brief SIMD-optimized CPU implementation of element-wise unary operations.
///
/// Supports both f32 and f16 via a single templated implementation.
/// Processing is tiled in groups of 8 rows and dispatched via
/// ComputeContext::cpu_parallel_for when available.
///
/// Design:
///   1. Rows are grouped into tiles of 8 for SIMD-friendly blocking
///   2. Each tile calls into tiled_unary kernel library (lane=8, v_f32x8/v_f16x8)
///   3. Tile iteration is parallelized via ctx.cpu_parallel_for; falls back to
///      sequential when no parallel hook is provided
///   4. Branch on op type is hoisted outside all loops

#include "nnops/ops/unary.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "simd_kernel/simd_unary.hpp"

#include <algorithm>
#include <cmath>

namespace nnops::backend::cpu {

// ============================================================
// Templated implementation (f32 and f16)
// ============================================================

template <typename T>
void unary_impl(const UnaryAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t total = input.numel();
    if (total == 0) { return; }

    const int64_t rank = input.rank();
    NNOPS_ASSERT(output.numel() == total);

    // Row-by-row layout (pitch-aware).
    const int64_t last_dim = (rank >= 1) ? input.shape(rank - 1) * input.channel_pack_size() : 1;
    const int64_t num_rows = (rank >= 2) ? input.total_rows() : total / last_dim;
    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t out_row_stride = output.row_stride_elems();

    const auto* in_ptr  = input.ptr<T>();
    auto*       out_ptr = output.ptr<T>();
    const bool  add_to  = attrs.add_to;

    constexpr int64_t TILE_M = 8;
    const int64_t num_tiles = (num_rows + TILE_M - 1) / TILE_M;

    // Shared tiled dispatch: splits rows into tiles of ≤8, then invokes
    // the given tiled kernel for each tile, parallelized when available.
    auto tiled_for_each = [&](auto&& kernel) {
        auto body = [&](int64_t ti) {
            int64_t r = ti * TILE_M;
            int64_t m = std::min(TILE_M, num_rows - r);
            kernel(in_ptr  + r * in_row_stride,
                   out_ptr + r * out_row_stride,
                   m, last_dim, in_row_stride, out_row_stride, add_to);
        };
        if (ctx.cpu_parallel_for)
            ctx.cpu_parallel_for(0, num_tiles, body);
        else
            for (int64_t t = 0; t < num_tiles; ++t) body(t);
    };

    // Dispatch to the appropriate tiled_unary kernel
    switch (attrs.type) {

    case UnaryType::Exp:
        tiled_for_each([](const T* in, T* out, int64_t m, int64_t n,
                          int64_t ip, int64_t op, bool at) {
            kernel::exp<T>(in, out, m, n, ip, op, at);
        });
        break;

    case UnaryType::Log:
        tiled_for_each([](const T* in, T* out, int64_t m, int64_t n,
                          int64_t ip, int64_t op, bool at) {
            kernel::log<T>(in, out, m, n, ip, op, at);
        });
        break;

    case UnaryType::Sin:
        tiled_for_each([](const T* in, T* out, int64_t m, int64_t n,
                          int64_t ip, int64_t op, bool at) {
            kernel::sin<T>(in, out, m, n, ip, op, at);
        });
        break;

    case UnaryType::Cos:
        tiled_for_each([](const T* in, T* out, int64_t m, int64_t n,
                          int64_t ip, int64_t op, bool at) {
            kernel::cos<T>(in, out, m, n, ip, op, at);
        });
        break;

    case UnaryType::Tan:
        tiled_for_each([](const T* in, T* out, int64_t m, int64_t n,
                          int64_t ip, int64_t op, bool at) {
            kernel::tan<T>(in, out, m, n, ip, op, at);
        });
        break;

    case UnaryType::Tanh:
        tiled_for_each([](const T* in, T* out, int64_t m, int64_t n,
                          int64_t ip, int64_t op, bool at) {
            kernel::tanh<T>(in, out, m, n, ip, op, at);
        });
        break;

    case UnaryType::Abs:
        tiled_for_each([](const T* in, T* out, int64_t m, int64_t n,
                          int64_t ip, int64_t op, bool at) {
            kernel::abs<T>(in, out, m, n, ip, op, at);
        });
        break;

    case UnaryType::Neg:
        tiled_for_each([](const T* in, T* out, int64_t m, int64_t n,
                          int64_t ip, int64_t op, bool at) {
            kernel::neg<T>(in, out, m, n, ip, op, at);
        });
        break;

    case UnaryType::Sqrt:
        tiled_for_each([](const T* in, T* out, int64_t m, int64_t n,
                          int64_t ip, int64_t op, bool at) {
            kernel::sqrt<T>(in, out, m, n, ip, op, at);
        });
        break;

    case UnaryType::Erf:
        tiled_for_each([](const T* in, T* out, int64_t m, int64_t n,
                          int64_t ip, int64_t op, bool at) {
            kernel::erf<T>(in, out, m, n, ip, op, at);
        });
        break;

    case UnaryType::Round:
        tiled_for_each([](const T* in, T* out, int64_t m, int64_t n,
                          int64_t ip, int64_t op, bool at) {
            kernel::round<T>(in, out, m, n, ip, op, at);
        });
        break;

    case UnaryType::Ceil:
        tiled_for_each([](const T* in, T* out, int64_t m, int64_t n,
                          int64_t ip, int64_t op, bool at) {
            kernel::ceil<T>(in, out, m, n, ip, op, at);
        });
        break;

    case UnaryType::Floor:
        tiled_for_each([](const T* in, T* out, int64_t m, int64_t n,
                          int64_t ip, int64_t op, bool at) {
            kernel::floor<T>(in, out, m, n, ip, op, at);
        });
        break;

    case UnaryType::Recip:
        tiled_for_each([](const T* in, T* out, int64_t m, int64_t n,
                          int64_t ip, int64_t op, bool at) {
            kernel::recip<T>(in, out, m, n, ip, op, at);
        });
        break;

    case UnaryType::Sign:
        tiled_for_each([](const T* in, T* out, int64_t m, int64_t n,
                          int64_t ip, int64_t op, bool at) {
            kernel::sign<T>(in, out, m, n, ip, op, at);
        });
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
