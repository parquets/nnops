/// @file activation.cpp
/// @brief SIMD-optimized CPU implementation of activation functions.
///
/// All 8 activation types are vectorized with the nnops SIMD abstraction layer.
/// Supports both f32 (v_f32x8) and f16 (v_f16x8) via a single templated implementation.
/// Processing is tiled in groups of 8 rows and dispatched via
/// ComputeContext::cpu_parallel_for when available.
///
/// Design:
///   1. Rows are grouped into tiles of 8 for SIMD-friendly blocking
///   2. Each tile calls into tiled_activation kernel library
///   3. Tile iteration is parallelized via ctx.cpu_parallel_for
///   4. All SIMD constants are pre-computed inside the tiled kernels

#include "nnops/ops/activation.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "simd_kernel/simd_activation.hpp"
#include "simd_kernel/activation_kernels.hpp"

#include <algorithm>
#include <cmath>

namespace nnops::backend::cpu {

// ============================================================
// Templated implementation (f32 and f16)
// ============================================================

template <typename T>
void activation_impl(const ActivationAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t total = input.numel();
    if (total == 0) { return; }

    const int64_t rank = input.rank();

    // Row-by-row layout (pitch-aware).
    const int64_t last_dim = (rank >= 1) ? input.shape(rank - 1) * input.channel_pack_size() : 1;
    const int64_t num_rows = (rank >= 2) ? input.total_rows() : total / last_dim;
    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t out_row_stride = output.row_stride_elems();

    const auto* in_ptr  = input.ptr<T>();
    auto*       out_ptr = output.ptr<T>();
    const bool  add_to  = attrs.add_to;

    constexpr int64_t TILE_M = 32;
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
        if (ctx.cpu_parallel_for) {
            ctx.cpu_parallel_for(0, num_tiles, body);
        }
        else {
            for (int64_t t = 0; t < num_tiles; ++t) {
                body(t);
            }
        }
    };

    switch (attrs.type) {

    case ActivationType::Relu:
        tiled_for_each([](const T* in, T* out, int64_t m, int64_t n,
                          int64_t ip, int64_t op, bool at) {
            kernel::relu<T>(in, out, m, n, ip, op, at);
        });
        break;

    case ActivationType::LeakyRelu:
        tiled_for_each([alpha = attrs.alpha](const T* in, T* out, int64_t m, int64_t n,
                                              int64_t ip, int64_t op, bool at) {
            kernel::leaky_relu<T>(in, out, m, n, ip, op, at, alpha);
        });
        break;

    case ActivationType::Sigmoid:
        tiled_for_each([](const T* in, T* out, int64_t m, int64_t n,
                          int64_t ip, int64_t op, bool at) {
            kernel::sigmoid<T>(in, out, m, n, ip, op, at);
        });
        break;

    case ActivationType::Tanh:
        tiled_for_each([](const T* in, T* out, int64_t m, int64_t n,
                          int64_t ip, int64_t op, bool at) {
            kernel::tanh<T>(in, out, m, n, ip, op, at);
        });
        break;

    case ActivationType::Gelu:
        tiled_for_each([](const T* in, T* out, int64_t m, int64_t n,
                          int64_t ip, int64_t op, bool at) {
            kernel::gelu<T>(in, out, m, n, ip, op, at);
        });
        break;

    case ActivationType::Silu:
        tiled_for_each([](const T* in, T* out, int64_t m, int64_t n,
                          int64_t ip, int64_t op, bool at) {
            kernel::silu<T>(in, out, m, n, ip, op, at);
        });
        break;

    case ActivationType::HardSwish:
        tiled_for_each([beta = attrs.beta](const T* in, T* out, int64_t m, int64_t n,
                                            int64_t ip, int64_t op, bool at) {
            kernel::hard_swish<T>(in, out, m, n, ip, op, at, beta);
        });
        break;

    case ActivationType::Elu:
        tiled_for_each([alpha = attrs.alpha](const T* in, T* out, int64_t m, int64_t n,
                                              int64_t ip, int64_t op, bool at) {
            kernel::elu<T>(in, out, m, n, ip, op, at, alpha);
        });
        break;

    }  // switch
}

// ============================================================
// Main entry point with dtype dispatch
// ============================================================

void activation_cpu(const ActivationAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        activation_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        activation_impl<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"activation_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
