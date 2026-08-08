/// @file clamp.cpp
/// @brief SIMD-optimized CPU implementation of element-wise Clamp.
///
/// Supports both f32 and f16 via a single templated implementation.
/// Processing is tiled in groups of 32 rows and dispatched via
/// ComputeContext::cpu_parallel_for when available.
///
/// Core operation: v_max(v_min(x, max_val), min_val) — 2 SIMD instructions.

#include "nnops/ops/clamp.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <algorithm>
#include <limits>

namespace nnops::backend::cpu {

template <typename T>
void clamp_impl(const ClampAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t total = input.numel();
    if (total == 0) { return; }

    const int64_t rank = input.rank();
    NNOPS_ASSERT(output.numel() == total);

    // Row-by-row layout (pitch-aware, supports packed layouts).
    const int64_t last_dim = (rank >= 1) ? input.shape(rank - 1) * input.channel_pack_size() : 1;
    const int64_t num_rows = (rank >= 2) ? input.total_rows() : total / last_dim;
    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t out_row_stride = output.row_stride_elems();

    const auto* in_ptr  = input.ptr<T>();
    auto*       out_ptr = output.ptr<T>();
    const bool  add_to  = attrs.add_to;
    const float min_val = attrs.min_val;
    const float max_val = attrs.max_val;

    constexpr int64_t TILE_M = 32;
    const int64_t num_tiles = (num_rows + TILE_M - 1) / TILE_M;

    // SIMD lane width
    constexpr int64_t LANE = simd::simd_lane_for<T>;

    auto body = [&](int64_t ti) {
        int64_t r = ti * TILE_M;
        int64_t m = std::min(TILE_M, num_rows - r);

        for (int64_t row = 0; row < m; ++row) {
            const T* in_row = in_ptr + (r + row) * in_row_stride;
            T* out_row = out_ptr + (r + row) * out_row_stride;

            int64_t i = 0;
            auto vmin = simd::v_set1(in_row, min_val);
            auto vmax = simd::v_set1(in_row, max_val);
            for (; i + LANE <= last_dim; i += LANE) {
                auto x = simd::v_load(&in_row[i]);
                auto y = simd::v_max(simd::v_min(x, vmax), vmin);
                simd::v_store_add(&out_row[i], y, add_to);
            }
            // Scalar tail
            for (; i < last_dim; ++i) {
                float v = simd::s_load(&in_row[i]);
                float rv = std::max(min_val, std::min(max_val, v));
                simd::s_store_add(&out_row[i], rv, add_to);
            }
        }
    };

    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, num_tiles, body);
    }
    else {
        for (int64_t t = 0; t < num_tiles; ++t) {
            body(t);
        }
    }
}

void clamp_cpu(const ClampAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx,
                 void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        clamp_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        clamp_impl<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"clamp_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
