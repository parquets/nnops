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
#include "simd_kernel/tiled_map.hpp"
#include "common/elementwise.hpp"

#include <algorithm>

namespace nnops::backend::cpu {

template <typename T>
void clamp_impl(const ClampAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx)
{
    const float min_val = attrs.min_val;
    const float max_val = attrs.max_val;

    // Bounds are invariant across the tensor, so precompute the vectors once
    // and reuse them for every row/tile.
    const auto vmin = simd::v_set1(inputs[0].ptr<T>(), min_val);
    const auto vmax = simd::v_set1(inputs[0].ptr<T>(), max_val);

    tiled_float_transform<T>(inputs[0], output, ctx,
        [&](const T* in, T* out, int64_t m, int64_t n, int64_t in_stride, int64_t out_stride) {
            kernel::tiled_map_simd<T, 1>({in}, {in_stride}, out, out_stride, m, n, attrs.add_to,
                [vmin, vmax](auto x) { return simd::v_max(simd::v_min(x, vmax), vmin); },
                [min_val, max_val](float v) { return std::max(min_val, std::min(max_val, v)); });
        });
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
