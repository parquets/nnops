/// @file layer_norm_ref.cpp
/// @brief Naive CPU reference implementation of layer normalization.
///
/// Uses Welford's online algorithm for numerically stable mean/variance.
/// Normalizes over X.shape[axis:] and broadcasts scale/bias.

#include "nnops/ops/layer_norm.hpp"
#include "nnops/core/parallel_for.hpp"

#include <cmath>
#include <vector>

namespace nnops::backend::cpu::reference {

void layer_norm_ref(const LayerNormAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* /*workspace*/)
{
    const auto& X     = inputs[0];
    const auto& scale = inputs[1];
    const bool has_bias = (inputs.size() >= 3 && !inputs[2].is_empty());

    const int64_t rank = X.rank();
    const float epsilon = attrs.epsilon;

    // Normalize axis
    int64_t axis = attrs.axis;
    if (axis < 0) axis += rank;
    NNOPS_ASSERT(axis >= 0 && axis < rank);

    // Number of rows and norm_size per row
    int64_t num_rows = 1;
    for (int64_t i = 0; i < axis; ++i) {
        num_rows *= X.shape(i);
    }
    int64_t norm_size = 1;
    for (int64_t i = axis; i < rank; ++i) {
        norm_size *= X.shape(i);
    }

    const auto* x_ptr  = X.ptr<float>();
    const auto* s_ptr  = scale.ptr<float>();
    const auto* b_ptr  = has_bias ? inputs[2].ptr<float>() : nullptr;
    auto* y_ptr = output.ptr<float>();

    // Compute the element stride of the axis dimension (accounts for pitch).
    // For [B, C, H, W] with axis=1: stride_elems(1) = H * row_stride.
    const int64_t outer_stride = (axis > 0) ? X.stride_elems(axis - 1) : 0;

    // Convert a flat index within the normalized tail (dims axis..rank-1)
    // to an element offset from the start of the normalization group.
    auto inner_offset = [&](int64_t flat_idx) -> int64_t {
        int64_t off = 0;
        int64_t rem = flat_idx;
        for (int64_t i = rank - 1; i >= axis; --i) {
            int64_t dim = X.shape(i);
            off += (rem % dim) * X.stride_elems(i);
            rem /= dim;
        }
        return off;
    };

    // Scale/bias offset: for now handle the common case where scale shape
    // matches the normalized dims.
    auto scale_offset = [&](int64_t flat_idx) -> int64_t {
        if (scale.numel() == 1) return 0;
        if (scale.rank() == 1 && norm_size == scale.shape(0)) return flat_idx;
        // Default: assume matching shape
        return flat_idx;
    };

    const auto process_row = [&](int64_t row) {
        int64_t row_base = row * outer_stride;

        // Welford's online algorithm for mean and variance
        float mean_val = 0.0f;
        float M2 = 0.0f;
        for (int64_t i = 0; i < norm_size; ++i) {
            int64_t off = row_base + inner_offset(i);
            float x = x_ptr[off];
            float delta = x - mean_val;
            mean_val += delta / static_cast<float>(i + 1);
            float delta2 = x - mean_val;
            M2 += delta * delta2;
        }
        float var_val = M2 / static_cast<float>(norm_size);
        float inv_std = 1.0f / std::sqrt(var_val + epsilon);

        // Apply normalization with scale and bias
        for (int64_t i = 0; i < norm_size; ++i) {
            int64_t off = row_base + inner_offset(i);
            int64_t s_idx = scale_offset(i);
            float s = s_ptr[s_idx];
            float b = b_ptr ? b_ptr[s_idx] : 0.0f;
            float val = (x_ptr[off] - mean_val) * inv_std * s + b;
            y_ptr[off] = attrs.add_to ? y_ptr[off] + val : val;
        }
    };

    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, num_rows, process_row);
    } else {
        for (int64_t i = 0; i < num_rows; ++i) {
            process_row(i);
        }
    }
}

}  // namespace nnops::backend::cpu::reference
