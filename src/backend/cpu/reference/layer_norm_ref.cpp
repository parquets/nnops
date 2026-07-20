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
                    const TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* /*workspace*/)
{
    const auto& X     = inputs[0];
    const auto& scale = inputs[1];
    const bool has_bias = (inputs.size() >= 3 && inputs[2].data() != nullptr);

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

    // Scale/bias broadcast strides within the normalized shape
    const int64_t scale_rank = scale.rank();
    // scale.shape should match the last (rank - axis) dims of X
    // For simple broadcasting: compute scale offset from position within norm
    auto compute_scale_idx = [&](int64_t row_start, int64_t elem_offset) -> int64_t {
        // Given a linear offset within the normalized dims, compute scale index
        if (scale_rank == 1 && norm_size == scale.shape(0)) {
            return elem_offset;
        }
        // For now, handle the common case: scale shape == last dims of X
        // More complex broadcasting can be added later
        if (scale.numel() == 1) return 0;
        return elem_offset;  // assume same shape
    };

    const auto* x_ptr  = X.data_as<float>();
    const auto* s_ptr  = scale.data_as<float>();
    const auto* b_ptr  = has_bias ? inputs[2].data_as<float>() : nullptr;
    auto* y_ptr = output.data_as<float>();

    const auto process_row = [&](int64_t row) {
        int64_t row_start = row * norm_size;

        // Welford's online algorithm for mean and variance
        float mean_val = 0.0f;
        float M2 = 0.0f;
        for (int64_t i = 0; i < norm_size; ++i) {
            float x = x_ptr[row_start + i];
            float delta = x - mean_val;
            mean_val += delta / static_cast<float>(i + 1);
            float delta2 = x - mean_val;
            M2 += delta * delta2;
        }
        float var_val = M2 / static_cast<float>(norm_size);
        float inv_std = 1.0f / std::sqrt(var_val + epsilon);

        // Apply normalization with scale and bias
        for (int64_t i = 0; i < norm_size; ++i) {
            int64_t s_idx = compute_scale_idx(row_start, i);
            float s = s_ptr[s_idx];
            float b = b_ptr ? b_ptr[s_idx] : 0.0f;
            y_ptr[row_start + i] = (x_ptr[row_start + i] - mean_val) * inv_std * s + b;
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
