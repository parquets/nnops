/// @file rms_norm_ref.cpp
/// @brief Naive CPU reference implementation of RMS normalization.
///
/// Computes RMS(x) = sqrt(mean(x^2) + epsilon), then y = x / RMS(x) * scale.
/// No mean subtraction and no bias (unlike LayerNorm).

#include "nnops/ops/rms_norm.hpp"
#include "nnops/core/parallel_for.hpp"

#include <cmath>

namespace nnops::backend::cpu::reference {

void rms_norm_ref(const RMSNormAttributes& attrs,
                  const TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx,
                  void* /*workspace*/)
{
    const auto& X     = inputs[0];
    const auto& scale = inputs[1];

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

    const auto* x_ptr = X.data_as<float>();
    const auto* s_ptr = scale.data_as<float>();
    auto* y_ptr = output.data_as<float>();

    // Scale is typically the same shape as norm_shape (last dims)
    const bool scale_is_scalar = (scale.numel() == 1);

    const auto process_row = [&](int64_t row) {
        int64_t row_start = row * norm_size;

        // Compute sum of squares
        float sum_sq = 0.0f;
        for (int64_t i = 0; i < norm_size; ++i) {
            float x = x_ptr[row_start + i];
            sum_sq += x * x;
        }
        float rms = std::sqrt(sum_sq / static_cast<float>(norm_size) + epsilon);

        // Apply RMS normalization with scale
        float inv_rms = 1.0f / rms;
        for (int64_t i = 0; i < norm_size; ++i) {
            float s = scale_is_scalar ? s_ptr[0] : s_ptr[i];
            y_ptr[row_start + i] = x_ptr[row_start + i] * inv_rms * s;
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
