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
                  TensorView& output,
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

    const auto* x_ptr = X.ptr<float>();
    const auto* s_ptr = scale.ptr<float>();
    auto* y_ptr = output.ptr<float>();

    // Compute the element stride of the axis dimension (accounts for pitch)
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

    // Scale is typically the same shape as norm_shape (last dims)
    const bool scale_is_scalar = (scale.numel() == 1);

    const auto process_row = [&](int64_t row) {
        int64_t row_base = row * outer_stride;

        // Compute sum of squares
        float sum_sq = 0.0f;
        for (int64_t i = 0; i < norm_size; ++i) {
            int64_t off = row_base + inner_offset(i);
            float x = x_ptr[off];
            sum_sq += x * x;
        }
        float rms = std::sqrt(sum_sq / static_cast<float>(norm_size) + epsilon);

        // Apply RMS normalization with scale
        float inv_rms = 1.0f / rms;
        for (int64_t i = 0; i < norm_size; ++i) {
            int64_t off = row_base + inner_offset(i);
            float s = scale_is_scalar ? s_ptr[0] : s_ptr[i];
            float val = x_ptr[off] * inv_rms * s;
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
