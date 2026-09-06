/// @file attention_ref.cpp
/// @brief Naive CPU reference implementation of multi-head scaled dot-product attention.

#include "nnops/ops/attention.hpp"
#include "nnops/core/parallel_for.hpp"

#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>

namespace nnops::backend::cpu::reference {

namespace {

/// Softmax over the last dimension of a [rows, cols] matrix, in-place.
void softmax_last_dim(float* data, int64_t rows, int64_t cols) {
    for (int64_t r = 0; r < rows; ++r) {
        float* row = data + r * cols;
        // Find max for numerical stability
        float max_val = -std::numeric_limits<float>::infinity();
        for (int64_t c = 0; c < cols; ++c) {
            max_val = std::max(max_val, row[c]);
        }
        // Exp and sum
        float sum = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            row[c] = std::exp(row[c] - max_val);
            sum += row[c];
        }
        // Normalize
        float inv_sum = 1.0f / sum;
        for (int64_t c = 0; c < cols; ++c) {
            row[c] *= inv_sum;
        }
    }
}

}  // namespace

void attention_ref(const AttentionAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx,
                   void* /*workspace*/)
{
    const auto& Q = inputs[0];
    const auto& K = inputs[1];
    const auto& V = inputs[2];
    const bool has_mask = inputs.size() > 3;

    // Detect layout: rank 3 = [B, S, H*D], rank 4 = [B, H, S, D]
    const bool merged_heads = (Q.rank() == 3);

    const int64_t B  = Q.shape(0);
    const int64_t H  = attrs.num_heads;
    int64_t Sq, Sk, Sv, D;
    if (merged_heads) {
        Sq = Q.shape(1);
        Sk = K.shape(1);
        Sv = V.shape(1);
        D  = Q.shape(2) / H;
    } else {
        Sq = Q.shape(2);
        Sk = K.shape(2);
        Sv = V.shape(2);
        D  = Q.shape(3);
    }

    const float scale = (attrs.scale == 0.0f)
        ? (1.0f / std::sqrt(static_cast<float>(D)))
        : attrs.scale;

    const auto* q_ptr = Q.ptr<float>();
    const auto* k_ptr = K.ptr<float>();
    const auto* v_ptr = V.ptr<float>();
    const auto* mask_ptr = has_mask ? inputs[3].ptr<float>() : nullptr;
    auto* out_ptr = output.ptr<float>();

    // Row strides in elements (from pitch, accounts for padding)
    const int64_t q_row_stride = Q.row_stride_elems();  // merged: >= H*D, explicit: >= D
    const int64_t k_row_stride = K.row_stride_elems();
    const int64_t v_row_stride = V.row_stride_elems();
    const int64_t o_row_stride = output.row_stride_elems();

    // Helper to read Q/K/V elements accounting for merged vs. explicit layout
    auto read_elem = [&](const float* ptr, int64_t b, int64_t h,
                         int64_t s, int64_t d, int64_t row_stride, bool merged) -> float {
        if (merged) {
            return ptr[(b * Sq + s) * row_stride + h * D + d];
        } else {
            return ptr[((b * H + h) * Sq + s) * row_stride + d];
        }
    };

    auto write_elem = [&](float* ptr, int64_t b, int64_t h,
                          int64_t s, int64_t d, float val, int64_t row_stride, bool merged) {
        int64_t idx = merged
            ? (b * Sq + s) * row_stride + h * D + d
            : ((b * H + h) * Sq + s) * row_stride + d;
        ptr[idx] = val;
    };

    // Per-head compute lambda
    const auto compute_head_batch = [&](int64_t idx) {
        const int64_t b = idx / H;
        const int64_t h = idx % H;

        // Allocate scratch for scores: [Sq, Sk]
        std::vector<float> scores(static_cast<size_t>(Sq * Sk), 0.0f);

        // Step 1: scores = Q @ K^T * scale
        for (int64_t i = 0; i < Sq; ++i) {
            for (int64_t j = 0; j < Sk; ++j) {
                float dot = 0.0f;
                for (int64_t d = 0; d < D; ++d) {
                    float qv = read_elem(q_ptr, b, h, i, d, q_row_stride, merged_heads);
                    float kv = read_elem(k_ptr, b, h, j, d, k_row_stride, merged_heads);
                    dot += qv * kv;
                }
                scores[i * Sk + j] = dot * scale;
            }
        }

        // Step 2: Apply explicit mask if provided
        if (has_mask) {
            for (int64_t i = 0; i < Sq; ++i) {
                for (int64_t j = 0; j < Sk; ++j) {
                    scores[i * Sk + j] += mask_ptr[i * Sk + j];
                }
            }
        }

        // Step 4: Softmax along last dimension
        softmax_last_dim(scores.data(), Sq, Sk);

        // Step 5: output = attn @ V
        for (int64_t i = 0; i < Sq; ++i) {
            for (int64_t d = 0; d < D; ++d) {
                float sum = 0.0f;
                for (int64_t j = 0; j < Sk; ++j) {
                    float attn = scores[i * Sk + j];
                    if (std::isinf(attn) || std::isnan(attn)) { continue; }
                    float vv = read_elem(v_ptr, b, h, j, d, v_row_stride, merged_heads);
                    sum += attn * vv;
                }
                write_elem(out_ptr, b, h, i, d, sum, o_row_stride, merged_heads);
            }
        }
    };

    const int64_t total = B * H;
    ctx.cpu.run(0, total,
            [&](int64_t idx) { compute_head_batch(idx); });
}

}  // namespace nnops::backend::cpu::reference
