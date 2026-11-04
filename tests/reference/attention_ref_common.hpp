#pragma once
/// @file attention_ref_common.hpp
/// @brief Naive f32 attention, shared by the three operators' references.
///
/// Rank-3 Q/K/V are merged [B, S, H*D] with the heads interleaved inside each
/// row; rank-4 are explicit [B, H, S, D]. Query head h reads kv head
/// h / (H / H_kv), and when @p v_is_k the value is the leading @p v_head_dim
/// channels of the key row (MLA's latent) rather than a third tensor.
///
/// Per head: the Q·K dot over the key width, scaled, plus the optional
/// [Sq, Sk] mask, softmaxed per row over its causal prefix, then multiplied by
/// V into the output. A row whose causal prefix is empty is all zeros.
///
/// This is deliberately the slowest possible spelling — no blocking, no
/// packing — so it can serve as ground truth for the tiled kernels.

#include "nnops/core/tensor_view.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/parallel_for.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <vector>

namespace nnops::backend::cpu::reference::detail {

inline void attention_ref_generic(int64_t num_heads, int64_t num_kv_heads,
                                  int64_t v_head_dim,
                                  float scale, bool causal, bool v_is_k,
                                  TensorView& output,
                                  std::span<const TensorView> inputs,
                                  const ComputeContext& ctx)
{
    const auto& Q = inputs[0];
    const auto& K = inputs[1];
    const auto& V = v_is_k ? inputs[1] : inputs[2];
    const bool has_mask = inputs.size() > (v_is_k ? 2u : 3u);

    const bool merged = (Q.rank() == 3);
    const int64_t B   = Q.shape(0);
    const int64_t H   = num_heads;
    const int64_t Hkv = num_kv_heads > 0 ? num_kv_heads : num_heads;

    int64_t Sq, Sk, Sv, Dqk;
    if (merged) {
        Sq  = Q.shape(1);
        Sk  = K.shape(1);
        Sv  = V.shape(1);
        Dqk = Q.shape(2) / H;
    } else {
        Sq  = Q.shape(2);
        Sk  = K.shape(2);
        Sv  = V.shape(2);
        Dqk = Q.shape(3);
    }
    const int64_t Dv = v_is_k ? v_head_dim : Dqk;

    const float s = (scale == 0.0f)
        ? (1.0f / std::sqrt(static_cast<float>(Dqk)))
        : scale;

    const auto* q_ptr = Q.ptr<float>();
    const auto* k_ptr = K.ptr<float>();
    const auto* v_ptr = V.ptr<float>();
    const auto* mask_ptr = has_mask ? inputs[v_is_k ? 2 : 3].ptr<float>() : nullptr;
    auto* out_ptr = output.ptr<float>();

    const int64_t q_rs = Q.row_stride_elems();
    const int64_t k_rs = K.row_stride_elems();
    const int64_t v_rs = V.row_stride_elems();
    const int64_t o_rs = output.row_stride_elems();
    // A latent key row is Dqk wide per head; the value is its prefix.
    const int64_t v_head_stride = v_is_k ? Dqk : Dv;

    const auto compute_head_batch = [&](int64_t idx) {
        const int64_t b   = idx / H;
        const int64_t h   = idx % H;
        const int64_t hkv = h / (H / Hkv);

        const float* q_head;
        const float* k_head;
        const float* v_head;
        float* o_head;
        if (merged) {
            q_head = q_ptr + b * Sq * q_rs + h * Dqk;
            k_head = k_ptr + b * Sk * k_rs + hkv * Dqk;
            v_head = v_ptr + b * Sv * v_rs + hkv * v_head_stride;
            o_head = out_ptr + b * Sq * o_rs + h * Dv;
        } else {
            q_head = q_ptr + (b * H + h) * Sq * q_rs;
            k_head = k_ptr + (b * Hkv + hkv) * Sk * k_rs;
            v_head = v_ptr + (b * Hkv + hkv) * Sv * v_rs;
            o_head = out_ptr + (b * H + h) * Sq * o_rs;
        }

        std::vector<float> scores(static_cast<size_t>(Sq * Sk), 0.0f);

        for (int64_t i = 0; i < Sq; ++i) {
            for (int64_t j = 0; j < Sk; ++j) {
                float dot = 0.0f;
                for (int64_t d = 0; d < Dqk; ++d) {
                    dot += q_head[i * q_rs + d] * k_head[j * k_rs + d];
                }
                scores[static_cast<size_t>(i * Sk + j)] = dot * s;
            }
        }

        if (mask_ptr) {
            for (int64_t i = 0; i < Sq; ++i) {
                for (int64_t j = 0; j < Sk; ++j) {
                    scores[static_cast<size_t>(i * Sk + j)] += mask_ptr[i * Sk + j];
                }
            }
        }

        // Causal prefix, bottom-right aligned; the tail is never read.
        for (int64_t i = 0; i < Sq; ++i) {
            int64_t n = Sk;
            if (causal) {
                n = i + (Sk - Sq) + 1;
                n = std::max<int64_t>(0, std::min<int64_t>(n, Sk));
            }
            float* row = scores.data() + i * Sk;

            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t j = 0; j < n; ++j) { max_val = std::max(max_val, row[j]); }

            if (n == 0) {
                for (int64_t j = 0; j < Sk; ++j) { row[j] = 0.0f; }
                continue;
            }

            float sum = 0.0f;
            for (int64_t j = 0; j < n; ++j) {
                row[j] = std::exp(row[j] - max_val);
                sum += row[j];
            }
            const float inv_sum = 1.0f / sum;
            for (int64_t j = 0; j < n; ++j) { row[j] *= inv_sum; }
            for (int64_t j = n; j < Sk; ++j) { row[j] = 0.0f; }
        }

        for (int64_t i = 0; i < Sq; ++i) {
            for (int64_t d = 0; d < Dv; ++d) {
                float sum = 0.0f;
                for (int64_t j = 0; j < Sk; ++j) {
                    const float attn = scores[static_cast<size_t>(i * Sk + j)];
                    if (std::isinf(attn) || std::isnan(attn)) { continue; }
                    sum += attn * v_head[j * v_rs + d];
                }
                o_head[i * o_rs + d] = sum;
            }
        }
    };

    const int64_t total = B * H;
    ctx.cpu.run(0, total, [&](int64_t idx) { compute_head_batch(idx); });
}

}  // namespace nnops::backend::cpu::reference::detail
