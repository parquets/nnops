/// @file non_max_suppression.cpp
/// @brief CPU NonMaxSuppression — per-(batch, class) greedy IoU suppression.
///
/// One (batch, class) at a time: score-threshold the candidates, keep the
/// highest-scoring `top_k` (a partial sort, O(n log k), not a full sort of the
/// anchors), then suppress boxes whose IoU with an already-kept box is above
/// the threshold. The candidates and the kept indices come from the CPU
/// backend's scratch pool, so getWorkspaceSize stays 0.
///
/// The per-class order is (score desc, box index asc), which makes ties
/// deterministic — ONNX leaves ties unspecified. The output is written in
/// (batch, class) order and truncated at the capacity the caller allocated;
/// num_selected[0] is the number of rows actually written.

#include "nnops/ops/non_max_suppression.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "common/dtype_dispatch.hpp"
#include "common/memory_pool.hpp"

#include <algorithm>
#include <cmath>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

/// ONNX corner form: y1, x1, y2, x2.
struct Box {
    float y1, x1, y2, x2;
};

inline float box_area(const Box& b)
{
    const float w = std::max(0.0f, b.x2 - b.x1);
    const float h = std::max(0.0f, b.y2 - b.y1);
    return w * h;
}

inline float box_iou(const Box& a, const Box& b)
{
    const float iw = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    const float ih = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    const float inter = iw * ih;
    if (inter <= 0.0f) {
        return 0.0f;
    }
    const float uni = box_area(a) + box_area(b) - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

struct Candidate {
    float score;
    int64_t index;
};

template <typename T>
void nms_impl(const NonMaxSuppressionAttributes& attrs, TensorView& indices_out,
              TensorView& count_out, std::span<const TensorView> inputs)
{
    const auto& boxes = inputs[0];
    const auto& scores = inputs[1];

    const int64_t B = boxes.shape(0);
    const int64_t N = boxes.shape(1);
    const int64_t C = scores.shape(1);

    const bool has_max = inputs.size() >= 3 && !inputs[2].is_empty();
    const bool has_iou = inputs.size() >= 4 && !inputs[3].is_empty();
    const bool has_score = inputs.size() >= 5 && !inputs[4].is_empty();

    int64_t max_out = 0;
    if (has_max) {
        const auto& v = inputs[2];
        max_out = (v.data_type() == DataType::s32) ? static_cast<int64_t>(v.ptr<int32_t>()[0])
                                                   : v.ptr<int64_t>()[0];
        if (max_out < 0) {
            max_out = 0;
        }
    }
    const float iou_thr = has_iou ? inputs[3].ptr<float>()[0] : 0.0f;
    const float score_thr = has_score ? inputs[4].ptr<float>()[0] : 0.0f;

    const bool center_box = attrs.center_point_box == 1;
    const int64_t top_k = attrs.top_k;

    auto* out = indices_out.ptr<int64_t>();
    const int64_t cap = indices_out.shape(0);

    const auto* boxes_ptr = boxes.ptr<T>();
    const auto* scores_ptr = scores.ptr<T>();
    const int64_t bs0 = boxes.stride_elems(0);
    const int64_t bs1 = boxes.stride_elems(1);
    const int64_t ss0 = scores.stride_elems(0);
    const int64_t ss1 = scores.stride_elems(1);

    int64_t written = 0;
    if (max_out == 0 || cap == 0 || N == 0) {
        count_out.ptr<int64_t>()[0] = 0;
        return;
    }

    PoolArray<Candidate> cand(static_cast<size_t>(N));
    PoolArray<int64_t> kept(static_cast<size_t>(std::min(max_out, N)));

    const auto box_at = [&](int64_t batch_base, int64_t i) {
        const T* p = boxes_ptr + batch_base + i * bs1;
        const float a = s_load(&p[0]);
        const float b = s_load(&p[1]);
        const float c = s_load(&p[2]);
        const float d = s_load(&p[3]);
        if (center_box) {
            // [x_center, y_center, width, height] -> [y1, x1, y2, x2].
            return Box{b - d * 0.5f, a - c * 0.5f, b + d * 0.5f, a + c * 0.5f};
        }
        // ONNX's corner form is [y1, x1, y2, x2].
        return Box{a, b, c, d};
    };

    for (int64_t bi = 0; bi < B && written < cap; ++bi) {
        const int64_t base_b = bi * bs0;
        for (int64_t ci = 0; ci < C && written < cap; ++ci) {
            const int64_t base_s = bi * ss0 + ci * ss1;

            int64_t m = 0;
            for (int64_t i = 0; i < N; ++i) {
                const float s = s_load(&scores_ptr[base_s + i]);
                if (s > score_thr) {
                    cand[static_cast<size_t>(m)] = Candidate{s, i};
                    ++m;
                }
            }
            if (m == 0) {
                continue;
            }

            const auto by_score = [](const Candidate& a, const Candidate& b) {
                if (a.score != b.score) {
                    return a.score > b.score;
                }
                return a.index < b.index;
            };
            if (top_k > 0 && m > top_k) {
                std::partial_sort(cand.begin(), cand.begin() + top_k, cand.begin() + m,
                                  by_score);
                m = top_k;
            } else {
                std::sort(cand.begin(), cand.begin() + m, by_score);
            }

            int64_t nkept = 0;
            for (int64_t i = 0; i < m && nkept < max_out; ++i) {
                const int64_t idx = cand[static_cast<size_t>(i)].index;
                const Box box = box_at(base_b, idx);
                bool keep = true;
                for (int64_t k = 0; k < nkept; ++k) {
                    if (box_iou(box, box_at(base_b, kept[static_cast<size_t>(k)])) > iou_thr) {
                        keep = false;
                        break;
                    }
                }
                if (!keep) {
                    continue;
                }
                kept[static_cast<size_t>(nkept)] = idx;
                ++nkept;
                if (written < cap) {
                    out[written * 3 + 0] = bi;
                    out[written * 3 + 1] = ci;
                    out[written * 3 + 2] = idx;
                    ++written;
                }
            }
        }
    }

    count_out.ptr<int64_t>()[0] = written;
}

} // anonymous namespace

Status nms_cpu(const NonMaxSuppressionAttributes& attrs, TensorView& indices,
               TensorView& num_selected, std::span<const TensorView> inputs,
               const ComputeContext& ctx, void* /*workspace*/)
{
    (void)ctx;  // one (batch, class) at a time; see the file comment
    dispatch_f32_f16(inputs[0].data_type(), "nms_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        nms_impl<T>(attrs, indices, num_selected, inputs);
    });
    return Status::ok;
}

} // namespace nnops::backend::cpu
