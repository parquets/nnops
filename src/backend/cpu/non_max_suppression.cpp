/// @file non_max_suppression.cpp
/// @brief CPU NonMaxSuppression — per-(batch, class) greedy IoU suppression.
///
/// One (batch, class) at a time: score-threshold the candidates, keep the
/// highest-scoring `top_k` (partial sort for a small k, quickselect + sort when
/// k is a large slice — never a full sort of the anchors), then suppress boxes
/// whose IoU with an already-kept box is above the threshold. The suppression
/// scans eight kept boxes per vector: the selected set is held as separate
/// x1/y1/x2/y2/area planes, so the check is max over lanes with no gathers.
///
/// The box corners and areas are decoded once per batch, not once per class and
/// not once per comparison: the suppression loop then reads a packed
/// [x1, y1, x2, y2, area] array instead of re-reading and re-normalizing the
/// boxes for every pair (the shape onnxruntime's SuppressByIOU pays for). The
/// candidates and the kept indices come from the CPU backend's scratch pool,
/// so getWorkspaceSize stays 0.
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
#include <limits>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

/// Corner form, normalized (x1 <= x2, y1 <= y2), with the area precomputed.
struct Box {
    float x1, y1, x2, y2, area;
};

/// Scalar IoU, only for the < 8-box tail of the selected set.
inline float box_iou(const Box& a, const Box& b)
{
    if (a.area <= 0.0f || b.area <= 0.0f) {
        return 0.0f;
    }
    const float iw = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    if (iw <= 0.0f) {
        return 0.0f;
    }
    const float ih = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
    if (ih <= 0.0f) {
        return 0.0f;
    }
    const float inter = iw * ih;
    const float uni = a.area + b.area - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

/// Largest IoU between @p box and the `n` selected boxes in the structure-of-
/// arrays planes. The division is by `max(union, tiny)`: a degenerate union
/// carries zero intersection, so the guarded quotient is 0 rather than NaN,
/// and the max over lanes needs no mask arithmetic.
inline float max_iou_against_selected(const Box& box, const float* sx1, const float* sy1,
                                      const float* sx2, const float* sy2, const float* sarea,
                                      int64_t n)
{
    constexpr int64_t L = simd_default_lane_f32;
    float max_iou = 0.0f;
    int64_t k = 0;
    for (; k + L <= n; k += L) {
        const auto ix1 = v_max(v_load(sx1 + k), v_set1_f32x8(box.x1));
        const auto ix2 = v_min(v_load(sx2 + k), v_set1_f32x8(box.x2));
        const auto iw = v_max(v_sub(ix2, ix1), v_zero_f32x8());
        const auto iy1 = v_max(v_load(sy1 + k), v_set1_f32x8(box.y1));
        const auto iy2 = v_min(v_load(sy2 + k), v_set1_f32x8(box.y2));
        const auto ih = v_max(v_sub(iy2, iy1), v_zero_f32x8());
        const auto inter = v_mul(iw, ih);
        const auto uni =
            v_sub(v_add(v_load(sarea + k), v_set1_f32x8(box.area)), inter);
        const auto ratio = v_div(inter, v_max(uni, v_set1_f32x8(1e-30f)));
        max_iou = std::max(max_iou, v_reduce_max(ratio));
    }
    for (; k < n; ++k) {
        max_iou = std::max(max_iou, box_iou(box, Box{sx1[k], sy1[k], sx2[k], sy2[k], sarea[k]}));
    }
    return max_iou;
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

    const size_t keep_cap = static_cast<size_t>(std::min(max_out, N));
    PoolArray<Candidate> cand(static_cast<size_t>(N));
    PoolArray<Box> corners(static_cast<size_t>(N));
    // The selected set of the current (batch, class), as separate planes so the
    // suppression check is a straight vector walk.
    PoolArray<float> sx1(keep_cap), sy1(keep_cap), sx2(keep_cap), sy2(keep_cap),
        sarea(keep_cap);

    for (int64_t bi = 0; bi < B && written < cap; ++bi) {
        const int64_t base_b = bi * bs0;

        // Decode the batch's boxes once; both classes and every IoU read this.
        for (int64_t i = 0; i < N; ++i) {
            const T* p = boxes_ptr + base_b + i * bs1;
            const float a = s_load(&p[0]);
            const float b = s_load(&p[1]);
            const float c = s_load(&p[2]);
            const float d = s_load(&p[3]);
            Box bx;
            if (center_box) {
                // [x_center, y_center, width, height].
                bx.x1 = a - c * 0.5f;
                bx.x2 = a + c * 0.5f;
                bx.y1 = b - d * 0.5f;
                bx.y2 = b + d * 0.5f;
            } else {
                // ONNX's corner form is [y1, x1, y2, x2].
                bx.y1 = a;
                bx.y2 = c;
                bx.x1 = b;
                bx.x2 = d;
            }
            if (bx.x2 < bx.x1) {
                std::swap(bx.x1, bx.x2);
            }
            if (bx.y2 < bx.y1) {
                std::swap(bx.y1, bx.y2);
            }
            bx.area = (bx.x2 - bx.x1) * (bx.y2 - bx.y1);
            corners[static_cast<size_t>(i)] = bx;
        }

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
                // partial_sort is O(m log k) and wins while k is a small slice
                // of the candidates; quickselect + sort of the survivors is
                // O(m + k log k) and wins once k is more than ~1/16 of them
                // (measured: 1.5x faster at k = 1000 of 8000, 2x slower at
                // k = 100).
                if (top_k * 16 < m) {
                    std::partial_sort(cand.begin(), cand.begin() + top_k, cand.begin() + m,
                                      by_score);
                } else {
                    std::nth_element(cand.begin(), cand.begin() + top_k, cand.begin() + m,
                                     by_score);
                    std::sort(cand.begin(), cand.begin() + top_k, by_score);
                }
                m = top_k;
            } else {
                std::sort(cand.begin(), cand.begin() + m, by_score);
            }

            int64_t nkept = 0;
            for (int64_t i = 0; i < m && nkept < max_out; ++i) {
                const int64_t idx = cand[static_cast<size_t>(i)].index;
                const Box& box = corners[static_cast<size_t>(idx)];
                if (nkept > 0 &&
                    max_iou_against_selected(box, sx1.data(), sy1.data(), sx2.data(), sy2.data(),
                                             sarea.data(), nkept) > iou_thr) {
                    continue;
                }
                sx1[static_cast<size_t>(nkept)] = box.x1;
                sy1[static_cast<size_t>(nkept)] = box.y1;
                sx2[static_cast<size_t>(nkept)] = box.x2;
                sy2[static_cast<size_t>(nkept)] = box.y2;
                sarea[static_cast<size_t>(nkept)] = box.area;
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
