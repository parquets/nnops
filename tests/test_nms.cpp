/// @file test_nms.cpp
/// @brief NonMaxSuppression on the CPU backend.
///
/// Hand cases pin the greedy order, the strict `IoU > threshold` rule, the
/// per-class independence, the TopK pre-selection and the two box encodings.
/// The optional ONNX inputs are covered by omission (defaults select nothing)
/// and by explicit scalars, and the capacity rule is checked by handing the
/// operator an output smaller than the selection.

#include "nnops/ops/non_max_suppression.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/test_helpers.hpp"

#include <vector>

using namespace nnops;

namespace {

struct Case {
    std::vector<int64_t> out;  // flattened [rows, 3]
    int64_t count = 0;
};

/// Run one call; returns the (batch, class, box) rows the op wrote.
Case run_nms(const NonMaxSuppressionAttributes& attrs, const TensorView& boxes,
             const TensorView& scores, const TensorView* max_out, const TensorView* iou,
             const TensorView* score_thr, int64_t capacity)
{
    auto op = NonMaxSuppression::create(attrs, Backend::CPU);

    std::vector<TensorView> ins = {boxes, scores};
    std::vector<TensorDesc> descs = {boxes.desc(), scores.desc()};
    if (max_out != nullptr) { ins.push_back(*max_out); descs.push_back(max_out->desc()); }
    if (iou != nullptr) { ins.push_back(*iou); descs.push_back(iou->desc()); }
    if (score_thr != nullptr) { ins.push_back(*score_thr); descs.push_back(score_thr->desc()); }

    // Exercise the descriptor path too; the capacity here is the caller's.
    (void)op->getOutputTensorDesc(descs);

    const int64_t idx_shape[] = {capacity, 3};
    std::vector<int64_t> idx(static_cast<size_t>(capacity * 3), -1);
    TensorView idx_view(idx_shape, DataType::s64, idx.data());

    const int64_t cnt_shape[] = {1};
    int64_t cnt = -1;
    TensorView cnt_view(cnt_shape, DataType::s64, &cnt);

    TensorView outs[] = {idx_view, cnt_view};
    if (op->compute(outs, ins) != Status::ok) {
        return {};
    }
    Case c;
    c.count = cnt;
    c.out = idx;
    return c;
}

TensorView scalar_f32(float& storage) {
    const int64_t s[] = {1};
    return TensorView(s, DataType::f32, &storage);
}
TensorView scalar_s64(int64_t& storage) {
    const int64_t s[] = {1};
    return TensorView(s, DataType::s64, &storage);
}

} // anonymous namespace

NNOPS_TEST(nms_hand_greedy) {
    // Corner form [y1, x1, y2, x2].
    const int64_t bs[] = {1, 4, 4};
    float boxes[] = {
        0.0f, 0.0f, 1.0f, 1.0f,  // 0
        0.0f, 0.0f, 1.0f, 2.0f,  // 1: IoU 0.5 with 0
        5.0f, 5.0f, 6.0f, 6.0f,  // 2: disjoint
        0.0f, 0.0f, 2.0f, 2.0f,  // 3: IoU 0.25 with 0
    };
    const int64_t ss[] = {1, 1, 4};
    float scores[] = {0.9f, 0.8f, 0.7f, 0.6f};

    TensorView boxes_v(bs, DataType::f32, boxes);
    TensorView scores_v(ss, DataType::f32, scores);

    NonMaxSuppressionAttributes attrs;
    int64_t max_out = 10;
    float iou = 0.4f;
    auto mo = scalar_s64(max_out);
    auto io = scalar_f32(iou);

    Case c = run_nms(attrs, boxes_v, scores_v, &mo, &io, nullptr, 10);
    NNOPS_EXPECT_EQ(c.count, int64_t{3});
    NNOPS_EXPECT_EQ(c.out[0], int64_t{0});
    NNOPS_EXPECT_EQ(c.out[1], int64_t{0});
    NNOPS_EXPECT_EQ(c.out[2], int64_t{0});  // box 0 kept first
    NNOPS_EXPECT_EQ(c.out[5], int64_t{2});  // disjoint box 2
    NNOPS_EXPECT_EQ(c.out[8], int64_t{3});  // box 3 survives IoU 0.25

    // 0.5 is not strictly greater than the threshold, so box 1 survives.
    iou = 0.5f;
    Case c2 = run_nms(attrs, boxes_v, scores_v, &mo, &io, nullptr, 10);
    NNOPS_EXPECT_EQ(c2.count, int64_t{4});
    NNOPS_EXPECT_EQ(c2.out[5], int64_t{1});
    NNOPS_EXPECT_EQ(c2.out[8], int64_t{2});
    NNOPS_EXPECT_EQ(c2.out[11], int64_t{3});
}

NNOPS_TEST(nms_capacity_and_max_output) {
    const int64_t bs[] = {1, 4, 4};
    float boxes[] = {0.0f, 0.0f, 1.0f, 1.0f, 5.0f, 5.0f, 6.0f, 6.0f, 9.0f, 9.0f, 10.0f, 10.0f,
                     20.0f, 20.0f, 21.0f, 21.0f};
    const int64_t ss[] = {1, 1, 4};
    float scores[] = {0.9f, 0.8f, 0.7f, 0.6f};
    TensorView boxes_v(bs, DataType::f32, boxes);
    TensorView scores_v(ss, DataType::f32, scores);

    NonMaxSuppressionAttributes attrs;
    int64_t max_out = 2;
    float iou = 0.5f;
    auto mo = scalar_s64(max_out);
    auto io = scalar_f32(iou);

    // max_output_boxes_per_class caps the selection at 2.
    Case c = run_nms(attrs, boxes_v, scores_v, &mo, &io, nullptr, 10);
    NNOPS_EXPECT_EQ(c.count, int64_t{2});
    NNOPS_EXPECT_EQ(c.out[2], int64_t{0});
    NNOPS_EXPECT_EQ(c.out[5], int64_t{1});

    // A capacity smaller than the selection truncates the write; the count
    // reports what actually landed.
    max_out = 10;
    Case c2 = run_nms(attrs, boxes_v, scores_v, &mo, &io, nullptr, 1);
    NNOPS_EXPECT_EQ(c2.count, int64_t{1});
    NNOPS_EXPECT_EQ(c2.out[2], int64_t{0});
}

NNOPS_TEST(nms_score_threshold_and_top_k) {
    const int64_t bs[] = {1, 4, 4};
    float boxes[] = {0.0f, 0.0f, 1.0f, 1.0f, 5.0f, 5.0f, 6.0f, 6.0f, 9.0f, 9.0f, 10.0f, 10.0f,
                     20.0f, 20.0f, 21.0f, 21.0f};
    const int64_t ss[] = {1, 1, 4};
    float scores[] = {0.9f, 0.8f, 0.7f, 0.6f};
    TensorView boxes_v(bs, DataType::f32, boxes);
    TensorView scores_v(ss, DataType::f32, scores);

    NonMaxSuppressionAttributes attrs;
    int64_t max_out = 10;
    float iou = 0.5f;
    float score_thr = 0.85f;
    auto mo = scalar_s64(max_out);
    auto io = scalar_f32(iou);
    auto so = scalar_f32(score_thr);

    // Only box 0 clears the score threshold.
    Case c = run_nms(attrs, boxes_v, scores_v, &mo, &io, &so, 10);
    NNOPS_EXPECT_EQ(c.count, int64_t{1});
    NNOPS_EXPECT_EQ(c.out[2], int64_t{0});

    // top_k keeps the two best candidates before suppression.
    attrs.top_k = 2;
    score_thr = 0.0f;
    Case c2 = run_nms(attrs, boxes_v, scores_v, &mo, &io, &so, 10);
    NNOPS_EXPECT_EQ(c2.count, int64_t{2});
    NNOPS_EXPECT_EQ(c2.out[2], int64_t{0});
    NNOPS_EXPECT_EQ(c2.out[5], int64_t{1});
}

NNOPS_TEST(nms_multiple_classes_are_independent) {
    // One batch, two classes; class 1 sees the same boxes with lower scores.
    const int64_t bs[] = {1, 2, 4};
    float boxes[] = {0.0f, 0.0f, 1.0f, 1.0f, 5.0f, 5.0f, 6.0f, 6.0f};
    const int64_t ss[] = {1, 2, 2};
    float scores[] = {
        0.9f, 0.8f,  // class 0
        0.4f, 0.95f, // class 1: box 1 wins
    };
    TensorView boxes_v(bs, DataType::f32, boxes);
    TensorView scores_v(ss, DataType::f32, scores);

    NonMaxSuppressionAttributes attrs;
    int64_t max_out = 1;
    float iou = 0.5f;
    auto mo = scalar_s64(max_out);
    auto io = scalar_f32(iou);

    Case c = run_nms(attrs, boxes_v, scores_v, &mo, &io, nullptr, 10);
    NNOPS_EXPECT_EQ(c.count, int64_t{2});
    NNOPS_EXPECT_EQ(c.out[0], int64_t{0});  // batch
    NNOPS_EXPECT_EQ(c.out[1], int64_t{0});  // class 0
    NNOPS_EXPECT_EQ(c.out[2], int64_t{0});  // box 0
    NNOPS_EXPECT_EQ(c.out[3], int64_t{0});
    NNOPS_EXPECT_EQ(c.out[4], int64_t{1});  // class 1
    NNOPS_EXPECT_EQ(c.out[5], int64_t{1});  // box 1
}

NNOPS_TEST(nms_center_point_box) {
    // [x_center, y_center, width, height]; the same geometry as the corner case.
    const int64_t bs[] = {1, 3, 4};
    float boxes[] = {
        0.5f, 0.5f, 1.0f, 1.0f,  // corners [0,0,1,1]
        1.0f, 0.5f, 2.0f, 1.0f,  // corners [0,0,2,1] — IoU 0.5
        5.5f, 5.5f, 1.0f, 1.0f,  // disjoint
    };
    const int64_t ss[] = {1, 1, 3};
    float scores[] = {0.9f, 0.8f, 0.7f};
    TensorView boxes_v(bs, DataType::f32, boxes);
    TensorView scores_v(ss, DataType::f32, scores);

    NonMaxSuppressionAttributes attrs;
    attrs.center_point_box = 1;
    int64_t max_out = 10;
    float iou = 0.4f;
    auto mo = scalar_s64(max_out);
    auto io = scalar_f32(iou);

    Case c = run_nms(attrs, boxes_v, scores_v, &mo, &io, nullptr, 10);
    NNOPS_EXPECT_EQ(c.count, int64_t{2});
    NNOPS_EXPECT_EQ(c.out[2], int64_t{0});
    NNOPS_EXPECT_EQ(c.out[5], int64_t{2});
}

NNOPS_TEST(nms_omitted_optionals_select_nothing) {
    // ONNX defaults every optional scalar to 0, and a 0 max output means no
    // rows. Both an absent slot and an explicit 0 say the same thing.
    const int64_t bs[] = {1, 2, 4};
    float boxes[] = {0.0f, 0.0f, 1.0f, 1.0f, 5.0f, 5.0f, 6.0f, 6.0f};
    const int64_t ss[] = {1, 1, 2};
    float scores[] = {0.9f, 0.8f};
    TensorView boxes_v(bs, DataType::f32, boxes);
    TensorView scores_v(ss, DataType::f32, scores);

    NonMaxSuppressionAttributes attrs;
    Case c = run_nms(attrs, boxes_v, scores_v, nullptr, nullptr, nullptr, 4);
    NNOPS_EXPECT_EQ(c.count, int64_t{0});
}

NNOPS_TEST(nms_f16) {
    const int64_t bs[] = {1, 3, 4};
    float boxes_f32[] = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 2.0f, 5.0f, 5.0f, 6.0f, 6.0f};
    float scores_f32[] = {0.9f, 0.8f, 0.7f};
    auto boxes_h = test::f32_to_f16(std::vector<float>(boxes_f32, boxes_f32 + 12));
    auto scores_h = test::f32_to_f16(std::vector<float>(scores_f32, scores_f32 + 3));
    const int64_t ss[] = {1, 1, 3};
    TensorView boxes_v(bs, DataType::f16, boxes_h.data());
    TensorView scores_v(ss, DataType::f16, scores_h.data());

    NonMaxSuppressionAttributes attrs;
    int64_t max_out = 10;
    float iou = 0.4f;
    auto mo = scalar_s64(max_out);
    auto io = scalar_f32(iou);

    Case c = run_nms(attrs, boxes_v, scores_v, &mo, &io, nullptr, 10);
    NNOPS_EXPECT_EQ(c.count, int64_t{2});
    NNOPS_EXPECT_EQ(c.out[2], int64_t{0});
    NNOPS_EXPECT_EQ(c.out[5], int64_t{2});
}

NNOPS_TEST(nms_validation) {
    const int64_t bs[] = {1, 2, 4};
    float boxes[8] = {};
    const int64_t ss[] = {1, 1, 2};
    float scores[2] = {};
    TensorView boxes_v(bs, DataType::f32, boxes);
    TensorView scores_v(ss, DataType::f32, scores);

    TensorDesc out0;
    out0.rank = 2;
    out0.dtype = DataType::s64;
    out0.dims.resize(2);
    out0.dims[0] = 4;
    out0.dims[1] = 3;
    TensorDesc out1;
    out1.rank = 1;
    out1.dtype = DataType::s64;
    out1.dims.resize(1);
    out1.dims[0] = 1;
    const TensorDesc outs[] = {out0, out1};
    const auto out_span = std::span<const TensorDesc>(outs, 2);

    {
        NonMaxSuppressionAttributes a;
        a.center_point_box = 2;
        auto op = NonMaxSuppression::create(a, Backend::CPU);
        const TensorDesc ins[] = {boxes_v.desc(), scores_v.desc()};
        NNOPS_EXPECT_TRUE(op->validate(std::span<const TensorDesc>(ins, 2), out_span) ==
                          Status::invalid_argument);
    }
    {
        // Score count does not match the box count.
        const int64_t bad_ss[] = {1, 1, 3};
        TensorView bad_scores(bad_ss, DataType::f32, scores);
        NonMaxSuppressionAttributes a;
        auto op = NonMaxSuppression::create(a, Backend::CPU);
        const TensorDesc ins[] = {boxes_v.desc(), bad_scores.desc()};
        NNOPS_EXPECT_TRUE(op->validate(std::span<const TensorDesc>(ins, 2), out_span) ==
                          Status::invalid_argument);
    }
    {
        // Boxes must carry four coordinates.
        const int64_t bad_bs[] = {1, 2, 5};
        float bad_boxes[10] = {};
        TensorView bad_boxes_v(bad_bs, DataType::f32, bad_boxes);
        NonMaxSuppressionAttributes a;
        auto op = NonMaxSuppression::create(a, Backend::CPU);
        const TensorDesc ins[] = {bad_boxes_v.desc(), scores_v.desc()};
        NNOPS_EXPECT_TRUE(op->validate(std::span<const TensorDesc>(ins, 2), out_span) ==
                          Status::invalid_argument);
    }
    {
        // The count output must be a one-element s64 vector.
        NonMaxSuppressionAttributes a;
        auto op = NonMaxSuppression::create(a, Backend::CPU);
        TensorDesc bad1 = out1;
        bad1.dtype = DataType::f32;
        const TensorDesc bad_outs[] = {out0, bad1};
        const TensorDesc ins[] = {boxes_v.desc(), scores_v.desc()};
        NNOPS_EXPECT_TRUE(op->validate(std::span<const TensorDesc>(ins, 2),
                                       std::span<const TensorDesc>(bad_outs, 2)) ==
                          Status::invalid_argument);
    }
}
