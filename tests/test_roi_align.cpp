/// @file test_roi_align.cpp
/// @brief RoiAlign on the CPU backend.
///
/// A hand case pins the bilinear semantics (the 2×2 image pooled to 1×1 is its
/// mean; the legacy coordinate mode lands on the border pixel), then a naive
/// scalar reference — the same documented formula, written independently of the
/// kernel — checks random cases across Avg/Max, both coordinate modes, an
/// explicit sampling ratio and the adaptive one, spatial scaling, several
/// batches and out-of-image ROIs.

#include "nnops/ops/roi_align.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace nnops;

namespace {

TensorDesc make_desc(std::initializer_list<int64_t> dims, DataType dt, TensorLayout layout)
{
    TensorDesc d;
    d.dtype = dt;
    d.layout = layout;
    d.rank = static_cast<int64_t>(dims.size());
    d.dims.resize(dims.size());
    size_t i = 0;
    for (int64_t v : dims) { d.dims[i++] = v; }
    return d;
}

/// Naive scalar twin of the operator, over dense NHWC data.
std::vector<float> roi_align_ref(const std::vector<float>& x, int64_t N, int64_t H, int64_t W,
                                 int64_t C, const std::vector<float>& rois,
                                 const std::vector<int64_t>& batch,
                                 const RoiAlignAttributes& a)
{
    const int64_t R = static_cast<int64_t>(rois.size()) / 4;
    const int64_t OH = a.output_height;
    const int64_t OW = a.output_width;
    const bool max_mode = a.mode == RoiAlignMode::Max;
    std::vector<float> out(static_cast<size_t>(R * OH * OW * C), 0.0f);

    const auto pixel = [&](int64_t n, int64_t h, int64_t w, int64_t c) -> float {
        if (h < 0 || h >= H || w < 0 || w >= W) { return 0.0f; }
        return x[static_cast<size_t>(((n * H + h) * W + w) * C + c)];
    };

    for (int64_t r = 0; r < R; ++r) {
        const int64_t n = batch[static_cast<size_t>(r)];
        if (n < 0 || n >= N) { continue; }
        float sw = rois[static_cast<size_t>(r * 4 + 0)] * a.spatial_scale;
        float sh = rois[static_cast<size_t>(r * 4 + 1)] * a.spatial_scale;
        float ew = rois[static_cast<size_t>(r * 4 + 2)] * a.spatial_scale;
        float eh = rois[static_cast<size_t>(r * 4 + 3)] * a.spatial_scale;
        if (a.coordinate_mode == RoiAlignCoordinateMode::HalfPixel) {
            sw -= 0.5f; sh -= 0.5f; ew -= 0.5f; eh -= 0.5f;
        }
        float rw = ew - sw;
        float rh = eh - sh;
        if (a.coordinate_mode == RoiAlignCoordinateMode::OutputHalfPixel) {
            // The ONNX reference's 1x1 floor for legacy coordinates.
            rw = std::max(rw, 1.0f);
            rh = std::max(rh, 1.0f);
        }
        const float bw = rw / static_cast<float>(OW);
        const float bh = rh / static_cast<float>(OH);
        const int64_t gh = a.sampling_ratio > 0
                               ? a.sampling_ratio
                               : std::max<int64_t>(1, static_cast<int64_t>(std::ceil(rh / OH)));
        const int64_t gw = a.sampling_ratio > 0
                               ? a.sampling_ratio
                               : std::max<int64_t>(1, static_cast<int64_t>(std::ceil(rw / OW)));
        const float count = static_cast<float>(gh * gw);

        for (int64_t ph = 0; ph < OH; ++ph) {
            for (int64_t pw = 0; pw < OW; ++pw) {
                std::vector<float> acc(static_cast<size_t>(C), max_mode ? -1e30f : 0.0f);
                for (int64_t iy = 0; iy < gh; ++iy) {
                    const float y = sh + static_cast<float>(ph) * bh +
                                    (static_cast<float>(iy) + 0.5f) * bh / static_cast<float>(gh);
                    for (int64_t ix = 0; ix < gw; ++ix) {
                        const float xx = sw + static_cast<float>(pw) * bw +
                                         (static_cast<float>(ix) + 0.5f) * bw /
                                             static_cast<float>(gw);
                        const bool outside = y < -1.0f || y > static_cast<float>(H) ||
                                             xx < -1.0f || xx > static_cast<float>(W);
                        for (int64_t c = 0; c < C; ++c) {
                            float v = 0.0f;
                            if (!outside) {
                                const float yc = std::max(y, 0.0f);
                                const float xc = std::max(xx, 0.0f);
                                int64_t yl = static_cast<int64_t>(std::floor(yc));
                                int64_t xl = static_cast<int64_t>(std::floor(xc));
                                int64_t yh = yl + 1;
                                int64_t xh = xl + 1;
                                float dy = yc - static_cast<float>(yl);
                                float dx = xc - static_cast<float>(xl);
                                if (yl >= H - 1) { yh = yl = H - 1; dy = 0.0f; }
                                if (xl >= W - 1) { xh = xl = W - 1; dx = 0.0f; }
                                v = (1 - dy) * (1 - dx) * pixel(n, yl, xl, c) +
                                    (1 - dy) * dx * pixel(n, yl, xh, c) +
                                    dy * (1 - dx) * pixel(n, yh, xl, c) +
                                    dy * dx * pixel(n, yh, xh, c);
                            }
                            acc[static_cast<size_t>(c)] =
                                max_mode ? std::max(acc[static_cast<size_t>(c)], v)
                                         : acc[static_cast<size_t>(c)] + v;
                        }
                    }
                }
                for (int64_t c = 0; c < C; ++c) {
                    const size_t o = static_cast<size_t>(((r * OH + ph) * OW + pw) * C + c);
                    out[o] = max_mode ? acc[static_cast<size_t>(c)]
                                      : acc[static_cast<size_t>(c)] / count;
                }
            }
        }
    }
    return out;
}

} // anonymous namespace

NNOPS_TEST(roi_align_hand_2x2_to_1x1) {
    // X = [[1, 2], [3, 4]] (1 batch, C = 1).
    const int64_t xs[] = {1, 2, 2, 1};
    float x_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    TensorView x(xs, DataType::f32, x_data, TensorLayout::NHWC);

    float rois_data[] = {0.0f, 0.0f, 2.0f, 2.0f};
    const int64_t rs[] = {1, 4};
    TensorView rois(rs, DataType::f32, rois_data);

    for (int mode = 0; mode < 2; ++mode) {
        RoiAlignAttributes attrs;
        attrs.output_height = 1;
        attrs.output_width = 1;
        attrs.sampling_ratio = 1;
        attrs.coordinate_mode = mode == 0 ? RoiAlignCoordinateMode::HalfPixel
                                          : RoiAlignCoordinateMode::OutputHalfPixel;
        auto op = RoiAlign::create(attrs, Backend::CPU);

        const TensorDesc descs[] = {x.desc(), rois.desc()};
        auto desc = op->getOutputTensorDesc(descs);
        NNOPS_EXPECT_EQ(desc[0].rank, 4);
        NNOPS_EXPECT_EQ(desc[0].dims[0], int64_t{1});
        NNOPS_EXPECT_EQ(desc[0].dims[1], int64_t{1});
        NNOPS_EXPECT_EQ(desc[0].dims[2], int64_t{1});
        NNOPS_EXPECT_EQ(desc[0].dims[3], int64_t{1});

        float out_data[1] = {-1.0f};
        TensorView out(std::span<const int64_t>(desc[0].dims.data(), 4), DataType::f32,
                       out_data, TensorLayout::NHWC);
        const TensorView ins[] = {x, rois};
        NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::ok);

        // HalfPixel samples (0.5, 0.5) — the image mean; the legacy mode
        // samples (1.0, 1.0), clamped to the bottom-right pixel.
        NNOPS_EXPECT_NEAR(out_data[0], mode == 0 ? 2.5f : 4.0f, 1e-6f);
    }
}

NNOPS_TEST(roi_align_legacy_clamps_degenerate_roi) {
    // A 0.2-wide ROI in legacy coordinates is forced to 1x1, so the single
    // sample sits at (0.5, 0.5) — the 2x2 image mean (the ONNX reference's rule).
    const int64_t xs[] = {1, 2, 2, 1};
    float x_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    TensorView x(xs, DataType::f32, x_data, TensorLayout::NHWC);
    float rois_data[] = {0.0f, 0.0f, 0.2f, 0.2f};
    const int64_t rs[] = {1, 4};
    TensorView rois(rs, DataType::f32, rois_data);

    RoiAlignAttributes attrs;
    attrs.output_height = 1;
    attrs.output_width = 1;
    attrs.sampling_ratio = 1;
    attrs.coordinate_mode = RoiAlignCoordinateMode::OutputHalfPixel;
    auto op = RoiAlign::create(attrs, Backend::CPU);

    const TensorDesc descs[] = {x.desc(), rois.desc()};
    auto desc = op->getOutputTensorDesc(descs);
    float out_data[1] = {};
    TensorView out(std::span<const int64_t>(desc[0].dims.data(), 4), DataType::f32, out_data,
                   TensorLayout::NHWC);
    const TensorView ins[] = {x, rois};
    NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::ok);
    NNOPS_EXPECT_NEAR(out_data[0], 2.5f, 1e-6f);
}

NNOPS_TEST(roi_align_hand_2x2_to_2x2) {
    const int64_t xs[] = {1, 2, 2, 1};
    float x_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    TensorView x(xs, DataType::f32, x_data, TensorLayout::NHWC);
    float rois_data[] = {0.0f, 0.0f, 2.0f, 2.0f};
    const int64_t rs[] = {1, 4};
    TensorView rois(rs, DataType::f32, rois_data);

    RoiAlignAttributes attrs;
    attrs.output_height = 2;
    attrs.output_width = 2;
    attrs.sampling_ratio = 1;
    auto op = RoiAlign::create(attrs, Backend::CPU);

    const TensorDesc descs[] = {x.desc(), rois.desc()};
    auto desc = op->getOutputTensorDesc(descs);
    float out_data[4] = {};
    TensorView out(std::span<const int64_t>(desc[0].dims.data(), 4), DataType::f32, out_data,
                   TensorLayout::NHWC);
    const TensorView ins[] = {x, rois};
    NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::ok);

    // HalfPixel starts the grid at (0, 0) and steps 1, so each 1×1 bin samples
    // exactly one pixel: 1, 2, 3, 4.
    NNOPS_EXPECT_NEAR(out_data[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[3], 4.0f, 1e-6f);
}

NNOPS_TEST(roi_align_random_matches_reference) {
    const int64_t N = 2, H = 5, W = 6, C = 3, R = 5;
    auto [x_vec, _x] = test::make_random_tensor({N, H, W, C}, -2.0f, 2.0f, 700);
    auto [roi_vec, _r] = test::make_random_tensor({R, 4}, 0.0f, 9.0f, 701);
    // Keep x1 < x2 and y1 < y2 so the ROIs are well-formed.
    std::vector<float> rois(roi_vec);
    for (int64_t r = 0; r < R; ++r) {
        float& x1 = rois[static_cast<size_t>(r * 4 + 0)];
        float& y1 = rois[static_cast<size_t>(r * 4 + 1)];
        float& x2 = rois[static_cast<size_t>(r * 4 + 2)];
        float& y2 = rois[static_cast<size_t>(r * 4 + 3)];
        if (x2 < x1) { std::swap(x1, x2); }
        if (y2 < y1) { std::swap(y1, y2); }
    }
    std::vector<int64_t> batch = {0, 1, 0, 1, 0};

    const int64_t xs[] = {N, H, W, C};
    TensorView x(xs, DataType::f32, x_vec.data(), TensorLayout::NHWC);
    const int64_t rs[] = {R, 4};
    TensorView rois_v(rs, DataType::f32, rois.data());
    const int64_t bs[] = {R};
    TensorView batch_v(bs, DataType::s64, batch.data());

    for (int mode = 0; mode < 2; ++mode) {
        for (int coord = 0; coord < 2; ++coord) {
            for (int64_t ratio : {int64_t{1}, int64_t{2}, int64_t{0}}) {
                for (float scale : {1.0f, 0.5f}) {
                    RoiAlignAttributes attrs;
                    attrs.mode = mode == 0 ? RoiAlignMode::Avg : RoiAlignMode::Max;
                    attrs.coordinate_mode =
                        coord == 0 ? RoiAlignCoordinateMode::HalfPixel
                                   : RoiAlignCoordinateMode::OutputHalfPixel;
                    attrs.output_height = 3;
                    attrs.output_width = 4;
                    attrs.sampling_ratio = ratio;
                    attrs.spatial_scale = scale;

                    auto op = RoiAlign::create(attrs, Backend::CPU);
                    const TensorDesc descs[] = {x.desc(), rois_v.desc(), batch_v.desc()};
                    auto desc = op->getOutputTensorDesc(descs);

                    std::vector<float> got(static_cast<size_t>(desc[0].numel()));
                    TensorView out(std::span<const int64_t>(desc[0].dims.data(), 4),
                                   DataType::f32, got.data(), TensorLayout::NHWC);
                    const TensorView ins[] = {x, rois_v, batch_v};
                    NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::ok);

                    const std::vector<float> want =
                        roi_align_ref(x_vec, N, H, W, C, rois, batch, attrs);
                    for (size_t i = 0; i < got.size(); ++i) {
                        NNOPS_EXPECT_NEAR(got[i], want[i], 2e-5f);
                    }
                }
            }
        }
    }
}

NNOPS_TEST(roi_align_bad_batch_index_is_zero) {
    const int64_t xs[] = {1, 2, 2, 1};
    float x_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    TensorView x(xs, DataType::f32, x_data, TensorLayout::NHWC);
    float rois_data[] = {0.0f, 0.0f, 2.0f, 2.0f};
    const int64_t rs[] = {1, 4};
    TensorView rois(rs, DataType::f32, rois_data);
    int64_t bad[] = {7};
    const int64_t bs[] = {1};
    TensorView batch(bs, DataType::s64, bad);

    RoiAlignAttributes attrs;
    attrs.output_height = 1;
    attrs.output_width = 1;
    auto op = RoiAlign::create(attrs, Backend::CPU);

    const TensorDesc descs[] = {x.desc(), rois.desc(), batch.desc()};
    auto desc = op->getOutputTensorDesc(descs);
    float out_data[1] = {9.0f};
    TensorView out(std::span<const int64_t>(desc[0].dims.data(), 4), DataType::f32, out_data,
                   TensorLayout::NHWC);
    const TensorView ins[] = {x, rois, batch};
    NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::ok);
    NNOPS_EXPECT_NEAR(out_data[0], 0.0f, 1e-6f);
}

NNOPS_TEST(roi_align_non_finite_roi_is_zero) {
    // A NaN coordinate cannot index the image: its taps are all outside, so the
    // ROI comes back as zeros rather than an out-of-range load.
    const int64_t xs[] = {1, 2, 2, 1};
    float x_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    TensorView x(xs, DataType::f32, x_data, TensorLayout::NHWC);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    float rois_data[] = {nan, 0.0f, 2.0f, 2.0f};
    const int64_t rs[] = {1, 4};
    TensorView rois(rs, DataType::f32, rois_data);

    RoiAlignAttributes attrs;
    attrs.output_height = 2;
    attrs.output_width = 2;
    attrs.sampling_ratio = 2;
    auto op = RoiAlign::create(attrs, Backend::CPU);

    const TensorDesc descs[] = {x.desc(), rois.desc()};
    auto desc = op->getOutputTensorDesc(descs);
    std::vector<float> out_data(static_cast<size_t>(desc[0].numel()), 9.0f);
    TensorView out(std::span<const int64_t>(desc[0].dims.data(), 4), DataType::f32,
                   out_data.data(), TensorLayout::NHWC);
    const TensorView ins[] = {x, rois};
    NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::ok);
    for (float v : out_data) {
        NNOPS_EXPECT_NEAR(v, 0.0f, 1e-6f);
    }
}

NNOPS_TEST(roi_align_f16_matches_f32) {
    const int64_t N = 1, H = 4, W = 5, C = 4, R = 3;
    auto [x_f32, _] = test::make_random_tensor({N, H, W, C}, -1.0f, 1.0f, 710);
    auto [roi_vec, __] = test::make_random_tensor({R, 4}, 0.0f, 4.0f, 711);
    std::vector<float> rois(roi_vec);
    for (int64_t r = 0; r < R; ++r) {
        float& x1 = rois[static_cast<size_t>(r * 4 + 0)];
        float& y1 = rois[static_cast<size_t>(r * 4 + 1)];
        float& x2 = rois[static_cast<size_t>(r * 4 + 2)];
        float& y2 = rois[static_cast<size_t>(r * 4 + 3)];
        if (x2 < x1) { std::swap(x1, x2); }
        if (y2 < y1) { std::swap(y1, y2); }
    }
    auto x_h = test::f32_to_f16(x_f32);

    const int64_t xs[] = {N, H, W, C};
    const int64_t rs[] = {R, 4};
    TensorView x16(xs, DataType::f16, x_h.data(), TensorLayout::NHWC);
    TensorView rois_v(rs, DataType::f32, rois.data());

    RoiAlignAttributes attrs;
    attrs.output_height = 2;
    attrs.output_width = 3;
    attrs.sampling_ratio = 2;
    attrs.mode = RoiAlignMode::Avg;
    auto op = RoiAlign::create(attrs, Backend::CPU);

    const TensorDesc descs[] = {x16.desc(), rois_v.desc()};
    auto desc = op->getOutputTensorDesc(descs);
    std::vector<nnops::backend::cpu::half> got(static_cast<size_t>(desc[0].numel()));
    TensorView out(std::span<const int64_t>(desc[0].dims.data(), 4), DataType::f16, got.data(),
                   TensorLayout::NHWC);
    const TensorView ins[] = {x16, rois_v};
    NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::ok);

    const std::vector<float> x_back = test::f16_to_f32(x_h);
    const std::vector<int64_t> batch_zero(static_cast<size_t>(R), 0);
    const std::vector<float> want =
        roi_align_ref(x_back, N, H, W, C, rois, batch_zero, attrs);
    const std::vector<float> got_f32 = test::f16_to_f32(got);
    for (size_t i = 0; i < got_f32.size(); ++i) {
        NNOPS_EXPECT_NEAR(got_f32[i], want[i], 3e-3f);
    }
}

NNOPS_TEST(roi_align_validation) {
    const int64_t xs[] = {1, 4, 4, 8};
    std::vector<float> xb(128, 0.0f);
    TensorView x(xs, DataType::f32, xb.data(), TensorLayout::NHWC);
    const int64_t rs[] = {2, 4};
    std::vector<float> rb(8, 0.0f);
    TensorView rois(rs, DataType::f32, rb.data());
    const TensorDesc descs[] = {x.desc(), rois.desc()};
    const TensorDesc out_desc[] = {make_desc({2, 2, 2, 8}, DataType::f32, TensorLayout::NHWC)};

    // Not NHWC.
    {
        TensorView xd(xs, DataType::f32, xb.data(), TensorLayout::Dense);
        RoiAlignAttributes a;
        a.output_height = 2;
        a.output_width = 2;
        auto op = RoiAlign::create(a, Backend::CPU);
        const TensorDesc bad[] = {xd.desc(), rois.desc()};
        NNOPS_EXPECT_TRUE(op->validate(std::span<const TensorDesc>(bad, 2),
                                       std::span<const TensorDesc>(out_desc, 1)) ==
                          Status::unsupported);
    }
    // Bad ROI width and a bad sampling ratio.
    {
        const int64_t bad_rs[] = {2, 5};
        TensorView bad_rois(bad_rs, DataType::f32, rb.data());
        RoiAlignAttributes a;
        a.output_height = 2;
        a.output_width = 2;
        auto op = RoiAlign::create(a, Backend::CPU);
        const TensorDesc bad[] = {x.desc(), bad_rois.desc()};
        NNOPS_EXPECT_TRUE(op->validate(std::span<const TensorDesc>(bad, 2),
                                       std::span<const TensorDesc>(out_desc, 1)) ==
                          Status::invalid_argument);
    }
    {
        RoiAlignAttributes a;
        a.output_height = 2;
        a.output_width = 2;
        a.sampling_ratio = -1;
        auto op = RoiAlign::create(a, Backend::CPU);
        NNOPS_EXPECT_TRUE(op->validate(std::span<const TensorDesc>(descs, 2),
                                       std::span<const TensorDesc>(out_desc, 1)) ==
                          Status::invalid_argument);
    }
    // Wrong output width.
    {
        RoiAlignAttributes a;
        a.output_height = 2;
        a.output_width = 2;
        auto op = RoiAlign::create(a, Backend::CPU);
        const TensorDesc bad_out[] = {make_desc({2, 3, 3, 8}, DataType::f32, TensorLayout::NHWC)};
        NNOPS_EXPECT_TRUE(op->validate(std::span<const TensorDesc>(descs, 2),
                                       std::span<const TensorDesc>(bad_out, 1)) ==
                          Status::invalid_argument);
    }
}
