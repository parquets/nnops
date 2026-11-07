/// @file test_resize.cpp
/// @brief Tests for the Resize (interpolation) operator — channel-first and
/// channel-last.
///
/// All tests use the class API path:
/// Channel-first in -> LayoutConvert -> channel-last -> Resize::compute ->
/// LayoutConvert -> channel-first -> compare. The parity check between the two
/// paths also runs the channel-first Resize directly.

#include "nnops/ops/resize.hpp"
#include "nnops/ops/layout_convert.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"

#include <algorithm>
#include <vector>
#include <cmath>

using namespace nnops;

// Channel-first scalar reference — the ground truth.
namespace nnops::backend::cpu::reference {
    void resize_ref(const ResizeAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

namespace {

/// 2D+3D roundtrip: convert -> resize channel-last -> convert back -> compare
/// with the channel-first reference.
void test_channel_last_vs_channel_first(const std::vector<int64_t>& in_shape,
                          const ResizeAttributes& attrs)
{
    const int64_t rank = static_cast<int64_t>(in_shape.size());
    const int64_t srank = rank - 2;  // 2 or 3

    const TensorLayout channel_first_layout = (srank == 3) ? TensorLayout::NCDHW : TensorLayout::NCHW;
    const TensorLayout cl_layout = (srank == 3) ? TensorLayout::NDHWC : TensorLayout::NHWC;

    auto [in_vec, in_default] = test::make_random_tensor(in_shape);
    TensorView in_channel_first(in_default.shape_span(), DataType::f32,
                         in_default.ptr<float>(), channel_first_layout);

    std::vector<int64_t> oshape_ref = in_shape;
    if (srank == 3) {
        oshape_ref[2] = attrs.output_size[0];
        oshape_ref[3] = attrs.output_size[1];
        oshape_ref[4] = attrs.output_size[2];
    } else {
        oshape_ref[2] = attrs.output_size[1];
        oshape_ref[3] = attrs.output_size[2];
    }

    size_t ref_elems = 1;
    for (auto d : oshape_ref) {
        ref_elems *= static_cast<size_t>(d);
    }
    std::vector<float> ref_out(ref_elems);
    TensorView out_ref(oshape_ref, DataType::f32, ref_out.data(), channel_first_layout);

    const float prefill_val = 0.5f;
    if (attrs.add_to) {
        for (auto& v : ref_out) {
            v = prefill_val;
        }
    }

    {
        ComputeContext ctx;
        const TensorView ref_ins[] = {in_channel_first};
        backend::cpu::reference::resize_ref(attrs, out_ref, ref_ins, ctx, nullptr);
    }

    {
        auto to_cl_op = LayoutConvert::create(cl_layout, Backend::CPU);
        auto d = in_channel_first.desc();
        const TensorDesc cl_in_desc[] = {d};
        auto cl_descs = to_cl_op->getOutputTensorDesc(cl_in_desc);

        NNOPS_EXPECT_EQ(cl_descs[0].rank, rank);
        NNOPS_EXPECT_EQ(cl_descs[0].layout, cl_layout);
        NNOPS_EXPECT_EQ(cl_descs[0].dtype, DataType::f32);
        NNOPS_EXPECT_EQ(cl_descs[0].dims[0], in_shape[0]);
        NNOPS_EXPECT_EQ(cl_descs[0].dims[1], in_shape[1]);

        std::vector<char> cl_buf(cl_descs[0].storage_bytes());
        auto in_cl = test::make_packed(cl_descs[0], cl_buf.data());
        const TensorView cl_in_descs[] = {in_channel_first};
        to_cl_op->compute(in_cl, cl_in_descs);

        auto resize_op = Resize::create(attrs, Backend::CPU);
        auto pd = in_cl.desc();
        const TensorDesc resize_in[] = {pd};
        auto resize_descs = resize_op->getOutputTensorDesc(resize_in);

        NNOPS_EXPECT_EQ(resize_descs[0].rank, rank);
        NNOPS_EXPECT_EQ(resize_descs[0].layout, cl_layout);
        NNOPS_EXPECT_EQ(resize_descs[0].dtype, DataType::f32);
        NNOPS_EXPECT_EQ(resize_descs[0].dims[0], in_shape[0]);  // N unchanged
        NNOPS_EXPECT_EQ(resize_descs[0].dims[1], in_shape[1]);  // C unchanged
        if (srank == 3) {
            NNOPS_EXPECT_EQ(resize_descs[0].dims[2], attrs.output_size[0]);
            NNOPS_EXPECT_EQ(resize_descs[0].dims[3], attrs.output_size[1]);
            NNOPS_EXPECT_EQ(resize_descs[0].dims[4], attrs.output_size[2]);
        } else {
            NNOPS_EXPECT_EQ(resize_descs[0].dims[2], attrs.output_size[1]);
            NNOPS_EXPECT_EQ(resize_descs[0].dims[3], attrs.output_size[2]);
        }

        std::vector<char> resize_buf(resize_descs[0].storage_bytes());
        if (attrs.add_to) {
            float* fbuf = reinterpret_cast<float*>(resize_buf.data());
            size_t nfloats = resize_descs[0].storage_bytes() / sizeof(float);
            std::fill(fbuf, fbuf + nfloats, prefill_val);
        }
        auto out_cl = test::make_packed(resize_descs[0], resize_buf.data());
        const TensorView resize_ins[] = {in_cl};
        resize_op->compute(out_cl, resize_ins);

        auto unto_cl_op = LayoutConvert::create(channel_first_layout, Backend::CPU);
        auto od = out_cl.desc();
        const TensorDesc uncl_in_desc[] = {od};
        auto uncl_descs = unto_cl_op->getOutputTensorDesc(uncl_in_desc);

        NNOPS_EXPECT_EQ(uncl_descs[0].rank, rank);
        NNOPS_EXPECT_EQ(uncl_descs[0].layout, channel_first_layout);
        NNOPS_EXPECT_EQ(uncl_descs[0].dtype, DataType::f32);
        NNOPS_EXPECT_EQ(uncl_descs[0].dims[0], in_shape[0]);
        NNOPS_EXPECT_EQ(uncl_descs[0].dims[1], in_shape[1]);
        if (srank == 3) {
            NNOPS_EXPECT_EQ(uncl_descs[0].dims[2], attrs.output_size[0]);
            NNOPS_EXPECT_EQ(uncl_descs[0].dims[3], attrs.output_size[1]);
            NNOPS_EXPECT_EQ(uncl_descs[0].dims[4], attrs.output_size[2]);
        } else {
            NNOPS_EXPECT_EQ(uncl_descs[0].dims[2], attrs.output_size[1]);
            NNOPS_EXPECT_EQ(uncl_descs[0].dims[3], attrs.output_size[2]);
        }

        std::vector<char> channel_first_buf(uncl_descs[0].storage_bytes());
        auto res_channel_first = test::make_channel_first(uncl_descs[0], channel_first_buf.data());
        const TensorView uncl_in_descs[] = {out_cl};
        unto_cl_op->compute(res_channel_first, uncl_in_descs);

        const float tol = (attrs.mode == ResizeMode::Nearest) ? 1e-4f : 1e-3f;
        NNOPS_EXPECT_TRUE(test::allclose(res_channel_first, out_ref, tol, tol));
    }
}

/// convert -> resize channel-last -> convert back, then hand the channel-first result
/// to the caller's validator.
template <typename ValidateFn>
void run_resize_roundtrip(const std::vector<int64_t>& in_shape,
                           const std::vector<int64_t>& out_shape,
                           const std::vector<float>& in_data,
                           const ResizeAttributes& attrs,
                           ValidateFn validate_fn)
{
    const int64_t rank = static_cast<int64_t>(in_shape.size());
    const int64_t srank = rank - 2;
    const TensorLayout channel_first_layout = (srank == 3) ? TensorLayout::NCDHW : TensorLayout::NCHW;
    const TensorLayout cl_layout = (srank == 3) ? TensorLayout::NDHWC : TensorLayout::NHWC;

    // TensorView takes a mutable buffer; copy the const input.
    std::vector<float> in_buf = in_data;
    TensorView in_channel_first(std::span<const int64_t>(in_shape), DataType::f32, in_buf.data(), channel_first_layout);

    auto to_cl_op = LayoutConvert::create(cl_layout, Backend::CPU);
    auto d = in_channel_first.desc();
    const TensorDesc cl_in_desc[] = {d};
    auto cl_descs = to_cl_op->getOutputTensorDesc(cl_in_desc);

    NNOPS_EXPECT_EQ(cl_descs[0].rank, rank);
    NNOPS_EXPECT_EQ(cl_descs[0].layout, cl_layout);
    NNOPS_EXPECT_EQ(cl_descs[0].dtype, DataType::f32);

    std::vector<char> cl_buf(cl_descs[0].storage_bytes());
    auto in_cl = test::make_packed(cl_descs[0], cl_buf.data());
    const TensorView cl_in_descs[] = {in_channel_first};
    to_cl_op->compute(in_cl, cl_in_descs);

    auto resize_op = Resize::create(attrs, Backend::CPU);
    auto pd = in_cl.desc();
    const TensorDesc resize_in[] = {pd};
    auto resize_descs = resize_op->getOutputTensorDesc(resize_in);

    NNOPS_EXPECT_EQ(resize_descs[0].rank, rank);
    NNOPS_EXPECT_EQ(resize_descs[0].layout, cl_layout);
    NNOPS_EXPECT_EQ(resize_descs[0].dtype, DataType::f32);

    std::vector<char> resize_buf(resize_descs[0].storage_bytes());
    auto out_cl = test::make_packed(resize_descs[0], resize_buf.data());
    const TensorView resize_ins[] = {in_cl};
    resize_op->compute(out_cl, resize_ins);

    auto unto_cl_op = LayoutConvert::create(channel_first_layout, Backend::CPU);
    auto od = out_cl.desc();
    const TensorDesc uncl_in_desc[] = {od};
    auto uncl_descs = unto_cl_op->getOutputTensorDesc(uncl_in_desc);

    NNOPS_EXPECT_EQ(uncl_descs[0].rank, rank);
    NNOPS_EXPECT_EQ(uncl_descs[0].layout, channel_first_layout);
    NNOPS_EXPECT_EQ(uncl_descs[0].dtype, DataType::f32);

    std::vector<char> channel_first_buf(uncl_descs[0].storage_bytes());
    auto res_channel_first = test::make_channel_first(uncl_descs[0], channel_first_buf.data());
    const TensorView uncl_in_descs[] = {out_cl};
    unto_cl_op->compute(res_channel_first, uncl_in_descs);

    validate_fn(res_channel_first);
}

}  // anonymous namespace

// ============================================================
// 2D Nearest-neighbor hand-verified tests
// ============================================================

NNOPS_TEST(resize_2d_nearest_upsample) {
    const std::vector<int64_t> ishape = {1, 1, 2, 2};
    const std::vector<int64_t> oshape = {1, 1, 4, 4};
    const std::vector<float> in_data = {1, 2, 3, 4};

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 4, 4};

    run_resize_roundtrip(ishape, oshape, in_data, attrs, [](const TensorView& result) {
        const float* r = result.ptr<float>();
        // scale=0.5, each input pixel maps to a 2x2 block
        NNOPS_EXPECT_NEAR(r[0],  1.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[3],  2.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[4],  3.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[7],  4.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[12], 3.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[15], 4.0f, 1e-6f);
    });
}

NNOPS_TEST(resize_2d_nearest_downsample) {
    const std::vector<int64_t> ishape = {1, 1, 4, 4};
    const std::vector<int64_t> oshape = {1, 1, 2, 2};
    const std::vector<float> in_data = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
       13, 14, 15, 16
    };

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 2, 2};

    run_resize_roundtrip(ishape, oshape, in_data, attrs, [](const TensorView& result) {
        const float* r = result.ptr<float>();
        // scale=2.0, dst 0->src 0, dst 1->src 2
        NNOPS_EXPECT_NEAR(r[0], 1.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[1], 3.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[2], 9.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[3], 11.0f, 1e-6f);
    });
}

NNOPS_TEST(resize_2d_nearest_identity) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 8, 8};
    test_channel_last_vs_channel_first({1, 3, 8, 8}, attrs);
}

// ============================================================
// 2D Linear (bilinear) tests
// ============================================================

NNOPS_TEST(resize_2d_linear_align_corners_corners_exact) {
    const std::vector<int64_t> ishape = {1, 1, 2, 2};
    const std::vector<int64_t> oshape = {1, 1, 4, 4};
    const std::vector<float> in_data = {1, 2, 3, 4};

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::AlignCorners;
    attrs.output_size = {0, 4, 4};

    run_resize_roundtrip(ishape, oshape, in_data, attrs, [](const TensorView& result) {
        const float* r = result.ptr<float>();
        NNOPS_EXPECT_NEAR(r[0],  1.0f, 1e-5f);
        NNOPS_EXPECT_NEAR(r[3],  2.0f, 1e-5f);
        NNOPS_EXPECT_NEAR(r[12], 3.0f, 1e-5f);
        NNOPS_EXPECT_NEAR(r[15], 4.0f, 1e-5f);
        for (int64_t i = 0; i < result.numel(); ++i) {
            NNOPS_EXPECT_TRUE(!std::isnan(r[i]));
            NNOPS_EXPECT_TRUE(!std::isinf(r[i]));
        }
    });
}

NNOPS_TEST(resize_2d_linear_identity) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 4, 4};
    test_channel_last_vs_channel_first({1, 1, 4, 4}, attrs);
}

NNOPS_TEST(resize_2d_linear_upsample_no_nan) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 20, 20};
    test_channel_last_vs_channel_first({1, 2, 8, 8}, attrs);
}

// ============================================================
// 2D Cubic (bicubic) tests
// ============================================================

NNOPS_TEST(resize_2d_cubic_identity) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Cubic;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 8, 8};
    test_channel_last_vs_channel_first({1, 3, 8, 8}, attrs);
}

NNOPS_TEST(resize_2d_cubic_upsample) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Cubic;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 20, 20};
    test_channel_last_vs_channel_first({1, 2, 8, 8}, attrs);
}

NNOPS_TEST(resize_2d_cubic_downsample) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Cubic;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 5, 7};
    test_channel_last_vs_channel_first({1, 3, 11, 13}, attrs);
}

NNOPS_TEST(resize_2d_cubic_align_corners) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Cubic;
    attrs.coord_mode = CoordinateTransformMode::AlignCorners;
    attrs.output_size = {0, 12, 12};
    test_channel_last_vs_channel_first({1, 2, 6, 6}, attrs);
}

NNOPS_TEST(resize_2d_cubic_partial_channels) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Cubic;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 16, 16};
    test_channel_last_vs_channel_first({1, 5, 8, 8}, attrs);
}

NNOPS_TEST(resize_2d_cubic_add_to) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Cubic;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 8, 8};
    attrs.add_to = true;
    test_channel_last_vs_channel_first({1, 3, 4, 4}, attrs);
}

NNOPS_TEST(resize_2d_cubic_coeff_half) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Cubic;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 20, 20};
    attrs.cubic_coeff_a = -0.5f;  // PIL's BICUBIC
    test_channel_last_vs_channel_first({1, 3, 10, 10}, attrs);
}

NNOPS_TEST(resize_2d_cubic_constant_preserved) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Cubic;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 6, 6};
    run_resize_roundtrip({1, 1, 3, 3}, {1, 1, 6, 6},
                         std::vector<float>(9, 7.0f), attrs,
                         [](const TensorView& result) {
        const float* r = result.ptr<float>();
        for (int64_t i = 0; i < result.numel(); ++i) {
            NNOPS_EXPECT_NEAR(r[i], 7.0f, 1e-5f);
        }
    });
}

NNOPS_TEST(resize_2d_cubic_ramp_hand_verified) {
    // 4x4, every row the horizontal ramp {0,1,2,3}, upsampled half-pixel to
    // 8x8: src = o/2 - 0.25. The a = -0.75 kernel is not first-order
    // reproducing (only a = -0.5 is), so the values below come from the Keys
    // weights; the edges also fold their outside taps into the border pixel.
    const std::vector<float> in_data = {
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
    };

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Cubic;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 8, 8};

    run_resize_roundtrip({1, 1, 4, 4}, {1, 1, 8, 8}, in_data, attrs,
                         [](const TensorView& result) {
        const float* r = result.ptr<float>();
        // Interior, src 1.25: w = {-0.10546875, 0.87890625, 0.26171875,
        // -0.03515625} over p = {0,1,2,3}; src 1.75 mirrors it.
        NNOPS_EXPECT_NEAR(r[3], 1.296875f, 1e-5f);
        NNOPS_EXPECT_NEAR(r[4], 1.703125f, 1e-5f);
        // Left edge (src -0.25): (w0+w1+w2)*p0 + w3*p1.
        NNOPS_EXPECT_NEAR(r[0], -0.10546875f, 1e-5f);
        // Right edge (src 3.25): w0*p2 + (w1+w2+w3)*p3.
        NNOPS_EXPECT_NEAR(r[7], 3.10546875f, 1e-5f);
    });
}

NNOPS_TEST(resize_2d_cubic_ramp_interior_2d) {
    // v[r][c] = 3r + 2c + 5, and a = -0.5 (Catmull-Rom) reproduces a linear
    // ramp, so output coords whose four taps all land inside are exact.
    std::vector<float> in_data(36);
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            in_data[static_cast<size_t>(r * 6 + c)] = 3.0f * r + 2.0f * c + 5.0f;
        }
    }

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Cubic;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 12, 12};
    attrs.cubic_coeff_a = -0.5f;

    run_resize_roundtrip({1, 1, 6, 6}, {1, 1, 12, 12}, in_data, attrs,
                         [](const TensorView& result) {
        const float* r = result.ptr<float>();
        const auto value = [](int oh, int ow) {
            const float sh = 0.5f * oh - 0.25f;
            const float sw = 0.5f * ow - 0.25f;
            return 3.0f * sh + 2.0f * sw + 5.0f;
        };
        for (int oh = 3; oh <= 8; ++oh) {
            for (int ow = 3; ow <= 8; ++ow) {
                NNOPS_EXPECT_NEAR(r[oh * 12 + ow], value(oh, ow), 1e-4f);
            }
        }
    });
}

NNOPS_TEST(resize_3d_cubic_unsupported) {
    std::vector<float> in(8, 1.0f);
    const int64_t ishape[] = {1, 1, 2, 2, 2};
    TensorView x(ishape, DataType::f32, in.data(), TensorLayout::NCDHW);

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Cubic;
    attrs.output_size = {4, 4, 4};
    auto op = Resize::create(attrs, Backend::CPU);

    auto d = x.desc();
    const TensorDesc da[] = {d};
    auto descs = op->getOutputTensorDesc(da);
    std::vector<float> out(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_channel_first(descs[0], out.data());
    const TensorView ins[] = {x};
    NNOPS_EXPECT_EQ(op->compute(y, ins), Status::unsupported);
}

// ============================================================
// 3D Nearest-neighbor tests
// ============================================================

NNOPS_TEST(resize_3d_nearest_basic) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {4, 4, 4};
    test_channel_last_vs_channel_first({1, 1, 2, 2, 2}, attrs);
}

NNOPS_TEST(resize_3d_nearest_identity) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {4, 4, 4};
    test_channel_last_vs_channel_first({1, 2, 4, 4, 4}, attrs);
}

// ============================================================
// 3D Linear (trilinear) tests
// ============================================================

NNOPS_TEST(resize_3d_linear_basic) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {8, 12, 12};
    test_channel_last_vs_channel_first({1, 1, 4, 6, 6}, attrs);
}

NNOPS_TEST(resize_3d_linear_identity) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {4, 4, 4};
    test_channel_last_vs_channel_first({1, 2, 4, 4, 4}, attrs);
}

// ============================================================
// Random stress tests (multi-channel, partial C8)
// ============================================================

NNOPS_TEST(resize_2d_multichannel) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 16, 16};
    test_channel_last_vs_channel_first({2, 3, 32, 32}, attrs);
}

NNOPS_TEST(resize_3d_multichannel) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {4, 8, 8};
    test_channel_last_vs_channel_first({1, 3, 8, 16, 16}, attrs);
}

NNOPS_TEST(resize_add_to) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 8, 8};
    attrs.add_to = true;
    test_channel_last_vs_channel_first({1, 1, 4, 4}, attrs);
}

NNOPS_TEST(resize_2d_linear_half_pixel) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 16, 16};
    test_channel_last_vs_channel_first({1, 1, 8, 8}, attrs);
}

// ============================================================
// Partial C8 tests (zero-padded)
// ============================================================

NNOPS_TEST(resize_partial_channels_nearest) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 16, 16};
    test_channel_last_vs_channel_first({1, 5, 8, 8}, attrs);
}

NNOPS_TEST(resize_partial_channels_linear) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 16, 16};
    test_channel_last_vs_channel_first({1, 3, 8, 8}, attrs);
}

NNOPS_TEST(resize_single_channel) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 12, 12};
    test_channel_last_vs_channel_first({1, 1, 4, 4}, attrs);
}

NNOPS_TEST(resize_multi_channel_nearest) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 8, 8};
    test_channel_last_vs_channel_first({2, 17, 16, 16}, attrs);
}

// ============================================================
// Crop + resize tests
// ============================================================

NNOPS_TEST(resize_2d_crop_nearest_upsample) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 8, 8};
    attrs.crop_start = {0, 2, 2};
    attrs.crop_end   = {0, 6, 6};
    test_channel_last_vs_channel_first({1, 3, 8, 8}, attrs);
}

NNOPS_TEST(resize_2d_crop_nearest_downsample) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 4, 4};
    attrs.crop_start = {0, 4, 4};
    attrs.crop_end   = {0, 12, 12};
    test_channel_last_vs_channel_first({1, 3, 16, 16}, attrs);
}

NNOPS_TEST(resize_2d_crop_bilinear_upsample) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 12, 12};
    attrs.crop_start = {0, 3, 3};
    attrs.crop_end   = {0, 7, 7};
    test_channel_last_vs_channel_first({1, 3, 10, 10}, attrs);
}

NNOPS_TEST(resize_2d_crop_bilinear_align_corners) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::AlignCorners;
    attrs.output_size = {0, 8, 8};
    attrs.crop_start = {0, 1, 1};
    attrs.crop_end   = {0, 5, 5};
    test_channel_last_vs_channel_first({1, 1, 6, 6}, attrs);
}

NNOPS_TEST(resize_2d_crop_full_image) {
    // Crop equals full image — should produce same result as no-crop
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 16, 16};
    attrs.crop_start = {0, 0, 0};
    attrs.crop_end   = {0, 8, 8};
    test_channel_last_vs_channel_first({1, 3, 8, 8}, attrs);
}

NNOPS_TEST(resize_2d_crop_single_pixel_output) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 1, 1};
    attrs.crop_start = {0, 3, 5};
    attrs.crop_end   = {0, 5, 7};
    test_channel_last_vs_channel_first({1, 3, 8, 12}, attrs);
}

NNOPS_TEST(resize_2d_crop_asymmetric_edge) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 8, 8};
    attrs.crop_start = {0, 0, 0};
    attrs.crop_end   = {0, 6, 6};
    test_channel_last_vs_channel_first({1, 3, 10, 10}, attrs);
}

NNOPS_TEST(resize_2d_crop_bottom_right_corner) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 8, 8};
    attrs.crop_start = {0, 6, 6};
    attrs.crop_end   = {0, 12, 12};
    test_channel_last_vs_channel_first({1, 3, 12, 12}, attrs);
}

NNOPS_TEST(resize_2d_crop_multichannel) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 8, 8};
    attrs.crop_start = {0, 4, 4};
    attrs.crop_end   = {0, 12, 12};
    test_channel_last_vs_channel_first({2, 7, 16, 16}, attrs);
}

NNOPS_TEST(resize_2d_crop_partial_channels) {
    // 3 channels -> one zero-padded C8 block.
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 16, 16};
    attrs.crop_start = {0, 2, 2};
    attrs.crop_end   = {0, 10, 10};
    test_channel_last_vs_channel_first({1, 3, 12, 12}, attrs);
}

NNOPS_TEST(resize_2d_crop_add_to) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 8, 8};
    attrs.add_to = true;
    attrs.crop_start = {0, 3, 3};
    attrs.crop_end   = {0, 9, 9};
    test_channel_last_vs_channel_first({1, 3, 12, 12}, attrs);
}

// ============================================================
// 3D crop + resize tests
// ============================================================

NNOPS_TEST(resize_3d_crop_nearest) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {4, 4, 4};
    attrs.crop_start = {1, 2, 2};
    attrs.crop_end   = {3, 6, 6};
    test_channel_last_vs_channel_first({1, 3, 4, 8, 8}, attrs);
}

NNOPS_TEST(resize_3d_crop_trilinear) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {4, 6, 6};
    attrs.crop_start = {2, 2, 2};
    attrs.crop_end   = {6, 10, 10};
    test_channel_last_vs_channel_first({1, 3, 8, 12, 12}, attrs);
}

NNOPS_TEST(resize_3d_crop_full_volume) {
    // Crop equals entire 3D volume — same as no-crop
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {4, 8, 8};
    attrs.crop_start = {0, 0, 0};
    attrs.crop_end   = {4, 8, 8};
    test_channel_last_vs_channel_first({1, 1, 4, 8, 8}, attrs);
}

NNOPS_TEST(resize_3d_crop_partial_channels) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {4, 8, 8};
    attrs.crop_start = {1, 2, 2};
    attrs.crop_end   = {5, 10, 10};
    test_channel_last_vs_channel_first({1, 5, 6, 12, 12}, attrs);
}

NNOPS_TEST(resize_3d_crop_align_corners) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::AlignCorners;
    attrs.output_size = {4, 6, 6};
    attrs.crop_start = {1, 1, 1};
    attrs.crop_end   = {5, 7, 7};
    test_channel_last_vs_channel_first({1, 2, 6, 8, 8}, attrs);
}

// ============================================================
// Hand-verified crop + resize: known input → known output
// ============================================================

NNOPS_TEST(resize_2d_crop_nearest_hand_verified) {
    // 1x1x4x4 input:
    //   1   2   3   4
    //   5   6   7   8
    //   9  10  11  12
    //  13  14  15  16
    //
    // Crop [1:3, 1:3) = {6,7; 10,11}, nearest-upsample Asymmetric to 4x4.
    //
    // scale = 2/4 = 0.5.  Asymmetric: src = dst * scale.
    //   dst=0 → src=0.0, round=0
    //   dst=1 → src=0.5, round=1  (std::round half-away-from-zero)
    //   dst=2 → src=1.0, round=1
    //   dst=3 → src=1.5, round=2 → clamped to the last crop index (1)
    //
    // Expected output (4x4):
    //   6   7   7   7
    //  10  11  11  11
    //  10  11  11  11
    //  10  11  11  11
    const std::vector<int64_t> ishape = {1, 1, 4, 4};
    const std::vector<float> in_data = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
       13, 14, 15, 16
    };

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 4, 4};
    attrs.crop_start = {0, 1, 1};
    attrs.crop_end   = {0, 3, 3};

    run_resize_roundtrip(ishape, {1, 1, 4, 4}, in_data, attrs, [](const TensorView& result) {
        const float* r = result.ptr<float>();
        NNOPS_EXPECT_NEAR(r[0],  6.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[1],  7.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[2],  7.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[3],  7.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[4], 10.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[5], 11.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[6], 11.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[7], 11.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[8], 10.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[9], 11.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[10], 11.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[11], 11.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[12], 10.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[13], 11.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[14], 11.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[15], 11.0f, 1e-6f);
    });
}

NNOPS_TEST(resize_2d_crop_bilinear_hand_verified) {
    // 1x1x3x3 input:
    //   1  2  3
    //   4  5  6
    //   7  8  9
    //
    // Crop [0:2, 0:2) = {1,2; 4,5}, bilinear-upsample AlignCorners to 3x3
    // src_h = dst * (2-1)/(3-1) = dst * 0.5
    // dst_h=0 -> src_h=0.0 (row 0, col 0), dst_h=2 -> src_h=1.0 (row 1, col 1)
    // dst_h=1 -> src_h=0.5 (interpolate row 0/1)
    const std::vector<int64_t> ishape = {1, 1, 3, 3};
    const std::vector<float> in_data = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::AlignCorners;
    attrs.output_size = {0, 3, 3};
    attrs.crop_start = {0, 0, 0};
    attrs.crop_end   = {0, 2, 2};

    run_resize_roundtrip(ishape, {1, 1, 3, 3}, in_data, attrs, [](const TensorView& result) {
        const float* r = result.ptr<float>();
        NNOPS_EXPECT_NEAR(r[0], 1.0f, 1e-5f);  // (0,0): v00=1
        NNOPS_EXPECT_NEAR(r[1], 1.5f, 1e-5f);  // (0,1): h=0,w=0.5 → 0.5*1+0.5*2=1.5
        NNOPS_EXPECT_NEAR(r[2], 2.0f, 1e-5f);  // (0,2): v01=2
        NNOPS_EXPECT_NEAR(r[3], 2.5f, 1e-5f);  // (1,0): h=0.5,w=0 → 0.5*1+0.5*4=2.5
        NNOPS_EXPECT_NEAR(r[4], 3.0f, 1e-5f);  // (1,1): center interp of all 4
        NNOPS_EXPECT_NEAR(r[5], 3.5f, 1e-5f);  // (1,2): h=0.5,w=1 → 0.5*2+0.5*5=3.5
        NNOPS_EXPECT_NEAR(r[6], 4.0f, 1e-5f);  // (2,0): v10=4
        NNOPS_EXPECT_NEAR(r[7], 4.5f, 1e-5f);  // (2,1): h=1,w=0.5 → 0.5*4+0.5*5=4.5
        NNOPS_EXPECT_NEAR(r[8], 5.0f, 1e-5f);  // (2,2): v11=5
        for (int64_t i = 0; i < result.numel(); ++i) {
            NNOPS_EXPECT_TRUE(!std::isnan(r[i]));
            NNOPS_EXPECT_TRUE(!std::isinf(r[i]));
        }
    });
}
