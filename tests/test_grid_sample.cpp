/// @file test_grid_sample.cpp
/// @brief GridSample tests — channel-first and channel-last activations.
///
/// Every case runs the class API end-to-end on a channel-last input:
///   Channel-first -> LayoutConvert -> GridSample -> LayoutConvert -> channel-first
/// and checks the channel-first result against known values where the sample position
/// is predictable, else only finiteness. The grid stays channel-first.
/// Covers bilinear/nearest, zeros/border/reflection padding, align_corners,
/// add_to (residual onto a pre-filled output), batch, 3D, and an odd C.

#include "nnops/ops/grid_sample.hpp"
#include "nnops/ops/layout_convert.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

// ============================================================
// 2D Bilinear tests
// ============================================================

NNOPS_TEST(grid_sample_2d_bilinear_identity) {
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t oshape[] = {1, 1, 2, 2};
    const int64_t gshape[] = {1, 2, 2, 2};

    float in_data[4] = {1, 2, 3, 4};
    float grid_data[8] = {
        -0.5f, -0.5f,
        -0.5f,  0.5f,
         0.5f, -0.5f,
         0.5f,  0.5f,
    };

    TensorView in_channel_first(ishape, DataType::f32, in_data, TensorLayout::NCHW);
    TensorView grid(gshape, DataType::f32, grid_data, TensorLayout::NCHW);

    auto to_cl_op = LayoutConvert::create(TensorLayout::NHWC);
    {
        auto d = in_channel_first.desc();
        const TensorDesc cl_in_arr[] = {d};
        auto cl_descs = to_cl_op->getOutputTensorDesc(cl_in_arr);
        NNOPS_EXPECT_EQ(cl_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(cl_descs[0].layout, TensorLayout::NHWC);
        NNOPS_EXPECT_EQ(cl_descs[0].dtype, DataType::f32);

        std::vector<char> cl_buf(cl_descs[0].storage_bytes());
        auto in_cl = test::make_packed(cl_descs[0], cl_buf.data());
        const TensorView cl_conv_ins[] = {in_channel_first};
        to_cl_op->compute(in_cl, cl_conv_ins);

        auto gs_op = GridSample::create({});
        auto d_in = in_cl.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NHWC);
        NNOPS_EXPECT_EQ(gs_descs[0].dtype, DataType::f32);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_cl = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        auto to_channel_first_op = LayoutConvert::create(TensorLayout::NCHW);
        auto d_out = out_cl.desc();
        const TensorDesc channel_first_in_arr[] = {d_out};
        auto channel_first_descs = to_channel_first_op->getOutputTensorDesc(channel_first_in_arr);
        NNOPS_EXPECT_EQ(channel_first_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(channel_first_descs[0].layout, TensorLayout::NCHW);
        NNOPS_EXPECT_EQ(channel_first_descs[0].dtype, DataType::f32);

        std::vector<char> channel_first_buf(channel_first_descs[0].storage_bytes());
        auto res_channel_first = test::make_channel_first(channel_first_descs[0], channel_first_buf.data());
        const TensorView channel_first_ins[] = {out_cl};
        to_channel_first_op->compute(res_channel_first, channel_first_ins);

        NNOPS_EXPECT_NEAR(res_channel_first.ptr<float>()[0], 1.0f, 1e-5f);
        NNOPS_EXPECT_NEAR(res_channel_first.ptr<float>()[1], 2.0f, 1e-5f);
        NNOPS_EXPECT_NEAR(res_channel_first.ptr<float>()[2], 3.0f, 1e-5f);
        NNOPS_EXPECT_NEAR(res_channel_first.ptr<float>()[3], 4.0f, 1e-5f);
    }
}

NNOPS_TEST(grid_sample_2d_bilinear_align_corners) {
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t gshape[] = {1, 4, 4, 2};

    float in_data[4] = {1, 2, 3, 4};

    TensorView in_channel_first(ishape, DataType::f32, in_data, TensorLayout::NCHW);

    std::vector<float> grid_buf(1 * 4 * 4 * 2);
    for (int64_t oh = 0; oh < 4; ++oh) {
        for (int64_t ow = 0; ow < 4; ++ow) {
            float gy = static_cast<float>(oh) / 3.0f * 2.0f - 1.0f;
            float gx = static_cast<float>(ow) / 3.0f * 2.0f - 1.0f;
            grid_buf[(oh * 4 + ow) * 2]     = gy;
            grid_buf[(oh * 4 + ow) * 2 + 1] = gx;
        }
    }
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::NCHW);

    auto to_cl_op = LayoutConvert::create(TensorLayout::NHWC);
    {
        auto d = in_channel_first.desc();
        const TensorDesc cl_in_arr[] = {d};
        auto cl_descs = to_cl_op->getOutputTensorDesc(cl_in_arr);
        NNOPS_EXPECT_EQ(cl_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(cl_descs[0].layout, TensorLayout::NHWC);

        std::vector<char> cl_buf(cl_descs[0].storage_bytes());
        auto in_cl = test::make_packed(cl_descs[0], cl_buf.data());
        const TensorView cl_conv_ins[] = {in_channel_first};
        to_cl_op->compute(in_cl, cl_conv_ins);

        GridSampleAttributes attrs;
        attrs.align_corners = true;
        auto gs_op = GridSample::create(attrs);
        auto d_in = in_cl.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NHWC);
        NNOPS_EXPECT_EQ(gs_descs[0].dtype, DataType::f32);
        NNOPS_EXPECT_EQ(gs_descs[0].dims[0], 1);
        NNOPS_EXPECT_EQ(gs_descs[0].dims[1], 1);
        NNOPS_EXPECT_EQ(gs_descs[0].dims[2], 4);
        NNOPS_EXPECT_EQ(gs_descs[0].dims[3], 4);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_cl = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        auto to_channel_first_op = LayoutConvert::create(TensorLayout::NCHW);
        auto d_out = out_cl.desc();
        const TensorDesc channel_first_in_arr[] = {d_out};
        auto channel_first_descs = to_channel_first_op->getOutputTensorDesc(channel_first_in_arr);
        NNOPS_EXPECT_EQ(channel_first_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(channel_first_descs[0].layout, TensorLayout::NCHW);

        std::vector<char> channel_first_buf(channel_first_descs[0].storage_bytes());
        auto res_channel_first = test::make_channel_first(channel_first_descs[0], channel_first_buf.data());
        const TensorView channel_first_ins[] = {out_cl};
        to_channel_first_op->compute(res_channel_first, channel_first_ins);

        NNOPS_EXPECT_NEAR(res_channel_first.ptr<float>()[0],  1.0f, 1e-5f);
        NNOPS_EXPECT_NEAR(res_channel_first.ptr<float>()[3],  2.0f, 1e-5f);
        NNOPS_EXPECT_NEAR(res_channel_first.ptr<float>()[12], 3.0f, 1e-5f);
        NNOPS_EXPECT_NEAR(res_channel_first.ptr<float>()[15], 4.0f, 1e-5f);

        for (int64_t i = 0; i < 16; ++i) {
            NNOPS_EXPECT_TRUE(!std::isnan(res_channel_first.ptr<float>()[i]));
            NNOPS_EXPECT_TRUE(!std::isinf(res_channel_first.ptr<float>()[i]));
        }
    }
}

NNOPS_TEST(grid_sample_2d_bilinear_random) {
    auto [in_vec, in_channel_first] = test::make_random_tensor({1, 2, 8, 8});
    const int64_t gshape[] = {1, 6, 6, 2};

    std::vector<float> grid_buf(1 * 6 * 6 * 2);
    for (size_t i = 0; i < grid_buf.size(); ++i) {
        grid_buf[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::NCHW);

    auto to_cl_op = LayoutConvert::create(TensorLayout::NHWC);
    {
        auto d = in_channel_first.desc();
        const TensorDesc cl_in_arr[] = {d};
        auto cl_descs = to_cl_op->getOutputTensorDesc(cl_in_arr);
        NNOPS_EXPECT_EQ(cl_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(cl_descs[0].layout, TensorLayout::NHWC);

        std::vector<char> cl_buf(cl_descs[0].storage_bytes());
        auto in_cl = test::make_packed(cl_descs[0], cl_buf.data());
        const TensorView cl_conv_ins[] = {in_channel_first};
        to_cl_op->compute(in_cl, cl_conv_ins);

        auto gs_op = GridSample::create({});
        auto d_in = in_cl.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NHWC);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_cl = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        auto to_channel_first_op = LayoutConvert::create(TensorLayout::NCHW);
        auto d_out = out_cl.desc();
        const TensorDesc channel_first_in_arr[] = {d_out};
        auto channel_first_descs = to_channel_first_op->getOutputTensorDesc(channel_first_in_arr);
        NNOPS_EXPECT_EQ(channel_first_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(channel_first_descs[0].layout, TensorLayout::NCHW);

        std::vector<char> channel_first_buf(channel_first_descs[0].storage_bytes());
        auto res_channel_first = test::make_channel_first(channel_first_descs[0], channel_first_buf.data());
        const TensorView channel_first_ins[] = {out_cl};
        to_channel_first_op->compute(res_channel_first, channel_first_ins);

        for (size_t i = 0; i < 72; ++i) {
            NNOPS_EXPECT_TRUE(!std::isnan(res_channel_first.ptr<float>()[i]));
            NNOPS_EXPECT_TRUE(!std::isinf(res_channel_first.ptr<float>()[i]));
        }
    }
}

// ============================================================
// 2D Nearest-neighbor tests
// ============================================================

NNOPS_TEST(grid_sample_2d_nearest_basic) {
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t gshape[] = {1, 2, 2, 2};

    float in_data[4] = {1, 2, 3, 4};
    float grid_data[8] = {
        0.5f, 0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f, 0.5f,
    };

    TensorView in_channel_first(ishape, DataType::f32, in_data, TensorLayout::NCHW);
    TensorView grid(gshape, DataType::f32, grid_data, TensorLayout::NCHW);

    auto to_cl_op = LayoutConvert::create(TensorLayout::NHWC);
    {
        auto d = in_channel_first.desc();
        const TensorDesc cl_in_arr[] = {d};
        auto cl_descs = to_cl_op->getOutputTensorDesc(cl_in_arr);
        NNOPS_EXPECT_EQ(cl_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(cl_descs[0].layout, TensorLayout::NHWC);

        std::vector<char> cl_buf(cl_descs[0].storage_bytes());
        auto in_cl = test::make_packed(cl_descs[0], cl_buf.data());
        const TensorView cl_conv_ins[] = {in_channel_first};
        to_cl_op->compute(in_cl, cl_conv_ins);

        GridSampleAttributes attrs;
        attrs.mode = GridSampleMode::Nearest;
        auto gs_op = GridSample::create(attrs);
        auto d_in = in_cl.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NHWC);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_cl = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        auto to_channel_first_op = LayoutConvert::create(TensorLayout::NCHW);
        auto d_out = out_cl.desc();
        const TensorDesc channel_first_in_arr[] = {d_out};
        auto channel_first_descs = to_channel_first_op->getOutputTensorDesc(channel_first_in_arr);
        NNOPS_EXPECT_EQ(channel_first_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(channel_first_descs[0].layout, TensorLayout::NCHW);

        std::vector<char> channel_first_buf(channel_first_descs[0].storage_bytes());
        auto res_channel_first = test::make_channel_first(channel_first_descs[0], channel_first_buf.data());
        const TensorView channel_first_ins[] = {out_cl};
        to_channel_first_op->compute(res_channel_first, channel_first_ins);

        for (int i = 0; i < 4; ++i) {
            NNOPS_EXPECT_NEAR(res_channel_first.ptr<float>()[i], 4.0f, 1e-5f);
        }
    }
}

NNOPS_TEST(grid_sample_2d_nearest_random) {
    auto [in_vec, in_channel_first] = test::make_random_tensor({1, 3, 10, 10});
    const int64_t gshape[] = {1, 5, 5, 2};

    std::vector<float> grid_buf(1 * 5 * 5 * 2);
    for (size_t i = 0; i < grid_buf.size(); ++i) {
        grid_buf[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::NCHW);

    auto to_cl_op = LayoutConvert::create(TensorLayout::NHWC);
    {
        auto d = in_channel_first.desc();
        const TensorDesc cl_in_arr[] = {d};
        auto cl_descs = to_cl_op->getOutputTensorDesc(cl_in_arr);
        NNOPS_EXPECT_EQ(cl_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(cl_descs[0].layout, TensorLayout::NHWC);

        std::vector<char> cl_buf(cl_descs[0].storage_bytes());
        auto in_cl = test::make_packed(cl_descs[0], cl_buf.data());
        const TensorView cl_conv_ins[] = {in_channel_first};
        to_cl_op->compute(in_cl, cl_conv_ins);

        GridSampleAttributes attrs;
        attrs.mode = GridSampleMode::Nearest;
        auto gs_op = GridSample::create(attrs);
        auto d_in = in_cl.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NHWC);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_cl = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        auto to_channel_first_op = LayoutConvert::create(TensorLayout::NCHW);
        auto d_out = out_cl.desc();
        const TensorDesc channel_first_in_arr[] = {d_out};
        auto channel_first_descs = to_channel_first_op->getOutputTensorDesc(channel_first_in_arr);
        NNOPS_EXPECT_EQ(channel_first_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(channel_first_descs[0].layout, TensorLayout::NCHW);

        std::vector<char> channel_first_buf(channel_first_descs[0].storage_bytes());
        auto res_channel_first = test::make_channel_first(channel_first_descs[0], channel_first_buf.data());
        const TensorView channel_first_ins[] = {out_cl};
        to_channel_first_op->compute(res_channel_first, channel_first_ins);

        for (size_t i = 0; i < 75; ++i) {
            NNOPS_EXPECT_TRUE(!std::isnan(res_channel_first.ptr<float>()[i]));
            NNOPS_EXPECT_TRUE(!std::isinf(res_channel_first.ptr<float>()[i]));
        }
    }
}

// ============================================================
// Padding mode tests
// ============================================================

NNOPS_TEST(grid_sample_2d_zeros_padding) {
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t gshape[] = {1, 2, 2, 2};

    float in_data[4] = {1, 2, 3, 4};
    float grid_data[8] = {
        10.0f, 10.0f, 10.0f, 10.0f,
        10.0f, 10.0f, 10.0f, 10.0f,
    };

    TensorView in_channel_first(ishape, DataType::f32, in_data, TensorLayout::NCHW);
    TensorView grid(gshape, DataType::f32, grid_data, TensorLayout::NCHW);

    auto to_cl_op = LayoutConvert::create(TensorLayout::NHWC);
    {
        auto d = in_channel_first.desc();
        const TensorDesc cl_in_arr[] = {d};
        auto cl_descs = to_cl_op->getOutputTensorDesc(cl_in_arr);
        NNOPS_EXPECT_EQ(cl_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(cl_descs[0].layout, TensorLayout::NHWC);

        std::vector<char> cl_buf(cl_descs[0].storage_bytes());
        auto in_cl = test::make_packed(cl_descs[0], cl_buf.data());
        const TensorView cl_conv_ins[] = {in_channel_first};
        to_cl_op->compute(in_cl, cl_conv_ins);

        auto gs_op = GridSample::create({});
        auto d_in = in_cl.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NHWC);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_cl = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        auto to_channel_first_op = LayoutConvert::create(TensorLayout::NCHW);
        auto d_out = out_cl.desc();
        const TensorDesc channel_first_in_arr[] = {d_out};
        auto channel_first_descs = to_channel_first_op->getOutputTensorDesc(channel_first_in_arr);
        NNOPS_EXPECT_EQ(channel_first_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(channel_first_descs[0].layout, TensorLayout::NCHW);

        std::vector<char> channel_first_buf(channel_first_descs[0].storage_bytes());
        auto res_channel_first = test::make_channel_first(channel_first_descs[0], channel_first_buf.data());
        const TensorView channel_first_ins[] = {out_cl};
        to_channel_first_op->compute(res_channel_first, channel_first_ins);

        for (int i = 0; i < 4; ++i) {
            NNOPS_EXPECT_NEAR(res_channel_first.ptr<float>()[i], 0.0f, 1e-5f);
        }
    }
}

NNOPS_TEST(grid_sample_2d_border_padding) {
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t gshape[] = {1, 2, 2, 2};

    float in_data[4] = {10, 20, 30, 40};
    float grid_data[8] = {
         0.0f, -10.0f, 0.0f, -10.0f,
         0.0f, -10.0f, 0.0f, -10.0f,
    };

    TensorView in_channel_first(ishape, DataType::f32, in_data, TensorLayout::NCHW);
    TensorView grid(gshape, DataType::f32, grid_data, TensorLayout::NCHW);

    auto to_cl_op = LayoutConvert::create(TensorLayout::NHWC);
    {
        auto d = in_channel_first.desc();
        const TensorDesc cl_in_arr[] = {d};
        auto cl_descs = to_cl_op->getOutputTensorDesc(cl_in_arr);
        NNOPS_EXPECT_EQ(cl_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(cl_descs[0].layout, TensorLayout::NHWC);

        std::vector<char> cl_buf(cl_descs[0].storage_bytes());
        auto in_cl = test::make_packed(cl_descs[0], cl_buf.data());
        const TensorView cl_conv_ins[] = {in_channel_first};
        to_cl_op->compute(in_cl, cl_conv_ins);

        GridSampleAttributes attrs;
        attrs.padding_mode = GridSamplePaddingMode::Border;
        auto gs_op = GridSample::create(attrs);
        auto d_in = in_cl.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NHWC);
        NNOPS_EXPECT_EQ(gs_descs[0].dtype, DataType::f32);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_cl = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        auto to_channel_first_op = LayoutConvert::create(TensorLayout::NCHW);
        auto d_out = out_cl.desc();
        const TensorDesc channel_first_in_arr[] = {d_out};
        auto channel_first_descs = to_channel_first_op->getOutputTensorDesc(channel_first_in_arr);
        NNOPS_EXPECT_EQ(channel_first_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(channel_first_descs[0].layout, TensorLayout::NCHW);

        std::vector<char> channel_first_buf(channel_first_descs[0].storage_bytes());
        auto res_channel_first = test::make_channel_first(channel_first_descs[0], channel_first_buf.data());
        const TensorView channel_first_ins[] = {out_cl};
        to_channel_first_op->compute(res_channel_first, channel_first_ins);

        for (int i = 0; i < 4; ++i) {
            NNOPS_EXPECT_TRUE(!std::isnan(res_channel_first.ptr<float>()[i]));
            NNOPS_EXPECT_TRUE(!std::isinf(res_channel_first.ptr<float>()[i]));
        }
    }
}

NNOPS_TEST(grid_sample_2d_reflection_padding) {
    auto [in_vec, in_channel_first] = test::make_random_tensor({1, 1, 4, 4});
    const int64_t gshape[] = {1, 4, 4, 2};

    std::vector<float> grid_buf(1 * 4 * 4 * 2);
    for (int64_t oh = 0; oh < 4; ++oh) {
        for (int64_t ow = 0; ow < 4; ++ow) {
            float gy = (static_cast<float>(oh) / 3.0f) * 2.4f - 1.2f;
            float gx = (static_cast<float>(ow) / 3.0f) * 2.4f - 1.2f;
            grid_buf[(oh * 4 + ow) * 2]     = gy;
            grid_buf[(oh * 4 + ow) * 2 + 1] = gx;
        }
    }
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::NCHW);

    auto to_cl_op = LayoutConvert::create(TensorLayout::NHWC);
    {
        auto d = in_channel_first.desc();
        const TensorDesc cl_in_arr[] = {d};
        auto cl_descs = to_cl_op->getOutputTensorDesc(cl_in_arr);
        NNOPS_EXPECT_EQ(cl_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(cl_descs[0].layout, TensorLayout::NHWC);

        std::vector<char> cl_buf(cl_descs[0].storage_bytes());
        auto in_cl = test::make_packed(cl_descs[0], cl_buf.data());
        const TensorView cl_conv_ins[] = {in_channel_first};
        to_cl_op->compute(in_cl, cl_conv_ins);

        GridSampleAttributes attrs;
        attrs.padding_mode = GridSamplePaddingMode::Reflection;
        auto gs_op = GridSample::create(attrs);
        auto d_in = in_cl.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NHWC);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_cl = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        auto to_channel_first_op = LayoutConvert::create(TensorLayout::NCHW);
        auto d_out = out_cl.desc();
        const TensorDesc channel_first_in_arr[] = {d_out};
        auto channel_first_descs = to_channel_first_op->getOutputTensorDesc(channel_first_in_arr);
        NNOPS_EXPECT_EQ(channel_first_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(channel_first_descs[0].layout, TensorLayout::NCHW);

        std::vector<char> channel_first_buf(channel_first_descs[0].storage_bytes());
        auto res_channel_first = test::make_channel_first(channel_first_descs[0], channel_first_buf.data());
        const TensorView channel_first_ins[] = {out_cl};
        to_channel_first_op->compute(res_channel_first, channel_first_ins);

        for (size_t i = 0; i < 16; ++i) {
            NNOPS_EXPECT_TRUE(!std::isnan(res_channel_first.ptr<float>()[i]));
            NNOPS_EXPECT_TRUE(!std::isinf(res_channel_first.ptr<float>()[i]));
        }
    }
}

// ============================================================
// 3D tests
// ============================================================

NNOPS_TEST(grid_sample_3d_bilinear_basic) {
    const int64_t ishape[] = {1, 1, 2, 2, 2};
    const int64_t gshape[] = {1, 2, 2, 2, 3};

    std::vector<float> in_buf(8);
    for (int i = 0; i < 8; ++i) {
        in_buf[i] = static_cast<float>(i + 1);
    }

    TensorView in_ncdhw(ishape, DataType::f32, in_buf.data(), TensorLayout::NCDHW);

    std::vector<float> grid_buf(1 * 2 * 2 * 2 * 3);
    for (int64_t od = 0; od < 2; ++od) {
        for (int64_t oh = 0; oh < 2; ++oh) {
            for (int64_t ow = 0; ow < 2; ++ow) {
                float gd = (od == 0) ? -0.5f : 0.5f;
                float gh = (oh == 0) ? -0.5f : 0.5f;
                float gw = (ow == 0) ? -0.5f : 0.5f;
                size_t idx = (od * 2 * 2 + oh * 2 + ow) * 3;
                grid_buf[idx]     = gd;
                grid_buf[idx + 1] = gh;
                grid_buf[idx + 2] = gw;
            }
        }
    }
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::NCHW);

    auto to_cl_op = LayoutConvert::create(TensorLayout::NDHWC);
    {
        auto d = in_ncdhw.desc();
        const TensorDesc cl_in_arr[] = {d};
        auto cl_descs = to_cl_op->getOutputTensorDesc(cl_in_arr);
        NNOPS_EXPECT_EQ(cl_descs[0].rank, 5);
        NNOPS_EXPECT_EQ(cl_descs[0].layout, TensorLayout::NDHWC);
        NNOPS_EXPECT_EQ(cl_descs[0].dtype, DataType::f32);

        std::vector<char> cl_buf(cl_descs[0].storage_bytes());
        auto in_cl = test::make_packed(cl_descs[0], cl_buf.data());
        const TensorView cl_conv_ins[] = {in_ncdhw};
        to_cl_op->compute(in_cl, cl_conv_ins);

        auto gs_op = GridSample::create({});
        auto d_in = in_cl.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 5);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NDHWC);
        NNOPS_EXPECT_EQ(gs_descs[0].dtype, DataType::f32);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_cl = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        auto to_channel_first_op = LayoutConvert::create(TensorLayout::NCDHW);
        auto d_out = out_cl.desc();
        const TensorDesc channel_first_in_arr[] = {d_out};
        auto channel_first_descs = to_channel_first_op->getOutputTensorDesc(channel_first_in_arr);
        NNOPS_EXPECT_EQ(channel_first_descs[0].rank, 5);
        NNOPS_EXPECT_EQ(channel_first_descs[0].layout, TensorLayout::NCDHW);
        NNOPS_EXPECT_EQ(channel_first_descs[0].dtype, DataType::f32);

        std::vector<char> channel_first_buf(channel_first_descs[0].storage_bytes());
        auto res_ncdhw = test::make_channel_first(channel_first_descs[0], channel_first_buf.data());
        const TensorView channel_first_ins[] = {out_cl};
        to_channel_first_op->compute(res_ncdhw, channel_first_ins);

        for (size_t i = 0; i < 8; ++i) {
            NNOPS_EXPECT_NEAR(res_ncdhw.ptr<float>()[i], static_cast<float>(i + 1), 1e-4f);
        }
    }
}

NNOPS_TEST(grid_sample_3d_nearest_basic) {
    auto [in_vec, in_ncdhw] = test::make_random_tensor({1, 1, 4, 4, 4});
    const int64_t gshape[] = {1, 3, 3, 3, 3};

    std::vector<float> grid_buf(1 * 3 * 3 * 3 * 3);
    for (size_t i = 0; i < grid_buf.size(); ++i) {
        grid_buf[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::NCHW);

    auto to_cl_op = LayoutConvert::create(TensorLayout::NDHWC);
    {
        auto d = in_ncdhw.desc();
        const TensorDesc cl_in_arr[] = {d};
        auto cl_descs = to_cl_op->getOutputTensorDesc(cl_in_arr);
        NNOPS_EXPECT_EQ(cl_descs[0].rank, 5);
        NNOPS_EXPECT_EQ(cl_descs[0].layout, TensorLayout::NDHWC);

        std::vector<char> cl_buf(cl_descs[0].storage_bytes());
        auto in_cl = test::make_packed(cl_descs[0], cl_buf.data());
        const TensorView cl_conv_ins[] = {in_ncdhw};
        to_cl_op->compute(in_cl, cl_conv_ins);

        GridSampleAttributes attrs;
        attrs.mode = GridSampleMode::Nearest;
        auto gs_op = GridSample::create(attrs);
        auto d_in = in_cl.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 5);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NDHWC);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_cl = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        auto to_channel_first_op = LayoutConvert::create(TensorLayout::NCDHW);
        auto d_out = out_cl.desc();
        const TensorDesc channel_first_in_arr[] = {d_out};
        auto channel_first_descs = to_channel_first_op->getOutputTensorDesc(channel_first_in_arr);
        NNOPS_EXPECT_EQ(channel_first_descs[0].rank, 5);
        NNOPS_EXPECT_EQ(channel_first_descs[0].layout, TensorLayout::NCDHW);

        std::vector<char> channel_first_buf(channel_first_descs[0].storage_bytes());
        auto res_ncdhw = test::make_channel_first(channel_first_descs[0], channel_first_buf.data());
        const TensorView channel_first_ins[] = {out_cl};
        to_channel_first_op->compute(res_ncdhw, channel_first_ins);

        for (size_t i = 0; i < 27; ++i) {
            NNOPS_EXPECT_TRUE(!std::isnan(res_ncdhw.ptr<float>()[i]));
            NNOPS_EXPECT_TRUE(!std::isinf(res_ncdhw.ptr<float>()[i]));
        }
    }
}

// ============================================================
// Add-to mode
// ============================================================

NNOPS_TEST(grid_sample_add_to) {
    auto [in_vec, in_channel_first] = test::make_random_tensor({1, 1, 4, 4});
    const int64_t gshape[] = {1, 4, 4, 2};

    std::vector<float> grid_buf(1 * 4 * 4 * 2);
    for (int64_t oh = 0; oh < 4; ++oh) {
        for (int64_t ow = 0; ow < 4; ++ow) {
            float gy = (static_cast<float>(oh) + 0.5f) / 2.0f - 1.0f;
            float gx = (static_cast<float>(ow) + 0.5f) / 2.0f - 1.0f;
            grid_buf[(oh * 4 + ow) * 2]     = gy;
            grid_buf[(oh * 4 + ow) * 2 + 1] = gx;
        }
    }
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::NCHW);

    auto to_cl_op = LayoutConvert::create(TensorLayout::NHWC);
    {
        auto d = in_channel_first.desc();
        const TensorDesc cl_in_arr[] = {d};
        auto cl_descs = to_cl_op->getOutputTensorDesc(cl_in_arr);
        NNOPS_EXPECT_EQ(cl_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(cl_descs[0].layout, TensorLayout::NHWC);

        std::vector<char> cl_buf(cl_descs[0].storage_bytes());
        auto in_cl = test::make_packed(cl_descs[0], cl_buf.data());
        const TensorView cl_conv_ins[] = {in_channel_first};
        to_cl_op->compute(in_cl, cl_conv_ins);

        GridSampleAttributes attrs;
        attrs.add_to = true;
        auto gs_op = GridSample::create(attrs);
        auto d_in = in_cl.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NHWC);
        NNOPS_EXPECT_EQ(gs_descs[0].dtype, DataType::f32);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_cl = test::make_packed(gs_descs[0], gs_buf.data());

        int64_t rows = out_cl.total_rows();
        int64_t row_elems = out_cl.row_stride_elems();
        for (int64_t r = 0; r < rows; ++r) {
            float* row = out_cl.ptr<float>(r);
            for (int64_t e = 0; e < row_elems; ++e) {
                row[e] = 0.5f;
            }
        }

        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        auto to_channel_first_op = LayoutConvert::create(TensorLayout::NCHW);
        auto d_out = out_cl.desc();
        const TensorDesc channel_first_in_arr[] = {d_out};
        auto channel_first_descs = to_channel_first_op->getOutputTensorDesc(channel_first_in_arr);
        NNOPS_EXPECT_EQ(channel_first_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(channel_first_descs[0].layout, TensorLayout::NCHW);

        std::vector<char> channel_first_buf(channel_first_descs[0].storage_bytes());
        auto res_channel_first = test::make_channel_first(channel_first_descs[0], channel_first_buf.data());
        const TensorView channel_first_ins[] = {out_cl};
        to_channel_first_op->compute(res_channel_first, channel_first_ins);

        for (size_t i = 0; i < 16; ++i) {
            NNOPS_EXPECT_TRUE(!std::isnan(res_channel_first.ptr<float>()[i]));
            NNOPS_EXPECT_TRUE(!std::isinf(res_channel_first.ptr<float>()[i]));
        }
    }
}

// ============================================================
// Multi-sample batch test
// ============================================================

NNOPS_TEST(grid_sample_multisample) {
    auto [in_vec, in_channel_first] = test::make_random_tensor({2, 1, 4, 4});
    const int64_t gshape[] = {2, 3, 3, 2};

    std::vector<float> grid_buf(2 * 3 * 3 * 2);
    for (size_t i = 0; i < grid_buf.size(); ++i) {
        grid_buf[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::NCHW);

    auto to_cl_op = LayoutConvert::create(TensorLayout::NHWC);
    {
        auto d = in_channel_first.desc();
        const TensorDesc cl_in_arr[] = {d};
        auto cl_descs = to_cl_op->getOutputTensorDesc(cl_in_arr);
        NNOPS_EXPECT_EQ(cl_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(cl_descs[0].layout, TensorLayout::NHWC);

        std::vector<char> cl_buf(cl_descs[0].storage_bytes());
        auto in_cl = test::make_packed(cl_descs[0], cl_buf.data());
        const TensorView cl_conv_ins[] = {in_channel_first};
        to_cl_op->compute(in_cl, cl_conv_ins);

        auto gs_op = GridSample::create({});
        auto d_in = in_cl.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NHWC);
        NNOPS_EXPECT_EQ(gs_descs[0].dtype, DataType::f32);
        NNOPS_EXPECT_EQ(gs_descs[0].dims[0], 2);
        NNOPS_EXPECT_EQ(gs_descs[0].dims[1], 1);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_cl = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        auto to_channel_first_op = LayoutConvert::create(TensorLayout::NCHW);
        auto d_out = out_cl.desc();
        const TensorDesc channel_first_in_arr[] = {d_out};
        auto channel_first_descs = to_channel_first_op->getOutputTensorDesc(channel_first_in_arr);
        NNOPS_EXPECT_EQ(channel_first_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(channel_first_descs[0].layout, TensorLayout::NCHW);

        std::vector<char> channel_first_buf(channel_first_descs[0].storage_bytes());
        auto res_channel_first = test::make_channel_first(channel_first_descs[0], channel_first_buf.data());
        const TensorView channel_first_ins[] = {out_cl};
        to_channel_first_op->compute(res_channel_first, channel_first_ins);

        for (size_t i = 0; i < 18; ++i) {
            NNOPS_EXPECT_TRUE(!std::isnan(res_channel_first.ptr<float>()[i]));
            NNOPS_EXPECT_TRUE(!std::isinf(res_channel_first.ptr<float>()[i]));
        }
    }
}

// ============================================================
// Align corners: 2D edge cases
// ============================================================

NNOPS_TEST(grid_sample_2d_align_corners_corners_exact) {
    const int64_t ishape[] = {1, 1, 3, 3};
    const int64_t gshape[] = {1, 3, 3, 2};

    float in_data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    TensorView in_channel_first(ishape, DataType::f32, in_data, TensorLayout::NCHW);

    std::vector<float> grid_buf(1 * 3 * 3 * 2);
    for (int64_t oh = 0; oh < 3; ++oh) {
        for (int64_t ow = 0; ow < 3; ++ow) {
            float gy = static_cast<float>(oh) / 1.0f - 1.0f;
            float gx = static_cast<float>(ow) / 1.0f - 1.0f;
            grid_buf[(oh * 3 + ow) * 2]     = gy;
            grid_buf[(oh * 3 + ow) * 2 + 1] = gx;
        }
    }
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::NCHW);

    auto to_cl_op = LayoutConvert::create(TensorLayout::NHWC);
    {
        auto d = in_channel_first.desc();
        const TensorDesc cl_in_arr[] = {d};
        auto cl_descs = to_cl_op->getOutputTensorDesc(cl_in_arr);
        NNOPS_EXPECT_EQ(cl_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(cl_descs[0].layout, TensorLayout::NHWC);

        std::vector<char> cl_buf(cl_descs[0].storage_bytes());
        auto in_cl = test::make_packed(cl_descs[0], cl_buf.data());
        const TensorView cl_conv_ins[] = {in_channel_first};
        to_cl_op->compute(in_cl, cl_conv_ins);

        GridSampleAttributes attrs;
        attrs.align_corners = true;
        auto gs_op = GridSample::create(attrs);
        auto d_in = in_cl.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NHWC);
        NNOPS_EXPECT_EQ(gs_descs[0].dtype, DataType::f32);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_cl = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        auto to_channel_first_op = LayoutConvert::create(TensorLayout::NCHW);
        auto d_out = out_cl.desc();
        const TensorDesc channel_first_in_arr[] = {d_out};
        auto channel_first_descs = to_channel_first_op->getOutputTensorDesc(channel_first_in_arr);
        NNOPS_EXPECT_EQ(channel_first_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(channel_first_descs[0].layout, TensorLayout::NCHW);

        std::vector<char> channel_first_buf(channel_first_descs[0].storage_bytes());
        auto res_channel_first = test::make_channel_first(channel_first_descs[0], channel_first_buf.data());
        const TensorView channel_first_ins[] = {out_cl};
        to_channel_first_op->compute(res_channel_first, channel_first_ins);

        for (int i = 0; i < 9; ++i) {
            NNOPS_EXPECT_NEAR(res_channel_first.ptr<float>()[i], in_data[i], 1e-5f);
        }
    }
}

// ============================================================
// Partial C8 tests (zero-padded, all C8 blocks use SIMD)
// ============================================================

NNOPS_TEST(grid_sample_partial_channels) {
    auto [in_vec, in_channel_first] = test::make_random_tensor({1, 3, 4, 4});
    const int64_t gshape[] = {1, 4, 4, 2};

    std::vector<float> grid_buf(1 * 4 * 4 * 2);
    for (int64_t oh = 0; oh < 4; ++oh) {
        for (int64_t ow = 0; ow < 4; ++ow) {
            float gy = (static_cast<float>(oh) + 0.5f) / 2.0f - 1.0f;
            float gx = (static_cast<float>(ow) + 0.5f) / 2.0f - 1.0f;
            grid_buf[(oh * 4 + ow) * 2]     = gy;
            grid_buf[(oh * 4 + ow) * 2 + 1] = gx;
        }
    }
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::NCHW);

    auto to_cl_op = LayoutConvert::create(TensorLayout::NHWC);
    {
        auto d = in_channel_first.desc();
        const TensorDesc cl_in_arr[] = {d};
        auto cl_descs = to_cl_op->getOutputTensorDesc(cl_in_arr);
        NNOPS_EXPECT_EQ(cl_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(cl_descs[0].layout, TensorLayout::NHWC);

        std::vector<char> cl_buf(cl_descs[0].storage_bytes());
        auto in_cl = test::make_packed(cl_descs[0], cl_buf.data());
        const TensorView cl_conv_ins[] = {in_channel_first};
        to_cl_op->compute(in_cl, cl_conv_ins);

        auto gs_op = GridSample::create({});
        auto d_in = in_cl.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NHWC);
        NNOPS_EXPECT_EQ(gs_descs[0].dtype, DataType::f32);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_cl = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        auto to_channel_first_op = LayoutConvert::create(TensorLayout::NCHW);
        auto d_out = out_cl.desc();
        const TensorDesc channel_first_in_arr[] = {d_out};
        auto channel_first_descs = to_channel_first_op->getOutputTensorDesc(channel_first_in_arr);
        NNOPS_EXPECT_EQ(channel_first_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(channel_first_descs[0].layout, TensorLayout::NCHW);

        std::vector<char> channel_first_buf(channel_first_descs[0].storage_bytes());
        auto res_channel_first = test::make_channel_first(channel_first_descs[0], channel_first_buf.data());
        const TensorView channel_first_ins[] = {out_cl};
        to_channel_first_op->compute(res_channel_first, channel_first_ins);

        for (size_t i = 0; i < 48; ++i) {
            NNOPS_EXPECT_TRUE(!std::isnan(res_channel_first.ptr<float>()[i]));
            NNOPS_EXPECT_TRUE(!std::isinf(res_channel_first.ptr<float>()[i]));
        }
    }
}
