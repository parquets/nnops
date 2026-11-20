/// @file test_grid_sample.cpp
/// @brief GridSample tests on NHWC activations.
///
/// Every case runs the class API end-to-end on an NHWC input:
///   dense -> NHWC view -> GridSample -> dense copy
/// and checks the dense-order result against known values where the sample
/// position is predictable, else only finiteness. The grid stays dense.
/// Covers bilinear/nearest, zeros/border/reflection padding, align_corners,
/// add_to (residual onto a pre-filled output), batch, 3D, and an odd C.

#include "nnops/ops/grid_sample.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"
#include "common/channel_last.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

using namespace nnops;

namespace {

/// Dense [N, C, spatial...] dims → channel-last [N, spatial..., C].
std::vector<int64_t> cl_dims_of(const TensorView& cf)
{
    const int64_t rank = cf.rank();
    std::vector<int64_t> d(static_cast<size_t>(rank));
    d[0] = cf.shape(0);
    for (int64_t i = 2; i < rank; ++i) { d[static_cast<size_t>(i - 1)] = cf.shape(i); }
    d[static_cast<size_t>(rank - 1)] = cf.shape(1);
    return d;
}

/// Channel-last [N, spatial..., C] dims → dense [N, C, spatial...].
std::vector<int64_t> dense_dims_of(const TensorView& cl)
{
    const int64_t rank = cl.rank();
    std::vector<int64_t> d(static_cast<size_t>(rank));
    d[0] = cl.shape(0);
    d[1] = cl.shape(rank - 1);
    for (int64_t i = 1; i + 1 < rank; ++i) { d[static_cast<size_t>(i + 1)] = cl.shape(i); }
    return d;
}

/// Pixel-major reorder of a dense view.
std::vector<float> to_channel_last_data(const TensorView& cf)
{
    const int64_t N = cf.shape(0);
    const int64_t C = cf.shape(1);
    const int64_t S = cf.numel() / (N * C);
    return test::to_channel_last(
        std::span<const float>(cf.ptr<float>(), static_cast<size_t>(cf.numel())),
        N, C, S);
}

/// Logical [N,C,S] copy of an NHWC view.
std::vector<float> to_channel_first_data(const TensorView& cl)
{
    const int64_t N = cl.shape(0);
    const int64_t C = cl.shape(cl.rank() - 1);
    const int64_t S = cl.numel() / (N * C);
    return test::to_channel_first(
        std::span<const float>(cl.ptr<float>(), static_cast<size_t>(cl.numel())),
        N, C, S);
}

}  // namespace

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

    TensorView in_channel_first(ishape, DataType::f32, in_data, TensorLayout::Dense);
    TensorView grid(gshape, DataType::f32, grid_data, TensorLayout::Dense);

    {
        std::vector<float> cl_buf = to_channel_last_data(in_channel_first);
        const std::vector<int64_t> in_cl_dims = cl_dims_of(in_channel_first);
        TensorView in_cl(in_cl_dims, DataType::f32, cl_buf.data(),
                         TensorLayout::NHWC);

        auto gs_op = GridSample::create({});
        auto d_in = in_cl.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NHWC);
        NNOPS_EXPECT_EQ(gs_descs[0].dtype, DataType::f32);

        std::vector<float> gs_buf(static_cast<size_t>(gs_descs[0].numel()));
        TensorView out_cl(
            std::span<const int64_t>(gs_descs[0].dims.data(),
                                     static_cast<size_t>(gs_descs[0].rank)),
            DataType::f32, gs_buf.data(), gs_descs[0].layout);
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        std::vector<float> res_buf = to_channel_first_data(out_cl);
        const std::vector<int64_t> res_channel_first_dims = dense_dims_of(out_cl);
        TensorView res_channel_first(res_channel_first_dims, DataType::f32, res_buf.data(),
                                     TensorLayout::Dense);

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

    TensorView in_channel_first(ishape, DataType::f32, in_data, TensorLayout::Dense);

    std::vector<float> grid_buf(1 * 4 * 4 * 2);
    for (int64_t oh = 0; oh < 4; ++oh) {
        for (int64_t ow = 0; ow < 4; ++ow) {
            float gy = static_cast<float>(oh) / 3.0f * 2.0f - 1.0f;
            float gx = static_cast<float>(ow) / 3.0f * 2.0f - 1.0f;
            grid_buf[(oh * 4 + ow) * 2]     = gy;
            grid_buf[(oh * 4 + ow) * 2 + 1] = gx;
        }
    }
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::Dense);

    {
        std::vector<float> cl_buf = to_channel_last_data(in_channel_first);
        const std::vector<int64_t> in_cl_dims = cl_dims_of(in_channel_first);
        TensorView in_cl(in_cl_dims, DataType::f32, cl_buf.data(),
                         TensorLayout::NHWC);

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
        // [N, OH, OW, C]
        NNOPS_EXPECT_EQ(gs_descs[0].dims[0], 1);
        NNOPS_EXPECT_EQ(gs_descs[0].dims[1], 4);
        NNOPS_EXPECT_EQ(gs_descs[0].dims[2], 4);
        NNOPS_EXPECT_EQ(gs_descs[0].dims[3], 1);

        std::vector<float> gs_buf(static_cast<size_t>(gs_descs[0].numel()));
        TensorView out_cl(
            std::span<const int64_t>(gs_descs[0].dims.data(),
                                     static_cast<size_t>(gs_descs[0].rank)),
            DataType::f32, gs_buf.data(), gs_descs[0].layout);
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        std::vector<float> res_buf = to_channel_first_data(out_cl);
        const std::vector<int64_t> res_channel_first_dims = dense_dims_of(out_cl);
        TensorView res_channel_first(res_channel_first_dims, DataType::f32, res_buf.data(),
                                     TensorLayout::Dense);

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
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::Dense);

    {
        std::vector<float> cl_buf = to_channel_last_data(in_channel_first);
        const std::vector<int64_t> in_cl_dims = cl_dims_of(in_channel_first);
        TensorView in_cl(in_cl_dims, DataType::f32, cl_buf.data(),
                         TensorLayout::NHWC);

        auto gs_op = GridSample::create({});
        auto d_in = in_cl.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NHWC);

        std::vector<float> gs_buf(static_cast<size_t>(gs_descs[0].numel()));
        TensorView out_cl(
            std::span<const int64_t>(gs_descs[0].dims.data(),
                                     static_cast<size_t>(gs_descs[0].rank)),
            DataType::f32, gs_buf.data(), gs_descs[0].layout);
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        std::vector<float> res_buf = to_channel_first_data(out_cl);
        const std::vector<int64_t> res_channel_first_dims = dense_dims_of(out_cl);
        TensorView res_channel_first(res_channel_first_dims, DataType::f32, res_buf.data(),
                                     TensorLayout::Dense);

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

    TensorView in_channel_first(ishape, DataType::f32, in_data, TensorLayout::Dense);
    TensorView grid(gshape, DataType::f32, grid_data, TensorLayout::Dense);

    {
        std::vector<float> cl_buf = to_channel_last_data(in_channel_first);
        const std::vector<int64_t> in_cl_dims = cl_dims_of(in_channel_first);
        TensorView in_cl(in_cl_dims, DataType::f32, cl_buf.data(),
                         TensorLayout::NHWC);

        GridSampleAttributes attrs;
        attrs.mode = GridSampleMode::Nearest;
        auto gs_op = GridSample::create(attrs);
        auto d_in = in_cl.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NHWC);

        std::vector<float> gs_buf(static_cast<size_t>(gs_descs[0].numel()));
        TensorView out_cl(
            std::span<const int64_t>(gs_descs[0].dims.data(),
                                     static_cast<size_t>(gs_descs[0].rank)),
            DataType::f32, gs_buf.data(), gs_descs[0].layout);
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        std::vector<float> res_buf = to_channel_first_data(out_cl);
        const std::vector<int64_t> res_channel_first_dims = dense_dims_of(out_cl);
        TensorView res_channel_first(res_channel_first_dims, DataType::f32, res_buf.data(),
                                     TensorLayout::Dense);

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
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::Dense);

    {
        std::vector<float> cl_buf = to_channel_last_data(in_channel_first);
        const std::vector<int64_t> in_cl_dims = cl_dims_of(in_channel_first);
        TensorView in_cl(in_cl_dims, DataType::f32, cl_buf.data(),
                         TensorLayout::NHWC);

        GridSampleAttributes attrs;
        attrs.mode = GridSampleMode::Nearest;
        auto gs_op = GridSample::create(attrs);
        auto d_in = in_cl.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NHWC);

        std::vector<float> gs_buf(static_cast<size_t>(gs_descs[0].numel()));
        TensorView out_cl(
            std::span<const int64_t>(gs_descs[0].dims.data(),
                                     static_cast<size_t>(gs_descs[0].rank)),
            DataType::f32, gs_buf.data(), gs_descs[0].layout);
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        std::vector<float> res_buf = to_channel_first_data(out_cl);
        const std::vector<int64_t> res_channel_first_dims = dense_dims_of(out_cl);
        TensorView res_channel_first(res_channel_first_dims, DataType::f32, res_buf.data(),
                                     TensorLayout::Dense);

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

    TensorView in_channel_first(ishape, DataType::f32, in_data, TensorLayout::Dense);
    TensorView grid(gshape, DataType::f32, grid_data, TensorLayout::Dense);

    {
        std::vector<float> cl_buf = to_channel_last_data(in_channel_first);
        const std::vector<int64_t> in_cl_dims = cl_dims_of(in_channel_first);
        TensorView in_cl(in_cl_dims, DataType::f32, cl_buf.data(),
                         TensorLayout::NHWC);

        auto gs_op = GridSample::create({});
        auto d_in = in_cl.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NHWC);

        std::vector<float> gs_buf(static_cast<size_t>(gs_descs[0].numel()));
        TensorView out_cl(
            std::span<const int64_t>(gs_descs[0].dims.data(),
                                     static_cast<size_t>(gs_descs[0].rank)),
            DataType::f32, gs_buf.data(), gs_descs[0].layout);
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        std::vector<float> res_buf = to_channel_first_data(out_cl);
        const std::vector<int64_t> res_channel_first_dims = dense_dims_of(out_cl);
        TensorView res_channel_first(res_channel_first_dims, DataType::f32, res_buf.data(),
                                     TensorLayout::Dense);

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

    TensorView in_channel_first(ishape, DataType::f32, in_data, TensorLayout::Dense);
    TensorView grid(gshape, DataType::f32, grid_data, TensorLayout::Dense);

    {
        std::vector<float> cl_buf = to_channel_last_data(in_channel_first);
        const std::vector<int64_t> in_cl_dims = cl_dims_of(in_channel_first);
        TensorView in_cl(in_cl_dims, DataType::f32, cl_buf.data(),
                         TensorLayout::NHWC);

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

        std::vector<float> gs_buf(static_cast<size_t>(gs_descs[0].numel()));
        TensorView out_cl(
            std::span<const int64_t>(gs_descs[0].dims.data(),
                                     static_cast<size_t>(gs_descs[0].rank)),
            DataType::f32, gs_buf.data(), gs_descs[0].layout);
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        std::vector<float> res_buf = to_channel_first_data(out_cl);
        const std::vector<int64_t> res_channel_first_dims = dense_dims_of(out_cl);
        TensorView res_channel_first(res_channel_first_dims, DataType::f32, res_buf.data(),
                                     TensorLayout::Dense);

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
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::Dense);

    {
        std::vector<float> cl_buf = to_channel_last_data(in_channel_first);
        const std::vector<int64_t> in_cl_dims = cl_dims_of(in_channel_first);
        TensorView in_cl(in_cl_dims, DataType::f32, cl_buf.data(),
                         TensorLayout::NHWC);

        GridSampleAttributes attrs;
        attrs.padding_mode = GridSamplePaddingMode::Reflection;
        auto gs_op = GridSample::create(attrs);
        auto d_in = in_cl.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NHWC);

        std::vector<float> gs_buf(static_cast<size_t>(gs_descs[0].numel()));
        TensorView out_cl(
            std::span<const int64_t>(gs_descs[0].dims.data(),
                                     static_cast<size_t>(gs_descs[0].rank)),
            DataType::f32, gs_buf.data(), gs_descs[0].layout);
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        std::vector<float> res_buf = to_channel_first_data(out_cl);
        const std::vector<int64_t> res_channel_first_dims = dense_dims_of(out_cl);
        TensorView res_channel_first(res_channel_first_dims, DataType::f32, res_buf.data(),
                                     TensorLayout::Dense);

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

    TensorView in_ncdhw(ishape, DataType::f32, in_buf.data(), TensorLayout::Dense);

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
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::Dense);

    {
        std::vector<float> cl_buf = to_channel_last_data(in_ncdhw);
        const std::vector<int64_t> in_cl_dims = cl_dims_of(in_ncdhw);
        TensorView in_cl(in_cl_dims, DataType::f32, cl_buf.data(),
                         TensorLayout::NDHWC);

        auto gs_op = GridSample::create({});
        auto d_in = in_cl.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 5);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NDHWC);
        NNOPS_EXPECT_EQ(gs_descs[0].dtype, DataType::f32);

        std::vector<float> gs_buf(static_cast<size_t>(gs_descs[0].numel()));
        TensorView out_cl(
            std::span<const int64_t>(gs_descs[0].dims.data(),
                                     static_cast<size_t>(gs_descs[0].rank)),
            DataType::f32, gs_buf.data(), gs_descs[0].layout);
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        std::vector<float> res_buf = to_channel_first_data(out_cl);
        const std::vector<int64_t> res_ncdhw_dims = dense_dims_of(out_cl);
        TensorView res_ncdhw(res_ncdhw_dims, DataType::f32, res_buf.data(),
                             TensorLayout::Dense);

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
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::Dense);

    {
        std::vector<float> cl_buf = to_channel_last_data(in_ncdhw);
        const std::vector<int64_t> in_cl_dims = cl_dims_of(in_ncdhw);
        TensorView in_cl(in_cl_dims, DataType::f32, cl_buf.data(),
                         TensorLayout::NDHWC);

        GridSampleAttributes attrs;
        attrs.mode = GridSampleMode::Nearest;
        auto gs_op = GridSample::create(attrs);
        auto d_in = in_cl.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 5);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NDHWC);

        std::vector<float> gs_buf(static_cast<size_t>(gs_descs[0].numel()));
        TensorView out_cl(
            std::span<const int64_t>(gs_descs[0].dims.data(),
                                     static_cast<size_t>(gs_descs[0].rank)),
            DataType::f32, gs_buf.data(), gs_descs[0].layout);
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        std::vector<float> res_buf = to_channel_first_data(out_cl);
        const std::vector<int64_t> res_ncdhw_dims = dense_dims_of(out_cl);
        TensorView res_ncdhw(res_ncdhw_dims, DataType::f32, res_buf.data(),
                             TensorLayout::Dense);

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
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::Dense);

    {
        std::vector<float> cl_buf = to_channel_last_data(in_channel_first);
        const std::vector<int64_t> in_cl_dims = cl_dims_of(in_channel_first);
        TensorView in_cl(in_cl_dims, DataType::f32, cl_buf.data(),
                         TensorLayout::NHWC);

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

        std::vector<float> gs_buf(static_cast<size_t>(gs_descs[0].numel()));
        TensorView out_cl(
            std::span<const int64_t>(gs_descs[0].dims.data(),
                                     static_cast<size_t>(gs_descs[0].rank)),
            DataType::f32, gs_buf.data(), gs_descs[0].layout);

        std::fill(gs_buf.begin(), gs_buf.end(), 0.5f);

        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        std::vector<float> res_buf = to_channel_first_data(out_cl);
        const std::vector<int64_t> res_channel_first_dims = dense_dims_of(out_cl);
        TensorView res_channel_first(res_channel_first_dims, DataType::f32, res_buf.data(),
                                     TensorLayout::Dense);

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
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::Dense);

    {
        std::vector<float> cl_buf = to_channel_last_data(in_channel_first);
        const std::vector<int64_t> in_cl_dims = cl_dims_of(in_channel_first);
        TensorView in_cl(in_cl_dims, DataType::f32, cl_buf.data(),
                         TensorLayout::NHWC);

        auto gs_op = GridSample::create({});
        auto d_in = in_cl.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NHWC);
        NNOPS_EXPECT_EQ(gs_descs[0].dtype, DataType::f32);
        // [N, OH, OW, C]
        NNOPS_EXPECT_EQ(gs_descs[0].dims[0], 2);
        NNOPS_EXPECT_EQ(gs_descs[0].dims[1], 3);
        NNOPS_EXPECT_EQ(gs_descs[0].dims[2], 3);
        NNOPS_EXPECT_EQ(gs_descs[0].dims[3], 1);

        std::vector<float> gs_buf(static_cast<size_t>(gs_descs[0].numel()));
        TensorView out_cl(
            std::span<const int64_t>(gs_descs[0].dims.data(),
                                     static_cast<size_t>(gs_descs[0].rank)),
            DataType::f32, gs_buf.data(), gs_descs[0].layout);
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        std::vector<float> res_buf = to_channel_first_data(out_cl);
        const std::vector<int64_t> res_channel_first_dims = dense_dims_of(out_cl);
        TensorView res_channel_first(res_channel_first_dims, DataType::f32, res_buf.data(),
                                     TensorLayout::Dense);

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

    TensorView in_channel_first(ishape, DataType::f32, in_data, TensorLayout::Dense);

    std::vector<float> grid_buf(1 * 3 * 3 * 2);
    for (int64_t oh = 0; oh < 3; ++oh) {
        for (int64_t ow = 0; ow < 3; ++ow) {
            float gy = static_cast<float>(oh) / 1.0f - 1.0f;
            float gx = static_cast<float>(ow) / 1.0f - 1.0f;
            grid_buf[(oh * 3 + ow) * 2]     = gy;
            grid_buf[(oh * 3 + ow) * 2 + 1] = gx;
        }
    }
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::Dense);

    {
        std::vector<float> cl_buf = to_channel_last_data(in_channel_first);
        const std::vector<int64_t> in_cl_dims = cl_dims_of(in_channel_first);
        TensorView in_cl(in_cl_dims, DataType::f32, cl_buf.data(),
                         TensorLayout::NHWC);

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

        std::vector<float> gs_buf(static_cast<size_t>(gs_descs[0].numel()));
        TensorView out_cl(
            std::span<const int64_t>(gs_descs[0].dims.data(),
                                     static_cast<size_t>(gs_descs[0].rank)),
            DataType::f32, gs_buf.data(), gs_descs[0].layout);
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        std::vector<float> res_buf = to_channel_first_data(out_cl);
        const std::vector<int64_t> res_channel_first_dims = dense_dims_of(out_cl);
        TensorView res_channel_first(res_channel_first_dims, DataType::f32, res_buf.data(),
                                     TensorLayout::Dense);

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
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::Dense);

    {
        std::vector<float> cl_buf = to_channel_last_data(in_channel_first);
        const std::vector<int64_t> in_cl_dims = cl_dims_of(in_channel_first);
        TensorView in_cl(in_cl_dims, DataType::f32, cl_buf.data(),
                         TensorLayout::NHWC);

        auto gs_op = GridSample::create({});
        auto d_in = in_cl.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NHWC);
        NNOPS_EXPECT_EQ(gs_descs[0].dtype, DataType::f32);

        std::vector<float> gs_buf(static_cast<size_t>(gs_descs[0].numel()));
        TensorView out_cl(
            std::span<const int64_t>(gs_descs[0].dims.data(),
                                     static_cast<size_t>(gs_descs[0].rank)),
            DataType::f32, gs_buf.data(), gs_descs[0].layout);
        const TensorView gs_ins[] = {in_cl, grid};
        gs_op->compute(out_cl, gs_ins);

        std::vector<float> res_buf = to_channel_first_data(out_cl);
        const std::vector<int64_t> res_channel_first_dims = dense_dims_of(out_cl);
        TensorView res_channel_first(res_channel_first_dims, DataType::f32, res_buf.data(),
                                     TensorLayout::Dense);

        for (size_t i = 0; i < 48; ++i) {
            NNOPS_EXPECT_TRUE(!std::isnan(res_channel_first.ptr<float>()[i]));
            NNOPS_EXPECT_TRUE(!std::isinf(res_channel_first.ptr<float>()[i]));
        }
    }
}
