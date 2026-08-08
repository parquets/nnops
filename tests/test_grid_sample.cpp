/// @file test_grid_sample.cpp
/// @brief Tests for the GridSample operator — NCHWC8/NCDHWC8 only.
///
/// All tests use the class API with LayoutConvert for pack/unpack:
///   NCHW in → LayoutConvert(pack) → NCHWC8 → GridSample::compute → LayoutConvert(unpack) → NCHW → compare
///
/// Covers bilinear and nearest-neighbor interpolation with zeros, border,
/// and reflection padding modes.

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

    TensorView in_nchw(ishape, DataType::f32, in_data, TensorLayout::NCHW);
    TensorView grid(gshape, DataType::f32, grid_data, TensorLayout::NCHW);

    // Pack input to NCHWC8
    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8);
    {
        auto d = in_nchw.desc();
        const TensorDesc pack_in_arr[] = {d};
        auto pack_descs = pack_op->getOutputTensorDesc(pack_in_arr);
        NNOPS_EXPECT_EQ(pack_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(pack_descs[0].layout, TensorLayout::NCHWC8);
        NNOPS_EXPECT_EQ(pack_descs[0].dtype, DataType::f32);

        std::vector<char> packed_buf(pack_descs[0].storage_bytes());
        auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
        const TensorView pack_ins[] = {in_nchw};
        pack_op->compute(in_c8, pack_ins);

        // GridSample on NCHWC8
        auto gs_op = GridSample::create({});
        auto d_in = in_c8.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NCHWC8);
        NNOPS_EXPECT_EQ(gs_descs[0].dtype, DataType::f32);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_c8 = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_c8, grid};
        gs_op->compute(out_c8, gs_ins);

        // Unpack to NCHW
        auto unpack_op = LayoutConvert::create(TensorLayout::NCHW);
        auto d_out = out_c8.desc();
        const TensorDesc unpack_in_arr[] = {d_out};
        auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in_arr);
        NNOPS_EXPECT_EQ(unpack_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(unpack_descs[0].layout, TensorLayout::NCHW);
        NNOPS_EXPECT_EQ(unpack_descs[0].dtype, DataType::f32);

        std::vector<char> unpack_buf(unpack_descs[0].storage_bytes());
        auto res_nchw = test::make_planar(unpack_descs[0], unpack_buf.data());
        const TensorView unpack_ins[] = {out_c8};
        unpack_op->compute(res_nchw, unpack_ins);

        NNOPS_EXPECT_NEAR(res_nchw.ptr<float>()[0], 1.0f, 1e-5f);
        NNOPS_EXPECT_NEAR(res_nchw.ptr<float>()[1], 2.0f, 1e-5f);
        NNOPS_EXPECT_NEAR(res_nchw.ptr<float>()[2], 3.0f, 1e-5f);
        NNOPS_EXPECT_NEAR(res_nchw.ptr<float>()[3], 4.0f, 1e-5f);
    }
}

NNOPS_TEST(grid_sample_2d_bilinear_align_corners) {
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t gshape[] = {1, 4, 4, 2};

    float in_data[4] = {1, 2, 3, 4};

    TensorView in_nchw(ishape, DataType::f32, in_data, TensorLayout::NCHW);

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

    // Pack input to NCHWC8
    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8);
    {
        auto d = in_nchw.desc();
        const TensorDesc pack_in_arr[] = {d};
        auto pack_descs = pack_op->getOutputTensorDesc(pack_in_arr);
        NNOPS_EXPECT_EQ(pack_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(pack_descs[0].layout, TensorLayout::NCHWC8);

        std::vector<char> packed_buf(pack_descs[0].storage_bytes());
        auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
        const TensorView pack_ins[] = {in_nchw};
        pack_op->compute(in_c8, pack_ins);

        // GridSample on NCHWC8
        GridSampleAttributes attrs;
        attrs.align_corners = true;
        auto gs_op = GridSample::create(attrs);
        auto d_in = in_c8.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NCHWC8);
        NNOPS_EXPECT_EQ(gs_descs[0].dtype, DataType::f32);
        NNOPS_EXPECT_EQ(gs_descs[0].dims[0], 1);
        NNOPS_EXPECT_EQ(gs_descs[0].dims[1], 1);
        NNOPS_EXPECT_EQ(gs_descs[0].dims[2], 4);
        NNOPS_EXPECT_EQ(gs_descs[0].dims[3], 4);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_c8 = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_c8, grid};
        gs_op->compute(out_c8, gs_ins);

        // Unpack to NCHW
        auto unpack_op = LayoutConvert::create(TensorLayout::NCHW);
        auto d_out = out_c8.desc();
        const TensorDesc unpack_in_arr[] = {d_out};
        auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in_arr);
        NNOPS_EXPECT_EQ(unpack_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(unpack_descs[0].layout, TensorLayout::NCHW);

        std::vector<char> unpack_buf(unpack_descs[0].storage_bytes());
        auto res_nchw = test::make_planar(unpack_descs[0], unpack_buf.data());
        const TensorView unpack_ins[] = {out_c8};
        unpack_op->compute(res_nchw, unpack_ins);

        NNOPS_EXPECT_NEAR(res_nchw.ptr<float>()[0],  1.0f, 1e-5f);
        NNOPS_EXPECT_NEAR(res_nchw.ptr<float>()[3],  2.0f, 1e-5f);
        NNOPS_EXPECT_NEAR(res_nchw.ptr<float>()[12], 3.0f, 1e-5f);
        NNOPS_EXPECT_NEAR(res_nchw.ptr<float>()[15], 4.0f, 1e-5f);

        for (int64_t i = 0; i < 16; ++i) {
            NNOPS_EXPECT_TRUE(!std::isnan(res_nchw.ptr<float>()[i]));
            NNOPS_EXPECT_TRUE(!std::isinf(res_nchw.ptr<float>()[i]));
        }
    }
}

NNOPS_TEST(grid_sample_2d_bilinear_random) {
    auto [in_vec, in_nchw] = test::make_random_tensor({1, 2, 8, 8});
    const int64_t gshape[] = {1, 6, 6, 2};

    std::vector<float> grid_buf(1 * 6 * 6 * 2);
    for (size_t i = 0; i < grid_buf.size(); ++i) {
        grid_buf[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::NCHW);

    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8);
    {
        auto d = in_nchw.desc();
        const TensorDesc pack_in_arr[] = {d};
        auto pack_descs = pack_op->getOutputTensorDesc(pack_in_arr);
        NNOPS_EXPECT_EQ(pack_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(pack_descs[0].layout, TensorLayout::NCHWC8);

        std::vector<char> packed_buf(pack_descs[0].storage_bytes());
        auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
        const TensorView pack_ins[] = {in_nchw};
        pack_op->compute(in_c8, pack_ins);

        auto gs_op = GridSample::create({});
        auto d_in = in_c8.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NCHWC8);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_c8 = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_c8, grid};
        gs_op->compute(out_c8, gs_ins);

        auto unpack_op = LayoutConvert::create(TensorLayout::NCHW);
        auto d_out = out_c8.desc();
        const TensorDesc unpack_in_arr[] = {d_out};
        auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in_arr);
        NNOPS_EXPECT_EQ(unpack_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(unpack_descs[0].layout, TensorLayout::NCHW);

        std::vector<char> unpack_buf(unpack_descs[0].storage_bytes());
        auto res_nchw = test::make_planar(unpack_descs[0], unpack_buf.data());
        const TensorView unpack_ins[] = {out_c8};
        unpack_op->compute(res_nchw, unpack_ins);

        for (size_t i = 0; i < 72; ++i) {
            NNOPS_EXPECT_TRUE(!std::isnan(res_nchw.ptr<float>()[i]));
            NNOPS_EXPECT_TRUE(!std::isinf(res_nchw.ptr<float>()[i]));
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

    TensorView in_nchw(ishape, DataType::f32, in_data, TensorLayout::NCHW);
    TensorView grid(gshape, DataType::f32, grid_data, TensorLayout::NCHW);

    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8);
    {
        auto d = in_nchw.desc();
        const TensorDesc pack_in_arr[] = {d};
        auto pack_descs = pack_op->getOutputTensorDesc(pack_in_arr);
        NNOPS_EXPECT_EQ(pack_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(pack_descs[0].layout, TensorLayout::NCHWC8);

        std::vector<char> packed_buf(pack_descs[0].storage_bytes());
        auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
        const TensorView pack_ins[] = {in_nchw};
        pack_op->compute(in_c8, pack_ins);

        GridSampleAttributes attrs;
        attrs.mode = GridSampleMode::Nearest;
        auto gs_op = GridSample::create(attrs);
        auto d_in = in_c8.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NCHWC8);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_c8 = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_c8, grid};
        gs_op->compute(out_c8, gs_ins);

        auto unpack_op = LayoutConvert::create(TensorLayout::NCHW);
        auto d_out = out_c8.desc();
        const TensorDesc unpack_in_arr[] = {d_out};
        auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in_arr);
        NNOPS_EXPECT_EQ(unpack_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(unpack_descs[0].layout, TensorLayout::NCHW);

        std::vector<char> unpack_buf(unpack_descs[0].storage_bytes());
        auto res_nchw = test::make_planar(unpack_descs[0], unpack_buf.data());
        const TensorView unpack_ins[] = {out_c8};
        unpack_op->compute(res_nchw, unpack_ins);

        for (int i = 0; i < 4; ++i) {
            NNOPS_EXPECT_NEAR(res_nchw.ptr<float>()[i], 4.0f, 1e-5f);
        }
    }
}

NNOPS_TEST(grid_sample_2d_nearest_random) {
    auto [in_vec, in_nchw] = test::make_random_tensor({1, 3, 10, 10});
    const int64_t gshape[] = {1, 5, 5, 2};

    std::vector<float> grid_buf(1 * 5 * 5 * 2);
    for (size_t i = 0; i < grid_buf.size(); ++i) {
        grid_buf[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::NCHW);

    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8);
    {
        auto d = in_nchw.desc();
        const TensorDesc pack_in_arr[] = {d};
        auto pack_descs = pack_op->getOutputTensorDesc(pack_in_arr);
        NNOPS_EXPECT_EQ(pack_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(pack_descs[0].layout, TensorLayout::NCHWC8);

        std::vector<char> packed_buf(pack_descs[0].storage_bytes());
        auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
        const TensorView pack_ins[] = {in_nchw};
        pack_op->compute(in_c8, pack_ins);

        GridSampleAttributes attrs;
        attrs.mode = GridSampleMode::Nearest;
        auto gs_op = GridSample::create(attrs);
        auto d_in = in_c8.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NCHWC8);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_c8 = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_c8, grid};
        gs_op->compute(out_c8, gs_ins);

        auto unpack_op = LayoutConvert::create(TensorLayout::NCHW);
        auto d_out = out_c8.desc();
        const TensorDesc unpack_in_arr[] = {d_out};
        auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in_arr);
        NNOPS_EXPECT_EQ(unpack_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(unpack_descs[0].layout, TensorLayout::NCHW);

        std::vector<char> unpack_buf(unpack_descs[0].storage_bytes());
        auto res_nchw = test::make_planar(unpack_descs[0], unpack_buf.data());
        const TensorView unpack_ins[] = {out_c8};
        unpack_op->compute(res_nchw, unpack_ins);

        for (size_t i = 0; i < 75; ++i) {
            NNOPS_EXPECT_TRUE(!std::isnan(res_nchw.ptr<float>()[i]));
            NNOPS_EXPECT_TRUE(!std::isinf(res_nchw.ptr<float>()[i]));
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

    TensorView in_nchw(ishape, DataType::f32, in_data, TensorLayout::NCHW);
    TensorView grid(gshape, DataType::f32, grid_data, TensorLayout::NCHW);

    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8);
    {
        auto d = in_nchw.desc();
        const TensorDesc pack_in_arr[] = {d};
        auto pack_descs = pack_op->getOutputTensorDesc(pack_in_arr);
        NNOPS_EXPECT_EQ(pack_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(pack_descs[0].layout, TensorLayout::NCHWC8);

        std::vector<char> packed_buf(pack_descs[0].storage_bytes());
        auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
        const TensorView pack_ins[] = {in_nchw};
        pack_op->compute(in_c8, pack_ins);

        auto gs_op = GridSample::create({});
        auto d_in = in_c8.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NCHWC8);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_c8 = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_c8, grid};
        gs_op->compute(out_c8, gs_ins);

        auto unpack_op = LayoutConvert::create(TensorLayout::NCHW);
        auto d_out = out_c8.desc();
        const TensorDesc unpack_in_arr[] = {d_out};
        auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in_arr);
        NNOPS_EXPECT_EQ(unpack_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(unpack_descs[0].layout, TensorLayout::NCHW);

        std::vector<char> unpack_buf(unpack_descs[0].storage_bytes());
        auto res_nchw = test::make_planar(unpack_descs[0], unpack_buf.data());
        const TensorView unpack_ins[] = {out_c8};
        unpack_op->compute(res_nchw, unpack_ins);

        for (int i = 0; i < 4; ++i) {
            NNOPS_EXPECT_NEAR(res_nchw.ptr<float>()[i], 0.0f, 1e-5f);
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

    TensorView in_nchw(ishape, DataType::f32, in_data, TensorLayout::NCHW);
    TensorView grid(gshape, DataType::f32, grid_data, TensorLayout::NCHW);

    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8);
    {
        auto d = in_nchw.desc();
        const TensorDesc pack_in_arr[] = {d};
        auto pack_descs = pack_op->getOutputTensorDesc(pack_in_arr);
        NNOPS_EXPECT_EQ(pack_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(pack_descs[0].layout, TensorLayout::NCHWC8);

        std::vector<char> packed_buf(pack_descs[0].storage_bytes());
        auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
        const TensorView pack_ins[] = {in_nchw};
        pack_op->compute(in_c8, pack_ins);

        GridSampleAttributes attrs;
        attrs.padding_mode = GridSamplePaddingMode::Border;
        auto gs_op = GridSample::create(attrs);
        auto d_in = in_c8.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NCHWC8);
        NNOPS_EXPECT_EQ(gs_descs[0].dtype, DataType::f32);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_c8 = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_c8, grid};
        gs_op->compute(out_c8, gs_ins);

        auto unpack_op = LayoutConvert::create(TensorLayout::NCHW);
        auto d_out = out_c8.desc();
        const TensorDesc unpack_in_arr[] = {d_out};
        auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in_arr);
        NNOPS_EXPECT_EQ(unpack_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(unpack_descs[0].layout, TensorLayout::NCHW);

        std::vector<char> unpack_buf(unpack_descs[0].storage_bytes());
        auto res_nchw = test::make_planar(unpack_descs[0], unpack_buf.data());
        const TensorView unpack_ins[] = {out_c8};
        unpack_op->compute(res_nchw, unpack_ins);

        for (int i = 0; i < 4; ++i) {
            NNOPS_EXPECT_TRUE(!std::isnan(res_nchw.ptr<float>()[i]));
            NNOPS_EXPECT_TRUE(!std::isinf(res_nchw.ptr<float>()[i]));
        }
    }
}

NNOPS_TEST(grid_sample_2d_reflection_padding) {
    auto [in_vec, in_nchw] = test::make_random_tensor({1, 1, 4, 4});
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

    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8);
    {
        auto d = in_nchw.desc();
        const TensorDesc pack_in_arr[] = {d};
        auto pack_descs = pack_op->getOutputTensorDesc(pack_in_arr);
        NNOPS_EXPECT_EQ(pack_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(pack_descs[0].layout, TensorLayout::NCHWC8);

        std::vector<char> packed_buf(pack_descs[0].storage_bytes());
        auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
        const TensorView pack_ins[] = {in_nchw};
        pack_op->compute(in_c8, pack_ins);

        GridSampleAttributes attrs;
        attrs.padding_mode = GridSamplePaddingMode::Reflection;
        auto gs_op = GridSample::create(attrs);
        auto d_in = in_c8.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NCHWC8);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_c8 = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_c8, grid};
        gs_op->compute(out_c8, gs_ins);

        auto unpack_op = LayoutConvert::create(TensorLayout::NCHW);
        auto d_out = out_c8.desc();
        const TensorDesc unpack_in_arr[] = {d_out};
        auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in_arr);
        NNOPS_EXPECT_EQ(unpack_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(unpack_descs[0].layout, TensorLayout::NCHW);

        std::vector<char> unpack_buf(unpack_descs[0].storage_bytes());
        auto res_nchw = test::make_planar(unpack_descs[0], unpack_buf.data());
        const TensorView unpack_ins[] = {out_c8};
        unpack_op->compute(res_nchw, unpack_ins);

        for (size_t i = 0; i < 16; ++i) {
            NNOPS_EXPECT_TRUE(!std::isnan(res_nchw.ptr<float>()[i]));
            NNOPS_EXPECT_TRUE(!std::isinf(res_nchw.ptr<float>()[i]));
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

    // Pack input to NCDHWC8
    auto pack_op = LayoutConvert::create(TensorLayout::NCDHWC8);
    {
        auto d = in_ncdhw.desc();
        const TensorDesc pack_in_arr[] = {d};
        auto pack_descs = pack_op->getOutputTensorDesc(pack_in_arr);
        NNOPS_EXPECT_EQ(pack_descs[0].rank, 5);
        NNOPS_EXPECT_EQ(pack_descs[0].layout, TensorLayout::NCDHWC8);
        NNOPS_EXPECT_EQ(pack_descs[0].dtype, DataType::f32);

        std::vector<char> packed_buf(pack_descs[0].storage_bytes());
        auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
        const TensorView pack_ins[] = {in_ncdhw};
        pack_op->compute(in_c8, pack_ins);

        // GridSample on NCDHWC8
        auto gs_op = GridSample::create({});
        auto d_in = in_c8.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 5);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NCDHWC8);
        NNOPS_EXPECT_EQ(gs_descs[0].dtype, DataType::f32);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_c8 = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_c8, grid};
        gs_op->compute(out_c8, gs_ins);

        // Unpack to NCDHW
        auto unpack_op = LayoutConvert::create(TensorLayout::NCDHW);
        auto d_out = out_c8.desc();
        const TensorDesc unpack_in_arr[] = {d_out};
        auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in_arr);
        NNOPS_EXPECT_EQ(unpack_descs[0].rank, 5);
        NNOPS_EXPECT_EQ(unpack_descs[0].layout, TensorLayout::NCDHW);
        NNOPS_EXPECT_EQ(unpack_descs[0].dtype, DataType::f32);

        std::vector<char> unpack_buf(unpack_descs[0].storage_bytes());
        auto res_ncdhw = test::make_planar(unpack_descs[0], unpack_buf.data());
        const TensorView unpack_ins[] = {out_c8};
        unpack_op->compute(res_ncdhw, unpack_ins);

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

    auto pack_op = LayoutConvert::create(TensorLayout::NCDHWC8);
    {
        auto d = in_ncdhw.desc();
        const TensorDesc pack_in_arr[] = {d};
        auto pack_descs = pack_op->getOutputTensorDesc(pack_in_arr);
        NNOPS_EXPECT_EQ(pack_descs[0].rank, 5);
        NNOPS_EXPECT_EQ(pack_descs[0].layout, TensorLayout::NCDHWC8);

        std::vector<char> packed_buf(pack_descs[0].storage_bytes());
        auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
        const TensorView pack_ins[] = {in_ncdhw};
        pack_op->compute(in_c8, pack_ins);

        GridSampleAttributes attrs;
        attrs.mode = GridSampleMode::Nearest;
        auto gs_op = GridSample::create(attrs);
        auto d_in = in_c8.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 5);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NCDHWC8);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_c8 = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_c8, grid};
        gs_op->compute(out_c8, gs_ins);

        auto unpack_op = LayoutConvert::create(TensorLayout::NCDHW);
        auto d_out = out_c8.desc();
        const TensorDesc unpack_in_arr[] = {d_out};
        auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in_arr);
        NNOPS_EXPECT_EQ(unpack_descs[0].rank, 5);
        NNOPS_EXPECT_EQ(unpack_descs[0].layout, TensorLayout::NCDHW);

        std::vector<char> unpack_buf(unpack_descs[0].storage_bytes());
        auto res_ncdhw = test::make_planar(unpack_descs[0], unpack_buf.data());
        const TensorView unpack_ins[] = {out_c8};
        unpack_op->compute(res_ncdhw, unpack_ins);

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
    auto [in_vec, in_nchw] = test::make_random_tensor({1, 1, 4, 4});
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

    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8);
    {
        auto d = in_nchw.desc();
        const TensorDesc pack_in_arr[] = {d};
        auto pack_descs = pack_op->getOutputTensorDesc(pack_in_arr);
        NNOPS_EXPECT_EQ(pack_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(pack_descs[0].layout, TensorLayout::NCHWC8);

        std::vector<char> packed_buf(pack_descs[0].storage_bytes());
        auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
        const TensorView pack_ins[] = {in_nchw};
        pack_op->compute(in_c8, pack_ins);

        GridSampleAttributes attrs;
        attrs.add_to = true;
        auto gs_op = GridSample::create(attrs);
        auto d_in = in_c8.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NCHWC8);
        NNOPS_EXPECT_EQ(gs_descs[0].dtype, DataType::f32);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_c8 = test::make_packed(gs_descs[0], gs_buf.data());

        // Pre-fill output with 0.5f for add_to
        int64_t rows = out_c8.total_rows();
        int64_t row_elems = out_c8.row_stride_elems();
        for (int64_t r = 0; r < rows; ++r) {
            float* row = out_c8.ptr<float>(r);
            for (int64_t e = 0; e < row_elems; ++e) {
                row[e] = 0.5f;
            }
        }

        const TensorView gs_ins[] = {in_c8, grid};
        gs_op->compute(out_c8, gs_ins);

        auto unpack_op = LayoutConvert::create(TensorLayout::NCHW);
        auto d_out = out_c8.desc();
        const TensorDesc unpack_in_arr[] = {d_out};
        auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in_arr);
        NNOPS_EXPECT_EQ(unpack_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(unpack_descs[0].layout, TensorLayout::NCHW);

        std::vector<char> unpack_buf(unpack_descs[0].storage_bytes());
        auto res_nchw = test::make_planar(unpack_descs[0], unpack_buf.data());
        const TensorView unpack_ins[] = {out_c8};
        unpack_op->compute(res_nchw, unpack_ins);

        for (size_t i = 0; i < 16; ++i) {
            NNOPS_EXPECT_TRUE(!std::isnan(res_nchw.ptr<float>()[i]));
            NNOPS_EXPECT_TRUE(!std::isinf(res_nchw.ptr<float>()[i]));
        }
    }
}

// ============================================================
// Multi-sample batch test
// ============================================================

NNOPS_TEST(grid_sample_multisample) {
    auto [in_vec, in_nchw] = test::make_random_tensor({2, 1, 4, 4});
    const int64_t gshape[] = {2, 3, 3, 2};

    std::vector<float> grid_buf(2 * 3 * 3 * 2);
    for (size_t i = 0; i < grid_buf.size(); ++i) {
        grid_buf[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::NCHW);

    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8);
    {
        auto d = in_nchw.desc();
        const TensorDesc pack_in_arr[] = {d};
        auto pack_descs = pack_op->getOutputTensorDesc(pack_in_arr);
        NNOPS_EXPECT_EQ(pack_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(pack_descs[0].layout, TensorLayout::NCHWC8);

        std::vector<char> packed_buf(pack_descs[0].storage_bytes());
        auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
        const TensorView pack_ins[] = {in_nchw};
        pack_op->compute(in_c8, pack_ins);

        auto gs_op = GridSample::create({});
        auto d_in = in_c8.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NCHWC8);
        NNOPS_EXPECT_EQ(gs_descs[0].dtype, DataType::f32);
        NNOPS_EXPECT_EQ(gs_descs[0].dims[0], 2);
        NNOPS_EXPECT_EQ(gs_descs[0].dims[1], 1);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_c8 = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_c8, grid};
        gs_op->compute(out_c8, gs_ins);

        auto unpack_op = LayoutConvert::create(TensorLayout::NCHW);
        auto d_out = out_c8.desc();
        const TensorDesc unpack_in_arr[] = {d_out};
        auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in_arr);
        NNOPS_EXPECT_EQ(unpack_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(unpack_descs[0].layout, TensorLayout::NCHW);

        std::vector<char> unpack_buf(unpack_descs[0].storage_bytes());
        auto res_nchw = test::make_planar(unpack_descs[0], unpack_buf.data());
        const TensorView unpack_ins[] = {out_c8};
        unpack_op->compute(res_nchw, unpack_ins);

        for (size_t i = 0; i < 18; ++i) {
            NNOPS_EXPECT_TRUE(!std::isnan(res_nchw.ptr<float>()[i]));
            NNOPS_EXPECT_TRUE(!std::isinf(res_nchw.ptr<float>()[i]));
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

    TensorView in_nchw(ishape, DataType::f32, in_data, TensorLayout::NCHW);

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

    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8);
    {
        auto d = in_nchw.desc();
        const TensorDesc pack_in_arr[] = {d};
        auto pack_descs = pack_op->getOutputTensorDesc(pack_in_arr);
        NNOPS_EXPECT_EQ(pack_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(pack_descs[0].layout, TensorLayout::NCHWC8);

        std::vector<char> packed_buf(pack_descs[0].storage_bytes());
        auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
        const TensorView pack_ins[] = {in_nchw};
        pack_op->compute(in_c8, pack_ins);

        GridSampleAttributes attrs;
        attrs.align_corners = true;
        auto gs_op = GridSample::create(attrs);
        auto d_in = in_c8.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NCHWC8);
        NNOPS_EXPECT_EQ(gs_descs[0].dtype, DataType::f32);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_c8 = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_c8, grid};
        gs_op->compute(out_c8, gs_ins);

        auto unpack_op = LayoutConvert::create(TensorLayout::NCHW);
        auto d_out = out_c8.desc();
        const TensorDesc unpack_in_arr[] = {d_out};
        auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in_arr);
        NNOPS_EXPECT_EQ(unpack_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(unpack_descs[0].layout, TensorLayout::NCHW);

        std::vector<char> unpack_buf(unpack_descs[0].storage_bytes());
        auto res_nchw = test::make_planar(unpack_descs[0], unpack_buf.data());
        const TensorView unpack_ins[] = {out_c8};
        unpack_op->compute(res_nchw, unpack_ins);

        for (int i = 0; i < 9; ++i) {
            NNOPS_EXPECT_NEAR(res_nchw.ptr<float>()[i], in_data[i], 1e-5f);
        }
    }
}

// ============================================================
// Partial C8 tests (zero-padded, all C8 blocks use SIMD)
// ============================================================

NNOPS_TEST(grid_sample_partial_c8) {
    auto [in_vec, in_nchw] = test::make_random_tensor({1, 3, 4, 4});
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

    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8);
    {
        auto d = in_nchw.desc();
        const TensorDesc pack_in_arr[] = {d};
        auto pack_descs = pack_op->getOutputTensorDesc(pack_in_arr);
        NNOPS_EXPECT_EQ(pack_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(pack_descs[0].layout, TensorLayout::NCHWC8);

        std::vector<char> packed_buf(pack_descs[0].storage_bytes());
        auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
        const TensorView pack_ins[] = {in_nchw};
        pack_op->compute(in_c8, pack_ins);

        auto gs_op = GridSample::create({});
        auto d_in = in_c8.desc();
        auto d_grid = grid.desc();
        const TensorDesc gs_in_arr[] = {d_in, d_grid};
        auto gs_descs = gs_op->getOutputTensorDesc(gs_in_arr);
        NNOPS_EXPECT_EQ(gs_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(gs_descs[0].layout, TensorLayout::NCHWC8);
        NNOPS_EXPECT_EQ(gs_descs[0].dtype, DataType::f32);

        std::vector<char> gs_buf(gs_descs[0].storage_bytes());
        auto out_c8 = test::make_packed(gs_descs[0], gs_buf.data());
        const TensorView gs_ins[] = {in_c8, grid};
        gs_op->compute(out_c8, gs_ins);

        auto unpack_op = LayoutConvert::create(TensorLayout::NCHW);
        auto d_out = out_c8.desc();
        const TensorDesc unpack_in_arr[] = {d_out};
        auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in_arr);
        NNOPS_EXPECT_EQ(unpack_descs[0].rank, 4);
        NNOPS_EXPECT_EQ(unpack_descs[0].layout, TensorLayout::NCHW);

        std::vector<char> unpack_buf(unpack_descs[0].storage_bytes());
        auto res_nchw = test::make_planar(unpack_descs[0], unpack_buf.data());
        const TensorView unpack_ins[] = {out_c8};
        unpack_op->compute(res_nchw, unpack_ins);

        for (size_t i = 0; i < 48; ++i) {
            NNOPS_EXPECT_TRUE(!std::isnan(res_nchw.ptr<float>()[i]));
            NNOPS_EXPECT_TRUE(!std::isinf(res_nchw.ptr<float>()[i]));
        }
    }
}
