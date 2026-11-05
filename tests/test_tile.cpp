/// @file test_tile.cpp
/// @brief Tile and Expand tests — per-axis repetition, NumPy broadcast, a rank
/// increase and a -1 target axis.
///
/// Tile's output is in.dims * repeats, axis by axis. Expand right-aligns the
/// target against the input: a size-1 (or missing) input axis repeats, any other
/// axis must land on its own extent, and -1 copies the input's extent.

#include "nnops/ops/tile.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <vector>

using namespace nnops;

NNOPS_TEST(tile_1d) {
    float data[] = {1, 2};
    const int64_t shape[] = {2};
    TensorView in(shape, DataType::f32, data);

    TileAttributes attrs;
    attrs.repeats = {3};
    auto op = Tile::create(attrs);

    const TensorDesc in_arr[] = {in.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 6);

    float out[6] = {};
    TensorView ov = test::make_channel_first(descs[0], out);
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    const float want[6] = {1, 2, 1, 2, 1, 2};
    for (int i = 0; i < 6; ++i) { NNOPS_EXPECT_NEAR(out[i], want[i], 1e-6f); }
}

NNOPS_TEST(tile_2d) {
    float data[] = {1, 2, 3, 4};
    const int64_t shape[] = {2, 2};
    TensorView in(shape, DataType::f32, data);

    TileAttributes attrs;
    attrs.repeats = {2, 2};
    auto op = Tile::create(attrs);

    const TensorDesc in_arr[] = {in.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 4);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 4);

    float out[16] = {};
    TensorView ov = test::make_channel_first(descs[0], out);
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    const float want[16] = {
        1, 2, 1, 2,
        3, 4, 3, 4,
        1, 2, 1, 2,
        3, 4, 3, 4,
    };
    for (int i = 0; i < 16; ++i) { NNOPS_EXPECT_NEAR(out[i], want[i], 1e-6f); }
}

NNOPS_TEST(tile_int64_exact) {
    int64_t data[] = {-1, 1234567890123LL};
    const int64_t shape[] = {2};
    TensorView in(shape, DataType::s64, data);

    TileAttributes attrs;
    attrs.repeats = {2};
    auto op = Tile::create(attrs);

    const TensorDesc in_arr[] = {in.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    int64_t out[4] = {};
    TensorView ov = test::make_channel_first(descs[0], out);
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    NNOPS_EXPECT_EQ(out[0], data[0]);
    NNOPS_EXPECT_EQ(out[1], data[1]);
    NNOPS_EXPECT_EQ(out[2], data[0]);
    NNOPS_EXPECT_EQ(out[3], data[1]);
}

NNOPS_TEST(expand_row_broadcast) {
    float data[] = {1, 2, 3};
    const int64_t shape[] = {1, 3};
    TensorView in(shape, DataType::f32, data);

    ExpandAttributes attrs;
    attrs.shape = {2, 3};
    auto op = Expand::create(attrs);

    const TensorDesc in_arr[] = {in.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 3);

    float out[6] = {};
    TensorView ov = test::make_channel_first(descs[0], out);
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    const float want[6] = {1, 2, 3, 1, 2, 3};
    for (int i = 0; i < 6; ++i) { NNOPS_EXPECT_NEAR(out[i], want[i], 1e-6f); }
}

NNOPS_TEST(expand_rank_increase) {
    float data[] = {1, 2, 3};
    const int64_t shape[] = {3};
    TensorView in(shape, DataType::f32, data);

    ExpandAttributes attrs;
    attrs.shape = {2, 2, 3};
    auto op = Expand::create(attrs);

    const TensorDesc in_arr[] = {in.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].rank, 3);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[2], 3);

    float out[12] = {};
    TensorView ov = test::make_channel_first(descs[0], out);
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    for (int i = 0; i < 12; ++i) {
        NNOPS_EXPECT_NEAR(out[i], data[i % 3], 1e-6f);
    }
}

NNOPS_TEST(expand_minus_one_keeps_axis) {
    float data[] = {1, 2, 3};
    const int64_t shape[] = {1, 3};
    TensorView in(shape, DataType::f32, data);

    ExpandAttributes attrs;
    attrs.shape = {4, -1};
    auto op = Expand::create(attrs);

    const TensorDesc in_arr[] = {in.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 4);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 3);

    float out[12] = {};
    TensorView ov = test::make_channel_first(descs[0], out);
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    for (int i = 0; i < 12; ++i) {
        NNOPS_EXPECT_NEAR(out[i], data[i % 3], 1e-6f);
    }
}

NNOPS_TEST(tile_expand_reject_bad_attrs) {
    float data[] = {1, 2, 3, 4, 5, 6};
    const int64_t shape[] = {2, 3};
    TensorView in(shape, DataType::f32, data);
    const TensorDesc in_arr[] = {in.desc()};

    {
        TileAttributes attrs;
        attrs.repeats = {2};  // needs one per axis
        auto op = Tile::create(attrs);
        const TensorDesc out_arr[] = {in.desc()};
        NNOPS_EXPECT_TRUE(op->validate(in_arr, out_arr) == Status::invalid_argument);
    }
    {
        ExpandAttributes attrs;
        attrs.shape = {4, 3};  // 2 does not broadcast to 4
        auto op = Expand::create(attrs);
        auto descs = op->getOutputTensorDesc(in_arr);
        NNOPS_EXPECT_TRUE(op->validate(in_arr, descs) == Status::invalid_argument);
    }
}
