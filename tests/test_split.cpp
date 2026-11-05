/// @file test_split.cpp
/// @brief Split tests — explicit lengths, an even split, a negative axis, and
/// the rejection of lengths that do not cover the axis.
///
/// Every output is a contiguous region of the input along `axis`, so the sums of
/// the slice lengths must equal the axis extent. With `split` empty the axis is
/// divided evenly across num_outputs.

#include "nnops/ops/split.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <vector>

using namespace nnops;

NNOPS_TEST(split_explicit_lengths) {
    float data[] = {0, 1, 2, 3, 4, 5, 6, 7};
    const int64_t shape[] = {2, 4};
    TensorView in(shape, DataType::f32, data);

    SplitAttributes attrs;
    attrs.axis = 1;
    attrs.split = {1, 3};
    auto op = Split::create(attrs);

    const TensorDesc in_arr[] = {in.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs.size(), 2u);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 1);
    NNOPS_EXPECT_EQ(descs[1].dims[1], 3);

    std::vector<float> b0(static_cast<size_t>(descs[0].numel()));
    std::vector<float> b1(static_cast<size_t>(descs[1].numel()));
    TensorView outs[] = {test::make_channel_first(descs[0], b0.data()),
                         test::make_channel_first(descs[1], b1.data())};
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(op->compute(std::span<TensorView>(outs), ins) == Status::ok);

    NNOPS_EXPECT_NEAR(b0[0], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(b0[1], 4.0f, 1e-6f);
    const float want1[6] = {1, 2, 3, 5, 6, 7};
    for (int i = 0; i < 6; ++i) { NNOPS_EXPECT_NEAR(b1[i], want1[i], 1e-6f); }
}

NNOPS_TEST(split_even_and_negative_axis) {
    float data[] = {0, 1, 2, 3, 4, 5};
    const int64_t shape[] = {2, 3};
    TensorView in(shape, DataType::f32, data);

    SplitAttributes attrs;
    attrs.axis = -1;
    attrs.num_outputs = 3;
    auto op = Split::create(attrs);

    const TensorDesc in_arr[] = {in.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs.size(), 3u);
    for (const auto& d : descs) { NNOPS_EXPECT_EQ(d.dims[1], 1); }

    std::vector<float> b[3];
    TensorView outs[3];
    for (int j = 0; j < 3; ++j) {
        b[j].assign(static_cast<size_t>(descs[j].numel()), -1.0f);
        outs[j] = test::make_channel_first(descs[j], b[j].data());
    }
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(op->compute(std::span<TensorView>(outs, 3), ins) == Status::ok);

    const float want[3][2] = {{0, 3}, {1, 4}, {2, 5}};
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 2; ++i) {
            NNOPS_EXPECT_NEAR(b[j][i], want[j][i], 1e-6f);
        }
    }
}

NNOPS_TEST(split_f16_exact) {
    using nnops::backend::cpu::half;
    using nnops::backend::cpu::half_to_bits;
    half data[4];
    const float vals[4] = {0.5f, -1.25f, 3.0f, 42.0f};
    for (int i = 0; i < 4; ++i) { simd::s_store(&data[i], vals[i]); }

    const int64_t shape[] = {4};
    TensorView in(shape, DataType::f16, data);

    SplitAttributes attrs;
    attrs.axis = 0;
    attrs.split = {2, 2};
    auto op = Split::create(attrs);

    const TensorDesc in_arr[] = {in.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    half b0[2], b1[2];
    TensorView outs[] = {test::make_channel_first(descs[0], b0),
                         test::make_channel_first(descs[1], b1)};
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(op->compute(std::span<TensorView>(outs), ins) == Status::ok);

    NNOPS_EXPECT_EQ(half_to_bits(b0[0]), half_to_bits(data[0]));
    NNOPS_EXPECT_EQ(half_to_bits(b0[1]), half_to_bits(data[1]));
    NNOPS_EXPECT_EQ(half_to_bits(b1[0]), half_to_bits(data[2]));
    NNOPS_EXPECT_EQ(half_to_bits(b1[1]), half_to_bits(data[3]));
}

NNOPS_TEST(split_rejects_bad_lengths) {
    float data[] = {0, 1, 2, 3, 4, 5};
    const int64_t shape[] = {2, 3};
    TensorView in(shape, DataType::f32, data);
    const TensorDesc in_arr[] = {in.desc()};

    {
        SplitAttributes attrs;
        attrs.axis = 1;
        attrs.split = {1, 1};  // sums to 2, the axis is 3
        auto op = Split::create(attrs);
        auto descs = op->getOutputTensorDesc(in_arr);
        NNOPS_EXPECT_TRUE(op->validate(in_arr, descs) == Status::invalid_argument);
    }
    {
        SplitAttributes attrs;
        attrs.axis = 1;
        attrs.num_outputs = 2;  // 3 does not divide evenly
        auto op = Split::create(attrs);
        auto descs = op->getOutputTensorDesc(in_arr);
        NNOPS_EXPECT_TRUE(op->validate(in_arr, descs) == Status::invalid_argument);
    }
}
