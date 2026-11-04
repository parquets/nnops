/// @file test_gather.cpp
/// @brief Gather family tests — one axis, per-element, and ND tuples.
///
/// Gather replaces the gathered axis with the index shape; GatherElements keeps
/// both ranks equal and swaps only the axis coordinate; GatherND reads a K-tuple
/// into the leading data axes. Indices are s64/s32 and may be negative (wrapping
/// from the end of the axis). The element copy is bit-exact for f16 and s64.
/// Out-of-range indices are rejected rather than read.

#include "nnops/ops/gather.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <vector>

using namespace nnops;

NNOPS_TEST(gather_axis0) {
    float data[] = {0, 1, 2, 3, 4, 5};
    const int64_t d_shape[] = {3, 2};
    TensorView d(d_shape, DataType::f32, data);

    int64_t idx[] = {1};
    const int64_t i_shape[] = {1};
    TensorView indices(i_shape, DataType::s64, idx);

    auto op = Gather::create(GatherAttributes{0});
    const TensorDesc in_arr[] = {d.desc(), indices.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 1);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 2);

    float out[2] = {};
    TensorView ov = test::make_planar(descs[0], out);
    const TensorView ins[] = {d, indices};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);
    NNOPS_EXPECT_NEAR(out[0], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out[1], 3.0f, 1e-6f);
}

NNOPS_TEST(gather_axis1_2d_indices) {
    float data[] = {0, 1, 2, 3, 4, 5};
    const int64_t d_shape[] = {2, 3};
    TensorView d(d_shape, DataType::f32, data);

    int64_t idx[] = {2, 0, 1, 1};
    const int64_t i_shape[] = {2, 2};
    TensorView indices(i_shape, DataType::s64, idx);

    auto op = Gather::create(GatherAttributes{1});
    const TensorDesc in_arr[] = {d.desc(), indices.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 3);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[2], 2);

    float out[8] = {};
    TensorView ov = test::make_planar(descs[0], out);
    const TensorView ins[] = {d, indices};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    // out[n, i, j] = data[n, idx[i, j]] — the index shape replaces the axis
    const float want[8] = {2, 0, 1, 1, 5, 3, 4, 4};
    for (int i = 0; i < 8; ++i) { NNOPS_EXPECT_NEAR(out[i], want[i], 1e-6f); }
}

NNOPS_TEST(gather_negative_and_s32) {
    float data[] = {10, 20, 30, 40};
    const int64_t d_shape[] = {4};
    TensorView d(d_shape, DataType::f32, data);

    int32_t idx[] = {-1, 0, -4};
    const int64_t i_shape[] = {3};
    TensorView indices(i_shape, DataType::s32, idx);

    auto op = Gather::create(GatherAttributes{0});
    const TensorDesc in_arr[] = {d.desc(), indices.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    float out[3] = {};
    TensorView ov = test::make_planar(descs[0], out);
    const TensorView ins[] = {d, indices};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    NNOPS_EXPECT_NEAR(out[0], 40.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out[1], 10.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out[2], 10.0f, 1e-6f);
}

NNOPS_TEST(gather_f16_exact) {
    using nnops::backend::cpu::half;
    using nnops::backend::cpu::half_to_bits;
    half data[4];
    const float vals[4] = {0.25f, -1.75f, 3.5f, 100.0f};
    for (int i = 0; i < 4; ++i) { simd::s_store(&data[i], vals[i]); }

    const int64_t d_shape[] = {4};
    TensorView d(d_shape, DataType::f16, data);

    int64_t idx[] = {3, 1};
    const int64_t i_shape[] = {2};
    TensorView indices(i_shape, DataType::s64, idx);

    auto op = Gather::create(GatherAttributes{0});
    const TensorDesc in_arr[] = {d.desc(), indices.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    half out[2];
    TensorView ov = test::make_planar(descs[0], out);
    const TensorView ins[] = {d, indices};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    NNOPS_EXPECT_EQ(half_to_bits(out[0]), half_to_bits(data[3]));
    NNOPS_EXPECT_EQ(half_to_bits(out[1]), half_to_bits(data[1]));
}

NNOPS_TEST(gather_rejects_out_of_range) {
    float data[] = {0, 1, 2};
    const int64_t d_shape[] = {3};
    TensorView d(d_shape, DataType::f32, data);

    int64_t idx[] = {3};  // one past the end
    const int64_t i_shape[] = {1};
    TensorView indices(i_shape, DataType::s64, idx);

    auto op = Gather::create(GatherAttributes{0});
    const TensorDesc in_arr[] = {d.desc(), indices.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    float out[1] = {};
    TensorView ov = test::make_planar(descs[0], out);
    const TensorView ins[] = {d, indices};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::invalid_argument);
}

NNOPS_TEST(gather_elements_axis1) {
    float data[] = {1, 2, 3, 4, 5, 6};
    const int64_t d_shape[] = {2, 3};
    TensorView d(d_shape, DataType::f32, data);

    int64_t idx[] = {2, 0, 0, 1};
    const int64_t i_shape[] = {2, 2};
    TensorView indices(i_shape, DataType::s64, idx);

    auto op = GatherElements::create(GatherElementsAttributes{1});
    const TensorDesc in_arr[] = {d.desc(), indices.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 2);

    float out[4] = {};
    TensorView ov = test::make_planar(descs[0], out);
    const TensorView ins[] = {d, indices};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    const float want[4] = {3, 1, 4, 5};
    for (int i = 0; i < 4; ++i) { NNOPS_EXPECT_NEAR(out[i], want[i], 1e-6f); }
}

NNOPS_TEST(gather_elements_negative) {
    float data[] = {1, 2, 3, 4, 5, 6};
    const int64_t d_shape[] = {2, 3};
    TensorView d(d_shape, DataType::f32, data);

    int64_t idx[] = {-1, -2, -3, -1};
    const int64_t i_shape[] = {2, 2};
    TensorView indices(i_shape, DataType::s64, idx);

    auto op = GatherElements::create(GatherElementsAttributes{1});
    const TensorDesc in_arr[] = {d.desc(), indices.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    float out[4] = {};
    TensorView ov = test::make_planar(descs[0], out);
    const TensorView ins[] = {d, indices};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    // 2, 1 for row 0 and 0, 2 for row 1
    const float want[4] = {3, 2, 4, 6};
    for (int i = 0; i < 4; ++i) { NNOPS_EXPECT_NEAR(out[i], want[i], 1e-6f); }
}

NNOPS_TEST(gather_nd_basic) {
    float data[] = {0, 1, 2, 3, 4, 5, 6, 7};
    const int64_t d_shape[] = {2, 2, 2};
    TensorView d(d_shape, DataType::f32, data);

    int64_t idx[] = {0, 0, 1, 1};  // two tuples of K = 2
    const int64_t i_shape[] = {2, 2};
    TensorView indices(i_shape, DataType::s64, idx);

    auto op = GatherND::create(GatherNDAttributes{});
    const TensorDesc in_arr[] = {d.desc(), indices.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 2);

    float out[4] = {};
    TensorView ov = test::make_planar(descs[0], out);
    const TensorView ins[] = {d, indices};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    const float want[4] = {0, 1, 6, 7};
    for (int i = 0; i < 4; ++i) { NNOPS_EXPECT_NEAR(out[i], want[i], 1e-6f); }
}

NNOPS_TEST(gather_nd_batch_dims) {
    float data[] = {0, 1, 2, 3, 4, 5, 6, 7};
    const int64_t d_shape[] = {2, 2, 2};
    TensorView d(d_shape, DataType::f32, data);

    int64_t idx[] = {0, 1};  // one tuple per batch entry, K = 1
    const int64_t i_shape[] = {2, 1};
    TensorView indices(i_shape, DataType::s64, idx);

    auto op = GatherND::create(GatherNDAttributes{1});
    const TensorDesc in_arr[] = {d.desc(), indices.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 2);

    float out[4] = {};
    TensorView ov = test::make_planar(descs[0], out);
    const TensorView ins[] = {d, indices};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    // batch 0 tuple 0 → data[0][0]; batch 1 tuple 1 → data[1][1]
    const float want[4] = {0, 1, 6, 7};
    for (int i = 0; i < 4; ++i) { NNOPS_EXPECT_NEAR(out[i], want[i], 1e-6f); }
}

NNOPS_TEST(gather_nd_rejects_out_of_range) {
    float data[] = {0, 1, 2, 3};
    const int64_t d_shape[] = {2, 2};
    TensorView d(d_shape, DataType::f32, data);

    int64_t idx[] = {0, 2};  // column 2 does not exist
    const int64_t i_shape[] = {1, 2};
    TensorView indices(i_shape, DataType::s64, idx);

    auto op = GatherND::create(GatherNDAttributes{});
    const TensorDesc in_arr[] = {d.desc(), indices.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    float out[1] = {};
    TensorView ov = test::make_planar(descs[0], out);
    const TensorView ins[] = {d, indices};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::invalid_argument);
}
