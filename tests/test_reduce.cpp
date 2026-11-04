/// @file test_reduce.cpp
/// @brief Tests for the Reduce operator (Sum, Min, Max, Mean).
///
/// Planar 1D/2D/3D, keepdims, negative axes, channel-last and f16. The 1D/2D
/// expected values are hand-computed next to the asserts; the random-data
/// tests build a scalar reference over the same buffer. A negative axis wraps
/// from the end, so the negative-axis test runs axis=2 and axis=-1 on one
/// input and compares the two outputs.
///
/// The channel-last section converts a planar input to NHWC/NDHWC, reduces, and
/// compares the result against the planar reduce in logical order. The layout
/// only changes where the values live: the axis means the same thing either
/// way. A rank-dropping reduce on a channel-last input produces a planar
/// output -- dropping an axis shifts whichever axis C sat on, so the tag cannot
/// describe the shorter rank any more -- while keepdims keeps the rank and the
/// channel-last tag.
///
/// f16 goes through the SIMD f16 path and is compared against the f32
/// reference at 5e-2 rather than round-tripped.

#include "nnops/ops/reduce.hpp"
#include "nnops/ops/layout_convert.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"

#include <cmath>
#include <random>

using namespace nnops;

// ============================================================
// 1D tests
// ============================================================

NNOPS_TEST(reduce_1d_sum) {
    const int64_t shape[] = {5};
    float in_data[]  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float out_data[1] = {0};

    TensorView input(shape, DataType::f32, in_data);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Sum;
    attrs.axis = 0;

    auto op = Reduce::create(attrs, Backend::CPU);
    auto d = input.desc();
    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 0);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    auto output = nnops::test::make_planar(descs[0], out_data);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_data[0], 15.0f, 1e-6f);
}

NNOPS_TEST(reduce_1d_max) {
    const int64_t shape[] = {5};
    float in_data[]  = {1.0f, 5.0f, 3.0f, 2.0f, 4.0f};
    float out_data[1] = {0};

    TensorView input(shape, DataType::f32, in_data);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Max;
    attrs.axis = 0;

    auto op = Reduce::create(attrs, Backend::CPU);
    auto d = input.desc();
    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 0);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    auto output = nnops::test::make_planar(descs[0], out_data);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_data[0], 5.0f, 1e-6f);
}

NNOPS_TEST(reduce_1d_min) {
    const int64_t shape[] = {5};
    float in_data[]  = {3.0f, 1.0f, 5.0f, 2.0f, 4.0f};
    float out_data[1] = {0};

    TensorView input(shape, DataType::f32, in_data);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Min;
    attrs.axis = 0;

    auto op = Reduce::create(attrs, Backend::CPU);
    auto d = input.desc();
    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 0);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    auto output = nnops::test::make_planar(descs[0], out_data);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_data[0], 1.0f, 1e-6f);
}

NNOPS_TEST(reduce_1d_mean) {
    const int64_t shape[] = {5};
    float in_data[]  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float out_data[1] = {0};

    TensorView input(shape, DataType::f32, in_data);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Mean;
    attrs.axis = 0;

    auto op = Reduce::create(attrs, Backend::CPU);
    auto d = input.desc();
    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 0);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    auto output = nnops::test::make_planar(descs[0], out_data);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_data[0], 3.0f, 1e-6f);
}

NNOPS_TEST(reduce_1d_keepdims) {
    const int64_t shape[] = {5};
    float in_data[]  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float out_data[1] = {0};

    TensorView input(shape, DataType::f32, in_data);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Sum;
    attrs.axis = 0;
    attrs.keepdims = true;

    auto op = Reduce::create(attrs, Backend::CPU);
    auto d = input.desc();
    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 1);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    auto output = nnops::test::make_planar(descs[0], out_data);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    NNOPS_EXPECT_EQ(output.rank(), 1);
    NNOPS_EXPECT_EQ(output.shape(0), 1);
    NNOPS_EXPECT_NEAR(out_data[0], 15.0f, 1e-6f);
}

// ============================================================
// 2D tests
// ============================================================

NNOPS_TEST(reduce_2d_axis0_sum) {
    const int64_t shape[] = {3, 4};
    // [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    float in_data[] = {1,2,3,4, 5,6,7,8, 9,10,11,12};
    float out_data[4] = {0};

    TensorView input(shape, DataType::f32, in_data);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Sum;
    attrs.axis = 0;

    auto op = Reduce::create(attrs, Backend::CPU);
    auto d = input.desc();
    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 4);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    auto output = nnops::test::make_planar(descs[0], out_data);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    // Column sums: 1+5+9=15, 2+6+10=18, 3+7+11=21, 4+8+12=24
    NNOPS_EXPECT_NEAR(out_data[0], 15.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 18.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 21.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[3], 24.0f, 1e-6f);
}

NNOPS_TEST(reduce_2d_axis1_max) {
    const int64_t shape[] = {2, 3};
    // [[1, 5, 3], [4, 2, 6]]
    float in_data[] = {1, 5, 3, 4, 2, 6};
    float out_data[2] = {0};

    TensorView input(shape, DataType::f32, in_data);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Max;
    attrs.axis = 1;

    auto op = Reduce::create(attrs, Backend::CPU);
    auto d = input.desc();
    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    auto output = nnops::test::make_planar(descs[0], out_data);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_data[0], 5.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 6.0f, 1e-6f);
}

NNOPS_TEST(reduce_2d_axis1_min) {
    const int64_t shape[] = {2, 3};
    // [[3, 1, 5], [4, 2, 6]]
    float in_data[] = {3, 1, 5, 4, 2, 6};
    float out_data[2] = {0};

    TensorView input(shape, DataType::f32, in_data);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Min;
    attrs.axis = 1;

    auto op = Reduce::create(attrs, Backend::CPU);
    auto d = input.desc();
    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    auto output = nnops::test::make_planar(descs[0], out_data);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_data[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 2.0f, 1e-6f);
}

NNOPS_TEST(reduce_2d_axis0_mean) {
    const int64_t shape[] = {2, 3};
    // [[1, 2, 3], [4, 5, 6]]
    float in_data[] = {1, 2, 3, 4, 5, 6};
    float out_data[3] = {0};

    TensorView input(shape, DataType::f32, in_data);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Mean;
    attrs.axis = 0;

    auto op = Reduce::create(attrs, Backend::CPU);
    auto d = input.desc();
    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    auto output = nnops::test::make_planar(descs[0], out_data);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    // (1+4)/2=2.5, (2+5)/2=3.5, (3+6)/2=4.5
    NNOPS_EXPECT_NEAR(out_data[0], 2.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 3.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 4.5f, 1e-6f);
}

NNOPS_TEST(reduce_2d_keepdims) {
    const int64_t shape[] = {3, 4};
    float in_data[12] = {};
    for (int i = 0; i < 12; ++i) { in_data[i] = static_cast<float>(i + 1); }
    float out_data[3] = {0};

    TensorView input(shape, DataType::f32, in_data);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Sum;
    attrs.axis = 1;
    attrs.keepdims = true;

    auto op = Reduce::create(attrs, Backend::CPU);
    auto d = input.desc();
    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 1);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    auto output = nnops::test::make_planar(descs[0], out_data);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    NNOPS_EXPECT_EQ(output.rank(), 2);
    NNOPS_EXPECT_EQ(output.shape(0), 3);
    NNOPS_EXPECT_EQ(output.shape(1), 1);
    // Row 0: 1+2+3+4=10, Row 1: 5+6+7+8=26, Row 2: 9+10+11+12=42
    NNOPS_EXPECT_NEAR(out_data[0], 10.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 26.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 42.0f, 1e-6f);
}

// ============================================================
// Negative axis
// ============================================================

NNOPS_TEST(reduce_negative_axis) {
    const int64_t shape[] = {3, 4, 5};
    auto [in_vec, input] = test::make_random_tensor(shape, 0.0f, 1.0f, 99);

    ReduceAttributes attrs1;
    attrs1.type = ReduceType::Sum;
    attrs1.axis = 2;  // last axis (positive)

    auto op1 = Reduce::create(attrs1, Backend::CPU);
    auto d = input.desc();
    const TensorDesc arr[] = {d};
    auto descs1 = op1->getOutputTensorDesc(arr);

    NNOPS_EXPECT_EQ(descs1[0].rank, 2);
    NNOPS_EXPECT_EQ(descs1[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs1[0].dims[1], 4);

    std::vector<float> out1(static_cast<size_t>(descs1[0].numel()));
    auto output1 = nnops::test::make_planar(descs1[0], out1.data());
    const TensorView ins1[] = {input};
    op1->compute(output1, ins1);

    ReduceAttributes attrs2;
    attrs2.type = ReduceType::Sum;
    attrs2.axis = -1;  // last axis (negative)

    auto op2 = Reduce::create(attrs2, Backend::CPU);
    auto descs2 = op2->getOutputTensorDesc(arr);

    NNOPS_EXPECT_EQ(descs2[0].rank, 2);
    NNOPS_EXPECT_EQ(descs2[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs2[0].dims[1], 4);

    std::vector<float> out2(static_cast<size_t>(descs2[0].numel()));
    auto output2 = nnops::test::make_planar(descs2[0], out2.data());
    const TensorView ins2[] = {input};
    op2->compute(output2, ins2);

    for (size_t i = 0; i < out1.size(); ++i) {
        NNOPS_EXPECT_NEAR(out1[i], out2[i], 1e-6f);
    }
}

// ============================================================
// Random data — all four types
// ============================================================

NNOPS_TEST(reduce_random_sum) {
    auto [in_vec, input] = test::make_random_tensor({5, 8}, -10.0f, 10.0f, 200);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Sum;
    attrs.axis = 1;

    auto op = Reduce::create(attrs, Backend::CPU);
    auto d = input.desc();
    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 5);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    auto output = nnops::test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    for (int r = 0; r < 5; ++r) {
        float expected = 0.0f;
        for (int c = 0; c < 8; ++c) { expected += in_vec[r * 8 + c]; }
        NNOPS_EXPECT_NEAR(out_buf[r], expected, 1e-4f);
    }
}

NNOPS_TEST(reduce_random_max) {
    auto [in_vec, input] = test::make_random_tensor({3, 10}, -100.0f, 100.0f, 300);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Max;
    attrs.axis = 0;

    auto op = Reduce::create(attrs, Backend::CPU);
    auto d = input.desc();
    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 10);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    auto output = nnops::test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    for (int c = 0; c < 10; ++c) {
        float expected = -std::numeric_limits<float>::infinity();
        for (int r = 0; r < 3; ++r) {
            if (in_vec[r * 10 + c] > expected) { expected = in_vec[r * 10 + c]; }
        }
        NNOPS_EXPECT_NEAR(out_buf[c], expected, 1e-6f);
    }
}

NNOPS_TEST(reduce_random_min) {
    auto [in_vec, input] = test::make_random_tensor({4, 7}, -50.0f, 50.0f, 400);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Min;
    attrs.axis = 0;

    auto op = Reduce::create(attrs, Backend::CPU);
    auto d = input.desc();
    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 7);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    auto output = nnops::test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    for (int c = 0; c < 7; ++c) {
        float expected = std::numeric_limits<float>::infinity();
        for (int r = 0; r < 4; ++r) {
            if (in_vec[r * 7 + c] < expected) { expected = in_vec[r * 7 + c]; }
        }
        NNOPS_EXPECT_NEAR(out_buf[c], expected, 1e-6f);
    }
}

NNOPS_TEST(reduce_random_mean) {
    auto [in_vec, input] = test::make_random_tensor({6, 5}, -20.0f, 20.0f, 500);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Mean;
    attrs.axis = 1;

    auto op = Reduce::create(attrs, Backend::CPU);
    auto d = input.desc();
    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 6);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    auto output = nnops::test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    for (int r = 0; r < 6; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < 5; ++c) { sum += in_vec[r * 5 + c]; }
        NNOPS_EXPECT_NEAR(out_buf[r], sum / 5.0f, 1e-4f);
    }
}

// ============================================================
// 3D tests
// ============================================================

NNOPS_TEST(reduce_3d_middle_axis) {
    const int64_t shape[] = {2, 3, 4};
    auto [in_vec, input] = test::make_random_tensor(shape, -1.0f, 1.0f, 600);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Sum;
    attrs.axis = 1;

    auto op = Reduce::create(attrs, Backend::CPU);
    auto d = input.desc();
    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 4);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    auto output = nnops::test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    for (int i = 0; i < 2; ++i) {
        for (int k = 0; k < 4; ++k) {
            float expected = 0.0f;
            for (int j = 0; j < 3; ++j) {
                expected += in_vec[i * 12 + j * 4 + k];
            }
            NNOPS_EXPECT_NEAR(out_buf[i * 4 + k], expected, 1e-4f);
        }
    }
}

NNOPS_TEST(reduce_3d_first_axis) {
    const int64_t shape[] = {3, 2, 5};
    auto [in_vec, input] = test::make_random_tensor(shape, -5.0f, 5.0f, 700);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Mean;
    attrs.axis = 0;

    auto op = Reduce::create(attrs, Backend::CPU);
    auto d = input.desc();
    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 5);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    auto output = nnops::test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    for (int j = 0; j < 2; ++j) {
        for (int k = 0; k < 5; ++k) {
            float sum = 0.0f;
            for (int i = 0; i < 3; ++i) {
                sum += in_vec[i * 10 + j * 5 + k];
            }
            NNOPS_EXPECT_NEAR(out_buf[j * 5 + k], sum / 3.0f, 1e-4f);
        }
    }
}

// ============================================================
// Channel-last layout tests (NHWC / NDHWC)
// ============================================================

namespace {

/// Convert a planar view into `target` and return the view; `buf` outlives it.
TensorView to_layout(const TensorView& planar, TensorLayout target,
                     std::vector<char>& buf)
{
    auto op = LayoutConvert::create(target);
    const TensorDesc in[] = {planar.desc()};
    auto d = op->getOutputTensorDesc(in)[0];
    buf.resize(d.storage_bytes());
    TensorView out = test::make_packed(d, buf.data());
    const TensorView ins[] = {planar};
    op->compute(out, ins);
    return out;
}

/// Compare two same-dimension views in logical order (any layout).
bool same_logical_f32(const TensorView& a, const TensorView& b)
{
    if (a.rank() != b.rank()) {
        return false;
    }
    for (int64_t d = 0; d < a.rank(); ++d) {
        if (a.shape(d) != b.shape(d)) {
            return false;
        }
    }
    int64_t idx[8] = {};
    const int64_t rank = a.rank();
    const int64_t total = a.numel();
    for (int64_t flat = 0; flat < total; ++flat) {
        int64_t off_a = 0;
        int64_t off_b = 0;
        for (int64_t d = 0; d < rank; ++d) {
            off_a += idx[d] * a.stride_elems(d);
            off_b += idx[d] * b.stride_elems(d);
        }
        const float x = a.ptr<float>()[off_a];
        const float y = b.ptr<float>()[off_b];
        // Mean's reciprocal multiply can differ by an ulp from a divide.
        if (std::fabs(x - y) > 1e-5f + 1e-5f * std::fabs(y)) {
            return false;
        }
        for (int64_t d = rank - 1; d >= 0; --d) {
            if (++idx[d] < a.shape(d)) {
                break;
            }
            idx[d] = 0;
        }
    }
    return true;
}

/// Run the reduce on a planar input and on its channel-last twin; the results
/// must agree element for element in logical order.
void check_reduce_layout(const std::vector<int64_t>& shape,
                         ReduceType type, int64_t axis, bool keepdims)
{
    const uint64_t seed = static_cast<uint64_t>(
        100 + axis * 7 + (keepdims ? 1 : 0) + static_cast<int>(type) * 31);
    auto [in_vec, in_planar] = test::make_random_tensor(shape, -2.0f, 2.0f, seed);

    ReduceAttributes attrs;
    attrs.type = type;
    attrs.axis = axis;
    attrs.keepdims = keepdims;
    auto op = Reduce::create(attrs, Backend::CPU);

    const TensorDesc planar_in[] = {in_planar.desc()};
    auto ref_desc = op->getOutputTensorDesc(planar_in);
    std::vector<float> ref_buf(static_cast<size_t>(ref_desc[0].numel()));
    auto ref_out = test::make_planar(ref_desc[0], ref_buf.data());
    const TensorView ref_ins[] = {in_planar};
    op->compute(ref_out, ref_ins);

    std::vector<char> cl_buf;
    const TensorLayout cl_layout =
        (shape.size() == 5) ? TensorLayout::NDHWC : TensorLayout::NHWC;
    auto in_cl = to_layout(in_planar, cl_layout, cl_buf);

    const TensorDesc cl_in[] = {in_cl.desc()};
    auto cl_desc = op->getOutputTensorDesc(cl_in);
    NNOPS_EXPECT_EQ(cl_desc[0].rank, ref_desc[0].rank);
    for (int64_t d = 0; d < cl_desc[0].rank; ++d) {
        NNOPS_EXPECT_EQ(cl_desc[0].dims[static_cast<size_t>(d)],
                        ref_desc[0].dims[static_cast<size_t>(d)]);
    }
    // A rank-preserving output stays channel-last; a rank-dropping one is planar.
    const TensorLayout planar_layout =
        (shape.size() == 5) ? TensorLayout::NCDHW : TensorLayout::NCHW;
    NNOPS_EXPECT_EQ(cl_desc[0].layout, keepdims ? cl_layout : planar_layout);

    std::vector<float> out_buf(
        static_cast<size_t>(cl_desc[0].storage_bytes() / sizeof(float)));
    auto out_cl = test::make_packed(cl_desc[0], out_buf.data());
    const TensorView cl_ins[] = {in_cl};
    op->compute(out_cl, cl_ins);

    NNOPS_EXPECT_TRUE(same_logical_f32(out_cl, ref_out));
}

}  // anonymous namespace

NNOPS_TEST(reduce_nhwc_axis_c_sum) {
    // Hand-checked: C is contiguous in NHWC, so this is the SIMD row path.
    const int64_t N = 1, C = 4, H = 1, W = 3;
    float in_planar_data[] = {
        1.0f, 2.0f, 3.0f,        // c=0
        10.0f, 20.0f, 30.0f,     // c=1
        100.0f, 200.0f, 300.0f,  // c=2
        -1.0f, -2.0f, -3.0f,     // c=3
    };
    const int64_t shape[] = {N, C, H, W};
    TensorView in_planar(shape, DataType::f32, in_planar_data);

    std::vector<char> cl_buf;
    auto in_cl = to_layout(in_planar, TensorLayout::NHWC, cl_buf);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Sum;
    attrs.axis = 1;
    auto op = Reduce::create(attrs, Backend::CPU);

    const TensorDesc in_desc[] = {in_cl.desc()};
    auto descs = op->getOutputTensorDesc(in_desc);
    NNOPS_EXPECT_EQ(descs[0].rank, 3);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    auto output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {in_cl};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], 1.0f + 10.0f + 100.0f - 1.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[1], 2.0f + 20.0f + 200.0f - 2.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[2], 3.0f + 30.0f + 300.0f - 3.0f, 1e-4f);
}

NNOPS_TEST(reduce_nhwc_axis_c_keepdims) {
    check_reduce_layout({2, 5, 3, 4}, ReduceType::Sum, 1, true);
    check_reduce_layout({2, 5, 3, 4}, ReduceType::Mean, 1, true);
}

NNOPS_TEST(reduce_nhwc_all_axes) {
    const std::vector<int64_t> shape = {2, 4, 3, 5};
    const ReduceType types[] = {ReduceType::Sum, ReduceType::Max,
                                ReduceType::Min, ReduceType::Mean};
    for (const auto type : types) {
        for (int64_t axis = 0; axis < 4; ++axis) {
            check_reduce_layout(shape, type, axis, false);
            check_reduce_layout(shape, type, axis, true);
        }
        check_reduce_layout(shape, type, -1, false);
    }
}

NNOPS_TEST(reduce_ndhwc_all_axes) {
    const std::vector<int64_t> shape = {2, 3, 2, 3, 4};
    const ReduceType types[] = {ReduceType::Sum, ReduceType::Max, ReduceType::Mean};
    for (const auto type : types) {
        for (int64_t axis = 0; axis < 5; ++axis) {
            check_reduce_layout(shape, type, axis, false);
            check_reduce_layout(shape, type, axis, true);
        }
    }
}

// ============================================================
// f16 tests — SIMD path
// ============================================================

NNOPS_TEST(reduce_random_f16) {
    auto [f32_vec, _f32] = test::make_random_tensor({5, 10}, -2.0f, 2.0f, 888);
    auto f16_vec = test::f32_to_f16(f32_vec);

    const int64_t shape[] = {5, 10};
    TensorView input(shape, DataType::f16, f16_vec.data());

    struct TestCase { ReduceType type; };
    TestCase cases[] = {
        {ReduceType::Sum}, {ReduceType::Max},
        {ReduceType::Min}, {ReduceType::Mean},
    };

    for (const auto& tc : cases) {
        ReduceAttributes attrs;
        attrs.type = tc.type;
        attrs.axis = 1;
        auto op = Reduce::create(attrs, Backend::CPU);

        auto d = input.desc();
        const TensorDesc arr[] = {d};
        auto descs = op->getOutputTensorDesc(arr);

        NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

        std::vector<nnops::backend::cpu::half> out_buf(static_cast<size_t>(descs[0].numel()));
        auto output = test::make_planar(descs[0], out_buf.data());
        const TensorView ins[] = {input};
        op->compute(output, ins);

        for (int r = 0; r < 5; ++r) {
            float expected = 0.0f;
            switch (tc.type) {
            case ReduceType::Sum:
            case ReduceType::Mean:
                expected = 0.0f;
                for (int c = 0; c < 10; ++c) {
                    expected += f32_vec[static_cast<size_t>(r * 10 + c)];
                }
                if (tc.type == ReduceType::Mean) { expected /= 10.0f; }
                break;
            case ReduceType::Max:
                expected = -std::numeric_limits<float>::infinity();
                for (int c = 0; c < 10; ++c) {
                    if (f32_vec[static_cast<size_t>(r * 10 + c)] > expected) {
                        expected = f32_vec[static_cast<size_t>(r * 10 + c)];
                    }
                }
                break;
            case ReduceType::Min:
                expected = std::numeric_limits<float>::infinity();
                for (int c = 0; c < 10; ++c) {
                    if (f32_vec[static_cast<size_t>(r * 10 + c)] < expected) {
                        expected = f32_vec[static_cast<size_t>(r * 10 + c)];
                    }
                }
                break;
            }
            float result = simd::s_load(&out_buf[static_cast<size_t>(r)]);
            NNOPS_EXPECT_TRUE(std::isfinite(result));
            NNOPS_EXPECT_NEAR(result, expected, 5e-2f);
        }
    }
}
