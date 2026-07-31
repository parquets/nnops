/// @file test_reduce.cpp
/// @brief Tests for the Reduce operator (Sum, Min, Max, Mean).
///
/// Covers: 1D/2D/3D reductions, keepdims, all-axes, class API,
/// and random data validation against reference.

#include "nnops/ops/reduce.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"

using namespace nnops;

// ============================================================
// 1D tests — hand-verified
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
// 2D tests — hand-verified
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

    // Results should be identical
    for (size_t i = 0; i < out1.size(); ++i) {
        NNOPS_EXPECT_NEAR(out1[i], out2[i], 1e-6f);
    }
}

// ============================================================
// Random data tests — all 4 reduction types
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
