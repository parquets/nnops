/// @file test_reduce.cpp
/// @brief Tests for the Reduce operator (Sum, Min, Max, Mean).
///
/// Covers: 1D/2D/3D reductions, keepdims, all-axes, class API,
/// and random data validation against reference.

#include "nnops/ops/reduce.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"

#include <random>

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

// ============================================================
// Packed layout tests (NCHWC8)
//
// NCHWC8 is rank 4: [N, C, H, W] where C is the LOGICAL channel
// count. Physical row layout: [w0_l0..w0_l7, w1_l0..w1_l7, ...].
// num_channel_blocks = ceil(C / 8), row_stride = W * 8 elements.
// total_rows = N * num_channel_blocks * H.
// ============================================================

/// Helper: create NCHWC8 TensorView with correct pitch, plus buffer.
/// Creates a rank-4 tensor [N, C, H, W] with NCHWC8 layout.
static std::pair<std::vector<float>, TensorView>
make_nchwc8_4d(int64_t N, int64_t C, int64_t H, int64_t W)
{
    TensorDesc desc;
    desc.rank = 4;
    desc.dims = {N, C, H, W};
    desc.dtype = DataType::f32;
    desc.layout = TensorLayout::NCHWC8;

    size_t nbytes = desc.storage_bytes();
    std::vector<float> buf(nbytes / sizeof(float));
    TensorView tv = test::make_packed(desc, buf.data());
    return {std::move(buf), tv};
}

/// Helper: create an NCHWC8 output tensor from a descriptor (handles any rank).
static std::pair<std::vector<float>, TensorView>
make_nchwc8_out(const TensorDesc& desc)
{
    size_t nbytes = desc.storage_bytes();
    std::vector<float> buf(nbytes / sizeof(float));
    TensorView tv = test::make_packed(desc, buf.data());
    return {std::move(buf), tv};
}

/// Helper: fill NCHWC8 data with sequential values.
static void fill_nchwc8(float* data,
                         int64_t N, int64_t C, int64_t H, int64_t W,
                         float base_val, float step)
{
    int64_t num_c8 = (C + 7) / 8;
    int64_t row_stride = W * 8;
    int64_t c8_stride = H * row_stride;
    int64_t n_stride = num_c8 * c8_stride;
    float val = base_val;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c8 = 0; c8 < num_c8; ++c8) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    for (int64_t l = 0; l < 8; ++l) {
                        int64_t off = n * n_stride + c8 * c8_stride
                                      + h * row_stride + w * 8 + l;
                        data[off] = val;
                        val += step;
                    }
                }
            }
        }
    }
}

/// Helper: read a value from NCHWC8 physical storage at logical (n, c, h, w).
static float read_nchwc8(const float* data,
                          int64_t N, int64_t C, int64_t H, int64_t W,
                          int64_t n, int64_t c, int64_t h, int64_t w)
{
    int64_t num_c8 = (C + 7) / 8;
    int64_t c8 = c / 8;
    int64_t lane = c % 8;
    int64_t row_stride = W * 8;
    int64_t c8_stride = H * row_stride;
    int64_t n_stride = num_c8 * c8_stride;
    int64_t off = n * n_stride + c8 * c8_stride + h * row_stride + w * 8 + lane;
    return data[off];
}

NNOPS_TEST(reduce_nchwc8_axis_w_sum) {
    // NCHWC8 [1, 8, 1, 4] — one full C8 block (C=8), W=4
    // Reduce over W (axis=-1) with Sum. Each C lane is independently summed.
    const int64_t N = 1, C = 8, H = 1, W = 4;

    auto [in_data, input] = make_nchwc8_4d(N, C, H, W);
    fill_nchwc8(in_data.data(), N, C, H, W, 1.0f, 1.0f);

    auto d_in = input.desc();

    ReduceAttributes attrs;
    attrs.type = ReduceType::Sum;
    attrs.axis = -1;  // W dimension (rank-1)
    auto op = Reduce::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 3);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 1);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 8);
    NNOPS_EXPECT_EQ(descs[0].dims[2], 1);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHWC8);

    auto [out_buf, output] = make_nchwc8_out(descs[0]);

    const TensorView ins[] = {input};
    op->compute(output, ins);

    // Output is rank 3 [1, 8, 1] NCHWC8.
    // Physical storage: 8 elements at offset 0 (one position with 8 lanes).
    // Lane l values at input: fill(0)+l=1+l, fill(8)+l=9+l, fill(16)+l=17+l, fill(24)+l=25+l
    // Sum per lane: (1+l)+(9+l)+(17+l)+(25+l) = 4*l + 52
    for (int lane = 0; lane < 8; ++lane) {
        float expected = 4.0f * static_cast<float>(lane) + 52.0f;
        NNOPS_EXPECT_NEAR(out_buf[static_cast<size_t>(lane)], expected, 1e-4f);
    }

    // Also verify per-lane against direct computation from input
    for (int lane = 0; lane < 8; ++lane) {
        float sum = 0.0f;
        for (int w = 0; w < 4; ++w) {
            sum += read_nchwc8(in_data.data(), N, C, H, W, 0, lane, 0, w);
        }
        NNOPS_EXPECT_NEAR(out_buf[static_cast<size_t>(lane)], sum, 1e-4f);
    }
}

NNOPS_TEST(reduce_nchwc8_axis_w_max) {
    // NCHWC8 [1, 8, 1, 3] — per-lane max over W
    const int64_t N = 1, C = 8, H = 1, W = 3;

    auto [in_data, input] = make_nchwc8_4d(N, C, H, W);
    // Fill with deliberate values so per-lane maxes are predictable
    int64_t num_c8 = (C + 7) / 8;
    int64_t row_stride = W * 8;
    int64_t c8_stride = H * row_stride;
    int64_t n_stride = num_c8 * c8_stride;
    for (int lane = 0; lane < 8; ++lane) {
        for (int w = 0; w < W; ++w) {
            int64_t off = 0 * n_stride + 0 * c8_stride + 0 * row_stride + w * 8 + lane;
            // Lane l: values 10*l+w+1, so max is 10*l+W at w=W-1
            in_data[off] = 10.0f * static_cast<float>(lane) + static_cast<float>(w) + 1.0f;
        }
    }

    auto d_in = input.desc();

    ReduceAttributes attrs;
    attrs.type = ReduceType::Max;
    attrs.axis = -1;
    auto op = Reduce::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_arr);

    auto [out_buf, output] = make_nchwc8_out(descs[0]);

    const TensorView ins[] = {input};
    op->compute(output, ins);

    for (int lane = 0; lane < 8; ++lane) {
        float expected = 10.0f * static_cast<float>(lane) + 3.0f; // max at w=2
        NNOPS_EXPECT_NEAR(out_buf[static_cast<size_t>(lane)], expected, 1e-4f);
    }
}

NNOPS_TEST(reduce_nchwc8_axis_w_mean) {
    // NCHWC8 [1, 8, 1, 4] — per-lane mean over W
    const int64_t N = 1, C = 8, H = 1, W = 4;

    auto [in_data, input] = make_nchwc8_4d(N, C, H, W);
    fill_nchwc8(in_data.data(), N, C, H, W, 1.0f, 1.0f);

    auto d_in = input.desc();

    ReduceAttributes attrs;
    attrs.type = ReduceType::Mean;
    attrs.axis = -1;
    auto op = Reduce::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_arr);

    auto [out_buf, output] = make_nchwc8_out(descs[0]);

    const TensorView ins[] = {input};
    op->compute(output, ins);

    // Per-lane means: ((1+l) + (9+l) + (17+l) + (25+l)) / 4 = (4*l + 52) / 4 = l + 13
    for (int lane = 0; lane < 8; ++lane) {
        float expected = static_cast<float>(lane) + 13.0f;
        NNOPS_EXPECT_NEAR(out_buf[static_cast<size_t>(lane)], expected, 1e-4f);
    }
}

NNOPS_TEST(reduce_nchwc8_axis_w_min) {
    // NCHWC8 [1, 8, 1, 3] — per-lane min over W
    const int64_t N = 1, C = 8, H = 1, W = 3;

    auto [in_data, input] = make_nchwc8_4d(N, C, H, W);
    int64_t num_c8 = (C + 7) / 8;
    int64_t row_stride = W * 8;
    int64_t c8_stride = H * row_stride;
    int64_t n_stride = num_c8 * c8_stride;
    for (int lane = 0; lane < 8; ++lane) {
        for (int w = 0; w < W; ++w) {
            int64_t off = 0 * n_stride + 0 * c8_stride + 0 * row_stride + w * 8 + lane;
            in_data[off] = 20.0f - static_cast<float>(w) * static_cast<float>(lane + 1);
        }
    }

    ReduceAttributes attrs;
    attrs.type = ReduceType::Min;
    attrs.axis = -1;
    auto op = Reduce::create(attrs, Backend::CPU);

    const TensorDesc in_arr2[] = {input.desc()};
    auto descs2 = op->getOutputTensorDesc(in_arr2);
    auto [out_buf, output] = make_nchwc8_out(descs2[0]);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    for (int lane = 0; lane < 8; ++lane) {
        // Min is at w=2: 20 - 2*(lane+1) = 18 - 2*lane
        float expected = 20.0f - 2.0f * static_cast<float>(lane + 1);
        NNOPS_EXPECT_NEAR(out_buf[static_cast<size_t>(lane)], expected, 1e-4f);
    }
}

NNOPS_TEST(reduce_nchwc8_random) {
    // Random NCHWC8 [2, 16, 3, 5] — two full C8 blocks (C=16).
    const int64_t N = 2, C = 16, H = 3, W = 5;
    const int64_t num_c8 = (C + 7) / 8;
    const int64_t row_stride = W * 8;

    auto [in_data, input] = make_nchwc8_4d(N, C, H, W);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (auto& v : in_data) {
        v = dist(rng);
    }

    int64_t n_stride = num_c8 * H * row_stride;
    int64_t c8_stride = H * row_stride;

    // Test all four reduce types over W (axis=-1)
    struct TestCase { ReduceType type; };
    TestCase cases[] = {
        {ReduceType::Sum},
        {ReduceType::Max},
        {ReduceType::Min},
        {ReduceType::Mean},
    };

    for (const auto& tc : cases) {
        ReduceAttributes attrs;
        attrs.type = tc.type;
        attrs.axis = -1;
        auto op = Reduce::create(attrs, Backend::CPU);

        auto d_in2 = input.desc();
        const TensorDesc in_arr2[] = {d_in2};
        auto descs2 = op->getOutputTensorDesc(in_arr2);
        auto [out_buf, output] = make_nchwc8_out(descs2[0]);
        const TensorView ins[] = {input};
        op->compute(output, ins);

        // Per-lane verification: for each (n, c8, h, lane), reduce over W.
        // Output is rank 3 [N, C, H] NCHWC8, densely packed.
        int64_t out_positions = static_cast<int64_t>(N) * num_c8 * H;
        for (int64_t pos = 0; pos < out_positions; ++pos) {
            int64_t n = pos / (num_c8 * H);
            int64_t c8 = (pos / H) % num_c8;
            int64_t h = pos % H;
            for (int lane = 0; lane < 8; ++lane) {
                // Gather all W values for this lane
                float result;
                switch (tc.type) {
                case ReduceType::Sum: {
                    float sum = 0.0f;
                    for (int64_t w = 0; w < W; ++w) {
                        int64_t off = n * n_stride + c8 * c8_stride
                                      + h * row_stride + w * 8 + lane;
                        sum += in_data[off];
                    }
                    result = sum;
                    break;
                }
                case ReduceType::Max: {
                    float best = -std::numeric_limits<float>::infinity();
                    for (int64_t w = 0; w < W; ++w) {
                        int64_t off = n * n_stride + c8 * c8_stride
                                      + h * row_stride + w * 8 + lane;
                        if (in_data[off] > best) {
                            best = in_data[off];
                        }
                    }
                    result = best;
                    break;
                }
                case ReduceType::Min: {
                    float best = std::numeric_limits<float>::infinity();
                    for (int64_t w = 0; w < W; ++w) {
                        int64_t off = n * n_stride + c8 * c8_stride
                                      + h * row_stride + w * 8 + lane;
                        if (in_data[off] < best) {
                            best = in_data[off];
                        }
                    }
                    result = best;
                    break;
                }
                case ReduceType::Mean: {
                    float sum = 0.0f;
                    for (int64_t w = 0; w < W; ++w) {
                        int64_t off = n * n_stride + c8 * c8_stride
                                      + h * row_stride + w * 8 + lane;
                        sum += in_data[off];
                    }
                    result = sum / static_cast<float>(W);
                    break;
                }
                }
                int64_t out_off = pos * 8 + lane;
                NNOPS_EXPECT_NEAR(out_buf[static_cast<size_t>(out_off)], result, 1e-4f);
            }
        }
    }
}

NNOPS_TEST(reduce_nchwc8_keepdims) {
    // NCHWC8 [1, 8, 2, 3] — reduce over W with keepdims=true
    const int64_t N = 1, C = 8, H = 2, W = 3;

    auto [in_data, input] = make_nchwc8_4d(N, C, H, W);
    fill_nchwc8(in_data.data(), N, C, H, W, 0.0f, 1.0f);

    auto d_in = input.desc();

    ReduceAttributes attrs;
    attrs.type = ReduceType::Sum;
    attrs.axis = -1;
    attrs.keepdims = true;
    auto op = Reduce::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 4);
    NNOPS_EXPECT_EQ(descs[0].dims[3], 1);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHWC8);

    auto [out_buf, output] = make_nchwc8_out(descs[0]);

    const TensorView ins[] = {input};
    op->compute(output, ins);

    // Verify per-lane sums for each (n, c8, h) position
    int64_t num_c8 = (C + 7) / 8;
    int64_t in_row_stride = W * 8;
    int64_t in_c8_stride = H * in_row_stride;
    int64_t in_n_stride = num_c8 * in_c8_stride;
    int64_t out_row_stride = 1 * 8;
    int64_t out_c8_stride = H * out_row_stride;
    int64_t out_n_stride = num_c8 * out_c8_stride;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c8 = 0; c8 < num_c8; ++c8) {
            for (int64_t h = 0; h < H; ++h) {
                for (int lane = 0; lane < 8; ++lane) {
                    float sum = 0.0f;
                    for (int64_t w = 0; w < W; ++w) {
                        int64_t off = n * in_n_stride + c8 * in_c8_stride
                                      + h * in_row_stride + w * 8 + lane;
                        sum += in_data[off];
                    }
                    int64_t out_off = n * out_n_stride + c8 * out_c8_stride
                                      + h * out_row_stride + lane;
                    NNOPS_EXPECT_NEAR(out_buf[out_off], sum, 1e-4f);
                }
            }
        }
    }
}
