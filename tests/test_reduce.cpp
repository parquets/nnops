/// @file test_reduce.cpp
/// @brief Tests for the Reduce operator (Sum, Min, Max, Mean).
///
/// Covers: 1D/2D/3D reductions, keepdims, all-axes, class API,
/// functional API, and random data validation against reference.

#include "nnops/ops/reduce.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

using namespace nnops;

// ============================================================
// 1D tests — hand-verified
// ============================================================

NNOPS_TEST(reduce_1d_sum) {
    const int64_t shape[] = {5};
    float in_data[]  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float out_data[1] = {0};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(std::span<const int64_t>(shape, 1), DataType::f32, out_data);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Sum;
    attrs.axes = {0};
    reduce(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 15.0f, 1e-6f);
}

NNOPS_TEST(reduce_1d_max) {
    const int64_t shape[] = {5};
    float in_data[]  = {1.0f, 5.0f, 3.0f, 2.0f, 4.0f};
    float out_data[1] = {0};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(std::span<const int64_t>(shape, 1), DataType::f32, out_data);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Max;
    attrs.axes = {0};
    reduce(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 5.0f, 1e-6f);
}

NNOPS_TEST(reduce_1d_min) {
    const int64_t shape[] = {5};
    float in_data[]  = {3.0f, 1.0f, 5.0f, 2.0f, 4.0f};
    float out_data[1] = {0};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(std::span<const int64_t>(shape, 1), DataType::f32, out_data);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Min;
    attrs.axes = {0};
    reduce(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 1.0f, 1e-6f);
}

NNOPS_TEST(reduce_1d_mean) {
    const int64_t shape[] = {5};
    float in_data[]  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float out_data[1] = {0};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(std::span<const int64_t>(shape, 1), DataType::f32, out_data);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Mean;
    attrs.axes = {0};
    reduce(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 3.0f, 1e-6f);
}

NNOPS_TEST(reduce_1d_keepdims) {
    const int64_t shape[] = {5};
    float in_data[]  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    const int64_t out_shape[] = {1};
    float out_data[1] = {0};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(out_shape, DataType::f32, out_data);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Sum;
    attrs.axes = {0};
    attrs.keepdims = true;
    reduce(input, output, attrs);

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
    const int64_t out_shape[] = {4};
    float out_data[4] = {0};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(out_shape, DataType::f32, out_data);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Sum;
    attrs.axes = {0};
    reduce(input, output, attrs);

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
    const int64_t out_shape[] = {2};
    float out_data[2] = {0};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(out_shape, DataType::f32, out_data);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Max;
    attrs.axes = {1};
    reduce(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 5.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 6.0f, 1e-6f);
}

NNOPS_TEST(reduce_2d_axis1_min) {
    const int64_t shape[] = {2, 3};
    // [[3, 1, 5], [4, 2, 6]]
    float in_data[] = {3, 1, 5, 4, 2, 6};
    const int64_t out_shape[] = {2};
    float out_data[2] = {0};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(out_shape, DataType::f32, out_data);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Min;
    attrs.axes = {1};
    reduce(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 2.0f, 1e-6f);
}

NNOPS_TEST(reduce_2d_axis0_mean) {
    const int64_t shape[] = {2, 3};
    // [[1, 2, 3], [4, 5, 6]]
    float in_data[] = {1, 2, 3, 4, 5, 6};
    const int64_t out_shape[] = {3};
    float out_data[3] = {0};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(out_shape, DataType::f32, out_data);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Mean;
    attrs.axes = {0};
    reduce(input, output, attrs);

    // (1+4)/2=2.5, (2+5)/2=3.5, (3+6)/2=4.5
    NNOPS_EXPECT_NEAR(out_data[0], 2.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 3.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 4.5f, 1e-6f);
}

NNOPS_TEST(reduce_2d_keepdims) {
    const int64_t shape[] = {3, 4};
    float in_data[12] = {};
    for (int i = 0; i < 12; ++i) in_data[i] = static_cast<float>(i + 1);
    const int64_t out_shape[] = {3, 1};
    float out_data[3] = {0};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(out_shape, DataType::f32, out_data);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Sum;
    attrs.axes = {1};
    attrs.keepdims = true;
    reduce(input, output, attrs);

    NNOPS_EXPECT_EQ(output.rank(), 2);
    NNOPS_EXPECT_EQ(output.shape(0), 3);
    NNOPS_EXPECT_EQ(output.shape(1), 1);
    // Row 0: 1+2+3+4=10, Row 1: 5+6+7+8=26, Row 2: 9+10+11+12=42
    NNOPS_EXPECT_NEAR(out_data[0], 10.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 26.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 42.0f, 1e-6f);
}

// ============================================================
// All-axes reduction (axes = empty)
// ============================================================

NNOPS_TEST(reduce_all_axes_sum) {
    const int64_t shape[] = {2, 3, 4};
    auto [in_vec, input] = test::make_random_tensor(shape, -1.0f, 1.0f, 42);

    float out_data[1] = {0};
    const int64_t out_shape[] = {1};
    TensorView output(out_shape, DataType::f32, out_data);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Sum;
    // empty axes → reduce all
    reduce(input, output, attrs);

    // Compute expected sum
    float expected = 0.0f;
    for (float v : in_vec) expected += v;
    NNOPS_EXPECT_NEAR(out_data[0], expected, 1e-4f);
}

NNOPS_TEST(reduce_all_axes_max) {
    const int64_t shape[] = {2, 5};
    auto [in_vec, input] = test::make_random_tensor(shape, -10.0f, 10.0f, 123);

    float out_data[1] = {0};
    const int64_t out_shape[] = {1};
    TensorView output(out_shape, DataType::f32, out_data);

    ReduceAttributes attrs;
    attrs.type = ReduceType::Max;
    reduce(input, output, attrs);

    float expected = -std::numeric_limits<float>::infinity();
    for (float v : in_vec) if (v > expected) expected = v;
    NNOPS_EXPECT_NEAR(out_data[0], expected, 1e-6f);
}

// ============================================================
// Multi-axis reduction
// ============================================================

NNOPS_TEST(reduce_multi_axis) {
    const int64_t shape[] = {2, 3, 4};
    auto [in_vec, input] = test::make_random_tensor(shape, -1.0f, 1.0f, 77);

    // Reduce axes 0 and 2 → output shape = {3}
    const int64_t out_shape[] = {3};
    std::vector<float> out_buf(3);
    TensorView output(out_shape, DataType::f32, out_buf.data());

    ReduceAttributes attrs;
    attrs.type = ReduceType::Sum;
    attrs.axes = {0, 2};
    reduce(input, output, attrs);

    // Manual verification: for each middle dim, sum over first and last
    for (int j = 0; j < 3; ++j) {
        float expected = 0.0f;
        for (int i = 0; i < 2; ++i)
            for (int k = 0; k < 4; ++k)
                expected += in_vec[i * 12 + j * 4 + k];
        NNOPS_EXPECT_NEAR(out_buf[j], expected, 1e-4f);
    }
}

// ============================================================
// Negative axis
// ============================================================

NNOPS_TEST(reduce_negative_axis) {
    const int64_t shape[] = {3, 4, 5};
    auto [in_vec, input] = test::make_random_tensor(shape, 0.0f, 1.0f, 99);

    const int64_t out_shape1[] = {3, 4};
    const int64_t out_shape2[] = {3, 4};
    std::vector<float> out1(12), out2(12);
    TensorView output1(out_shape1, DataType::f32, out1.data());
    TensorView output2(out_shape2, DataType::f32, out2.data());

    ReduceAttributes attrs1;
    attrs1.type = ReduceType::Sum;
    attrs1.axes = {2};  // last axis (positive)
    reduce(input, output1, attrs1);

    ReduceAttributes attrs2;
    attrs2.type = ReduceType::Sum;
    attrs2.axes = {-1};  // last axis (negative)
    reduce(input, output2, attrs2);

    // Results should be identical
    for (int i = 0; i < 12; ++i)
        NNOPS_EXPECT_NEAR(out1[i], out2[i], 1e-6f);
}

// ============================================================
// Class API parity
// ============================================================

NNOPS_TEST(reduce_class_api) {
    const int64_t shape[] = {4, 6};
    auto [in_vec, input] = test::make_random_tensor(shape, -1.0f, 1.0f, 55);

    const int64_t out_shape[] = {4};
    std::vector<float> out1(4), out2(4);
    TensorView output1(out_shape, DataType::f32, out1.data());
    TensorView output2(out_shape, DataType::f32, out2.data());

    ReduceAttributes attrs;
    attrs.type = ReduceType::Mean;
    attrs.axes = {1};

    // Functional API
    reduce(input, output1, attrs);

    // Class API
    auto op = Reduce::create(attrs, Backend::CPU);
    const TensorView ins[] = {input};
    op->compute(output2, ins);

    NNOPS_EXPECT_TRUE(test::allclose(output1, output2));
}

// ============================================================
// Random data tests — all 4 reduction types
// ============================================================

NNOPS_TEST(reduce_random_sum) {
    auto [in_vec, input] = test::make_random_tensor({5, 8}, -10.0f, 10.0f, 200);
    std::vector<float> out_buf(5);
    const int64_t out_shape[] = {5};
    TensorView output(out_shape, DataType::f32, out_buf.data());

    ReduceAttributes attrs;
    attrs.type = ReduceType::Sum;
    attrs.axes = {1};
    reduce(input, output, attrs);

    for (int r = 0; r < 5; ++r) {
        float expected = 0.0f;
        for (int c = 0; c < 8; ++c) expected += in_vec[r * 8 + c];
        NNOPS_EXPECT_NEAR(out_buf[r], expected, 1e-4f);
    }
}

NNOPS_TEST(reduce_random_max) {
    auto [in_vec, input] = test::make_random_tensor({3, 10}, -100.0f, 100.0f, 300);
    std::vector<float> out_buf(10);
    const int64_t out_shape[] = {10};
    TensorView output(out_shape, DataType::f32, out_buf.data());

    ReduceAttributes attrs;
    attrs.type = ReduceType::Max;
    attrs.axes = {0};
    reduce(input, output, attrs);

    for (int c = 0; c < 10; ++c) {
        float expected = -std::numeric_limits<float>::infinity();
        for (int r = 0; r < 3; ++r)
            if (in_vec[r * 10 + c] > expected) expected = in_vec[r * 10 + c];
        NNOPS_EXPECT_NEAR(out_buf[c], expected, 1e-6f);
    }
}

NNOPS_TEST(reduce_random_min) {
    auto [in_vec, input] = test::make_random_tensor({4, 7}, -50.0f, 50.0f, 400);
    std::vector<float> out_buf(7);
    const int64_t out_shape[] = {7};
    TensorView output(out_shape, DataType::f32, out_buf.data());

    ReduceAttributes attrs;
    attrs.type = ReduceType::Min;
    attrs.axes = {0};
    reduce(input, output, attrs);

    for (int c = 0; c < 7; ++c) {
        float expected = std::numeric_limits<float>::infinity();
        for (int r = 0; r < 4; ++r)
            if (in_vec[r * 7 + c] < expected) expected = in_vec[r * 7 + c];
        NNOPS_EXPECT_NEAR(out_buf[c], expected, 1e-6f);
    }
}

NNOPS_TEST(reduce_random_mean) {
    auto [in_vec, input] = test::make_random_tensor({6, 5}, -20.0f, 20.0f, 500);
    std::vector<float> out_buf(6);
    const int64_t out_shape[] = {6};
    TensorView output(out_shape, DataType::f32, out_buf.data());

    ReduceAttributes attrs;
    attrs.type = ReduceType::Mean;
    attrs.axes = {1};
    reduce(input, output, attrs);

    for (int r = 0; r < 6; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < 5; ++c) sum += in_vec[r * 5 + c];
        NNOPS_EXPECT_NEAR(out_buf[r], sum / 5.0f, 1e-4f);
    }
}

// ============================================================
// 3D tests
// ============================================================

NNOPS_TEST(reduce_3d_middle_axis) {
    const int64_t shape[] = {2, 3, 4};
    auto [in_vec, input] = test::make_random_tensor(shape, -1.0f, 1.0f, 600);

    const int64_t out_shape[] = {2, 4};
    std::vector<float> out_buf(8);
    TensorView output(out_shape, DataType::f32, out_buf.data());

    ReduceAttributes attrs;
    attrs.type = ReduceType::Sum;
    attrs.axes = {1};
    reduce(input, output, attrs);

    for (int i = 0; i < 2; ++i) {
        for (int k = 0; k < 4; ++k) {
            float expected = 0.0f;
            for (int j = 0; j < 3; ++j)
                expected += in_vec[i * 12 + j * 4 + k];
            NNOPS_EXPECT_NEAR(out_buf[i * 4 + k], expected, 1e-4f);
        }
    }
}

NNOPS_TEST(reduce_3d_first_axis) {
    const int64_t shape[] = {3, 2, 5};
    auto [in_vec, input] = test::make_random_tensor(shape, -5.0f, 5.0f, 700);

    const int64_t out_shape[] = {2, 5};
    std::vector<float> out_buf(10);
    TensorView output(out_shape, DataType::f32, out_buf.data());

    ReduceAttributes attrs;
    attrs.type = ReduceType::Mean;
    attrs.axes = {0};
    reduce(input, output, attrs);

    for (int j = 0; j < 2; ++j) {
        for (int k = 0; k < 5; ++k) {
            float sum = 0.0f;
            for (int i = 0; i < 3; ++i)
                sum += in_vec[i * 10 + j * 5 + k];
            NNOPS_EXPECT_NEAR(out_buf[j * 5 + k], sum / 3.0f, 1e-4f);
        }
    }
}
