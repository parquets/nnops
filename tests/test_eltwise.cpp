/// @file test_eltwise.cpp
/// @brief Unit tests for Eltwise operator (CPU backend).

#include "nnops/ops/eltwise.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

// ============================================================
// Hand-verified tests
// ============================================================

NNOPS_TEST(eltwise_add_1d) {
    const int64_t shape[] = {4};
    float a_data[]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[]  = {5.0f, 6.0f, 7.0f, 8.0f};
    float out_data[4] = {};

    TensorView a(shape, DataType::f32, a_data);
    TensorView b(shape, DataType::f32, b_data);
    TensorView output(shape, DataType::f32, out_data);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Add;
    eltwise(a, b, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 6.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 8.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 10.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[3], 12.0f, 1e-6f);
}

NNOPS_TEST(eltwise_sub_1d) {
    const int64_t shape[] = {3};
    float a_data[]  = {10.0f, 20.0f, 30.0f};
    float b_data[]  = {3.0f, 5.0f, 7.0f};
    float out_data[3] = {};

    TensorView a(shape, DataType::f32, a_data);
    TensorView b(shape, DataType::f32, b_data);
    TensorView output(shape, DataType::f32, out_data);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Sub;
    eltwise(a, b, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 7.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 15.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 23.0f, 1e-6f);
}

NNOPS_TEST(eltwise_mul_1d) {
    const int64_t shape[] = {4};
    float a_data[]  = {2.0f, 3.0f, 4.0f, 5.0f};
    float b_data[]  = {3.0f, 4.0f, 5.0f, 6.0f};
    float out_data[4] = {};

    TensorView a(shape, DataType::f32, a_data);
    TensorView b(shape, DataType::f32, b_data);
    TensorView output(shape, DataType::f32, out_data);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Mul;
    eltwise(a, b, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 6.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 12.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 20.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[3], 30.0f, 1e-6f);
}

NNOPS_TEST(eltwise_div_1d) {
    const int64_t shape[] = {3};
    float a_data[]  = {6.0f, 12.0f, 20.0f};
    float b_data[]  = {2.0f, 3.0f, 4.0f};
    float out_data[3] = {};

    TensorView a(shape, DataType::f32, a_data);
    TensorView b(shape, DataType::f32, b_data);
    TensorView output(shape, DataType::f32, out_data);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Div;
    eltwise(a, b, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 4.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 5.0f, 1e-6f);
}

NNOPS_TEST(eltwise_add_to) {
    const int64_t shape[] = {3};
    float a_data[]  = {2.0f, 3.0f, 4.0f};
    float b_data[]  = {1.0f, 2.0f, 3.0f};
    float out_data[] = {10.0f, 20.0f, 30.0f};  // initial values

    TensorView a(shape, DataType::f32, a_data);
    TensorView b(shape, DataType::f32, b_data);
    TensorView output(shape, DataType::f32, out_data);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Add;
    attrs.add_to = true;
    eltwise(a, b, output, attrs);

    // output = original + (a + b)
    NNOPS_EXPECT_NEAR(out_data[0], 10.0f + 2.0f + 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 20.0f + 3.0f + 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 30.0f + 4.0f + 3.0f, 1e-6f);
}

// ============================================================
// Random data tests
// ============================================================

NNOPS_TEST(eltwise_random_add) {
    auto [a_vec, a] = test::make_random_tensor({1000}, -10.0f, 10.0f, 42);
    auto [b_vec, b] = test::make_random_tensor({1000}, -10.0f, 10.0f, 99);
    std::vector<float> out_buf(1000);
    TensorView output(a.shape_span(), DataType::f32, out_buf.data());

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Add;
    eltwise(a, b, output, attrs);

    for (int i = 0; i < 1000; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], a_vec[i] + b_vec[i], 1e-4f);
    }
}

NNOPS_TEST(eltwise_random_mul) {
    auto [a_vec, a] = test::make_random_tensor({500}, -2.0f, 2.0f, 77);
    auto [b_vec, b] = test::make_random_tensor({500}, -2.0f, 2.0f, 88);
    std::vector<float> out_buf(500);
    TensorView output(a.shape_span(), DataType::f32, out_buf.data());

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Mul;
    eltwise(a, b, output, attrs);

    for (int i = 0; i < 500; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], a_vec[i] * b_vec[i], 1e-4f);
    }
}

// ============================================================
// 2D tests
// ============================================================

NNOPS_TEST(eltwise_2d_add) {
    const int64_t shape[] = {2, 3};
    float a_data[] = {1, 2, 3, 4, 5, 6};
    float b_data[] = {6, 5, 4, 3, 2, 1};
    float out_data[6] = {};

    TensorView a(shape, DataType::f32, a_data);
    TensorView b(shape, DataType::f32, b_data);
    TensorView output(shape, DataType::f32, out_data);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Add;
    eltwise(a, b, output, attrs);

    for (int i = 0; i < 6; ++i) {
        NNOPS_EXPECT_NEAR(out_data[i], 7.0f, 1e-6f);
    }
}

// ============================================================
// Class API parity
// ============================================================

NNOPS_TEST(eltwise_class_api) {
    const int64_t shape[] = {4};
    float a_data[]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[]  = {1.0f, 1.0f, 1.0f, 1.0f};
    float out1[4] = {}, out2[4] = {};

    TensorView a(shape, DataType::f32, a_data);
    TensorView b(shape, DataType::f32, b_data);
    TensorView out1_view(shape, DataType::f32, out1);
    TensorView out2_view(shape, DataType::f32, out2);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Add;

    // Functional API
    eltwise(a, b, out1_view, attrs);

    // Class API
    auto op = Eltwise::create(attrs, Backend::CPU);
    const TensorView ins[] = {a, b};
    op->compute(out2_view, ins);

    NNOPS_EXPECT_TRUE(test::allclose(out1_view, out2_view));
}
