/// Unit tests for Softmax operator (CPU reference).

#include "nnops/ops/softmax.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <cmath>
#include <algorithm>

using namespace nnops;

NNOPS_TEST(softmax_last_axis) {
    // 1D input, softmax along axis 0
    const int64_t shape[] = {3};
    float in_data[]  = {1.0f, 2.0f, 3.0f};
    float out_data[3] = {};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(shape, DataType::f32, out_data);

    softmax(input, output);  // default axis=-1 → axis 0 for 1D

    // exp(1)=2.718, exp(2)=7.389, exp(3)=20.086 → sum=30.193
    float sum_exp = 0.0f;
    for (int i = 0; i < 3; ++i) sum_exp += std::exp(in_data[i]);
    NNOPS_EXPECT_NEAR(out_data[0], std::exp(1.0f) / sum_exp, 1e-4f);
    NNOPS_EXPECT_NEAR(out_data[1], std::exp(2.0f) / sum_exp, 1e-4f);
    NNOPS_EXPECT_NEAR(out_data[2], std::exp(3.0f) / sum_exp, 1e-4f);
    // Should sum to 1
    NNOPS_EXPECT_NEAR(out_data[0] + out_data[1] + out_data[2], 1.0f, 1e-5f);
}

NNOPS_TEST(softmax_2d_axis_1) {
    // 2x3, softmax along axis=1 (normalize rows)
    const int64_t shape[] = {2, 3};
    float in_data[]  = {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f};
    float out_data[6] = {};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(shape, DataType::f32, out_data);

    SoftmaxAttributes attrs;
    attrs.axis = 1;
    softmax(input, output, attrs);

    // Row 0: same as 1D test
    float sum0 = 0.0f;
    for (int i = 0; i < 3; ++i) sum0 += std::exp(in_data[i]);
    NNOPS_EXPECT_NEAR(out_data[0], std::exp(1.0f) / sum0, 1e-4f);
    NNOPS_EXPECT_NEAR(out_data[1], std::exp(2.0f) / sum0, 1e-4f);
    NNOPS_EXPECT_NEAR(out_data[2], std::exp(3.0f) / sum0, 1e-4f);
    // Row 1: all equal → each gets 1/3
    NNOPS_EXPECT_NEAR(out_data[3], 1.0f / 3.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(out_data[4], 1.0f / 3.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(out_data[5], 1.0f / 3.0f, 1e-4f);
}

NNOPS_TEST(softmax_log_softmax) {
    const int64_t shape[] = {2};
    float in_data[]  = {0.0f, 1.0f};
    float out_data[2] = {};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(shape, DataType::f32, out_data);

    SoftmaxAttributes attrs;
    attrs.log_softmax = true;
    softmax(input, output, attrs);

    // log_softmax(0,1): max=1, shifted={-1,0}, exp={0.3679,1}, sum=1.3679, log_sum=0.3133
    // out[0] = -1 - 0.3133 = -1.3133, out[1] = 0 - 0.3133 = -0.3133
    float max_val = std::max(in_data[0], in_data[1]);
    float shifted0 = in_data[0] - max_val;
    float shifted1 = in_data[1] - max_val;
    float log_sum = std::log(std::exp(shifted0) + std::exp(shifted1));
    NNOPS_EXPECT_NEAR(out_data[0], shifted0 - log_sum, 1e-4f);
    NNOPS_EXPECT_NEAR(out_data[1], shifted1 - log_sum, 1e-4f);
}

NNOPS_TEST(softmax_random) {
    auto [in_vec, input] = test::make_random_tensor({4, 8}, -2.0f, 2.0f);
    std::vector<float> out_buf(32);
    TensorView output(input.shape_span(), DataType::f32, out_buf.data());

    SoftmaxAttributes attrs;
    attrs.axis = 1;
    softmax(input, output, attrs);

    // Each row should sum to 1
    for (int r = 0; r < 4; ++r) {
        float row_sum = 0.0f;
        for (int c = 0; c < 8; ++c) {
            row_sum += out_buf[r * 8 + c];
            NNOPS_EXPECT_TRUE(out_buf[r * 8 + c] >= 0.0f);
        }
        NNOPS_EXPECT_NEAR(row_sum, 1.0f, 1e-4f);
    }
}

NNOPS_TEST(softmax_class_api) {
    const int64_t shape[] = {4};
    float in_data[]  = {-1.0f, 0.0f, 1.0f, 2.0f};
    float out1[4] = {}, out2[4] = {};

    TensorView input(shape, DataType::f32, in_data);
    TensorView out1_view(shape, DataType::f32, out1);
    TensorView out2_view(shape, DataType::f32, out2);

    // Functional
    softmax(input, out1_view);
    // Class
    auto op = Softmax::create(Backend::CPU);
    const TensorView ins[] = {input};
    op->compute(out2_view, ins);

    NNOPS_EXPECT_TRUE(test::allclose(out1_view, out2_view));
}
