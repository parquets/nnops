/// Unit tests for RMSNorm operator (CPU reference).

#include "nnops/ops/rms_norm.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

NNOPS_TEST(rmsnorm_simple_last_axis) {
    const int64_t shape[] = {4};
    const int64_t scale_shape[] = {4};

    float x_data[]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float s_data[]  = {1.0f, 1.0f, 1.0f, 1.0f};
    float out_data[4] = {};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);
    TensorView y(shape, DataType::f32, out_data);

    rms_norm(x, s, y);

    // rms = sqrt(mean(x^2) + eps) = sqrt((1+4+9+16)/4 + 1e-5) = sqrt(7.5 + 1e-5)
    float sum_sq = 1.0f + 4.0f + 9.0f + 16.0f;
    float rms = std::sqrt(sum_sq / 4.0f + 1e-5f);
    for (int i = 0; i < 4; ++i) {
        float expected = x_data[i] / rms;
        NNOPS_EXPECT_NEAR(out_data[i], expected, 1e-3f);
    }
}

NNOPS_TEST(rmsnorm_with_scale) {
    const int64_t shape[] = {2};
    const int64_t scale_shape[] = {2};

    float x_data[]  = {2.0f, 4.0f};
    float s_data[]  = {0.5f, 2.0f};
    float out_data[2] = {};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);
    TensorView y(shape, DataType::f32, out_data);

    rms_norm(x, s, y);

    // rms = sqrt((4 + 16) / 2 + 1e-5) = sqrt(10 + 1e-5)
    float rms = std::sqrt(10.0f + 1e-5f);
    NNOPS_EXPECT_NEAR(out_data[0], 2.0f / rms * 0.5f, 1e-3f);
    NNOPS_EXPECT_NEAR(out_data[1], 4.0f / rms * 2.0f, 1e-3f);
}

NNOPS_TEST(rmsnorm_2d_axis_1) {
    const int64_t shape[] = {2, 2};
    const int64_t scale_shape[] = {2};

    float x_data[]  = {1.0f, 1.0f,
                       2.0f, 0.0f};
    float s_data[]  = {1.0f, 1.0f};
    float out_data[4] = {};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);
    TensorView y(shape, DataType::f32, out_data);

    RMSNormAttributes attrs;
    attrs.axis = 1;
    rms_norm(x, s, y, attrs);

    // Row 0: [1,1] → rms = sqrt((1+1)/2 + eps) = sqrt(1 + eps) ≈ 1
    // Row 1: [2,0] → rms = sqrt((4+0)/2 + eps) = sqrt(2 + eps)
    float rms0 = std::sqrt(1.0f + 1e-5f);
    float rms1 = std::sqrt(2.0f + 1e-5f);
    NNOPS_EXPECT_NEAR(out_data[0], 1.0f / rms0, 1e-3f);
    NNOPS_EXPECT_NEAR(out_data[1], 1.0f / rms0, 1e-3f);
    NNOPS_EXPECT_NEAR(out_data[2], 2.0f / rms1, 1e-3f);
    NNOPS_EXPECT_NEAR(out_data[3], 0.0f, 1e-3f);
}

NNOPS_TEST(rmsnorm_random) {
    auto [in_vec, input] = test::make_random_tensor({3, 6}, -1.0f, 1.0f);
    std::vector<float> scale_buf(6, 1.0f);
    std::vector<float> out_buf(18);

    int64_t shape_x[] = {3, 6};
    int64_t shape_s[] = {6};

    TensorView x(shape_x, DataType::f32, in_vec.data());
    TensorView s(shape_s, DataType::f32, scale_buf.data());
    TensorView y(shape_x, DataType::f32, out_buf.data());

    rms_norm(x, s, y);

    // With scale=1: each row's RMS should be ≈ 1
    for (int r = 0; r < 3; ++r) {
        float sum_sq = 0.0f;
        for (int c = 0; c < 6; ++c) {
            sum_sq += y.data_as<float>()[r * 6 + c] *
                      y.data_as<float>()[r * 6 + c];
        }
        float rms = std::sqrt(sum_sq / 6.0f);
        NNOPS_EXPECT_NEAR(rms, 1.0f, 0.1f);
    }
}

NNOPS_TEST(rmsnorm_class_api) {
    const int64_t shape[] = {4};
    float x_data[]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float s_data[]  = {1.0f, 1.0f, 1.0f, 1.0f};
    float out1[4] = {}, out2[4] = {};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(shape, DataType::f32, s_data);
    TensorView y1(shape, DataType::f32, out1);
    TensorView y2(shape, DataType::f32, out2);

    rms_norm(x, s, y1);

    auto op = RMSNorm::create(Backend::CPU);
    const TensorView ins[] = {x, s};
    op->compute(y2, ins);

    NNOPS_EXPECT_TRUE(test::allclose(y1, y2));
}
