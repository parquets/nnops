/// Unit tests for LayerNorm operator (CPU reference).

#include "nnops/ops/layer_norm.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

NNOPS_TEST(layernorm_simple_last_axis) {
    // 1x4, normalize last dim
    const int64_t shape[] = {4};
    const int64_t scale_shape[] = {4};

    float x_data[]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float s_data[]  = {1.0f, 1.0f, 1.0f, 1.0f};
    float out_data[4] = {};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);
    TensorView y(shape, DataType::f32, out_data);

    layer_norm(x, s, y);

    // mean = 2.5, var = 1.25
    float mean = 2.5f;
    float var = 1.25f;
    float eps = 1e-5f;
    float inv_std = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < 4; ++i) {
        float expected = (x_data[i] - mean) * inv_std;
        NNOPS_EXPECT_NEAR(out_data[i], expected, 1e-3f);
    }
}

NNOPS_TEST(layernorm_with_bias) {
    const int64_t shape[] = {2};
    const int64_t scale_shape[] = {2};

    float x_data[]  = {0.0f, 2.0f};
    float s_data[]  = {2.0f, 3.0f};
    float b_data[]  = {1.0f, -1.0f};
    float out_data[2] = {};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);
    TensorView b(scale_shape, DataType::f32, b_data);
    TensorView y(shape, DataType::f32, out_data);

    layer_norm(x, s, b, y);

    // mean = 1.0, var = 1.0 (x=[0,2] → mean=1, squared diffs: 1+1=2, var=2/2=1)
    float mean = 1.0f;
    float var = 1.0f;
    float inv_std = 1.0f / std::sqrt(var + 1e-5f);
    for (int i = 0; i < 2; ++i) {
        float norm = (x_data[i] - mean) * inv_std;
        float expected = norm * s_data[i] + b_data[i];
        NNOPS_EXPECT_NEAR(out_data[i], expected, 1e-3f);
    }
}

NNOPS_TEST(layernorm_2d_axis_1) {
    // 2x3, normalize along axis=1 (normalize each row)
    const int64_t shape[] = {2, 3};
    const int64_t scale_shape[] = {3};

    float x_data[]  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float s_data[]  = {1.0f, 1.0f, 1.0f};
    float out_data[6] = {};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);
    TensorView y(shape, DataType::f32, out_data);

    LayerNormAttributes attrs;
    attrs.axis = 1;
    layer_norm(x, s, y, attrs);

    // Row 0: mean=(1+2+3)/3=2, var=((1-2)^2+(2-2)^2+(3-2)^2)/3=(1+0+1)/3=0.6667
    // Row 1: mean=(4+5+6)/3=5, var=((4-5)^2+(5-5)^2+(6-5)^2)/3=0.6667
    float var = (1.0f + 0.0f + 1.0f) / 3.0f;  // 2/3
    float inv_std = 1.0f / std::sqrt(var + 1e-5f);

    NNOPS_EXPECT_NEAR(out_data[0], (1.0f - 2.0f) * inv_std, 1e-3f);
    NNOPS_EXPECT_NEAR(out_data[1], (2.0f - 2.0f) * inv_std, 1e-3f);
    NNOPS_EXPECT_NEAR(out_data[2], (3.0f - 2.0f) * inv_std, 1e-3f);
    NNOPS_EXPECT_NEAR(out_data[3], (4.0f - 5.0f) * inv_std, 1e-3f);
    NNOPS_EXPECT_NEAR(out_data[4], (5.0f - 5.0f) * inv_std, 1e-3f);
    NNOPS_EXPECT_NEAR(out_data[5], (6.0f - 5.0f) * inv_std, 1e-3f);
}

NNOPS_TEST(layernorm_random) {
    auto [in_vec, input] = test::make_random_tensor({4, 8}, -1.0f, 1.0f);
    std::vector<float> scale_buf(8, 1.0f);
    std::vector<float> out_buf(32);

    int64_t shape_x[] = {4, 8};
    int64_t shape_s[] = {8};

    TensorView x(shape_x, DataType::f32, in_vec.data());
    TensorView s(shape_s, DataType::f32, scale_buf.data());
    TensorView y(shape_x, DataType::f32, out_buf.data());

    layer_norm(x, s, y);

    // With scale=1: each row should have mean ≈ 0, std ≈ 1
    for (int r = 0; r < 4; ++r) {
        float row_mean = 0.0f;
        for (int c = 0; c < 8; ++c) { row_mean += out_buf[r * 8 + c]; }
        row_mean /= 8.0f;
        NNOPS_EXPECT_NEAR(row_mean, 0.0f, 1e-3f);

        float row_var = 0.0f;
        for (int c = 0; c < 8; ++c) {
            float d = out_buf[r * 8 + c] - row_mean;
            row_var += d * d;
        }
        row_var /= 8.0f;
        NNOPS_EXPECT_NEAR(row_var, 1.0f, 0.1f);
    }
}

NNOPS_TEST(layernorm_class_api) {
    const int64_t shape[] = {3};
    float x_data[]  = {1.0f, 2.0f, 3.0f};
    float s_data[]  = {1.0f, 1.0f, 1.0f};
    float out1[3] = {}, out2[3] = {};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(shape, DataType::f32, s_data);
    TensorView y1(shape, DataType::f32, out1);
    TensorView y2(shape, DataType::f32, out2);

    layer_norm(x, s, y1);

    auto op = LayerNorm::create(Backend::CPU);
    const TensorView ins[] = {x, s};
    op->compute(y2, ins);

    NNOPS_EXPECT_TRUE(test::allclose(y1, y2));
}
