/// Unit tests for LayerNorm operator (CPU reference).
/// Uses LayerNorm class API exclusively — no functional layer_norm() calls.

#include "nnops/ops/layer_norm.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
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

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);

    LayerNormAttributes attrs;
    auto op = LayerNorm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {dx, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    // Validate output descriptor
    NNOPS_EXPECT_EQ(outs[0].rank, 1);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t{4});
    NNOPS_EXPECT_EQ(outs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {x, s};
    op->compute(out, ins);

    // mean = 2.5, var = 1.25
    float mean = 2.5f;
    float var = 1.25f;
    float eps = 1e-5f;
    float inv_std = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < 4; ++i) {
        float expected = (x_data[i] - mean) * inv_std;
        NNOPS_EXPECT_NEAR(out_buf[i], expected, 1e-3f);
    }
}

NNOPS_TEST(layernorm_with_bias) {
    const int64_t shape[] = {2};
    const int64_t scale_shape[] = {2};

    float x_data[]  = {0.0f, 2.0f};
    float s_data[]  = {2.0f, 3.0f};
    float b_data[]  = {1.0f, -1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);
    TensorView b(scale_shape, DataType::f32, b_data);

    LayerNormAttributes attrs;
    auto op = LayerNorm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    auto db = b.desc();
    const TensorDesc desc_arr[] = {dx, ds, db};
    auto outs = op->getOutputTensorDesc(desc_arr);

    // Validate output descriptor
    NNOPS_EXPECT_EQ(outs[0].rank, 1);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t{2});
    NNOPS_EXPECT_EQ(outs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {x, s, b};
    op->compute(out, ins);

    // mean = 1.0, var = 1.0 (x=[0,2] -> mean=1, squared diffs: 1+1=2, var=2/2=1)
    float mean = 1.0f;
    float var = 1.0f;
    float inv_std = 1.0f / std::sqrt(var + 1e-5f);
    for (int i = 0; i < 2; ++i) {
        float norm = (x_data[i] - mean) * inv_std;
        float expected = norm * s_data[i] + b_data[i];
        NNOPS_EXPECT_NEAR(out_buf[i], expected, 1e-3f);
    }
}

NNOPS_TEST(layernorm_2d_axis_1) {
    // 2x3, normalize along axis=1 (normalize each row)
    const int64_t shape[] = {2, 3};
    const int64_t scale_shape[] = {3};

    float x_data[]  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float s_data[]  = {1.0f, 1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);

    LayerNormAttributes attrs;
    attrs.axis = 1;
    auto op = LayerNorm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {dx, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    // Validate output descriptor
    NNOPS_EXPECT_EQ(outs[0].rank, 2);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t{2});
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t{3});
    NNOPS_EXPECT_EQ(outs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {x, s};
    op->compute(out, ins);

    // Row 0: mean=(1+2+3)/3=2, var=((1-2)^2+(2-2)^2+(3-2)^2)/3=(1+0+1)/3=0.6667
    // Row 1: mean=(4+5+6)/3=5, var=((4-5)^2+(5-5)^2+(6-5)^2)/3=0.6667
    float var = (1.0f + 0.0f + 1.0f) / 3.0f;  // 2/3
    float inv_std = 1.0f / std::sqrt(var + 1e-5f);

    NNOPS_EXPECT_NEAR(out_buf[0], (1.0f - 2.0f) * inv_std, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[1], (2.0f - 2.0f) * inv_std, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[2], (3.0f - 2.0f) * inv_std, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[3], (4.0f - 5.0f) * inv_std, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[4], (5.0f - 5.0f) * inv_std, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[5], (6.0f - 5.0f) * inv_std, 1e-3f);
}

NNOPS_TEST(layernorm_random) {
    auto [in_vec, input] = test::make_random_tensor({4, 8}, -1.0f, 1.0f);
    std::vector<float> scale_buf(8, 1.0f);

    int64_t shape_s[] = {8};
    TensorView s(shape_s, DataType::f32, scale_buf.data());

    LayerNormAttributes attrs;
    auto op = LayerNorm::create(attrs, Backend::CPU);

    auto di = input.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {di, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    // Validate output descriptor
    NNOPS_EXPECT_EQ(outs[0].rank, 2);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t{4});
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t{8});
    NNOPS_EXPECT_EQ(outs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {input, s};
    op->compute(out, ins);

    // With scale=1: each row should have mean ~ 0, std ~ 1
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
