/// Unit tests for BatchNorm operator (CPU reference, inference only).

#include "nnops/ops/batch_norm.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

NNOPS_TEST(batchnorm_simple_1d) {
    // 1D input [N=4], C=1
    const int64_t shape_x[] = {4};
    const int64_t shape_c[] = {1};

    float x_data[]     = {1.0f, 2.0f, 3.0f, 4.0f};
    float scale_data[] = {1.0f};
    float bias_data[]  = {0.0f};
    float mean_data[]  = {2.5f};  // (1+2+3+4)/4
    float var_data[]   = {1.25f}; // variance
    float out_data[4]  = {};

    TensorView x(shape_x, DataType::f32, x_data);
    TensorView s(shape_c, DataType::f32, scale_data);
    TensorView b(shape_c, DataType::f32, bias_data);
    TensorView m(shape_c, DataType::f32, mean_data);
    TensorView v(shape_c, DataType::f32, var_data);

    // Class API
    BatchNormAttributes attrs;
    auto op = BatchNorm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    auto db = b.desc();
    auto dm = m.desc();
    auto dv = v.desc();
    const TensorDesc desc_arr[] = {dx, ds, db, dm, dv};
    auto descs = op->getOutputTensorDesc(desc_arr);

    // Validate output descriptor
    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 4);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    auto out = test::make_planar(descs[0], out_data);

    const TensorView ins[] = {x, s, b, m, v};
    TensorView outs[] = {out};
    op->compute(outs, ins);

    // Manual: (x - 2.5) / sqrt(1.25 + 1e-5) * 1.0 + 0.0
    float eps = 1e-5f;
    float inv_std = 1.0f / std::sqrt(1.25f + eps);
    for (int i = 0; i < 4; ++i) {
        float expected = (x_data[i] - 2.5f) * inv_std;
        NNOPS_EXPECT_NEAR(out_data[i], expected, 1e-4f);
    }
}

NNOPS_TEST(batchnorm_2d_with_scale_bias) {
    // N=2, C=2, spatial dim: 1x1 (total 2 elements per channel)
    const int64_t shape_x[] = {2, 2};
    const int64_t shape_c[] = {2};

    float x_data[]     = {1.0f, 3.0f,   // sample 0: ch0=1, ch1=3
                          5.0f, 7.0f};  // sample 1: ch0=5, ch1=7
    float scale_data[] = {2.0f, 3.0f};
    float bias_data[]  = {1.0f, -1.0f};
    float mean_data[]  = {0.0f, 0.0f};
    float var_data[]   = {1.0f, 1.0f};
    float out_data[4]  = {};

    TensorView x(shape_x, DataType::f32, x_data);
    TensorView s(shape_c, DataType::f32, scale_data);
    TensorView b(shape_c, DataType::f32, bias_data);
    TensorView m(shape_c, DataType::f32, mean_data);
    TensorView v(shape_c, DataType::f32, var_data);

    // Class API
    BatchNormAttributes attrs;
    auto op = BatchNorm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    auto db = b.desc();
    auto dm = m.desc();
    auto dv = v.desc();
    const TensorDesc desc_arr[] = {dx, ds, db, dm, dv};
    auto descs = op->getOutputTensorDesc(desc_arr);

    // Validate output descriptor
    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 2);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    auto out = test::make_planar(descs[0], out_data);

    const TensorView ins[] = {x, s, b, m, v};
    TensorView outs[] = {out};
    op->compute(outs, ins);

    float eps = 1e-5f;
    float inv_std = 1.0f / std::sqrt(1.0f + eps);
    float ns0 = inv_std * 2.0f;  // ch0 new_scale
    float nb0 = 1.0f - 0.0f * ns0;  // ch0 new_bias
    float ns1 = inv_std * 3.0f;  // ch1 new_scale
    float nb1 = -1.0f - 0.0f * ns1;  // ch1 new_bias

    NNOPS_EXPECT_NEAR(out_data[0], 1.0f * ns0 + nb0, 1e-4f);   // ch0, sample 0
    NNOPS_EXPECT_NEAR(out_data[1], 3.0f * ns1 + nb1, 1e-4f);   // ch1, sample 0
    NNOPS_EXPECT_NEAR(out_data[2], 5.0f * ns0 + nb0, 1e-4f);   // ch0, sample 1
    NNOPS_EXPECT_NEAR(out_data[3], 7.0f * ns1 + nb1, 1e-4f);   // ch1, sample 1
}

NNOPS_TEST(batchnorm_random) {
    auto [in_vec, input] = test::make_random_tensor({2, 4, 3}, -1.0f, 1.0f);
    std::vector<float> scale_buf(4, 1.0f);
    std::vector<float> bias_buf(4, 0.0f);
    std::vector<float> mean_buf(4, 0.0f);
    std::vector<float> var_buf(4, 1.0f);
    std::vector<float> out_buf(2 * 4 * 3);

    int64_t shape_c[] = {4};
    int64_t shape_x[] = {2, 4, 3};

    TensorView x(shape_x, DataType::f32, in_vec.data());
    TensorView s(shape_c, DataType::f32, scale_buf.data());
    TensorView b(shape_c, DataType::f32, bias_buf.data());
    TensorView m(shape_c, DataType::f32, mean_buf.data());
    TensorView v(shape_c, DataType::f32, var_buf.data());

    // Class API
    BatchNormAttributes attrs;
    auto op = BatchNorm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    auto db = b.desc();
    auto dm = m.desc();
    auto dv = v.desc();
    const TensorDesc desc_arr[] = {dx, ds, db, dm, dv};
    auto descs = op->getOutputTensorDesc(desc_arr);

    // Validate output descriptor
    NNOPS_EXPECT_EQ(descs[0].rank, 3);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 4);
    NNOPS_EXPECT_EQ(descs[0].dims[2], 3);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    auto out = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {x, s, b, m, v};
    TensorView outs[] = {out};
    op->compute(outs, ins);

    // With mean=0, var=1, scale=1, bias=0: output should equal input
    NNOPS_EXPECT_TRUE(test::allclose(x, out, 1e-3f, 1e-3f));
}

// ============================================================
// f16 test
// ============================================================

NNOPS_TEST(batchnorm_random_f16) {
    // 2D f16 spatial: N=2, C=4
    auto [x_f32_vec, _] = test::make_random_tensor({2, 4, 3}, -1.0f, 1.0f, 200);
    std::vector<float> scale_buf(4, 1.0f);
    std::vector<float> bias_buf(4, 0.0f);
    std::vector<float> mean_buf(4, 0.0f);
    std::vector<float> var_buf(4, 1.0f);

    auto x_f16 = test::f32_to_f16(x_f32_vec);
    auto s_f16 = test::f32_to_f16(scale_buf);
    auto b_f16 = test::f32_to_f16(bias_buf);
    auto m_f16 = test::f32_to_f16(mean_buf);
    auto v_f16 = test::f32_to_f16(var_buf);

    const int64_t shape_x[] = {2, 4, 3};
    const int64_t shape_c[] = {4};
    TensorView x(shape_x, DataType::f16, x_f16.data());
    TensorView s(shape_c, DataType::f16, s_f16.data());
    TensorView b(shape_c, DataType::f16, b_f16.data());
    TensorView m(shape_c, DataType::f16, m_f16.data());
    TensorView v(shape_c, DataType::f16, v_f16.data());

    BatchNormAttributes attrs;
    auto op = BatchNorm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    auto db = b.desc();
    auto dm = m.desc();
    auto dv = v.desc();
    const TensorDesc desc_arr[] = {dx, ds, db, dm, dv};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

    std::vector<nnops::backend::cpu::half> out_buf(static_cast<size_t>(descs[0].numel()));
    auto out = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {x, s, b, m, v};
    TensorView outs[] = {out};
    op->compute(outs, ins);

    // With mean=0, var=1, scale=1, bias=0: output should ≈ input
    for (size_t i = 0; i < out_buf.size(); ++i) {
        float result = simd::s_load(&out_buf[i]);
        NNOPS_EXPECT_TRUE(std::isfinite(result));
        NNOPS_EXPECT_NEAR(result, x_f32_vec[i], 1e-2f);
    }
}

// ============================================================
// Packed NCHWC8 test
// ============================================================

NNOPS_TEST(batchnorm_nchwc8_spatial) {
    // NCHWC8 [2, 8, 3, 4] — packed spatial batch norm
    // This exercises the pack > 1 path
    const int64_t N = 2, C = 8, H = 3, W = 4;

    // Create NCHW f32 reference data and pack to NCHWC8
    auto [f32_buf, _] = test::make_random_tensor({N, C, H, W}, -1.0f, 1.0f, 300);

    const int64_t num_c8 = (C + 7) / 8;
    const int64_t row_stride = W * 8;
    size_t total = static_cast<size_t>(N * num_c8 * H * row_stride);
    std::vector<float> in_data(total, 0.0f);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            int64_t c8 = c / 8;
            int64_t lane = c % 8;
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    int64_t planar_idx = n * (C * H * W) + c * (H * W) + h * W + w;
                    int64_t packed_off = n * (num_c8 * H * row_stride)
                                         + c8 * (H * row_stride)
                                         + h * row_stride + w * 8 + lane;
                    in_data[static_cast<size_t>(packed_off)] = f32_buf[static_cast<size_t>(planar_idx)];
                }
            }
        }
    }

    TensorDesc desc;
    desc.rank = 4;
    desc.dims = {N, C, H, W};
    desc.dtype = DataType::f32;
    desc.layout = TensorLayout::NCHWC8;
    TensorView x = test::make_packed(desc, in_data.data());

    // Scale/bias/mean/var: one per channel
    std::vector<float> scale(C, 1.0f);
    std::vector<float> bias(C, 0.0f);
    std::vector<float> mean(C, 0.0f);
    std::vector<float> var(C, 1.0f);

    const int64_t shape_c[] = {C};
    TensorView s(shape_c, DataType::f32, scale.data());
    TensorView b(shape_c, DataType::f32, bias.data());
    TensorView m(shape_c, DataType::f32, mean.data());
    TensorView v(shape_c, DataType::f32, var.data());

    BatchNormAttributes attrs;
    attrs.spatial = true;
    auto op = BatchNorm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    auto db = b.desc();
    auto dm = m.desc();
    auto dv = v.desc();
    const TensorDesc desc_arr[] = {dx, ds, db, dm, dv};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHWC8);

    std::vector<float> out_buf(total, 0.0f);
    auto out = test::make_packed(descs[0], out_buf.data());

    const TensorView ins[] = {x, s, b, m, v};
    TensorView outs[] = {out};
    op->compute(outs, ins);

    // Verify all values are finite and close to input (mean=0, var=1, scale=1, bias=0)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            int64_t c8 = c / 8;
            int64_t lane = c % 8;
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    int64_t planar_idx = n * (C * H * W) + c * (H * W) + h * W + w;
                    int64_t packed_off = n * (num_c8 * H * row_stride)
                                         + c8 * (H * row_stride)
                                         + h * row_stride + w * 8 + lane;
                    float expected = f32_buf[static_cast<size_t>(planar_idx)];
                    float result = out_buf[static_cast<size_t>(packed_off)];
                    NNOPS_EXPECT_TRUE(std::isfinite(result));
                    NNOPS_EXPECT_NEAR(result, expected, 1e-3f);
                }
            }
        }
    }
}
