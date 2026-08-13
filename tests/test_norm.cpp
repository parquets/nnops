/// Unit tests for the consolidated Norm operator (CPU reference).
///
/// Covers BatchNorm, LayerNorm, RMSNorm, GroupNorm, and L2Norm through the
/// shared Norm class / NormAttributes, distinguished by NormType.
/// Uses the class API, plus functional norm() parity checks.

#include "nnops/ops/norm.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

// ============================================================
// BatchNorm
// ============================================================

NNOPS_TEST(norm_batchnorm_simple_1d) {
    const int64_t shape_x[] = {4};
    const int64_t shape_c[] = {1};

    float x_data[]     = {1.0f, 2.0f, 3.0f, 4.0f};
    float scale_data[] = {1.0f};
    float bias_data[]  = {0.0f};
    float mean_data[]  = {2.5f};
    float var_data[]   = {1.25f};
    float out_data[4]  = {};

    TensorView x(shape_x, DataType::f32, x_data);
    TensorView s(shape_c, DataType::f32, scale_data);
    TensorView b(shape_c, DataType::f32, bias_data);
    TensorView m(shape_c, DataType::f32, mean_data);
    TensorView v(shape_c, DataType::f32, var_data);

    NormAttributes attrs;
    attrs.type = NormType::BatchNorm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    auto db = b.desc();
    auto dm = m.desc();
    auto dv = v.desc();
    const TensorDesc desc_arr[] = {dx, ds, db, dm, dv};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 4);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    auto out = test::make_planar(descs[0], out_data);

    const TensorView ins[] = {x, s, b, m, v};
    TensorView outs[] = {out};
    op->compute(outs, ins);

    float eps = 1e-5f;
    float inv_std = 1.0f / std::sqrt(1.25f + eps);
    for (int i = 0; i < 4; ++i) {
        float expected = (x_data[i] - 2.5f) * inv_std;
        NNOPS_EXPECT_NEAR(out_data[i], expected, 1e-4f);
    }
}

NNOPS_TEST(norm_batchnorm_2d_with_scale_bias) {
    const int64_t shape_x[] = {2, 2};
    const int64_t shape_c[] = {2};

    float x_data[]     = {1.0f, 3.0f, 5.0f, 7.0f};
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

    NormAttributes attrs;
    attrs.type = NormType::BatchNorm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    auto db = b.desc();
    auto dm = m.desc();
    auto dv = v.desc();
    const TensorDesc desc_arr[] = {dx, ds, db, dm, dv};
    auto descs = op->getOutputTensorDesc(desc_arr);

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
    float ns0 = inv_std * 2.0f;
    float nb0 = 1.0f - 0.0f * ns0;
    float ns1 = inv_std * 3.0f;
    float nb1 = -1.0f - 0.0f * ns1;

    NNOPS_EXPECT_NEAR(out_data[0], 1.0f * ns0 + nb0, 1e-4f);
    NNOPS_EXPECT_NEAR(out_data[1], 3.0f * ns1 + nb1, 1e-4f);
    NNOPS_EXPECT_NEAR(out_data[2], 5.0f * ns0 + nb0, 1e-4f);
    NNOPS_EXPECT_NEAR(out_data[3], 7.0f * ns1 + nb1, 1e-4f);
}

NNOPS_TEST(norm_batchnorm_random) {
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

    NormAttributes attrs;
    attrs.type = NormType::BatchNorm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    auto db = b.desc();
    auto dm = m.desc();
    auto dv = v.desc();
    const TensorDesc desc_arr[] = {dx, ds, db, dm, dv};
    auto descs = op->getOutputTensorDesc(desc_arr);

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

NNOPS_TEST(norm_batchnorm_random_f16) {
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

    NormAttributes attrs;
    attrs.type = NormType::BatchNorm;
    auto op = Norm::create(attrs, Backend::CPU);

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

    for (size_t i = 0; i < out_buf.size(); ++i) {
        float result = simd::s_load(&out_buf[i]);
        NNOPS_EXPECT_TRUE(std::isfinite(result));
        NNOPS_EXPECT_NEAR(result, x_f32_vec[i], 1e-2f);
    }
}

NNOPS_TEST(norm_batchnorm_nchwc8_spatial) {
    const int64_t N = 2, C = 8, H = 3, W = 4;

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

    std::vector<float> scale(C, 1.0f);
    std::vector<float> bias(C, 0.0f);
    std::vector<float> mean(C, 0.0f);
    std::vector<float> var(C, 1.0f);

    const int64_t shape_c[] = {C};
    TensorView s(shape_c, DataType::f32, scale.data());
    TensorView b(shape_c, DataType::f32, bias.data());
    TensorView m(shape_c, DataType::f32, mean.data());
    TensorView v(shape_c, DataType::f32, var.data());

    NormAttributes attrs;
    attrs.type = NormType::BatchNorm;
    attrs.spatial = true;
    auto op = Norm::create(attrs, Backend::CPU);

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

// ============================================================
// LayerNorm
// ============================================================

NNOPS_TEST(norm_layernorm_simple_last_axis) {
    const int64_t shape[] = {4};
    const int64_t scale_shape[] = {4};

    float x_data[]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float s_data[]  = {1.0f, 1.0f, 1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);

    NormAttributes attrs;
    attrs.type = NormType::LayerNorm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {dx, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(outs[0].rank, 1);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t{4});
    NNOPS_EXPECT_EQ(outs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {x, s};
    op->compute(out, ins);

    float mean = 2.5f;
    float var = 1.25f;
    float eps = 1e-5f;
    float inv_std = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < 4; ++i) {
        float expected = (x_data[i] - mean) * inv_std;
        NNOPS_EXPECT_NEAR(out_buf[i], expected, 1e-3f);
    }
}

NNOPS_TEST(norm_layernorm_with_bias) {
    const int64_t shape[] = {2};
    const int64_t scale_shape[] = {2};

    float x_data[]  = {0.0f, 2.0f};
    float s_data[]  = {2.0f, 3.0f};
    float b_data[]  = {1.0f, -1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);
    TensorView b(scale_shape, DataType::f32, b_data);

    NormAttributes attrs;
    attrs.type = NormType::LayerNorm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    auto db = b.desc();
    const TensorDesc desc_arr[] = {dx, ds, db};
    auto outs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(outs[0].rank, 1);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t{2});
    NNOPS_EXPECT_EQ(outs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {x, s, b};
    op->compute(out, ins);

    float mean = 1.0f;
    float var = 1.0f;
    float inv_std = 1.0f / std::sqrt(var + 1e-5f);
    for (int i = 0; i < 2; ++i) {
        float norm = (x_data[i] - mean) * inv_std;
        float expected = norm * s_data[i] + b_data[i];
        NNOPS_EXPECT_NEAR(out_buf[i], expected, 1e-3f);
    }
}

NNOPS_TEST(norm_layernorm_2d_axis_1) {
    const int64_t shape[] = {2, 3};
    const int64_t scale_shape[] = {3};

    float x_data[]  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float s_data[]  = {1.0f, 1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);

    NormAttributes attrs;
    attrs.type = NormType::LayerNorm;
    attrs.axis = 1;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {dx, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(outs[0].rank, 2);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t{2});
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t{3});
    NNOPS_EXPECT_EQ(outs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {x, s};
    op->compute(out, ins);

    float var = (1.0f + 0.0f + 1.0f) / 3.0f;
    float inv_std = 1.0f / std::sqrt(var + 1e-5f);

    NNOPS_EXPECT_NEAR(out_buf[0], (1.0f - 2.0f) * inv_std, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[1], (2.0f - 2.0f) * inv_std, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[2], (3.0f - 2.0f) * inv_std, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[3], (4.0f - 5.0f) * inv_std, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[4], (5.0f - 5.0f) * inv_std, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[5], (6.0f - 5.0f) * inv_std, 1e-3f);
}

NNOPS_TEST(norm_layernorm_random) {
    auto [in_vec, input] = test::make_random_tensor({4, 8}, -1.0f, 1.0f);
    std::vector<float> scale_buf(8, 1.0f);

    int64_t shape_s[] = {8};
    TensorView s(shape_s, DataType::f32, scale_buf.data());

    NormAttributes attrs;
    attrs.type = NormType::LayerNorm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto di = input.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {di, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(outs[0].rank, 2);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t{4});
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t{8});
    NNOPS_EXPECT_EQ(outs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {input, s};
    op->compute(out, ins);

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

NNOPS_TEST(norm_layernorm_random_f16) {
    auto [x_f32_vec, _] = test::make_random_tensor({4, 8}, -1.0f, 1.0f, 400);
    auto [s_f32_vec, __] = test::make_random_tensor({8}, 0.5f, 2.0f, 401);
    auto [b_f32_vec, ___] = test::make_random_tensor({8}, -0.5f, 0.5f, 402);

    auto x_f16 = test::f32_to_f16(x_f32_vec);
    auto s_f16 = test::f32_to_f16(s_f32_vec);
    auto b_f16 = test::f32_to_f16(b_f32_vec);

    const int64_t x_shape[] = {4, 8};
    const int64_t c_shape[] = {8};
    TensorView x(x_shape, DataType::f16, x_f16.data());
    TensorView s(c_shape, DataType::f16, s_f16.data());
    TensorView b(c_shape, DataType::f16, b_f16.data());

    NormAttributes attrs;
    attrs.type = NormType::LayerNorm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    auto db = b.desc();
    const TensorDesc desc_arr[] = {dx, ds, db};
    auto outs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::f16);

    std::vector<nnops::backend::cpu::half> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {x, s, b};
    op->compute(out, ins);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        float v = simd::s_load(&out_buf[i]);
        NNOPS_EXPECT_TRUE(std::isfinite(v));
        NNOPS_EXPECT_TRUE(std::abs(v) < 100.0f);
    }
}

// ============================================================
// RMSNorm
// ============================================================

NNOPS_TEST(norm_rmsnorm_simple_last_axis) {
    const int64_t shape[] = {4};
    const int64_t scale_shape[] = {4};

    float x_data[]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float s_data[]  = {1.0f, 1.0f, 1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);

    NormAttributes attrs;
    attrs.type = NormType::RMSNorm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto d_x = x.desc();
    auto d_s = s.desc();
    const TensorDesc desc_arr[] = {d_x, d_s};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(1));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(4));
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {x, s};
    op->compute(y, ins);

    float sum_sq = 1.0f + 4.0f + 9.0f + 16.0f;
    float rms = std::sqrt(sum_sq / 4.0f + 1e-5f);
    for (int i = 0; i < 4; ++i) {
        float expected = x_data[i] / rms;
        NNOPS_EXPECT_NEAR(out_buf[i], expected, 1e-3f);
    }
}

NNOPS_TEST(norm_rmsnorm_with_scale) {
    const int64_t shape[] = {2};
    const int64_t scale_shape[] = {2};

    float x_data[]  = {2.0f, 4.0f};
    float s_data[]  = {0.5f, 2.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);

    NormAttributes attrs;
    attrs.type = NormType::RMSNorm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto d_x = x.desc();
    auto d_s = s.desc();
    const TensorDesc desc_arr[] = {d_x, d_s};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(1));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {x, s};
    op->compute(y, ins);

    float rms = std::sqrt(10.0f + 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[0], 2.0f / rms * 0.5f, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[1], 4.0f / rms * 2.0f, 1e-3f);
}

NNOPS_TEST(norm_rmsnorm_2d_axis_1) {
    const int64_t shape[] = {2, 2};
    const int64_t scale_shape[] = {2};

    float x_data[]  = {1.0f, 1.0f, 2.0f, 0.0f};
    float s_data[]  = {1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);

    NormAttributes attrs;
    attrs.type = NormType::RMSNorm;
    attrs.axis = 1;
    auto op = Norm::create(attrs, Backend::CPU);

    auto d_x = x.desc();
    auto d_s = s.desc();
    const TensorDesc desc_arr[] = {d_x, d_s};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {x, s};
    op->compute(y, ins);

    float rms0 = std::sqrt(1.0f + 1e-5f);
    float rms1 = std::sqrt(2.0f + 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[0], 1.0f / rms0, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[1], 1.0f / rms0, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[2], 2.0f / rms1, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[3], 0.0f, 1e-3f);
}

NNOPS_TEST(norm_rmsnorm_random) {
    auto [in_vec, input] = test::make_random_tensor({3, 6}, -1.0f, 1.0f);
    std::vector<float> scale_buf(6, 1.0f);

    const int64_t scale_shape[] = {6};
    TensorView s(scale_shape, DataType::f32, scale_buf.data());

    NormAttributes attrs;
    attrs.type = NormType::RMSNorm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto d_x = input.desc();
    auto d_s = s.desc();
    const TensorDesc desc_arr[] = {d_x, d_s};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(3));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(6));
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {input, s};
    op->compute(y, ins);

    for (int r = 0; r < 3; ++r) {
        float sum_sq = 0.0f;
        for (int c = 0; c < 6; ++c) {
            sum_sq += y.ptr<float>()[r * 6 + c] * y.ptr<float>()[r * 6 + c];
        }
        float rms = std::sqrt(sum_sq / 6.0f);
        NNOPS_EXPECT_NEAR(rms, 1.0f, 0.1f);
    }
}

NNOPS_TEST(norm_rmsnorm_random_f16) {
    auto [x_f32_vec, _] = test::make_random_tensor({3, 6}, -1.0f, 1.0f, 500);
    auto [s_f32_vec, __] = test::make_random_tensor({6}, 0.5f, 2.0f, 501);

    auto x_f16 = test::f32_to_f16(x_f32_vec);
    auto s_f16 = test::f32_to_f16(s_f32_vec);

    const int64_t x_shape[] = {3, 6};
    const int64_t s_shape[] = {6};
    TensorView x(x_shape, DataType::f16, x_f16.data());
    TensorView s(s_shape, DataType::f16, s_f16.data());

    NormAttributes attrs;
    attrs.type = NormType::RMSNorm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto d_x = x.desc();
    auto d_s = s.desc();
    const TensorDesc desc_arr[] = {d_x, d_s};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

    std::vector<nnops::backend::cpu::half> out_buf(static_cast<size_t>(descs[0].numel()));
    auto y = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {x, s};
    op->compute(y, ins);

    for (int r = 0; r < 3; ++r) {
        float sum_sq = 0.0f;
        for (int c = 0; c < 6; ++c) {
            float v = simd::s_load(&out_buf[static_cast<size_t>(r * 6 + c)]);
            NNOPS_EXPECT_TRUE(std::isfinite(v));
            sum_sq += v * v;
        }
        float rms = std::sqrt(sum_sq / 6.0f);
        NNOPS_EXPECT_TRUE(rms > 0.01f && rms < 100.0f);
    }
}

// ============================================================
// L2Norm
// ============================================================

NNOPS_TEST(norm_l2norm_simple_last_axis) {
    const int64_t shape[] = {2};

    float x_data[] = {3.0f, 4.0f};

    TensorView x(shape, DataType::f32, x_data);

    NormAttributes attrs;
    attrs.type = NormType::L2Norm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    const TensorDesc desc_arr[] = {dx};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(1));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {x};
    op->compute(y, ins);

    // norm = sqrt(9 + 16 + eps) ≈ 5
    float norm_val = std::sqrt(25.0f + 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[0], 3.0f / norm_val, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[1], 4.0f / norm_val, 1e-3f);
}

NNOPS_TEST(norm_l2norm_2d_axis_1) {
    const int64_t shape[] = {2, 2};

    float x_data[] = {3.0f, 4.0f, 0.0f, 5.0f};

    TensorView x(shape, DataType::f32, x_data);

    NormAttributes attrs;
    attrs.type = NormType::L2Norm;
    attrs.axis = 1;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    const TensorDesc desc_arr[] = {dx};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(2));

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {x};
    op->compute(y, ins);

    float n0 = std::sqrt(25.0f + 1e-5f);
    float n1 = std::sqrt(25.0f + 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[0], 3.0f / n0, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[1], 4.0f / n0, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[2], 0.0f / n1, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[3], 5.0f / n1, 1e-3f);
}

NNOPS_TEST(norm_l2norm_random) {
    auto [in_vec, input] = test::make_random_tensor({4, 8}, -1.0f, 1.0f);

    NormAttributes attrs;
    attrs.type = NormType::L2Norm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = input.desc();
    const TensorDesc desc_arr[] = {dx};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(4));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(8));

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(y, ins);

    // Each row's L2 norm (sum of squares) should be ≈ 1
    for (int r = 0; r < 4; ++r) {
        float sum_sq = 0.0f;
        for (int c = 0; c < 8; ++c) {
            sum_sq += out_buf[r * 8 + c] * out_buf[r * 8 + c];
        }
        NNOPS_EXPECT_NEAR(sum_sq, 1.0f, 0.1f);
    }
}

NNOPS_TEST(norm_l2norm_add_to) {
    const int64_t shape[] = {2};

    float x_data[] = {3.0f, 4.0f};

    TensorView x(shape, DataType::f32, x_data);

    NormAttributes attrs;
    attrs.type = NormType::L2Norm;
    attrs.add_to = true;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    const TensorDesc desc_arr[] = {dx};
    auto descs = op->getOutputTensorDesc(desc_arr);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 10.0f);
    auto y = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {x};
    op->compute(y, ins);

    float norm_val = std::sqrt(25.0f + 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[0], 10.0f + 3.0f / norm_val, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[1], 10.0f + 4.0f / norm_val, 1e-3f);
}

NNOPS_TEST(norm_l2norm_random_f16) {
    auto [x_f32_vec, _] = test::make_random_tensor({3, 6}, -1.0f, 1.0f, 700);
    auto x_f16 = test::f32_to_f16(x_f32_vec);

    const int64_t x_shape[] = {3, 6};
    TensorView x(x_shape, DataType::f16, x_f16.data());

    NormAttributes attrs;
    attrs.type = NormType::L2Norm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    const TensorDesc desc_arr[] = {dx};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

    std::vector<nnops::backend::cpu::half> out_buf(static_cast<size_t>(descs[0].numel()));
    auto y = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {x};
    op->compute(y, ins);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        float v = simd::s_load(&out_buf[i]);
        NNOPS_EXPECT_TRUE(std::isfinite(v));
    }
}

// ============================================================
// GroupNorm
// ============================================================

NNOPS_TEST(norm_group_norm_2_groups) {
    const int64_t shape[] = {1, 4, 1, 2};
    const int64_t scale_shape[] = {4};

    float x_data[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
    };
    float s_data[] = {1.0f, 1.0f, 1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);

    NormAttributes attrs;
    attrs.type = NormType::GroupNorm;
    attrs.num_groups = 2;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {dx, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(outs[0].rank, 4);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t{1});
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t{4});
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t{1});
    NNOPS_EXPECT_EQ(outs[0].dims[3], int64_t{2});

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {x, s};
    op->compute(out, ins);

    float eps = 1e-5f;

    float g0_mean = 2.5f, g0_var = 1.25f;
    float g0_inv_std = 1.0f / std::sqrt(g0_var + eps);
    NNOPS_EXPECT_NEAR(out_buf[0], (1.0f - g0_mean) * g0_inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[1], (2.0f - g0_mean) * g0_inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[2], (3.0f - g0_mean) * g0_inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[3], (4.0f - g0_mean) * g0_inv_std, 1e-4f);

    float g1_mean = 6.5f, g1_var = 1.25f;
    float g1_inv_std = 1.0f / std::sqrt(g1_var + eps);
    NNOPS_EXPECT_NEAR(out_buf[4], (5.0f - g1_mean) * g1_inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[5], (6.0f - g1_mean) * g1_inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[6], (7.0f - g1_mean) * g1_inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[7], (8.0f - g1_mean) * g1_inv_std, 1e-4f);
}

NNOPS_TEST(norm_group_norm_with_scale_bias) {
    const int64_t shape[] = {1, 2, 1, 2};
    const int64_t scale_shape[] = {2};

    float x_data[] = {0.0f, 2.0f, 0.0f, 4.0f};
    float s_data[] = {2.0f, 3.0f};
    float b_data[] = {1.0f, -1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);
    TensorView b(scale_shape, DataType::f32, b_data);

    NormAttributes attrs;
    attrs.type = NormType::GroupNorm;
    attrs.num_groups = 1;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    auto db = b.desc();
    const TensorDesc desc_arr[] = {dx, ds, db};
    auto outs = op->getOutputTensorDesc(desc_arr);

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {x, s, b};
    op->compute(out, ins);

    float mean = 1.5f, var = 2.75f;
    float eps = 1e-5f;
    float inv_std = 1.0f / std::sqrt(var + eps);

    NNOPS_EXPECT_NEAR(out_buf[0], (0.0f - mean) * inv_std * 2.0f + 1.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[1], (2.0f - mean) * inv_std * 2.0f + 1.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[2], (0.0f - mean) * inv_std * 3.0f + (-1.0f), 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[3], (4.0f - mean) * inv_std * 3.0f + (-1.0f), 1e-4f);
}

NNOPS_TEST(norm_group_norm_equals_instancenorm_when_g_equals_c) {
    const int64_t shape[] = {2, 4, 2};
    const int64_t scale_shape[] = {4};

    std::vector<float> x_data = {
        1.0f, 2.0f,   3.0f, 4.0f,   5.0f, 6.0f,   7.0f, 8.0f,
        9.0f, 10.0f,  11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
    };
    std::vector<float> s_data = {1.0f, 1.0f, 1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data.data());
    TensorView s(scale_shape, DataType::f32, s_data.data());

    NormAttributes attrs;
    attrs.type = NormType::GroupNorm;
    attrs.num_groups = 4;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {dx, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());
    const TensorView ins[] = {x, s};
    op->compute(out, ins);

    float eps = 1e-5f;

    for (int n = 0; n < 2; ++n) {
        for (int c = 0; c < 4; ++c) {
            int base = n * 8 + c * 2;
            float v0 = x_data[base];
            float v1 = x_data[base + 1];
            float mean = (v0 + v1) * 0.5f;
            float var = ((v0 - mean) * (v0 - mean) + (v1 - mean) * (v1 - mean)) * 0.5f;
            float inv_std = 1.0f / std::sqrt(var + eps);
            NNOPS_EXPECT_NEAR(out_buf[base], (v0 - mean) * inv_std, 1e-4f);
            NNOPS_EXPECT_NEAR(out_buf[base + 1], (v1 - mean) * inv_std, 1e-4f);
        }
    }
}

NNOPS_TEST(norm_group_norm_add_to) {
    const int64_t shape[] = {1, 2, 2};
    const int64_t scale_shape[] = {2};

    float x_data[]  = {1.0f, 3.0f, 5.0f, 7.0f};
    float s_data[]  = {1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);

    NormAttributes attrs;
    attrs.type = NormType::GroupNorm;
    attrs.num_groups = 1;
    attrs.add_to = true;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {dx, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()), 10.0f);
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {x, s};
    op->compute(out, ins);

    float mean = 4.0f;
    float var = 5.0f;
    float eps = 1e-5f;
    float inv_std = 1.0f / std::sqrt(var + eps);

    NNOPS_EXPECT_NEAR(out_buf[0], 10.0f + (1.0f - mean) * inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[1], 10.0f + (3.0f - mean) * inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[2], 10.0f + (5.0f - mean) * inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[3], 10.0f + (7.0f - mean) * inv_std, 1e-4f);
}

NNOPS_TEST(norm_group_norm_random_f32) {
    const int64_t shape[] = {2, 6, 4, 8};
    const int64_t scale_shape[] = {6};

    auto [x_vec, x] = test::make_random_tensor(shape, -10.0f, 10.0f, 500);
    auto [s_vec, s] = test::make_random_tensor(scale_shape, 0.5f, 2.0f, 501);
    auto [b_vec, b] = test::make_random_tensor(scale_shape, -1.0f, 1.0f, 502);

    for (int64_t g : {1, 2, 3, 6}) {
        NormAttributes attrs;
        attrs.type = NormType::GroupNorm;
        attrs.num_groups = g;

        auto dx = x.desc();
        auto ds = s.desc();
        auto db = b.desc();

        auto op_simd = Norm::create(attrs, Backend::CPU);
        const TensorDesc desc_simd[] = {dx, ds, db};
        auto outs_simd = op_simd->getOutputTensorDesc(desc_simd);
        std::vector<float> out_simd_buf(static_cast<size_t>(outs_simd[0].numel()));
        auto out_simd = test::make_planar(outs_simd[0], out_simd_buf.data());
        const TensorView ins_simd[] = {x, s, b};
        op_simd->compute(out_simd, ins_simd);

        float out_mean = 0.0f, out_var = 0.0f;
        int64_t total = outs_simd[0].numel();
        for (int64_t i = 0; i < total; ++i) {
            out_mean += out_simd_buf[i];
        }
        out_mean /= static_cast<float>(total);
        for (int64_t i = 0; i < total; ++i) {
            float d = out_simd_buf[i] - out_mean;
            out_var += d * d;
        }
        out_var /= static_cast<float>(total);

        NNOPS_EXPECT_TRUE(out_var > 0.01f && out_var < 100.0f);
    }
}

NNOPS_TEST(norm_group_norm_random_f16) {
    const int64_t shape[] = {1, 4, 4, 4};
    const int64_t scale_shape[] = {4};

    using nnops::backend::cpu::half;

    auto [x_f32_vec, x_f32] = test::make_random_tensor(shape, -5.0f, 5.0f, 600);
    auto [s_f32_vec, s_f32] = test::make_random_tensor(scale_shape, 0.5f, 2.0f, 601);

    const int64_t x_numel = x_f32.numel();
    const int64_t s_numel = s_f32.numel();

    std::vector<half> x_data(static_cast<size_t>(x_numel));
    std::vector<half> s_data(static_cast<size_t>(s_numel));
    for (int64_t i = 0; i < x_numel; ++i) {
        simd::s_store(&x_data[static_cast<size_t>(i)], x_f32_vec[static_cast<size_t>(i)]);
    }
    for (int64_t i = 0; i < s_numel; ++i) {
        simd::s_store(&s_data[static_cast<size_t>(i)], s_f32_vec[static_cast<size_t>(i)]);
    }

    TensorView x(shape, DataType::f16, x_data.data());
    TensorView s(scale_shape, DataType::f16, s_data.data());

    NormAttributes attrs;
    attrs.type = NormType::GroupNorm;
    attrs.num_groups = 2;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {dx, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    std::vector<half> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {x, s};
    op->compute(out, ins);

    int64_t total = outs[0].numel();
    for (int64_t i = 0; i < total; ++i) {
        float val = simd::s_load(&out_buf[static_cast<size_t>(i)]);
        NNOPS_EXPECT_TRUE(std::isfinite(val));
    }
}

NNOPS_TEST(norm_group_norm_rank3_ncw) {
    const int64_t shape[] = {1, 3, 4};
    const int64_t scale_shape[] = {3};

    float x_data[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
    };
    float s_data[] = {1.0f, 1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);

    NormAttributes attrs;
    attrs.type = NormType::GroupNorm;
    attrs.num_groups = 1;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {dx, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(outs[0].rank, 3);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t{1});
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t{3});
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t{4});

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());
    const TensorView ins[] = {x, s};
    op->compute(out, ins);

    float mean = 6.5f;
    float var = 0.0f;
    for (int i = 0; i < 12; ++i) {
        var += (x_data[i] - mean) * (x_data[i] - mean);
    }
    var /= 12.0f;
    float eps = 1e-5f;
    float inv_std = 1.0f / std::sqrt(var + eps);

    for (int i = 0; i < 12; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], (x_data[i] - mean) * inv_std, 1e-4f);
    }
}

NNOPS_TEST(norm_group_norm_functional_api) {
    const int64_t shape[] = {1, 2, 2};
    const int64_t scale_shape[] = {2};

    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> s_data = {1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data.data());
    TensorView s(scale_shape, DataType::f32, s_data.data());

    auto dx = x.desc();
    auto out_desc = TensorDesc{dx.dims, dx.rank, dx.dtype, dx.layout};
    std::vector<float> out_buf1(static_cast<size_t>(out_desc.numel()));
    std::vector<float> out_buf2(static_cast<size_t>(out_desc.numel()));
    auto out1 = test::make_planar(out_desc, out_buf1.data());
    auto out2 = test::make_planar(out_desc, out_buf2.data());

    NormAttributes attrs;
    attrs.type = NormType::GroupNorm;
    attrs.num_groups = 1;

    auto op = Norm::create(attrs, Backend::CPU);
    const TensorView ins_class[] = {x, s};
    op->compute(out1, ins_class);

    norm(x, s, out2, attrs);

    for (size_t i = 0; i < out_buf1.size(); ++i) {
        NNOPS_EXPECT_NEAR(out_buf1[i], out_buf2[i], 1e-6f);
    }
}

NNOPS_TEST(norm_group_norm_functional_api_with_bias) {
    const int64_t shape[] = {1, 2, 2};
    const int64_t scale_shape[] = {2};

    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> s_data = {1.0f, 1.0f};
    std::vector<float> b_data = {0.5f, -0.5f};

    TensorView x(shape, DataType::f32, x_data.data());
    TensorView s(scale_shape, DataType::f32, s_data.data());
    TensorView b(scale_shape, DataType::f32, b_data.data());

    auto dx = x.desc();
    auto out_desc = TensorDesc{dx.dims, dx.rank, dx.dtype, dx.layout};
    std::vector<float> out_buf1(static_cast<size_t>(out_desc.numel()));
    std::vector<float> out_buf2(static_cast<size_t>(out_desc.numel()));
    auto out1 = test::make_planar(out_desc, out_buf1.data());
    auto out2 = test::make_planar(out_desc, out_buf2.data());

    NormAttributes attrs;
    attrs.type = NormType::GroupNorm;
    attrs.num_groups = 1;

    auto op = Norm::create(attrs, Backend::CPU);
    const TensorView ins_class[] = {x, s, b};
    op->compute(out1, ins_class);

    norm(x, s, b, out2, attrs);

    for (size_t i = 0; i < out_buf1.size(); ++i) {
        NNOPS_EXPECT_NEAR(out_buf1[i], out_buf2[i], 1e-6f);
    }
}

NNOPS_TEST(norm_group_norm_shape_inference) {
    const int64_t shape[] = {2, 8, 4, 4};
    const int64_t scale_shape[] = {8};

    std::vector<float> x_data(2 * 8 * 4 * 4);
    std::vector<float> s_data(8);

    TensorView x(shape, DataType::f32, x_data.data());
    TensorView s(scale_shape, DataType::f32, s_data.data());

    NormAttributes attrs;
    attrs.type = NormType::GroupNorm;
    attrs.num_groups = 4;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {dx, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(outs[0].rank, 4);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t{2});
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t{8});
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t{4});
    NNOPS_EXPECT_EQ(outs[0].dims[3], int64_t{4});
    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(outs[0].layout, TensorLayout::NCHW);
}

// ============================================================
// OpType / Backend / default-attributes checks
// ============================================================

NNOPS_TEST(norm_create_optype_backend) {
    NormAttributes attrs;
    attrs.type = NormType::LayerNorm;
    auto op = Norm::create(attrs, Backend::CPU);
    NNOPS_EXPECT_EQ(static_cast<int>(op->getOpType()), static_cast<int>(OpType::Norm));
    NNOPS_EXPECT_EQ(static_cast<int>(op->getBackend()), static_cast<int>(Backend::CPU));
}

NNOPS_TEST(norm_default_attrs) {
    NormAttributes attrs;
    NNOPS_EXPECT_EQ(static_cast<int>(attrs.type), static_cast<int>(NormType::LayerNorm));
    NNOPS_EXPECT_EQ(attrs.axis, int64_t(-1));
    NNOPS_EXPECT_EQ(attrs.num_groups, int64_t(1));
    NNOPS_EXPECT_NEAR(attrs.epsilon, 1e-5f, 1e-9f);
    NNOPS_EXPECT_EQ(attrs.spatial, true);
    NNOPS_EXPECT_EQ(attrs.add_to, false);
}
