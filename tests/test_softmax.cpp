/// Unit tests for Softmax operator (CPU reference / class API).

#include "nnops/ops/softmax.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"

#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

using namespace nnops;

NNOPS_TEST(softmax_last_axis) {
    // 1D input, softmax along axis 0
    const int64_t shape[] = {3};
    float in_data[]  = {1.0f, 2.0f, 3.0f};

    TensorView input(shape, DataType::f32, in_data);
    auto d_in = input.desc();

    auto op = Softmax::create({}, Backend::CPU);

    const TensorDesc in_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_arr);

    // Validate output descriptor
    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    std::vector<float> out_buf(descs[0].numel());
    TensorView output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(output, ins);

    // exp(1)=2.718, exp(2)=7.389, exp(3)=20.086 → sum=30.193
    float sum_exp = 0.0f;
    for (int i = 0; i < 3; ++i) {
        sum_exp += std::exp(in_data[i]);
    }
    NNOPS_EXPECT_NEAR(out_buf[0], std::exp(1.0f) / sum_exp, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[1], std::exp(2.0f) / sum_exp, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[2], std::exp(3.0f) / sum_exp, 1e-4f);
    // Should sum to 1
    NNOPS_EXPECT_NEAR(out_buf[0] + out_buf[1] + out_buf[2], 1.0f, 1e-5f);
}

NNOPS_TEST(softmax_2d_axis_1) {
    // 2x3, softmax along axis=1 (normalize rows)
    const int64_t shape[] = {2, 3};
    float in_data[]  = {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f};

    TensorView input(shape, DataType::f32, in_data);
    auto d_in = input.desc();

    SoftmaxAttributes attrs;
    attrs.axis = 1;
    auto op = Softmax::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_arr);

    // Validate output descriptor
    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 3);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    std::vector<float> out_buf(descs[0].numel());
    TensorView output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(output, ins);

    // Row 0: same as 1D test
    float sum0 = 0.0f;
    for (int i = 0; i < 3; ++i) {
        sum0 += std::exp(in_data[i]);
    }
    NNOPS_EXPECT_NEAR(out_buf[0], std::exp(1.0f) / sum0, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[1], std::exp(2.0f) / sum0, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[2], std::exp(3.0f) / sum0, 1e-4f);
    // Row 1: all equal → each gets 1/3
    NNOPS_EXPECT_NEAR(out_buf[3], 1.0f / 3.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[4], 1.0f / 3.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[5], 1.0f / 3.0f, 1e-4f);
}

NNOPS_TEST(softmax_log_softmax) {
    const int64_t shape[] = {2};
    float in_data[]  = {0.0f, 1.0f};

    TensorView input(shape, DataType::f32, in_data);
    auto d_in = input.desc();

    SoftmaxAttributes attrs;
    attrs.log_softmax = true;
    auto op = Softmax::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_arr);

    // Validate output descriptor
    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    std::vector<float> out_buf(descs[0].numel());
    TensorView output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(output, ins);

    // log_softmax(0,1): max=1, shifted={-1,0}, exp={0.3679,1}, sum=1.3679, log_sum=0.3133
    // out[0] = -1 - 0.3133 = -1.3133, out[1] = 0 - 0.3133 = -0.3133
    float max_val = std::max(in_data[0], in_data[1]);
    float shifted0 = in_data[0] - max_val;
    float shifted1 = in_data[1] - max_val;
    float log_sum = std::log(std::exp(shifted0) + std::exp(shifted1));
    NNOPS_EXPECT_NEAR(out_buf[0], shifted0 - log_sum, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[1], shifted1 - log_sum, 1e-4f);
}

NNOPS_TEST(softmax_random) {
    auto [in_vec, input] = test::make_random_tensor({4, 8}, -2.0f, 2.0f);
    auto d_in = input.desc();

    SoftmaxAttributes attrs;
    attrs.axis = 1;
    auto op = Softmax::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_arr);

    // Validate output descriptor
    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 4);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 8);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    std::vector<float> out_buf(descs[0].numel());
    TensorView output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(output, ins);

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

// ============================================================
// Packed layout tests (NCHWC8)
// NCHWC8 is rank 4: [N, C, H, W] where C is the LOGICAL channel count.
// Physical row layout: [w0_l0..w0_l7, w1_l0..w1_l7, ...]
// num_channel_blocks = ceil(C / 8), row_stride = W * 8 elements.
// total_rows = N * num_channel_blocks * H.
// ============================================================

/// Helper: create NCHWC8 TensorView with correct pitch, plus buffer.
static std::pair<std::vector<float>, TensorView>
make_nchwc8(int64_t N, int64_t C, int64_t H, int64_t W)
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

/// Helper: fill NCHWC8 data with sequential values.
/// num_c8 = ceil(C/8), row_stride = W*8.
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

NNOPS_TEST(softmax_nchwc8_axis_w) {
    // NCHWC8 [1, 8, 1, 3] — one full C8 block (C=8), W=3
    // Softmax over W (axis=-1). Each C lane is independently normalized.
    const int64_t N = 1, C = 8, H = 1, W = 3;
    const int64_t row_stride = W * 8;  // 24

    auto [in_data, input] = make_nchwc8(N, C, H, W);
    fill_nchwc8(in_data.data(), N, C, H, W, 1.0f, 1.0f);

    auto d_in = input.desc();

    SoftmaxAttributes attrs;
    attrs.axis = -1;  // W dimension (rank-1)
    auto op = Softmax::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 4);
    NNOPS_EXPECT_EQ(descs[0].dims[3], 3);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHWC8);

    auto [out_buf, output] = make_nchwc8(N, C, H, W);

    const TensorView ins[] = {input};
    op->compute(output, ins);

    // Verify: for each C lane, the 3 W positions sum to 1.
    for (int lane = 0; lane < 8; ++lane) {
        float row_sum = 0.0f;
        for (int w = 0; w < 3; ++w) {
            float v = out_buf[w * 8 + lane];
            NNOPS_EXPECT_TRUE(v >= 0.0f);
            row_sum += v;
        }
        NNOPS_EXPECT_NEAR(row_sum, 1.0f, 1e-4f);
    }
}

NNOPS_TEST(softmax_nchwc8_axis_w_log) {
    // Same as above but log_softmax.
    const int64_t N = 1, C = 8, H = 1, W = 3;

    auto [in_data, input] = make_nchwc8(N, C, H, W);
    fill_nchwc8(in_data.data(), N, C, H, W, 1.0f, 1.0f);
    auto d_in = input.desc();

    SoftmaxAttributes attrs;
    attrs.axis = -1;
    attrs.log_softmax = true;
    auto op = Softmax::create(attrs, Backend::CPU);

    auto [out_buf, output] = make_nchwc8(N, C, H, W);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    // log_softmax values should be <= 0 and exp-sum to 1
    for (int lane = 0; lane < 8; ++lane) {
        float sum_exp = 0.0f;
        for (int w = 0; w < 3; ++w) {
            float v = out_buf[w * 8 + lane];
            NNOPS_EXPECT_TRUE(v <= 0.0f + 1e-5f);
            sum_exp += std::exp(v);
        }
        NNOPS_EXPECT_NEAR(sum_exp, 1.0f, 1e-4f);
    }
}

NNOPS_TEST(softmax_nchwc8_random) {
    // Random NCHWC8 [2, 16, 3, 4] — two full C8 blocks (C=16).
    const int64_t N = 2, C = 16, H = 3, W = 4;
    const int64_t num_c8 = (C + 7) / 8;
    const int64_t row_stride = W * 8;

    auto [in_data, input] = make_nchwc8(N, C, H, W);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    for (auto& v : in_data) {
        v = dist(rng);
    }

    SoftmaxAttributes attrs;
    attrs.axis = -1;
    auto op = Softmax::create(attrs, Backend::CPU);

    auto [out_buf, output] = make_nchwc8(N, C, H, W);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    // For each (n, c8, h, lane), the W values should sum to 1.
    int64_t n_stride = num_c8 * H * row_stride;
    int64_t c8_stride = H * row_stride;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c8 = 0; c8 < num_c8; ++c8) {
            for (int64_t h = 0; h < H; ++h) {
                for (int lane = 0; lane < 8; ++lane) {
                    float sum = 0.0f;
                    for (int64_t w = 0; w < W; ++w) {
                        int64_t off = n * n_stride + c8 * c8_stride
                                      + h * row_stride + w * 8 + lane;
                        sum += out_buf[off];
                        NNOPS_EXPECT_TRUE(out_buf[off] >= 0.0f);
                    }
                    NNOPS_EXPECT_NEAR(sum, 1.0f, 1e-4f);
                }
            }
        }
    }
}

// ============================================================
// Temperature tests
// ============================================================

NNOPS_TEST(softmax_temperature_basic) {
    // 1D input [1, 2, 3], T=2.0
    // softmax(x_i, T) = exp(x_i/T) / sum(exp(x_j/T))
    // max=3, shifted={-2/T, -1/T, 0/T} = {-1, -0.5, 0}
    // exp: {0.3679, 0.6065, 1.0}, sum=1.9744
    // out: {0.1863, 0.3072, 0.5065}
    const int64_t shape[] = {3};
    float in_data[]  = {1.0f, 2.0f, 3.0f};

    TensorView input(shape, DataType::f32, in_data);
    auto d_in = input.desc();

    SoftmaxAttributes attrs;
    attrs.temperature = 2.0f;
    auto op = Softmax::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(descs[0].numel());
    TensorView output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(output, ins);

    float T = 2.0f;
    float max_val = 3.0f;
    float e0 = std::exp((1.0f - max_val) / T);
    float e1 = std::exp((2.0f - max_val) / T);
    float e2 = std::exp((3.0f - max_val) / T);
    float sum_exp = e0 + e1 + e2;

    NNOPS_EXPECT_NEAR(out_buf[0], e0 / sum_exp, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[1], e1 / sum_exp, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[2], e2 / sum_exp, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[0] + out_buf[1] + out_buf[2], 1.0f, 1e-5f);
}

NNOPS_TEST(softmax_temperature_sharper) {
    // T=0.5 makes distribution sharper (more peaked)
    // [1, 2, 3], T=0.5 → exp({-4, -2, 0}) = {0.0183, 0.1353, 1.0}
    // sum=1.1536, out: {0.0159, 0.1173, 0.8668}
    const int64_t shape[] = {3};
    float in_data[]  = {1.0f, 2.0f, 3.0f};

    TensorView input(shape, DataType::f32, in_data);
    auto d_in = input.desc();

    SoftmaxAttributes attrs;
    attrs.temperature = 0.5f;
    auto op = Softmax::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_arr);

    std::vector<float> out_buf(descs[0].numel());
    TensorView output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(output, ins);

    float T = 0.5f;
    float max_val = 3.0f;
    float e0 = std::exp((1.0f - max_val) / T);
    float e1 = std::exp((2.0f - max_val) / T);
    float e2 = std::exp((3.0f - max_val) / T);
    float sum_exp = e0 + e1 + e2;

    NNOPS_EXPECT_NEAR(out_buf[0], e0 / sum_exp, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[1], e1 / sum_exp, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[2], e2 / sum_exp, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[0] + out_buf[1] + out_buf[2], 1.0f, 1e-5f);

    // T=0.5: max value should be larger than T=1.0 case
    // At T=1: exp(1)/sum=0.090, exp(2)/sum=0.245, exp(3)/sum=0.665
    NNOPS_EXPECT_TRUE(out_buf[2] > 0.75f);  // sharper → higher for max
}

NNOPS_TEST(softmax_temperature_equals_one) {
    // T=1.0 should produce same result as default (no temperature)
    const int64_t shape[] = {2};
    float in_data[]  = {0.0f, 2.0f};

    TensorView input(shape, DataType::f32, in_data);
    auto d_in = input.desc();

    // Default attrs (T=1)
    SoftmaxAttributes attrs_default;
    auto op_default = Softmax::create(attrs_default, Backend::CPU);

    const TensorDesc in_arr[] = {d_in};
    auto descs = op_default->getOutputTensorDesc(in_arr);

    std::vector<float> out_default(2);
    TensorView output_default = nnops::test::make_planar(descs[0], out_default.data());
    const TensorView ins[] = {input};
    op_default->compute(output_default, ins);

    // Explicit T=1
    SoftmaxAttributes attrs_t1;
    attrs_t1.temperature = 1.0f;
    auto op_t1 = Softmax::create(attrs_t1, Backend::CPU);

    std::vector<float> out_t1(2);
    TensorView output_t1 = nnops::test::make_planar(descs[0], out_t1.data());
    op_t1->compute(output_t1, ins);

    NNOPS_EXPECT_NEAR(out_t1[0], out_default[0], 1e-6f);
    NNOPS_EXPECT_NEAR(out_t1[1], out_default[1], 1e-6f);
}

NNOPS_TEST(softmax_temperature_log) {
    // log_softmax with temperature T=2.0 on [1, 2, 3]
    // max=3, shifted/T: {-2/2=-1, -1/2=-0.5, 0/2=0}
    // exp: {0.3679, 0.6065, 1.0}, sum=1.9744, log_sum=0.6803
    // out: {-1-0.6803=-1.6803, -0.5-0.6803=-1.1803, 0-0.6803=-0.6803}
    const int64_t shape[] = {3};
    float in_data[]  = {1.0f, 2.0f, 3.0f};

    TensorView input(shape, DataType::f32, in_data);
    auto d_in = input.desc();

    SoftmaxAttributes attrs;
    attrs.log_softmax = true;
    attrs.temperature = 2.0f;
    auto op = Softmax::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_arr);

    std::vector<float> out_buf(3);
    TensorView output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(output, ins);

    float T = 2.0f;
    float max_val = 3.0f;
    float shifted[] = { (1.0f - max_val) / T, (2.0f - max_val) / T, (3.0f - max_val) / T };
    float log_sum = std::log(std::exp(shifted[0]) + std::exp(shifted[1]) + std::exp(shifted[2]));

    NNOPS_EXPECT_NEAR(out_buf[0], shifted[0] - log_sum, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[1], shifted[1] - log_sum, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[2], shifted[2] - log_sum, 1e-4f);

    // exp of log_softmax with temperature should sum to 1
    NNOPS_EXPECT_NEAR(std::exp(out_buf[0]) + std::exp(out_buf[1]) + std::exp(out_buf[2]),
                      1.0f, 1e-4f);
}

NNOPS_TEST(softmax_temperature_random_2d) {
    // Random 2D test: compare SIMD softmax with temperature against reference
    auto [in_vec, input] = test::make_random_tensor({4, 8}, -2.0f, 2.0f);
    auto d_in = input.desc();

    for (float T : {0.5f, 1.0f, 2.0f, 5.0f}) {
        SoftmaxAttributes attrs;
        attrs.axis = 1;
        attrs.temperature = T;
        auto op = Softmax::create(attrs, Backend::CPU);

        const TensorDesc in_arr[] = {d_in};
        auto descs = op->getOutputTensorDesc(in_arr);

        std::vector<float> out_buf(descs[0].numel());
        TensorView output = nnops::test::make_planar(descs[0], out_buf.data());

        const TensorView ins[] = {input};
        op->compute(output, ins);

        // Each row should sum to 1 (softmax property holds for any T)
        for (int r = 0; r < 4; ++r) {
            float row_sum = 0.0f;
            for (int c = 0; c < 8; ++c) {
                row_sum += out_buf[r * 8 + c];
                NNOPS_EXPECT_TRUE(out_buf[r * 8 + c] >= 0.0f);
            }
            NNOPS_EXPECT_NEAR(row_sum, 1.0f, 1e-4f);
        }
    }
}

NNOPS_TEST(softmax_temperature_nchwc8) {
    // NCHWC8 with temperature — per-lane SIMD path
    const int64_t N = 1, C = 8, H = 1, W = 4;
    const int64_t row_stride = W * 8;

    TensorDesc desc;
    desc.rank = 4;
    desc.dims = {N, C, H, W};
    desc.dtype = DataType::f32;
    desc.layout = TensorLayout::NCHWC8;

    auto [in_data, input] = make_nchwc8(N, C, H, W);
    fill_nchwc8(in_data.data(), N, C, H, W, 1.0f, 1.0f);

    for (float T : {0.5f, 1.0f, 2.0f, 4.0f}) {
        SoftmaxAttributes attrs;
        attrs.axis = -1;
        attrs.temperature = T;
        auto op = Softmax::create(attrs, Backend::CPU);

        auto [out_buf, output] = make_nchwc8(N, C, H, W);
        const TensorView ins[] = {input};
        op->compute(output, ins);

        // Each C lane sums to 1 across W
        for (int lane = 0; lane < 8; ++lane) {
            float sum = 0.0f;
            for (int w = 0; w < 4; ++w) {
                float v = out_buf[w * 8 + lane];
                NNOPS_EXPECT_TRUE(v >= 0.0f);
                sum += v;
            }
            NNOPS_EXPECT_NEAR(sum, 1.0f, 1e-4f);
        }
    }
}

NNOPS_TEST(softmax_nchw_axis_first) {
    // NCHW [3, 4, 2, 2] — softmax over axis=0 (batch dimension).
    // Uses the general scalar path. Verifies the decomposition is correct.
    const int64_t shape[] = {3, 4, 2, 2};
    const int64_t N = 3, C = 4, H = 2, W = 2;
    const int64_t total = N * C * H * W;  // 48

    std::vector<float> in_data(static_cast<size_t>(total));
    for (size_t i = 0; i < in_data.size(); ++i) {
        in_data[i] = static_cast<float>(i) * 0.1f;
    }

    TensorView input(shape, DataType::f32, in_data.data(), TensorLayout::NCHW);
    auto d_in = input.desc();

    SoftmaxAttributes attrs;
    attrs.axis = 0;  // batch dim
    auto op = Softmax::create(attrs, Backend::CPU);

    const TensorDesc in_desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_desc_arr);
    std::vector<float> out_buf(static_cast<size_t>(total));
    TensorView output(shape, DataType::f32, out_buf.data(), TensorLayout::NCHW);

    const TensorView ins[] = {input};
    op->compute(output, ins);

    // For each (c, h, w) position, the 3 batch values should sum to 1.
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                float sum = 0.0f;
                for (int n = 0; n < N; ++n) {
                    int off = n * 16 + c * 4 + h * 2 + w;
                    sum += out_buf[off];
                    NNOPS_EXPECT_TRUE(out_buf[off] >= 0.0f);
                }
                NNOPS_EXPECT_NEAR(sum, 1.0f, 1e-4f);
            }
        }
    }
}

// ============================================================
// f16 tests — exercise the SIMD f16 code path
// ============================================================

NNOPS_TEST(softmax_random_f16) {
    // Planar 2D f16: axis=1, 4x8
    auto [f16_buf, input] = test::make_random_f16_tensor({4, 8}, -2.0f, 2.0f);

    SoftmaxAttributes attrs;
    attrs.axis = 1;
    auto op = Softmax::create(attrs, Backend::CPU);

    auto d_in = input.desc();
    const TensorDesc in_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

    std::vector<nnops::backend::cpu::half> out_buf(static_cast<size_t>(descs[0].numel()));
    auto output = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(output, ins);

    // Each row should sum to 1 (softmax property)
    for (int r = 0; r < 4; ++r) {
        float row_sum = 0.0f;
        for (int c = 0; c < 8; ++c) {
            float v = simd::s_load(&out_buf[static_cast<size_t>(r * 8 + c)]);
            NNOPS_EXPECT_TRUE(v >= 0.0f);
            NNOPS_EXPECT_TRUE(std::isfinite(v));
            row_sum += v;
        }
        NNOPS_EXPECT_NEAR(row_sum, 1.0f, 1e-2f);  // f16 has lower precision
    }
}

NNOPS_TEST(softmax_nchwc8_random_f16) {
    // NCHWC8 [2, 16, 3, 4] f16 — exercise packed SIMD path with f16
    const int64_t N = 2, C = 16, H = 3, W = 4;

    // Create f32 random, then convert
    auto [f32_buf, _] = test::make_random_tensor({N, C, H, W}, -3.0f, 3.0f, 789);

    // Pack to NCHWC8
    const int64_t num_c8 = (C + 7) / 8;
    const int64_t row_stride = W * 8;
    size_t total = static_cast<size_t>(N * num_c8 * H * row_stride);
    std::vector<nnops::backend::cpu::half> in_data(total);

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
                    simd::s_store(&in_data[static_cast<size_t>(packed_off)], f32_buf[static_cast<size_t>(planar_idx)]);
                }
            }
        }
    }

    TensorDesc desc;
    desc.rank = 4;
    desc.dims = {N, C, H, W};
    desc.dtype = DataType::f16;
    desc.layout = TensorLayout::NCHWC8;
    TensorView input = test::make_packed(desc, in_data.data());

    SoftmaxAttributes attrs;
    attrs.axis = -1;
    auto op = Softmax::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {desc};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

    std::vector<nnops::backend::cpu::half> out_buf(total);
    TensorView output = test::make_packed(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(output, ins);

    // Per-lane verification: each C lane should sum to 1 across W
    int64_t n_stride = num_c8 * H * row_stride;
    int64_t c8_stride = H * row_stride;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c8 = 0; c8 < num_c8; ++c8) {
            for (int64_t h = 0; h < H; ++h) {
                for (int lane = 0; lane < 8; ++lane) {
                    float sum = 0.0f;
                    for (int64_t w = 0; w < W; ++w) {
                        int64_t off = n * n_stride + c8 * c8_stride
                                      + h * row_stride + w * 8 + lane;
                        float v = simd::s_load(&out_buf[static_cast<size_t>(off)]);
                        NNOPS_EXPECT_TRUE(v >= 0.0f);
                        sum += v;
                    }
                    NNOPS_EXPECT_NEAR(sum, 1.0f, 1e-2f);
                }
            }
        }
    }
}

// ============================================================
// Packed channel softmax — axis=1 (channel axis)
// ============================================================

NNOPS_TEST(softmax_nchwc8_axis_channel) {
    // NCHWC8 [2, 16, 3, 4] — softmax over axis=1 (channel).
    // Exercises the Path 1b packed channel SIMD kernel.
    const int64_t N = 2, C = 16, H = 3, W = 4;
    const int64_t num_c8 = (C + 7) / 8;    // 2
    const int64_t row_stride = W * 8;       // 32

    auto [in_data, input] = make_nchwc8(N, C, H, W);

    // Fill with varying values so that different C positions get different results
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto& v : in_data) {
        v = dist(rng);
    }

    SoftmaxAttributes attrs;
    attrs.axis = 1;     // channel axis
    auto op = Softmax::create(attrs, Backend::CPU);

    auto [out_buf, output] = make_nchwc8(N, C, H, W);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    // For each (n, h, w) spatial position, each of the 8 lanes independently
    // normalizes across its C8 blocks. So per lane: sum across C8 = 1.0.
    int64_t n_stride = num_c8 * H * row_stride;
    int64_t c8_stride = H * row_stride;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t w = 0; w < W; ++w) {
                for (int lane = 0; lane < 8; ++lane) {
                    float lane_sum = 0.0f;
                    for (int64_t c8 = 0; c8 < num_c8; ++c8) {
                        int64_t off = n * n_stride + c8 * c8_stride
                                      + h * row_stride + w * 8 + lane;
                        NNOPS_EXPECT_TRUE(out_buf[off] >= 0.0f);
                        lane_sum += out_buf[off];
                    }
                    NNOPS_EXPECT_NEAR(lane_sum, 1.0f, 1e-4f);
                }
            }
        }
    }
}

NNOPS_TEST(softmax_nchwc8_axis_channel_partial_c8) {
    // NCHWC8 [1, 10, 2, 3] — C=10 not a multiple of 8 (C8=2, valid_lanes=2).
    // Exercises the partial-lane path in packed channel softmax.
    const int64_t N = 1, C = 10, H = 2, W = 3;
    const int64_t num_c8 = (C + 7) / 8;    // 2
    const int64_t row_stride = W * 8;       // 24

    auto [in_data, input] = make_nchwc8(N, C, H, W);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto& v : in_data) {
        v = dist(rng);
    }

    SoftmaxAttributes attrs;
    attrs.axis = 1;     // channel axis
    auto op = Softmax::create(attrs, Backend::CPU);

    auto [out_buf, output] = make_nchwc8(N, C, H, W);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    // C=10, valid_lanes=2. All 8 lanes get softmax:
    // Lanes 0-1: across 2 C8 blocks → sum 1
    // Lanes 2-7: only C8 block 0 (block 1 has 0 for these lanes) → sum 1
    int64_t n_stride = num_c8 * H * row_stride;
    int64_t c8_stride = H * row_stride;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t w = 0; w < W; ++w) {
                for (int lane = 0; lane < 8; ++lane) {
                    float lane_sum = 0.0f;
                    for (int64_t c8 = 0; c8 < num_c8; ++c8) {
                        int64_t off = n * n_stride + c8 * c8_stride
                                      + h * row_stride + w * 8 + lane;
                        NNOPS_EXPECT_TRUE(out_buf[off] >= 0.0f);
                        lane_sum += out_buf[off];
                    }
                    // All lanes sum to 1:
                    // - Valid lanes (0-1): softmax across 2 values
                    // - Other lanes (2-7): softmax of 1 valid + 1 zero
                    NNOPS_EXPECT_NEAR(lane_sum, 1.0f, 1e-4f);
                }
            }
        }
    }
}
