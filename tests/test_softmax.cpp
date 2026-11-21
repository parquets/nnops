/// @file test_softmax.cpp
/// @brief Softmax operator tests (CPU, class API): softmax and log-softmax
///        along an axis, temperature and f16, on NHWC storage only.
///
/// The layout tests run every axis of a 4-D tensor against a reference that
/// walks the same physical indexing (NHWC [N][H][W][C]) in its own code, so an
/// axis handled through the wrong stride cannot pass. Channel-packed weight
/// tags are rejected.

#include "nnops/ops/softmax.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <random>

using namespace nnops;

NNOPS_TEST(softmax_last_axis) {
    const int64_t shape[] = {3};
    float in_data[]  = {1.0f, 2.0f, 3.0f};

    TensorView input(shape, DataType::f32, in_data, TensorLayout::NHWC);
    auto d_in = input.desc();

    auto op = Softmax::create({}, Backend::CPU);

    const TensorDesc in_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NHWC);

    std::vector<float> out_buf(descs[0].numel());
    TensorView output = nnops::test::make_channel_first(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(output, ins);

    float sum_exp = 0.0f;
    for (int i = 0; i < 3; ++i) {
        sum_exp += std::exp(in_data[i]);
    }
    NNOPS_EXPECT_NEAR(out_buf[0], std::exp(1.0f) / sum_exp, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[1], std::exp(2.0f) / sum_exp, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[2], std::exp(3.0f) / sum_exp, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[0] + out_buf[1] + out_buf[2], 1.0f, 1e-5f);
}

NNOPS_TEST(softmax_2d_axis_1) {
    const int64_t shape[] = {2, 3};
    float in_data[]  = {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f};

    TensorView input(shape, DataType::f32, in_data, TensorLayout::NHWC);
    auto d_in = input.desc();

    SoftmaxAttributes attrs;
    attrs.axis = 1;
    auto op = Softmax::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 3);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NHWC);

    std::vector<float> out_buf(descs[0].numel());
    TensorView output = nnops::test::make_channel_first(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(output, ins);

    float sum0 = 0.0f;
    for (int i = 0; i < 3; ++i) {
        sum0 += std::exp(in_data[i]);
    }
    NNOPS_EXPECT_NEAR(out_buf[0], std::exp(1.0f) / sum0, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[1], std::exp(2.0f) / sum0, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[2], std::exp(3.0f) / sum0, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[3], 1.0f / 3.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[4], 1.0f / 3.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[5], 1.0f / 3.0f, 1e-4f);
}

NNOPS_TEST(softmax_log_softmax) {
    const int64_t shape[] = {2};
    float in_data[]  = {0.0f, 1.0f};

    TensorView input(shape, DataType::f32, in_data, TensorLayout::NHWC);
    auto d_in = input.desc();

    SoftmaxAttributes attrs;
    attrs.log_softmax = true;
    auto op = Softmax::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NHWC);

    std::vector<float> out_buf(descs[0].numel());
    TensorView output = nnops::test::make_channel_first(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(output, ins);

    float max_val = std::max(in_data[0], in_data[1]);
    float shifted0 = in_data[0] - max_val;
    float shifted1 = in_data[1] - max_val;
    float log_sum = std::log(std::exp(shifted0) + std::exp(shifted1));
    NNOPS_EXPECT_NEAR(out_buf[0], shifted0 - log_sum, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[1], shifted1 - log_sum, 1e-4f);
}

NNOPS_TEST(softmax_random) {
    const int64_t shape[] = {4, 8};
    auto [in_vec, in_default] = test::make_random_tensor({4, 8}, -2.0f, 2.0f);
    (void)in_default;
    const TensorView input(shape, DataType::f32, in_vec.data(), TensorLayout::NHWC);
    auto d_in = input.desc();

    SoftmaxAttributes attrs;
    attrs.axis = 1;
    auto op = Softmax::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 4);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 8);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NHWC);

    std::vector<float> out_buf(descs[0].numel());
    TensorView output = nnops::test::make_channel_first(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(output, ins);

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
// Layout coverage — dense and NHWC
// ============================================================
//
// A channel-last tensor's logical dims [N][H][W][C] are its physical order.
// Every axis is compared against a reference that re-derives the offsets from
// that order on its own, so an axis walked with the wrong stride cannot pass.
// Dimensions stay <= 16 so a group fits a stack array.

template <typename Off>
static std::vector<float> layout_softmax_reference(const std::vector<float>& in,
                                                   const int64_t dims[4],
                                                   int64_t axis, bool log_softmax,
                                                   Off off)
{
    const int64_t D = dims[axis];
    int64_t others[3] = {0, 0, 0};
    int64_t n_others = 0;
    for (int64_t d = 0; d < 4; ++d) {
        if (d != axis) { others[n_others++] = d; }
    }

    int64_t groups = 1;
    for (int64_t i = 0; i < n_others; ++i) { groups *= dims[others[i]]; }

    std::vector<float> out(in.size(), 0.0f);
    for (int64_t g = 0; g < groups; ++g) {
        int64_t pos[4] = {0, 0, 0, 0};
        int64_t rem = g;
        for (int64_t i = n_others - 1; i >= 0; --i) {
            pos[others[i]] = rem % dims[others[i]];
            rem /= dims[others[i]];
        }

        float vals[16];
        float m = -std::numeric_limits<float>::infinity();
        for (int64_t k = 0; k < D; ++k) {
            pos[axis] = k;
            vals[k] = in[static_cast<size_t>(off(pos[0], pos[1], pos[2], pos[3]))];
            if (vals[k] > m) { m = vals[k]; }
        }
        float sum = 0.0f;
        for (int64_t k = 0; k < D; ++k) {
            vals[k] = std::exp(vals[k] - m);
            sum += vals[k];
        }
        for (int64_t k = 0; k < D; ++k) {
            pos[axis] = k;
            out[static_cast<size_t>(off(pos[0], pos[1], pos[2], pos[3]))] =
                log_softmax ? std::log(vals[k] / sum) : vals[k] / sum;
        }
    }
    return out;
}

template <typename Off>
static void check_all_axes(std::vector<float>& in, const int64_t dims[4],
                           TensorLayout layout, Off off)
{
    const int64_t shape[] = {dims[0], dims[1], dims[2], dims[3]};
    TensorView input(shape, DataType::f32, in.data(), layout);

    for (int64_t axis = 0; axis < 4; ++axis) {
        for (const bool log_softmax : {false, true}) {
            const auto expected = layout_softmax_reference(in, dims, axis, log_softmax, off);

            SoftmaxAttributes attrs;
            attrs.axis = axis;
            attrs.log_softmax = log_softmax;
            auto op = Softmax::create(attrs, Backend::CPU);

            std::vector<float> out_buf(in.size(), 0.0f);
            TensorView output(shape, DataType::f32, out_buf.data(), layout);

            const TensorView ins[] = {input};
            NNOPS_EXPECT_TRUE(is_ok(op->compute(output, ins)));

            for (size_t i = 0; i < out_buf.size(); ++i) {
                NNOPS_EXPECT_NEAR(out_buf[i], expected[i], 1e-5f);
            }
        }
    }
}

NNOPS_TEST(softmax_all_axes) {
    // The strided path lanes across channels, so C = 5 is a partial vector and
    // C = 12 is a full one plus a half; the channel axis itself is one NHWC
    // row, whose own tail is the last group's.
    for (const int64_t C : {int64_t{5}, int64_t{12}}) {
        const int64_t dims[4] = {2, 3, 4, C};
        const int64_t H = dims[1], W = dims[2];

        auto [in_data, view] = test::make_random_tensor({dims[0], C, H, W}, -2.0f, 2.0f, 13);
        (void)view;

        check_all_axes(in_data, dims, TensorLayout::NHWC,
                       [&](int64_t n, int64_t h, int64_t w, int64_t c) {
                           return ((n * H + h) * W + w) * C + c;
                       });
    }
}

NNOPS_TEST(softmax_2d_channel) {
    // Rank-2 NHWC is [N][C]: axis 1 is the channel axis and one contiguous pixel.
    const int64_t N = 3, C = 5;
    const int64_t shape[] = {N, C};

    auto [in_data, view] = test::make_random_tensor({N, C}, -2.0f, 2.0f, 19);
    (void)view;
    TensorView input(shape, DataType::f32, in_data.data(), TensorLayout::NHWC);

    SoftmaxAttributes attrs;
    attrs.axis = 1;
    auto op = Softmax::create(attrs, Backend::CPU);

    std::vector<float> out_buf(static_cast<size_t>(N * C), 0.0f);
    TensorView output(shape, DataType::f32, out_buf.data(), TensorLayout::NHWC);

    const TensorView ins[] = {input};
    NNOPS_EXPECT_TRUE(is_ok(op->compute(output, ins)));

    for (int64_t n = 0; n < N; ++n) {
        const float* row = in_data.data() + n * C;
        float m = *std::max_element(row, row + C);
        float sum = 0.0f;
        for (int64_t c = 0; c < C; ++c) { sum += std::exp(row[c] - m); }
        for (int64_t c = 0; c < C; ++c) {
            NNOPS_EXPECT_NEAR(out_buf[static_cast<size_t>(n * C + c)],
                              std::exp(row[c] - m) / sum, 1e-5f);
        }
    }
}

NNOPS_TEST(softmax_channel_axis_f16) {
    const int64_t N = 1, C = 10, H = 2, W = 2;
    const int64_t shape[] = {N, H, W, C};

    auto [f32_buf, view] = test::make_random_tensor({N, C, H, W}, -2.0f, 2.0f, 23);
    (void)view;
    auto f16_buf = test::f32_to_f16(f32_buf);
    TensorView input(shape, DataType::f16, f16_buf.data(), TensorLayout::NHWC);

    SoftmaxAttributes attrs;
    attrs.axis = 3;  // the channel axis is the last one
    auto op = Softmax::create(attrs, Backend::CPU);

    std::vector<nnops::backend::cpu::half> out_buf(static_cast<size_t>(N * C * H * W));
    TensorView output(shape, DataType::f16, out_buf.data(), TensorLayout::NHWC);

    const TensorView ins[] = {input};
    NNOPS_EXPECT_TRUE(is_ok(op->compute(output, ins)));

    // Compare against the same f16 inputs read back as f32, so the only error
    // left is the f16 output rounding.
    const auto exact = test::f16_to_f32(f16_buf);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t w = 0; w < W; ++w) {
                float m = -std::numeric_limits<float>::infinity();
                for (int64_t c = 0; c < C; ++c) {
                    m = std::max(m, exact[static_cast<size_t>(((n * H + h) * W + w) * C + c)]);
                }
                float sum = 0.0f;
                for (int64_t c = 0; c < C; ++c) {
                    sum += std::exp(exact[static_cast<size_t>(((n * H + h) * W + w) * C + c)] - m);
                }
                for (int64_t c = 0; c < C; ++c) {
                    const size_t i = static_cast<size_t>(((n * H + h) * W + w) * C + c);
                    NNOPS_EXPECT_NEAR(simd::s_load(&out_buf[i]),
                                      std::exp(exact[i] - m) / sum, 2e-3f);
                }
            }
        }
    }
}

NNOPS_TEST(softmax_rejects_weight_tags) {
    auto op = Softmax::create(Backend::CPU);

    const int64_t shape[] = {4, 128, 32};
    TensorDesc in_desc;
    in_desc.dims = std::span<const int64_t>(shape, 3);
    in_desc.rank = 3;
    in_desc.dtype = DataType::f32;
    in_desc.layout = TensorLayout::PackedHWIO;
    const TensorDesc out_desc = in_desc;

    const TensorDesc ins[] = {in_desc};
    const TensorDesc outs[] = {out_desc};
    NNOPS_EXPECT_EQ(op->validate(ins, outs), Status::unsupported);
}

// ============================================================
// Temperature tests
// ============================================================

NNOPS_TEST(softmax_temperature_basic) {
    const int64_t shape[] = {3};
    float in_data[]  = {1.0f, 2.0f, 3.0f};

    TensorView input(shape, DataType::f32, in_data, TensorLayout::NHWC);
    auto d_in = input.desc();

    SoftmaxAttributes attrs;
    attrs.temperature = 2.0f;
    auto op = Softmax::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(descs[0].numel());
    TensorView output = nnops::test::make_channel_first(descs[0], out_buf.data());

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
    const int64_t shape[] = {3};
    float in_data[]  = {1.0f, 2.0f, 3.0f};

    TensorView input(shape, DataType::f32, in_data, TensorLayout::NHWC);
    auto d_in = input.desc();

    SoftmaxAttributes attrs;
    attrs.temperature = 0.5f;
    auto op = Softmax::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_arr);

    std::vector<float> out_buf(descs[0].numel());
    TensorView output = nnops::test::make_channel_first(descs[0], out_buf.data());

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

    // T=0.5 sharpens: at T=1 the max is 0.665, so >0.75 means sharper.
    NNOPS_EXPECT_TRUE(out_buf[2] > 0.75f);
}

NNOPS_TEST(softmax_temperature_equals_one) {
    const int64_t shape[] = {2};
    float in_data[]  = {0.0f, 2.0f};

    TensorView input(shape, DataType::f32, in_data, TensorLayout::NHWC);
    auto d_in = input.desc();

    SoftmaxAttributes attrs_default;
    auto op_default = Softmax::create(attrs_default, Backend::CPU);

    const TensorDesc in_arr[] = {d_in};
    auto descs = op_default->getOutputTensorDesc(in_arr);

    std::vector<float> out_default(2);
    TensorView output_default = nnops::test::make_channel_first(descs[0], out_default.data());
    const TensorView ins[] = {input};
    op_default->compute(output_default, ins);

    SoftmaxAttributes attrs_t1;
    attrs_t1.temperature = 1.0f;
    auto op_t1 = Softmax::create(attrs_t1, Backend::CPU);

    std::vector<float> out_t1(2);
    TensorView output_t1 = nnops::test::make_channel_first(descs[0], out_t1.data());
    op_t1->compute(output_t1, ins);

    NNOPS_EXPECT_NEAR(out_t1[0], out_default[0], 1e-6f);
    NNOPS_EXPECT_NEAR(out_t1[1], out_default[1], 1e-6f);
}

NNOPS_TEST(softmax_temperature_log) {
    const int64_t shape[] = {3};
    float in_data[]  = {1.0f, 2.0f, 3.0f};

    TensorView input(shape, DataType::f32, in_data, TensorLayout::NHWC);
    auto d_in = input.desc();

    SoftmaxAttributes attrs;
    attrs.log_softmax = true;
    attrs.temperature = 2.0f;
    auto op = Softmax::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_arr);

    std::vector<float> out_buf(3);
    TensorView output = nnops::test::make_channel_first(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(output, ins);

    float T = 2.0f;
    float max_val = 3.0f;
    float shifted[] = { (1.0f - max_val) / T, (2.0f - max_val) / T, (3.0f - max_val) / T };
    float log_sum = std::log(std::exp(shifted[0]) + std::exp(shifted[1]) + std::exp(shifted[2]));

    NNOPS_EXPECT_NEAR(out_buf[0], shifted[0] - log_sum, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[1], shifted[1] - log_sum, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[2], shifted[2] - log_sum, 1e-4f);

    NNOPS_EXPECT_NEAR(std::exp(out_buf[0]) + std::exp(out_buf[1]) + std::exp(out_buf[2]),
                      1.0f, 1e-4f);
}

NNOPS_TEST(softmax_temperature_random_2d) {
    const int64_t shape[] = {4, 8};
    auto [in_vec, in_default] = test::make_random_tensor({4, 8}, -2.0f, 2.0f);
    (void)in_default;
    const TensorView input(shape, DataType::f32, in_vec.data(), TensorLayout::NHWC);
    auto d_in = input.desc();

    for (float T : {0.5f, 1.0f, 2.0f, 5.0f}) {
        SoftmaxAttributes attrs;
        attrs.axis = 1;
        attrs.temperature = T;
        auto op = Softmax::create(attrs, Backend::CPU);

        const TensorDesc in_arr[] = {d_in};
        auto descs = op->getOutputTensorDesc(in_arr);

        std::vector<float> out_buf(descs[0].numel());
        TensorView output = nnops::test::make_channel_first(descs[0], out_buf.data());

        const TensorView ins[] = {input};
        op->compute(output, ins);

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

NNOPS_TEST(softmax_random_f16) {
    const int64_t shape[] = {4, 8};
    auto [f16_buf, in_default] = test::make_random_f16_tensor({4, 8}, -2.0f, 2.0f);
    (void)in_default;
    const TensorView input(shape, DataType::f16, f16_buf.data(), TensorLayout::NHWC);

    SoftmaxAttributes attrs;
    attrs.axis = 1;
    auto op = Softmax::create(attrs, Backend::CPU);

    auto d_in = input.desc();
    const TensorDesc in_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

    std::vector<nnops::backend::cpu::half> out_buf(static_cast<size_t>(descs[0].numel()));
    auto output = test::make_channel_first(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(output, ins);

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

// ============================================================
// Layout and pitch: the contiguous fast path needs a compact view
// ============================================================

namespace {

/// Row-wise softmax reference over the physical C run of every pixel, read
/// through the view's own strides (so a pitched row is addressed correctly).
/// `storage_elems` is the buffer's element span, which exceeds numel() for a
/// pitched view; untouched entries stay NaN.
void softmax_last_axis_ref(const TensorView& x, size_t storage_elems, std::vector<double>& want)
{
    const int64_t rank = x.rank();
    const int64_t C = x.shape(rank - 1);
    const int64_t pixels = x.numel() / C;
    want.assign(storage_elems, std::numeric_limits<double>::quiet_NaN());
    for (int64_t p = 0; p < pixels; ++p) {
        // Position p in the storage order minus the C axis: decompose over the
        // leading axes, innermost first.
        int64_t rem = p;
        int64_t base = 0;
        for (int64_t d = rank - 2; d >= 0; --d) {
            const int64_t n = x.shape(d);
            base += (rem % n) * x.stride_elems(d);
            rem /= n;
        }
        const float* row = x.ptr<float>() + base;
        double mx = -1e30;
        for (int64_t c = 0; c < C; ++c) { mx = std::max(mx, static_cast<double>(row[c])); }
        double sum = 0.0;
        for (int64_t c = 0; c < C; ++c) { sum += std::exp(row[c] - mx); }
        for (int64_t c = 0; c < C; ++c) {
            want[static_cast<size_t>(base + c)] = std::exp(row[c] - mx) / sum;
        }
    }
}

} // namespace

NNOPS_TEST(softmax_last_axis_pitched_row) {
    // [1,2,3,3] with the default 32-byte row pitch: a row is 36 bytes of data
    // and 64 bytes of storage, so consecutive pixels are *not* C elements apart
    // and the contiguous fast path must not be taken.
    const int64_t H = 2, W = 3, C = 3;
    const int64_t pitch_bytes = 64;
    const int64_t pitch_elems = pitch_bytes / 4;
    std::vector<float> x(static_cast<size_t>(H * pitch_elems), 0.0f);
    std::vector<float> y(static_cast<size_t>(H * pitch_elems), -7.0f);
    for (int64_t h = 0; h < H; ++h) {
        for (int64_t i = 0; i < W * C; ++i) {
            x[static_cast<size_t>(h * pitch_elems + i)] =
                0.1f * static_cast<float>((h * 5 + i) % 7) - 0.3f;
        }
    }
    const int64_t shape[] = {1, H, W, C};
    TensorView in(std::span<const int64_t>(shape, 4), DataType::f32, x.data(), pitch_bytes,
                  TensorLayout::NHWC);
    TensorView out(std::span<const int64_t>(shape, 4), DataType::f32, y.data(), pitch_bytes,
                   TensorLayout::NHWC);

    SoftmaxAttributes attrs;
    attrs.axis = -1;
    auto op = Softmax::create(attrs, Backend::CPU);
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::ok);

    std::vector<double> want;
    softmax_last_axis_ref(in, y.size(), want);
    for (size_t i = 0; i < want.size(); ++i) {
        if (std::isnan(want[i])) { continue; }  // padding lanes carry no data
        NNOPS_EXPECT_NEAR(y[i], want[i], 1e-6);
    }
}

NNOPS_TEST(softmax_dense_layout) {
    // Rank-2 Dense [4, 8] over the last axis: the same walk, no channel claim.
    const int64_t shape[] = {4, 8};
    std::vector<float> x(32), y(32, -7.0f);
    for (int64_t r = 0; r < 4; ++r) {
        for (int64_t c = 0; c < 8; ++c) {
            x[static_cast<size_t>(r * 8 + c)] = 0.2f * static_cast<float>((r + c) % 5) - 0.4f;
        }
    }
    TensorView in(std::span<const int64_t>(shape, 2), DataType::f32, x.data());
    TensorView out(std::span<const int64_t>(shape, 2), DataType::f32, y.data());

    SoftmaxAttributes attrs;
    attrs.axis = -1;
    auto op = Softmax::create(attrs, Backend::CPU);
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::ok);

    std::vector<double> want;
    softmax_last_axis_ref(in, y.size(), want);
    for (size_t i = 0; i < want.size(); ++i) {
        NNOPS_EXPECT_NEAR(y[i], want[i], 1e-6);
    }
}
