/// @file test_softmax.cpp
/// @brief Softmax operator tests (CPU, class API): softmax and log-softmax
///        along an axis, temperature, f16, both regular layouts (channel-first and
///        channel-last) and quantized input.
///
/// The layout tests run every axis of a 4-D tensor against a reference that
/// walks the same physical indexing (channel-first [N][C][H][W], channel-last
/// [N][H][W][C]) in its own code, so an axis handled through the wrong stride
/// cannot pass. Padded row pitch and channel-packed rejection are covered too.
///
/// Quantized input (s8/u8) is dequantized before the float softmax, so the
/// output is always floating point (f32 by default, f16 if supplied) and never
/// requantized. PerTensor and PerToken scales are both covered.

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

    TensorView input(shape, DataType::f32, in_data);
    auto d_in = input.desc();

    auto op = Softmax::create({}, Backend::CPU);

    const TensorDesc in_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

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

    TensorView input(shape, DataType::f32, in_data);
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
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

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

    TensorView input(shape, DataType::f32, in_data);
    auto d_in = input.desc();

    SoftmaxAttributes attrs;
    attrs.log_softmax = true;
    auto op = Softmax::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

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
    auto [in_vec, input] = test::make_random_tensor({4, 8}, -2.0f, 2.0f);
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
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

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
// Layout coverage — channel-first and channel-last
// ============================================================
//
// Both families keep the logical dims [N][C][H][W]; only the physical order
// differs. Every axis is compared against a reference that re-derives the
// offsets from that order on its own, so an axis walked with the wrong stride
// cannot pass. Dimensions stay <= 16 so a group fits a stack array.

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

NNOPS_TEST(softmax_channel_first_all_axes) {
    const int64_t dims[4] = {2, 3, 4, 5};
    const int64_t C = dims[1], H = dims[2], W = dims[3];

    auto [in_data, view] = test::make_random_tensor({dims[0], C, H, W}, -2.0f, 2.0f, 11);
    (void)view;

    check_all_axes(in_data, dims, TensorLayout::NCHW,
                   [&](int64_t n, int64_t c, int64_t h, int64_t w) {
                       return ((n * C + c) * H + h) * W + w;
                   });
}

NNOPS_TEST(softmax_channel_last_all_axes) {
    // The strided path lanes across channels, so C = 5 is a partial vector and
    // C = 12 is a full one plus a half; the channel axis itself is a channel-last
    // row, whose own tail is the last group's.
    for (const int64_t C : {int64_t{5}, int64_t{12}}) {
        const int64_t dims[4] = {2, C, 3, 4};
        const int64_t H = dims[2], W = dims[3];

        auto [in_data, view] = test::make_random_tensor({dims[0], C, H, W}, -2.0f, 2.0f, 13);
        (void)view;

        check_all_axes(in_data, dims, TensorLayout::NHWC,
                       [&](int64_t n, int64_t c, int64_t h, int64_t w) {
                           return ((n * H + h) * W + w) * C + c;
                       });
    }
}

NNOPS_TEST(softmax_channel_last_2d_channel) {
    // Rank-2 channel-last is still [N][C]: axis 1 is one contiguous pixel.
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

NNOPS_TEST(softmax_channel_last_channel_axis_f16) {
    const int64_t N = 1, C = 10, H = 2, W = 2;
    const int64_t shape[] = {N, C, H, W};

    auto [f32_buf, view] = test::make_random_tensor({N, C, H, W}, -2.0f, 2.0f, 23);
    (void)view;
    auto f16_buf = test::f32_to_f16(f32_buf);
    TensorView input(shape, DataType::f16, f16_buf.data(), TensorLayout::NHWC);

    SoftmaxAttributes attrs;
    attrs.axis = 1;
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

NNOPS_TEST(softmax_channel_first_pitched_rows) {
    // A pitch one 32-byte step past the dense row. The last-axis path addresses
    // whole rows; the strided path may only lane across dims that stay inside
    // one row. The padding doubles as an overrun detector.
    const int64_t N = 2, C = 3, H = 2, W = 4;
    const int64_t dims[4] = {N, C, H, W};
    const int64_t shape[] = {N, C, H, W};
    const int64_t rows = N * C * H;
    const int64_t pitch = W * static_cast<int64_t>(sizeof(float)) + 32;
    const int64_t pe = pitch / static_cast<int64_t>(sizeof(float));
    const float kPad = -777.0f;

    auto [rnd, view] = test::make_random_tensor({N, C, H, W}, -2.0f, 2.0f, 17);
    (void)view;

    std::vector<float> in_buf(static_cast<size_t>(rows * pe), kPad);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t w = 0; w < W; ++w) {
            in_buf[static_cast<size_t>(r * pe + w)] = rnd[static_cast<size_t>(r * W + w)];
        }
    }

    TensorView input(shape, DataType::f32, in_buf.data(), pitch, TensorLayout::NCHW);

    for (int64_t axis = 0; axis < 4; ++axis) {
        const auto expected = layout_softmax_reference(
            in_buf, dims, axis, /*log_softmax=*/false,
            [&](int64_t n, int64_t c, int64_t h, int64_t w) {
                return ((n * C + c) * H + h) * pe + w;
            });

        SoftmaxAttributes attrs;
        attrs.axis = axis;
        auto op = Softmax::create(attrs, Backend::CPU);

        std::vector<float> out_buf(in_buf.size(), kPad);
        TensorView output(shape, DataType::f32, out_buf.data(), pitch, TensorLayout::NCHW);

        const TensorView ins[] = {input};
        NNOPS_EXPECT_TRUE(is_ok(op->compute(output, ins)));

        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t w = 0; w < W; ++w) {
                NNOPS_EXPECT_NEAR(out_buf[static_cast<size_t>(r * pe + w)],
                                  expected[static_cast<size_t>(r * pe + w)], 1e-5f);
            }
            for (int64_t w = W; w < pe; ++w) {
                NNOPS_EXPECT_EQ(out_buf[static_cast<size_t>(r * pe + w)], kPad);
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
    in_desc.layout = TensorLayout::PackedOIHW;
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

    TensorView input(shape, DataType::f32, in_data);
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

    TensorView input(shape, DataType::f32, in_data);
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

    TensorView input(shape, DataType::f32, in_data);
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

NNOPS_TEST(softmax_channel_first_axis_first) {
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

NNOPS_TEST(softmax_random_f16) {
    auto [f16_buf, input] = test::make_random_f16_tensor({4, 8}, -2.0f, 2.0f);

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
// Quantized input (s8/u8) → float output (f32/f16)
// ============================================================

NNOPS_TEST(softmax_quant_s8_per_tensor) {
    // in {2, 4, 6} x scale 0.5 → dequantized {1, 2, 3}.
    const int64_t shape[] = {3};
    int8_t in_data[] = {2, 4, 6};

    QuantParams qp;
    qp.scale = 0.5f;
    qp.granularity = QuantGranularity::PerTensor;
    TensorView input(shape, DataType::s8, in_data, TensorLayout::NCHW, qp);

    auto op = Softmax::create({}, Backend::CPU);
    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(descs[0].numel());
    TensorView output = nnops::test::make_channel_first(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(output, ins, {}, nullptr);

    float sum_exp = std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f);
    NNOPS_EXPECT_NEAR(out_buf[0], std::exp(1.0f) / sum_exp, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[1], std::exp(2.0f) / sum_exp, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[2], std::exp(3.0f) / sum_exp, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[0] + out_buf[1] + out_buf[2], 1.0f, 1e-5f);
}

NNOPS_TEST(softmax_quant_u8_per_tensor) {
    // (in - 128) * 0.5 for {130, 132, 134} → dequantized {1, 2, 3}.
    const int64_t shape[] = {3};
    uint8_t in_data[] = {130, 132, 134};

    QuantParams qp;
    qp.scale = 0.5f;
    qp.zero_point = 128;
    qp.granularity = QuantGranularity::PerTensor;
    TensorView input(shape, DataType::u8, in_data, TensorLayout::NCHW, qp);

    auto op = Softmax::create({}, Backend::CPU);
    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(descs[0].numel());
    TensorView output = nnops::test::make_channel_first(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(output, ins, {}, nullptr);

    float sum_exp = std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f);
    NNOPS_EXPECT_NEAR(out_buf[0], std::exp(1.0f) / sum_exp, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[1], std::exp(2.0f) / sum_exp, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[2], std::exp(3.0f) / sum_exp, 1e-4f);
}

NNOPS_TEST(softmax_quant_s8_per_token) {
    // Per-token s8, rows {1,2,3} and {2,4,6} with scales {1, 0.5}.
    // Both rows dequantize to {1, 2, 3}, so softmax over axis=1 is identical.
    const int64_t shape[] = {2, 3};
    int8_t in_data[] = {1, 2, 3,  2, 4, 6};
    float scale_data[] = {1.0f, 0.5f};

    QuantParams qp;
    qp.granularity = QuantGranularity::PerToken;
    qp.scale_data = scale_data;
    qp.num_scales = 2;
    TensorView input(shape, DataType::s8, in_data, TensorLayout::NCHW, qp);

    SoftmaxAttributes attrs;
    attrs.axis = 1;
    auto op = Softmax::create(attrs, Backend::CPU);
    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(descs[0].numel());
    TensorView output = nnops::test::make_channel_first(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(output, ins, {}, nullptr);

    float sum_exp = std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f);
    const float e0 = std::exp(1.0f) / sum_exp;
    const float e1 = std::exp(2.0f) / sum_exp;
    const float e2 = std::exp(3.0f) / sum_exp;
    for (int r = 0; r < 2; ++r) {
        NNOPS_EXPECT_NEAR(out_buf[r * 3 + 0], e0, 1e-4f);
        NNOPS_EXPECT_NEAR(out_buf[r * 3 + 1], e1, 1e-4f);
        NNOPS_EXPECT_NEAR(out_buf[r * 3 + 2], e2, 1e-4f);
    }
}

NNOPS_TEST(softmax_quant_s8_f16_output) {
    const int64_t shape[] = {3};
    int8_t in_data[] = {2, 4, 6};

    QuantParams qp;
    qp.scale = 0.5f;
    qp.granularity = QuantGranularity::PerTensor;
    TensorView input(shape, DataType::s8, in_data, TensorLayout::NCHW, qp);

    auto op = Softmax::create({}, Backend::CPU);

    std::vector<nnops::backend::cpu::half> out_buf(3);
    TensorView output(shape, DataType::f16, out_buf.data(), TensorLayout::NCHW);

    const TensorDesc in_arr[] = {input.desc()};
    const TensorDesc out_arr[] = {output.desc()};
    const TensorView ins[] = {input};
    op->compute(output, ins, {}, nullptr);

    float sum_exp = std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f);
    NNOPS_EXPECT_NEAR(simd::s_load(&out_buf[0]), std::exp(1.0f) / sum_exp, 1e-2f);
    NNOPS_EXPECT_NEAR(simd::s_load(&out_buf[1]), std::exp(2.0f) / sum_exp, 1e-2f);
    NNOPS_EXPECT_NEAR(simd::s_load(&out_buf[2]), std::exp(3.0f) / sum_exp, 1e-2f);
}

