/// @file test_activation.cpp
/// @brief Activation (CPU): f32 activations against closed-form values, the f16
/// path, and the fused dequant → activation → quant route.

#include "nnops/ops/activation.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"
#include "common/channel_last.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

NNOPS_TEST(activation_relu) {
    const int64_t shape[] = {8};
    float in_data[]  = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 10.0f, -10.0f};
    float expected[]  = { 0.0f,  0.0f, 0.0f, 1.0f, 2.0f,  0.0f, 10.0f,  0.0f};

    TensorView input(shape, DataType::f32, in_data);
    auto d = input.desc();

    ActivationAttributes attrs;
    attrs.type = ActivationType::Relu;
    auto op = Activation::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_TRUE(descs[0].rank == 1);
    NNOPS_EXPECT_TRUE(descs[0].dims[0] == 8);
    NNOPS_EXPECT_TRUE(descs[0].layout == TensorLayout::NCHW);
    NNOPS_EXPECT_TRUE(descs[0].dtype == DataType::f32);

    std::vector<float> buf(static_cast<size_t>(descs[0].numel()));
    auto out = nnops::test::make_planar(descs[0], buf.data());

    const TensorView ins[] = {input};
    op->compute(out, ins);

    for (int i = 0; i < 8; ++i) {
        NNOPS_EXPECT_NEAR(buf[i], expected[i], 1e-6f);
    }
}

NNOPS_TEST(activation_leaky_relu) {
    const int64_t shape[] = {4};
    float in_data[]  = {-2.0f, 0.0f, 1.0f, -1.0f};
    const float alpha = 0.01f;

    TensorView input(shape, DataType::f32, in_data);
    auto d = input.desc();

    ActivationAttributes attrs;
    attrs.type = ActivationType::LeakyRelu;
    attrs.alpha = alpha;
    auto op = Activation::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_TRUE(descs[0].rank == 1);
    NNOPS_EXPECT_TRUE(descs[0].dims[0] == 4);
    NNOPS_EXPECT_TRUE(descs[0].layout == TensorLayout::NCHW);
    NNOPS_EXPECT_TRUE(descs[0].dtype == DataType::f32);

    std::vector<float> buf(static_cast<size_t>(descs[0].numel()));
    auto out = nnops::test::make_planar(descs[0], buf.data());

    const TensorView ins[] = {input};
    op->compute(out, ins);

    NNOPS_EXPECT_NEAR(buf[0], alpha * (-2.0f), 1e-6f);
    NNOPS_EXPECT_NEAR(buf[1], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(buf[2], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(buf[3], alpha * (-1.0f), 1e-6f);
}

NNOPS_TEST(activation_sigmoid) {
    const int64_t shape[] = {3};
    float in_data[]  = {0.0f, 10.0f, -10.0f};

    TensorView input(shape, DataType::f32, in_data);
    auto d = input.desc();

    ActivationAttributes attrs;
    attrs.type = ActivationType::Sigmoid;
    auto op = Activation::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_TRUE(descs[0].rank == 1);
    NNOPS_EXPECT_TRUE(descs[0].dims[0] == 3);
    NNOPS_EXPECT_TRUE(descs[0].layout == TensorLayout::NCHW);
    NNOPS_EXPECT_TRUE(descs[0].dtype == DataType::f32);

    std::vector<float> buf(static_cast<size_t>(descs[0].numel()));
    auto out = nnops::test::make_planar(descs[0], buf.data());

    const TensorView ins[] = {input};
    op->compute(out, ins);

    NNOPS_EXPECT_NEAR(buf[0], 0.5f, 1e-6f);       // sigmoid(0)
    NNOPS_EXPECT_NEAR(buf[1], 0.99995f, 1e-4f);   // sigmoid(10)
    NNOPS_EXPECT_NEAR(buf[2], 0.000045f, 1e-4f);  // sigmoid(-10)
}

NNOPS_TEST(activation_tanh) {
    const int64_t shape[] = {3};
    float in_data[]  = {0.0f, 2.0f, -2.0f};

    TensorView input(shape, DataType::f32, in_data);
    auto d = input.desc();

    ActivationAttributes attrs;
    attrs.type = ActivationType::Tanh;
    auto op = Activation::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_TRUE(descs[0].rank == 1);
    NNOPS_EXPECT_TRUE(descs[0].dims[0] == 3);
    NNOPS_EXPECT_TRUE(descs[0].layout == TensorLayout::NCHW);
    NNOPS_EXPECT_TRUE(descs[0].dtype == DataType::f32);

    std::vector<float> buf(static_cast<size_t>(descs[0].numel()));
    auto out = nnops::test::make_planar(descs[0], buf.data());

    const TensorView ins[] = {input};
    op->compute(out, ins);

    NNOPS_EXPECT_NEAR(buf[0], 0.0f, 1e-6f);     // tanh(0)
    NNOPS_EXPECT_NEAR(buf[1], 0.9640f, 1e-3f);  // tanh(2)
    NNOPS_EXPECT_NEAR(buf[2], -0.9640f, 1e-3f); // tanh(-2)
}

NNOPS_TEST(activation_gelu) {
    const int64_t shape[] = {2};
    float in_data[]  = {0.0f, 1.0f};

    TensorView input(shape, DataType::f32, in_data);
    auto d = input.desc();

    ActivationAttributes attrs;
    attrs.type = ActivationType::Gelu;
    auto op = Activation::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_TRUE(descs[0].rank == 1);
    NNOPS_EXPECT_TRUE(descs[0].dims[0] == 2);
    NNOPS_EXPECT_TRUE(descs[0].layout == TensorLayout::NCHW);
    NNOPS_EXPECT_TRUE(descs[0].dtype == DataType::f32);

    std::vector<float> buf(static_cast<size_t>(descs[0].numel()));
    auto out = nnops::test::make_planar(descs[0], buf.data());

    const TensorView ins[] = {input};
    op->compute(out, ins);

    NNOPS_EXPECT_NEAR(buf[0], 0.0f, 1e-3f);    // GELU(0)
    NNOPS_EXPECT_NEAR(buf[1], 0.8413f, 1e-3f); // GELU(1)
}

NNOPS_TEST(activation_silu) {
    const int64_t shape[] = {2};
    float in_data[]  = {0.0f, 1.0f};

    TensorView input(shape, DataType::f32, in_data);
    auto d = input.desc();

    ActivationAttributes attrs;
    attrs.type = ActivationType::Silu;
    auto op = Activation::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_TRUE(descs[0].rank == 1);
    NNOPS_EXPECT_TRUE(descs[0].dims[0] == 2);
    NNOPS_EXPECT_TRUE(descs[0].layout == TensorLayout::NCHW);
    NNOPS_EXPECT_TRUE(descs[0].dtype == DataType::f32);

    std::vector<float> buf(static_cast<size_t>(descs[0].numel()));
    auto out = nnops::test::make_planar(descs[0], buf.data());

    const TensorView ins[] = {input};
    op->compute(out, ins);

    NNOPS_EXPECT_NEAR(buf[0], 0.0f, 1e-6f);    // SiLU(0)
    NNOPS_EXPECT_NEAR(buf[1], 0.7311f, 1e-3f); // SiLU(1)
}

NNOPS_TEST(activation_random_relu) {
    auto [in_vec, input] = test::make_random_tensor({1000}, -5.0f, 5.0f);
    auto d = input.desc();

    ActivationAttributes attrs;
    attrs.type = ActivationType::Relu;
    auto op = Activation::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_TRUE(descs[0].rank == 1);
    NNOPS_EXPECT_TRUE(descs[0].dims[0] == 1000);
    NNOPS_EXPECT_TRUE(descs[0].layout == TensorLayout::NCHW);
    NNOPS_EXPECT_TRUE(descs[0].dtype == DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    auto out = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(out, ins);

    for (int i = 0; i < 1000; ++i) {
        NNOPS_EXPECT_TRUE(out_buf[i] >= 0.0f);
    }
}

// ============================================================
// f16 tests — exercise the SIMD f16 code path
// ============================================================

NNOPS_TEST(activation_random_relu_f16) {
    auto [f32_vec, _] = test::make_random_tensor({500}, -5.0f, 5.0f, 800);

    auto f16_vec = test::f32_to_f16(f32_vec);

    const int64_t shape[] = {500};
    TensorView input(shape, DataType::f16, f16_vec.data());

    ActivationAttributes attrs;
    attrs.type = ActivationType::Relu;
    auto op = Activation::create(attrs, Backend::CPU);

    auto d = input.desc();
    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

    std::vector<nnops::backend::cpu::half> out_buf(500);
    auto out = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(out, ins);

    for (int i = 0; i < 500; ++i) {
        float result = simd::s_load(&out_buf[i]);
        NNOPS_EXPECT_TRUE(result >= -1e-5f);  // ReLU: >= 0
        float expected = f32_vec[i] > 0.0f ? f32_vec[i] : 0.0f;
        NNOPS_EXPECT_NEAR(result, expected, 1e-2f);
    }
}

// ============================================================
// Quantized input/output tests (fused dequant → activation → quant)
// ============================================================

NNOPS_TEST(activation_quant_per_tensor_relu_s8) {
    const int64_t shape[] = {6};
    int8_t in_data[] = {-4, -2, 0, 2, 4, 6};

    QuantParams qp;
    qp.scale = 0.5f;
    qp.granularity = QuantGranularity::PerTensor;
    TensorView input(shape, DataType::s8, in_data, TensorLayout::NCHW, qp);

    ActivationAttributes attrs;
    attrs.type = ActivationType::Relu;
    auto op = Activation::create(attrs, Backend::CPU);

    int8_t out_buf[6] = {};
    TensorView output(shape, DataType::s8, out_buf, TensorLayout::NCHW, qp);

    const TensorView ins[] = {input};
    op->compute(output, ins);

    // dequant: [-2,-1,0,1,2,3] → relu: [0,0,0,1,2,3] → /0.5: [0,0,0,2,4,6]
    const int8_t expected[] = {0, 0, 0, 2, 4, 6};
    for (int i = 0; i < 6; ++i) {
        NNOPS_EXPECT_EQ(out_buf[i], expected[i]);
    }
}

NNOPS_TEST(activation_quant_per_tensor_sigmoid_s8) {
    const int64_t shape[] = {2};
    int8_t in_data[] = {0, 2};

    QuantParams qp;
    qp.scale = 1.0f;
    qp.granularity = QuantGranularity::PerTensor;
    TensorView input(shape, DataType::s8, in_data, TensorLayout::NCHW, qp);

    ActivationAttributes attrs;
    attrs.type = ActivationType::Sigmoid;
    auto op = Activation::create(attrs, Backend::CPU);

    int8_t out_buf[2] = {};
    TensorView output(shape, DataType::s8, out_buf, TensorLayout::NCHW, qp);

    const TensorView ins[] = {input};
    op->compute(output, ins);

    // sigmoid(0)=0.5 → ties-to-even → 0; sigmoid(2)=0.8808 → 1
    NNOPS_EXPECT_EQ(out_buf[0], 0);
    NNOPS_EXPECT_EQ(out_buf[1], 1);
}

NNOPS_TEST(activation_quant_per_tensor_relu_u8) {
    const int64_t shape[] = {4};
    uint8_t in_data[] = {126, 127, 128, 129};

    QuantParams qp;
    qp.scale = 0.5f;
    qp.zero_point = 128;
    qp.granularity = QuantGranularity::PerTensor;
    TensorView input(shape, DataType::u8, in_data, TensorLayout::NCHW, qp);

    ActivationAttributes attrs;
    attrs.type = ActivationType::Relu;
    auto op = Activation::create(attrs, Backend::CPU);

    uint8_t out_buf[4] = {};
    TensorView output(shape, DataType::u8, out_buf, TensorLayout::NCHW, qp);

    const TensorView ins[] = {input};
    op->compute(output, ins);

    // dequant: [-1,-0.5,0,0.5] → relu: [0,0,0,0.5] → /0.5+128: [128,128,128,129]
    NNOPS_EXPECT_EQ(out_buf[0], 128);
    NNOPS_EXPECT_EQ(out_buf[1], 128);
    NNOPS_EXPECT_EQ(out_buf[2], 128);
    NNOPS_EXPECT_EQ(out_buf[3], 129);
}

// ============================================================
// Graph-support set — the closed form is recomputed here in double-free fp32
// arithmetic, so a wrong branch (or a mixed-up alpha/beta) shows up directly.
// ============================================================

template <class F>
static void check_activation_f32(const std::vector<int64_t>& shape,
                                 const ActivationAttributes& attrs, F&& ref,
                                 float tol = 1e-4f)
{
    auto [in_vec, input] = test::make_random_tensor(shape, -3.0f, 3.0f);
    auto op = Activation::create(attrs, Backend::CPU);

    const TensorDesc in_desc[] = {input.desc()};
    auto out_descs = op->getOutputTensorDesc(in_desc);

    std::vector<float> buf(static_cast<size_t>(out_descs[0].numel()));
    auto out = nnops::test::make_planar(out_descs[0], buf.data());

    const TensorView ins[] = {input};
    NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));

    for (int64_t i = 0; i < input.numel(); ++i) {
        NNOPS_EXPECT_NEAR(buf[static_cast<size_t>(i)],
                          ref(in_vec[static_cast<size_t>(i)]), tol);
    }
}

NNOPS_TEST(activation_hard_sigmoid) {
    ActivationAttributes attrs;
    attrs.type  = ActivationType::HardSigmoid;
    attrs.alpha = 0.2f;
    attrs.beta  = 0.5f;
    check_activation_f32({257}, attrs, [](float x) {
        return std::min(std::max(0.2f * x + 0.5f, 0.0f), 1.0f);
    });
}

NNOPS_TEST(activation_softplus) {
    ActivationAttributes attrs;
    attrs.type = ActivationType::Softplus;
    attrs.beta = 2.0f;
    check_activation_f32({257}, attrs, [](float x) {
        const float z = 2.0f * x;
        return (std::max(z, 0.0f) + std::log1p(std::exp(-std::fabs(z)))) / 2.0f;
    });
}

NNOPS_TEST(activation_selu) {
    const float alpha = 1.6732632f;
    const float scale = 1.0507010f;
    ActivationAttributes attrs;
    attrs.type  = ActivationType::Selu;
    attrs.alpha = alpha;
    attrs.beta  = scale;
    check_activation_f32({257}, attrs, [=](float x) {
        return scale * (x > 0.0f ? x : alpha * (std::exp(x) - 1.0f));
    });
}

NNOPS_TEST(activation_mish) {
    ActivationAttributes attrs;
    attrs.type = ActivationType::Mish;
    check_activation_f32({257}, attrs, [](float x) {
        return x * std::tanh(std::max(x, 0.0f) +
                             std::log1p(std::exp(-std::fabs(x))));
    });
}

NNOPS_TEST(activation_gelu_exact) {
    ActivationAttributes attrs;
    attrs.type = ActivationType::GeluExact;
    check_activation_f32({257}, attrs, [](float x) {
        return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f));
    });
}

NNOPS_TEST(activation_quick_gelu) {
    ActivationAttributes attrs;
    attrs.type  = ActivationType::QuickGelu;
    attrs.alpha = 1.702f;
    check_activation_f32({257}, attrs, [](float x) {
        return x / (1.0f + std::exp(-1.702f * x));
    });
}

// ============================================================
// PRelu — the slope is a second input, one per logical channel (dim 1)
// ============================================================

NNOPS_TEST(activation_prelu_planar) {
    const int64_t H = 5, W = 7;
    auto [in_vec, input] = test::make_random_tensor({1, 3, H, W}, -4.0f, 4.0f);
    float slope_data[] = {2.0f, 0.5f, -1.0f};
    const int64_t sshape[] = {3};
    TensorView slope(sshape, DataType::f32, slope_data);

    ActivationAttributes attrs;
    attrs.type = ActivationType::PRelu;
    auto op = Activation::create(attrs);

    const TensorDesc in_descs[] = {input.desc(), slope.desc()};
    auto out_descs = op->getOutputTensorDesc(in_descs);
    NNOPS_EXPECT_TRUE(out_descs[0].rank == 4 && out_descs[0].dims[1] == 3);

    std::vector<float> buf(static_cast<size_t>(out_descs[0].numel()));
    auto out = nnops::test::make_planar(out_descs[0], buf.data());

    const TensorView ins[] = {input, slope};
    NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));

    for (int64_t i = 0; i < input.numel(); ++i) {
        const int64_t c = (i / (H * W)) % 3;
        const float x = in_vec[static_cast<size_t>(i)];
        const float expected = x > 0.0f ? x : slope_data[c] * x;
        NNOPS_EXPECT_NEAR(buf[static_cast<size_t>(i)], expected, 1e-6f);
    }
}

NNOPS_TEST(activation_prelu_nhwc) {
    constexpr int64_t N = 2, C = 5, H = 4, W = 6;
    auto [in_vec, in_planar] = test::make_random_tensor({N, C, H, W}, -4.0f, 4.0f);
    float slope_data[C];
    for (int64_t c = 0; c < C; ++c) { slope_data[c] = 0.1f * (c + 1); }

    auto cl = test::to_channel_last<float>(in_vec, N, C, H * W);
    // Logical shape stays [N, C, H, W]; the NHWC tag changes only the storage.
    const int64_t in_shape[] = {N, C, H, W};
    TensorView input(in_shape, DataType::f32, cl.data(), TensorLayout::NHWC);

    const int64_t sshape[] = {C};
    TensorView slope(sshape, DataType::f32, slope_data);

    ActivationAttributes attrs;
    attrs.type = ActivationType::PRelu;
    auto op = Activation::create(attrs);

    const TensorDesc in_descs[] = {input.desc(), slope.desc()};
    auto out_descs = op->getOutputTensorDesc(in_descs);
    NNOPS_EXPECT_TRUE(out_descs[0].layout == TensorLayout::NHWC);

    std::vector<float> buf(static_cast<size_t>(out_descs[0].numel()));
    TensorView out(std::span<const int64_t>(in_shape, 4), DataType::f32,
                   buf.data(), TensorLayout::NHWC);

    const TensorView ins[] = {input, slope};
    NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));

    const auto got = test::to_channel_first<float>(buf, N, C, H * W);
    for (int64_t i = 0; i < N * C * H * W; ++i) {
        const int64_t c = (i / (H * W)) % C;
        const float x = in_vec[static_cast<size_t>(i)];
        const float expected = x > 0.0f ? x : slope_data[c] * x;
        NNOPS_EXPECT_NEAR(got[static_cast<size_t>(i)], expected, 1e-6f);
    }
}

NNOPS_TEST(activation_prelu_single_slope) {
    auto [in_vec, input] = test::make_random_tensor({1, 4, 3, 3}, -4.0f, 4.0f);
    float slope_data[] = {0.25f};
    const int64_t sshape[] = {1};
    TensorView slope(sshape, DataType::f32, slope_data);

    ActivationAttributes attrs;
    attrs.type = ActivationType::PRelu;
    auto op = Activation::create(attrs);

    const TensorDesc in_descs[] = {input.desc(), slope.desc()};
    auto out_descs = op->getOutputTensorDesc(in_descs);
    std::vector<float> buf(static_cast<size_t>(out_descs[0].numel()));
    auto out = nnops::test::make_planar(out_descs[0], buf.data());

    const TensorView ins[] = {input, slope};
    NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));

    for (int64_t i = 0; i < input.numel(); ++i) {
        const float x = in_vec[static_cast<size_t>(i)];
        NNOPS_EXPECT_NEAR(buf[static_cast<size_t>(i)],
                          x > 0.0f ? x : 0.25f * x, 1e-6f);
    }
}

NNOPS_TEST(activation_prelu_rejects_bad_slope) {
    TensorDesc x;
    x.rank = 4; x.layout = TensorLayout::NCHW; x.dtype = DataType::f32;
    for (int64_t d : {1, 3, 4, 4}) { x.dims.push_back(d); }

    TensorDesc slope;
    slope.rank = 1; slope.layout = TensorLayout::NCHW; slope.dtype = DataType::f32;
    slope.dims.push_back(2);  // neither 3 nor 1

    ActivationAttributes attrs;
    attrs.type = ActivationType::PRelu;
    auto op = Activation::create(attrs);

    TensorDesc out = x;
    const TensorDesc ins[] = {x, slope};
    const TensorDesc outs[] = {out};
    NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::invalid_argument);
}
