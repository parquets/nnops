/// @file test_pad.cpp
/// @brief Pad tests — the four modes, a negative crop, and the dtype paths.
///
/// The pads attribute is [begin_0 … begin_{R-1}, end_0 … end_{R-1}], so a rank-1
/// input reads {begin, end} and a rank-2 one four values. Constant fills with
/// constant_value (int dtypes round it); Edge repeats the border; Reflect mirrors
/// without repeating it; Wrap tiles. A negative pad crops, which the same
/// per-axis map handles because the source coordinate simply starts inside the
/// axis. f16 and the int dtypes are copied as bits, never through float.

#include "nnops/ops/pad.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <vector>

using namespace nnops;

NNOPS_TEST(pad_constant_1d) {
    float src[] = {1, 2, 3};
    const int64_t shape[] = {3};
    TensorView in(shape, DataType::f32, src);

    PadAttributes attrs;
    attrs.pads = {1, 1};
    attrs.constant_value = 0.0f;
    auto op = Pad::create(attrs);

    const TensorDesc in_arr[] = {in.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 5);

    float out[5] = {};
    TensorView ov = test::make_channel_first(descs[0], out);
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    NNOPS_EXPECT_NEAR(out[0], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out[1], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out[2], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out[3], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out[4], 0.0f, 1e-6f);
}

NNOPS_TEST(pad_constant_2d_value) {
    float src[] = {1, 2, 3, 4};
    const int64_t shape[] = {2, 2};
    TensorView in(shape, DataType::f32, src);

    PadAttributes attrs;
    attrs.pads = {1, 0, 1, 0};   // one row before, one after
    attrs.constant_value = -1.0f;
    auto op = Pad::create(attrs);

    const TensorDesc in_arr[] = {in.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 4);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 2);

    float out[8] = {};
    TensorView ov = test::make_channel_first(descs[0], out);
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    const float want[8] = {-1, -1, 1, 2, 3, 4, -1, -1};
    for (int i = 0; i < 8; ++i) { NNOPS_EXPECT_NEAR(out[i], want[i], 1e-6f); }
}

NNOPS_TEST(pad_edge) {
    float src[] = {1, 2, 3};
    const int64_t shape[] = {3};
    TensorView in(shape, DataType::f32, src);

    PadAttributes attrs;
    attrs.pads = {2, 0};
    attrs.mode = PadAttributes::Mode::Edge;
    auto op = Pad::create(attrs);

    const TensorDesc in_arr[] = {in.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 5);

    float out[5] = {};
    TensorView ov = test::make_channel_first(descs[0], out);
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    const float want[5] = {1, 1, 1, 2, 3};
    for (int i = 0; i < 5; ++i) { NNOPS_EXPECT_NEAR(out[i], want[i], 1e-6f); }
}

NNOPS_TEST(pad_reflect) {
    float src[] = {1, 2, 3, 4};
    const int64_t shape[] = {4};
    TensorView in(shape, DataType::f32, src);

    PadAttributes attrs;
    attrs.pads = {2, 1};
    attrs.mode = PadAttributes::Mode::Reflect;
    auto op = Pad::create(attrs);

    const TensorDesc in_arr[] = {in.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 7);

    float out[7] = {};
    TensorView ov = test::make_channel_first(descs[0], out);
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    // -2→2, -1→1, 4→2
    const float want[7] = {3, 2, 1, 2, 3, 4, 3};
    for (int i = 0; i < 7; ++i) { NNOPS_EXPECT_NEAR(out[i], want[i], 1e-6f); }
}

NNOPS_TEST(pad_wrap) {
    float src[] = {1, 2, 3};
    const int64_t shape[] = {3};
    TensorView in(shape, DataType::f32, src);

    PadAttributes attrs;
    attrs.pads = {1, 2};
    attrs.mode = PadAttributes::Mode::Wrap;
    auto op = Pad::create(attrs);

    const TensorDesc in_arr[] = {in.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 6);

    float out[6] = {};
    TensorView ov = test::make_channel_first(descs[0], out);
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    const float want[6] = {3, 1, 2, 3, 1, 2};
    for (int i = 0; i < 6; ++i) { NNOPS_EXPECT_NEAR(out[i], want[i], 1e-6f); }
}

NNOPS_TEST(pad_negative_crop) {
    float src[] = {1, 2, 3, 4, 5};
    const int64_t shape[] = {5};
    TensorView in(shape, DataType::f32, src);

    PadAttributes attrs;
    attrs.pads = {-1, 0};
    auto op = Pad::create(attrs);

    const TensorDesc in_arr[] = {in.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 4);

    float out[4] = {};
    TensorView ov = test::make_channel_first(descs[0], out);
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    const float want[4] = {2, 3, 4, 5};
    for (int i = 0; i < 4; ++i) { NNOPS_EXPECT_NEAR(out[i], want[i], 1e-6f); }
}

NNOPS_TEST(pad_f16_bits_and_int64) {
    using nnops::backend::cpu::half;

    half h[2];
    simd::s_store(&h[0], 1.5f);
    simd::s_store(&h[1], -2.25f);
    const int64_t h_shape[] = {2};
    TensorView hin(h_shape, DataType::f16, h);

    PadAttributes attrs;
    attrs.pads = {1, 1};
    attrs.constant_value = 0.5f;
    auto op = Pad::create(attrs);

    const TensorDesc h_arr[] = {hin.desc()};
    auto h_descs = op->getOutputTensorDesc(h_arr);
    half h_out[4];
    TensorView h_ov = test::make_channel_first(h_descs[0], h_out);
    const TensorView h_ins[] = {hin};
    NNOPS_EXPECT_TRUE(op->compute(h_ov, h_ins) == Status::ok);

    NNOPS_EXPECT_NEAR(simd::s_load(&h_out[0]), 0.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(simd::s_load(&h_out[1]), 1.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(simd::s_load(&h_out[2]), -2.25f, 1e-6f);
    NNOPS_EXPECT_NEAR(simd::s_load(&h_out[3]), 0.5f, 1e-6f);

    int64_t i[2] = {7, -9};
    const int64_t i_shape[] = {2};
    TensorView iin(i_shape, DataType::s64, i);
    PadAttributes iattrs;
    iattrs.pads = {1, 1};
    iattrs.constant_value = 2.6f;  // rounds to 3
    auto iop = Pad::create(iattrs);

    const TensorDesc i_arr[] = {iin.desc()};
    auto i_descs = iop->getOutputTensorDesc(i_arr);
    int64_t i_out[4] = {};
    TensorView i_ov = test::make_channel_first(i_descs[0], i_out);
    const TensorView i_ins[] = {iin};
    NNOPS_EXPECT_TRUE(iop->compute(i_ov, i_ins) == Status::ok);

    NNOPS_EXPECT_EQ(i_out[0], int64_t(3));
    NNOPS_EXPECT_EQ(i_out[1], int64_t(7));
    NNOPS_EXPECT_EQ(i_out[2], int64_t(-9));
    NNOPS_EXPECT_EQ(i_out[3], int64_t(3));
}

NNOPS_TEST(pad_rejects_bad_attributes) {
    float src[] = {1, 2, 3};
    const int64_t shape[] = {3};
    TensorView in(shape, DataType::f32, src);

    {
        PadAttributes attrs;
        attrs.pads = {1};  // needs 2 * rank
        auto op = Pad::create(attrs);
        const TensorDesc in_arr[] = {in.desc()};
        const TensorDesc out_arr[] = {in.desc()};
        NNOPS_EXPECT_TRUE(op->validate(in_arr, out_arr) == Status::invalid_argument);
    }
    {
        PadAttributes attrs;
        attrs.pads = {4, 0};
        attrs.mode = PadAttributes::Mode::Reflect;  // reflection past the axis
        auto op = Pad::create(attrs);
        const TensorDesc in_arr[] = {in.desc()};
        const TensorDesc out_arr[] = {in.desc()};
        NNOPS_EXPECT_TRUE(op->validate(in_arr, out_arr) == Status::invalid_argument);
    }
}
