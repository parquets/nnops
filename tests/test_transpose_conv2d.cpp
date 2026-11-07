/// @file test_transpose_conv2d.cpp
/// @brief TransposeConv2D correctness tests — channel-first and channel-last.
///
/// Every case round-trips the channel-last path: channel-first input ->
/// LayoutConvert -> compute -> LayoutConvert back to channel-first, then compare.
/// Weight stays channel-first [IC, OC/G, KH, KW] (no prepacking), so one weight
/// tensor feeds both the channel-last op and the scalar golden.
///
/// Two kinds of case: hand-computed small shapes that pin exact output values
/// and output geometry, and random roundtrips against
/// reference::transpose_conv2d_channel_first_ref (tests/reference).
///
/// Channel-last input:  logical [N, IC, IH, IW] stored [N, IH, IW, IC]
/// Channel-last output: logical [N, OC, OH, OW] stored [N, OH, OW, OC]

#include "nnops/ops/transpose_conv2d.hpp"
#include "nnops/ops/layout_convert.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/channel_last.hpp"

#include <span>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>

using namespace nnops;

// Defined in tests/reference/; the channel-first scalar golden used for cross-checks.
namespace nnops::backend::cpu::reference {
    void transpose_conv2d_channel_first_ref(const TransposeConv2DAttributes& attrs,
                                    TensorView& output,
                                    std::span<const TensorView> inputs,
                                    const ComputeContext& ctx,
                                    void* workspace);
}

/// Roundtrip: convert input to channel-last -> TransposeConv2D::compute -> convert back
/// -> compare with the channel-first scalar golden.
static void test_channel_last_vs_ref(
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& w_shape,
    const TransposeConv2DAttributes& attrs,
    bool has_bias = false)
{
    const int64_t OC_per_G = w_shape[1];
    const int64_t G = attrs.groups;
    const int64_t OC = OC_per_G * G;

    auto [in_vec, in_channel_first] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_channel_first]  = test::make_random_tensor(w_shape);
    std::vector<float> b_vec;
    TensorView b_channel_first;
    if (has_bias) {
        auto p = test::make_random_tensor({OC});
        b_vec = std::move(p.first);
        b_channel_first = p.second;
    }

    auto tconv_op_ref = TransposeConv2D::create(attrs, Backend::CPU);

    TensorDesc in_desc_ref = in_channel_first.desc();
    TensorDesc w_desc_ref  = w_channel_first.desc();
    std::vector<TensorDesc> ref_descs;
    if (has_bias) {
        TensorDesc b_desc_ref = b_channel_first.desc();
        const TensorDesc ref_desc_arr[] = {in_desc_ref, w_desc_ref, b_desc_ref};
        ref_descs = tconv_op_ref->getOutputTensorDesc(ref_desc_arr);
    } else {
        const TensorDesc ref_desc_arr[] = {in_desc_ref, w_desc_ref};
        ref_descs = tconv_op_ref->getOutputTensorDesc(ref_desc_arr);
    }

    std::vector<float> ref_buf(static_cast<size_t>(ref_descs[0].numel()));
    auto out_ref = test::make_channel_first(ref_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        if (has_bias) {
            const TensorView ref_arr[] = {in_channel_first, w_channel_first, b_channel_first};
            backend::cpu::reference::transpose_conv2d_channel_first_ref(
                attrs, out_ref, ref_arr, ctx, nullptr);
        } else {
            const TensorView ref_arr[] = {in_channel_first, w_channel_first};
            backend::cpu::reference::transpose_conv2d_channel_first_ref(
                attrs, out_ref, ref_arr, ctx, nullptr);
        }
    }

    auto lc_in = LayoutConvert::create(TensorLayout::NHWC, Backend::CPU);
    TensorDesc in_channel_first_desc = in_channel_first.desc();
    const TensorDesc lc_in_desc_arr[] = {in_channel_first_desc};
    auto lc_in_descs = lc_in->getOutputTensorDesc(lc_in_desc_arr);

    std::vector<float> cl_in_buf(
        static_cast<size_t>(lc_in_descs[0].storage_bytes()));
    auto in_cl = test::make_packed(lc_in_descs[0], cl_in_buf.data());
    {
        const TensorView lc_ins[] = {in_channel_first};
        TensorView lc_outs[] = {in_cl};
        lc_in->compute(lc_outs, lc_ins);
    }

    (void)w_vec;  // w_channel_first references it

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    TensorDesc in_cl_desc = in_cl.desc();
    std::vector<TensorDesc> out_descs;
    if (has_bias) {
        const TensorDesc tc_desc_arr[] = {in_cl_desc, w_desc_ref, b_channel_first.desc()};
        out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);
    } else {
        const TensorDesc tc_desc_arr[] = {in_cl_desc, w_desc_ref};
        out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);
    }

    std::vector<float> cl_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_cl = test::make_packed(out_descs[0], cl_out_buf.data());

    {
        ComputeContext ctx;
        if (has_bias) {
            const TensorView ins[] = {in_cl, w_channel_first, b_channel_first};
            TensorView outs[] = {out_cl};
            tconv_op->compute(outs, ins, ctx);
        } else {
            const TensorView ins[] = {in_cl, w_channel_first};
            TensorView outs[] = {out_cl};
            tconv_op->compute(outs, ins, ctx);
        }
    }

    auto to_channel_first = LayoutConvert::create(TensorLayout::NCHW, Backend::CPU);
    TensorDesc out_cl_desc = out_cl.desc();
    const TensorDesc channel_first_out_desc_arr[] = {out_cl_desc};
    auto channel_first_out_descs = to_channel_first->getOutputTensorDesc(channel_first_out_desc_arr);

    std::vector<float> result_buf(
        static_cast<size_t>(channel_first_out_descs[0].numel()));
    auto result = test::make_channel_first(channel_first_out_descs[0], result_buf.data());
    {
        const TensorView lc_ins[] = {out_cl};
        TensorView lc_outs[] = {result};
        to_channel_first->compute(lc_outs, lc_ins);
    }

    NNOPS_EXPECT_TRUE(test::allclose(result, out_ref, 1e-4f, 1e-4f));
}

// ============================================================
// Hand-verified small tests (check exact expected values)
// ============================================================

NNOPS_TEST(tconv2d_basic_1x1_kernel) {
    // Input 2x2 all 3.0, IC=2; weight [2, 1, 1, 1] all 2.0, so every output
    // is 2 channels * 3.0 * 2.0 = 12.0.
    const int64_t ishape[] = {1, 2, 2, 2};
    const int64_t wshape[] = {2, 1, 1, 1};

    std::vector<float> in_buf(8, 3.0f);  // 1*2*2*2 = 8
    std::vector<float> w_buf(2, 2.0f);   // 2*1*1*1 = 2

    TensorView in_channel_first(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_channel_first(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);

    auto lc_in = LayoutConvert::create(TensorLayout::NHWC, Backend::CPU);
    TensorDesc in_desc = in_channel_first.desc();
    const TensorDesc lc_in_desc_arr[] = {in_desc};
    auto lc_in_descs = lc_in->getOutputTensorDesc(lc_in_desc_arr);

    std::vector<float> cl_in_buf(
        static_cast<size_t>(lc_in_descs[0].storage_bytes()));
    auto in_cl = test::make_packed(lc_in_descs[0], cl_in_buf.data());
    {
        const TensorView lc_ins[] = {in_channel_first};
        TensorView lc_outs[] = {in_cl};
        lc_in->compute(lc_outs, lc_ins);
    }

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {1, 1};
    attrs.stride      = {1, 1};
    attrs.dilation    = {1, 1};
    attrs.padding     = {0, 0};

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    TensorDesc in_cl_desc = in_cl.desc();
    TensorDesc w_desc = w_channel_first.desc();
    const TensorDesc tc_desc_arr[] = {in_cl_desc, w_desc};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    // [1, 1, 2, 2]
    NNOPS_EXPECT_EQ(out_descs[0].dims[0], 1);
    NNOPS_EXPECT_EQ(out_descs[0].dims[1], 1);
    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 2);
    NNOPS_EXPECT_EQ(out_descs[0].dims[3], 2);

    std::vector<float> cl_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_cl = test::make_packed(out_descs[0], cl_out_buf.data());

    {
        const TensorView ins[] = {in_cl, w_channel_first};
        TensorView outs[] = {out_cl};
        tconv_op->compute(outs, ins);
    }

    auto to_channel_first = LayoutConvert::create(TensorLayout::NCHW, Backend::CPU);
    TensorDesc out_cl_desc = out_cl.desc();
    const TensorDesc channel_first_out_desc_arr[] = {out_cl_desc};
    auto channel_first_out_descs = to_channel_first->getOutputTensorDesc(channel_first_out_desc_arr);

    std::vector<float> result_buf(
        static_cast<size_t>(channel_first_out_descs[0].numel()));
    auto result = test::make_channel_first(channel_first_out_descs[0], result_buf.data());
    {
        const TensorView lc_ins[] = {out_cl};
        TensorView lc_outs[] = {result};
        to_channel_first->compute(lc_outs, lc_ins);
    }

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(result_buf[i], 12.0f, 1e-4f);
    }
}

NNOPS_TEST(tconv2d_basic_2x2_kernel) {
    // 2x2 kernel with only kh=0,kw=0 non-zero: an identity copy, so the 2x2
    // input lands in the top-left corner of a 3x3 output.
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t wshape[] = {1, 1, 2, 2};

    std::vector<float> in_buf = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> w_buf  = {1.0f, 0.0f, 0.0f, 0.0f};

    TensorView in_channel_first(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_channel_first(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);

    auto lc_in = LayoutConvert::create(TensorLayout::NHWC, Backend::CPU);
    TensorDesc in_desc = in_channel_first.desc();
    const TensorDesc lc_in_desc_arr[] = {in_desc};
    auto lc_in_descs = lc_in->getOutputTensorDesc(lc_in_desc_arr);

    std::vector<float> cl_in_buf(
        static_cast<size_t>(lc_in_descs[0].storage_bytes()));
    auto in_cl = test::make_packed(lc_in_descs[0], cl_in_buf.data());
    {
        const TensorView lc_ins[] = {in_channel_first};
        TensorView lc_outs[] = {in_cl};
        lc_in->compute(lc_outs, lc_ins);
    }

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {2, 2};
    attrs.stride      = {1, 1};
    attrs.dilation    = {1, 1};
    attrs.padding     = {0, 0};

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    TensorDesc in_cl_desc = in_cl.desc();
    TensorDesc w_desc = w_channel_first.desc();
    const TensorDesc tc_desc_arr[] = {in_cl_desc, w_desc};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    // [1, 1, 3, 3]
    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 3);
    NNOPS_EXPECT_EQ(out_descs[0].dims[3], 3);

    std::vector<float> cl_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_cl = test::make_packed(out_descs[0], cl_out_buf.data());
    {
        const TensorView ins[] = {in_cl, w_channel_first};
        TensorView outs[] = {out_cl};
        tconv_op->compute(outs, ins);
    }

    auto to_channel_first = LayoutConvert::create(TensorLayout::NCHW, Backend::CPU);
    TensorDesc out_cl_desc = out_cl.desc();
    const TensorDesc channel_first_out_desc_arr[] = {out_cl_desc};
    auto channel_first_out_descs = to_channel_first->getOutputTensorDesc(channel_first_out_desc_arr);

    std::vector<float> result_buf(
        static_cast<size_t>(channel_first_out_descs[0].numel()));
    auto result = test::make_channel_first(channel_first_out_descs[0], result_buf.data());
    {
        const TensorView lc_ins[] = {out_cl};
        TensorView lc_outs[] = {result};
        to_channel_first->compute(lc_outs, lc_ins);
    }

    // [[1, 2, 0], [3, 4, 0], [0, 0, 0]]
    NNOPS_EXPECT_NEAR(result_buf[0], 1.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(result_buf[1], 2.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(result_buf[2], 0.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(result_buf[3], 3.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(result_buf[4], 4.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(result_buf[5], 0.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(result_buf[6], 0.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(result_buf[7], 0.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(result_buf[8], 0.0f, 1e-4f);
}

NNOPS_TEST(tconv2d_stride_2) {
    // All-ones 2x2 weight, stride 2: OH = OW = (2-1)*2 + (2-1) + 1 = 4.
    // Each input pixel spreads into a 2x2 patch, so the four inputs tile it.
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t wshape[] = {1, 1, 2, 2};

    std::vector<float> in_buf = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> w_buf  = {1.0f, 1.0f, 1.0f, 1.0f};

    TensorView in_channel_first(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_channel_first(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);

    auto lc_in = LayoutConvert::create(TensorLayout::NHWC, Backend::CPU);
    const TensorDesc lc_in_desc_arr[] = {in_channel_first.desc()};
    auto lc_in_descs = lc_in->getOutputTensorDesc(lc_in_desc_arr);

    std::vector<float> cl_in_buf(
        static_cast<size_t>(lc_in_descs[0].storage_bytes()));
    auto in_cl = test::make_packed(lc_in_descs[0], cl_in_buf.data());
    {
        const TensorView lc_ins[] = {in_channel_first};
        TensorView lc_outs[] = {in_cl};
        lc_in->compute(lc_outs, lc_ins);
    }

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {2, 2};
    attrs.stride      = {2, 2};
    attrs.dilation    = {1, 1};
    attrs.padding     = {0, 0};

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    const TensorDesc tc_desc_arr[] = {in_cl.desc(), w_channel_first.desc()};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 4);
    NNOPS_EXPECT_EQ(out_descs[0].dims[3], 4);

    std::vector<float> cl_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_cl = test::make_packed(out_descs[0], cl_out_buf.data());
    {
        const TensorView ins[] = {in_cl, w_channel_first};
        TensorView outs[] = {out_cl};
        tconv_op->compute(outs, ins);
    }

    auto to_channel_first = LayoutConvert::create(TensorLayout::NCHW, Backend::CPU);
    const TensorDesc channel_first_out_desc_arr[] = {out_cl.desc()};
    auto channel_first_out_descs = to_channel_first->getOutputTensorDesc(channel_first_out_desc_arr);

    std::vector<float> result_buf(
        static_cast<size_t>(channel_first_out_descs[0].numel()));
    auto result = test::make_channel_first(channel_first_out_descs[0], result_buf.data());
    {
        const TensorView lc_ins[] = {out_cl};
        TensorView lc_outs[] = {result};
        to_channel_first->compute(lc_outs, lc_ins);
    }

    auto tconv_op_ref = TransposeConv2D::create(attrs, Backend::CPU);
    const TensorDesc ref_desc_arr[] = {in_channel_first.desc(), w_channel_first.desc()};
    auto ref_descs = tconv_op_ref->getOutputTensorDesc(ref_desc_arr);

    std::vector<float> ref_buf(static_cast<size_t>(ref_descs[0].numel()));
    auto out_ref = test::make_channel_first(ref_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        const TensorView ref_arr[] = {in_channel_first, w_channel_first};
        backend::cpu::reference::transpose_conv2d_channel_first_ref(
            attrs, out_ref, ref_arr, ctx, nullptr);
    }

    NNOPS_EXPECT_TRUE(test::allclose(result, out_ref, 1e-4f, 1e-4f));
}

NNOPS_TEST(tconv2d_with_padding) {
    // Padding=1, stride=1, 2x2 kernel on a 2x2 input:
    // OH = OW = (2-1)*1 - 2 + (2-1) + 1 = 1.
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t wshape[] = {1, 1, 2, 2};

    std::vector<float> in_buf = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> w_buf  = {2.0f, 0.0f, 0.0f, 0.0f};  // only kh=0,kw=0

    TensorView in_channel_first(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_channel_first(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);

    auto lc_in = LayoutConvert::create(TensorLayout::NHWC, Backend::CPU);
    const TensorDesc lc_in_desc_arr[] = {in_channel_first.desc()};
    auto lc_in_descs = lc_in->getOutputTensorDesc(lc_in_desc_arr);

    std::vector<float> cl_in_buf(
        static_cast<size_t>(lc_in_descs[0].storage_bytes()));
    auto in_cl = test::make_packed(lc_in_descs[0], cl_in_buf.data());
    {
        const TensorView lc_ins[] = {in_channel_first};
        TensorView lc_outs[] = {in_cl};
        lc_in->compute(lc_outs, lc_ins);
    }

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {2, 2};
    attrs.stride      = {1, 1};
    attrs.dilation    = {1, 1};
    attrs.padding     = {1, 1};

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    const TensorDesc tc_desc_arr[] = {in_cl.desc(), w_channel_first.desc()};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 1);
    NNOPS_EXPECT_EQ(out_descs[0].dims[3], 1);

    std::vector<float> cl_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_cl = test::make_packed(out_descs[0], cl_out_buf.data());
    {
        const TensorView ins[] = {in_cl, w_channel_first};
        TensorView outs[] = {out_cl};
        tconv_op->compute(outs, ins);
    }

    auto to_channel_first = LayoutConvert::create(TensorLayout::NCHW, Backend::CPU);
    const TensorDesc channel_first_out_desc_arr[] = {out_cl.desc()};
    auto channel_first_out_descs = to_channel_first->getOutputTensorDesc(channel_first_out_desc_arr);

    std::vector<float> result_buf(
        static_cast<size_t>(channel_first_out_descs[0].numel()));
    auto result = test::make_channel_first(channel_first_out_descs[0], result_buf.data());
    {
        const TensorView lc_ins[] = {out_cl};
        TensorView lc_outs[] = {result};
        to_channel_first->compute(lc_outs, lc_ins);
    }

    // input[0][0] maps to (oh, ow) = (-1, -1) and is clipped; only input[1][1]
    // survives, at (0, 0): 4 * 2 = 8.
    NNOPS_EXPECT_NEAR(result_buf[0], 8.0f, 1e-4f);

    auto tconv_op_ref = TransposeConv2D::create(attrs, Backend::CPU);
    const TensorDesc ref_desc_arr[] = {in_channel_first.desc(), w_channel_first.desc()};
    auto ref_descs = tconv_op_ref->getOutputTensorDesc(ref_desc_arr);
    std::vector<float> ref_buf(static_cast<size_t>(ref_descs[0].numel()));
    auto out_ref = test::make_channel_first(ref_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        const TensorView ref_arr[] = {in_channel_first, w_channel_first};
        backend::cpu::reference::transpose_conv2d_channel_first_ref(
            attrs, out_ref, ref_arr, ctx, nullptr);
    }
    NNOPS_EXPECT_TRUE(test::allclose(result, out_ref, 1e-4f, 1e-4f));
}

NNOPS_TEST(tconv2d_with_bias) {
    // 1x1 kernel plus bias: each output = 1.0 * 2.0 + 5.0 = 7.0.
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t wshape[] = {1, 1, 1, 1};
    const int64_t bshape[] = {1};

    std::vector<float> in_buf(4, 1.0f);
    std::vector<float> w_buf  = {2.0f};
    std::vector<float> b_buf  = {5.0f};

    TensorView in_channel_first(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_channel_first(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);
    TensorView b_channel_first(bshape, DataType::f32, b_buf.data(), TensorLayout::NCHW);

    auto lc_in = LayoutConvert::create(TensorLayout::NHWC, Backend::CPU);
    const TensorDesc lc_in_desc_arr[] = {in_channel_first.desc()};
    auto lc_in_descs = lc_in->getOutputTensorDesc(lc_in_desc_arr);

    std::vector<float> cl_in_buf(
        static_cast<size_t>(lc_in_descs[0].storage_bytes()));
    auto in_cl = test::make_packed(lc_in_descs[0], cl_in_buf.data());
    {
        const TensorView lc_ins[] = {in_channel_first};
        TensorView lc_outs[] = {in_cl};
        lc_in->compute(lc_outs, lc_ins);
    }

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {1, 1};
    attrs.stride      = {1, 1};
    attrs.dilation    = {1, 1};
    attrs.padding     = {0, 0};

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    const TensorDesc tc_desc_arr[] = {in_cl.desc(), w_channel_first.desc(), b_channel_first.desc()};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 2);
    NNOPS_EXPECT_EQ(out_descs[0].dims[3], 2);

    std::vector<float> cl_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_cl = test::make_packed(out_descs[0], cl_out_buf.data());
    {
        const TensorView ins[] = {in_cl, w_channel_first, b_channel_first};
        TensorView outs[] = {out_cl};
        tconv_op->compute(outs, ins);
    }

    auto to_channel_first = LayoutConvert::create(TensorLayout::NCHW, Backend::CPU);
    const TensorDesc channel_first_out_desc_arr[] = {out_cl.desc()};
    auto channel_first_out_descs = to_channel_first->getOutputTensorDesc(channel_first_out_desc_arr);

    std::vector<float> result_buf(
        static_cast<size_t>(channel_first_out_descs[0].numel()));
    auto result = test::make_channel_first(channel_first_out_descs[0], result_buf.data());
    {
        const TensorView lc_ins[] = {out_cl};
        TensorView lc_outs[] = {result};
        to_channel_first->compute(lc_outs, lc_ins);
    }

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(result_buf[i], 7.0f, 1e-4f);
    }
}

NNOPS_TEST(tconv2d_epilogue_relu) {
    // ReLU epilogue: -3.0 * 1.0 + (-1.0) = -4.0 clamps to 0.
    const int64_t ishape[] = {1, 1, 1, 1};
    const int64_t wshape[] = {1, 1, 1, 1};

    std::vector<float> in_buf = {-3.0f};
    std::vector<float> w_buf  = {1.0f};
    std::vector<float> b_buf  = {-1.0f};

    TensorView in_channel_first(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_channel_first(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);
    const int64_t bshape[] = {1};
    TensorView b_channel_first(bshape, DataType::f32, b_buf.data(), TensorLayout::NCHW);

    auto lc_in = LayoutConvert::create(TensorLayout::NHWC, Backend::CPU);
    const TensorDesc lc_in_desc_arr[] = {in_channel_first.desc()};
    auto lc_in_descs = lc_in->getOutputTensorDesc(lc_in_desc_arr);

    std::vector<float> cl_in_buf(
        static_cast<size_t>(lc_in_descs[0].storage_bytes()));
    auto in_cl = test::make_packed(lc_in_descs[0], cl_in_buf.data());
    {
        const TensorView lc_ins[] = {in_channel_first};
        TensorView lc_outs[] = {in_cl};
        lc_in->compute(lc_outs, lc_ins);
    }

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {1, 1};
    attrs.stride      = {1, 1};
    attrs.dilation    = {1, 1};
    attrs.padding     = {0, 0};
    attrs.epilogue.type = EpilogueActivateType::Relu;

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    const TensorDesc tc_desc_arr[] = {in_cl.desc(), w_channel_first.desc(), b_channel_first.desc()};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    std::vector<float> cl_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_cl = test::make_packed(out_descs[0], cl_out_buf.data());
    {
        const TensorView ins[] = {in_cl, w_channel_first, b_channel_first};
        TensorView outs[] = {out_cl};
        tconv_op->compute(outs, ins);
    }

    auto to_channel_first = LayoutConvert::create(TensorLayout::NCHW, Backend::CPU);
    const TensorDesc channel_first_out_desc_arr[] = {out_cl.desc()};
    auto channel_first_out_descs = to_channel_first->getOutputTensorDesc(channel_first_out_desc_arr);

    std::vector<float> result_buf(
        static_cast<size_t>(channel_first_out_descs[0].numel()));
    auto result = test::make_channel_first(channel_first_out_descs[0], result_buf.data());
    {
        const TensorView lc_ins[] = {out_cl};
        TensorView lc_outs[] = {result};
        to_channel_first->compute(lc_outs, lc_ins);
    }

    NNOPS_EXPECT_NEAR(result_buf[0], 0.0f, 1e-4f);
}

NNOPS_TEST(tconv2d_groups) {
    // groups=2 with IC=OC=2: channel c sees weight[c] alone, so
    // 10.0 * 2.0 = 20.0 and 20.0 * 3.0 = 60.0.
    const int64_t ishape[] = {1, 2, 1, 1};
    const int64_t wshape[] = {2, 1, 1, 1};

    std::vector<float> in_buf = {10.0f, 20.0f};
    std::vector<float> w_buf  = {2.0f, 3.0f};

    TensorView in_channel_first(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_channel_first(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);

    auto lc_in = LayoutConvert::create(TensorLayout::NHWC, Backend::CPU);
    const TensorDesc lc_in_desc_arr[] = {in_channel_first.desc()};
    auto lc_in_descs = lc_in->getOutputTensorDesc(lc_in_desc_arr);

    std::vector<float> cl_in_buf(
        static_cast<size_t>(lc_in_descs[0].storage_bytes()));
    auto in_cl = test::make_packed(lc_in_descs[0], cl_in_buf.data());
    {
        const TensorView lc_ins[] = {in_channel_first};
        TensorView lc_outs[] = {in_cl};
        lc_in->compute(lc_outs, lc_ins);
    }

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {1, 1};
    attrs.stride      = {1, 1};
    attrs.dilation    = {1, 1};
    attrs.padding     = {0, 0};
    attrs.groups      = 2;

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    const TensorDesc tc_desc_arr[] = {in_cl.desc(), w_channel_first.desc()};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    NNOPS_EXPECT_EQ(out_descs[0].dims[1], 2);  // OC = 2

    std::vector<float> cl_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_cl = test::make_packed(out_descs[0], cl_out_buf.data());
    {
        const TensorView ins[] = {in_cl, w_channel_first};
        TensorView outs[] = {out_cl};
        tconv_op->compute(outs, ins);
    }

    auto to_channel_first = LayoutConvert::create(TensorLayout::NCHW, Backend::CPU);
    const TensorDesc channel_first_out_desc_arr[] = {out_cl.desc()};
    auto channel_first_out_descs = to_channel_first->getOutputTensorDesc(channel_first_out_desc_arr);

    std::vector<float> result_buf(
        static_cast<size_t>(channel_first_out_descs[0].numel()));
    auto result = test::make_channel_first(channel_first_out_descs[0], result_buf.data());
    {
        const TensorView lc_ins[] = {out_cl};
        TensorView lc_outs[] = {result};
        to_channel_first->compute(lc_outs, lc_ins);
    }

    NNOPS_EXPECT_NEAR(result_buf[0], 20.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(result_buf[1], 60.0f, 1e-4f);
}

NNOPS_TEST(tconv2d_output_padding) {
    // Stride=2, output_padding=1: OH = OW = (2-1)*2 + (1-1) + 1 + 1 = 4,
    // where output_padding=0 would give 3.
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t wshape[] = {1, 1, 1, 1};

    std::vector<float> in_buf(4, 1.0f);
    std::vector<float> w_buf  = {1.0f};

    TensorView in_channel_first(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_channel_first(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);

    auto lc_in = LayoutConvert::create(TensorLayout::NHWC, Backend::CPU);
    const TensorDesc lc_in_desc_arr[] = {in_channel_first.desc()};
    auto lc_in_descs = lc_in->getOutputTensorDesc(lc_in_desc_arr);

    std::vector<float> cl_in_buf(
        static_cast<size_t>(lc_in_descs[0].storage_bytes()));
    auto in_cl = test::make_packed(lc_in_descs[0], cl_in_buf.data());
    {
        const TensorView lc_ins[] = {in_channel_first};
        TensorView lc_outs[] = {in_cl};
        lc_in->compute(lc_outs, lc_ins);
    }

    TransposeConv2DAttributes attrs;
    attrs.kernel_size   = {1, 1};
    attrs.stride        = {2, 2};
    attrs.dilation      = {1, 1};
    attrs.padding       = {0, 0};
    attrs.output_padding = {1, 1};

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    const TensorDesc tc_desc_arr[] = {in_cl.desc(), w_channel_first.desc()};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 4);
    NNOPS_EXPECT_EQ(out_descs[0].dims[3], 4);

    auto tconv_op_ref = TransposeConv2D::create(attrs, Backend::CPU);
    const TensorDesc ref_desc_arr[] = {in_channel_first.desc(), w_channel_first.desc()};
    auto ref_descs = tconv_op_ref->getOutputTensorDesc(ref_desc_arr);
    std::vector<float> ref_buf(static_cast<size_t>(ref_descs[0].numel()));
    auto out_ref = test::make_channel_first(ref_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        const TensorView ref_arr[] = {in_channel_first, w_channel_first};
        backend::cpu::reference::transpose_conv2d_channel_first_ref(
            attrs, out_ref, ref_arr, ctx, nullptr);
    }

    std::vector<float> cl_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_cl = test::make_packed(out_descs[0], cl_out_buf.data());
    {
        const TensorView ins[] = {in_cl, w_channel_first};
        TensorView outs[] = {out_cl};
        tconv_op->compute(outs, ins);
    }

    auto to_channel_first = LayoutConvert::create(TensorLayout::NCHW, Backend::CPU);
    const TensorDesc channel_first_out_desc_arr[] = {out_cl.desc()};
    auto channel_first_out_descs = to_channel_first->getOutputTensorDesc(channel_first_out_desc_arr);

    std::vector<float> result_buf(
        static_cast<size_t>(channel_first_out_descs[0].numel()));
    auto result = test::make_channel_first(channel_first_out_descs[0], result_buf.data());
    {
        const TensorView lc_ins[] = {out_cl};
        TensorView lc_outs[] = {result};
        to_channel_first->compute(lc_outs, lc_ins);
    }

    NNOPS_EXPECT_TRUE(test::allclose(result, out_ref, 1e-4f, 1e-4f));
}

NNOPS_TEST(tconv2d_dilation) {
    // Dilation=2 spaces the 2x2 kernel's taps 2 apart, so a 3x3 input gives a
    // 5x5 output: OH = OW = (3-1)*1 + 2*(2-1) + 1 = 5.
    const int64_t ishape[] = {1, 1, 3, 3};
    const int64_t wshape[] = {1, 1, 2, 2};

    std::vector<float> in_buf(9, 1.0f);
    std::vector<float> w_buf  = {5.0f, 4.0f, 3.0f, 2.0f};

    TensorView in_channel_first(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_channel_first(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);

    auto lc_in = LayoutConvert::create(TensorLayout::NHWC, Backend::CPU);
    const TensorDesc lc_in_desc_arr[] = {in_channel_first.desc()};
    auto lc_in_descs = lc_in->getOutputTensorDesc(lc_in_desc_arr);

    std::vector<float> cl_in_buf(
        static_cast<size_t>(lc_in_descs[0].storage_bytes()));
    auto in_cl = test::make_packed(lc_in_descs[0], cl_in_buf.data());
    {
        const TensorView lc_ins[] = {in_channel_first};
        TensorView lc_outs[] = {in_cl};
        lc_in->compute(lc_outs, lc_ins);
    }

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {2, 2};
    attrs.stride      = {1, 1};
    attrs.dilation    = {2, 2};
    attrs.padding     = {0, 0};

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    const TensorDesc tc_desc_arr[] = {in_cl.desc(), w_channel_first.desc()};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 5);
    NNOPS_EXPECT_EQ(out_descs[0].dims[3], 5);

    std::vector<float> cl_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_cl = test::make_packed(out_descs[0], cl_out_buf.data());
    {
        const TensorView ins[] = {in_cl, w_channel_first};
        TensorView outs[] = {out_cl};
        tconv_op->compute(outs, ins);
    }

    auto to_channel_first = LayoutConvert::create(TensorLayout::NCHW, Backend::CPU);
    const TensorDesc channel_first_out_desc_arr[] = {out_cl.desc()};
    auto channel_first_out_descs = to_channel_first->getOutputTensorDesc(channel_first_out_desc_arr);

    std::vector<float> result_buf(
        static_cast<size_t>(channel_first_out_descs[0].numel()));
    auto result = test::make_channel_first(channel_first_out_descs[0], result_buf.data());
    {
        const TensorView lc_ins[] = {out_cl};
        TensorView lc_outs[] = {result};
        to_channel_first->compute(lc_outs, lc_ins);
    }

    auto tconv_op_ref = TransposeConv2D::create(attrs, Backend::CPU);
    const TensorDesc ref_desc_arr[] = {in_channel_first.desc(), w_channel_first.desc()};
    auto ref_descs = tconv_op_ref->getOutputTensorDesc(ref_desc_arr);
    std::vector<float> ref_buf(static_cast<size_t>(ref_descs[0].numel()));
    auto out_ref = test::make_channel_first(ref_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        const TensorView ref_arr[] = {in_channel_first, w_channel_first};
        backend::cpu::reference::transpose_conv2d_channel_first_ref(
            attrs, out_ref, ref_arr, ctx, nullptr);
    }

    NNOPS_EXPECT_TRUE(test::allclose(result, out_ref, 1e-4f, 1e-4f));
}

// ============================================================
// Channel-last roundtrip tests (random data, compare with channel-first ref golden)
// ============================================================

NNOPS_TEST(tconv2d_roundtrip_f32) {
    TransposeConv2DAttributes attrs;
    attrs.kernel_size  = {3, 3};
    attrs.stride       = {2, 2};
    attrs.dilation     = {1, 1};
    attrs.padding      = {1, 1};
    attrs.groups       = 1;

    test_channel_last_vs_ref({1, 4, 8, 8}, {4, 8, 3, 3}, attrs);
}

NNOPS_TEST(tconv2d_roundtrip_f32_bias) {
    TransposeConv2DAttributes attrs;
    attrs.kernel_size  = {3, 3};
    attrs.stride       = {2, 2};
    attrs.dilation     = {1, 1};
    attrs.padding      = {1, 1};
    attrs.groups       = 1;

    test_channel_last_vs_ref({1, 4, 8, 8}, {4, 8, 3, 3}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(tconv2d_roundtrip_f32_groups) {
    TransposeConv2DAttributes attrs;
    attrs.kernel_size  = {3, 3};
    attrs.stride       = {1, 1};
    attrs.dilation     = {1, 1};
    attrs.padding      = {0, 0};
    attrs.groups       = 2;

    test_channel_last_vs_ref({1, 4, 4, 4}, {4, 4, 3, 3}, attrs);
}

NNOPS_TEST(tconv2d_roundtrip_f32_nopad) {
    TransposeConv2DAttributes attrs;
    attrs.kernel_size  = {2, 2};
    attrs.stride       = {1, 1};
    attrs.dilation     = {1, 1};
    attrs.padding      = {0, 0};
    attrs.groups       = 1;

    test_channel_last_vs_ref({1, 4, 4, 4}, {4, 6, 2, 2}, attrs);
}

NNOPS_TEST(tconv2d_roundtrip_f32_dilation) {
    TransposeConv2DAttributes attrs;
    attrs.kernel_size  = {3, 3};
    attrs.stride       = {1, 1};
    attrs.dilation     = {2, 2};
    attrs.padding      = {0, 0};
    attrs.groups       = 1;

    test_channel_last_vs_ref({1, 4, 8, 8}, {4, 4, 3, 3}, attrs);
}

NNOPS_TEST(tconv2d_roundtrip_f32_multi_channel) {
    // > 8 channels to test channel packing across multiple C8 groups
    TransposeConv2DAttributes attrs;
    attrs.kernel_size  = {3, 3};
    attrs.stride       = {2, 2};
    attrs.dilation     = {1, 1};
    attrs.padding      = {1, 1};
    attrs.groups       = 1;

    test_channel_last_vs_ref({1, 16, 8, 8}, {16, 32, 3, 3}, attrs);
}

// ============================================================
// Channel-first col-GEMM route
// ============================================================

/// One channel-first case against the scalar golden. The shapes here reach the parts
/// of the route the hand-verified cases are too small for: an output row wider
/// than one tile, a col large enough to be chunked, and a group count that makes
/// OC/G narrower than a vector.
static void test_channel_first_vs_ref(const std::vector<int64_t>& in_shape,
                               const std::vector<int64_t>& w_shape,
                               const TransposeConv2DAttributes& attrs,
                               bool has_bias = false)
{
    const int64_t OC = w_shape[1] * attrs.groups;

    auto [in_vec, in_channel_first] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_channel_first]  = test::make_random_tensor(w_shape);
    std::vector<float> b_vec;
    TensorView b_channel_first;
    if (has_bias) {
        auto p = test::make_random_tensor({OC});
        b_vec = std::move(p.first);
        b_channel_first = p.second;
    }

    auto op = TransposeConv2D::create(attrs, Backend::CPU);

    std::vector<TensorDesc> out_descs;
    if (has_bias) {
        const TensorDesc d[] = {in_channel_first.desc(), w_channel_first.desc(), b_channel_first.desc()};
        out_descs = op->getOutputTensorDesc(d);
    } else {
        const TensorDesc d[] = {in_channel_first.desc(), w_channel_first.desc()};
        out_descs = op->getOutputTensorDesc(d);
    }

    std::vector<float> out_buf(static_cast<size_t>(out_descs[0].numel()));
    auto out = test::make_channel_first(out_descs[0], out_buf.data());
    {
        TensorView outs[] = {out};
        ComputeContext ctx;
        if (has_bias) {
            const TensorView ins[] = {in_channel_first, w_channel_first, b_channel_first};
            op->compute(outs, ins, ctx);
        } else {
            const TensorView ins[] = {in_channel_first, w_channel_first};
            op->compute(outs, ins, ctx);
        }
    }

    std::vector<float> ref_buf(static_cast<size_t>(out_descs[0].numel()));
    auto out_ref = test::make_channel_first(out_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        if (has_bias) {
            const TensorView ins[] = {in_channel_first, w_channel_first, b_channel_first};
            backend::cpu::reference::transpose_conv2d_channel_first_ref(
                attrs, out_ref, ins, ctx, nullptr);
        } else {
            const TensorView ins[] = {in_channel_first, w_channel_first};
            backend::cpu::reference::transpose_conv2d_channel_first_ref(
                attrs, out_ref, ins, ctx, nullptr);
        }
    }

    NNOPS_EXPECT_TRUE(test::allclose(out, out_ref, 1e-4f, 1e-4f));
}

NNOPS_TEST(tconv2d_channel_first_stride2) {
    // OW = 19, so an output row spans several four-pixel tiles, and OC = 32
    // gives the GEMM enough columns to block.
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride      = {2, 2};
    attrs.dilation    = {1, 1};
    attrs.padding     = {1, 1};
    attrs.groups      = 1;

    test_channel_first_vs_ref({2, 64, 10, 10}, {64, 32, 3, 3}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(tconv2d_channel_first_col_chunked) {
    // 5x5 over 64x64 with 256 output channels is ~100K col elements per oc row,
    // past the 8 MiB budget, so the M rows are split and the scatter has to fold
    // several GEMM calls back together.
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {5, 5};
    attrs.stride      = {1, 1};
    attrs.dilation    = {1, 1};
    attrs.padding     = {2, 2};
    attrs.groups      = 1;

    test_channel_first_vs_ref({1, 16, 64, 64}, {16, 256, 5, 5}, attrs);
}

NNOPS_TEST(tconv2d_channel_first_narrow_oc) {
    // OC/G = 4: the tile's two-vector block never applies and the oc tail runs
    // the scalar path, so this is the shape that would silently drop a group.
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride      = {1, 1};
    attrs.dilation    = {1, 1};
    attrs.padding     = {1, 1};
    attrs.groups      = 4;

    test_channel_first_vs_ref({1, 32, 12, 12}, {32, 4, 3, 3}, attrs, /*has_bias=*/true);
}

/// Channel-last roundtrip against the channel-first golden with a mixed-sign
/// residual in the destination, so that the pre- and post-activation orders of
/// add_to cannot agree by accident.
static void test_channel_last_residual_vs_ref(
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& w_shape,
    const TransposeConv2DAttributes& attrs,
    bool has_bias = false)
{
    const int64_t N  = in_shape[0];
    const int64_t IC = in_shape[1];
    const int64_t IH = in_shape[2];
    const int64_t IW = in_shape[3];
    const int64_t OC = w_shape[1] * attrs.groups;

    auto [in_vec, in_cf] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_cf]  = test::make_random_tensor(w_shape);
    (void)w_vec;

    std::vector<float> b_vec;
    TensorView b_cf;
    if (has_bias) {
        auto p = test::make_random_tensor({OC});
        b_vec = std::move(p.first);
        b_cf  = p.second;
    }

    auto op = TransposeConv2D::create(attrs, Backend::CPU);
    std::vector<TensorDesc> out_descs;
    if (has_bias) {
        const TensorDesc d[] = {in_cf.desc(), w_cf.desc(), b_cf.desc()};
        out_descs = op->getOutputTensorDesc(d);
    } else {
        const TensorDesc d[] = {in_cf.desc(), w_cf.desc()};
        out_descs = op->getOutputTensorDesc(d);
    }
    const int64_t on  = out_descs[0].dims[0];
    const int64_t och = out_descs[0].dims[1];
    const int64_t oh  = out_descs[0].dims[2];
    const int64_t ow  = out_descs[0].dims[3];

    auto [res_vec, res_view] = test::make_random_tensor(
        {on, och, oh, ow}, -1.0f, 1.0f, 11);
    (void)res_view;

    std::vector<float> ref_buf(res_vec);
    auto out_ref = test::make_channel_first(out_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        if (has_bias) {
            const TensorView ins[] = {in_cf, w_cf, b_cf};
            backend::cpu::reference::transpose_conv2d_channel_first_ref(
                attrs, out_ref, ins, ctx, nullptr);
        } else {
            const TensorView ins[] = {in_cf, w_cf};
            backend::cpu::reference::transpose_conv2d_channel_first_ref(
                attrs, out_ref, ins, ctx, nullptr);
        }
    }

    auto in_cl = test::channel_first_to_channel_last<float>(
        std::span<const float>(in_vec.data(), in_vec.size()), on, IC, IH, IW);
    auto out_cl = test::channel_first_to_channel_last<float>(
        std::span<const float>(res_vec.data(), res_vec.size()), on, och, oh, ow);

    const std::vector<int64_t> ishape = {on, IC, IH, IW};
    const std::vector<int64_t> oshape = {on, och, oh, ow};
    TensorView in_view(ishape, DataType::f32, in_cl.data(), TensorLayout::NHWC);
    TensorView out_view(oshape, DataType::f32, out_cl.data(), TensorLayout::NHWC);
    {
        TensorView outs[] = {out_view};
        ComputeContext ctx;
        if (has_bias) {
            const TensorView ins[] = {in_view, w_cf, b_cf};
            op->compute(outs, ins, ctx);
        } else {
            const TensorView ins[] = {in_view, w_cf};
            op->compute(outs, ins, ctx);
        }
    }

    auto got = test::channel_last_to_channel_first<float>(
        std::span<const float>(out_cl.data(), out_cl.size()), on, och, oh, ow);
    TensorView got_view(oshape, DataType::f32, got.data(), TensorLayout::NCHW);
    NNOPS_EXPECT_TRUE(test::allclose(got_view, out_ref, 1e-4f, 1e-4f));
}

NNOPS_TEST(tconv2d_channel_first_add_to) {
    // add_to keeps whatever the destination already held, and the merge pass is
    // the only writer — this is the case that would clobber it.
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride      = {2, 2};
    attrs.dilation    = {1, 1};
    attrs.padding     = {1, 1};
    attrs.groups      = 1;
    attrs.add_to      = true;

    auto [in_vec, in_channel_first]   = test::make_random_tensor({1, 16, 8, 8});
    auto [w_vec,  w_channel_first]    = test::make_random_tensor({16, 32, 3, 3});
    auto [res_vec, res_view] = test::make_random_tensor({1, 32, 15, 15});

    auto op = TransposeConv2D::create(attrs, Backend::CPU);
    const TensorDesc d[] = {in_channel_first.desc(), w_channel_first.desc()};
    const auto out_descs = op->getOutputTensorDesc(d);
    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 15);
    NNOPS_EXPECT_EQ(out_descs[0].dims[3], 15);

    std::vector<float> out_buf(res_vec);
    auto out = test::make_channel_first(out_descs[0], out_buf.data());
    {
        TensorView outs[] = {out};
        const TensorView ins[] = {in_channel_first, w_channel_first};
        ComputeContext ctx;
        op->compute(outs, ins, ctx);
    }

    std::vector<float> ref_buf(res_vec);
    auto out_ref = test::make_channel_first(out_descs[0], ref_buf.data());
    {
        const TensorView ins[] = {in_channel_first, w_channel_first};
        ComputeContext ctx;
        backend::cpu::reference::transpose_conv2d_channel_first_ref(
            attrs, out_ref, ins, ctx, nullptr);
    }

    NNOPS_EXPECT_TRUE(test::allclose(out, out_ref, 1e-4f, 1e-4f));
}

NNOPS_TEST(tconv2d_channel_first_pre_act) {
    // ReLU plus a mixed-sign residual: the post-activation order clamps the
    // convolution before the shortcut arrives, the pre-activation order after.
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride      = {2, 2};
    attrs.dilation    = {1, 1};
    attrs.padding     = {1, 1};
    attrs.groups      = 1;
    attrs.epilogue.type = EpilogueActivateType::Relu;
    attrs.add_to      = true;
    attrs.residual_pre_activation = true;

    auto [in_vec, in_channel_first]   = test::make_random_tensor({1, 16, 8, 8});
    auto [w_vec,  w_channel_first]    = test::make_random_tensor({16, 32, 3, 3});
    auto [res_vec, res_view] = test::make_random_tensor({1, 32, 15, 15}, -1.0f, 1.0f, 11);

    auto op = TransposeConv2D::create(attrs, Backend::CPU);
    const TensorDesc d[] = {in_channel_first.desc(), w_channel_first.desc()};
    const auto out_descs = op->getOutputTensorDesc(d);

    std::vector<float> out_buf(res_vec);
    auto out = test::make_channel_first(out_descs[0], out_buf.data());
    {
        TensorView outs[] = {out};
        const TensorView ins[] = {in_channel_first, w_channel_first};
        ComputeContext ctx;
        op->compute(outs, ins, ctx);
    }

    std::vector<float> ref_buf(res_vec);
    auto out_ref = test::make_channel_first(out_descs[0], ref_buf.data());
    {
        const TensorView ins[] = {in_channel_first, w_channel_first};
        ComputeContext ctx;
        backend::cpu::reference::transpose_conv2d_channel_first_ref(
            attrs, out_ref, ins, ctx, nullptr);
    }

    NNOPS_EXPECT_TRUE(test::allclose(out, out_ref, 1e-4f, 1e-4f));
}

// ============================================================
// f16, both layouts
// ============================================================

/// f16 against an f32 golden computed from the same values *after* their f16
/// rounding, so the comparison prices the kernel's summation order rather than
/// the operand quantisation.
static void test_f16_vs_ref(const std::vector<int64_t>& in_shape,
                            const std::vector<int64_t>& w_shape,
                            const TransposeConv2DAttributes& attrs,
                            bool channel_last)
{
    using half = nnops::backend::cpu::half;

    const int64_t N  = in_shape[0];
    const int64_t IC = in_shape[1];
    const int64_t IH = in_shape[2];
    const int64_t IW = in_shape[3];
    const int64_t OC = w_shape[1] * attrs.groups;

    auto [in_vec, in_f32_view] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_f32_view]  = test::make_random_tensor(w_shape);

    const auto in16 = test::f32_to_f16(in_vec);
    const auto w16  = test::f32_to_f16(w_vec);
    auto in_r = test::f16_to_f32(in16);
    auto w_r  = test::f16_to_f32(w16);

    auto op = TransposeConv2D::create(attrs, Backend::CPU);
    const TensorDesc d[] = {in_f32_view.desc(), w_f32_view.desc()};
    const auto out_descs = op->getOutputTensorDesc(d);
    const int64_t OH = out_descs[0].dims[2];
    const int64_t OW = out_descs[0].dims[3];
    const auto numel = static_cast<size_t>(out_descs[0].numel());

    TensorView in_ref(in_shape, DataType::f32, in_r.data(), TensorLayout::NCHW);
    TensorView w_ref(w_shape, DataType::f32, w_r.data(), TensorLayout::NCHW);
    std::vector<float> ref_buf(numel);
    auto out_ref = test::make_channel_first(out_descs[0], ref_buf.data());
    {
        const TensorView ins[] = {in_ref, w_ref};
        ComputeContext ctx;
        backend::cpu::reference::transpose_conv2d_channel_first_ref(
            attrs, out_ref, ins, ctx, nullptr);
    }

    TensorView w16_view(w_shape, DataType::f16,
                        const_cast<half*>(w16.data()), TensorLayout::NCHW);
    std::vector<half> out16(numel);
    std::vector<float> got;

    if (channel_last) {
        auto in_cl = test::channel_first_to_channel_last<half>(
            std::span<const half>(in16.data(), in16.size()), N, IC, IH, IW);
        TensorView in16_view(in_shape, DataType::f16, in_cl.data(),
                             TensorLayout::NHWC);
        TensorView out16_view(out_descs[0].dims, DataType::f16, out16.data(),
                              TensorLayout::NHWC);
        TensorView outs[] = {out16_view};
        const TensorView ins[] = {in16_view, w16_view};
        ComputeContext ctx;
        op->compute(outs, ins, ctx);

        got = test::f16_to_f32(test::channel_last_to_channel_first<half>(
            std::span<const half>(out16.data(), out16.size()), N, OC, OH, OW));
    } else {
        TensorView in16_view(in_shape, DataType::f16,
                             const_cast<half*>(in16.data()), TensorLayout::NCHW);
        TensorView out16_view(out_descs[0].dims, DataType::f16, out16.data(),
                              TensorLayout::NCHW);
        TensorView outs[] = {out16_view};
        const TensorView ins[] = {in16_view, w16_view};
        ComputeContext ctx;
        op->compute(outs, ins, ctx);

        got = test::f16_to_f32(out16);
    }

    TensorView got_view(out_descs[0].dims, DataType::f32, got.data(),
                        TensorLayout::NCHW);
    NNOPS_EXPECT_TRUE(test::allclose(got_view, out_ref, test::kF16AccumTol,
                                     test::kF16AccumTol));
}

NNOPS_TEST(tconv2d_f16_channel_first) {
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride      = {2, 2};
    attrs.dilation    = {1, 1};
    attrs.padding     = {1, 1};
    attrs.groups      = 1;

    test_f16_vs_ref({1, 32, 12, 12}, {32, 64, 3, 3}, attrs, /*channel_last=*/false);
}

NNOPS_TEST(tconv2d_f16_channel_last) {
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride      = {2, 2};
    attrs.dilation    = {1, 1};
    attrs.padding     = {1, 1};
    attrs.groups      = 1;

    test_f16_vs_ref({1, 32, 12, 12}, {32, 64, 3, 3}, attrs, /*channel_last=*/true);
}

NNOPS_TEST(tconv2d_channel_last_wide_oc) {
    // Wide enough that the channel-last tile repeats its oc block several times
    // over one row of pixels.
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride      = {2, 2};
    attrs.dilation    = {1, 1};
    attrs.padding     = {1, 1};
    attrs.groups      = 1;

    test_channel_last_vs_ref({1, 64, 16, 16}, {64, 128, 3, 3}, attrs);
}

// ============================================================
// Channel-last col routes (depthwise, in-place filter, chunked col)
// ============================================================

NNOPS_TEST(tconv2d_channel_last_depthwise) {
    // g == IC == OC, so every group holds one channel and the route vectorises
    // across groups. The group offset has to be exactly g on both operands or
    // the channels silently swap.
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride      = {2, 2};
    attrs.dilation    = {1, 1};
    attrs.padding     = {1, 1};
    attrs.groups      = 16;

    test_channel_last_vs_ref({1, 16, 9, 9}, {16, 1, 3, 3}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(tconv2d_channel_last_depthwise_stride1) {
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride      = {1, 1};
    attrs.dilation    = {1, 1};
    attrs.padding     = {1, 1};
    attrs.groups      = 12;
    attrs.epilogue    = {EpilogueActivateType::LeakyRelu};
    attrs.epilogue.alpha = 0.05f;

    test_channel_last_vs_ref({1, 12, 10, 10}, {12, 1, 3, 3}, attrs);
}

NNOPS_TEST(tconv2d_channel_last_direct_weight) {
    // Fewer pixels than the filter has input channels: the col is transposed
    // per pixel instead of the weight, so B is the channel-first filter read in place.
    // OC/G = 12 leaves the vector loop a tail as well.
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {4, 4};
    attrs.stride      = {2, 2};
    attrs.dilation    = {1, 1};
    attrs.padding     = {1, 1};
    attrs.groups      = 2;
    attrs.epilogue    = {EpilogueActivateType::Relu6};

    test_channel_last_vs_ref({1, 32, 3, 3}, {32, 12, 4, 4}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(tconv2d_channel_last_col_chunked) {
    // The 8 MiB col budget from the other side: the col is [pixels][tap·oc]
    // here, so it splits on oc and each chunk runs its own GEMM and scatter.
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {5, 5};
    attrs.stride      = {1, 1};
    attrs.dilation    = {1, 1};
    attrs.padding     = {2, 2};
    attrs.groups      = 1;

    test_channel_last_vs_ref({1, 16, 64, 64}, {16, 256, 5, 5}, attrs);
}

NNOPS_TEST(tconv2d_channel_last_batched_groups) {
    // Every batch item runs its own GEMM against the one packed filter, so the
    // group and batch offsets both have to survive.
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride      = {1, 1};
    attrs.dilation    = {1, 1};
    attrs.padding     = {1, 1};
    attrs.groups      = 2;

    test_channel_last_vs_ref({3, 16, 10, 10}, {16, 8, 3, 3}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(tconv2d_channel_last_add_to) {
    // add_to through the col route: only the merge writes, so the destination's
    // own values are still there to add into.
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {4, 4};
    attrs.stride      = {2, 2};
    attrs.dilation    = {1, 1};
    attrs.padding     = {1, 1};
    attrs.groups      = 1;
    attrs.add_to      = true;
    attrs.epilogue    = {EpilogueActivateType::Relu};

    auto [in_vec, in_channel_first] = test::make_random_tensor({1, 16, 8, 8});
    auto [w_vec,  w_channel_first]  = test::make_random_tensor({16, 32, 4, 4});

    auto op = TransposeConv2D::create(attrs, Backend::CPU);
    const TensorDesc d[] = {in_channel_first.desc(), w_channel_first.desc()};
    const auto out_descs = op->getOutputTensorDesc(d);
    const int64_t ON = out_descs[0].dims[0];
    const int64_t OC = out_descs[0].dims[1];
    const int64_t OH = out_descs[0].dims[2];
    const int64_t OW = out_descs[0].dims[3];

    auto [res_vec, res_view] = test::make_random_tensor(
        std::span<const int64_t>(out_descs[0].dims.begin(), out_descs[0].dims.size()),
        -1.0f, 1.0f, 11);
    (void)res_view;

    std::vector<float> ref_buf(res_vec);
    auto out_ref = test::make_channel_first(out_descs[0], ref_buf.data());
    {
        const TensorView ins[] = {in_channel_first, w_channel_first};
        ComputeContext ctx;
        backend::cpu::reference::transpose_conv2d_channel_first_ref(
            attrs, out_ref, ins, ctx, nullptr);
    }

    auto in_cl = test::channel_first_to_channel_last<float>(
        std::span<const float>(in_vec.data(), in_vec.size()), ON, 16, 8, 8);
    auto out_cl = test::channel_first_to_channel_last<float>(
        std::span<const float>(res_vec.data(), res_vec.size()), ON, OC, OH, OW);

    const std::vector<int64_t> ishape = {ON, 16, 8, 8};
    const std::vector<int64_t> oshape = {ON, OC, OH, OW};
    TensorView in_view(ishape, DataType::f32, in_cl.data(), TensorLayout::NHWC);
    TensorView out_view(oshape, DataType::f32, out_cl.data(), TensorLayout::NHWC);
    {
        TensorView outs[] = {out_view};
        const TensorView ins[] = {in_view, w_channel_first};
        ComputeContext ctx;
        op->compute(outs, ins, ctx);
    }

    auto got = test::channel_last_to_channel_first<float>(
        std::span<const float>(out_cl.data(), out_cl.size()), ON, OC, OH, OW);
    TensorView got_view(oshape, DataType::f32, got.data(), TensorLayout::NCHW);
    NNOPS_EXPECT_TRUE(test::allclose(got_view, out_ref, 1e-4f, 1e-4f));
}

// ============================================================
// residual_pre_activation — channel-last routes
// ============================================================

NNOPS_TEST(tconv2d_channel_last_pre_act) {
    // 4x4 stride 2 is past the pixel kernel's tap budget, so this is the col
    // route: the merge is the only writer and folds the destination in first.
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {4, 4};
    attrs.stride      = {2, 2};
    attrs.dilation    = {1, 1};
    attrs.padding     = {1, 1};
    attrs.groups      = 1;
    attrs.epilogue.type = EpilogueActivateType::Relu;
    attrs.add_to      = true;
    attrs.residual_pre_activation = true;

    test_channel_last_residual_vs_ref({1, 16, 8, 8}, {16, 32, 4, 4}, attrs);
}

NNOPS_TEST(tconv2d_channel_last_pre_act_bias_groups) {
    // Two groups and a bias, with an odd output width so the oc tail and the
    // row edges both carry the residual.
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride      = {1, 1};
    attrs.dilation    = {1, 1};
    attrs.padding     = {1, 1};
    attrs.groups      = 2;
    attrs.epilogue.type = EpilogueActivateType::Relu6;
    attrs.add_to      = true;
    attrs.residual_pre_activation = true;

    test_channel_last_residual_vs_ref({2, 8, 9, 10}, {8, 8, 3, 3}, attrs, true);
}

NNOPS_TEST(tconv2d_channel_last_pixel_pre_act) {
    // 2x2 stride 2 keeps the filter under the pixel kernel's threshold, so this
    // runs the four-pixel tile and the per-pixel edge path with the residual.
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {2, 2};
    attrs.stride      = {2, 2};
    attrs.dilation    = {1, 1};
    attrs.padding     = {0, 0};
    attrs.groups      = 1;
    attrs.epilogue.type = EpilogueActivateType::Relu;
    attrs.add_to      = true;
    attrs.residual_pre_activation = true;

    test_channel_last_residual_vs_ref({1, 16, 10, 11}, {16, 32, 2, 2}, attrs, true);
}

NNOPS_TEST(tconv2d_channel_last_depthwise_pre_act) {
    // g == IC == OC, so this takes the channel-last depthwise merge.
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride      = {2, 2};
    attrs.dilation    = {1, 1};
    attrs.padding     = {1, 1};
    attrs.groups      = 16;
    attrs.epilogue.type = EpilogueActivateType::LeakyRelu;
    attrs.epilogue.alpha = 0.05f;
    attrs.add_to      = true;
    attrs.residual_pre_activation = true;

    test_channel_last_residual_vs_ref({1, 16, 9, 9}, {16, 1, 3, 3}, attrs, true);
}
