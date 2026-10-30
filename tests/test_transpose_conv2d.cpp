/// @file test_transpose_conv2d.cpp
/// @brief TransposeConv2D correctness tests — NCHWC8 only.
///
/// TransposeConv2D is packed-only, so every case round-trips: NCHW input ->
/// LayoutConvert NCHWC8 -> compute -> LayoutConvert back to NCHW, then compare.
/// Weight stays planar NCHW [IC, OC/G, KH, KW] (no prepacking), so one weight
/// tensor feeds both the packed op and the scalar golden.
///
/// Two kinds of case: hand-computed small shapes that pin exact output values
/// and output geometry, and random roundtrips against
/// reference::transpose_conv2d_nchw_ref (tests/reference).
///
/// Input:  NCHWC8 [N, ceil(IC/8), IH, IW, 8]
/// Output: NCHWC8 [N, ceil(OC/8), OH, OW, 8]

#include "nnops/ops/transpose_conv2d.hpp"
#include "nnops/ops/layout_convert.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>

using namespace nnops;

// Defined in tests/reference/; the NCHW scalar golden used for cross-checks.
namespace nnops::backend::cpu::reference {
    void transpose_conv2d_nchw_ref(const TransposeConv2DAttributes& attrs,
                                    TensorView& output,
                                    std::span<const TensorView> inputs,
                                    const ComputeContext& ctx,
                                    void* workspace);
}

/// Roundtrip: pack input → TransposeConv2D::compute → unpack → compare with NCHW ref golden.
static void test_nchwc8_vs_ref(
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& w_shape,
    const TransposeConv2DAttributes& attrs,
    bool has_bias = false)
{
    const int64_t OC_per_G = w_shape[1];
    const int64_t G = attrs.groups;
    const int64_t OC = OC_per_G * G;

    auto [in_vec, in_nchw] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_nchw]  = test::make_random_tensor(w_shape);
    std::vector<float> b_vec;
    TensorView b_nchw;
    if (has_bias) {
        auto p = test::make_random_tensor({OC});
        b_vec = std::move(p.first);
        b_nchw = p.second;
    }

    auto tconv_op_ref = TransposeConv2D::create(attrs, Backend::CPU);

    TensorDesc in_desc_ref = in_nchw.desc();
    TensorDesc w_desc_ref  = w_nchw.desc();
    std::vector<TensorDesc> ref_descs;
    if (has_bias) {
        TensorDesc b_desc_ref = b_nchw.desc();
        const TensorDesc ref_desc_arr[] = {in_desc_ref, w_desc_ref, b_desc_ref};
        ref_descs = tconv_op_ref->getOutputTensorDesc(ref_desc_arr);
    } else {
        const TensorDesc ref_desc_arr[] = {in_desc_ref, w_desc_ref};
        ref_descs = tconv_op_ref->getOutputTensorDesc(ref_desc_arr);
    }

    std::vector<float> ref_buf(static_cast<size_t>(ref_descs[0].numel()));
    auto out_ref = test::make_planar(ref_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        if (has_bias) {
            const TensorView ref_arr[] = {in_nchw, w_nchw, b_nchw};
            backend::cpu::reference::transpose_conv2d_nchw_ref(
                attrs, out_ref, ref_arr, ctx, nullptr);
        } else {
            const TensorView ref_arr[] = {in_nchw, w_nchw};
            backend::cpu::reference::transpose_conv2d_nchw_ref(
                attrs, out_ref, ref_arr, ctx, nullptr);
        }
    }

    auto lc_in = LayoutConvert::create(TensorLayout::NCHWC8, Backend::CPU);
    TensorDesc in_planar_desc = in_nchw.desc();
    const TensorDesc lc_in_desc_arr[] = {in_planar_desc};
    auto lc_in_descs = lc_in->getOutputTensorDesc(lc_in_desc_arr);

    std::vector<float> packed_in_buf(
        static_cast<size_t>(lc_in_descs[0].storage_bytes()));
    auto in_c8 = test::make_packed(lc_in_descs[0], packed_in_buf.data());
    {
        const TensorView lc_ins[] = {in_nchw};
        TensorView lc_outs[] = {in_c8};
        lc_in->compute(lc_outs, lc_ins);
    }

    (void)w_vec;  // w_nchw references it

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    TensorDesc in_c8_desc = in_c8.desc();
    std::vector<TensorDesc> out_descs;
    if (has_bias) {
        const TensorDesc tc_desc_arr[] = {in_c8_desc, w_desc_ref, b_nchw.desc()};
        out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);
    } else {
        const TensorDesc tc_desc_arr[] = {in_c8_desc, w_desc_ref};
        out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);
    }

    std::vector<float> packed_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_c8 = test::make_packed(out_descs[0], packed_out_buf.data());

    {
        ComputeContext ctx;
        if (has_bias) {
            const TensorView ins[] = {in_c8, w_nchw, b_nchw};
            TensorView outs[] = {out_c8};
            tconv_op->compute(outs, ins, ctx);
        } else {
            const TensorView ins[] = {in_c8, w_nchw};
            TensorView outs[] = {out_c8};
            tconv_op->compute(outs, ins, ctx);
        }
    }

    auto lc_out = LayoutConvert::create(TensorLayout::NCHW, Backend::CPU);
    TensorDesc out_c8_desc = out_c8.desc();
    const TensorDesc lc_out_desc_arr[] = {out_c8_desc};
    auto lc_out_descs = lc_out->getOutputTensorDesc(lc_out_desc_arr);

    std::vector<float> result_buf(
        static_cast<size_t>(lc_out_descs[0].numel()));
    auto result = test::make_planar(lc_out_descs[0], result_buf.data());
    {
        const TensorView lc_ins[] = {out_c8};
        TensorView lc_outs[] = {result};
        lc_out->compute(lc_outs, lc_ins);
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

    TensorView in_nchw(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_nchw(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);

    auto lc_in = LayoutConvert::create(TensorLayout::NCHWC8, Backend::CPU);
    TensorDesc in_desc = in_nchw.desc();
    const TensorDesc lc_in_desc_arr[] = {in_desc};
    auto lc_in_descs = lc_in->getOutputTensorDesc(lc_in_desc_arr);

    std::vector<float> packed_in_buf(
        static_cast<size_t>(lc_in_descs[0].storage_bytes()));
    auto in_c8 = test::make_packed(lc_in_descs[0], packed_in_buf.data());
    {
        const TensorView lc_ins[] = {in_nchw};
        TensorView lc_outs[] = {in_c8};
        lc_in->compute(lc_outs, lc_ins);
    }

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {1, 1};
    attrs.stride      = {1, 1};
    attrs.dilation    = {1, 1};
    attrs.padding     = {0, 0};

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    TensorDesc in_c8_desc = in_c8.desc();
    TensorDesc w_desc = w_nchw.desc();
    const TensorDesc tc_desc_arr[] = {in_c8_desc, w_desc};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    // [1, 1, 2, 2]
    NNOPS_EXPECT_EQ(out_descs[0].dims[0], 1);
    NNOPS_EXPECT_EQ(out_descs[0].dims[1], 1);
    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 2);
    NNOPS_EXPECT_EQ(out_descs[0].dims[3], 2);

    std::vector<float> packed_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_c8 = test::make_packed(out_descs[0], packed_out_buf.data());

    {
        const TensorView ins[] = {in_c8, w_nchw};
        TensorView outs[] = {out_c8};
        tconv_op->compute(outs, ins);
    }

    auto lc_out = LayoutConvert::create(TensorLayout::NCHW, Backend::CPU);
    TensorDesc out_c8_desc = out_c8.desc();
    const TensorDesc lc_out_desc_arr[] = {out_c8_desc};
    auto lc_out_descs = lc_out->getOutputTensorDesc(lc_out_desc_arr);

    std::vector<float> result_buf(
        static_cast<size_t>(lc_out_descs[0].numel()));
    auto result = test::make_planar(lc_out_descs[0], result_buf.data());
    {
        const TensorView lc_ins[] = {out_c8};
        TensorView lc_outs[] = {result};
        lc_out->compute(lc_outs, lc_ins);
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

    TensorView in_nchw(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_nchw(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);

    auto lc_in = LayoutConvert::create(TensorLayout::NCHWC8, Backend::CPU);
    TensorDesc in_desc = in_nchw.desc();
    const TensorDesc lc_in_desc_arr[] = {in_desc};
    auto lc_in_descs = lc_in->getOutputTensorDesc(lc_in_desc_arr);

    std::vector<float> packed_in_buf(
        static_cast<size_t>(lc_in_descs[0].storage_bytes()));
    auto in_c8 = test::make_packed(lc_in_descs[0], packed_in_buf.data());
    {
        const TensorView lc_ins[] = {in_nchw};
        TensorView lc_outs[] = {in_c8};
        lc_in->compute(lc_outs, lc_ins);
    }

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {2, 2};
    attrs.stride      = {1, 1};
    attrs.dilation    = {1, 1};
    attrs.padding     = {0, 0};

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    TensorDesc in_c8_desc = in_c8.desc();
    TensorDesc w_desc = w_nchw.desc();
    const TensorDesc tc_desc_arr[] = {in_c8_desc, w_desc};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    // [1, 1, 3, 3]
    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 3);
    NNOPS_EXPECT_EQ(out_descs[0].dims[3], 3);

    std::vector<float> packed_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_c8 = test::make_packed(out_descs[0], packed_out_buf.data());
    {
        const TensorView ins[] = {in_c8, w_nchw};
        TensorView outs[] = {out_c8};
        tconv_op->compute(outs, ins);
    }

    auto lc_out = LayoutConvert::create(TensorLayout::NCHW, Backend::CPU);
    TensorDesc out_c8_desc = out_c8.desc();
    const TensorDesc lc_out_desc_arr[] = {out_c8_desc};
    auto lc_out_descs = lc_out->getOutputTensorDesc(lc_out_desc_arr);

    std::vector<float> result_buf(
        static_cast<size_t>(lc_out_descs[0].numel()));
    auto result = test::make_planar(lc_out_descs[0], result_buf.data());
    {
        const TensorView lc_ins[] = {out_c8};
        TensorView lc_outs[] = {result};
        lc_out->compute(lc_outs, lc_ins);
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

    TensorView in_nchw(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_nchw(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);

    auto lc_in = LayoutConvert::create(TensorLayout::NCHWC8, Backend::CPU);
    const TensorDesc lc_in_desc_arr[] = {in_nchw.desc()};
    auto lc_in_descs = lc_in->getOutputTensorDesc(lc_in_desc_arr);

    std::vector<float> packed_in_buf(
        static_cast<size_t>(lc_in_descs[0].storage_bytes()));
    auto in_c8 = test::make_packed(lc_in_descs[0], packed_in_buf.data());
    {
        const TensorView lc_ins[] = {in_nchw};
        TensorView lc_outs[] = {in_c8};
        lc_in->compute(lc_outs, lc_ins);
    }

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {2, 2};
    attrs.stride      = {2, 2};
    attrs.dilation    = {1, 1};
    attrs.padding     = {0, 0};

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    const TensorDesc tc_desc_arr[] = {in_c8.desc(), w_nchw.desc()};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 4);
    NNOPS_EXPECT_EQ(out_descs[0].dims[3], 4);

    std::vector<float> packed_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_c8 = test::make_packed(out_descs[0], packed_out_buf.data());
    {
        const TensorView ins[] = {in_c8, w_nchw};
        TensorView outs[] = {out_c8};
        tconv_op->compute(outs, ins);
    }

    auto lc_out = LayoutConvert::create(TensorLayout::NCHW, Backend::CPU);
    const TensorDesc lc_out_desc_arr[] = {out_c8.desc()};
    auto lc_out_descs = lc_out->getOutputTensorDesc(lc_out_desc_arr);

    std::vector<float> result_buf(
        static_cast<size_t>(lc_out_descs[0].numel()));
    auto result = test::make_planar(lc_out_descs[0], result_buf.data());
    {
        const TensorView lc_ins[] = {out_c8};
        TensorView lc_outs[] = {result};
        lc_out->compute(lc_outs, lc_ins);
    }

    auto tconv_op_ref = TransposeConv2D::create(attrs, Backend::CPU);
    const TensorDesc ref_desc_arr[] = {in_nchw.desc(), w_nchw.desc()};
    auto ref_descs = tconv_op_ref->getOutputTensorDesc(ref_desc_arr);

    std::vector<float> ref_buf(static_cast<size_t>(ref_descs[0].numel()));
    auto out_ref = test::make_planar(ref_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        const TensorView ref_arr[] = {in_nchw, w_nchw};
        backend::cpu::reference::transpose_conv2d_nchw_ref(
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

    TensorView in_nchw(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_nchw(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);

    auto lc_in = LayoutConvert::create(TensorLayout::NCHWC8, Backend::CPU);
    const TensorDesc lc_in_desc_arr[] = {in_nchw.desc()};
    auto lc_in_descs = lc_in->getOutputTensorDesc(lc_in_desc_arr);

    std::vector<float> packed_in_buf(
        static_cast<size_t>(lc_in_descs[0].storage_bytes()));
    auto in_c8 = test::make_packed(lc_in_descs[0], packed_in_buf.data());
    {
        const TensorView lc_ins[] = {in_nchw};
        TensorView lc_outs[] = {in_c8};
        lc_in->compute(lc_outs, lc_ins);
    }

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {2, 2};
    attrs.stride      = {1, 1};
    attrs.dilation    = {1, 1};
    attrs.padding     = {1, 1};

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    const TensorDesc tc_desc_arr[] = {in_c8.desc(), w_nchw.desc()};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 1);
    NNOPS_EXPECT_EQ(out_descs[0].dims[3], 1);

    std::vector<float> packed_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_c8 = test::make_packed(out_descs[0], packed_out_buf.data());
    {
        const TensorView ins[] = {in_c8, w_nchw};
        TensorView outs[] = {out_c8};
        tconv_op->compute(outs, ins);
    }

    auto lc_out = LayoutConvert::create(TensorLayout::NCHW, Backend::CPU);
    const TensorDesc lc_out_desc_arr[] = {out_c8.desc()};
    auto lc_out_descs = lc_out->getOutputTensorDesc(lc_out_desc_arr);

    std::vector<float> result_buf(
        static_cast<size_t>(lc_out_descs[0].numel()));
    auto result = test::make_planar(lc_out_descs[0], result_buf.data());
    {
        const TensorView lc_ins[] = {out_c8};
        TensorView lc_outs[] = {result};
        lc_out->compute(lc_outs, lc_ins);
    }

    // input[0][0] maps to (oh, ow) = (-1, -1) and is clipped; only input[1][1]
    // survives, at (0, 0): 4 * 2 = 8.
    NNOPS_EXPECT_NEAR(result_buf[0], 8.0f, 1e-4f);

    auto tconv_op_ref = TransposeConv2D::create(attrs, Backend::CPU);
    const TensorDesc ref_desc_arr[] = {in_nchw.desc(), w_nchw.desc()};
    auto ref_descs = tconv_op_ref->getOutputTensorDesc(ref_desc_arr);
    std::vector<float> ref_buf(static_cast<size_t>(ref_descs[0].numel()));
    auto out_ref = test::make_planar(ref_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        const TensorView ref_arr[] = {in_nchw, w_nchw};
        backend::cpu::reference::transpose_conv2d_nchw_ref(
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

    TensorView in_nchw(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_nchw(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);
    TensorView b_nchw(bshape, DataType::f32, b_buf.data(), TensorLayout::NCHW);

    auto lc_in = LayoutConvert::create(TensorLayout::NCHWC8, Backend::CPU);
    const TensorDesc lc_in_desc_arr[] = {in_nchw.desc()};
    auto lc_in_descs = lc_in->getOutputTensorDesc(lc_in_desc_arr);

    std::vector<float> packed_in_buf(
        static_cast<size_t>(lc_in_descs[0].storage_bytes()));
    auto in_c8 = test::make_packed(lc_in_descs[0], packed_in_buf.data());
    {
        const TensorView lc_ins[] = {in_nchw};
        TensorView lc_outs[] = {in_c8};
        lc_in->compute(lc_outs, lc_ins);
    }

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {1, 1};
    attrs.stride      = {1, 1};
    attrs.dilation    = {1, 1};
    attrs.padding     = {0, 0};

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    const TensorDesc tc_desc_arr[] = {in_c8.desc(), w_nchw.desc(), b_nchw.desc()};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 2);
    NNOPS_EXPECT_EQ(out_descs[0].dims[3], 2);

    std::vector<float> packed_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_c8 = test::make_packed(out_descs[0], packed_out_buf.data());
    {
        const TensorView ins[] = {in_c8, w_nchw, b_nchw};
        TensorView outs[] = {out_c8};
        tconv_op->compute(outs, ins);
    }

    auto lc_out = LayoutConvert::create(TensorLayout::NCHW, Backend::CPU);
    const TensorDesc lc_out_desc_arr[] = {out_c8.desc()};
    auto lc_out_descs = lc_out->getOutputTensorDesc(lc_out_desc_arr);

    std::vector<float> result_buf(
        static_cast<size_t>(lc_out_descs[0].numel()));
    auto result = test::make_planar(lc_out_descs[0], result_buf.data());
    {
        const TensorView lc_ins[] = {out_c8};
        TensorView lc_outs[] = {result};
        lc_out->compute(lc_outs, lc_ins);
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

    TensorView in_nchw(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_nchw(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);
    const int64_t bshape[] = {1};
    TensorView b_nchw(bshape, DataType::f32, b_buf.data(), TensorLayout::NCHW);

    auto lc_in = LayoutConvert::create(TensorLayout::NCHWC8, Backend::CPU);
    const TensorDesc lc_in_desc_arr[] = {in_nchw.desc()};
    auto lc_in_descs = lc_in->getOutputTensorDesc(lc_in_desc_arr);

    std::vector<float> packed_in_buf(
        static_cast<size_t>(lc_in_descs[0].storage_bytes()));
    auto in_c8 = test::make_packed(lc_in_descs[0], packed_in_buf.data());
    {
        const TensorView lc_ins[] = {in_nchw};
        TensorView lc_outs[] = {in_c8};
        lc_in->compute(lc_outs, lc_ins);
    }

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {1, 1};
    attrs.stride      = {1, 1};
    attrs.dilation    = {1, 1};
    attrs.padding     = {0, 0};
    attrs.epilogue.type = EpilogueActivateType::Relu;

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    const TensorDesc tc_desc_arr[] = {in_c8.desc(), w_nchw.desc(), b_nchw.desc()};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    std::vector<float> packed_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_c8 = test::make_packed(out_descs[0], packed_out_buf.data());
    {
        const TensorView ins[] = {in_c8, w_nchw, b_nchw};
        TensorView outs[] = {out_c8};
        tconv_op->compute(outs, ins);
    }

    auto lc_out = LayoutConvert::create(TensorLayout::NCHW, Backend::CPU);
    const TensorDesc lc_out_desc_arr[] = {out_c8.desc()};
    auto lc_out_descs = lc_out->getOutputTensorDesc(lc_out_desc_arr);

    std::vector<float> result_buf(
        static_cast<size_t>(lc_out_descs[0].numel()));
    auto result = test::make_planar(lc_out_descs[0], result_buf.data());
    {
        const TensorView lc_ins[] = {out_c8};
        TensorView lc_outs[] = {result};
        lc_out->compute(lc_outs, lc_ins);
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

    TensorView in_nchw(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_nchw(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);

    auto lc_in = LayoutConvert::create(TensorLayout::NCHWC8, Backend::CPU);
    const TensorDesc lc_in_desc_arr[] = {in_nchw.desc()};
    auto lc_in_descs = lc_in->getOutputTensorDesc(lc_in_desc_arr);

    std::vector<float> packed_in_buf(
        static_cast<size_t>(lc_in_descs[0].storage_bytes()));
    auto in_c8 = test::make_packed(lc_in_descs[0], packed_in_buf.data());
    {
        const TensorView lc_ins[] = {in_nchw};
        TensorView lc_outs[] = {in_c8};
        lc_in->compute(lc_outs, lc_ins);
    }

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {1, 1};
    attrs.stride      = {1, 1};
    attrs.dilation    = {1, 1};
    attrs.padding     = {0, 0};
    attrs.groups      = 2;

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    const TensorDesc tc_desc_arr[] = {in_c8.desc(), w_nchw.desc()};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    NNOPS_EXPECT_EQ(out_descs[0].dims[1], 2);  // OC = 2

    std::vector<float> packed_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_c8 = test::make_packed(out_descs[0], packed_out_buf.data());
    {
        const TensorView ins[] = {in_c8, w_nchw};
        TensorView outs[] = {out_c8};
        tconv_op->compute(outs, ins);
    }

    auto lc_out = LayoutConvert::create(TensorLayout::NCHW, Backend::CPU);
    const TensorDesc lc_out_desc_arr[] = {out_c8.desc()};
    auto lc_out_descs = lc_out->getOutputTensorDesc(lc_out_desc_arr);

    std::vector<float> result_buf(
        static_cast<size_t>(lc_out_descs[0].numel()));
    auto result = test::make_planar(lc_out_descs[0], result_buf.data());
    {
        const TensorView lc_ins[] = {out_c8};
        TensorView lc_outs[] = {result};
        lc_out->compute(lc_outs, lc_ins);
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

    TensorView in_nchw(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_nchw(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);

    auto lc_in = LayoutConvert::create(TensorLayout::NCHWC8, Backend::CPU);
    const TensorDesc lc_in_desc_arr[] = {in_nchw.desc()};
    auto lc_in_descs = lc_in->getOutputTensorDesc(lc_in_desc_arr);

    std::vector<float> packed_in_buf(
        static_cast<size_t>(lc_in_descs[0].storage_bytes()));
    auto in_c8 = test::make_packed(lc_in_descs[0], packed_in_buf.data());
    {
        const TensorView lc_ins[] = {in_nchw};
        TensorView lc_outs[] = {in_c8};
        lc_in->compute(lc_outs, lc_ins);
    }

    TransposeConv2DAttributes attrs;
    attrs.kernel_size   = {1, 1};
    attrs.stride        = {2, 2};
    attrs.dilation      = {1, 1};
    attrs.padding       = {0, 0};
    attrs.output_padding = {1, 1};

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    const TensorDesc tc_desc_arr[] = {in_c8.desc(), w_nchw.desc()};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 4);
    NNOPS_EXPECT_EQ(out_descs[0].dims[3], 4);

    auto tconv_op_ref = TransposeConv2D::create(attrs, Backend::CPU);
    const TensorDesc ref_desc_arr[] = {in_nchw.desc(), w_nchw.desc()};
    auto ref_descs = tconv_op_ref->getOutputTensorDesc(ref_desc_arr);
    std::vector<float> ref_buf(static_cast<size_t>(ref_descs[0].numel()));
    auto out_ref = test::make_planar(ref_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        const TensorView ref_arr[] = {in_nchw, w_nchw};
        backend::cpu::reference::transpose_conv2d_nchw_ref(
            attrs, out_ref, ref_arr, ctx, nullptr);
    }

    std::vector<float> packed_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_c8 = test::make_packed(out_descs[0], packed_out_buf.data());
    {
        const TensorView ins[] = {in_c8, w_nchw};
        TensorView outs[] = {out_c8};
        tconv_op->compute(outs, ins);
    }

    auto lc_out = LayoutConvert::create(TensorLayout::NCHW, Backend::CPU);
    const TensorDesc lc_out_desc_arr[] = {out_c8.desc()};
    auto lc_out_descs = lc_out->getOutputTensorDesc(lc_out_desc_arr);

    std::vector<float> result_buf(
        static_cast<size_t>(lc_out_descs[0].numel()));
    auto result = test::make_planar(lc_out_descs[0], result_buf.data());
    {
        const TensorView lc_ins[] = {out_c8};
        TensorView lc_outs[] = {result};
        lc_out->compute(lc_outs, lc_ins);
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

    TensorView in_nchw(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_nchw(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);

    auto lc_in = LayoutConvert::create(TensorLayout::NCHWC8, Backend::CPU);
    const TensorDesc lc_in_desc_arr[] = {in_nchw.desc()};
    auto lc_in_descs = lc_in->getOutputTensorDesc(lc_in_desc_arr);

    std::vector<float> packed_in_buf(
        static_cast<size_t>(lc_in_descs[0].storage_bytes()));
    auto in_c8 = test::make_packed(lc_in_descs[0], packed_in_buf.data());
    {
        const TensorView lc_ins[] = {in_nchw};
        TensorView lc_outs[] = {in_c8};
        lc_in->compute(lc_outs, lc_ins);
    }

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {2, 2};
    attrs.stride      = {1, 1};
    attrs.dilation    = {2, 2};
    attrs.padding     = {0, 0};

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    const TensorDesc tc_desc_arr[] = {in_c8.desc(), w_nchw.desc()};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 5);
    NNOPS_EXPECT_EQ(out_descs[0].dims[3], 5);

    std::vector<float> packed_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_c8 = test::make_packed(out_descs[0], packed_out_buf.data());
    {
        const TensorView ins[] = {in_c8, w_nchw};
        TensorView outs[] = {out_c8};
        tconv_op->compute(outs, ins);
    }

    auto lc_out = LayoutConvert::create(TensorLayout::NCHW, Backend::CPU);
    const TensorDesc lc_out_desc_arr[] = {out_c8.desc()};
    auto lc_out_descs = lc_out->getOutputTensorDesc(lc_out_desc_arr);

    std::vector<float> result_buf(
        static_cast<size_t>(lc_out_descs[0].numel()));
    auto result = test::make_planar(lc_out_descs[0], result_buf.data());
    {
        const TensorView lc_ins[] = {out_c8};
        TensorView lc_outs[] = {result};
        lc_out->compute(lc_outs, lc_ins);
    }

    auto tconv_op_ref = TransposeConv2D::create(attrs, Backend::CPU);
    const TensorDesc ref_desc_arr[] = {in_nchw.desc(), w_nchw.desc()};
    auto ref_descs = tconv_op_ref->getOutputTensorDesc(ref_desc_arr);
    std::vector<float> ref_buf(static_cast<size_t>(ref_descs[0].numel()));
    auto out_ref = test::make_planar(ref_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        const TensorView ref_arr[] = {in_nchw, w_nchw};
        backend::cpu::reference::transpose_conv2d_nchw_ref(
            attrs, out_ref, ref_arr, ctx, nullptr);
    }

    NNOPS_EXPECT_TRUE(test::allclose(result, out_ref, 1e-4f, 1e-4f));
}

// ============================================================
// NCHWC8 roundtrip tests (random data, compare with NCHW ref golden)
// ============================================================

NNOPS_TEST(tconv2d_roundtrip_f32) {
    TransposeConv2DAttributes attrs;
    attrs.kernel_size  = {3, 3};
    attrs.stride       = {2, 2};
    attrs.dilation     = {1, 1};
    attrs.padding      = {1, 1};
    attrs.groups       = 1;

    test_nchwc8_vs_ref({1, 4, 8, 8}, {4, 8, 3, 3}, attrs);
}

NNOPS_TEST(tconv2d_roundtrip_f32_bias) {
    TransposeConv2DAttributes attrs;
    attrs.kernel_size  = {3, 3};
    attrs.stride       = {2, 2};
    attrs.dilation     = {1, 1};
    attrs.padding      = {1, 1};
    attrs.groups       = 1;

    test_nchwc8_vs_ref({1, 4, 8, 8}, {4, 8, 3, 3}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(tconv2d_roundtrip_f32_groups) {
    TransposeConv2DAttributes attrs;
    attrs.kernel_size  = {3, 3};
    attrs.stride       = {1, 1};
    attrs.dilation     = {1, 1};
    attrs.padding      = {0, 0};
    attrs.groups       = 2;

    test_nchwc8_vs_ref({1, 4, 4, 4}, {4, 4, 3, 3}, attrs);
}

NNOPS_TEST(tconv2d_roundtrip_f32_nopad) {
    TransposeConv2DAttributes attrs;
    attrs.kernel_size  = {2, 2};
    attrs.stride       = {1, 1};
    attrs.dilation     = {1, 1};
    attrs.padding      = {0, 0};
    attrs.groups       = 1;

    test_nchwc8_vs_ref({1, 4, 4, 4}, {4, 6, 2, 2}, attrs);
}

NNOPS_TEST(tconv2d_roundtrip_f32_dilation) {
    TransposeConv2DAttributes attrs;
    attrs.kernel_size  = {3, 3};
    attrs.stride       = {1, 1};
    attrs.dilation     = {2, 2};
    attrs.padding      = {0, 0};
    attrs.groups       = 1;

    test_nchwc8_vs_ref({1, 4, 8, 8}, {4, 4, 3, 3}, attrs);
}

NNOPS_TEST(tconv2d_roundtrip_f32_multi_channel) {
    // > 8 channels to test channel packing across multiple C8 groups
    TransposeConv2DAttributes attrs;
    attrs.kernel_size  = {3, 3};
    attrs.stride       = {2, 2};
    attrs.dilation     = {1, 1};
    attrs.padding      = {1, 1};
    attrs.groups       = 1;

    test_nchwc8_vs_ref({1, 16, 8, 8}, {16, 32, 3, 3}, attrs);
}
