/// Unit tests for TransposeConv2D operator — NCHWC8 only.
///
/// Pattern: NCHW input → LayoutConvert → NCHWC8, run TransposeConv2D::compute,
/// LayoutConvert back to NCHW, compare with NCHW scalar golden reference.
///
/// Weight layout: planar NCHW [IC, OC/G, KH, KW].
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

// Forward declare the NCHW reference kernel (ground truth for tests).
namespace nnops::backend::cpu::reference {
    void transpose_conv2d_nchw_ref(const TransposeConv2DAttributes& attrs,
                                    TensorView& output,
                                    std::span<const TensorView> inputs,
                                    const ComputeContext& ctx,
                                    void* workspace);
}

// ============================================================
// Helpers
// ============================================================

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

    // -- Random NCHW input + weight --
    auto [in_vec, in_nchw] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_nchw]  = test::make_random_tensor(w_shape);
    std::vector<float> b_vec;
    TensorView b_nchw;
    if (has_bias) {
        auto p = test::make_random_tensor({OC});
        b_vec = std::move(p.first);
        b_nchw = p.second;
    }

    // -- NCHW scalar reference (golden) --
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

    // -- LayoutConvert: NCHW → NCHWC8 --
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

    // -- Weight stays planar NCHW (no prepacking) --
    (void)w_vec;  // w_nchw references it

    // -- getOutputTensorDesc for NCHWC8 output --
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

    // -- Allocate NCHWC8 output --
    std::vector<float> packed_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_c8 = test::make_packed(out_descs[0], packed_out_buf.data());

    // -- TransposeConv2D::compute --
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

    // -- LayoutConvert back: NCHWC8 → NCHW --
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

    // -- Compare --
    NNOPS_EXPECT_TRUE(test::allclose(result, out_ref, 1e-4f, 1e-4f));
}

// ============================================================
// Hand-verified small tests (check exact expected values)
// ============================================================

NNOPS_TEST(tconv2d_basic_1x1_kernel) {
    // 1x1 kernel: transpose conv is just input * weight (upsampled version).
    // Input: [1, 2, 2, 2] all 3.0, IC=2
    // Weight: [2, 1, 1, 1] all 2.0, OC=1
    // Output: [1, 1, 2, 2] = all 12.0 (2 channels * 3.0 * 2.0)
    const int64_t ishape[] = {1, 2, 2, 2};
    const int64_t wshape[] = {2, 1, 1, 1};

    std::vector<float> in_buf(8, 3.0f);  // 1*2*2*2 = 8
    std::vector<float> w_buf(2, 2.0f);   // 2*1*1*1 = 2

    TensorView in_nchw(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_nchw(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);

    // LayoutConvert: NCHW → NCHWC8
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

    // getOutputTensorDesc
    TensorDesc in_c8_desc = in_c8.desc();
    TensorDesc w_desc = w_nchw.desc();
    const TensorDesc tc_desc_arr[] = {in_c8_desc, w_desc};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    // Output should be [1, 1, 2, 2]
    NNOPS_EXPECT_EQ(out_descs[0].dims[0], 1);
    NNOPS_EXPECT_EQ(out_descs[0].dims[1], 1);
    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 2);
    NNOPS_EXPECT_EQ(out_descs[0].dims[3], 2);

    // Allocate output
    std::vector<float> packed_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_c8 = test::make_packed(out_descs[0], packed_out_buf.data());

    // Compute
    {
        const TensorView ins[] = {in_c8, w_nchw};
        TensorView outs[] = {out_c8};
        tconv_op->compute(outs, ins);
    }

    // LayoutConvert back: NCHWC8 → NCHW
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

    // Each output = IC * input_val * weight_val = 2 * 3.0 * 2.0 = 12.0
    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(result_buf[i], 12.0f, 1e-4f);
    }
}

NNOPS_TEST(tconv2d_basic_2x2_kernel) {
    // 2x2 kernel, only kh=0,kw=0 non-zero (identity-like transfer).
    // Input: [1, 1, 2, 2] = [[1, 2], [3, 4]]
    // Weight: [1, 1, 2, 2] = [[1, 0], [0, 0]]
    // Output: [1, 1, 3, 3]
    // Only output[0:2][0:2] has values matching input.
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t wshape[] = {1, 1, 2, 2};

    std::vector<float> in_buf = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> w_buf  = {1.0f, 0.0f, 0.0f, 0.0f};

    TensorView in_nchw(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_nchw(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);

    // LayoutConvert: NCHW → NCHWC8
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

    // Output should be [1, 1, 3, 3]
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

    // LayoutConvert back: NCHWC8 → NCHW
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

    // Expected: [[1, 2, 0], [3, 4, 0], [0, 0, 0]]
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
    // Stride=2, all-ones weight → scatters input to output with stride gaps.
    // Input: [1, 1, 2, 2] = [[1, 2], [3, 4]]
    // Weight: [1, 1, 2, 2] = [[1, 1], [1, 1]]
    // OH = (2-1)*2 - 0 + 1*(2-1) + 0 + 1 = 2 + 1 + 1 = 4
    // OW = 4
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

    // Verify with NCHW reference
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
    // Padding=1: input is offset within output.
    // Input: [1, 1, 2, 2] = [[1, 2], [3, 4]]
    // Kernel: [1, 1, 2, 2] weights so only kh=0,kw=0 contributes
    // Stride=1, Padding=1
    // OH = (2-1)*1 - 2 + 1*(2-1) + 0 + 1 = 1 - 2 + 1 + 1 = 1
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

    // OH = (2-1)*1 - 2*1 + 1*(2-1) + 0 + 1 = 1 - 2 + 1 + 1 = 1
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

    // With padding=1, input[0][0] maps to oh=0*1+0-1=-1 (clipped)
    // Only input[1][1] maps to oh=1,ow=1 which is in bounds: contribution = 4*2 = 8
    NNOPS_EXPECT_NEAR(result_buf[0], 8.0f, 1e-4f);

    // Verify with NCHW reference
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
    // 1x1 kernel with bias.
    // Input: [1, 1, 2, 2] all 1.0
    // Weight: [1, 1, 1, 1] = [2.0]
    // Bias: [1] = [5.0]
    // OH = 2, OW = 2
    // Each output = 1.0 * 2.0 + 5.0 = 7.0
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
    // Transpose conv with ReLU epilogue: negative results become 0.
    // Input: [1, 1, 1, 1] = [-3.0]
    // Weight: [1, 1, 1, 1] = [1.0]
    // Bias: [1] = [-1.0]
    // Result before ReLU: -3.0 * 1.0 + (-1.0) = -4.0
    // After ReLU: 0.0
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
    // Groups=2: IC=2, OC=2, each group has IC_per_G=1, OC_per_G=1
    // Input channel 0 → output channel 0 (weight[0][0])
    // Input channel 1 → output channel 1 (weight[1][0])
    // Input: [1, 2, 1, 1] = [[10.0], [20.0]]
    // Weight: [2, 1, 1, 1] = [2.0, 3.0]
    //   weight[0][0] = 2.0, weight[1][0] = 3.0
    // Output: [1, 2, 1, 1] = [20.0, 60.0]
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

    // Channel 0: 10.0 * 2.0 = 20.0
    // Channel 1: 20.0 * 3.0 = 60.0
    NNOPS_EXPECT_NEAR(result_buf[0], 20.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(result_buf[1], 60.0f, 1e-4f);
}

NNOPS_TEST(tconv2d_output_padding) {
    // Output padding adds extra rows/columns to output.
    // Input: [1, 1, 2, 2] all 1.0
    // Weight: [1, 1, 1, 1] = [1.0]
    // Stride=2, output_padding=1
    // OH = (2-1)*2 - 0 + 1*(1-1) + 1 + 1 = 2 + 0 + 1 + 1 = 4
    // Without output_padding: OH = (2-1)*2 + 1 = 3
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

    // OH = (2-1)*2 - 2*0 + 1*(1-1) + 1 + 1 = 2 + 0 + 1 + 1 = 4
    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 4);
    NNOPS_EXPECT_EQ(out_descs[0].dims[3], 4);

    // Verify with NCHW reference
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

    // Compute NCHWC8
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
    // Dilated kernel: weights are spaced apart.
    // Input: [1, 1, 3, 3] all 1.0
    // Weight: [1, 1, 2, 2] = [[5, 4], [3, 2]]
    // Dilation=2, Stride=1, Padding=0
    // OH = (3-1)*1 - 0 + 2*(2-1) + 0 + 1 = 2 + 2 + 1 = 5
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

    // Verify with NCHW reference
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
