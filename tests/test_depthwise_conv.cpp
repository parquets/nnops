/// Unit tests for DepthwiseConv operator — NCHWC8/NCDHWC8 only.
///
/// Pattern: NCHW → LayoutConvert → NCHWC8, prepack weight via
/// prepackWeights(query→allocate→pack), getOutputTensorDesc for output shape,
/// DepthwiseConv::compute, LayoutConvert back to NCHW, compare with ref.
///
/// 3D: NCDHW → LayoutConvert → NCDHWC8, same pattern with 3D layouts.

#include "nnops/ops/depthwise_conv.hpp"
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

// Forward declare the reference kernel (NCHW scalar, ground truth).
namespace nnops::backend::cpu::reference {
    void depthwise_conv_ref(const DepthwiseConvAttributes& attrs,
                            TensorView& output,
                            std::span<const TensorView> inputs,
                            const ComputeContext& ctx,
                            void* workspace);
}

// ============================================================
// 2D Helpers
// ============================================================

/// Generic 2D NCHWC8 roundtrip: pack input, prepack weight, run backend,
/// unpack output, compare against NCHW scalar reference.
static void test_nchwc8_vs_ref(
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& w_shape,
    const DepthwiseConvAttributes& attrs,
    bool has_bias = false)
{
    const int64_t C = w_shape[0];

    // -- Random NCHW input + weight --
    auto [in_vec, in_nchw] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_nchw]  = test::make_random_tensor(w_shape);
    std::vector<float> b_vec;
    TensorView b_nchw;
    if (has_bias) {
        auto p = test::make_random_tensor({C});
        b_vec = std::move(p.first);
        b_nchw = p.second;
    }

    // -- NCHW scalar reference (golden) --
    auto dw_op_ref = DepthwiseConv::create(attrs, Backend::CPU);

    TensorDesc in_desc_ref = in_nchw.desc();
    TensorDesc w_desc_ref  = w_nchw.desc();
    const TensorDesc ref_desc_arr[] = {in_desc_ref, w_desc_ref};
    auto ref_descs = dw_op_ref->getOutputTensorDesc(ref_desc_arr);

    std::vector<float> ref_buf(static_cast<size_t>(ref_descs[0].numel()));
    auto out_ref = test::make_planar(ref_descs[0], ref_buf.data());
    if (attrs.add_to) {
        for (auto& v : ref_buf) v = 1.0f;
    }
    {
        ComputeContext ctx;
        if (has_bias) {
            const TensorView ref_arr[] = {in_nchw, w_nchw, b_nchw};
            backend::cpu::reference::depthwise_conv_ref(
                attrs, out_ref, ref_arr, ctx, nullptr);
        } else {
            const TensorView ref_arr[] = {in_nchw, w_nchw};
            backend::cpu::reference::depthwise_conv_ref(
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

    // -- Prepack weight: query → allocate → pack --
    auto dw_op = DepthwiseConv::create(attrs, Backend::CPU);

    // Step 1: query packed shapes
    TensorView packed_w_query;
    TensorView packed_b_query;
    if (has_bias) {
        const TensorView w_arr_q[] = {w_nchw, b_nchw};
        TensorView pw_arr_q[] = {packed_w_query, packed_b_query};
        dw_op->prepackWeights(w_arr_q, pw_arr_q);
        packed_w_query = pw_arr_q[0];
        packed_b_query = pw_arr_q[1];
    } else {
        const TensorView w_arr_q[] = {w_nchw};
        TensorView pw_arr_q[] = {packed_w_query};
        dw_op->prepackWeights(w_arr_q, pw_arr_q);
        packed_w_query = pw_arr_q[0];
    }

    // Step 2: allocate
    std::vector<float> pw_buf(static_cast<size_t>(packed_w_query.numel()));
    auto pw_view = test::make_planar(packed_w_query.desc(), pw_buf.data());

    std::vector<float> pb_buf;
    TensorView pb_view;
    if (has_bias) {
        pb_buf.resize(static_cast<size_t>(packed_b_query.numel()));
        pb_view = test::make_planar(packed_b_query.desc(), pb_buf.data());
    }

    // Step 3: pack
    if (has_bias) {
        const TensorView w_arr_p[] = {w_nchw, b_nchw};
        TensorView pw_arr_p[] = {pw_view, pb_view};
        dw_op->prepackWeights(w_arr_p, pw_arr_p);
    } else {
        const TensorView w_arr_p[] = {w_nchw};
        TensorView pw_arr_p[] = {pw_view};
        dw_op->prepackWeights(w_arr_p, pw_arr_p);
    }

    // -- getOutputTensorDesc for NCHWC8 output --
    TensorDesc in_c8_desc = in_c8.desc();
    TensorDesc pw_desc = pw_view.desc();
    std::vector<TensorDesc> out_descs;
    if (has_bias) {
        TensorDesc pb_desc = pb_view.desc();
        const TensorDesc dw_desc_arr[] = {in_c8_desc, pw_desc, pb_desc};
        out_descs = dw_op->getOutputTensorDesc(dw_desc_arr);
    } else {
        const TensorDesc dw_desc_arr[] = {in_c8_desc, pw_desc};
        out_descs = dw_op->getOutputTensorDesc(dw_desc_arr);
    }

    // -- Allocate NCHWC8 output --
    std::vector<float> packed_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_c8 = test::make_packed(out_descs[0], packed_out_buf.data());
    if (attrs.add_to) {
        std::fill(packed_out_buf.begin(), packed_out_buf.end(), 1.0f);
    }

    // -- DepthwiseConv::compute --
    {
        ComputeContext ctx;
        if (has_bias) {
            const TensorView ins[] = {in_c8, pw_view, pb_view};
            TensorView outs[] = {out_c8};
            dw_op->compute(outs, ins, ctx);
        } else {
            const TensorView ins[] = {in_c8, pw_view};
            TensorView outs[] = {out_c8};
            dw_op->compute(outs, ins, ctx);
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

NNOPS_TEST(dwconv_basic_no_pad) {
    // 1x1x4x4 input all-ones, 1x1x3x3 kernel all-0.5, stride=1, pad=0
    // Expected: 2x2 output, each = 9 * (1.0 * 0.5) = 4.5
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t wshape[] = {1, 1, 3, 3};

    std::vector<float> in_buf(16, 1.0f);
    std::vector<float> w_buf(9, 0.5f);

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

    // Prepack weight: query → allocate → pack
    DepthwiseConvAttributes attrs;
    attrs.kernel_size = {1, 3, 3};
    attrs.stride      = {1, 1, 1};
    attrs.dilation    = {1, 1, 1};
    attrs.padding     = {0, 0, 0};

    auto dw_op = DepthwiseConv::create(attrs, Backend::CPU);

    // Query
    TensorView pw_query;
    {
        const TensorView w_arr_q[] = {w_nchw};
        TensorView pw_arr_q[] = {pw_query};
        dw_op->prepackWeights(w_arr_q, pw_arr_q);
        pw_query = pw_arr_q[0];
    }
    // Allocate
    std::vector<float> pw_buf(static_cast<size_t>(pw_query.numel()));
    auto pw_view = test::make_planar(pw_query.desc(), pw_buf.data());
    // Pack
    {
        const TensorView w_arr_p[] = {w_nchw};
        TensorView pw_arr_p[] = {pw_view};
        dw_op->prepackWeights(w_arr_p, pw_arr_p);
    }

    // getOutputTensorDesc
    TensorDesc in_c8_desc = in_c8.desc();
    TensorDesc pw_desc = pw_view.desc();
    const TensorDesc dw_desc_arr[] = {in_c8_desc, pw_desc};
    auto out_descs = dw_op->getOutputTensorDesc(dw_desc_arr);

    // Allocate output
    std::vector<float> packed_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_c8 = test::make_packed(out_descs[0], packed_out_buf.data());

    // Compute
    {
        const TensorView ins[] = {in_c8, pw_view};
        TensorView outs[] = {out_c8};
        dw_op->compute(outs, ins);
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

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(result_buf[i], 4.5f, 1e-4f);
    }
}

NNOPS_TEST(dwconv_stride_2) {
    // 1x1x6x6 input all-ones, 1x1x3x3 kernel all-0.5, stride=2
    const int64_t ishape[] = {1, 1, 6, 6};
    const int64_t wshape[] = {1, 1, 3, 3};

    std::vector<float> in_buf(36, 1.0f);
    std::vector<float> w_buf(9, 0.5f);

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

    // Prepack weight: query → allocate → pack
    DepthwiseConvAttributes attrs;
    attrs.kernel_size = {1, 3, 3};
    attrs.stride      = {1, 2, 2};
    attrs.dilation    = {1, 1, 1};
    attrs.padding     = {0, 0, 0};

    auto dw_op = DepthwiseConv::create(attrs, Backend::CPU);

    TensorView pw_query;
    {
        const TensorView w_arr_q[] = {w_nchw};
        TensorView pw_arr_q[] = {pw_query};
        dw_op->prepackWeights(w_arr_q, pw_arr_q);
        pw_query = pw_arr_q[0];
    }
    std::vector<float> pw_buf(static_cast<size_t>(pw_query.numel()));
    auto pw_view = test::make_planar(pw_query.desc(), pw_buf.data());
    {
        const TensorView w_arr_p[] = {w_nchw};
        TensorView pw_arr_p[] = {pw_view};
        dw_op->prepackWeights(w_arr_p, pw_arr_p);
    }

    // getOutputTensorDesc
    TensorDesc in_c8_desc = in_c8.desc();
    TensorDesc pw_desc = pw_view.desc();
    const TensorDesc dw_desc_arr[] = {in_c8_desc, pw_desc};
    auto out_descs = dw_op->getOutputTensorDesc(dw_desc_arr);

    std::vector<float> packed_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_c8 = test::make_packed(out_descs[0], packed_out_buf.data());

    {
        const TensorView ins[] = {in_c8, pw_view};
        TensorView outs[] = {out_c8};
        dw_op->compute(outs, ins);
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

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(result_buf[i], 4.5f, 1e-4f);
    }
}

NNOPS_TEST(dwconv_padding_1) {
    // 1x1x2x2 input all-ones, 1x1x3x3 kernel all-0.5, pad=1
    // 4 valid positions per output corner → 4 * 0.5 = 2.0
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t wshape[] = {1, 1, 3, 3};

    std::vector<float> in_buf(4, 1.0f);
    std::vector<float> w_buf(9, 0.5f);

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

    // Prepack weight: query → allocate → pack
    DepthwiseConvAttributes attrs;
    attrs.kernel_size = {1, 3, 3};
    attrs.stride      = {1, 1, 1};
    attrs.dilation    = {1, 1, 1};
    attrs.padding     = {0, 1, 1};

    auto dw_op = DepthwiseConv::create(attrs, Backend::CPU);

    TensorView pw_query;
    {
        const TensorView w_arr_q[] = {w_nchw};
        TensorView pw_arr_q[] = {pw_query};
        dw_op->prepackWeights(w_arr_q, pw_arr_q);
        pw_query = pw_arr_q[0];
    }
    std::vector<float> pw_buf(static_cast<size_t>(pw_query.numel()));
    auto pw_view = test::make_planar(pw_query.desc(), pw_buf.data());
    {
        const TensorView w_arr_p[] = {w_nchw};
        TensorView pw_arr_p[] = {pw_view};
        dw_op->prepackWeights(w_arr_p, pw_arr_p);
    }

    // getOutputTensorDesc
    TensorDesc in_c8_desc = in_c8.desc();
    TensorDesc pw_desc = pw_view.desc();
    const TensorDesc dw_desc_arr[] = {in_c8_desc, pw_desc};
    auto out_descs = dw_op->getOutputTensorDesc(dw_desc_arr);

    std::vector<float> packed_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_c8 = test::make_packed(out_descs[0], packed_out_buf.data());

    {
        const TensorView ins[] = {in_c8, pw_view};
        TensorView outs[] = {out_c8};
        dw_op->compute(outs, ins);
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

    NNOPS_EXPECT_NEAR(result_buf[0], 2.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(result_buf[1], 2.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(result_buf[2], 2.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(result_buf[3], 2.0f, 1e-4f);
}

NNOPS_TEST(dwconv_with_bias) {
    // 1x1x4x4 input all-1.0, 1x1x3x3 kernel all-0.5, bias=0.5 → each = 5.0
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t wshape[] = {1, 1, 3, 3};
    const int64_t bshape[] = {1};

    std::vector<float> in_buf(16, 1.0f);
    std::vector<float> w_buf(9, 0.5f);
    std::vector<float> b_buf = {0.5f};

    TensorView in_nchw(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_nchw(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);
    TensorView b_nchw(bshape, DataType::f32, b_buf.data(), TensorLayout::NCHW);

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

    // Prepack weight+bias: query → allocate → pack
    DepthwiseConvAttributes attrs;
    attrs.kernel_size = {1, 3, 3};
    attrs.stride      = {1, 1, 1};
    attrs.dilation    = {1, 1, 1};
    attrs.padding     = {0, 0, 0};

    auto dw_op = DepthwiseConv::create(attrs, Backend::CPU);

    // Query
    TensorView pw_query, pb_query;
    {
        const TensorView w_arr_q[] = {w_nchw, b_nchw};
        TensorView pw_arr_q[] = {pw_query, pb_query};
        dw_op->prepackWeights(w_arr_q, pw_arr_q);
        pw_query = pw_arr_q[0];
        pb_query = pw_arr_q[1];
    }
    // Allocate
    std::vector<float> pw_buf(static_cast<size_t>(pw_query.numel()));
    auto pw_view = test::make_planar(pw_query.desc(), pw_buf.data());
    std::vector<float> pb_buf(static_cast<size_t>(pb_query.numel()));
    auto pb_view = test::make_planar(pb_query.desc(), pb_buf.data());
    // Pack
    {
        const TensorView w_arr_p[] = {w_nchw, b_nchw};
        TensorView pw_arr_p[] = {pw_view, pb_view};
        dw_op->prepackWeights(w_arr_p, pw_arr_p);
    }

    // getOutputTensorDesc
    TensorDesc in_c8_desc = in_c8.desc();
    TensorDesc pw_desc = pw_view.desc();
    TensorDesc pb_desc = pb_view.desc();
    const TensorDesc dw_desc_arr[] = {in_c8_desc, pw_desc, pb_desc};
    auto out_descs = dw_op->getOutputTensorDesc(dw_desc_arr);

    std::vector<float> packed_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_c8 = test::make_packed(out_descs[0], packed_out_buf.data());

    {
        const TensorView ins[] = {in_c8, pw_view, pb_view};
        TensorView outs[] = {out_c8};
        dw_op->compute(outs, ins);
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

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(result_buf[i], 5.0f, 1e-4f);
    }
}

NNOPS_TEST(dwconv_dilation) {
    // 1x1x5x5 input all-ones, 1x1x3x3 kernel all-0.5, dilation=2
    // Output: 1x1. 9 kernel positions, sum = 4.5
    const int64_t ishape[] = {1, 1, 5, 5};
    const int64_t wshape[] = {1, 1, 3, 3};

    std::vector<float> in_buf(25, 1.0f);
    std::vector<float> w_buf(9, 0.5f);

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

    // Prepack weight: query → allocate → pack
    DepthwiseConvAttributes attrs;
    attrs.kernel_size = {1, 3, 3};
    attrs.stride      = {1, 1, 1};
    attrs.dilation    = {1, 2, 2};
    attrs.padding     = {0, 0, 0};

    auto dw_op = DepthwiseConv::create(attrs, Backend::CPU);

    TensorView pw_query;
    {
        const TensorView w_arr_q[] = {w_nchw};
        TensorView pw_arr_q[] = {pw_query};
        dw_op->prepackWeights(w_arr_q, pw_arr_q);
        pw_query = pw_arr_q[0];
    }
    std::vector<float> pw_buf(static_cast<size_t>(pw_query.numel()));
    auto pw_view = test::make_planar(pw_query.desc(), pw_buf.data());
    {
        const TensorView w_arr_p[] = {w_nchw};
        TensorView pw_arr_p[] = {pw_view};
        dw_op->prepackWeights(w_arr_p, pw_arr_p);
    }

    // getOutputTensorDesc
    TensorDesc in_c8_desc = in_c8.desc();
    TensorDesc pw_desc = pw_view.desc();
    const TensorDesc dw_desc_arr[] = {in_c8_desc, pw_desc};
    auto out_descs = dw_op->getOutputTensorDesc(dw_desc_arr);

    std::vector<float> packed_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_c8 = test::make_packed(out_descs[0], packed_out_buf.data());

    {
        const TensorView ins[] = {in_c8, pw_view};
        TensorView outs[] = {out_c8};
        dw_op->compute(outs, ins);
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

    NNOPS_EXPECT_NEAR(result_buf[0], 4.5f, 1e-4f);
}

NNOPS_TEST(dwconv_add_to) {
    // add_to with pre-filled 1.0 → each output = 4.5 + 1.0 = 5.5
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t wshape[] = {1, 1, 3, 3};

    std::vector<float> in_buf(16, 1.0f);
    std::vector<float> w_buf(9, 0.5f);

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

    // Prepack weight: query → allocate → pack
    DepthwiseConvAttributes attrs;
    attrs.kernel_size = {1, 3, 3};
    attrs.stride      = {1, 1, 1};
    attrs.dilation    = {1, 1, 1};
    attrs.padding     = {0, 0, 0};
    attrs.add_to      = true;

    auto dw_op = DepthwiseConv::create(attrs, Backend::CPU);

    TensorView pw_query;
    {
        const TensorView w_arr_q[] = {w_nchw};
        TensorView pw_arr_q[] = {pw_query};
        dw_op->prepackWeights(w_arr_q, pw_arr_q);
        pw_query = pw_arr_q[0];
    }
    std::vector<float> pw_buf(static_cast<size_t>(pw_query.numel()));
    auto pw_view = test::make_planar(pw_query.desc(), pw_buf.data());
    {
        const TensorView w_arr_p[] = {w_nchw};
        TensorView pw_arr_p[] = {pw_view};
        dw_op->prepackWeights(w_arr_p, pw_arr_p);
    }

    // getOutputTensorDesc
    TensorDesc in_c8_desc = in_c8.desc();
    TensorDesc pw_desc = pw_view.desc();
    const TensorDesc dw_desc_arr[] = {in_c8_desc, pw_desc};
    auto out_descs = dw_op->getOutputTensorDesc(dw_desc_arr);

    // Pre-fill output with 1.0
    std::vector<float> packed_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()), 1.0f);
    auto out_c8 = test::make_packed(out_descs[0], packed_out_buf.data());

    {
        const TensorView ins[] = {in_c8, pw_view};
        TensorView outs[] = {out_c8};
        dw_op->compute(outs, ins);
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

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(result_buf[i], 5.5f, 1e-4f);
    }
}

NNOPS_TEST(dwconv_relu_epilogue) {
    // Channel 0: positive kernel → output positive → ReLU passes through
    // Channel 1: negative kernel → output negative → ReLU clips to 0
    const int64_t ishape[] = {1, 2, 4, 4};
    const int64_t wshape[] = {2, 1, 3, 3};

    std::vector<float> in_buf(32, 1.0f);
    std::vector<float> w_buf(18);
    for (int i = 0; i < 9; ++i) {
        w_buf[i] = 0.5f;        // ch0: positive
        w_buf[i + 9] = -0.5f;   // ch1: negative
    }

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

    // Prepack weight: query → allocate → pack
    DepthwiseConvAttributes attrs;
    attrs.kernel_size = {1, 3, 3};
    attrs.stride      = {1, 1, 1};
    attrs.dilation    = {1, 1, 1};
    attrs.padding     = {0, 0, 0};
    attrs.epilogue.type = EpilogueActivateType::Relu;

    auto dw_op = DepthwiseConv::create(attrs, Backend::CPU);

    TensorView pw_query;
    {
        const TensorView w_arr_q[] = {w_nchw};
        TensorView pw_arr_q[] = {pw_query};
        dw_op->prepackWeights(w_arr_q, pw_arr_q);
        pw_query = pw_arr_q[0];
    }
    std::vector<float> pw_buf(static_cast<size_t>(pw_query.numel()));
    auto pw_view = test::make_planar(pw_query.desc(), pw_buf.data());
    {
        const TensorView w_arr_p[] = {w_nchw};
        TensorView pw_arr_p[] = {pw_view};
        dw_op->prepackWeights(w_arr_p, pw_arr_p);
    }

    // getOutputTensorDesc
    TensorDesc in_c8_desc = in_c8.desc();
    TensorDesc pw_desc = pw_view.desc();
    const TensorDesc dw_desc_arr[] = {in_c8_desc, pw_desc};
    auto out_descs = dw_op->getOutputTensorDesc(dw_desc_arr);

    std::vector<float> packed_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_c8 = test::make_packed(out_descs[0], packed_out_buf.data());

    {
        const TensorView ins[] = {in_c8, pw_view};
        TensorView outs[] = {out_c8};
        dw_op->compute(outs, ins);
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

    // Channel 0: 4.5 → ReLU → 4.5
    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(result_buf[i], 4.5f, 1e-4f);
    }
    // Channel 1: -4.5 → ReLU → 0.0
    for (int i = 4; i < 8; ++i) {
        NNOPS_EXPECT_NEAR(result_buf[i], 0.0f, 1e-4f);
    }
}

NNOPS_TEST(dwconv_multi_channel) {
    // C=2, different kernels per channel
    // Ch0: kernel all 1.0, input all 2.0 → 9 * 2.0 * 1.0 = 18.0
    // Ch1: kernel all 0.5, input all 3.0 → 9 * 3.0 * 0.5 = 13.5
    const int64_t ishape[] = {1, 2, 3, 3};
    const int64_t wshape[] = {2, 1, 3, 3};

    std::vector<float> in_buf(18);
    for (int i = 0; i < 9; ++i) {
        in_buf[i] = 2.0f;
        in_buf[i + 9] = 3.0f;
    }
    std::vector<float> w_buf(18);
    for (int i = 0; i < 9; ++i) {
        w_buf[i] = 1.0f;
        w_buf[i + 9] = 0.5f;
    }

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

    // Prepack weight: query → allocate → pack
    DepthwiseConvAttributes attrs;
    attrs.kernel_size = {1, 3, 3};
    attrs.stride      = {1, 1, 1};
    attrs.dilation    = {1, 1, 1};
    attrs.padding     = {0, 0, 0};

    auto dw_op = DepthwiseConv::create(attrs, Backend::CPU);

    TensorView pw_query;
    {
        const TensorView w_arr_q[] = {w_nchw};
        TensorView pw_arr_q[] = {pw_query};
        dw_op->prepackWeights(w_arr_q, pw_arr_q);
        pw_query = pw_arr_q[0];
    }
    std::vector<float> pw_buf(static_cast<size_t>(pw_query.numel()));
    auto pw_view = test::make_planar(pw_query.desc(), pw_buf.data());
    {
        const TensorView w_arr_p[] = {w_nchw};
        TensorView pw_arr_p[] = {pw_view};
        dw_op->prepackWeights(w_arr_p, pw_arr_p);
    }

    // getOutputTensorDesc
    TensorDesc in_c8_desc = in_c8.desc();
    TensorDesc pw_desc = pw_view.desc();
    const TensorDesc dw_desc_arr[] = {in_c8_desc, pw_desc};
    auto out_descs = dw_op->getOutputTensorDesc(dw_desc_arr);

    std::vector<float> packed_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_c8 = test::make_packed(out_descs[0], packed_out_buf.data());

    {
        const TensorView ins[] = {in_c8, pw_view};
        TensorView outs[] = {out_c8};
        dw_op->compute(outs, ins);
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

    NNOPS_EXPECT_NEAR(result_buf[0], 18.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(result_buf[1], 13.5f, 1e-4f);
}

// ============================================================
// C=8 hand-verified test (exercises SIMD path with full C8)
// ============================================================

NNOPS_TEST(dwconv_full_c8_hand_check) {
    // C=8, 3x3 input, 2x2 kernel, stride=1 → output 2x2
    // Each channel has known values for manual verification
    const int64_t ishape[] = {1, 8, 3, 3};
    const int64_t wshape[] = {8, 1, 2, 2};

    // Input: channel c has value (c+1)*10 at position (0,0), decreasing to right/bottom
    std::vector<float> in_buf(72);  // 1*8*3*3
    std::vector<float> w_buf(32);   // 8*2*2
    for (int c = 0; c < 8; ++c) {
        for (int h = 0; h < 3; ++h) {
            for (int w = 0; w < 3; ++w) {
                in_buf[c*9 + h*3 + w] = (c + 1) * 10.0f + h + w * 0.1f;
            }
        }
        for (int kh = 0; kh < 2; ++kh) {
            for (int kw = 0; kw < 2; ++kw) {
                w_buf[c*4 + kh*2 + kw] = 0.5f + c * 0.1f;
            }
        }
    }

    TensorView in_nchw(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_nchw(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);

    // NCHW reference
    DepthwiseConvAttributes attrs;
    attrs.kernel_size = {1, 2, 2};
    attrs.stride      = {1, 1, 1};
    attrs.dilation    = {1, 1, 1};
    attrs.padding     = {0, 0, 0};

    auto dw_op_ref = DepthwiseConv::create(attrs, Backend::CPU);
    TensorDesc in_desc_ref = in_nchw.desc();
    TensorDesc w_desc_ref  = w_nchw.desc();
    const TensorDesc ref_desc_arr[] = {in_desc_ref, w_desc_ref};
    auto ref_descs = dw_op_ref->getOutputTensorDesc(ref_desc_arr);

    std::vector<float> ref_buf(static_cast<size_t>(ref_descs[0].numel()));
    auto out_ref = test::make_planar(ref_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        const TensorView ref_arr[] = {in_nchw, w_nchw};
        backend::cpu::reference::depthwise_conv_ref(
            attrs, out_ref, ref_arr, ctx, nullptr);
    }

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

    // Prepack weight: query → allocate → pack
    auto dw_op = DepthwiseConv::create(attrs, Backend::CPU);

    TensorView pw_query;
    {
        const TensorView w_arr_q[] = {w_nchw};
        TensorView pw_arr_q[] = {pw_query};
        dw_op->prepackWeights(w_arr_q, pw_arr_q);
        pw_query = pw_arr_q[0];
    }
    std::vector<float> pw_buf(static_cast<size_t>(pw_query.numel()));
    auto pw_view = test::make_planar(pw_query.desc(), pw_buf.data());
    {
        const TensorView w_arr_p[] = {w_nchw};
        TensorView pw_arr_p[] = {pw_view};
        dw_op->prepackWeights(w_arr_p, pw_arr_p);
    }

    // getOutputTensorDesc
    TensorDesc in_c8_desc = in_c8.desc();
    TensorDesc pw_desc = pw_view.desc();
    const TensorDesc dw_desc_arr[] = {in_c8_desc, pw_desc};
    auto out_descs = dw_op->getOutputTensorDesc(dw_desc_arr);

    std::vector<float> packed_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_c8 = test::make_packed(out_descs[0], packed_out_buf.data());

    {
        const TensorView ins[] = {in_c8, pw_view};
        TensorView outs[] = {out_c8};
        dw_op->compute(outs, ins);
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

    // Compare each element
    for (int i = 0; i < 32; ++i) {
        NNOPS_EXPECT_NEAR(result_buf[i], ref_buf[i], 1e-4f);
    }
}

// ============================================================
// NCHWC8 vs Reference random tests
// ============================================================

NNOPS_TEST(dwconv_nchwc8_vs_ref_basic) {
    test_nchwc8_vs_ref({2, 8, 16, 16}, {8, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}});
}

NNOPS_TEST(dwconv_nchwc8_vs_ref_stride2) {
    test_nchwc8_vs_ref({1, 8, 12, 12}, {8, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 2, 2}, {1, 1, 1}, {0, 0, 0}});
}

NNOPS_TEST(dwconv_nchwc8_vs_ref_stride2_odd) {
    test_nchwc8_vs_ref({1, 4, 7, 7}, {4, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 2, 2}, {1, 1, 1}, {0, 0, 0}});
}

NNOPS_TEST(dwconv_nchwc8_vs_ref_padding) {
    test_nchwc8_vs_ref({1, 8, 8, 8}, {8, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}});
}

NNOPS_TEST(dwconv_nchwc8_vs_ref_dilation) {
    test_nchwc8_vs_ref({1, 4, 10, 10}, {4, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 2, 2}, {0, 0, 0}});
}

NNOPS_TEST(dwconv_nchwc8_vs_ref_bias) {
    test_nchwc8_vs_ref({1, 8, 8, 8}, {8, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}}, true);
}

NNOPS_TEST(dwconv_nchwc8_vs_ref_add_to) {
    DepthwiseConvAttributes attrs{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}};
    attrs.add_to = true;
    test_nchwc8_vs_ref({1, 8, 8, 8}, {8, 1, 3, 3}, attrs);
}

NNOPS_TEST(dwconv_nchwc8_vs_ref_partial_c8) {
    // C=3 → partial C8
    test_nchwc8_vs_ref({1, 3, 8, 8}, {3, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}});
}

NNOPS_TEST(dwconv_nchwc8_vs_ref_partial_c8_bias) {
    test_nchwc8_vs_ref({1, 5, 8, 8}, {5, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}}, true);
}

NNOPS_TEST(dwconv_nchwc8_vs_ref_single_channel) {
    test_nchwc8_vs_ref({1, 1, 8, 8}, {1, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}});
}

NNOPS_TEST(dwconv_nchwc8_vs_ref_multi_c8) {
    // C=17 → 3 C8 blocks (2 full + 1 partial)
    test_nchwc8_vs_ref({1, 17, 8, 8}, {17, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}});
}

NNOPS_TEST(dwconv_nchwc8_vs_ref_small_kernel) {
    test_nchwc8_vs_ref({1, 8, 16, 16}, {8, 1, 1, 1},
        DepthwiseConvAttributes{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}});
}

NNOPS_TEST(dwconv_nchwc8_vs_ref_batch) {
    test_nchwc8_vs_ref({4, 8, 16, 16}, {8, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}});
}

NNOPS_TEST(dwconv_nchwc8_vs_ref_large) {
    test_nchwc8_vs_ref({1, 16, 32, 32}, {16, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}});
}

NNOPS_TEST(dwconv_nchwc8_vs_ref_1x1_kernel_stride2) {
    test_nchwc8_vs_ref({1, 8, 8, 8}, {8, 1, 1, 1},
        DepthwiseConvAttributes{{1, 1, 1}, {1, 2, 2}, {1, 1, 1}, {0, 0, 0}});
}

// ============================================================
// Direct NCHWC8 data test — bypasses LayoutConvert/prepack
// ============================================================

NNOPS_TEST(dwconv_direct_nchwc8_data) {
    // C=8, IH=2, IW=2, KH=1, KW=1, pad=0 → OH=2, OW=2
    // Manually populate NCHWC8 arrays (no LayoutConvert/prepack needed)
    const int64_t ishape[] = {1, 8, 2, 2};
    const int64_t ROW = 16;  // IW*8 = 16, no alignment needed (already 32B-aligned)

    // NCHWC8 input: manual layout [C8=1, IH=2, IW=2, 8]
    std::vector<float> in_c8_buf(32, 0.0f);  // C8*IH*ROW
    for (int ih = 0; ih < 2; ++ih) {
        for (int iw = 0; iw < 2; ++iw) {
            for (int c = 0; c < 8; ++c) {
                in_c8_buf[ih * ROW + iw * 8 + c] = (c + 1) * 10.0f + ih + iw * 0.1f;
            }
        }
    }
    int64_t in_pitch = ROW * 4;
    TensorView in_c8(ishape, DataType::f32, in_c8_buf.data(), in_pitch, TensorLayout::NCHWC8);

    // Packed weight: manual [C8=1, KH=1, KW=1, 8]
    std::vector<float> pw_buf(8);
    for (int c = 0; c < 8; ++c) {
        pw_buf[c] = 0.5f + c * 0.1f;
    }
    const int64_t pw_shape[] = {1, 1, 1, 8};
    TensorView pw_view(std::span<const int64_t>(pw_shape, 4), DataType::f32,
                        pw_buf.data(), TensorLayout::PackedWeight);

    // getOutputTensorDesc
    DepthwiseConvAttributes attrs;
    attrs.kernel_size = {1, 1, 1};
    attrs.stride      = {1, 1, 1};
    attrs.dilation    = {1, 1, 1};
    attrs.padding     = {0, 0, 0};

    auto dw_op = DepthwiseConv::create(attrs, Backend::CPU);

    TensorDesc in_desc = in_c8.desc();
    TensorDesc pw_desc = pw_view.desc();
    const TensorDesc dw_desc_arr[] = {in_desc, pw_desc};
    auto out_descs = dw_op->getOutputTensorDesc(dw_desc_arr);

    std::vector<float> out_c8_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()), 0.0f);
    auto out_c8 = test::make_packed(out_descs[0], out_c8_buf.data());

    {
        const TensorView ins[] = {in_c8, pw_view};
        TensorView outs[] = {out_c8};
        dw_op->compute(outs, ins);
    }

    // Manual check: result for channel c at (oh, ow) = in[oh][ow][c] * w[c][0][0]
    const float* out_ptr = out_c8.ptr<float>();
    int64_t out_row = out_c8.row_stride_elems();
    for (int c = 0; c < 8; ++c) {
        for (int oh = 0; oh < 2; ++oh) {
            for (int ow = 0; ow < 2; ++ow) {
                float expected = ((c + 1) * 10.0f + oh + ow * 0.1f) * (0.5f + c * 0.1f);
                float actual = out_ptr[oh * out_row + ow * 8 + c];
                NNOPS_EXPECT_NEAR(actual, expected, 1e-4f);
            }
        }
    }
}

// ============================================================
// Prepack interface tests
// ============================================================

NNOPS_TEST(dwconv_prepack_query) {
    // Verify that prepackWeights with empty output fills correct metadata
    const int64_t wshape[] = {5, 1, 3, 3};
    std::vector<float> w_buf(45);
    TensorView weight(wshape, DataType::f32, w_buf.data());

    auto op = DepthwiseConv::create(DepthwiseConvAttributes{}, Backend::CPU);
    TensorView packed_w;
    {
        const TensorView w_arr[] = {weight};
        TensorView pw_arr[] = {packed_w};
        op->prepackWeights(w_arr, pw_arr);
        packed_w = pw_arr[0];  // copy back metadata
    }

    // C=5 → C8=1
    NNOPS_EXPECT_EQ(packed_w.rank(), 4);
    NNOPS_EXPECT_EQ(packed_w.shape(0), 1);  // C8
    NNOPS_EXPECT_EQ(packed_w.shape(1), 3);  // KH
    NNOPS_EXPECT_EQ(packed_w.shape(2), 3);  // KW
    NNOPS_EXPECT_EQ(packed_w.shape(3), 8);  // lanes
    NNOPS_EXPECT_EQ(packed_w.data_type(), DataType::f32);
}

NNOPS_TEST(dwconv_prepack_query_bias) {
    const int64_t wshape[] = {10, 1, 3, 3};
    const int64_t bshape[] = {10};
    std::vector<float> w_buf(90);
    std::vector<float> b_buf(10);
    TensorView weight(wshape, DataType::f32, w_buf.data());
    TensorView bias(bshape, DataType::f32, b_buf.data());

    auto op = DepthwiseConv::create(DepthwiseConvAttributes{}, Backend::CPU);
    TensorView packed_w, packed_b;
    {
        const TensorView w_arr[] = {weight, bias};
        TensorView pw_arr[] = {packed_w, packed_b};
        op->prepackWeights(w_arr, pw_arr);
        packed_w = pw_arr[0];
        packed_b = pw_arr[1];
    }

    // C=10 → C8=2
    NNOPS_EXPECT_EQ(packed_w.shape(0), 2);
    NNOPS_EXPECT_EQ(packed_b.shape(0), 2);
    NNOPS_EXPECT_EQ(packed_b.shape(1), 8);
}

// ============================================================
// 3D NCDHWC8 Helpers
// ============================================================

/// Generic 3D NCDHWC8 roundtrip: pack input, prepack weight, run backend,
/// unpack output, compare against NCDHW scalar reference.
static void test_ncdhwc8_vs_ref(
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& w_shape,
    const DepthwiseConvAttributes& attrs,
    bool has_bias = false)
{
    const int64_t C = w_shape[0];

    // -- Random NCDHW input + weight --
    auto [in_vec, in_ncdhw] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_ncdhw]  = test::make_random_tensor(w_shape);
    std::vector<float> b_vec;
    TensorView b_nchw;
    if (has_bias) {
        auto p = test::make_random_tensor({C});
        b_vec = std::move(p.first);
        b_nchw = p.second;
    }

    // -- NCDHW scalar reference (golden) --
    auto dw_op_ref = DepthwiseConv::create(attrs, Backend::CPU);

    TensorDesc in_desc_ref = in_ncdhw.desc();
    TensorDesc w_desc_ref  = w_ncdhw.desc();
    const TensorDesc ref_desc_arr[] = {in_desc_ref, w_desc_ref};
    auto ref_descs = dw_op_ref->getOutputTensorDesc(ref_desc_arr);

    std::vector<float> ref_buf(static_cast<size_t>(ref_descs[0].numel()));
    auto out_ref = test::make_planar(ref_descs[0], ref_buf.data());
    if (attrs.add_to) {
        for (auto& v : ref_buf) v = 1.0f;
    }
    {
        ComputeContext ctx;
        if (has_bias) {
            const TensorView ref_arr[] = {in_ncdhw, w_ncdhw, b_nchw};
            backend::cpu::reference::depthwise_conv_ref(
                attrs, out_ref, ref_arr, ctx, nullptr);
        } else {
            const TensorView ref_arr[] = {in_ncdhw, w_ncdhw};
            backend::cpu::reference::depthwise_conv_ref(
                attrs, out_ref, ref_arr, ctx, nullptr);
        }
    }

    // -- LayoutConvert: NCDHW → NCDHWC8 --
    auto lc_in = LayoutConvert::create(TensorLayout::NCDHWC8, Backend::CPU);
    TensorDesc in_planar_desc = in_ncdhw.desc();
    const TensorDesc lc_in_desc_arr[] = {in_planar_desc};
    auto lc_in_descs = lc_in->getOutputTensorDesc(lc_in_desc_arr);

    std::vector<float> packed_in_buf(
        static_cast<size_t>(lc_in_descs[0].storage_bytes()));
    auto in_c8 = test::make_packed(lc_in_descs[0], packed_in_buf.data());
    {
        const TensorView lc_ins[] = {in_ncdhw};
        TensorView lc_outs[] = {in_c8};
        lc_in->compute(lc_outs, lc_ins);
    }

    // -- Prepack weight: query → allocate → pack --
    auto dw_op = DepthwiseConv::create(attrs, Backend::CPU);

    // Step 1: query packed shapes
    TensorView packed_w_query;
    TensorView packed_b_query;
    if (has_bias) {
        const TensorView w_arr_q[] = {w_ncdhw, b_nchw};
        TensorView pw_arr_q[] = {packed_w_query, packed_b_query};
        dw_op->prepackWeights(w_arr_q, pw_arr_q);
        packed_w_query = pw_arr_q[0];
        packed_b_query = pw_arr_q[1];
    } else {
        const TensorView w_arr_q[] = {w_ncdhw};
        TensorView pw_arr_q[] = {packed_w_query};
        dw_op->prepackWeights(w_arr_q, pw_arr_q);
        packed_w_query = pw_arr_q[0];
    }

    // Step 2: allocate
    std::vector<float> pw_buf(static_cast<size_t>(packed_w_query.numel()));
    auto pw_view = test::make_planar(packed_w_query.desc(), pw_buf.data());

    std::vector<float> pb_buf;
    TensorView pb_view;
    if (has_bias) {
        pb_buf.resize(static_cast<size_t>(packed_b_query.numel()));
        pb_view = test::make_planar(packed_b_query.desc(), pb_buf.data());
    }

    // Step 3: pack
    if (has_bias) {
        const TensorView w_arr_p[] = {w_ncdhw, b_nchw};
        TensorView pw_arr_p[] = {pw_view, pb_view};
        dw_op->prepackWeights(w_arr_p, pw_arr_p);
    } else {
        const TensorView w_arr_p[] = {w_ncdhw};
        TensorView pw_arr_p[] = {pw_view};
        dw_op->prepackWeights(w_arr_p, pw_arr_p);
    }

    // -- getOutputTensorDesc for NCDHWC8 output --
    TensorDesc in_c8_desc = in_c8.desc();
    TensorDesc pw_desc = pw_view.desc();
    std::vector<TensorDesc> out_descs;
    if (has_bias) {
        TensorDesc pb_desc = pb_view.desc();
        const TensorDesc dw_desc_arr[] = {in_c8_desc, pw_desc, pb_desc};
        out_descs = dw_op->getOutputTensorDesc(dw_desc_arr);
    } else {
        const TensorDesc dw_desc_arr[] = {in_c8_desc, pw_desc};
        out_descs = dw_op->getOutputTensorDesc(dw_desc_arr);
    }

    // -- Allocate NCDHWC8 output --
    std::vector<float> packed_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_c8 = test::make_packed(out_descs[0], packed_out_buf.data());
    if (attrs.add_to) {
        std::fill(packed_out_buf.begin(), packed_out_buf.end(), 1.0f);
    }

    // -- DepthwiseConv::compute --
    {
        ComputeContext ctx;
        if (has_bias) {
            const TensorView ins[] = {in_c8, pw_view, pb_view};
            TensorView outs[] = {out_c8};
            dw_op->compute(outs, ins, ctx);
        } else {
            const TensorView ins[] = {in_c8, pw_view};
            TensorView outs[] = {out_c8};
            dw_op->compute(outs, ins, ctx);
        }
    }

    // -- LayoutConvert back: NCDHWC8 → NCDHW --
    auto lc_out = LayoutConvert::create(TensorLayout::NCDHW, Backend::CPU);
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
// 3D hand-verified small test
// ============================================================

NNOPS_TEST(dwconv_3d_basic) {
    // 1x2x3x3x3 input, 1x2x2x2x2 kernel (KD=2, KH=2, KW=2)
    // C=2, all-ones input, all-0.5 kernel, stride=1, pad=0
    // Each output = 2*2*2 * 1.0 * 0.5 = 8 * 0.5 = 4.0
    const int64_t ishape[] = {1, 2, 3, 3, 3};
    const int64_t wshape[] = {2, 1, 2, 2, 2};

    std::vector<float> in_buf(54, 1.0f);   // 1*2*3*3*3
    std::vector<float> w_buf(16, 0.5f);     // 2*1*2*2*2

    TensorView in_ncdhw(ishape, DataType::f32, in_buf.data(), TensorLayout::NCDHW);
    TensorView w_ncdhw(wshape, DataType::f32, w_buf.data(), TensorLayout::NCDHW);

    // NCDHW reference
    DepthwiseConvAttributes attrs;
    attrs.kernel_size = {2, 2, 2};
    attrs.stride      = {1, 1, 1};
    attrs.dilation    = {1, 1, 1};
    attrs.padding     = {0, 0, 0};

    auto dw_op_ref = DepthwiseConv::create(attrs, Backend::CPU);
    TensorDesc in_desc_ref = in_ncdhw.desc();
    TensorDesc w_desc_ref = w_ncdhw.desc();
    const TensorDesc ref_desc_arr[] = {in_desc_ref, w_desc_ref};
    auto ref_descs = dw_op_ref->getOutputTensorDesc(ref_desc_arr);

    std::vector<float> ref_buf(static_cast<size_t>(ref_descs[0].numel()));
    auto out_ref = test::make_planar(ref_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        const TensorView ref_arr[] = {in_ncdhw, w_ncdhw};
        backend::cpu::reference::depthwise_conv_ref(
            attrs, out_ref, ref_arr, ctx, nullptr);
    }

    // LayoutConvert: NCDHW → NCDHWC8
    auto lc_in = LayoutConvert::create(TensorLayout::NCDHWC8, Backend::CPU);
    TensorDesc in_desc = in_ncdhw.desc();
    const TensorDesc lc_in_desc_arr[] = {in_desc};
    auto lc_in_descs = lc_in->getOutputTensorDesc(lc_in_desc_arr);

    std::vector<float> packed_in_buf(
        static_cast<size_t>(lc_in_descs[0].storage_bytes()));
    auto in_c8 = test::make_packed(lc_in_descs[0], packed_in_buf.data());
    {
        const TensorView lc_ins[] = {in_ncdhw};
        TensorView lc_outs[] = {in_c8};
        lc_in->compute(lc_outs, lc_ins);
    }

    // Prepack weight: query → allocate → pack
    auto dw_op = DepthwiseConv::create(attrs, Backend::CPU);

    TensorView pw_query;
    {
        const TensorView w_arr_q[] = {w_ncdhw};
        TensorView pw_arr_q[] = {pw_query};
        dw_op->prepackWeights(w_arr_q, pw_arr_q);
        pw_query = pw_arr_q[0];
    }
    std::vector<float> pw_buf(static_cast<size_t>(pw_query.numel()));
    auto pw_view = test::make_planar(pw_query.desc(), pw_buf.data());
    {
        const TensorView w_arr_p[] = {w_ncdhw};
        TensorView pw_arr_p[] = {pw_view};
        dw_op->prepackWeights(w_arr_p, pw_arr_p);
    }

    // getOutputTensorDesc
    TensorDesc in_c8_desc = in_c8.desc();
    TensorDesc pw_desc = pw_view.desc();
    const TensorDesc dw_desc_arr[] = {in_c8_desc, pw_desc};
    auto out_descs = dw_op->getOutputTensorDesc(dw_desc_arr);

    std::vector<float> packed_out_buf(
        static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_c8 = test::make_packed(out_descs[0], packed_out_buf.data());

    {
        const TensorView ins[] = {in_c8, pw_view};
        TensorView outs[] = {out_c8};
        dw_op->compute(outs, ins);
    }

    // LayoutConvert back: NCDHWC8 → NCDHW
    auto lc_out = LayoutConvert::create(TensorLayout::NCDHW, Backend::CPU);
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

    for (int i = 0; i < 16; ++i) {
        NNOPS_EXPECT_NEAR(result_buf[i], 4.0f, 1e-4f);
    }
}

// ============================================================
// 3D NCDHWC8 vs Reference random tests
// ============================================================

NNOPS_TEST(dwconv_3d_ncdhwc8_vs_ref_basic) {
    test_ncdhwc8_vs_ref({1, 8, 8, 8, 8}, {8, 1, 3, 3, 3},
        DepthwiseConvAttributes{{3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}});
}

NNOPS_TEST(dwconv_3d_ncdhwc8_vs_ref_stride) {
    test_ncdhwc8_vs_ref({1, 4, 6, 6, 6}, {4, 1, 2, 2, 2},
        DepthwiseConvAttributes{{2, 2, 2}, {1, 2, 2}, {1, 1, 1}, {0, 0, 0}});
}

NNOPS_TEST(dwconv_3d_ncdhwc8_vs_ref_padding) {
    test_ncdhwc8_vs_ref({1, 8, 4, 4, 4}, {8, 1, 2, 2, 2},
        DepthwiseConvAttributes{{2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
}

NNOPS_TEST(dwconv_3d_ncdhwc8_vs_ref_bias) {
    test_ncdhwc8_vs_ref({1, 4, 4, 4, 4}, {4, 1, 2, 2, 2},
        DepthwiseConvAttributes{{2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}}, true);
}

NNOPS_TEST(dwconv_3d_ncdhwc8_vs_ref_partial_c8) {
    test_ncdhwc8_vs_ref({1, 3, 4, 4, 4}, {3, 1, 2, 2, 2},
        DepthwiseConvAttributes{{2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}});
}

NNOPS_TEST(dwconv_3d_ncdhwc8_vs_ref_small_kernel) {
    test_ncdhwc8_vs_ref({1, 8, 6, 6, 6}, {8, 1, 1, 1, 1},
        DepthwiseConvAttributes{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}});
}

NNOPS_TEST(dwconv_3d_ncdhwc8_vs_ref_add_to) {
    DepthwiseConvAttributes attrs{{2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}};
    attrs.add_to = true;
    test_ncdhwc8_vs_ref({1, 4, 4, 4, 4}, {4, 1, 2, 2, 2}, attrs);
}
