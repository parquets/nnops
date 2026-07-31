/// Unit tests for DepthwiseConv operator — NCHWC8 only.
///
/// All tests explicitly go through the NCHWC8 path:
///   NCHW in → pack_nchw_to_nchwc8 → dwconv(NCHWC8, prepacked_w) → unpack → compare
///
/// The NCHW scalar reference (depthwise_conv_ref) is the golden ground truth.

#include "nnops/ops/depthwise_conv.hpp"
#include "backend/cpu/layout_convert.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

// Forward declare the reference kernel (NCHW scalar, ground truth).
namespace nnops::backend::cpu::reference {
    void depthwise_conv_ref(const DepthwiseConvAttributes& attrs,
                               TensorView& output,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* workspace);
}
namespace nnops::backend::cpu {
    void depthwise_conv_cpu(const DepthwiseConvAttributes& attrs,
                               TensorView& output,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* workspace);
}

// ============================================================
// Helpers
// ============================================================

/// Compute 32-byte-aligned pitch in bytes for NCHWC8 row stride.
inline int64_t nchwc8_pitch(int64_t W, int64_t elem_size = 4) {
    return ((W * 8 * elem_size + 31) / 32) * 32;
}

/// Compute output spatial dims for 2D depthwise convolution.
inline std::pair<int64_t, int64_t> dwconv_out_2d(
    int64_t IH, int64_t IW, int64_t KH, int64_t KW,
    int64_t SH, int64_t SW, int64_t PH, int64_t PW,
    int64_t DH = 1, int64_t DW = 1)
{
    int64_t OH = (IH + 2 * PH - DH * (KH - 1) - 1) / SH + 1;
    int64_t OW = (IW + 2 * PW - DW * (KW - 1) - 1) / SW + 1;
    return {OH, OW};
}

/// Prepack NCHW weight [C, 1, KH, KW] → dense [C8, KH, KW, 8].
/// Uses the operator's prepackWeights interface (query → allocate → pack).
static void prepack_dwconv_weight(
    const TensorView& weight_nchw,
    std::vector<float>& packed_buf,
    TensorView& packed_view,
    bool has_bias,
    const TensorView* bias_nchw,
    std::vector<float>& bias_buf,
    TensorView& bias_view)
{
    const int64_t C  = weight_nchw.shape(0);
    const int64_t KH = weight_nchw.shape(2);
    const int64_t KW = weight_nchw.shape(3);
    const int64_t C8 = (C + 7) / 8;
    const DataType dtype = weight_nchw.data_type();

    // Packed weight
    {
        const int64_t w_shape[] = {C8, KH, KW, 8};
        packed_buf.resize(static_cast<size_t>(C8 * KH * KW * 8));
        packed_view = TensorView(std::span<const int64_t>(w_shape, 4), dtype,
                                  packed_buf.data(), TensorLayout::PackedWeight);
    }

    // Packed bias
    if (has_bias) {
        const int64_t b_shape[] = {C8, 8};
        bias_buf.resize(static_cast<size_t>(C8 * 8));
        bias_view = TensorView(std::span<const int64_t>(b_shape, 2), dtype,
                                bias_buf.data(), TensorLayout::PackedWeight);
    }

    // Use operator's prepack interface
    auto op = DepthwiseConv::create(DepthwiseConvAttributes{});
    if (has_bias) {
        const TensorView w_arr[] = {weight_nchw, *bias_nchw};
        TensorView pw_arr[] = {packed_view, bias_view};
        op->prepackWeights(w_arr, pw_arr);
    } else {
        const TensorView w_arr[] = {weight_nchw};
        TensorView pw_arr[] = {packed_view};
        op->prepackWeights(w_arr, pw_arr);
    }
}

/// Generic NCHWC8 roundtrip test: pack input, prepack weight, run backend,
/// unpack output, compare against NCHW scalar reference.
static void test_nchwc8_vs_ref(
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& w_shape,
    DepthwiseConvAttributes attrs,
    bool has_bias = false)
{
    const int64_t N = in_shape[0], C = in_shape[1], IH = in_shape[2], IW = in_shape[3];
    const int64_t C8 = (C + 7) / 8;
    const int64_t KH = attrs.kernel_size[1], KW = attrs.kernel_size[2];
    const int64_t SH = attrs.stride[1], SW = attrs.stride[2];
    const int64_t DH = attrs.dilation[1], DW = attrs.dilation[2];
    const int64_t PH = attrs.padding[1], PW = attrs.padding[2];
    auto [OH, OW] = dwconv_out_2d(IH, IW, KH, KW, SH, SW, PH, PW, DH, DW);

    // Random NCHW input + weight
    auto [in_vec, in_nchw] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_nchw]  = test::make_random_tensor(w_shape);
    std::vector<float> b_vec;
    TensorView b_nchw;
    if (has_bias) {
        auto p = test::make_random_tensor({C});
        b_vec = std::move(p.first);
        b_nchw = p.second;
    }

    // NCHW scalar reference (golden)
    std::vector<float> ref_out(static_cast<size_t>(N * C * OH * OW));
    const std::vector<int64_t> oshape_ref = {N, C, OH, OW};
    TensorView out_ref(oshape_ref, DataType::f32, ref_out.data(), TensorLayout::NCHW);
    if (attrs.add_to) {
        for (auto& v : ref_out) v = 1.0f;
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

    // Pack input to NCHWC8
    int64_t in_pitch = nchwc8_pitch(IW);
    int64_t in_pitch_elems = in_pitch / 4;
    std::vector<float> packed_in(static_cast<size_t>(N * C8 * IH * in_pitch_elems));
    TensorView in_c8(in_shape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    // Prepack weight + bias
    std::vector<float> pw_buf, pb_buf;
    TensorView pw_view, pb_view;
    prepack_dwconv_weight(w_nchw, pw_buf, pw_view,
                           has_bias, has_bias ? &b_nchw : nullptr,
                           pb_buf, pb_view);

    // NCHWC8 output
    int64_t out_pitch = nchwc8_pitch(OW);
    int64_t out_pitch_elems = out_pitch / 4;
    std::vector<float> packed_out(static_cast<size_t>(N * C8 * OH * out_pitch_elems));
    TensorView out_c8(oshape_ref, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    // For add_to: pre-fill output with 1.0
    if (attrs.add_to) {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c8i = 0; c8i < C8; ++c8i) {
                for (int64_t oh = 0; oh < OH; ++oh) {
                    float* row = packed_out.data() + ((n * C8 + c8i) * OH + oh) * out_pitch_elems;
                    for (int64_t ow = 0; ow < OW; ++ow)
                        for (int64_t l = 0; l < 8; ++l)
                            row[ow * 8 + l] = 1.0f;
                }
            }
        }
    }

    // Run backend
    {
        ComputeContext ctx;
        if (has_bias) {
            const TensorView ins_arr[] = {in_c8, pw_view, pb_view};
            backend::cpu::depthwise_conv_cpu(attrs, out_c8, ins_arr, ctx, nullptr);
        } else {
            const TensorView ins_arr[] = {in_c8, pw_view};
            backend::cpu::depthwise_conv_cpu(attrs, out_c8, ins_arr, ctx, nullptr);
        }
    }

    // Unpack and compare
    std::vector<float> result(static_cast<size_t>(N * C * OH * OW));
    TensorView res_nchw(oshape_ref, DataType::f32, result.data(), TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);
    NNOPS_EXPECT_TRUE(test::allclose(res_nchw, out_ref, 1e-4f, 1e-4f));
}

// ============================================================
// Hand-verified small tests (check exact expected values)
// ============================================================

NNOPS_TEST(dwconv_basic_no_pad) {
    // 1x1x4x4 input all-ones, 1x1x3x3 kernel all-0.5, stride=1, pad=0
    // Expected: 2x2 output, each = 9 * (1.0 * 0.5) = 4.5
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t wshape[] = {1, 1, 3, 3};
    const int64_t C8 = 1, IH = 4, IW = 4, KH = 3, KW = 3, OH = 2, OW = 2;

    std::vector<float> in_buf(16, 1.0f);
    std::vector<float> w_buf(9, 0.5f);

    TensorView in_nchw(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_nchw(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);

    // Pack input → NCHWC8
    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    // Prepack weight
    std::vector<float> pw_buf(static_cast<size_t>(C8 * KH * KW * 8));
    const int64_t pw_shape[] = {C8, KH, KW, 8};
    TensorView pw_view(std::span<const int64_t>(pw_shape, 4), DataType::f32,
                        pw_buf.data(), TensorLayout::PackedWeight);
    {
        auto op = DepthwiseConv::create(DepthwiseConvAttributes{});
        const TensorView w_arr[] = {w_nchw};
        TensorView pw_arr[] = {pw_view};
        op->prepackWeights(w_arr, pw_arr);
    }

    // Run
    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(C8 * OH * out_pitch / 4));
    const int64_t oshape[] = {1, 1, OH, OW};
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    DepthwiseConvAttributes attrs;
    attrs.kernel_size = {1, 3, 3};
    attrs.stride = {1, 1, 1};
    attrs.padding = {0, 0, 0};

    {
        ComputeContext ctx;
        const TensorView ins[] = {in_c8, pw_view};
        backend::cpu::depthwise_conv_cpu(attrs, out_c8, ins, ctx, nullptr);
    }

    // Unpack
    std::vector<float> result(4);
    TensorView res_nchw(oshape, DataType::f32, result.data(), TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(result[i], 4.5f, 1e-4f);
    }
}

NNOPS_TEST(dwconv_stride_2) {
    // 1x1x6x6 input all-ones, 1x1x3x3 kernel all-0.5, stride=2
    const int64_t ishape[] = {1, 1, 6, 6};
    const int64_t wshape[] = {1, 1, 3, 3};
    const int64_t C8 = 1, IH = 6, IW = 6, KH = 3, KW = 3, OH = 2, OW = 2;

    std::vector<float> in_buf(36, 1.0f);
    std::vector<float> w_buf(9, 0.5f);

    TensorView in_nchw(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_nchw(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);

    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    std::vector<float> pw_buf(static_cast<size_t>(C8 * KH * KW * 8));
    const int64_t pw_shape[] = {C8, KH, KW, 8};
    TensorView pw_view(std::span<const int64_t>(pw_shape, 4), DataType::f32,
                        pw_buf.data(), TensorLayout::PackedWeight);
    {
        auto op = DepthwiseConv::create(DepthwiseConvAttributes{});
        const TensorView w_arr[] = {w_nchw};
        TensorView pw_arr[] = {pw_view};
        op->prepackWeights(w_arr, pw_arr);
    }

    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(C8 * OH * out_pitch / 4));
    const int64_t oshape[] = {1, 1, OH, OW};
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    DepthwiseConvAttributes attrs;
    attrs.kernel_size = {1, 3, 3};
    attrs.stride = {1, 2, 2};
    attrs.padding = {0, 0, 0};

    {
        ComputeContext ctx;
        const TensorView ins[] = {in_c8, pw_view};
        backend::cpu::depthwise_conv_cpu(attrs, out_c8, ins, ctx, nullptr);
    }

    std::vector<float> result(4);
    TensorView res_nchw(oshape, DataType::f32, result.data(), TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(result[i], 4.5f, 1e-4f);
    }
}

NNOPS_TEST(dwconv_padding_1) {
    // 1x1x2x2 input all-ones, 1x1x3x3 kernel all-0.5, pad=1
    // 4 valid positions per output corner → 4 * 0.5 = 2.0
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t wshape[] = {1, 1, 3, 3};
    const int64_t C8 = 1, IH = 2, IW = 2, KH = 3, KW = 3, OH = 2, OW = 2;

    std::vector<float> in_buf(4, 1.0f);
    std::vector<float> w_buf(9, 0.5f);

    TensorView in_nchw(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_nchw(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);

    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    std::vector<float> pw_buf(static_cast<size_t>(C8 * KH * KW * 8));
    const int64_t pw_shape[] = {C8, KH, KW, 8};
    TensorView pw_view(std::span<const int64_t>(pw_shape, 4), DataType::f32,
                        pw_buf.data(), TensorLayout::PackedWeight);
    {
        auto op = DepthwiseConv::create(DepthwiseConvAttributes{});
        const TensorView w_arr[] = {w_nchw};
        TensorView pw_arr[] = {pw_view};
        op->prepackWeights(w_arr, pw_arr);
    }

    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(C8 * OH * out_pitch / 4));
    const int64_t oshape[] = {1, 1, OH, OW};
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    DepthwiseConvAttributes attrs;
    attrs.kernel_size = {1, 3, 3};
    attrs.stride = {1, 1, 1};
    attrs.padding = {0, 1, 1};

    {
        ComputeContext ctx;
        const TensorView ins[] = {in_c8, pw_view};
        backend::cpu::depthwise_conv_cpu(attrs, out_c8, ins, ctx, nullptr);
    }

    std::vector<float> result(4);
    TensorView res_nchw(oshape, DataType::f32, result.data(), TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    NNOPS_EXPECT_NEAR(result[0], 2.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(result[1], 2.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(result[2], 2.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(result[3], 2.0f, 1e-4f);
}

NNOPS_TEST(dwconv_with_bias) {
    // 1x1x4x4 input all-1.0, 1x1x3x3 kernel all-0.5, bias=0.5 → each = 5.0
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t wshape[] = {1, 1, 3, 3};
    const int64_t bshape[] = {1};
    const int64_t C8 = 1, KH = 3, KW = 3, OH = 2, OW = 2;

    std::vector<float> in_buf(16, 1.0f);
    std::vector<float> w_buf(9, 0.5f);
    std::vector<float> b_buf = {0.5f};

    TensorView in_nchw(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_nchw(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);
    TensorView b_nchw(bshape, DataType::f32, b_buf.data(), TensorLayout::NCHW);

    // Pack input
    int64_t in_pitch = nchwc8_pitch(4);
    std::vector<float> packed_in(static_cast<size_t>(C8 * 4 * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    // Prepack
    std::vector<float> pw_buf(static_cast<size_t>(C8 * KH * KW * 8));
    const int64_t pw_shape[] = {C8, KH, KW, 8};
    TensorView pw_view(std::span<const int64_t>(pw_shape, 4), DataType::f32,
                        pw_buf.data(), TensorLayout::PackedWeight);
    std::vector<float> pb_buf(static_cast<size_t>(C8 * 8));
    const int64_t pb_shape[] = {C8, 8};
    TensorView pb_view(std::span<const int64_t>(pb_shape, 2), DataType::f32,
                        pb_buf.data(), TensorLayout::PackedWeight);
    {
        auto op = DepthwiseConv::create(DepthwiseConvAttributes{});
        const TensorView w_arr[] = {w_nchw, b_nchw};
        TensorView pw_arr[] = {pw_view, pb_view};
        op->prepackWeights(w_arr, pw_arr);
    }

    // Run
    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(C8 * OH * out_pitch / 4));
    const int64_t oshape[] = {1, 1, OH, OW};
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    DepthwiseConvAttributes attrs;
    attrs.kernel_size = {1, 3, 3};
    attrs.stride = {1, 1, 1};
    attrs.padding = {0, 0, 0};

    {
        ComputeContext ctx;
        const TensorView ins[] = {in_c8, pw_view, pb_view};
        backend::cpu::depthwise_conv_cpu(attrs, out_c8, ins, ctx, nullptr);
    }

    std::vector<float> result(4);
    TensorView res_nchw(oshape, DataType::f32, result.data(), TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(result[i], 5.0f, 1e-4f);
    }
}

NNOPS_TEST(dwconv_dilation) {
    // 1x1x5x5 input all-ones, 1x1x3x3 kernel all-0.5, dilation=2
    // Output: 1x1. 9 kernel positions, sum = 4.5
    const int64_t ishape[] = {1, 1, 5, 5};
    const int64_t wshape[] = {1, 1, 3, 3};
    const int64_t C8 = 1, IH = 5, IW = 5, KH = 3, KW = 3, OH = 1, OW = 1;

    std::vector<float> in_buf(25, 1.0f);
    std::vector<float> w_buf(9, 0.5f);

    TensorView in_nchw(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_nchw(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);

    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    std::vector<float> pw_buf(static_cast<size_t>(C8 * KH * KW * 8));
    const int64_t pw_shape[] = {C8, KH, KW, 8};
    TensorView pw_view(std::span<const int64_t>(pw_shape, 4), DataType::f32,
                        pw_buf.data(), TensorLayout::PackedWeight);
    {
        auto op = DepthwiseConv::create(DepthwiseConvAttributes{});
        const TensorView w_arr[] = {w_nchw};
        TensorView pw_arr[] = {pw_view};
        op->prepackWeights(w_arr, pw_arr);
    }

    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(C8 * OH * out_pitch / 4));
    const int64_t oshape[] = {1, 1, OH, OW};
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    DepthwiseConvAttributes attrs;
    attrs.kernel_size = {1, 3, 3};
    attrs.stride   = {1, 1, 1};
    attrs.dilation = {1, 2, 2};
    attrs.padding  = {0, 0, 0};

    {
        ComputeContext ctx;
        const TensorView ins[] = {in_c8, pw_view};
        backend::cpu::depthwise_conv_cpu(attrs, out_c8, ins, ctx, nullptr);
    }

    std::vector<float> result(1);
    TensorView res_nchw(oshape, DataType::f32, result.data(), TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);
    NNOPS_EXPECT_NEAR(result[0], 4.5f, 1e-4f);
}

NNOPS_TEST(dwconv_add_to) {
    // add_to with pre-filled 1.0 → each output = 4.5 + 1.0 = 5.5
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t wshape[] = {1, 1, 3, 3};
    const int64_t C8 = 1, KH = 3, KW = 3, OH = 2, OW = 2;

    std::vector<float> in_buf(16, 1.0f);
    std::vector<float> w_buf(9, 0.5f);

    TensorView in_nchw(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_nchw(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);

    int64_t in_pitch = nchwc8_pitch(4);
    std::vector<float> packed_in(static_cast<size_t>(C8 * 4 * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    std::vector<float> pw_buf(static_cast<size_t>(C8 * KH * KW * 8));
    const int64_t pw_shape[] = {C8, KH, KW, 8};
    TensorView pw_view(std::span<const int64_t>(pw_shape, 4), DataType::f32,
                        pw_buf.data(), TensorLayout::PackedWeight);
    {
        auto op = DepthwiseConv::create(DepthwiseConvAttributes{});
        const TensorView w_arr[] = {w_nchw};
        TensorView pw_arr[] = {pw_view};
        op->prepackWeights(w_arr, pw_arr);
    }

    // Pre-fill NCHWC8 output with 1.0
    int64_t out_pitch = nchwc8_pitch(OW);
    int64_t out_pitch_elems = out_pitch / 4;
    std::vector<float> packed_out(static_cast<size_t>(C8 * OH * out_pitch_elems), 1.0f);
    const int64_t oshape[] = {1, 1, OH, OW};
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    DepthwiseConvAttributes attrs;
    attrs.kernel_size = {1, 3, 3};
    attrs.stride = {1, 1, 1};
    attrs.padding = {0, 0, 0};
    attrs.add_to = true;

    {
        ComputeContext ctx;
        const TensorView ins[] = {in_c8, pw_view};
        backend::cpu::depthwise_conv_cpu(attrs, out_c8, ins, ctx, nullptr);
    }

    std::vector<float> result(4);
    TensorView res_nchw(oshape, DataType::f32, result.data(), TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(result[i], 5.5f, 1e-4f);
    }
}

NNOPS_TEST(dwconv_relu_epilogue) {
    // Channel 0: positive kernel → output positive → ReLU passes through
    // Channel 1: negative kernel → output negative → ReLU clips to 0
    const int64_t ishape[] = {1, 2, 4, 4};
    const int64_t wshape[] = {2, 1, 3, 3};
    const int64_t C = 2, C8 = 1, IH = 4, IW = 4, KH = 3, KW = 3, OH = 2, OW = 2;

    std::vector<float> in_buf(32, 1.0f);
    std::vector<float> w_buf(18);
    for (int i = 0; i < 9; ++i) {
        w_buf[i] = 0.5f;        // ch0: positive
        w_buf[i + 9] = -0.5f;   // ch1: negative
    }

    TensorView in_nchw(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
    TensorView w_nchw(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);

    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    std::vector<float> pw_buf(static_cast<size_t>(C8 * KH * KW * 8));
    const int64_t pw_shape[] = {C8, KH, KW, 8};
    TensorView pw_view(std::span<const int64_t>(pw_shape, 4), DataType::f32,
                        pw_buf.data(), TensorLayout::PackedWeight);
    {
        auto op = DepthwiseConv::create(DepthwiseConvAttributes{});
        const TensorView w_arr[] = {w_nchw};
        TensorView pw_arr[] = {pw_view};
        op->prepackWeights(w_arr, pw_arr);
    }

    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(C8 * OH * out_pitch / 4));
    const int64_t oshape[] = {1, 2, OH, OW};
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    DepthwiseConvAttributes attrs;
    attrs.kernel_size = {1, 3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0, 0};
    attrs.epilogue.type = EpilogueActivateType::Relu;

    {
        ComputeContext ctx;
        const TensorView ins[] = {in_c8, pw_view};
        backend::cpu::depthwise_conv_cpu(attrs, out_c8, ins, ctx, nullptr);
    }

    std::vector<float> result(8);
    TensorView res_nchw(oshape, DataType::f32, result.data(), TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    // Channel 0: 4.5 → ReLU → 4.5
    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(result[i], 4.5f, 1e-4f);
    }
    // Channel 1: -4.5 → ReLU → 0.0
    for (int i = 4; i < 8; ++i) {
        NNOPS_EXPECT_NEAR(result[i], 0.0f, 1e-4f);
    }
}

NNOPS_TEST(dwconv_multi_channel) {
    // C=2, different kernels per channel
    // Ch0: kernel all 1.0, input all 2.0 → 9 * 2.0 * 1.0 = 18.0
    // Ch1: kernel all 0.5, input all 3.0 → 9 * 3.0 * 0.5 = 13.5
    const int64_t ishape[] = {1, 2, 3, 3};
    const int64_t wshape[] = {2, 1, 3, 3};
    const int64_t C = 2, C8 = 1, IH = 3, IW = 3, KH = 3, KW = 3, OH = 1, OW = 1;

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

    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    std::vector<float> pw_buf(static_cast<size_t>(C8 * KH * KW * 8));
    const int64_t pw_shape[] = {C8, KH, KW, 8};
    TensorView pw_view(std::span<const int64_t>(pw_shape, 4), DataType::f32,
                        pw_buf.data(), TensorLayout::PackedWeight);
    {
        auto op = DepthwiseConv::create(DepthwiseConvAttributes{});
        const TensorView w_arr[] = {w_nchw};
        TensorView pw_arr[] = {pw_view};
        op->prepackWeights(w_arr, pw_arr);
    }

    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(C8 * OH * out_pitch / 4));
    const int64_t oshape[] = {1, 2, OH, OW};
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    DepthwiseConvAttributes attrs;
    attrs.kernel_size = {1, 3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0, 0};

    {
        ComputeContext ctx;
        const TensorView ins[] = {in_c8, pw_view};
        backend::cpu::depthwise_conv_cpu(attrs, out_c8, ins, ctx, nullptr);
    }

    std::vector<float> result(2);
    TensorView res_nchw(oshape, DataType::f32, result.data(), TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    NNOPS_EXPECT_NEAR(result[0], 18.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(result[1], 13.5f, 1e-4f);
}

// ============================================================
// C=8 hand-verified test (exercises SIMD path with full C8)
// ============================================================

NNOPS_TEST(dwconv_full_c8_hand_check) {
    // C=8, 3x3 input, 2x2 kernel, stride=1 → output 2x2
    // Each channel has known values for manual verification
    const int64_t ishape[] = {1, 8, 3, 3};
    const int64_t wshape[] = {8, 1, 2, 2};
    const int64_t C8 = 1, IH = 3, IW = 3, KH = 2, KW = 2, OH = 2, OW = 2;

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
    std::vector<float> ref_buf(32);  // 1*8*2*2
    const int64_t oshape[] = {1, 8, OH, OW};
    TensorView out_ref(oshape, DataType::f32, ref_buf.data(), TensorLayout::NCHW);
    {
        ComputeContext ctx;
        const TensorView ref_arr[] = {in_nchw, w_nchw};
        backend::cpu::reference::depthwise_conv_ref(
            DepthwiseConvAttributes{{1, 2, 2}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}},
            out_ref, ref_arr, ctx, nullptr);
    }

    // NCHWC8 path
    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    std::vector<float> pw_buf(static_cast<size_t>(C8 * KH * KW * 8));
    const int64_t pw_shape[] = {C8, KH, KW, 8};
    TensorView pw_view(std::span<const int64_t>(pw_shape, 4), DataType::f32,
                        pw_buf.data(), TensorLayout::PackedWeight);
    {
        auto op = DepthwiseConv::create(DepthwiseConvAttributes{});
        const TensorView w_arr[] = {w_nchw};
        TensorView pw_arr[] = {pw_view};
        op->prepackWeights(w_arr, pw_arr);
    }

    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(C8 * OH * out_pitch / 4));
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    DepthwiseConvAttributes attrs;
    attrs.kernel_size = {1, 2, 2};
    attrs.stride = {1, 1, 1};
    attrs.padding = {0, 0, 0};

    {
        ComputeContext ctx;
        const TensorView ins_arr[] = {in_c8, pw_view};
        backend::cpu::depthwise_conv_cpu(attrs, out_c8, ins_arr, ctx, nullptr);
    }

    std::vector<float> result(32);
    TensorView res_nchw(oshape, DataType::f32, result.data(), TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    // Compare each element
    for (int i = 0; i < 32; ++i) {
        NNOPS_EXPECT_NEAR(result[i], ref_buf[i], 1e-4f);
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
    // C=3 → partial C8 (valid_lanes=3)
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
// Direct NCHWC8 data test — bypasses pack_nchw/prepack
// ============================================================

NNOPS_TEST(dwconv_direct_nchwc8_data) {
    // C=8, IH=2, IW=2, KH=1, KW=1, pad=0 → OH=2, OW=2
    // Manually populate NCHWC8 arrays (no pack/prepack needed)
    const int64_t ishape[] = {1, 8, 2, 2};
    const int64_t C8 = 1, IH = 2, IW = 2, OH = 2, OW = 2;
    const int64_t ROW = 16;  // IW*8 = 16, no alignment needed (already 32B-aligned)

    // NCHWC8 input: manual layout [C8=1, IH=2, IW=2, 8]
    // Row 0 (ih=0): positions (0,0,ch0..7), (0,1,ch0..7)
    // Row 1 (ih=1): positions (1,0,ch0..7), (1,1,ch0..7)
    std::vector<float> in_c8_buf(C8 * IH * ROW, 0.0f);
    for (int ih = 0; ih < IH; ++ih) {
        for (int iw = 0; iw < IW; ++iw) {
            for (int c = 0; c < 8; ++c) {
                in_c8_buf[ih * ROW + iw * 8 + c] = (c + 1) * 10.0f + ih + iw * 0.1f;
            }
        }
    }
    int64_t in_pitch = ROW * 4;
    TensorView in_c8(ishape, DataType::f32, in_c8_buf.data(), in_pitch, TensorLayout::NCHWC8);

    // Packed weight: manual [C8=1, KH=1, KW=1, 8]
    // All 8 channel weights at (0,0) contiguous
    std::vector<float> pw_buf(8);
    for (int c = 0; c < 8; ++c) {
        pw_buf[c] = 0.5f + c * 0.1f;
    }
    const int64_t pw_shape[] = {1, 1, 1, 8};
    TensorView pw_view(std::span<const int64_t>(pw_shape, 4), DataType::f32,
                        pw_buf.data(), TensorLayout::PackedWeight);

    // Output: NCHWC8
    std::vector<float> out_c8_buf(C8 * OH * ROW, 0.0f);
    const int64_t oshape[] = {1, 8, OH, OW};
    int64_t out_pitch = ROW * 4;
    TensorView out_c8(oshape, DataType::f32, out_c8_buf.data(), out_pitch, TensorLayout::NCHWC8);

    DepthwiseConvAttributes attrs;
    attrs.kernel_size = {1, 1, 1};
    attrs.stride = {1, 1, 1};
    attrs.padding = {0, 0, 0};

    {
        ComputeContext ctx;
        const TensorView ins_arr[] = {in_c8, pw_view};
        backend::cpu::depthwise_conv_cpu(attrs, out_c8, ins_arr, ctx, nullptr);
    }

    // Manual check: result for channel c at (oh, ow) = in[oh][ow][c] * w[c][0][0]
    const auto* out_ptr = out_c8.ptr<float>();
    int64_t out_row = out_c8.row_stride_elems();
    for (int c = 0; c < 8; ++c) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
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

    auto op = DepthwiseConv::create(DepthwiseConvAttributes{});
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

    auto op = DepthwiseConv::create(DepthwiseConvAttributes{});
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
// 3D helpers
// ============================================================

/// Compute output spatial dims for 3D depthwise convolution.
inline std::tuple<int64_t, int64_t, int64_t> dwconv_out_3d(
    int64_t ID, int64_t IH, int64_t IW,
    int64_t KD, int64_t KH, int64_t KW,
    int64_t SD, int64_t SH, int64_t SW,
    int64_t PD, int64_t PH, int64_t PW,
    int64_t DD = 1, int64_t DH = 1, int64_t DW = 1)
{
    int64_t OD = (ID + 2 * PD - DD * (KD - 1) - 1) / SD + 1;
    int64_t OH = (IH + 2 * PH - DH * (KH - 1) - 1) / SH + 1;
    int64_t OW = (IW + 2 * PW - DW * (KW - 1) - 1) / SW + 1;
    return {OD, OH, OW};
}

/// Prepack NCDHW weight [C, 1, KD, KH, KW] → dense [C8, KD, KH, KW, 8].
static void prepack_dwconv_weight_3d(
    const TensorView& weight_ncdhw,
    std::vector<float>& packed_buf,
    TensorView& packed_view,
    bool has_bias,
    const TensorView* bias_nchw,
    std::vector<float>& bias_buf,
    TensorView& bias_view)
{
    const int64_t C  = weight_ncdhw.shape(0);
    const int64_t KD = weight_ncdhw.shape(2);
    const int64_t KH = weight_ncdhw.shape(3);
    const int64_t KW = weight_ncdhw.shape(4);
    const int64_t C8 = (C + 7) / 8;
    const DataType dtype = weight_ncdhw.data_type();

    // Packed weight
    {
        const int64_t w_shape[] = {C8, KD, KH, KW, 8};
        packed_buf.resize(static_cast<size_t>(C8 * KD * KH * KW * 8));
        packed_view = TensorView(std::span<const int64_t>(w_shape, 5), dtype,
                                  packed_buf.data(), TensorLayout::PackedWeight);
    }

    // Packed bias
    if (has_bias) {
        const int64_t b_shape[] = {C8, 8};
        bias_buf.resize(static_cast<size_t>(C8 * 8));
        bias_view = TensorView(std::span<const int64_t>(b_shape, 2), dtype,
                                bias_buf.data(), TensorLayout::PackedWeight);
    }

    // Use operator's prepack interface
    auto op = DepthwiseConv::create(DepthwiseConvAttributes{});
    if (has_bias) {
        const TensorView w_arr[] = {weight_ncdhw, *bias_nchw};
        TensorView pw_arr[] = {packed_view, bias_view};
        op->prepackWeights(w_arr, pw_arr);
    } else {
        const TensorView w_arr[] = {weight_ncdhw};
        TensorView pw_arr[] = {packed_view};
        op->prepackWeights(w_arr, pw_arr);
    }
}

/// Generic 3D NCDHWC8 roundtrip test.
static void test_ncdhwc8_vs_ref(
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& w_shape,
    DepthwiseConvAttributes attrs,
    bool has_bias = false)
{
    const int64_t N = in_shape[0], C = in_shape[1];
    const int64_t ID = in_shape[2], IH = in_shape[3], IW = in_shape[4];
    const int64_t C8 = (C + 7) / 8;
    const int64_t KD = attrs.kernel_size[0], KH = attrs.kernel_size[1], KW = attrs.kernel_size[2];
    const int64_t SD = attrs.stride[0], SH = attrs.stride[1], SW = attrs.stride[2];
    const int64_t DD = attrs.dilation[0], DH = attrs.dilation[1], DW = attrs.dilation[2];
    const int64_t PD = attrs.padding[0], PH = attrs.padding[1], PW = attrs.padding[2];
    auto [OD, OH, OW] = dwconv_out_3d(ID, IH, IW, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW);

    // Random NCDHW input + weight
    auto [in_vec, in_ncdhw] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_ncdhw]  = test::make_random_tensor(w_shape);
    std::vector<float> b_vec;
    TensorView b_nchw;
    if (has_bias) {
        auto p = test::make_random_tensor({C});
        b_vec = std::move(p.first);
        b_nchw = p.second;
    }

    // NCDHW scalar reference (golden)
    std::vector<float> ref_out(static_cast<size_t>(N * C * OD * OH * OW));
    const std::vector<int64_t> oshape_ref = {N, C, OD, OH, OW};
    TensorView out_ref(oshape_ref, DataType::f32, ref_out.data(), TensorLayout::NCDHW);
    if (attrs.add_to) {
        for (auto& v : ref_out) v = 1.0f;
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

    // Pack input to NCDHWC8
    int64_t in_pitch = nchwc8_pitch(IW);
    int64_t in_pitch_elems = in_pitch / 4;
    int64_t in_d_elems = IH * in_pitch_elems;
    std::vector<float> packed_in(static_cast<size_t>(N * C8 * ID * in_d_elems));
    TensorView in_c8(in_shape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCDHWC8);
    pack_ncdhw_to_ncdhwc8(in_ncdhw, in_c8);

    // Prepack weight + bias
    std::vector<float> pw_buf, pb_buf;
    TensorView pw_view, pb_view;
    prepack_dwconv_weight_3d(w_ncdhw, pw_buf, pw_view,
                               has_bias, has_bias ? &b_nchw : nullptr,
                               pb_buf, pb_view);

    // NCDHWC8 output
    int64_t out_pitch = nchwc8_pitch(OW);
    int64_t out_pitch_elems = out_pitch / 4;
    int64_t out_d_elems = OH * out_pitch_elems;
    std::vector<float> packed_out(static_cast<size_t>(N * C8 * OD * out_d_elems));
    TensorView out_c8(oshape_ref, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCDHWC8);

    // For add_to: pre-fill output with 1.0
    if (attrs.add_to) {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c8i = 0; c8i < C8; ++c8i) {
                for (int64_t od = 0; od < OD; ++od) {
                    for (int64_t oh = 0; oh < OH; ++oh) {
                        float* row = packed_out.data() + ((n * C8 + c8i) * OD + od) * out_d_elems + oh * out_pitch_elems;
                        for (int64_t ow = 0; ow < OW; ++ow)
                            for (int64_t l = 0; l < 8; ++l)
                                row[ow * 8 + l] = 1.0f;
                    }
                }
            }
        }
    }

    // Run backend
    {
        ComputeContext ctx;
        if (has_bias) {
            const TensorView ins_arr[] = {in_c8, pw_view, pb_view};
            backend::cpu::depthwise_conv_cpu(attrs, out_c8, ins_arr, ctx, nullptr);
        } else {
            const TensorView ins_arr[] = {in_c8, pw_view};
            backend::cpu::depthwise_conv_cpu(attrs, out_c8, ins_arr, ctx, nullptr);
        }
    }

    // Unpack and compare
    std::vector<float> result(static_cast<size_t>(N * C * OD * OH * OW));
    TensorView res_ncdhw(oshape_ref, DataType::f32, result.data(), TensorLayout::NCDHW);
    unpack_ncdhwc8_to_ncdhw(out_c8, res_ncdhw);
    NNOPS_EXPECT_TRUE(test::allclose(res_ncdhw, out_ref, 1e-4f, 1e-4f));
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
    const int64_t C = 2, C8 = 1, ID = 3, IH = 3, IW = 3;
    const int64_t KD = 2, KH = 2, KW = 2;
    const int64_t OD = 2, OH = 2, OW = 2;  // (3-2)/1+1 = 2

    std::vector<float> in_buf(54, 1.0f);   // 1*2*3*3*3
    std::vector<float> w_buf(16, 0.5f);     // 2*1*2*2*2

    TensorView in_ncdhw(ishape, DataType::f32, in_buf.data(), TensorLayout::NCDHW);
    TensorView w_ncdhw(wshape, DataType::f32, w_buf.data(), TensorLayout::NCDHW);

    // NCDHW reference
    std::vector<float> ref_buf(16);  // 1*2*2*2*2
    const int64_t oshape[] = {1, 2, OD, OH, OW};
    TensorView out_ref(oshape, DataType::f32, ref_buf.data(), TensorLayout::NCDHW);
    {
        ComputeContext ctx;
        DepthwiseConvAttributes attrs;
        attrs.kernel_size = {KD, KH, KW};
        attrs.stride = {1, 1, 1};
        attrs.padding = {0, 0, 0};
        const TensorView ref_arr[] = {in_ncdhw, w_ncdhw};
        backend::cpu::reference::depthwise_conv_ref(attrs, out_ref, ref_arr, ctx, nullptr);
    }

    // Pack input → NCDHWC8
    int64_t in_pitch = nchwc8_pitch(IW);
    int64_t in_pitch_elems = in_pitch / 4;
    int64_t in_d_elems = IH * in_pitch_elems;
    std::vector<float> packed_in(static_cast<size_t>(C8 * ID * in_d_elems));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCDHWC8);
    pack_ncdhw_to_ncdhwc8(in_ncdhw, in_c8);

    // Prepack weight
    std::vector<float> pw_buf(static_cast<size_t>(C8 * KD * KH * KW * 8));
    const int64_t pw_shape[] = {C8, KD, KH, KW, 8};
    TensorView pw_view(std::span<const int64_t>(pw_shape, 5), DataType::f32,
                        pw_buf.data(), TensorLayout::PackedWeight);
    {
        auto op = DepthwiseConv::create(DepthwiseConvAttributes{});
        const TensorView w_arr[] = {w_ncdhw};
        TensorView pw_arr[] = {pw_view};
        op->prepackWeights(w_arr, pw_arr);
    }

    // Run
    int64_t out_pitch = nchwc8_pitch(OW);
    int64_t out_pitch_elems = out_pitch / 4;
    int64_t out_d_elems = OH * out_pitch_elems;
    std::vector<float> packed_out(static_cast<size_t>(C8 * OD * out_d_elems));
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCDHWC8);

    DepthwiseConvAttributes attrs;
    attrs.kernel_size = {KD, KH, KW};
    attrs.stride = {1, 1, 1};
    attrs.padding = {0, 0, 0};

    {
        ComputeContext ctx;
        const TensorView ins[] = {in_c8, pw_view};
        backend::cpu::depthwise_conv_cpu(attrs, out_c8, ins, ctx, nullptr);
    }

    // Unpack
    std::vector<float> result(16);
    TensorView res_ncdhw(oshape, DataType::f32, result.data(), TensorLayout::NCDHW);
    unpack_ncdhwc8_to_ncdhw(out_c8, res_ncdhw);

    for (int i = 0; i < 16; ++i) {
        NNOPS_EXPECT_NEAR(result[i], 4.0f, 1e-4f);
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
