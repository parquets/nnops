/// @file test_transpose_conv2d.cpp
/// @brief TransposeConv2D correctness tests on NHWC activations.
///
/// Every case round-trips: the input is built in the reference's dense order,
/// moved to NHWC, computed, moved back, then compared. Weight stays OIHW
/// [IC, OC/G, KH, KW] (no prepacking), so one weight tensor feeds both the op
/// and the scalar golden.
///
/// Two kinds of case: hand-computed small shapes that pin exact output values
/// and output geometry, and random roundtrips against
/// reference::transpose_conv2d_channel_first_ref (tests/reference).
///
/// Input:  [N, IH, IW, IC] (channel-last)
/// Output: [N, OH, OW, OC]

#include "nnops/ops/transpose_conv2d.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/channel_last.hpp"
#include "common/cpu_pool.hpp"

#include <atomic>
#include <span>
#include <thread>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>

using namespace nnops;

// Defined in tests/reference/; the scalar golden, in dense order, used for cross-checks.
namespace nnops::backend::cpu::reference {
    void transpose_conv2d_channel_first_ref(const TransposeConv2DAttributes& attrs,
                                    TensorView& output,
                                    std::span<const TensorView> inputs,
                                    const ComputeContext& ctx,
                                    void* workspace);
}

namespace {

/// Channel-last [N, spatial..., C] dims for a dense [N, C, spatial...] view.
std::vector<int64_t> cl_dims_of(const TensorView& cf)
{
    const int64_t rank = cf.rank();
    std::vector<int64_t> d(static_cast<size_t>(rank));
    d[0] = cf.shape(0);
    for (int64_t i = 2; i < rank; ++i) { d[static_cast<size_t>(i - 1)] = cf.shape(i); }
    d[static_cast<size_t>(rank - 1)] = cf.shape(1);
    return d;
}

/// Dense [N, C, spatial...] dims for a channel-last view.
std::vector<int64_t> dense_dims_of(const TensorView& cl)
{
    const int64_t rank = cl.rank();
    std::vector<int64_t> d(static_cast<size_t>(rank));
    d[0] = cl.shape(0);
    d[1] = cl.shape(rank - 1);
    for (int64_t i = 1; i + 1 < rank; ++i) { d[static_cast<size_t>(i + 1)] = cl.shape(i); }
    return d;
}

/// The channel-last output desc for a dense input: shape inference speaks
/// channel-last whatever the storage.
TensorDesc cl_out_desc_of(const TensorView& in_cf, const TensorView& w,
                          const TensorView* b,
                          const TransposeConv2DAttributes& attrs)
{
    auto op = TransposeConv2D::create(attrs, Backend::CPU);
    const std::vector<int64_t> cl = cl_dims_of(in_cf);
    TensorDesc in_cl = in_cf.desc();
    in_cl.dims = std::span<const int64_t>(cl.data(), cl.size());
    in_cl.layout = (in_cf.rank() == 5) ? TensorLayout::NDHWC : TensorLayout::NHWC;
    std::vector<TensorDesc> ds = {in_cl, w.desc()};
    if (b != nullptr) { ds.push_back(b->desc()); }
    return op->getOutputTensorDesc(ds)[0];
}

/// Dense [N, C, OH, OW] desc built from a channel-last one.
TensorDesc dense_desc_of(const TensorDesc& cl)
{
    const int64_t rank = cl.rank;
    TensorDesc d;
    d.rank = rank;
    d.dtype = cl.dtype;
    d.layout = TensorLayout::Dense;
    d.dims.resize(static_cast<size_t>(rank));
    d.dims[0] = cl.dims[0];
    d.dims[1] = cl.dims[static_cast<size_t>(rank - 1)];
    for (int64_t i = 1; i + 1 < rank; ++i) {
        d.dims[static_cast<size_t>(i + 1)] = cl.dims[static_cast<size_t>(i)];
    }
    return d;
}

}  // namespace

/// Roundtrip: convert input to NHWC -> TransposeConv2D::compute -> convert back
/// -> compare with the scalar golden.
static void test_vs_ref(
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

    // The reference keeps the dense order: the query is channel-last, the
    // buffer it writes is [N, C, OH, OW].
    const TensorDesc ref_desc =
        dense_desc_of(cl_out_desc_of(in_channel_first, w_channel_first,
                                     has_bias ? &b_channel_first : nullptr, attrs));

    std::vector<float> ref_buf(static_cast<size_t>(ref_desc.numel()));
    auto out_ref = test::make_channel_first(ref_desc, ref_buf.data());
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

    std::vector<float> cl_in_buf = test::to_channel_last(
        std::span<const float>(in_channel_first.ptr<float>(),
                               static_cast<size_t>(in_channel_first.numel())),
        in_channel_first.shape(0), in_channel_first.shape(1),
        in_channel_first.numel() / (in_channel_first.shape(0) * in_channel_first.shape(1)));
    const std::vector<int64_t> in_cl_dims = cl_dims_of(in_channel_first);
    TensorView in_cl(in_cl_dims, DataType::f32, cl_in_buf.data(),
                     TensorLayout::NHWC);

    (void)w_vec;  // w_channel_first references it

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    TensorDesc in_cl_desc = in_cl.desc();
    std::vector<TensorDesc> out_descs;
    if (has_bias) {
        const TensorDesc tc_desc_arr[] = {in_cl_desc, w_channel_first.desc(),
                                          b_channel_first.desc()};
        out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);
    } else {
        const TensorDesc tc_desc_arr[] = {in_cl_desc, w_channel_first.desc()};
        out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);
    }

    std::vector<float> cl_out_buf(static_cast<size_t>(out_descs[0].numel()));
    TensorView out_cl(std::span<const int64_t>(out_descs[0].dims.data(),
                                               static_cast<size_t>(out_descs[0].rank)),
                      DataType::f32, cl_out_buf.data(), TensorLayout::NHWC);

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

    std::vector<float> result_buf = test::to_channel_first(
        std::span<const float>(out_cl.ptr<float>(), static_cast<size_t>(out_cl.numel())),
        out_cl.shape(0), out_cl.shape(out_cl.rank() - 1),
        out_cl.numel() / (out_cl.shape(0) * out_cl.shape(out_cl.rank() - 1)));
    const std::vector<int64_t> result_dims = dense_dims_of(out_cl);
    TensorView result(result_dims, DataType::f32, result_buf.data(),
                      TensorLayout::Dense);

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

    TensorView in_channel_first(ishape, DataType::f32, in_buf.data(), TensorLayout::Dense);
    TensorView w_channel_first(wshape, DataType::f32, w_buf.data(), TensorLayout::Dense);

    std::vector<float> cl_in_buf = test::to_channel_last(
        std::span<const float>(in_channel_first.ptr<float>(),
                               static_cast<size_t>(in_channel_first.numel())),
        in_channel_first.shape(0), in_channel_first.shape(1),
        in_channel_first.numel() / (in_channel_first.shape(0) * in_channel_first.shape(1)));
    const std::vector<int64_t> in_cl_dims = cl_dims_of(in_channel_first);
    TensorView in_cl(in_cl_dims, DataType::f32, cl_in_buf.data(),
                     TensorLayout::NHWC);

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

    // [N, OH, OW, OC] = [1, 2, 2, 1]
    NNOPS_EXPECT_EQ(out_descs[0].dims[0], 1);
    NNOPS_EXPECT_EQ(out_descs[0].dims[1], 2);
    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 2);
    NNOPS_EXPECT_EQ(out_descs[0].dims[3], 1);

    std::vector<float> cl_out_buf(static_cast<size_t>(out_descs[0].numel()));
    TensorView out_cl(std::span<const int64_t>(out_descs[0].dims.data(),
                                               static_cast<size_t>(out_descs[0].rank)),
                      DataType::f32, cl_out_buf.data(), TensorLayout::NHWC);

    {
        const TensorView ins[] = {in_cl, w_channel_first};
        TensorView outs[] = {out_cl};
        tconv_op->compute(outs, ins);
    }

    std::vector<float> result_buf = test::to_channel_first(
        std::span<const float>(out_cl.ptr<float>(), static_cast<size_t>(out_cl.numel())),
        out_cl.shape(0), out_cl.shape(out_cl.rank() - 1),
        out_cl.numel() / (out_cl.shape(0) * out_cl.shape(out_cl.rank() - 1)));
    const std::vector<int64_t> result_dims = dense_dims_of(out_cl);
    TensorView result(result_dims, DataType::f32, result_buf.data(),
                      TensorLayout::Dense);

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

    TensorView in_channel_first(ishape, DataType::f32, in_buf.data(), TensorLayout::Dense);
    TensorView w_channel_first(wshape, DataType::f32, w_buf.data(), TensorLayout::Dense);

    std::vector<float> cl_in_buf = test::to_channel_last(
        std::span<const float>(in_channel_first.ptr<float>(),
                               static_cast<size_t>(in_channel_first.numel())),
        in_channel_first.shape(0), in_channel_first.shape(1),
        in_channel_first.numel() / (in_channel_first.shape(0) * in_channel_first.shape(1)));
    const std::vector<int64_t> in_cl_dims = cl_dims_of(in_channel_first);
    TensorView in_cl(in_cl_dims, DataType::f32, cl_in_buf.data(),
                     TensorLayout::NHWC);

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

    // [N, OH, OW, OC] = [1, 3, 3, 1]
    NNOPS_EXPECT_EQ(out_descs[0].dims[1], 3);
    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 3);
    NNOPS_EXPECT_EQ(out_descs[0].dims[3], 1);

    std::vector<float> cl_out_buf(static_cast<size_t>(out_descs[0].numel()));
    TensorView out_cl(std::span<const int64_t>(out_descs[0].dims.data(),
                                               static_cast<size_t>(out_descs[0].rank)),
                      DataType::f32, cl_out_buf.data(), TensorLayout::NHWC);
    {
        const TensorView ins[] = {in_cl, w_channel_first};
        TensorView outs[] = {out_cl};
        tconv_op->compute(outs, ins);
    }

    std::vector<float> result_buf = test::to_channel_first(
        std::span<const float>(out_cl.ptr<float>(), static_cast<size_t>(out_cl.numel())),
        out_cl.shape(0), out_cl.shape(out_cl.rank() - 1),
        out_cl.numel() / (out_cl.shape(0) * out_cl.shape(out_cl.rank() - 1)));
    const std::vector<int64_t> result_dims = dense_dims_of(out_cl);
    TensorView result(result_dims, DataType::f32, result_buf.data(),
                      TensorLayout::Dense);

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

    TensorView in_channel_first(ishape, DataType::f32, in_buf.data(), TensorLayout::Dense);
    TensorView w_channel_first(wshape, DataType::f32, w_buf.data(), TensorLayout::Dense);

    std::vector<float> cl_in_buf = test::to_channel_last(
        std::span<const float>(in_channel_first.ptr<float>(),
                               static_cast<size_t>(in_channel_first.numel())),
        in_channel_first.shape(0), in_channel_first.shape(1),
        in_channel_first.numel() / (in_channel_first.shape(0) * in_channel_first.shape(1)));
    const std::vector<int64_t> in_cl_dims = cl_dims_of(in_channel_first);
    TensorView in_cl(in_cl_dims, DataType::f32, cl_in_buf.data(),
                     TensorLayout::NHWC);

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {2, 2};
    attrs.stride      = {2, 2};
    attrs.dilation    = {1, 1};
    attrs.padding     = {0, 0};

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    const TensorDesc tc_desc_arr[] = {in_cl.desc(), w_channel_first.desc()};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    NNOPS_EXPECT_EQ(out_descs[0].dims[1], 4);
    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 4);

    std::vector<float> cl_out_buf(static_cast<size_t>(out_descs[0].numel()));
    TensorView out_cl(std::span<const int64_t>(out_descs[0].dims.data(),
                                               static_cast<size_t>(out_descs[0].rank)),
                      DataType::f32, cl_out_buf.data(), TensorLayout::NHWC);
    {
        const TensorView ins[] = {in_cl, w_channel_first};
        TensorView outs[] = {out_cl};
        tconv_op->compute(outs, ins);
    }

    std::vector<float> result_buf = test::to_channel_first(
        std::span<const float>(out_cl.ptr<float>(), static_cast<size_t>(out_cl.numel())),
        out_cl.shape(0), out_cl.shape(out_cl.rank() - 1),
        out_cl.numel() / (out_cl.shape(0) * out_cl.shape(out_cl.rank() - 1)));
    const std::vector<int64_t> result_dims = dense_dims_of(out_cl);
    TensorView result(result_dims, DataType::f32, result_buf.data(),
                      TensorLayout::Dense);

    auto tconv_op_ref = TransposeConv2D::create(attrs, Backend::CPU);
    const TensorDesc ref_desc =
        dense_desc_of(cl_out_desc_of(in_channel_first, w_channel_first, nullptr, attrs));

    std::vector<float> ref_buf(static_cast<size_t>(ref_desc.numel()));
    auto out_ref = test::make_channel_first(ref_desc, ref_buf.data());
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

    TensorView in_channel_first(ishape, DataType::f32, in_buf.data(), TensorLayout::Dense);
    TensorView w_channel_first(wshape, DataType::f32, w_buf.data(), TensorLayout::Dense);

    std::vector<float> cl_in_buf = test::to_channel_last(
        std::span<const float>(in_channel_first.ptr<float>(),
                               static_cast<size_t>(in_channel_first.numel())),
        in_channel_first.shape(0), in_channel_first.shape(1),
        in_channel_first.numel() / (in_channel_first.shape(0) * in_channel_first.shape(1)));
    const std::vector<int64_t> in_cl_dims = cl_dims_of(in_channel_first);
    TensorView in_cl(in_cl_dims, DataType::f32, cl_in_buf.data(),
                     TensorLayout::NHWC);

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

    std::vector<float> cl_out_buf(static_cast<size_t>(out_descs[0].numel()));
    TensorView out_cl(std::span<const int64_t>(out_descs[0].dims.data(),
                                               static_cast<size_t>(out_descs[0].rank)),
                      DataType::f32, cl_out_buf.data(), TensorLayout::NHWC);
    {
        const TensorView ins[] = {in_cl, w_channel_first};
        TensorView outs[] = {out_cl};
        tconv_op->compute(outs, ins);
    }

    std::vector<float> result_buf = test::to_channel_first(
        std::span<const float>(out_cl.ptr<float>(), static_cast<size_t>(out_cl.numel())),
        out_cl.shape(0), out_cl.shape(out_cl.rank() - 1),
        out_cl.numel() / (out_cl.shape(0) * out_cl.shape(out_cl.rank() - 1)));
    const std::vector<int64_t> result_dims = dense_dims_of(out_cl);
    TensorView result(result_dims, DataType::f32, result_buf.data(),
                      TensorLayout::Dense);

    // input[0][0] maps to (oh, ow) = (-1, -1) and is clipped; only input[1][1]
    // survives, at (0, 0): 4 * 2 = 8.
    NNOPS_EXPECT_NEAR(result_buf[0], 8.0f, 1e-4f);

    const TensorDesc ref_desc =
        dense_desc_of(cl_out_desc_of(in_channel_first, w_channel_first, nullptr, attrs));
    std::vector<float> ref_buf(static_cast<size_t>(ref_desc.numel()));
    auto out_ref = test::make_channel_first(ref_desc, ref_buf.data());
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

    TensorView in_channel_first(ishape, DataType::f32, in_buf.data(), TensorLayout::Dense);
    TensorView w_channel_first(wshape, DataType::f32, w_buf.data(), TensorLayout::Dense);
    TensorView b_channel_first(bshape, DataType::f32, b_buf.data(), TensorLayout::Dense);

    std::vector<float> cl_in_buf = test::to_channel_last(
        std::span<const float>(in_channel_first.ptr<float>(),
                               static_cast<size_t>(in_channel_first.numel())),
        in_channel_first.shape(0), in_channel_first.shape(1),
        in_channel_first.numel() / (in_channel_first.shape(0) * in_channel_first.shape(1)));
    const std::vector<int64_t> in_cl_dims = cl_dims_of(in_channel_first);
    TensorView in_cl(in_cl_dims, DataType::f32, cl_in_buf.data(),
                     TensorLayout::NHWC);

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {1, 1};
    attrs.stride      = {1, 1};
    attrs.dilation    = {1, 1};
    attrs.padding     = {0, 0};

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    const TensorDesc tc_desc_arr[] = {in_cl.desc(), w_channel_first.desc(), b_channel_first.desc()};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    NNOPS_EXPECT_EQ(out_descs[0].dims[1], 2);
    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 2);

    std::vector<float> cl_out_buf(static_cast<size_t>(out_descs[0].numel()));
    TensorView out_cl(std::span<const int64_t>(out_descs[0].dims.data(),
                                               static_cast<size_t>(out_descs[0].rank)),
                      DataType::f32, cl_out_buf.data(), TensorLayout::NHWC);
    {
        const TensorView ins[] = {in_cl, w_channel_first, b_channel_first};
        TensorView outs[] = {out_cl};
        tconv_op->compute(outs, ins);
    }

    std::vector<float> result_buf = test::to_channel_first(
        std::span<const float>(out_cl.ptr<float>(), static_cast<size_t>(out_cl.numel())),
        out_cl.shape(0), out_cl.shape(out_cl.rank() - 1),
        out_cl.numel() / (out_cl.shape(0) * out_cl.shape(out_cl.rank() - 1)));
    const std::vector<int64_t> result_dims = dense_dims_of(out_cl);
    TensorView result(result_dims, DataType::f32, result_buf.data(),
                      TensorLayout::Dense);

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

    TensorView in_channel_first(ishape, DataType::f32, in_buf.data(), TensorLayout::Dense);
    TensorView w_channel_first(wshape, DataType::f32, w_buf.data(), TensorLayout::Dense);
    const int64_t bshape[] = {1};
    TensorView b_channel_first(bshape, DataType::f32, b_buf.data(), TensorLayout::Dense);

    std::vector<float> cl_in_buf = test::to_channel_last(
        std::span<const float>(in_channel_first.ptr<float>(),
                               static_cast<size_t>(in_channel_first.numel())),
        in_channel_first.shape(0), in_channel_first.shape(1),
        in_channel_first.numel() / (in_channel_first.shape(0) * in_channel_first.shape(1)));
    const std::vector<int64_t> in_cl_dims = cl_dims_of(in_channel_first);
    TensorView in_cl(in_cl_dims, DataType::f32, cl_in_buf.data(),
                     TensorLayout::NHWC);

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {1, 1};
    attrs.stride      = {1, 1};
    attrs.dilation    = {1, 1};
    attrs.padding     = {0, 0};
    attrs.epilogue.type = EpilogueActivateType::Relu;

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    const TensorDesc tc_desc_arr[] = {in_cl.desc(), w_channel_first.desc(), b_channel_first.desc()};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    std::vector<float> cl_out_buf(static_cast<size_t>(out_descs[0].numel()));
    TensorView out_cl(std::span<const int64_t>(out_descs[0].dims.data(),
                                               static_cast<size_t>(out_descs[0].rank)),
                      DataType::f32, cl_out_buf.data(), TensorLayout::NHWC);
    {
        const TensorView ins[] = {in_cl, w_channel_first, b_channel_first};
        TensorView outs[] = {out_cl};
        tconv_op->compute(outs, ins);
    }

    std::vector<float> result_buf = test::to_channel_first(
        std::span<const float>(out_cl.ptr<float>(), static_cast<size_t>(out_cl.numel())),
        out_cl.shape(0), out_cl.shape(out_cl.rank() - 1),
        out_cl.numel() / (out_cl.shape(0) * out_cl.shape(out_cl.rank() - 1)));
    const std::vector<int64_t> result_dims = dense_dims_of(out_cl);
    TensorView result(result_dims, DataType::f32, result_buf.data(),
                      TensorLayout::Dense);

    NNOPS_EXPECT_NEAR(result_buf[0], 0.0f, 1e-4f);
}

NNOPS_TEST(tconv2d_groups) {
    // groups=2 with IC=OC=2: channel c sees weight[c] alone, so
    // 10.0 * 2.0 = 20.0 and 20.0 * 3.0 = 60.0.
    const int64_t ishape[] = {1, 2, 1, 1};
    const int64_t wshape[] = {2, 1, 1, 1};

    std::vector<float> in_buf = {10.0f, 20.0f};
    std::vector<float> w_buf  = {2.0f, 3.0f};

    TensorView in_channel_first(ishape, DataType::f32, in_buf.data(), TensorLayout::Dense);
    TensorView w_channel_first(wshape, DataType::f32, w_buf.data(), TensorLayout::Dense);

    std::vector<float> cl_in_buf = test::to_channel_last(
        std::span<const float>(in_channel_first.ptr<float>(),
                               static_cast<size_t>(in_channel_first.numel())),
        in_channel_first.shape(0), in_channel_first.shape(1),
        in_channel_first.numel() / (in_channel_first.shape(0) * in_channel_first.shape(1)));
    const std::vector<int64_t> in_cl_dims = cl_dims_of(in_channel_first);
    TensorView in_cl(in_cl_dims, DataType::f32, cl_in_buf.data(),
                     TensorLayout::NHWC);

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {1, 1};
    attrs.stride      = {1, 1};
    attrs.dilation    = {1, 1};
    attrs.padding     = {0, 0};
    attrs.groups      = 2;

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    const TensorDesc tc_desc_arr[] = {in_cl.desc(), w_channel_first.desc()};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    NNOPS_EXPECT_EQ(out_descs[0].dims[3], 2);  // OC = 2, the last axis

    std::vector<float> cl_out_buf(static_cast<size_t>(out_descs[0].numel()));
    TensorView out_cl(std::span<const int64_t>(out_descs[0].dims.data(),
                                               static_cast<size_t>(out_descs[0].rank)),
                      DataType::f32, cl_out_buf.data(), TensorLayout::NHWC);
    {
        const TensorView ins[] = {in_cl, w_channel_first};
        TensorView outs[] = {out_cl};
        tconv_op->compute(outs, ins);
    }

    std::vector<float> result_buf = test::to_channel_first(
        std::span<const float>(out_cl.ptr<float>(), static_cast<size_t>(out_cl.numel())),
        out_cl.shape(0), out_cl.shape(out_cl.rank() - 1),
        out_cl.numel() / (out_cl.shape(0) * out_cl.shape(out_cl.rank() - 1)));
    const std::vector<int64_t> result_dims = dense_dims_of(out_cl);
    TensorView result(result_dims, DataType::f32, result_buf.data(),
                      TensorLayout::Dense);

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

    TensorView in_channel_first(ishape, DataType::f32, in_buf.data(), TensorLayout::Dense);
    TensorView w_channel_first(wshape, DataType::f32, w_buf.data(), TensorLayout::Dense);

    std::vector<float> cl_in_buf = test::to_channel_last(
        std::span<const float>(in_channel_first.ptr<float>(),
                               static_cast<size_t>(in_channel_first.numel())),
        in_channel_first.shape(0), in_channel_first.shape(1),
        in_channel_first.numel() / (in_channel_first.shape(0) * in_channel_first.shape(1)));
    const std::vector<int64_t> in_cl_dims = cl_dims_of(in_channel_first);
    TensorView in_cl(in_cl_dims, DataType::f32, cl_in_buf.data(),
                     TensorLayout::NHWC);

    TransposeConv2DAttributes attrs;
    attrs.kernel_size   = {1, 1};
    attrs.stride        = {2, 2};
    attrs.dilation      = {1, 1};
    attrs.padding       = {0, 0};
    attrs.output_padding = {1, 1};

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    const TensorDesc tc_desc_arr[] = {in_cl.desc(), w_channel_first.desc()};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    NNOPS_EXPECT_EQ(out_descs[0].dims[1], 4);
    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 4);

    const TensorDesc ref_desc =
        dense_desc_of(cl_out_desc_of(in_channel_first, w_channel_first, nullptr, attrs));
    std::vector<float> ref_buf(static_cast<size_t>(ref_desc.numel()));
    auto out_ref = test::make_channel_first(ref_desc, ref_buf.data());
    {
        ComputeContext ctx;
        const TensorView ref_arr[] = {in_channel_first, w_channel_first};
        backend::cpu::reference::transpose_conv2d_channel_first_ref(
            attrs, out_ref, ref_arr, ctx, nullptr);
    }

    std::vector<float> cl_out_buf(static_cast<size_t>(out_descs[0].numel()));
    TensorView out_cl(std::span<const int64_t>(out_descs[0].dims.data(),
                                               static_cast<size_t>(out_descs[0].rank)),
                      DataType::f32, cl_out_buf.data(), TensorLayout::NHWC);
    {
        const TensorView ins[] = {in_cl, w_channel_first};
        TensorView outs[] = {out_cl};
        tconv_op->compute(outs, ins);
    }

    std::vector<float> result_buf = test::to_channel_first(
        std::span<const float>(out_cl.ptr<float>(), static_cast<size_t>(out_cl.numel())),
        out_cl.shape(0), out_cl.shape(out_cl.rank() - 1),
        out_cl.numel() / (out_cl.shape(0) * out_cl.shape(out_cl.rank() - 1)));
    const std::vector<int64_t> result_dims = dense_dims_of(out_cl);
    TensorView result(result_dims, DataType::f32, result_buf.data(),
                      TensorLayout::Dense);

    NNOPS_EXPECT_TRUE(test::allclose(result, out_ref, 1e-4f, 1e-4f));
}

NNOPS_TEST(tconv2d_dilation) {
    // Dilation=2 spaces the 2x2 kernel's taps 2 apart, so a 3x3 input gives a
    // 5x5 output: OH = OW = (3-1)*1 + 2*(2-1) + 1 = 5.
    const int64_t ishape[] = {1, 1, 3, 3};
    const int64_t wshape[] = {1, 1, 2, 2};

    std::vector<float> in_buf(9, 1.0f);
    std::vector<float> w_buf  = {5.0f, 4.0f, 3.0f, 2.0f};

    TensorView in_channel_first(ishape, DataType::f32, in_buf.data(), TensorLayout::Dense);
    TensorView w_channel_first(wshape, DataType::f32, w_buf.data(), TensorLayout::Dense);

    std::vector<float> cl_in_buf = test::to_channel_last(
        std::span<const float>(in_channel_first.ptr<float>(),
                               static_cast<size_t>(in_channel_first.numel())),
        in_channel_first.shape(0), in_channel_first.shape(1),
        in_channel_first.numel() / (in_channel_first.shape(0) * in_channel_first.shape(1)));
    const std::vector<int64_t> in_cl_dims = cl_dims_of(in_channel_first);
    TensorView in_cl(in_cl_dims, DataType::f32, cl_in_buf.data(),
                     TensorLayout::NHWC);

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {2, 2};
    attrs.stride      = {1, 1};
    attrs.dilation    = {2, 2};
    attrs.padding     = {0, 0};

    auto tconv_op = TransposeConv2D::create(attrs, Backend::CPU);

    const TensorDesc tc_desc_arr[] = {in_cl.desc(), w_channel_first.desc()};
    auto out_descs = tconv_op->getOutputTensorDesc(tc_desc_arr);

    NNOPS_EXPECT_EQ(out_descs[0].dims[1], 5);
    NNOPS_EXPECT_EQ(out_descs[0].dims[2], 5);

    std::vector<float> cl_out_buf(static_cast<size_t>(out_descs[0].numel()));
    TensorView out_cl(std::span<const int64_t>(out_descs[0].dims.data(),
                                               static_cast<size_t>(out_descs[0].rank)),
                      DataType::f32, cl_out_buf.data(), TensorLayout::NHWC);
    {
        const TensorView ins[] = {in_cl, w_channel_first};
        TensorView outs[] = {out_cl};
        tconv_op->compute(outs, ins);
    }

    std::vector<float> result_buf = test::to_channel_first(
        std::span<const float>(out_cl.ptr<float>(), static_cast<size_t>(out_cl.numel())),
        out_cl.shape(0), out_cl.shape(out_cl.rank() - 1),
        out_cl.numel() / (out_cl.shape(0) * out_cl.shape(out_cl.rank() - 1)));
    const std::vector<int64_t> result_dims = dense_dims_of(out_cl);
    TensorView result(result_dims, DataType::f32, result_buf.data(),
                      TensorLayout::Dense);

    const TensorDesc ref_desc =
        dense_desc_of(cl_out_desc_of(in_channel_first, w_channel_first, nullptr, attrs));
    std::vector<float> ref_buf(static_cast<size_t>(ref_desc.numel()));
    auto out_ref = test::make_channel_first(ref_desc, ref_buf.data());
    {
        ComputeContext ctx;
        const TensorView ref_arr[] = {in_channel_first, w_channel_first};
        backend::cpu::reference::transpose_conv2d_channel_first_ref(
            attrs, out_ref, ref_arr, ctx, nullptr);
    }

    NNOPS_EXPECT_TRUE(test::allclose(result, out_ref, 1e-4f, 1e-4f));
}

// ============================================================
// Roundtrip tests (random data, compare with the scalar golden)
// ============================================================

NNOPS_TEST(tconv2d_roundtrip_f32) {
    TransposeConv2DAttributes attrs;
    attrs.kernel_size  = {3, 3};
    attrs.stride       = {2, 2};
    attrs.dilation     = {1, 1};
    attrs.padding      = {1, 1};
    attrs.groups       = 1;

    test_vs_ref({1, 4, 8, 8}, {4, 8, 3, 3}, attrs);
}

NNOPS_TEST(tconv2d_roundtrip_f32_bias) {
    TransposeConv2DAttributes attrs;
    attrs.kernel_size  = {3, 3};
    attrs.stride       = {2, 2};
    attrs.dilation     = {1, 1};
    attrs.padding      = {1, 1};
    attrs.groups       = 1;

    test_vs_ref({1, 4, 8, 8}, {4, 8, 3, 3}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(tconv2d_roundtrip_f32_groups) {
    TransposeConv2DAttributes attrs;
    attrs.kernel_size  = {3, 3};
    attrs.stride       = {1, 1};
    attrs.dilation     = {1, 1};
    attrs.padding      = {0, 0};
    attrs.groups       = 2;

    test_vs_ref({1, 4, 4, 4}, {4, 4, 3, 3}, attrs);
}

NNOPS_TEST(tconv2d_roundtrip_f32_nopad) {
    TransposeConv2DAttributes attrs;
    attrs.kernel_size  = {2, 2};
    attrs.stride       = {1, 1};
    attrs.dilation     = {1, 1};
    attrs.padding      = {0, 0};
    attrs.groups       = 1;

    test_vs_ref({1, 4, 4, 4}, {4, 6, 2, 2}, attrs);
}

NNOPS_TEST(tconv2d_roundtrip_f32_dilation) {
    TransposeConv2DAttributes attrs;
    attrs.kernel_size  = {3, 3};
    attrs.stride       = {1, 1};
    attrs.dilation     = {2, 2};
    attrs.padding      = {0, 0};
    attrs.groups       = 1;

    test_vs_ref({1, 4, 8, 8}, {4, 4, 3, 3}, attrs);
}

NNOPS_TEST(tconv2d_roundtrip_f32_multi_channel) {
    // > 8 channels to test channel packing across multiple C8 groups
    TransposeConv2DAttributes attrs;
    attrs.kernel_size  = {3, 3};
    attrs.stride       = {2, 2};
    attrs.dilation     = {1, 1};
    attrs.padding      = {1, 1};
    attrs.groups       = 1;

    test_vs_ref({1, 16, 8, 8}, {16, 32, 3, 3}, attrs);
}

/// Roundtrip against the scalar golden with a mixed-sign
/// residual in the destination, so that the pre- and post-activation orders of
/// add_to cannot agree by accident.
static void test_residual_vs_ref(
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
    // Channel-last query: [N, OH, OW, OC]; the reference reads dense order.
    const TensorDesc cl_out = cl_out_desc_of(in_cf, w_cf,
                                             has_bias ? &b_cf : nullptr, attrs);
    const TensorDesc dense_out = dense_desc_of(cl_out);
    const int64_t on  = cl_out.dims[0];
    const int64_t oh  = cl_out.dims[1];
    const int64_t ow  = cl_out.dims[2];
    const int64_t och = cl_out.dims[3];

    auto [res_vec, res_view] = test::make_random_tensor(
        {on, och, oh, ow}, -1.0f, 1.0f, 11);
    (void)res_view;

    std::vector<float> ref_buf(res_vec);
    auto out_ref = test::make_channel_first(dense_out, ref_buf.data());
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

    const std::vector<int64_t> ishape = {on, IH, IW, IC};
    const std::vector<int64_t> oshape = {on, oh, ow, och};
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
    const std::vector<int64_t> got_shape = {on, och, oh, ow};
    TensorView got_view(got_shape, DataType::f32, got.data(), TensorLayout::Dense);
    NNOPS_EXPECT_TRUE(test::allclose(got_view, out_ref, 1e-4f, 1e-4f));
}

// ============================================================
// f16
// ============================================================

/// f16 against an f32 golden computed from the same values *after* their f16
/// rounding, so the comparison prices the kernel's summation order rather than
/// the operand quantisation.
static void test_f16_vs_ref(const std::vector<int64_t>& in_shape,
                            const std::vector<int64_t>& w_shape,
                            const TransposeConv2DAttributes& attrs)
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
    // Channel-last query; the reference reads dense [N, OC, OH, OW].
    const TensorDesc cl_desc =
        cl_out_desc_of(in_f32_view, w_f32_view, nullptr, attrs);
    const int64_t OH = cl_desc.dims[1];
    const int64_t OW = cl_desc.dims[2];
    const auto numel = static_cast<size_t>(cl_desc.numel());
    const TensorDesc ref_desc = dense_desc_of(cl_desc);

    TensorView in_ref(in_shape, DataType::f32, in_r.data(), TensorLayout::Dense);
    TensorView w_ref(w_shape, DataType::f32, w_r.data(), TensorLayout::Dense);
    std::vector<float> ref_buf(numel);
    auto out_ref = test::make_channel_first(ref_desc, ref_buf.data());
    {
        const TensorView ins[] = {in_ref, w_ref};
        ComputeContext ctx;
        backend::cpu::reference::transpose_conv2d_channel_first_ref(
            attrs, out_ref, ins, ctx, nullptr);
    }

    TensorView w16_view(w_shape, DataType::f16,
                        const_cast<half*>(w16.data()), TensorLayout::Dense);
    std::vector<half> out16(numel);

    auto in_cl = test::channel_first_to_channel_last<half>(
        std::span<const half>(in16.data(), in16.size()), N, IC, IH, IW);
    // Channel-last dims: [N, IH, IW, IC].
    const int64_t in_cl_shape[] = {N, IH, IW, IC};
    TensorView in16_view(std::span<const int64_t>(in_cl_shape, 4), DataType::f16,
                         in_cl.data(), TensorLayout::NHWC);
    TensorView out16_view(std::span<const int64_t>(cl_desc.dims.data(), 4),
                          DataType::f16, out16.data(), TensorLayout::NHWC);
    TensorView outs[] = {out16_view};
    const TensorView ins[] = {in16_view, w16_view};
    ComputeContext ctx;
    op->compute(outs, ins, ctx);

    auto got = test::f16_to_f32(test::channel_last_to_channel_first<half>(
        std::span<const half>(out16.data(), out16.size()), N, OC, OH, OW));

    TensorView got_view(std::span<const int64_t>(ref_desc.dims.data(), 4),
                        DataType::f32, got.data(), TensorLayout::Dense);
    NNOPS_EXPECT_TRUE(test::allclose(got_view, out_ref, test::kF16AccumTol,
                                     test::kF16AccumTol));
}

NNOPS_TEST(tconv2d_f16) {
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride      = {2, 2};
    attrs.dilation    = {1, 1};
    attrs.padding     = {1, 1};
    attrs.groups      = 1;

    test_f16_vs_ref({1, 32, 12, 12}, {32, 64, 3, 3}, attrs);
}

NNOPS_TEST(tconv2d_wide_oc) {
    // Wide enough that the tile repeats its oc block several times
    // over one row of pixels.
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride      = {2, 2};
    attrs.dilation    = {1, 1};
    attrs.padding     = {1, 1};
    attrs.groups      = 1;

    test_vs_ref({1, 64, 16, 16}, {64, 128, 3, 3}, attrs);
}

// ============================================================
// Col routes (depthwise, in-place filter, chunked col)
// ============================================================

NNOPS_TEST(tconv2d_depthwise) {
    // g == IC == OC, so every group holds one channel and the route vectorises
    // across groups. The group offset has to be exactly g on both operands or
    // the channels silently swap.
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride      = {2, 2};
    attrs.dilation    = {1, 1};
    attrs.padding     = {1, 1};
    attrs.groups      = 16;

    test_vs_ref({1, 16, 9, 9}, {16, 1, 3, 3}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(tconv2d_depthwise_stride1) {
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride      = {1, 1};
    attrs.dilation    = {1, 1};
    attrs.padding     = {1, 1};
    attrs.groups      = 12;
    attrs.epilogue    = {EpilogueActivateType::LeakyRelu};
    attrs.epilogue.alpha = 0.05f;

    test_vs_ref({1, 12, 10, 10}, {12, 1, 3, 3}, attrs);
}

NNOPS_TEST(tconv2d_direct_weight) {
    // Fewer pixels than the filter has input channels: the col is transposed
    // per pixel instead of the weight, so B is the OIHW filter read in place.
    // OC/G = 12 leaves the vector loop a tail as well.
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {4, 4};
    attrs.stride      = {2, 2};
    attrs.dilation    = {1, 1};
    attrs.padding     = {1, 1};
    attrs.groups      = 2;
    attrs.epilogue    = {EpilogueActivateType::Relu6};

    test_vs_ref({1, 32, 3, 3}, {32, 12, 4, 4}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(tconv2d_col_chunked) {
    // The 8 MiB col budget from the other side: the col is [pixels][tap·oc]
    // here, so it splits on oc and each chunk runs its own GEMM and scatter.
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {5, 5};
    attrs.stride      = {1, 1};
    attrs.dilation    = {1, 1};
    attrs.padding     = {2, 2};
    attrs.groups      = 1;

    test_vs_ref({1, 16, 64, 64}, {16, 256, 5, 5}, attrs);
}

NNOPS_TEST(tconv2d_batched_groups) {
    // Every batch item runs its own GEMM against the one packed filter, so the
    // group and batch offsets both have to survive.
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride      = {1, 1};
    attrs.dilation    = {1, 1};
    attrs.padding     = {1, 1};
    attrs.groups      = 2;

    test_vs_ref({3, 16, 10, 10}, {16, 8, 3, 3}, attrs, /*has_bias=*/true);
}

NNOPS_TEST(tconv2d_add_to) {
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
    const TensorDesc cl_out = cl_out_desc_of(in_channel_first, w_channel_first, nullptr, attrs);
    const TensorDesc dense_out = dense_desc_of(cl_out);
    const int64_t ON = cl_out.dims[0];
    const int64_t OH = cl_out.dims[1];
    const int64_t OW = cl_out.dims[2];
    const int64_t OC = cl_out.dims[3];

    auto [res_vec, res_view] = test::make_random_tensor(
        {ON, OC, OH, OW}, -1.0f, 1.0f, 11);
    (void)res_view;

    std::vector<float> ref_buf(res_vec);
    auto out_ref = test::make_channel_first(dense_out, ref_buf.data());
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

    const std::vector<int64_t> ishape = {ON, 8, 8, 16};
    const std::vector<int64_t> oshape = {ON, OH, OW, OC};
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
    const std::vector<int64_t> got_shape = {ON, OC, OH, OW};
    TensorView got_view(got_shape, DataType::f32, got.data(), TensorLayout::Dense);
    NNOPS_EXPECT_TRUE(test::allclose(got_view, out_ref, 1e-4f, 1e-4f));
}

// ============================================================
// residual_pre_activation — CPU routes
// ============================================================

NNOPS_TEST(tconv2d_pre_act) {
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

    test_residual_vs_ref({1, 16, 8, 8}, {16, 32, 4, 4}, attrs);
}

NNOPS_TEST(tconv2d_pre_act_bias_groups) {
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

    test_residual_vs_ref({2, 8, 9, 10}, {8, 8, 3, 3}, attrs, true);
}

NNOPS_TEST(tconv2d_pixel_pre_act) {
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

    test_residual_vs_ref({1, 16, 10, 11}, {16, 32, 2, 2}, attrs, true);
}

NNOPS_TEST(tconv2d_depthwise_pre_act) {
    // g == IC == OC, so this takes the depthwise merge.
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

    test_residual_vs_ref({1, 16, 9, 9}, {16, 1, 3, 3}, attrs, true);
}

// ============================================================
// Thread identity
// ============================================================

/// Fixed-size pool over a CpuBackend (the test_conv2d.cpp pattern).
struct SimplePool {
    explicit SimplePool(int nthreads) : nthreads_(nthreads) {
        if (nnops::test::bench_use_openmp()) {
            cpu = nnops::openmp_backend(nthreads);
            return;
        }
        cpu.parallel_for = [this](int64_t begin, int64_t end, const ParallelForBody& body) {
            this->parallel_for(begin, end, body);
        };
        cpu.num_threads = [this]() { return nthreads_; };
        cpu.thread_id = []() { return current_thread_id_; };
    }

    void parallel_for(int64_t begin, int64_t end, const ParallelForBody& body) {
        std::atomic<int64_t> next{begin};
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(nthreads_));
        for (int t = 0; t < nthreads_; ++t) {
            workers.emplace_back([&, t]() {
                current_thread_id_ = t;
                for (;;) {
                    int64_t i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= end) { break; }
                    body(i);
                }
            });
        }
        for (auto& w : workers) { w.join(); }
    }

    int nthreads_;
    CpuBackend cpu;
    inline static thread_local int current_thread_id_ = 0;
};

NNOPS_TEST(tconv2d_threaded_matches_serial) {
    // The kernels split the image, and an output row's merge stays inside its
    // own task, so a thread count may only change a GEMM's accumulation order.
    // The residual is off here: the add_to cases live in
    // test_residual_vs_ref, where the destination is built permuted.
    const std::vector<int64_t> in_shape = {2, 32, 12, 12};
    const std::vector<int64_t> w_shape  = {32, 16, 3, 3};

    auto [in_vec, in_cf] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_cf]  = test::make_random_tensor(w_shape);
    (void)w_vec;

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride      = {2, 2};
    attrs.dilation    = {1, 1};
    attrs.padding     = {1, 1};
    attrs.groups      = 1;
    attrs.epilogue    = {EpilogueActivateType::Relu6};

    auto cl_op = TransposeConv2D::create(attrs, Backend::CPU);

    std::vector<float> cl_in_buf = test::to_channel_last(
        std::span<const float>(in_cf.ptr<float>(), static_cast<size_t>(in_cf.numel())),
        in_cf.shape(0), in_cf.shape(1),
        in_cf.numel() / (in_cf.shape(0) * in_cf.shape(1)));
    const std::vector<int64_t> in_cl_dims = cl_dims_of(in_cf);
    TensorView in_cl(in_cl_dims, DataType::f32, cl_in_buf.data(),
                     TensorLayout::NHWC);

    const TensorDesc cl_in_desc = in_cl.desc();
    const TensorDesc w_desc = w_cf.desc();
    const TensorDesc cl_d[] = {cl_in_desc, w_desc};
    const auto cl_out_descs = cl_op->getOutputTensorDesc(cl_d);
    const auto cl_bytes = static_cast<size_t>(cl_out_descs[0].storage_bytes());

    std::vector<float> cl_ref_buf(cl_bytes);
    {
        TensorView out = test::make_packed(cl_out_descs[0], cl_ref_buf.data());
        TensorView outs[] = {out};
        const TensorView ins[] = {in_cl, w_cf};
        NNOPS_EXPECT_EQ(cl_op->compute(outs, ins, ComputeContext{}), Status::ok);
    }

    for (int nt : {2, 4, 20}) {
        std::vector<float> out_buf(cl_bytes);
        TensorView out = test::make_packed(cl_out_descs[0], out_buf.data());
        SimplePool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        TensorView outs[] = {out};
        const TensorView ins[] = {in_cl, w_cf};
        NNOPS_EXPECT_EQ(cl_op->compute(outs, ins, ctx), Status::ok);
        for (size_t i = 0; i < cl_ref_buf.size(); ++i) {
            NNOPS_EXPECT_NEAR(out_buf[i], cl_ref_buf[i], 1e-5f);
        }
    }
}
