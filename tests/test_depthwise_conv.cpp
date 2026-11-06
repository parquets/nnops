/// @file test_depthwise_conv.cpp
/// @brief DepthwiseConv tests — channel-first and channel-last.
///
/// Channel-first: the raw [C, 1, (KD,) KH, KW] filter goes straight to the operator,
/// and the output is compared with the scalar reference kernel. One case is
/// hand-computed so the suite is not only reference against reference.
///
/// Channel-last: the same values in both orders — channel-first plus the raw
/// channel-first filter for the reference, channel-last plus TensorFlow's
/// [KD][KH][KW][C] filter for the kernel. The route reads the filter as the
/// graph holds it, so the only plumbing is the transpose between the orders. A
/// call covers 8 channels, or 16 (two vectors) where the 3x3 kernel can take
/// both halves — which the C values below pick out: sub-vector, partial, full
/// and multi-block.

#include "nnops/ops/conv2d.hpp"
#include "nnops/ops/depthwise_conv.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/channel_last.hpp"

#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>

using namespace nnops;

// The reference kernel is defined in the backend, so declare it here.
namespace nnops::backend::cpu::reference {
    void depthwise_conv_ref(const DepthwiseConvAttributes& attrs,
                            TensorView& output,
                            std::span<const TensorView> inputs,
                            const ComputeContext& ctx,
                            void* workspace);
}

static void test_channel_first_vs_ref(const std::vector<int64_t>& in_shape,
                               const std::vector<int64_t>& w_shape,
                               const DepthwiseConvAttributes& attrs,
                               bool has_bias = false)
{
    auto [in_vec, in] = test::make_random_tensor(in_shape);
    auto [w_vec,  w]  = test::make_random_tensor(w_shape);
    std::vector<float> b_vec;
    TensorView b;
    if (has_bias) {
        auto p = test::make_random_tensor({w_shape[0]});
        b_vec = std::move(p.first);
        b = p.second;
    }

    auto dw_op = DepthwiseConv::create(attrs, Backend::CPU);

    TensorDesc in_desc = in.desc();
    TensorDesc w_desc  = w.desc();
    const TensorDesc desc_arr[] = {in_desc, w_desc};
    auto out_descs = dw_op->getOutputTensorDesc(desc_arr);

    std::vector<float> out_buf(static_cast<size_t>(out_descs[0].numel()));
    auto out = test::make_channel_first(out_descs[0], out_buf.data());
    std::vector<float> ref_buf(out_buf.size());
    auto ref = test::make_channel_first(out_descs[0], ref_buf.data());
    if (attrs.add_to) {
        std::fill(out_buf.begin(), out_buf.end(), 1.0f);
        std::fill(ref_buf.begin(), ref_buf.end(), 1.0f);
    }

    {
        ComputeContext ctx;
        if (has_bias) {
            const TensorView ref_ins[] = {in, w, b};
            backend::cpu::reference::depthwise_conv_ref(attrs, ref, ref_ins, ctx, nullptr);
        } else {
            const TensorView ref_ins[] = {in, w};
            backend::cpu::reference::depthwise_conv_ref(attrs, ref, ref_ins, ctx, nullptr);
        }
    }

    if (has_bias) {
        const TensorView ins[] = {in, w, b};
        NNOPS_EXPECT_TRUE(is_ok(dw_op->compute(out, ins)));
    } else {
        const TensorView ins[] = {in, w};
        NNOPS_EXPECT_TRUE(is_ok(dw_op->compute(out, ins)));
    }
    NNOPS_EXPECT_TRUE(test::allclose(out, ref, 1e-4f, 1e-4f));
}

NNOPS_TEST(dwconv_channel_first_vs_ref_basic) {
    // C=4, odd output width 13: the plane loop, a last vector 5 of 8 lanes.
    test_channel_first_vs_ref({1, 4, 12, 15}, {4, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}});
}

NNOPS_TEST(dwconv_channel_first_vs_ref_padding_stride2) {
    // Both W edges padded, stride 2: the vector load deinterleaves.
    test_channel_first_vs_ref({1, 6, 14, 13}, {6, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 2, 2}, {1, 1, 1}, {0, 1, 1}});
}

NNOPS_TEST(dwconv_channel_first_vs_ref_dilation) {
    test_channel_first_vs_ref({1, 4, 10, 10}, {4, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 2, 2}, {0, 0, 0}});
}

NNOPS_TEST(dwconv_channel_first_vs_ref_bias) {
    test_channel_first_vs_ref({2, 8, 8, 8}, {8, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}}, true);
}

NNOPS_TEST(dwconv_channel_first_vs_ref_add_to) {
    DepthwiseConvAttributes attrs{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}};
    attrs.add_to = true;
    test_channel_first_vs_ref({1, 3, 8, 8}, {3, 1, 3, 3}, attrs);
}

NNOPS_TEST(dwconv_channel_first_vs_ref_relu_epilogue) {
    DepthwiseConvAttributes attrs{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}};
    attrs.epilogue.type = EpilogueActivateType::Relu;
    test_channel_first_vs_ref({1, 5, 6, 11}, {5, 1, 3, 3}, attrs);
}

NNOPS_TEST(dwconv_channel_first_vs_ref_1x1_kernel_stride2) {
    test_channel_first_vs_ref({1, 8, 9, 9}, {8, 1, 1, 1},
        DepthwiseConvAttributes{{1, 1, 1}, {1, 2, 2}, {1, 1, 1}, {0, 0, 0}});
}

/// The channel-first route in f16. The f32 reference runs on the same values widened
/// back from f16, so the only error left is the kernel's own f16 arithmetic.
static void test_channel_first_f16_vs_ref(const std::vector<int64_t>& in_shape,
                                   const std::vector<int64_t>& w_shape,
                                   const DepthwiseConvAttributes& attrs,
                                   bool has_bias = false)
{
    using half = backend::cpu::half;
    auto [in_vec, in_f32] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_f32]  = test::make_random_tensor(w_shape);
    std::vector<float> b_vec;
    TensorView b_f32;
    if (has_bias) {
        auto p = test::make_random_tensor({w_shape[0]});
        b_vec = std::move(p.first);
        b_f32 = p.second;
    }

    // Round in place so the reference and the kernel see the same values.
    std::vector<half> in_f16(in_vec.size()), w_f16(w_vec.size()), b_f16(b_vec.size());
    for (size_t i = 0; i < in_f16.size(); ++i) {
        simd::s_store(&in_f16[i], in_vec[i]);
        in_vec[i] = simd::s_load(&in_f16[i]);
    }
    for (size_t i = 0; i < w_f16.size(); ++i) {
        simd::s_store(&w_f16[i], w_vec[i]);
        w_vec[i] = simd::s_load(&w_f16[i]);
    }
    for (size_t i = 0; i < b_f16.size(); ++i) {
        simd::s_store(&b_f16[i], b_vec[i]);
        b_vec[i] = simd::s_load(&b_f16[i]);
    }

    TensorView in(std::span<const int64_t>(in_shape), DataType::f16, in_f16.data());
    TensorView w(std::span<const int64_t>(w_shape), DataType::f16, w_f16.data());
    const int64_t b_shape[] = {w_shape[0]};
    TensorView b(std::span<const int64_t>(b_shape, 1), DataType::f16, b_f16.data());

    auto op = DepthwiseConv::create(attrs, Backend::CPU);
    const TensorDesc in_d[] = {in.desc(), w.desc(), b.desc()};
    auto out_descs = op->getOutputTensorDesc(
        std::span<const TensorDesc>(in_d, has_bias ? 3 : 2));

    std::vector<float> ref_buf(static_cast<size_t>(out_descs[0].numel()),
                               attrs.add_to ? 1.0f : 0.0f);
    auto ref = test::make_channel_first(out_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        const TensorView r[] = {in_f32, w_f32, b_f32};
        backend::cpu::reference::depthwise_conv_ref(
            attrs, ref, std::span<const TensorView>(r, has_bias ? 3 : 2), ctx, nullptr);
    }

    std::vector<half> out_buf(static_cast<size_t>(out_descs[0].numel()));
    auto out = test::make_channel_first(out_descs[0], out_buf.data());
    for (auto& v : out_buf) { simd::s_store(&v, attrs.add_to ? 1.0f : 0.0f); }

    const TensorView ins[] = {in, w, b};
    NNOPS_EXPECT_TRUE(
        is_ok(op->compute(out, std::span<const TensorView>(ins, has_bias ? 3 : 2))));

    const auto got = test::f16_to_f32(out_buf);
    for (size_t i = 0; i < got.size(); ++i) {
        NNOPS_EXPECT_NEAR(got[i], ref_buf[i], test::kF16AccumTol);
    }
}

NNOPS_TEST(dwconv_channel_first_f16_vs_ref) {
    // Four channel blocks: the first and last skip the folded width mask (their
    // masked load would read past the tensor), so only the middle blocks take
    // it — and in f16 that mask was built as raw bits, i.e. the subnormal
    // 0x0001 instead of one, which erased every masked tap.
    test_channel_first_f16_vs_ref({1, 32, 12, 15}, {32, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}}, true);
    // One width vector per row: every vector is an edge vector.
    test_channel_first_f16_vs_ref({1, 24, 9, 7}, {24, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}});
    // Stride 2 on the width, and dilation on both axes.
    test_channel_first_f16_vs_ref({2, 20, 9, 11}, {20, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 2, 2}, {1, 1, 1}, {0, 1, 1}});
    test_channel_first_f16_vs_ref({1, 16, 10, 10}, {16, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 2, 2}, {0, 0, 0}});
    // 3D, which walks the depth loop.
    test_channel_first_f16_vs_ref({1, 17, 4, 6, 9}, {17, 1, 3, 3, 3},
        DepthwiseConvAttributes{{3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, true);
}

NNOPS_TEST(dwconv_channel_first_3d_vs_ref) {
    test_channel_first_vs_ref({1, 4, 4, 6, 9}, {4, 1, 3, 3, 3},
        DepthwiseConvAttributes{{3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, true);
}

// The one hand-computed case: everything else compares against
// depthwise_conv_ref, itself a second implementation that could drift with the
// kernel.
NNOPS_TEST(dwconv_channel_first_3d_hand_check) {
    // 1x2x3x3x3 input of ones, 1x2x2x2x2 kernel of 0.5, stride 1, pad 0:
    // each output = 2*2*2 taps * 1.0 * 0.5 = 4.0.
    const int64_t ishape[] = {1, 2, 3, 3, 3};
    const int64_t wshape[] = {2, 1, 2, 2, 2};

    std::vector<float> in_buf(54, 1.0f);   // 1*2*3*3*3
    std::vector<float> w_buf(16, 0.5f);    // 2*1*2*2*2

    TensorView in(ishape, DataType::f32, in_buf.data(), TensorLayout::NCDHW);
    TensorView w(wshape, DataType::f32, w_buf.data(), TensorLayout::NCDHW);

    DepthwiseConvAttributes attrs{{2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}};
    auto op = DepthwiseConv::create(attrs, Backend::CPU);

    const TensorDesc in_desc_arr[] = {in.desc(), w.desc()};
    auto out_descs = op->getOutputTensorDesc(in_desc_arr);

    std::vector<float> out_buf(static_cast<size_t>(out_descs[0].numel()));
    auto out = test::make_channel_first(out_descs[0], out_buf.data());
    {
        const TensorView ins[] = {in, w};
        TensorView outs[] = {out};
        NNOPS_EXPECT_TRUE(op->compute(outs, ins) == Status::ok);
    }

    for (float v : out_buf) { NNOPS_EXPECT_NEAR(v, 4.0f, 1e-4f); }
}

// ============================================================
// Quantized depthwise conv — s8/u8 in and out, PerTensor activation and
// output, PerChannel weight.
// ============================================================

static QuantParams make_per_tensor(float scale, int32_t zp) {
    QuantParams qp;
    qp.granularity = QuantGranularity::PerTensor;
    qp.scale = scale;
    qp.zero_point = zp;
    return qp;
}

static QuantParams make_per_channel(const std::vector<float>& scales,
                                    const std::vector<int32_t>& zps) {
    QuantParams qp;
    qp.granularity = QuantGranularity::PerChannel;
    qp.scale = 1.0f;      // unused when per-channel
    qp.zero_point = 0;
    qp.scale_data = scales.data();
    qp.zero_point_data = zps.data();
    qp.num_scales = static_cast<int64_t>(scales.size());
    return qp;
}

// ============================================================
// Layout contract — the channel-first path reads the raw channel-first filter, so a
// channel-last weight is rejected at descriptor level rather than misread.
// ============================================================

NNOPS_TEST(dwconv_channel_first_rejects_channel_last_weight) {
    TensorDesc in;
    in.rank = 4;
    for (int64_t d : {1, 4, 8, 8}) { in.dims.push_back(d); }

    TensorDesc w;
    w.rank   = 4;
    w.layout = TensorLayout::NHWC;
    for (int64_t d : {1, 3, 3, 8}) { w.dims.push_back(d); }

    TensorDesc out;
    out.rank = 4;
    for (int64_t d : {1, 4, 6, 6}) { out.dims.push_back(d); }

    DepthwiseConvAttributes attrs{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}};
    auto op = DepthwiseConv::create(attrs, Backend::CPU);
    const TensorDesc ins[] = {in, w};
    const TensorDesc outs[] = {out};
    NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::unsupported);
}

// ============================================================
// Channel-last depthwise conv — channel-last activation, HWCM filter
// ============================================================

/// Taps of a depthwise filter, whose shape is [C, 1, (KD,) KH, KW].
static int64_t dw_taps(const std::vector<int64_t>& w_shape)
{
    int64_t t = 1;
    for (size_t i = 2; i < w_shape.size(); ++i) { t *= w_shape[i]; }
    return t;
}

/// Product of the spatial dims of an [N, C, (D,) H, W] shape — a shape vector
/// or a TensorDesc's dims, which is a SmallVector.
template <class Shape>
static int64_t dw_spatial_elems(const Shape& shape)
{
    int64_t s = 1;
    for (size_t i = 2; i < shape.size(); ++i) { s *= shape[i]; }
    return s;
}

template <typename T>
static void test_channel_last_vs_ref(const std::vector<int64_t>& in_shape,
                             const std::vector<int64_t>& w_shape,
                             const DepthwiseConvAttributes& attrs,
                             bool has_bias,
                             float tol)
{
    constexpr DataType dt = std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
    const auto to_T = [](float v) -> T {
        if constexpr (std::is_same_v<T, float>) {
            return v;
        } else {
            T h;
            simd::s_store(&h, v);
            return h;
        }
    };
    const auto as_f = [](T v) -> float {
        if constexpr (std::is_same_v<T, float>) { return v; }
        else { return simd::s_load(&v); }
    };

    const int64_t N     = in_shape[0];
    const int64_t C     = in_shape[1];
    const int64_t inS   = dw_spatial_elems(in_shape);
    const int64_t taps  = dw_taps(w_shape);
    const int64_t srank = static_cast<int64_t>(in_shape.size()) - 2;
    const TensorLayout cl = (srank == 3) ? TensorLayout::NDHWC : TensorLayout::NHWC;

    auto [in_vec, in_channel_first] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_channel_first]  = test::make_random_tensor(w_shape);
    std::vector<float> b_vec;
    TensorView b_channel_first;
    if (has_bias) {
        auto p = test::make_random_tensor({C});
        b_vec = std::move(p.first);
        b_channel_first = p.second;
    }

    auto op = DepthwiseConv::create(attrs, Backend::CPU);

    // ---- Reference, in the channel-first order ----------------------------------
    const TensorDesc ref_in[] = {in_channel_first.desc(), w_channel_first.desc(), b_channel_first.desc()};
    auto ref_descs = op->getOutputTensorDesc(
        std::span<const TensorDesc>(ref_in, has_bias ? 3 : 2));
    const int64_t outS  = dw_spatial_elems(ref_descs[0].dims);
    const int64_t out_n = N * C * outS;

    std::vector<float> ref_buf(static_cast<size_t>(out_n),
                               attrs.add_to ? 1.0f : 0.0f);
    auto ref = test::make_channel_first(ref_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        const TensorView r[] = {in_channel_first, w_channel_first, b_channel_first};
        backend::cpu::reference::depthwise_conv_ref(
            attrs, ref, std::span<const TensorView>(r, has_bias ? 3 : 2), ctx, nullptr);
    }

    // ---- The same values, in the channel-last order ----------------------
    const auto in_cl_f = test::to_channel_last<float>(in_vec, N, C, inS);
    const auto w_cl_f  = test::depthwise_to_hwcm<float>(w_vec, C, taps);
    std::vector<T> in_cl(in_cl_f.size());
    std::vector<T> w_cl(w_cl_f.size());
    for (size_t i = 0; i < in_cl.size(); ++i) { in_cl[i] = to_T(in_cl_f[i]); }
    for (size_t i = 0; i < w_cl.size(); ++i)  { w_cl[i]  = to_T(w_cl_f[i]); }

    TensorView in_v(std::span<const int64_t>(in_shape), dt, in_cl.data(), cl);
    const int64_t KD = (srank == 3) ? w_shape[2] : 1;
    const int64_t KH = w_shape[w_shape.size() - 2];
    const int64_t KW = w_shape.back();
    const int64_t wsh[] = {KD, KH, KW, C};
    TensorView w_v(std::span<const int64_t>(wsh, 4), dt, w_cl.data(),
                   TensorLayout::HWCM);

    std::vector<T> b_cl;
    TensorView b_v;
    if (has_bias) {
        b_cl.resize(static_cast<size_t>(C));
        for (int64_t c = 0; c < C; ++c) {
            b_cl[static_cast<size_t>(c)] = to_T(b_vec[static_cast<size_t>(c)]);
        }
        const int64_t bsh[] = {C};
        b_v = TensorView(std::span<const int64_t>(bsh, 1), dt, b_cl.data());
    }

    const TensorDesc cl_in[] = {in_v.desc(), w_v.desc(), b_v.desc()};
    auto out_descs = op->getOutputTensorDesc(
        std::span<const TensorDesc>(cl_in, has_bias ? 3 : 2));
    NNOPS_EXPECT_TRUE(out_descs[0].layout == cl);
    NNOPS_EXPECT_EQ(out_descs[0].numel(), out_n);

    std::vector<T> out_buf(static_cast<size_t>(out_n),
                           to_T(attrs.add_to ? 1.0f : 0.0f));
    TensorView out_v(std::span<const int64_t>(out_descs[0].dims.data(),
                                              static_cast<size_t>(out_descs[0].rank)),
                     dt, out_buf.data(), cl);

    std::vector<TensorView> ins = {in_v, w_v};
    if (has_bias) { ins.push_back(b_v); }
    NNOPS_EXPECT_TRUE(is_ok(op->compute(out_v, ins)));

    // ---- Back to the channel-first order for the comparison ---------------------
    const auto got = test::to_channel_first<T>(out_buf, N, C, outS);
    for (int64_t i = 0; i < out_n; ++i) {
        NNOPS_EXPECT_NEAR(as_f(got[static_cast<size_t>(i)]),
                          ref_buf[static_cast<size_t>(i)], tol);
    }
}

NNOPS_TEST(dwconv_channel_last_vs_ref_basic) {
    test_channel_last_vs_ref<float>({2, 8, 16, 16}, {8, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}},
        false, 1e-4f);
}

NNOPS_TEST(dwconv_channel_last_vs_ref_partial_block) {
    // C=5: an 8-lane call with three lanes masked; OH=7 leaves a three-row group.
    test_channel_last_vs_ref<float>({1, 5, 9, 11}, {5, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}},
        false, 1e-4f);
}

NNOPS_TEST(dwconv_channel_last_vs_ref_stride2_padding) {
    test_channel_last_vs_ref<float>({1, 6, 14, 13}, {6, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 2, 2}, {1, 1, 1}, {0, 1, 1}},
        false, 1e-4f);
}

NNOPS_TEST(dwconv_channel_last_vs_ref_dilation) {
    test_channel_last_vs_ref<float>({1, 4, 10, 10}, {4, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 2, 2}, {0, 0, 0}},
        false, 1e-4f);
}

NNOPS_TEST(dwconv_channel_last_vs_ref_bias) {
    test_channel_last_vs_ref<float>({2, 8, 8, 8}, {8, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}},
        true, 1e-4f);
}

NNOPS_TEST(dwconv_channel_last_vs_ref_add_to) {
    DepthwiseConvAttributes attrs{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}};
    attrs.add_to = true;
    test_channel_last_vs_ref<float>({1, 3, 8, 8}, {3, 1, 3, 3}, attrs, false, 1e-4f);
}

NNOPS_TEST(dwconv_channel_last_vs_ref_relu) {
    DepthwiseConvAttributes attrs{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}};
    attrs.epilogue.type = EpilogueActivateType::Relu;
    test_channel_last_vs_ref<float>({1, 5, 6, 11}, {5, 1, 3, 3}, attrs, false, 1e-4f);
}

NNOPS_TEST(dwconv_channel_last_vs_ref_1x1_stride2) {
    test_channel_last_vs_ref<float>({1, 8, 9, 9}, {8, 1, 1, 1},
        DepthwiseConvAttributes{{1, 1, 1}, {1, 2, 2}, {1, 1, 1}, {0, 0, 0}},
        false, 1e-4f);
}

NNOPS_TEST(dwconv_channel_last_vs_ref_multi_block) {
    // C=17: one full sixteen-wide block plus a one-lane block; the plane is
    // small enough that the filter axis splits across tasks.
    test_channel_last_vs_ref<float>({1, 17, 7, 7}, {17, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}},
        true, 1e-4f);
}

NNOPS_TEST(dwconv_channel_last_vs_ref_f16) {
    test_channel_last_vs_ref<backend::cpu::half>({1, 8, 10, 10}, {8, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}},
        false, 5e-3f);
}

NNOPS_TEST(dwconv_channel_last_vs_ref_f16_wide) {
    // The two-vector 3x3 kernel is generic over T; half is a different
    // `v_load`/`v_fmadd`, so it gets its own wide case.
    test_channel_last_vs_ref<backend::cpu::half>({1, 32, 10, 10}, {32, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}},
        true, 5e-3f);
}

NNOPS_TEST(dwconv_channel_last_vs_ref_wide_stride2) {
    // C=16 is exactly one sixteen-wide block: the two-vector 3x3 kernel's
    // stride-2 instantiation, whose row table reads R*2+2 input rows instead of
    // R+2 — a different index set from the stride-1 cases above.
    test_channel_last_vs_ref<float>({1, 16, 15, 15}, {16, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 2, 2}, {1, 1, 1}, {0, 1, 1}},
        true, 1e-4f);
}

NNOPS_TEST(dwconv_channel_last_vs_ref_wide_block) {
    // C=48: three full sixteen-wide blocks, each handed to the two-vector 3x3
    // kernel at its own width. Passing the *remaining* channel count instead
    // routes the full blocks through the generic path — the same values, so
    // only a timing run would notice.
    test_channel_last_vs_ref<float>({1, 48, 9, 9}, {48, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}},
        false, 1e-4f);
}

NNOPS_TEST(dwconv_ndhwc_vs_ref_3d) {
    test_channel_last_vs_ref<float>({1, 4, 4, 6, 9}, {4, 1, 3, 3, 3},
        DepthwiseConvAttributes{{3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
        true, 1e-4f);
}

// ---- Quantized, channel-last: same task grid, HWCM weight ----------------

template <typename InT, typename OutT>
static void dwconv_q_channel_last_test(const std::vector<int64_t>& in_shape,
                               const std::vector<int64_t>& w_shape,
                               const DepthwiseConvAttributes& attrs,
                               const QuantParams& in_qp,
                               const QuantParams& w_qp,
                               const QuantParams& out_qp,
                               bool has_bias)
{
    constexpr DataType in_dt  = std::is_same_v<InT, int8_t> ? DataType::s8 : DataType::u8;
    constexpr DataType out_dt = std::is_same_v<OutT, int8_t> ? DataType::s8 : DataType::u8;

    auto numel = [](const std::vector<int64_t>& s) {
        int64_t n = 1;
        for (int64_t d : s) { n *= d; }
        return n;
    };
    const int64_t N     = in_shape[0];
    const int64_t C     = in_shape[1];
    const int64_t inS   = dw_spatial_elems(in_shape);
    const int64_t taps  = dw_taps(w_shape);
    const int64_t srank = static_cast<int64_t>(in_shape.size()) - 2;
    const TensorLayout cl = (srank == 3) ? TensorLayout::NDHWC : TensorLayout::NHWC;

    std::vector<InT> in_data(static_cast<size_t>(numel(in_shape)));
    std::vector<InT> w_data(static_cast<size_t>(numel(w_shape)));
    for (size_t i = 0; i < in_data.size(); ++i) {
        int32_t v = static_cast<int32_t>((i * 7 + 3) % 9) - 4;
        if constexpr (std::is_same_v<InT, uint8_t>) { v += 4; }
        in_data[i] = static_cast<InT>(v);
    }
    for (size_t i = 0; i < w_data.size(); ++i) {
        int32_t v = static_cast<int32_t>((i * 5 + 1) % 5) - 2;
        if constexpr (std::is_same_v<InT, uint8_t>) { v += 2; }
        w_data[i] = static_cast<InT>(v);
    }
    std::vector<float> bias_data;
    if (has_bias) {
        bias_data.resize(static_cast<size_t>(C));
        for (int64_t c = 0; c < C; ++c) {
            bias_data[static_cast<size_t>(c)] = (c % 5) * 0.25f;
        }
    }

    TensorView in_channel_first(std::span<const int64_t>(in_shape), in_dt, in_data.data(),
                       TensorLayout::NCHW, in_qp);
    TensorView w_channel_first(std::span<const int64_t>(w_shape), in_dt, w_data.data(),
                      TensorLayout::NCHW, w_qp);
    TensorView b_channel_first;
    if (has_bias) {
        const int64_t bsh[] = {C};
        b_channel_first = TensorView(std::span<const int64_t>(bsh, 1), DataType::f32,
                            bias_data.data());
    }

    auto op = DepthwiseConv::create(attrs, Backend::CPU);

    // ---- Reference, in the channel-first order ----------------------------------
    const TensorDesc ref_in[] = {in_channel_first.desc(), w_channel_first.desc(), b_channel_first.desc()};
    auto ref_descs = op->getOutputTensorDesc(
        std::span<const TensorDesc>(ref_in, has_bias ? 3 : 2));
    const int64_t outS = dw_spatial_elems(ref_descs[0].dims);
    std::vector<OutT> ref_buf(static_cast<size_t>(N * C * outS));
    TensorView ref_out(std::span<const int64_t>(ref_descs[0].dims), out_dt,
                       ref_buf.data(), TensorLayout::NCHW, out_qp);
    {
        ComputeContext ctx;
        const TensorView r[] = {in_channel_first, w_channel_first, b_channel_first};
        backend::cpu::reference::depthwise_conv_ref(
            attrs, ref_out, std::span<const TensorView>(r, has_bias ? 3 : 2),
            ctx, nullptr);
    }

    // ---- Channel-last: channel-last activation, HWCM filter -----------------------
    auto in_cl = test::to_channel_last<InT>(in_data, N, C, inS);
    auto w_cl  = test::depthwise_to_hwcm<InT>(w_data, C, taps);

    TensorView in_v(std::span<const int64_t>(in_shape), in_dt, in_cl.data(), cl,
                    in_qp);
    const int64_t KD = (srank == 3) ? w_shape[2] : 1;
    const int64_t KH = w_shape[w_shape.size() - 2];
    const int64_t KW = w_shape.back();
    const int64_t wsh[] = {KD, KH, KW, C};
    TensorView w_v(std::span<const int64_t>(wsh, 4), in_dt, w_cl.data(),
                   TensorLayout::HWCM, w_qp);

    std::vector<TensorView> ins = {in_v, w_v};
    if (has_bias) {
        const int64_t bsh[] = {C};
        ins.push_back(TensorView(std::span<const int64_t>(bsh, 1), DataType::f32,
                                 bias_data.data()));
    }

    const TensorDesc cl_in[] = {in_v.desc(), w_v.desc(), b_channel_first.desc()};
    auto out_descs = op->getOutputTensorDesc(
        std::span<const TensorDesc>(cl_in, has_bias ? 3 : 2));
    NNOPS_EXPECT_TRUE(out_descs[0].layout == cl);

    std::vector<OutT> out_buf(static_cast<size_t>(N * C * outS));
    TensorView out_v(std::span<const int64_t>(out_descs[0].dims.data(),
                                              static_cast<size_t>(out_descs[0].rank)),
                     out_dt, out_buf.data(), cl, out_qp);
    NNOPS_EXPECT_TRUE(is_ok(op->compute(out_v, ins)));

    // ---- Compare dequantized, within one output step ----------------------
    const auto got = test::to_channel_first<OutT>(out_buf, N, C, outS);
    const float out_scale = out_qp.scale;
    const float out_zero  = static_cast<float>(out_qp.zero_point);
    const int64_t n = N * C * outS;
    for (int64_t i = 0; i < n; ++i) {
        const float fa = (static_cast<float>(ref_buf[static_cast<size_t>(i)]) - out_zero) * out_scale;
        const float fb = (static_cast<float>(got[static_cast<size_t>(i)]) - out_zero) * out_scale;
        NNOPS_EXPECT_NEAR(fa, fb, out_scale * 1.001f);
    }
}

NNOPS_TEST(dwconv_q_channel_last_s8_to_s8) {
    DepthwiseConvAttributes attrs{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}};
    std::vector<float> wscales(8, 1.0f);
    std::vector<int32_t> wzps(8, 0);
    dwconv_q_channel_last_test<int8_t, int8_t>({1, 8, 8, 8}, {8, 1, 3, 3}, attrs,
        make_per_tensor(1.0f, 0), make_per_channel(wscales, wzps),
        make_per_tensor(1.0f, 0), false);
}

NNOPS_TEST(dwconv_q_channel_last_u8_to_u8_partial_bias) {
    DepthwiseConvAttributes attrs{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}};
    std::vector<float> wscales(5, 1.0f);
    std::vector<int32_t> wzps(5, 0);
    dwconv_q_channel_last_test<uint8_t, uint8_t>({1, 5, 6, 6}, {5, 1, 3, 3}, attrs,
        make_per_tensor(1.0f, 0), make_per_channel(wscales, wzps),
        make_per_tensor(1.0f, 128), true);
}

NNOPS_TEST(dwconv_q_channel_last_multi_block_s8) {
    // C=17: two full vectors, a one-lane tail, and a split channel axis.
    DepthwiseConvAttributes attrs{{1, 2, 2}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}};
    std::vector<float> wscales(17, 1.0f);
    std::vector<int32_t> wzps(17, 0);
    dwconv_q_channel_last_test<int8_t, int8_t>({1, 17, 5, 5}, {17, 1, 2, 2}, attrs,
        make_per_tensor(1.0f, 0), make_per_channel(wscales, wzps),
        make_per_tensor(1.0f, 0), false);
}

// ============================================================
// Conv2D's depthwise geometry — group == in_channels runs on these kernels
// ============================================================

/// A Conv2D whose group count equals its input channels is the depthwise
/// geometry, and compute() hands it to these kernels instead of the grouped
/// GEMM. The filter arrives in the order the GEMM path carries rather than the
/// depthwise one, but the two index alike, so nothing is converted. Compare it
/// against depthwise_conv_ref on the same values, in both activation layouts.
static void conv2d_depthwise_vs_ref(const std::vector<int64_t>& in_shape,
                                    const DepthwiseConvAttributes& dw,
                                    bool has_bias,
                                    bool channel_last)
{
    const int64_t N    = in_shape[0];
    const int64_t C    = in_shape[1];
    const int64_t H    = in_shape[2];
    const int64_t W    = in_shape[3];
    const int64_t S    = H * W;
    const int64_t KH   = dw.kernel_size[1];
    const int64_t KW   = dw.kernel_size[2];
    const int64_t taps = KH * KW;

    auto [in_vec, in_cf] = test::make_random_tensor(in_shape);
    auto [w_vec,  w_cf]  = test::make_random_tensor({C, 1, KH, KW});
    std::vector<float> b_vec;
    TensorView b_cf;
    if (has_bias) {
        auto p = test::make_random_tensor({C});
        b_vec = std::move(p.first);
        b_cf  = p.second;
    }

    // ---- The reference: the depthwise operator's own shape rule and kernel ---
    auto dw_op = DepthwiseConv::create(dw, Backend::CPU);
    const TensorDesc ref_in[] = {in_cf.desc(), w_cf.desc(), b_cf.desc()};
    auto ref_descs = dw_op->getOutputTensorDesc(
        std::span<const TensorDesc>(ref_in, has_bias ? 3 : 2));
    const int64_t out_n = ref_descs[0].numel();

    std::vector<float> ref_buf(static_cast<size_t>(out_n), dw.add_to ? 1.0f : 0.0f);
    auto ref = test::make_channel_first(ref_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        const TensorView r[] = {in_cf, w_cf, b_cf};
        backend::cpu::reference::depthwise_conv_ref(
            dw, ref, std::span<const TensorView>(r, has_bias ? 3 : 2), ctx, nullptr);
    }

    // ---- Conv2D at the same geometry ----------------------------------------
    Conv2DAttributes attrs;
    attrs.kernel_size = {KH, KW};
    attrs.stride      = {dw.stride[1], dw.stride[2]};
    attrs.dilation    = {dw.dilation[1], dw.dilation[2]};
    attrs.padding     = {dw.padding[1], dw.padding[2]};
    attrs.groups      = C;
    attrs.epilogue    = dw.epilogue;
    attrs.add_to      = dw.add_to;

    auto conv = Conv2D::create(attrs, Backend::CPU);

    std::vector<float> out_buf(static_cast<size_t>(out_n), dw.add_to ? 1.0f : 0.0f);
    if (channel_last) {
        auto in_cl = test::to_channel_last<float>(in_vec, N, C, S);
        auto w_cl  = test::depthwise_to_hwcm<float>(w_vec, C, taps);

        // A tensor's dims stay logical [N, C, H, W]; the layout tag is what
        // says the storage is pixel-major.
        const int64_t in_cl_shape[] = {N, C, H, W};
        TensorView in_v(in_cl_shape, DataType::f32, in_cl.data(), TensorLayout::NHWC);
        const int64_t w_cl_shape[] = {KH, KW, 1, C};
        TensorView w_v(w_cl_shape, DataType::f32, w_cl.data(), TensorLayout::HWIO);

        const TensorDesc cl_in[] = {in_v.desc(), w_v.desc(), b_cf.desc()};
        auto out_descs = conv->getOutputTensorDesc(
            std::span<const TensorDesc>(cl_in, has_bias ? 3 : 2));
        NNOPS_EXPECT_EQ(out_descs[0].numel(), out_n);
        NNOPS_EXPECT_TRUE(out_descs[0].layout == TensorLayout::NHWC);

        auto out = test::make_channel_first(out_descs[0], out_buf.data());
        const TensorView ins[] = {in_v, w_v, b_cf};
        NNOPS_EXPECT_TRUE(is_ok(conv->compute(
            out, std::span<const TensorView>(ins, has_bias ? 3 : 2))));
        out_buf = test::to_channel_first<float>(
            out_buf, N, C, out_descs[0].numel() / (N * C));
    } else {
        const TensorDesc cf_in[] = {in_cf.desc(), w_cf.desc(), b_cf.desc()};
        auto out_descs = conv->getOutputTensorDesc(
            std::span<const TensorDesc>(cf_in, has_bias ? 3 : 2));
        NNOPS_EXPECT_EQ(out_descs[0].numel(), out_n);

        auto out = test::make_channel_first(out_descs[0], out_buf.data());
        const TensorView ins[] = {in_cf, w_cf, b_cf};
        NNOPS_EXPECT_TRUE(is_ok(conv->compute(
            out, std::span<const TensorView>(ins, has_bias ? 3 : 2))));
    }

    for (int64_t i = 0; i < out_n; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[static_cast<size_t>(i)],
                          ref_buf[static_cast<size_t>(i)], 1e-4f);
    }
}

NNOPS_TEST(conv2d_group_eq_channels_depthwise_channel_first) {
    // Padding, bias and a relu epilogue: the pieces the depthwise route has to
    // carry across, not just the multiply-add.
    DepthwiseConvAttributes dw{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}};
    dw.epilogue.type = EpilogueActivateType::Relu;
    conv2d_depthwise_vs_ref({1, 6, 9, 9}, dw, true, false);
}

NNOPS_TEST(conv2d_group_eq_channels_depthwise_add_to) {
    DepthwiseConvAttributes dw{{1, 3, 3}, {1, 2, 2}, {1, 1, 1}, {0, 1, 1}};
    dw.add_to = true;
    conv2d_depthwise_vs_ref({2, 8, 10, 11}, dw, false, false);
}

NNOPS_TEST(conv2d_group_eq_channels_depthwise_channel_last) {
    // The same route under an NHWC activation and an HWIO [KH, KW, 1, C] filter.
    DepthwiseConvAttributes dw{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}};
    dw.epilogue.type = EpilogueActivateType::Relu;
    conv2d_depthwise_vs_ref({1, 8, 8, 9}, dw, true, true);
}

NNOPS_TEST(conv2d_group_eq_channels_depthwise_channel_last_add_to) {
    // C=17 leaves a one-lane tail past two full vectors, and the filter is
    // smaller than its padding-free plane, so the output is not the input's size.
    DepthwiseConvAttributes dw{{1, 2, 2}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}};
    dw.add_to = true;
    conv2d_depthwise_vs_ref({1, 17, 7, 6}, dw, false, true);
}
