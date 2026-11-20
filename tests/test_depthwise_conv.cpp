/// @file test_depthwise_conv.cpp
/// @brief DepthwiseConv tests on NHWC activations.
///
/// The same values in both orders — dense plus the raw [C, 1, ...] filter for
/// the reference, NHWC plus TensorFlow's [KD][KH][KW][C]
/// filter for the kernel. The route reads the filter as the graph holds it, so
/// the only plumbing is the transpose between the orders. A call covers 8
/// channels, or 16 (two vectors) where the 3x3 kernel can take both halves —
/// which the C values below pick out: sub-vector, partial, full and multi-block.
///
/// The shape sweep picks the geometries that strand an edge mask: a plane one
/// row or one column wide, C of 1 and past a vector, a filter larger than a
/// vector, stride 2 with dilation 2 (the two compose, so the tap at the far
/// edge is read and the one past it is not), and 3D volumes with a kernel
/// shorter than the rank on one axis.

#include "nnops/ops/conv2d.hpp"
#include "nnops/ops/depthwise_conv.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/channel_last.hpp"

#include <vector>
#include <cstdio>
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
// Depthwise conv — NHWC activation, HWCM filter
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

/// Dense [N, C, spatial...] desc for a channel-last [N, spatial..., C] output —
/// the order the references read and write.
static TensorDesc dense_from_cl(const TensorDesc& cl, int64_t N, int64_t C)
{
    TensorDesc d;
    d.rank = cl.rank;
    d.dtype = cl.dtype;
    d.layout = TensorLayout::Dense;
    d.dims.push_back(N);
    d.dims.push_back(C);
    for (int64_t i = 1; i + 1 < cl.rank; ++i) { d.dims.push_back(cl.dims[i]); }
    return d;
}

/// Channel-last [N, spatial..., C] dims for a dense [N, C, spatial...] shape.
static void cl_dims_from(const std::vector<int64_t>& dense, int64_t* out)
{
    const int64_t rank = static_cast<int64_t>(dense.size());
    out[0] = dense[0];
    for (int64_t d = 2; d < rank; ++d) { out[d - 1] = dense[d]; }
    out[rank - 1] = dense[1];
}

/// A dense [N, C, spatial...] desc re-expressed channel-last, for shape queries.
static TensorDesc cl_desc_of(const TensorDesc& dense, const std::vector<int64_t>& dense_shape)
{
    TensorDesc d = dense;
    int64_t dims[8] = {};
    cl_dims_from(dense_shape, dims);
    d.dims = std::span<const int64_t>(dims, dense_shape.size());
    d.layout = (dense_shape.size() == 5) ? TensorLayout::NDHWC : TensorLayout::NHWC;
    return d;
}

template <typename T>
static void test_vs_ref(const std::vector<int64_t>& in_shape,
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

    // ---- Reference, in dense order ----------------------------------------------
    const TensorDesc ref_in[] = {cl_desc_of(in_channel_first.desc(), in_shape),
                                 w_channel_first.desc(), b_channel_first.desc()};
    auto ref_descs = op->getOutputTensorDesc(
        std::span<const TensorDesc>(ref_in, has_bias ? 3 : 2));
    const TensorDesc ref_desc = dense_from_cl(ref_descs[0], N, C);
    const int64_t outS  = ref_desc.numel() / (N * C);
    const int64_t out_n = N * C * outS;

    const auto residual_at = [](int64_t i) {
        return static_cast<float>(i % 5) * 0.5f - 1.0f;
    };
    std::vector<float> ref_buf(static_cast<size_t>(out_n));
    for (int64_t i = 0; attrs.add_to && i < out_n; ++i) {
        ref_buf[static_cast<size_t>(i)] = residual_at(i);
    }
    auto ref = test::make_channel_first(ref_desc, ref_buf.data());
    {
        ComputeContext ctx;
        const TensorView r[] = {in_channel_first, w_channel_first, b_channel_first};
        backend::cpu::reference::depthwise_conv_ref(
            attrs, ref, std::span<const TensorView>(r, has_bias ? 3 : 2), ctx, nullptr);
    }

    // ---- The same values, in NHWC order ----------------------------------
    const auto in_cl_f = test::to_channel_last<float>(in_vec, N, C, inS);
    const auto w_cl_f  = test::depthwise_to_hwcm<float>(w_vec, C, taps);
    std::vector<T> in_cl(in_cl_f.size());
    std::vector<T> w_cl(w_cl_f.size());
    for (size_t i = 0; i < in_cl.size(); ++i) { in_cl[i] = to_T(in_cl_f[i]); }
    for (size_t i = 0; i < w_cl.size(); ++i)  { w_cl[i]  = to_T(w_cl_f[i]); }

    int64_t in_shape_cl[8] = {};
    cl_dims_from(in_shape, in_shape_cl);
    TensorView in_v(std::span<const int64_t>(in_shape_cl, in_shape.size()), dt,
                    in_cl.data(), cl);
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

    // The destination is pixel-major, so the shortcut has to be permuted the
    // same way the activation is.
    std::vector<T> res_cf(static_cast<size_t>(out_n));
    for (int64_t i = 0; attrs.add_to && i < out_n; ++i) {
        res_cf[static_cast<size_t>(i)] = to_T(residual_at(i));
    }
    std::vector<T> out_buf(attrs.add_to
                               ? test::to_channel_last<T>(res_cf, N, C, outS)
                               : std::vector<T>(static_cast<size_t>(out_n)));
    TensorView out_v(std::span<const int64_t>(out_descs[0].dims.data(),
                                              static_cast<size_t>(out_descs[0].rank)),
                     dt, out_buf.data(), cl);

    std::vector<TensorView> ins = {in_v, w_v};
    if (has_bias) { ins.push_back(b_v); }
    NNOPS_EXPECT_TRUE(is_ok(op->compute(out_v, ins)));

    // ---- Back to dense order for the comparison ---------------------------------
    const auto got = test::to_channel_first<T>(out_buf, N, C, outS);
    for (int64_t i = 0; i < out_n; ++i) {
        NNOPS_EXPECT_NEAR(as_f(got[static_cast<size_t>(i)]),
                          ref_buf[static_cast<size_t>(i)], tol);
    }
}

NNOPS_TEST(dwconv_vs_ref_basic) {
    test_vs_ref<float>({2, 8, 16, 16}, {8, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}},
        false, 1e-4f);
}

NNOPS_TEST(dwconv_vs_ref_partial_block) {
    // C=5: an 8-lane call with three lanes masked; OH=7 leaves a three-row group.
    test_vs_ref<float>({1, 5, 9, 11}, {5, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}},
        false, 1e-4f);
}

NNOPS_TEST(dwconv_vs_ref_stride2_padding) {
    test_vs_ref<float>({1, 6, 14, 13}, {6, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 2, 2}, {1, 1, 1}, {0, 1, 1}},
        false, 1e-4f);
}

NNOPS_TEST(dwconv_vs_ref_dilation) {
    test_vs_ref<float>({1, 4, 10, 10}, {4, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 2, 2}, {0, 0, 0}},
        false, 1e-4f);
}

NNOPS_TEST(dwconv_vs_ref_bias) {
    test_vs_ref<float>({2, 8, 8, 8}, {8, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}},
        true, 1e-4f);
}

NNOPS_TEST(dwconv_vs_ref_add_to) {
    DepthwiseConvAttributes attrs{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}};
    attrs.add_to = true;
    test_vs_ref<float>({1, 3, 8, 8}, {3, 1, 3, 3}, attrs, false, 1e-4f);
}

NNOPS_TEST(dwconv_vs_ref_relu) {
    DepthwiseConvAttributes attrs{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}};
    attrs.epilogue.type = EpilogueActivateType::Relu;
    test_vs_ref<float>({1, 5, 6, 11}, {5, 1, 3, 3}, attrs, false, 1e-4f);
}

NNOPS_TEST(dwconv_vs_ref_pre_act) {
    DepthwiseConvAttributes attrs{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}};
    attrs.add_to = true;
    attrs.residual_pre_activation = true;
    attrs.epilogue.type = EpilogueActivateType::Relu;
    test_vs_ref<float>({1, 5, 6, 11}, {5, 1, 3, 3}, attrs, false, 1e-4f);
}

NNOPS_TEST(dwconv_vs_ref_pre_act_wide) {
    // C=17: the two-vector 3x3 kernel followed by the one-lane tail, both
    // reading their own destination before the epilogue.
    DepthwiseConvAttributes attrs{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}};
    attrs.add_to = true;
    attrs.residual_pre_activation = true;
    attrs.epilogue.type = EpilogueActivateType::Relu;
    test_vs_ref<float>({1, 17, 7, 7}, {17, 1, 3, 3}, attrs, true, 1e-4f);
}

NNOPS_TEST(dwconv_vs_ref_pre_act_generic) {
    // A 1x1 filter keeps the 1x1 shape off the 3x3 kernels and on the generic
    // row kernel.
    DepthwiseConvAttributes attrs{{1, 1, 1}, {1, 2, 2}, {1, 1, 1}, {0, 0, 0}};
    attrs.add_to = true;
    attrs.residual_pre_activation = true;
    attrs.epilogue.type = EpilogueActivateType::Relu;
    test_vs_ref<float>({1, 8, 9, 9}, {8, 1, 1, 1}, attrs, false, 1e-4f);
}

NNOPS_TEST(dwconv_vs_ref_f16_pre_act) {
    DepthwiseConvAttributes attrs{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}};
    attrs.add_to = true;
    attrs.residual_pre_activation = true;
    attrs.epilogue.type = EpilogueActivateType::Relu;
    test_vs_ref<backend::cpu::half>({1, 8, 10, 10}, {8, 1, 3, 3},
        attrs, false, 5e-3f);
}

NNOPS_TEST(dwconv_vs_ref_1x1_stride2) {
    test_vs_ref<float>({1, 8, 9, 9}, {8, 1, 1, 1},
        DepthwiseConvAttributes{{1, 1, 1}, {1, 2, 2}, {1, 1, 1}, {0, 0, 0}},
        false, 1e-4f);
}

NNOPS_TEST(dwconv_vs_ref_multi_block) {
    // C=17: one full sixteen-wide block plus a one-lane block; the plane is
    // small enough that the filter axis splits across tasks.
    test_vs_ref<float>({1, 17, 7, 7}, {17, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}},
        true, 1e-4f);
}

NNOPS_TEST(dwconv_vs_ref_f16) {
    test_vs_ref<backend::cpu::half>({1, 8, 10, 10}, {8, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}},
        false, 5e-3f);
}

NNOPS_TEST(dwconv_vs_ref_f16_wide) {
    // The two-vector 3x3 kernel is generic over T; half is a different
    // `v_load`/`v_fmadd`, so it gets its own wide case.
    test_vs_ref<backend::cpu::half>({1, 32, 10, 10}, {32, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}},
        true, 5e-3f);
}

NNOPS_TEST(dwconv_vs_ref_wide_stride2) {
    // C=16 is exactly one sixteen-wide block: the two-vector 3x3 kernel's
    // stride-2 instantiation, whose row table reads R*2+2 input rows instead of
    // R+2 — a different index set from the stride-1 cases above.
    test_vs_ref<float>({1, 16, 15, 15}, {16, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 2, 2}, {1, 1, 1}, {0, 1, 1}},
        true, 1e-4f);
}

NNOPS_TEST(dwconv_vs_ref_wide_block) {
    // C=48: three full sixteen-wide blocks, each handed to the two-vector 3x3
    // kernel at its own width. Passing the *remaining* channel count instead
    // routes the full blocks through the generic path — the same values, so
    // only a timing run would notice.
    test_vs_ref<float>({1, 48, 9, 9}, {48, 1, 3, 3},
        DepthwiseConvAttributes{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}},
        false, 1e-4f);
}

NNOPS_TEST(dwconv_ndhwc_vs_ref_3d) {
    test_vs_ref<float>({1, 4, 4, 6, 9}, {4, 1, 3, 3, 3},
        DepthwiseConvAttributes{{3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
        true, 1e-4f);
}

// ---- Quantized: same task grid, HWCM weight ------------------------------

template <typename InT, typename OutT>
static void dwconv_q_test(const std::vector<int64_t>& in_shape,
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
                       TensorLayout::Dense, in_qp);
    TensorView w_channel_first(std::span<const int64_t>(w_shape), in_dt, w_data.data(),
                      TensorLayout::Dense, w_qp);
    TensorView b_channel_first;
    if (has_bias) {
        const int64_t bsh[] = {C};
        b_channel_first = TensorView(std::span<const int64_t>(bsh, 1), DataType::f32,
                            bias_data.data());
    }

    auto op = DepthwiseConv::create(attrs, Backend::CPU);

    // ---- Reference, in dense order ----------------------------------------------
    const TensorDesc ref_in[] = {cl_desc_of(in_channel_first.desc(), in_shape),
                                 w_channel_first.desc(), b_channel_first.desc()};
    auto ref_descs = op->getOutputTensorDesc(
        std::span<const TensorDesc>(ref_in, has_bias ? 3 : 2));
    const TensorDesc ref_dense = dense_from_cl(ref_descs[0], N, C);
    const int64_t outS = ref_dense.numel() / (N * C);
    std::vector<OutT> ref_buf(static_cast<size_t>(N * C * outS));
    // A signed shortcut in the output's domain, so the pre- and post-epilogue
    // orders give different answers under a clamp or a ReLU.
    const auto residual_at = [&](int64_t i) {
        const int32_t z = out_qp.zero_point;
        return static_cast<OutT>(z + static_cast<int32_t>(i % 5) - 2);
    };
    for (int64_t i = 0; attrs.add_to && i < N * C * outS; ++i) {
        ref_buf[static_cast<size_t>(i)] = residual_at(i);
    }
    TensorView ref_out(std::span<const int64_t>(ref_dense.dims.data(),
                                                static_cast<size_t>(ref_dense.rank)),
                       out_dt, ref_buf.data(), TensorLayout::Dense, out_qp);
    {
        ComputeContext ctx;
        const TensorView r[] = {in_channel_first, w_channel_first, b_channel_first};
        backend::cpu::reference::depthwise_conv_ref(
            attrs, ref_out, std::span<const TensorView>(r, has_bias ? 3 : 2),
            ctx, nullptr);
    }

    // ---- The run: NHWC activation, HWCM filter ------------------------------------
    auto in_cl = test::to_channel_last<InT>(in_data, N, C, inS);
    auto w_cl  = test::depthwise_to_hwcm<InT>(w_data, C, taps);

    int64_t in_shape_cl[8] = {};
    cl_dims_from(in_shape, in_shape_cl);
    TensorView in_v(std::span<const int64_t>(in_shape_cl, in_shape.size()), in_dt,
                    in_cl.data(), cl, in_qp);
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

    std::vector<OutT> res_cf(static_cast<size_t>(N * C * outS));
    for (int64_t i = 0; attrs.add_to && i < N * C * outS; ++i) {
        res_cf[static_cast<size_t>(i)] = residual_at(i);
    }
    std::vector<OutT> out_buf(attrs.add_to
                                  ? test::to_channel_last<OutT>(res_cf, N, C, outS)
                                  : std::vector<OutT>(static_cast<size_t>(N * C * outS)));
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

NNOPS_TEST(dwconv_q_s8_to_s8) {
    DepthwiseConvAttributes attrs{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}};
    std::vector<float> wscales(8, 1.0f);
    std::vector<int32_t> wzps(8, 0);
    dwconv_q_test<int8_t, int8_t>({1, 8, 8, 8}, {8, 1, 3, 3}, attrs,
        make_per_tensor(1.0f, 0), make_per_channel(wscales, wzps),
        make_per_tensor(1.0f, 0), false);
}

NNOPS_TEST(dwconv_q_u8_to_u8_partial_bias) {
    DepthwiseConvAttributes attrs{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}};
    std::vector<float> wscales(5, 1.0f);
    std::vector<int32_t> wzps(5, 0);
    dwconv_q_test<uint8_t, uint8_t>({1, 5, 6, 6}, {5, 1, 3, 3}, attrs,
        make_per_tensor(1.0f, 0), make_per_channel(wscales, wzps),
        make_per_tensor(1.0f, 128), true);
}

NNOPS_TEST(dwconv_q_s8_pre_act) {
    // The quantized route dequantizes the destination into the sum, so the
    // clamp sees the residual on the correct side of the output quantizer.
    DepthwiseConvAttributes attrs{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}};
    attrs.add_to = true;
    attrs.residual_pre_activation = true;
    attrs.epilogue.type = EpilogueActivateType::Relu;
    std::vector<float> wscales(8, 1.0f);
    std::vector<int32_t> wzps(8, 0);
    dwconv_q_test<int8_t, int8_t>({1, 8, 8, 8}, {8, 1, 3, 3}, attrs,
        make_per_tensor(1.0f, 0), make_per_channel(wscales, wzps),
        make_per_tensor(1.0f, 0), true);
}

NNOPS_TEST(dwconv_q_u8_pre_act_clamp) {
    // u8 with a non-zero zero point and a clamp: the residual arrives as a
    // quantized value and must be dequantized around that zero point.
    DepthwiseConvAttributes attrs{{1, 2, 2}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}};
    attrs.add_to = true;
    attrs.residual_pre_activation = true;
    attrs.epilogue.min_clip = -0.25f;
    attrs.epilogue.max_clip = 0.75f;
    std::vector<float> wscales(5, 1.0f);
    std::vector<int32_t> wzps(5, 0);
    dwconv_q_test<uint8_t, uint8_t>({1, 5, 6, 6}, {5, 1, 2, 2}, attrs,
        make_per_tensor(1.0f, 128), make_per_channel(wscales, wzps),
        make_per_tensor(1.0f, 128), true);
}

NNOPS_TEST(dwconv_q_multi_block_s8) {
    // C=17: two full vectors, a one-lane tail, and a split channel axis.
    DepthwiseConvAttributes attrs{{1, 2, 2}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}};
    std::vector<float> wscales(17, 1.0f);
    std::vector<int32_t> wzps(17, 0);
    dwconv_q_test<int8_t, int8_t>({1, 17, 5, 5}, {17, 1, 2, 2}, attrs,
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
/// against depthwise_conv_ref on the same values.
static void conv2d_depthwise_vs_ref(const std::vector<int64_t>& in_shape,
                                    const DepthwiseConvAttributes& dw,
                                    bool has_bias)
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
    const TensorDesc ref_in[] = {cl_desc_of(in_cf.desc(), in_shape), w_cf.desc(),
                                 b_cf.desc()};
    auto ref_descs = dw_op->getOutputTensorDesc(
        std::span<const TensorDesc>(ref_in, has_bias ? 3 : 2));
    const TensorDesc ref_desc = dense_from_cl(ref_descs[0], N, C);
    const int64_t out_n = ref_descs[0].numel();

    const auto residual_at = [](int64_t i) {
        return static_cast<float>(i % 5) * 0.5f - 1.0f;
    };
    std::vector<float> ref_buf(static_cast<size_t>(out_n));
    for (int64_t i = 0; dw.add_to && i < out_n; ++i) {
        ref_buf[static_cast<size_t>(i)] = residual_at(i);
    }
    auto ref = test::make_channel_first(ref_desc, ref_buf.data());
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
    attrs.residual_pre_activation = dw.residual_pre_activation;

    auto conv = Conv2D::create(attrs, Backend::CPU);

    std::vector<float> out_buf(static_cast<size_t>(out_n));
    for (int64_t i = 0; dw.add_to && i < out_n; ++i) {
        out_buf[static_cast<size_t>(i)] = residual_at(i);
    }
    if (dw.add_to) {
        out_buf = test::to_channel_last<float>(out_buf, N, C, out_n / (N * C));
    }
    auto in_cl = test::to_channel_last<float>(in_vec, N, C, S);
    auto w_cl  = test::depthwise_to_hwcm<float>(w_vec, C, taps);

    // The dims are channel-last: [N, H, W, C].
    const int64_t in_cl_shape[] = {N, H, W, C};
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

    for (int64_t i = 0; i < out_n; ++i) {
    }
    for (int64_t i = 0; i < out_n; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[static_cast<size_t>(i)],
                          ref_buf[static_cast<size_t>(i)], 1e-4f);
    }
}

NNOPS_TEST(conv2d_group_eq_channels_depthwise) {
    // Padding, bias and a relu epilogue: the pieces the depthwise route has to
    // carry across, not just the multiply-add. Also an NHWC activation and an
    // HWIO [KH, KW, 1, C] filter.
    DepthwiseConvAttributes dw{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}};
    dw.epilogue.type = EpilogueActivateType::Relu;
    conv2d_depthwise_vs_ref({1, 8, 8, 9}, dw, true);
}

NNOPS_TEST(conv2d_group_eq_channels_depthwise_pre_act) {
    // The reroute forwards the flag: a Conv2D with group == in_channels must
    // not silently apply its shortcut after the activation.
    DepthwiseConvAttributes dw{{1, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 1, 1}};
    dw.add_to = true;
    dw.residual_pre_activation = true;
    dw.epilogue.type = EpilogueActivateType::Relu;
    conv2d_depthwise_vs_ref({1, 8, 8, 9}, dw, true);
}

NNOPS_TEST(conv2d_group_eq_channels_depthwise_add_to) {
    // C=17 leaves a one-lane tail past two full vectors, and the filter is
    // smaller than its padding-free plane, so the output is not the input's size.
    DepthwiseConvAttributes dw{{1, 2, 2}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}};
    dw.add_to = true;
    conv2d_depthwise_vs_ref({1, 17, 7, 6}, dw, false);
}
