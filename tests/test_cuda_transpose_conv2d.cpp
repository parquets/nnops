/// @file test_cuda_transpose_conv2d.cpp
/// @brief CUDA TransposeConv2D, channel-first and channel-last, against the CPU
///        backend.
///
/// Every case builds one activation/filter pair, runs the operator on both
/// backends over the same bytes, and compares. The CPU route is the oracle
/// because it already defines this operator's contract for both memory orders,
/// so the output size this geometry produces (output_padding included), the
/// bias placement, the padding ring and the two add_to orderings are pinned by
/// the operator itself rather than by a hand-written reference.
///
/// The cases split along the dispatch. The residue-class GEMM takes any stride
/// and dilation with OC/G wide enough for an n tile, so its cases walk stride 1
/// and 2, a stride-2 4x4, a dilated 3x3, groups 1/2/4 and tile-crossing planes.
/// Two geometries fall to the output-per-thread scripts and keep their own
/// cases: a depthwise group (one channel on both sides), which is the
/// channel-last 16 B lane when the pixels are channel-last and the planar script
/// otherwise, and an 8-channel group, which is below the GEMM's n gate.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/transpose_conv2d.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/cuda_test_helper.hpp"
#include "common/channel_last.hpp"

#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

struct TConvCase {
    int64_t n = 1, ic = 8, ih = 14, iw = 14;
    int64_t oc = 16;
    int64_t kh = 3, kw = 3;
    int64_t sh = 1, sw = 1;
    int64_t dh = 1, dw = 1;
    int64_t ph = 1, pw = 1;
    int64_t oph = 0, opw = 0;
    int64_t groups = 1;
    DataType dtype = DataType::f32;
    bool channel_last = false;
    bool use_bias = false;
    bool add_to = false;
    bool pre_act = false;
    // Channel-first only: a 32 B row pitch on the activation and on the output.
    bool padded = false;
    bool pad_output = false;
    Epilogue epilogue{};
};

struct Outputs {
    HostTensor tensor;
    int64_t oh = 0, ow = 0;
};

/// Run one case on both backends and compare; returns the reference result.
Outputs run_case(const TConvCase& c, const char* what)
{
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {c.kh, c.kw};
    attrs.stride = {c.sh, c.sw};
    attrs.dilation = {c.dh, c.dw};
    attrs.padding = {c.ph, c.pw};
    attrs.output_padding = {c.oph, c.opw};
    attrs.groups = c.groups;
    attrs.epilogue = c.epilogue;
    attrs.add_to = c.add_to;
    attrs.residual_pre_activation = c.pre_act;

    const TensorLayout layout =
        c.channel_last ? TensorLayout::NHWC : TensorLayout::NCHW;
    // The filter is [IC, OC/G, KH, KW] in both memory orders.
    const int64_t ocg = c.oc / c.groups;

    const bool pad_in = c.padded && !c.channel_last;
    HostTensor act = make_host_random(std::vector<int64_t>{c.n, c.ic, c.ih, c.iw},
                                      c.dtype, pad_in, 23, -0.5f, 0.5f, layout);
    HostTensor wgt = make_host_random(
        std::vector<int64_t>{c.ic, ocg, c.kh, c.kw}, c.dtype, false, 29, -0.25f,
        0.25f, TensorLayout::NCHW);
    HostTensor bias;
    if (c.use_bias) {
        bias = make_host_random(std::vector<int64_t>{c.oc}, c.dtype, false, 31,
                                -0.25f, 0.25f);
    }

    std::vector<TensorDesc> descs = {act.view.desc(), wgt.view.desc()};
    if (c.use_bias) { descs.push_back(bias.view.desc()); }

    auto op_cpu = TransposeConv2D::create(attrs, Backend::CPU);
    const std::vector<TensorDesc> outs = op_cpu->getOutputTensorDesc(descs);
    NNOPS_EXPECT_TRUE(!outs.empty());
    const TensorDesc od = outs.front();

    Outputs res;
    res.oh = od.dims[2];
    res.ow = od.dims[3];

    const bool out_padded = c.pad_output && !c.channel_last;
    const auto seed = [&](uint64_t s) {
        return c.add_to ? make_host_random(od.dims, od.dtype, out_padded, s, 0.0f,
                                           0.5f, layout)
                        : make_host_zeroed(od.dims, od.dtype, out_padded, layout);
    };
    HostTensor out_ref = seed(37);
    HostTensor out_dev = seed(37);

    std::vector<TensorView> cpu_ins = {act.view, wgt.view};
    if (c.use_bias) { cpu_ins.push_back(bias.view); }
    cpu_run(*op_cpu, out_ref.view, cpu_ins);

    auto op_dev = TransposeConv2D::create(attrs, Backend::CUDA);
    DeviceTensor dact = to_device(act);
    DeviceTensor dwgt = to_device(wgt);
    DeviceTensor dbias;
    std::vector<TensorView> dev_ins = {dact.view, dwgt.view};
    if (c.use_bias) {
        dbias = to_device(bias);
        dev_ins.push_back(dbias.view);
    }
    DeviceTensor dout = alloc_like(out_dev.view, out_dev);
    cuda_run(*op_dev, dout.view, dev_ins);
    from_device(dout, out_dev);

    const float tol_abs = c.dtype == DataType::f16 ? kF16AccumTol : 1e-3f;
    const float tol_rel = c.dtype == DataType::f16 ? 2e-2f : 1e-4f;
    expect_same(out_dev, out_ref, tol_abs, tol_rel, what);

    res.tensor = std::move(out_ref);
    return res;
}

}  // namespace

// ============================================================
// Residue-class GEMM
// ============================================================

NNOPS_TEST(cuda_transpose_conv2d_stride) {
    {
        TConvCase c;   // stride-1 3x3: one class
        c.ic = 8; c.oc = 16;
        run_case(c, "cuda_tconv_stride");
    }
    {
        TConvCase c;   // stride-2 3x3: four classes, PH = 1
        c.ic = 8; c.oc = 16; c.sh = c.sw = 2;
        run_case(c, "cuda_tconv_stride");
    }
    {
        TConvCase c;   // the 4x4 stride-2 upsampler a decoder actually runs
        c.ic = 16; c.oc = 32; c.ih = c.iw = 7; c.kh = c.kw = 4; c.sh = c.sw = 2;
        c.ph = c.pw = 1;
        run_case(c, "cuda_tconv_stride");
    }
    {
        TConvCase c;   // stride 2 with output_padding on a non-square kernel
        c.ic = 8; c.oc = 16; c.kh = 3; c.kw = 2; c.sh = 2; c.sw = 2;
        c.ph = 1; c.pw = 0; c.oph = 1; c.opw = 0;
        run_case(c, "cuda_tconv_stride");
    }
    {
        TConvCase c;   // stride 1 with a non-square kernel and padding
        c.ic = 6; c.oc = 16; c.kh = 2; c.kw = 5; c.ph = 0; c.pw = 2;
        run_case(c, "cuda_tconv_stride");
    }
}

NNOPS_TEST(cuda_transpose_conv2d_dilation) {
    // Dilation changes the tap lattice a class sees: DH and SH share a factor,
    // so only some residues have any solution at all.
    for (int64_t d : {2, 3}) {
        for (int64_t s : {1, 2}) {
            TConvCase c;
            c.ic = 8; c.oc = 16; c.kh = c.kw = 3; c.dh = c.dw = d;
            c.sh = c.sw = s; c.ph = c.pw = d;
            run_case(c, "cuda_tconv_dilation");
        }
    }
}

NNOPS_TEST(cuda_transpose_conv2d_groups) {
    for (int64_t g : {2, 4}) {
        TConvCase c;
        c.ic = 16; c.oc = 32; c.groups = g; c.sh = c.sw = 2;
        c.ph = c.pw = 1;
        run_case(c, "cuda_tconv_groups");
    }
    {   // A group of 8 output channels: below the GEMM's n tile, so the script.
        TConvCase c;
        c.ic = 8; c.oc = 8; c.groups = 1;
        run_case(c, "cuda_tconv_groups");
    }
}

NNOPS_TEST(cuda_transpose_conv2d_depthwise) {
    // One channel on both sides of every group: the planar script, in both
    // memory orders (channel-last keeps a 16 B lane).
    for (bool last : {false, true}) {
        for (DataType dt : {DataType::f32, DataType::f16}) {
            TConvCase c;
            c.ic = 16; c.oc = 16; c.groups = 16; c.dtype = dt;
            c.channel_last = last; c.ih = c.iw = 9; c.ph = c.pw = 1;
            run_case(c, "cuda_tconv_depthwise");
            c.sh = c.sw = 2;
            run_case(c, "cuda_tconv_depthwise");
        }
    }
}

NNOPS_TEST(cuda_transpose_conv2d_wide) {
    // Past one CTA tile on both axes, and past one k tile.
    for (DataType dt : {DataType::f32, DataType::f16}) {
        TConvCase c;
        c.ic = 40; c.oc = 96; c.ih = c.iw = 12; c.dtype = dt;
        run_case(c, "cuda_tconv_wide");
    }
    {   // An m axis of 49 pixels against a 64-pixel tile, several batches.
        TConvCase c;
        c.n = 3; c.ic = 8; c.oc = 32; c.ih = c.iw = 7; c.ph = c.pw = 0;
        run_case(c, "cuda_tconv_wide");
    }
}

// ============================================================
// Channel-last
// ============================================================

NNOPS_TEST(cuda_transpose_conv2d_channel_last) {
    for (DataType dt : {DataType::f32, DataType::f16}) {
        TConvCase c;
        c.ic = 16; c.oc = 32; c.channel_last = true; c.dtype = dt;
        c.ih = c.iw = 11;
        run_case(c, "cuda_tconv_channel_last");
        c.sh = c.sw = 2;
        run_case(c, "cuda_tconv_channel_last");
        c.kh = c.kw = 4;
        c.oph = 1; c.opw = 1;
        run_case(c, "cuda_tconv_channel_last");
    }
    {   // A channel count the 16 B lane does not divide.
        TConvCase c;
        c.ic = 6; c.oc = 12; c.channel_last = true; c.ih = c.iw = 8;
        run_case(c, "cuda_tconv_channel_last");
    }
}

NNOPS_TEST(cuda_transpose_conv2d_channel_last_groups) {
    TConvCase c;
    c.ic = 16; c.oc = 32; c.groups = 4; c.channel_last = true;
    c.sh = c.sw = 2;
    run_case(c, "cuda_tconv_channel_last_groups");

    TConvCase d;
    d.ic = 32; d.oc = 128; d.groups = 2; d.channel_last = true;
    d.ih = d.iw = 10; d.dh = d.dw = 2; d.ph = d.pw = 2;
    run_case(d, "cuda_tconv_channel_last_groups");
}

// ============================================================
// Epilogue, add_to, pitched output
// ============================================================

NNOPS_TEST(cuda_transpose_conv2d_epilogue) {
    Epilogue relu;
    relu.type = EpilogueActivateType::Relu;
    Epilogue silu;
    silu.type = EpilogueActivateType::Silu;

    for (bool pre : {false, true}) {
        for (bool last : {false, true}) {
            for (const Epilogue& ep : {relu, silu}) {
                TConvCase c;
                c.ic = 8; c.oc = 16; c.channel_last = last; c.use_bias = true;
                c.add_to = true; c.pre_act = pre; c.epilogue = ep;
                run_case(c, "cuda_tconv_epilogue");
            }
        }
    }
    {   // A pure clip with no activation type.
        Epilogue clip;
        clip.min_clip = -0.02f;
        clip.max_clip = 0.03f;
        TConvCase c;
        c.ic = 8; c.oc = 16; c.use_bias = true; c.epilogue = clip;
        run_case(c, "cuda_tconv_epilogue");
    }
}

NNOPS_TEST(cuda_transpose_conv2d_pitched) {
    // A 32 B row pitch on both the activation and the destination: every store
    // goes through the operator's own strides, so the tail bytes stay put.
    for (DataType dt : {DataType::f32, DataType::f16}) {
        TConvCase c;
        c.ic = 8; c.oc = 16; c.ih = c.iw = 14; c.dtype = dt;
        c.padded = true; c.pad_output = true; c.oph = 1; c.opw = 1;
        run_case(c, "cuda_tconv_pitched");
    }
}

NNOPS_TEST(cuda_transpose_conv2d_layouts_agree) {
    // One transposed convolution computed in both memory orders must agree: the
    // channel-first result is converted to channel-last and held against the
    // channel-last route on the same filter bytes.
    const int64_t n = 2, ic = 8, ih = 9, iw = 7, oc = 16, kh = 3, kw = 3;
    const int64_t oh = (ih - 1) * 2 - 2 + kh, ow = (iw - 1) * 2 - 2 + kw;

    HostTensor act = make_host_random(std::vector<int64_t>{n, ic, ih, iw},
                                      DataType::f32, false, 47, -0.5f, 0.5f);
    HostTensor wgt = make_host_random(std::vector<int64_t>{ic, oc, kh, kw},
                                      DataType::f32, false, 53, -0.25f, 0.25f);

    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {kh, kw};
    attrs.stride = {2, 2};
    attrs.padding = {1, 1};

    auto op = TransposeConv2D::create(attrs, Backend::CUDA);
    const std::vector<int64_t> os = {n, oc, oh, ow};

    HostTensor out_cf = make_host_zeroed(os, DataType::f32, false);
    {
        DeviceTensor dact = to_device(act);
        DeviceTensor dwgt = to_device(wgt);
        DeviceTensor dout = alloc_like(out_cf.view, out_cf);
        const TensorView ins[] = {dact.view, dwgt.view};
        cuda_run(*op, dout.view, ins);
        from_device(dout, out_cf);
    }

    // The same values in the other memory order; the filter keeps its bytes.
    const float* act_p = reinterpret_cast<const float*>(act.bytes.data());
    const std::vector<float> act_cf(act_p, act_p + n * ic * ih * iw);
    const std::vector<float> act_last =
        channel_first_to_channel_last<float>(act_cf, n, ic, ih, iw);
    CudaBuffer d_act_last(act_last.size() * sizeof(float));
    d_act_last.upload(act_last.data(), act_last.size() * sizeof(float));
    CudaBuffer d_wgt(wgt.bytes.size());
    d_wgt.upload(wgt.bytes.data(), wgt.bytes.size());

    HostTensor out_last = make_host_zeroed(os, DataType::f32, false,
                                           TensorLayout::NHWC);
    {
        DeviceTensor dout = alloc_like(out_last.view, out_last);
        const int64_t ash[] = {n, ic, ih, iw};
        const int64_t wsh[] = {ic, oc, kh, kw};
        TensorView a_last(std::span<const int64_t>(ash, 4), DataType::f32,
                          d_act_last.data(), TensorLayout::NHWC);
        TensorView w(std::span<const int64_t>(wsh, 4), DataType::f32, d_wgt.data());
        const TensorView ins[] = {a_last, w};
        cuda_run(*op, dout.view, ins);
        from_device(dout, out_last);
    }

    const float* cf = reinterpret_cast<const float*>(out_cf.bytes.data());
    const std::vector<float> cf_flat(cf, cf + n * oc * oh * ow);
    const std::vector<float> cf_as_last =
        channel_first_to_channel_last<float>(cf_flat, n, oc, oh, ow);
    const float* gl = reinterpret_cast<const float*>(out_last.bytes.data());
    expect_close(cf_as_last, std::vector<float>(gl, gl + cf_as_last.size()), 1e-3f,
                 1e-4f, "cuda_tconv_layouts_agree");
}

#endif  // NNOPS_HAS_CUDA
