/// @file test_cuda_conv2d.cpp
/// @brief CUDA Conv2D, channel-first and channel-last, against the CPU backend.
///
/// Each case builds one activation/filter pair, runs the operator on both
/// backends over the same bytes, and compares. The CPU route is the oracle
/// because it is the reference the library itself ships for both orders, so the
/// plane sizes, the padding ring, the bias placement and the add_to ordering are
/// all pinned by the operator's own contract.
///
/// Geometry coverage follows what the implicit GEMM branches on: kernel size
/// 1x1/3x3/5x5 and a non-square 3x2, stride 1 and 2, dilation 2, groups 1/2 and
/// a depthwise-shaped group count, plane sizes that leave a partial pixel tile
/// (a 7x7 output is 49 pixels against a 64-pixel tile), and input channels that
/// do not fill a k tile. Both memory orders run the same cases, and one test
/// converts a channel-first result into the channel-last order and demands the two
/// routes agree, which catches an address error that both backends would have
/// to share.
///
/// One case cannot use the CPU as the oracle and says so where it runs:
/// epilogue.min_clip/max_clip, which Conv2D folds nowhere (see
/// core/epilogue.hpp) — the case documents that the fields are inert here
/// rather than asserting a clamp. A padded destination is a normal oracle case:
/// the CPU planner pins its oh-block to one row for a pitched output, so both
/// backends write the same padded rows.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/conv2d.hpp"
#include "nnops/detail/half.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/cuda_test_helper.hpp"
#include "common/channel_last.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

struct ConvCase {
    int64_t n = 1, ic = 8, ih = 14, iw = 14;
    int64_t oc = 16;
    int64_t kh = 3, kw = 3;
    int64_t sh = 1, sw = 1;
    int64_t dh = 1, dw = 1;
    int64_t ph = 1, pw = 1;
    int64_t groups = 1;
    DataType dtype = DataType::f32;
    bool channel_last = false;
    bool use_bias = false;
    bool add_to = false;
    // Channel-first only: a 32 B row pitch on the activation, and one on the output.
    // A pitched output pins the CPU planner's oh-block to one row (its GEMM
    // flattens a block into one contiguous run), so both backends have to
    // reproduce the same values in the same padded rows.
    bool padded = false;
    bool pad_output = false;
    Epilogue epilogue{};
};

struct Outputs {
    HostTensor tensor;
    int64_t oh = 0, ow = 0;
};

/// Run one case on both backends and compare. Returns the reference result so a
/// caller can cross-check two routes against each other.
Outputs run_case(const ConvCase& c, const char* what)
{
    Conv2DAttributes attrs;
    attrs.kernel_size = {c.kh, c.kw};
    attrs.stride = {c.sh, c.sw};
    attrs.dilation = {c.dh, c.dw};
    attrs.padding = {c.ph, c.pw};
    attrs.groups = c.groups;
    attrs.epilogue = c.epilogue;
    attrs.add_to = c.add_to;

    const TensorLayout act_layout =
        c.channel_last ? TensorLayout::NHWC : TensorLayout::NCHW;
    const TensorLayout wgt_layout =
        c.channel_last ? TensorLayout::HWIO : TensorLayout::NCHW;

    const int64_t icg = c.ic / c.groups;
    const int64_t ocg = c.oc / c.groups;

    HostTensor act = make_host_random(std::vector<int64_t>{c.n, c.ic, c.ih, c.iw}, c.dtype,
                                      c.padded && !c.channel_last, 23,
                                      -0.5f, 0.5f, act_layout);
    HostTensor wgt = make_host_random(
        c.channel_last ? std::vector<int64_t>{c.kh, c.kw, icg, c.oc}
                       : std::vector<int64_t>{c.oc, icg, c.kh, c.kw},
        c.dtype, false, 29, -0.25f, 0.25f, wgt_layout);
    HostTensor bias;
    if (c.use_bias) {
        bias = make_host_random(std::vector<int64_t>{c.oc}, c.dtype, false, 31, -0.25f, 0.25f);
    }
    (void)ocg;

    std::vector<TensorDesc> descs = {act.view.desc(), wgt.view.desc()};
    if (c.use_bias) { descs.push_back(bias.view.desc()); }

    auto op_cpu = Conv2D::create(attrs, Backend::CPU);
    const std::vector<TensorDesc> outs = op_cpu->getOutputTensorDesc(descs);
    NNOPS_EXPECT_TRUE(!outs.empty());
    const TensorDesc od = outs.front();

    Outputs res;
    res.oh = od.dims[2];
    res.ow = od.dims[3];

    const bool out_padded = c.pad_output && !c.channel_last;
    const auto seed = [&](uint64_t s) {
        return c.add_to ? make_host_random(od.dims, od.dtype, out_padded, s, 0.0f, 0.5f,
                                           act_layout)
                        : make_host_zeroed(od.dims, od.dtype, out_padded, act_layout);
    };
    HostTensor out_ref = seed(37);
    HostTensor out_dev = seed(37);

    std::vector<TensorView> cpu_ins = {act.view, wgt.view};
    if (c.use_bias) { cpu_ins.push_back(bias.view); }
    cpu_run(*op_cpu, out_ref.view, cpu_ins);

    auto op_dev = Conv2D::create(attrs, Backend::CUDA);
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

NNOPS_TEST(cuda_conv2d_channel_first_geometry) {
    {
        ConvCase c;   // 3x3 s1 p1 over 14x14
        c.ic = 8; c.oc = 16;
        run_case(c, "cuda_conv2d_channel_first_geometry");
    }
    {
        ConvCase c;   // stride 2
        c.ic = 8; c.oc = 16; c.sh = c.sw = 2; c.ph = c.pw = 0;
        run_case(c, "cuda_conv2d_channel_first_geometry");
    }
    {
        ConvCase c;   // dilation 2 on a 5x5 filter
        c.ic = 8; c.oc = 16; c.kh = c.kw = 5; c.dh = c.dw = 2; c.ph = c.pw = 4;
        run_case(c, "cuda_conv2d_channel_first_geometry");
    }
    {
        ConvCase c;   // 1x1 pointwise
        c.ic = 8; c.oc = 16; c.kh = c.kw = 1; c.ph = c.pw = 0;
        run_case(c, "cuda_conv2d_channel_first_geometry");
    }
    {
        ConvCase c;   // non-square filter and stride
        c.ic = 6; c.oc = 12; c.kh = 3; c.kw = 2; c.sh = 1; c.sw = 2;
        c.ph = 1; c.pw = 0;
        run_case(c, "cuda_conv2d_channel_first_geometry");
    }
    {
        ConvCase c;   // 7x7 plane: 49 pixels against a 64-pixel tile
        c.ic = 16; c.oc = 32; c.ih = c.iw = 7; c.ph = c.pw = 1;
        c.pad_output = true;
        run_case(c, "cuda_conv2d_channel_first_geometry");
    }
    {
        ConvCase c;   // multi-batch
        c.n = 3; c.ic = 5; c.oc = 9; c.ih = 9; c.iw = 11;
        run_case(c, "cuda_conv2d_channel_first_geometry");
    }
}

NNOPS_TEST(cuda_conv2d_channel_first_groups) {
    {
        ConvCase c;   // two groups
        c.ic = 8; c.oc = 16; c.groups = 2;
        run_case(c, "cuda_conv2d_channel_first_groups");
    }
    {
        ConvCase c;   // depthwise-shaped: one group per channel
        c.ic = 8; c.oc = 8; c.groups = 8;
        run_case(c, "cuda_conv2d_channel_first_groups");
    }
    {
        ConvCase c;   // four groups, non-square stride
        c.ic = 12; c.oc = 16; c.groups = 4; c.sh = c.sw = 2; c.ph = c.pw = 0;
        run_case(c, "cuda_conv2d_channel_first_groups");
    }
}

NNOPS_TEST(cuda_conv2d_channel_first_f16) {
    ConvCase c;
    c.ic = 8; c.oc = 16; c.dtype = DataType::f16; c.use_bias = true;
    run_case(c, "cuda_conv2d_channel_first_f16");

    ConvCase d;
    d.ic = 16; d.oc = 24; d.dtype = DataType::f16; d.padded = true;
    d.ih = d.iw = 7; d.groups = 2;
    d.pad_output = true;   // 7 f16 = 14 B a row: both planes get a 32 B pitch
    run_case(d, "cuda_conv2d_channel_first_f16");
}

NNOPS_TEST(cuda_conv2d_channel_first_epilogue) {
    // Relu with add_to: the residual lands after the activation.
    Epilogue relu;
    relu.type = EpilogueActivateType::Relu;

    ConvCase c;
    c.ic = 8; c.oc = 16; c.use_bias = true; c.epilogue = relu;
    c.add_to = true;
    run_case(c, "cuda_conv2d_channel_first_epilogue");

    // Relu6 is the other activation Conv folds into the store clamp.
    Epilogue relu6;
    relu6.type = EpilogueActivateType::Relu6;

    ConvCase d;
    d.ic = 8; d.oc = 16; d.use_bias = true; d.epilogue = relu6;
    run_case(d, "cuda_conv2d_channel_first_epilogue");

    // An activation that is not a pure clamp runs through the epilogue itself.
    Epilogue gelu;
    gelu.type = EpilogueActivateType::Gelu;

    ConvCase e;
    e.ic = 8; e.oc = 16; e.use_bias = true; e.epilogue = gelu;
    run_case(e, "cuda_conv2d_channel_first_epilogue");

    // epilogue.min_clip/max_clip are MatMul's: Conv2D folds only Relu/Relu6 and
    // ignores the fields (see core/epilogue.hpp), on both backends.
    Epilogue clip;
    clip.min_clip = -0.02f;
    clip.max_clip = 0.03f;

    ConvCase f;
    f.ic = 8; f.oc = 16; f.use_bias = true; f.epilogue = clip;
    run_case(f, "cuda_conv2d_channel_first_epilogue");
}

NNOPS_TEST(cuda_conv2d_channel_last_geometry) {
    {
        ConvCase c;
        c.ic = 8; c.oc = 16; c.channel_last = true;
        run_case(c, "cuda_conv2d_channel_last_geometry");
    }
    {
        ConvCase c;   // stride 2
        c.ic = 8; c.oc = 16; c.channel_last = true; c.sh = c.sw = 2;
        c.ph = c.pw = 0;
        run_case(c, "cuda_conv2d_channel_last_geometry");
    }
    {
        ConvCase c;   // 1x1
        c.ic = 16; c.oc = 32; c.channel_last = true; c.kh = c.kw = 1;
        c.ph = c.pw = 0;
        run_case(c, "cuda_conv2d_channel_last_geometry");
    }
    {
        ConvCase c;   // non-square filter, dilation 2
        c.ic = 6; c.oc = 12; c.channel_last = true; c.kh = 3; c.kw = 2;
        c.dh = c.dw = 2; c.ph = 2; c.pw = 0;
        run_case(c, "cuda_conv2d_channel_last_geometry");
    }
    {
        ConvCase c;   // 7x7 plane, several batches
        c.n = 2; c.ic = 16; c.oc = 24; c.channel_last = true;
        c.ih = c.iw = 7;
        run_case(c, "cuda_conv2d_channel_last_geometry");
    }
}

NNOPS_TEST(cuda_conv2d_channel_last_groups) {
    ConvCase c;
    c.ic = 8; c.oc = 16; c.channel_last = true; c.groups = 2;
    run_case(c, "cuda_conv2d_channel_last_groups");

    ConvCase d;
    d.ic = 8; d.oc = 8; d.channel_last = true; d.groups = 8;
    run_case(d, "cuda_conv2d_channel_last_groups");
}

NNOPS_TEST(cuda_conv2d_channel_last_f16_add_to) {
    Epilogue ep;
    ep.type = EpilogueActivateType::Relu;

    ConvCase c;
    c.ic = 8; c.oc = 16; c.channel_last = true; c.dtype = DataType::f16;
    c.use_bias = true; c.add_to = true; c.epilogue = ep;
    run_case(c, "cuda_conv2d_channel_last_f16_add_to");
}

NNOPS_TEST(cuda_conv2d_wide_channels) {
    // More than 64 output channels and more than 64 pixels: this is the shape
    // that exercises the second piece of a CTA tile wider than the 64-wide MMA
    // tile, which every smaller case here masks out completely.
    ConvCase c;
    c.ic = 64; c.oc = 128; c.ih = c.iw = 14; c.dtype = DataType::f16;
    run_case(c, "cuda_conv2d_wide_channels");

    ConvCase d = c;
    d.channel_last = true;
    d.use_bias = true;
    run_case(d, "cuda_conv2d_wide_channels");
}

NNOPS_TEST(cuda_conv2d_layouts_agree) {
    // One convolution computed in both orders must give the same numbers: run
    // the channel-first route, convert its output to the channel-last order, and hold
    // it against the channel-last route on the transposed filter.
    const int64_t n = 2, ic = 8, ih = 12, iw = 10, oc = 16, kh = 3, kw = 3;
    const int64_t oh = ih, ow = iw;
    const int64_t icg = ic, ocg = oc;

    auto [act_flat, act_view] = make_random_tensor({n, ic, ih, iw}, -0.5f, 0.5f, 41);
    auto [wgt_flat, wgt_view] = make_random_tensor({oc, icg, kh, kw}, -0.25f, 0.25f, 43);
    (void)act_view;
    (void)wgt_view;

    std::vector<float> act_channel_last = channel_first_to_channel_last<float>(act_flat, n, ic, ih, iw);
    std::vector<float> wgt_hwio = oihw_to_hwio<float>(wgt_flat, oc, icg, kh, kw);

    Conv2DAttributes attrs;
    attrs.kernel_size = {kh, kw};
    attrs.padding = {1, 1};

    const int64_t as_channel_first[] = {n, ic, ih, iw};
    const int64_t ws_oihw[] = {oc, icg, kh, kw};
    const int64_t as_channel_last[] = {n, ic, ih, iw};
    const int64_t ws_hwio[] = {kh, kw, icg, oc};
    const int64_t os[] = {n, oc, oh, ow};

    // The operator reads the views' pointers on the device, so every operand
    // (and the two destinations) has to live in device memory.
    CudaBuffer d_act_channel_first(act_flat.size() * sizeof(float));
    CudaBuffer d_wgt_channel_first(wgt_flat.size() * sizeof(float));
    CudaBuffer d_act_last(act_channel_last.size() * sizeof(float));
    CudaBuffer d_wgt_last(wgt_hwio.size() * sizeof(float));
    const size_t out_bytes = static_cast<size_t>(n * oc * oh * ow) * sizeof(float);
    CudaBuffer d_out_channel_first(out_bytes);
    CudaBuffer d_out_last(out_bytes);
    d_act_channel_first.upload(act_flat.data(), act_flat.size() * sizeof(float));
    d_wgt_channel_first.upload(wgt_flat.data(), wgt_flat.size() * sizeof(float));
    d_act_last.upload(act_channel_last.data(), act_channel_last.size() * sizeof(float));
    d_wgt_last.upload(wgt_hwio.data(), wgt_hwio.size() * sizeof(float));
    d_out_channel_first.zero();
    d_out_last.zero();

    TensorView a_channel_first(as_channel_first, DataType::f32, d_act_channel_first.data());
    TensorView w_channel_first(ws_oihw, DataType::f32, d_wgt_channel_first.data());
    TensorView a_last(as_channel_last, DataType::f32, d_act_last.data(), TensorLayout::NHWC);
    TensorView w_last(ws_hwio, DataType::f32, d_wgt_last.data(), TensorLayout::HWIO);
    TensorView o_channel_first(os, DataType::f32, d_out_channel_first.data());
    TensorView o_last(os, DataType::f32, d_out_last.data(), TensorLayout::NHWC);

    std::vector<float> out_channel_first(static_cast<size_t>(n * oc * oh * ow), 0.0f);
    std::vector<float> out_last(out_channel_first.size(), 0.0f);

    auto op = Conv2D::create(attrs, Backend::CUDA);
    {
        const TensorView ins[] = {a_channel_first, w_channel_first};
        cuda_run(*op, o_channel_first, ins);
    }
    {
        const TensorView ins[] = {a_last, w_last};
        cuda_run(*op, o_last, ins);
    }
    d_out_channel_first.download(out_channel_first.data(), out_bytes);
    d_out_last.download(out_last.data(), out_bytes);

    std::vector<float> channel_first_as_last =
        channel_first_to_channel_last<float>(out_channel_first, n, oc, oh, ow);
    expect_close(channel_first_as_last, out_last, 1e-3f, 1e-4f, "cuda_conv2d_layouts_agree");
}

NNOPS_TEST(cuda_conv2d_pitched_output) {
    // The store goes through the operator's own address arithmetic, so a 32 B
    // row pitch has to leave the tail bytes alone and still land every element
    // where the strides say. Both backends run the same padded destination: the
    // CPU planner pins its oh-block to one row for it, and the two results are
    // compared row by row with the sentinel tails checked on both.
    for (DataType dt : {DataType::f32, DataType::f16}) {
        const std::vector<int64_t> act_shape = {1, 8, 14, 14};
        const std::vector<int64_t> wgt_shape = {16, 8, 3, 3};
        const std::vector<int64_t> out_shape = {1, 16, 14, 14};

        Conv2DAttributes attrs;
        attrs.kernel_size = {3, 3};
        attrs.padding = {1, 1};

        HostTensor act = make_host_random(act_shape, dt, false, 23);
        HostTensor wgt = make_host_random(wgt_shape, dt, false, 29, -0.25f, 0.25f);

        // Every byte, padding tails included, starts as the sentinel.
        HostTensor want = make_host_random(out_shape, dt, true, 5, 1000.0f, 1000.0f);
        HostTensor got = make_host_random(out_shape, dt, true, 5, 1000.0f, 1000.0f);

        auto op_cpu = Conv2D::create(attrs, Backend::CPU);
        const TensorView cpu_ins[] = {act.view, wgt.view};
        cpu_run(*op_cpu, want.view, cpu_ins);

        auto op = Conv2D::create(attrs, Backend::CUDA);
        DeviceTensor dact = to_device(act);
        DeviceTensor dwgt = to_device(wgt);
        DeviceTensor dgot = alloc_like(got.view, got);
        {
            const TensorView ins[] = {dact.view, dwgt.view};
            cuda_run(*op, dgot.view, ins);
        }
        from_device(dgot, got);

        const int64_t esz = data_type_size(dt);
        const int64_t rows = got.view.total_rows();
        const int64_t cols = got.view.numel() / rows;
        const int64_t tail = (got.view.pitch() - cols * esz) / esz;
        int bad_tail = 0;
        for (const HostTensor* t : {&got, &want}) {
            for (int64_t r = 0; r < rows; ++r) {
                const std::byte* p = t->bytes.data() + r * t->view.pitch() + cols * esz;
                for (float v : to_floats(p, dt, static_cast<size_t>(tail))) {
                    if (v != 1000.0f) { ++bad_tail; }
                }
            }
        }

        const bool f16 = dt == DataType::f16;
        expect_same(got, want, f16 ? kF16AccumTol : 1e-5f, f16 ? 2e-2f : 1e-5f,
                    "cuda_conv2d_pitched_output");
        NNOPS_EXPECT_EQ(bad_tail, 0);
    }
}

NNOPS_TEST(cuda_conv2d_i8) {
    // The quantized route on CUDA: a pre-quantized u8/s8 activation against an
    // s8 OIHW filter, channel-first, compared bit for bit with the CPU route.
    // Integer destinations are exact by construction (the accumulate is int32);
    // the dequantizing ones use the fp tolerance.
    struct I8Conv {
        int64_t n, ic, ih, iw, oc, kh, kw, sh, sw, dh, dw, ph, pw, groups;
        DataType adt, odt;
        bool bias, relu;
        uint64_t seed;
        bool cl = false;   // NHWC activation + HWIO filter
        int32_t zp_x = 999;  // 999 keeps the default; else an explicit value
        bool sym_w = false;  // all-zero per-channel weight zero points
    };
    // Every (activation, destination) pair the quantized route resolves, in
    // both memory orders: channel-first OIHW above the channel-last HWIO block.
    const I8Conv cases[] = {
        // Isolation: s8 activation with a zero point of 0 and symmetric weights,
        // so the raw accumulate is the whole answer.
        {1, 4, 8, 8, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, DataType::s8, DataType::s8, false, false, 77, false, 0, true},
        // Only the -zp_a·R_b[n] half is live (the weight zero points are zero).
        {1, 4, 8, 8, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, DataType::u8, DataType::s8, false, false, 79, false, 128, true},
        // Only the -(R_a[m] - K·zp_a)·zp_b[n] half is live (the activation zero
        // point is zero).
        {1, 4, 8, 8, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, DataType::s8, DataType::s8, false, false, 78, false, 0, false},
        {2, 8, 9, 7, 12, 3, 3, 1, 1, 1, 1, 1, 1, 2, DataType::u8, DataType::s8, true, true, 91},
        {2, 8, 9, 7, 12, 3, 3, 2, 1, 1, 1, 1, 1, 2, DataType::s8, DataType::s8, true, false, 93},
        // 1x1 has no padding and a filter row shorter than a 16 B vector.
        {1, 4, 8, 8, 6, 1, 1, 1, 1, 1, 1, 0, 0, 1, DataType::u8, DataType::u8, true, true, 97},
        {1, 4, 8, 8, 6, 3, 3, 1, 1, 2, 1, 1, 1, 1, DataType::s8, DataType::u8, true, true, 99},
        {1, 4, 8, 8, 6, 3, 3, 1, 1, 1, 1, 1, 1, 1, DataType::u8, DataType::f32, true, true, 101},
        {1, 4, 8, 8, 6, 3, 3, 1, 1, 1, 1, 1, 1, 1, DataType::s8, DataType::f32, true, true, 103},
        {1, 4, 8, 8, 6, 3, 3, 1, 1, 1, 1, 1, 1, 1, DataType::u8, DataType::f16, true, true, 107},
        {1, 4, 8, 8, 6, 3, 3, 1, 1, 1, 1, 1, 1, 1, DataType::s8, DataType::f16, false, false, 109},

        // Channel-last: IC/G a multiple of 16 so the vector A-fill runs.
        {1, 16, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, DataType::u8, DataType::s8, true, true, 111, true},
        {1, 16, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, DataType::s8, DataType::s8, true, false, 113, true},
        {1, 16, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, DataType::u8, DataType::u8, true, true, 115, true},
        {1, 16, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, DataType::s8, DataType::u8, true, true, 117, true},
        {1, 16, 8, 8, 8, 1, 1, 1, 1, 1, 1, 0, 0, 1, DataType::u8, DataType::f32, true, true, 119, true},
        {1, 16, 8, 8, 8, 1, 1, 1, 1, 1, 1, 0, 0, 1, DataType::s8, DataType::f32, false, false, 121, true},
        {1, 16, 8, 8, 8, 3, 3, 2, 1, 1, 1, 1, 1, 1, DataType::u8, DataType::f16, true, true, 123, true},
        {1, 16, 8, 8, 8, 3, 3, 1, 1, 2, 1, 1, 1, 1, DataType::s8, DataType::f16, true, true, 125, true},
        // Groups split ICG to 8, so this one takes the scalar channel-last fill.
        {1, 16, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, 2, DataType::u8, DataType::s8, true, true, 127, true},

        // Depthwise-shaped groups (IC/G = OC/G = 1) stay on the quantized GEMM.
        {1, 4, 8, 8, 4, 3, 3, 1, 1, 1, 1, 1, 1, 4, DataType::u8, DataType::s8, true, true, 129},
        {1, 16, 8, 8, 16, 3, 3, 1, 1, 1, 1, 1, 1, 16, DataType::s8, DataType::u8, true, true, 131, true},
    };

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
        const I8Conv& c = cases[ci];
        const int64_t icg = c.ic / c.groups;
        test::XorShift128 rng(c.seed);
        const int64_t an = c.n * c.ic * c.ih * c.iw;
        std::vector<uint8_t> a(static_cast<size_t>(an));
        for (auto& e : a) { e = static_cast<uint8_t>(rng.next_u64() & 0xFFu); }
        const int64_t wn = c.oc * icg * c.kh * c.kw;
        std::vector<int8_t> w(static_cast<size_t>(wn));
        for (auto& e : w) {
            e = static_cast<int8_t>(int32_t(rng.next_u64() % 255) - 127);
        }

        std::vector<float> w_s(static_cast<size_t>(c.oc));
        std::vector<int32_t> w_zp(static_cast<size_t>(c.oc));
        for (int64_t i = 0; i < c.oc; ++i) {
            w_s[static_cast<size_t>(i)] = 0.008f + 0.001f * float(i % 5);
            w_zp[static_cast<size_t>(i)] = c.sym_w ? 0 : int32_t(i % 7) - 3;
        }

        QuantParams qa;
        qa.scale = 0.0078125f;
        qa.zero_point = c.zp_x != 999 ? c.zp_x : (c.adt == DataType::u8 ? 128 : -6);
        QuantParams qw;
        qw.granularity = QuantGranularity::PerChannel;
        qw.scale_data = w_s.data();
        qw.zero_point_data = w_zp.data();
        qw.num_scales = c.oc;
        qw.scale = 0.008f;
        qw.zero_point = 0;
        QuantParams qc;
        qc.scale = 0.03f;
        qc.zero_point = c.odt == DataType::u8 ? 110 : 0;

        const bool int_out = c.odt == DataType::s32 || c.odt == DataType::s8 ||
                             c.odt == DataType::u8;
        const DataType bdt = int_out ? DataType::s32 : c.odt;
        std::vector<uint8_t> bias_bytes;
        if (c.bias) {
            bias_bytes.resize(static_cast<size_t>(c.oc) *
                              size_t(data_type_size(bdt)));
            for (int64_t i = 0; i < c.oc; ++i) {
                if (bdt == DataType::s32) {
                    reinterpret_cast<int32_t*>(bias_bytes.data())[i] =
                        int32_t(rng.next_u64() % 2001) - 1000;
                } else if (bdt == DataType::f16) {
                    reinterpret_cast<nnops::backend::cpu::half*>(
                        bias_bytes.data())[i] =
                        nnops::backend::cpu::float_to_half(
                            rng.next_float(-0.5f, 0.5f));
                } else {
                    reinterpret_cast<float*>(bias_bytes.data())[i] =
                        rng.next_float(-0.5f, 0.5f);
                }
            }
        }

        Conv2DAttributes attrs;
        attrs.kernel_size = {c.kh, c.kw};
        attrs.stride = {c.sh, c.sw};
        attrs.dilation = {c.dh, c.dw};
        attrs.padding = {c.ph, c.pw};
        attrs.groups = c.groups;
        attrs.output_dtype = c.odt;
        if (c.relu) { attrs.epilogue.type = EpilogueActivateType::Relu; }

        const auto view = [](const std::vector<int64_t>& sh, DataType dt,
                             void* d, TensorLayout lay,
                             const QuantParams& qp = {}) {
            return TensorView(std::span<const int64_t>(sh.data(), sh.size()), dt,
                              d, lay, qp);
        };
        const TensorLayout alay = c.cl ? TensorLayout::NHWC : TensorLayout::NCHW;
        const TensorLayout wlay = c.cl ? TensorLayout::HWIO : TensorLayout::NCHW;
        const std::vector<int64_t> ashape = {c.n, c.ic, c.ih, c.iw};
        const std::vector<int64_t> wshape =
            c.cl ? std::vector<int64_t>{c.kh, c.kw, icg, c.oc}
                 : std::vector<int64_t>{c.oc, icg, c.kh, c.kw};

        TensorView av = view(ashape, c.adt, a.data(), alay, qa);
        TensorView wv = view(wshape, DataType::s8, w.data(), wlay, qw);
        TensorView biasv;
        std::vector<TensorView> ins = {av, wv};
        std::vector<TensorDesc> descs = {av.desc(), wv.desc()};
        if (c.bias) {
            biasv = view({c.oc}, bdt, bias_bytes.data(), TensorLayout::NCHW);
            ins.push_back(biasv);
            descs.push_back(biasv.desc());
        }

        auto op_cpu = Conv2D::create(attrs, Backend::CPU);
        const TensorDesc od = op_cpu->getOutputTensorDesc(descs).front();
        const std::vector<int64_t> odims(od.dims.begin(), od.dims.begin() + od.rank);
        int64_t on = 1;
        for (auto d : odims) { on *= d; }
        const size_t obytes = static_cast<size_t>(on) *
                              size_t(data_type_size(od.dtype));
        std::vector<uint8_t> out_cpu(obytes, 0);
        std::vector<uint8_t> out_dev(obytes, 0);
        TensorView cv = view(odims, od.dtype, out_cpu.data(), od.layout, qc);
        cpu_run(*op_cpu, cv, ins);

        CudaBuffer da(static_cast<size_t>(an)), dw(static_cast<size_t>(wn));
        da.upload(a.data(), a.size());
        dw.upload(w.data(), w.size());
        TensorView dav = view(ashape, c.adt, da.data(), alay, qa);
        TensorView dwv = view(wshape, DataType::s8, dw.data(), wlay, qw);
        CudaBuffer dbias;
        std::vector<TensorView> dins = {dav, dwv};
        if (c.bias) {
            dbias = CudaBuffer(bias_bytes.size());
            dbias.upload(bias_bytes.data(), bias_bytes.size());
            dins.push_back(view({c.oc}, bdt, dbias.data(), TensorLayout::NCHW));
        }
        CudaBuffer dout(obytes);
        dout.zero();
        TensorView dcv = view(odims, od.dtype, dout.data(), od.layout, qc);
        auto op_dev = Conv2D::create(attrs, Backend::CUDA);
        cuda_run(*op_dev, dcv, dins);
        dout.download(out_dev.data(), obytes);

        if (int_out) {
            if (std::memcmp(out_cpu.data(), out_dev.data(), obytes) != 0) {
                const size_t esz = size_t(data_type_size(od.dtype));
                const size_t n = obytes / esz;
                size_t bad = 0;
                while (bad < n &&
                       std::memcmp(out_cpu.data() + bad * esz,
                                   out_dev.data() + bad * esz, esz) == 0) {
                    ++bad;
                }
                char msg[512];
                int off = std::snprintf(msg, sizeof(msg),
                              "conv2d_i8 case %zu (cl=%d act=%d out=%d): elem %zu; "
                              "cpu:", ci, int(c.cl), int(c.adt), int(c.odt), bad);
                for (size_t k = 0; k < 8 && k < n; ++k) {
                    off += std::snprintf(msg + off, sizeof(msg) - size_t(off), " %d",
                                         od.dtype == DataType::s8
                                             ? int(int8_t(out_cpu[k]))
                                             : int(out_cpu[k]));
                }
                off += std::snprintf(msg + off, sizeof(msg) - size_t(off), " | cuda:");
                for (size_t k = 0; k < 8 && k < n; ++k) {
                    off += std::snprintf(msg + off, sizeof(msg) - size_t(off), " %d",
                                         od.dtype == DataType::s8
                                             ? int(int8_t(out_dev[k]))
                                             : int(out_dev[k]));
                }
                throw std::runtime_error(msg);
            }
            continue;
        }
        const size_t n = obytes / size_t(data_type_size(od.dtype));
        for (size_t i = 0; i < n; ++i) {
            float x = 0.0f, y = 0.0f;
            if (od.dtype == DataType::f16) {
                x = nnops::backend::cpu::half_to_float(
                    reinterpret_cast<const nnops::backend::cpu::half*>(
                        out_dev.data())[i]);
                y = nnops::backend::cpu::half_to_float(
                    reinterpret_cast<const nnops::backend::cpu::half*>(
                        out_cpu.data())[i]);
            } else {
                x = reinterpret_cast<const float*>(out_dev.data())[i];
                y = reinterpret_cast<const float*>(out_cpu.data())[i];
            }
            NNOPS_EXPECT_TRUE(std::fabs(x - y) <=
                              1e-4f * std::max(1.0f, std::fabs(y)));
        }
    }

    // An s32 destination has no Conv2D route on either backend (MatMul keeps it).
    {
        const int64_t ashape[] = {1, 4, 8, 8};
        const int64_t wshape[] = {6, 4, 3, 3};
        std::vector<uint8_t> a(1 * 4 * 8 * 8, 1);
        std::vector<int8_t> w(6 * 4 * 3 * 3, 1);
        QuantParams qa;
        qa.scale = 0.01f;
        qa.zero_point = 128;
        QuantParams qw;
        qw.scale = 0.01f;
        qw.zero_point = 0;
        TensorView av(ashape, DataType::u8, a.data(), TensorLayout::NCHW, qa);
        TensorView wv(wshape, DataType::s8, w.data(), TensorLayout::NCHW, qw);
        const int64_t cshape[] = {1, 6, 6, 6};
        std::vector<int32_t> out(1 * 6 * 6 * 6, 0);
        TensorView cv(cshape, DataType::s32, out.data());

        Conv2DAttributes attrs;
        attrs.kernel_size = {3, 3};
        attrs.padding = {1, 1};
        attrs.output_dtype = DataType::s32;
        auto op = Conv2D::create(attrs, Backend::CUDA);
        const TensorView ins[] = {av, wv};
        NNOPS_EXPECT_EQ(op->compute({&cv, 1}, ins, cuda_context(), nullptr),
                        Status::unsupported);
    }
}

#endif  // NNOPS_HAS_CUDA
