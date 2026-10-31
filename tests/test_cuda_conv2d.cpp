/// @file test_cuda_conv2d.cpp
/// @brief CUDA Conv2D, planar and channel-last, against the CPU backend.
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
/// converts a planar result into the channel-last order and demands the two
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
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/cuda_test_helper.hpp"
#include "common/channel_last.hpp"

#include <cstdio>
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
    // Planar only: a 32 B row pitch on the activation, and one on the output.
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

NNOPS_TEST(cuda_conv2d_planar_geometry) {
    {
        ConvCase c;   // 3x3 s1 p1 over 14x14
        c.ic = 8; c.oc = 16;
        run_case(c, "cuda_conv2d_planar_geometry");
    }
    {
        ConvCase c;   // stride 2
        c.ic = 8; c.oc = 16; c.sh = c.sw = 2; c.ph = c.pw = 0;
        run_case(c, "cuda_conv2d_planar_geometry");
    }
    {
        ConvCase c;   // dilation 2 on a 5x5 filter
        c.ic = 8; c.oc = 16; c.kh = c.kw = 5; c.dh = c.dw = 2; c.ph = c.pw = 4;
        run_case(c, "cuda_conv2d_planar_geometry");
    }
    {
        ConvCase c;   // 1x1 pointwise
        c.ic = 8; c.oc = 16; c.kh = c.kw = 1; c.ph = c.pw = 0;
        run_case(c, "cuda_conv2d_planar_geometry");
    }
    {
        ConvCase c;   // non-square filter and stride
        c.ic = 6; c.oc = 12; c.kh = 3; c.kw = 2; c.sh = 1; c.sw = 2;
        c.ph = 1; c.pw = 0;
        run_case(c, "cuda_conv2d_planar_geometry");
    }
    {
        ConvCase c;   // 7x7 plane: 49 pixels against a 64-pixel tile
        c.ic = 16; c.oc = 32; c.ih = c.iw = 7; c.ph = c.pw = 1;
        c.pad_output = true;
        run_case(c, "cuda_conv2d_planar_geometry");
    }
    {
        ConvCase c;   // multi-batch
        c.n = 3; c.ic = 5; c.oc = 9; c.ih = 9; c.iw = 11;
        run_case(c, "cuda_conv2d_planar_geometry");
    }
}

NNOPS_TEST(cuda_conv2d_planar_groups) {
    {
        ConvCase c;   // two groups
        c.ic = 8; c.oc = 16; c.groups = 2;
        run_case(c, "cuda_conv2d_planar_groups");
    }
    {
        ConvCase c;   // depthwise-shaped: one group per channel
        c.ic = 8; c.oc = 8; c.groups = 8;
        run_case(c, "cuda_conv2d_planar_groups");
    }
    {
        ConvCase c;   // four groups, non-square stride
        c.ic = 12; c.oc = 16; c.groups = 4; c.sh = c.sw = 2; c.ph = c.pw = 0;
        run_case(c, "cuda_conv2d_planar_groups");
    }
}

NNOPS_TEST(cuda_conv2d_planar_f16) {
    ConvCase c;
    c.ic = 8; c.oc = 16; c.dtype = DataType::f16; c.use_bias = true;
    run_case(c, "cuda_conv2d_planar_f16");

    ConvCase d;
    d.ic = 16; d.oc = 24; d.dtype = DataType::f16; d.padded = true;
    d.ih = d.iw = 7; d.groups = 2;
    d.pad_output = true;   // 7 f16 = 14 B a row: both planes get a 32 B pitch
    run_case(d, "cuda_conv2d_planar_f16");
}

NNOPS_TEST(cuda_conv2d_planar_epilogue) {
    // Relu with add_to: the residual lands after the activation.
    Epilogue relu;
    relu.type = EpilogueActivateType::Relu;

    ConvCase c;
    c.ic = 8; c.oc = 16; c.use_bias = true; c.epilogue = relu;
    c.add_to = true;
    run_case(c, "cuda_conv2d_planar_epilogue");

    // Relu6 is the other activation Conv folds into the store clamp.
    Epilogue relu6;
    relu6.type = EpilogueActivateType::Relu6;

    ConvCase d;
    d.ic = 8; d.oc = 16; d.use_bias = true; d.epilogue = relu6;
    run_case(d, "cuda_conv2d_planar_epilogue");

    // An activation that is not a pure clamp runs through the epilogue itself.
    Epilogue gelu;
    gelu.type = EpilogueActivateType::Gelu;

    ConvCase e;
    e.ic = 8; e.oc = 16; e.use_bias = true; e.epilogue = gelu;
    run_case(e, "cuda_conv2d_planar_epilogue");

    // epilogue.min_clip/max_clip are MatMul's: Conv2D folds only Relu/Relu6 and
    // ignores the fields (see core/epilogue.hpp), on both backends.
    Epilogue clip;
    clip.min_clip = -0.02f;
    clip.max_clip = 0.03f;

    ConvCase f;
    f.ic = 8; f.oc = 16; f.use_bias = true; f.epilogue = clip;
    run_case(f, "cuda_conv2d_planar_epilogue");
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

NNOPS_TEST(cuda_conv2d_layouts_agree) {
    // One convolution computed in both orders must give the same numbers: run
    // the planar route, convert its output to the channel-last order, and hold
    // it against the channel-last route on the transposed filter.
    const int64_t n = 2, ic = 8, ih = 12, iw = 10, oc = 16, kh = 3, kw = 3;
    const int64_t oh = ih, ow = iw;
    const int64_t icg = ic, ocg = oc;

    auto [act_flat, act_view] = make_random_tensor({n, ic, ih, iw}, -0.5f, 0.5f, 41);
    auto [wgt_flat, wgt_view] = make_random_tensor({oc, icg, kh, kw}, -0.25f, 0.25f, 43);
    (void)act_view;
    (void)wgt_view;

    std::vector<float> act_nhwc = nchw_to_nhwc<float>(act_flat, n, ic, ih, iw);
    std::vector<float> wgt_hwio = oihw_to_hwio<float>(wgt_flat, oc, icg, kh, kw);

    Conv2DAttributes attrs;
    attrs.kernel_size = {kh, kw};
    attrs.padding = {1, 1};

    const int64_t as_nchw[] = {n, ic, ih, iw};
    const int64_t ws_oihw[] = {oc, icg, kh, kw};
    const int64_t as_nhwc[] = {n, ic, ih, iw};
    const int64_t ws_hwio[] = {kh, kw, icg, oc};
    const int64_t os[] = {n, oc, oh, ow};

    // The operator reads the views' pointers on the device, so every operand
    // (and the two destinations) has to live in device memory.
    CudaBuffer d_act_planar(act_flat.size() * sizeof(float));
    CudaBuffer d_wgt_planar(wgt_flat.size() * sizeof(float));
    CudaBuffer d_act_last(act_nhwc.size() * sizeof(float));
    CudaBuffer d_wgt_last(wgt_hwio.size() * sizeof(float));
    const size_t out_bytes = static_cast<size_t>(n * oc * oh * ow) * sizeof(float);
    CudaBuffer d_out_planar(out_bytes);
    CudaBuffer d_out_last(out_bytes);
    d_act_planar.upload(act_flat.data(), act_flat.size() * sizeof(float));
    d_wgt_planar.upload(wgt_flat.data(), wgt_flat.size() * sizeof(float));
    d_act_last.upload(act_nhwc.data(), act_nhwc.size() * sizeof(float));
    d_wgt_last.upload(wgt_hwio.data(), wgt_hwio.size() * sizeof(float));
    d_out_planar.zero();
    d_out_last.zero();

    TensorView a_planar(as_nchw, DataType::f32, d_act_planar.data());
    TensorView w_planar(ws_oihw, DataType::f32, d_wgt_planar.data());
    TensorView a_last(as_nhwc, DataType::f32, d_act_last.data(), TensorLayout::NHWC);
    TensorView w_last(ws_hwio, DataType::f32, d_wgt_last.data(), TensorLayout::HWIO);
    TensorView o_planar(os, DataType::f32, d_out_planar.data());
    TensorView o_last(os, DataType::f32, d_out_last.data(), TensorLayout::NHWC);

    std::vector<float> out_planar(static_cast<size_t>(n * oc * oh * ow), 0.0f);
    std::vector<float> out_last(out_planar.size(), 0.0f);

    auto op = Conv2D::create(attrs, Backend::CUDA);
    {
        const TensorView ins[] = {a_planar, w_planar};
        cuda_run(*op, o_planar, ins);
    }
    {
        const TensorView ins[] = {a_last, w_last};
        cuda_run(*op, o_last, ins);
    }
    d_out_planar.download(out_planar.data(), out_bytes);
    d_out_last.download(out_last.data(), out_bytes);

    std::vector<float> planar_as_last =
        nchw_to_nhwc<float>(out_planar, n, oc, oh, ow);
    expect_close(planar_as_last, out_last, 1e-3f, 1e-4f, "cuda_conv2d_layouts_agree");
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

#endif  // NNOPS_HAS_CUDA
