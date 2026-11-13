/// @file test_cuda_depthwise_transpose_conv.cpp
/// @brief CUDA DepthwiseTransposeConv2D against the CPU backend.
///
/// The operator has its own CUDA entry, so the cases exercise it directly
/// rather than through the grouped TransposeConv2D that used to be the only way
/// in. The CPU route is the oracle: it already pins the output size (with
/// output_padding), the bias placement and the two add_to orderings, and the
/// comparison is the same tolerance-based one the other convolution suites use.
///
/// The cases split along the dispatch. Channel-first geometry takes the
/// residue-class lane kernels, with a 32 B row pitch on the activation and on
/// the output in one case. Channel-last takes the 16 B channel lane, plus one
/// case whose channel count is not a whole vector so the entry's scalar
/// fallback runs. Both filter forms appear: the [C][1][KH][KW] the grouped
/// route hands over, and the [KH][KW][C] a channel-last graph carries.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/depthwise_transpose_conv.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/cuda_test_helper.hpp"

#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

struct DwCase {
    int64_t n = 1, c = 17, ih = 9, iw = 9;
    int64_t kh = 3, kw = 3;
    int64_t sh = 1, sw = 1;
    int64_t dh = 1, dw = 1;
    int64_t ph = 1, pw = 1;
    int64_t oph = 0, opw = 0;
    DataType dtype = DataType::f32;
    bool channel_last = false;
    bool use_bias = false;
    bool add_to = false;
    bool pre_act = false;
    /// Channel-first only: a 32 B row pitch on the activation and the output.
    bool padded = false;
    /// Channel-last activation carrying the [C][1][KH][KW] filter, the form the
    /// grouped TransposeConv2D passes through unconverted.
    bool filter_channel_first = false;
    Epilogue epilogue{};
};

/// Run one case on both backends over the same bytes and compare.
void run_case(const DwCase& c, const char* what)
{
    DepthwiseTransposeConv2DAttributes attrs;
    attrs.kernel_size = {c.kh, c.kw};
    attrs.stride = {c.sh, c.sw};
    attrs.dilation = {c.dh, c.dw};
    attrs.padding = {c.ph, c.pw};
    attrs.output_padding = {c.oph, c.opw};
    attrs.add_to = c.add_to;
    attrs.residual_pre_activation = c.pre_act;

    const TensorLayout layout =
        c.channel_last ? TensorLayout::NHWC : TensorLayout::NCHW;
    const bool pad = c.padded && !c.channel_last;

    HostTensor act = make_host_random(std::vector<int64_t>{c.n, c.c, c.ih, c.iw},
                                      c.dtype, pad, 23, -0.5f, 0.5f, layout);
    const bool hwcm = c.channel_last && !c.filter_channel_first;
    HostTensor wgt = hwcm
        ? make_host_random(std::vector<int64_t>{c.kh, c.kw, c.c}, c.dtype, false,
                           29, -0.25f, 0.25f, TensorLayout::HWCM)
        : make_host_random(std::vector<int64_t>{c.c, 1, c.kh, c.kw}, c.dtype,
                           false, 29, -0.25f, 0.25f);
    HostTensor bias;
    if (c.use_bias) {
        bias = make_host_random(std::vector<int64_t>{c.c}, c.dtype, false, 31,
                                -0.25f, 0.25f);
    }

    std::vector<TensorDesc> descs = {act.view.desc(), wgt.view.desc()};
    if (c.use_bias) { descs.push_back(bias.view.desc()); }

    auto op_cpu = DepthwiseTransposeConv2D::create(attrs, Backend::CPU);
    const std::vector<TensorDesc> outs = op_cpu->getOutputTensorDesc(descs);
    NNOPS_EXPECT_TRUE(!outs.empty());
    const TensorDesc od = outs.front();

    // The residual seed is the same on both sides; a plain case starts at zero.
    const auto seed = [&](uint64_t s) {
        return c.add_to ? make_host_random(od.dims, od.dtype, pad, s, 0.0f, 0.5f,
                                           layout)
                        : make_host_zeroed(od.dims, od.dtype, pad, layout);
    };
    HostTensor out_ref = seed(37);
    HostTensor out_dev = seed(37);

    std::vector<TensorView> cpu_ins = {act.view, wgt.view};
    if (c.use_bias) { cpu_ins.push_back(bias.view); }
    cpu_run(*op_cpu, out_ref.view, cpu_ins);

    auto op_dev = DepthwiseTransposeConv2D::create(attrs, Backend::CUDA);
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
}

}  // namespace

// ============================================================
// Channel-first
// ============================================================

NNOPS_TEST(cuda_dwt_basic) {
    for (DataType dt : {DataType::f32, DataType::f16}) {
        DwCase c;
        c.dtype = dt;
        run_case(c, "cuda_dwt_basic");
        c.use_bias = true;
        run_case(c, "cuda_dwt_bias");
    }
}

NNOPS_TEST(cuda_dwt_geometry) {
    {   // Stride 2 with output_padding: the residue classes and the extra column.
        DwCase c;
        c.c = 12; c.ih = c.iw = 7; c.sh = c.sw = 2; c.oph = 1; c.opw = 1;
        run_case(c, "cuda_dwt_stride2");
    }
    {   // A dilated 3x3, so the taps move within the window.
        DwCase c;
        c.c = 9; c.ih = c.iw = 8; c.dh = c.dw = 2;
        run_case(c, "cuda_dwt_dilated");
    }
    {   // 1x1 kernel: the interior window is a single tap.
        DwCase c;
        c.c = 20; c.kh = c.kw = 1; c.ph = c.pw = 0;
        run_case(c, "cuda_dwt_1x1");
    }
    {   // An even kernel on an even stride, no padding.
        DwCase c;
        c.c = 6; c.ih = c.iw = 6; c.kh = c.kw = 4; c.sh = c.sw = 2;
        c.ph = c.pw = 0;
        run_case(c, "cuda_dwt_even");
    }
}

NNOPS_TEST(cuda_dwt_pitched) {
    // A 32 B pitch on both the activation and the output.
    DwCase c;
    c.c = 10; c.ih = c.iw = 7; c.padded = true;
    run_case(c, "cuda_dwt_pitched");
    c.use_bias = true;
    c.sh = c.sw = 2;
    run_case(c, "cuda_dwt_pitched_stride2");
}

NNOPS_TEST(cuda_dwt_residual) {
    // A positive residual survives a ReLU either way; a negative one tells the
    // pre- and post-activation orders apart, so the seed has mixed signs and
    // the epilogue is a ReLU.
    for (bool pre : {false, true}) {
        DwCase c;
        c.c = 8; c.ih = c.iw = 6; c.add_to = true; c.pre_act = pre;
        c.epilogue.type = EpilogueActivateType::Relu;
        run_case(c, "cuda_dwt_residual");
    }
}

// ============================================================
// Channel-last
// ============================================================

NNOPS_TEST(cuda_dwt_channel_last) {
    for (DataType dt : {DataType::f32, DataType::f16}) {
        DwCase c;
        c.c = 16; c.dtype = dt; c.channel_last = true; c.use_bias = true;
        run_case(c, "cuda_dwt_last");
        c.sh = c.sw = 2;
        run_case(c, "cuda_dwt_last_stride2");
    }
}

NNOPS_TEST(cuda_dwt_channel_last_narrow) {
    // Channels that are not a whole 16 B vector and a row that is not aligned:
    // the entry's scalar fallback, not the 16 B lane.
    for (DataType dt : {DataType::f32, DataType::f16}) {
        for (int64_t ch : {int64_t(5), int64_t(7)}) {
            DwCase c;
            c.c = ch; c.ih = c.iw = 5; c.dtype = dt; c.channel_last = true;
            run_case(c, "cuda_dwt_last_narrow");
        }
    }
}

NNOPS_TEST(cuda_dwt_channel_last_filter_form) {
    // The grouped route hands a channel-last activation its [C][1][KH][KW]
    // filter; the entry reads it in place rather than repacking.
    DwCase c;
    c.c = 12; c.ih = c.iw = 6; c.channel_last = true; c.filter_channel_first = true;
    run_case(c, "cuda_dwt_last_cf_filter");
}

NNOPS_TEST(cuda_dwt_wide) {
    // Past one channel tile and several batches.
    DwCase c;
    c.n = 3; c.c = 40; c.ih = c.iw = 11;
    run_case(c, "cuda_dwt_wide");
}

#endif  // NNOPS_HAS_CUDA
