/// @file test_cuda_transpose_conv2d.cpp
/// @brief CUDA TransposeConv2D against the CPU backend.
///
/// Every case builds one activation/filter pair, runs the operator on both
/// backends over the same bytes, and compares. The CPU route is the oracle, so
/// the output size this geometry produces (output_padding included), the bias
/// placement, the padding ring and the two add_to orderings are pinned by the
/// operator itself rather than by a hand-written reference.
///
/// The cases split along the dispatch. The residue-class GEMM takes any stride
/// and dilation with OC/G wide enough for an n tile. A depthwise group (one
/// channel on both sides) falls to the 16 B channel lane.
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
    bool use_bias = false;
    bool add_to = false;
    bool pre_act = false;
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

    const TensorLayout layout = TensorLayout::NHWC;
    // The filter is [IC, OC/G, KH, KW].
    const int64_t ocg = c.oc / c.groups;

    HostTensor act = make_host_random(std::vector<int64_t>{c.n, c.ic, c.ih, c.iw},
                                      c.dtype, false, 23, -0.5f, 0.5f, layout);
    HostTensor wgt = make_host_random(
        std::vector<int64_t>{c.ic, ocg, c.kh, c.kw}, c.dtype, false, 29, -0.25f,
        0.25f, TensorLayout::Dense);
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

    const auto seed = [&](uint64_t s) {
        return c.add_to ? make_host_random(od.dims, od.dtype, false, s, 0.0f,
                                           0.5f, layout)
                        : make_host_zeroed(od.dims, od.dtype, false, layout);
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

NNOPS_TEST(cuda_transpose_conv2d_depthwise) {
    // One channel on both sides of every group: the 16 B channel lane.
    for (DataType dt : {DataType::f32, DataType::f16}) {
        TConvCase c;
        c.ic = 16; c.oc = 16; c.groups = 16; c.dtype = dt;
        c.ih = c.iw = 9; c.ph = c.pw = 1;
        run_case(c, "cuda_tconv_depthwise");
        c.sh = c.sw = 2;
        run_case(c, "cuda_tconv_depthwise");
    }
}

// ============================================================
// NHWC
// ============================================================

NNOPS_TEST(cuda_transpose_conv2d) {
    for (DataType dt : {DataType::f32, DataType::f16}) {
        TConvCase c;
        c.ic = 16; c.oc = 32; c.dtype = dt;
        c.ih = c.iw = 11;
        run_case(c, "cuda_tconv");
        c.sh = c.sw = 2;
        run_case(c, "cuda_tconv");
        c.kh = c.kw = 4;
        c.oph = 1; c.opw = 1;
        run_case(c, "cuda_tconv");
    }
    {   // A channel count the 16 B lane does not divide.
        TConvCase c;
        c.ic = 6; c.oc = 12; c.ih = c.iw = 8;
        run_case(c, "cuda_tconv");
    }
}

NNOPS_TEST(cuda_transpose_conv2d_groups) {
    TConvCase c;
    c.ic = 16; c.oc = 32; c.groups = 4;
    c.sh = c.sw = 2;
    run_case(c, "cuda_tconv_channel_last_groups");

    TConvCase d;
    d.ic = 32; d.oc = 128; d.groups = 2;
    d.ih = d.iw = 10; d.dh = d.dw = 2; d.ph = d.pw = 2;
    run_case(d, "cuda_tconv_channel_last_groups");
}

// ============================================================
// Epilogue and add_to
// ============================================================

NNOPS_TEST(cuda_transpose_conv2d_epilogue) {
    Epilogue relu;
    relu.type = EpilogueActivateType::Relu;
    Epilogue silu;
    silu.type = EpilogueActivateType::Silu;

    for (bool pre : {false, true}) {
        for (const Epilogue& ep : {relu, silu}) {
            TConvCase c;
            c.ic = 8; c.oc = 16; c.use_bias = true;
            c.add_to = true; c.pre_act = pre; c.epilogue = ep;
            run_case(c, "cuda_tconv_epilogue");
        }
    }
}

#endif  // NNOPS_HAS_CUDA
