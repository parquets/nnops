/// @file test_cuda_depthwise_transpose_conv.cpp
/// @brief CUDA DepthwiseTransposeConv2D against the CPU backend.
///
/// The operator has its own CUDA entry, so the cases exercise it directly
/// rather than through the grouped TransposeConv2D that used to be the only way
/// in. The CPU route is the oracle: it already pins the output size (with
/// output_padding), the bias placement and the two add_to orderings, and the
/// comparison is the same tolerance-based one the other convolution suites use.
///
/// The cases take the 16 B channel lane, plus cases whose channel count is not
/// a whole vector so the entry's scalar fallback runs. Both filter forms
/// appear: the [C][1][KH][KW] the grouped route hands over, and the [KH][KW][C]
/// an NHWC graph carries.
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
    bool use_bias = false;
    bool add_to = false;
    bool pre_act = false;
    /// NHWC activation carrying the [C][1][KH][KW] filter, the form the
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

    const TensorLayout layout = TensorLayout::NHWC;

    HostTensor act = make_host_random(std::vector<int64_t>{c.n, c.c, c.ih, c.iw},
                                      c.dtype, false, 23, -0.5f, 0.5f, layout);
    const bool hwcm = !c.filter_channel_first;
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
        return c.add_to ? make_host_random(od.dims, od.dtype, false, s, 0.0f, 0.5f,
                                           layout)
                        : make_host_zeroed(od.dims, od.dtype, false, layout);
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
// NHWC
// ============================================================

NNOPS_TEST(cuda_dwt_nhwc) {
    for (DataType dt : {DataType::f32, DataType::f16}) {
        DwCase c;
        c.c = 16; c.dtype = dt; c.use_bias = true;
        run_case(c, "cuda_dwt_last");
        c.sh = c.sw = 2;
        run_case(c, "cuda_dwt_last_stride2");
    }
}

NNOPS_TEST(cuda_dwt_nhwc_narrow) {
    // Channels that are not a whole 16 B vector and a row that is not aligned:
    // the entry's scalar fallback, not the 16 B lane.
    for (DataType dt : {DataType::f32, DataType::f16}) {
        for (int64_t ch : {int64_t(5), int64_t(7)}) {
            DwCase c;
            c.c = ch; c.ih = c.iw = 5; c.dtype = dt;
            run_case(c, "cuda_dwt_last_narrow");
        }
    }
}

NNOPS_TEST(cuda_dwt_nhwc_filter_form) {
    // The grouped route hands an NHWC activation its [C][1][KH][KW]
    // filter; the entry reads it in place rather than repacking.
    DwCase c;
    c.c = 12; c.ih = c.iw = 6; c.filter_channel_first = true;
    run_case(c, "cuda_dwt_last_cf_filter");
}

#endif  // NNOPS_HAS_CUDA
