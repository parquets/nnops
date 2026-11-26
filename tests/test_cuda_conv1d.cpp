/// @file test_cuda_conv1d.cpp
/// @brief CUDA Conv1D against the CPU backend.
///
/// Conv1D has no kernel of its own on either backend: it is a rank-4 lift onto
/// the 2D implicit GEMM, and a depthwise geometry takes the 2D op's routing to
/// the depthwise kernel. Both are the operator's own decision on the CPU, so
/// the CPU run is the oracle for both poses here.
///
/// The rank-3 view is NWC — the memory order only, the logical dims stay
/// [N, C, L] — and the filter is HWIO [K, IC/G, OC].
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/conv1d.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/cuda_test_helper.hpp"

#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

struct C1Case {
    int64_t n = 1, l = 12, ic = 8, oc = 16;
    int64_t k = 3, s = 1, d = 1, p = 1;
    int64_t groups = 1;
    DataType dtype = DataType::f32;
    bool use_bias = false;
    bool add_to = false;
    bool padded = false;
    Conv1DAttributes::AutoPad auto_pad = Conv1DAttributes::AutoPad::NOTSET;
    Epilogue epilogue{};
};

void run_case(const C1Case& c, const char* what)
{
    Conv1DAttributes attrs;
    attrs.kernel_size = c.k;
    attrs.stride = c.s;
    attrs.dilation = c.d;
    attrs.padding = c.p;
    attrs.groups = c.groups;
    attrs.epilogue = c.epilogue;
    attrs.add_to = c.add_to;
    attrs.auto_pad = c.auto_pad;

    const int64_t icg = c.ic / c.groups;
    HostTensor act = make_host_random(std::vector<int64_t>{c.n, c.l, c.ic}, c.dtype, c.padded, 17,
                                      -0.5f, 0.5f, TensorLayout::NHWC);
    HostTensor wgt = make_host_random(std::vector<int64_t>{c.k, icg, c.oc}, c.dtype, false, 19,
                                      -0.25f, 0.25f, TensorLayout::HWIO);
    HostTensor bias;
    if (c.use_bias) {
        bias = make_host_random(std::vector<int64_t>{c.oc}, c.dtype, false, 23, -0.25f, 0.25f);
    }

    std::vector<TensorDesc> descs = {act.view.desc(), wgt.view.desc()};
    if (c.use_bias) {
        descs.push_back(bias.view.desc());
    }
    auto op_cpu = Conv1D::create(attrs, Backend::CPU);
    const std::vector<TensorDesc> outs = op_cpu->getOutputTensorDesc(descs);
    NNOPS_EXPECT_TRUE(!outs.empty());
    const TensorDesc od = outs.front();

    const auto seed = [&](uint64_t s) {
        return c.add_to ? make_host_random(od.dims, od.dtype, false, s, 0.0f, 0.5f,
                                           TensorLayout::NHWC)
                        : make_host_zeroed(od.dims, od.dtype, false, TensorLayout::NHWC);
    };
    HostTensor out_ref = seed(29);
    HostTensor out_dev = seed(29);

    std::vector<TensorView> cpu_ins = {act.view, wgt.view};
    if (c.use_bias) {
        cpu_ins.push_back(bias.view);
    }
    cpu_run(*op_cpu, out_ref.view, cpu_ins);

    auto op_dev = Conv1D::create(attrs, Backend::CUDA);
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

} // anonymous namespace

NNOPS_TEST(cuda_conv1d_shapes)
{
    for (DataType dt : {DataType::f32, DataType::f16}) {
        C1Case c;
        c.dtype = dt;
        run_case(c, "cuda_conv1d");
        // A stride the kernel strides over, and a 1-tap filter (the pointwise
        // pose, which the 2D route hands to the dense GEMM).
        c.s = 2;
        c.p = 0;
        run_case(c, "cuda_conv1d_s2");
        c.k = 1;
        c.p = 0;
        run_case(c, "cuda_conv1d_k1");
        // Two taps with no padding: the only in-range position per output.
        c.k = 2;
        c.s = 1;
        run_case(c, "cuda_conv1d_k2");
    }
}

NNOPS_TEST(cuda_conv1d_dilated_padded)
{
    C1Case c;
    c.k = 3;
    c.d = 2;
    c.p = 2;
    c.l = 10;
    run_case(c, "cuda_conv1d_dilated");

    // A channel count the vector fills do not divide.
    C1Case d;
    d.ic = 6;
    d.oc = 12;
    d.k = 3;
    run_case(d, "cuda_conv1d_narrow");
}

NNOPS_TEST(cuda_conv1d_groups)
{
    C1Case c;
    c.ic = 16;
    c.oc = 32;
    c.groups = 4;
    run_case(c, "cuda_conv1d_groups");

    // Depthwise: one channel a group on both sides takes the 2D op's routing to
    // the depthwise kernel, which is what the lift has to preserve.
    C1Case dw;
    dw.ic = 16;
    dw.oc = 16;
    dw.groups = 16;
    dw.k = 3;
    dw.l = 9;
    for (DataType dt : {DataType::f32, DataType::f16}) {
        dw.dtype = dt;
        run_case(dw, "cuda_conv1d_depthwise");
    }
}

NNOPS_TEST(cuda_conv1d_epilogue)
{
    Epilogue relu6;
    relu6.type = EpilogueActivateType::Relu6;
    Epilogue silu;
    silu.type = EpilogueActivateType::Silu;

    for (const Epilogue& ep : {relu6, silu}) {
        C1Case c;
        c.use_bias = true;
        c.epilogue = ep;
        c.add_to = true;
        run_case(c, "cuda_conv1d_epilogue");
    }

    // The clamp bounds travel with the epilogue on every backend.
    // The clip rides with a non-foldable activation: the CPU's conv family
    // folds Relu/Relu6 into an accumulator clamp and skips its epilogue pass, so
    // a bound set beside one of those — or beside the identity — never reaches
    // the output, and CUDA's unconditional clamp would disagree with it.
    Epilogue clipped;
    clipped.type = EpilogueActivateType::LeakyRelu;
    clipped.alpha = 0.1f;
    clipped.min_clip = -0.1f;
    clipped.max_clip = 0.2f;
    C1Case c;
    c.epilogue = clipped;
    c.use_bias = true;
    run_case(c, "cuda_conv1d_clip");
}

NNOPS_TEST(cuda_conv1d_batched)
{
    C1Case c;
    c.n = 3;
    c.l = 17;
    c.ic = 12;
    c.oc = 20;
    c.k = 4;
    c.p = 2;
    run_case(c, "cuda_conv1d_batched");
}

NNOPS_TEST(cuda_conv1d_auto_pad)
{
    // auto_pad is resolved inside the kernel on this op — Conv1D::compute does
    // not rewrite the descriptor the way Conv2D's and Conv3D's do — so the lift
    // has to resolve it exactly as the CPU entry does.
    for (auto ap : {Conv1DAttributes::AutoPad::SAME_UPPER, Conv1DAttributes::AutoPad::SAME_LOWER,
                    Conv1DAttributes::AutoPad::VALID}) {
        C1Case c;
        c.k = 3;
        c.s = 2;
        c.p = 0;
        c.l = 11;
        c.auto_pad = ap;
        run_case(c, "cuda_conv1d_auto_pad");
    }
}

NNOPS_TEST(cuda_conv1d_padded_activation)
{
    // A pitched row: the lift keeps the pitch, so the gather still steps whole
    // L·C rows.
    C1Case c;
    c.n = 2;
    c.l = 9;
    c.ic = 6;
    c.oc = 10;
    c.padded = true;
    run_case(c, "cuda_conv1d_padded");
}

#endif // NNOPS_HAS_CUDA
