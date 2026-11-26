/// @file test_cuda_conv3d.cpp
/// @brief CUDA Conv3D against the CPU backend.
///
/// Every case builds one activation/filter pair, runs the operator on both
/// backends over the same bytes, and compares. The CPU route is the oracle, so
/// the output size this geometry produces, the bias placement, the padding ring
/// and add_to's order are pinned by the operator itself rather than by a
/// hand-written reference.
///
/// The activation is NDHWC [N, ID, IH, IW, IC] and the filter DHWIO
/// [KD, KH, KW, IC/G, OC] — the orders validate() pairs, one channel-last and
/// one not, which is the case the kernel's stride walks exist for.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/conv3d.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/cuda_test_helper.hpp"

#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

struct C3Case {
    int64_t n = 1, id = 5, ih = 6, iw = 7, ic = 8, oc = 16;
    int64_t kd = 3, kh = 3, kw = 3;
    int64_t sd = 1, sh = 1, sw = 1;
    int64_t dd = 1, dh = 1, dw = 1;
    int64_t pd = 1, ph = 1, pw = 1;
    int64_t groups = 1;
    DataType dtype = DataType::f32;
    bool use_bias = false;
    bool add_to = false;
    bool padded = false;
    Epilogue epilogue{};
};

void run_case(const C3Case& c, const char* what)
{
    Conv3DAttributes attrs;
    attrs.kernel_size = {c.kd, c.kh, c.kw};
    attrs.stride = {c.sd, c.sh, c.sw};
    attrs.dilation = {c.dd, c.dh, c.dw};
    attrs.padding = {c.pd, c.ph, c.pw};
    attrs.groups = c.groups;
    attrs.epilogue = c.epilogue;
    attrs.add_to = c.add_to;

    const int64_t icg = c.ic / c.groups;
    HostTensor act = make_host_random(std::vector<int64_t>{c.n, c.id, c.ih, c.iw, c.ic}, c.dtype,
                                      false, 11, -0.5f, 0.5f, TensorLayout::NDHWC);
    HostTensor wgt = make_host_random(std::vector<int64_t>{c.kd, c.kh, c.kw, icg, c.oc}, c.dtype,
                                      false, 13, -0.25f, 0.25f, TensorLayout::DHWIO);
    HostTensor bias;
    if (c.use_bias) {
        bias = make_host_random(std::vector<int64_t>{c.oc}, c.dtype, false, 17, -0.25f, 0.25f);
    }

    std::vector<TensorDesc> descs = {act.view.desc(), wgt.view.desc()};
    if (c.use_bias) {
        descs.push_back(bias.view.desc());
    }
    auto op_cpu = Conv3D::create(attrs, Backend::CPU);
    const std::vector<TensorDesc> outs = op_cpu->getOutputTensorDesc(descs);
    NNOPS_EXPECT_TRUE(!outs.empty());
    const TensorDesc od = outs.front();

    const auto seed = [&](uint64_t s) {
        return c.add_to ? make_host_random(od.dims, od.dtype, c.padded, s, 0.0f, 0.5f,
                                           TensorLayout::NDHWC)
                        : make_host_zeroed(od.dims, od.dtype, c.padded, TensorLayout::NDHWC);
    };
    HostTensor out_ref = seed(19);
    HostTensor out_dev = seed(19);

    std::vector<TensorView> cpu_ins = {act.view, wgt.view};
    if (c.use_bias) {
        cpu_ins.push_back(bias.view);
    }
    cpu_run(*op_cpu, out_ref.view, cpu_ins);

    auto op_dev = Conv3D::create(attrs, Backend::CUDA);
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

NNOPS_TEST(cuda_conv3d_patch_embed)
{
    // The pose a vision transformer's 3-D patch embedding runs: a k3s1p1 filter
    // over three or four input channels, one tap stride.
    for (DataType dt : {DataType::f32, DataType::f16}) {
        C3Case c;
        c.ic = 4;
        c.oc = 24;
        c.dtype = dt;
        run_case(c, "cuda_conv3d_k3p1");
    }

    C3Case c;
    c.ic = 3;
    c.oc = 12;
    c.kd = c.kh = c.kw = 2;
    c.pd = c.ph = c.pw = 0;
    c.sd = c.sh = c.sw = 2;
    run_case(c, "cuda_conv3d_k2s2");
}

NNOPS_TEST(cuda_conv3d_per_axis)
{
    // Each spatial axis carries its own kernel and stride, which is what the tap
    // decode has to keep straight.
    C3Case c;
    c.kd = 2;
    c.kh = 3;
    c.kw = 1;
    c.pd = 0;
    c.ph = 1;
    c.pw = 0;
    c.sd = 1;
    c.sh = 2;
    c.sw = 1;
    run_case(c, "cuda_conv3d_per_axis");

    C3Case d;
    d.dd = 2;
    d.dh = 1;
    d.dw = 3;
    d.pd = 2;
    d.ph = 1;
    d.pw = 3;
    run_case(d, "cuda_conv3d_dilated");
}

NNOPS_TEST(cuda_conv3d_groups)
{
    C3Case c;
    c.ic = 16;
    c.oc = 32;
    c.groups = 4;
    run_case(c, "cuda_conv3d_groups");

    // Depthwise: one channel a group on both sides.
    C3Case dw;
    dw.ic = 16;
    dw.oc = 16;
    dw.groups = 16;
    dw.id = dw.ih = dw.iw = 4;
    for (DataType dt : {DataType::f32, DataType::f16}) {
        dw.dtype = dt;
        run_case(dw, "cuda_conv3d_depthwise");
    }
}

NNOPS_TEST(cuda_conv3d_epilogue)
{
    Epilogue relu;
    relu.type = EpilogueActivateType::Relu;
    Epilogue silu;
    silu.type = EpilogueActivateType::Silu;

    for (const Epilogue& ep : {relu, silu}) {
        C3Case c;
        c.use_bias = true;
        c.add_to = true;
        c.epilogue = ep;
        run_case(c, "cuda_conv3d_epilogue");
    }

    // The clip rides with a non-foldable activation: the CPU's conv family
    // folds Relu/Relu6 into an accumulator clamp and skips its epilogue pass, so
    // a bound set beside one of those — or beside the identity — never reaches
    // the output, and CUDA's unconditional clamp would disagree with it.
    Epilogue clipped;
    clipped.type = EpilogueActivateType::LeakyRelu;
    clipped.alpha = 0.1f;
    clipped.min_clip = -0.1f;
    clipped.max_clip = 0.2f;
    C3Case c;
    c.use_bias = true;
    c.epilogue = clipped;
    run_case(c, "cuda_conv3d_clip");
}

NNOPS_TEST(cuda_conv3d_tiles)
{
    // More than one block on both axes of the tile: M past 64 voxels and OCG
    // past 64 channels.
    C3Case c;
    c.id = c.ih = c.iw = 5;
    c.ic = 8;
    c.oc = 80;
    c.use_bias = true;
    run_case(c, "cuda_conv3d_tiles");

    // A padded output: every row the kernel stores through carries a pitch.
    C3Case p;
    p.id = p.ih = p.iw = 4;
    p.ic = 5;
    p.oc = 40;
    p.padded = true;
    run_case(p, "cuda_conv3d_padded_out");
}

NNOPS_TEST(cuda_conv3d_batched)
{
    C3Case c;
    c.n = 2;
    c.id = 3;
    c.ih = 4;
    c.iw = 5;
    c.ic = 6;
    c.oc = 18;
    c.groups = 3;
    c.use_bias = true;
    run_case(c, "cuda_conv3d_batched");
}

#endif // NNOPS_HAS_CUDA
