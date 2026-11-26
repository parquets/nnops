/// @file test_cuda_transpose_conv3d.cpp
/// @brief CUDA TransposeConv3D against the CPU backend.
///
/// Every case builds one activation/filter pair, runs the operator on both
/// backends over the same bytes, and compares. The CPU route is the oracle, so
/// the output size this geometry produces (output_padding included), the bias
/// placement and the two add_to orderings are pinned by the operator itself.
///
/// The activation is NDHWC [N, ID, IH, IW, IC] and the filter is
/// [IC, OC/G, KD, KH, KW] — input channel outermost, the opposite of Conv3D's
/// filter, which is what makes the group offset land on a different axis.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/transpose_conv3d.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/cuda_test_helper.hpp"

#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

struct T3Case {
    int64_t n = 1, id = 4, ih = 5, iw = 6, ic = 8, oc = 16;
    int64_t kd = 3, kh = 3, kw = 3;
    int64_t sd = 1, sh = 1, sw = 1;
    int64_t dd = 1, dh = 1, dw = 1;
    int64_t pd = 1, ph = 1, pw = 1;
    int64_t opd = 0, oph = 0, opw = 0;
    int64_t groups = 1;
    DataType dtype = DataType::f32;
    bool use_bias = false;
    bool add_to = false;
    bool pre_act = false;
    bool padded = false;
    Epilogue epilogue{};
};

void run_case(const T3Case& c, const char* what)
{
    TransposeConv3DAttributes attrs;
    attrs.kernel_size = {c.kd, c.kh, c.kw};
    attrs.stride = {c.sd, c.sh, c.sw};
    attrs.dilation = {c.dd, c.dh, c.dw};
    attrs.padding = {c.pd, c.ph, c.pw};
    attrs.output_padding = {c.opd, c.oph, c.opw};
    attrs.groups = c.groups;
    attrs.epilogue = c.epilogue;
    attrs.add_to = c.add_to;
    attrs.residual_pre_activation = c.pre_act;

    const int64_t ocg = c.oc / c.groups;
    HostTensor act = make_host_random(std::vector<int64_t>{c.n, c.id, c.ih, c.iw, c.ic}, c.dtype,
                                      false, 31, -0.5f, 0.5f, TensorLayout::NDHWC);
    HostTensor wgt = make_host_random(std::vector<int64_t>{c.ic, ocg, c.kd, c.kh, c.kw}, c.dtype,
                                      false, 37, -0.25f, 0.25f);
    HostTensor bias;
    if (c.use_bias) {
        bias = make_host_random(std::vector<int64_t>{c.oc}, c.dtype, false, 41, -0.25f, 0.25f);
    }

    std::vector<TensorDesc> descs = {act.view.desc(), wgt.view.desc()};
    if (c.use_bias) {
        descs.push_back(bias.view.desc());
    }
    auto op_cpu = TransposeConv3D::create(attrs, Backend::CPU);
    const std::vector<TensorDesc> outs = op_cpu->getOutputTensorDesc(descs);
    NNOPS_EXPECT_TRUE(!outs.empty());
    const TensorDesc od = outs.front();

    const auto seed = [&](uint64_t s) {
        return c.add_to ? make_host_random(od.dims, od.dtype, c.padded, s, 0.0f, 0.5f,
                                           TensorLayout::NDHWC)
                        : make_host_zeroed(od.dims, od.dtype, c.padded, TensorLayout::NDHWC);
    };
    HostTensor out_ref = seed(43);
    HostTensor out_dev = seed(43);

    std::vector<TensorView> cpu_ins = {act.view, wgt.view};
    if (c.use_bias) {
        cpu_ins.push_back(bias.view);
    }
    cpu_run(*op_cpu, out_ref.view, cpu_ins);

    auto op_dev = TransposeConv3D::create(attrs, Backend::CUDA);
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

NNOPS_TEST(cuda_transpose_conv3d_smallest)
{
    // One voxel, one tap, no stride: the smallest product there is.
    T3Case c;
    c.id = c.ih = c.iw = 1;
    c.ic = 4;
    c.oc = 4;
    c.kd = c.kh = c.kw = 1;
    c.pd = c.ph = c.pw = 0;
    run_case(c, "cuda_tconv3d_smallest");
}

NNOPS_TEST(cuda_transpose_conv3d_voxel_rows)
{
    // Two voxels a side, still one tap: the output is strided.
    T3Case c;
    c.id = c.ih = c.iw = 2;
    c.ic = 4;
    c.oc = 4;
    c.kd = c.kh = c.kw = 1;
    c.pd = c.ph = c.pw = 0;
    run_case(c, "cuda_tconv3d_voxel_rows");
}

NNOPS_TEST(cuda_transpose_conv3d_single_tap_stride)
{
    // One tap, stride two: only the aligned outputs contribute.
    T3Case c;
    c.id = c.ih = c.iw = 2;
    c.ic = 4;
    c.oc = 4;
    c.kd = c.kh = c.kw = 1;
    c.pd = c.ph = c.pw = 0;
    c.sd = c.sh = c.sw = 2;
    run_case(c, "cuda_tconv3d_single_tap_stride");
}

NNOPS_TEST(cuda_transpose_conv3d_stride)
{
    for (DataType dt : {DataType::f32, DataType::f16}) {
        T3Case c;
        c.dtype = dt;
        run_case(c, "cuda_tconv3d_s1");
        // A stride leaves output voxels with fewer contributing taps; the
        // residue test is what picks them.
        c.sd = c.sh = c.sw = 2;
        c.pd = c.ph = c.pw = 0;
        run_case(c, "cuda_tconv3d_s2");
        c.opd = c.oph = c.opw = 1;
        run_case(c, "cuda_tconv3d_s2_opad");
        // A stride on one axis only, which mixes the two regimes.
        c.sd = 1;
        c.sh = 2;
        c.sw = 3;
        c.pd = c.ph = c.pw = 1;
        c.opd = c.oph = c.opw = 0;
        run_case(c, "cuda_tconv3d_mixed_stride");
    }
}

NNOPS_TEST(cuda_transpose_conv3d_filters)
{
    T3Case c;
    c.kd = c.kh = c.kw = 2;
    c.pd = c.ph = c.pw = 0;
    run_case(c, "cuda_tconv3d_k2");

    T3Case d;
    d.kd = 4;
    d.kh = 2;
    d.kw = 1;
    d.pd = 3;
    d.ph = 0;
    d.pw = 0;
    d.dd = 2;
    run_case(d, "cuda_tconv3d_dilated");
}

NNOPS_TEST(cuda_transpose_conv3d_groups)
{
    T3Case c;
    c.ic = 16;
    c.oc = 32;
    c.groups = 4;
    run_case(c, "cuda_tconv3d_groups");

    T3Case dw;
    dw.ic = 16;
    dw.oc = 16;
    dw.groups = 16;
    dw.id = dw.ih = dw.iw = 3;
    for (DataType dt : {DataType::f32, DataType::f16}) {
        dw.dtype = dt;
        run_case(dw, "cuda_tconv3d_depthwise");
    }
}

NNOPS_TEST(cuda_transpose_conv3d_epilogue)
{
    Epilogue relu;
    relu.type = EpilogueActivateType::Relu;
    Epilogue silu;
    silu.type = EpilogueActivateType::Silu;

    // Both add_to orders: the residual after the activation, and folded in
    // before it. The operator supports both on this route.
    for (bool pre : {false, true}) {
        for (const Epilogue& ep : {relu, silu}) {
            T3Case c;
            c.use_bias = true;
            c.add_to = true;
            c.pre_act = pre;
            c.epilogue = ep;
            run_case(c, "cuda_tconv3d_epilogue");
        }
    }

    // The clip is the second half of the epilogue contract, and it composes
    // with the activation rather than being replaced by it: on its own, beside
    // the two activations the CPU folds into an accumulator clamp, and beside a
    // non-foldable one. All four must reach the same output.
    for (auto type : {EpilogueActivateType::None, EpilogueActivateType::Relu,
                      EpilogueActivateType::Relu6, EpilogueActivateType::LeakyRelu}) {
        Epilogue clipped;
        clipped.type = type;
        clipped.alpha = 0.1f;
        clipped.min_clip = -0.1f;
        clipped.max_clip = 0.2f;
        T3Case k;
        k.use_bias = true;
        k.epilogue = clipped;
        run_case(k, "cuda_tconv3d_clip");
    }
    // The same bounds on a grouped geometry, whose group loop is a different
    // walk of the same finish.
    {
        Epilogue clipped;
        clipped.type = EpilogueActivateType::Relu;
        clipped.min_clip = -0.1f;
        clipped.max_clip = 0.2f;
        T3Case g;
        g.ic = 16;
        g.oc = 32;
        g.groups = 4;
        g.epilogue = clipped;
        run_case(g, "cuda_tconv3d_clip_groups");
    }

}

NNOPS_TEST(cuda_transpose_conv3d_tiles)
{
    // More than one block on both tile axes, and a padded output row.
    T3Case c;
    c.id = c.ih = c.iw = 4;
    c.oc = 80;
    c.use_bias = true;
    run_case(c, "cuda_tconv3d_tiles");

    T3Case p;
    p.id = p.ih = p.iw = 3;
    p.ic = 5;
    p.oc = 40;
    p.padded = true;
    run_case(p, "cuda_tconv3d_padded_out");
}

NNOPS_TEST(cuda_transpose_conv3d_batched)
{
    T3Case c;
    c.n = 2;
    c.id = 2;
    c.ih = 3;
    c.iw = 4;
    c.ic = 6;
    c.oc = 18;
    c.groups = 3;
    c.sd = c.sh = c.sw = 2;
    c.pd = c.ph = c.pw = 0;
    c.use_bias = true;
    run_case(c, "cuda_tconv3d_batched");
}

#endif // NNOPS_HAS_CUDA
