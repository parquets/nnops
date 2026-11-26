/// @file bench_cuda_conv3d.cpp
/// @brief CUDA Conv3D and TransposeConv3D throughput.
///
/// M is the output voxels, N the output channels of one group, K the reduction
/// IC/G·KD·KH·KW — the same product in both directions, which is why they share
/// a kernel. GFLOPS is 2·batch·OC·OD·OH·OW·(IC/G)·KD·KH·KW, the MACs actually
/// performed; a transposed convolution's zero-divisible taps are multiply-free
/// in principle but this kernel's gather still walks them, so its row counts the
/// full K and its number is a floor.
///
/// The layers are the ones a vision transformer's patch embedding runs (a 3-D
/// k3s1p1 over a small channel count, where K is tiny and the tile cannot fill),
/// a video-model 3-D convolution, and the matching upsampling transposes.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/conv3d.hpp"
#include "nnops/ops/transpose_conv3d.hpp"
#include "common/bench_harness.hpp"
#include "common/cuda_bench_helper.hpp"
#include "common/cuda_test_helper.hpp"

#include <cstdio>
#include <string>
#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

struct C3Bench {
    int64_t n = 1, id = 16, ih = 16, iw = 16, ic = 8, oc = 64;
    int64_t kd = 3, kh = 3, kw = 3;
    int64_t sd = 1, sh = 1, sw = 1;
    int64_t pd = 1, ph = 1, pw = 1;
    int64_t groups = 1;
    DataType dtype = DataType::f32;

    int64_t span(int64_t i, int64_t k, int64_t s, int64_t d, int64_t p) const
    {
        return (i + 2 * p - d * (k - 1) - 1) / s + 1;
    }
    int64_t od() const { return span(id, kd, sd, 1, pd); }
    int64_t oh() const { return span(ih, kh, sh, 1, ph); }
    int64_t ow() const { return span(iw, kw, sw, 1, pw); }
    double flops() const
    {
        return 2.0 * double(n) * double(oc) * double(od()) * double(oh()) * double(ow()) *
               double(ic / groups) * double(kd) * double(kh) * double(kw);
    }
};

std::string c3_geometry(const C3Bench& c, const char* tag)
{
    char buf[192];
    std::snprintf(buf, sizeof(buf), "%s n%lld %lldx%lldx%lldx%lld k%lld s%lld%s",
                  tag, (long long)c.n, (long long)c.id, (long long)c.ih, (long long)c.iw,
                  (long long)c.ic, (long long)c.kd, (long long)c.sd,
                  c.groups > 1 ? " g" : "");
    std::string s(buf);
    if (c.groups > 1) {
        s += std::to_string(c.groups);
    }
    s += c.dtype == DataType::f16 ? " f16" : " f32";
    return s;
}

double run_conv3d(const C3Bench& c, bool print = true)
{
    Conv3DAttributes attrs;
    attrs.kernel_size = {c.kd, c.kh, c.kw};
    attrs.stride = {c.sd, c.sh, c.sw};
    attrs.padding = {c.pd, c.ph, c.pw};
    attrs.groups = c.groups;

    HostTensor act = make_host_random(std::vector<int64_t>{c.n, c.id, c.ih, c.iw, c.ic}, c.dtype,
                                      false, 23, -0.5f, 0.5f, TensorLayout::NDHWC);
    HostTensor wgt = make_host_random(
        std::vector<int64_t>{c.kd, c.kh, c.kw, c.ic / c.groups, c.oc}, c.dtype, false, 29,
        -0.25f, 0.25f, TensorLayout::DHWIO);
    HostTensor out = make_host_zeroed(
        std::vector<int64_t>{c.n, c.od(), c.oh(), c.ow(), c.oc}, c.dtype, false,
        TensorLayout::NDHWC);

    DeviceTensor dact = to_device(act);
    DeviceTensor dwgt = to_device(wgt);
    DeviceTensor dout = to_device(out);

    auto op = Conv3D::create(attrs, Backend::CUDA);
    const TensorView ins[] = {dact.view, dwgt.view};
    TensorView o = dout.view;
    const double ms = cuda_median_ms([&] { cuda_launch(*op, o, ins); }, 3, 11);
    if (print) {
        print_cuda_row("conv3d", c3_geometry(c, "ndhwc"), ms, cuda_gflops(ms, c.flops()));
    }
    return cuda_gflops(ms, c.flops());
}

/// The transposed pose: the same product with o + pad − k·dil divided by the
/// stride. The output size is the operator's formula, so output_padding is 0.
double run_tconv3d(const C3Bench& c, bool print = true)
{
    TransposeConv3DAttributes attrs;
    attrs.kernel_size = {c.kd, c.kh, c.kw};
    attrs.stride = {c.sd, c.sh, c.sw};
    attrs.padding = {c.pd, c.ph, c.pw};
    attrs.groups = c.groups;

    const int64_t od = (c.id - 1) * c.sd - 2 * c.pd + (c.kd - 1) + 1;
    const int64_t oh = (c.ih - 1) * c.sh - 2 * c.ph + (c.kh - 1) + 1;
    const int64_t ow = (c.iw - 1) * c.sw - 2 * c.pw + (c.kw - 1) + 1;

    HostTensor act = make_host_random(std::vector<int64_t>{c.n, c.id, c.ih, c.iw, c.ic}, c.dtype,
                                      false, 31, -0.5f, 0.5f, TensorLayout::NDHWC);
    HostTensor wgt = make_host_random(
        std::vector<int64_t>{c.ic, c.oc / c.groups, c.kd, c.kh, c.kw}, c.dtype, false, 37,
        -0.25f, 0.25f);
    HostTensor out = make_host_zeroed(
        std::vector<int64_t>{c.n, od, oh, ow, c.oc}, c.dtype, false, TensorLayout::NDHWC);

    DeviceTensor dact = to_device(act);
    DeviceTensor dwgt = to_device(wgt);
    DeviceTensor dout = to_device(out);

    auto op = TransposeConv3D::create(attrs, Backend::CUDA);
    const TensorView ins[] = {dact.view, dwgt.view};
    TensorView o = dout.view;
    const double ms = cuda_median_ms([&] { cuda_launch(*op, o, ins); }, 3, 11);
    const double flops = 2.0 * double(c.n) * double(c.oc) * double(od * oh * ow) *
                         double(c.ic / c.groups) * double(c.kd * c.kh * c.kw);
    if (print) {
        print_cuda_row("tconv3d", c3_geometry(c, "ndhwc"), ms, cuda_gflops(ms, flops));
    }
    return cuda_gflops(ms, flops);
}

} // anonymous namespace

NNOPS_BENCH(cuda_conv3d)
{
    print_cuda_header("CUDA Conv3D (NDHWC x DHWIO)");

    // A patch embedding: K = 3·27, so the tile is the reduction.
    C3Bench embed;
    embed.ic = 8;
    embed.oc = 96;
    embed.id = embed.ih = embed.iw = 16;
    run_conv3d(embed);
    embed.dtype = DataType::f16;
    run_conv3d(embed);

    // A video-shaped convolution: enough K and N to fill the tile.
    C3Bench video;
    video.ic = 128;
    video.oc = 128;
    video.id = 8;
    video.ih = video.iw = 32;
    run_conv3d(video);
    video.dtype = DataType::f16;
    run_conv3d(video);

    // A grouped layer, and the depthwise pose a graph would reroute.
    C3Bench g = video;
    g.groups = 4;
    run_conv3d(g);
    C3Bench dw = video;
    dw.ic = dw.oc = 64;
    dw.groups = 64;
    run_conv3d(dw);
}

NNOPS_BENCH(cuda_transpose_conv3d)
{
    print_cuda_header("CUDA TransposeConv3D (NDHWC x [IC][OC/G][KD][KH][KW])");

    // A 2x upsampling stage of a video decoder.
    C3Bench up;
    up.ic = 128;
    up.oc = 128;
    up.id = 4;
    up.ih = up.iw = 16;
    up.sd = up.sh = up.sw = 2;
    up.pd = up.ph = up.pw = 1;
    run_tconv3d(up);
    up.dtype = DataType::f16;
    run_tconv3d(up);

    // Unit stride at the same size, where every tap contributes.
    C3Bench same = up;
    same.dtype = DataType::f32;
    same.sd = same.sh = same.sw = 1;
    same.id = 8;
    same.ih = same.iw = 32;
    run_tconv3d(same);

    // A small-channel patch-decoder pose and a grouped one.
    C3Bench small;
    small.ic = 16;
    small.oc = 32;
    small.id = small.ih = small.iw = 12;
    small.sd = small.sh = small.sw = 2;
    small.pd = small.ph = small.pw = 1;
    run_tconv3d(small);
    small.groups = 4;
    small.oc = 64;
    run_tconv3d(small);
}

#endif // NNOPS_HAS_CUDA
