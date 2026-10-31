/// @file bench_cuda_conv2d.cpp
/// @brief CUDA Conv2D throughput, planar NCHW and channel-last NHWC.
///
/// Both memory orders run the same implicit-GEMM kernel: M is the output
/// pixels, N is the output channels per group, K is IC/G·KH·KW, and the only
/// difference is where each operand comes from — a gathered planar activation
/// against a coalesced channel-last row, and an OIHW filter read k-major
/// against an HWIO filter read n-major. So the two tables answer which layout
/// the gather is cheaper in for a given layer, and the third section puts them
/// head to head on the same geometry so the answer is a ratio, not two numbers
/// taken minutes apart.
///
/// The layers are the ones a CNN actually runs: a 7x7 stride-2 stem, the 3x3
/// stack of a ResNet stage, the 1x1 expansions and projections of a MobileNet
/// block, a grouped layer, and a depthwise-shaped one (groups = IC, which the
/// separate DepthwiseConv operator would own in a graph — it is here because the
/// conv kernel still accepts it and a graph that does not reroute pays this).
///
/// GFLOPS is 2·batch·OC·OH·OW·(IC/G)·KH·KW, i.e. the MACs actually performed;
/// the zero-padding ring is not counted, since a stride-2 border tap and an
/// interior tap cost the same in this kernel's gather.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/conv2d.hpp"
#include "common/bench_harness.hpp"
#include "common/cuda_test_helper.hpp"
#include "common/cuda_bench_helper.hpp"

#include <cstdio>
#include <string>
#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

struct ConvBenchCase {
    int64_t n = 1, ic = 64, ih = 56, iw = 56, oc = 64;
    int64_t kh = 3, kw = 3;
    int64_t sh = 1, sw = 1;
    int64_t dh = 1, dw = 1;
    int64_t ph = 1, pw = 1;
    int64_t groups = 1;
    DataType dtype = DataType::f32;
    bool channel_last = false;

    int64_t oh() const { return (ih + 2 * ph - dh * (kh - 1) - 1) / sh + 1; }
    int64_t ow() const { return (iw + 2 * pw - dw * (kw - 1) - 1) / sw + 1; }
    int64_t icg() const { return ic / groups; }
    int64_t ocg() const { return oc / groups; }
    double flops() const
    {
        return 2.0 * double(n) * double(oc) * double(oh()) * double(ow()) *
               double(icg()) * double(kh) * double(kw);
    }
};

std::string conv_geometry(const ConvBenchCase& c, const char* layout)
{
    char buf[192];
    int n = std::snprintf(buf, sizeof(buf),
                          "%s n%lld %lldx%lldx%lld k%lld%s%lld s%lld%s%lld",
                          layout, (long long)c.n, (long long)c.ih,
                          (long long)c.iw, (long long)c.ic, (long long)c.kh,
                          c.kw != c.kh ? "x" : "", (long long)c.kw,
                          (long long)c.sh, c.sw != c.sh ? "x" : "",
                          (long long)c.sw);
    // A grouped layer computes the same output as a dense one but only a
    // fraction of the MACs, so its row is meaningless without the group count.
    if (c.groups > 1 && n < static_cast<int>(sizeof(buf))) {
        n += std::snprintf(buf + n, sizeof(buf) - size_t(n), " g%lld",
                           (long long)c.groups);
    }
    if (c.dh > 1 && n < static_cast<int>(sizeof(buf))) {
        std::snprintf(buf + n, sizeof(buf) - size_t(n), " d%lld", (long long)c.dh);
    }
    return buf;
}

/// Build the case's three device tensors and the operator, upload once, and
/// return the median GPU milliseconds of a single compute().
double measure(const ConvBenchCase& c)
{
    Conv2DAttributes attrs;
    attrs.kernel_size = {c.kh, c.kw};
    attrs.stride = {c.sh, c.sw};
    attrs.dilation = {c.dh, c.dw};
    attrs.padding = {c.ph, c.pw};
    attrs.groups = c.groups;

    const TensorLayout act_layout =
        c.channel_last ? TensorLayout::NHWC : TensorLayout::NCHW;
    const TensorLayout wgt_layout =
        c.channel_last ? TensorLayout::HWIO : TensorLayout::NCHW;

    HostTensor act = make_host_random(std::vector<int64_t>{c.n, c.ic, c.ih, c.iw}, c.dtype, false, 23,
                                      -0.5f, 0.5f, act_layout);
    HostTensor wgt = make_host_random(
        c.channel_last ? std::vector<int64_t>{c.kh, c.kw, c.icg(), c.oc}
                       : std::vector<int64_t>{c.oc, c.icg(), c.kh, c.kw},
        c.dtype, false, 29, -0.25f, 0.25f, wgt_layout);
    HostTensor out =
        make_host_zeroed(std::vector<int64_t>{c.n, c.oc, c.oh(), c.ow()}, c.dtype,
                         false, act_layout);

    DeviceTensor dact = to_device(act);
    DeviceTensor dwgt = to_device(wgt);
    DeviceTensor dout = to_device(out);

    auto op = Conv2D::create(attrs, Backend::CUDA);
    const TensorView ins[] = {dact.view, dwgt.view};
    TensorView o = dout.view;
    auto launch = [&] { cuda_launch(*op, o, ins); };
    return cuda_median_ms(launch);
}

/// Time one convolution and print its row. Returns GFLOPS.
double run_conv(const ConvBenchCase& c, const char* label)
{
    const double ms = measure(c);
    print_cuda_row(label, conv_geometry(c, c.channel_last ? "nhwc" : "nchw"), ms,
                   cuda_gflops(ms, c.flops()));
    return cuda_gflops(ms, c.flops());
}

std::vector<ConvBenchCase> cnn_layers()
{
    std::vector<ConvBenchCase> v;

    ConvBenchCase stem;   // 224x224x3, 7x7 s2 p3 -> 112x112x64
    stem.ic = 3; stem.ih = stem.iw = 224; stem.oc = 64;
    stem.kh = stem.kw = 7; stem.sh = stem.sw = 2; stem.ph = stem.pw = 3;
    v.push_back(stem);

    ConvBenchCase s56;   // 56x56x64, 3x3
    s56.ic = s56.oc = 64; s56.ih = s56.iw = 56;
    v.push_back(s56);

    ConvBenchCase expand;   // 56x56x64 -> 256, 1x1
    expand.ic = 64; expand.oc = 256; expand.ih = expand.iw = 56;
    expand.kh = expand.kw = 1; expand.ph = expand.pw = 0;
    v.push_back(expand);

    ConvBenchCase s28;
    s28.ic = 128; s28.oc = 128; s28.ih = s28.iw = 28;
    v.push_back(s28);

    ConvBenchCase s14;
    s14.ic = 256; s14.oc = 256; s14.ih = s14.iw = 14;
    v.push_back(s14);

    ConvBenchCase s7;   // 7x7x512, 1x1
    s7.ic = 512; s7.oc = 512; s7.ih = s7.iw = 7;
    s7.kh = s7.kw = 1; s7.ph = s7.pw = 0;
    v.push_back(s7);

    ConvBenchCase grouped;   // 56x56, 4 groups
    grouped.ic = grouped.oc = 64; grouped.ih = grouped.iw = 56; grouped.groups = 4;
    v.push_back(grouped);

    ConvBenchCase dw;   // depthwise-shaped: one group per channel
    dw.ic = dw.oc = 64; dw.ih = dw.iw = 112; dw.groups = 64;
    v.push_back(dw);

    ConvBenchCase batched;   // a four-frame batch
    batched.n = 4; batched.ic = batched.oc = 64; batched.ih = batched.iw = 56;
    v.push_back(batched);

    ConvBenchCase wide;   // 5x5, 128 channels
    wide.ic = wide.oc = 128; wide.ih = wide.iw = 28;
    wide.kh = wide.kw = 5; wide.ph = wide.pw = 2;
    v.push_back(wide);

    return v;
}

void run_table(DataType dt, bool channel_last, const char* label)
{
    for (ConvBenchCase c : cnn_layers()) {
        c.dtype = dt;
        c.channel_last = channel_last;
        run_conv(c, label);
    }
}

}  // namespace

NNOPS_BENCH(cuda_conv2d_planar) {
    print_cuda_header("CUDA Conv2D planar NCHW (f32)");
    run_table(DataType::f32, false, "f32");
}

NNOPS_BENCH(cuda_conv2d_channel_last) {
    print_cuda_header("CUDA Conv2D channel-last NHWC (f32)");
    run_table(DataType::f32, true, "f32");
}

NNOPS_BENCH(cuda_conv2d_f16) {
    print_cuda_header("CUDA Conv2D f16 (tensor core)");
    run_table(DataType::f16, false, "f16 nchw");
    run_table(DataType::f16, true, "f16 nhwc");
}

NNOPS_BENCH(cuda_conv2d_layouts) {
    print_cuda_header("CUDA Conv2D layout A/B, same geometry (f32)");
    // The planar arm is measured once before and once after each channel-last
    // arm, so the repeat is the drift floor the ratio has to clear.
    for (ConvBenchCase c : cnn_layers()) {
        ConvBenchCase last = c;
        last.channel_last = true;

        const double planar = measure(c);
        const double last_ms = measure(last);
        const double planar_again = measure(c);
        const double floor = planar_again / planar;

        // The shared geometry string, minus the layout prefix.
        const std::string geo = conv_geometry(c, "");
        std::printf("  %-36s nchw %8.4f ms   nhwc %8.4f ms   nhwc/nchw %.2f   "
                    "drift %.3f\n", geo.c_str() + 1, planar, last_ms,
                    last_ms / planar, floor);
    }
}

NNOPS_BENCH(cuda_conv2d_shapes) {
    print_cuda_header("CUDA Conv2D stride, dilation and small planes (f32 nhwc)");
    // The shapes the implicit GEMM is most likely to mis-tile: planes whose
    // pixel count is not a multiple of the 64-pixel tile (49 and 196 pixels), a
    // stride-2 border, and a dilated 3x3 whose taps leave the plane.
    for (int64_t side : {7, 14, 28}) {
        ConvBenchCase c;
        c.ic = c.oc = 64;
        c.ih = c.iw = side;
        c.channel_last = true;
        run_conv(c, "nhwc s1");
    }
    {
        ConvBenchCase c;
        c.ic = c.oc = 128; c.ih = c.iw = 56; c.sh = c.sw = 2; c.ph = c.pw = 0;
        c.channel_last = true;
        run_conv(c, "nhwc s2");
    }
    {
        ConvBenchCase c;
        c.ic = c.oc = 64; c.ih = c.iw = 28; c.dh = c.dw = 2; c.ph = c.pw = 2;
        c.channel_last = true;
        run_conv(c, "nhwc d2");
    }
}

#endif  // NNOPS_HAS_CUDA
