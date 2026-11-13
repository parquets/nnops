/// @file bench_cuda_transpose_conv2d.cpp
/// @brief CUDA TransposeConv2D throughput, channel-first and channel-last.
///
/// The residue-class GEMM poses one stride-1 correlation per output class, with
/// M the output pixels of that class, N the output channels per group and K the
/// (IC/G)-times-taps reduction (the taps are the ones the class actually hits).
/// Both memory orders run that kernel and the difference is the gather: a
/// channel-first activation read per pixel, a channel-last one read per
/// channel lane. The third table puts the two head to head on one geometry.
///
/// The layers are the ones a decoder runs — stride-2 4x4 / 3x3 upsamplers, a
/// 2x2 pixel shuffle, stride-1 refinement passes — plus the awkward cases: a 5x5
/// kernel, a grouped layer, a depthwise group (which is the channel-last 16 B
/// lane, or the output-per-thread script in channel-first), a batch, and a
/// dilated 3x3.
///
/// GFLOPS is 2·N·OC·(IC/G)·IH·IW·KH·KW — the contributions, one full kernel per
/// input pixel, the same count the CPU bench uses. Counting output taps instead
/// inflates a stride-2 layer by 4x for no work the kernel does.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/transpose_conv2d.hpp"
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

struct TConvBenchCase {
    const char* name = "";
    int64_t n = 1, ic = 64, ih = 32, iw = 32, oc = 64;
    int64_t kh = 3, kw = 3;
    int64_t sh = 1, sw = 1;
    int64_t dh = 1, dw = 1;
    int64_t ph = 1, pw = 1;
    int64_t oph = 0, opw = 0;
    int64_t groups = 1;
    DataType dtype = DataType::f32;
    bool channel_last = false;

    int64_t oh() const { return (ih - 1) * sh - 2 * ph + dh * (kh - 1) + oph + 1; }
    int64_t ow() const { return (iw - 1) * sw - 2 * pw + dw * (kw - 1) + opw + 1; }
    int64_t icg() const { return ic / groups; }
    int64_t ocg() const { return oc / groups; }
    double flops() const
    {
        return 2.0 * double(n) * double(oc) * double(ih) * double(iw) *
               double(icg()) * double(kh) * double(kw);
    }
};

std::string tconv_geometry(const TConvBenchCase& c, const char* layout)
{
    char buf[224];
    int n = std::snprintf(buf, sizeof(buf),
                          "%s %s n%lld %lldx%lldx%lld k%lld%s%lld s%lld%s%lld -> "
                          "%lldx%lld",
                          layout, c.name, (long long)c.n, (long long)c.ih,
                          (long long)c.iw, (long long)c.ic, (long long)c.kh,
                          c.kw != c.kh ? "x" : "", (long long)c.kw,
                          (long long)c.sh, c.sw != c.sh ? "x" : "",
                          (long long)c.sw, (long long)c.oh(), (long long)c.ow());
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
double measure(const TConvBenchCase& c)
{
    TransposeConv2DAttributes attrs;
    attrs.kernel_size = {c.kh, c.kw};
    attrs.stride = {c.sh, c.sw};
    attrs.dilation = {c.dh, c.dw};
    attrs.padding = {c.ph, c.pw};
    attrs.output_padding = {c.oph, c.opw};
    attrs.groups = c.groups;

    const TensorLayout act_layout =
        c.channel_last ? TensorLayout::NHWC : TensorLayout::NCHW;
    const TensorLayout out_layout = act_layout;

    HostTensor act = make_host_random(std::vector<int64_t>{c.n, c.ic, c.ih, c.iw},
                                      c.dtype, false, 23, -0.5f, 0.5f, act_layout);
    // The filter is [IC, OC/G, KH, KW] in both memory orders.
    HostTensor wgt = make_host_random(
        std::vector<int64_t>{c.ic, c.ocg(), c.kh, c.kw}, c.dtype, false, 29, -0.25f,
        0.25f, TensorLayout::NCHW);
    HostTensor out = make_host_zeroed(
        std::vector<int64_t>{c.n, c.oc, c.oh(), c.ow()}, c.dtype, false, out_layout);

    DeviceTensor dact = to_device(act);
    DeviceTensor dwgt = to_device(wgt);
    DeviceTensor dout = to_device(out);

    auto op = TransposeConv2D::create(attrs, Backend::CUDA);
    const TensorView ins[] = {dact.view, dwgt.view};
    TensorView o = dout.view;
    auto launch = [&] { cuda_launch(*op, o, ins); };
    return cuda_median_ms(launch);
}

double run_tconv(const TConvBenchCase& c, const char* label)
{
    const double ms = measure(c);
    print_cuda_row(label,
                   tconv_geometry(c, c.channel_last ? "channel_last" : "channel_first"),
                   ms, cuda_gflops(ms, c.flops()));
    return cuda_gflops(ms, c.flops());
}

/// The decoder layer list from bench_transpose_conv2d.cpp.
std::vector<TConvBenchCase> decoder_layers()
{
    std::vector<TConvBenchCase> v;
    auto add = [&v](const char* name, int64_t n, int64_t ic, int64_t ih, int64_t iw,
                    int64_t oc, int64_t kh, int64_t kw, int64_t sh, int64_t sw,
                    int64_t ph, int64_t pw, int64_t groups = 1, int64_t dh = 1,
                    int64_t dw = 1) {
        TConvBenchCase c;
        c.name = name;
        c.n = n; c.ic = ic; c.ih = ih; c.iw = iw; c.oc = oc;
        c.kh = kh; c.kw = kw; c.sh = sh; c.sw = sw; c.ph = ph; c.pw = pw;
        c.groups = groups; c.dh = dh; c.dw = dw;
        v.push_back(c);
    };

    add("up", 1, 256, 14, 14, 128, 4, 4, 2, 2, 1, 1);   // deeplab
    add("up", 1, 512, 7, 7, 256, 4, 4, 2, 2, 1, 1);     // seg
    add("up", 1, 512, 8, 8, 256, 4, 4, 2, 2, 1, 1);     // gan
    add("up", 1, 64, 32, 32, 64, 3, 3, 2, 2, 1, 1);     // sr
    add("up", 1, 32, 64, 64, 32, 3, 3, 2, 2, 1, 1);     // sr
    add("pixshuffle", 1, 256, 56, 56, 64, 2, 2, 2, 2, 0, 0);
    add("refine", 1, 64, 64, 64, 64, 3, 3, 1, 1, 1, 1);
    add("refine", 1, 128, 32, 32, 128, 3, 3, 1, 1, 1, 1);
    add("big k", 1, 32, 32, 32, 32, 5, 5, 1, 1, 2, 2);
    add("group", 1, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 4);
    add("depthwise", 1, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1, 64);
    add("depthwise s2", 1, 64, 28, 28, 64, 3, 3, 2, 2, 1, 1, 64);
    add("depthwise wide", 1, 128, 32, 32, 128, 3, 3, 1, 1, 1, 1, 128);
    add("depthwise n8", 8, 32, 64, 64, 32, 3, 3, 1, 1, 1, 1, 32);
    add("batch", 8, 64, 28, 28, 64, 3, 3, 1, 1, 1, 1);
    add("dilated", 1, 64, 32, 32, 64, 3, 3, 1, 1, 0, 0, 1, 2, 2);
    return v;
}

void run_table(DataType dt, bool channel_last, const char* label)
{
    for (TConvBenchCase c : decoder_layers()) {
        c.dtype = dt;
        c.channel_last = channel_last;
        run_tconv(c, label);
    }
}

}  // namespace

NNOPS_BENCH(cuda_transpose_conv2d_channel_first) {
    print_cuda_header("CUDA TransposeConv2D channel-first (f32)");
    run_table(DataType::f32, false, "f32");
}

NNOPS_BENCH(cuda_transpose_conv2d_channel_last) {
    print_cuda_header("CUDA TransposeConv2D channel-last (f32)");
    run_table(DataType::f32, true, "f32");
}

NNOPS_BENCH(cuda_transpose_conv2d_f16) {
    print_cuda_header("CUDA TransposeConv2D f16 (tensor core)");
    run_table(DataType::f16, false, "f16 channel_first");
    run_table(DataType::f16, true, "f16 channel_last");
}

NNOPS_BENCH(cuda_transpose_conv2d_layouts) {
    print_cuda_header("CUDA TransposeConv2D layout A/B, same geometry (f32)");
    // The channel-first arm is measured once before and once after each
    // channel-last arm, so the repeat is the drift floor the ratio must clear.
    for (TConvBenchCase c : decoder_layers()) {
        TConvBenchCase last = c;
        last.channel_last = true;

        const double channel_first = measure(c);
        const double last_ms = measure(last);
        const double channel_first_again = measure(c);
        const double floor = channel_first_again / channel_first;

        const std::string geo = tconv_geometry(c, "");
        std::printf("  %-52s channel_first %8.4f ms   channel_last %8.4f ms   "
                    "channel_last/channel_first %.2f   drift %.3f\n",
                    geo.c_str() + 1, channel_first, last_ms, last_ms / channel_first,
                    floor);
    }
}

NNOPS_BENCH(cuda_transpose_conv2d_small) {
    print_cuda_header("CUDA TransposeConv2D small planes and output_padding (f32)");
    // The class's M is the output pixels of one residue class, so a 7x7 input
    // with stride 2 gives a 13x13 output and ~42 pixels per class — below the
    // 64-pixel tile, and four times over. And output_padding, which only grows
    // the destination.
    for (bool last : {false, true}) {
        const char* layout = last ? "channel_last" : "channel_first";
        for (int64_t side : {7, 14}) {
            TConvBenchCase c;
            c.name = "s2 small";
            c.ic = c.oc = 64;
            c.ih = c.iw = side;
            c.sh = c.sw = 2;
            c.channel_last = last;
            run_tconv(c, layout);
        }
        TConvBenchCase p;
        p.name = "s2 outpad";
        p.ic = p.oc = 64; p.ih = p.iw = 16; p.sh = p.sw = 2;
        p.oph = p.opw = 1;
        p.channel_last = last;
        run_tconv(p, layout);
    }
}

#endif  // NNOPS_HAS_CUDA
