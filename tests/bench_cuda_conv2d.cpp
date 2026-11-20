/// @file bench_cuda_conv2d.cpp
/// @brief CUDA Conv2D throughput.
///
/// M is the output pixels, N is the output channels per group, K is
/// IC/G·KH·KW. The route reads a coalesced activation row and an HWIO filter
/// n-major.
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
#include "detail/bench_knobs.hpp"
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

    const TensorLayout act_layout = TensorLayout::NHWC;
    const TensorLayout wgt_layout = TensorLayout::HWIO;

    HostTensor act = make_host_random(std::vector<int64_t>{c.n, c.ih, c.iw, c.ic}, c.dtype, false, 23,
                                      -0.5f, 0.5f, act_layout);
    HostTensor wgt = make_host_random(
        std::vector<int64_t>{c.kh, c.kw, c.icg(), c.oc},
        c.dtype, false, 29, -0.25f, 0.25f, wgt_layout);
    HostTensor out =
        make_host_zeroed(std::vector<int64_t>{c.n, c.oh(), c.ow(), c.oc}, c.dtype,
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
    print_cuda_row(label, conv_geometry(c, "nhwc"), ms,
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

/// The same stack on planes large enough that the pixel axis alone fills the
/// machine several times over. The CNN set above is one image at a time, so
/// most of its rows are under one wave of the 64-pixel tile and price latency,
/// not the tile: 56x56x64 is 49 blocks against 114 resident. These rows answer
/// what the kernel does when the grid is not the excuse.
std::vector<ConvBenchCase> wide_layers()
{
    std::vector<ConvBenchCase> v;
    ConvBenchCase c;

    c.ic = c.oc = 64; c.ih = c.iw = 224;            v.push_back(c); // 3x3
    c.ic = c.oc = 64; c.ih = c.iw = 112;            v.push_back(c);
    c.ic = 128; c.oc = 128; c.ih = c.iw = 112;      v.push_back(c);
    c.ic = 256; c.oc = 256; c.ih = c.iw = 56;       v.push_back(c);

    c = ConvBenchCase{}; c.ic = 64; c.oc = 256; c.ih = c.iw = 112;
    c.kh = c.kw = 1; c.ph = c.pw = 0;               v.push_back(c); // pointwise
    c = ConvBenchCase{}; c.ic = 64; c.oc = 128; c.ih = c.iw = 224;
    c.sh = c.sw = 2; c.ph = c.pw = 0;               v.push_back(c); // stride 2

    // Past a wave several times over and at the widths the large-image
    // backbones use: 448x448x64 and 224x224x128 are each 14.8 GFLOP a call, so
    // a row reads as a steady-state rate rather than a launch.
    c = ConvBenchCase{}; c.ic = 64; c.oc = 64; c.ih = c.iw = 448;  v.push_back(c);
    c = ConvBenchCase{}; c.ic = 64; c.oc = 64; c.ih = c.iw = 512;  v.push_back(c);
    c = ConvBenchCase{}; c.ic = 128; c.oc = 128; c.ih = c.iw = 224; v.push_back(c);
    c = ConvBenchCase{}; c.n = 4; c.ic = c.oc = 64; c.ih = c.iw = 224; v.push_back(c);
    c = ConvBenchCase{}; c.ic = 256; c.oc = 256; c.ih = c.iw = 224;
    c.kh = c.kw = 1; c.ph = c.pw = 0;               v.push_back(c); // pointwise

    return v;
}

/// int8: a pre-quantized u8 activation against an s8 filter, per-tensor
/// activation parameters and per-channel weight ones, into an s8 destination.
/// The same layer set as the fp tables, so the rows are comparable.
double run_conv_i8(const ConvBenchCase& c, const char* label, bool packed = false,
                   bool print = true, DataType adt = DataType::u8)
{
    Conv2DAttributes attrs;
    attrs.kernel_size = {c.kh, c.kw};
    attrs.stride = {c.sh, c.sw};
    attrs.dilation = {c.dh, c.dw};
    attrs.padding = {c.ph, c.pw};
    attrs.groups = c.groups;
    attrs.output_dtype = DataType::s8;

    const int64_t icg = c.icg();
    test::XorShift128 rng(23);

    std::vector<uint8_t> a(static_cast<size_t>(c.n * c.ic * c.ih * c.iw));
    for (auto& e : a) { e = static_cast<uint8_t>(rng.next_u64() & 0xFFu); }
    std::vector<int8_t> w(static_cast<size_t>(c.oc * icg * c.kh * c.kw));
    for (auto& e : w) {
        e = static_cast<int8_t>(int32_t(rng.next_u64() % 25) - 12);
    }
    std::vector<float> w_s(static_cast<size_t>(c.oc));
    std::vector<int32_t> w_zp(static_cast<size_t>(c.oc));
    for (int64_t i = 0; i < c.oc; ++i) {
        w_s[static_cast<size_t>(i)] = 0.008f + 0.001f * float(i % 5);
        w_zp[static_cast<size_t>(i)] = int32_t(i % 7) - 3;
    }

    QuantParams qa;
    qa.scale = 0.0078125f;
    qa.zero_point = adt == DataType::u8 ? 128 : 0;
    QuantParams qw;
    qw.granularity = QuantGranularity::PerChannel;
    qw.scale_data = w_s.data();
    qw.zero_point_data = w_zp.data();
    qw.num_scales = c.oc;
    qw.scale = 0.008f;
    qw.zero_point = 0;
    QuantParams qc;
    qc.scale = 0.03f;
    qc.zero_point = 0;

    const auto view = [](const std::vector<int64_t>& sh, DataType dt, void* d,
                         TensorLayout lay, const QuantParams& qp) {
        return TensorView(std::span<const int64_t>(sh.data(), sh.size()), dt, d,
                          lay, qp);
    };
    const TensorLayout alay = TensorLayout::NHWC;
    const TensorLayout wlay = TensorLayout::HWIO;
    const std::vector<int64_t> ashape = {c.n, c.ih, c.iw, c.ic};
    const std::vector<int64_t> wshape = {c.kh, c.kw, icg, c.oc};

    TensorView av = view(ashape, adt, a.data(), alay, qa);
    TensorView wv = view(wshape, DataType::s8, w.data(), wlay, qw);
    const TensorDesc descs[] = {av.desc(), wv.desc()};
    auto op_cpu = Conv2D::create(attrs, Backend::CPU);
    const TensorDesc od = op_cpu->getOutputTensorDesc(descs).front();
    const std::vector<int64_t> odims(od.dims.begin(), od.dims.begin() + od.rank);
    int64_t on = 1;
    for (auto d : odims) { on *= d; }

    CudaBuffer da(a.size()), dw(w.size());
    CudaBuffer dout(static_cast<size_t>(on));
    da.upload(a.data(), a.size());
    dw.upload(w.data(), w.size());
    dout.zero();
    TensorView dav = view(ashape, DataType::u8, da.data(), alay, qa);
    TensorView dwv = view(wshape, DataType::s8, dw.data(), wlay, qw);
    TensorView dcv = view(odims, od.dtype, dout.data(), od.layout, qc);

    auto op = Conv2D::create(attrs, Backend::CUDA);

    // The packed arm packs once, outside the timer, exactly as a caller with a
    // loaded model would.
    CudaBuffer dblob;
    std::vector<int64_t> bshape;
    TensorView bv;
    if (packed) {
        std::vector<TensorView> qin = {dwv};
        std::vector<TensorView> qout = {TensorView{}};
        op->prepackWeights(qin, qout, cuda_context());
        if (qout[0].rank() != 2) {
            throw std::runtime_error("conv2d int8: the packed filter query is empty");
        }
        dblob = CudaBuffer(static_cast<size_t>(qout[0].numel()));
        bshape = {qout[0].shape(0), qout[0].shape(1)};
        bv = view(bshape, DataType::s8, dblob.data(), qout[0].layout(), qw);
        std::vector<TensorView> pin = {dwv};
        std::vector<TensorView> pout = {bv};
        op->prepackWeights(pin, pout, cuda_context());
        NNOPS_CUDA_CHECK(cudaDeviceSynchronize());
    }
    std::vector<TensorView> ins = {dav, packed ? bv : dwv};
    const double ms = cuda_median_ms([&] { cuda_launch(*op, dcv, ins); });
    if (print) {
        print_cuda_row(label, conv_geometry(c, "nhwc"), ms, cuda_gflops(ms, c.flops()));
    }
    return cuda_gflops(ms, c.flops());
}

void run_table_i8(const char* label)
{
    for (ConvBenchCase c : cnn_layers()) {
        run_conv_i8(c, label);
    }
}

void run_table(DataType dt, const char* label)
{
    for (ConvBenchCase c : cnn_layers()) {
        c.dtype = dt;
        run_conv(c, label);
    }
}

/// The wide-plane set, all three dtypes back to back so the rows stay
/// comparable: the same geometry is one row each for f32, f16 and int8.
void run_table_wide()
{
    for (ConvBenchCase c : wide_layers()) {
        run_conv_i8(c, "i8");
        c.dtype = DataType::f32;
        run_conv(c, "f32");
        c.dtype = DataType::f16;
        run_conv(c, "f16");
    }
}

}  // namespace

NNOPS_BENCH(cuda_conv2d) {
    print_cuda_header("CUDA Conv2D (f32)");
    run_table(DataType::f32, "f32");
}

NNOPS_BENCH(cuda_conv2d_i8) {
    print_cuda_header("CUDA Conv2D int8 (u8 x s8 -> s8)");
    run_table_i8("i8 nhwc");
}

/// The prepacked filter against the raw one, both arms alternated inside this
/// process: the filter gather is the one part of the int8 kernel a caller can
/// pay for once, and this is the only instrument on this host that can see the
/// difference.
NNOPS_BENCH(cuda_conv2d_i8_packed) {
    print_cuda_header("CUDA Conv2D int8: raw filter vs PackedHWIOCuda");
    for (ConvBenchCase c : wide_layers()) {
        double best[2] = {0.0, 0.0};
        for (int rep = 0; rep < 3; ++rep) {
            for (int arm = 0; arm < 2; ++arm) {
                best[arm] = std::max(best[arm], run_conv_i8(c, "i8", arm == 1, false));
            }
        }
        std::printf("  %-8s %-34s raw %8.0f packed %8.0f GFLOPS  %.2fx\n", "i8",
                    conv_geometry(c, "nhwc").c_str(), best[0], best[1], best[1] / best[0]);
    }
}

NNOPS_BENCH(cuda_conv2d_i8_s8) {
    print_cuda_header("CUDA Conv2D int8 s8 x s8 -> s8: raw vs packed, and the u8 twin");
    for (ConvBenchCase c : wide_layers()) {
        // 0: raw s8, 1: packed s8, 2: packed u8 (the same kernel on the unsigned
        // activation, so the zero-point and signed-dp4a cost shows up as a ratio).
        double best[3] = {0.0, 0.0, 0.0};
        for (int rep = 0; rep < 3; ++rep) {
            best[0] = std::max(best[0], run_conv_i8(c, "i8s8", false, false, DataType::s8));
            best[1] = std::max(best[1], run_conv_i8(c, "i8s8", true, false, DataType::s8));
            best[2] = std::max(best[2], run_conv_i8(c, "i8", true, false, DataType::u8));
        }
        std::printf("  %-8s %-34s raw %8.0f packed %8.0f (u8 %8.0f) GFLOPS  %.2fx\n", "i8s8",
                    conv_geometry(c, "nhwc").c_str(), best[0], best[1], best[2],
                    best[1] / best[0]);
    }
}

NNOPS_BENCH(cuda_conv2d_i8_splitk) {
    print_cuda_header("CUDA Conv2D int8 s8: split-K off vs the automatic rule");
    for (ConvBenchCase c : cnn_layers()) {
        double best[2] = {0.0, 0.0};
        for (int rep = 0; rep < 3; ++rep) {
            nnops::detail::cuda_splitk.store(0, std::memory_order_relaxed);
            best[0] = std::max(best[0], run_conv_i8(c, "i8", true, false, DataType::s8));
            nnops::detail::cuda_splitk.store(-1, std::memory_order_relaxed);
            best[1] = std::max(best[1], run_conv_i8(c, "i8", true, false, DataType::s8));
        }
        std::printf("  %-8s %-34s whole %8.0f split %8.0f GFLOPS  %.2fx\n", "i8s8",
                    conv_geometry(c, "nhwc").c_str(), best[0], best[1], best[1] / best[0]);
    }
}

NNOPS_BENCH(cuda_conv2d_f16) {
    print_cuda_header("CUDA Conv2D f16 (tensor core)");
    run_table(DataType::f16, "f16");
}

NNOPS_BENCH(cuda_conv2d_wide) {
    print_cuda_header("CUDA Conv2D wide planes (all dtypes)");
    run_table_wide();
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
        run_conv(c, "s1");
    }
    {
        ConvBenchCase c;
        c.ic = c.oc = 128; c.ih = c.iw = 56; c.sh = c.sw = 2; c.ph = c.pw = 0;
        run_conv(c, "s2");
    }
    {
        ConvBenchCase c;
        c.ic = c.oc = 64; c.ih = c.iw = 28; c.dh = c.dw = 2; c.ph = c.pw = 2;
        run_conv(c, "d2");
    }
}

NNOPS_BENCH(cuda_depthwise_layers) {
    print_cuda_header("CUDA depthwise (groups = IC, rerouted out of the GEMM)");
    // The depthwise stack a MobileNetV2 block walks, plus the 5x5.
    struct Dw { int64_t side, c, k, s; };
    const Dw layers[] = {
        {112, 32, 3, 1}, {112, 64, 3, 2}, {56, 64, 3, 1}, {56, 128, 3, 2},
        {28, 128, 3, 1}, {28, 256, 3, 2}, {14, 256, 3, 1}, {14, 512, 3, 2},
        {7, 512, 3, 1}, {56, 64, 5, 1},
    };
    for (DataType dt : {DataType::f32, DataType::f16}) {
        for (const Dw& l : layers) {
            ConvBenchCase c;
            c.ic = c.oc = l.c;
            c.ih = c.iw = l.side;
            c.kh = c.kw = l.k;
            c.sh = c.sw = l.s;
            c.ph = c.pw = l.k / 2;
            c.groups = l.c;
            c.dtype = dt;
                run_conv(c, dt == DataType::f32 ? "f32" : "f16");
        }
    }
}

#endif  // NNOPS_HAS_CUDA
