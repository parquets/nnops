/// @file bench_cuda_resize.cpp
/// @brief CUDA resize throughput, channel-first and channel-last.
///
/// Resize performs no MACs, so the second column is not a FLOP rate: it is the
/// bandwidth a call moves at, counting the output it writes and the input image
/// once. For an upsample that understates the read traffic (a source pixel feeds
/// several outputs and is re-read), and for a downsample it overstates it by the
/// same factor — but it is the same convention for every row, so rows are
/// comparable, and it is the number a resize is actually bounded by.
///
/// The layers are the ones a model resizes: the 2x nearest and bilinear passes of
/// a U-Net decoder, a 4x pixel-shuffle-style nearest, a bilinear downsample, the
/// bicubic pass a super-resolution head uses, an RGB three-channel image (which
/// falls to the pixel script rather than the 16 B lane), and a trilinear 3D
/// volume. Each runs in both memory orders and both dtypes.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/resize.hpp"
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

struct RszBenchCase {
    const char* name = "";
    int64_t n = 1, c = 64;
    int64_t ih = 64, iw = 64;
    int64_t oh = 128, ow = 128;
    int64_t id = 1, od = 1;
    bool three_d = false;
    ResizeMode mode = ResizeMode::Nearest;
    CoordinateTransformMode coord = CoordinateTransformMode::HalfPixel;
    DataType dtype = DataType::f32;
    bool channel_last = false;

    double elems(int64_t nn, int64_t cc, int64_t dd, int64_t hh, int64_t ww) const
    {
        return double(nn) * double(cc) * double(dd) * double(hh) * double(ww);
    }
};

std::string rsz_geometry(const RszBenchCase& c, const char* layout)
{
    const char* mode = c.mode == ResizeMode::Nearest
                           ? "nearest"
                           : (c.mode == ResizeMode::Linear ? "linear" : "cubic");
    char buf[224];
    if (c.three_d) {
        std::snprintf(buf, sizeof(buf),
                      "%s %s n%lld %lldx%lldx%lldx%lld -> %lldx%lldx%lld", layout,
                      mode, (long long)c.n, (long long)c.c, (long long)c.id,
                      (long long)c.ih, (long long)c.iw, (long long)c.od,
                      (long long)c.oh, (long long)c.ow);
    } else {
        std::snprintf(buf, sizeof(buf), "%s %s n%lld %lldx%lldx%lld -> %lldx%lld",
                      layout, mode, (long long)c.n, (long long)c.c,
                      (long long)c.ih, (long long)c.iw, (long long)c.oh,
                      (long long)c.ow);
    }
    return buf;
}

/// Median GPU milliseconds of one compute() on the case's fixed buffers.
double measure(const RszBenchCase& c)
{
    ResizeAttributes attrs;
    attrs.mode = c.mode;
    attrs.coord_mode = c.coord;
    attrs.output_size = {c.od, c.oh, c.ow};

    const TensorLayout layout =
        c.channel_last ? (c.three_d ? TensorLayout::NDHWC : TensorLayout::NHWC)
                       : (c.three_d ? TensorLayout::NCDHW : TensorLayout::NCHW);

    const std::vector<int64_t> in_shape = c.three_d
        ? std::vector<int64_t>{c.n, c.c, c.id, c.ih, c.iw}
        : std::vector<int64_t>{c.n, c.c, c.ih, c.iw};
    const std::vector<int64_t> out_shape = c.three_d
        ? std::vector<int64_t>{c.n, c.c, c.od, c.oh, c.ow}
        : std::vector<int64_t>{c.n, c.c, c.oh, c.ow};

    HostTensor act = make_host_random(in_shape, c.dtype, false, 23, -0.5f, 0.5f,
                                      layout);
    HostTensor out = make_host_zeroed(out_shape, c.dtype, false, layout);

    DeviceTensor dact = to_device(act);
    DeviceTensor dout = to_device(out);

    auto op = Resize::create(attrs, Backend::CUDA);
    const TensorView ins[] = {dact.view};
    TensorView o = dout.view;
    auto launch = [&] { cuda_launch(*op, o, ins); };
    return cuda_median_ms(launch);
}

double run_rsz(const RszBenchCase& c, const char* label)
{
    const double ms = measure(c);
    const double esz = double(data_type_size(c.dtype));
    const double bytes =
        (c.elems(c.n, c.c, c.od, c.oh, c.ow) + c.elems(c.n, c.c, c.id, c.ih, c.iw)) * esz;
    const double gbps = ms > 0.0 ? bytes / (ms * 1e6) : 0.0;
    std::printf("  %-18s %-46s %9.4f ms %9.2f GB/s\n", label,
                rsz_geometry(c, c.channel_last ? "channel_last" : "channel_first")
                    .c_str(),
                ms, gbps);
    return gbps;
}

/// The resize pass list a model actually runs.
std::vector<RszBenchCase> resize_layers()
{
    std::vector<RszBenchCase> v;
    auto add = [&v](const char* name, int64_t n, int64_t c, int64_t ih, int64_t iw,
                    int64_t oh, int64_t ow, ResizeMode mode,
                    CoordinateTransformMode coord = CoordinateTransformMode::HalfPixel) {
        RszBenchCase b;
        b.name = name;
        b.n = n; b.c = c; b.ih = ih; b.iw = iw; b.oh = oh; b.ow = ow;
        b.mode = mode; b.coord = coord;
        v.push_back(b);
    };

    add("up x2", 1, 64, 64, 64, 128, 128, ResizeMode::Nearest);
    add("up x2", 1, 256, 32, 32, 64, 64, ResizeMode::Nearest);
    add("up x2", 1, 64, 64, 64, 128, 128, ResizeMode::Linear);
    add("up x2", 1, 256, 32, 32, 64, 64, ResizeMode::Linear);
    add("up x4", 1, 64, 16, 16, 64, 64, ResizeMode::Nearest);
    add("up x4", 1, 64, 16, 16, 64, 64, ResizeMode::Linear);
    add("down x2", 1, 64, 128, 128, 64, 64, ResizeMode::Linear);
    add("down x2", 1, 64, 128, 128, 64, 64, ResizeMode::Nearest);
    add("sr bicubic", 1, 32, 64, 64, 192, 192, ResizeMode::Cubic);
    add("rgb", 1, 3, 256, 256, 512, 512, ResizeMode::Linear);
    add("rgb nearest", 1, 3, 256, 256, 512, 512, ResizeMode::Nearest);
    add("wide C", 1, 512, 32, 32, 64, 64, ResizeMode::Linear);
    add("batch", 8, 64, 32, 32, 64, 64, ResizeMode::Linear);

    RszBenchCase vol;
    vol.name = "3d";
    vol.three_d = true;
    vol.n = 1; vol.c = 32; vol.id = 16; vol.ih = 32; vol.iw = 32;
    vol.od = 32; vol.oh = 64; vol.ow = 64;
    vol.mode = ResizeMode::Linear;
    v.push_back(vol);
    return v;
}

void run_table(DataType dt, bool channel_last, const char* label)
{
    for (RszBenchCase c : resize_layers()) {
        c.dtype = dt;
        c.channel_last = channel_last;
        run_rsz(c, label);
    }
}

}  // namespace

NNOPS_BENCH(cuda_resize_channel_first) {
    print_cuda_header("CUDA Resize channel-first (f32)");
    run_table(DataType::f32, false, "f32");
}

NNOPS_BENCH(cuda_resize_channel_last) {
    print_cuda_header("CUDA Resize channel-last (f32)");
    run_table(DataType::f32, true, "f32");
}

NNOPS_BENCH(cuda_resize_f16) {
    print_cuda_header("CUDA Resize f16");
    run_table(DataType::f16, false, "f16 channel_first");
    run_table(DataType::f16, true, "f16 channel_last");
}

#endif  // NNOPS_HAS_CUDA
