/// @file bench_cuda_pooling.cpp
/// @brief CUDA pooling throughput, channel-first and channel-last.
///
/// Pooling performs no MACs, so the throughput column is not a FLOP rate: the
/// numerator is the number of window taps the op reads (N·C·OH·OW·K·K, twice
/// that for the printed figure, matching the other benches' 2· convention), and
/// the mix of max, average, exclude_pad and global cases is what decides where
/// the kernel's time goes.
///
/// The shapes are the ones a CNN pools: the 3x3 stride-2 windows of a ResNet
/// stage, the 2x2 windows of the older networks, a 3x3 stride-1 average, and the
/// 7x7 global reduction a head ends with. Each runs in both memory orders and
/// both dtypes, and the layer table is walked twice per dtype — once per
/// layout — so a row can be read as a ratio against its twin.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/pooling.hpp"
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

struct PoolBenchCase {
    int64_t n = 1, c = 64, ih = 56, iw = 56;
    int64_t k = 3, s = 2, pad = 1;
    PoolingType type = PoolingType::Max;
    bool exclude_pad = false;
    bool global = false;
    DataType dtype = DataType::f32;
    bool channel_last = false;

    int64_t oh() const
    {
        return global ? 1 : (ih + 2 * pad - k) / s + 1;
    }
    int64_t ow() const { return oh(); }
    double taps() const
    {
        if (global) { return 2.0 * double(n) * double(c) * double(ih) * double(iw); }
        return 2.0 * double(n) * double(c) * double(oh()) * double(ow()) *
               double(k) * double(k);
    }
};

std::string pool_geometry(const PoolBenchCase& c, const char* layout)
{
    char buf[192];
    if (c.global) {
        std::snprintf(buf, sizeof(buf), "%s n%lld %lldx%lldx%lld global %s", layout,
                      (long long)c.n, (long long)c.ih, (long long)c.iw,
                      (long long)c.c,
                      c.type == PoolingType::Max ? "max" : "avg");
    } else {
        std::snprintf(buf, sizeof(buf),
                      "%s n%lld %lldx%lldx%lld k%lld s%lld p%lld %s%s", layout,
                      (long long)c.n, (long long)c.ih, (long long)c.iw,
                      (long long)c.c, (long long)c.k, (long long)c.s,
                      (long long)c.pad,
                      c.type == PoolingType::Max ? "max" : "avg",
                      c.exclude_pad ? " xp" : "");
    }
    return buf;
}

double measure(const PoolBenchCase& c)
{
    PoolingAttributes attrs;
    attrs.type = c.type;
    attrs.kernel_shape = {c.k, c.k, c.k};
    attrs.stride = {c.s, c.s, c.s};
    attrs.padding = {c.pad, c.pad, c.pad};
    attrs.exclude_pad = c.exclude_pad;
    attrs.global = c.global;

    const TensorLayout layout = c.channel_last ? TensorLayout::NHWC : TensorLayout::NCHW;

    HostTensor in = make_host_random(std::vector<int64_t>{c.n, c.c, c.ih, c.iw},
                                     c.dtype, false, 23, -1.0f, 1.0f, layout);

    auto op = Pooling::create(attrs, Backend::CUDA);
    const TensorDesc in_desc = in.view.desc();
    const std::vector<TensorDesc> ods = op->getOutputTensorDesc({&in_desc, 1});
    const TensorDesc od = ods.front();

    HostTensor out = make_host_zeroed(od.dims, c.dtype, false, layout);
    DeviceTensor din = to_device(in);
    DeviceTensor dout = to_device(out);

    const TensorView ins[] = {din.view};
    TensorView o = dout.view;
    auto launch = [&] { cuda_launch(*op, o, ins); };
    return cuda_median_ms(launch);
}

void run_pool(const PoolBenchCase& c, const char* label)
{
    const double ms = measure(c);
    print_cuda_row(label, pool_geometry(c, c.channel_last ? "channel_last" : "channel_first"),
                   ms, cuda_gflops(ms, c.taps()));
}

std::vector<PoolBenchCase> cnn_pools()
{
    std::vector<PoolBenchCase> v;
    const auto add = [&](int64_t c, int64_t side, int64_t k, int64_t s, int64_t pad,
                         PoolingType type, bool exclude_pad) {
        PoolBenchCase p;
        p.c = c; p.ih = p.iw = side;
        p.k = k; p.s = s; p.pad = pad;
        p.type = type; p.exclude_pad = exclude_pad;
        v.push_back(p);
    };

    // The 3x3 stride-2 window a ResNet stage opens with.
    add(64, 112, 3, 2, 1, PoolingType::Max, false);
    add(64, 56, 3, 2, 1, PoolingType::Max, false);
    add(128, 28, 3, 2, 1, PoolingType::Max, false);
    add(256, 14, 3, 2, 1, PoolingType::Max, false);
    // 2x2 stride 2, the older-network form, and a stride-1 3x3 average.
    add(64, 112, 2, 2, 0, PoolingType::Max, false);
    add(128, 56, 2, 2, 0, PoolingType::Max, false);
    add(64, 56, 3, 1, 1, PoolingType::Average, false);
    add(64, 56, 3, 1, 1, PoolingType::Average, true);
    // The head's global reduction.
    add(512, 7, 7, 1, 0, PoolingType::Average, false);
    add(512, 7, 7, 1, 0, PoolingType::Max, false);
    return v;
}

void run_table(DataType dt, bool channel_last, const char* label)
{
    for (PoolBenchCase c : cnn_pools()) {
        c.dtype = dt;
        c.channel_last = channel_last;
        c.global = c.k == 7 && c.s == 1 && c.pad == 0;
        run_pool(c, label);
    }
}

}  // namespace

NNOPS_BENCH(cuda_pooling) {
    print_cuda_header("CUDA pooling (f32 channel_first)");
    run_table(DataType::f32, false, "f32");
    print_cuda_header("CUDA pooling (f32 channel_last)");
    run_table(DataType::f32, true, "f32");
    print_cuda_header("CUDA pooling (f16 channel_first)");
    run_table(DataType::f16, false, "f16");
    print_cuda_header("CUDA pooling (f16 channel_last)");
    run_table(DataType::f16, true, "f16");
}

#endif  // NNOPS_HAS_CUDA
