/// @file bench_cuda_conv1d.cpp
/// @brief CUDA Conv1D throughput.
///
/// Conv1D has no kernel of its own: it is a rank-4 lift onto the 2D implicit
/// GEMM with a 1-wide height, so these rows are what the 2D kernel does when one
/// of its spatial axes is a single row. That is the point of measuring them —
/// the 2D plan cuts an oh axis that does not exist here, and the column split is
/// what has to carry the grid.
///
/// M is the output positions, N the output channels of one group, K is
/// IC/G·KW. GFLOPS is 2·batch·OC·OW·(IC/G)·KW.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/conv1d.hpp"
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

struct C1Bench {
    int64_t n = 1, l = 256, ic = 64, oc = 64;
    int64_t k = 3, s = 1, d = 1, p = 1;
    int64_t groups = 1;
    DataType dtype = DataType::f32;

    int64_t ow() const { return (l + 2 * p - d * (k - 1) - 1) / s + 1; }
    double flops() const
    {
        return 2.0 * double(n) * double(oc) * double(ow()) * double(ic / groups) * double(k);
    }
};

std::string c1_geometry(const C1Bench& c)
{
    char buf[160];
    std::snprintf(buf, sizeof(buf), "nwc n%lld l%lld c%lld k%lld s%lld%s", (long long)c.n,
                  (long long)c.l, (long long)c.ic, (long long)c.k, (long long)c.s,
                  c.groups > 1 ? " g" : "");
    std::string s(buf);
    if (c.groups > 1) {
        s += std::to_string(c.groups);
    }
    s += c.dtype == DataType::f16 ? " f16" : " f32";
    return s;
}

void run_case(const C1Bench& c, const char* tag)
{
    Conv1DAttributes attrs;
    attrs.kernel_size = c.k;
    attrs.stride = c.s;
    attrs.dilation = c.d;
    attrs.padding = c.p;
    attrs.groups = c.groups;

    HostTensor act = make_host_random(std::vector<int64_t>{c.n, c.l, c.ic}, c.dtype, false, 23,
                                      -0.5f, 0.5f, TensorLayout::NHWC);
    HostTensor wgt = make_host_random(std::vector<int64_t>{c.k, c.ic / c.groups, c.oc}, c.dtype,
                                      false, 29, -0.25f, 0.25f, TensorLayout::HWIO);
    HostTensor out = make_host_zeroed(std::vector<int64_t>{c.n, c.ow(), c.oc}, c.dtype, false,
                                      TensorLayout::NHWC);

    DeviceTensor dact = to_device(act);
    DeviceTensor dwgt = to_device(wgt);
    DeviceTensor dout = to_device(out);

    auto op = Conv1D::create(attrs, Backend::CUDA);
    const TensorView ins[] = {dact.view, dwgt.view};
    TensorView o = dout.view;
    const double ms = cuda_median_ms([&] { cuda_launch(*op, o, ins); }, 3, 11);
    print_cuda_row(tag, c1_geometry(c), ms, cuda_gflops(ms, c.flops()));
}

} // anonymous namespace

NNOPS_BENCH(cuda_conv1d)
{
    print_cuda_header("CUDA Conv1D (NWC x HWIO, lifted onto the 2D kernel)");

    // A speech-model front end: many long positions, few channels.
    C1Bench speech;
    speech.l = 1024;
    speech.ic = 32;
    speech.oc = 64;
    run_case(speech, "conv1d");
    speech.dtype = DataType::f16;
    run_case(speech, "conv1d");

    // A wide layer where the channels dominate the reduction.
    C1Bench wide;
    wide.l = 512;
    wide.ic = wide.oc = 256;
    run_case(wide, "conv1d");
    wide.dtype = DataType::f16;
    run_case(wide, "conv1d");

    // A stride the lift carries into the 2D plan, and a depthwise geometry that
    // the 2D op's routing hands to the depthwise kernel.
    C1Bench s2 = speech;
    s2.dtype = DataType::f32;
    s2.s = 2;
    s2.p = 0;
    run_case(s2, "conv1d");

    C1Bench dw;
    dw.l = 1024;
    dw.ic = dw.oc = 64;
    dw.groups = 64;
    run_case(dw, "conv1d");
}

#endif // NNOPS_HAS_CUDA
