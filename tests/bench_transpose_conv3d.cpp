/// @file bench_transpose_conv3d.cpp
/// @brief TransposeConv3D throughput, channel-last.
///
/// One arm: the operator on an NDHWC activation, converted once outside every
/// timer, so a row prices the kernel and the route it selects.
///
/// Measurement rules follow the other benches: repeatedly timed calls, median
/// across rounds.
///
/// FLOPs are 2·N·OC·(IC/G)·ID·IH·IW·KD·KH·KW — the contributions, one full
/// kernel per input voxel. Counting output-taps instead inflates a stride-2
/// layer 8x on three axes and hides the point of the col GEMM.
///
/// The rows are video-decoder shapes: 2x2x2 / 3x3x3 / 4x4x4 upsamplers, a
/// stride-1 refinement pass, a temporal-only expander, a grouped case, a batch
/// and a dilated pass. `NNOPS_BENCH_THREADS` pins the ladder, as in the others.

#include "nnops/ops/transpose_conv3d.hpp"
#include "common/bench_harness.hpp"
#include "common/channel_last.hpp"
#include "common/cpu_pool.hpp"
#include "common/random_tensor.hpp"
#include "common/test_helpers.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <thread>
#include <vector>

using namespace nnops;

namespace {

struct BenchPool {
    explicit BenchPool(int nthreads) : nthreads_(nthreads) {
        if (nnops::test::bench_use_openmp()) {
            cpu = nnops::openmp_backend(nthreads);
            return;
        }
        cpu.parallel_for = [this](int64_t begin, int64_t end, const ParallelForBody& body) {
            this->parallel_for(begin, end, body);
        };
        cpu.num_threads = [this]() { return nthreads_; };
        cpu.thread_id = []() { return current_thread_id_; };
    }

    void parallel_for(int64_t begin, int64_t end, const ParallelForBody& body) {
        std::atomic<int64_t> next{begin};
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(nthreads_));
        for (int t = 0; t < nthreads_; ++t) {
            workers.emplace_back([&, t]() {
                current_thread_id_ = t;
                for (;;) {
                    int64_t i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= end) { break; }
                    body(i);
                }
            });
        }
        for (auto& w : workers) { w.join(); }
    }

    int nthreads_;
    CpuBackend cpu;
    inline static thread_local int current_thread_id_ = 0;
};

template <class Fn>
double time_ms(Fn&& fn, int iters) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) { fn(); }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() /
           std::max(iters, 1);
}

double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Case {
    const char* name;
    std::vector<int64_t> in_shape;
    std::vector<int64_t> w_shape;
    TransposeConv3DAttributes attrs;
};

/// One entry per shape family a 3-D decoder runs.
std::vector<Case> make_cases()
{
    std::vector<Case> cs;

    auto add = [&](const char* name, std::vector<int64_t> in, std::vector<int64_t> w,
                   std::array<int64_t, 3> k, std::array<int64_t, 3> s,
                   std::array<int64_t, 3> d, std::array<int64_t, 3> p, int64_t g = 1) {
        Case c{name, std::move(in), std::move(w), {}};
        c.attrs.kernel_size = k;
        c.attrs.stride = s;
        c.attrs.dilation = d;
        c.attrs.padding = p;
        c.attrs.groups = g;
        cs.push_back(std::move(c));
    };

    add("up 2x2x2  1x64x8x14x14 -> 32", {1, 64, 8, 14, 14}, {64, 32, 2, 2, 2},
        {2, 2, 2}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0});
    add("up 3x3x3  1x64x8x14x14 -> 32", {1, 64, 8, 14, 14}, {64, 32, 3, 3, 3},
        {3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1});
    add("up 4x4x4  1x32x4x7x7   -> 16", {1, 32, 4, 7, 7}, {32, 16, 4, 4, 4},
        {4, 4, 4}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1});
    add("refine    1x32x4x16x16 -> 32", {1, 32, 4, 16, 16}, {32, 32, 3, 3, 3},
        {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1});
    add("temporal  1x16x16x8x8  -> 16", {1, 16, 16, 8, 8}, {16, 16, 3, 1, 1},
        {3, 1, 1}, {2, 1, 1}, {1, 1, 1}, {0, 0, 0});
    add("grouped   1x64x4x8x8   -> 16 g16", {1, 64, 4, 8, 8}, {64, 4, 3, 3, 3},
        {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 16);
    add("depthwise 1x32x4x8x8   -> 32 g32", {1, 32, 4, 8, 8}, {32, 1, 3, 3, 3},
        {3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, 32);
    add("batch 4   4x32x4x8x8   -> 32", {4, 32, 4, 8, 8}, {32, 32, 3, 3, 3},
        {3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1});
    add("dilated   1x32x4x8x8   -> 32", {1, 32, 4, 8, 8}, {32, 32, 3, 3, 3},
        {3, 3, 3}, {1, 1, 1}, {2, 2, 2}, {2, 2, 2});

    return cs;
}

void run_one(const Case& c, int nt, int rounds)
{
    const int64_t N = c.in_shape[0];
    const int64_t G = c.attrs.groups;
    const int64_t OC = c.w_shape[1] * G;
    const int64_t IC_per_G = c.w_shape[0] / G;

    auto [in_vec, in_view] = test::make_random_tensor(c.in_shape);
    auto [w_vec, w_view] = test::make_random_tensor(c.w_shape);
    (void)in_view;
    (void)w_vec;

    // The same values in NDHWC, converted once, outside every timer.
    const int64_t IC = c.in_shape[1];
    const int64_t S_in = c.in_shape[2] * c.in_shape[3] * c.in_shape[4];
    const auto in_cl = test::to_channel_last<float>(in_vec, N, IC, S_in);
    TensorView in_cl_view(c.in_shape, DataType::f32,
                          const_cast<float*>(in_cl.data()), TensorLayout::NDHWC);

    auto op = TransposeConv3D::create(c.attrs, Backend::CPU);
    const TensorDesc d[] = {in_cl_view.desc(), w_view.desc()};
    const auto descs = op->getOutputTensorDesc(d);
    const auto numel = static_cast<size_t>(descs[0].numel());
    const auto out_dims = std::span<const int64_t>(
        descs[0].dims.data(), static_cast<size_t>(descs[0].rank));

    std::vector<float> out_cl_buf(numel, 0.0f);
    TensorView out_cl(out_dims, DataType::f32, out_cl_buf.data(),
                      TensorLayout::NDHWC);

    BenchPool pool(nt);
    ComputeContext ctx;
    ctx.cpu = pool.cpu;
    const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

    const TensorView ins_cl[] = {in_cl_view, w_view};
    TensorView outs_cl[] = {out_cl};

    auto arm_cl = [&]() { op->compute(outs_cl, ins_cl, use_ctx); };

    const int op_iters = 3;

    arm_cl();

    std::vector<double> cl_ms;
    for (int r = 0; r < rounds; ++r) {
        cl_ms.push_back(time_ms(arm_cl, op_iters));
    }

    const double cl = median_of(cl_ms);

    const double flops = 2.0 * static_cast<double>(N) * static_cast<double>(OC) *
                         static_cast<double>(IC_per_G) *
                         static_cast<double>(c.in_shape[2] * c.in_shape[3] *
                                             c.in_shape[4]) *
                         static_cast<double>(c.attrs.kernel_size[0] *
                                             c.attrs.kernel_size[1] *
                                             c.attrs.kernel_size[2]);

    std::printf("    %-34s | %8.3f | %7.1f\n",
                c.name, cl, flops / (cl * 1e6));
}

}  // namespace

NNOPS_BENCH(tconv3d) {
    const auto cases = make_cases();
    int nth = 1;
    if (const char* env = std::getenv("NNOPS_BENCH_THREADS")) {
        nth = std::atoi(env);
        if (nth <= 0) { nth = 1; }
    }

    std::printf("\n=== TransposeConv3D fp32, %dT — NDHWC ===\n", nth);
    std::printf("    %-34s | %8s | %7s\n",
                "case", "NDHWC ms", "GF");
    for (const auto& c : cases) { run_one(c, nth, 3); }
}
