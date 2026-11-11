/// @file bench_conv3d.cpp
/// @brief Conv3D — the im2col → GEMM route against the scalar gather it replaced.
///
/// Two arms on identical values: the operator, and
/// reference::conv3d_channel_first_ref from tests/reference, the output-driven
/// scalar reduction every voxel used to walk. The ref arm is not a competitor's
/// kernel — it is the shape of the code the col GEMM was built against, so the
/// ratio says what the structure bought.
///
/// Measurement rules follow the other benches: interleaved, with a repeated
/// operator pass as the null control, per-round min, median across rounds. The
/// scalar arm is two orders of magnitude slower, so it gets one call per timing
/// pass where the operator gets ten.
///
/// FLOPs are 2·N·OC·(IC/G)·OD·OH·OW·KD·KH·KW — one MAC per output tap, the
/// standard count for a forward convolution.
///
/// The rows are video-decoder shapes: 3x3x3 refiners and downsamples, the
/// temporal-only 3x1x1, the pointwise 1x1x1 the video head spends its time in,
/// a grouped case, a depthwise case, a batch and a large kernel.
/// `NNOPS_BENCH_THREADS` pins the ladder, as in the others.

#include "nnops/ops/conv3d.hpp"
#include "common/bench_harness.hpp"
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

namespace nnops::backend::cpu::reference {
    void conv3d_channel_first_ref(const Conv3DAttributes& attrs,
                                  TensorView& output,
                                  std::span<const TensorView> inputs,
                                  const ComputeContext& ctx,
                                  void* workspace);
}

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
    Conv3DAttributes attrs;
};

/// One entry per shape family a 3-D backbone runs.
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

    add("stem 1x3x16x64x64 -> 32 k3", {1, 3, 16, 64, 64}, {32, 3, 3, 3, 3},
        {3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1});
    add("refine 1x64x8x16x16 -> 64 k3", {1, 64, 8, 16, 16}, {64, 64, 3, 3, 3},
        {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1});
    add("down 1x64x8x32x32 -> 128 k3s2", {1, 64, 8, 32, 32}, {128, 64, 3, 3, 3},
        {3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1});
    add("temporal 1x64x16x14x14 -> 64 k3x1", {1, 64, 16, 14, 14}, {64, 64, 3, 1, 1},
        {3, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1});
    add("pointwise 1x128x8x14x14 -> 64", {1, 128, 8, 14, 14}, {64, 128, 1, 1, 1},
        {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0});
    add("grouped 1x64x4x8x8 -> 64 g8", {1, 64, 4, 8, 8}, {64, 8, 3, 3, 3},
        {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 8);
    add("depthwise 1x32x4x8x8 -> 32 g32", {1, 32, 4, 8, 8}, {32, 1, 3, 3, 3},
        {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 32);
    add("batch 4 4x32x4x8x8 -> 32 k3", {4, 32, 4, 8, 8}, {32, 32, 3, 3, 3},
        {3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1});
    add("large k 1x32x8x8x8 -> 32 k5", {1, 32, 8, 8, 8}, {32, 32, 5, 5, 5},
        {5, 5, 5}, {1, 1, 1}, {1, 1, 1}, {2, 2, 2});
    add("dilated 1x32x4x8x8 -> 32 k3d2", {1, 32, 4, 8, 8}, {32, 32, 3, 3, 3},
        {3, 3, 3}, {1, 1, 1}, {2, 2, 2}, {2, 2, 2});

    return cs;
}

void run_one(const Case& c, int nt, int rounds)
{
    const int64_t N = c.in_shape[0];
    const int64_t G = c.attrs.groups;
    const int64_t OC = c.w_shape[0];
    const int64_t IC_per_G = c.w_shape[1];

    auto [in_vec, in_view] = test::make_random_tensor(c.in_shape);
    auto [w_vec, w_view] = test::make_random_tensor(c.w_shape);

    auto op = Conv3D::create(c.attrs, Backend::CPU);
    const TensorDesc d[] = {in_view.desc(), w_view.desc()};
    const auto descs = op->getOutputTensorDesc(d);
    const auto numel = static_cast<size_t>(descs[0].numel());

    std::vector<float> out_buf(numel, 0.0f);
    auto out = test::make_channel_first(descs[0], out_buf.data());

    BenchPool pool(nt);
    ComputeContext ctx;
    ctx.cpu = pool.cpu;
    const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

    const TensorView ins[] = {in_view, w_view};
    TensorView outs[] = {out};

    auto arm_op = [&]() { op->compute(outs, ins, use_ctx); };
    auto arm_ref = [&]() {
        backend::cpu::reference::conv3d_channel_first_ref(c.attrs, out, ins, use_ctx,
                                                          nullptr);
    };

    // The operator arm is sub-millisecond on the smaller rows, where a single
    // pass is all timer noise; it gets enough calls to fill a few milliseconds.
    const int op_iters = 10;
    const int ref_iters = 1;

    arm_op();
    arm_ref();

    std::vector<double> op_ms, ref_ms, null_pct;
    for (int r = 0; r < rounds; ++r) {
        const double a1 = time_ms(arm_op, op_iters);
        const double b1 = time_ms(arm_ref, ref_iters);
        const double a2 = time_ms(arm_op, op_iters);   // null control
        op_ms.push_back(std::min(a1, a2));
        ref_ms.push_back(b1);
        null_pct.push_back(std::abs(a2 - a1) / std::min(a1, a2) * 100.0);
    }

    const double op_t = median_of(op_ms);
    const double ref_t = median_of(ref_ms);
    const double null_pct_med = median_of(null_pct);

    const int64_t OD = descs[0].dims[2];
    const int64_t OH = descs[0].dims[3];
    const int64_t OW = descs[0].dims[4];
    const double flops = 2.0 * static_cast<double>(N) * static_cast<double>(OC) *
                         static_cast<double>(IC_per_G) *
                         static_cast<double>(OD * OH * OW) *
                         static_cast<double>(c.attrs.kernel_size[0] *
                                             c.attrs.kernel_size[1] *
                                             c.attrs.kernel_size[2]);

    std::printf("    %-36s | %8.3f %9.3f | %7.1f | %6.2f %6.1f%%\n", c.name,
                op_t, ref_t, flops / (op_t * 1e6), ref_t / op_t, null_pct_med);
}

}  // namespace

NNOPS_BENCH(conv3d) {
    const auto cases = make_cases();
    int nth = 1;
    if (const char* env = std::getenv("NNOPS_BENCH_THREADS")) {
        nth = std::atoi(env);
        if (nth <= 0) { nth = 1; }
    }

    std::printf("\n=== Conv3D fp32, %dT — im2col GEMM vs the scalar gather ===\n", nth);
    std::printf("    %-36s | %8s %9s | %7s | %6s %6s\n",
                "case", "op ms", "ref ms", "op GF", "vs ref", "null");
    for (const auto& c : cases) { run_one(c, nth, 3); }
}
