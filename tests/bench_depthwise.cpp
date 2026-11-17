/// @file bench_depthwise.cpp
/// @brief Depthwise conv throughput.
///
/// The channel lane vectorizes: one load feeds 16 channels at one output
/// position, four output rows are unrolled, and each tap broadcasts one weight
/// vector into four independent accumulates.
///
/// The layout conversion is done once, outside every timed region: an engine
/// converts a tensor once and reuses it, so billing the bench's convert per call
/// would price the harness, not the kernel.
///
/// Each round times the same call repeatedly and reports the median, so the
/// column is the kernel's own cost.

#include "nnops/ops/depthwise_conv.hpp"
#include "common/bench_harness.hpp"
#include "common/channel_last.hpp"
#include "common/cpu_pool.hpp"
#include "common/random_tensor.hpp"
#include "common/test_helpers.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

using namespace nnops;

namespace {

// OpenMP when the build has it and NNOPS_BENCH_OMP allows, else spawn-per-call.
struct BenchPool {
    explicit BenchPool(int nthreads) : nthreads_(nthreads) {
        if (test::bench_use_openmp()) {
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

std::vector<int> bench_threads() {
    std::vector<int> threads;
    if (const char* env = std::getenv("NNOPS_BENCH_THREADS")) {
        for (const char* p = env; *p;) {
            char* end = nullptr;
            const long v = std::strtol(p, &end, 10);
            if (end != p && v > 0) { threads.push_back(static_cast<int>(v)); }
            p = (end != p) ? end : p + 1;
            if (*p == ',') { ++p; }
        }
        if (!threads.empty()) { return threads; }
    }
    threads = {1, 2, 4, 8};
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw > 1 && std::find(threads.begin(), threads.end(), hw) == threads.end()) {
        threads.push_back(hw);
    }
    return threads;
}

/// Mean wall time of one call, in microseconds.
template <typename Fn>
double time_us(Fn&& fn, int iters) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) { fn(); }
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() /
           std::max(iters, 1);
}

double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct DepthwiseCase {
    const char* name;
    int64_t C, IH, IW;
    int64_t KH = 3, KW = 3, SH = 1, SW = 1, PH = 1, PW = 1;
    /// The model's stages ride relu6; the plain cases do not — one run compares both.
    bool relu6 = false;
};

void run_case(const DepthwiseCase& c, const std::vector<int>& threads, int iters,
              int rounds) {
    const int64_t ishape[] = {1, c.C, c.IH, c.IW};
    const int64_t wshape[] = {c.C, 1, c.KH, c.KW};
    const int64_t bshape[] = {c.C};

    auto [in_vec, in_raw] = test::make_random_tensor(ishape, -1.0f, 1.0f, 31);
    auto [w_vec, w_raw]   = test::make_random_tensor(wshape, -1.0f, 1.0f, 32);
    auto [b_vec, b_v]     = test::make_random_tensor(bshape, -0.5f, 0.5f, 33);
    (void)in_raw;
    (void)w_raw;
    (void)b_vec;

    DepthwiseConvAttributes attrs{};
    attrs.kernel_size = {1, c.KH, c.KW};
    attrs.stride      = {1, c.SH, c.SW};
    attrs.dilation    = {1, 1, 1};
    attrs.padding     = {0, c.PH, c.PW};
    if (c.relu6) { attrs.epilogue.type = nnops::EpilogueActivateType::Relu6; }

    // TensorFlow's orders, converted once like a real graph.
    auto in_cl_vec = test::to_channel_last<float>(
        std::span<const float>(in_vec.data(), in_vec.size()), 1, c.C, c.IH * c.IW);
    auto w_cl_vec = test::depthwise_to_hwcm<float>(
        std::span<const float>(w_vec.data(), w_vec.size()), c.C, c.KH * c.KW);
    TensorView in_cl(ishape, DataType::f32, in_cl_vec.data(), TensorLayout::NHWC);
    const int64_t w_cl_shape[] = {1, c.KH, c.KW, c.C};
    TensorView w_cl(std::span<const int64_t>(w_cl_shape, 4), DataType::f32,
                    w_cl_vec.data(), TensorLayout::HWCM);

    auto op = DepthwiseConv::create(attrs, Backend::CPU);
    const TensorView ins[] = {in_cl, w_cl, b_v};
    const TensorDesc descs[] = {in_cl.desc(), w_cl.desc(), b_v.desc()};
    auto out_desc = op->getOutputTensorDesc(descs)[0];
    const int64_t OH = out_desc.dims[2];
    const int64_t OW = out_desc.dims[3];
    std::vector<float> out_buf(static_cast<size_t>(out_desc.numel()));
    auto out = test::make_channel_first(out_desc, out_buf.data());
    const auto call = [&](const ComputeContext& ctx) {
        TensorView outs[] = {out};
        op->compute(outs, ins, ctx);
    };

    const double flops = 2.0 * static_cast<double>(c.C) * OH * OW * c.KH * c.KW;
    std::printf("  %s\n", c.name);
    std::printf("    in[1,%lld,%lld,%lld] w[%lld,1,%lldx%lld] s%lld p%lld -> out[1,%lld,%lld,%lld]\n",
                (long long)c.C, (long long)c.IH, (long long)c.IW,
                (long long)c.C, (long long)c.KH, (long long)c.KW,
                (long long)c.SH, (long long)c.PH,
                (long long)c.C, (long long)OH, (long long)OW);

    for (int nt : threads) {
        BenchPool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

        call(use_ctx);                // warm-up / first touch

        std::vector<double> cl;
        for (int r = 0; r < rounds; ++r) {
            cl.push_back(time_us([&] { call(use_ctx); }, iters));
        }
        const double tc = median_of(cl);

        std::printf("    %2dT  nhwc %8.1f us (%6.1f GFLOPS)\n",
                    nt, tc, flops / (tc * 1e3));
    }
}


}  // namespace

NNOPS_BENCH(depthwise) {
    const std::vector<int> threads = bench_threads();
    const int iters = 20;
    const int rounds = 5;

    std::printf("depthwise conv (lane=channel)\n");
    std::printf("  pool=%s  threads=", test::bench_pool_name());
    for (size_t i = 0; i < threads.size(); ++i) {
        std::printf("%d%s", threads[i], i + 1 < threads.size() ? "," : "\n");
    }

    const DepthwiseCase cases[] = {
        {"dw 3x3 C=8   @112 s1",    8, 112, 112},   // half a channel vector
        {"dw 3x3 C=16  @112 s1",   16, 112, 112},
        {"dw 3x3 C=32  @112 s1",   32, 112, 112},
        {"dw 3x3 C=64  @112 s1",   64, 112, 112},
        {"dw 3x3 C=96  @112 s2",   96, 112, 112, 3, 3, 2, 2},
        {"dw 3x3 C=144 @56  s1",  144,  56,  56},
        {"dw 3x3 C=240 @28  s1",  240,  28,  28},
        {"dw 3x3 C=576 @14  s1",  576,  14,  14},
        {"dw 3x3 C=960 @7   s1",  960,   7,   7},
        {"dw 3x3 C=256 @200x300", 256, 200, 300},
        {"dw 3x3 C=256 @100x150", 256, 100, 150},
        {"dw 3x3 C=256 @60x80",   256,  60,  80},
        {"dw 3x3 C=256 @17x23",   256,  17,  23},
        {"dw 5x5 C=64  @56  s1",   64,  56,  56, 5, 5, 1, 1, 2, 2},
        {"dw 1x1 C=144 @56",      144,  56,  56, 1, 1, 1, 1, 0, 0},
    };

    for (const auto& c : cases) {
        run_case(c, threads, iters, rounds);
    }
}

NNOPS_BENCH(depthwise_mobilenetv2_shapes) {
    const std::vector<int> threads = bench_threads();
    const int iters = 20;
    const int rounds = 5;

    std::printf("depthwise 3x3: MobileNetV2's own layer shapes\n");
    std::printf("  pool=%s  threads=", test::bench_pool_name());
    for (size_t i = 0; i < threads.size(); ++i) {
        std::printf("%d%s", threads[i], i + 1 < threads.size() ? "," : "\n");
    }

    const DepthwiseCase cases[] = {
        {"mnv2 dw @112 C=32  s1",  32, 112, 112},
        {"mnv2 dw @112 C=96  s2",  96, 112, 112, 3, 3, 2, 2},
        {"mnv2 dw @56  C=144 s1", 144,  56,  56},
        {"mnv2 dw @56  C=144 s2", 144,  56,  56, 3, 3, 2, 2},
        {"mnv2 dw @28  C=192 s1", 192,  28,  28},
        {"mnv2 dw @28  C=192 s2", 192,  28,  28, 3, 3, 2, 2},
        {"mnv2 dw @14  C=384 s1", 384,  14,  14},
        {"mnv2 dw @14  C=576 s2", 576,  14,  14, 3, 3, 2, 2},
        {"mnv2 dw @7   C=576 s1", 576,   7,   7},
        {"mnv2 dw @7   C=960 s1", 960,   7,   7},
    };

    for (const auto& c : cases) {
        run_case(c, threads, iters, rounds);
    }
}
