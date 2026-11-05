/// @file bench_moe.cpp
/// @brief MoE end-to-end on the shapes a decoder runs.
///
/// One row per routing pattern: a Mixtral-style 8-expert top-2 prefill and the
/// same layer at T=1 (decode), a DeepSeek-V3-style 64-expert top-8 with sigmoid
/// gating and renormalized weights, a SwiGLU expert (three GEMMs per route), and
/// a small grouped case. The routing itself is random logits, so the expert
/// load is uneven the way a real batch is — which is exactly what a per-expert
/// parallel split has to absorb.
///
/// Measurement follows the other benches: per-round min of two back-to-back
/// calls, median across rounds, and the drift between those two calls printed
/// as the null control. FLOPs count the routes (k per token), each running
/// fc1 + fc2, plus fc3 for SwiGLU.

#include "nnops/ops/moe.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "common/bench_harness.hpp"
#include "common/cpu_pool.hpp"
#include "common/random_tensor.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

using namespace nnops;
using half = nnops::backend::cpu::half;

namespace {

/// The same two schedulers the other benches carry: OpenMP when the build has
/// it, else a thread-per-call pool.
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
    threads = {1};
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw > 1) { threads.push_back(hw); }
    return threads;
}

template <class T>
constexpr DataType dtype_of() {
    return std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
}

template <class T>
void fill_random(std::vector<T>& v, uint64_t seed, float lo = -1.0f, float hi = 1.0f) {
    nnops::test::XorShift128 rng(seed);
    for (size_t i = 0; i < v.size(); ++i) {
        const float f = rng.next_float(lo, hi);
        if constexpr (std::is_same_v<T, float>) {
            v[i] = f;
        } else {
            nnops::simd::s_store(&v[i], f);
        }
    }
}

/// One decode-layer MoE. `swiglu` adds the third projection (Separate layout).
struct Shape {
    const char* name;
    int64_t T, H, I, E, k;
    MoEActivation act = MoEActivation::Silu;
    MoERouterGating gating = MoERouterGating::Softmax;
    bool normalize = false;
    float scale = 1.0f;
    bool swiglu = false;
};

const std::vector<Shape>& sweep() {
    static const std::vector<Shape> s = {
        {"mixtral 8x  k2  prefill T=256", 256, 2048, 4096, 8, 2},
        {"mixtral 8x  k2  decode  T=1",     1, 2048, 4096, 8, 2},
        {"deepseek 64x k8 prefill T=256", 256, 2048,  512, 64, 8,
         MoEActivation::Silu, MoERouterGating::Sigmoid, true, 2.5f},
        {"swiglu 8x   k2  T=128",         128, 1024, 2048, 8, 2,
         MoEActivation::SwiGLU, MoERouterGating::Softmax, false, 1.0f, true},
        {"small 16x   k4  T=64",           64,  512,  512, 16, 4,
         MoEActivation::Gelu, MoERouterGating::Softmax, false, 1.0f, false},
    };
    return s;
}

template <class T>
struct Layer {
    explicit Layer(const Shape& c) : s(c) {
        constexpr DataType dt = dtype_of<T>();
        const int64_t tok[] = {c.T, c.H};
        const int64_t gate[] = {c.T, c.E};
        const int64_t f1[] = {c.E, c.I, c.H};
        const int64_t b1[] = {c.E, c.I};
        const int64_t f2[] = {c.E, c.H, c.I};
        const int64_t b2[] = {c.E, c.H};
        const int64_t f3[] = {c.E, c.I, c.H};
        const int64_t b3[] = {c.E, c.I};
        const int64_t out[] = {c.T, c.H};

        auto alloc = [](std::vector<T>& v, const int64_t* d, int rank, uint64_t seed) {
            int64_t n = 1;
            for (int i = 0; i < rank; ++i) { n *= d[i]; }
            v.resize(static_cast<size_t>(n));
            fill_random(v, seed);
        };
        alloc(input, tok, 2, 1);
        alloc(router, gate, 2, 2);
        alloc(fc1w, f1, 3, 3);
        alloc(fc1b, b1, 2, 4);
        alloc(fc2w, f2, 3, 5);
        alloc(fc2b, b2, 2, 6);
        if (c.swiglu) {
            alloc(fc3w, f3, 3, 7);
            alloc(fc3b, b3, 2, 8);
        }
        out_buf.assign(static_cast<size_t>(c.T * c.H), T{});

        in_v       = TensorView(std::span<const int64_t>(tok, 2), dt, input.data());
        router_v   = TensorView(std::span<const int64_t>(gate, 2), dt, router.data());
        fc1w_v     = TensorView(std::span<const int64_t>(f1, 3), dt, fc1w.data());
        fc1b_v     = TensorView(std::span<const int64_t>(b1, 2), dt, fc1b.data());
        fc2w_v     = TensorView(std::span<const int64_t>(f2, 3), dt, fc2w.data());
        fc2b_v     = TensorView(std::span<const int64_t>(b2, 2), dt, fc2b.data());
        if (c.swiglu) {
            fc3w_v = TensorView(std::span<const int64_t>(f3, 3), dt, fc3w.data());
            fc3b_v = TensorView(std::span<const int64_t>(b3, 2), dt, fc3b.data());
        }
        out_v      = TensorView(std::span<const int64_t>(out, 2), dt, out_buf.data());

        attrs.k = c.k;
        attrs.activation = c.act;
        attrs.router_gating = c.gating;
        attrs.normalize_routing_weights = c.normalize;
        attrs.routed_scaling_factor = c.scale;
        attrs.swiglu_layout = SwiGLULayout::Separate;
        op = MoE::create(attrs, Backend::CPU);

        ins = {in_v, router_v, fc1w_v, fc1b_v, fc2w_v, fc2b_v,
               fc3w_v, fc3b_v, TensorView{}};

        // Routes: k per token, each fc1 + fc2 (+ fc3 for SwiGLU).
        const double routes = static_cast<double>(c.k) * static_cast<double>(c.T);
        work = routes * 4.0 * static_cast<double>(c.H) * static_cast<double>(c.I);
        if (c.swiglu) { work += routes * 2.0 * static_cast<double>(c.H) * static_cast<double>(c.I); }
    }

    void run(const ComputeContext& ctx) {
        TensorView outs[] = {out_v};
        op->compute(outs, ins, ctx, nullptr);
    }

    Shape s;
    MoEAttributes attrs{};
    std::unique_ptr<MoE> op;
    std::vector<T> input, router, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b;
    std::vector<T> out_buf;
    TensorView in_v, router_v, fc1w_v, fc1b_v, fc2w_v, fc2b_v, fc3w_v, fc3b_v, out_v;
    std::vector<TensorView> ins;
    double work = 0.0;
};

template <class T>
void run_table(const char* tag) {
    const std::vector<int> threads = bench_threads();
    const int rounds = 3;

    std::printf("  %s\n", tag);
    std::printf("    %-40s %5s %5s %5s %3s %2s | %9s %8s %5s\n",
                "case", "T", "H", "I", "E", "k", "ms", "GFLOPS", "null");

    for (const Shape& c : sweep()) {
        Layer<T> l(c);
        for (int nt : threads) {
            BenchPool pool(nt);
            ComputeContext ctx;
            ctx.cpu = pool.cpu;
            const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

            const int iters = c.T <= 8 ? 20 : (c.T <= 128 ? 8 : 3);
            l.run(use_ctx);   // warm-up / page faults

            std::vector<double> ms, null;
            for (int r = 0; r < rounds; ++r) {
                const double a1 = time_ms([&] { l.run(use_ctx); }, iters);
                const double a2 = time_ms([&] { l.run(use_ctx); }, iters);
                ms.push_back(std::min(a1, a2));
                null.push_back(std::abs(a2 - a1) / std::min(a1, a2) * 100.0);
            }
            const double best = median_of(ms);
            std::printf("    %-40s %5lld %5lld %5lld %3lld %2lld | %3dT %8.3f %8.1f %4.1f%%\n",
                        (nt == threads.front()) ? c.name : "",
                        (long long)c.T, (long long)c.H, (long long)c.I,
                        (long long)c.E, (long long)c.k,
                        nt, best, l.work / best / 1e6, median_of(null));
        }
    }
}

}  // anonymous namespace

NNOPS_BENCH(moe_f32) { run_table<float>("f32"); }
NNOPS_BENCH(moe_f16) { run_table<half>("f16"); }
