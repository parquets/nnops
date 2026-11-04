/// @file bench_transpose_conv2d.cpp
/// @brief TransposeConv2D — planar vs channel-last on the same values.
///
/// One row per layer, two arms: `nchw` runs the planar route (NCHW activation,
/// planar [IC, OC/G, KH, KW] filter, scatter through a col matrix) and `nhwc`
/// the channel-last one (NHWC activation, the same filter, output channels
/// vectorised). Both are handed identical values — the channel-last input is
/// converted once, outside every timer — so a row prices the layout and the
/// kernel it selects, nothing else.
///
/// The rows are the layers a decoder actually runs: stride-2 4x4 and 3x3
/// upsamplers (DeepLab, GAN generators), stride-1 3x3 refinement passes
/// (super-resolution), a 2x2 pixel-shuffle-style layer, and grouped / batched
/// cases. They are deliberately *small*-channel on the input side and
/// large-channel on the output side, which is what a decoder looks like and
/// what makes the K of the channel reduction asymmetric.
///
/// Measurement rules follow the other benches: interleaved A/B, A B A B, with
/// the second A as the null control (same op, same buffers), per-round min,
/// median across rounds. A ratio is only called real when it clears that null
/// floor. FLOPs count 2·N·OC·(IC/G)·IH·IW·KH·KW — the contributions, one full
/// kernel per input pixel. Counting output-taps instead would inflate a stride-2
/// layer by 4x and hide the whole point of the col GEMM.
///
/// Two tables, one per value type, each with the thread ladder folded in:
/// every shape is also printed as a scan of its own geometry, so a regression
/// can be read against OW, the tap count and the col size rather than against a
/// single rate.

#include "nnops/ops/transpose_conv2d.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "common/bench_harness.hpp"
#include "common/channel_last.hpp"
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
#include <thread>
#include <type_traits>
#include <vector>

using namespace nnops;
using half = nnops::backend::cpu::half;

namespace {

// ============================================================
// Pool / timing
// ============================================================

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

/// One interleaved A/B measurement at @p nt threads, in per-call ms.
struct AbResult {
    double a_ms = 0.0, b_ms = 0.0;   ///< median over rounds of the per-round min
    double null_pct = 0.0;           ///< median A2/A1 drift — the noise floor
};

template <class FA, class FB>
AbResult measure_ab(int nt, FA&& a, FB&& b, int iters, int rounds) {
    BenchPool pool(nt);
    ComputeContext ctx;
    ctx.cpu = pool.cpu;
    const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

    auto run_a = [&]() { a(use_ctx); };
    auto run_b = [&]() { b(use_ctx); };
    run_a(); run_b(); run_a(); run_b();   // warm-up / page faults

    std::vector<double> a_ms, b_ms, null_pct;
    for (int r = 0; r < rounds; ++r) {
        const double a1 = time_ms(run_a, iters);
        const double b1 = time_ms(run_b, iters);
        const double a2 = time_ms(run_a, iters);   // null control
        const double b2 = time_ms(run_b, iters);
        a_ms.push_back(std::min(a1, a2));
        b_ms.push_back(std::min(b1, b2));
        null_pct.push_back(std::abs(a2 - a1) / std::min(a1, a2) * 100.0);
    }
    return {median_of(a_ms), median_of(b_ms), median_of(null_pct)};
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

// ============================================================
// The layer list
// ============================================================

struct Shape {
    const char* name;
    int64_t N, IC, IH, IW, OC, KH, KW;
    int64_t SH = 1, SW = 1, PH = 0, PW = 0, groups = 1;
    int64_t DH = 1, DW = 1;
};

/// Decoder upsamplers and refinement passes, then the awkward cases.
const std::vector<Shape>& sweep() {
    static const std::vector<Shape> s = {
        // ---- stride-2 upsamplers: the layer this op exists for ---------------
        {"deeplab up 1x256x14x14 -> 128 4x4 s2 p1", 1, 256, 14, 14, 128, 4, 4, 2, 2, 1, 1},
        {"seg up     1x512x7x7   -> 256 4x4 s2 p1", 1, 512,  7,  7, 256, 4, 4, 2, 2, 1, 1},
        {"gan up     1x512x8x8   -> 256 4x4 s2 p1", 1, 512,  8,  8, 256, 4, 4, 2, 2, 1, 1},
        {"sr up      1x64x32x32  -> 64  3x3 s2 p1", 1,  64, 32, 32,  64, 3, 3, 2, 2, 1, 1},
        {"sr up      1x32x64x64  -> 32  3x3 s2 p1", 1,  32, 64, 64,  32, 3, 3, 2, 2, 1, 1},
        {"pixshuffle 1x256x56x56 -> 64  2x2 s2",    1, 256, 56, 56,  64, 2, 2, 2, 2, 0, 0},
        // ---- stride-1 refinement --------------------------------------------
        {"refine     1x64x64x64  -> 64  3x3 p1",    1,  64, 64, 64,  64, 3, 3, 1, 1, 1, 1},
        {"refine     1x128x32x32 -> 128 3x3 p1",    1, 128, 32, 32, 128, 3, 3, 1, 1, 1, 1},
        {"big k      1x32x32x32  -> 32  5x5 p2",    1,  32, 32, 32,  32, 5, 5, 1, 1, 2, 2},
        // ---- grouped / batched / dilated ------------------------------------
        {"group      1x128x28x28 -> 128 3x3 g4 p1", 1, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1, 4},
        {"depthwise  1x64x56x56  -> 64  3x3 g64",   1,  64, 56, 56,  64, 3, 3, 1, 1, 1, 1, 64},
        {"batch      8x64x28x28  -> 64  3x3 p1",    8,  64, 28, 28,  64, 3, 3, 1, 1, 1, 1},
        {"dilated    1x64x32x32  -> 64  3x3 d2",    1,  64, 32, 32,  64, 3, 3, 1, 1, 0, 0, 1, 2, 2},
    };
    return s;
}

// ============================================================
// One layer, both layouts, identical values
// ============================================================

template <class T>
constexpr DataType dtype_of() {
    return std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
}

template <class T>
std::vector<T> random_vec(std::span<const int64_t> shape, uint64_t seed) {
    int64_t numel = 1;
    for (int64_t d : shape) { numel *= d; }
    std::vector<T> v(static_cast<size_t>(numel));
    test::XorShift128 rng(seed);
    for (int64_t i = 0; i < numel; ++i) {
        const float f = rng.next_float(-1.0f, 1.0f);
        if constexpr (std::is_same_v<T, float>) {
            v[static_cast<size_t>(i)] = f;
        } else {
            simd::s_store(&v[static_cast<size_t>(i)], f);
        }
    }
    return v;
}

template <class T>
struct Layer {
    explicit Layer(const Shape& c) : s(c) {
        const int64_t icg = c.IC / c.groups;
        const int64_t ish[] = {c.N, c.IC, c.IH, c.IW};
        const int64_t wsh[] = {c.IC, c.OC / c.groups, c.KH, c.KW};  // [IC][OC/G][KH][KW]
        const int64_t bsh[] = {c.OC};
        constexpr DataType dt = dtype_of<T>();

        in_planar = random_vec<T>(std::span<const int64_t>(ish, 4), 1);
        weight    = random_vec<T>(std::span<const int64_t>(wsh, 4), 2);
        bias      = random_vec<T>(std::span<const int64_t>(bsh, 1), 3);
        in_cl     = test::nchw_to_nhwc<T>(std::span<const T>(in_planar.data(), in_planar.size()),
                                          c.N, c.IC, c.IH, c.IW);

        in_nchw = TensorView(std::span<const int64_t>(ish, 4), dt, in_planar.data(),
                             TensorLayout::NCHW);
        in_nhwc = TensorView(std::span<const int64_t>(ish, 4), dt, in_cl.data(),
                             TensorLayout::NHWC);
        w       = TensorView(std::span<const int64_t>(wsh, 4), dt, weight.data(),
                             TensorLayout::NCHW);
        b       = TensorView(std::span<const int64_t>(bsh, 1), dt, bias.data(),
                             TensorLayout::NCHW);

        attrs.kernel_size = {c.KH, c.KW};
        attrs.stride      = {c.SH, c.SW};
        attrs.dilation    = {c.DH, c.DW};
        attrs.padding     = {c.PH, c.PW};
        attrs.groups      = c.groups;

        op = TransposeConv2D::create(attrs, Backend::CPU);

        const TensorDesc dn[] = {in_nchw.desc(), w.desc(), b.desc()};
        const auto od = op->getOutputTensorDesc(dn);
        OH = od[0].dims[2];
        OW = od[0].dims[3];
        out_planar.assign(static_cast<size_t>(od[0].numel()), T{});
        out_cl.assign(static_cast<size_t>(od[0].numel()), T{});

        out_nchw = TensorView(od[0].dims, dt, out_planar.data(), TensorLayout::NCHW);
        out_nhwc = TensorView(od[0].dims, dt, out_cl.data(), TensorLayout::NHWC);

        ins_nchw = {in_nchw, w, b};
        ins_nhwc = {in_nhwc, w, b};
        // Contributions, not output-taps: each input pixel runs its whole
        // kernel once, so a stride-2 layer costs a quarter, not a whole.
        work = 2.0 * static_cast<double>(c.N) * c.OC * c.IH * c.IW *
               static_cast<double>(icg) * c.KH * c.KW;
    }

    void run_nchw(const ComputeContext& ctx) { op->compute(out_nchw, ins_nchw, ctx, nullptr); }
    void run_nhwc(const ComputeContext& ctx) { op->compute(out_nhwc, ins_nhwc, ctx, nullptr); }

    Shape s;
    TransposeConv2DAttributes attrs{};
    std::unique_ptr<TransposeConv2D> op;
    std::vector<T> in_planar, in_cl, weight, bias;
    std::vector<T> out_planar, out_cl;
    TensorView in_nchw, in_nhwc, w, b, out_nchw, out_nhwc;
    std::vector<TensorView> ins_nchw, ins_nhwc;
    int64_t OH = 0, OW = 0;
    double work = 0.0;
};

// ============================================================
// Tables
// ============================================================

template <class T>
void run_table(const char* tag) {
    const std::vector<int> threads = bench_threads();
    const int iters = 20, rounds = 5;

    std::printf("  %s — nchw/ms is the planar route, nhwc/ms the channel-last one\n", tag);
    std::printf("    %-45s %6s %6s %6s | %8s %8s | %7s %7s | %6s %s\n",
                "case", "OH", "OW", "taps", "nchw ms", "nhwc ms",
                "nchw GF", "nhwc GF", "r", "null");

    double sum_nchw = 0.0, sum_nhwc = 0.0, sum_work = 0.0;
    for (const Shape& c : sweep()) {
        Layer<T> l(c);
        const double taps = static_cast<double>(c.KH * c.KW) /
                            static_cast<double>(c.SH * c.SW);
        for (int nt : threads) {
            const AbResult r = measure_ab(nt,
                [&](const ComputeContext& x) { l.run_nchw(x); },
                [&](const ComputeContext& x) { l.run_nhwc(x); },
                iters, rounds);
            const bool real = std::abs(r.a_ms - r.b_ms) / r.a_ms * 100.0 > r.null_pct;
            std::printf("    %-45s %6lld %6lld %6.1f | %8.3f %8.3f | %7.1f %7.1f | %6.2f %4.1f%%%s\n",
                        (nt == threads.front()) ? c.name : "",
                        (long long)l.OH, (long long)l.OW, taps,
                        r.a_ms, r.b_ms,
                        l.work / r.a_ms / 1e6, l.work / r.b_ms / 1e6,
                        r.a_ms / r.b_ms, r.null_pct,
                        real ? "" : " (noise)");
            if (nt == threads.front()) {
                sum_nchw += r.a_ms;
                sum_nhwc += r.b_ms;
                sum_work += l.work;
            }
        }
    }
    std::printf("    %-45s %6s %6s %6s | %8.3f %8.3f | %7.1f %7.1f | %6.2f\n",
                "sum of the layers (1T)", "", "", "",
                sum_nchw, sum_nhwc,
                sum_work / sum_nchw / 1e6, sum_work / sum_nhwc / 1e6,
                sum_nchw / sum_nhwc);
}

}  // anonymous namespace

NNOPS_BENCH(tconv2d_f32) { run_table<float>("f32"); }
NNOPS_BENCH(tconv2d_f16) { run_table<half>("f16"); }
