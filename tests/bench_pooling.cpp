/// @file bench_pooling.cpp
/// @brief Global pooling micro-benchmark: the whole-plane scan against the two
///        routes it replaced, plus an NCHW-vs-NHWC layout comparison.
///
/// Three arms over the same input, interleaved inside one process so the
/// numbers share an instrument:
///
///   global   Pooling with attrs.global — the whole-plane scan, one output
///            element per channel plane
///   window   Pooling with the window spelled out as the full extent — the
///            generic walk, which is what the quantized path falls back to
///   reduce   two Reduce(Mean) calls (over W, then H) — what the model runners
///            did before global pooling existed
///
/// The shapes are the ones the two model runners actually pool, plus one wide
/// backbone plane to show what the scan does when it is not dwarfed by dispatch.
///
/// The layouts bench holds shape and kernel fixed and varies only the memory
/// order: NCHW keeps a plane contiguous along W, NHWC along C. It times the
/// same two modes in each layout — the global mean and a 3x3 stride-2 max — and
/// a null re-run of NCHW reports instrument drift.
///
/// `NNOPS_BENCH_THREADS=1` (or "1,4") pins the ladder, as in the other benches.

#include "nnops/ops/layout_convert.hpp"
#include "nnops/ops/pooling.hpp"
#include "nnops/ops/reduce.hpp"
#include "common/bench_harness.hpp"
#include "common/channel_last.hpp"
#include "common/cpu_pool.hpp"
#include "common/random_tensor.hpp"
#include "common/test_helpers.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using namespace nnops;
using half = nnops::backend::cpu::half;

namespace {

// Same two schedulers as the other benches — see common/cpu_pool.hpp.
struct BenchPool {
    explicit BenchPool(int nthreads) : nthreads_(nthreads) {
        if (nnops::test::bench_use_openmp()) {
            cpu = nnops::openmp_backend(nthreads);
            return;
        }
        cpu.parallel_for = [this](int64_t begin, int64_t end, const ParallelForBody& body) {
            parallel_for(begin, end, body);
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

template <typename T>
auto make_random(std::span<const int64_t> shape, uint64_t seed) {
    if constexpr (std::is_same_v<T, float>) {
        return test::make_random_tensor(shape, -1.0f, 1.0f, seed);
    } else {
        return test::make_random_f16_tensor(shape, -1.0f, 1.0f, seed);
    }
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct GlobalCase {
    const char* name;
    int64_t N, C, H, W;
    bool f16 = false;
};

template <typename Fn>
double time_us(int iters, Fn&& fn) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) { fn(); }
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

template <typename T>
void run_case(const GlobalCase& c, const std::vector<int>& threads, int rounds,
              int iters)
{
    const int64_t shape[] = {c.N, c.C, c.H, c.W};
    auto [in_vec, in_nchw] = make_random<T>(shape, 1);

    auto make_arm = [&](bool global) {
        PoolingAttributes attrs;
        attrs.type = PoolingType::Average;
        attrs.global = global;
        if (!global) {
            attrs.kernel_shape = {1, c.H, c.W};
            attrs.stride = {1, 1, 1};
            attrs.padding = {0, 0, 0};
        }
        auto op = Pooling::create(attrs, Backend::CPU);
        TensorDesc d = in_nchw.desc();
        const TensorDesc ins[] = {d};
        auto descs = op->getOutputTensorDesc(ins);
        return std::pair{std::move(op), std::move(descs)};
    };

    auto [global_op, global_desc] = make_arm(/*global=*/true);
    auto [window_op, window_desc] = make_arm(/*global=*/false);
    const TensorView& x = in_nchw;

    std::vector<T> g_buf(static_cast<size_t>(global_desc[0].numel()));
    std::vector<T> w_buf(static_cast<size_t>(window_desc[0].numel()));
    auto g_out = test::make_planar(global_desc[0], g_buf.data());
    auto w_out = test::make_planar(window_desc[0], w_buf.data());

    // Axes are rank-relative: the input is [N,C,H,W], so W is 3 and H is 2.
    auto reduce_w = Reduce::create(ReduceAttributes{ReduceType::Mean, 3, true}, Backend::CPU);
    auto reduce_h = Reduce::create(ReduceAttributes{ReduceType::Mean, 2, true}, Backend::CPU);
    TensorDesc rd = in_nchw.desc();
    const TensorDesc r_in[] = {rd};
    auto mid_desc = reduce_w->getOutputTensorDesc(r_in);
    std::vector<T> mid_buf(static_cast<size_t>(mid_desc[0].numel()));
    auto mid = test::make_planar(mid_desc[0], mid_buf.data());
    std::vector<T> r_buf(static_cast<size_t>(c.N * c.C));
    auto r_out = test::make_planar(global_desc[0], r_buf.data());

    std::printf("  %-34s [%lld,%lld,%lld,%lld] %s\n", c.name, (long long)c.N,
                (long long)c.C, (long long)c.H, (long long)c.W,
                c.f16 ? "f16" : "f32");

    const double plane_bytes = static_cast<double>(c.N) * c.C * c.H * c.W *
                               (c.f16 ? 2.0 : 4.0);

    for (int nt : threads) {
        BenchPool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use = (nt > 1) ? ctx : ComputeContext{};

        const TensorView g_ins[] = {x};
        const TensorView w_ins[] = {x};
        const TensorView r_ins[] = {in_nchw};
        const TensorView mid_ins[] = {mid};
        TensorView g_outs[] = {g_out};
        TensorView w_outs[] = {w_out};
        TensorView mid_outs[] = {mid};
        TensorView r_outs[] = {r_out};

        const auto run_global = [&] { return global_op->compute(g_outs, g_ins, use, nullptr); };
        const auto run_window = [&] { return window_op->compute(w_outs, w_ins, use, nullptr); };
        const auto run_reduce = [&] {
            const Status s1 = reduce_w->compute(mid_outs, r_ins, use, nullptr);
            return is_ok(s1) ? reduce_h->compute(r_outs, mid_ins, use, nullptr) : s1;
        };

        // One checked pass each: an arm that refuses would time as instant and read as a win.
        const bool ok_global = is_ok(run_global());
        const bool ok_window = is_ok(run_window());
        const bool ok_reduce = is_ok(run_reduce());

        // Every arm has to agree with a host-side mean, not just with each
        // other: a compute that writes nothing would otherwise time as a win.
        // The output is planar, one pixel per channel.
        if (nt == threads.front()) {
            const double tol = c.f16 ? 2e-3 : 1e-4;  // f16 lanes round over the walk
            auto worst = [&](const std::vector<T>& a, const T* want, size_t n) {
                double d = 0.0;
                size_t at = 0;
                for (size_t i = 0; i < a.size() && i < n; ++i) {
                    const double e = std::abs(static_cast<double>(simd::s_load(&a[i])) -
                                              static_cast<double>(simd::s_load(&want[i])));
                    if (e > d) { d = e; at = i; }
                }
                if (d > tol) {
                    std::printf("    !! arm off by %.3g @%zu (want %.5f, got %.5f)\n", d, at,
                                static_cast<double>(simd::s_load(&want[at])),
                                static_cast<double>(simd::s_load(&a[at])));
                }
                return d;
            };

            // Ground truth: the mean of each channel plane of the planar input.
            std::vector<T> want(static_cast<size_t>(c.N * c.C));
            const int64_t plane = c.H * c.W;
            for (int64_t ch = 0; ch < c.N * c.C; ++ch) {
                double sum = 0.0;
                for (int64_t k = 0; k < plane; ++k) {
                    sum += static_cast<double>(simd::s_load(&in_vec[static_cast<size_t>(ch * plane + k)]));
                }
                simd::s_store(&want[static_cast<size_t>(ch)],
                              static_cast<float>(sum / static_cast<double>(plane)));
            }
            worst(g_buf, want.data(), want.size());
            worst(w_buf, want.data(), want.size());
            if (ok_reduce) { worst(r_buf, want.data(), want.size()); }
        }

        std::vector<double> t_g, t_w, t_r;
        for (int r = 0; r < rounds; ++r) {
            if (ok_global) { t_g.push_back(time_us(iters, run_global)); }
            if (ok_window) { t_w.push_back(time_us(iters, run_window)); }
            if (ok_reduce) { t_r.push_back(time_us(iters, run_reduce)); }
        }

        std::printf("    %2dT  ", nt);
        const double g = median(t_g);
        std::printf("global %8.2f us (%6.1f GB/s)", g, plane_bytes / g / 1e3);
        std::printf(" | window %8.2f us (%.1fx)", median(t_w), median(t_w) / g);
        if (ok_reduce) {
            std::printf(" | reduce %8.2f us (%.1fx)", median(t_r), median(t_r) / g);
        } else {
            std::printf(" | reduce unsupported");
        }
        std::printf("\n");
    }
}

// ============================================================
// Layouts: NHWC against NCHW
// ============================================================

struct LayoutCase {
    const char* name;
    int64_t C, H, W;
    bool f16 = false;
};

template <typename T>
void run_layout_case(const LayoutCase& c, const std::vector<int>& threads,
                     int rounds, int iters)
{
    constexpr DataType dt = std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
    const int64_t shape[] = {1, c.C, c.H, c.W};
    auto [in_vec, in_nchw] = make_random<T>(shape, 11);

    // NHWC: all C channels dense per pixel.
    auto cl_vec = test::to_channel_last<T>(
        std::span<const T>(in_vec.data(), in_vec.size()), 1, c.C, c.H * c.W);
    TensorView in_cl(shape, dt, cl_vec.data(), TensorLayout::NHWC);

    const TensorView* ins[2] = {&in_nchw, &in_cl};
    const char* const names[2] = {"NCHW", "NHWC"};

    struct Arm {
        TensorView in;
        std::unique_ptr<Pooling> op;
        std::vector<T> buf;
        TensorView out;
    };

    auto make_mode = [&](const PoolingAttributes& attrs) {
        std::array<Arm, 2> arms;
        for (int i = 0; i < 2; ++i) {
            arms[i].in = *ins[i];
            arms[i].op = Pooling::create(attrs, Backend::CPU);
            TensorDesc d = ins[i]->desc();
            const TensorDesc ad[] = {d};
            auto od = arms[i].op->getOutputTensorDesc(ad);
            const int64_t storage = od[0].storage_bytes() / static_cast<int64_t>(sizeof(T));
            arms[i].buf.assign(static_cast<size_t>(std::max(od[0].numel(), storage)), T{});
            arms[i].out = test::make_planar(od[0], arms[i].buf.data());
        }
        return arms;
    };

    PoolingAttributes avg_attrs;
    avg_attrs.type = PoolingType::Average;
    avg_attrs.global = true;
    // 3x3 stride 2 with one pad: the stem max-pool the ImageNet CNNs run, and a
    // full-extent window would time a reduction rather than a pooling.
    PoolingAttributes max_attrs;
    max_attrs.type = PoolingType::Max;
    max_attrs.kernel_shape = {1, 3, 3};
    max_attrs.stride = {1, 2, 2};
    max_attrs.padding = {0, 1, 1};

    auto avg_arms = make_mode(avg_attrs);
    auto max_arms = make_mode(max_attrs);
    const char* const mode_names[2] = {"avg (global)", "max 3x3 s2"};

    // Ground truth: the plane's mean and max on the host from the planar input;
    // every arm of every mode has to land on these.
    const int64_t plane = c.H * c.W;
    std::vector<T> want_avg(static_cast<size_t>(c.C));
    for (int64_t ch = 0; ch < c.C; ++ch) {
        double sum = 0.0;
        const T* p = in_vec.data() + ch * plane;
        for (int64_t k = 0; k < plane; ++k) {
            sum += static_cast<double>(simd::s_load(p + k));
        }
        simd::s_store(&want_avg[static_cast<size_t>(ch)],
                      static_cast<float>(sum / static_cast<double>(plane)));
    }

    // The window reference: the planar max over a 3x3 stride-2 padded window.
    const int64_t OH = (c.H + 2 - 3) / 2 + 1;
    const int64_t OW = (c.W + 2 - 3) / 2 + 1;
    std::vector<T> want_max(static_cast<size_t>(c.C * OH * OW));
    for (int64_t ch = 0; ch < c.C; ++ch) {
        const T* p = in_vec.data() + ch * plane;
        for (int64_t oh = 0; oh < OH; ++oh) {
            for (int64_t ow = 0; ow < OW; ++ow) {
                double hi = -1e30;
                for (int64_t kh = 0; kh < 3; ++kh) {
                    const int64_t ih = oh * 2 + kh - 1;
                    if (ih < 0 || ih >= c.H) { continue; }
                    for (int64_t kw = 0; kw < 3; ++kw) {
                        const int64_t iw = ow * 2 + kw - 1;
                        if (iw < 0 || iw >= c.W) { continue; }
                        hi = std::max(hi, static_cast<double>(
                                          simd::s_load(p + ih * c.W + iw)));
                    }
                }
                simd::s_store(&want_max[static_cast<size_t>((ch * OH + oh) * OW + ow)],
                              static_cast<float>(hi));
            }
        }
    }

    std::printf("  %-28s [1,%lld,%lld,%lld] %s\n", c.name, (long long)c.C,
                (long long)c.H, (long long)c.W, c.f16 ? "f16" : "f32");

    for (int nt : threads) {
        BenchPool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use = (nt > 1) ? ctx : ComputeContext{};

        for (int m = 0; m < 2; ++m) {
            auto& arms = (m == 0) ? avg_arms : max_arms;
            const std::vector<T>& want = (m == 0) ? want_avg : want_max;

            auto run = [&](int i) {
                TensorView o[] = {arms[i].out};
                const TensorView x[] = {arms[i].in};
                return arms[i].op->compute(o, x, use, nullptr);
            };

            const bool ok[2] = {is_ok(run(0)), is_ok(run(1))};
            if (nt == threads.front()) {
                const double tol = c.f16 ? 2e-3 : 1e-4;
                for (int i = 0; i < 2; ++i) {
                    if (!ok[i]) {
                        std::printf("    !! %s/%s refused\n", mode_names[m], names[i]);
                        continue;
                    }
                    double worst = 0.0;
                    int64_t at = 0;
                    for (int64_t ch = 0; ch < c.C; ++ch) {
                        for (int64_t e = 0; e < ((m == 0) ? 1 : OH * OW); ++e) {
                            // Channel-last: a pixel's channels are dense.
                            const bool cl =
                                arms[i].out.layout() == TensorLayout::NHWC;
                            int64_t k = (m == 0) ? ch : ch * OH * OW + e;
                            if (cl) { k = (m == 0) ? ch : e * c.C + ch; }
                            const double got = static_cast<double>(
                                simd::s_load(arms[i].buf.data() + k));
                            const double have = static_cast<double>(simd::s_load(
                                &want[static_cast<size_t>((m == 0) ? ch
                                                                  : ch * OH * OW + e)]));
                            const double d = std::abs(got - have);
                            if (d > worst) { worst = d; at = ch; }
                        }
                    }
                    if (worst > tol) {
                        std::printf("    !! %s/%s off by %.3g @%lld\n", mode_names[m],
                                    names[i], worst, (long long)at);
                    }
                }
            }

            std::vector<double> a1, cl, a2;
            for (int r = 0; r < rounds; ++r) {
                a1.push_back(time_us(iters, [&] { run(0); }));
                cl.push_back(time_us(iters, [&] { run(1); }));
                a2.push_back(time_us(iters, [&] { run(0); }));
            }
            const double tp = median(a1);
            const double tc = median(cl);
            const double null_gap = (median(a2) / tp - 1.0) * 100.0;

            std::printf("    %-12s %2dT  NCHW %8.2f us | NHWC %8.2f us"
                        " | planar/nhwc %5.2fx  (null %+4.1f%%)\n",
                        mode_names[m], nt, tp, tc, tp / tc, null_gap);
        }
    }
}

}  // namespace

NNOPS_BENCH(pooling_global) {
    const std::vector<int> threads = bench_threads();
    const int rounds = 5;
    const int iters = 50;

    std::printf("global pooling: whole-plane scan vs the full-extent window walk "
                "vs two Reduce(Mean)\n");
    std::printf("  pool=%s threads=", nnops::test::bench_pool_name());
    for (size_t i = 0; i < threads.size(); ++i) {
        std::printf("%d%s", threads[i], i + 1 < threads.size() ? "," : "\n");
    }

    const GlobalCase cases[] = {
        {"resnet18 avgpool",        1,  512, 7, 7},
        {"mobilenetv2 avgpool",     1, 1280, 7, 7},
        {"backbone plane 56x56",    1,  256, 56, 56},
        {"resnet18 avgpool (f16)",  1,  512, 7, 7, true},
    };

    for (const auto& c : cases) {
        if (c.f16) {
            run_case<half>(c, threads, rounds, iters);
        } else {
            run_case<float>(c, threads, rounds, iters);
        }
    }
}

NNOPS_BENCH(pooling_layouts) {
    const std::vector<int> threads = bench_threads();
    const int rounds = 5;
    const int iters = 50;

    std::printf("pooling layouts: NCHW against NHWC over one shape and kernel\n");
    std::printf("  pool=%s threads=", nnops::test::bench_pool_name());
    for (size_t i = 0; i < threads.size(); ++i) {
        std::printf("%d%s", threads[i], i + 1 < threads.size() ? "," : "\n");
    }

    const LayoutCase cases[] = {
        {"resnet18 avgpool",      512,  7,  7},
        {"mobilenetv2 avgpool",  1280,  7,  7},
        {"backbone plane 56x56",  256, 56, 56},
        {"narrow 100 channels",   100, 14, 14},
        {"resnet18 avgpool f16",  512,  7,  7, true},
    };

    for (const auto& c : cases) {
        if (c.f16) {
            run_layout_case<half>(c, threads, rounds, iters);
        } else {
            run_layout_case<float>(c, threads, rounds, iters);
        }
    }
}
