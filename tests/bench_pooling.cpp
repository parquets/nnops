/// @file bench_pooling.cpp
/// @brief Channel-last pooling micro-benchmarks: the global mean against a
///        full-extent window, and the 3x3 window routes.
///
/// The layouts bench holds shape and kernel fixed: a global mean and a 3x3
/// stride-2 max, both channel-last, with the pixel channels dense.
///
/// The window bench isolates the channel-last 3x3 route: the straight-line
/// `pool_channel_last_rows_3x3` against the rolled `pool_channel_last_rows` walk, over the same buffers,
/// by flipping `detail::pooling_3x3_roll`. Max/avg, stride 1/2, one-block and
/// multi-block shapes, so a per-call or per-block cost shows up separately
/// from the tap work.
///
/// `NNOPS_BENCH_THREADS=1` (or "1,4") pins the ladder, as in the other benches.

#include "nnops/ops/pooling.hpp"
#include "nnops/detail/bench_knobs.hpp"
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

template <typename Fn>
double time_us(int iters, Fn&& fn) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) { fn(); }
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

// ============================================================
// Layouts: channel-last
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
    auto [in_vec, in_raw] = make_random<T>(shape, 11);
    (void)in_raw;

    // Channel-last: all C channels dense per pixel.
    auto cl_vec = test::to_channel_last<T>(
        std::span<const T>(in_vec.data(), in_vec.size()), 1, c.C, c.H * c.W);
    TensorView in_cl(shape, dt, cl_vec.data(), TensorLayout::NHWC);

    struct Arm {
        TensorView in;
        std::unique_ptr<Pooling> op;
        std::vector<T> buf;
        TensorView out;
    };

    auto make_mode = [&](const PoolingAttributes& attrs) {
        Arm arm;
        arm.in = in_cl;
        arm.op = Pooling::create(attrs, Backend::CPU);
        TensorDesc d = in_cl.desc();
        const TensorDesc ad[] = {d};
        auto od = arm.op->getOutputTensorDesc(ad);
        const int64_t storage = od[0].storage_bytes() / static_cast<int64_t>(sizeof(T));
        arm.buf.assign(static_cast<size_t>(std::max(od[0].numel(), storage)), T{});
        arm.out = test::make_channel_first(od[0], arm.buf.data());
        return arm;
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

    auto avg_arm = make_mode(avg_attrs);
    auto max_arm = make_mode(max_attrs);
    const char* const mode_names[2] = {"avg (global)", "max 3x3 s2"};

    // Ground truth: the plane's mean and max on the host from the generated input;
    // every mode has to land on these.
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

    // The window reference: the per-channel max over a 3x3 stride-2 padded window.
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
            auto& arm = (m == 0) ? avg_arm : max_arm;
            const std::vector<T>& want = (m == 0) ? want_avg : want_max;

            auto run = [&] {
                TensorView o[] = {arm.out};
                const TensorView x[] = {arm.in};
                return arm.op->compute(o, x, use, nullptr);
            };

            const bool ok = is_ok(run());
            if (nt == threads.front()) {
                const double tol = c.f16 ? 2e-3 : 1e-4;
                if (!ok) {
                    std::printf("    !! %s refused\n", mode_names[m]);
                } else {
                    double worst = 0.0;
                    int64_t at = 0;
                    for (int64_t ch = 0; ch < c.C; ++ch) {
                        for (int64_t e = 0; e < ((m == 0) ? 1 : OH * OW); ++e) {
                            // Channel-last: a pixel's channels are dense.
                            const int64_t k = (m == 0) ? ch : e * c.C + ch;
                            const double got = static_cast<double>(
                                simd::s_load(arm.buf.data() + k));
                            const double have = static_cast<double>(simd::s_load(
                                &want[static_cast<size_t>((m == 0) ? ch
                                                                  : ch * OH * OW + e)]));
                            const double d = std::abs(got - have);
                            if (d > worst) { worst = d; at = ch; }
                        }
                    }
                    if (worst > tol) {
                        std::printf("    !! %s off by %.3g @%lld\n", mode_names[m],
                                    worst, (long long)at);
                    }
                }
            }

            std::vector<double> t;
            for (int r = 0; r < rounds; ++r) {
                t.push_back(time_us(iters, run));
            }
            const double ms = median(t);

            std::printf("    %-12s %2dT  channel-last %8.2f us\n",
                        mode_names[m], nt, ms);
        }
    }
}

// ============================================================
// Window 3x3: the straight-line route against the rolled walk
// ============================================================

struct WindowCase {
    const char* name;
    int64_t C, H, W;
    bool f16 = false;
};

/// Flips `detail::pooling_3x3_roll`, so one process times both routes over the
/// same buffers.
void set_pool_route(bool roll)
{
    ::nnops::detail::pooling_3x3_roll.store(roll ? 1 : 0, std::memory_order_relaxed);
}

template <typename T>
void run_window_case(const WindowCase& c, const std::vector<int>& threads,
                     int rounds, int iters)
{
    constexpr DataType dt = std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
    const int64_t shape[] = {1, c.C, c.H, c.W};
    auto [in_vec, in_raw] = make_random<T>(shape, 31);
    (void)in_raw;
    auto cl_vec = test::to_channel_last<T>(
        std::span<const T>(in_vec.data(), in_vec.size()), 1, c.C, c.H * c.W);
    TensorView in_cl(shape, dt, cl_vec.data(), TensorLayout::NHWC);

    struct Mode {
        const char* name;
        PoolingType type;
        int64_t stride;
    };
    const Mode modes[] = {
        {"max 3x3 s2", PoolingType::Max, 2},
        {"avg 3x3 s2", PoolingType::Average, 2},
        {"max 3x3 s1", PoolingType::Max, 1},
        {"avg 3x3 s1", PoolingType::Average, 1},
    };
    const double in_bytes = static_cast<double>(c.C) * c.H * c.W *
                            (c.f16 ? 2.0 : 4.0);

    std::printf("  %-26s [1,%lld,%lld,%lld] %s\n", c.name, (long long)c.C,
                (long long)c.H, (long long)c.W, c.f16 ? "f16" : "f32");

    for (const Mode& m : modes) {
        PoolingAttributes attrs;
        attrs.type = m.type;
        attrs.kernel_shape = {1, 3, 3};
        attrs.stride = {1, m.stride, m.stride};
        attrs.padding = {0, 1, 1};
        auto op = Pooling::create(attrs, Backend::CPU);
        const TensorDesc d = in_cl.desc();
        const TensorDesc ad[] = {d};
        auto od = op->getOutputTensorDesc(ad);
        const int64_t elems = std::max<int64_t>(
            od[0].numel(), static_cast<int64_t>(od[0].storage_bytes() / sizeof(T)));
        std::vector<T> a_buf(static_cast<size_t>(elems)), b_buf(static_cast<size_t>(elems));
        auto a_out = test::make_channel_first(od[0], a_buf.data());
        auto b_out = test::make_channel_first(od[0], b_buf.data());

        const TensorView x[] = {in_cl};
        TensorView oa[] = {a_out};
        TensorView ob[] = {b_out};

        std::printf("    %-10s", m.name);
        for (int nt : threads) {
            BenchPool pool(nt);
            ComputeContext ctx;
            ctx.cpu = pool.cpu;
            const ComputeContext& use = (nt > 1) ? ctx : ComputeContext{};

            set_pool_route(false);
            const Status sa = op->compute(oa, x, use, nullptr);
            set_pool_route(true);
            const Status sb = op->compute(ob, x, use, nullptr);
            set_pool_route(false);

            if (!is_ok(sa) || !is_ok(sb)) {
                std::printf("  %2dT unsupported", nt);
                continue;
            }

            // Max is exact between the routes; avg reassociates the taps.
            if (nt == threads.front()) {
                const double tol = (m.type == PoolingType::Max)
                                       ? 0.0 : (c.f16 ? 2e-3 : 1e-5);
                double worst = 0.0;
                for (int64_t i = 0; i < elems; ++i) {
                    const double d0 = std::abs(
                        static_cast<double>(simd::s_load(&a_buf[static_cast<size_t>(i)])) -
                        static_cast<double>(simd::s_load(&b_buf[static_cast<size_t>(i)])));
                    worst = std::max(worst, d0);
                }
                if (worst > tol) {
                    std::printf(" [routes differ %.3g]", worst);
                }
            }

            std::vector<double> ta, tb;
            for (int r = 0; r < rounds; ++r) {
                set_pool_route(false);
                ta.push_back(time_us(iters, [&] { op->compute(oa, x, use, nullptr); }));
                set_pool_route(true);
                tb.push_back(time_us(iters, [&] { op->compute(ob, x, use, nullptr); }));
            }
            set_pool_route(false);
            const double tn = median(ta), tr = median(tb);
            std::printf("  %2dT 3x3 %8.2f us | rolled %8.2f us | %5.2fx | %6.1f GB/s",
                        nt, tn, tr, tr / tn, in_bytes / tn / 1e3);
        }
        std::printf("\n");
    }
}

}  // namespace

NNOPS_BENCH(pooling_channel_last) {
    const std::vector<int> threads = bench_threads();
    const int rounds = 5;
    const int iters = 50;

    std::printf("pooling channel-last over one shape and kernel\n");
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

NNOPS_BENCH(pooling_window_3x3) {
    const std::vector<int> threads = bench_threads();
    const int rounds = 3;
    const int iters = 200;  // the 56x56 plane is ~100 us a call, still ~0.1 s a round

    std::printf("pooling window 3x3: the straight-line route against the rolled walk\n");
    std::printf("  pool=%s threads=", nnops::test::bench_pool_name());
    for (size_t i = 0; i < threads.size(); ++i) {
        std::printf("%d%s", threads[i], i + 1 < threads.size() ? "," : "\n");
    }

    const WindowCase cases[] = {
        {"one block 7x7",         16,  7,  7},
        {"two blocks 7x7",        32,  7,  7},
        {"one block 56x56",       16, 56, 56},
        {"resnet18 pool 7x7",    512,  7,  7},
        {"narrow 100 @14x14",    100, 14, 14},
        {"backbone 256 @56x56",  256, 56, 56},
        {"resnet18 pool 7x7 f16", 512, 7, 7, true},
        {"backbone 256 f16",     256, 56, 56, true},
    };

    for (const auto& c : cases) {
        if (c.f16) {
            run_window_case<half>(c, threads, rounds, iters);
        } else {
            run_window_case<float>(c, threads, rounds, iters);
        }
    }
}


