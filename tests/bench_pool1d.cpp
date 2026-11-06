/// @file bench_pool1d.cpp
/// @brief Pool1D against Pooling on the identical lift — the acceptance bench
///        for the 1D route, in both layouts and both quantized types.
///
/// Pool1D *is* 2D pooling on the [N, C, 1, L] lift with a {1, K} window, so the
/// only thing to prove is that wrapping the 2D operator costs nothing. Each row
/// is therefore one tensor image handed to two operators: the 1D op, and the
/// Pooling call the caller would have made had they shaped the rank-4 views
/// themselves. Every row should read 1.00x: the adaptive output of length one
/// used to be the exception, because the 1D op answered it with the global
/// plane scan while the 2D one walked the window — that shortcut now lives in
/// the backend (pooling.cpp), so both arms take it and the row ties too.
///
/// The cache to keep in mind when reading a row: the 2D pooling grid is one
/// kernel call per (n, c, od, oh), so with H = 1 it is one call per channel and
/// the L axis is a scan inside it. A pooling rate that moves with the channel
/// count is measuring the window walk; one that moves with the pool size is
/// measuring the barrier. Everything here is bandwidth as a result, and the
/// tables print GB/s of activation read plus output written next to the time.
///
/// Both layouts are timed because the 2D kernel has two routes — channel-first
/// scans one contiguous plane, channel-last lanes the channel four rows at a
/// time — and each must survive the wrap. The quantized route exists only
/// channel-last (the SIMD lane is the channel) and has no adaptive form, which
/// the unsupported rows say outright rather than timing a rejected call.
///
/// Measurement rules follow the other benches: interleaved A/B with the second
/// A as the null control, per-round min, median across rounds, and a ratio only
/// called real when it clears that floor. The order is swapped on alternate
/// rounds, so the pool's wake-up cost lands on each arm equally. The last
/// column is the 1D arm's own scaling from the narrowest pool to each other.
///
/// `NNOPS_BENCH_THREADS=1` (or "1,4") pins the ladder, as in the other benches.

#include "nnops/ops/pool1d.hpp"
#include "nnops/ops/pooling.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "backend/cpu/conv/conv1d_common.h"
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
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

using namespace nnops;
using half = nnops::backend::cpu::half;
namespace cpu = nnops::backend::cpu;

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
    CpuBackend backend_cpu;
    CpuBackend& cpu = backend_cpu;
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

struct AbResult {
    double a_ms = 0.0, b_ms = 0.0;   ///< median over rounds of the per-round min
    double null_pct = 0.0;           ///< median A2/A1 drift — the noise floor
    bool consistent = false;         ///< every round put the same arm ahead
};

template <class FA, class FB>
AbResult measure_ab(int nt, FA&& a, FB&& b, int iters, int rounds) {
    BenchPool pool(nt);
    ComputeContext ctx;
    ctx.cpu = pool.cpu;
    const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

    auto run_a = [&]() { (void)a(use_ctx); };
    auto run_b = [&]() { (void)b(use_ctx); };
    run_a(); run_b(); run_a(); run_b();

    std::vector<double> a_ms, b_ms, null_pct;
    int b_ahead = 0, a_ahead = 0;
    for (int r = 0; r < rounds; ++r) {
        // Alternate which arm opens the round: a small layer pays the pool's
        // wake-up on the first call of a region, and a fixed order would hand
        // that cost to A on every round.
        double a1, b1, a2, b2;
        if (r % 2 == 0) {
            a1 = time_ms(run_a, iters);
            b1 = time_ms(run_b, iters);
            a2 = time_ms(run_a, iters);
            b2 = time_ms(run_b, iters);
        } else {
            b1 = time_ms(run_b, iters);
            a1 = time_ms(run_a, iters);
            b2 = time_ms(run_b, iters);
            a2 = time_ms(run_a, iters);
        }
        const double ar = std::min(a1, a2);
        const double br = std::min(b1, b2);
        a_ms.push_back(ar);
        b_ms.push_back(br);
        (br > ar ? b_ahead : a_ahead) += 1;
        null_pct.push_back(std::abs(a2 - a1) / std::min(a1, a2) * 100.0);
    }
    // A real difference keeps its sign in every round; a magnitude that merely
    // clears the drift once is what these layers produce at random.
    return {median_of(a_ms), median_of(b_ms), median_of(null_pct),
            b_ahead == rounds || a_ahead == rounds};
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
    threads = {1, 2, 4, 8};
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw > 1 && std::find(threads.begin(), threads.end(), hw) == threads.end()) {
        threads.push_back(hw);
    }
    return threads;
}

/// A scanning kernel is priced in elements, not FLOPs: ~20M elements a region.
int auto_iters(int64_t elems) {
    return static_cast<int>(std::clamp(2e7 / std::max<double>(
                                           static_cast<double>(elems), 1.0),
                                       3.0, 400.0));
}

// ============================================================
// The layer list
// ============================================================

struct PoolShape {
    const char* name;
    int64_t N, C, L;
    Pool1DType type = Pool1DType::Max;
    int64_t K = 2, S = 2, P = 0, D = 1;
    bool global = false;
    bool exclude_pad = false;
    int64_t adaptive = 0;
    bool cl = false;
};

/// Audio-model pooling: the squeeze-excite global mean, the backbone max-pools,
/// the adaptive head, and the awkward geometries — then two of the same layers
/// channel-last, where the 2D kernel takes its other route.
const std::vector<PoolShape>& pool_sweep() {
    static const std::vector<PoolShape> s = {
        {"cf se global avg 1x256x400",            1, 256,  400, Pool1DType::Average,
         2, 2, 0, 1, /*global=*/true},
        {"cf backbone max k2 s2 1x64x4000",       1,  64, 4000, Pool1DType::Max, 2, 2},
        {"cf backbone max k3 s2 p1 1x128x1000",   1, 128, 1000, Pool1DType::Max, 3, 2, 1},
        {"cf stem max k5 s4 p2 1x16x8000",        1,  16, 8000, Pool1DType::Max, 5, 4, 2},
        {"cf avg k3 s1 p1 exclude_pad 1x128x200", 1, 128,  200, Pool1DType::Average,
         3, 1, 1, 1, false, /*exclude_pad=*/true},
        {"cf adaptive avg -> 16 1x128x512",       1, 128,  512, Pool1DType::Average,
         2, 2, 0, 1, false, false, /*adaptive=*/16},
        {"cf dilated max k3 d2 p2 1x64x2000",     1,  64, 2000, Pool1DType::Max, 3, 1, 2, 2},
        // The one case where the 1D op does not forward the call unchanged:
        // an adaptive pool to length 1 is answered by the global plane scan.
        {"cf adaptive avg -> 1 1x256x400",        1, 256,  400, Pool1DType::Average,
         2, 2, 0, 1, false, false, /*adaptive=*/1},
        // ---- the same two routes channel-last ---------------------------------
        {"cl backbone max k2 s2 1x64x4000",       1,  64, 4000, Pool1DType::Max, 2, 2, 0, 1,
         false, false, 0, /*cl=*/true},
        {"cl backbone max k3 s2 p1 1x128x1000",   1, 128, 1000, Pool1DType::Max, 3, 2, 1, 1,
         false, false, 0, /*cl=*/true},
        {"cl se global avg 1x256x400",            1, 256,  400, Pool1DType::Average,
         2, 2, 0, 1, /*global=*/true, false, 0, /*cl=*/true},
        {"cl adaptive avg -> 16 1x128x512",       1, 128,  512, Pool1DType::Average,
         2, 2, 0, 1, false, false, /*adaptive=*/16, /*cl=*/true},
    };
    return s;
}

Pool1DAttributes attrs_of(const PoolShape& c) {
    Pool1DAttributes a{};
    a.type         = c.type;
    a.kernel_shape = c.K;
    a.stride       = c.S;
    a.padding      = c.P;
    a.dilation     = c.D;
    a.global       = c.global;
    a.exclude_pad  = c.exclude_pad;
    a.output_size  = c.adaptive;
    return a;
}

/// The 2D attributes Pool1D itself builds: a {1, K} window on the wide lift.
PoolingAttributes attrs_2d(const PoolShape& c) {
    PoolingAttributes a{};
    a.type         = (c.type == Pool1DType::Max) ? PoolingType::Max
                                                 : PoolingType::Average;
    a.kernel_shape = {1, 1, c.K};
    a.stride       = {1, 1, c.S};
    a.padding      = {0, 0, c.P};
    a.dilation     = {1, 1, c.D};
    a.exclude_pad  = c.exclude_pad;
    a.global       = c.global;
    a.output_size  = {0, 1, c.adaptive};
    return a;
}

// ============================================================
// One layer, two operators, whatever the dtype
// ============================================================

template <class T>
constexpr DataType dtype_of() {
    return std::is_same_v<T, float>   ? DataType::f32
           : std::is_same_v<T, half>  ? DataType::f16
           : std::is_same_v<T, uint8_t> ? DataType::u8
                                        : DataType::s8;
}

template <class T>
std::vector<T> random_vec(int64_t n, uint64_t seed) {
    std::vector<T> v(static_cast<size_t>(n));
    test::XorShift128 rng(seed);
    for (int64_t i = 0; i < n; ++i) {
        const float f = rng.next_float(-1.0f, 1.0f);
        if constexpr (std::is_same_v<T, float>) {
            v[static_cast<size_t>(i)] = f;
        } else if constexpr (std::is_same_v<T, half>) {
            simd::s_store(&v[static_cast<size_t>(i)], f);
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            v[static_cast<size_t>(i)] = static_cast<uint8_t>(rng.next_u64() % 256);
        } else {
            v[static_cast<size_t>(i)] = static_cast<int8_t>(
                static_cast<int32_t>(rng.next_u64() % 256) - 128);
        }
    }
    return v;
}

/// One Pool1D layer and the Pooling call on its lift. @p quant selects the
/// integer route, which carries a QuantParams on both views and is
/// channel-last only.
template <class T>
struct PoolPair {
    explicit PoolPair(const PoolShape& c, bool quant = false) : s(c) {
        constexpr DataType dt = dtype_of<T>();
        const int64_t numel = c.N * c.C * c.L;

        if constexpr (std::is_integral_v<T>) {
            // The integer route: the whole storage range, so the kernel is not
            // timed on a stream that happens to compress.
            buf.assign(static_cast<size_t>(numel), T{});
            test::XorShift128 rng(6);
            for (int64_t i = 0; i < numel; ++i) {
                const int32_t v = static_cast<int32_t>(rng.next_u64() % 256);
                buf[static_cast<size_t>(i)] =
                    std::is_same_v<T, uint8_t> ? static_cast<T>(v)
                                               : static_cast<T>(v - 128);
            }
        } else {
            buf = random_vec<T>(numel, 5);
        }
        if (c.cl) {
            buf = test::to_channel_last<T>(
                std::span<const T>(buf.data(), buf.size()), c.N, c.C, c.L);
        }

        QuantParams qp;
        qp.granularity = QuantGranularity::PerTensor;
        qp.scale = 1.0f / 32.0f;
        qp.zero_point = std::is_same_v<T, uint8_t> ? 128 : 0;

        const int64_t sh[] = {c.N, c.C, c.L};
        const TensorLayout lay = c.cl ? TensorLayout::NHWC : TensorLayout::NCHW;
        in = TensorView(std::span<const int64_t>(sh, 3), dt, buf.data(), lay,
                        quant ? qp : QuantParams{});

        op1d = Pool1D::create(attrs_of(c), Backend::CPU);
        const TensorDesc d[] = {in.desc()};
        const auto od = op1d->getOutputTensorDesc(d);
        OL = od[0].dims[2];

        out_buf.assign(static_cast<size_t>(od[0].numel()), T{});
        out = TensorView(od[0].dims, dt, out_buf.data(), lay,
                         quant ? qp : QuantParams{});
        ins1d = {in};

        // The control arm: the operator's own lift, straight to Pooling.
        in4  = cpu::lift_wide(in);
        out4 = cpu::lift_wide(out);
        op2d = Pooling::create(attrs_2d(c), Backend::CPU);
        ins2d = {in4};

        // Read plus write, in bytes: what a streaming rate should be.
        bytes = static_cast<double>(numel + od[0].numel()) *
                static_cast<double>(data_type_size(dt));
    }

    Status run_1d(const ComputeContext& ctx) { return op1d->compute(out, ins1d, ctx, nullptr); }
    Status run_2d(const ComputeContext& ctx) { return op2d->compute(out4, ins2d, ctx, nullptr); }

    PoolShape s;
    std::vector<T> buf, out_buf;
    TensorView in, out, in4, out4;
    std::vector<TensorView> ins1d, ins2d;
    std::unique_ptr<Pool1D> op1d;
    std::unique_ptr<Pooling> op2d;
    int64_t OL = 0;
    double bytes = 0.0;
};

// ============================================================
// Table
// ============================================================

template <class T>
void run_table(const char* tag, bool quant) {
    const std::vector<int> threads = bench_threads();
    const int rounds = 4;   // even, so the alternating order balances
    std::printf("  %s — Pool1D against Pooling on the same [N, C, 1, L] lift"
                " (2d/1d above 1.00 means the 1D route won)\n", tag);
    std::printf("    %-42s %7s %7s | %9s %8s | %9s %8s | %6s %6s | %s\n",
                "case", "L_in", "L_out", "1d ms", "GB/s", "2d ms", "GB/s",
                "2d/1d", "null", "1d 1T->nT");

    double sum_a = 0.0, sum_b = 0.0, sum_bytes = 0.0;
    for (const PoolShape& c : pool_sweep()) {
        // Integer pooling is channel-last only; the float tables run both.
        if (quant && !c.cl) { continue; }
        PoolPair<T> l(c, quant);
        const Status sa = l.run_1d(ComputeContext{});
        const Status sb = l.run_2d(ComputeContext{});
        if (!is_ok(sa) || !is_ok(sb)) {
            std::printf("    %-42s %7lld %7lld | %s%s%s\n", c.name,
                        (long long)c.L, (long long)l.OL,
                        is_ok(sa) ? "" : "1d unsupported ",
                        is_ok(sb) ? "" : "2d unsupported ",
                        (!is_ok(sa) && !is_ok(sb)) ? "(both reject the geometry)" : "");
            continue;
        }
        const int iters = auto_iters(c.N * c.C * c.L);
        double base_gbs = 0.0;
        for (size_t i = 0; i < threads.size(); ++i) {
            const AbResult m = measure_ab(threads[i],
                [&](const ComputeContext& x) { return l.run_1d(x); },
                [&](const ComputeContext& x) { return l.run_2d(x); },
                iters, rounds);
            const double a_gbs = l.bytes / m.a_ms / 1e6;
            if (i == 0) { base_gbs = a_gbs; }
            const bool real = m.consistent &&
                std::abs(m.a_ms - m.b_ms) / m.a_ms * 100.0 > m.null_pct;
            char scale[32];
            std::snprintf(scale, sizeof(scale), "%2dT %.2fx", threads[i],
                          base_gbs > 0.0 ? a_gbs / base_gbs : 0.0);
            std::printf("    %-42s %7lld %7lld | %9.4f %8.1f | %9.4f %8.1f | %5.2fx %5.1f%% | %s%s\n",
                        i == 0 ? c.name : "", (long long)c.L, (long long)l.OL,
                        m.a_ms, a_gbs, m.b_ms, l.bytes / m.b_ms / 1e6,
                        m.b_ms / m.a_ms, m.null_pct, scale,
                        real ? "" : " (noise)");
            if (i == 0) {
                sum_a += m.a_ms;
                sum_b += m.b_ms;
                sum_bytes += l.bytes;
            }
        }
    }
    std::printf("    %-42s %7s %7s | %9.4f %8.1f | %9.4f %8.1f | %5.2fx\n",
                "sum of the layers (1T)", "", "", sum_a, sum_bytes / sum_a / 1e6,
                sum_b, sum_bytes / sum_b / 1e6, sum_b / sum_a);
}

}  // namespace

// ============================================================
// Benches
// ============================================================

NNOPS_BENCH(pool1d_f32_vs_pooling_lift) {
    run_table<float>("pool1d f32", false);
}

NNOPS_BENCH(pool1d_f16_vs_pooling_lift) {
    run_table<half>("pool1d f16", false);
}

NNOPS_BENCH(pool1d_int8_u8_vs_pooling_lift) {
    run_table<uint8_t>("pool1d u8 (act/out), channel-last", true);
}

NNOPS_BENCH(pool1d_int8_s8_vs_pooling_lift) {
    run_table<int8_t>("pool1d s8 (act/out), channel-last", true);
}
