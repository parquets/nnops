/// @file bench_conv2d_nhwc.cpp
/// @brief Channel-last vs planar Conv2D on the same layer — an interleaved A/B.
///
/// One row per layer. `nchw` is the planar route (NCHW activations, OIHW weight,
/// im2col + GEMM) and `nhwc` is the channel-last one (NHWC activations, HWIO
/// weight, pixel tiles straight out of the row). Both run on identical values,
/// converted once outside the timed region, so what a row shows is the data
/// layout and everything it decides: the gather's access pattern, whether a 1x1
/// needs an im2col buffer at all, and how a pixel tile fills the GEMM.
///
/// The measurement is the interleaved A/B the rest of the benches use
/// (bench_conv2d.cpp): within a round the two alternate, the second planar run
/// is the null control, and a verdict is only called real when it clears the
/// drift floor. Three tables, one per value type: f32, f16, and the quantized
/// s8-weight / u8-activation route.
///
/// The shapes are the layers a channel-last framework actually runs — 1x1
/// expansions and projections, 3x3 stacks, 5x5/7x7 stems, grouped and batched
/// layers — because that is the population the layout has to win on, not one
/// square.

#include "nnops/ops/conv2d.hpp"
#include "common/bench_harness.hpp"
#include "common/channel_last.hpp"
#include "common/cpu_pool.hpp"
#include "common/test_helpers.hpp"
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
// Pool / timing — the same instruments as bench_conv2d.cpp
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
};

/// ResNet / MobileNet / Inception layer shapes, a few per family, plus the
/// awkward ones (a narrow plane, a wide plane, a fan-out 1x1, a batched layer).
const std::vector<Shape>& sweep() {
    static const std::vector<Shape> s = {
        // ---- 1x1: the dominant shape in a modern network --------------------
        {"pointwise 1x1 56x56  64->64",     1,   64,  56,  56,   64, 1, 1},
        {"expand    1x1 56x56  24->144",    1,   24,  56,  56,  144, 1, 1},
        {"project   1x1 56x56 144->24",     1,  144,  56,  56,   24, 1, 1},
        {"pointwise 1x1 28x28 128->128",    1,  128,  28,  28,  128, 1, 1},
        {"project   1x1 14x14 576->96",     1,  576,  14,  14,   96, 1, 1},
        {"classifier 1x1  7x7 1280->1280",  1, 1280,   7,   7, 1280, 1, 1},
        // ---- 3x3 ------------------------------------------------------------
        {"conv 3x3 s1 56x56  64->64",       1,   64,  56,  56,   64, 3, 3, 1, 1, 1, 1},
        {"conv 3x3 s2 56x56 128->128",      1,  128,  56,  56,  128, 3, 3, 2, 2, 1, 1},
        {"conv 3x3 s1 28x28 256->256",      1,  256,  28,  28,  256, 3, 3, 1, 1, 1, 1},
        {"conv 3x3 s2 14x14 512->512",      1,  512,  14,  14,  512, 3, 3, 2, 2, 1, 1},
        {"conv 3x3 s1  7x7  512->512",      1,  512,   7,   7,  512, 3, 3, 1, 1, 1, 1},
        // ---- wide kernels ---------------------------------------------------
        {"stem 7x7 s2 224x224 3->64",       1,    3, 224, 224,   64, 7, 7, 2, 2, 3, 3},
        {"conv 5x5 s1 28x28  32->32",       1,   32,  28,  28,   32, 5, 5, 1, 1, 2, 2},
        {"conv 5x5 s2 56x56  16->32",       1,   16,  56,  56,   32, 5, 5, 2, 2, 2, 2},
        // ---- grouped / batched ----------------------------------------------
        {"group 3x3 56x56 128->128 g4",     1,  128,  56,  56,  128, 3, 3, 1, 1, 1, 1, 4},
        {"depthwise-shaped 3x3 112x112 32", 1,   32, 112, 112,   32, 3, 3, 1, 1, 1, 1, 32},
        {"conv 3x3 s1 14x14 128->128 n8",   8,  128,  14,  14,  128, 3, 3, 1, 1, 1, 1},
    };
    return s;
}

// ============================================================
// Floating-point A/B: one layer in both layouts
// ============================================================

template <class T>
constexpr DataType dtype_of() {
    return std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
}

template <class T>
std::vector<T> random_vec(std::span<const int64_t> shape, uint64_t seed) {
    if constexpr (std::is_same_v<T, float>) {
        auto [v, _] = test::make_random_tensor(shape, -1.0f, 1.0f, seed);
        return v;
    } else {
        auto [v, _] = test::make_random_f16_tensor(shape, -1.0f, 1.0f, seed);
        return v;
    }
}

template <class T>
struct AbFp {
    explicit AbFp(const Shape& c) : s(c) {
        icg = c.IC / c.groups;
        const int64_t ish[] = {c.N, c.IC, c.IH, c.IW};
        const int64_t wsh[] = {c.OC, icg, c.KH, c.KW};        // OIHW
        const int64_t wsh_cl[] = {c.KH, c.KW, icg, c.OC};     // HWIO
        const int64_t bsh[] = {c.OC};
        constexpr DataType dt = dtype_of<T>();

        const auto in32 = random_vec<T>(std::span<const int64_t>(ish, 4), 1);
        const auto w32  = random_vec<T>(std::span<const int64_t>(wsh, 4), 2);
        bias     = random_vec<T>(std::span<const int64_t>(bsh, 1), 3);
        in_planar = in32;
        w_planar  = w32;
        in_cl     = test::nchw_to_nhwc<T>(std::span<const T>(in32.data(), in32.size()),
                                          c.N, c.IC, c.IH, c.IW);
        w_cl      = test::oihw_to_hwio<T>(std::span<const T>(w32.data(), w32.size()),
                                          c.OC, icg, c.KH, c.KW);

        in_nchw = TensorView(std::span<const int64_t>(ish, 4), dt, in_planar.data(),
                             TensorLayout::NCHW);
        w_nchw  = TensorView(std::span<const int64_t>(wsh, 4), dt, w_planar.data(),
                             TensorLayout::NCHW);
        in_nhwc = TensorView(std::span<const int64_t>(ish, 4), dt, in_cl.data(),
                             TensorLayout::NHWC);
        w_nhwc  = TensorView(std::span<const int64_t>(wsh_cl, 4), dt, w_cl.data(),
                             TensorLayout::HWIO);
        b_v     = TensorView(std::span<const int64_t>(bsh, 1), dt, bias.data());

        Conv2DAttributes attrs{};
        attrs.kernel_size = {c.KH, c.KW};
        attrs.stride = {c.SH, c.SW};
        attrs.padding = {c.PH, c.PW};
        attrs.groups = c.groups;
        op = Conv2D::create(attrs, Backend::CPU);

        const TensorDesc d0[] = {in_nchw.desc(), w_nchw.desc()};
        const auto od = op->getOutputTensorDesc(d0);
        OH = od[0].dims[2];
        OW = od[0].dims[3];
        work = 2.0 * static_cast<double>(c.N) * c.OC * OH * OW * icg * c.KH * c.KW;

        out_nchw_buf.assign(static_cast<size_t>(od[0].numel()), T{});
        out_nhwc_buf.assign(static_cast<size_t>(od[0].numel()), T{});
        out_nchw = TensorView(std::span<const int64_t>(od[0].dims.data(), 4), dt,
                              out_nchw_buf.data(), TensorLayout::NCHW);
        out_nhwc = TensorView(std::span<const int64_t>(od[0].dims.data(), 4), dt,
                              out_nhwc_buf.data(), TensorLayout::NHWC);
    }

    void run_nchw(const ComputeContext& ctx) {
        const TensorView ins[] = {in_nchw, w_nchw, b_v};
        op->compute(std::span<TensorView>(&out_nchw, 1), ins, ctx, nullptr);
    }
    void run_nhwc(const ComputeContext& ctx) {
        const TensorView ins[] = {in_nhwc, w_nhwc, b_v};
        op->compute(std::span<TensorView>(&out_nhwc, 1), ins, ctx, nullptr);
    }

    Shape s;
    int64_t icg = 1, OH = 0, OW = 0;
    double work = 0.0;
    std::unique_ptr<Conv2D> op;
    std::vector<T> in_planar, w_planar, in_cl, w_cl, bias;
    std::vector<T> out_nchw_buf, out_nhwc_buf;
    TensorView in_nchw, w_nchw, in_nhwc, w_nhwc, b_v;
    TensorView out_nchw, out_nhwc;
};

// ============================================================
// Quantized A/B: s8 weight, u8 activation, f32 output
// ============================================================

struct AbI8 {
    explicit AbI8(const Shape& c) : s(c) {
        icg = c.IC / c.groups;
        const int64_t ish[] = {c.N, c.IC, c.IH, c.IW};
        const int64_t wsh[] = {c.OC, icg, c.KH, c.KW};        // OIHW
        const int64_t wsh_cl[] = {c.KH, c.KW, icg, c.OC};     // HWIO
        const int64_t bsh[] = {c.OC};
        const float s_x = 1.0f / 128.0f;   // exact in binary, and it fits u8
        const int32_t zp_x = 128;

        const int64_t act_n = c.N * c.IC * c.IH * c.IW;
        act_planar.resize(static_cast<size_t>(act_n));
        {
            test::XorShift128 rng(11);
            for (auto& v : act_planar) {
                v = static_cast<uint8_t>(quant_u8(rng.next_float(-1.0f, 1.0f), s_x, zp_x));
            }
        }
        act_cl = test::nchw_to_nhwc<uint8_t>(
            std::span<const uint8_t>(act_planar.data(), act_planar.size()),
            c.N, c.IC, c.IH, c.IW);

        w_oihw.resize(static_cast<size_t>(c.OC * icg * c.KH * c.KW));
        {
            test::XorShift128 rng(12);
            for (auto& v : w_oihw) {
                v = static_cast<int8_t>(quant_s8(rng.next_float(-1.0f, 1.0f), 127.0f));
            }
        }
        w_hwio = test::oihw_to_hwio<int8_t>(
            std::span<const int8_t>(w_oihw.data(), w_oihw.size()), c.OC, icg, c.KH, c.KW);

        w_s.resize(static_cast<size_t>(c.OC));
        w_zp.assign(static_cast<size_t>(c.OC), 0);
        bias.resize(static_cast<size_t>(c.OC));
        for (int64_t oc = 0; oc < c.OC; ++oc) {
            w_s[static_cast<size_t>(oc)] = 0.01f * static_cast<float>(1 + oc % 5);
        }
        {
            test::XorShift128 rng(13);
            for (auto& v : bias) { v = rng.next_float(-0.5f, 0.5f); }
        }

        Conv2DAttributes attrs{};
        attrs.kernel_size = {c.KH, c.KW};
        attrs.stride = {c.SH, c.SW};
        attrs.padding = {c.PH, c.PW};
        attrs.groups = c.groups;
        op = Conv2D::create(attrs, Backend::CPU);

        QuantParams qa;
        qa.granularity = QuantGranularity::PerTensor;
        qa.scale = s_x;
        qa.zero_point = zp_x;
        QuantParams qw;
        qw.granularity = QuantGranularity::PerChannel;
        qw.scale_data = w_s.data();
        qw.zero_point_data = w_zp.data();
        qw.num_scales = c.OC;

        in_nchw = TensorView(std::span<const int64_t>(ish, 4), DataType::u8,
                             act_planar.data(), TensorLayout::NCHW, qa);
        w_nchw  = TensorView(std::span<const int64_t>(wsh, 4), DataType::s8,
                             w_oihw.data(), TensorLayout::NCHW, qw);
        in_nhwc = TensorView(std::span<const int64_t>(ish, 4), DataType::u8,
                             act_cl.data(), TensorLayout::NHWC, qa);
        w_nhwc  = TensorView(std::span<const int64_t>(wsh_cl, 4), DataType::s8,
                             w_hwio.data(), TensorLayout::HWIO, qw);
        b_v     = TensorView(std::span<const int64_t>(bsh, 1), DataType::f32, bias.data());

        const TensorDesc d0[] = {in_nchw.desc(), w_nchw.desc(), b_v.desc()};
        const auto od = op->getOutputTensorDesc(d0);
        OH = od[0].dims[2];
        OW = od[0].dims[3];
        work = 2.0 * static_cast<double>(c.N) * c.OC * OH * OW * icg * c.KH * c.KW;

        out_nchw_buf.assign(static_cast<size_t>(od[0].numel()), 0.0f);
        out_nhwc_buf.assign(static_cast<size_t>(od[0].numel()), 0.0f);
        out_nchw = TensorView(std::span<const int64_t>(od[0].dims.data(), 4), DataType::f32,
                              out_nchw_buf.data(), TensorLayout::NCHW);
        out_nhwc = TensorView(std::span<const int64_t>(od[0].dims.data(), 4), DataType::f32,
                              out_nhwc_buf.data(), TensorLayout::NHWC);
    }

    static int32_t quant_u8(float x, float inv, int32_t zp) {
        const int32_t q = static_cast<int32_t>(std::nearbyintf(
            std::fmaf(x, inv, static_cast<float>(zp))));
        return std::min(std::max(q, 0), 255);
    }
    static int32_t quant_s8(float x, float inv) {
        const int32_t q = static_cast<int32_t>(std::nearbyintf(x * inv));
        return std::min(std::max(q, -128), 127);
    }

    void run_nchw(const ComputeContext& ctx) {
        const TensorView ins[] = {in_nchw, w_nchw, b_v};
        op->compute(std::span<TensorView>(&out_nchw, 1), ins, ctx, nullptr);
    }
    void run_nhwc(const ComputeContext& ctx) {
        const TensorView ins[] = {in_nhwc, w_nhwc, b_v};
        op->compute(std::span<TensorView>(&out_nhwc, 1), ins, ctx, nullptr);
    }

    Shape s;
    int64_t icg = 1, OH = 0, OW = 0;
    double work = 0.0;
    std::unique_ptr<Conv2D> op;
    std::vector<uint8_t> act_planar, act_cl;
    std::vector<int8_t> w_oihw, w_hwio;
    std::vector<float> w_s, bias;
    std::vector<int32_t> w_zp;
    std::vector<float> out_nchw_buf, out_nhwc_buf;
    TensorView in_nchw, w_nchw, in_nhwc, w_nhwc, b_v;
    TensorView out_nchw, out_nhwc;
};

// ============================================================
// The tables
// ============================================================

/// One table: every layer at every thread count, planar vs channel-last.
template <class MakeOp>
void run_table(const char* title, MakeOp make_op) {
    const int iters = 10, rounds = 3;
    std::printf("\n== %s ==\n", title);
    for (int nt : bench_threads()) {
        std::printf("  %d thread%s\n", nt, nt == 1 ? "" : "s");
        std::printf("    %-34s %8s %8s | %8s %8s | %s | %s\n",
                    "layer", "nchw ms", "GFLOPS", "nhwc ms", "GFLOPS",
                    "nhwc time", "verdict");
        int wins = 0, losses = 0, ties = 0;
        double tot_planar = 0.0, tot_cl = 0.0;
        for (const auto& c : sweep()) {
            auto op = make_op(c);
            const AbResult r = measure_ab(nt,
                [&](const ComputeContext& x) { op->run_nchw(x); },
                [&](const ComputeContext& x) { op->run_nhwc(x); },
                iters, rounds);
            const double ratio = r.b_ms / r.a_ms;   // channel-last / planar time
            const bool real = std::abs(r.a_ms - r.b_ms) / r.a_ms * 100.0 > r.null_pct;
            const char* verdict = !real ? "tie (noise)"
                                : (ratio < 1.0 ? "nhwc wins" : "nchw wins");
            if (!real) { ++ties; } else if (ratio < 1.0) { ++wins; } else { ++losses; }
            tot_planar += r.a_ms;
            tot_cl += r.b_ms;
            std::printf("    %-34s %8.3f %8.1f | %8.3f %8.1f | %8.2fx | %s\n",
                        c.name, r.a_ms, op->work / r.a_ms / 1e6,
                        r.b_ms, op->work / r.b_ms / 1e6, ratio, verdict);
        }
        std::printf("    %-34s %8.3f %8s | %8.3f %8s | %8.2fx | "
                    "%d win, %d lose, %d tie\n",
                    "sum of the layers", tot_planar, "", tot_cl, "",
                    tot_cl / tot_planar, wins, losses, ties);
    }
}

}  // namespace

NNOPS_BENCH(conv2d_nhwc_vs_nchw_f32) {
    run_table("f32: planar (NCHW+OIHW) vs channel-last (NHWC+HWIO)",
              [](const Shape& c) { return std::make_unique<AbFp<float>>(c); });
}

NNOPS_BENCH(conv2d_nhwc_vs_nchw_f16) {
    run_table("f16: planar (NCHW+OIHW) vs channel-last (NHWC+HWIO)",
              [](const Shape& c) { return std::make_unique<AbFp<half>>(c); });
}

NNOPS_BENCH(conv2d_nhwc_vs_nchw_int8) {
    run_table("int8: s8 weight x u8 activation -> f32, planar vs channel-last",
              [](const Shape& c) { return std::make_unique<AbI8>(c); });
}
