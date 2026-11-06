/// @file bench_conv1d.cpp
/// @brief Conv1D against Conv2D on the identical lift — the acceptance bench
///        for the 1D route.
///
/// Conv1D owns no kernel: it lifts [N, C, L] to [N, C, 1, L] and calls the 2D
/// implicit-GEMM entry, which cuts the columns when the output is one row
/// (wide_split_chunks in conv2d_implicit_gemm.cpp). Both arms are therefore the
/// same image handed to two operators, and should read 1.00x — anything else is
/// the adapter's own cost or a drift artifact. The depthwise rows are the
/// exception: Conv1D takes the 2D op's depthwise routing while the 2D arm here
/// is forced to ImplicitGemm, so those compare the depthwise scan against the
/// grouped GEMM rather than the two operators.
///
/// Measurement rules follow the other benches: interleaved A/B with the second
/// A as the null control (same op, same buffers), per-round min, median across
/// rounds. The order is swapped on alternate rounds — a small layer pays its
/// pool's wake-up on the first call of a region, so a fixed order would charge
/// that to the 1D arm every round and report it as a kernel difference. A row
/// is called a real difference only when all four rounds put the same arm
/// ahead and the margin clears the null: a ratio that clears the drift in one
/// round is what a 50 µs layer on a 6P+8E part produces at random.
///
/// FLOPs are 2·N·OC·OL·(IC/G)·K for Conv1D. The int8 table counts MACs the same
/// way and print GOPS, with a u8 activation at scale 1/128 and an s8 weight with
/// per-channel scales — the deployment shape. No bias in the int8 rows, so they
/// price the accumulation and the dequant, not the epilogue add.
///
/// `NNOPS_BENCH_THREADS=1` (or "1,4") pins the ladder, as in the other benches.

#include "nnops/ops/conv1d.hpp"
#include "nnops/ops/conv2d.hpp"
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

/// One interleaved A/B measurement at @p nt threads, in per-call ms.
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

    auto run_a = [&]() { a(use_ctx); };
    auto run_b = [&]() { b(use_ctx); };
    run_a(); run_b(); run_a(); run_b();   // warm-up / page faults

    std::vector<double> a_ms, b_ms, null_pct;
    int b_ahead = 0, a_ahead = 0;
    for (int r = 0; r < rounds; ++r) {
        // Alternate which arm opens the round. A small layer pays the pool's
        // wake-up on the first call of a region, so a fixed order would hand
        // that cost to A on every round and call it a kernel difference.
        double a1, b1, a2, b2;
        if (r % 2 == 0) {
            a1 = time_ms(run_a, iters);
            b1 = time_ms(run_b, iters);
            a2 = time_ms(run_a, iters);   // null control
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

/// Enough iterations for ~a second of the smallest layers, few for the largest:
/// 1e9 FLOPs per region, clamped so even a 2.5 GFLOP layer is timed 3 times.
int auto_iters(double flops_per_call) {
    return static_cast<int>(std::clamp(1e9 / std::max(flops_per_call, 1.0),
                                       3.0, 100.0));
}

// ============================================================
// Data
// ============================================================

template <class T>
constexpr DataType dtype_of() {
    return std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
}

template <class T>
std::vector<T> random_vec(std::span<const int64_t> shape, uint64_t seed,
                          float amp = 1.0f) {
    int64_t numel = 1;
    for (int64_t d : shape) { numel *= d; }
    std::vector<T> v(static_cast<size_t>(numel));
    test::XorShift128 rng(seed);
    for (int64_t i = 0; i < numel; ++i) {
        const float f = rng.next_float(-amp, amp);
        if constexpr (std::is_same_v<T, float>) {
            v[static_cast<size_t>(i)] = f;
        } else {
            simd::s_store(&v[static_cast<size_t>(i)], f);
        }
    }
    return v;
}

/// float -> s8/u8 storage value, with the arithmetic the kernels use.
int32_t quant_i8(float x, float inv_scale, int32_t zp, bool u8) {
    const float v = std::fmaf(x, inv_scale, static_cast<float>(zp));
    const int32_t q = static_cast<int32_t>(std::nearbyintf(v));
    return std::min(std::max(q, u8 ? 0 : -128), u8 ? 255 : 127);
}

QuantParams q_per_tensor(float scale, int32_t zp) {
    QuantParams qp;
    qp.granularity = QuantGranularity::PerTensor;
    qp.scale = scale;
    qp.zero_point = zp;
    return qp;
}

QuantParams q_per_channel(const float* scale, const int32_t* zp, int64_t n) {
    QuantParams qp;
    qp.granularity = QuantGranularity::PerChannel;
    qp.scale_data = scale;
    qp.zero_point_data = zp;
    qp.num_scales = n;
    return qp;
}

// ============================================================
// Conv1D — the layer list
// ============================================================

struct ConvShape {
    const char* name;
    int64_t N, IC, L, OC, K;
    int64_t S = 1, P = 0, D = 1, G = 1;
    bool cl = false;   ///< channel-last activation + HWIO filter
};

/// Audio front ends and sequence backbones, then the awkward geometries. The
/// `cl` variants are the same layer in the other layout — a different 2D route,
/// held to the same rule.
const std::vector<ConvShape>& conv_sweep() {
    static const std::vector<ConvShape> s = {
        // ---- wav2vec2's feature extractor: five strided convs on raw audio -----
        {"cf wav2vec fe c1  1x1x16000  -> 512 k10 s5", 1,   1, 16000, 512, 10, 5, 4},
        {"cf wav2vec fe c2  1x512x3199 -> 512 k3 s2",  1, 512,  3199, 512,  3, 2, 1},
        {"cf wav2vec fe c3  1x512x1600 -> 512 k3 s2",  1, 512,  1600, 512,  3, 2, 1},
        // ---- squeeze-excite / conformer blocks ---------------------------------
        {"cf se conv 1x64x400 k3 p1",                   1,  64,   400,  64,  3, 1, 1},
        {"cf se pointwise 1x64x400 k1",                 1,  64,   400,  64,  1},
        {"cf tcn dilated 1x128x1000 k3 d4 p4",          1, 128,  1000, 128,  3, 1, 4, 4},
        // ---- the awkward cases --------------------------------------------------
        {"cf depthwise 1x256x4000 k5 g256",             1, 256,  4000, 256,  5, 1, 2, 1, 256},
        {"cf grouped 1x256x1000 k3 g4",                 1, 256,  1000, 256,  3, 1, 1, 1, 4},
        {"cf big k 1x32x2000 k15",                      1,  32,  2000,  32, 15, 1, 7},
        {"cf batch 8x64x500 k3 p1",                     8,  64,   500,  64,  3, 1, 1},
        // ---- the same layers channel-last --------------------------------------
        {"cl wav2vec fe c2  1x512x3199 -> 512 k3 s2",  1, 512,  3199, 512,  3, 2, 1, 1,
         true},
        {"cl se conv 1x64x400 k3 p1",                   1,  64,   400,  64,  3, 1, 1, 1,
         true},
        {"cl tcn dilated 1x128x1000 k3 d4 p4",          1, 128,  1000, 128,  3, 1, 4, 4,
         true},
        {"cl grouped 1x256x1000 k3 g4",                 1, 256,  1000, 256,  3, 1, 1, 1, 4,
         true},
        {"cl depthwise 1x256x4000 k5 g256",             1, 256,  4000, 256,  5, 1, 2, 1, 256,
         true},
        {"cl big k 1x32x2000 k15",                      1,  32,  2000,  32, 15, 1, 7, 1,
         true},
    };
    return s;
}

// ============================================================
// Conv1D — the operator and the 2D call on the same lift
// ============================================================

/// One layer built twice over one tensor image: the 1D op (which may split L)
/// and Conv2D on the identical lift (which cannot). Both write the same output
/// buffer, so a row differs only in the code path, never in the data.
template <class T>
struct Conv1DPair {
    explicit Conv1DPair(const ConvShape& c) : s(c) {
        const int64_t icg = c.IC / c.G;
        const int64_t ish[] = {c.N, c.IC, c.L};
        constexpr DataType dt = dtype_of<T>();

        in_buf = random_vec<T>(std::span<const int64_t>(ish, 3), 1);
        if (c.cl) {
            in_layout = TensorLayout::NHWC;
            in_buf = test::to_channel_last<T>(
                std::span<const T>(in_buf.data(), in_buf.size()), c.N, c.IC, c.L);
        }

        // The filter is written once and stored as OIHW or HWIO: the same
        // numbers, in the order the layout wants.
        const std::vector<T> w_oihw = random_vec<T>(
            std::span<const int64_t>(std::array<int64_t, 3>{c.OC, icg, c.K}.data(), 3), 2);
        if (c.cl) {
            w_buf = test::oihw_to_hwio<T>(w_oihw, c.OC, icg, 1, c.K);
            w_dims = {c.K, icg, c.OC};
            w_layout = TensorLayout::HWIO;
        } else {
            w_buf = w_oihw;
            w_dims = {c.OC, icg, c.K};
            w_layout = TensorLayout::NCHW;
        }
        b_buf = random_vec<T>(std::span<const int64_t>(&c.OC, 1), 3, 0.05f);

        in = TensorView(std::span<const int64_t>(ish, 3), dt, in_buf.data(),
                        in_layout);
        w  = TensorView(std::span<const int64_t>(w_dims.data(), 3), dt,
                        w_buf.data(), w_layout);
        b  = TensorView(std::span<const int64_t>(&c.OC, 1), dt, b_buf.data(),
                        TensorLayout::NCHW);

        Conv1DAttributes a1{};
        a1.kernel_size = c.K;
        a1.stride      = c.S;
        a1.dilation    = c.D;
        a1.padding     = c.P;
        a1.groups      = c.G;
        op1d = Conv1D::create(a1, Backend::CPU);

        const TensorDesc d1[] = {in.desc(), w.desc(), b.desc()};
        const auto od = op1d->getOutputTensorDesc(d1);
        OL = od[0].dims[2];
        out_buf.assign(static_cast<size_t>(od[0].numel()), T{});
        out = TensorView(od[0].dims, dt, out_buf.data(),
                         c.cl ? TensorLayout::NHWC : TensorLayout::NCHW);
        ins1d = {in, w, b};

        // The control arm: the operator's own lift, handed straight to Conv2D.
        TensorView in4 = cpu::lift_wide(in);
        TensorView w4  = cpu::lift_filter_wide(w);
        out4 = cpu::lift_wide(out);

        Conv2DAttributes a2{};
        a2.kernel_size = {1, c.K};
        a2.stride      = {1, c.S};
        a2.dilation    = {1, c.D};
        a2.padding     = {0, c.P};
        a2.groups      = c.G;
        // As Conv1D does: a square Winograd tile has no 1-wide analogue.
        a2.algorithm   = ConvAlgorithm::ImplicitGemm;
        op2d = Conv2D::create(a2, Backend::CPU);
        ins2d = {in4, w4, b};

        work = 2.0 * static_cast<double>(c.N) * c.OC * OL *
               static_cast<double>(icg) * c.K;
    }

    Status run_1d(const ComputeContext& ctx) { return op1d->compute(out, ins1d, ctx, nullptr); }
    Status run_2d(const ComputeContext& ctx) { return op2d->compute(out4, ins2d, ctx, nullptr); }

    ConvShape s;
    TensorLayout in_layout = TensorLayout::NCHW;
    TensorLayout w_layout = TensorLayout::NCHW;
    std::array<int64_t, 3> w_dims{};
    std::vector<T> in_buf, w_buf, b_buf, out_buf;
    TensorView in, w, b, out, out4;
    std::vector<TensorView> ins1d, ins2d;
    std::unique_ptr<Conv1D> op1d;
    std::unique_ptr<Conv2D> op2d;
    int64_t OL = 0;
    double work = 0.0;
};

/// The quantized counterpart: u8 activation (per tensor), s8 weight (per
/// channel), f32 destination — the deployment shape, in either layout.
struct I8Conv1DPair {
    explicit I8Conv1DPair(const ConvShape& c) : s(c) {
        const int64_t icg = c.IC / c.G;
        const int64_t in_n = c.N * c.IC * c.L;
        const int64_t w_n  = c.OC * icg * c.K;
        const float s_x = 0.0078125f;   // 1/128 — exact, and the fp range fits
        const int32_t zp_x = 128;

        in_buf.assign(static_cast<size_t>(in_n), 0);
        {
            test::XorShift128 rng(1);
            for (int64_t i = 0; i < in_n; ++i) {
                in_buf[static_cast<size_t>(i)] = static_cast<uint8_t>(
                    quant_i8(rng.next_float(-1.0f, 1.0f), 1.0f / s_x, zp_x, true));
            }
        }
        std::vector<int8_t> w_oihw(static_cast<size_t>(w_n));
        w_s.assign(static_cast<size_t>(c.OC), 0.0f);
        w_zp.assign(static_cast<size_t>(c.OC), 0);
        {
            test::XorShift128 rng(2);
            for (int64_t i = 0; i < w_n; ++i) {
                w_oihw[static_cast<size_t>(i)] = static_cast<int8_t>(
                    quant_i8(rng.next_float(-1.0f, 1.0f), 127.0f, 0, false));
            }
            for (int64_t oc = 0; oc < c.OC; ++oc) {
                w_s[static_cast<size_t>(oc)] = 0.01f * static_cast<float>(1 + oc % 5);
            }
        }
        if (c.cl) {
            in_buf = test::to_channel_last<uint8_t>(
                std::span<const uint8_t>(in_buf.data(), in_buf.size()), c.N, c.IC, c.L);
            w_buf = test::oihw_to_hwio<int8_t>(w_oihw, c.OC, icg, 1, c.K);
            w_dims = {c.K, icg, c.OC};
            w_layout = TensorLayout::HWIO;
        } else {
            w_buf = w_oihw;
            w_dims = {c.OC, icg, c.K};
            w_layout = TensorLayout::NCHW;
        }
        b_buf = random_vec<float>(std::span<const int64_t>(&c.OC, 1), 3, 0.05f);

        const int64_t ish[] = {c.N, c.IC, c.L};
        const QuantParams qa = q_per_tensor(s_x, zp_x);
        const QuantParams qw = q_per_channel(w_s.data(), w_zp.data(), c.OC);

        in = TensorView(std::span<const int64_t>(ish, 3), DataType::u8,
                        in_buf.data(), c.cl ? TensorLayout::NHWC : TensorLayout::NCHW,
                        qa);
        w  = TensorView(std::span<const int64_t>(w_dims.data(), 3), DataType::s8,
                        w_buf.data(), w_layout, qw);
        b  = TensorView(std::span<const int64_t>(&c.OC, 1), DataType::f32,
                        b_buf.data(), TensorLayout::NCHW);

        Conv1DAttributes a1{};
        a1.kernel_size  = c.K;
        a1.stride       = c.S;
        a1.dilation     = c.D;
        a1.padding      = c.P;
        a1.groups       = c.G;
        a1.output_dtype = DataType::f32;
        op1d = Conv1D::create(a1, Backend::CPU);

        const TensorDesc d1[] = {in.desc(), w.desc(), b.desc()};
        const auto od = op1d->getOutputTensorDesc(d1);
        OL = od[0].dims[2];
        out_buf.assign(static_cast<size_t>(od[0].numel()), 0.0f);
        out = TensorView(od[0].dims, DataType::f32, out_buf.data(),
                         c.cl ? TensorLayout::NHWC : TensorLayout::NCHW);
        ins1d = {in, w, b};

        TensorView in4 = cpu::lift_wide(in);
        TensorView w4  = cpu::lift_filter_wide(w);
        out4 = cpu::lift_wide(out);

        Conv2DAttributes a2{};
        a2.kernel_size = {1, c.K};
        a2.stride      = {1, c.S};
        a2.dilation    = {1, c.D};
        a2.padding     = {0, c.P};
        a2.groups      = c.G;
        a2.algorithm   = ConvAlgorithm::ImplicitGemm;
        op2d = Conv2D::create(a2, Backend::CPU);
        ins2d = {in4, w4, b};

        work = 2.0 * static_cast<double>(c.N) * c.OC * OL *
               static_cast<double>(icg) * c.K;
    }

    Status run_1d(const ComputeContext& ctx) { return op1d->compute(out, ins1d, ctx, nullptr); }
    Status run_2d(const ComputeContext& ctx) { return op2d->compute(out4, ins2d, ctx, nullptr); }

    ConvShape s;
    TensorLayout w_layout = TensorLayout::NCHW;
    std::array<int64_t, 3> w_dims{};
    std::vector<uint8_t> in_buf;
    std::vector<int8_t> w_buf;
    std::vector<float> w_s, b_buf, out_buf;
    std::vector<int32_t> w_zp;
    TensorView in, w, b, out, out4;
    std::vector<TensorView> ins1d, ins2d;
    std::unique_ptr<Conv1D> op1d;
    std::unique_ptr<Conv2D> op2d;
    int64_t OL = 0;
    double work = 0.0;
};

// ============================================================
// Tables
// ============================================================

struct Row {
    const char* name;
    int64_t L_in = 0, L_out = 0;
    double work = 0.0;
};

template <class Pair>
void run_pair_table(const char* title, const char* la, const char* lb,
                    std::vector<Pair>& layers,
                    Status (Pair::*run_a)(const ComputeContext&),
                    Status (Pair::*run_b)(const ComputeContext&)) {
    const std::vector<int> threads = bench_threads();
    const int rounds = 4;   // even, so the alternating order balances
    std::printf("  %s\n", title);
    std::printf("    %-42s %7s %7s | %9s %8s | %9s %8s | %s/%s %6s %s\n",
                "case", "L_in", "L_out", la, "rate", lb, "rate", lb, la, "null",
                "pool");

    double sum_a = 0.0, sum_b = 0.0, sum_work = 0.0;
    for (Pair& l : layers) {
        const Row r{l.s.name, l.s.L, l.OL, l.work};
        const Status sa = (l.*run_a)(ComputeContext{});
        const Status sb = (l.*run_b)(ComputeContext{});
        if (!is_ok(sa) || !is_ok(sb)) {
            std::printf("    %-42s %7lld %7lld | %s%s%s\n", r.name,
                        (long long)r.L_in, (long long)r.L_out,
                        is_ok(sa) ? "" : "1d unsupported ",
                        is_ok(sb) ? "" : "2d unsupported ",
                        (!is_ok(sa) && !is_ok(sb)) ? "(both reject the geometry)" : "");
            continue;
        }
        const int iters = auto_iters(l.work);
        for (size_t i = 0; i < threads.size(); ++i) {
            const AbResult m = measure_ab(threads[i],
                [&](const ComputeContext& x) { return (l.*run_a)(x); },
                [&](const ComputeContext& x) { return (l.*run_b)(x); },
                iters, rounds);
            // B/A: above 1.00 the 2D call took longer, i.e. the 1D route won.
            const bool real = m.consistent &&
                std::abs(m.a_ms - m.b_ms) / m.a_ms * 100.0 > m.null_pct;
            std::printf("    %-42s %7lld %7lld | %9.3f %8.1f | %9.3f %8.1f | %5.2fx %5.1f%% %3dT%s\n",
                        i == 0 ? r.name : "", (long long)r.L_in, (long long)r.L_out,
                        m.a_ms, r.work / m.a_ms / 1e6,
                        m.b_ms, r.work / m.b_ms / 1e6,
                        m.b_ms / m.a_ms, m.null_pct, threads[i],
                        real ? "" : " (noise)");
            if (i == 0) {
                sum_a += m.a_ms;
                sum_b += m.b_ms;
                sum_work += l.work;
            }
        }
    }
    std::printf("    %-42s %7s %7s | %9.3f %8.1f | %9.3f %8.1f | %5.2fx\n",
                "sum of the layers (1T)", "", "", sum_a, sum_work / sum_a / 1e6,
                sum_b, sum_work / sum_b / 1e6, sum_b / sum_a);
}

template <class T>
std::vector<Conv1DPair<T>> conv_layers() {
    std::vector<Conv1DPair<T>> v;
    for (const ConvShape& c : conv_sweep()) { v.emplace_back(c); }
    return v;
}

}  // namespace

// ============================================================
// Benches
// ============================================================

NNOPS_BENCH(conv1d_f32_vs_conv2d_lift) {
    auto layers = conv_layers<float>();
    run_pair_table("conv1d f32 — Conv1D against Conv2D on the same lift (2D/1D = 1.00x when the adapter is free)",
                   "1d", "2d", layers,
                   &Conv1DPair<float>::run_1d, &Conv1DPair<float>::run_2d);
}

NNOPS_BENCH(conv1d_f16_vs_conv2d_lift) {
    auto layers = conv_layers<half>();
    run_pair_table("conv1d f16 — Conv1D against Conv2D on the same lift",
                   "1d", "2d", layers,
                   &Conv1DPair<half>::run_1d, &Conv1DPair<half>::run_2d);
}

NNOPS_BENCH(conv1d_int8_vs_conv2d_lift) {
    std::vector<I8Conv1DPair> layers;
    for (const ConvShape& c : conv_sweep()) { layers.emplace_back(c); }
    run_pair_table("conv1d int8 — u8 act / s8 weight, dequant to f32; same lift",
                   "1d", "2d", layers,
                   &I8Conv1DPair::run_1d, &I8Conv1DPair::run_2d);
}
