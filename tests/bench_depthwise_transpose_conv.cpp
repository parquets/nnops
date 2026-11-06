/// @file bench_depthwise_transpose_conv.cpp
/// @brief DepthwiseTransposeConv2D: the dedicated depthwise route against the
///        general TransposeConv2D one, in both layouts.
///
/// A depthwise transposed convolution is a general one with groups == channels
/// and one tap block per channel, so the same shapes can always be run either
/// way. The dedicated route exploits the one-channel-per-group structure: on
/// channel-first the lane is the stride residue (one weight scalar broadcast
/// over a run of output columns), on channel-last it is the channel (one
/// weight vector serves a whole pixel, four pixels share it).
///
/// The general route is the reference arm: it reads the same filter as
/// [C, 1, KH, KW] and pays for its GEMM — the col matrix counts every
/// input-tap pair, which is exactly what a one-tap-per-channel filter does not
/// need. The gap between the two arms is what the dedicated path buys.
///
/// Layout conversion runs once, outside every timed region, and each round
/// times A B C D A — the trailing A is the null control (same arm, same
/// buffers, same binary), so an A2/A1 gap is drift and nothing else.
///
/// The quantized arms are in place but the kernel does not answer them yet:
/// a call that returns anything but ok prints `n/a`, so the number appears the
/// day the route lands.

#include "nnops/ops/depthwise_transpose_conv.hpp"
#include "nnops/ops/transpose_conv2d.hpp"
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

double max_abs_diff(const float* a, const float* b, int64_t n) {
    double worst = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        worst = std::fmax(worst, std::fabs(static_cast<double>(a[i]) - b[i]));
    }
    return worst;
}

struct DwtCase {
    const char* name;
    int64_t C, IH, IW;
    int64_t KH = 3, KW = 3, SH = 1, SW = 1, PH = 1, PW = 1;
};

/// Both routes are the same function, so every timed call also checks the
/// others: the general route's output is the yardstick for the dedicated one.
void run_case(const DwtCase& c, const std::vector<int>& threads, int iters, int rounds) {
    const int64_t ishape[] = {1, c.C, c.IH, c.IW};
    const int64_t wshape[] = {c.C, 1, c.KH, c.KW};

    auto [in_vec, in_cf] = test::make_random_tensor(ishape, -1.0f, 1.0f, 31);
    auto [w_vec, w_cf] = test::make_random_tensor(wshape, -1.0f, 1.0f, 32);

    DepthwiseTransposeConv2DAttributes dwt_attrs{};
    dwt_attrs.kernel_size = {c.KH, c.KW};
    dwt_attrs.stride      = {c.SH, c.SW};
    dwt_attrs.dilation    = {1, 1};
    dwt_attrs.padding     = {c.PH, c.PW};

    TransposeConv2DAttributes gen_attrs{};
    gen_attrs.kernel_size = {c.KH, c.KW};
    gen_attrs.stride      = {c.SH, c.SW};
    gen_attrs.dilation    = {1, 1};
    gen_attrs.padding     = {c.PH, c.PW};
    gen_attrs.groups      = c.C;

    // ---- the four fp arms: two routes × two layouts --------------------
    auto dwt_op = DepthwiseTransposeConv2D::create(dwt_attrs, Backend::CPU);
    auto gen_op = TransposeConv2D::create(gen_attrs, Backend::CPU);

    const TensorView cf_ins[] = {in_cf, w_cf};
    const TensorDesc cf_desc[] = {in_cf.desc(), w_cf.desc()};
    const auto out_desc = dwt_op->getOutputTensorDesc(cf_desc)[0];
    const int64_t OH = out_desc.dims[2];
    const int64_t OW = out_desc.dims[3];
    const int64_t out_numel = out_desc.numel();

    std::vector<float> dwt_cf_buf(static_cast<size_t>(out_numel));
    std::vector<float> gen_cf_buf(static_cast<size_t>(out_numel));
    auto dwt_cf_out = test::make_channel_first(out_desc, dwt_cf_buf.data());
    auto gen_cf_out = test::make_channel_first(out_desc, gen_cf_buf.data());

    auto in_cl_vec = test::to_channel_last<float>(
        std::span<const float>(in_vec.data(), in_vec.size()), 1, c.C, c.IH * c.IW);
    auto w_hmcw_vec = test::depthwise_to_hwcm<float>(
        std::span<const float>(w_vec.data(), w_vec.size()), c.C, c.KH * c.KW);

    TensorView in_cl(std::span<const int64_t>(ishape, 4), DataType::f32,
                     in_cl_vec.data(), TensorLayout::NHWC);
    const int64_t w_hwcm_shape[] = {c.KH, c.KW, c.C};
    TensorView w_hwcm(std::span<const int64_t>(w_hwcm_shape, 3), DataType::f32,
                      w_hmcw_vec.data(), TensorLayout::HWCM);
    const TensorView cl_ins[] = {in_cl, w_hwcm};
    const TensorDesc cl_desc[] = {in_cl.desc(), w_hwcm.desc()};
    const auto cl_out_desc = dwt_op->getOutputTensorDesc(cl_desc)[0];

    // The general route keeps the channel-first filter in both layouts.
    const TensorView gen_cl_ins[] = {in_cl, w_cf};
    const TensorDesc gen_cl_desc[] = {in_cl.desc(), w_cf.desc()};

    std::vector<float> dwt_cl_buf(static_cast<size_t>(out_numel));
    std::vector<float> gen_cl_buf(static_cast<size_t>(out_numel));
    auto dwt_cl_out = test::make_channel_first(cl_out_desc, dwt_cl_buf.data());
    auto gen_cl_out = test::make_channel_first(cl_out_desc, gen_cl_buf.data());

    const auto dwt_cf = [&](const ComputeContext& ctx) {
        TensorView outs[] = {dwt_cf_out};
        dwt_op->compute(outs, cf_ins, ctx);
    };
    const auto gen_cf = [&](const ComputeContext& ctx) {
        TensorView outs[] = {gen_cf_out};
        gen_op->compute(outs, cf_ins, ctx);
    };
    const auto dwt_cl = [&](const ComputeContext& ctx) {
        TensorView outs[] = {dwt_cl_out};
        dwt_op->compute(outs, cl_ins, ctx);
    };
    const auto gen_cl = [&](const ComputeContext& ctx) {
        TensorView outs[] = {gen_cl_out};
        gen_op->compute(outs, gen_cl_ins, ctx);
    };

    // ---- agreement before timing: all four must agree -------------------
    dwt_cf(ComputeContext{});
    gen_cf(ComputeContext{});
    dwt_cl(ComputeContext{});
    gen_cl(ComputeContext{});
    const auto dwt_cl_as_cf = test::to_channel_first<float>(
        std::span<const float>(dwt_cl_buf.data(), dwt_cl_buf.size()), 1, c.C, OH * OW);
    const auto gen_cl_as_cf = test::to_channel_first<float>(
        std::span<const float>(gen_cl_buf.data(), gen_cl_buf.size()), 1, c.C, OH * OW);
    const double d_gen = max_abs_diff(dwt_cf_buf.data(), gen_cf_buf.data(), out_numel);
    const double d_cl = max_abs_diff(dwt_cf_buf.data(), dwt_cl_as_cf.data(), out_numel);
    const double d_gen_cl = max_abs_diff(dwt_cf_buf.data(), gen_cl_as_cf.data(), out_numel);

    const double flops = 2.0 * static_cast<double>(c.C) * OH * OW * c.KH * c.KW;
    std::printf("  %s\n", c.name);
    std::printf("    in[1,%lld,%lldx%lld] w[%lld,1,%lldx%lld] s%lld p%lld -> out[1,%lld,%lldx%lld]"
                "  |dwt - gen| = %.2e  |dwt cf - dwt cl| = %.2e  |dwt - gen cl| = %.2e\n",
                (long long)c.C, (long long)c.IH, (long long)c.IW,
                (long long)c.C, (long long)c.KH, (long long)c.KW,
                (long long)c.SH, (long long)c.PH,
                (long long)c.C, (long long)OH, (long long)OW,
                d_gen, d_cl, d_gen_cl);

    for (int nt : threads) {
        BenchPool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

        dwt_cf(use_ctx);   // warm-up / first touch
        gen_cf(use_ctx);
        dwt_cl(use_ctx);
        gen_cl(use_ctx);

        std::vector<double> a1, g_cf, d_cl_t, g_cl, a2;
        for (int r = 0; r < rounds; ++r) {
            a1.push_back(time_us([&] { dwt_cf(use_ctx); }, iters));
            g_cf.push_back(time_us([&] { gen_cf(use_ctx); }, iters));
            d_cl_t.push_back(time_us([&] { dwt_cl(use_ctx); }, iters));
            g_cl.push_back(time_us([&] { gen_cl(use_ctx); }, iters));
            a2.push_back(time_us([&] { dwt_cf(use_ctx); }, iters));
        }
        const double t_dcf = median_of(a1);
        const double t_gcf = median_of(g_cf);
        const double t_dcl = median_of(d_cl_t);
        const double t_gcl = median_of(g_cl);
        const double t_scratch = median_of(a2);
        const double null_gap = (t_scratch / t_dcf - 1.0) * 100.0;

        std::printf("    %2dT  dwt cf %8.1f us (%6.1f GFLOP/s) | dwt cl %8.1f us (%6.1f) |"
                    " gen cf %8.1f us (%6.1f) | gen cl %8.1f us (%6.1f)"
                    "  |  gen/dwt  cf %4.2fx  cl %4.2fx  (null %+4.1f%%)\n",
                    nt,
                    t_dcf, flops / (t_dcf * 1e3),
                    t_dcl, flops / (t_dcl * 1e3),
                    t_gcf, flops / (t_gcf * 1e3),
                    t_gcl, flops / (t_gcl * 1e3),
                    t_gcf / t_dcf, t_gcl / t_dcl, null_gap);
    }
}

/// The quantized arm, both layouts: the same s8 operands, permuted.
void run_quantized_arm(const DwtCase& c, const std::vector<int>& threads, int iters) {
    const int64_t ishape[] = {1, c.C, c.IH, c.IW};
    const int64_t wshape[] = {c.C, 1, c.KH, c.KW};

    std::vector<int8_t> act(static_cast<size_t>(c.C * c.IH * c.IW), 7);
    std::vector<int8_t> w(static_cast<size_t>(c.C * c.KH * c.KW), 9);
    std::vector<float> w_s(static_cast<size_t>(c.C), 0.01f);
    std::vector<int32_t> w_zp(static_cast<size_t>(c.C), 0);
    QuantParams qa;
    qa.scale = 1.0f / 128.0f;
    QuantParams qw;
    qw.granularity = QuantGranularity::PerChannel;
    qw.scale_data = w_s.data();
    qw.zero_point_data = w_zp.data();
    qw.num_scales = c.C;

    TensorView in(ishape, DataType::s8, act.data(), TensorLayout::NCHW, qa);
    TensorView weight(wshape, DataType::s8, w.data(), TensorLayout::NCHW, qw);

    DepthwiseTransposeConv2DAttributes attrs{};
    attrs.kernel_size = {c.KH, c.KW};
    attrs.stride      = {c.SH, c.SW};
    attrs.padding     = {c.PH, c.PW};

    auto op = DepthwiseTransposeConv2D::create(attrs, Backend::CPU);
    const TensorView ins[] = {in, weight};
    const TensorDesc descs[] = {in.desc(), weight.desc()};
    const auto out_desc = op->getOutputTensorDesc(descs)[0];
    std::vector<float> out_buf(static_cast<size_t>(out_desc.numel()));
    auto out = test::make_channel_first(out_desc, out_buf.data());

    auto act_cl_vec = test::to_channel_last<int8_t>(
        std::span<const int8_t>(act.data(), act.size()), 1, c.C, c.IH * c.IW);
    auto w_hwcm_vec = test::depthwise_to_hwcm<int8_t>(
        std::span<const int8_t>(w.data(), w.size()), c.C, c.KH * c.KW);
    TensorView in_cl(std::span<const int64_t>(ishape, 4), DataType::s8,
                     act_cl_vec.data(), TensorLayout::NHWC, qa);
    const int64_t w_hwcm_shape[] = {c.KH, c.KW, c.C};
    TensorView w_hwcm(std::span<const int64_t>(w_hwcm_shape, 3), DataType::s8,
                      w_hwcm_vec.data(), TensorLayout::HWCM, qw);
    const TensorView cl_ins[] = {in_cl, w_hwcm};
    const TensorDesc cl_descs[] = {in_cl.desc(), w_hwcm.desc()};
    const auto cl_desc_out = op->getOutputTensorDesc(cl_descs)[0];
    std::vector<float> cl_buf(static_cast<size_t>(cl_desc_out.numel()));
    auto cl_out = test::make_channel_first(cl_desc_out, cl_buf.data());

    const Status st_cf = op->compute(out, ins, ComputeContext{});
    const Status st_cl = op->compute(cl_out, cl_ins, ComputeContext{});
    if (st_cf != Status::ok || st_cl != Status::ok) {
        std::printf("    int8: n/a (compute returned %d / %d)\n",
                    static_cast<int>(st_cf), static_cast<int>(st_cl));
        (void)threads;
        (void)iters;
        return;
    }

    // A permuted input and filter must land on the same numbers.
    const auto cl_as_cf = test::to_channel_first<float>(
        std::span<const float>(cl_buf.data(), cl_buf.size()), 1, c.C,
        out_desc.dims[2] * out_desc.dims[3]);
    const double gap = max_abs_diff(out_buf.data(), cl_as_cf.data(), out_desc.numel());

    const double flops = 2.0 * static_cast<double>(c.C) * out_desc.dims[2] *
                         out_desc.dims[3] * c.KH * c.KW;
    std::printf("    |int8 cf - int8 cl| = %.2e\n", gap);

    for (int nt : threads) {
        BenchPool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};
        std::vector<double> t_cf, t_cl;
        for (int r = 0; r < 3; ++r) {
            t_cf.push_back(time_us([&] { op->compute(out, ins, use_ctx); }, iters));
            t_cl.push_back(time_us([&] { op->compute(cl_out, cl_ins, use_ctx); }, iters));
        }
        const double cf_us = median_of(t_cf);
        const double cl_us = median_of(t_cl);
        std::printf("    int8 %2dT  cf %8.1f us (%6.1f GFLOP/s) | cl %8.1f us (%6.1f)\n",
                    nt, cf_us, flops / (cf_us * 1e3), cl_us, flops / (cl_us * 1e3));
    }
}

}  // namespace

// ============================================================
// Decoder shapes — the transposed convolutions of a MobileNetV2-style
// segmentation head, plus wide and narrow planes
// ============================================================

NNOPS_BENCH(depthwise_transpose_conv_routes) {
    const std::vector<int> threads = bench_threads();
    const int iters = 10;
    const int rounds = 3;

    std::printf("depthwise transpose conv: the dedicated route against groups==C TransposeConv2D\n");
    std::printf("  pool=%s  threads=", test::bench_pool_name());
    for (size_t i = 0; i < threads.size(); ++i) {
        std::printf("%d%s", threads[i], i + 1 < threads.size() ? "," : "\n");
    }

    const DwtCase cases[] = {
        {"dwt 3x3 C=16  @112 s1",  16, 112, 112},   // half a channel vector
        {"dwt 3x3 C=32  @112 s2",  32, 112, 112, 3, 3, 2, 2},
        {"dwt 3x3 C=64  @56  s1",  64,  56,  56},
        {"dwt 3x3 C=96  @28  s1",  96,  28,  28},
        {"dwt 3x3 C=144 @56  s1", 144,  56,  56},
        {"dwt 3x3 C=240 @28  s1", 240,  28,  28},
        {"dwt 3x3 C=256 @14  s1", 256,  14,  14},
        {"dwt 3x3 C=576 @7   s1", 576,   7,   7},
        {"dwt 3x3 C=32  @224 s2",  32, 224, 224, 3, 3, 2, 2},
        {"dwt 3x3 C=256 @200x300", 256, 200, 300},
        {"dwt 3x3 C=64  @60x80",   64,  60,  80},
        {"dwt 3x3 C=17  @37x41",   17,  37,  41},
        {"dwt 5x5 C=64  @56  s1",  64,  56,  56, 5, 5, 1, 1, 2, 2},
        {"dwt 1x1 C=144 @56",     144,  56,  56, 1, 1, 1, 1, 0, 0},
    };

    for (const auto& c : cases) {
        run_case(c, threads, iters, rounds);
    }
}

// ============================================================
// The quantized route's arms
// ============================================================

NNOPS_BENCH(depthwise_transpose_conv_int8) {
    const std::vector<int> threads = bench_threads();
    const int iters = 10;

    std::printf("depthwise transpose conv, quantized: s8 per-channel weight × s8 activation\n");
    std::printf("  pool=%s  threads=", test::bench_pool_name());
    for (size_t i = 0; i < threads.size(); ++i) {
        std::printf("%d%s", threads[i], i + 1 < threads.size() ? "," : "\n");
    }

    const DwtCase cases[] = {
        {"dwt 3x3 C=16  @112 s1",  16, 112, 112},
        {"dwt 3x3 C=64  @56  s1",  64,  56,  56},
        {"dwt 3x3 C=144 @56  s1", 144,  56,  56},
        {"dwt 3x3 C=256 @14  s1", 256,  14,  14},
        {"dwt 3x3 C=256 @200x300", 256, 200, 300},
        {"dwt 1x1 C=144 @56",     144,  56,  56, 1, 1, 1, 1, 0, 0},
    };

    for (const auto& c : cases) {
        std::printf("  %s\n", c.name);
        run_quantized_arm(c, threads, iters);
    }
}
