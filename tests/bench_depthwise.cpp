/// @file bench_depthwise.cpp
/// @brief Depthwise conv: packed (NCHWC8) against planar (NCHW), interleaved.
///
/// These are two kernels, not two layouts of one kernel, and they are built on
/// opposite assumptions:
///
///   packed  the SIMD lane is a channel, so one load feeds 8 channels at the
///           same output position; four output rows are unrolled (h4) and each
///           tap broadcasts one weight vector into four independent accumulates.
///   planar  the SIMD lane is output width, so one load feeds 8 positions of
///           ONE row; there is one accumulate per row and the taps chain
///           through it (9 dependent FMAs for a 3x3).
///
/// Shape decides which wins: packed's width wastes lanes on a narrow plane
/// (7x7 gives 7 useful lanes per row, and 14 per four-row group), while planar's
/// dependent chain is immune to width but cannot hide FMA latency.
///
/// Layout conversion and weight prepack are done once, outside every timed
/// region: an engine converts a tensor once and reuses it, so billing the bench's
/// convert per call would price the harness, not the kernel.
///
/// Each round times A B A — the second A is the null control (same arm, same
/// buffers, same binary), so an A2/A1 gap is drift and nothing else. A verdict
/// is only read as real when it clears that floor.

#include "nnops/ops/depthwise_conv.hpp"
#include "backend/cpu/simd_kernel/simd_depthwise_conv.hpp"
#include "nnops/ops/layout_convert.hpp"
#include "common/bench_harness.hpp"
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
namespace k = nnops::kernel;

namespace {

// Same two schedulers as the other benches: OpenMP when the build has it (and
// NNOPS_BENCH_OMP does not opt out), else one thread per call.
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

struct DepthwiseCase {
    const char* name;
    int64_t C, IH, IW;
    int64_t KH = 3, KW = 3, SH = 1, SW = 1, PH = 1, PW = 1;
    /// The model's depthwise stages ride a relu6 epilogue; the plain cases do
    /// not. Kept on the case so the two can be compared in one run.
    bool relu6 = false;
};

void run_case(const DepthwiseCase& c, const std::vector<int>& threads, int iters,
              int rounds) {
    const int64_t ishape[] = {1, c.C, c.IH, c.IW};
    const int64_t wshape[] = {c.C, 1, c.KH, c.KW};
    const int64_t bshape[] = {c.C};

    auto [in_vec, in_nchw] = test::make_random_tensor(ishape, -1.0f, 1.0f, 31);
    auto [w_vec, w_nchw]   = test::make_random_tensor(wshape, -1.0f, 1.0f, 32);
    auto [b_vec, b_nchw]   = test::make_random_tensor(bshape, -0.5f, 0.5f, 33);

    DepthwiseConvAttributes attrs{};
    attrs.kernel_size = {1, c.KH, c.KW};
    attrs.stride      = {1, c.SH, c.SW};
    attrs.dilation    = {1, 1, 1};
    attrs.padding     = {0, c.PH, c.PW};
    if (c.relu6) { attrs.epilogue.type = nnops::EpilogueActivateType::Relu6; }

    // ---- planar arm: the tensors as they come out of a PyTorch model ----
    auto planar_op = DepthwiseConv::create(attrs, Backend::CPU);
    const TensorView planar_in[] = {in_nchw, w_nchw, b_nchw};
    const TensorDesc planar_in_desc[] = {in_nchw.desc(), w_nchw.desc(), b_nchw.desc()};
    auto planar_out_desc = planar_op->getOutputTensorDesc(planar_in_desc)[0];
    const int64_t OH = planar_out_desc.dims[2];
    const int64_t OW = planar_out_desc.dims[3];
    std::vector<float> planar_out_buf(static_cast<size_t>(planar_out_desc.numel()));
    auto planar_out = test::make_planar(planar_out_desc, planar_out_buf.data());
    const auto planar_call = [&](const ComputeContext& ctx) {
        TensorView outs[] = {planar_out};
        planar_op->compute(outs, planar_in, ctx);
    };

    // ---- packed arm: convert in, prepack the weight, allocate a packed out ----
    auto to_c8 = LayoutConvert::create(TensorLayout::NCHWC8, Backend::CPU);
    const TensorDesc in_desc = in_nchw.desc();
    auto in_c8_desc = to_c8->getOutputTensorDesc({&in_desc, 1})[0];
    std::vector<float> in_c8_buf(static_cast<size_t>(in_c8_desc.storage_bytes() / sizeof(float)));
    auto in_c8 = test::make_packed(in_c8_desc, in_c8_buf.data());
    {
        const TensorView i[] = {in_nchw};
        TensorView o[] = {in_c8};
        to_c8->compute(o, i);
    }

    auto packed_op = DepthwiseConv::create(attrs, Backend::CPU);
    TensorView pw_query;
    {
        const TensorView wq[] = {w_nchw};
        TensorView pq[] = {pw_query};
        packed_op->prepackWeights(wq, pq);
        pw_query = pq[0];
    }
    std::vector<float> pw_buf(static_cast<size_t>(pw_query.numel()));
    auto pw_view = test::make_planar(pw_query.desc(), pw_buf.data());
    {
        const TensorView wp[] = {w_nchw};
        TensorView po[] = {pw_view};
        packed_op->prepackWeights(wp, po);
    }

    const TensorView packed_in[] = {in_c8, pw_view, b_nchw};
    const TensorDesc packed_in_desc[] = {in_c8.desc(), pw_view.desc(), b_nchw.desc()};
    auto packed_out_desc = packed_op->getOutputTensorDesc(packed_in_desc)[0];
    std::vector<float> packed_out_buf(
        static_cast<size_t>(packed_out_desc.storage_bytes() / sizeof(float)));
    auto packed_out = test::make_packed(packed_out_desc, packed_out_buf.data());
    const auto packed_call = [&](const ComputeContext& ctx) {
        TensorView outs[] = {packed_out};
        packed_op->compute(outs, packed_in, ctx);
    };

    // ---- agreement: the two kernels must be computing the same function ----
    planar_call(ComputeContext{});
    packed_call(ComputeContext{});
    std::vector<float> packed_as_planar(static_cast<size_t>(planar_out_desc.numel()));
    auto packed_nchw = test::make_planar(planar_out_desc, packed_as_planar.data());
    {
        auto to_nchw = LayoutConvert::create(TensorLayout::NCHW, Backend::CPU);
        const TensorView i[] = {packed_out};
        TensorView o[] = {packed_nchw};
        to_nchw->compute(o, i);
    }
    const double d = max_abs_diff(planar_out_buf.data(), packed_as_planar.data(),
                                  planar_out_desc.numel());

    const double flops = 2.0 * static_cast<double>(c.C) * OH * OW * c.KH * c.KW;
    std::printf("  %s\n", c.name);
    std::printf("    in[1,%lld,%lld,%lld] w[%lld,1,%lldx%lld] s%lld p%lld -> out[1,%lld,%lld,%lld]"
                "  |planar - packed| = %.2e\n",
                (long long)c.C, (long long)c.IH, (long long)c.IW,
                (long long)c.C, (long long)c.KH, (long long)c.KW,
                (long long)c.SH, (long long)c.PH,
                (long long)c.C, (long long)OH, (long long)OW, d);

    for (int nt : threads) {
        BenchPool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

        planar_call(use_ctx);                // warm-up / first touch
        packed_call(use_ctx);

        std::vector<double> a1, b, a2;
        for (int r = 0; r < rounds; ++r) {
            a1.push_back(time_us([&] { planar_call(use_ctx); }, iters));
            b.push_back(time_us([&] { packed_call(use_ctx); }, iters));
            a2.push_back(time_us([&] { planar_call(use_ctx); }, iters));
        }
        const double tp = median_of(a1);
        const double tk = median_of(b);
        const double null_gap = (median_of(a2) / tp - 1.0) * 100.0;

        std::printf("    %2dT  planar %8.1f us (%6.1f GFLOPS) | packed %8.1f us (%6.1f GFLOPS)"
                    " | %5.2fx  (null %+4.1f%%)\n",
                    nt, tp, flops / (tp * 1e3), tk, flops / (tk * 1e3), tp / tk, null_gap);
    }
}


// ============================================================
// The unroll itself: planar h1 against planar h4
// ============================================================
//
// Same kernel family, same buffers, one variable: R. Single-threaded on
// purpose — the question is instruction-level parallelism, and a pool would
// only add scheduling noise on top of it. The h4 arm runs the tail rows through
// h1, exactly as the driver does.

/// One channel's worth of the driver's planar loop, over the whole plane.
///
/// Edge masks are built the way the driver builds them — once per call from the
/// shape — and then handed to every channel. The first and last channel plane
/// are the tensor's first and last there, so they keep the bounds-checked path,
/// exactly as the driver's block test does.
void planar_kernel(const float* in, float* out, const float* w, const float* b,
                   const DepthwiseCase& c, int64_t OH, int64_t OW, bool h4) {
    const Epilogue ep{};
    constexpr int64_t kMaskMaxKw = 16;

    for (int64_t ch = 0; ch < c.C; ++ch) {
        k::DwConvParams p{};
        p.ID = 1; p.IH = c.IH; p.IW = c.IW;
        p.OH = OH; p.OW = OW;
        p.KD = 1; p.KH = c.KH; p.KW = c.KW;
        p.SD = 1; p.SH = c.SH; p.SW = c.SW;
        p.DD = 1; p.DH = 1; p.DW = 1;
        p.PD = 0; p.PH = c.PH; p.PW = c.PW;
        p.out_row_stride = OW;
        p.in_d_stride    = c.IH * c.IW;
        p.in_row_stride  = c.IW;
        p.od = 0;

        uint8_t needed[2 * kMaskMaxKw] = {};
        alignas(32) float masks[2 * kMaskMaxKw * 8] = {};
        k::DwPlanarEdges edges =
            k::dwconv_planar_edge_masks<kMaskMaxKw, float>(p, masks, needed);
        if (ch == 0 || ch == c.C - 1) { edges.masks = nullptr; }

        const float* in_c  = in  + ch * c.IH * c.IW;
        float*       out_c = out + ch * OH * OW;
        const float* w_c   = w   + ch * c.KH * c.KW;

        int64_t oh = 0;
        if (h4) {
            for (; oh + 3 < OH; oh += 4) {
                p.oh = oh;
                k::dwconv_planar_h4(out_c + oh * OW, in_c, w_c, b[ch], p, ep, false, edges);
            }
        }
        for (; oh < OH; ++oh) {
            p.oh = oh;
            k::dwconv_planar(out_c + oh * OW, in_c, w_c, b[ch], p, ep, false, edges);
        }
    }
}

void run_unroll_case(const DepthwiseCase& c, int iters, int rounds) {
    const int64_t ishape[] = {1, c.C, c.IH, c.IW};
    const int64_t wshape[] = {c.C, 1, c.KH, c.KW};
    const int64_t bshape[] = {c.C};

    auto [in_vec, in_nchw] = test::make_random_tensor(ishape, -1.0f, 1.0f, 41);
    auto [w_vec, w_nchw]   = test::make_random_tensor(wshape, -1.0f, 1.0f, 42);
    auto [b_vec, b_nchw]   = test::make_random_tensor(bshape, -0.5f, 0.5f, 43);

    const int64_t OH = (c.IH + 2 * c.PH - c.KH) / c.SH + 1;
    const int64_t OW = (c.IW + 2 * c.PW - c.KW) / c.SW + 1;

    std::vector<float> out1(static_cast<size_t>(c.C * OH * OW));
    std::vector<float> out4(out1.size());

    const float* in = in_nchw.ptr<float>();
    const float* w  = w_nchw.ptr<float>();
    const float* b  = b_nchw.ptr<float>();

    const auto h1 = [&] { planar_kernel(in, out1.data(), w, b, c, OH, OW, false); };
    const auto h4 = [&] { planar_kernel(in, out4.data(), w, b, c, OH, OW, true); };

    h1();
    h4();
    const double d = max_abs_diff(out1.data(), out4.data(), static_cast<int64_t>(out1.size()));

    const double flops = 2.0 * static_cast<double>(c.C) * OH * OW * c.KH * c.KW;
    std::printf("  %s\n", c.name);
    std::printf("    out[1,%lld,%lld,%lld]  |h1 - h4| = %.2e\n",
                (long long)c.C, (long long)OH, (long long)OW, d);

    std::vector<double> a1, b4, a2;
    for (int r = 0; r < rounds; ++r) {
        a1.push_back(time_us(h1, iters));
        b4.push_back(time_us(h4, iters));
        a2.push_back(time_us(h1, iters));
    }
    const double t1 = median_of(a1);
    const double t4 = median_of(b4);
    std::printf("    h1 %8.1f us (%5.1f GFLOPS) | h4 %8.1f us (%5.1f GFLOPS) | %5.2fx"
                "  (null %+4.1f%%)\n",
                t1, flops / (t1 * 1e3), t4, flops / (t4 * 1e3), t1 / t4,
                (median_of(a2) / t1 - 1.0) * 100.0);
}

}  // namespace

NNOPS_BENCH(depthwise_planar_vs_packed) {
    const std::vector<int> threads = bench_threads();
    const int iters = 20;
    const int rounds = 5;

    std::printf("depthwise conv: planar (NCHW, lane=width) against packed (NCHWC8, lane=channel)\n");
    std::printf("  pool=%s  threads=", test::bench_pool_name());
    for (size_t i = 0; i < threads.size(); ++i) {
        std::printf("%d%s", threads[i], i + 1 < threads.size() ? "," : "\n");
    }

    const DepthwiseCase cases[] = {
        {"dw 3x3 C=32  @112 s1",   32, 112, 112},
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

NNOPS_BENCH(depthwise_planar_unroll) {
    const int iters = 20;
    const int rounds = 5;

    std::printf("depthwise planar: one output row per call (h1) against four (h4)\n");

    const DepthwiseCase cases[] = {
        {"dw 3x3 C=32  @112 s1",   32, 112, 112},
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
        run_unroll_case(c, iters, rounds);
    }
}

// ============================================================
// MobileNetV2's actual depthwise mix
// ============================================================
//
// The seventeen stages as the model runs them (the expansion conv decides the
// channel count: 6x the block width), each one a separate operator call there.
// The sum is what the model's `depthwise_fused` row should read if the kernel
// rate holds at these shapes; the difference is per-call cost. Reference FLOPs
// are printed so the two can be compared without redoing the arithmetic.

NNOPS_BENCH(depthwise_planar_mobilenet) {
    const std::vector<int> threads = bench_threads();
    const int iters = 20;
    const int rounds = 5;

    std::printf("depthwise planar: MobileNetV2's seventeen stages, 3x3 p1\n");
    std::printf("  pool=%s  threads=", test::bench_pool_name());
    for (size_t i = 0; i < threads.size(); ++i) {
        std::printf("%d%s", threads[i], i + 1 < threads.size() ? "," : "\n");
    }

    const DepthwiseCase cases[] = {
        {"f1   16@112 s1",    16, 112, 112, 3, 3, 1, 1, 1, 1, true},
        {"f2  144@112 s2",   144, 112, 112, 3, 3, 2, 2, 1, 1, true},
        {"f3  144@56  s1",   144,  56,  56, 3, 3, 1, 1, 1, 1, true},
        {"f4  192@56  s2",   192,  56,  56, 3, 3, 2, 2, 1, 1, true},
        {"f5  192@28  s1",   192,  28,  28, 3, 3, 1, 1, 1, 1, true},
        {"f6  192@28  s1",   192,  28,  28, 3, 3, 1, 1, 1, 1, true},
        {"f7  384@28  s2",   384,  28,  28, 3, 3, 2, 2, 1, 1, true},
        {"f8  384@14  s1",   384,  14,  14, 3, 3, 1, 1, 1, 1, true},
        {"f9  384@14  s1",   384,  14,  14, 3, 3, 1, 1, 1, 1, true},
        {"f10 384@14  s1",   384,  14,  14, 3, 3, 1, 1, 1, 1, true},
        {"f11 576@14  s1",   576,  14,  14, 3, 3, 1, 1, 1, 1, true},
        {"f12 576@14  s1",   576,  14,  14, 3, 3, 1, 1, 1, 1, true},
        {"f13 576@14  s1",   576,  14,  14, 3, 3, 1, 1, 1, 1, true},
        {"f14 960@14  s2",   960,  14,  14, 3, 3, 2, 2, 1, 1, true},
        {"f15 960@7   s1",   960,   7,   7, 3, 3, 1, 1, 1, 1, true},
        {"f16 960@7   s1",   960,   7,   7, 3, 3, 1, 1, 1, 1, true},
        {"f17 1920@7  s1",  1920,   7,   7, 3, 3, 1, 1, 1, 1, true},
    };

    for (const auto& c : cases) {
        run_case(c, threads, iters, rounds);
    }
}

// ============================================================
// Narrow planes: the masked edge load, and what the width tail costs
// ============================================================
//
// Widths 7 and 14 are MobileNetV2's last two stages; 8, 15 and 16 are the same
// plane one element wider, i.e. the same number of taps and the nearest width
// whose last vector is full. The pairs (7,8), (15,16) are therefore a direct
// read on the scalar store tail, with the edge masking held constant.

NNOPS_BENCH(depthwise_planar_narrow) {
    const std::vector<int> threads = bench_threads();
    const int iters = 20;
    const int rounds = 5;

    std::printf("depthwise planar: narrow widths, masked edges\n");
    std::printf("  pool=%s  threads=", test::bench_pool_name());
    for (size_t i = 0; i < threads.size(); ++i) {
        std::printf("%d%s", threads[i], i + 1 < threads.size() ? "," : "\n");
    }

    const DepthwiseCase cases[] = {
        {"dw 3x3 C=960 @7  s1",  960,  7,  7},
        {"dw 3x3 C=960 @8  s1",  960,  8,  8},
        {"dw 3x3 C=960 @9  s1",  960,  9,  9},
        {"dw 3x3 C=576 @14 s1",  576, 14, 14},
        {"dw 3x3 C=576 @15 s1",  576, 15, 15},
        {"dw 3x3 C=576 @16 s1",  576, 16, 16},
    };

    for (const auto& c : cases) {
        run_case(c, threads, iters, rounds);
    }
}
