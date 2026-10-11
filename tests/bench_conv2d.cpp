/// @file bench_conv2d.cpp
/// @brief Conv2D (im2col + GEMM) micro-benchmark — plan + thread scaling.
///
/// Mirrors bench_matmul.cpp: every case runs the plan-driven fused fast path and
/// reports the resolved plan plus GFLOPS at 1, 2, 4, 8 and hardware threads.
///
/// For each (batch, group) the output spatial plane is split into oh-blocks;
/// when (batch, group) alone under-subscribes the pool the oh-blocks are
/// partitioned into parallel chunks (plan.oh_chunks). The report shows the block
/// geometry (icn/ocn/oh_block), the resulting oh-block count, and the total
/// parallel task count (N*G*oh_chunks).

#include "nnops/ops/conv2d.hpp"
#include "nnops/ops/depthwise_conv.hpp"
#include "nnops/ops/layout_convert.hpp"
#include "backend/cpu/conv2d_im2col.h"   // get_conv2d_plan, Conv2DPlan
#include "backend/cpu/conv2d_winograd.h" // get_conv2d_winograd_plan
#include "common/bench_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <type_traits>
#include <vector>

using namespace nnops;
using half = nnops::backend::cpu::half;
namespace cpu = nnops::backend::cpu;

namespace {

// Minimal fixed-size thread pool exposing a CpuBackend (same as bench_matmul).
struct SimplePool {
    explicit SimplePool(int nthreads) : nthreads_(nthreads) {
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

// Build a random tensor of type T (float -> f32, half -> f16).
template <typename T>
auto make_random(std::span<const int64_t> shape, uint64_t seed) {
    if constexpr (std::is_same_v<T, float>) {
        return test::make_random_tensor(shape, -1.0f, 1.0f, seed);
    } else {
        return test::make_random_f16_tensor(shape, -1.0f, 1.0f, seed);
    }
}

struct Conv2DCase {
    const char* name;
    int64_t N, IC, IH, IW, OC, KH, KW;
    int64_t SH = 1, SW = 1, PH = 0, PW = 0, DH = 1, DW = 1, groups = 1;
    bool bias = false, f16 = false;
};

template <typename T>
void run_conv2d(const Conv2DCase& c, const std::vector<int>& threads, int iters) {
    const int64_t ishape[] = {c.N, c.IC, c.IH, c.IW};
    const int64_t wshape[] = {c.OC, c.IC / c.groups, c.KH, c.KW};

    auto [in_vec, input] = make_random<T>(ishape, 1);
    auto [w_vec, weight] = make_random<T>(wshape, 2);

    std::vector<T> b_vec;
    TensorView bias_view;
    if (c.bias) {
        const int64_t bshape[] = {c.OC};
        auto [bv, b] = make_random<T>(bshape, 3);
        b_vec = std::move(bv);
        bias_view = b;
    }

    Conv2DAttributes attrs{};
    attrs.kernel_size = {c.KH, c.KW};
    attrs.stride      = {c.SH, c.SW};
    attrs.dilation    = {c.DH, c.DW};
    attrs.padding     = {c.PH, c.PW};
    attrs.groups      = c.groups;

    auto op = Conv2D::create(attrs, Backend::CPU);

    std::vector<TensorDesc> descs = {input.desc(), weight.desc()};
    std::vector<TensorView> ins   = {input, weight};
    if (c.bias) { descs.push_back(bias_view.desc()); ins.push_back(bias_view); }
    auto out_descs = op->getOutputTensorDesc(descs);

    const int64_t OH  = out_descs[0].dims[2];
    const int64_t OW  = out_descs[0].dims[3];
    const int64_t icg = c.IC / c.groups;
    const double flops_per_iter = 2.0 * static_cast<double>(c.N) * c.OC * OH * OW * icg * c.KH * c.KW;

    // Plan resolved at the largest thread count (shows the max oh-chunk split).
    // The bench pool reports thread ids, so the per-thread scratch sizing applies.
    const cpu::Conv2DPlan plan = cpu::get_conv2d_plan(attrs, descs, out_descs, threads.back(),
                                                      /*use_thread_slots=*/true);

    std::vector<T> out_buf(static_cast<size_t>(out_descs[0].numel()));
    auto output = test::make_planar(out_descs[0], out_buf.data());

    std::printf("  %s\n", c.name);
    std::printf("    in[%lld,%lld,%lld,%lld]  w[%lld,%lld,%lldx%lld]  -> out[%lld,%lld,%lld,%lld]  %s\n",
                (long long)c.N, (long long)c.IC, (long long)c.IH, (long long)c.IW,
                (long long)c.OC, (long long)icg, (long long)c.KH, (long long)c.KW,
                (long long)c.N, (long long)c.OC, (long long)OH, (long long)OW,
                c.f16 ? "f16" : "f32");
    std::printf("    icn=%lld ocn=%lld oh_block=%lld oh_blocks=%lld  tasks=N*G*oh_chunks=%lld"
                "  ws=%zuB\n",
                (long long)plan.icn_block, (long long)plan.ocn_block,
                (long long)plan.oh_block, (long long)plan.num_oh_blocks,
                (long long)(c.N * c.groups * plan.oh_chunks), plan.workspace_size);

    auto compute = [&](const ComputeContext& ctx) { op->compute(output, ins, ctx, nullptr); };
    std::vector<double> gflops;
    for (int nt : threads) {
        SimplePool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

        for (int i = 0; i < 2; ++i) { compute(use_ctx); }   // warm-up / page faults

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) { compute(use_ctx); }
        auto end = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        gflops.push_back(flops_per_iter * static_cast<double>(iters) / (ms * 1e6));
    }

    const double g1 = gflops.empty() ? 0.0 : gflops[0];
    std::printf("    ");
    for (size_t i = 0; i < threads.size(); ++i) {
        const double speedup = g1 > 0.0 ? gflops[i] / g1 : 0.0;
        std::printf(" %2dT %7.1f GFLOPS (%.2fx)", threads[i], gflops[i], speedup);
        if (i + 1 < threads.size()) { std::printf(" |"); }
    }
    std::printf("\n");
}

void bench_conv2d_case(const Conv2DCase& c) {
    const int iters = 20;
    std::vector<int> threads{1, 2, 4, 8};
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw > 1 && std::find(threads.begin(), threads.end(), hw) == threads.end()) {
        threads.push_back(hw);
    }
    if (c.f16) { run_conv2d<half>(c, threads, iters); }
    else        { run_conv2d<float>(c, threads, iters); }
}

// ============================================================
// DepthwiseConv — dedicated NCHWC8 kernel (vs the im2col group path)
// ============================================================

struct DepthwiseCase {
    const char* name;
    int64_t N, C, IH, IW, KH, KW;
    int64_t SH = 1, SW = 1, PH = 0, PW = 0;
};

void run_depthwise(const DepthwiseCase& c, const std::vector<int>& threads, int iters) {
    const int64_t ishape[] = {c.N, c.C, c.IH, c.IW};
    const int64_t wshape[] = {c.C, 1, c.KH, c.KW};

    auto [in_vec, in_nchw] = test::make_random_tensor(ishape, -1.0f, 1.0f, 21);
    auto [w_vec, w_nchw]   = test::make_random_tensor(wshape, -1.0f, 1.0f, 22);

    DepthwiseConvAttributes attrs{};
    attrs.kernel_size = {1, c.KH, c.KW};
    attrs.stride      = {1, c.SH, c.SW};
    attrs.dilation    = {1, 1, 1};
    attrs.padding     = {0, c.PH, c.PW};

    // Pack input NCHW -> NCHWC8 (layout convert, done once outside timing).
    auto lc_in = LayoutConvert::create(TensorLayout::NCHWC8, Backend::CPU);
    TensorDesc in_desc = in_nchw.desc();
    const TensorDesc lc_in_arr[] = {in_desc};
    auto in_c8_descs = lc_in->getOutputTensorDesc(lc_in_arr);
    std::vector<float> in_c8_buf(static_cast<size_t>(in_c8_descs[0].storage_bytes()));
    auto in_c8 = test::make_packed(in_c8_descs[0], in_c8_buf.data());
    {
        const TensorView i[] = {in_nchw};
        TensorView o[] = {in_c8};
        lc_in->compute(o, i);
    }

    // Prepack weight: query -> allocate -> pack (once, outside timing).
    auto dw_op = DepthwiseConv::create(attrs, Backend::CPU);
    TensorView pw_query;
    {
        const TensorView wq[] = {w_nchw};
        TensorView pq[] = {pw_query};
        dw_op->prepackWeights(wq, pq);
        pw_query = pq[0];
    }
    std::vector<float> pw_buf(static_cast<size_t>(pw_query.numel()));
    auto pw_view = test::make_planar(pw_query.desc(), pw_buf.data());
    {
        const TensorView wp[] = {w_nchw};
        TensorView po[] = {pw_view};
        dw_op->prepackWeights(wp, po);
    }

    // Output desc (NCHWC8) + geometry.
    TensorDesc in_c8_desc = in_c8.desc();
    TensorDesc pw_desc = pw_view.desc();
    const TensorDesc dw_arr[] = {in_c8_desc, pw_desc};
    auto out_descs = dw_op->getOutputTensorDesc(dw_arr);
    const int64_t OH = out_descs[0].dims[2];
    const int64_t OW = out_descs[0].dims[3];
    const double flops_per_iter = 2.0 * static_cast<double>(c.N) * c.C * OH * OW * c.KH * c.KW;

    std::vector<float> out_buf(static_cast<size_t>(out_descs[0].storage_bytes()));
    auto out_c8 = test::make_packed(out_descs[0], out_buf.data());

    std::printf("  %s\n", c.name);
    std::printf("    in[%lld,%lld,%lld,%lld]  w[%lld,1,%lldx%lld]  -> out[%lld,%lld,%lld,%lld]  NCHWC8\n",
                (long long)c.N, (long long)c.C, (long long)c.IH, (long long)c.IW,
                (long long)c.C, (long long)c.KH, (long long)c.KW,
                (long long)c.N, (long long)c.C, (long long)OH, (long long)OW);

    auto compute = [&](const ComputeContext& ctx) {
        const TensorView ins[] = {in_c8, pw_view};
        TensorView outs[] = {out_c8};
        dw_op->compute(outs, ins, ctx, nullptr);
    };
    std::vector<double> gflops;
    for (int nt : threads) {
        SimplePool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

        for (int i = 0; i < 2; ++i) { compute(use_ctx); }   // warm-up / page faults

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) { compute(use_ctx); }
        auto end = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        gflops.push_back(flops_per_iter * static_cast<double>(iters) / (ms * 1e6));
    }

    const double g1 = gflops.empty() ? 0.0 : gflops[0];
    std::printf("    ");
    for (size_t i = 0; i < threads.size(); ++i) {
        const double speedup = g1 > 0.0 ? gflops[i] / g1 : 0.0;
        std::printf(" %2dT %7.1f GFLOPS (%.2fx)", threads[i], gflops[i], speedup);
        if (i + 1 < threads.size()) { std::printf(" |"); }
    }
    std::printf("\n");
}

void bench_depthwise_case(const DepthwiseCase& c) {
    const int iters = 20;
    std::vector<int> threads{1, 2, 4, 8};
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw > 1 && std::find(threads.begin(), threads.end(), hw) == threads.end()) {
        threads.push_back(hw);
    }
    run_depthwise(c, threads, iters);
}

// ============================================================
// Winograd F(2,3) vs im2col — interleaved A/B with a null control
// ============================================================

struct WinoCase {
    const char* name;
    int64_t N, IC, IH, IW, OC;
    int64_t PH = 1, PW = 1;
    bool bias = false, f16 = false;
};

/// Mean wall time of ONE call, in milliseconds.
///
/// Per-call, not per-batch: the caller divides a FLOP count taken per iteration
/// by this and prints the quotient as a per-call rate, so a batch total here
/// inflates the ms column by `iters` and the GFLOPS column by 1000/iters —
/// 20x and 50x at the 20 iterations this section runs. The speedup column is a
/// ratio and cancelled it, so the A/B verdicts stayed right while every
/// absolute number was wrong. The caller's divisor was off in the same
/// direction and was fixed in the same change.
template <typename Fn>
double time_ms(Fn&& fn, int iters) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) { fn(); }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() /
           std::max(iters, 1);
}

/// Interleaved A/B. Within one round the two variants alternate A B A B, and the
/// second A is the **null control**: same op, same binary, same buffers, so an
/// A2/A1 gap is pure measurement drift (fanless thermal soak, cache state) and
/// nothing else. Block-ordering the runs — all A, then all B — manufactures a
/// regression out of that drift, so the variants are interleaved and a speedup
/// is only reported as real when it clears the null floor.
///
/// Prepack of U happens once, outside every timed region.
template <typename T>
void run_wino_ab(const WinoCase& c, const std::vector<int>& threads,
                 int iters, int rounds) {
    const int64_t ishape[] = {c.N, c.IC, c.IH, c.IW};
    const int64_t wshape[] = {c.OC, c.IC, 3, 3};

    auto [in_vec, input] = make_random<T>(ishape, 11);
    auto [w_vec, weight] = make_random<T>(wshape, 12);

    std::vector<T> b_vec;
    TensorView bias_view;
    if (c.bias) {
        const int64_t bshape[] = {c.OC};
        auto [bv, b] = make_random<T>(bshape, 13);
        b_vec = std::move(bv);
        bias_view = b;
    }

    Conv2DAttributes base{};
    base.kernel_size = {3, 3};
    base.stride      = {1, 1};
    base.padding     = {c.PH, c.PW};

    Conv2DAttributes im_attrs = base;
    im_attrs.algorithm = ConvAlgorithm::ImplicitGemm;
    Conv2DAttributes wn_attrs = base;
    wn_attrs.algorithm = ConvAlgorithm::Winograd;

    auto im_op = Conv2D::create(im_attrs, Backend::CPU);
    auto wn_op = Conv2D::create(wn_attrs, Backend::CPU);

    std::vector<TensorDesc> descs = {input.desc(), weight.desc()};
    if (c.bias) { descs.push_back(bias_view.desc()); }
    auto out_descs = im_op->getOutputTensorDesc(descs);
    const int64_t OH = out_descs[0].dims[2];
    const int64_t OW = out_descs[0].dims[3];
    const double flops_per_iter =
        2.0 * static_cast<double>(c.N) * c.OC * OH * OW * c.IC * 9;
    // Winograd multiplies once per transform-domain product, not per k-area
    // term: 16 GEMM positions over tiles, instead of OH·OW·9 per (n, oc, ic).
    const int64_t TH = (OH + 1) / 2, TW_pad = ((OW + 1) / 2 + 7) / 8 * 8;
    const double flops_per_iter_wino =
        2.0 * static_cast<double>(c.N) * c.OC * (TH * TW_pad) * c.IC * 16;

    // Prepack U (query -> allocate -> pack), once and outside every timer.
    std::vector<T> u_buf;
    TensorView u_view;
    {
        TensorView q;
        const TensorView wi[] = {weight};
        TensorView wo[] = {q};
        wn_op->prepackWeights(wi, wo);
        q = wo[0];
        u_buf.assign(static_cast<size_t>(q.storage_bytes() / sizeof(T)), T{});
        u_view = TensorView(q.shape_span(), q.data_type(), u_buf.data(), q.pitch(),
                            TensorLayout::PackedWinoWeight);
        const TensorView wi2[] = {weight};
        TensorView wo2[] = {u_view};
        wn_op->prepackWeights(wi2, wo2);
    }

    std::vector<T> out_buf(static_cast<size_t>(out_descs[0].numel()));
    auto output = test::make_planar(out_descs[0], out_buf.data());

    std::vector<TensorView> im_ins = {input, weight};
    std::vector<TensorView> wn_ins = {input, u_view};
    if (c.bias) { im_ins.push_back(bias_view); wn_ins.push_back(bias_view); }

    // Report the resolved Winograd geometry (descs are the same for both ops).
    {
        std::vector<TensorDesc> wd = {input.desc(), u_view.desc()};
        if (c.bias) { wd.push_back(bias_view.desc()); }
        const TensorDesc wo_arr[] = {out_descs[0]};
        const auto wp = cpu::get_conv2d_winograd_plan(base, wd, wo_arr, 1, false);
        std::printf("    wino: tiles=%lldx%lld(pad %lld) icn=%lld ocn=%lld tile_blk=%lld"
                    " a_tasks=%lld b_tasks=%lld ws=%zuB\n",
                    (long long)wp.num_tile_rows, (long long)wp.num_tile_cols,
                    (long long)wp.num_tile_cols_pad, (long long)wp.icn_block,
                    (long long)wp.ocn_block, (long long)wp.tile_block,
                    (long long)wp.num_a_tasks, (long long)wp.num_b_tasks,
                    wp.workspace_size);
    }

    std::printf("  %s\n", c.name);
    std::printf("    in[%lld,%lld,%lld,%lld]  w[%lld,%lld,3x3]  -> out[%lld,%lld,%lld,%lld]  %s\n",
                (long long)c.N, (long long)c.IC, (long long)c.IH, (long long)c.IW,
                (long long)c.OC, (long long)c.IC, (long long)c.N, (long long)c.OC,
                (long long)OH, (long long)OW, c.f16 ? "f16" : "f32");

    for (int nt : threads) {
        SimplePool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

        auto run_im = [&]() { im_op->compute(output, im_ins, use_ctx, nullptr); };
        auto run_wn = [&]() { wn_op->compute(output, wn_ins, use_ctx, nullptr); };

        run_im(); run_wn(); run_im(); run_wn();   // warm-up / page faults

        std::vector<double> im_ms, wn_ms, null_pct;
        for (int r = 0; r < rounds; ++r) {
            const double a1 = time_ms(run_im, iters);
            const double b1 = time_ms(run_wn, iters);
            const double a2 = time_ms(run_im, iters);   // null control
            const double b2 = time_ms(run_wn, iters);
            const double a = std::min(a1, a2), b = std::min(b1, b2);
            im_ms.push_back(a);
            wn_ms.push_back(b);
            null_pct.push_back(std::abs(a2 - a1) / std::min(a1, a2) * 100.0);
        }
        auto median = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };
        const double im = median(im_ms), wn = median(wn_ms);
        // Median, not max: one descheduled round should not be allowed to
        // disable the verdict for every other round.
        const double nf = median(null_pct);
        const double speedup = im / wn;
        // The two GFLOPS columns are each variant's own useful-work rate; they
        // are not comparable (im2col does 9/4x the multiplies). The verdict is
        // the wall-time speedup, and only when it clears the null floor.
        const bool real = std::abs(im - wn) / im * 100.0 > nf;

        std::printf("    %2dT  im2col %7.3f ms (%7.1f GFLOPS) | wino %7.3f ms (%7.1f GFLOPS)"
                    " | speedup %.2fx%s | null %.1f%%\n",
                    nt, im, flops_per_iter / im / 1e6,
                    wn, flops_per_iter_wino / wn / 1e6,
                    speedup, real ? "" : " (within noise)", nf);
    }
}

void bench_wino_case(const WinoCase& c) {
    const int iters = 20, rounds = 5;
    std::vector<int> threads{1, 2, 4, 8};
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw > 1 && std::find(threads.begin(), threads.end(), hw) == threads.end()) {
        threads.push_back(hw);
    }
    if (c.f16) { run_wino_ab<half>(c, threads, iters, rounds); }
    else       { run_wino_ab<float>(c, threads, iters, rounds); }
}

}  // anonymous namespace

// ============================================================
// Routing / tiling sweep
// ============================================================

NNOPS_BENCH(conv2d_f32_resnet_first_7x7) {
    // ResNet first layer: stride-2 downsampling, 3 -> 64 channels.
    bench_conv2d_case({"f32 resnet-first 1x3x224x224 -> 64 x 7x7 s2 p3", 1, 3, 224, 224, 64, 7, 7,
                       /*SH=*/2, /*SW=*/2, /*PH=*/3, /*PW=*/3});
}

NNOPS_BENCH(conv2d_f32_resnet_block_3x3) {
    bench_conv2d_case({"f32 resnet-block 1x64x56x56 -> 64 x 3x3 s1 p1", 1, 64, 56, 56, 64, 3, 3,
                       /*SH=*/1, /*SW=*/1, /*PH=*/1, /*PW=*/1});
}

NNOPS_BENCH(conv2d_f32_pointwise_1x1) {
    // 1x1 bottleneck expand — large K (IC) and N (OC), karea == 1 skips im2col.
    bench_conv2d_case({"f32 pointwise 1x256x28x28 -> 512 x 1x1 s1", 1, 256, 28, 28, 512, 1, 1});
}

NNOPS_BENCH(conv2d_f32_depthwise_3x3) {
    // Depthwise (groups == IC) 3x3 — the grouped im2col path.
    bench_conv2d_case({"f32 depthwise 1x256x28x28 -> 256 x 3x3 g256 s1 p1", 1, 256, 28, 28, 256, 3, 3,
                       /*SH=*/1, /*SW=*/1, /*PH=*/1, /*PW=*/1, /*DH=*/1, /*DW=*/1, /*groups=*/256});
}

NNOPS_BENCH(conv2d_f32_batch) {
    // Batch=8 exercises the (batch, group) parallel dimension.
    bench_conv2d_case({"f32 batch 8x64x28x28 -> 128 x 3x3 s1 p1", 8, 64, 28, 28, 128, 3, 3,
                       /*SH=*/1, /*SW=*/1, /*PH=*/1, /*PW=*/1});
}

NNOPS_BENCH(conv2d_f16_resnet_block_3x3) {
    bench_conv2d_case({"f16 resnet-block 1x64x56x56 -> 64 x 3x3 s1 p1", 1, 64, 56, 56, 64, 3, 3,
                       /*SH=*/1, /*SW=*/1, /*PH=*/1, /*PW=*/1, /*DH=*/1, /*DW=*/1, /*groups=*/1,
                       /*bias=*/false, /*f16=*/true});
}

NNOPS_BENCH(conv2d_f16_resnet_first_7x7) {
    bench_conv2d_case({"f16 resnet-first 1x3x224x224 -> 64 x 7x7 s2 p3", 1, 3, 224, 224, 64, 7, 7,
                       /*SH=*/2, /*SW=*/2, /*PH=*/3, /*PW=*/3, /*DH=*/1, /*DW=*/1, /*groups=*/1,
                       /*bias=*/false, /*f16=*/true});
}

NNOPS_BENCH(conv2d_f16_pointwise_1x1) {
    bench_conv2d_case({"f16 pointwise 1x256x28x28 -> 512 x 1x1 s1", 1, 256, 28, 28, 512, 1, 1,
                       /*SH=*/1, /*SW=*/1, /*PH=*/0, /*PW=*/0, /*DH=*/1, /*DW=*/1, /*groups=*/1,
                       /*bias=*/false, /*f16=*/true});
}

NNOPS_BENCH(conv2d_f16_grouped_3x3) {
    // groups == IC: the grouped im2col path.
    bench_conv2d_case({"f16 depthwise 1x256x28x28 -> 256 x 3x3 g256 s1 p1", 1, 256, 28, 28, 256, 3, 3,
                       /*SH=*/1, /*SW=*/1, /*PH=*/1, /*PW=*/1, /*DH=*/1, /*DW=*/1, /*groups=*/256,
                       /*bias=*/false, /*f16=*/true});
}

NNOPS_BENCH(conv2d_f16_batch) {
    bench_conv2d_case({"f16 batch 8x64x28x28 -> 128 x 3x3 s1 p1", 8, 64, 28, 28, 128, 3, 3,
                       /*SH=*/1, /*SW=*/1, /*PH=*/1, /*PW=*/1, /*DH=*/1, /*DW=*/1, /*groups=*/1,
                       /*bias=*/false, /*f16=*/true});
}

// ============================================================
// DepthwiseConv — dedicated NCHWC8 kernel
// ============================================================

NNOPS_BENCH(dwconv_f32_3x3) {
    // Same shape as the im2col depthwise case above (SAME padding, OH=28) —
    // direct head-to-head.
    bench_depthwise_case({"f32 dwconv 1x256x28x28 -> 3x3 s1 p1", 1, 256, 28, 28, 3, 3,
                          /*SH=*/1, /*SW=*/1, /*PH=*/1, /*PW=*/1});
}

NNOPS_BENCH(dwconv_f32_3x3_large) {
    bench_depthwise_case({"f32 dwconv 1x256x56x56 -> 3x3 s1 p1", 1, 256, 56, 56, 3, 3,
                          /*SH=*/1, /*SW=*/1, /*PH=*/1, /*PW=*/1});
}

NNOPS_BENCH(dwconv_f32_stride2) {
    // MobileNet-style downsampling block.
    bench_depthwise_case({"f32 dwconv 1x256x56x56 -> 3x3 s2", 1, 256, 56, 56, 3, 3,
                          /*SH=*/2, /*SW=*/2});
}

NNOPS_BENCH(dwconv_f32_5x5) {
    bench_depthwise_case({"f32 dwconv 1x128x28x28 -> 5x5 s1 p2", 1, 128, 28, 28, 5, 5,
                          /*SH=*/1, /*SW=*/1, /*PH=*/2, /*PW=*/2});
}

NNOPS_BENCH(dwconv_f32_batch) {
    bench_depthwise_case({"f32 dwconv batch 8x128x28x28 -> 3x3 s1 p1", 8, 128, 28, 28, 3, 3,
                          /*SH=*/1, /*SW=*/1, /*PH=*/1, /*PW=*/1});
}

// ============================================================
// Winograd F(2,3) vs im2col
// ============================================================
//
// Winograd trades 2.25× fewer multiplies for transforms whose cost is fixed per
// tile and per channel, so it only wins once IC and OC are large. The first two
// cases are the regime it is for; the small-channel and single-channel cases are
// the degenerate end where it must lose — kept in the sweep so a regression in
// the guard shows up as a number, not an assumption.

NNOPS_BENCH(conv2d_wino_f32_64ch) {
    bench_wino_case({"f32 wino 1x64x56x56 -> 64 x 3x3 s1 p1", 1, 64, 56, 56, 64});
}

NNOPS_BENCH(conv2d_wino_f32_256ch) {
    bench_wino_case({"f32 wino 1x256x28x28 -> 256 x 3x3 s1 p1", 1, 256, 28, 28, 256});
}

NNOPS_BENCH(conv2d_wino_f32_32ch) {
    bench_wino_case({"f32 wino 1x32x56x56 -> 32 x 3x3 s1 p1", 1, 32, 56, 56, 32});
}

NNOPS_BENCH(conv2d_wino_f32_8ch_degenerate) {
    // Below the ~16-channel floor: the transforms and the extra passes dominate.
    bench_wino_case({"f32 wino 1x8x56x56 -> 8 x 3x3 s1 p1 (degenerate)", 1, 8, 56, 56, 8});
}

NNOPS_BENCH(conv2d_wino_f32_pad0_odd) {
    // pad 0 and an odd OW exercise the tile-grid edges (row1_ok, column masking).
    bench_wino_case({"f32 wino 1x64x55x55 -> 64 x 3x3 s1 p0", 1, 64, 55, 55, 64,
                     /*PH=*/0, /*PW=*/0});
}

NNOPS_BENCH(conv2d_wino_f16_64ch) {
    bench_wino_case({"f16 wino 1x64x56x56 -> 64 x 3x3 s1 p1", 1, 64, 56, 56, 64,
                     /*PH=*/1, /*PW=*/1, /*bias=*/false, /*f16=*/true});
}

NNOPS_BENCH(conv2d_wino_f16_256ch) {
    bench_wino_case({"f16 wino 1x256x28x28 -> 256 x 3x3 s1 p1", 1, 256, 28, 28, 256,
                     /*PH=*/1, /*PW=*/1, /*bias=*/false, /*f16=*/true});
}