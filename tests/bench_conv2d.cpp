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
///
/// Cases are point-shaped — one named network layer per bench — plus, at the end,
/// kernel-area sweep tables: one shape walked across kernel size, plane size and
/// group count. The plan's block sizes are a function of kernel *area*
/// (get_ic_block), so no single 3x3 point case can show what a 5x5 or 7x7 does to
/// them.

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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

using namespace nnops;
using half = nnops::backend::cpu::half;
using nnops::backend::cpu::float_to_half;
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
                "  ws=%zuB%s\n",
                (long long)plan.icn_block, (long long)plan.ocn_block,
                (long long)plan.oh_block, (long long)plan.num_oh_blocks,
                (long long)(c.N * c.groups * plan.oh_chunks), plan.workspace_size,
                plan.pointwise ? "  pointwise(no col buffer)" : "");

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

/// Thread ladder for every conv bench. `NNOPS_BENCH_THREADS=1` (or "1,4") pins
/// it: the multi-thread cells on a busy host mostly measure background load,
/// and a pinned run is the only way to get a clean per-core number.
std::vector<int> conv_bench_threads() {
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

void bench_conv2d_case(const Conv2DCase& c) {
    const int iters = 20;
    const std::vector<int> threads = conv_bench_threads();
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
    const std::vector<int> threads = conv_bench_threads();
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

/// Median of the per-round timings — one descheduled round should not decide
/// the verdict for every other round.
double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
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
    // Planned for the widest thread count in the sweep: the scratch is per
    // worker slot, so the serial plan is the one that would size it by task.
    {
        std::vector<TensorDesc> wd = {input.desc(), u_view.desc()};
        if (c.bias) { wd.push_back(bias_view.desc()); }
        const TensorDesc wo_arr[] = {out_descs[0]};
        const int max_nt = threads.empty() ? 1 : *std::max_element(threads.begin(),
                                                                  threads.end());
        const auto wp = cpu::get_conv2d_winograd_plan(base, wd, wo_arr, max_nt, true);
        std::printf("    wino: tiles=%lldx%lld(pad %lld) icn=%lld ocn=%lld tile_blk=%lld"
                    " tasks=%lld x %lld slots, slot=%lld el, ws=%zuB\n",
                    (long long)wp.num_tile_rows, (long long)wp.num_tile_cols,
                    (long long)wp.num_tile_cols_pad, (long long)wp.icn_block,
                    (long long)wp.ocn_block, (long long)wp.tile_block,
                    (long long)wp.num_tasks, (long long)wp.num_slots,
                    (long long)wp.slot_size, wp.workspace_size);
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
        const double im = median_of(im_ms), wn = median_of(wn_ms);
        const double nf = median_of(null_pct);
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
    const std::vector<int> threads = conv_bench_threads();
    if (c.f16) { run_wino_ab<half>(c, threads, iters, rounds); }
    else       { run_wino_ab<float>(c, threads, iters, rounds); }
}

// ============================================================
// Quantized Conv2D — s8 weight × u8/s8/f32/f16 activation → fp output
// ============================================================
//
// Rates are GOPS of int8 MACs, counted with the same
// 2·N·OC·OH·OW·(IC/G)·KH·KW expression the fp cases call FLOPs, so the A/B
// rows below compare wall time for the same amount of useful work.
//
// A pre-quantized activation (u8/s8) is the deployment shape: the tensor
// arrives quantized, so im2col is a byte copy. An fp activation is quantized
// while the tile is gathered — identical MACs, extra work per element — and
// `conv2d_i8_fp_act_overhead_ab` measures exactly that difference.

struct Conv2DI8Case {
    const char* name;
    int64_t N, IC, IH, IW, OC, KH, KW;
    int64_t SH = 1, SW = 1, PH = 0, PW = 0, DH = 1, DW = 1, groups = 1;
    bool bias = false;
    DataType act = DataType::u8;      // stored activation dtype
    DataType out = DataType::f32;
    DataType act_q = DataType::s8;    // quantization target for fp activations
    bool asym_w = false;              // per-channel weight zero point != 0
};

/// float → s8/u8 storage value, with the arithmetic the kernels use.
int32_t quant_i8(float x, float inv_scale, int32_t zp, bool u8) {
    const float v = std::fmaf(x, inv_scale, static_cast<float>(zp));
    const int32_t q = static_cast<int32_t>(std::nearbyintf(v));
    return std::min(std::max(q, u8 ? 0 : -128), u8 ? 255 : 127);
}

const char* dtype_name(DataType dt) {
    switch (dt) {
        case DataType::f32: return "f32";
        case DataType::f16: return "f16";
        case DataType::u8:  return "u8";
        case DataType::s8:  return "s8";
        default:            return "?";
    }
}

/// A fully built quantized convolution. Construction is the whole setup —
/// operand storage, the views over it, the op, the output and the plan — done
/// once outside every timed region; `run` is then the entire timed body.
struct I8Op {
    I8Op(const Conv2DI8Case& c, int plan_threads) {
        const int64_t N = c.N, IC = c.IC, IH = c.IH, IW = c.IW, OC = c.OC;
        icg = IC / c.groups;
        const bool act_u8 = (c.act == DataType::u8);
        const bool fp_act = (c.act == DataType::f32 || c.act == DataType::f16);
        const bool target_u8 = (c.act_q == DataType::u8);
        const float s_x = 0.0078125f;   // 1/128 — exact, and the fp range fits
        const int32_t zp_x = (act_u8 || (fp_act && target_u8)) ? 128 : 0;
        const int64_t act_n = N * IC * IH * IW;

        act.resize(static_cast<size_t>(act_n) * static_cast<size_t>(data_type_size(c.act)));
        {
            test::XorShift128 rng(31);
            const float inv = 1.0f / s_x;
            for (int64_t i = 0; i < act_n; ++i) {
                const float x = rng.next_float(-1.0f, 1.0f);
                if (c.act == DataType::f32) {
                    reinterpret_cast<float*>(act.data())[i] = x;
                } else if (c.act == DataType::f16) {
                    reinterpret_cast<half*>(act.data())[i] = float_to_half(x);
                } else {
                    act[static_cast<size_t>(i)] =
                        static_cast<uint8_t>(quant_i8(x, inv, zp_x, act_u8));
                }
            }
        }

        const int64_t w_n = OC * icg * c.KH * c.KW;
        w.resize(static_cast<size_t>(w_n));
        w_s.resize(static_cast<size_t>(OC));
        w_zp.resize(static_cast<size_t>(OC));
        {
            test::XorShift128 rng(32);
            for (int64_t i = 0; i < w_n; ++i) {
                w[static_cast<size_t>(i)] = static_cast<int8_t>(
                    quant_i8(rng.next_float(-1.0f, 1.0f), 127.0f, 0, /*u8=*/false));
            }
            for (int64_t oc = 0; oc < OC; ++oc) {
                w_s[static_cast<size_t>(oc)] = 0.01f * static_cast<float>(1 + oc % 5);
                w_zp[static_cast<size_t>(oc)] =
                    c.asym_w ? static_cast<int32_t>(-40 + (oc * 7) % 60) : 0;
            }
        }

        if (c.bias) {
            bias.resize(static_cast<size_t>(OC) * static_cast<size_t>(data_type_size(c.out)));
            test::XorShift128 rng(33);
            for (int64_t oc = 0; oc < OC; ++oc) {
                const float b = rng.next_float(-0.5f, 0.5f);
                if (c.out == DataType::f16) {
                    reinterpret_cast<half*>(bias.data())[oc] = float_to_half(b);
                } else {
                    reinterpret_cast<float*>(bias.data())[oc] = b;
                }
            }
        }

        // Per-tensor activation, per-channel weight — the contract's shape.
        qa.granularity = QuantGranularity::PerTensor;
        qa.scale = s_x;
        qa.zero_point = zp_x;
        qw.granularity = QuantGranularity::PerChannel;
        qw.scale_data = w_s.data();
        qw.zero_point_data = w_zp.data();
        qw.num_scales = OC;

        attrs.kernel_size = {c.KH, c.KW};
        attrs.stride      = {c.SH, c.SW};
        attrs.dilation    = {c.DH, c.DW};
        attrs.padding     = {c.PH, c.PW};
        attrs.groups      = c.groups;
        attrs.output_dtype = c.out;
        attrs.act_quant_dtype = c.act_q;

        const int64_t ishape[] = {N, IC, IH, IW};
        const int64_t wshape[] = {OC, icg, c.KH, c.KW};
        ins.push_back(TensorView(ishape, c.act, act.data(), TensorLayout::NCHW, qa));
        ins.push_back(TensorView(wshape, DataType::s8, w.data(), TensorLayout::NCHW, qw));
        if (c.bias) {
            const int64_t bshape[] = {OC};
            ins.push_back(TensorView(bshape, c.out, bias.data(), TensorLayout::NCHW));
        }

        std::vector<TensorDesc> descs;
        descs.reserve(ins.size());
        for (const auto& v : ins) { descs.push_back(v.desc()); }

        op = Conv2D::create(attrs, Backend::CPU);
        const auto outs = op->getOutputTensorDesc(descs);
        OH = outs[0].dims[2];
        OW = outs[0].dims[3];
        out_buf.assign(outs[0].nbytes(), 0);
        out = test::make_planar(outs[0], out_buf.data());

        // Resolved at the thread count the pair runs on, with per-thread scratch
        // sized in — the bench pool reports thread ids. A single thread runs on
        // a bare ComputeContext, which reports none, so it is sized per task.
        plan = cpu::get_conv2d_plan(attrs, descs, outs, plan_threads,
                                    /*use_thread_slots=*/plan_threads > 1);
        work = 2.0 * static_cast<double>(N) * OC * OH * OW *
               static_cast<double>(icg) * c.KH * c.KW;
    }

    void run(const ComputeContext& ctx) { op->compute(out, ins, ctx, nullptr); }

    Conv2DAttributes attrs{};
    QuantParams qa{}, qw{};
    std::vector<uint8_t> act, bias, out_buf;
    std::vector<int8_t>  w;
    std::vector<float>   w_s;
    std::vector<int32_t> w_zp;
    std::vector<TensorView> ins;
    TensorView out;
    std::unique_ptr<Conv2D> op;
    cpu::Conv2DPlan plan{};
    int64_t OH = 0, OW = 0, icg = 0;
    double work = 0.0;   // MACs per call
};

/// The fp32 baseline for the A/B — the same geometry through the fp
/// im2col+GEMM route, kept alive across rounds. `run_conv2d` builds, reports and
/// tears everything down in one call, which is the wrong shape for an
/// interleaved comparison.
struct Fp32Op {
    explicit Fp32Op(const Conv2DCase& c) {
        const int64_t ishape[] = {c.N, c.IC, c.IH, c.IW};
        const int64_t wshape[] = {c.OC, c.IC / c.groups, c.KH, c.KW};
        icg = c.IC / c.groups;

        in_buf.resize(static_cast<size_t>(c.N * c.IC * c.IH * c.IW));
        in = TensorView(ishape, DataType::f32, in_buf.data(), TensorLayout::NCHW);
        test::fill_random_float(in, -1.0f, 1.0f, 41);

        w_buf.resize(static_cast<size_t>(c.OC * icg * c.KH * c.KW));
        w = TensorView(wshape, DataType::f32, w_buf.data(), TensorLayout::NCHW);
        test::fill_random_float(w, -1.0f, 1.0f, 42);

        attrs.kernel_size = {c.KH, c.KW};
        attrs.stride      = {c.SH, c.SW};
        attrs.dilation    = {c.DH, c.DW};
        attrs.padding     = {c.PH, c.PW};
        attrs.groups      = c.groups;

        std::vector<TensorDesc> descs = {in.desc(), w.desc()};
        ins = {in, w};
        if (c.bias) {
            b_buf.resize(static_cast<size_t>(c.OC));
            bias = TensorView(std::span<const int64_t>(&c.OC, 1), DataType::f32,
                              b_buf.data(), TensorLayout::NCHW);
            test::fill_random_float(bias, -0.5f, 0.5f, 43);
            descs.push_back(bias.desc());
            ins.push_back(bias);
        }

        op = Conv2D::create(attrs, Backend::CPU);
        const auto outs = op->getOutputTensorDesc(descs);
        OH = outs[0].dims[2];
        OW = outs[0].dims[3];
        out_buf.assign(outs[0].nbytes(), 0);
        out = test::make_planar(outs[0], out_buf.data());
        work = 2.0 * static_cast<double>(c.N) * c.OC * OH * OW *
               static_cast<double>(icg) * c.KH * c.KW;
    }

    void run(const ComputeContext& ctx) { op->compute(out, ins, ctx, nullptr); }

    Conv2DAttributes attrs{};
    std::vector<float> in_buf, w_buf, b_buf, out_buf;
    TensorView in, w, bias, out;
    std::vector<TensorView> ins;
    std::unique_ptr<Conv2D> op;
    int64_t OH = 0, OW = 0, icg = 0;
    double work = 0.0;   // FLOPs per call
};

Conv2DI8Case as_i8(const Conv2DCase& c, DataType act) {
    Conv2DI8Case q{};
    q.name = c.name;
    q.N = c.N; q.IC = c.IC; q.IH = c.IH; q.IW = c.IW;
    q.OC = c.OC; q.KH = c.KH; q.KW = c.KW;
    q.SH = c.SH; q.SW = c.SW; q.PH = c.PH; q.PW = c.PW;
    q.DH = c.DH; q.DW = c.DW; q.groups = c.groups;
    q.bias = c.bias;
    q.act = act;
    return q;
}

void run_conv2d_i8(const Conv2DI8Case& c, const std::vector<int>& threads, int iters) {
    I8Op b(c, threads.back());

    std::printf("  %s\n", c.name);
    std::printf("    in[%lld,%lld,%lld,%lld] %s  w[%lld,%lld,%lldx%lld] s8  ->"
                "  out[%lld,%lld,%lld,%lld] %s%s%s%s\n",
                (long long)c.N, (long long)c.IC, (long long)c.IH, (long long)c.IW,
                dtype_name(c.act),
                (long long)c.OC, (long long)b.icg, (long long)c.KH, (long long)c.KW,
                (long long)c.N, (long long)c.OC, (long long)b.OH, (long long)b.OW,
                dtype_name(c.out),
                c.bias ? "  +bias" : "", c.asym_w ? "  asym-w" : "",
                c.act == DataType::f32 || c.act == DataType::f16
                    ? (c.act_q == DataType::u8 ? "  quant->u8" : "  quant->s8") : "");
    std::printf("    icn=%lld ocn=%lld oh_block=%lld oh_blocks=%lld  tasks=N*G*oh_chunks=%lld"
                "  ws=%zuB%s\n",
                (long long)b.plan.icn_block, (long long)b.plan.ocn_block,
                (long long)b.plan.oh_block, (long long)b.plan.num_oh_blocks,
                (long long)(c.N * c.groups * b.plan.oh_chunks), b.plan.workspace_size,
                b.plan.pointwise ? "  pointwise(no col buffer)" : "");

    std::vector<double> gops;
    for (int nt : threads) {
        SimplePool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

        auto compute = [&]() { b.run(use_ctx); };
        for (int i = 0; i < 2; ++i) { compute(); }   // warm-up / page faults
        const double ms = time_ms(compute, iters);
        gops.push_back(b.work / ms / 1e6);
    }

    const double g1 = gops.empty() ? 0.0 : gops[0];
    std::printf("    ");
    for (size_t i = 0; i < threads.size(); ++i) {
        const double speedup = g1 > 0.0 ? gops[i] / g1 : 0.0;
        std::printf(" %2dT %7.1f GOPS (%.2fx)", threads[i], gops[i], speedup);
        if (i + 1 < threads.size()) { std::printf(" |"); }
    }
    std::printf("\n");
}

void bench_conv2d_i8_case(const Conv2DI8Case& c) {
    run_conv2d_i8(c, conv_bench_threads(), 20);
}

/// Interleaved A/B, the same discipline as `run_wino_ab`: within a round the
/// variants alternate A B A B, and the second A is the null control — same op,
/// same binary, same buffers, so an A2/A1 gap is measurement drift and nothing
/// else. Per round the minimum is kept and the median across rounds is
/// reported; the verdict is called real only when it clears the null floor.
/// One interleaved A/B measurement at @p nt threads, in per-call ms.
struct AbResult {
    double a_ms = 0.0, b_ms = 0.0;   ///< median over rounds of the per-round min
    double null_pct = 0.0;           ///< median A2/A1 drift — the noise floor
};

template <class FA, class FB>
AbResult measure_ab(int nt, FA&& a, FB&& b, int iters, int rounds) {
    SimplePool pool(nt);
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

template <class FA, class FB>
void run_ab_pair(const char* title,
                 const char* la, double wa, const char* ua, FA&& a,
                 const char* lb, double wb, const char* ub, FB&& b,
                 const std::vector<int>& threads, int iters, int rounds) {
    std::printf("  %s\n", title);
    for (int nt : threads) {
        const AbResult r = measure_ab(nt, a, b, iters, rounds);
        // B's wall time as a fraction of A's: 0.5x means B costs half the time.
        const double ratio = r.b_ms / r.a_ms;
        const bool real = std::abs(r.a_ms - r.b_ms) / r.a_ms * 100.0 > r.null_pct;

        std::printf("    %2dT  %-6s %7.3f ms (%7.1f %s) | %-6s %7.3f ms (%7.1f %s)"
                    " | %s/%s time %.2fx%s | null %.1f%%\n",
                    nt, la, r.a_ms, wa / r.a_ms / 1e6, ua,
                    lb, r.b_ms, wb / r.b_ms / 1e6, ub,
                    lb, la, ratio, real ? "" : " (within noise)", r.null_pct);
    }
}

/// The headline number: the quantized route against the fp route on the same
/// geometry (u8 activation, which is the deployment shape).
void bench_conv2d_i8_vs_fp(const Conv2DCase& c, int iters, int rounds) {
    const std::vector<int> threads = conv_bench_threads();
    I8Op i8(as_i8(c, DataType::u8), threads.back());
    Fp32Op fp(c);
    run_ab_pair(c.name,
                "fp32", fp.work, "GFLOPS", [&](const ComputeContext& x) { fp.run(x); },
                "int8", i8.work, "GOPS",  [&](const ComputeContext& x) { i8.run(x); },
                threads, iters, rounds);
}

// ============================================================
// Kernel-area / shape sweeps
// ============================================================
//
// The point cases above each pin one shape, and 3x3 dominates them. The one
// thing 3x3 cannot show is what the kernel *area* does to the plan:
// get_ic_block() maps karea -> icn_block (64 at 1x1, 16 through 3x3, 8 through
// 5x5, 2 through 7x7, 1 past that), and a single GEMM call reduces over
// icn_block * karea. The ladder holds that K roughly flat as the kernel grows —
// the K column below reads 64 -> 144 -> 200 -> 98 for 1x1 -> 3x3 -> 5x5 -> 7x7 —
// so the im2col panel handed to one GEMM stays near 144-200 wide through 5x5.
// The 7x7 rung is left at 98: a 49 -> 4 bump measured -3.8% on a 112 plane.
// The 1x1 rows read 64 only because IC is 64 here; they take the plan's
// pointwise route (the pw column), so they never consult get_ic_block() and
// never build a panel — 64 is the whole channel count, not a cap.
//
// What is *not* flat is the traffic: every input channel's karea taps
// must be materialised for every output row, so the im2col write per output row
// is IC * karea and grows 49/9 from 3x3 to 7x7, and the weight panel is re-read
// once per ic-block (IC / icn_block of them). These tables put a number on that.
//
// The tile columns are the plan resolved at the *largest* pool in the row, not
// at 1 thread: oh_chunks and the oh-block shrink are per-pool decisions, so the
// widest pool is the geometry the narrower ones are compared against. blks
// (N*G*oh_blocks) is the hard ceiling on tasks — where blks is far below the
// thread count, the multi-thread columns are partly measuring the pool.
//
// Cells are adaptive — each runs until a wall-clock budget elapses — rather than
// the flat 20 the point cases use: a 7x7 at 112x112 is orders of magnitude more
// work per call than a 1x1 at 7x7, so no single iteration count covers both.
//
// Caveat on the multi-thread columns: SimplePool spawns and joins its workers
// inside every parallel_for, so each call pays thread start-up once. Measured on
// this host that is ~450 us at 20 threads — around a quarter of the f32 3x3 row
// in the first table, and the reason the smallest rows' 8T/20T cells read low.
// The 5x5/7x7 rows do 25/9 and 49/9 of the 3x3's work per call and are far less
// affected; read the small-shape 20T cells as a lower bound on the kernel.

struct ConvSweepShape {
    int64_t N, IC, IH, OC, K;    // square kernel; plane is IH x IW
    int64_t S = 1;               // stride (both axes)
    int64_t groups = 1;
    int64_t pad = -1;            // -1 == "same": ((K-1)*D)/2
    int64_t D = 1;               // dilation (both axes)
    int64_t IW = 0;              // 0 == square (IW = IH); set for wide frames
};

std::vector<int> sweep_threads() {
    return conv_bench_threads();
}

/// One conv problem, built once and reused across pool sizes.
template <typename T>
struct ConvSweepOp {
    std::unique_ptr<Conv2D> op;
    std::vector<T> in_buf, w_buf, out_buf;
    std::vector<TensorView> ins;
    TensorView out;
    int64_t icn, ocn, ohb, blks, tasks;
    bool pw;                     // plan.pointwise — no col buffer, B off the input
    double flops;

    ConvSweepOp(const ConvSweepShape& s, uint64_t seed, int64_t meta_threads) {
        const int64_t ph = s.pad < 0 ? ((s.K - 1) * s.D) / 2 : s.pad;
        const int64_t iw = s.IW > 0 ? s.IW : s.IH;
        const int64_t ishape[] = {s.N, s.IC, s.IH, iw};
        const int64_t wshape[] = {s.OC, s.IC / s.groups, s.K, s.K};

        auto [iv, in] = make_random<T>(ishape, seed);
        auto [wv, w]  = make_random<T>(wshape, seed + 1);
        in_buf = std::move(iv);
        w_buf  = std::move(wv);

        Conv2DAttributes attrs{};
        attrs.kernel_size = {s.K, s.K};
        attrs.stride      = {s.S, s.S};
        attrs.dilation    = {s.D, s.D};
        attrs.padding     = {ph, ph};
        attrs.groups      = s.groups;

        op = Conv2D::create(attrs, Backend::CPU);
        const TensorDesc descs[] = {in.desc(), w.desc()};
        auto out_descs = op->getOutputTensorDesc(descs);
        out_buf.assign(static_cast<size_t>(out_descs[0].numel()), T{});
        out = test::make_planar(out_descs[0], out_buf.data());
        ins = {in, w};

        const cpu::Conv2DPlan plan = cpu::get_conv2d_plan(
            attrs, descs, out_descs, meta_threads, /*use_thread_slots=*/true);
        icn   = plan.icn_block;
        ocn   = plan.ocn_block;
        ohb   = plan.oh_block;
        blks  = s.N * s.groups * plan.num_oh_blocks;
        tasks = s.N * s.groups * plan.oh_chunks;
        pw    = plan.pointwise;

        const int64_t OH = out_descs[0].dims[2], OW = out_descs[0].dims[3];
        flops = 2.0 * static_cast<double>(s.N) * static_cast<double>(s.OC) *
                static_cast<double>(OH) * static_cast<double>(OW) *
                static_cast<double>(s.IC / s.groups) *
                static_cast<double>(s.K) * static_cast<double>(s.K);
    }

    /// GFLOPS over one budgeted block: warm up, then iterate until @p target_ms
    /// of wall time has elapsed.
    double time_budget(const ComputeContext& ctx, double target_ms) {
        for (int i = 0; i < 2; ++i) { op->compute(out, ins, ctx, nullptr); }
        const auto t0 = std::chrono::high_resolution_clock::now();
        int64_t iters = 0;
        double ms = 0.0;
        constexpr int64_t kMaxIters = 20000;   // the floor is a 1x1 at 7x7
        do {
            op->compute(out, ins, ctx, nullptr);
            ++iters;
            ms = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - t0).count();
        } while (ms < target_ms && iters < kMaxIters);
        return flops * static_cast<double>(iters) / (ms * 1e6);
    }
};

void print_conv_sweep_header(const std::vector<int>& threads) {
    std::printf("   %3s %4s %4s %5s %4s %3s %4s %2s %2s | %4s %4s %4s %5s %5s %3s |",
                "N", "IC", "IH", "IW", "OC", "K", "g", "s", "d",
                "icn", "ocn", "ohb", "blks", "tasks", "pw");
    for (int t : threads) {
        const std::string label = std::to_string(t) + "T";
        std::printf(" %7s", label.c_str());
    }
    std::printf("   (GFLOPS)\n");
}

void print_conv_sweep_row(const ConvSweepShape& s, int64_t icn, int64_t ocn, int64_t ohb,
                          int64_t blks, int64_t tasks, bool pw,
                          const std::vector<double>& g) {
    std::printf("   %3lld %4lld %4lld %5lld %4lld %3lld %4lld %2lld %2lld |"
                " %4lld %4lld %4lld %5lld %5lld %3s |",
                static_cast<long long>(s.N), static_cast<long long>(s.IC),
                static_cast<long long>(s.IH),
                static_cast<long long>(s.IW > 0 ? s.IW : s.IH),
                static_cast<long long>(s.OC),
                static_cast<long long>(s.K), static_cast<long long>(s.groups),
                static_cast<long long>(s.S), static_cast<long long>(s.D),
                static_cast<long long>(icn), static_cast<long long>(ocn),
                static_cast<long long>(ohb), static_cast<long long>(blks),
                static_cast<long long>(tasks), pw ? "yes" : "-");
    for (double v : g) { std::printf(" %7.1f", v); }
    std::printf("\n");
}

template <typename T>
void sweep_conv(const char* tag, const std::vector<ConvSweepShape>& shapes,
                const std::vector<int>& threads, double target_ms) {
    const int64_t meta_threads = *std::max_element(threads.begin(), threads.end());
    std::printf("  %s conv sweep  budget=%.0f ms/cell\n", tag, target_ms);
    print_conv_sweep_header(threads);
    uint64_t seed = 7;
    for (const ConvSweepShape& s : shapes) {
        ConvSweepOp<T> g(s, seed, meta_threads);
        seed += 2;
        std::vector<double> gflops;
        for (int nt : threads) {
            SimplePool pool(nt);
            ComputeContext ctx;
            ctx.cpu = pool.cpu;
            const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};
            gflops.push_back(g.time_budget(use_ctx, target_ms));
        }
        print_conv_sweep_row(s, g.icn, g.ocn, g.ohb, g.blks, g.tasks, g.pw, gflops);
    }
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
// Larger kernels — 5x5, 7x7
// ============================================================
//
// karea drives icn_block (16 through 3x3, 8 through 5x5, 2 through 7x7), so
// everything past 3x3 runs on a narrower im2col panel and a deeper ic-block
// walk. The 56x56 cases deliberately share the resnet-block shape above them so
// the rows differ in the kernel and nothing else.

NNOPS_BENCH(conv2d_f32_5x5) {
    // Same in/out shape as conv2d_f32_resnet_block_3x3 — 25/9 the FLOPs, icn 8.
    bench_conv2d_case({"f32 5x5 1x64x56x56 -> 64 x 5x5 s1 p2", 1, 64, 56, 56, 64, 5, 5,
                       /*SH=*/1, /*SW=*/1, /*PH=*/2, /*PW=*/2});
}

NNOPS_BENCH(conv2d_f32_7x7) {
    // 49/9 of the 3x3 at the same shape, and 4x the 5x5 above it. karea 49 puts
    // icn_block at 2: the same output is produced from a panel a quarter as wide
    // as the 5x5's, rewritten four times as many times.
    bench_conv2d_case({"f32 7x7 1x64x56x56 -> 64 x 7x7 s1 p3", 1, 64, 56, 56, 64, 7, 7,
                       /*SH=*/1, /*SW=*/1, /*PH=*/3, /*PW=*/3});
}

NNOPS_BENCH(conv2d_f32_5x5_stride2) {
    // MobileNet-style downsample: 5x5 stride 2 is the largest kernel that
    // actually sits in a cheap block of a real network.
    bench_conv2d_case({"f32 5x5 s2 1x32x112x112 -> 32 x 5x5 s2 p2", 1, 32, 112, 112, 32, 5, 5,
                       /*SH=*/2, /*SW=*/2, /*PH=*/2, /*PW=*/2});
}

NNOPS_BENCH(conv2d_f32_5x5_dilated2) {
    // karea stays 25 (icn_block 8) while the gathered window is 9 wide — the
    // contrast case against a real 9x9, which plans icn_block 1.
    bench_conv2d_case({"f32 5x5 d2 1x64x56x56 -> 64 x 5x5 s1 p2 d2", 1, 64, 56, 56, 64, 5, 5,
                       /*SH=*/1, /*SW=*/1, /*PH=*/2, /*PW=*/2, /*DH=*/2, /*DW=*/2});
}

NNOPS_BENCH(conv2d_f32_5x5_grouped) {
    // The im2col grouped path at 5x5 — pairs with dwconv_f32_5x5 below, which is
    // the dedicated NCHWC8 kernel on the same shape.
    bench_conv2d_case({"f32 depthwise 5x5 1x128x28x28 -> 128 x 5x5 g128 s1 p2",
                       1, 128, 28, 28, 128, 5, 5, /*SH=*/1, /*SW=*/1, /*PH=*/2, /*PW=*/2,
                       /*DH=*/1, /*DW=*/1, /*groups=*/128});
}

NNOPS_BENCH(conv2d_f16_5x5) {
    bench_conv2d_case({"f16 5x5 1x64x56x56 -> 64 x 5x5 s1 p2", 1, 64, 56, 56, 64, 5, 5,
                       /*SH=*/1, /*SW=*/1, /*PH=*/2, /*PW=*/2, /*DH=*/1, /*DW=*/1, /*groups=*/1,
                       /*bias=*/false, /*f16=*/true});
}

NNOPS_BENCH(conv2d_f16_7x7) {
    bench_conv2d_case({"f16 7x7 1x64x56x56 -> 64 x 7x7 s1 p3", 1, 64, 56, 56, 64, 7, 7,
                       /*SH=*/1, /*SW=*/1, /*PH=*/3, /*PW=*/3, /*DH=*/1, /*DW=*/1, /*groups=*/1,
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
// tile and per channel, so the win grows with the channel count: from 32 channels
// up it is ahead of im2col here, and at 8 it is a wash. The 8-channel case is the
// floor of the range this kernel is worth calling for — kept in the sweep so a
// regression in the transforms shows up as a number, not an assumption.

NNOPS_BENCH(conv2d_wino_f32_64ch) {
    bench_wino_case({"f32 wino 1x64x56x56 -> 64 x 3x3 s1 p1", 1, 64, 56, 56, 64});
}

NNOPS_BENCH(conv2d_wino_f32_batch4) {
    // Batch is part of the task grid, not a serial loop around one dispatch, so
    // N=4 should scale with threads the same way N=1 does rather than paying the
    // per-(batch, group) phases and dispatches four times over.
    bench_wino_case({"f32 wino 4x64x56x56 -> 64 x 3x3 s1 p1", 4, 64, 56, 56, 64});
}

NNOPS_BENCH(conv2d_wino_f32_256ch) {
    bench_wino_case({"f32 wino 1x256x28x28 -> 256 x 3x3 s1 p1", 1, 256, 28, 28, 256});
}

NNOPS_BENCH(conv2d_wino_f32_32ch) {
    bench_wino_case({"f32 wino 1x32x56x56 -> 32 x 3x3 s1 p1", 1, 32, 56, 56, 32});
}

NNOPS_BENCH(conv2d_wino_f32_8ch_degenerate) {
    // The floor: at 8 channels the transforms eat the 2.25x multiply saving, so
    // this is the case where Winograd is only at parity with im2col.
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

// ============================================================
// Kernel-area sweeps — what 5x5 and 7x7 cost against 3x3
// ============================================================
//
// Same shapes, read as a table: the icn column is the plan's answer to karea,
// and the gflops columns are what that answer buys.

NNOPS_BENCH(conv2d_f32_kernel_sweep) {
    // One shape (1x64x56x56 -> 64), one padding rule, only the kernel moves.
    // Read icn against K: the plan trades icn_block away roughly in step with
    // karea, so the K of one GEMM call stays in the 144-200 band while the FLOPs
    // above it scale with karea^2. The dilated 3x3 is the outlier — karea 9
    // (icn 16) with a 5-wide gather window, so it plans like a 3x3 and gathers
    // like a 5x5; only the `d` column tells it apart from the row above.
    sweep_conv<float>("f32", {
        {1, 64, 56, 64, 1},                        // pointwise: karea  1, icn 64
        {1, 64, 56, 64, 3},                        // karea  9, icn 16
        {1, 64, 56, 64, 3, 1, 1, -1, /*D=*/2},     // dilated 3x3: karea 9, window 5
        {1, 64, 56, 64, 5},                        // karea 25, icn 8
        {1, 64, 56, 64, 7},                        // karea 49, icn 2
    }, sweep_threads(), 80.0);
}

NNOPS_BENCH(conv2d_f32_large_kernel_shape_sweep) {
    // Kernel area against plane size. At 28x28 the whole plane is 7 oh-blocks,
    // so the tasks ceiling (blks) is well under the pool; at 112x112 it is not.
    sweep_conv<float>("f32", {
        {1, 64,  28, 64, 3},
        {1, 64,  28, 64, 5},
        {1, 64,  28, 64, 7},
        {1, 64, 112, 64, 3},
        {1, 64, 112, 64, 5},
        {1, 64, 112, 64, 7},
    }, sweep_threads(), 80.0);
}

NNOPS_BENCH(conv2d_f32_5x5_group_sweep) {
    // Groups against a fixed 5x5: raising g shrinks ic_per_group, which min()s
    // into icn_block, so g = 16 already sits at icn_block 1 — the same rank-1
    // panel a 9x9 reaches with g = 1. Task count rises with g throughout, so
    // this isolates tile size from pool fill.
    sweep_conv<float>("f32", {
        {1, 64, 28, 64, 5, 1, /*groups=*/1},
        {1, 64, 28, 64, 5, 1, /*groups=*/4},
        {1, 64, 28, 64, 5, 1, /*groups=*/16},
        {1, 64, 28, 64, 5, 1, /*groups=*/64},   // depthwise 5x5
    }, sweep_threads(), 80.0);
}

NNOPS_BENCH(conv2d_f16_kernel_sweep) {
    sweep_conv<half>("f16", {
        {1, 64, 56, 64, 1},
        {1, 64, 56, 64, 3},
        {1, 64, 56, 64, 5},
        {1, 64, 56, 64, 7},
    }, sweep_threads(), 80.0);
}

// ============================================================
// im2col parity set vs MLAS — single thread
// ============================================================
//
// The first twelve rows are the shape set onnxruntime's MLAS convolution
// benchmark runs (onnxruntime/test/mlas/bench/bench_sconv.cpp: the ResNet50 set
// plus the TeamsModel depthwise layer), expressed in this harness's
// (N, IC, IH, OC, K, stride, groups, pad, dilation) form. Each one is therefore
// a head-to-head with an MLAS `MlasConv` number on the identical problem:
//
//   row                              MLAS parameters (N, G, Cpg, Fpg, I, K, P, S)
//   1x3x224x224 -> 64 x 7x7 s2       1, 1, 3,   64,  224, 7x7, 3, 2
//   1x64x56x56 -> 64 x 3x3           1, 1, 64,  64,  56,  3x3, 1, 1
//   1x64x56x56 -> 256 x 1x1          1, 1, 64,  256, 56,  1x1, 0, 1
//   1x128x28x28 -> 512 x 3x3 s2      1, 1, 128, 512, 28,  3x3, 1, 2
//   1x256x14x14 -> 1024 x 1x1        1, 1, 256, 1024, 14, 1x1, 0, 1
//   1x512x7x7 -> 2048 x 1x1          1, 1, 512, 2048, 7,  1x1, 0, 1
//   1x64x56x56 -> 64 x 3x3 g64       1, 64, 1,  1,   56,  3x3, 1, 1
//   1x24x72x72 -> 12 x 3x3 g2 p0     1, 2, 12,  6,   72,  3x3, 0, 1
//   1x256x378x378 -> 256 3x3 g256    1, 256, 1, 1,   378, 3x3, 1, 1
//   8x64x56x56 -> 64 x 3x3           8, 1, 64,  64,  56,  3x3, 1, 1
//   8x64x56x56 -> 256 x 1x1          8, 1, 64,  256, 56,  1x1, 0, 1
//   1x256x224x224 -> 256 x 3x3       1, 1, 256, 256, 224, 3x3, 1, 1
//
// The rows past them are the cases the MLAS set does not reach: karea 81 (the
// plan's icn 1 floor), a small-plane depthwise layer, a depthwise 5x5, a
// dilated 5x5 on a 112 plane, and a grouped 3x3 whose output is 4x the input.
//
// Single thread only: at one thread every thread-count knob disappears from
// both harnesses — no pool geometry, no per-call thread creation (this harness's
// SimplePool spawns its workers inside every parallel_for, which at 8T/20T costs
// more than the small rows below take in total). The rate printed here is the
// kernel, and the kernel is the only thing the two implementations can be
// compared on without also measuring the harnesses.
//
// The 1x1 rows are the exception to "im2col" in the section title: at karea 1
// with unit stride and zero padding both sides run the GEMM straight off the
// input tensor (the plan's `pointwise` flag; MLAS takes
// MlasConvAlgorithmGemmDirect), so those rows compare the GEMM underneath. They
// are the pw=yes rows, and note what the icn column reads there: IC itself, not
// the 64 the 1x1 rung of get_ic_block() would give — with the col buffer gone
// the whole group fits in a single K pass.

NNOPS_BENCH(conv2d_f32_im2col_parity_1t) {
    sweep_conv<float>("f32 im2col parity", {
        // --- MLAS ResNet50 + TeamsModel set ---
        {1,   3, 224,   64, 7, /*S=*/2},                       // resnet50 conv1 7x7 s2
        {1,  64,  56,   64, 3},                                // resnet50 3x3
        {1,  64,  56,  256, 1},                                // resnet50 1x1 pw
        {1, 128,  28,  512, 3, /*S=*/2},                       // resnet50 3x3 s2
        {1, 256,  14, 1024, 1},                                // resnet50 1x1 pw (14x14)
        {1, 512,   7, 2048, 1},                                // resnet50 1x1 pw (7x7)
        {1,  64,  56,   64, 3, 1, /*groups=*/64},              // mobilenet depthwise
        {1,  24,  72,   12, 3, 1, /*groups=*/2,  /*pad=*/0},   // grouped g2, no pad
        {1, 256, 378,  256, 3, 1, /*groups=*/256},             // teams depthwise 378x378
        {8,  64,  56,   64, 3},                                // batch 8, 3x3
        {8,  64,  56,  256, 1},                                // batch 8, 1x1 pw
        {1, 256, 224,  256, 3},                                // 224 plane, 256x256
        // --- past the MLAS set ---
        {1,  64,  56,   64, 9},                                // karea 81 -> icn 1
        {1, 256,  28,  256, 3, 1, /*groups=*/256},             // depthwise on a small plane
        {1, 128,  28,  128, 5, 1, /*groups=*/128},             // depthwise 5x5
        {1,  64, 112,   64, 5, 1, /*groups=*/1, /*pad=*/4, /*D=*/2},  // dilated 5x5, 112 plane
        {1,  64,  28,  256, 3, 1, /*groups=*/4},               // grouped, OC = 4x IC
    }, {1}, /*target_ms=*/300.0);
}

// ============================================================
// Classic classification backbones — full layer stacks
// ============================================================
//
// The parity set above measures single layers at one thread. These walk the
// whole stack of the three backbones people actually profile, one row per
// distinct layer shape in execution order (N = 1, the training/inference batch
// every published number uses). Rows that repeat inside a stage appear twice
// only in VGG, where the depth of the stage is the point: block3 has three
// 256-channel convs, blocks 4 and 5 have three each in VGG-16.
//
// Layer shapes follow torchvision's resnet50/resnet18 and vgg16 weight shapes;
// stride-2 is on the first 1x1 of every ResNet stage (the bottleneck
// downsample), as in the reference implementation.

NNOPS_BENCH(conv2d_f32_resnet50_stack) {
    sweep_conv<float>("f32 resnet50", {
        {1,    3, 224,   64, 7, /*S=*/2},                      // conv1
        {1,   64,  56,   64, 1},                               // conv2_1 1x1
        {1,   64,  56,   64, 3},                               // conv2_1 3x3
        {1,   64,  56,  256, 1},                               // conv2_1 1x1 expand
        {1,  256,  56,  128, 1, /*S=*/2},                      // conv3_1 1x1 downsample
        {1,  128,  28,  128, 3},                               // conv3_1 3x3
        {1,  128,  28,  512, 1},                               // conv3_1 1x1 expand
        {1,  512,  28,  256, 1, /*S=*/2},                      // conv4_1 1x1 downsample
        {1,  256,  14,  256, 3},                               // conv4_1 3x3
        {1,  256,  14, 1024, 1},                               // conv4_1 1x1 expand
        {1, 1024,  14,  512, 1, /*S=*/2},                      // conv5_1 1x1 downsample
        {1,  512,   7,  512, 3},                               // conv5_1 3x3
        {1,  512,   7, 2048, 1},                               // conv5_1 1x1 expand
    }, sweep_threads(), 250.0);
}

NNOPS_BENCH(conv2d_f32_resnet18_stack) {
    sweep_conv<float>("f32 resnet18", {
        {1,   3, 224,  64, 7, /*S=*/2},                        // conv1
        {1,  64,  56,  64, 3},                                 // layer1
        {1,  64,  56, 128, 3, /*S=*/2},                        // layer2 downsample
        {1, 128,  28, 128, 3},                                 // layer2
        {1, 128,  28, 256, 3, /*S=*/2},                        // layer3 downsample
        {1, 256,  14, 256, 3},                                 // layer3
        {1, 256,  14, 512, 3, /*S=*/2},                        // layer4 downsample
        {1, 512,   7, 512, 3},                                 // layer4
    }, sweep_threads(), 250.0);
}

NNOPS_BENCH(conv2d_f32_vgg16_stack) {
    sweep_conv<float>("f32 vgg16", {
        {1,   3, 224,  64, 3},                                 // block1
        {1,  64, 224,  64, 3},
        {1,  64, 112, 128, 3},                                 // block2
        {1, 128, 112, 128, 3},
        {1, 128,  56, 256, 3},                                 // block3 (x3)
        {1, 256,  56, 256, 3},
        {1, 256,  56, 256, 3},
        {1, 256,  28, 512, 3},                                 // block4 (x3)
        {1, 512,  28, 512, 3},
        {1, 512,  28, 512, 3},
        {1, 512,  14, 512, 3},                                 // block5 (x3)
        {1, 512,  14, 512, 3},
        {1, 512,  14, 512, 3},
    }, sweep_threads(), 250.0);
}

// ============================================================
// Wide frames — ResNet / VGG at 1280 x 720 and 1280 x 960
// ============================================================
//
// Every stack above is a 224-square plane. A frame is not square: at 1280 wide
// the N side of the GEMM spans many nr_max panels on its own, and an oh-block
// of 4 rows is a far smaller share of the plane, so the multi-block and
// multi-chunk walk is much deeper than on any 224 row. These tables are the
// front stages of each model at that frame size — the layers whose plane is
// large enough for the width to show. Deeper stages sit at 1/8, 1/16 and 1/32
// of the frame and are the same shapes the 224 stacks already cover. Input and
// output tensors here are up to ~236 MB per side (1x64x720x1280 f32), so this
// is the one table that measures the memory system as much as the kernel.

NNOPS_BENCH(conv2d_f32_720p_stack) {
    sweep_conv<float>("f32 1280x720", {
        // ResNet50 front: the stem drops to 360x640, conv2_x runs at 180x320.
        {1,   3, 720,  64, 7, /*S=*/2, 1, -1, 1, 1280},   // conv1 7x7 s2
        {1,  64, 180,  64, 3, 1, 1, -1, 1, 320},           // conv2_x 3x3
        {1,  64, 180, 256, 1, 1, 1,  0, 1, 320},           // conv2_x 1x1 expand
        // VGG16 front: block1 is full resolution, block2 half of it.
        {1,   3, 720,  64, 3, 1, 1, -1, 1, 1280},          // block1 a
        {1,  64, 720,  64, 3, 1, 1, -1, 1, 1280},          // block1 b
        {1,  64, 360, 128, 3, 1, 1, -1, 1, 640},           // block2 a
        {1, 128, 360, 128, 3, 1, 1, -1, 1, 640},           // block2 b
    }, sweep_threads(), 250.0);
}

NNOPS_BENCH(conv2d_f32_960p_stack) {
    sweep_conv<float>("f32 1280x960", {
        {1,   3, 960,  64, 7, /*S=*/2, 1, -1, 1, 1280},   // conv1 7x7 s2
        {1,  64, 240,  64, 3, 1, 1, -1, 1, 320},           // conv2_x 3x3
        {1,  64, 240, 256, 1, 1, 1,  0, 1, 320},           // conv2_x 1x1 expand
        {1,   3, 960,  64, 3, 1, 1, -1, 1, 1280},          // block1 a
        {1,  64, 960,  64, 3, 1, 1, -1, 1, 1280},          // block1 b
        {1,  64, 480, 128, 3, 1, 1, -1, 1, 640},           // block2 a
        {1, 128, 480, 128, 3, 1, 1, -1, 1, 640},           // block2 b
    }, sweep_threads(), 250.0);
}

NNOPS_BENCH(conv2d_f16_resnet50_stack) {
    sweep_conv<half>("f16 resnet50", {
        {1,    3, 224,   64, 7, /*S=*/2},
        {1,   64,  56,   64, 1},
        {1,   64,  56,   64, 3},
        {1,   64,  56,  256, 1},
        {1,  256,  56,  128, 1, /*S=*/2},
        {1,  128,  28,  128, 3},
        {1,  128,  28,  512, 1},
        {1,  512,  28,  256, 1, /*S=*/2},
        {1,  256,  14,  256, 3},
        {1,  256,  14, 1024, 1},
        {1, 1024,  14,  512, 1, /*S=*/2},
        {1,  512,   7,  512, 3},
        {1,  512,   7, 2048, 1},
    }, sweep_threads(), 250.0);
}

NNOPS_BENCH(conv2d_f16_vgg16_stack) {
    sweep_conv<half>("f16 vgg16", {
        {1,   3, 224,  64, 3},
        {1,  64, 224,  64, 3},
        {1,  64, 112, 128, 3},
        {1, 128, 112, 128, 3},
        {1, 128,  56, 256, 3},
        {1, 256,  56, 256, 3},
        {1, 256,  56, 256, 3},
        {1, 256,  28, 512, 3},
        {1, 512,  28, 512, 3},
        {1, 512,  28, 512, 3},
        {1, 512,  14, 512, 3},
        {1, 512,  14, 512, 3},
        {1, 512,  14, 512, 3},
    }, sweep_threads(), 250.0);
}

// ============================================================
// Quantized Conv2D — s8 weight × u8/s8/f32/f16 activation → fp output
// ============================================================

NNOPS_BENCH(conv2d_i8_u8_3x3) {
    // Pre-quantized activation — the deployment shape: im2col is a byte copy.
    bench_conv2d_i8_case({"i8 u8-act 1x64x56x56 -> 64 x 3x3 s1 p1 +bias",
                          1, 64, 56, 56, 64, 3, 3,
                          /*SH=*/1, /*SW=*/1, /*PH=*/1, /*PW=*/1,
                          /*DH=*/1, /*DW=*/1, /*groups=*/1, /*bias=*/true});
}

NNOPS_BENCH(conv2d_i8_f32act_3x3) {
    // Same geometry, fp activation: quantized inside the im2col gather.
    bench_conv2d_i8_case({"i8 f32-act 1x64x56x56 -> 64 x 3x3 s1 p1 +bias",
                          1, 64, 56, 56, 64, 3, 3,
                          /*SH=*/1, /*SW=*/1, /*PH=*/1, /*PW=*/1,
                          /*DH=*/1, /*DW=*/1, /*groups=*/1, /*bias=*/true,
                          DataType::f32});
}

NNOPS_BENCH(conv2d_i8_pointwise_1x1) {
    // 1x1 with a pre-quantized activation takes the pointwise route (no col).
    bench_conv2d_i8_case({"i8 u8-act pointwise 1x256x28x28 -> 256 x 1x1 +bias",
                          1, 256, 28, 28, 256, 1, 1,
                          /*SH=*/1, /*SW=*/1, /*PH=*/0, /*PW=*/0,
                          /*DH=*/1, /*DW=*/1, /*groups=*/1, /*bias=*/true});
}

NNOPS_BENCH(conv2d_i8_mid_3x3) {
    bench_conv2d_i8_case({"i8 u8-act 1x128x28x28 -> 128 x 3x3 s1 p1 +bias",
                          1, 128, 28, 28, 128, 3, 3,
                          /*SH=*/1, /*SW=*/1, /*PH=*/1, /*PW=*/1,
                          /*DH=*/1, /*DW=*/1, /*groups=*/1, /*bias=*/true});
}

NNOPS_BENCH(conv2d_i8_groups4) {
    // Grouped: K per group is IC/G·karea, so the ic-block count collapses.
    bench_conv2d_i8_case({"i8 u8-act grouped 1x128x56x56 -> 128 x 3x3 g4 +bias",
                          1, 128, 56, 56, 128, 3, 3,
                          /*SH=*/1, /*SW=*/1, /*PH=*/1, /*PW=*/1,
                          /*DH=*/1, /*DW=*/1, /*groups=*/4, /*bias=*/true});
}

NNOPS_BENCH(conv2d_i8_f16_act_out) {
    // f16 in, f16 out: the quantize-in-gather path with an f16 epilogue.
    bench_conv2d_i8_case({"i8 f16-act/f16-out 1x64x56x56 -> 64 x 3x3 s1 p1 +bias",
                          1, 64, 56, 56, 64, 3, 3,
                          /*SH=*/1, /*SW=*/1, /*PH=*/1, /*PW=*/1,
                          /*DH=*/1, /*DW=*/1, /*groups=*/1, /*bias=*/true,
                          DataType::f16, DataType::f16});
}

NNOPS_BENCH(conv2d_i8_vs_fp_ab) {
    // The headline number: how much of the fp wall time the quantized route
    // recovers, same geometry, same useful work.
    bench_conv2d_i8_vs_fp({"int8 vs fp32 1x64x56x56 -> 64 x 3x3 s1 p1 +bias",
                           1, 64, 56, 56, 64, 3, 3,
                           /*SH=*/1, /*SW=*/1, /*PH=*/1, /*PW=*/1,
                           /*DH=*/1, /*DW=*/1, /*groups=*/1, /*bias=*/true},
                          10, 5);
}

NNOPS_BENCH(conv2d_i8_fp_act_overhead_ab) {
    // The price of quantizing on the fly: identical MACs, a conversion per
    // gathered element. Pointwise is where it should hurt most — the 1x1 case
    // turns im2col into a pure quantizing copy.
    const int iters = 10, rounds = 5;
    const std::vector<int> threads = conv_bench_threads();

    auto bench_one = [&](const Conv2DI8Case& u8c) {
        Conv2DI8Case f32c = u8c;
        f32c.act = DataType::f32;
        I8Op a(u8c, threads.back());
        I8Op b(f32c, threads.back());
        run_ab_pair(u8c.name,
                    "u8", a.work, "GOPS", [&](const ComputeContext& x) { a.run(x); },
                    "f32", b.work, "GOPS", [&](const ComputeContext& x) { b.run(x); },
                    threads, iters, rounds);
    };

    bench_one({"i8 fp-act overhead 1x256x28x28 -> 256 x 1x1 +bias",
               1, 256, 28, 28, 256, 1, 1,
               /*SH=*/1, /*SW=*/1, /*PH=*/0, /*PW=*/0,
               /*DH=*/1, /*DW=*/1, /*groups=*/1, /*bias=*/true});
    bench_one({"i8 fp-act overhead 1x64x56x56 -> 64 x 3x3 s1 p1 +bias",
               1, 64, 56, 56, 64, 3, 3,
               /*SH=*/1, /*SW=*/1, /*PH=*/1, /*PW=*/1,
               /*DH=*/1, /*DW=*/1, /*groups=*/1, /*bias=*/true});
}

/// The layer ladder the comparison sweeps walk: a ResNet-ish stack plus a
/// grouped and a batched case, every one with bias.
const std::vector<Conv2DCase>& conv_sweep_shapes() {
    static const std::vector<Conv2DCase> shapes = {
        {"1x3x224x224  -> 64  7x7 s2 p3",  1,   3, 224, 224,  64, 7, 7,
         2, 2, 3, 3, 1, 1, 1, true},
        {"1x64x56x56   -> 64  3x3 p1",     1,  64,  56,  56,  64, 3, 3,
         1, 1, 1, 1, 1, 1, 1, true},
        {"1x64x56x56   -> 256 1x1",        1,  64,  56,  56, 256, 1, 1,
         1, 1, 0, 0, 1, 1, 1, true},
        {"1x128x28x28  -> 128 3x3 p1",     1, 128,  28,  28, 128, 3, 3,
         1, 1, 1, 1, 1, 1, 1, true},
        {"1x256x28x28  -> 256 1x1",        1, 256,  28,  28, 256, 1, 1,
         1, 1, 0, 0, 1, 1, 1, true},
        {"1x256x14x14  -> 256 3x3 p1",     1, 256,  14,  14, 256, 3, 3,
         1, 1, 1, 1, 1, 1, 1, true},
        {"1x512x14x14  -> 512 1x1",        1, 512,  14,  14, 512, 1, 1,
         1, 1, 0, 0, 1, 1, 1, true},
        {"1x128x56x56  -> 128 3x3 g4 p1",  1, 128,  56,  56, 128, 3, 3,
         1, 1, 1, 1, 1, 1, 4, true},
        {"4x64x28x28   -> 128 3x3 p1",     4,  64,  28,  28, 128, 3, 3,
         1, 1, 1, 1, 1, 1, 1, true},
    };
    return shapes;
}

NNOPS_BENCH(conv2d_i8_vs_fp_sweep) {
    // The two routes side by side over one network's layer ladder. Both columns
    // count the same 2·N·OC·OH·OW·(IC/G)·KH·KW MACs, so GFLOPS and GOPS are
    // directly comparable, and the last column is B's wall time over A's.
    // Every row is an interleaved A/B with a null control, so a row that lands
    // inside the noise floor says so instead of reporting drift as a win.
    const std::vector<Conv2DCase>& shapes = conv_sweep_shapes();

    const int iters = 10, rounds = 3;
    const std::vector<int> threads = conv_bench_threads();
    for (int nt : threads) {
        std::printf("\n  %d thread%s — fp32 = im2col+GEMM, int8 = s8 weight x u8 activation\n",
                    nt, nt == 1 ? "" : "s");
        std::printf("    %-31s %7s %8s | %7s %8s | %6s | %s\n",
                    "case", "fp32 ms", "GFLOPS", "int8 ms", "GOPS", "int8/fp32", "null");
        double tot_a = 0.0, tot_b = 0.0;
        for (const auto& c : shapes) {
            I8Op i8(as_i8(c, DataType::u8), nt);
            Fp32Op fp(c);
            const AbResult r = measure_ab(nt,
                                          [&](const ComputeContext& x) { fp.run(x); },
                                          [&](const ComputeContext& x) { i8.run(x); },
                                          iters, rounds);
            tot_a += r.a_ms;
            tot_b += r.b_ms;
            const bool real = std::abs(r.a_ms - r.b_ms) / r.a_ms * 100.0 > r.null_pct;
            std::printf("    %-31s %7.3f %8.1f | %7.3f %8.1f | %5.2fx | %4.1f%%%s\n",
                        c.name, r.a_ms, fp.work / r.a_ms / 1e6,
                        r.b_ms, i8.work / r.b_ms / 1e6,
                        r.b_ms / r.a_ms, r.null_pct, real ? "" : " (noise)");
        }
        // Sum of the per-layer times: the speedup a straight fp→int8 swap of the
        // whole stack would deliver, unweighted by how often each layer runs.
        std::printf("    %-31s %7.3f %8s | %7.3f %8s | %5.2fx |\n",
                    "sum of the layers", tot_a, "", tot_b, "", tot_b / tot_a);
    }
}

NNOPS_BENCH(conv2d_i8_fpact_vs_fp_sweep) {
    // In-kernel quantization, priced against fp32: the same fp32 activation goes
    // through the int8 route, which quantizes it into a byte plane up front.
    // The fp32 column is the alternative, so `f32act/fp32` is the answer to "I
    // cannot pre-quantize my tensors — is this route still worth it". The `u8`
    // column is what pre-quantizing would add on top; `f32/u8` is therefore the
    // price of quantizing on the fly.
    //
    // Two interleaved pairs per row (fp32 vs f32act, u8 vs f32act) rather than
    // one three-way, so each ratio is measured against a baseline carrying the
    // same drift.
    const std::vector<Conv2DCase>& shapes = conv_sweep_shapes();

    const int iters = 10, rounds = 3;
    const std::vector<int> threads = conv_bench_threads();
    for (int nt : threads) {
        std::printf("\n  %d thread%s — fp32 route vs the int8 route with an f32 activation,"
                    " and with a pre-quantized u8 one\n", nt, nt == 1 ? "" : "s");
        std::printf("    %-31s %7s %8s | %8s %8s | %s | %7s %8s | %s | %s %s\n",
                    "case", "fp32 ms", "GFLOPS", "f32act ms", "GOPS",
                    "f32act/fp32", "u8 ms", "GOPS", "f32act/u8", "plane KB", "arena KB");
        double tot_a = 0.0, tot_b = 0.0, tot_u = 0.0;
        for (const auto& c : shapes) {
            I8Op u8op(as_i8(c, DataType::u8), nt);
            I8Op fop(as_i8(c, DataType::f32), nt);
            Fp32Op fp(c);

            const AbResult rf = measure_ab(nt,
                                           [&](const ComputeContext& x) { fp.run(x); },
                                           [&](const ComputeContext& x) { fop.run(x); },
                                           iters, rounds);
            const AbResult ru = measure_ab(nt,
                                           [&](const ComputeContext& x) { u8op.run(x); },
                                           [&](const ComputeContext& x) { fop.run(x); },
                                           iters, rounds);
            tot_a += rf.a_ms;
            tot_b += rf.b_ms;
            tot_u += ru.a_ms;
            const bool real = std::abs(rf.a_ms - rf.b_ms) / rf.a_ms * 100.0 > rf.null_pct;
            std::printf("    %-31s %7.3f %8.1f | %8.3f %8.1f | %9.2fx | %7.3f %8.1f | %8.2fx%s"
                        " | %6lld %7lld\n",
                        c.name, rf.a_ms, fp.work / rf.a_ms / 1e6,
                        rf.b_ms, fop.work / rf.b_ms / 1e6, rf.b_ms / rf.a_ms,
                        ru.a_ms, u8op.work / ru.a_ms / 1e6, ru.b_ms / ru.a_ms,
                        real ? "" : " (noise)",
                        (long long)(fop.plan.quant_size / 1024),
                        (long long)(fop.plan.workspace_size / 1024));
        }
        std::printf("    %-31s %7.3f %8s | %8.3f %8s | %9.2fx | %7.3f %8s | %8.2fx |\n",
                    "sum of the layers", tot_a, "", tot_b, "", tot_b / tot_a,
                    tot_u, "", tot_b / tot_u);
    }
}