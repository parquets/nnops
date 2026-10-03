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

// ============================================================
// DepthwiseConv — dedicated NCHWC8 kernel
// ============================================================

NNOPS_BENCH(dwconv_f32_3x3) {
    // Same shape as the im2col depthwise case above — direct head-to-head.
    bench_depthwise_case({"f32 dwconv 1x256x28x28 -> 3x3 s1 p1", 1, 256, 28, 28, 3, 3});
}

NNOPS_BENCH(dwconv_f32_3x3_large) {
    bench_depthwise_case({"f32 dwconv 1x256x56x56 -> 3x3 s1 p1", 1, 256, 56, 56, 3, 3});
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
    bench_depthwise_case({"f32 dwconv batch 8x128x28x28 -> 3x3 s1 p1", 8, 128, 28, 28, 3, 3});
}