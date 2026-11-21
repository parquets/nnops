/// @file bench_aux_ops.cpp
/// @brief The generic CPU operators outside conv / matmul / attention / rnn:
///        data movement (concat, split, tile/expand, pad, gather, scatter,
///        flatten), the broadcast element-wise family, spatial resize and
///        sampling, the norm that is not a row norm, and a model-shaped
///        reduction/selection tail (topk, argmax, quantize/dequantize).
///
/// Every row moved to a *block* copy or to the thread pool: the shapes here are
/// the ones a graph actually runs (a channel concat, a QKV split, the gqa
/// repeat, an embedding gather, a vocabulary top-k), and the arms are 1T versus
/// the pool, since that is where the change shows.
///
/// `NNOPS_BENCH_THREADS=1,4` pins the ladder; unset runs 1T and all cores.

#include "nnops/ops/argminmax.hpp"
#include "nnops/ops/concat.hpp"
#include "nnops/ops/eltwise.hpp"
#include "nnops/ops/flatten.hpp"
#include "nnops/ops/gather.hpp"
#include "nnops/ops/norm.hpp"
#include "nnops/ops/pad.hpp"
#include "nnops/ops/quant_linear.hpp"
#include "nnops/ops/resize.hpp"
#include "nnops/ops/scatter.hpp"
#include "nnops/ops/split.hpp"
#include "nnops/ops/tile.hpp"
#include "nnops/ops/topk.hpp"
#include "nnops/ops/where.hpp"
#include "common/bench_harness.hpp"
#include "common/cpu_pool.hpp"
#include "common/test_helpers.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>
#include <thread>
#include <vector>

using namespace nnops;

namespace {

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

template <class Fn> double time_ms(Fn&& fn, int iters) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) { fn(); }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / std::max(iters, 1);
}

double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
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

/// Elements actually moved by the op: the throughput column.
struct Row {
    const char* name;
    int64_t bytes;
    int iters;
};

/// Run one arm of one row and print it: `1T  0.123 ms  45.6 GB/s`.
template <class Fn>
void arm(const Row& row, int nt, Fn&& run) {
    BenchPool pool(nt);
    ComputeContext ctx;
    ctx.cpu = pool.cpu;
    const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};
    (void)use_ctx;

    std::vector<double> times;
    for (int round = 0; round < 3; ++round) {
        times.push_back(time_ms([&] { run(use_ctx); }, row.iters));
    }
    const double ms = median_of(times);
    const double gbs = static_cast<double>(row.bytes) / (ms * 1.0e6);
    std::printf("  %-34s %2dT %8.3f ms %7.1f GB/s\n", row.name, nt, ms, gbs);
}

template <class Fn>
void row_all_arms(const Row& row, const std::vector<int>& threads, Fn&& run) {
    for (int nt : threads) { arm(row, nt, run); }
}

/// A channel-last [N,H,W,C] f32 view over `buf` with the natural pitch.
TensorView nhwc(std::vector<float>& buf, int64_t N, int64_t H, int64_t W, int64_t C) {
    const int64_t shape[] = {N, H, W, C};
    return TensorView(std::span<const int64_t>(shape, 4), DataType::f32, buf.data(),
                      TensorLayout::NHWC);
}

std::vector<float> random_f32(int64_t n, uint32_t seed) {
    std::vector<float> v(static_cast<size_t>(n));
    uint32_t s = seed;
    for (int64_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        v[static_cast<size_t>(i)] = (static_cast<float>(s >> 9) * (1.0f / 8388608.0f)) - 1.0f;
    }
    return v;
}

} // namespace

NNOPS_BENCH(concat_split) {
    std::printf("--- concat / split (NHWC, block copy + pool) ---\n");
    const std::vector<int> threads = bench_threads();

    // concat on C: [1,56,56,256] + [1,56,56,256] -> [1,56,56,512]
    {
        constexpr int64_t N = 1, H = 56, W = 56, C = 256;
        std::vector<float> a = random_f32(N * H * W * C, 1);
        std::vector<float> b = random_f32(N * H * W * C, 2);
        std::vector<float> o(static_cast<size_t>(2 * N * H * W * C));
        TensorView va = nhwc(a, N, H, W, C);
        TensorView vb = nhwc(b, N, H, W, C);
        const int64_t oshape[] = {N, H, W, 2 * C};
        TensorView vo(std::span<const int64_t>(oshape, 4), DataType::f32, o.data(),
                      TensorLayout::NHWC);
        ConcatAttributes attrs;
        attrs.axis = -1;
        auto op = Concat::create(attrs, Backend::CPU);
        const TensorView ins[] = {va, vb};
        const Row row{"concat C 2x[1,56,56,256]", 2 * N * H * W * C * 4 * 2, 20};
        row_all_arms(row, threads, [&](const ComputeContext& ctx) { op->compute(vo, ins, ctx); });
    }

    // split on C: [1,56,56,768] -> 3 x [1,56,56,256] (a QKV projection output)
    {
        constexpr int64_t N = 1, H = 56, W = 56, C = 768;
        std::vector<float> a = random_f32(N * H * W * C, 3);
        std::vector<float> o(static_cast<size_t>(N * H * W * C));
        TensorView va = nhwc(a, N, H, W, C);
        const int64_t sshape[] = {N, H, W, C / 3};
        TensorView v0(std::span<const int64_t>(sshape, 4), DataType::f32, o.data(),
                      TensorLayout::NHWC);
        TensorView v1(std::span<const int64_t>(sshape, 4), DataType::f32, o.data() + (N * H * W * C) / 3,
                      TensorLayout::NHWC);
        TensorView v2(std::span<const int64_t>(sshape, 4), DataType::f32, o.data() + 2 * (N * H * W * C) / 3,
                      TensorLayout::NHWC);
        SplitAttributes attrs;
        attrs.axis = -1;
        attrs.split = {C / 3, C / 3, C / 3};
        auto op = Split::create(attrs, Backend::CPU);
        const TensorView ins[] = {va};
        TensorView outs[] = {v0, v1, v2};
        const Row row{"split C [1,56,56,768] x3", N * H * W * C * 4 * 2, 20};
        row_all_arms(row, threads,
                     [&](const ComputeContext& ctx) { op->compute(outs, ins, ctx); });
    }
}

NNOPS_BENCH(tile_expand_pad) {
    std::printf("--- tile / expand / pad (block copy + pool) ---\n");
    const std::vector<int> threads = bench_threads();

    // expand: the gqa repeat [1,8,1,128] -> [1,8,2048,128]
    {
        constexpr int64_t B = 1, Hh = 8, S = 2048, D = 128;
        std::vector<float> a = random_f32(B * Hh * D, 4);
        std::vector<float> o(static_cast<size_t>(B * Hh * S * D));
        const int64_t ishape[] = {B, Hh, 1, D};
        TensorView va(std::span<const int64_t>(ishape, 4), DataType::f32, a.data());
        const int64_t oshape[] = {B, Hh, S, D};
        TensorView vo(std::span<const int64_t>(oshape, 4), DataType::f32, o.data());
        ExpandAttributes attrs;
        auto op = Expand::create(attrs, Backend::CPU);
        const TensorView ins[] = {va};
        const Row row{"expand kv [1,8,1,128]->x2048", B * Hh * S * D * 4 * 2, 10};
        row_all_arms(row, threads, [&](const ComputeContext& ctx) { op->compute(vo, ins, ctx); });
    }

    // tile: [1,56,56,64] x2 on H
    {
        constexpr int64_t N = 1, H = 56, W = 56, C = 64;
        std::vector<float> a = random_f32(N * H * W * C, 5);
        std::vector<float> o(static_cast<size_t>(2 * N * H * W * C));
        TensorView va = nhwc(a, N, H, W, C);
        const int64_t oshape[] = {N, 2 * H, W, C};
        TensorView vo(std::span<const int64_t>(oshape, 4), DataType::f32, o.data(),
                      TensorLayout::NHWC);
        TileAttributes attrs;
        attrs.repeats = {1, 2, 1, 1};
        auto op = Tile::create(attrs, Backend::CPU);
        const TensorView ins[] = {va};
        const Row row{"tile H x2 [1,56,56,64]", 2 * N * H * W * C * 4 * 2, 20};
        row_all_arms(row, threads, [&](const ComputeContext& ctx) { op->compute(vo, ins, ctx); });
    }

    // pad: [1,56,56,64] constant 1 on H and W
    {
        constexpr int64_t N = 1, H = 56, W = 56, C = 64;
        std::vector<float> a = random_f32(N * H * W * C, 6);
        std::vector<float> o(static_cast<size_t>(N * (H + 2) * (W + 2) * C));
        TensorView va = nhwc(a, N, H, W, C);
        const int64_t oshape[] = {N, H + 2, W + 2, C};
        TensorView vo(std::span<const int64_t>(oshape, 4), DataType::f32, o.data(),
                      TensorLayout::NHWC);
        PadAttributes attrs;
        attrs.pads = {0, 1, 1, 0, 0, 1, 1, 0};
        attrs.mode = PadAttributes::Mode::Constant;
        attrs.constant_value = 0.0f;
        auto op = Pad::create(attrs, Backend::CPU);
        const TensorView ins[] = {va};
        const Row row{"pad 1px [1,56,56,64]", N * (H + 2) * (W + 2) * C * 4 * 2, 20};
        row_all_arms(row, threads, [&](const ComputeContext& ctx) { op->compute(vo, ins, ctx); });
    }
}

NNOPS_BENCH(gather_scatter_flatten) {
    std::printf("--- gather / scatter / flatten (block copy) ---\n");
    const std::vector<int> threads = bench_threads();

    // gather: an embedding table [32000,4096] read at 128 token ids
    {
        constexpr int64_t V = 32000, D = 4096, T = 128;
        std::vector<float> w = random_f32(V * D, 7);
        std::vector<float> o(static_cast<size_t>(T * D));
        std::vector<int32_t> idx(static_cast<size_t>(T));
        for (int64_t i = 0; i < T; ++i) {
            idx[static_cast<size_t>(i)] = static_cast<int32_t>((i * 7919) % V);
        }
        const int64_t wshape[] = {V, D};
        const int64_t ishape[] = {T};
        const int64_t oshape[] = {T, D};
        TensorView vw(std::span<const int64_t>(wshape, 2), DataType::f32, w.data());
        TensorView vi(std::span<const int64_t>(ishape, 1), DataType::s32, idx.data());
        TensorView vo(std::span<const int64_t>(oshape, 2), DataType::f32, o.data());
        GatherAttributes attrs;
        attrs.axis = 0;
        auto op = Gather::create(attrs, Backend::CPU);
        const TensorView ins[] = {vw, vi};
        const Row row{"gather embed [32000,4096] x128", T * D * 4 * 2, 10};
        row_all_arms(row, threads, [&](const ComputeContext& ctx) { op->compute(vo, ins, ctx); });
    }

    // scatter_nd: 256 rows of a [4096, 1024] table (a kv-cache write)
    {
        constexpr int64_t R = 4096, D = 1024, T = 256;
        std::vector<float> data = random_f32(R * D, 8);
        std::vector<float> upd = random_f32(T * D, 9);
        std::vector<float> o(static_cast<size_t>(R * D));
        std::vector<int64_t> idx(static_cast<size_t>(T));
        for (int64_t i = 0; i < T; ++i) {
            idx[static_cast<size_t>(i)] = (i * 13) % R;
        }
        const int64_t dshape[] = {R, D};
        const int64_t ishape[] = {T, 1};
        const int64_t ushape[] = {T, D};
        TensorView vd(std::span<const int64_t>(dshape, 2), DataType::f32, data.data());
        TensorView vi(std::span<const int64_t>(ishape, 2), DataType::s64, idx.data());
        TensorView vu(std::span<const int64_t>(ushape, 2), DataType::f32, upd.data());
        TensorView vo(std::span<const int64_t>(dshape, 2), DataType::f32, o.data());
        ScatterNDAttributes attrs;
        auto op = ScatterND::create(attrs, Backend::CPU);
        const TensorView ins[] = {vd, vi, vu};
        const Row row{"scatter_nd 256 rows of [4096,1024]", R * D * 4 * 2 + T * D * 4, 10};
        row_all_arms(row, threads, [&](const ComputeContext& ctx) { op->compute(vo, ins, ctx); });
    }

    // flatten: NHWC [1,56,56,256] read as dense (the same dims, storage order)
    {
        constexpr int64_t N = 1, H = 56, W = 56, C = 256;
        std::vector<float> a = random_f32(N * H * W * C, 10);
        std::vector<float> o(static_cast<size_t>(N * H * W * C));
        TensorView va = nhwc(a, N, H, W, C);
        const int64_t oshape[] = {N, H, W, C};
        TensorView vo(std::span<const int64_t>(oshape, 4), DataType::f32, o.data());
        FlattenAttributes attrs;
        auto op = Flatten::create(attrs, Backend::CPU);
        const TensorView ins[] = {va};
        const Row row{"flatten NHWC->dense [1,56,56,256]", N * H * W * C * 4 * 2, 20};
        row_all_arms(row, threads, [&](const ComputeContext& ctx) { op->compute(vo, ins, ctx); });
    }
}

NNOPS_BENCH(broadcast_eltwise) {
    std::printf("--- element-wise with a broadcast operand (8-lane run) ---\n");
    const std::vector<int> threads = bench_threads();

    // bias add: [1,2048,4096] + [4096]
    {
        constexpr int64_t M = 2048, Nn = 4096;
        std::vector<float> a = random_f32(M * Nn, 11);
        std::vector<float> b = random_f32(Nn, 12);
        std::vector<float> o(static_cast<size_t>(M * Nn));
        const int64_t ashape[] = {M, Nn};
        const int64_t bshape[] = {Nn};
        TensorView va(std::span<const int64_t>(ashape, 2), DataType::f32, a.data());
        TensorView vb(std::span<const int64_t>(bshape, 1), DataType::f32, b.data());
        TensorView vo(std::span<const int64_t>(ashape, 2), DataType::f32, o.data());
        EltwiseAttributes attrs;
        attrs.type = EltwiseType::Add;
        auto op = Eltwise::create(attrs, Backend::CPU);
        const TensorView ins[] = {va, vb};
        const Row row{"add [2048,4096] + [4096]", M * Nn * 4 * 3, 10};
        row_all_arms(row, threads, [&](const ComputeContext& ctx) { op->compute(vo, ins, ctx); });
    }

    // scale a row by a [1,1] scalar tensor
    {
        constexpr int64_t M = 2048, Nn = 4096;
        std::vector<float> a = random_f32(M * Nn, 13);
        std::vector<float> b(1, 1.5f);
        std::vector<float> o(static_cast<size_t>(M * Nn));
        const int64_t ashape[] = {M, Nn};
        const int64_t bshape[] = {1};
        TensorView va(std::span<const int64_t>(ashape, 2), DataType::f32, a.data());
        TensorView vb(std::span<const int64_t>(bshape, 1), DataType::f32, b.data());
        TensorView vo(std::span<const int64_t>(ashape, 2), DataType::f32, o.data());
        EltwiseAttributes attrs;
        attrs.type = EltwiseType::Mul;
        auto op = Eltwise::create(attrs, Backend::CPU);
        const TensorView ins[] = {va, vb};
        const Row row{"mul [2048,4096] * scalar", M * Nn * 4 * 3, 10};
        row_all_arms(row, threads, [&](const ComputeContext& ctx) { op->compute(vo, ins, ctx); });
    }

    // where: a [1,1,4096] mask over two [2048,4096] branches
    {
        constexpr int64_t M = 2048, Nn = 4096;
        std::vector<float> x = random_f32(M * Nn, 14);
        std::vector<float> y = random_f32(M * Nn, 15);
        std::vector<float> o(static_cast<size_t>(M * Nn));
        std::vector<uint8_t> mask(static_cast<size_t>(Nn));
        for (int64_t i = 0; i < Nn; ++i) {
            mask[static_cast<size_t>(i)] = static_cast<uint8_t>(i & 1);
        }
        const int64_t mshape[] = {1, 1, Nn};
        const int64_t ashape[] = {M, Nn};
        TensorView vm(std::span<const int64_t>(mshape, 3), DataType::u8, mask.data());
        TensorView vx(std::span<const int64_t>(ashape, 2), DataType::f32, x.data());
        TensorView vy(std::span<const int64_t>(ashape, 2), DataType::f32, y.data());
        TensorView vo(std::span<const int64_t>(ashape, 2), DataType::f32, o.data());
        WhereAttributes attrs;
        auto op = Where::create(attrs, Backend::CPU);
        const TensorView ins[] = {vm, vx, vy};
        const Row row{"where mask[1,1,4096] [2048,4096]", M * Nn * 4 * 4, 10};
        row_all_arms(row, threads, [&](const ComputeContext& ctx) { op->compute(vo, ins, ctx); });
    }
}

NNOPS_BENCH(resize_norm_batchnorm) {
    std::printf("--- resize / spatial BatchNorm (row pool) ---\n");
    const std::vector<int> threads = bench_threads();

    // bilinear upsample x2: [1,64,64,256] -> [1,128,128,256]
    {
        constexpr int64_t N = 1, H = 64, W = 64, C = 256;
        std::vector<float> a = random_f32(N * H * W * C, 16);
        std::vector<float> o(static_cast<size_t>(N * 2 * H * 2 * W * C));
        TensorView va = nhwc(a, N, H, W, C);
        const int64_t oshape[] = {N, 2 * H, 2 * W, C};
        TensorView vo(std::span<const int64_t>(oshape, 4), DataType::f32, o.data(),
                      TensorLayout::NHWC);
        ResizeAttributes attrs;
        attrs.mode = ResizeMode::Linear;
        attrs.coord_mode = CoordinateTransformMode::HalfPixel;
        auto op = Resize::create(attrs, Backend::CPU);
        const TensorView ins[] = {va};
        const Row row{"resize bilinear x2 [1,64,64,256]", N * 4 * H * W * C * 4 * 2, 10};
        row_all_arms(row, threads, [&](const ComputeContext& ctx) { op->compute(vo, ins, ctx); });
    }

    // spatial BatchNorm in place: [1,56,56,256]
    {
        constexpr int64_t N = 1, H = 56, W = 56, C = 256;
        std::vector<float> a = random_f32(N * H * W * C, 17);
        std::vector<float> o(static_cast<size_t>(N * H * W * C));
        std::vector<float> s(C, 1.0f), bb(C, 0.0f), mm(C, 0.0f), vv(C, 1.0f);
        TensorView va = nhwc(a, N, H, W, C);
        TensorView vo(std::span<const int64_t>(va.shape_span().data(), 4), DataType::f32,
                      o.data(), TensorLayout::NHWC);
        const int64_t cshape[] = {C};
        TensorView vs(std::span<const int64_t>(cshape, 1), DataType::f32, s.data());
        TensorView vb(std::span<const int64_t>(cshape, 1), DataType::f32, bb.data());
        TensorView vm(std::span<const int64_t>(cshape, 1), DataType::f32, mm.data());
        TensorView vv2(std::span<const int64_t>(cshape, 1), DataType::f32, vv.data());
        NormAttributes attrs;
        attrs.type = NormType::BatchNorm;
        attrs.spatial = true;
        auto op = Norm::create(attrs, Backend::CPU);
        const TensorView ins[] = {va, vs, vb, vm, vv2};
        const Row row{"batch_norm spatial [1,56,56,256]", N * H * W * C * 4 * 2, 20};
        row_all_arms(row, threads, [&](const ComputeContext& ctx) { op->compute(vo, ins, ctx); });
    }
}

NNOPS_BENCH(select_tail) {
    std::printf("--- topk / argmax / quantize (one long row) ---\n");
    const std::vector<int> threads = bench_threads();

    // top-5 over a vocabulary row
    {
        constexpr int64_t V = 151936, K = 5;
        std::vector<float> a = random_f32(V, 18);
        std::vector<float> ov(static_cast<size_t>(K));
        std::vector<int64_t> oi(static_cast<size_t>(K));
        const int64_t ashape[] = {1, V};
        const int64_t oshape[] = {1, K};
        TensorView va(std::span<const int64_t>(ashape, 2), DataType::f32, a.data());
        TensorView vov(std::span<const int64_t>(oshape, 2), DataType::f32, ov.data());
        TensorView voi(std::span<const int64_t>(oshape, 2), DataType::s64, oi.data());
        TopKAttributes attrs;
        attrs.type = TopKType::Max;
        attrs.axis = -1;
        attrs.k = K;
        attrs.sorted = true;
        auto op = TopK::create(attrs, Backend::CPU);
        const TensorView ins[] = {va};
        TensorView outs[] = {vov, voi};
        const Row row{"topk 5 of 151936", V * 4, 20};
        row_all_arms(row, threads,
                     [&](const ComputeContext& ctx) { op->compute(outs, ins, ctx); });
    }

    // top-64 over the same row: k large enough that the heap replaces a sort
    {
        constexpr int64_t V = 151936, K = 64;
        std::vector<float> a = random_f32(V, 20);
        std::vector<float> ov(static_cast<size_t>(K));
        std::vector<int64_t> oi(static_cast<size_t>(K));
        const int64_t ashape[] = {1, V};
        const int64_t oshape[] = {1, K};
        TensorView va(std::span<const int64_t>(ashape, 2), DataType::f32, a.data());
        TensorView vov(std::span<const int64_t>(oshape, 2), DataType::f32, ov.data());
        TensorView voi(std::span<const int64_t>(oshape, 2), DataType::s64, oi.data());
        TopKAttributes attrs;
        attrs.type = TopKType::Max;
        attrs.axis = -1;
        attrs.k = K;
        attrs.sorted = true;
        auto op = TopK::create(attrs, Backend::CPU);
        const TensorView ins[] = {va};
        TensorView outs[] = {vov, voi};
        const Row row{"topk 64 of 151936", V * 4, 20};
        row_all_arms(row, threads,
                     [&](const ComputeContext& ctx) { op->compute(outs, ins, ctx); });
    }

    // argmax over a vocabulary row
    {
        constexpr int64_t V = 151936;
        std::vector<float> a = random_f32(V, 19);
        std::vector<int64_t> oi(1);
        const int64_t ashape[] = {1, V};
        const int64_t oshape[] = {1};
        TensorView va(std::span<const int64_t>(ashape, 2), DataType::f32, a.data());
        TensorView voi(std::span<const int64_t>(oshape, 1), DataType::s64, oi.data());
        ArgMinMaxAttributes attrs;
        attrs.type = ArgMinMaxType::Max;
        attrs.axis = -1;
        auto op = ArgMax::create(attrs, Backend::CPU);
        const TensorView ins[] = {va};
        const Row row{"argmax of 151936", V * 4, 20};
        row_all_arms(row, threads, [&](const ComputeContext& ctx) { op->compute(voi, ins, ctx); });
    }

    // per-token dequantize of an activation [2048,4096]
    {
        constexpr int64_t M = 2048, Nn = 4096;
        std::vector<int8_t> a(static_cast<size_t>(M * Nn));
        for (int64_t i = 0; i < M * Nn; ++i) {
            a[static_cast<size_t>(i)] = static_cast<int8_t>((i * 31) % 251 - 125);
        }
        std::vector<float> o(static_cast<size_t>(M * Nn));
        std::vector<float> s(static_cast<size_t>(M), 0.01f);
        std::vector<int8_t> z(static_cast<size_t>(M), 0);
        const int64_t ashape[] = {M, Nn};
        const int64_t sshape[] = {M};
        TensorView va(std::span<const int64_t>(ashape, 2), DataType::s8, a.data());
        TensorView vs(std::span<const int64_t>(sshape, 1), DataType::f32, s.data());
        TensorView vz(std::span<const int64_t>(sshape, 1), DataType::s8, z.data());
        TensorView vo(std::span<const int64_t>(ashape, 2), DataType::f32, o.data());
        QuantLinearAttributes attrs;
        attrs.axis = -1;
        auto op = DequantizeLinear::create(attrs, Backend::CPU);
        const TensorView ins[] = {va, vs, vz};
        const Row row{"dequant per-token [2048,4096]", M * Nn * 5, 10};
        row_all_arms(row, threads, [&](const ComputeContext& ctx) { op->compute(vo, ins, ctx); });
    }
}
