/// @file bench_conv3d.cpp
/// @brief Conv3D — the im2col → GEMM route against the scalar gather it replaced.
///
/// Two arms on identical values: the operator, and
/// reference::conv3d_channel_first_ref from tests/reference, the output-driven
/// scalar reduction every voxel used to walk. The ref arm is not a competitor's
/// kernel — it is the shape of the code the col GEMM was built against, so the
/// ratio says what the structure bought.
///
/// Measurement rules follow the other benches: interleaved, with a repeated
/// operator pass as the null control, per-round min, median across rounds. The
/// scalar arm is two orders of magnitude slower, so it gets one call per timing
/// pass where the operator gets ten.
///
/// FLOPs are 2·N·OC·(IC/G)·OD·OH·OW·KD·KH·KW — one MAC per output tap, the
/// standard count for a forward convolution.
///
/// The rows are video-decoder shapes: 3x3x3 refiners and downsamples, the
/// temporal-only 3x1x1, the pointwise 1x1x1 the video head spends its time in,
/// a grouped case, a depthwise case, a batch and a large kernel.
/// `NNOPS_BENCH_THREADS` pins the ladder, as in the others.

#include "nnops/ops/conv3d.hpp"
#include "common/bench_harness.hpp"
#include "common/cpu_pool.hpp"
#include "common/random_tensor.hpp"
#include "common/test_helpers.hpp"
#include "common/channel_last.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <thread>
#include <vector>

using namespace nnops;

namespace nnops::backend::cpu::reference {
    void conv3d_channel_first_ref(const Conv3DAttributes& attrs,
                                  TensorView& output,
                                  std::span<const TensorView> inputs,
                                  const ComputeContext& ctx,
                                  void* workspace);
}

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

struct Case {
    const char* name;
    std::vector<int64_t> in_shape;
    std::vector<int64_t> w_shape;
    Conv3DAttributes attrs;
};

/// One entry per shape family a 3-D backbone runs.
std::vector<Case> make_cases()
{
    std::vector<Case> cs;

    auto add = [&](const char* name, std::vector<int64_t> in, std::vector<int64_t> w,
                   std::array<int64_t, 3> k, std::array<int64_t, 3> s,
                   std::array<int64_t, 3> d, std::array<int64_t, 3> p, int64_t g = 1) {
        Case c{name, std::move(in), std::move(w), {}};
        c.attrs.kernel_size = k;
        c.attrs.stride = s;
        c.attrs.dilation = d;
        c.attrs.padding = p;
        c.attrs.groups = g;
        cs.push_back(std::move(c));
    };

    add("stem 1x3x16x64x64 -> 32 k3", {1, 3, 16, 64, 64}, {3, 3, 3, 3, 32},
        {3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1});
    add("refine 1x64x8x16x16 -> 64 k3", {1, 64, 8, 16, 16}, {3, 3, 3, 64, 64},
        {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1});
    add("down 1x64x8x32x32 -> 128 k3s2", {1, 64, 8, 32, 32}, {3, 3, 3, 64, 128},
        {3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1});
    add("temporal 1x64x16x14x14 -> 64 k3x1", {1, 64, 16, 14, 14}, {3, 1, 1, 64, 64},
        {3, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1});
    add("pointwise 1x128x8x14x14 -> 64", {1, 128, 8, 14, 14}, {1, 1, 1, 128, 64},
        {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0});
    add("grouped 1x64x4x8x8 -> 64 g8", {1, 64, 4, 8, 8}, {3, 3, 3, 8, 64},
        {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 8);
    add("depthwise 1x32x4x8x8 -> 32 g32", {1, 32, 4, 8, 8}, {3, 3, 3, 1, 32},
        {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 32);
    add("batch 4 4x32x4x8x8 -> 32 k3", {4, 32, 4, 8, 8}, {3, 3, 3, 32, 32},
        {3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1});
    add("large k 1x32x8x8x8 -> 32 k5", {1, 32, 8, 8, 8}, {5, 5, 5, 32, 32},
        {5, 5, 5}, {1, 1, 1}, {1, 1, 1}, {2, 2, 2});
    add("dilated 1x32x4x8x8 -> 32 k3d2", {1, 32, 4, 8, 8}, {3, 3, 3, 32, 32},
        {3, 3, 3}, {1, 1, 1}, {2, 2, 2}, {2, 2, 2});

    return cs;
}

void run_one(const Case& c, int nt, int rounds)
{
    const int64_t N = c.in_shape[0];
    const int64_t G = c.attrs.groups;
    const int64_t OC = c.w_shape[4];
    const int64_t IC_per_G = c.w_shape[3];

    auto [in_vec, in_view] = test::make_random_tensor(c.in_shape);
    auto [w_vec, w_dense] = test::make_random_tensor(c.w_shape);
    TensorView w_view(c.w_shape, DataType::f32, w_vec.data(), TensorLayout::DHWIO);
    (void)w_dense;

    // The operator reads a channel-last [N, ID, IH, IW, IC] activation; the
    // scalar reference keeps the dense [N, IC, ID, IH, IW] one.
    const int64_t S_in = c.in_shape[2] * c.in_shape[3] * c.in_shape[4];
    const auto in_cl_vec = test::to_channel_last<float>(in_vec, N, c.in_shape[1], S_in);
    const int64_t in_cl_dims[] = {c.in_shape[0], c.in_shape[2], c.in_shape[3],
                                  c.in_shape[4], c.in_shape[1]};
    TensorView in_cl(std::span<const int64_t>(in_cl_dims, 5), DataType::f32,
                     const_cast<float*>(in_cl_vec.data()), TensorLayout::NDHWC);

    auto op = Conv3D::create(c.attrs, Backend::CPU);
    const TensorDesc d[] = {in_cl.desc(), w_view.desc()};
    const auto descs = op->getOutputTensorDesc(d);
    const auto numel = static_cast<size_t>(descs[0].numel());

    // Channel-last output for the operator, dense for the reference.
    const int64_t ref_dims[] = {c.in_shape[0], OC, descs[0].dims[1],
                                descs[0].dims[2], descs[0].dims[3]};
    std::vector<float> out_buf(numel, 0.0f);
    auto out = test::make_channel_first(descs[0], out_buf.data());
    std::vector<float> ref_buf(numel, 0.0f);
    TensorView ref_out(std::span<const int64_t>(ref_dims, 5), DataType::f32,
                       ref_buf.data(), TensorLayout::Dense);

    BenchPool pool(nt);
    ComputeContext ctx;
    ctx.cpu = pool.cpu;
    const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

    const TensorView ins[] = {in_cl, w_view};
    TensorView outs[] = {out};

    auto arm_op = [&]() { op->compute(outs, ins, use_ctx); };
    auto arm_ref = [&]() {
        const TensorView ref_ins[] = {in_view, w_view};
        backend::cpu::reference::conv3d_channel_first_ref(c.attrs, ref_out, ref_ins,
                                                          use_ctx, nullptr);
    };

    // The operator arm is sub-millisecond on the smaller rows, where a single
    // pass is all timer noise; it gets enough calls to fill a few milliseconds.
    const int op_iters = 10;
    const int ref_iters = 1;

    arm_op();
    arm_ref();

    std::vector<double> op_ms, ref_ms, null_pct;
    for (int r = 0; r < rounds; ++r) {
        const double a1 = time_ms(arm_op, op_iters);
        const double b1 = time_ms(arm_ref, ref_iters);
        const double a2 = time_ms(arm_op, op_iters);   // null control
        op_ms.push_back(std::min(a1, a2));
        ref_ms.push_back(b1);
        null_pct.push_back(std::abs(a2 - a1) / std::min(a1, a2) * 100.0);
    }

    const double op_t = median_of(op_ms);
    const double ref_t = median_of(ref_ms);
    const double null_pct_med = median_of(null_pct);

    const int64_t OD = descs[0].dims[1];
    const int64_t OH = descs[0].dims[2];
    const int64_t OW = descs[0].dims[3];
    const double flops = 2.0 * static_cast<double>(N) * static_cast<double>(OC) *
                         static_cast<double>(IC_per_G) *
                         static_cast<double>(OD * OH * OW) *
                         static_cast<double>(c.attrs.kernel_size[0] *
                                             c.attrs.kernel_size[1] *
                                             c.attrs.kernel_size[2]);

    std::printf("    %-36s | %8.3f %9.3f | %7.1f | %6.2f %6.1f%%\n", c.name,
                op_t, ref_t, flops / (op_t * 1e6), ref_t / op_t, null_pct_med);
}

/// The quantized route on the same shapes: u8 activation, s8 weight, u8 out.
void run_one_i8(const Case& c, int nt, int rounds)
{
    const int64_t N = c.in_shape[0];
    const int64_t OC = c.w_shape[4];
    const int64_t IC_per_G = c.w_shape[3];
    const int64_t IC = c.in_shape[1];
    const int64_t karea = c.attrs.kernel_size[0] * c.attrs.kernel_size[1] *
                          c.attrs.kernel_size[2];

    // NDHWC activation, as stored.
    const int64_t in_n = N * c.in_shape[2] * c.in_shape[3] * c.in_shape[4] * IC;
    std::vector<uint8_t> act(static_cast<size_t>(in_n));
    {
        test::XorShift128 rng(11);
        for (auto& v : act) { v = static_cast<uint8_t>(rng.next_u64() & 0xFFu); }
    }
    std::vector<int8_t> w(static_cast<size_t>(OC * IC_per_G * karea));
    {
        test::XorShift128 rng(12);
        for (auto& v : w) { v = static_cast<int8_t>(int32_t(rng.next_u64() % 255) - 127); }
    }
    std::vector<float> w_s(static_cast<size_t>(OC), 0.01f);
    QuantParams qa;
    qa.scale = 0.0078125f;
    QuantParams qw;
    qw.granularity = QuantGranularity::PerChannel;
    qw.scale_data = w_s.data();
    qw.num_scales = OC;

    Conv3DAttributes attrs = c.attrs;
    attrs.output_dtype = DataType::u8;

    const int64_t in_dims[] = {c.in_shape[0], c.in_shape[2], c.in_shape[3], c.in_shape[4],
                               c.in_shape[1]};
    const int64_t w_dims[] = {c.attrs.kernel_size[0], c.attrs.kernel_size[1],
                              c.attrs.kernel_size[2], IC_per_G, OC};
    TensorView in(std::span<const int64_t>(in_dims, 5), DataType::u8, act.data(),
                  TensorLayout::NDHWC, qa);
    TensorView wt(std::span<const int64_t>(w_dims, 5), DataType::s8, w.data(),
                  TensorLayout::DHWIO);
    auto op = Conv3D::create(attrs, Backend::CPU);
    const TensorDesc d[] = {in.desc(), wt.desc()};
    const auto descs = op->getOutputTensorDesc(d);
    std::vector<uint8_t> out_buf(descs[0].nbytes(), 0);
    TensorView out(descs[0].dims, DataType::u8, out_buf.data(), TensorLayout::NDHWC);

    BenchPool pool(nt);
    ComputeContext ctx;
    ctx.cpu = pool.cpu;
    const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};
    const TensorView ins[] = {in, wt};
    TensorView outs[] = {out};
    auto arm = [&]() { op->compute(outs, ins, use_ctx); };
    arm();

    std::vector<double> ms;
    for (int r = 0; r < rounds; ++r) { ms.push_back(time_ms(arm, 20)); }
    const double t = median_of(ms);

    const int64_t OD = descs[0].dims[1], OH = descs[0].dims[2], OW = descs[0].dims[3];
    const double gops = 2.0 * static_cast<double>(N) * static_cast<double>(OC) *
                        static_cast<double>(IC_per_G) *
                        static_cast<double>(OD * OH * OW) * static_cast<double>(karea) / 1e9;
    std::printf("    %-36s | %8.3f %9.1f\n", c.name, t, gops / (t * 1e-3));
}

/// The prepacked filter against the raw one on the same values — what
/// Conv3D::prepackWeights buys, since the raw route re-packs B inside MatMul on
/// every call.
void run_one_prepack(const Case& c, int nt, int rounds)
{
    const int64_t N = c.in_shape[0];

    auto [in_vec, in_dense] = test::make_random_tensor(c.in_shape);
    auto [w_vec, w_dense] = test::make_random_tensor(c.w_shape);
    TensorView w(c.w_shape, DataType::f32, w_vec.data(), TensorLayout::DHWIO);
    (void)in_dense;
    (void)w_dense;

    const int64_t S_in = c.in_shape[2] * c.in_shape[3] * c.in_shape[4];
    const auto in_cl_vec = test::to_channel_last<float>(in_vec, N, c.in_shape[1], S_in);
    const int64_t in_cl_dims[] = {c.in_shape[0], c.in_shape[2], c.in_shape[3],
                                  c.in_shape[4], c.in_shape[1]};
    TensorView in_cl(std::span<const int64_t>(in_cl_dims, 5), DataType::f32,
                     const_cast<float*>(in_cl_vec.data()), TensorLayout::NDHWC);

    auto op = Conv3D::create(c.attrs, Backend::CPU);
    std::vector<uint8_t> blob;
    TensorView wpacked;
    {
        TensorView q;
        const TensorView wi[] = {w};
        TensorView wq[] = {q};
        op->prepackWeights(wi, wq);
        q = wq[0];
        if (q.rank() == 0) {
            std::printf("    %-36s | no packed form\n", c.name);
            return;
        }
        blob.assign(q.storage_bytes(), 0);
        wpacked = TensorView(q.shape_span(), q.data_type(), blob.data(), q.pitch(), q.layout(),
                             q.quant_params());
        TensorView wf[] = {wpacked};
        op->prepackWeights(wi, wf);
    }

    const TensorDesc d[] = {in_cl.desc(), w.desc()};
    const auto descs = op->getOutputTensorDesc(d);
    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto out = test::make_channel_first(descs[0], out_buf.data());
    TensorView outs[] = {out};

    BenchPool pool(nt);
    ComputeContext ctx;
    ctx.cpu = pool.cpu;
    const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

    const TensorView raw_ins[] = {in_cl, w};
    const TensorView packed_ins[] = {in_cl, wpacked};
    auto arm_raw = [&]() { op->compute(outs, raw_ins, use_ctx); };
    auto arm_packed = [&]() { op->compute(outs, packed_ins, use_ctx); };
    arm_raw();
    arm_packed();

    std::vector<double> raw_ms, packed_ms;
    for (int r = 0; r < rounds; ++r) {
        const double a1 = time_ms(arm_raw, 10);
        const double b1 = time_ms(arm_packed, 10);
        const double a2 = time_ms(arm_raw, 10); // null control
        raw_ms.push_back(std::min(a1, a2));
        packed_ms.push_back(b1);
    }
    const double raw_t = median_of(raw_ms);
    const double packed_t = median_of(packed_ms);
    std::printf("    %-36s | %8.3f %9.3f | %6.2f\n", c.name, raw_t, packed_t,
                raw_t / packed_t);
}

}  // namespace

NNOPS_BENCH(conv3d_prepack) {
    const auto cases = make_cases();
    int nth = 1;
    if (const char* env = std::getenv("NNOPS_BENCH_THREADS")) {
        nth = std::atoi(env);
        if (nth <= 0) { nth = 1; }
    }
    std::printf("\n=== Conv3D fp32, %dT — raw filter vs prepacked blob ===\n", nth);
    std::printf("    %-36s | %8s %9s | %6s\n", "case", "raw ms", "pack ms", "raw/pack");
    for (const auto& c : cases) { run_one_prepack(c, nth, 3); }
}

NNOPS_BENCH(conv3d) {
    const auto cases = make_cases();
    int nth = 1;
    if (const char* env = std::getenv("NNOPS_BENCH_THREADS")) {
        nth = std::atoi(env);
        if (nth <= 0) { nth = 1; }
    }

    std::printf("\n=== Conv3D fp32, %dT — im2col GEMM vs the scalar gather ===\n", nth);
    std::printf("    %-36s | %8s %9s | %7s | %6s %6s\n",
                "case", "op ms", "ref ms", "op GF", "vs ref", "null");
    for (const auto& c : cases) { run_one(c, nth, 3); }
}

NNOPS_BENCH(conv3d_i8) {
    const auto cases = make_cases();
    int nth = 1;
    if (const char* env = std::getenv("NNOPS_BENCH_THREADS")) {
        nth = std::atoi(env);
        if (nth <= 0) { nth = 1; }
    }
    std::printf("\n=== Conv3D int8 (u8 x s8 -> u8), %dT ===\n", nth);
    std::printf("    %-36s | %8s %9s\n", "case", "op ms", "GOPS");
    for (const auto& c : cases) { run_one_i8(c, nth, 3); }
}
