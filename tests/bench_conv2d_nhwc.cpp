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
/// square. The depthwise row is the one exception to "Conv2D": it runs the
/// DepthwiseConv operator, whose two layouts are the shapes a depthwise layer
/// actually dispatches with (see bench_depthwise for that operator's own table).

#include "nnops/ops/conv2d.hpp"
#include "nnops/ops/depthwise_conv.hpp"
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
    /// Depthwise layers run on the DepthwiseConv operator, not Conv2D with
    /// groups == IC: the two are different kernels, and routing the row through
    /// Conv2D measures a shape that operator never sees.
    bool dw = false;
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
        {"depthwise 3x3 112x112 32",        1,   32, 112, 112,   32, 3, 3, 1, 1, 1, 1, 32, true},
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

/// One table row: the same layer run through each layout's entry point. A row is
/// a conv or a depthwise conv, and the table does not care which — only that the
/// two arms compute the same function on the layout each one wants.
struct AbLayer {
    virtual ~AbLayer() = default;
    virtual void run_nchw(const ComputeContext&) = 0;
    virtual void run_nhwc(const ComputeContext&) = 0;
    double work = 0.0;   ///< 2 * MACs, for the GFLOPS column
};

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
struct AbFp : AbLayer {
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

    void run_nchw(const ComputeContext& ctx) override {
        const TensorView ins[] = {in_nchw, w_nchw, b_v};
        op->compute(std::span<TensorView>(&out_nchw, 1), ins, ctx, nullptr);
    }
    void run_nhwc(const ComputeContext& ctx) override {
        const TensorView ins[] = {in_nhwc, w_nhwc, b_v};
        op->compute(std::span<TensorView>(&out_nhwc, 1), ins, ctx, nullptr);
    }

    Shape s;
    int64_t icg = 1, OH = 0, OW = 0;
    std::unique_ptr<Conv2D> op;
    std::vector<T> in_planar, w_planar, in_cl, w_cl, bias;
    std::vector<T> out_nchw_buf, out_nhwc_buf;
    TensorView in_nchw, w_nchw, in_nhwc, w_nhwc, b_v;
    TensorView out_nchw, out_nhwc;
};

// ============================================================
// Depthwise A/B: the DepthwiseConv operator in both layouts
// ============================================================
//
// The same layer as the conv rows, but on the operator a depthwise layer
// actually dispatches to: planar takes NCHW + a [C][1][KH][KW] filter, the
// channel-last arm takes NHWC + TensorFlow's HWCM filter. Feeding this shape to
// Conv2D with groups == IC instead measures a route no model uses.

template <class T>
struct AbDw : AbLayer {
    explicit AbDw(const Shape& c) : s(c) {
        const int64_t C = c.IC;
        const int64_t ish[]    = {c.N, C, c.IH, c.IW};
        const int64_t wsh[]    = {C, 1, c.KH, c.KW};          // [C][1][KH][KW]
        const int64_t wsh_cl[] = {c.KH, c.KW, C, 1};          // HWCM
        const int64_t bsh[]    = {C};
        constexpr DataType dt = dtype_of<T>();

        const auto in32 = random_vec<T>(std::span<const int64_t>(ish, 4), 1);
        const auto w32  = random_vec<T>(std::span<const int64_t>(wsh, 4), 2);
        bias = random_vec<T>(std::span<const int64_t>(bsh, 1), 3);
        in_planar = in32;
        in_cl     = test::nchw_to_nhwc<T>(std::span<const T>(in32.data(), in32.size()),
                                          c.N, C, c.IH, c.IW);
        w_planar  = w32;
        w_hwcm    = test::depthwise_to_hwcm<T>(std::span<const T>(w32.data(), w32.size()),
                                               C, c.KH * c.KW);

        in_nchw = TensorView(std::span<const int64_t>(ish, 4), dt, in_planar.data(),
                             TensorLayout::NCHW);
        w_nchw  = TensorView(std::span<const int64_t>(wsh, 4), dt, w_planar.data(),
                             TensorLayout::NCHW);
        in_nhwc = TensorView(std::span<const int64_t>(ish, 4), dt, in_cl.data(),
                             TensorLayout::NHWC);
        w_nhwc  = TensorView(std::span<const int64_t>(wsh_cl, 4), dt, w_hwcm.data(),
                             TensorLayout::HWCM);
        b_v     = TensorView(std::span<const int64_t>(bsh, 1), dt, bias.data());

        DepthwiseConvAttributes attrs{};
        attrs.kernel_size = {1, c.KH, c.KW};
        attrs.stride      = {1, c.SH, c.SW};
        attrs.dilation    = {1, 1, 1};
        attrs.padding     = {0, c.PH, c.PW};
        op_nchw = DepthwiseConv::create(attrs, Backend::CPU);
        op_nhwc = DepthwiseConv::create(attrs, Backend::CPU);

        const TensorDesc d0[] = {in_nchw.desc(), w_nchw.desc(), b_v.desc()};
        const auto od = op_nchw->getOutputTensorDesc(d0);
        OH = od[0].dims[2];
        OW = od[0].dims[3];
        work = 2.0 * static_cast<double>(c.N) * C * OH * OW * c.KH * c.KW;

        out_nchw_buf.assign(static_cast<size_t>(od[0].numel()), T{});
        out_nhwc_buf.assign(static_cast<size_t>(od[0].numel()), T{});
        out_nchw = TensorView(std::span<const int64_t>(od[0].dims.data(), 4), dt,
                              out_nchw_buf.data(), TensorLayout::NCHW);
        out_nhwc = TensorView(std::span<const int64_t>(od[0].dims.data(), 4), dt,
                              out_nhwc_buf.data(), TensorLayout::NHWC);
    }

    void run_nchw(const ComputeContext& ctx) override {
        const TensorView ins[] = {in_nchw, w_nchw, b_v};
        op_nchw->compute(std::span<TensorView>(&out_nchw, 1), ins, ctx, nullptr);
    }
    void run_nhwc(const ComputeContext& ctx) override {
        const TensorView ins[] = {in_nhwc, w_nhwc, b_v};
        op_nhwc->compute(std::span<TensorView>(&out_nhwc, 1), ins, ctx, nullptr);
    }

    Shape s;
    int64_t OH = 0, OW = 0;
    std::unique_ptr<DepthwiseConv> op_nchw, op_nhwc;
    std::vector<T> in_planar, w_planar, in_cl, w_hwcm, bias;
    std::vector<T> out_nchw_buf, out_nhwc_buf;
    TensorView in_nchw, w_nchw, in_nhwc, w_nhwc, b_v;
    TensorView out_nchw, out_nhwc;
};

// ============================================================
// Quantized A/B: s8 weight, u8 activation, f32 output
// ============================================================

struct AbI8 : AbLayer {
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

    void run_nchw(const ComputeContext& ctx) override {
        const TensorView ins[] = {in_nchw, w_nchw, b_v};
        op->compute(std::span<TensorView>(&out_nchw, 1), ins, ctx, nullptr);
    }
    void run_nhwc(const ComputeContext& ctx) override {
        const TensorView ins[] = {in_nhwc, w_nhwc, b_v};
        op->compute(std::span<TensorView>(&out_nhwc, 1), ins, ctx, nullptr);
    }

    Shape s;
    int64_t icg = 1, OH = 0, OW = 0;
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
/// @p make_dw builds the depthwise rows (nullptr for a table with no such arm).
template <class MakeOp, class MakeDw>
void run_table(const char* title, MakeOp make_op, MakeDw make_dw) {
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
            std::unique_ptr<AbLayer> op = c.dw ? make_dw(c) : make_op(c);
            if (!op) { continue; }   // this table has no arm for the row
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

// ============================================================
// Channel-last Winograd vs channel-last implicit GEMM
// ============================================================
//
// A is the implicit-GEMM route reading the raw HWIO filter — what a channel-last
// conv runs without Winograd; B is the Winograd route on the same activation
// and the same filter, transformed once into the U B-panels outside every
// timer. Both write the same NHWC output, so the row is purely the algorithm.
//
// Only k3s1 can route to Winograd, so the list is the k3s1 half of the layer
// sweep plus the small planes ResNet/MobileNet end on — the question is where
// the policy should stop, and 14x14/7x7 is where it stops.

const std::vector<Shape>& wino_sweep() {
    static const std::vector<Shape> s = {
        {"conv 3x3 s1  224x224   3->64", 1,    3, 224, 224,   64, 3, 3, 1, 1, 1, 1},
        {"conv 3x3 s1  224x224  64->64", 1,   64, 224, 224,   64, 3, 3, 1, 1, 1, 1},
        {"conv 3x3 s1 112x112  32->32",  1,   32, 112, 112,   32, 3, 3, 1, 1, 1, 1},
        {"conv 3x3 s1 112x112  64->64",  1,   64, 112, 112,   64, 3, 3, 1, 1, 1, 1},
        {"conv 3x3 s1  56x56  32->32",   1,   32,  56,  56,   32, 3, 3, 1, 1, 1, 1},
        {"conv 3x3 s1  56x56  64->64",   1,   64,  56,  56,   64, 3, 3, 1, 1, 1, 1},
        {"conv 3x3 s1  56x56 128->128",  1,  128,  56,  56,  128, 3, 3, 1, 1, 1, 1},
        {"conv 3x3 s1  56x56 256->256",  1,  256,  56,  56,  256, 3, 3, 1, 1, 1, 1},
        {"conv 3x3 s1  28x28  32->32",   1,   32,  28,  28,   32, 3, 3, 1, 1, 1, 1},
        {"conv 3x3 s1  28x28  64->64",   1,   64,  28,  28,   64, 3, 3, 1, 1, 1, 1},
        {"conv 3x3 s1  28x28 128->128",  1,  128,  28,  28,  128, 3, 3, 1, 1, 1, 1},
        {"conv 3x3 s1  28x28 256->256",  1,  256,  28,  28,  256, 3, 3, 1, 1, 1, 1},
        {"conv 3x3 s1  28x28 512->512",  1,  512,  28,  28,  512, 3, 3, 1, 1, 1, 1},
        {"conv 3x3 s1  14x14  32->32",   1,   32,  14,  14,   32, 3, 3, 1, 1, 1, 1},
        {"conv 3x3 s1  14x14  64->64",   1,   64,  14,  14,   64, 3, 3, 1, 1, 1, 1},
        {"conv 3x3 s1  14x14 128->128",  1,  128,  14,  14,  128, 3, 3, 1, 1, 1, 1},
        {"conv 3x3 s1  14x14 256->256",  1,  256,  14,  14,  256, 3, 3, 1, 1, 1, 1},
        {"conv 3x3 s1   7x7   64->64",   1,   64,   7,   7,   64, 3, 3, 1, 1, 1, 1},
        {"conv 3x3 s1   7x7  256->256",  1,  256,   7,   7,  256, 3, 3, 1, 1, 1, 1},
        {"conv 3x3 s1   7x7  512->512",  1,  512,   7,   7,  512, 3, 3, 1, 1, 1, 1},
        {"conv 3x3 s1  28x28 128->128 n4", 4, 128, 28,  28,  128, 3, 3, 1, 1, 1, 1},
    };
    return s;
}

template <class T>
struct AbWino {
    explicit AbWino(const Shape& c) : s(c) {
        const int64_t ish[] = {c.N, c.IC, c.IH, c.IW};
        const int64_t wsh[] = {c.KH, c.KW, c.IC, c.OC};   // HWIO
        constexpr DataType dt = dtype_of<T>();

        // The activation and the filter are generated planar and converted once:
        // both routes then see identical values, and the conversion is outside
        // every timed region.
        const auto in32 = random_vec<T>(std::span<const int64_t>(ish, 4), 1);
        const auto w32  = random_vec<T>(std::span<const int64_t>(wsh, 4), 2);
        in_cl = test::nchw_to_nhwc<T>(std::span<const T>(in32.data(), in32.size()),
                                      c.N, c.IC, c.IH, c.IW);
        w_hwio = w32;   // already [KH][KW][IC][OC]

        in_v = TensorView(std::span<const int64_t>(ish, 4), dt, in_cl.data(),
                          TensorLayout::NHWC);
        w_v  = TensorView(std::span<const int64_t>(wsh, 4), dt, w_hwio.data(),
                          TensorLayout::HWIO);

        Conv2DAttributes attrs{};
        attrs.kernel_size = {c.KH, c.KW};
        attrs.stride = {c.SH, c.SW};
        attrs.padding = {c.PH, c.PW};
        attrs.groups = c.groups;
        im_attrs = attrs;
        im_attrs.algorithm = ConvAlgorithm::ImplicitGemm;
        wn_attrs = attrs;
        wn_attrs.algorithm = ConvAlgorithm::Winograd;
        im_op = Conv2D::create(im_attrs, Backend::CPU);
        wn_op = Conv2D::create(wn_attrs, Backend::CPU);

        const TensorDesc d0[] = {in_v.desc(), w_v.desc()};
        const auto od = im_op->getOutputTensorDesc(d0);
        out_desc = od[0];
        OH = od[0].dims[2];
        OW = od[0].dims[3];
        const int64_t ntiles = ((OH + 1) / 2) * ((OW + 1) / 2);
        work_im = 2.0 * static_cast<double>(c.N) * c.OC * OH * OW * c.IC * 9.0;
        work_wn = 2.0 * static_cast<double>(c.N) * c.OC * ntiles * c.IC * 16.0;

        out_buf.assign(static_cast<size_t>(od[0].numel()), T{});
        out_v = test::make_planar(out_desc, out_buf.data());

        // Prepack U (query -> allocate -> pack), once and outside every timer.
        {
            TensorView q;
            TensorView wi[] = {w_v};
            TensorView wo[] = {q};
            wn_op->prepackWeights(wi, wo);
            q = wo[0];
            u_buf.assign(static_cast<size_t>(q.storage_bytes() / sizeof(T)), T{});
            u_v = TensorView(q.shape_span(), q.data_type(), u_buf.data(), q.pitch(),
                             q.layout());
            TensorView wi2[] = {w_v};
            TensorView wo2[] = {u_v};
            wn_op->prepackWeights(wi2, wo2);
        }
    }

    void run_im(const ComputeContext& ctx) {
        const TensorView ins[] = {in_v, w_v};
        im_op->compute(std::span<TensorView>(&out_v, 1), ins, ctx, nullptr);
    }
    void run_wn(const ComputeContext& ctx) {
        const TensorView ins[] = {in_v, u_v};
        wn_op->compute(std::span<TensorView>(&out_v, 1), ins, ctx, nullptr);
    }

    Shape s;
    int64_t OH = 0, OW = 0;
    double work_im = 0.0, work_wn = 0.0;
    Conv2DAttributes im_attrs, wn_attrs;
    std::unique_ptr<Conv2D> im_op, wn_op;
    std::vector<T> in_cl, w_hwio, u_buf, out_buf;
    TensorView in_v, w_v, u_v, out_v;
    TensorDesc out_desc{};
};

template <class T>
void run_wino_table(const char* title) {
    const int iters = 10, rounds = 3;
    std::printf("\n== %s ==\n", title);
    for (int nt : bench_threads()) {
        std::printf("  %d thread%s\n", nt, nt == 1 ? "" : "s");
        std::printf("    %-32s %8s %8s | %8s %8s | %s | %s\n",
                    "layer", "im gemm ms", "GFLOPS", "wino ms", "GFLOPS",
                    "wino time", "verdict");
        int wins = 0, losses = 0, ties = 0;
        double tot_im = 0.0, tot_wn = 0.0;
        for (const auto& c : wino_sweep()) {
            auto op = std::make_unique<AbWino<T>>(c);
            const AbResult r = measure_ab(nt,
                [&](const ComputeContext& x) { op->run_im(x); },
                [&](const ComputeContext& x) { op->run_wn(x); },
                iters, rounds);
            const double ratio = r.b_ms / r.a_ms;
            const bool real = std::abs(r.a_ms - r.b_ms) / r.a_ms * 100.0 > r.null_pct;
            const char* verdict = !real ? "tie (noise)"
                                : (ratio < 1.0 ? "wino wins" : "im gemm wins");
            if (!real) { ++ties; } else if (ratio < 1.0) { ++wins; } else { ++losses; }
            tot_im += r.a_ms;
            tot_wn += r.b_ms;
            std::printf("    %-32s %8.3f %8.1f | %8.3f %8.1f | %8.2fx | %s\n",
                        c.name, r.a_ms, op->work_im / r.a_ms / 1e6,
                        r.b_ms, op->work_wn / r.b_ms / 1e6, ratio, verdict);
        }
        std::printf("    %-32s %8.3f %8s | %8.3f %8s | %8.2fx | "
                    "%d win, %d lose, %d tie\n",
                    "sum of the layers", tot_im, "", tot_wn, "",
                    tot_wn / tot_im, wins, losses, ties);
    }
}

// ============================================================
// The layer list as a graph: every conv cold
// ============================================================
//
// The A/B above re-runs one layer until it is L2-resident, which is the right
// instrument for the kernel and the wrong one for the network: inside a forward
// pass a layer's input, weights and output arrive cold, and the layers around it
// evict what it just touched. MobileNetV2's 35 convs are the list below, in the
// order the model issues them; one pass over it is the graph's conv time.
//
// Same layers, same kernels — the difference from the table above is what the
// neighbours did to the caches. It is where the small-plane expansions and the
// big-plane head show their real cost, and it is the number a change to the
// pointwise route has to move.

const std::vector<Shape>& mobile_graph() {
    static const std::vector<Shape> s = {
        {"stem   3x3 s2 224   3->32",    1,    3, 224, 224,   32, 3, 3, 2, 2, 1, 1},
        {"proj   @112      32->16",       1,   32, 112, 112,   16, 1, 1},
        {"expand @112      16->96",       1,   16, 112, 112,   96, 1, 1},
        {"proj   @56       96->24",       1,   96,  56,  56,   24, 1, 1},
        {"expand @56       24->144",      1,   24,  56,  56,  144, 1, 1},
        {"proj   @56      144->24",       1,  144,  56,  56,   24, 1, 1},
        {"expand @56       24->144",      1,   24,  56,  56,  144, 1, 1},
        {"proj   @28      144->32",       1,  144,  28,  28,   32, 1, 1},
        {"expand @28       32->192",      1,   32,  28,  28,  192, 1, 1},
        {"proj   @28      192->32",       1,  192,  28,  28,   32, 1, 1},
        {"expand @28       32->192",      1,   32,  28,  28,  192, 1, 1},
        {"proj   @28      192->32",       1,  192,  28,  28,   32, 1, 1},
        {"expand @28       32->192",      1,   32,  28,  28,  192, 1, 1},
        {"proj   @14      192->64",       1,  192,  14,  14,   64, 1, 1},
        {"expand @14       64->384",      1,   64,  14,  14,  384, 1, 1},
        {"proj   @14      384->64",       1,  384,  14,  14,   64, 1, 1},
        {"expand @14       64->384",      1,   64,  14,  14,  384, 1, 1},
        {"proj   @14      384->64",       1,  384,  14,  14,   64, 1, 1},
        {"expand @14       64->384",      1,   64,  14,  14,  384, 1, 1},
        {"proj   @14      384->64",       1,  384,  14,  14,   64, 1, 1},
        {"expand @14       64->384",      1,   64,  14,  14,  384, 1, 1},
        {"proj   @14      384->96",       1,  384,  14,  14,   96, 1, 1},
        {"expand @14       96->576",      1,   96,  14,  14,  576, 1, 1},
        {"proj   @14      576->96",       1,  576,  14,  14,   96, 1, 1},
        {"expand @14       96->576",      1,   96,  14,  14,  576, 1, 1},
        {"proj   @14      576->96",       1,  576,  14,  14,   96, 1, 1},
        {"expand @14       96->576",      1,   96,  14,  14,  576, 1, 1},
        {"proj   @7       576->160",      1,  576,   7,   7,  160, 1, 1},
        {"expand @7       160->960",      1,  160,   7,   7,  960, 1, 1},
        {"proj   @7       960->160",      1,  960,   7,   7,  160, 1, 1},
        {"expand @7       160->960",      1,  160,   7,   7,  960, 1, 1},
        {"proj   @7       960->160",      1,  960,   7,   7,  160, 1, 1},
        {"expand @7       160->960",      1,  160,   7,   7,  960, 1, 1},
        {"proj   @7       960->320",      1,  960,   7,   7,  320, 1, 1},
        {"head   @7       320->1280",     1,  320,   7,   7, 1280, 1, 1},
    };
    return s;
}

}  // namespace

/// One pass over the conv list in graph order, every call timed on its own so a
/// row's cost is the cold one. Rows are medians over @c passes, the total is the
/// median pass.
NNOPS_BENCH(conv2d_nhwc_mobile_graph) {
    const int passes = 5;
    std::printf("\n== f32 channel-last: MobileNetV2's convs as a graph (cold) ==\n");
    for (int nt : bench_threads()) {
        std::printf("  %d thread%s\n", nt, nt == 1 ? "" : "s");
        std::vector<std::unique_ptr<AbFp<float>>> ops;
        for (const auto& c : mobile_graph()) {
            ops.push_back(std::make_unique<AbFp<float>>(c));
        }
        BenchPool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

        for (auto& op : ops) { op->run_nhwc(use_ctx); }   // warm-up

        std::vector<std::vector<double>> row(ops.size());
        std::vector<double> pass_ms;
        for (int pass = 0; pass < passes; ++pass) {
            double sum = 0.0;
            for (size_t i = 0; i < ops.size(); ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                ops[i]->run_nhwc(use_ctx);
                const double ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - t0).count();
                row[i].push_back(ms);
                sum += ms;
            }
            pass_ms.push_back(sum);
        }
        for (auto& r : row) { std::sort(r.begin(), r.end()); }
        std::sort(pass_ms.begin(), pass_ms.end());

        std::printf("    %-24s %8s %8s\n", "layer", "ms", "GFLOPS");
        double sum = 0.0;
        for (size_t i = 0; i < ops.size(); ++i) {
            const double ms = row[i][row[i].size() / 2];
            sum += ms;
            std::printf("    %-24s %8.3f %8.1f\n", ops[i]->s.name, ms,
                        ops[i]->work / ms / 1e6);
        }
        std::printf("    %-24s %8.3f\n", "sum of the rows", sum);
        std::printf("    %-24s %8.3f ms per pass (median of %d)\n",
                    "the graph", pass_ms[pass_ms.size() / 2], passes);
    }
}

NNOPS_BENCH(conv2d_nhwc_wino_vs_implicit_f32) {
    run_wino_table<float>(
        "f32: channel-last implicit GEMM (HWIO) vs channel-last Winograd (U panels)");
}

NNOPS_BENCH(conv2d_nhwc_wino_vs_implicit_f16) {
    run_wino_table<half>(
        "f16: channel-last implicit GEMM (HWIO) vs channel-last Winograd (U panels)");
}

NNOPS_BENCH(conv2d_nhwc_vs_nchw_f32) {
    run_table("f32: planar (NCHW+OIHW) vs channel-last (NHWC+HWIO)",
              [](const Shape& c) -> std::unique_ptr<AbLayer> {
                  return std::make_unique<AbFp<float>>(c);
              },
              [](const Shape& c) -> std::unique_ptr<AbLayer> {
                  return std::make_unique<AbDw<float>>(c);
              });
}

NNOPS_BENCH(conv2d_nhwc_vs_nchw_f16) {
    run_table("f16: planar (NCHW+OIHW) vs channel-last (NHWC+HWIO)",
              [](const Shape& c) -> std::unique_ptr<AbLayer> {
                  return std::make_unique<AbFp<half>>(c);
              },
              [](const Shape& c) -> std::unique_ptr<AbLayer> {
                  return std::make_unique<AbDw<half>>(c);
              });
}

NNOPS_BENCH(conv2d_nhwc_vs_nchw_int8) {
    run_table("int8: s8 weight x u8 activation -> f32, planar vs channel-last",
              [](const Shape& c) -> std::unique_ptr<AbLayer> {
                  return std::make_unique<AbI8>(c);
              },
              // A quantized depthwise arm needs a per-channel weight zero-point
              // and its own output type; bench_depthwise owns that comparison.
              [](const Shape&) -> std::unique_ptr<AbLayer> { return nullptr; });
}
