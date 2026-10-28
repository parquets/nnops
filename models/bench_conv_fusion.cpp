/// @file bench_conv_fusion.cpp
/// @brief conv -> BN -> activation: one fused operator call against the three
///        it replaces, on the layer shapes the two model runners actually run.
///
///   unfused  Conv2D/DepthwiseConv, then Norm(BatchNorm), then the activation —
///            three calls, three passes over the activation
///   fused    one conv call: the BN folded into the weight and a bias, the
///            activation riding the conv's epilogue
///
/// Both arms go through the same wrappers the runners use (models/common/
/// net_ops.hpp), so allocation and shape inference are in both numbers — which
/// is the honest comparison: a fused layer really does allocate one activation
/// instead of three.
///
/// The arms are interleaved inside one process, so they share an instrument;
/// the reported value is the median of `rounds` timings of `iters` calls.
/// `NNOPS_BENCH_THREADS=1,4,8` pins the thread ladder (the runners themselves
/// are sequential; this shows what the fusion is worth when the conv has a
/// pool behind it).

#include "common/net_ops.hpp"
#include "nnops/core/openmp_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

using model::Buf;
using nnops::TensorView;

namespace {

/// A tensor the bench owns: storage, its shape, and a view over both.
struct Owned {
    std::vector<float> data;
    std::vector<int64_t> shape;
    TensorView view;

    Owned() = default;
    Owned(const Owned&) = delete;
    Owned& operator=(const Owned&) = delete;

    void reset(std::vector<int64_t> sh, uint32_t seed, float lo, float hi)
    {
        shape = std::move(sh);
        int64_t n = 1;
        for (int64_t d : shape) { n *= d; }
        data.resize(static_cast<size_t>(n));
        uint32_t s = seed != 0 ? seed : 1;
        for (auto& v : data) {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            const float u = static_cast<float>(s & 0xFFFFFFu) / 16777216.0f;
            v = lo + (hi - lo) * u;
        }
        view = TensorView(shape, nnops::DataType::f32, data.data());
    }
};

/// Every tensor the bench has ever made, kept for the life of the process. The
/// wrappers cache prepacked weights — the folded filter, its Winograd U — by
/// pointer, so a freed tensor's address coming back would hand the next case
/// the previous case's prepack. The runners get this for free: the artifact
/// owns every tensor from load to exit.
std::vector<std::unique_ptr<Owned>>& arena()
{
    static std::vector<std::unique_ptr<Owned>> tensors;
    return tensors;
}

Owned& make(std::vector<int64_t> shape, uint32_t seed, float lo, float hi)
{
    arena().push_back(std::make_unique<Owned>());
    Owned& t = *arena().back();
    t.reset(std::move(shape), seed, lo, hi);
    return t;
}

struct FusionCase {
    const char* name;
    std::vector<int64_t> in;  // N, C, H, W
    int64_t oc = 0;           // output channels (depthwise: ignored, C == OC)
    int64_t k = 1, s = 1, p = 0;
    bool depthwise = false;
    nnops::EpilogueActivateType act = nnops::EpilogueActivateType::Relu6;
};

nnops::Conv2DAttributes conv_attrs(const FusionCase& c)
{
    nnops::Conv2DAttributes attrs;
    attrs.kernel_size = {c.k, c.k};
    attrs.stride = {c.s, c.s};
    attrs.padding = {c.p, c.p};
    return attrs;
}

const char* route_of(const FusionCase& c)
{
    if (c.depthwise) { return "depthwise"; }
    const auto attrs = conv_attrs(c);
    const int64_t ic = c.in[1];
    return nnops::winograd_applicable(attrs, ic, c.in[2], c.in[3]) ? "wino" : "im2col";
}

const char* act_name(nnops::EpilogueActivateType a)
{
    switch (a) {
    case nnops::EpilogueActivateType::None:  return "bn only";
    case nnops::EpilogueActivateType::Relu:  return "+relu";
    case nnops::EpilogueActivateType::Relu6: return "+relu6";
    default:                                 return "+?";
    }
}

std::vector<int> bench_threads()
{
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
    return {1, 4, 8};
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

template <typename Fn>
double time_us(int iters, Fn&& fn)
{
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) { fn(); }
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

/// How far the folded chain is from the three-call one, measured the way the
/// model checkers measure: worst |a-b| / (atol + rtol|a|), so a value below 1
/// means they agree to the standard the PyTorch checkpoints are held to.
/// A pure relative difference is the wrong instrument here — after a relu the
/// outputs bottom out at zero, and 1e-6 against 5e-6 would read as 400%.
struct Agreement {
    double max_abs = 0.0;
    double tol_ratio = 0.0;
};

Agreement agreement(const TensorView& ref, const TensorView& got)
{
    constexpr double atol = 1e-3;
    constexpr double rtol = 1e-3;
    const int64_t n = ref.numel();
    const float* pa = ref.ptr<float>();
    const float* pb = got.ptr<float>();
    Agreement a;
    for (int64_t i = 0; i < n; ++i) {
        const double d = std::fabs(static_cast<double>(pa[i]) - static_cast<double>(pb[i]));
        a.max_abs = std::fmax(a.max_abs, d);
        a.tol_ratio = std::fmax(
            a.tol_ratio, d / (atol + rtol * std::fabs(static_cast<double>(pa[i]))));
    }
    return a;
}

void run_case(const FusionCase& c, const std::vector<int>& threads, int rounds,
              int iters)
{
    const int64_t oc = c.depthwise ? c.in[1] : c.oc;
    Owned& x = make(c.in, 1, -1.0f, 1.0f);
    Owned& w = c.depthwise
        ? make({c.in[1], 1, c.k, c.k}, 2, -1.0f, 1.0f)
        : make({oc, c.in[1], c.k, c.k}, 2, -1.0f, 1.0f);
    // BN statistics in the range a trained network's are: gamma/var near 1.
    Owned& gamma = make({oc}, 3, 0.5f, 1.5f);
    Owned& beta = make({oc}, 4, -0.5f, 0.5f);
    Owned& mean = make({oc}, 5, -0.5f, 0.5f);
    Owned& var = make({oc}, 6, 0.5f, 1.5f);
    const model::BatchNormParams bn{gamma.view, beta.view, mean.view, var.view, 1e-5f};

    const auto unfused = [&]() -> Buf {
        Buf y = c.depthwise
            ? model::depthwise_conv(x.view, w.view, c.k, c.s, c.p)
            : model::conv2d(x.view, w.view, c.k, c.s, c.p);
        y = model::batch_norm(y.view, bn);
        switch (c.act) {
        case nnops::EpilogueActivateType::None:  return y;
        case nnops::EpilogueActivateType::Relu:  return model::relu(y.view);
        default:                                 return model::relu6(y.view);
        }
    };
    const auto fused = [&]() -> Buf {
        return c.depthwise
            ? model::depthwise_bn_act(x.view, w.view, bn, c.act, c.k, c.s, c.p)
            : model::conv2d_bn_act(x.view, w.view, bn, c.act, c.k, c.s, c.p);
    };

    std::printf("  %-38s %-9s %s\n", c.name, route_of(c), act_name(c.act));

    for (int nt : threads) {
        model::context().cpu = (nt > 1) ? nnops::openmp_backend(nt) : nnops::CpuBackend{};

        // One checked pass each — also the warm-up that builds the folded
        // weight, its Winograd U, and every buffer's first touch.
        const Buf ref = unfused();
        const Buf got = fused();
        if (nt == threads.front()) {
            const Agreement a = agreement(ref.view, got.view);
            std::printf("    agree: max|d|=%.1e  worst|d|/(atol+rtol|ref|)=%.3f%s\n",
                        a.max_abs, a.tol_ratio, a.tol_ratio > 1.0 ? "  << FAILS" : "");
        }

        std::vector<double> t_unfused, t_fused;
        for (int r = 0; r < rounds; ++r) {
            t_unfused.push_back(time_us(iters, unfused));
            t_fused.push_back(time_us(iters, fused));
        }
        const double u = median(t_unfused);
        const double f = median(t_fused);
        std::printf("    %2dT  unfused %9.2f us | fused %9.2f us (%5.2fx, %+6.1f%%)\n",
                    nt, u, f, u / f, (f / u - 1.0) * 100.0);
    }
    model::context().cpu = nnops::CpuBackend{};
}

}  // namespace

int main()
{
    const std::vector<int> threads = bench_threads();
    const int rounds = 5;
    const int iters = 5;

    std::printf("conv-bn-act fusion: three operator calls against one folded call\n");
    std::printf("  pool=%s threads=",
                nnops::openmp_available() ? "openmp" : "sequential");
    for (size_t i = 0; i < threads.size(); ++i) {
        std::printf("%d%s", threads[i], i + 1 < threads.size() ? "," : "\n");
    }

    const std::vector<FusionCase> cases = {
        {"resnet18 conv1  7x7/2",              {1, 3, 224, 224},   64, 7, 2, 3, false, nnops::EpilogueActivateType::Relu},
        {"resnet18 layer1 3x3/1 @56",          {1, 64, 56, 56},    64, 3, 1, 1, false, nnops::EpilogueActivateType::Relu},
        {"resnet18 layer4 3x3/1 @7",           {1, 512, 7, 7},    512, 3, 1, 1, false, nnops::EpilogueActivateType::Relu},
        {"mobilenet stem 3x3/2",               {1, 3, 224, 224},   32, 3, 2, 1, false, nnops::EpilogueActivateType::Relu6},
        {"mobilenet expand 1x1 24->144 @56",   {1, 24, 56, 56},   144, 1, 1, 0, false, nnops::EpilogueActivateType::Relu6},
        {"mobilenet expand 1x1 96->576 @14",   {1, 96, 14, 14},   576, 1, 1, 0, false, nnops::EpilogueActivateType::Relu6},
        {"mobilenet dw 3x3/1 C=144 @56",       {1, 144, 56, 56},  144, 3, 1, 1, true,  nnops::EpilogueActivateType::Relu6},
        {"mobilenet project 1x1 144->24 @56",  {1, 144, 56, 56},   24, 1, 1, 0, false, nnops::EpilogueActivateType::None},
    };

    for (const auto& c : cases) {
        run_case(c, threads, rounds, iters);
    }
    return 0;
}
