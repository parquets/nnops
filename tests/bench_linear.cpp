/// @file bench_linear.cpp
/// @brief Linear micro-benchmarks.
///
/// Linear is the MatMul GEMM with W stored [N,K], an optional per-column bias
/// and an fp output for quantized operands; it drives the MatMul kernel instead
/// of reimplementing it. The pairing sections alternate it with the
/// MatMul(transpose_b) call it stands for, round by round inside one process —
/// on this host a run of one operator drifts a few percent against the next, the
/// same order as the gaps being measured (bench_matmul says the same at length).
///
/// Sections:
///   linear_fp_vs_matmul      f32 Linear against MatMul(tb), no bias: the ratio
///                            is the cost of the Linear layer itself, ~1.00.
///   linear_bias_epilogue     what bias, the fused activation and add_to add —
///                            the deferred output pass Linear pays when the GEMM
///                            cannot own the write-back.
///   linear_prepacked_weight  a weight packed once at load against packed per call.
///   linear_f16_vs_matmul     the same pairing in f16.
///   linear_int8              u8s8 / s8s8 -> f32 / f16 dequant, per-token X and
///                            per-channel W.
///   linear_int8_int_out      the same operands with an s32 / s8 / u8 destination
///                            and its int32 bias, against the f32 dequantizing row.
///   linear_threads           block-granularity scaling of the plain f32 call.
///
/// The decode rows (M=1) are the ones that matter for the operator's real use:
/// a transformer inference step is one token against a [N,K] weight, and that is
/// exactly the shape where a second pass over C is most expensive relative to
/// the GEMM.

#include "nnops/ops/linear.hpp"
#include "nnops/ops/matmul.hpp"
#include "nnops/core/quant_params.hpp"
#include "nnops/detail/half.hpp"
#include "common/bench_harness.hpp"
#include "common/cpu_pool.hpp"
#include "common/random_tensor.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

using namespace nnops;
using nnops::backend::cpu::half;

namespace {

using SteadyClock = std::chrono::high_resolution_clock;

// OpenMP when the build has it and NNOPS_BENCH_OMP allows, else spawn-per-call.
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

template <typename T>
constexpr DataType dtype_of() {
    return std::is_same_v<T, half> ? DataType::f16 : DataType::f32;
}

template <typename T>
std::vector<T> random_fp(int64_t n, uint64_t seed) {
    std::vector<T> v(static_cast<size_t>(n));
    nnops::test::XorShift128 rng(seed);
    for (auto& e : v) {
        const float f = rng.next_float(-1.0f, 1.0f);
        if constexpr (std::is_same_v<T, half>) {
            e = nnops::backend::cpu::float_to_half(f);
        } else {
            e = f;
        }
    }
    return v;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

/// Drop the pooled scratch before a cell so every cell starts from the same
/// allocator state; without it, later sections ran their decode rows up to 3x
/// slower than the same row in isolation, their free lists warm with other sizes.
void fresh_state() {
    nnops::release_scratch_memory();
}

std::vector<int> thread_ladder() {
    std::vector<int> threads{1, 2, 4, 8};
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw > 1 && std::find(threads.begin(), threads.end(), hw) == threads.end()) {
        threads.push_back(hw);
    }
    return threads;
}

/// Iterate @p fn until @p target_ms of wall clock has elapsed; return GFLOPS.
///
/// A budget, not a fixed count, keeps a decode row and a 1024^3 row on the
/// machine equally long, which is what makes rows of one table comparable on a
/// host that drifts under load. The kMinIters floor matters as much: at 2-3
/// calls the first one's cold scratch has read as a 3x error on this host;
/// eight calls bound it to ~12%.
template <typename F>
double time_gflops(F&& fn, double flops, double target_ms) {
    constexpr int64_t kMaxIters = 200000;
    constexpr int64_t kMinIters = 8;
    for (int i = 0; i < 3; ++i) { fn(); }
    const auto t0 = SteadyClock::now();
    int64_t iters = 0;
    double ms = 0.0;
    do {
        fn();
        ++iters;
        ms = std::chrono::duration<double, std::milli>(SteadyClock::now() - t0).count();
    } while ((ms < target_ms || iters < kMinIters) && iters < kMaxIters);
    return flops * static_cast<double>(iters) / (ms * 1e6);
}

struct Shape { int64_t M, K, N; };

double gemm_flops(const Shape& s) {
    return 2.0 * static_cast<double>(s.M) * static_cast<double>(s.N)
         * static_cast<double>(s.K);
}

// ---- fp problems -----------------------------------------------------------

/// A [M,K], W [N,K], optional bias [N], C [M,N], plus the Linear op and the
/// MatMul(transpose_b) call it stands for.
template <typename T>
struct FpProblem {
    std::vector<T> a_buf, w_buf, bias_buf, out_buf;
    TensorView out;
    std::vector<TensorView> ins;
    std::unique_ptr<OpBase> linear;
    std::unique_ptr<OpBase> matmul;
    double flops;

    FpProblem(const Shape& s, const LinearAttributes& la, bool with_bias,
              uint64_t seed = 1)
        : a_buf(random_fp<T>(s.M * s.K, seed)),
          w_buf(random_fp<T>(s.N * s.K, seed + 1)),
          bias_buf(with_bias ? random_fp<T>(s.N, seed + 2) : std::vector<T>{}),
          out_buf(static_cast<size_t>(s.M * s.N), T{})
    {
        constexpr DataType dt = dtype_of<T>();
        const int64_t as[] = {s.M, s.K};
        const int64_t ws[] = {s.N, s.K};
        const int64_t os[] = {s.M, s.N};
        ins.emplace_back(std::span<const int64_t>(as, 2), dt, a_buf.data());
        ins.emplace_back(std::span<const int64_t>(ws, 2), dt, w_buf.data());
        if (with_bias) {
            const int64_t bs[] = {s.N};
            ins.emplace_back(std::span<const int64_t>(bs, 1), dt, bias_buf.data());
        }
        out = TensorView(std::span<const int64_t>(os, 2), dt, out_buf.data());

        linear = Linear::create(la, Backend::CPU);
        if (!with_bias) {
            MatMulAttributes ma;
            ma.transpose_b = true;
            ma.epilogue = la.epilogue;
            matmul = MatMul::create(ma, Backend::CPU);
        }
        flops = gemm_flops(s);
    }

    double run(OpBase& op, const ComputeContext& ctx, double target_ms) {
        fresh_state();   // warm the pool inside the call, not in the reading
        return time_gflops(
            [&] {
                op.compute(std::span<TensorView>(&out, 1),
                           std::span<const TensorView>(ins.data(), ins.size()),
                           ctx, nullptr);
            },
            flops, target_ms);
    }
};

template <typename T>
void bench_fp_vs_matmul(const char* tag, const std::vector<Shape>& shapes,
                        const std::vector<int>& threads, double target_ms,
                        int rounds) {
    std::printf("  %s Linear vs MatMul(tb), no bias  budget=%.0f ms/cell  rounds=%d\n",
                tag, target_ms, rounds);
    std::printf("   %5s %5s %5s |", "M", "K", "N");
    for (int nt : threads) {
        std::printf("  %2dT:  Linear    MatMul   L/M |", nt);
    }
    std::printf("\n");

    for (const Shape& s : shapes) {
        FpProblem<T> p(s, LinearAttributes{}, /*with_bias=*/false);
        std::printf("   %5lld %5lld %5lld |", static_cast<long long>(s.M),
                    static_cast<long long>(s.K), static_cast<long long>(s.N));
        for (int nt : threads) {
            BenchPool pool(nt);
            ComputeContext ctx;
            if (nt > 1) { ctx.cpu = pool.cpu; }

            std::vector<double> vl, vm;
            for (int r = 0; r < rounds; ++r) {
                vl.push_back(p.run(*p.linear, ctx, target_ms));
                vm.push_back(p.run(*p.matmul, ctx, target_ms));
            }
            const double gl = median(vl), gm = median(vm);
            std::printf(" %9.1f %9.1f %5.3f |", gl, gm, gm > 0.0 ? gl / gm : 0.0);
        }
        std::printf("\n");
    }
}

// ---- quantized problems ----------------------------------------------------

/// A u8|s8 x s8 problem with an LLM checkpoint's parameters: X per token, W per
/// channel. `out_dt` selects the output pass: f32/f16 dequantize, s32 keeps the
/// int32 accumulate, s8/u8 requantize with the output's QuantParams. An integer
/// destination takes its bias as int32.
template <typename OutT>
struct QuantProblem {
    std::vector<int8_t> a_buf, w_buf;   // byte storage; X's dtype decides s8/u8
    std::vector<float> sx, sw;
    std::vector<int32_t> zx, zw;
    std::vector<int32_t> ibias;         // bias storage, integer destinations
    std::vector<OutT> fbias;            // bias storage, fp destinations
    std::vector<OutT> out_buf;
    TensorView out;
    std::vector<TensorView> ins;
    QuantParams out_q;
    std::unique_ptr<OpBase> linear;
    std::unique_ptr<OpBase> matmul;
    double flops;

    QuantProblem(const Shape& s, DataType a_dt, DataType out_dt,
                 uint64_t seed = 1, bool with_bias = false)
        : a_buf(static_cast<size_t>(s.M * s.K)),
          w_buf(static_cast<size_t>(s.N * s.K)),
          sx(static_cast<size_t>(s.M)),
          sw(static_cast<size_t>(s.N)),
          zx(static_cast<size_t>(s.M)),
          zw(static_cast<size_t>(s.N)),
          out_buf(static_cast<size_t>(s.M * s.N), OutT{})
    {
        nnops::test::XorShift128 rng(seed);
        const bool x_unsigned = (a_dt == DataType::u8);
        for (auto& v : a_buf) {
            const uint32_t byte = static_cast<uint32_t>(rng.next_u64() & 0xFF);
            v = x_unsigned ? static_cast<int8_t>(byte)
                           : static_cast<int8_t>(byte - 128);
        }
        for (auto& v : w_buf) {
            v = static_cast<int8_t>(static_cast<int32_t>(rng.next_u64() % 255) - 127);
        }
        // Realistic magnitudes: activations coarser than weights.
        for (size_t i = 0; i < sx.size(); ++i) { sx[i] = 0.01f + 0.001f * (i % 7); }
        for (size_t i = 0; i < sw.size(); ++i) { sw[i] = 0.002f + 0.0004f * (i % 11); }
        // s8 is symmetric (zp 0); u8 carries the usual asymmetric offset.
        for (auto& z : zx) { z = x_unsigned ? 128 : 0; }
        for (auto& z : zw) { z = 0; }

        QuantParams qx;
        qx.granularity = QuantGranularity::PerToken;
        qx.scale_data = sx.data();
        qx.zero_point_data = zx.data();
        qx.num_scales = s.M;

        QuantParams qw;
        qw.granularity = QuantGranularity::PerChannel;
        qw.scale_data = sw.data();
        qw.zero_point_data = zw.data();
        qw.num_scales = s.N;

        // An s8/u8 destination requantizes with the output's own parameters.
        const bool int_out = (out_dt == DataType::s32 || out_dt == DataType::s8 ||
                              out_dt == DataType::u8);
        if (out_dt == DataType::s8 || out_dt == DataType::u8) {
            out_q.granularity = QuantGranularity::PerTensor;
            out_q.scale = 0.0625f;
            out_q.zero_point = (out_dt == DataType::u8) ? 128 : 0;
        }

        const int64_t as[] = {s.M, s.K};
        const int64_t ws[] = {s.N, s.K};
        const int64_t os[] = {s.M, s.N};
        ins.emplace_back(std::span<const int64_t>(as, 2), a_dt,
                         a_buf.data(), TensorLayout::NCHW, qx);
        ins.emplace_back(std::span<const int64_t>(ws, 2), DataType::s8,
                         w_buf.data(), TensorLayout::NCHW, qw);
        if (with_bias) {
            const int64_t bs[] = {s.N};
            if (int_out) {
                ibias.resize(static_cast<size_t>(s.N));
                for (auto& b : ibias) {
                    b = static_cast<int32_t>(rng.next_u64() % 4096) - 2048;
                }
                ins.emplace_back(std::span<const int64_t>(bs, 1), DataType::s32,
                                 ibias.data(), TensorLayout::NCHW);
            } else {
                fbias.resize(static_cast<size_t>(s.N));
                for (auto& b : fbias) {
                    if constexpr (std::is_same_v<OutT, half>) {
                        b = nnops::backend::cpu::float_to_half(rng.next_float(-0.5f, 0.5f));
                    } else {
                        b = rng.next_float(-0.5f, 0.5f);
                    }
                }
                ins.emplace_back(std::span<const int64_t>(bs, 1), out_dt,
                                 fbias.data(), TensorLayout::NCHW);
            }
        }
        out = TensorView(std::span<const int64_t>(os, 2), out_dt, out_buf.data(),
                         TensorLayout::NCHW, out_q);

        LinearAttributes la;
        la.output_dtype = out_dt;
        linear = Linear::create(la, Backend::CPU);

        MatMulAttributes ma;
        ma.transpose_b = true;
        ma.output_dtype = out_dt;
        matmul = MatMul::create(ma, Backend::CPU);
        flops = gemm_flops(s);
    }

    double run(OpBase& op, const ComputeContext& ctx, double target_ms) {
        fresh_state();   // warm the pool inside the call, not in the reading
        return time_gflops(
            [&] {
                op.compute(std::span<TensorView>(&out, 1),
                           std::span<const TensorView>(ins.data(), ins.size()),
                           ctx, nullptr);
            },
            flops, target_ms);
    }
};

/// One int8 row: Linear against the MatMul call it stands for, same operands
/// and output dtype.
template <typename OutT>
void bench_quant_row(const Shape& s, DataType a_dt, DataType out_dt, int nt,
                     double target_ms, int rounds, bool with_bias = false) {
    QuantProblem<OutT> p(s, a_dt, out_dt, 1, with_bias);
    BenchPool pool(nt);
    ComputeContext ctx;
    if (nt > 1) { ctx.cpu = pool.cpu; }

    std::vector<double> vl, vm;
    for (int r = 0; r < rounds; ++r) {
        vl.push_back(p.run(*p.linear, ctx, target_ms));
        vm.push_back(p.run(*p.matmul, ctx, target_ms));
    }
    const double gl = median(vl), gm = median(vm);
    const char* oname = out_dt == DataType::f16   ? "f16"
                      : out_dt == DataType::s32   ? "s32"
                      : out_dt == DataType::s8    ? "s8"
                      : out_dt == DataType::u8    ? "u8"
                                                  : "f32";
    std::printf("   %5lld %5lld %5lld  %-7s -> %-3s%s | %9.1f %9.1f %5.3f |\n",
                static_cast<long long>(s.M), static_cast<long long>(s.K),
                static_cast<long long>(s.N),
                a_dt == DataType::u8 ? "u8s8" : "s8s8", oname,
                with_bias ? " +b" : "   ",
                gl, gm, gm > 0.0 ? gl / gm : 0.0);
}

}  // anonymous namespace

// ============================================================
// f32 — Linear against the MatMul call it stands for
// ============================================================

NNOPS_BENCH(linear_fp_vs_matmul) {
    const std::vector<Shape> shapes{
        {1, 4096, 4096},     // decode: one token against a 64 MB weight
        {8, 4096, 4096},     // small speculative batch
        {128, 4096, 4096},   // prefill
        {1024, 1024, 1024},  // square, FLOP bound
    };
    bench_fp_vs_matmul<float>("f32", shapes, {1, 4}, /*target_ms=*/100.0,
                             /*rounds=*/3);
}

// ============================================================
// f32 — what bias / epilogue / add_to add to the plain call
// ============================================================

NNOPS_BENCH(linear_bias_epilogue) {
    struct Variant {
        const char* name;
        EpilogueActivateType ep;
        bool bias;
        bool add_to;
    };
    // Ordered by what the GEMM can own: no bias and no residual fuse into the
    // output tile; a bias costs one pass over C; add_to with an identity
    // epilogue rides MatMul's beta, and add_to with any other epilogue needs an
    // M×N temporary and a combine pass.
    const Variant variants[] = {
        {"plain",               EpilogueActivateType::None, false, false},
        {"relu (fused)",        EpilogueActivateType::Relu, false, false},
        {"bias",                EpilogueActivateType::None, true,  false},
        {"bias + relu",         EpilogueActivateType::Relu, true,  false},
        {"add_to (beta)",       EpilogueActivateType::None, false, true},
        {"add_to + relu (tmp)", EpilogueActivateType::Relu, false, true},
    };
    const std::vector<Shape> shapes{{1, 4096, 4096}, {128, 4096, 4096}};
    constexpr int kRounds = 3;
    constexpr double kBudgetMs = 100.0;

    std::printf("  f32 Linear write-back variants  1T  budget=%.0f ms/cell  rounds=%d\n",
                kBudgetMs, kRounds);
    std::printf("   %5s %5s %5s | %-18s %9s %6s |\n", "M", "K", "N",
                "variant", "GFLOPS", "vs");

    for (const Shape& s : shapes) {
        double base = 0.0;
        for (const Variant& v : variants) {
            LinearAttributes la;
            la.epilogue.type = v.ep;
            la.add_to = v.add_to;
            FpProblem<float> p(s, la, v.bias);

            std::vector<double> g;
            for (int r = 0; r < kRounds; ++r) {
                g.push_back(p.run(*p.linear, ComputeContext{}, kBudgetMs));
            }
            const double gf = median(g);
            if (base == 0.0) { base = gf; }
            std::printf("   %5lld %5lld %5lld | %-18s %9.1f %6.3f |\n",
                        static_cast<long long>(s.M), static_cast<long long>(s.K),
                        static_cast<long long>(s.N), v.name, gf,
                        base > 0.0 ? gf / base : 0.0);
        }
        std::printf("\n");
    }
}

// ============================================================
// The prepacked weight — what prepackWeights() takes out of the call
// ============================================================
//
// Each call packs W into the GEMM's uniform panels; a weight packed once at load
// is read in place instead. Same Linear, same numbers, so the ratio is the pack.

template <typename T>
void bench_prepack_vs_raw(const char* tag, const std::vector<Shape>& shapes,
                          const std::vector<int>& threads, double target_ms,
                          int rounds) {
    std::printf("  %s Linear, weight packed once vs packed per call  "
                "budget=%.0f ms/cell  rounds=%d\n", tag, target_ms, rounds);
    std::printf("   %5s %5s %5s |", "M", "K", "N");
    for (int nt : threads) {
        std::printf("  %2dT:  packed     raw    P/R |", nt);
    }
    std::printf("\n");

    for (const Shape& s : shapes) {
        // Same seed, distinct buffers: one keeps the raw weight, the other gets the blob.
        FpProblem<T> raw(s, LinearAttributes{}, /*with_bias=*/false);
        FpProblem<T> pk(s, LinearAttributes{}, /*with_bias=*/false);

        // Pack once, outside every timer — what a graph engine does at load.
        auto op = Linear::create(LinearAttributes{}, Backend::CPU);
        TensorView query;
        const TensorView win[] = {raw.ins[1]};
        TensorView q[] = {query};
        op->prepackWeights(win, q);
        // 64-byte aligned like every buffer the pool hands out: a packed panel
        // starting 16 bytes into a line splits half its vector loads.
        std::vector<std::byte> hold(
            static_cast<size_t>(q[0].desc().numel()) * sizeof(T) + 64);
        const size_t mis = reinterpret_cast<uintptr_t>(hold.data()) % 64;
        T* blob = reinterpret_cast<T*>(hold.data() + ((64 - mis) % 64));
        TensorView packed(q[0].shape_span(), dtype_of<T>(), blob,
                          q[0].pitch(), q[0].layout());
        TensorView pack[] = {packed};
        op->prepackWeights(win, pack);
        pk.ins[1] = packed;

        std::printf("   %5lld %5lld %5lld |", static_cast<long long>(s.M),
                    static_cast<long long>(s.K), static_cast<long long>(s.N));
        for (int nt : threads) {
            BenchPool pool(nt);
            ComputeContext ctx;
            if (nt > 1) { ctx.cpu = pool.cpu; }

            std::vector<double> vp, vr;
            for (int r = 0; r < rounds; ++r) {
                vp.push_back(pk.run(*pk.linear, ctx, target_ms));
                vr.push_back(raw.run(*raw.linear, ctx, target_ms));
            }
            const double gp = median(vp), gr = median(vr);
            std::printf(" %9.1f %9.1f %5.3f |", gp, gr, gr > 0.0 ? gp / gr : 0.0);
        }
        std::printf("\n");
    }
}

NNOPS_BENCH(linear_prepacked_weight) {
    // P/R below 1 is not a defect in the blob: at a large M the per-call pack
    // is a pass over B that leaves the panels it just wrote in L2, so the GEMM
    // reads them from L2 while the blob arm re-reads them from L3. The loss
    // tracks that working set — 512^2 (1 MB) ties, 1024^2 (4 MB) loses 11%,
    // 2048^2 (16 MB) loses 13% — and needs a large M to show. Alignment is not
    // the cause: a 64-byte aligned blob measures the same as a 16-byte one.
    const std::vector<Shape> shapes{
        {1, 4096, 4096},     // decode: the per-call pack reads a 64 MB weight
        {8, 4096, 4096},     // small speculative batch
        {128, 4096, 4096},   // prefill: the pack amortizes over the rows
        {1024, 1024, 1024},  // square, FLOP bound: the shape class that loses
        {1, 1280, 1000},     // MobileNetV2's logits
        {512, 512, 512},     // 1 MB of panels — fits L2, so the arms tie
        {2048, 2048, 2048},  // 16 MB — the loss grows with the working set
    };
    bench_prepack_vs_raw<float>("f32", shapes, {1, 4}, /*target_ms=*/100.0,
                                /*rounds=*/3);
    bench_prepack_vs_raw<half>("f16", shapes, {1, 4}, /*target_ms=*/100.0,
                               /*rounds=*/3);
}

// ============================================================
// f16 — the same pairing
// ============================================================

NNOPS_BENCH(linear_f16_vs_matmul) {
    const std::vector<Shape> shapes{
        {1, 4096, 4096},
        {128, 4096, 4096},
        {1024, 1024, 1024},
    };
    bench_fp_vs_matmul<half>("f16", shapes, {1, 4}, /*target_ms=*/100.0,
                             /*rounds=*/3);
}

// ============================================================
// Quantized operands — u8s8 / s8s8 -> fp
// ============================================================

NNOPS_BENCH(linear_int8) {
    const std::vector<Shape> shapes{
        {1, 4096, 4096},
        {128, 4096, 4096},
        {1024, 1024, 1024},
    };
    constexpr int kRounds = 3;
    constexpr double kBudgetMs = 100.0;
    constexpr int kThreads = 1;

    std::printf("  quantized Linear vs MatMul(tb), per-token X x per-channel W"
                "  %dT  budget=%.0f ms/cell  rounds=%d\n", kThreads, kBudgetMs,
                kRounds);
    std::printf("   %5s %5s %5s  %-7s    %-3s | %9s %9s %5s |\n", "M", "K", "N",
                "dtype", "out", "Linear", "MatMul", "L/M");

    for (const Shape& s : shapes) {
        bench_quant_row<float>(s, DataType::s8, DataType::f32, kThreads,
                               kBudgetMs, kRounds);
        bench_quant_row<float>(s, DataType::u8, DataType::f32, kThreads,
                               kBudgetMs, kRounds);
        bench_quant_row<half>(s, DataType::s8, DataType::f16, kThreads,
                              kBudgetMs, kRounds);
    }
}

// ============================================================
// Quantized operands with an integer destination — s32 / s8 / u8
// ============================================================
//
// The output dtype decides the closing pass: s32 adds the int32 bias to the
// compensated accumulate (the reductions stay off when the zero points are
// symmetric); s8/u8 requantize with the output's QuantParams, so the reduction
// sweep runs and the per-element walk does the round-and-clamp. f32 is the
// dequantizing baseline; every row carries a bias, the input these routes fuse.

NNOPS_BENCH(linear_int8_int_out) {
    const std::vector<Shape> shapes{
        {1, 4096, 4096},     // decode: the output pass is the largest share
        {128, 4096, 4096},   // prefill: the GEMM dominates
    };
    constexpr int kRounds = 3;
    constexpr double kBudgetMs = 100.0;
    constexpr int kThreads = 1;

    std::printf("  quantized Linear, integer destinations vs MatMul(tb)"
                "  %dT  budget=%.0f ms/cell  rounds=%d\n", kThreads, kBudgetMs,
                kRounds);
    std::printf("   %5s %5s %5s  %-7s    %-6s | %9s %9s %5s |\n", "M", "K", "N",
                "dtype", "out", "Linear", "MatMul", "L/M");

    for (const Shape& s : shapes) {
        bench_quant_row<float>(s, DataType::s8, DataType::f32, kThreads,
                               kBudgetMs, kRounds, /*with_bias=*/true);
        bench_quant_row<int32_t>(s, DataType::s8, DataType::s32, kThreads,
                                 kBudgetMs, kRounds, /*with_bias=*/true);
        bench_quant_row<int8_t>(s, DataType::s8, DataType::s8, kThreads,
                                kBudgetMs, kRounds, /*with_bias=*/true);
        bench_quant_row<uint8_t>(s, DataType::s8, DataType::u8, kThreads,
                                 kBudgetMs, kRounds, /*with_bias=*/true);
    }
}

// ============================================================
// Thread scaling of the plain call
// ============================================================

NNOPS_BENCH(linear_threads) {
    // The GEMM splits over (batch, n-block) or (batch, m-block); at M=1 the only
    // axis left is N, so decode rows scale with the column-block count and stop
    // when the weight stream saturates.
    const std::vector<Shape> shapes{
        {1, 4096, 4096},
        {8, 4096, 4096},
        {128, 4096, 4096},
        {1024, 1024, 1024},
    };
    const std::vector<int> threads = thread_ladder();
    constexpr double kBudgetMs = 100.0;

    std::printf("  f32 Linear, plain  1..N threads  budget=%.0f ms/cell\n", kBudgetMs);
    std::printf("   %5s %5s %5s |", "M", "K", "N");
    for (int nt : threads) { std::printf(" %7s", (std::to_string(nt) + "T").c_str()); }
    std::printf("   (GFLOPS)\n");

    for (const Shape& s : shapes) {
        FpProblem<float> p(s, LinearAttributes{}, /*with_bias=*/false);
        std::printf("   %5lld %5lld %5lld |", static_cast<long long>(s.M),
                    static_cast<long long>(s.K), static_cast<long long>(s.N));
        for (int nt : threads) {
            BenchPool pool(nt);
            ComputeContext ctx;
            if (nt > 1) { ctx.cpu = pool.cpu; }
            std::printf(" %7.1f", p.run(*p.linear, ctx, kBudgetMs));
        }
        std::printf("\n");
    }
}
