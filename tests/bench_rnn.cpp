/// @file bench_rnn.cpp
/// @brief Recurrent-family benchmark — RNN, LSTM and GRU, f32, f16 and bf16.
///
/// Two numbers matter for a recurrent op and they pull in opposite directions:
///
///  - GFLOPS, the usual throughput reading — 2·T·D·B·G·H·(I+H) MAC-flops, which
///    is the whole op including the chunked input projection;
///  - ms per timestep, the latency a decoding loop actually feels, since a
///    timestep cannot start before the previous one's recurrent product lands.
///
/// A case with a large T and a small B is dominated by that dependency chain (the
/// projection streams out of cache and the GEMM tiles are short); a large B with
/// a small H shifts it towards the projection. The shapes below cover both, plus
/// the widths where H is past the k-block cap and the recurrent product splits.
///
/// BenchPool carries the two arms of common/cpu_pool.hpp: an OpenMP team when
/// the build has one (unless NNOPS_BENCH_OMP opts out), else a pool that spawns
/// per call.

#include "nnops/ops/gru.hpp"
#include "nnops/ops/lstm.hpp"
#include "nnops/ops/rnn.hpp"
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
#include <thread>
#include <type_traits>
#include <vector>

using namespace nnops;
using nnops::simd::s_load;
using nnops::simd::s_store;
using nnops::backend::cpu::bfloat16;
namespace cpu = nnops::backend::cpu;
using cpu::half;

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

/// The element type a run_recurrent<T> instantiation runs in.
template <typename T>
constexpr DataType dt_of() noexcept {
    if constexpr (std::is_same_v<T, float>) {
        return DataType::f32;
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        return DataType::bf16;
    } else {
        return DataType::f16;
    }
}

template <typename T>
constexpr const char* dt_name() noexcept {
    if constexpr (std::is_same_v<T, float>) {
        return "f32";
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        return "bf16";
    } else {
        return "f16";
    }
}

template <typename T>
auto make_random(std::span<const int64_t> shape, uint64_t seed) {
    if constexpr (std::is_same_v<T, float>) {
        return test::make_random_tensor(shape, -1.0f, 1.0f, seed);
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        return test::make_random_bf16_tensor(shape, -1.0f, 1.0f, seed);
    } else {
        return test::make_random_f16_tensor(shape, -1.0f, 1.0f, seed);
    }
}

/// The weights as a decode loop would hold them: prepackWeights() once, before
/// the clock, and only the blob handed to compute(). The pack is a per-call cost
/// otherwise, and at T=1 it is most of the call — every row here is the steady
/// state, which is the only number a caller can act on.
struct PackedBlob {
    std::vector<std::byte> buf;
    TensorView view;
};

inline PackedBlob prepack(OpBase& op, std::span<const TensorView> ins) {
    PackedBlob b;
    TensorView query{};
    op.prepackWeights(ins, std::span<TensorView>(&query, 1), {});
    // The answer carries no storage, so rank — not is_empty() — separates a
    // blob from "no packed form".
    if (query.rank() == 0) { return b; }
    b.buf.assign(static_cast<size_t>(query.numel()) *
                     static_cast<size_t>(data_type_size(query.data_type())) + 64,
                 std::byte{0});
    const int64_t shape[1] = {query.numel()};
    b.view = TensorView(std::span<const int64_t>(shape, 1), query.data_type(),
                        b.buf.data(), TensorLayout::PackedRecurrent);
    TensorView fill[1] = {b.view};
    op.prepackWeights(ins, fill, {});
    b.view = fill[0];
    return b;
}

struct RecurrentCase {
    const char* name;
    RecurrentMode mode = RecurrentMode::Lstm;
    int64_t T = 32, B = 1, I = 128, H = 128;
    RnnDirection dir = RnnDirection::Forward;
    bool f16 = false;
    bool lbr = false;  ///< GRU linear_before_reset: one pass per step instead of two
    /// s8 W and R with one symmetric scale per gate row; X stays floating, so
    /// the kernel quantizes every activation row (the W8A8 route).
    bool q8 = false;
    /// bf16 arm: the BFMMLA route, distinct from f16's (both are 16-bit, and
    /// on aarch64 each has its own packed-B layout). Last so the positional
    /// initializers above keep their meaning.
    bool bf16 = false;
};

/// W or R as a real quantization would leave it: each row rounded to s8 against
/// its own max, so the quantized route sees the same weights the fp route does
/// (up to the rounding) and the two timings are comparable.
template <typename T>
void quantize_rows(const std::vector<T>& src, int64_t rows, int64_t cols,
                   std::vector<int8_t>& q, std::vector<float>& sc) {
    q.resize(src.size());
    sc.assign(static_cast<size_t>(rows), 1.0f);
    for (int64_t r = 0; r < rows; ++r) {
        const T* row = src.data() + r * cols;
        float mx = 0.0f;
        for (int64_t k = 0; k < cols; ++k) {
            mx = std::max(mx, std::fabs(s_load(row + k)));
        }
        const float s = mx > 0.0f ? mx / 127.0f : 1.0f;
        sc[static_cast<size_t>(r)] = s;
        for (int64_t k = 0; k < cols; ++k) {
            const float v = std::min(std::max(s_load(row + k) / s, -127.0f), 127.0f);
            q[static_cast<size_t>(r * cols + k)] = static_cast<int8_t>(std::lround(v));
        }
    }
}

template <typename T>
void run_recurrent(const RecurrentCase& c, const std::vector<int>& threads, int iters) {
    const int64_t G = recurrent_gates(c.mode);
    const int64_t D = rnn_num_directions(c.dir);

    const int64_t xshape[] = {c.T, c.B, c.I};
    const int64_t wshape[] = {D, G * c.H, c.I};
    const int64_t rshape[] = {D, G * c.H, c.H};
    const int64_t bshape[] = {D, 2 * G * c.H};
    const int64_t hshape[] = {D, c.B, c.H};
    const int64_t yshape[] = {c.T, D, c.B, c.H};

    auto [x_vec, x] = make_random<T>(xshape, 11);
    auto [w_vec, w] = make_random<T>(wshape, 12);
    auto [r_vec, r] = make_random<T>(rshape, 13);
    auto [b_vec, b] = make_random<T>(bshape, 14);
    auto [h_vec, h0] = make_random<T>(hshape, 15);
    auto [c_vec, c0] = make_random<T>(hshape, 16);

    const bool lstm = c.mode == RecurrentMode::Lstm;

    // The quantized route's storage and views. Declared before `ins` so the
    // views' scale pointers outlive every compute call.
    std::vector<int8_t> wq, rq;
    std::vector<float> wsc, rsc;
    QuantParams qw, qr;
    TensorView w_view = w;
    TensorView r_view = r;
    if (c.q8) {
        quantize_rows(w_vec, D * G * c.H, c.I, wq, wsc);
        quantize_rows(r_vec, D * G * c.H, c.H, rq, rsc);
        qw.granularity = QuantGranularity::PerToken;
        qw.scale_data = wsc.data();
        qw.num_scales = D * G * c.H;
        qr = qw;
        qr.scale_data = rsc.data();
        w_view = TensorView(std::span<const int64_t>(wshape, 3), DataType::s8,
                            wq.data(), TensorLayout::NCHW, qw);
        r_view = TensorView(std::span<const int64_t>(rshape, 3), DataType::s8,
                            rq.data(), TensorLayout::NCHW, qr);
    }

    std::vector<TensorView> ins = {x, w_view, r_view, b, TensorView{}, h0};
    if (lstm) { ins.push_back(c0); }

    std::vector<T> y_buf(static_cast<size_t>(c.T * D * c.B * c.H));
    std::vector<T> yh_buf(static_cast<size_t>(D * c.B * c.H));
    std::vector<T> yc_buf(static_cast<size_t>(D * c.B * c.H));
    const DataType dt = dt_of<T>();
    std::vector<TensorView> outs = {
        TensorView(std::span<const int64_t>(yshape, 4), dt, y_buf.data()),
        TensorView(std::span<const int64_t>(hshape, 3), dt, yh_buf.data()),
    };
    if (lstm) {
        outs.push_back(TensorView(std::span<const int64_t>(hshape, 3), dt, yc_buf.data()));
    }

    std::unique_ptr<OpBase> op;
    if (c.mode == RecurrentMode::Lstm) {
        LstmAttributes a;
        a.hidden_size = c.H;
        a.direction = c.dir;
        op = Lstm::create(a, Backend::CPU);
    } else if (c.mode == RecurrentMode::Gru) {
        GruAttributes a;
        a.hidden_size = c.H;
        a.direction = c.dir;
        a.linear_before_reset = c.lbr;
        op = Gru::create(a, Backend::CPU);
    } else {
        RnnAttributes a;
        a.hidden_size = c.H;
        a.direction = c.dir;
        op = Rnn::create(a, Backend::CPU);
    }
    const PackedBlob blob = prepack(*op, ins);
    ins[1] = blob.view;

    // Every gate column of every batch row of every timestep, input and
    // recurrent halves: 2 flops per MAC.
    const double flops_per_iter = 2.0 * static_cast<double>(c.T * D * c.B * G * c.H) *
                                  static_cast<double>(c.I + c.H);

    std::printf("  %s\n", c.name);
    std::printf("    T=%lld B=%lld I=%lld H=%lld  G=%lld D=%lld  %s%s\n",
                (long long)c.T, (long long)c.B, (long long)c.I, (long long)c.H,
                (long long)G, (long long)D, c.bf16 ? "bf16" : c.f16 ? "f16" : "f32",
                c.q8 ? " W8A8" : "");

    auto compute = [&](const ComputeContext& ctx) { op->compute(outs, ins, ctx, nullptr); };

    std::vector<double> gflops, ms_per_step;
    for (int nt : threads) {
        BenchPool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

        for (int i = 0; i < 2; ++i) { compute(use_ctx); }

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) { compute(use_ctx); }
        auto end = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        const double per = ms / static_cast<double>(iters);
        gflops.push_back(flops_per_iter * static_cast<double>(iters) / (ms * 1e6));
        ms_per_step.push_back(per / static_cast<double>(c.T));
    }

    const double g1 = gflops.empty() ? 0.0 : gflops[0];
    std::printf("    ");
    for (size_t i = 0; i < threads.size(); ++i) {
        const double speedup = g1 > 0.0 ? gflops[i] / g1 : 0.0;
        std::printf(" %2dT %7.1f GF (%.2fx) %7.3f ms/step", threads[i], gflops[i],
                    speedup, ms_per_step[i]);
        if (i + 1 < threads.size()) { std::printf(" |"); }
    }
    std::printf("\n");
}

void bench_case(const RecurrentCase& c) {
    const int iters = 20;
    std::vector<int> threads{1, 2, 4};
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw > 1 && std::find(threads.begin(), threads.end(), hw) == threads.end()) {
        threads.push_back(hw);
    }
    if (c.bf16)     { run_recurrent<bfloat16>(c, threads, iters); }
    else if (c.f16) { run_recurrent<half>(c, threads, iters); }
    else            { run_recurrent<float>(c, threads, iters); }
}

}  // namespace

// ============================================================
// LSTM — the widest gate tile (4·H columns)
// ============================================================

NNOPS_BENCH(lstm_f32_small) {
    // One sentence at a time: the recurrent product is the whole cost.
    bench_case({"lstm f32 T32 B1 I128 H128", RecurrentMode::Lstm, 32, 1, 128, 128});
}

NNOPS_BENCH(lstm_f32_batched) {
    // Batched: the same T with 8 rows, so the projection and the recurrent tile
    // both have full m.
    bench_case({"lstm f32 T32 B8 I128 H128", RecurrentMode::Lstm, 32, 8, 128, 128});
}

NNOPS_BENCH(lstm_f32_wide) {
    // H past the 128-column k-block cap: the recurrent product splits in k.
    bench_case({"lstm f32 T16 B2 I256 H256", RecurrentMode::Lstm, 16, 2, 256, 256});
}

NNOPS_BENCH(lstm_f32_bidirectional) {
    bench_case({"lstm f32 bidir T32 B2 I128 H128", RecurrentMode::Lstm, 32, 2, 128, 128,
                RnnDirection::Bidirectional});
}

NNOPS_BENCH(lstm_f32_long) {
    // A long sequence, B=1: 256 steps of dependent work in a row.
    bench_case({"lstm f32 T256 B1 I64 H256", RecurrentMode::Lstm, 256, 1, 64, 256});
}

NNOPS_BENCH(lstm_f16_small) {
    bench_case({"lstm f16 T32 B1 I128 H128", RecurrentMode::Lstm, 32, 1, 128, 128,
                RnnDirection::Forward, /*f16=*/true});
}

NNOPS_BENCH(lstm_f16_wide) {
    bench_case({"lstm f16 T16 B2 I256 H256", RecurrentMode::Lstm, 16, 2, 256, 256,
                RnnDirection::Forward, /*f16=*/true});
}

#ifdef NNOPS_ARCH_AARCH64
NNOPS_BENCH(lstm_bf16_small) {
    RecurrentCase c{"lstm bf16 T32 B1 I128 H128", RecurrentMode::Lstm, 32, 1, 128, 128};
    c.bf16 = true;
    bench_case(c);
}
#endif  // NNOPS_ARCH_AARCH64

#ifdef NNOPS_ARCH_AARCH64
NNOPS_BENCH(lstm_bf16_wide) {
    RecurrentCase c{"lstm bf16 T16 B2 I256 H256", RecurrentMode::Lstm, 16, 2, 256, 256};
    c.bf16 = true;
    bench_case(c);
}
#endif  // NNOPS_ARCH_AARCH64

// ============================================================
// GRU — two phases per step, 3·H columns each
// ============================================================

NNOPS_BENCH(gru_f32_small) {
    bench_case({"gru f32 T32 B1 I128 H128", RecurrentMode::Gru, 32, 1, 128, 128});
}

NNOPS_BENCH(gru_f32_batched) {
    bench_case({"gru f32 T32 B8 I128 H128", RecurrentMode::Gru, 32, 8, 128, 128});
}

/// linear_before_reset = 1 against = 0, interleaved in one process: the two
/// arms take turns so a clock or cache-state drift along the run hits both. The
/// merged arm runs one recurrent pass per step where the other runs two.
template <typename T>
void ab_gru_lbr(int64_t steps, int64_t B, int64_t I, int64_t H,
                const std::vector<int>& threads, int iters, int reps) {
    constexpr int64_t G = 3, D = 1;
    const int64_t xshape[] = {steps, B, I};
    const int64_t wshape[] = {D, G * H, I};
    const int64_t rshape[] = {D, G * H, H};
    const int64_t bshape[] = {D, 2 * G * H};
    const int64_t hshape[] = {D, B, H};
    const int64_t yshape[] = {steps, D, B, H};

    auto [x_vec, x] = make_random<T>(xshape, 11);
    auto [w_vec, w] = make_random<T>(wshape, 12);
    auto [r_vec, r] = make_random<T>(rshape, 13);
    auto [b_vec, b] = make_random<T>(bshape, 14);
    auto [h_vec, h0] = make_random<T>(hshape, 15);

    std::vector<T> y_buf(static_cast<size_t>(steps * D * B * H));
    std::vector<T> yh_buf(static_cast<size_t>(D * B * H));
    const DataType dt = dt_of<T>();
    std::vector<TensorView> outs = {
        TensorView(std::span<const int64_t>(yshape, 4), dt, y_buf.data()),
        TensorView(std::span<const int64_t>(hshape, 3), dt, yh_buf.data()),
    };
    std::vector<TensorView> ins = {x, w, r, b, TensorView{}, h0};

    GruAttributes merged;
    merged.hidden_size = H;
    merged.linear_before_reset = true;
    GruAttributes split;
    split.hidden_size = H;
    auto op_merged = Gru::create(merged, Backend::CPU);
    auto op_split = Gru::create(split, Backend::CPU);

    std::printf("  gru lbr vs 2-pass  T=%lld B=%lld I=%lld H=%lld\n",
                (long long)steps, (long long)B, (long long)I, (long long)H);

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    auto time_it = [&](OpBase& op, const ComputeContext& ctx) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) { op.compute(outs, ins, ctx, nullptr); }
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() /
               static_cast<double>(iters);
    };

    std::printf("    ");
    for (size_t i = 0; i < threads.size(); ++i) {
        BenchPool pool(threads[i]);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use_ctx = (threads[i] > 1) ? ctx : ComputeContext{};
        time_it(*op_merged, use_ctx);
        time_it(*op_split, use_ctx);

        std::vector<double> tm, ts;
        for (int rep = 0; rep < reps; ++rep) {
            if (rep & 1) {
                ts.push_back(time_it(*op_split, use_ctx));
                tm.push_back(time_it(*op_merged, use_ctx));
            } else {
                tm.push_back(time_it(*op_merged, use_ctx));
                ts.push_back(time_it(*op_split, use_ctx));
            }
        }
        const double m = median(tm), sp = median(ts);
        const double per = 1e3 / static_cast<double>(steps);
        std::printf(" %2dT %6.2f vs %6.2f us/step (%.2fx)%s", threads[i], m * per,
                    sp * per, sp / m, (i + 1 < threads.size()) ? " |" : "\n");
    }
}

NNOPS_BENCH(gru_lbr_ab) {
    const std::vector<int> threads{1, 2, 4, 8};
    ab_gru_lbr<float>(32, 1, 128, 128, threads, 20, 9);
    ab_gru_lbr<float>(32, 8, 128, 128, threads, 20, 9);
    ab_gru_lbr<float>(64, 1, 64, 256, threads, 20, 9);
}

NNOPS_BENCH(gru_f32_bidirectional) {
    bench_case({"gru f32 bidir T32 B2 I128 H128", RecurrentMode::Gru, 32, 2, 128, 128,
                RnnDirection::Bidirectional});
}

NNOPS_BENCH(gru_f16_small) {
    bench_case({"gru f16 T32 B1 I128 H128", RecurrentMode::Gru, 32, 1, 128, 128,
                RnnDirection::Forward, /*f16=*/true});
}

#ifdef NNOPS_ARCH_AARCH64
NNOPS_BENCH(gru_bf16_small) {
    RecurrentCase c{"gru bf16 T32 B1 I128 H128", RecurrentMode::Gru, 32, 1, 128, 128};
    c.bf16 = true;
    bench_case(c);
}
#endif  // NNOPS_ARCH_AARCH64

// ============================================================
// RNN — one gate, the cheapest epilogue
// ============================================================

NNOPS_BENCH(rnn_f32_small) {
    bench_case({"rnn f32 T32 B1 I128 H128", RecurrentMode::Rnn, 32, 1, 128, 128});
}

NNOPS_BENCH(rnn_f32_batched) {
    bench_case({"rnn f32 T32 B8 I128 H128", RecurrentMode::Rnn, 32, 8, 128, 128});
}

NNOPS_BENCH(rnn_f16_small) {
    bench_case({"rnn f16 T32 B1 I128 H128", RecurrentMode::Rnn, 32, 1, 128, 128,
                RnnDirection::Forward, /*f16=*/true});
}

#ifdef NNOPS_ARCH_AARCH64
NNOPS_BENCH(rnn_bf16_small) {
    RecurrentCase c{"rnn bf16 T32 B1 I128 H128", RecurrentMode::Rnn, 32, 1, 128, 128};
    c.bf16 = true;
    bench_case(c);
}
#endif  // NNOPS_ARCH_AARCH64

// ============================================================
// Prepacked weights — the same call, with the pack hoisted out
// ============================================================

/// Raw weights against a blob built once beforehand, the two arms taking turns
/// in one process. Only `ins[1]` differs between them, so the delta is the
/// gate interleave and the W/R packs the kernel would otherwise redo on every
/// call — which is what a decode loop pays per token.
template <typename T>
void ab_rnn_prepack(RecurrentMode mode, int64_t steps, int64_t B, int64_t I,
                    int64_t H, const std::vector<int>& threads, int iters,
                    int reps) {
    const int64_t G = recurrent_gates(mode);
    constexpr int64_t D = 1;
    const int64_t xshape[] = {steps, B, I};
    const int64_t wshape[] = {D, G * H, I};
    const int64_t rshape[] = {D, G * H, H};
    const int64_t bshape[] = {D, 2 * G * H};
    const int64_t hshape[] = {D, B, H};
    const int64_t yshape[] = {steps, D, B, H};

    auto [x_vec, x] = make_random<T>(xshape, 11);
    auto [w_vec, w] = make_random<T>(wshape, 12);
    auto [r_vec, r] = make_random<T>(rshape, 13);
    auto [b_vec, b] = make_random<T>(bshape, 14);
    auto [h_vec, h0] = make_random<T>(hshape, 15);
    auto [c_vec, c0] = make_random<T>(hshape, 16);

    std::vector<T> y_buf(static_cast<size_t>(steps * D * B * H));
    std::vector<T> yh_buf(static_cast<size_t>(D * B * H));
    std::vector<T> yc_buf(static_cast<size_t>(D * B * H));
    const DataType dt = dt_of<T>();
    std::vector<TensorView> ins = {x, w, r, b, TensorView{}, h0};
    std::vector<TensorView> outs = {
        TensorView(std::span<const int64_t>(yshape, 4), dt, y_buf.data()),
        TensorView(std::span<const int64_t>(hshape, 3), dt, yh_buf.data()),
    };
    if (mode == RecurrentMode::Lstm) {
        ins.push_back(c0);
        outs.push_back(TensorView(std::span<const int64_t>(hshape, 3), dt, yc_buf.data()));
    }

    std::unique_ptr<OpBase> op;
    if (mode == RecurrentMode::Lstm) {
        LstmAttributes a;
        a.hidden_size = H;
        op = Lstm::create(a, Backend::CPU);
    } else if (mode == RecurrentMode::Gru) {
        GruAttributes a;
        a.hidden_size = H;
        op = Gru::create(a, Backend::CPU);
    } else {
        RnnAttributes a;
        a.hidden_size = H;
        op = Rnn::create(a, Backend::CPU);
    }

    // One pack, outside every timing loop: that is the whole contract.
    TensorView q{};
    op->prepackWeights(ins, std::span<TensorView>(&q, 1), {});
    std::vector<T> blob(static_cast<size_t>(q.numel()), T{});
    const int64_t pkshape[1] = {q.numel()};
    TensorView packed(std::span<const int64_t>(pkshape, 1), dt, blob.data(),
                      TensorLayout::PackedRecurrent);
    op->prepackWeights(ins, std::span<TensorView>(&packed, 1), {});
    std::vector<TensorView> ins_packed = ins;
    ins_packed[1] = packed;

    const char* name = mode == RecurrentMode::Lstm ? "lstm"
                     : mode == RecurrentMode::Gru  ? "gru" : "rnn";
    std::printf("  %s prepack  T=%lld B=%lld I=%lld H=%lld  %s (blob %lld elems)\n",
                name, (long long)steps, (long long)B,
                (long long)I, (long long)H,
                dt_name<T>(), (long long)q.numel());

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    auto time_it = [&](std::span<const TensorView> in, const ComputeContext& ctx) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) { op->compute(outs, in, ctx, nullptr); }
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() /
               static_cast<double>(iters);
    };

    std::printf("    ");
    for (size_t i = 0; i < threads.size(); ++i) {
        BenchPool pool(threads[i]);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use_ctx = (threads[i] > 1) ? ctx : ComputeContext{};
        time_it(ins, use_ctx);
        (void)time_it(ins_packed, use_ctx);

        std::vector<double> raw, pk;
        for (int rep = 0; rep < reps; ++rep) {
            if (rep & 1) {
                pk.push_back(time_it(ins_packed, use_ctx));
                raw.push_back(time_it(ins, use_ctx));
            } else {
                raw.push_back(time_it(ins, use_ctx));
                pk.push_back(time_it(ins_packed, use_ctx));
            }
        }
        const double a = median(raw), c = median(pk);
        std::printf(" %2dT %7.1f vs %7.1f us/call (%.2fx)%s", threads[i], a * 1e3,
                    c * 1e3, a / c, (i + 1 < threads.size()) ? " |" : "\n");
    }
}

NNOPS_BENCH(rnn_prepack_ab) {
    const std::vector<int> threads{1, 4};
    // A decode step: the pack is most of the call, so this is where hoisting it
    // out is worth the most.
    ab_rnn_prepack<float>(RecurrentMode::Lstm, 1, 1, 256, 256, threads, 50, 9);
    ab_rnn_prepack<float>(RecurrentMode::Gru, 1, 1, 256, 256, threads, 50, 9);
    // Short sequences, then one long enough that the pack is a minority.
    ab_rnn_prepack<float>(RecurrentMode::Lstm, 8, 1, 256, 256, threads, 20, 9);
    ab_rnn_prepack<float>(RecurrentMode::Lstm, 32, 1, 128, 128, threads, 20, 9);
    ab_rnn_prepack<float>(RecurrentMode::Lstm, 32, 8, 128, 128, threads, 20, 9);
    ab_rnn_prepack<half>(RecurrentMode::Lstm, 32, 1, 256, 256, threads, 20, 9);
}

// ============================================================
// W8A8 — s8 weights, activations quantized inside the kernel
// ============================================================

NNOPS_BENCH(q8_lstm_decode) {
    // One timestep, one row: the decode step, where both products are short and
    // the int8 route's per-call setup is the whole question.
    bench_case({"lstm T1 B1 I256 H256 W8A8", RecurrentMode::Lstm, 1, 1, 256, 256,
                RnnDirection::Forward, /*f16=*/false, /*lbr=*/false, /*q8=*/true});
}

NNOPS_BENCH(q8_lstm_batched) {
    // Batched and long enough that the tile has full m: the case where the
    // narrower int8 panels can actually pay.
    bench_case({"lstm T32 B8 I128 H128 W8A8", RecurrentMode::Lstm, 32, 8, 128, 128,
                RnnDirection::Forward, false, false, /*q8=*/true});
}

NNOPS_BENCH(q8_lstm_wide) {
    // H past the 256-value int8 k cap: the recurrent product splits in k.
    bench_case({"lstm T16 B2 I256 H520 W8A8", RecurrentMode::Lstm, 16, 2, 256, 520,
                RnnDirection::Forward, false, false, /*q8=*/true});
}

NNOPS_BENCH(q8_gru_batched) {
    // Two phases per step, each with its own packed R panels.
    bench_case({"gru T32 B8 I128 H128 W8A8", RecurrentMode::Gru, 32, 8, 128, 128,
                RnnDirection::Forward, false, /*lbr=*/true, /*q8=*/true});
}

NNOPS_BENCH(q8_lstm_f16_compute) {
    // s8 weights, f16 gate arithmetic and state: the mixed reading.
    bench_case({"lstm f16 T32 B8 I128 H128 W8A8", RecurrentMode::Lstm, 32, 8, 128, 128,
                RnnDirection::Forward, /*f16=*/true, false, /*q8=*/true});
}

#ifdef NNOPS_ARCH_AARCH64
NNOPS_BENCH(q8_lstm_bf16_compute) {
    RecurrentCase c{"lstm bf16 T32 B8 I128 H128 W8A8", RecurrentMode::Lstm, 32, 8, 128, 128};
    c.q8 = true;
    c.bf16 = true;
    bench_case(c);
}
#endif  // NNOPS_ARCH_AARCH64

/// The fp arm against the W8A8 arm, interleaved in one process on the same
/// weights. The only difference is the two products' arithmetic, so the ratio is
/// what quantizing the layer buys end to end — including the per-step
/// quantize-and-dequantize passes the int8 route adds.
template <typename T>
void ab_rnn_q8(RecurrentMode mode, int64_t steps, int64_t B, int64_t I, int64_t H,
               const std::vector<int>& threads, int iters, int reps) {
    const int64_t G = recurrent_gates(mode);
    constexpr int64_t D = 1;
    const int64_t xshape[] = {steps, B, I};
    const int64_t wshape[] = {D, G * H, I};
    const int64_t rshape[] = {D, G * H, H};
    const int64_t bshape[] = {D, 2 * G * H};
    const int64_t hshape[] = {D, B, H};
    const int64_t yshape[] = {steps, D, B, H};

    auto [x_vec, x] = make_random<T>(xshape, 11);
    auto [w_vec, w] = make_random<T>(wshape, 12);
    auto [r_vec, r] = make_random<T>(rshape, 13);
    auto [b_vec, b] = make_random<T>(bshape, 14);
    auto [h_vec, h0] = make_random<T>(hshape, 15);
    auto [c_vec, c0] = make_random<T>(hshape, 16);

    std::vector<int8_t> wq, rq;
    std::vector<float> wsc, rsc;
    quantize_rows(w_vec, D * G * H, I, wq, wsc);
    quantize_rows(r_vec, D * G * H, H, rq, rsc);
    QuantParams qw, qr;
    qw.granularity = QuantGranularity::PerToken;
    qw.scale_data = wsc.data();
    qw.num_scales = D * G * H;
    qr = qw;
    qr.scale_data = rsc.data();
    const TensorView w8w(std::span<const int64_t>(wshape, 3), DataType::s8, wq.data(),
                         TensorLayout::NCHW, qw);
    const TensorView r8w(std::span<const int64_t>(rshape, 3), DataType::s8, rq.data(),
                         TensorLayout::NCHW, qr);

    std::vector<T> y_buf(static_cast<size_t>(steps * D * B * H));
    std::vector<T> yh_buf(static_cast<size_t>(D * B * H));
    std::vector<T> yc_buf(static_cast<size_t>(D * B * H));
    const DataType dt = dt_of<T>();
    std::vector<TensorView> outs = {
        TensorView(std::span<const int64_t>(yshape, 4), dt, y_buf.data()),
        TensorView(std::span<const int64_t>(hshape, 3), dt, yh_buf.data()),
    };
    std::vector<TensorView> ins_fp = {x, w, r, b, TensorView{}, h0};
    std::vector<TensorView> ins_q8 = {x, w8w, r8w, b, TensorView{}, h0};
    if (mode == RecurrentMode::Lstm) {
        ins_fp.push_back(c0);
        ins_q8.push_back(c0);
        outs.push_back(TensorView(std::span<const int64_t>(hshape, 3), dt, yc_buf.data()));
    }

    std::unique_ptr<OpBase> op;
    if (mode == RecurrentMode::Lstm) {
        LstmAttributes a;
        a.hidden_size = H;
        op = Lstm::create(a, Backend::CPU);
    } else if (mode == RecurrentMode::Gru) {
        GruAttributes a;
        a.hidden_size = H;
        op = Gru::create(a, Backend::CPU);
    } else {
        RnnAttributes a;
        a.hidden_size = H;
        op = Rnn::create(a, Backend::CPU);
    }

    const char* name = mode == RecurrentMode::Lstm ? "lstm"
                     : mode == RecurrentMode::Gru  ? "gru" : "rnn";
    // Both arms prepacked, so the ratio is the two routes' arithmetic and not
    // the two routes' packers — a per-call pack is 4-8x a one-step call.
    const PackedBlob blob_fp = prepack(*op, ins_fp);
    ins_fp[1] = blob_fp.view;
    const PackedBlob blob_q8 = prepack(*op, ins_q8);
    ins_q8[1] = blob_q8.view;
    std::printf("  %s fp vs W8A8  T=%lld B=%lld I=%lld H=%lld\n", name,
                (long long)steps, (long long)B, (long long)I, (long long)H);

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    auto time_it = [&](std::span<const TensorView> in, const ComputeContext& ctx) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) { op->compute(outs, in, ctx, nullptr); }
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() /
               static_cast<double>(iters);
    };

    std::printf("    ");
    for (size_t i = 0; i < threads.size(); ++i) {
        BenchPool pool(threads[i]);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use_ctx = (threads[i] > 1) ? ctx : ComputeContext{};
        time_it(ins_fp, use_ctx);
        time_it(ins_q8, use_ctx);

        std::vector<double> tf, tq;
        for (int rep = 0; rep < reps; ++rep) {
            if (rep & 1) {
                tq.push_back(time_it(ins_q8, use_ctx));
                tf.push_back(time_it(ins_fp, use_ctx));
            } else {
                tf.push_back(time_it(ins_fp, use_ctx));
                tq.push_back(time_it(ins_q8, use_ctx));
            }
        }
        const double f = median(tf), q = median(tq);
        std::printf(" %2dT fp %7.1f vs q8 %7.1f us/call (%.2fx)%s", threads[i],
                    f * 1e3, q * 1e3, f / q, (i + 1 < threads.size()) ? " |" : "\n");
    }
}

NNOPS_BENCH(rnn_q8_ab) {
    const std::vector<int> threads{1, 4};
    ab_rnn_q8<float>(RecurrentMode::Lstm, 1, 1, 256, 256, threads, 50, 9);
    ab_rnn_q8<float>(RecurrentMode::Lstm, 8, 1, 256, 256, threads, 20, 9);
    ab_rnn_q8<float>(RecurrentMode::Lstm, 32, 8, 128, 128, threads, 20, 9);
    ab_rnn_q8<float>(RecurrentMode::Lstm, 16, 2, 256, 520, threads, 20, 9);
    ab_rnn_q8<float>(RecurrentMode::Gru, 32, 1, 128, 128, threads, 20, 9);
    ab_rnn_q8<float>(RecurrentMode::Rnn, 32, 8, 128, 128, threads, 20, 9);
    ab_rnn_q8<half>(RecurrentMode::Lstm, 32, 8, 128, 128, threads, 20, 9);
}
