/// @file bench_cuda_rnn.cpp
/// @brief CUDA recurrent-family benchmark — RNN, GRU and LSTM, f32 and f16.
///
/// A recurrent layer's time is not one number. The input projection is a real
/// GEMM and scales with T·B, but the recurrent product and the gate pass are
/// serial in t, so a decode row (T = 1, small B) is a latency chain of a
/// hundred small launches and a prefill row (large T) is dominated by the
/// projection. Each row therefore reports both the whole op's GFLOPS — using
/// 2·T·D·B·G·H·(I+H), the same count bench_rnn uses — and microseconds per
/// timestep per direction, which is what a generation loop actually waits on.
///
/// The device route keeps the CPU decomposition: one chunked projection GEMM,
/// then one recurrent GEMM and one elementwise gate pass per phase per
/// timestep. A GRU without linear_before_reset runs two phases, so its step
/// count is twice the others'; that is visible in the per-step column and is
/// the honest number for that reading.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/gru.hpp"
#include "nnops/ops/lstm.hpp"
#include "detail/bench_knobs.hpp"
#include "nnops/ops/matmul.hpp"
#include "nnops/ops/rnn.hpp"
#include "common/bench_harness.hpp"
#include "common/cuda_bench_helper.hpp"
#include "common/cuda_test_helper.hpp"

#include <cstdio>
#include <string>
#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

struct Geometry {
    int64_t T = 1, B = 1, I = 256, H = 256;
    int64_t D = 1;
    DataType dt = DataType::f32;
};

std::string geometry_text(RecurrentMode mode, const Geometry& g)
{
    const char* name = mode == RecurrentMode::Rnn  ? "RNN" :
                       mode == RecurrentMode::Gru  ? "GRU" :
                                                     "LSTM";
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s T=%lld B=%lld I=%lld H=%lld D=%lld %s", name,
                  static_cast<long long>(g.T), static_cast<long long>(g.B),
                  static_cast<long long>(g.I), static_cast<long long>(g.H),
                  static_cast<long long>(g.D),
                  g.dt == DataType::f32 ? "f32" : "f16");
    return std::string(buf);
}

/// One timed run. The attributes are the caller's, so a row means exactly what
/// its label says (an LSTM with input_forget, a GRU with linear_before_reset).
template <class OpT, class AttrT>
double run_row(RecurrentMode mode, const AttrT& attrs, const Geometry& g, const char* tag,
               bool print = true)
{
    const int64_t G = recurrent_gates(mode);
    const bool lstm = mode == RecurrentMode::Lstm;

    std::vector<int64_t> x_shape = {g.T, g.B, g.I};
    std::vector<int64_t> w_shape = {g.D, G * g.H, g.I};
    std::vector<int64_t> r_shape = {g.D, G * g.H, g.H};
    std::vector<int64_t> b_shape = {g.D, 2 * G * g.H};
    std::vector<int64_t> h_shape = {g.D, g.B, g.H};
    std::vector<int64_t> y_shape = {g.T, g.D, g.B, g.H};

    HostTensor hx = make_host_random(x_shape, g.dt, false, 11, -0.5f, 0.5f);
    HostTensor hw = make_host_random(w_shape, g.dt, false, 13, -0.25f, 0.25f);
    HostTensor hr = make_host_random(r_shape, g.dt, false, 17, -0.25f, 0.25f);
    HostTensor hb = make_host_random(b_shape, g.dt, false, 19, -0.25f, 0.25f);
    HostTensor hy = make_host_zeroed(y_shape, g.dt, false);
    HostTensor hyh = make_host_zeroed(h_shape, g.dt, false);
    HostTensor hyc;
    if (lstm) {
        hyc = make_host_zeroed(h_shape, g.dt, false);
    }

    DeviceTensor dx = to_device(hx);
    DeviceTensor dw = to_device(hw);
    DeviceTensor dr = to_device(hr);
    DeviceTensor db = to_device(hb);
    DeviceTensor dy = to_device(hy);
    DeviceTensor dyh = to_device(hyh);
    DeviceTensor dyc;
    if (lstm) {
        dyc = to_device(hyc);
    }

    auto op = OpT::create(attrs, Backend::CUDA);
    std::vector<TensorView> ins = {dx.view, dw.view, dr.view, db.view};
    std::vector<TensorView> outs;
    outs.push_back(dy.view);
    outs.push_back(dyh.view);
    if (lstm) {
        outs.push_back(dyc.view);
    }

    auto launch = [&] {
        const Status st = op->compute(outs, ins, cuda_context(), nullptr);
        if (st != Status::ok) {
            throw std::runtime_error(std::string("cuda compute: ") + to_string(st));
        }
    };

    const double ms = cuda_median_ms(launch, 3, 11, 1);
    const double flops = 2.0 * double(g.T) * double(g.D) * double(g.B) * double(G) *
                         double(g.H) * (double(g.I) + double(g.H));
    const double steps = double(g.T) * double(g.D);
    const double us_per_step = steps > 0.0 ? ms * 1000.0 / steps : 0.0;

    if (print) {
        std::printf("  %-18s %-46s %9.4f ms %9.2f GFLOPS %8.2f us/step\n", tag,
                    geometry_text(mode, g).c_str(), ms, cuda_gflops(ms, flops),
                    us_per_step);
    }
    return ms;
}

LstmAttributes lstm_attrs(int64_t hidden)
{
    LstmAttributes a;
    a.hidden_size = hidden;
    return a;
}

GruAttributes gru_attrs(int64_t hidden, bool lbr)
{
    GruAttributes a;
    a.hidden_size = hidden;
    a.linear_before_reset = lbr;
    return a;
}

RnnAttributes rnn_attrs(int64_t hidden)
{
    RnnAttributes a;
    a.hidden_size = hidden;
    return a;
}

}  // namespace

NNOPS_BENCH(cuda_rnn_lstm) {
    print_cuda_header("CUDA LSTM — decode and prefill shapes");

    // Decode: one step, the recurrent GEMM is the whole op. Small batches are
    // the narrow-m pose; a larger B fills the gate-row tiles.
    for (DataType dt : {DataType::f32, DataType::f16}) {
        for (int64_t b : {1, 8, 32}) {
            Geometry g;
            g.T = 1;
            g.B = b;
            g.I = 1024;
            g.H = 1024;
            g.dt = dt;
            run_row<Lstm, LstmAttributes>(RecurrentMode::Lstm, lstm_attrs(g.H), g, "lstm");
        }
    }

    // Prefill: the chunked projection dominates and the arithmetic intensity is
    // the GEMM's rather than a vector-matrix product's.
    for (DataType dt : {DataType::f32, DataType::f16}) {
        for (int64_t t : {32, 128}) {
            Geometry g;
            g.T = t;
            g.B = 8;
            g.I = 1024;
            g.H = 1024;
            g.dt = dt;
            run_row<Lstm, LstmAttributes>(RecurrentMode::Lstm, lstm_attrs(g.H), g, "lstm");
        }
    }

    // A narrower layer, the shape a small model's first layer actually has.
    for (DataType dt : {DataType::f32, DataType::f16}) {
        Geometry g;
        g.T = 64;
        g.B = 16;
        g.I = 256;
        g.H = 256;
        g.dt = dt;
        run_row<Lstm, LstmAttributes>(RecurrentMode::Lstm, lstm_attrs(g.H), g, "lstm");
    }
}

NNOPS_BENCH(cuda_rnn_gru) {
    print_cuda_header("CUDA GRU — both reset readings");

    for (DataType dt : {DataType::f32, DataType::f16}) {
        for (bool lbr : {false, true}) {
            Geometry g;
            g.T = 1;
            g.B = 8;
            g.I = 1024;
            g.H = 1024;
            g.dt = dt;
            run_row<Gru, GruAttributes>(RecurrentMode::Gru, gru_attrs(g.H, lbr), g,
                                        lbr ? "gru lbr=1" : "gru lbr=0");
        }
    }
    for (DataType dt : {DataType::f32, DataType::f16}) {
        Geometry g;
        g.T = 64;
        g.B = 8;
        g.I = 512;
        g.H = 512;
        g.dt = dt;
        for (bool lbr : {false, true}) {
            run_row<Gru, GruAttributes>(RecurrentMode::Gru, gru_attrs(g.H, lbr), g,
                                        lbr ? "gru lbr=1" : "gru lbr=0");
        }
    }
}

NNOPS_BENCH(cuda_rnn_family) {
    print_cuda_header("CUDA recurrent family — the same geometry, three gate counts");

    // One geometry across the three operators: the gate width is 1·H, 3·H and
    // 4·H, so the GFLOPS column separates the projection's share from the
    // serial step's.
    for (DataType dt : {DataType::f32, DataType::f16}) {
        Geometry g;
        g.T = 16;
        g.B = 16;
        g.I = 512;
        g.H = 512;
        g.dt = dt;
        run_row<Rnn, RnnAttributes>(RecurrentMode::Rnn, rnn_attrs(g.H), g, "rnn");
        run_row<Gru, GruAttributes>(RecurrentMode::Gru, gru_attrs(g.H, true), g, "gru lbr=1");
        run_row<Lstm, LstmAttributes>(RecurrentMode::Lstm, lstm_attrs(g.H), g, "lstm");
    }

    // Bidirectional doubles the directions and the steps, but each direction
    // walks its own timeline, so per-step latency should be about flat.
    for (DataType dt : {DataType::f32, DataType::f16}) {
        Geometry g;
        g.T = 32;
        g.B = 8;
        g.I = 512;
        g.H = 512;
        g.D = 2;
        g.dt = dt;
        LstmAttributes a = lstm_attrs(g.H);
        a.direction = RnnDirection::Bidirectional;
        run_row<Lstm, LstmAttributes>(RecurrentMode::Lstm, a, g, "lstm bidir");
    }
}


/// The recurrent product is a short-M GEMM whose only reused operand is R, so
/// its rate is set by how cheaply the engine can stream R. The operator's own
/// layout is `[n][k]` (the transposed-B pose); this section prices that pose
/// against a one-time `[k][n]` transpose, which is the pose the narrow-m loop
/// was built for. It is the evidence behind the step route, and the price the
/// transpose has to earn back over the timeline.
NNOPS_BENCH(cuda_rnn_step_probe) {
    print_cuda_header("CUDA recurrent step — B pose A/B ([n][k] vs [k][n])");

    struct Shape {
        int64_t m, k, n;
    };
    const Shape shapes[] = {{1, 1024, 4096}, {8, 1024, 4096},  {16, 1024, 4096},
                            {32, 1024, 4096}, {8, 1024, 1024},  {8, 2048, 8192}};

    for (DataType dt : {DataType::f32, DataType::f16}) {
        for (const Shape& sh : shapes) {
            const std::vector<int64_t> a_shape = {sh.m, sh.k};
            const std::vector<int64_t> bt_shape = {sh.n, sh.k};
            const std::vector<int64_t> bk_shape = {sh.k, sh.n};
            const std::vector<int64_t> c_shape = {sh.m, sh.n};
            HostTensor ha = make_host_random(a_shape, dt, false, 11, -0.5f, 0.5f);
            HostTensor hbt = make_host_random(bt_shape, dt, false, 13, -0.25f, 0.25f);
            HostTensor hbk = make_host_random(bk_shape, dt, false, 17, -0.25f, 0.25f);
            HostTensor hc = make_host_zeroed(c_shape, dt, false);
            DeviceTensor da = to_device(ha);
            DeviceTensor dbt = to_device(hbt);
            DeviceTensor dbk = to_device(hbk);
            DeviceTensor dc = to_device(hc);

            MatMulAttributes tb;
            tb.transpose_b = true;
            MatMulAttributes tk;
            tk.transpose_b = false;
            auto op_tb = MatMul::create(tb, Backend::CUDA);
            auto op_tk = MatMul::create(tk, Backend::CUDA);
            std::vector<TensorView> ins_tb = {da.view, dbt.view};
            std::vector<TensorView> ins_tk = {da.view, dbk.view};
            TensorView out = dc.view;

            const double ms_tb = cuda_median_ms([&] { cuda_launch(*op_tb, out, ins_tb); });
            const double ms_tk = cuda_median_ms([&] { cuda_launch(*op_tk, out, ins_tk); });
            const double bytes = double(sh.n) * double(sh.k) * data_type_size(dt);

            char geo[128];
            std::snprintf(geo, sizeof(geo), "M=%lld K=%lld N=%lld %s",
                          static_cast<long long>(sh.m), static_cast<long long>(sh.k),
                          static_cast<long long>(sh.n),
                          dt == DataType::f32 ? "f32" : "f16");
            std::printf("  %-34s  [n][k] %8.4f ms (%6.1f GB/s)   [k][n] %8.4f ms (%6.1f GB/s)  %+6.1f%%\n",
                        geo, ms_tb, bytes / (ms_tb * 1e6), ms_tk, bytes / (ms_tk * 1e6),
                        100.0 * (ms_tb - ms_tk) / ms_tb);
        }
    }
}


/// The step product sits at the boundary of the engine's regimes: a handful of
/// rows, a gate row for N, and a K that is often too short for the narrow-m
/// loop. This sweeps the two knobs that change the pose for the shapes the step
/// actually poses, so the route's engine choice is a measurement rather than an
/// inheritance.
NNOPS_BENCH(cuda_rnn_step_knobs) {
    print_cuda_header("CUDA recurrent step — engine knobs");

    const auto time_pose = [](int64_t m, int64_t k, int64_t n, DataType dt) {
        const std::vector<int64_t> a_shape = {m, k};
        const std::vector<int64_t> b_shape = {n, k};
        const std::vector<int64_t> c_shape = {m, n};
        HostTensor ha = make_host_random(a_shape, dt, false, 11, -0.5f, 0.5f);
        HostTensor hb = make_host_random(b_shape, dt, false, 13, -0.25f, 0.25f);
        HostTensor hc = make_host_zeroed(c_shape, dt, false);
        DeviceTensor da = to_device(ha);
        DeviceTensor db = to_device(hb);
        DeviceTensor dc = to_device(hc);
        MatMulAttributes attrs;
        attrs.transpose_b = true;
        auto op = MatMul::create(attrs, Backend::CUDA);
        std::vector<TensorView> ins = {da.view, db.view};
        TensorView out = dc.view;
        return cuda_median_ms([&] { cuda_launch(*op, out, ins); });
    };

    auto& splitk = nnops::detail::cuda_splitk;
    auto& narrow = nnops::detail::cuda_narrow_m;
    const int split_arms[] = {-1, 0, 2, 4, 8, 16};

    struct Shape {
        int64_t m, k, n;
    };
    const Shape shapes[] = {{8, 1024, 2048}, {8, 1024, 1024}, {16, 1024, 1024},
                            {16, 1024, 512}, {1, 1024, 256},  {2, 1024, 1024}};

    for (DataType dt : {DataType::f32, DataType::f16}) {
        for (const Shape& sh : shapes) {
            std::printf("  %s M=%2lld K=%4lld N=%4lld  tile:",
                        dt == DataType::f32 ? "f32" : "f16",
                        static_cast<long long>(sh.m), static_cast<long long>(sh.k),
                        static_cast<long long>(sh.n));
            narrow.store(-1, std::memory_order_relaxed);
            for (int sp : split_arms) {
                splitk.store(sp, std::memory_order_relaxed);
                const double ms = time_pose(sh.m, sh.k, sh.n, dt);
                std::printf(" %s=%6.4f", sp < 0 ? "auto" : std::to_string(sp).c_str(), ms);
            }
            splitk.store(-1, std::memory_order_relaxed);

            if (dt == DataType::f16 && sh.m > 1) {
                narrow.store(1, std::memory_order_relaxed);
                std::printf(" | NT:");
                for (int sp : split_arms) {
                    splitk.store(sp, std::memory_order_relaxed);
                    const double ms = time_pose(sh.m, sh.k, sh.n, dt);
                    std::printf(" %s=%6.4f", sp < 0 ? "auto" : std::to_string(sp).c_str(),
                                ms);
                }
                narrow.store(-1, std::memory_order_relaxed);
                splitk.store(-1, std::memory_order_relaxed);
            }
            std::printf("\n");
        }
    }
}

/// A recurrent step is a strict dependency chain, so its cost is the kernel
/// plus whatever the chain pays between kernels. This section fits that split:
/// the same geometry at several T gives the marginal step, and the step
/// product alone (a bare MatMul of the same shape) gives the kernel. The
/// difference is the epilogue and the launch gap, which is what a fused or
/// persistent step would return.
NNOPS_BENCH(cuda_rnn_step_floor) {
    print_cuda_header("CUDA recurrent step — marginal step vs the bare product");

    struct Geom {
        int64_t b, i, h;
        DataType dt;
    };
    const Geom geoms[] = {{16, 256, 256, DataType::f32},
                          {8, 512, 512, DataType::f16},
                          {8, 1024, 1024, DataType::f16}};

    for (const Geom& gm : geoms) {
        // The bare step product: [b, h] x [G*h, h]^T.
        {
            const std::vector<int64_t> a_shape = {gm.b, gm.h};
            const std::vector<int64_t> b_shape = {4 * gm.h, gm.h};
            const std::vector<int64_t> c_shape = {gm.b, 4 * gm.h};
            HostTensor ha = make_host_random(a_shape, gm.dt, false, 11, -0.5f, 0.5f);
            HostTensor hb = make_host_random(b_shape, gm.dt, false, 13, -0.25f, 0.25f);
            HostTensor hc = make_host_zeroed(c_shape, gm.dt, false);
            DeviceTensor da = to_device(ha);
            DeviceTensor db = to_device(hb);
            DeviceTensor dc = to_device(hc);
            MatMulAttributes attrs;
            attrs.transpose_b = true;
            auto op = MatMul::create(attrs, Backend::CUDA);
            std::vector<TensorView> ins = {da.view, db.view};
            TensorView out = dc.view;
            const double gms = cuda_median_ms([&] { cuda_launch(*op, out, ins); });
            const double gms20 =
                cuda_median_ms([&] { cuda_launch(*op, out, ins); }, 2, 5, 20);
            std::printf("  LSTM B=%2lld I=%lld H=%lld %-3s  bare product single %7.4f ms"
                        "  back-to-back x20 %7.4f ms (%6.2f us)\n",
                        static_cast<long long>(gm.b), static_cast<long long>(gm.i),
                        static_cast<long long>(gm.h),
                        gm.dt == DataType::f32 ? "f32" : "f16", gms, gms20,
                        gms20 * 1000.0);
        }
        // The whole operator at increasing T: the slope is one full step.
        double prev_ms = 0.0;
        int64_t prev_t = 0;
        for (int64_t t : {1, 2, 4, 8, 16, 32}) {
            std::vector<int64_t> x_shape = {t, gm.b, gm.i};
            std::vector<int64_t> w_shape = {1, 4 * gm.h, gm.i};
            std::vector<int64_t> r_shape = {1, 4 * gm.h, gm.h};
            std::vector<int64_t> b_shape = {1, 8 * gm.h};
            std::vector<int64_t> h_shape = {1, gm.b, gm.h};
            std::vector<int64_t> y_shape = {t, 1, gm.b, gm.h};
            HostTensor hx = make_host_random(x_shape, gm.dt, false, 11);
            HostTensor hw = make_host_random(w_shape, gm.dt, false, 13);
            HostTensor hr = make_host_random(r_shape, gm.dt, false, 17);
            HostTensor hb2 = make_host_random(b_shape, gm.dt, false, 19);
            HostTensor hy = make_host_zeroed(y_shape, gm.dt, false);
            HostTensor hyh = make_host_zeroed(h_shape, gm.dt, false);
            HostTensor hyc = make_host_zeroed(h_shape, gm.dt, false);
            DeviceTensor dx = to_device(hx);
            DeviceTensor dw = to_device(hw);
            DeviceTensor dr = to_device(hr);
            DeviceTensor db2 = to_device(hb2);
            DeviceTensor dy = to_device(hy);
            DeviceTensor dyh = to_device(hyh);
            DeviceTensor dyc = to_device(hyc);
            LstmAttributes a;
            a.hidden_size = gm.h;
            auto op = Lstm::create(a, Backend::CUDA);
            std::vector<TensorView> ins = {dx.view, dw.view, dr.view, db2.view};
            std::vector<TensorView> outs = {dy.view, dyh.view, dyc.view};
            auto launch = [&] {
                const Status st = op->compute(outs, ins, cuda_context(), nullptr);
                if (st != Status::ok) {
                    throw std::runtime_error(std::string("cuda compute: ") + to_string(st));
                }
            };
            const double ms = cuda_median_ms(launch, 2, 7, 1);
            const double slope = prev_t > 0 ? (ms - prev_ms) / double(t - prev_t) : 0.0;
            std::printf("    T=%3lld  %8.4f ms   %8.2f us/step   marginal %8.2f us\n",
                        static_cast<long long>(t), ms, ms * 1000.0 / double(t),
                        slope * 1000.0);
            prev_ms = ms;
            prev_t = t;
        }
    }
}


/// A bidirectional layer has two independent timelines, and the two ways to run
/// them — one stream in sequence, or one each — are not a matter of theory on
/// this platform. The arms are interleaved inside one process because host
/// drift between two runs is larger than the difference.
NNOPS_BENCH(cuda_rnn_bidir_overlap) {
    print_cuda_header("CUDA bidirectional — concurrent vs serial (interleaved)");

    auto& serial = nnops::detail::cuda_rnn_serial;
    for (DataType dt : {DataType::f32, DataType::f16}) {
        Geometry g;
        g.T = 32;
        g.B = 8;
        g.I = 512;
        g.H = 512;
        g.D = 2;
        g.dt = dt;
        LstmAttributes a = lstm_attrs(g.H);
        a.direction = RnnDirection::Bidirectional;

        constexpr int kRounds = 6;
        double seq_ms = 0.0;
        double par_ms = 0.0;
        for (int r = 0; r < kRounds; ++r) {
            serial.store(1, std::memory_order_relaxed);
            seq_ms += run_row<Lstm, LstmAttributes>(RecurrentMode::Lstm, a, g, "serial", false);
            serial.store(0, std::memory_order_relaxed);
            par_ms += run_row<Lstm, LstmAttributes>(RecurrentMode::Lstm, a, g, "concurrent",
                                                    false);
        }
        serial.store(0, std::memory_order_relaxed);
        std::printf("  LSTM bidir T=32 B=8 I=H=512 %-3s  serial %8.4f ms   concurrent %8.4f ms"
                    "   %+5.1f%%\n",
                    dt == DataType::f32 ? "f32" : "f16", seq_ms / kRounds, par_ms / kRounds,
                    100.0 * (seq_ms - par_ms) / seq_ms);
    }
}

#endif  // NNOPS_HAS_CUDA
