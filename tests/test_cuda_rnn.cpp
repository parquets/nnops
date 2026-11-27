/// @file test_cuda_rnn.cpp
/// @brief CUDA recurrent family against the CPU backend on the same inputs.
///
/// Every case builds one random host fixture, runs the operator twice — once on
/// the CPU and once on the device over uploaded copies — and compares Y, Y_h and
/// (for an LSTM) Y_c. The CPU route is the oracle rather than a re-derived
/// reference, so the contract under test is the operator's own: the gate order,
/// clip, input_forget, both linear_before_reset readings, the sequence-length
/// mask and the optional inputs.
///
/// The device route is a chunked input projection on the dense GEMM engine plus
/// a per-step recurrent product and one elementwise gate pass, so the cases that
/// matter are the ones that fork that walk: forward / reverse / bidirectional,
/// H that is not a multiple of eight, a one-row batch, a pitched X and Y, T = 1,
/// a missing Y (only the final state), and f16 next to f32.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/nnops.hpp"

#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/cuda_test_helper.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

/// One case's shapes and optional-input mask. Times stay small: the point is
/// the walk, not the flop count, and every step is a launch on both sides.
struct Geo {
    int64_t T = 4, B = 2, I = 3, H = 6;
    RnnDirection dir = RnnDirection::Forward;
    DataType dt = DataType::f32;
    bool bias = true;
    bool seq = false;
    bool h0 = false;
    bool c0 = false;
    bool want_y = true;
    bool want_yc = true;
    /// Extra storage at the end of each X and Y row, so both views carry a
    /// pitch. The padding is filled like data; the comparison skips it, and a
    /// write past a row shows up.
    bool padded = false;
    uint64_t seed = 11;
};

std::pair<float, float> tolerance(DataType dt)
{
    // The device reorders the gate sums and rounds the gate buffer differently
    // from the CPU's f16 MMA, so the two f16 routes get the shared budget.
    return dt == DataType::f16 ? std::pair<float, float>{kF16AccumTol, 2e-2f}
                               : std::pair<float, float>{2e-4f, 2e-4f};
}

/// A [B] int32 sequence-length tensor. One row never steps, so the "past its
/// length" path is always exercised; the rest stop somewhere in the sequence.
HostTensor make_sequence(int64_t batch, int64_t seq, uint64_t seed)
{
    HostTensor t;
    t.bytes.assign(static_cast<size_t>(batch) * sizeof(int32_t), std::byte{0});
    auto* p = reinterpret_cast<int32_t*>(t.bytes.data());
    XorShift128 rng(seed);
    for (int64_t b = 0; b < batch; ++b) {
        p[b] = static_cast<int32_t>(rng.next_u64() % static_cast<uint64_t>(seq + 1));
    }
    if (batch > 0) {
        p[0] = 0;
    }
    const int64_t shape[1] = {batch};
    t.view = TensorView(std::span<const int64_t>(shape, 1), DataType::s32, t.bytes.data());
    return t;
}

template <class OpT, class AttrT>
void run_case(const Geo& g, RecurrentMode mode, const AttrT& attrs, const char* what)
{
    const int64_t G = recurrent_gates(mode);
    const int64_t D = rnn_num_directions(g.dir);
    const bool lstm = mode == RecurrentMode::Lstm;
    const std::pair<float, float> tol = tolerance(g.dt);

    std::vector<int64_t> x_shape = {g.T, g.B, g.I};
    std::vector<int64_t> w_shape = {D, G * g.H, g.I};
    std::vector<int64_t> r_shape = {D, G * g.H, g.H};
    std::vector<int64_t> b_shape = {D, 2 * G * g.H};
    std::vector<int64_t> h_shape = {D, g.B, g.H};
    std::vector<int64_t> y_shape = {g.T, D, g.B, g.H};

    HostTensor x = make_host_random(x_shape, g.dt, g.padded, g.seed + 1);
    HostTensor w = make_host_random(w_shape, g.dt, false, g.seed + 2);
    HostTensor r = make_host_random(r_shape, g.dt, false, g.seed + 3);
    HostTensor bias;
    HostTensor h0;
    HostTensor c0;
    HostTensor seq;
    if (g.bias) {
        bias = make_host_random(b_shape, g.dt, false, g.seed + 4, -1.0f, 1.0f);
    }
    if (g.seq) {
        seq = make_sequence(g.B, g.T, g.seed + 5);
    }
    if (g.h0) {
        h0 = make_host_random(h_shape, g.dt, false, g.seed + 6, -1.0f, 1.0f);
    }
    if (lstm && g.c0) {
        c0 = make_host_random(h_shape, g.dt, false, g.seed + 7, -1.0f, 1.0f);
    }

    // Inputs: 0 X, 1 W, 2 R, 3 B, 4 sequence_lens, 5 initial_h, 6 initial_c.
    // Trailing absence is a shorter list; a hole stays an empty view.
    std::vector<TensorView> ins = {x.view, w.view, r.view};
    const auto put = [&](size_t i, bool present, const HostTensor& t) {
        if (i >= ins.size()) {
            ins.resize(i + 1);
        }
        ins[i] = present ? t.view : TensorView{};
    };
    put(3, g.bias, bias);
    put(4, g.seq, seq);
    put(5, g.h0, h0);
    if (lstm) {
        put(6, g.c0, c0);
    }
    while (!ins.empty() && ins.back().is_empty()) {
        ins.pop_back();
    }

    // Outputs: 0 Y, 1 Y_h, 2 Y_c (LSTM only).
    HostTensor y;
    HostTensor yh = make_host_zeroed(h_shape, g.dt, false);
    HostTensor yc;
    if (g.want_y) {
        y = make_host_zeroed(y_shape, g.dt, g.padded);
    }
    if (lstm && g.want_yc) {
        yc = make_host_zeroed(h_shape, g.dt, false);
    }
    std::vector<TensorView> outs;
    std::vector<const HostTensor*> out_src;
    outs.push_back(g.want_y ? y.view : TensorView{});
    out_src.push_back(g.want_y ? &y : nullptr);
    outs.push_back(yh.view);
    out_src.push_back(&yh);
    if (lstm) {
        outs.push_back(g.want_yc ? yc.view : TensorView{});
        out_src.push_back(g.want_yc ? &yc : nullptr);
    }

    auto op_cpu = OpT::create(attrs, Backend::CPU);
    const Status cpu_st = op_cpu->compute(outs, ins, cpu_context(), nullptr);
    if (cpu_st != Status::ok) {
        throw std::runtime_error(std::string(what) + ": cpu compute failed");
    }

    // The same bytes on the device. An absent optional stays an empty view --
    // uploading a zero-size buffer would hand the kernel a live pointer to one
    // byte, which is not the same thing.
    std::vector<const HostTensor*> src(ins.size(), nullptr);
    src[0] = &x;
    src[1] = &w;
    src[2] = &r;
    if (g.bias) { src[3] = &bias; }
    if (g.seq) { src[4] = &seq; }
    if (g.h0) { src[5] = &h0; }
    if (lstm && g.c0) { src[6] = &c0; }

    std::vector<DeviceTensor> dins_store;
    dins_store.reserve(ins.size());
    std::vector<TensorView> dins;
    dins.reserve(ins.size());
    for (size_t i = 0; i < ins.size(); ++i) {
        if (src[i] == nullptr) {
            dins.push_back(TensorView{});
        } else {
            dins_store.push_back(to_device(*src[i]));
            dins.push_back(dins_store.back().view);
        }
    }

    std::vector<DeviceTensor> douts(outs.size());
    std::vector<TensorView> dv(outs.size());
    for (size_t i = 0; i < outs.size(); ++i) {
        if (outs[i].is_empty()) {
            dv[i] = TensorView{};
            continue;
        }
        const bool padded = (i == 0 && g.padded);
        HostTensor zero = make_host_zeroed(outs[i].shape_span(), g.dt, padded);
        douts[i] = alloc_like(outs[i], zero);
        dv[i] = douts[i].view;
    }

    auto op_dev = OpT::create(attrs, Backend::CUDA);
    const Status dev_st = op_dev->compute(dv, dins, cuda_context(), nullptr);
    if (dev_st != Status::ok) {
        throw std::runtime_error(std::string(what) + ": cuda compute returned " +
                                 to_string(dev_st));
    }
    CudaBuffer::sync();

    for (size_t i = 0; i < outs.size(); ++i) {
        if (outs[i].is_empty()) {
            continue;
        }
        const bool padded = (i == 0 && g.padded);
        HostTensor got = make_host_zeroed(outs[i].shape_span(), g.dt, padded);
        from_device(douts[i], got);
        expect_same(got, *out_src[i], tol.first, tol.second, what);
    }
}

RnnAttributes rnn_attrs(const Geo& g, RnnActivation act = RnnActivation::Tanh,
                        float clip = 0.0f)
{
    RnnAttributes a;
    a.hidden_size = g.H;
    a.direction = g.dir;
    a.activation = act;
    a.clip = clip;
    return a;
}

GruAttributes gru_attrs(const Geo& g, bool lbr, float clip = 0.0f)
{
    GruAttributes a;
    a.hidden_size = g.H;
    a.direction = g.dir;
    a.linear_before_reset = lbr;
    a.clip = clip;
    return a;
}

LstmAttributes lstm_attrs(const Geo& g, bool input_forget = false, float clip = 0.0f)
{
    LstmAttributes a;
    a.hidden_size = g.H;
    a.direction = g.dir;
    a.input_forget = input_forget;
    a.clip = clip;
    return a;
}

// =========================================================================
//  RNN
// =========================================================================

NNOPS_TEST(cuda_rnn_forward) {
    Geo g;
    g.T = 5;
    g.B = 3;
    g.I = 7;
    g.H = 6;
    run_case<Rnn, RnnAttributes>(g, RecurrentMode::Rnn, rnn_attrs(g), "rnn forward");
}

NNOPS_TEST(cuda_rnn_reverse_with_state_and_lens) {
    Geo g;
    g.T = 6;
    g.B = 4;
    g.I = 5;
    g.H = 5;
    g.dir = RnnDirection::Reverse;
    g.seq = true;
    g.h0 = true;
    g.padded = true;
    run_case<Rnn, RnnAttributes>(g, RecurrentMode::Rnn, rnn_attrs(g), "rnn reverse");
}

NNOPS_TEST(cuda_rnn_f16) {
    Geo g;
    g.T = 4;
    g.B = 2;
    g.I = 5;
    g.H = 9;
    g.dt = DataType::f16;
    g.seq = true;
    g.h0 = true;
    run_case<Rnn, RnnAttributes>(g, RecurrentMode::Rnn, rnn_attrs(g), "rnn f16");
}

NNOPS_TEST(cuda_rnn_relu_and_elu) {
    Geo g;
    g.T = 3;
    g.B = 2;
    g.I = 4;
    g.H = 7;
    for (RnnActivation act : {RnnActivation::Relu, RnnActivation::LeakyRelu,
                              RnnActivation::Elu}) {
        g.seed += 3;
        run_case<Rnn, RnnAttributes>(g, RecurrentMode::Rnn, rnn_attrs(g, act, 1.5f),
                                     "rnn nonlinear activation");
    }
}

// =========================================================================
//  GRU
// =========================================================================

NNOPS_TEST(cuda_gru_both_reset_readings) {
    Geo g;
    g.T = 5;
    g.B = 3;
    g.I = 4;
    g.H = 7;
    g.seq = true;
    g.h0 = true;
    for (bool lbr : {false, true}) {
        g.seed += 11;
        run_case<Gru, GruAttributes>(g, RecurrentMode::Gru, gru_attrs(g, lbr),
                                     lbr ? "gru linear_before_reset" : "gru reset-before");
    }
}

NNOPS_TEST(cuda_gru_bidirectional_f16) {
    Geo g;
    g.T = 4;
    g.B = 2;
    g.I = 6;
    g.H = 5;
    g.dt = DataType::f16;
    g.dir = RnnDirection::Bidirectional;
    g.bias = true;
    run_case<Gru, GruAttributes>(g, RecurrentMode::Gru, gru_attrs(g, false, 0.75f),
                                 "gru bidirectional f16");
}

// =========================================================================
//  LSTM
// =========================================================================

NNOPS_TEST(cuda_lstm_forward_with_cell) {
    Geo g;
    g.T = 5;
    g.B = 3;
    g.I = 6;
    g.H = 8;
    g.seq = true;
    g.h0 = true;
    g.c0 = true;
    run_case<Lstm, LstmAttributes>(g, RecurrentMode::Lstm, lstm_attrs(g), "lstm forward");
}

NNOPS_TEST(cuda_lstm_bidirectional_reverse_paths) {
    Geo g;
    g.T = 6;
    g.B = 2;
    g.I = 3;
    g.H = 5;
    g.dir = RnnDirection::Bidirectional;
    g.seq = true;
    g.h0 = true;
    g.c0 = true;
    g.padded = true;
    run_case<Lstm, LstmAttributes>(g, RecurrentMode::Lstm, lstm_attrs(g),
                                   "lstm bidirectional");
}

NNOPS_TEST(cuda_lstm_input_forget_and_clip) {
    Geo g;
    g.T = 4;
    g.B = 3;
    g.I = 5;
    g.H = 6;
    g.h0 = true;
    g.c0 = true;
    run_case<Lstm, LstmAttributes>(g, RecurrentMode::Lstm,
                                   lstm_attrs(g, /*input_forget=*/true, 0.5f),
                                   "lstm input_forget + clip");
}

NNOPS_TEST(cuda_lstm_f16) {
    Geo g;
    g.T = 4;
    g.B = 2;
    g.I = 5;
    g.H = 7;
    g.dt = DataType::f16;
    g.seq = true;
    g.h0 = true;
    g.c0 = true;
    run_case<Lstm, LstmAttributes>(g, RecurrentMode::Lstm, lstm_attrs(g), "lstm f16");
}

NNOPS_TEST(cuda_lstm_single_step_and_no_y) {
    Geo g;
    g.T = 1;
    g.B = 3;
    g.I = 4;
    g.H = 5;
    run_case<Lstm, LstmAttributes>(g, RecurrentMode::Lstm, lstm_attrs(g),
                                   "lstm single step");

    Geo only_state = g;
    only_state.want_y = false;
    only_state.h0 = true;
    only_state.c0 = true;
    run_case<Lstm, LstmAttributes>(only_state, RecurrentMode::Lstm, lstm_attrs(only_state),
                                   "lstm without Y");
}

NNOPS_TEST(cuda_lstm_single_row_batch) {
    // One row is the narrow-m pose of both products, and the only way a phase
    // grid collapses to one block.
    Geo g;
    g.T = 5;
    g.B = 1;
    g.I = 4;
    g.H = 9;
    g.seq = true;
    g.h0 = true;
    g.c0 = true;
    run_case<Lstm, LstmAttributes>(g, RecurrentMode::Lstm, lstm_attrs(g), "lstm B = 1");
}

NNOPS_TEST(cuda_lstm_hidden_one) {
    Geo g;
    g.T = 3;
    g.B = 2;
    g.I = 2;
    g.H = 1;
    run_case<Lstm, LstmAttributes>(g, RecurrentMode::Lstm, lstm_attrs(g), "lstm H = 1");
}

// =========================================================================
//  Wide hidden sizes: the hinted narrow products
// =========================================================================

/// The step and projection products are handed a split count of their own when
/// they are short and narrow (see narrow_split_hint). That path only opens for
/// a vectorisable pose — a hidden size on the 16 B bound and K ≥ 256 — which
/// the small fixtures above do not reach, so these cases walk it: H a multiple
/// of eight, K = H at 256 and 512, and both f32 and f16.
NNOPS_TEST(cuda_lstm_hinted_split_f32) {
    Geo g;
    g.T = 4;
    g.B = 8;
    g.I = 64;
    g.H = 256;
    g.seq = true;
    g.h0 = true;
    g.c0 = true;
    run_case<Lstm, LstmAttributes>(g, RecurrentMode::Lstm, lstm_attrs(g),
                                   "lstm hinted split f32");
}

NNOPS_TEST(cuda_lstm_hinted_split_f16) {
    Geo g;
    g.T = 4;
    g.B = 8;
    g.I = 64;
    g.H = 256;
    g.dt = DataType::f16;
    g.seq = true;
    g.h0 = true;
    g.c0 = true;
    run_case<Lstm, LstmAttributes>(g, RecurrentMode::Lstm, lstm_attrs(g),
                                   "lstm hinted split f16");
}

NNOPS_TEST(cuda_rnn_hinted_split_f32) {
    // RNN's gate row is H, so H = 512 puts both products on the narrowest grid.
    Geo g;
    g.T = 4;
    g.B = 8;
    g.I = 64;
    g.H = 512;
    g.seq = true;
    g.h0 = true;
    run_case<Rnn, RnnAttributes>(g, RecurrentMode::Rnn, rnn_attrs(g),
                                 "rnn hinted split f32");
}

NNOPS_TEST(cuda_gru_hinted_split_f16) {
    // Both readings: lbr splits the gate row into 2H + H phases, the other into
    // one 3H pass, so the two walk different hint values.
    Geo g;
    g.T = 4;
    g.B = 8;
    g.I = 64;
    g.H = 256;
    g.dt = DataType::f16;
    g.seq = true;
    g.h0 = true;
    for (bool lbr : {false, true}) {
        g.seed += 7;
        run_case<Gru, GruAttributes>(g, RecurrentMode::Gru, gru_attrs(g, lbr),
                                     lbr ? "gru hinted f16 lbr" : "gru hinted f16");
    }
}

NNOPS_TEST(cuda_lstm_decode_hinted_projection) {
    // T = 1 makes the input projection a short-M product as well, which is the
    // other place the hint is applied.
    Geo g;
    g.T = 1;
    g.B = 8;
    g.I = 256;
    g.H = 256;
    g.h0 = true;
    g.c0 = true;
    run_case<Lstm, LstmAttributes>(g, RecurrentMode::Lstm, lstm_attrs(g),
                                   "lstm decode hinted projection");
}

/// bf16 is admitted on the CUDA backend (the x86 CPU has no bf16 kernel), so the
/// oracle is the f32 CPU route over inputs that are already bf16-representable:
/// every value is rounded first, which makes the only difference between the two
/// runs the device's bf16 rounding of the gate products and the state.
NNOPS_TEST(cuda_lstm_bf16_against_rounded_f32) {
    const int64_t Tn = 4, B = 2, I = 5, H = 6, D = 1, G = 4;
    const std::vector<int64_t> x_shape = {Tn, B, I};
    const std::vector<int64_t> w_shape = {D, G * H, I};
    const std::vector<int64_t> r_shape = {D, G * H, H};
    const std::vector<int64_t> b_shape = {D, 2 * G * H};
    const std::vector<int64_t> h_shape = {D, B, H};
    const std::vector<int64_t> y_shape = {Tn, D, B, H};

    HostTensor x = bf16_rounded_host(make_host_random(x_shape, DataType::f32, false, 101));
    HostTensor w = bf16_rounded_host(make_host_random(w_shape, DataType::f32, false, 103));
    HostTensor r = bf16_rounded_host(make_host_random(r_shape, DataType::f32, false, 107));
    HostTensor b = bf16_rounded_host(
        make_host_random(b_shape, DataType::f32, false, 109, -1.0f, 1.0f));
    HostTensor h0 = bf16_rounded_host(
        make_host_random(h_shape, DataType::f32, false, 113, -1.0f, 1.0f));
    HostTensor c0 = bf16_rounded_host(
        make_host_random(h_shape, DataType::f32, false, 127, -1.0f, 1.0f));
    HostTensor y = make_host_zeroed(y_shape, DataType::f32, false);
    HostTensor yh = make_host_zeroed(h_shape, DataType::f32, false);
    HostTensor yc = make_host_zeroed(h_shape, DataType::f32, false);

    const std::vector<TensorView> ins = {x.view, w.view, r.view, b.view, TensorView{},
                                         h0.view, c0.view};
    std::vector<TensorView> outs = {y.view, yh.view, yc.view};

    LstmAttributes a;
    a.hidden_size = H;
    auto op_cpu = Lstm::create(a, Backend::CPU);
    if (op_cpu->compute(outs, ins, cpu_context(), nullptr) != Status::ok) {
        throw std::runtime_error("lstm bf16: cpu oracle failed");
    }

    HostTensor xb = to_bf16_host(x, false);
    HostTensor wb = to_bf16_host(w, false);
    HostTensor rb = to_bf16_host(r, false);
    HostTensor bb = to_bf16_host(b, false);
    HostTensor hb = to_bf16_host(h0, false);
    HostTensor cb = to_bf16_host(c0, false);
    DeviceTensor dx = to_device(xb);
    DeviceTensor dw = to_device(wb);
    DeviceTensor dr = to_device(rb);
    DeviceTensor db = to_device(bb);
    DeviceTensor dh = to_device(hb);
    DeviceTensor dc = to_device(cb);

    HostTensor zy = make_host_zeroed(y_shape, DataType::bf16, false);
    HostTensor zyh = make_host_zeroed(h_shape, DataType::bf16, false);
    HostTensor zyc = make_host_zeroed(h_shape, DataType::bf16, false);
    DeviceTensor dy = alloc_like(zy.view, zy);
    DeviceTensor dyh = alloc_like(zyh.view, zyh);
    DeviceTensor dyc = alloc_like(zyc.view, zyc);

    std::vector<TensorView> dins = {dx.view, dw.view, dr.view, db.view, TensorView{},
                                    dh.view, dc.view};
    std::vector<TensorView> douts = {dy.view, dyh.view, dyc.view};

    auto op_dev = Lstm::create(a, Backend::CUDA);
    const Status st = op_dev->compute(douts, dins, cuda_context(), nullptr);
    if (st != Status::ok) {
        throw std::runtime_error(std::string("lstm bf16: cuda compute returned ") +
                                 to_string(st));
    }
    CudaBuffer::sync();

    HostTensor gy = make_host_zeroed(y_shape, DataType::bf16, false);
    HostTensor gyh = make_host_zeroed(h_shape, DataType::bf16, false);
    HostTensor gyc = make_host_zeroed(h_shape, DataType::bf16, false);
    from_device(dy, gy);
    from_device(dyh, gyh);
    from_device(dyc, gyc);

    // Compare in one dtype: the reference is rounded to bf16 too, so the two
    // sides differ only by the device's own bf16 rounding. bf16 keeps seven
    // mantissa bits and the state is re-rounded every step, hence the budget.
    constexpr float kTol = 3e-2f;
    expect_same(gy, to_bf16_host(y, false), kTol, kTol, "lstm bf16 Y");
    expect_same(gyh, to_bf16_host(yh, false), kTol, kTol, "lstm bf16 Y_h");
    expect_same(gyc, to_bf16_host(yc, false), kTol, kTol, "lstm bf16 Y_c");
}

NNOPS_TEST(cuda_rnn_rejects_prepacked) {
    // A PackedRecurrent view is a CPU-packer blob this driver has no reader
    // for; it is refused rather than misread. (An s8 weight is the W8A8 route
    // the cases below cover.)
    std::vector<float> blob(64, 0.0f);
    std::vector<float> x(2 * 1 * 2, 0.0f);
    std::vector<float> r(1 * 2 * 2, 0.0f);
    const int64_t xs[] = {2, 1, 2};
    const int64_t ws[] = {64};
    const int64_t rs[] = {1, 2, 2};
    RnnAttributes a;
    a.hidden_size = 2;
    auto op = Rnn::create(a, Backend::CUDA);
    TensorView xv(xs, DataType::f32, x.data());
    TensorView wv(ws, DataType::f32, blob.data(), TensorLayout::PackedRecurrent);
    TensorView rv(rs, DataType::f32, r.data());
    const TensorView ins[] = {xv, wv, rv};
    const int64_t ys[] = {2, 1, 1, 2};
    std::vector<float> y(2 * 1 * 1 * 2, 0.0f);
    std::vector<float> yh(1 * 1 * 2, 0.0f);
    TensorView yv(ys, DataType::f32, y.data());
    const int64_t hs[] = {1, 1, 2};
    TensorView hv(hs, DataType::f32, yh.data());
    TensorView outs[] = {yv, hv};
    const Status st = op->compute(outs, ins, cuda_context(), nullptr);
    if (st != Status::unsupported) {
        throw std::runtime_error(std::string("prepacked W: status ") + to_string(st));
    }
}

}  // namespace

// =========================================================================
//  W8A8 — s8 weights, the activation quantized per row
// =========================================================================

/// One quantized layer: s8 W and R with symmetric per-gate-row scales (one
/// table per direction), the compute dtype's bias and state, and either a
/// floating X — quantized per row on the fly — or an s8 X with its own
/// per-token table.
template <class OpT, class AttrT>
void run_q8_case(const Geo& g, RecurrentMode mode, const AttrT& attrs, bool x_s8,
                 const char* what)
{
    const int64_t G = recurrent_gates(mode);
    const int64_t D = rnn_num_directions(g.dir);
    const bool lstm = mode == RecurrentMode::Lstm;
    const std::pair<float, float> tol = tolerance(g.dt);

    const std::vector<int64_t> x_shape = {g.T, g.B, g.I};
    const std::vector<int64_t> w_shape = {D, G * g.H, g.I};
    const std::vector<int64_t> r_shape = {D, G * g.H, g.H};
    const std::vector<int64_t> b_shape = {D, 2 * G * g.H};
    const std::vector<int64_t> h_shape = {D, g.B, g.H};
    const std::vector<int64_t> y_shape = {g.T, D, g.B, g.H};

    const int64_t rows = g.T * g.B;
    HostTensor x;
    QuantParams qx;
    std::vector<float> x_sc(static_cast<size_t>(rows));
    if (x_s8) {
        x = make_host_int(x_shape, DataType::s8, g.seed + 1, -127, 127);
        for (int64_t i = 0; i < rows; ++i) {
            x_sc[static_cast<size_t>(i)] = 0.01f + 0.002f * float(i % 5);
        }
        qx.granularity = QuantGranularity::PerToken;
        qx.scale_data = x_sc.data();
        qx.num_scales = rows;
        x.view = with_quant_params(x.view, qx);
    } else {
        x = make_host_random(x_shape, g.dt, g.padded, g.seed + 1);
    }

    test::XorShift128 rng(g.seed + 2);
    std::vector<int8_t> w(static_cast<size_t>(D * G * g.H * g.I));
    std::vector<int8_t> r(static_cast<size_t>(D * G * g.H * g.H));
    for (auto& e : w) { e = static_cast<int8_t>(int32_t(rng.next_u64() % 255) - 127); }
    for (auto& e : r) { e = static_cast<int8_t>(int32_t(rng.next_u64() % 255) - 127); }
    std::vector<float> w_sc(static_cast<size_t>(D * G * g.H));
    std::vector<float> r_sc(static_cast<size_t>(D * G * g.H));
    for (int64_t i = 0; i < D * G * g.H; ++i) {
        w_sc[static_cast<size_t>(i)] = 0.003f + 0.001f * float(i % 4);
        r_sc[static_cast<size_t>(i)] = 0.002f + 0.001f * float(i % 5);
    }
    QuantParams qw;
    qw.granularity = QuantGranularity::PerToken;
    qw.scale_data = w_sc.data();
    qw.num_scales = D * G * g.H;
    QuantParams qr;
    qr.granularity = QuantGranularity::PerToken;
    qr.scale_data = r_sc.data();
    qr.num_scales = D * G * g.H;

    HostTensor bias;
    HostTensor h0;
    HostTensor c0;
    if (g.bias) {
        bias = make_host_random(b_shape, g.dt, false, g.seed + 4, -1.0f, 1.0f);
    }
    if (g.h0) { h0 = make_host_random(h_shape, g.dt, false, g.seed + 6, -1.0f, 1.0f); }
    if (lstm && g.c0) { c0 = make_host_random(h_shape, g.dt, false, g.seed + 7, -1.0f, 1.0f); }

    TensorView wv = TensorView(std::span<const int64_t>(w_shape.data(), 3), DataType::s8,
                               w.data(), TensorLayout::Dense, qw);
    TensorView rv = TensorView(std::span<const int64_t>(r_shape.data(), 3), DataType::s8,
                               r.data(), TensorLayout::Dense, qr);
    std::vector<TensorView> ins = {x.view, wv, rv};
    const auto put = [&](size_t i, bool present, const HostTensor& t) {
        if (i >= ins.size()) { ins.resize(i + 1); }
        ins[i] = present ? t.view : TensorView{};
    };
    put(3, g.bias, bias);
    put(4, false, bias);
    put(5, g.h0, h0);
    if (lstm) { put(6, g.c0, c0); }
    while (!ins.empty() && ins.back().is_empty()) { ins.pop_back(); }

    HostTensor y = make_host_zeroed(y_shape, g.dt, g.padded);
    HostTensor yh = make_host_zeroed(h_shape, g.dt, false);
    HostTensor yc;
    if (lstm) { yc = make_host_zeroed(h_shape, g.dt, false); }
    std::vector<TensorView> outs = {y.view, yh.view};
    if (lstm) { outs.push_back(yc.view); }

    auto op_cpu = OpT::create(attrs, Backend::CPU);
    if (op_cpu->compute(outs, ins, cpu_context(), nullptr) != Status::ok) {
        throw std::runtime_error(std::string(what) + ": cpu compute failed");
    }

    const HostTensor* src[7] = {&x, nullptr, nullptr, g.bias ? &bias : nullptr, nullptr,
                                g.h0 ? &h0 : nullptr, (lstm && g.c0) ? &c0 : nullptr};

    std::vector<TensorView> dins(ins.size());
    std::vector<DeviceTensor> store;
    std::vector<CudaBuffer> wbuf(2);
    for (size_t i = 0; i < ins.size(); ++i) {
        if (i == 1 || i == 2) {
            const std::vector<int8_t>& bytes = (i == 1) ? w : r;
            wbuf[i - 1] = CudaBuffer(bytes.size());
            wbuf[i - 1].upload(bytes.data(), bytes.size());
            const std::vector<int64_t>& sh = (i == 1) ? w_shape : r_shape;
            dins[i] = TensorView(std::span<const int64_t>(sh.data(), 3), DataType::s8,
                                 wbuf[i - 1].data(), TensorLayout::Dense,
                                 (i == 1) ? qw : qr);
        } else if (src[i] != nullptr) {
            store.push_back(to_device(*src[i]));
            TensorView v = store.back().view;
            if (i == 0 && x_s8) { v = with_quant_params(v, qx); }
            dins[i] = v;
        } else {
            dins[i] = TensorView{};
        }
    }

    std::vector<DeviceTensor> douts(outs.size());
    std::vector<TensorView> dv(outs.size());
    for (size_t i = 0; i < outs.size(); ++i) {
        const bool padded = (i == 0 && g.padded);
        HostTensor zero = make_host_zeroed(outs[i].shape_span(), g.dt, padded);
        douts[i] = alloc_like(outs[i], zero);
        dv[i] = douts[i].view;
    }

    auto op_dev = OpT::create(attrs, Backend::CUDA);
    const Status st = op_dev->compute(dv, dins, cuda_context(), nullptr);
    if (st != Status::ok) {
        throw std::runtime_error(std::string(what) + ": cuda compute returned " +
                                 to_string(st));
    }
    CudaBuffer::sync();

    const HostTensor* want[] = {&y, &yh, lstm ? &yc : nullptr};
    for (size_t i = 0; i < outs.size(); ++i) {
        const bool padded = (i == 0 && g.padded);
        HostTensor got = make_host_zeroed(outs[i].shape_span(), g.dt, padded);
        from_device(douts[i], got);
        expect_same(got, *want[i], tol.first, tol.second, what);
    }
}

NNOPS_TEST(cuda_rnn_w8a8) {
    Geo g;
    g.seed = 91;
    run_q8_case<Rnn, RnnAttributes>(g, RecurrentMode::Rnn, rnn_attrs(g), false, "rnn w8a8");
}

NNOPS_TEST(cuda_lstm_w8a8_bidirectional) {
    Geo g;
    g.dir = RnnDirection::Bidirectional;
    g.bias = true;
    g.h0 = true;
    g.c0 = true;
    g.padded = true;
    g.seed = 93;
    run_q8_case<Lstm, LstmAttributes>(g, RecurrentMode::Lstm, lstm_attrs(g), false,
                                      "lstm w8a8 bidirectional");
}

NNOPS_TEST(cuda_gru_w8a8_f16) {
    for (bool lbr : {false, true}) {
        Geo g;
        g.dt = DataType::f16;
        g.B = 1;
        g.T = 3;
        g.seed = 95 + (lbr ? 1 : 0);
        run_q8_case<Gru, GruAttributes>(g, RecurrentMode::Gru, gru_attrs(g, lbr), false,
                                        lbr ? "gru lbr w8a8 f16" : "gru w8a8 f16");
    }
}

NNOPS_TEST(cuda_rnn_w8a8_s8_activation) {
    Geo g;
    g.seed = 97;
    run_q8_case<Rnn, RnnAttributes>(g, RecurrentMode::Rnn, rnn_attrs(g), true,
                                    "rnn w8a8 s8 activation");
}

#endif  // NNOPS_HAS_CUDA
