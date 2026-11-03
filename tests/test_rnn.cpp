/// @file test_rnn.cpp
/// @brief Unit tests for the recurrent family — RNN, GRU and LSTM.
///
/// Every operator is compared against `reference::*_ref`, a scalar transcription
/// of the ONNX equations in its own translation unit, so agreement is evidence
/// about the math rather than about a shared bug. The cases then walk the axes
/// the fused kernel forks on:
///
///  - hidden widths that are and are not multiples of 8 (the interleave pads to
///    blocks of eight, so H % 8 is where a Y store can run off the caller's row);
///  - forward, reverse and bidirectional (the reverse timeline is chunked
///    downwards, and its Y rows are indexed by the real t);
///  - sequence_lens, including a length of 0 (a row that never steps, whose
///    final state is zeroed) and a padded tail (whose Y rows are zero);
///  - bias, initial_h and initial_c as optional inputs, plus the LSTM's optional
///    Y_c and a run with no Y at all;
///  - GRU linear_before_reset both ways — that is a different accumulation, not
///    a different flop count;
///  - input_forget and clip, the two attributes that change the gate arithmetic;
///  - H past the k-block cap, so the recurrent product is split in k;
///  - f16, where the gate buffer itself is half precision;
///  - pitched X and Y, since both are read and written by row stride;
///  - a one-row batch, which takes the matrix-vector route for the recurrent
///    product (a sub-panel width, a k split, and several column blocks);
///  - a threaded run, compared against the sequential one.

#include "nnops/nnops.hpp"

#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/test_helpers.hpp"

#include "nnops/core/openmp_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

using namespace nnops;
using nnops::test::XorShift128;

// Test-local declarations; the references are defined in their own unit.
namespace nnops::backend::cpu::reference {
void rnn_ref(const RnnAttributes&, std::span<TensorView>, std::span<const TensorView>);
void gru_ref(const GruAttributes&, std::span<TensorView>, std::span<const TensorView>);
void lstm_ref(const LstmAttributes&, std::span<TensorView>, std::span<const TensorView>);
}  // namespace nnops::backend::cpu::reference

namespace ref = nnops::backend::cpu::reference;

namespace {

using nnops::backend::cpu::half;
using nnops::simd::s_load;
using nnops::simd::s_store;

/// f32's GEMM reorders the sums and the f16 route rounds the gate buffer, so the
/// two get different bars; both are absolute against a reference on the same
/// inputs.
constexpr float kF32Tol = 2e-4f;

template <typename T> constexpr DataType dt_of() {
    return std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
}

template <typename T>
void fill_random(std::vector<T>& v, XorShift128& rng, float lo = -1.0f, float hi = 1.0f) {
    for (auto& e : v) { s_store(&e, rng.next_float(lo, hi)); }
}

template <typename T>
float max_diff(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) { return 1e30f; }
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(s_load(&a[i]) - s_load(&b[i])));
    }
    return m;
}

/// One case's shapes and optional-input mask. Times are small on purpose: the
/// reference costs T·B·G·H·(I+H) and nothing here needs a long sequence.
struct Geo {
    int64_t T = 4, B = 2, I = 3, H = 6;
    RnnDirection dir = RnnDirection::Forward;
    bool bias = true;
    bool seq = false;
    bool h0 = false;
    bool c0 = false;
    bool want_y = true;
    bool want_yc = true;
    /// Extra elements at the end of each X and Y row, so both views are pitched
    /// the way a slice of a larger tensor is. The padding is filled like data and
    /// every output comparison covers it: a write past a row shows up.
    int64_t pad = 0;
    uint64_t seed = 11;
    /// W and R are s8 with one symmetric scale per gate row — the W8A8 route.
    /// The reference sees them dequantized, so it agrees to the activation
    /// quantisation's own error rather than to the packing.
    bool q8 = false;
    /// A multiplier on the quantized weights' row scales. At a wide hidden size
    /// the random weights drive every tanh into its flat region, where the
    /// quantisation noise of a 520-term product is amplified; a smaller scale
    /// keeps the case linear so the tolerance stays meaningful.
    float q8_wscale = 1.0f;
    /// X arrives as s8 with a per-row scale table instead of floating point.
    bool x_s8 = false;
};

/// Owns every operand of one run and the views over them.
template <typename T>
struct Fixture {
    std::vector<T> x, w, r, b, h0, c0;
    std::vector<int32_t> seq;
    std::vector<T> y, yh, yc;
    std::vector<TensorView> ins, outs;
    // The quantized route's storage: s8 bytes and their per-row scales, with
    // `w`/`r`/`xd` holding the dequantized values the reference reads.
    std::vector<int8_t> wq, rq, xq;
    std::vector<float> wscale, rscale, xscale;
    std::vector<T> xd;
    TensorView x_ref_view;
    std::vector<TensorView> ins_ref;

    void build(RecurrentMode mode, const Geo& g) {
        const DataType dt = dt_of<T>();
        const int64_t G = recurrent_gates(mode);
        const int64_t D = rnn_num_directions(g.dir);
        const bool lstm = mode == RecurrentMode::Lstm;

        const int64_t xw = g.I + g.pad;
        const int64_t yw = g.H + g.pad;
        const int64_t esz = static_cast<int64_t>(sizeof(T));

        x.assign(static_cast<size_t>(g.T * g.B * xw), T{});
        w.assign(static_cast<size_t>(D * G * g.H * g.I), T{});
        r.assign(static_cast<size_t>(D * G * g.H * g.H), T{});
        b.assign(g.bias ? static_cast<size_t>(D * 2 * G * g.H) : 0, T{});
        h0.assign(g.h0 ? static_cast<size_t>(D * g.B * g.H) : 0, T{});
        c0.assign(g.c0 ? static_cast<size_t>(D * g.B * g.H) : 0, T{});
        seq.assign(g.seq ? static_cast<size_t>(g.B) : 0, 0);
        y.assign(static_cast<size_t>(g.T * D * g.B * yw), T{});
        yh.assign(static_cast<size_t>(D * g.B * g.H), T{});
        yc.assign(lstm ? static_cast<size_t>(D * g.B * g.H) : 0, T{});

        XorShift128 rng(g.seed);
        fill_random(x, rng);
        // Weights in [-0.5, 0.5) so the gate pre-activations stay off the
        // sigmoid's and tanh's flat regions at the widths used here.
        fill_random(w, rng, -0.5f, 0.5f);
        fill_random(r, rng, -0.5f, 0.5f);
        fill_random(b, rng);
        fill_random(h0, rng);
        fill_random(c0, rng);
        for (auto& s : seq) {
            s = static_cast<int32_t>(rng.next_u64() % static_cast<uint64_t>(g.T + 1));
        }
        if (seq.size() > 1) { seq[0] = 0; }  // a row that never steps, always

        int64_t s_x[3] = {g.T, g.B, g.I};
        int64_t s_w[3] = {D, G * g.H, g.I};
        int64_t s_r[3] = {D, G * g.H, g.H};
        int64_t s_b[2] = {D, 2 * G * g.H};
        int64_t s_s[1] = {g.B};
        int64_t s_h[3] = {D, g.B, g.H};
        int64_t s_y[4] = {g.T, D, g.B, g.H};

        ins.clear();
        ins.push_back(g.pad > 0
                          ? TensorView(std::span<const int64_t>(s_x, 3), dt, x.data(),
                                       xw * esz)
                          : TensorView(std::span<const int64_t>(s_x, 3), dt, x.data()));
        ins.push_back(TensorView(std::span<const int64_t>(s_w, 3), dt, w.data()));
        ins.push_back(TensorView(std::span<const int64_t>(s_r, 3), dt, r.data()));
        ins.push_back(g.bias ? TensorView(std::span<const int64_t>(s_b, 2), dt, b.data())
                             : TensorView{});
        ins.push_back(g.seq ? TensorView(std::span<const int64_t>(s_s, 1), DataType::s32,
                                         seq.data())
                            : TensorView{});
        ins.push_back(g.h0 ? TensorView(std::span<const int64_t>(s_h, 3), dt, h0.data())
                           : TensorView{});
        if (lstm) {
            ins.push_back(g.c0 ? TensorView(std::span<const int64_t>(s_h, 3), dt, c0.data())
                               : TensorView{});
        }
        // Trailing absence is a shorter list; absence in the middle stays an
        // empty slot, which is the only way to name input 5 without input 4.
        while (!ins.empty() && ins.back().is_empty()) { ins.pop_back(); }

        outs.clear();
        if (!g.want_y) {
            outs.push_back(TensorView{});
        } else if (g.pad > 0) {
            outs.push_back(TensorView(std::span<const int64_t>(s_y, 4), dt, y.data(),
                                      yw * esz));
        } else {
            outs.push_back(TensorView(std::span<const int64_t>(s_y, 4), dt, y.data()));
        }
        outs.push_back(TensorView(std::span<const int64_t>(s_h, 3), dt, yh.data()));
        if (lstm) {
            outs.push_back(g.want_yc
                               ? TensorView(std::span<const int64_t>(s_h, 3), dt, yc.data())
                               : TensorView{});
        }

        // The reference reads the floating views taken before W/R/X turn into s8;
        // quantize_* rewrites the fp vectors they point at with the dequantized
        // values, so the comparison is against the quantization's own error.
        ins_ref = ins;
        if (g.q8) { quantize_weights(mode, g); }
        if (g.x_s8) { quantize_x(g); }
        if (g.x_s8) { ins_ref[0] = x_ref_view; }
    }

    /// W and R become s8 with one positive scale per gate row; the fp vectors the
    /// fixture already owns are rewritten as the dequantized values, which is what
    /// the reference solves with.
    void quantize_weights(RecurrentMode mode, const Geo& g) {
        const int64_t G = recurrent_gates(mode);
        const int64_t D = rnn_num_directions(g.dir);
        XorShift128 rng(g.seed * 3 + 1);
        wq.resize(w.size());
        rq.resize(r.size());
        wscale.assign(static_cast<size_t>(D * G * g.H), 0.0f);
        rscale.assign(static_cast<size_t>(D * G * g.H), 0.0f);

        const auto fill = [&](std::vector<int8_t>& q, std::vector<float>& sc,
                              std::vector<T>& deq, int64_t cols) {
            for (int64_t row = 0; row < D * G * g.H; ++row) {
                // Scales that differ per row: a table packed out of order does not
                // cancel out.
                const float s = (0.002f + 0.006f * rng.next_float(0.0f, 1.0f)) *
                                g.q8_wscale;
                sc[static_cast<size_t>(row)] = s;
                for (int64_t k = 0; k < cols; ++k) {
                    const auto v = static_cast<int8_t>(
                        static_cast<int>(rng.next_u64() % 255) - 127);
                    q[static_cast<size_t>(row * cols + k)] = v;
                    s_store(&deq[static_cast<size_t>(row * cols + k)],
                            static_cast<float>(v) * s);
                }
            }
        };
        fill(wq, wscale, w, g.I);
        fill(rq, rscale, r, g.H);

        const int64_t rows = D * G * g.H;
        QuantParams qpw;
        qpw.granularity = QuantGranularity::PerToken;
        qpw.scale_data = wscale.data();
        qpw.num_scales = rows;
        QuantParams qpr;
        qpr.granularity = QuantGranularity::PerToken;
        qpr.scale_data = rscale.data();
        qpr.num_scales = rows;

        int64_t s_w[3] = {D, G * g.H, g.I};
        int64_t s_r[3] = {D, G * g.H, g.H};
        // The pack reads the fp bias as the compute dtype, so `b` stays as built.
        ins[1] = TensorView(std::span<const int64_t>(s_w, 3), DataType::s8,
                            wq.data(), TensorLayout::NCHW, qpw);
        ins[2] = TensorView(std::span<const int64_t>(s_r, 3), DataType::s8,
                            rq.data(), TensorLayout::NCHW, qpr);
    }

    /// X as s8 with one scale per [t, b] row; `xd` holds the dequantized values,
    /// which the reference reads instead.
    void quantize_x(const Geo& g) {
        const int64_t xw = g.I + g.pad;
        const int64_t rows = g.T * g.B;
        xq.resize(static_cast<size_t>(rows * xw));
        xd.resize(static_cast<size_t>(rows * xw));
        xscale.assign(static_cast<size_t>(rows), 0.0f);
        XorShift128 rng(g.seed * 7 + 3);
        for (int64_t row = 0; row < rows; ++row) {
            const float s = 0.002f + 0.004f * rng.next_float(0.0f, 1.0f);
            xscale[static_cast<size_t>(row)] = s;
            for (int64_t k = 0; k < xw; ++k) {
                const auto v = static_cast<int8_t>(
                    static_cast<int>(rng.next_u64() % 255) - 127);
                xq[static_cast<size_t>(row * xw + k)] = v;
                s_store(&xd[static_cast<size_t>(row * xw + k)],
                        static_cast<float>(v) * s);
            }
        }
        QuantParams qp;
        qp.granularity = QuantGranularity::PerToken;
        qp.scale_data = xscale.data();
        qp.num_scales = rows;
        int64_t s_x[3] = {g.T, g.B, g.I};
        const int64_t esz = static_cast<int64_t>(sizeof(T));
        ins[0] = g.pad > 0
                     ? TensorView(std::span<const int64_t>(s_x, 3), DataType::s8,
                                  xq.data(), xw, TensorLayout::NCHW, qp)
                     : TensorView(std::span<const int64_t>(s_x, 3), DataType::s8,
                                  xq.data(), TensorLayout::NCHW, qp);
        x_ref_view = g.pad > 0
                         ? TensorView(std::span<const int64_t>(s_x, 3), dt_of<T>(),
                                      xd.data(), xw * esz)
                         : TensorView(std::span<const int64_t>(s_x, 3), dt_of<T>(),
                                      xd.data());
    }

    /// Zero the outputs, run the scalar reference, and return the operator's own
    /// results as floats so the two can be compared.
    void run_reference(RecurrentMode mode, const RnnAttributes& ra,
                       const GruAttributes& ga, const LstmAttributes& la) {
        got_y = y;
        got_yh = yh;
        got_yc = yc;
        std::fill(y.begin(), y.end(), T{});
        std::fill(yh.begin(), yh.end(), T{});
        std::fill(yc.begin(), yc.end(), T{});

        switch (mode) {
        case RecurrentMode::Rnn:
            ref::rnn_ref(ra, outs, ins_ref);
            break;
        case RecurrentMode::Gru:
            ref::gru_ref(ga, outs, ins_ref);
            break;
        case RecurrentMode::Lstm:
            ref::lstm_ref(la, outs, ins_ref);
            break;
        }
    }

    std::vector<T> got_y, got_yh, got_yc;
};

/// Shared tail of the three checks: run the op, then the reference, and compare
/// every output the case asked for.
template <typename T>
void compare_outputs(Fixture<T>& f, const Geo& g, bool lstm, float tol,
                     const char* what) {
    const float dy = g.want_y ? max_diff(f.got_y, f.y) : 0.0f;
    const float dh = max_diff(f.got_yh, f.yh);
    const float dc = (lstm && g.want_yc) ? max_diff(f.got_yc, f.yc) : 0.0f;
    if (dy > tol || dh > tol || dc > tol) {
        throw std::runtime_error(std::string(what) + ": Y " + std::to_string(dy) +
                                 ", Y_h " + std::to_string(dh) + ", Y_c " +
                                 std::to_string(dc) + " > " + std::to_string(tol));
    }
}

template <typename T>
void check_rnn(const Geo& g, const RnnAttributes& a, float tol) {
    Fixture<T> f;
    f.build(RecurrentMode::Rnn, g);
    auto op = Rnn::create(a, Backend::CPU);
    NNOPS_EXPECT_TRUE(is_ok(op->compute(f.outs, f.ins, ComputeContext{}, nullptr)));
    f.run_reference(RecurrentMode::Rnn, a, {}, {});
    compare_outputs(f, g, false, tol, "rnn");
}

template <typename T>
void check_gru(const Geo& g, const GruAttributes& a, float tol) {
    Fixture<T> f;
    f.build(RecurrentMode::Gru, g);
    auto op = Gru::create(a, Backend::CPU);
    NNOPS_EXPECT_TRUE(is_ok(op->compute(f.outs, f.ins, ComputeContext{}, nullptr)));
    f.run_reference(RecurrentMode::Gru, {}, a, {});
    compare_outputs(f, g, false, tol, "gru");
}

template <typename T>
void check_lstm(const Geo& g, const LstmAttributes& a, float tol) {
    Fixture<T> f;
    f.build(RecurrentMode::Lstm, g);
    auto op = Lstm::create(a, Backend::CPU);
    NNOPS_EXPECT_TRUE(is_ok(op->compute(f.outs, f.ins, ComputeContext{}, nullptr)));
    f.run_reference(RecurrentMode::Lstm, {}, {}, a);
    compare_outputs(f, g, true, tol, "lstm");
}

template <typename T>
void expect_same(const std::vector<T>& a, const std::vector<T>& b, const char* what) {
    const float d = max_diff(a, b);
    if (d != 0.0f) {
        throw std::runtime_error(std::string("prepacked run differs in ") + what +
                                 " by " + std::to_string(d));
    }
}

/// One op, twice: as its caller would run it with the raw weights, then with W
/// replaced by the blob prepackWeights() built from those same weights. The blob
/// holds the interleave and the packs the kernel would otherwise redo per call,
/// in the same order, so the two runs must agree bit for bit — a tolerance here
/// would hide a blob read at the wrong offset.
template <typename T, typename OpT, typename AttrT>
void check_prepacked(RecurrentMode mode, const Geo& g, const AttrT& a) {
    Fixture<T> f;
    f.build(mode, g);
    auto op = OpT::create(a, Backend::CPU);
    NNOPS_EXPECT_TRUE(is_ok(op->compute(f.outs, f.ins, ComputeContext{}, nullptr)));
    const std::vector<T> raw_y = f.y, raw_yh = f.yh, raw_yc = f.yc;

    // The documented two-step: an empty view asks for the descriptor, a
    // populated one receives the data.
    TensorView query{};
    op->prepackWeights(f.ins, std::span<TensorView>(&query, 1), {});
    // The answer carries no storage — rank separates it from "no packed form",
    // not is_empty(), which is the question that was asked.
    NNOPS_EXPECT_EQ(query.rank(), 1);
    NNOPS_EXPECT_TRUE(query.numel() > 0);
    NNOPS_EXPECT_TRUE(query.layout() == TensorLayout::PackedRecurrent);
    // A quantized blob is its own element size — s8, one byte per panel — so the
    // buffer is sized from what the query answered, not from the compute type.
    NNOPS_EXPECT_TRUE(query.data_type() == f.ins[1].data_type());

    const size_t blob_bytes =
        static_cast<size_t>(query.numel()) * data_type_size(query.data_type());
    std::vector<uint8_t> blob(blob_bytes, uint8_t{});
    const int64_t bs[1] = {query.numel()};
    TensorView packed(std::span<const int64_t>(bs, 1), query.data_type(), blob.data(),
                      TensorLayout::PackedRecurrent);
    op->prepackWeights(f.ins, std::span<TensorView>(&packed, 1), {});

    f.ins[1] = packed;
    std::fill(f.y.begin(), f.y.end(), T{});
    std::fill(f.yh.begin(), f.yh.end(), T{});
    std::fill(f.yc.begin(), f.yc.end(), T{});
    NNOPS_EXPECT_TRUE(is_ok(op->compute(f.outs, f.ins, ComputeContext{}, nullptr)));
    expect_same(f.y, raw_y, "Y");
    expect_same(f.yh, raw_yh, "Y_h");
    expect_same(f.yc, raw_yc, "Y_c");
}

// =========================================================================
//  RNN
// =========================================================================

NNOPS_TEST(rnn_forward_matches_reference) {
    RnnAttributes a;
    a.hidden_size = 6;
    Geo g;
    g.T = 5; g.B = 3; g.I = 7; g.H = 6;
    g.h0 = false;
    check_rnn<float>(g, a, kF32Tol);
}

NNOPS_TEST(rnn_hidden_padding_tails) {
    // H % 8 != 0 exercises the padded state lanes and the masked Y store.
    for (int64_t H : {1, 5, 8, 13}) {
        RnnAttributes a;
        a.hidden_size = H;
        Geo g;
        g.T = 3; g.B = 4; g.I = 5; g.H = H; g.bias = true;
        check_rnn<float>(g, a, kF32Tol);
    }
}

NNOPS_TEST(rnn_activation_clip_and_reverse) {
    for (auto act : {RnnActivation::Relu, RnnActivation::LeakyRelu,
                     RnnActivation::Elu, RnnActivation::Tanh}) {
        RnnAttributes a;
        a.hidden_size = 9;
        a.activation = act;
        a.activation_alpha = 0.25f;
        a.clip = 0.75f;
        a.direction = RnnDirection::Reverse;
        Geo g;
        g.T = 4; g.B = 2; g.I = 6; g.H = 9; g.dir = RnnDirection::Reverse;
        g.seq = true; g.h0 = true;
        check_rnn<float>(g, a, kF32Tol);
    }
}

NNOPS_TEST(rnn_bidirectional_matches_reference) {
    RnnAttributes a;
    a.hidden_size = 7;
    a.direction = RnnDirection::Bidirectional;
    Geo g;
    g.T = 4; g.B = 3; g.I = 4; g.H = 7; g.dir = RnnDirection::Bidirectional;
    g.h0 = false;
    check_rnn<float>(g, a, kF32Tol);
}

NNOPS_TEST(rnn_f16_matches_reference) {
    RnnAttributes a;
    a.hidden_size = 10;
    a.direction = RnnDirection::Bidirectional;
    Geo g;
    g.T = 3; g.B = 2; g.I = 6; g.H = 10; g.dir = RnnDirection::Bidirectional;
    check_rnn<half>(g, a, test::kF16AccumTol);
}

// =========================================================================
//  LSTM
// =========================================================================

NNOPS_TEST(lstm_forward_matches_reference) {
    LstmAttributes a;
    a.hidden_size = 8;
    Geo g;
    g.T = 5; g.B = 3; g.I = 6; g.H = 8;
    g.h0 = false; g.c0 = true;
    check_lstm<float>(g, a, kF32Tol);
}

NNOPS_TEST(lstm_hidden_padding_tails) {
    for (int64_t H : {1, 7, 8, 11}) {
        LstmAttributes a;
        a.hidden_size = H;
        Geo g;
        g.T = 3; g.B = 2; g.I = 4; g.H = H; g.h0 = true; g.c0 = true;
        check_lstm<float>(g, a, kF32Tol);
    }
}

NNOPS_TEST(lstm_input_forget_clip_and_reverse) {
    LstmAttributes a;
    a.hidden_size = 9;
    a.input_forget = true;
    a.clip = 1.25f;
    a.direction = RnnDirection::Reverse;
    Geo g;
    g.T = 4; g.B = 2; g.I = 5; g.H = 9; g.dir = RnnDirection::Reverse;
    g.seq = true; g.h0 = true; g.c0 = true;
    check_lstm<float>(g, a, kF32Tol);
}

NNOPS_TEST(lstm_bidirectional_sequence_lens) {
    LstmAttributes a;
    a.hidden_size = 6;
    a.direction = RnnDirection::Bidirectional;
    Geo g;
    g.T = 5; g.B = 4; g.I = 3; g.H = 6; g.dir = RnnDirection::Bidirectional;
    g.seq = true; g.h0 = true; g.c0 = true;
    check_lstm<float>(g, a, kF32Tol);
}

NNOPS_TEST(lstm_alternate_activations) {
    LstmAttributes a;
    a.hidden_size = 5;
    a.activation_f = RnnActivation::Sigmoid;
    a.activation_g = RnnActivation::Relu;
    a.activation_h = RnnActivation::Tanh;
    Geo g;
    g.T = 3; g.B = 2; g.I = 3; g.H = 5;
    check_lstm<float>(g, a, kF32Tol);
}

NNOPS_TEST(lstm_optional_outputs) {
    // No Y at all: only the final states come back.
    {
        LstmAttributes a;
        a.hidden_size = 4;
        Geo g;
        g.T = 3; g.B = 2; g.I = 3; g.H = 4; g.h0 = true;
        g.want_y = false;
        check_lstm<float>(g, a, kF32Tol);
    }
    // No Y_c either.
    {
        LstmAttributes a;
        a.hidden_size = 4;
        Geo g;
        g.T = 3; g.B = 2; g.I = 3; g.H = 4;
        g.want_yc = false;
        check_lstm<float>(g, a, kF32Tol);
    }
}

NNOPS_TEST(lstm_f16_matches_reference) {
    LstmAttributes a;
    a.hidden_size = 8;
    a.direction = RnnDirection::Reverse;
    Geo g;
    g.T = 3; g.B = 2; g.I = 5; g.H = 8; g.dir = RnnDirection::Reverse;
    g.h0 = false; g.c0 = true;
    check_lstm<half>(g, a, test::kF16AccumTol);
}

// =========================================================================
//  GRU
// =========================================================================

NNOPS_TEST(gru_linear_before_reset_both) {
    for (bool lbr : {false, true}) {
        GruAttributes a;
        a.hidden_size = 7;
        a.linear_before_reset = lbr;
        Geo g;
        g.T = 4; g.B = 3; g.I = 5; g.H = 7;
        g.h0 = false;
        check_gru<float>(g, a, kF32Tol);
    }
}

NNOPS_TEST(gru_hidden_padding_tails) {
    for (int64_t H : {1, 5, 8, 15}) {
        GruAttributes a;
        a.hidden_size = H;
        Geo g;
        g.T = 3; g.B = 2; g.I = 3; g.H = H; g.h0 = true;
        check_gru<float>(g, a, kF32Tol);
    }
}

NNOPS_TEST(gru_reverse_and_bidirectional) {
    for (bool lbr : {false, true}) {
        for (auto dir : {RnnDirection::Reverse, RnnDirection::Bidirectional}) {
            GruAttributes a;
            a.hidden_size = 6;
            a.linear_before_reset = lbr;
            a.direction = dir;
            a.clip = 2.0f;
            Geo g;
            g.T = 4; g.B = 3; g.I = 4; g.H = 6; g.dir = dir;
            g.seq = true; g.h0 = true;
            check_gru<float>(g, a, kF32Tol);
        }
    }
}

NNOPS_TEST(gru_f16_matches_reference) {
    for (bool lbr : {false, true}) {
        GruAttributes a;
        a.hidden_size = 9;
        a.linear_before_reset = lbr;
        Geo g;
        g.T = 3; g.B = 2; g.I = 5; g.H = 9; g.h0 = true;
        check_gru<half>(g, a, test::kF16AccumTol);
    }
}

// =========================================================================
//  W8A8 — s8 weights, per-row scale, dynamically quantized activations
// =========================================================================

/// The weights are exactly what the reference solves with (the fixture's s8 rows
/// dequantize to the fp values it is handed), so what is left is the activation's
/// own per-row quantisation: half an int8 step of that row's max, run through the
/// gate functions. A weight scale attached to the wrong column would be orders of
/// magnitude larger.
constexpr float kQ8Tol = 2e-2f;

NNOPS_TEST(q8_rnn_matches_reference) {
    for (int64_t H : {5, 8, 16}) {
        RnnAttributes a;
        a.hidden_size = H;
        Geo g;
        g.T = 4; g.B = 3; g.I = 7; g.H = H; g.h0 = true; g.q8 = true;
        check_rnn<float>(g, a, kQ8Tol);
    }
}

NNOPS_TEST(q8_lstm_matches_reference) {
    LstmAttributes a;
    a.hidden_size = 9;
    a.direction = RnnDirection::Bidirectional;
    Geo g;
    g.T = 4; g.B = 2; g.I = 5; g.H = 9; g.dir = RnnDirection::Bidirectional;
    g.seq = true; g.h0 = true; g.c0 = true; g.q8 = true;
    check_lstm<float>(g, a, kQ8Tol);
}

NNOPS_TEST(q8_gru_matches_reference) {
    for (bool lbr : {false, true}) {
        GruAttributes a;
        a.hidden_size = 7;
        a.linear_before_reset = lbr;
        a.clip = 1.5f;
        Geo g;
        g.T = 3; g.B = 3; g.I = 5; g.H = 7; g.h0 = true; g.q8 = true;
        check_gru<float>(g, a, kQ8Tol);
    }
}

NNOPS_TEST(q8_hidden_past_k_block_cap) {
    // H > 512 splits the recurrent product into two k blocks, so R's panels in
    // the second slab are addressed at a different offset from the first's — and
    // a wrong offset is an O(1) error, not a noise-level one. The weights are
    // scaled down because at this width the unscaled random ones saturate every
    // tanh, where the 520-term product's quantisation noise is amplified.
    RnnAttributes a;
    a.hidden_size = 520;
    Geo g;
    g.T = 3; g.B = 2; g.I = 6; g.H = 520; g.h0 = true; g.q8 = true;
    g.q8_wscale = 0.1f;
    check_rnn<float>(g, a, kQ8Tol);
}

NNOPS_TEST(q8_s8_activation_and_f16_compute) {
    // X arrives quantized with its own per-row table: the projection packs the
    // bytes as stored instead of quantizing them. And an f16 X makes the gate
    // math f16, while the weights stay s8.
    {
        LstmAttributes a;
        a.hidden_size = 8;
        Geo g;
        g.T = 3; g.B = 2; g.I = 6; g.H = 8; g.h0 = true; g.c0 = true;
        g.q8 = true; g.x_s8 = true;
        check_lstm<float>(g, a, kQ8Tol);
    }
    {
        RnnAttributes a;
        a.hidden_size = 10;
        Geo g;
        g.T = 3; g.B = 2; g.I = 5; g.H = 10; g.h0 = true; g.q8 = true;
        check_rnn<half>(g, a, test::kF16AccumTol + kQ8Tol);
    }
    {
        // Pitched X and Y on the quantized route: the panel pack and the Y store
        // both honour the row stride.
        RnnAttributes a;
        a.hidden_size = 6;
        Geo g;
        g.T = 3; g.B = 2; g.I = 4; g.H = 6; g.h0 = true; g.q8 = true; g.pad = 3;
        check_rnn<float>(g, a, kQ8Tol);
    }
}

NNOPS_TEST(q8_prepacked_matches_raw) {
    // The blob's panels and scales go through the same writer compute() uses, so
    // the prepacked run is bit-identical — a tolerance here would hide a region
    // read at the wrong offset.
    {
        LstmAttributes a;
        a.hidden_size = 9;
        Geo g;
        g.T = 4; g.B = 2; g.I = 5; g.H = 9; g.h0 = true; g.c0 = true; g.q8 = true;
        check_prepacked<float, Lstm>(RecurrentMode::Lstm, g, a);
    }
    for (bool lbr : {false, true}) {
        GruAttributes a;
        a.hidden_size = 7;
        a.linear_before_reset = lbr;
        Geo g;
        g.T = 3; g.B = 3; g.I = 5; g.H = 7; g.h0 = true; g.q8 = true;
        check_prepacked<float, Gru>(RecurrentMode::Gru, g, a);
    }
    {
        // A quantized blob for the f16 compute route: the bias rides the blob as
        // f32 and the driver converts it.
        RnnAttributes a;
        a.hidden_size = 11;
        Geo g;
        g.T = 3; g.B = 2; g.I = 6; g.H = 11; g.h0 = true; g.q8 = true;
        check_prepacked<half, Rnn>(RecurrentMode::Rnn, g, a);
    }
}

NNOPS_TEST(q8_rejects_asymmetric_quantisation) {
    // Symmetric only: a non-zero zero point needs correction terms the kernel
    // does not carry, so it is refused rather than silently dropped. The
    // descriptor cannot carry quant params, so this is compute()'s verdict.
    LstmAttributes a;
    a.hidden_size = 5;
    Geo g;
    g.T = 2; g.B = 2; g.I = 4; g.H = 5; g.q8 = true;
    Fixture<float> f;
    f.build(RecurrentMode::Lstm, g);
    auto op = Lstm::create(a, Backend::CPU);

    // The baseline runs.
    NNOPS_EXPECT_TRUE(is_ok(op->compute(f.outs, f.ins, ComputeContext{}, nullptr)));

    const int64_t D = 1, G = 4;
    int64_t s_w[3] = {D, G * g.H, g.I};
    {
        QuantParams qp;
        qp.granularity = QuantGranularity::PerToken;
        qp.scale_data = f.wscale.data();
        qp.num_scales = D * G * g.H;
        qp.zero_point = 1;  // asymmetric
        TensorView wv(std::span<const int64_t>(s_w, 3), DataType::s8, f.wq.data(),
                      TensorLayout::NCHW, qp);
        std::vector<TensorView> ins = f.ins;
        ins[1] = wv;
        NNOPS_EXPECT_EQ(op->compute(f.outs, ins, ComputeContext{}, nullptr),
                        Status::unsupported);
    }
    {
        // A scale table shorter than one row per gate row is not a per-row table.
        QuantParams qp;
        qp.granularity = QuantGranularity::PerToken;
        qp.scale_data = f.wscale.data();
        qp.num_scales = 3;
        TensorView wv(std::span<const int64_t>(s_w, 3), DataType::s8, f.wq.data(),
                      TensorLayout::NCHW, qp);
        std::vector<TensorView> ins = f.ins;
        ins[1] = wv;
        NNOPS_EXPECT_EQ(op->compute(f.outs, ins, ComputeContext{}, nullptr),
                        Status::unsupported);
    }
    {
        // A blob bakes the scales in and hides the weight from compute()'s
        // checks, so the packer applies the same rule: an asymmetric W gets no
        // packed form (a rank-0 answer), rather than a blob with wrong scales.
        QuantParams qp;
        qp.granularity = QuantGranularity::PerToken;
        qp.scale_data = f.wscale.data();
        qp.num_scales = D * G * g.H;
        qp.zero_point = -3;
        TensorView wv(std::span<const int64_t>(s_w, 3), DataType::s8, f.wq.data(),
                      TensorLayout::NCHW, qp);
        std::vector<TensorView> ins = f.ins;
        ins[1] = wv;
        TensorView q{};
        op->prepackWeights(ins, std::span<TensorView>(&q, 1), {});
        NNOPS_EXPECT_EQ(q.rank(), 0);
    }
}

NNOPS_TEST(q8_validation_is_dtype_only) {
    // validate() reads descriptors, which carry no quant params: it can only say
    // that an s8 weight pairs with an s8 R, and that the bias, the states and the
    // outputs are the activation's floating type.
    std::vector<TensorDesc> in(3);
    in[0].rank = 3; in[0].dims = {4, 2, 3}; in[0].dtype = DataType::f32;
    in[1].rank = 3; in[1].dims = {1, 4 * 5, 3}; in[1].dtype = DataType::s8;
    in[2].rank = 3; in[2].dims = {1, 4 * 5, 5}; in[2].dtype = DataType::s8;
    std::vector<TensorDesc> out(3);
    out[0].rank = 4; out[0].dims = {4, 1, 2, 5}; out[0].dtype = DataType::f32;
    out[1].rank = 3; out[1].dims = {1, 2, 5}; out[1].dtype = DataType::f32;
    out[2].rank = 3; out[2].dims = {1, 2, 5}; out[2].dtype = DataType::f32;

    LstmAttributes a;
    a.hidden_size = 5;
    auto op = Lstm::create(a);
    NNOPS_EXPECT_TRUE(is_ok(op->validate(in, out)));

    // An f16 activation with s8 weights is the f16 compute route.
    auto in_f16 = in;
    in_f16[0].dtype = DataType::f16;
    auto out_f16 = out;
    for (auto& o : out_f16) { o.dtype = DataType::f16; }
    NNOPS_EXPECT_TRUE(is_ok(op->validate(in_f16, out_f16)));

    // An s8 R is required beside an s8 W.
    auto bad = in;
    bad[2].dtype = DataType::f32;
    NNOPS_EXPECT_EQ(op->validate(bad, out), Status::invalid_argument);

    // The bias and states are the compute dtype, not the weight's.
    auto bad_bias = in;
    bad_bias.resize(4);
    bad_bias[3].rank = 2;
    bad_bias[3].dims = {1, 2 * 4 * 5};
    bad_bias[3].dtype = DataType::s8;
    NNOPS_EXPECT_EQ(op->validate(bad_bias, out), Status::invalid_argument);
}

// =========================================================================
//  Cross-cutting
// =========================================================================

NNOPS_TEST(hidden_past_the_k_block_cap) {
    // H > 128 splits the recurrent product into two k-blocks: the first builds
    // the tile, the second accumulates into it.
    LstmAttributes a;
    a.hidden_size = 136;
    Geo g;
    g.T = 2; g.B = 2; g.I = 6; g.H = 136;
    g.h0 = false; g.c0 = true;
    check_lstm<float>(g, a, 2e-3f);
}

NNOPS_TEST(sequence_lens_zero_row) {
    // Row 0 has length 0 (set by the fixture): it never steps, so Y is zero
    // everywhere and the final state is zeroed rather than the initial state.
    LstmAttributes a;
    a.hidden_size = 5;
    Geo g;
    g.T = 4; g.B = 3; g.I = 3; g.H = 5;
    g.seq = true; g.h0 = true; g.c0 = true;
    Fixture<float> f;
    f.build(RecurrentMode::Lstm, g);
    auto op = Lstm::create(a, Backend::CPU);
    NNOPS_EXPECT_TRUE(is_ok(op->compute(f.outs, f.ins, ComputeContext{}, nullptr)));
    for (int64_t d = 0; d < 2; ++d) {
        for (int64_t h = 0; h < 5; ++h) {
            const size_t at = static_cast<size_t>((d * g.B + 0) * g.H + h);
            if (d == 0) {  // one direction only
                NNOPS_EXPECT_NEAR(f.yh[at], 0.0f, 1e-6f);
            }
        }
    }
    f.run_reference(RecurrentMode::Lstm, {}, {}, a);
    compare_outputs(f, g, true, kF32Tol, "seq0");
}

NNOPS_TEST(threaded_matches_sequential) {
    LstmAttributes a;
    a.hidden_size = 12;
    a.direction = RnnDirection::Bidirectional;
    Geo g;
    g.T = 4; g.B = 16; g.I = 8; g.H = 12; g.dir = RnnDirection::Bidirectional;
    g.h0 = false; g.c0 = true;

    Fixture<float> seq_f, th_f;
    seq_f.build(RecurrentMode::Lstm, g);
    th_f.build(RecurrentMode::Lstm, g);
    th_f.y = seq_f.y;  // same initial contents, though both start zeroed

    auto op = Lstm::create(a, Backend::CPU);
    NNOPS_EXPECT_TRUE(is_ok(op->compute(seq_f.outs, seq_f.ins, ComputeContext{}, nullptr)));

    ComputeContext ctx;
    ctx.cpu = openmp_backend(4);
    NNOPS_EXPECT_TRUE(is_ok(op->compute(th_f.outs, th_f.ins, ctx, nullptr)));

    // Not bit-exact: the thread count picks a different tile grid, and the MMA
    // micro-kernels for different m-levels accumulate in different orders.
    NNOPS_EXPECT_TRUE(max_diff(seq_f.y, th_f.y) < 1e-5f);
    NNOPS_EXPECT_TRUE(max_diff(seq_f.yh, th_f.yh) < 1e-5f);
    NNOPS_EXPECT_TRUE(max_diff(seq_f.yc, th_f.yc) < 1e-5f);
}

NNOPS_TEST(batch_one_narrow_and_wide_hidden) {
    // A one-row batch still steps m-level 1 through the MMA, and these widths
    // pick its edges: H < 8 leaves RNN and the GRU h~ phase with a sub-panel
    // width, and H > 128 splits k across two blocks.
    {
        RnnAttributes a;
        a.hidden_size = 5;
        Geo g;
        g.T = 4; g.B = 1; g.I = 3; g.H = 5; g.h0 = true;
        check_rnn<float>(g, a, kF32Tol);
    }
    {
        GruAttributes a;
        a.hidden_size = 5;
        a.linear_before_reset = true;
        Geo g;
        g.T = 4; g.B = 1; g.I = 3; g.H = 5; g.h0 = true;
        check_gru<float>(g, a, kF32Tol);
    }
    {
        LstmAttributes a;
        a.hidden_size = 136;
        Geo g;
        g.T = 3; g.B = 1; g.I = 4; g.H = 136; g.h0 = false; g.c0 = true;
        check_lstm<float>(g, a, 2e-3f);
    }
    {
        LstmAttributes a;
        a.hidden_size = 9;
        a.direction = RnnDirection::Bidirectional;
        Geo g;
        g.T = 3; g.B = 1; g.I = 6; g.H = 9; g.dir = RnnDirection::Bidirectional;
        g.seq = true; g.h0 = true; g.c0 = true;
        check_lstm<float>(g, a, kF32Tol);
    }
    {
        GruAttributes a;
        a.hidden_size = 11;
        Geo g;
        g.T = 3; g.B = 1; g.I = 5; g.H = 11; g.h0 = true;
        check_gru<half>(g, a, test::kF16AccumTol);
    }
    {
        RnnAttributes a;
        a.hidden_size = 12;
        a.direction = RnnDirection::Reverse;
        Geo g;
        g.T = 3; g.B = 1; g.I = 4; g.H = 12; g.dir = RnnDirection::Reverse;
        check_rnn<half>(g, a, test::kF16AccumTol);
    }
}

NNOPS_TEST(batch_one_threaded_matches_sequential) {
    // H = 256 is wide enough that a one-row step splits its columns across the
    // team, so the GEMV runs over several column blocks in parallel and each
    // must write only its own.
    LstmAttributes a;
    a.hidden_size = 256;
    Geo g;
    g.T = 3; g.B = 1; g.I = 4; g.H = 256; g.h0 = true; g.c0 = true;

    Fixture<float> seq_f, th_f;
    seq_f.build(RecurrentMode::Lstm, g);
    th_f.build(RecurrentMode::Lstm, g);

    auto op = Lstm::create(a, Backend::CPU);
    NNOPS_EXPECT_TRUE(is_ok(op->compute(seq_f.outs, seq_f.ins, ComputeContext{}, nullptr)));

    ComputeContext ctx;
    ctx.cpu = openmp_backend(4);
    NNOPS_EXPECT_TRUE(is_ok(op->compute(th_f.outs, th_f.ins, ctx, nullptr)));

    // Each block owns a disjoint width of the gate row, so a block that also
    // reads its neighbour's within-row offset still lands on the reference.
    seq_f.run_reference(RecurrentMode::Lstm, {}, {}, a);
    th_f.run_reference(RecurrentMode::Lstm, {}, {}, a);
    compare_outputs(seq_f, g, true, kF32Tol, "lstm one block");
    compare_outputs(th_f, g, true, kF32Tol, "lstm split columns");
}

NNOPS_TEST(wide_step_threaded_matches_sequential) {
    // Rows as well as columns: H = 256 over four threads splits the step's grid
    // both ways, and each block must take its own width of the bias vectors.
    LstmAttributes a;
    a.hidden_size = 256;
    Geo g;
    g.T = 2; g.B = 8; g.I = 8; g.H = 256; g.h0 = true; g.c0 = true;

    Fixture<float> f;
    f.build(RecurrentMode::Lstm, g);
    auto op = Lstm::create(a, Backend::CPU);
    ComputeContext ctx;
    ctx.cpu = openmp_backend(4);
    NNOPS_EXPECT_TRUE(is_ok(op->compute(f.outs, f.ins, ctx, nullptr)));
    f.run_reference(RecurrentMode::Lstm, {}, {}, a);
    compare_outputs(f, g, true, kF32Tol, "lstm wide split");
}

NNOPS_TEST(pitched_views) {
    // X and Y are slices of larger tensors: their rows have a pitch the kernel
    // has to honour on both sides. The padding is filled like data and is part
    // of the comparison, so a store past a row lands where it can be seen.
    {
        LstmAttributes a;
        a.hidden_size = 9;
        a.direction = RnnDirection::Bidirectional;
        Geo g;
        g.T = 4; g.B = 2; g.I = 5; g.H = 9; g.dir = RnnDirection::Bidirectional;
        g.h0 = true; g.c0 = true; g.pad = 3;
        check_lstm<float>(g, a, kF32Tol);
    }
    {
        GruAttributes a;
        a.hidden_size = 11;
        a.linear_before_reset = true;
        Geo g;
        g.T = 3; g.B = 3; g.I = 7; g.H = 11; g.h0 = true; g.pad = 5;
        check_gru<half>(g, a, test::kF16AccumTol);
    }
    {
        RnnAttributes a;
        a.hidden_size = 6;
        a.direction = RnnDirection::Reverse;
        Geo g;
        g.T = 3; g.B = 2; g.I = 4; g.H = 6; g.dir = RnnDirection::Reverse;
        g.seq = true; g.pad = 1;
        check_rnn<float>(g, a, kF32Tol);
    }
}

NNOPS_TEST(prepacked_weights_match_raw) {
    // Every branch of the pack — RNN and LSTM's single phase, GRU's two, GRU's
    // merged three-gate phase, both precisions, bias present and absent,
    // bidirectional, a hidden that is not a multiple of 8, H past the k-block
    // cap, and a hidden inferred from R (which the query has to reproduce).
    {
        RnnAttributes a;
        a.hidden_size = 6;
        Geo g;
        g.T = 4; g.B = 3; g.I = 7; g.H = 6; g.h0 = true;
        check_prepacked<float, Rnn>(RecurrentMode::Rnn, g, a);
    }
    {
        RnnAttributes a;
        a.hidden_size = 5;
        Geo g;
        g.T = 3; g.B = 2; g.I = 4; g.H = 5; g.bias = false; g.h0 = true;
        check_prepacked<half, Rnn>(RecurrentMode::Rnn, g, a);
    }
    for (bool lbr : {false, true}) {
        GruAttributes a;
        a.hidden_size = 7;
        a.linear_before_reset = lbr;
        a.direction = RnnDirection::Bidirectional;
        Geo g;
        g.T = 3; g.B = 3; g.I = 5; g.H = 7; g.dir = RnnDirection::Bidirectional;
        g.seq = true; g.h0 = true;
        check_prepacked<float, Gru>(RecurrentMode::Gru, g, a);
    }
    {
        GruAttributes a;  // no bias: the blob carries the zeroes instead
        a.hidden_size = 9;
        Geo g;
        g.T = 3; g.B = 2; g.I = 6; g.H = 9; g.bias = false; g.h0 = true;
        check_prepacked<half, Gru>(RecurrentMode::Gru, g, a);
    }
    {
        LstmAttributes a;
        a.hidden_size = 9;
        a.direction = RnnDirection::Reverse;
        Geo g;
        g.T = 4; g.B = 2; g.I = 5; g.H = 9; g.dir = RnnDirection::Reverse;
        g.seq = true; g.h0 = true; g.c0 = true;
        check_prepacked<float, Lstm>(RecurrentMode::Lstm, g, a);
    }
    {
        LstmAttributes a;  // hidden_size left at 0: read off R by both sides
        Geo g;
        g.T = 3; g.B = 2; g.I = 4; g.H = 8; g.h0 = true; g.c0 = true;
        check_prepacked<half, Lstm>(RecurrentMode::Lstm, g, a);
    }
    {
        LstmAttributes a;  // H past the 128 k-block cap
        a.hidden_size = 136;
        Geo g;
        g.T = 2; g.B = 1; g.I = 4; g.H = 136; g.c0 = true;
        check_prepacked<float, Lstm>(RecurrentMode::Lstm, g, a);
    }
}

NNOPS_TEST(prepacked_weights_reject_foreign_blobs) {
    // A blob says what it was packed for, so a layer of another geometry — or a
    // buffer that never went through the packer — is refused, not misread.
    LstmAttributes wide;
    wide.hidden_size = 11;
    Geo gw;
    gw.T = 2; gw.B = 2; gw.I = 5; gw.H = 11;
    Fixture<float> fw;
    fw.build(RecurrentMode::Lstm, gw);
    auto opw = Lstm::create(wide, Backend::CPU);

    TensorView q{};
    opw->prepackWeights(fw.ins, std::span<TensorView>(&q, 1), {});
    NNOPS_EXPECT_TRUE(q.numel() > 0);
    std::vector<float> blob(static_cast<size_t>(q.numel()), 0.0f);
    const int64_t bs[1] = {q.numel()};
    TensorView packed(std::span<const int64_t>(bs, 1), DataType::f32, blob.data(),
                      TensorLayout::PackedRecurrent);
    opw->prepackWeights(fw.ins, std::span<TensorView>(&packed, 1), {});
    // The geometry it was packed for still runs.
    fw.ins[1] = packed;
    NNOPS_EXPECT_TRUE(is_ok(opw->compute(fw.outs, fw.ins, ComputeContext{}, nullptr)));

    // A narrower layer with the same I: only the header's hidden disagrees.
    LstmAttributes narrow;
    narrow.hidden_size = 6;
    Geo gn;
    gn.T = 2; gn.B = 2; gn.I = 5; gn.H = 6;
    Fixture<float> fn;
    fn.build(RecurrentMode::Lstm, gn);
    fn.ins[1] = packed;
    auto opn = Lstm::create(narrow, Backend::CPU);
    NNOPS_EXPECT_EQ(opn->compute(fn.outs, fn.ins, ComputeContext{}, nullptr),
                    Status::invalid_argument);

    // Zeroes of the right size: the magic does not match.
    std::vector<float> zeros(blob.size(), 0.0f);
    TensorView not_a_blob(std::span<const int64_t>(bs, 1), DataType::f32, zeros.data(),
                          TensorLayout::PackedRecurrent);
    fw.ins[1] = not_a_blob;
    NNOPS_EXPECT_EQ(opw->compute(fw.outs, fw.ins, ComputeContext{}, nullptr),
                    Status::invalid_argument);
}

NNOPS_TEST(hidden_size_inferred_from_R) {
    LstmAttributes a;  // hidden_size left at 0
    Geo g;
    g.T = 2; g.B = 2; g.I = 3; g.H = 4;
    check_lstm<float>(g, a, kF32Tol);
}

// =========================================================================
//  Shape inference and validation
// =========================================================================


NNOPS_TEST(shape_inference) {
    int64_t s_x[3] = {5, 3, 7};
    int64_t s_w[3] = {2, 4 * 6, 7};
    int64_t s_r[3] = {2, 4 * 6, 6};
    std::vector<TensorDesc> in(3);
    in[0].rank = 3; in[0].dims = {5, 3, 7}; in[0].dtype = DataType::f32;
    in[1].rank = 3; in[1].dims = {2, 24, 7}; in[1].dtype = DataType::f32;
    in[2].rank = 3; in[2].dims = {2, 24, 6}; in[2].dtype = DataType::f32;

    LstmAttributes la;
    la.hidden_size = 6;
    la.direction = RnnDirection::Bidirectional;
    auto lstm = Lstm::create(la);
    auto out = lstm->getOutputTensorDesc(in);
    NNOPS_EXPECT_EQ(out.size(), size_t{3});
    NNOPS_EXPECT_EQ(out[0].rank, 4);
    NNOPS_EXPECT_EQ(out[0].dims[0], 5);
    NNOPS_EXPECT_EQ(out[0].dims[1], 2);
    NNOPS_EXPECT_EQ(out[0].dims[2], 3);
    NNOPS_EXPECT_EQ(out[0].dims[3], 6);
    NNOPS_EXPECT_EQ(out[1].rank, 3);
    NNOPS_EXPECT_EQ(out[1].dims[0], 2);
    NNOPS_EXPECT_EQ(out[2].dims[2], 6);

    // hidden_size 0 is inferred from R, and RNN/GRU have no Y_c.
    int64_t s_rr[3] = {1, 12, 4};
    in[1].dims = {1, 12, 7};
    in[2].dims = {1, 12, 4};
    RnnAttributes ra;
    auto rnn = Rnn::create(ra);
    auto rout = rnn->getOutputTensorDesc(in);
    NNOPS_EXPECT_EQ(rout.size(), size_t{2});
    NNOPS_EXPECT_EQ(rout[0].dims[3], 4);
    NNOPS_EXPECT_EQ(rout[1].dims[0], 1);
    (void)s_x; (void)s_w; (void)s_r; (void)s_rr;
}

NNOPS_TEST(validation_rejects) {
    auto make = [](int64_t h) {
        std::vector<TensorDesc> in(3);
        in[0].rank = 3; in[0].dims = {4, 2, 3}; in[0].dtype = DataType::f32;
        in[1].rank = 3; in[1].dims = {1, 4 * h, 3}; in[1].dtype = DataType::f32;
        in[2].rank = 3; in[2].dims = {1, 4 * h, h}; in[2].dtype = DataType::f32;
        return in;
    };
    std::vector<TensorDesc> gout(3);
    gout[0].rank = 4; gout[0].dims = {4, 1, 2, 5}; gout[0].dtype = DataType::f32;
    gout[1].rank = 3; gout[1].dims = {1, 2, 5}; gout[1].dtype = DataType::f32;
    gout[2].rank = 3; gout[2].dims = {1, 2, 5}; gout[2].dtype = DataType::f32;

    LstmAttributes ok_a;
    ok_a.hidden_size = 5;
    auto lstm = Lstm::create(ok_a);
    NNOPS_EXPECT_TRUE(is_ok(lstm->validate(make(5), gout)));

    // An attribute hidden width that disagrees with R.
    {
        LstmAttributes wide;
        wide.hidden_size = 6;
        auto op = Lstm::create(wide);
        NNOPS_EXPECT_EQ(op->validate(make(5), gout), Status::invalid_argument);
    }

    // A gate count that is not 4·H (an RNN weight handed to an LSTM).
    {
        auto in = make(5);
        in[1].dims = {1, 3 * 5, 3};
        in[2].dims = {1, 3 * 5, 5};
        NNOPS_EXPECT_EQ(lstm->validate(in, gout), Status::invalid_argument);
    }
    // Peepholes are not implemented, and are refused rather than ignored. Input 7
    // is P, so the three optionals before it have to be named as absent.
    {
        auto in = make(5);
        in.resize(8);  // rank-0 descriptors: bias, sequence_lens, initial_h, initial_c
        TensorDesc p;
        p.rank = 1; p.dims = {15}; p.dtype = DataType::f32;
        in[7] = p;
        NNOPS_EXPECT_EQ(lstm->validate(in, gout), Status::unsupported);

        // The same list without P is fine.
        in[7] = TensorDesc{};
        NNOPS_EXPECT_TRUE(is_ok(lstm->validate(in, gout)));
    }
    // An activation with no kernel here (ScaledTanh and HardSigmoid are ONNX's,
    // not this library's).
    {
        LstmAttributes bad;
        bad.hidden_size = 5;
        bad.activation_f = static_cast<RnnActivation>(42);
        auto op = Lstm::create(bad);
        NNOPS_EXPECT_EQ(op->validate(make(5), gout), Status::unsupported);
    }
    // Channel-last is not a layout this kernel reads.
    {
        LstmAttributes last;
        last.hidden_size = 5;
        last.layout = 1;
        auto op = Lstm::create(last);
        NNOPS_EXPECT_EQ(op->validate(make(5), gout), Status::unsupported);
    }
    // Too few inputs.
    {
        auto in = make(5);
        in.resize(2);
        NNOPS_EXPECT_EQ(lstm->validate(in, gout), Status::invalid_argument);
    }
}

NNOPS_TEST(unsupported_backend_and_dtype) {
    // A backend with no kernel still constructs, and compute() says so.
    LstmAttributes a;
    a.hidden_size = 4;
    auto op = Lstm::create(a, Backend::CUDA);
    NNOPS_EXPECT_EQ(op->getBackend(), Backend::CUDA);

    // bf16 has no kernel: the descriptor check rejects it up front.
    std::vector<TensorDesc> in(3);
    in[0].rank = 3; in[0].dims = {2, 2, 3}; in[0].dtype = DataType::bf16;
    in[1].rank = 3; in[1].dims = {1, 16, 3}; in[1].dtype = DataType::bf16;
    in[2].rank = 3; in[2].dims = {1, 16, 4}; in[2].dtype = DataType::bf16;
    std::vector<TensorDesc> out(2);
    auto op2 = Lstm::create(a);
    NNOPS_EXPECT_EQ(op2->validate(in, out), Status::unsupported);
}

}  // namespace

