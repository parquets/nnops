/// @file recurrent_ref.cpp
/// @brief Scalar reference for RNN, GRU and LSTM — the oracle for the fused
///        CPU kernel.
///
/// Deliberately the plain transcription of the ONNX equations, one row of one
/// timestep at a time, with f32 accumulation over the caller's element type.
/// The time axis runs the way onnxruntime runs it: forward ascends, reverse
/// descends, and a row past its sequence length holds its state and writes a
/// zero Y row (ORT reverses per sequence, so a reversed row only visits
/// t < sequence_lens[b], which is the same thing).
///
/// GRU's gates are ordered z, r, h and LSTM's i, o, f, c, in both W/R and the
/// bias, which is ONNX's order and the only place it appears here.

#include "nnops/detail/simd/simd.hpp"
#include "nnops/ops/gru.hpp"
#include "nnops/ops/lstm.hpp"
#include "nnops/ops/recurrent.hpp"
#include "nnops/ops/rnn.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace nnops::backend::cpu::reference {
namespace {

using namespace nnops::simd;

struct RefCfg {
    RecurrentMode mode = RecurrentMode::Rnn;
    int64_t hidden = 0;
    int64_t directions = 1;
    RnnDirection direction = RnnDirection::Forward;
    RnnActivation act_f = RnnActivation::Sigmoid;
    RnnActivation act_g = RnnActivation::Tanh;
    RnnActivation act_h = RnnActivation::Tanh;
    float alpha_f = 0.0f, alpha_g = 0.0f, alpha_h = 0.0f;
    float clip = 0.0f;
    bool input_forget = false;
    bool linear_before_reset = false;
};

float clip1(float v, float clip) noexcept {
    if (clip <= 0.0f) { return v; }
    return std::min(std::max(v, -clip), clip);
}

float act1(float v, RnnActivation a, float alpha) noexcept {
    switch (a) {
    case RnnActivation::Sigmoid:   return 1.0f / (1.0f + std::exp(-v));
    case RnnActivation::Tanh:      return std::tanh(v);
    case RnnActivation::Relu:      return v > 0.0f ? v : 0.0f;
    case RnnActivation::LeakyRelu: return v > 0.0f ? v : alpha * v;
    case RnnActivation::Elu:       return v > 0.0f ? v : alpha * (std::exp(v) - 1.0f);
    }
    return v;
}

template <typename T>
void recurrent_ref_impl(const RefCfg& cfg, std::span<TensorView> outputs,
                        std::span<const TensorView> inputs)
{
    const auto& x = inputs[0];
    const auto& w = inputs[1];
    const auto& r = inputs[2];
    const int64_t Tn = x.shape(0);
    const int64_t B = x.shape(1);
    const int64_t I = x.shape(2);
    const int64_t H = cfg.hidden;
    const int64_t D = cfg.directions;
    const int64_t G = recurrent_gates(cfg.mode);
    const bool is_gru = cfg.mode == RecurrentMode::Gru;
    const bool is_lstm = cfg.mode == RecurrentMode::Lstm;

    TensorView& y = outputs[0];
    TensorView& yh = outputs[1];
    TensorView unused;
    TensorView& yc = outputs.size() > 2 ? outputs[2] : unused;

    const bool has_b = inputs.size() > 3 && !inputs[3].is_empty();
    const bool has_seq = inputs.size() > 4 && !inputs[4].is_empty();
    const bool has_h0 = inputs.size() > 5 && !inputs[5].is_empty();
    const bool has_c0 = inputs.size() > 6 && !inputs[6].is_empty();
    const int32_t* seq = has_seq ? inputs[4].ptr<int32_t>() : nullptr;
    const T* bias = has_b ? inputs[3].ptr<T>() : nullptr;
    const int64_t bias_d = 2 * G * H;

    std::vector<float> h(static_cast<size_t>(B * H));
    std::vector<float> c(static_cast<size_t>(B * H));
    std::vector<float> hn(static_cast<size_t>(B * H));
    std::vector<float> cn(static_cast<size_t>(B * H));
    std::vector<float> pre(static_cast<size_t>(G * H));
    std::vector<float> rv(static_cast<size_t>(H));  // GRU's reset gate
    std::vector<float> zv(static_cast<size_t>(H));  // GRU's update gate
    std::vector<float> rh(static_cast<size_t>(H));  // r ⊙ H_{t-1}, the h gate's
                                                    // recurrent operand

    for (int64_t d = 0; d < D; ++d) {
        // ---- the direction's initial state -------------------------------
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t hh = 0; hh < H; ++hh) {
                const size_t at = static_cast<size_t>(b * H + hh);
                h[at] = has_h0 ? s_load(inputs[5].ptr<T>(static_cast<size_t>(d * B + b)) + hh)
                               : 0.0f;
                c[at] = has_c0 ? s_load(inputs[6].ptr<T>(static_cast<size_t>(d * B + b)) + hh)
                               : 0.0f;
            }
        }

        const bool fwd = (D > 1) ? (d == 0) : (cfg.direction != RnnDirection::Reverse);
        for (int64_t step = 0; step < Tn; ++step) {
            const int64_t t = fwd ? step : Tn - 1 - step;

            for (int64_t b = 0; b < B; ++b) {
                const bool active = (seq == nullptr) || (t < seq[b]);
                T* y_row = y.is_empty()
                               ? nullptr
                               : y.ptr<T>(static_cast<size_t>((t * D + d) * B + b));
                const float* hp = &h[static_cast<size_t>(b * H)];
                float* hnp = &hn[static_cast<size_t>(b * H)];
                const float* cp = &c[static_cast<size_t>(b * H)];
                float* cnp = &cn[static_cast<size_t>(b * H)];

                if (!active) {
                    for (int64_t hh = 0; hh < H; ++hh) {
                        hnp[hh] = hp[hh];
                        cnp[hh] = cp[hh];
                        if (y_row) { s_store(y_row + hh, 0.0f); }
                    }
                    continue;
                }

                const T* xr = x.ptr<T>(static_cast<size_t>(t * B + b));
                const T* bd = bias ? bias + d * bias_d : nullptr;

                // The linear part of one gate. `use_r` swaps the h_prev operand
                // for `aux` (GRU's r⊙h_prev) and `with_rb` adds the R half of
                // the bias.
                auto gate = [&](int64_t g, int64_t hh, bool with_rb,
                                bool use_r, const float* aux) {
                    const int64_t gi = g * H + hh;
                    float acc = 0.0f;
                    const T* wr = w.ptr<T>(static_cast<size_t>(d * G * H + gi));
                    for (int64_t i = 0; i < I; ++i) {
                        acc += s_load(xr + i) * s_load(wr + i);
                    }
                    const T* rr = r.ptr<T>(static_cast<size_t>(d * G * H + gi));
                    for (int64_t k = 0; k < H; ++k) {
                        acc += (use_r ? aux[k] : hp[k]) * s_load(rr + k);
                    }
                    if (bd) {
                        acc += s_load(bd + gi);
                        if (with_rb) { acc += s_load(bd + G * H + gi); }
                    }
                    return acc;
                };

                if (cfg.mode == RecurrentMode::Rnn) {
                    for (int64_t hh = 0; hh < H; ++hh) {
                        const float v = act1(clip1(gate(0, hh, true, false, nullptr), cfg.clip),
                                             cfg.act_g, cfg.alpha_g);
                        hnp[hh] = v;
                        if (y_row) { s_store(y_row + hh, v); }
                    }

                } else if (is_lstm) {
                    for (int64_t hh = 0; hh < H; ++hh) {
                        const float vi = act1(clip1(gate(0, hh, true, false, nullptr), cfg.clip),
                                              cfg.act_f, cfg.alpha_f);
                        const float vo = act1(clip1(gate(1, hh, true, false, nullptr), cfg.clip),
                                              cfg.act_f, cfg.alpha_f);
                        const float vf = cfg.input_forget
                            ? 1.0f - vi
                            : act1(clip1(gate(2, hh, true, false, nullptr), cfg.clip),
                                   cfg.act_f, cfg.alpha_f);
                        const float vc = act1(clip1(gate(3, hh, true, false, nullptr), cfg.clip),
                                              cfg.act_g, cfg.alpha_g);
                        const float cell = vf * cp[hh] + vi * vc;
                        cnp[hh] = cell;
                        const float v = vo * act1(cell, cfg.act_h, cfg.alpha_h);
                        hnp[hh] = v;
                        if (y_row) { s_store(y_row + hh, v); }
                    }

                } else {  // GRU
                    // z and r first: h~ reads r.
                    for (int64_t hh = 0; hh < H; ++hh) { pre[hh] = gate(0, hh, true, false, nullptr); }
                    for (int64_t hh = 0; hh < H; ++hh) {
                        zv[hh] = act1(clip1(pre[hh], cfg.clip), cfg.act_f, cfg.alpha_f);
                    }
                    for (int64_t hh = 0; hh < H; ++hh) { pre[hh] = gate(1, hh, true, false, nullptr); }
                    for (int64_t hh = 0; hh < H; ++hh) {
                        rv[hh] = act1(clip1(pre[hh], cfg.clip), cfg.act_f, cfg.alpha_f);
                        rh[hh] = rv[hh] * hp[hh];
                    }
                    for (int64_t hh = 0; hh < H; ++hh) {
                        float ph;
                        if (cfg.linear_before_reset) {
                            const int64_t gi = 2 * H + hh;
                            const T* wr = w.ptr<T>(static_cast<size_t>(d * G * H + gi));
                            float a = 0.0f;
                            for (int64_t i = 0; i < I; ++i) { a += s_load(xr + i) * s_load(wr + i); }
                            if (bd) { a += s_load(bd + gi); }
                            const T* rr = r.ptr<T>(static_cast<size_t>(d * G * H + gi));
                            float q = 0.0f;
                            for (int64_t k = 0; k < H; ++k) { q += hp[k] * s_load(rr + k); }
                            if (bd) { q += s_load(bd + G * H + gi); }
                            ph = a + rv[hh] * q;
                        } else {
                            ph = gate(2, hh, true, true, rh.data());
                        }
                        const float htilde = act1(clip1(ph, cfg.clip), cfg.act_g, cfg.alpha_g);
                        const float z = zv[hh];
                        const float v = (1.0f - z) * htilde + z * hp[hh];
                        hnp[hh] = v;
                        if (y_row) { s_store(y_row + hh, v); }
                    }
                }
            }

            h.swap(hn);
            c.swap(cn);
        }

        // ---- the direction's final states --------------------------------
        if (!yh.is_empty()) {
            for (int64_t b = 0; b < B; ++b) {
                const bool zero = (seq != nullptr) && (seq[b] == 0);
                for (int64_t hh = 0; hh < H; ++hh) {
                    const float v = zero ? 0.0f : h[static_cast<size_t>(b * H + hh)];
                    s_store(yh.ptr<T>(static_cast<size_t>(d * B + b)) + hh, v);
                    if (is_lstm && !yc.is_empty()) {
                        const float cv = zero ? 0.0f : c[static_cast<size_t>(b * H + hh)];
                        s_store(yc.ptr<T>(static_cast<size_t>(d * B + b)) + hh, cv);
                    }
                }
            }
        }
    }
}

void run(const RefCfg& cfg, std::span<TensorView> outputs,
         std::span<const TensorView> inputs)
{
    if (inputs[0].data_type() == DataType::f32) {
        recurrent_ref_impl<float>(cfg, outputs, inputs);
    } else {
        recurrent_ref_impl<half>(cfg, outputs, inputs);
    }
}

}  // namespace

void rnn_ref(const RnnAttributes& a, std::span<TensorView> outputs,
             std::span<const TensorView> inputs)
{
    RefCfg cfg;
    cfg.mode = RecurrentMode::Rnn;
    cfg.hidden = a.hidden_size > 0 ? a.hidden_size
                                   : inputs[2].shape(inputs[2].rank() - 1);
    cfg.directions = rnn_num_directions(a.direction);
    cfg.direction = a.direction;
    cfg.act_f = cfg.act_g = cfg.act_h = a.activation;
    cfg.alpha_f = cfg.alpha_g = cfg.alpha_h = a.activation_alpha;
    cfg.clip = a.clip;
    run(cfg, outputs, inputs);
}

void gru_ref(const GruAttributes& a, std::span<TensorView> outputs,
             std::span<const TensorView> inputs)
{
    RefCfg cfg;
    cfg.mode = RecurrentMode::Gru;
    cfg.hidden = a.hidden_size > 0 ? a.hidden_size
                                   : inputs[2].shape(inputs[2].rank() - 1);
    cfg.directions = rnn_num_directions(a.direction);
    cfg.direction = a.direction;
    cfg.act_f = a.activation_f;
    cfg.act_g = a.activation_g;
    cfg.act_h = a.activation_g;
    cfg.alpha_f = a.alpha_f;
    cfg.alpha_g = a.alpha_g;
    cfg.alpha_h = a.alpha_g;
    cfg.clip = a.clip;
    cfg.linear_before_reset = a.linear_before_reset;
    run(cfg, outputs, inputs);
}

void lstm_ref(const LstmAttributes& a, std::span<TensorView> outputs,
              std::span<const TensorView> inputs)
{
    RefCfg cfg;
    cfg.mode = RecurrentMode::Lstm;
    cfg.hidden = a.hidden_size > 0 ? a.hidden_size
                                   : inputs[2].shape(inputs[2].rank() - 1);
    cfg.directions = rnn_num_directions(a.direction);
    cfg.direction = a.direction;
    cfg.act_f = a.activation_f;
    cfg.act_g = a.activation_g;
    cfg.act_h = a.activation_h;
    cfg.alpha_f = a.alpha_f;
    cfg.alpha_g = a.alpha_g;
    cfg.alpha_h = a.alpha_h;
    cfg.clip = a.clip;
    cfg.input_forget = a.input_forget;
    run(cfg, outputs, inputs);
}

}  // namespace nnops::backend::cpu::reference
