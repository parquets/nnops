#pragma once
/// @file recurrent_common.hpp
/// @brief The part RNN, GRU and LSTM share: ONNX's shape rules, the
/// attribute → RecurrentParams mapping, and the optional-input presence rules.
///
/// One kernel entry point serves all three (backend/cpu/rnn/recurrent.h)
/// because the gate interleave, the packed weights, the chunked input
/// projection and the per-step tile epilogue are the same code — what differs is
/// the arithmetic inside the epilogue, which is a template argument there.
///
/// Input order is ONNX's for all three; the optional ones are absent when their
/// view is empty, and a caller that stops early simply passes fewer:
///   0 X [T,B,I]  1 W [D,G·H,I]  2 R [D,G·H,H]  3 B [D,2·G·H]
///   4 sequence_lens [B] int32  5 initial_h [D,B,H]  6 initial_c [D,B,H] (LSTM)
///   7 P (LSTM peepholes — unsupported, so a non-empty view is rejected)
/// Outputs: 0 Y [T,D,B,H]  1 Y_h [D,B,H]  2 Y_c [D,B,H] (LSTM).

#include "backend/cpu/rnn/recurrent.h"
#include "nnops/detail/validate.hpp"
#include "nnops/ops/recurrent.hpp"

#include <algorithm>
#include <cstdint>
#include <span>

namespace nnops::detail {

/// A recurrent op's attributes, reduced to what the kernel and the checks read.
struct RecurrentSpec {
    RecurrentMode mode = RecurrentMode::Rnn;
    int64_t hidden_size = 0;  ///< 0 infers from R's last axis
    int64_t directions = 1;
    RnnDirection direction = RnnDirection::Forward;

    RnnActivation act_f = RnnActivation::Sigmoid;
    RnnActivation act_g = RnnActivation::Tanh;
    RnnActivation act_h = RnnActivation::Tanh;
    float alpha_f = 0.0f, beta_f = 0.0f;
    float alpha_g = 0.0f, beta_g = 0.0f;
    float alpha_h = 0.0f, beta_h = 0.0f;

    float clip = 0.0f;
    bool input_forget = false;
    bool linear_before_reset = false;
};

/// The kernel an op's Impl resolves for its backend. `nullptr` is "no kernel on
/// this backend", which compute() reports as Status::unsupported.
using RecurrentKernelFn = Status (*)(const backend::cpu::RecurrentParams&,
                                     std::span<TensorView>,
                                     std::span<const TensorView>,
                                     const ComputeContext&);

/// Hidden width: the attribute when set, else R's last axis.
inline int64_t recurrent_hidden(const RecurrentSpec& s,
                                std::span<const TensorDesc> inputs) noexcept
{
    if (s.hidden_size > 0) { return s.hidden_size; }
    if (inputs.size() < 3 || inputs[2].rank < 3) { return 0; }
    return inputs[2].dims[static_cast<size_t>(inputs[2].rank - 1)];
}

/// The recurrent family's W8A8 contract: an s8 weight makes the layer
/// quantized, and then the compute dtype — the gate math, the state and every
/// output — is the activation's floating type, f32 for an s8 activation.
/// `weight_dtype` is anything else for the plain floating route.
inline bool recurrent_is_quantized(DataType wdtype) noexcept {
    return wdtype == DataType::s8;
}

inline DataType recurrent_compute_dtype(DataType xdtype, DataType wdtype) noexcept {
    if (!recurrent_is_quantized(wdtype)) { return xdtype; }
    return (xdtype == DataType::f16) ? DataType::f16 : DataType::f32;
}

/// The quantized route's activation requirement. A floating activation is
/// quantized per row inside the kernel, so it needs nothing; an s8 one carries
/// its own per-row (or per-tensor) symmetric scale.
inline bool recurrent_activation_quant_ok(const TensorView& x) noexcept {
    if (x.data_type() != DataType::s8) { return true; }
    const QuantParams& q = x.quant_params();
    if (q.zero_point != 0 || q.zero_point_data != nullptr) { return false; }
    if (q.granularity == QuantGranularity::PerChannel) { return false; }
    const int64_t rows = x.shape(0) * x.shape(1);
    return q.num_scales == 0 || q.num_scales >= rows ||
           q.granularity == QuantGranularity::PerTensor;
}

/// A weight's per-row symmetric scales: per row of the whole [D, G·H, K]
/// tensor, or one table of G·H rows shared by every direction.
inline bool recurrent_weight_quant_ok(const TensorView& w, int64_t rows,
                                      int64_t per_direction) noexcept {
    const QuantParams& q = w.quant_params();
    if (q.zero_point != 0 || q.zero_point_data != nullptr) { return false; }
    if (q.granularity == QuantGranularity::PerChannel) { return false; }
    if (q.granularity == QuantGranularity::PerTensor) { return true; }
    return q.num_scales >= rows || q.num_scales >= per_direction;
}

/// The scale table a direction reads: the direction's own slice when the table
/// is per-row, otherwise the shared one. Null means "the per-tensor scalar".
inline const float* recurrent_scale_table(const QuantParams& q, int64_t d,
                                          int64_t per_direction) noexcept
{
    if (q.scale_data == nullptr) { return nullptr; }
    return (q.num_scales > per_direction) ? q.scale_data + d * per_direction
                                          : q.scale_data;
}

inline bool recurrent_activation_ok(RnnActivation a) noexcept {
    return rnn_activation_supported(a);
}

/// Descriptor-only twin of recurrent_run's checks: every rule the ONNX schema
/// states about the shapes, plus the ones this library adds (channel-first layout,
/// peepholes unsupported). An absent optional is a rank-0 descriptor.
inline Status recurrent_check(const RecurrentSpec& s,
                              std::span<const TensorDesc> inputs,
                              std::span<const TensorDesc> outputs,
                              bool with_cell)
{
    const size_t n_in_max = with_cell ? 8 : 6;
    NNOPS_TRY(check_counts(inputs.size(), outputs.size(), 3, n_in_max, 1));
    if (outputs.size() > (with_cell ? size_t{3} : size_t{2})) {
        return Status::invalid_argument;
    }

    const auto& x = inputs[0];
    const auto& w = inputs[1];
    const auto& r = inputs[2];
    // A prepacked W is opaque — its own header carries I, H, D and the gates,
    // and the kernel checks them — so only R is left to describe the geometry.
    const bool w_packed = is_packed_recurrent(w.layout);
    NNOPS_TRY(check_rank(x.rank, 3));
    NNOPS_TRY(check_rank(w.rank, w_packed ? 1 : 3));
    NNOPS_TRY(check_rank(r.rank, 3));

    if (!dtype_in<DataType::f32, DataType::f16, DataType::s8>(x.dtype) ||
        !dtype_in<DataType::f32, DataType::f16, DataType::s8>(w.dtype)) {
        return Status::unsupported;
    }
    const int64_t G = recurrent_gates(s.mode);
    const int64_t D = s.directions;
    const int64_t I = x.dims[2];
    const int64_t B = x.dims[1];
    const int64_t H = recurrent_hidden(s, inputs);
    /// The quantized route's floating half; the plain route's one dtype.
    const bool quantized = recurrent_is_quantized(w.dtype);
    const DataType cdtype = recurrent_compute_dtype(x.dtype, w.dtype);

    if (H <= 0 || D <= 0) { return Status::invalid_argument; }
    if (quantized ? (r.dtype != DataType::s8)
                  : (w.dtype != x.dtype || r.dtype != x.dtype)) {
        return Status::invalid_argument;
    }
    if (!is_layout_supported(x.layout, LayoutSupport::ChannelFirstOnly)) {
        return Status::invalid_argument;
    }
    if (!w_packed && (w.dims[0] != D || w.dims[1] != G * H || w.dims[2] != I)) {
        return Status::invalid_argument;
    }
    if (r.dims[0] != D || r.dims[1] != G * H || r.dims[2] != H) {
        return Status::invalid_argument;
    }

    // B: [D, 2·G·H], the W half then the R half.
    if (inputs.size() > 3 && inputs[3].rank > 0) {
        const auto& b = inputs[3];
        NNOPS_TRY(check_rank(b.rank, 2));
        if (b.dtype != cdtype || b.dims[0] != D || b.dims[1] != 2 * G * H) {
            return Status::invalid_argument;
        }
    }
    // sequence_lens: [B] int32.
    if (inputs.size() > 4 && inputs[4].rank > 0) {
        const auto& t = inputs[4];
        NNOPS_TRY(check_rank(t.rank, 1));
        if (t.dtype != DataType::s32 || t.dims[0] != B) {
            return Status::invalid_argument;
        }
    }
    // initial_h, initial_c: [D, B, H], in the compute dtype.
    const size_t n_state = std::min<size_t>(inputs.size(), with_cell ? 7 : 6);
    for (size_t i = 5; i < n_state; ++i) {
        const auto& t = inputs[i];
        if (t.rank == 0) { continue; }
        NNOPS_TRY(check_rank(t.rank, 3));
        if (t.dtype != cdtype || t.dims[0] != D || t.dims[1] != B || t.dims[2] != H) {
            return Status::invalid_argument;
        }
    }
    // P: LSTM peepholes, which this kernel does not implement.
    if (with_cell && inputs.size() > 7 && inputs[7].rank > 0) {
        return Status::unsupported;
    }

    if (!recurrent_activation_ok(s.act_f) || !recurrent_activation_ok(s.act_g) ||
        !recurrent_activation_ok(s.act_h)) {
        return Status::unsupported;
    }

    // The outputs, where the caller supplied them, must be the inferred shapes —
    // and, on the quantized route, the compute dtype, since W's s8 says nothing
    // about what the caller expects back.
    if (quantized) {
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (outputs[i].rank > 0 && outputs[i].dtype != cdtype) {
                return Status::invalid_argument;
            }
        }
    }
    if (!outputs.empty() && outputs[0].rank > 0) {
        const auto& y = outputs[0];
        NNOPS_TRY(check_rank(y.rank, 4));
        if (y.dims[0] != x.dims[0] || y.dims[1] != D || y.dims[2] != B ||
            y.dims[3] != H) {
            return Status::invalid_argument;
        }
    }
    for (size_t i = 1; i < outputs.size(); ++i) {
        const auto& t = outputs[i];
        if (t.rank == 0) { continue; }
        NNOPS_TRY(check_rank(t.rank, 3));
        if (t.dims[0] != D || t.dims[1] != B || t.dims[2] != H) {
            return Status::invalid_argument;
        }
    }
    return Status::ok;
}

/// The kernel's parameter block: the attributes plus the geometry, which the
/// caller reads off X (the sequence) or off W (the pack).
inline backend::cpu::RecurrentParams recurrent_params(const RecurrentSpec& s,
                                                      int64_t seq, int64_t batch,
                                                      int64_t input, int64_t hidden,
                                                      DataType dtype,
                                                      DataType weight_dtype)
{
    backend::cpu::RecurrentParams p;
    p.mode = s.mode;
    p.seq = seq;
    p.batch = batch;
    p.input = input;
    p.hidden = hidden;
    p.directions = s.directions;
    p.direction = s.direction;
    p.dtype = dtype;
    p.weight_dtype = weight_dtype;
    p.act_f = s.act_f;
    p.act_g = s.act_g;
    p.act_h = s.act_h;
    p.alpha_f = s.alpha_f;
    p.beta_f = s.beta_f;
    p.alpha_g = s.alpha_g;
    p.beta_g = s.beta_g;
    p.alpha_h = s.alpha_h;
    p.beta_h = s.beta_h;
    p.clip = s.clip;
    p.input_forget = s.input_forget;
    p.linear_before_reset = s.linear_before_reset;
    return p;
}

/// The recurrent family's prepacked weights, in the two-step shape Conv2D and
/// MatMul use: an empty outputs[0] queries the blob's descriptor, a populated
/// one receives the pack. The weights are the op's own inputs — W and R are
/// interleaved, packed and combined with the bias once, so compute() stops
/// rebuilding them per call. Only the CPU kernel packs; every other backend
/// leaves the query empty, which is the documented "no packed form" answer.
inline void recurrent_prepack(Backend backend, const RecurrentSpec& s,
                              std::span<const TensorView> inputs,
                              std::span<TensorView> outputs,
                              const ComputeContext& ctx)
{
    NNOPS_ASSERT(inputs.size() >= 3);
    NNOPS_ASSERT(!outputs.empty());
    if (backend != Backend::CPU) { return; }

    const auto& w = inputs[1];
    const auto& r = inputs[2];
    if (w.is_empty() || r.is_empty() || is_packed_recurrent(w.layout())) { return; }
    const DataType dt = w.data_type();
    if (!dtype_in<DataType::f32, DataType::f16, DataType::s8>(dt)) { return; }
    if (w.rank() != 3 || r.rank() != 3 ||
        r.data_type() != (recurrent_is_quantized(dt) ? DataType::s8 : dt)) {
        return;
    }

    const int64_t G = recurrent_gates(s.mode);
    const int64_t I = w.shape(2);
    const int64_t H = s.hidden_size > 0 ? s.hidden_size : r.shape(2);
    if (H <= 0 || w.shape(0) != s.directions || w.shape(1) != G * H ||
        r.shape(0) != s.directions || r.shape(1) != G * H || r.shape(2) != H) {
        return;
    }
    // A blob bakes the scales in and hides the weight from compute()'s checks,
    // so a quantisation the kernel would refuse must not pack either.
    if (recurrent_is_quantized(dt) &&
        (!recurrent_weight_quant_ok(w, s.directions * G * H, G * H) ||
         !recurrent_weight_quant_ok(r, s.directions * G * H, G * H))) {
        return;
    }

    // A quantized blob's own dtype is the weight's (s8); the bias it may carry
    // is stored as f32 whatever the compute dtype turns out to be, so the
    // compute dtype plays no part in the pack.
    auto p = recurrent_params(s, /*seq=*/0, /*batch=*/0, I, H,
                              recurrent_is_quantized(dt) ? DataType::f32 : dt, dt);
    const int64_t numel = backend::cpu::recurrent_packed_numel(p);
    const int64_t shape[1] = {numel};
    if (outputs[0].is_empty()) {
        outputs[0] = TensorView(std::span<const int64_t>(shape, 1), dt, nullptr,
                                TensorLayout::PackedRecurrent);
        return;
    }
    NNOPS_ASSERT(outputs[0].layout() == TensorLayout::PackedRecurrent &&
                 outputs[0].data_type() == dt && outputs[0].numel() >= numel &&
                 "recurrent prepackWeights: the buffer does not match the geometry");

    const TensorView* bias =
        (inputs.size() > 3 && !inputs[3].is_empty()) ? &inputs[3] : nullptr;
    // Called, not asserted-call: NNOPS_ASSERT is ((void)0) under NDEBUG, which
    // would make the whole pack a release-build no-op.
    const Status packed = backend::cpu::recurrent_pack_weights(p, w, r, bias,
                                                              outputs[0], ctx);
    NNOPS_ASSERT(is_ok(packed));
    (void)packed;
}

/// The quantized route's checks that need the views rather than the descriptors:
/// the scale tables and, above all, the zero points. Symmetric only — a
/// non-zero zero point would need the three-term correction the kernel does not
/// carry (per-row weight sums, per-step activation row sums, a column term), so
/// it is refused rather than silently dropped.
inline Status recurrent_q8_check(std::span<TensorView> outputs,
                                 std::span<const TensorView> inputs,
                                 int64_t directions)
{
    const auto& x = inputs[0];
    const auto& w = inputs[1];
    const auto& r = inputs[2];
    const bool w_packed = is_packed_recurrent(w.layout());
    const DataType cdtype = recurrent_compute_dtype(
        x.data_type(), w_packed ? DataType::s8 : w.data_type());

    if (!dtype_in<DataType::f32, DataType::f16, DataType::s8>(x.data_type())) {
        return Status::unsupported;
    }
    if (!w_packed) {
        const int64_t rows = w.shape(w.rank() - 2);
        if (!recurrent_weight_quant_ok(w, directions * rows, rows)) {
            return Status::unsupported;
        }
    }
    if (!recurrent_weight_quant_ok(r, directions * r.shape(r.rank() - 2),
                                   r.shape(r.rank() - 2))) {
        return Status::unsupported;
    }
    if (!recurrent_activation_quant_ok(x)) { return Status::unsupported; }

    const auto dtype_ok = [&](const TensorView& t) {
        return t.is_empty() || t.data_type() == cdtype;
    };
    if (inputs.size() > 3 && !dtype_ok(inputs[3])) { return Status::invalid_argument; }
    if (inputs.size() > 5 && !dtype_ok(inputs[5])) { return Status::invalid_argument; }
    if (inputs.size() > 6 && !dtype_ok(inputs[6])) { return Status::invalid_argument; }
    for (const auto& out : outputs) {
        if (!out.is_empty() && out.data_type() != cdtype) {
            return Status::invalid_argument;
        }
    }
    return Status::ok;
}

/// Compute-side entry: the optional views may be absent, so this checks the
/// required three by hand (check_views would reject an empty optional), then
/// hands the resolved numbers to the backend kernel.
inline Status recurrent_run(const RecurrentSpec& s, RecurrentKernelFn fn,
                            std::span<TensorView> outputs,
                            std::span<const TensorView> inputs,
                            const ComputeContext& ctx)
{
    if (!fn) { return Status::unsupported; }
    NNOPS_TRY(check_counts(inputs.size(), outputs.size(), 3, size_t(-1), 1));
    for (size_t i = 0; i < 3; ++i) {
        if (inputs[i].is_empty()) { return Status::invalid_argument; }
    }
    if (outputs.empty() || (outputs[0].is_empty() && outputs[1].is_empty())) {
        return Status::invalid_argument;
    }

    const auto& x = inputs[0];
    const auto& w = inputs[1];
    const auto& r = inputs[2];
    if (recurrent_is_quantized(w.data_type())) {
        NNOPS_TRY(recurrent_q8_check(outputs, inputs, s.directions));
    }

    auto p = recurrent_params(s, x.shape(0), x.shape(1), x.shape(2),
                              s.hidden_size > 0 ? s.hidden_size
                                                : r.shape(r.rank() - 1),
                              recurrent_compute_dtype(x.data_type(), w.data_type()),
                              w.data_type());
    return fn(p, outputs, inputs, ctx);
}

}  // namespace nnops::detail
