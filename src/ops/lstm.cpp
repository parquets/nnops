/// @file lstm.cpp
/// @brief LSTM operator dispatch.
///
///   i_t, o_t, f_t = f(clip(·)),  c~_t = g(clip(·))   (ONNX gate order i, o, f, c)
///   C_t = f_t ⊙ C_{t-1} + i_t ⊙ c~_t
///   H_t = o_t ⊙ h(C_t)
///
/// The four gates of one hidden block are adjacent columns of the fused gate
/// tile, so a tile pass produces C and H together — the cell state never exists
/// as a separate tensor, and neither does the gate buffer.
///
/// Peepholes (input 7) are not implemented: a non-empty P is rejected rather
/// than silently ignored, since ignoring them would be a wrong answer.

#include "nnops/ops/lstm.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "recurrent_common.hpp"

#include "backend/cpu/rnn/recurrent.h"

namespace nnops {

struct Lstm::Impl {
    using KernelFn = detail::RecurrentKernelFn;
    KernelFn kernel_fn = nullptr;
};
Lstm::~Lstm() = default;

namespace {

auto resolve_lstm_kernel(Backend backend) -> Lstm::Impl::KernelFn {
    switch (backend) {
    case Backend::CPU: return backend::cpu::recurrent_cpu;
    default:           return nullptr;
    }
}

detail::RecurrentSpec lstm_spec(const LstmAttributes& a) {
    detail::RecurrentSpec s;
    s.mode = RecurrentMode::Lstm;
    s.hidden_size = a.hidden_size;
    s.directions = rnn_num_directions(a.direction);
    s.direction = a.direction;
    s.act_f = a.activation_f;
    s.act_g = a.activation_g;
    s.act_h = a.activation_h;
    s.alpha_f = a.alpha_f;
    s.alpha_g = a.alpha_g;
    s.alpha_h = a.alpha_h;
    s.beta_f = a.beta_f;
    s.beta_g = a.beta_g;
    s.beta_h = a.beta_h;
    s.clip = a.clip;
    s.input_forget = a.input_forget;
    return s;
}

}  // anonymous namespace

NNOPS_OP_CREATE(Lstm, LstmAttributes)

Lstm::Lstm(const LstmAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_lstm_kernel(backend);
}

std::vector<TensorDesc> Lstm::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    if (inputs.size() < 3) { return {}; }
    return recurrent_output_shape(attrs_.hidden_size,
                                  rnn_num_directions(attrs_.direction),
                                  /*with_cell=*/true, inputs);
}

Status Lstm::validate(std::span<const TensorDesc> inputs,
                      std::span<const TensorDesc> outputs) const
{
    if (attrs_.layout != 0) { return Status::unsupported; }
    return detail::recurrent_check(lstm_spec(attrs_), inputs, outputs,
                                   /*with_cell=*/true);
}

Status Lstm::compute(std::span<TensorView> outputs,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace)
{
    (void)workspace;
    NNOPS_TRY(validate_views(outputs, inputs));
    return detail::recurrent_run(lstm_spec(attrs_), impl_->kernel_fn, outputs,
                                 inputs, ctx);
}

void Lstm::prepackWeights(std::span<const TensorView> inputs,
                        std::span<TensorView> outputs,
                        const ComputeContext& ctx)
{
    detail::recurrent_prepack(backend_, lstm_spec(attrs_), inputs, outputs, ctx);
}

}  // namespace nnops
