/// @file gru.cpp
/// @brief GRU operator dispatch.
///
///   z_t = f(clip(X_t Wz^T + H_{t-1} Rz^T + Wbz + Rbz))
///   r_t = f(clip(X_t Wr^T + H_{t-1} Rr^T + Wbr + Rbr))
///   h~_t = g(clip(X_t Wh^T + (r_t ⊙ H_{t-1}) Rh^T + Wbh + Rbh))     lbr = 0
///   h~_t = g(clip(X_t Wh^T + r_t ⊙ (H_{t-1} Rh^T + Rbh) + Wbh))     lbr = 1
///   H_t = (1 - z_t) ⊙ h~_t + z_t ⊙ H_{t-1}
///
/// Two phases per timestep, because h~ needs r. The first pass covers z and r and
/// leaves r⊙H_{t-1} (or r) behind for the second, whose epilogue finishes the
/// step; linear_before_reset moves Rbh across the reset gate and needs the
/// recurrent product for h unsummed, which is the only reason that variant costs
/// a second accumulator. Both cost the same flops.

#include "nnops/ops/gru.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "recurrent_common.hpp"

#include "backend/cpu/rnn/recurrent.h"

namespace nnops {

struct Gru::Impl {
    using KernelFn = detail::RecurrentKernelFn;
    KernelFn kernel_fn = nullptr;
};
Gru::~Gru() = default;

namespace {

auto resolve_gru_kernel(Backend backend) -> Gru::Impl::KernelFn {
    switch (backend) {
    case Backend::CPU: return backend::cpu::recurrent_cpu;
    default:           return nullptr;
    }
}

detail::RecurrentSpec gru_spec(const GruAttributes& a) {
    detail::RecurrentSpec s;
    s.mode = RecurrentMode::Gru;
    s.hidden_size = a.hidden_size;
    s.directions = rnn_num_directions(a.direction);
    s.direction = a.direction;
    s.act_f = a.activation_f;
    s.act_g = a.activation_g;
    s.act_h = a.activation_g;
    s.alpha_f = a.alpha_f;
    s.alpha_g = a.alpha_g;
    s.alpha_h = a.alpha_g;
    s.beta_f = a.beta_f;
    s.beta_g = a.beta_g;
    s.beta_h = a.beta_g;
    s.clip = a.clip;
    s.linear_before_reset = a.linear_before_reset;
    return s;
}

}  // anonymous namespace

NNOPS_OP_CREATE(Gru, GruAttributes)

Gru::Gru(const GruAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_gru_kernel(backend);
}

std::vector<TensorDesc> Gru::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    if (inputs.size() < 3) { return {}; }
    return recurrent_output_shape(attrs_.hidden_size,
                                  rnn_num_directions(attrs_.direction),
                                  /*with_cell=*/false, inputs);
}

Status Gru::validate(std::span<const TensorDesc> inputs,
                     std::span<const TensorDesc> outputs) const
{
    if (attrs_.layout != 0) { return Status::unsupported; }
    return detail::recurrent_check(gru_spec(attrs_), inputs, outputs,
                                   /*with_cell=*/false);
}

Status Gru::compute(std::span<TensorView> outputs,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace)
{
    (void)workspace;
    NNOPS_TRY(validate_views(outputs, inputs));
    return detail::recurrent_run(gru_spec(attrs_), impl_->kernel_fn, outputs,
                                 inputs, ctx);
}

void Gru::prepackWeights(std::span<const TensorView> inputs,
                        std::span<TensorView> outputs,
                        const ComputeContext& ctx)
{
    detail::recurrent_prepack(backend_, gru_spec(attrs_), inputs, outputs, ctx);
}

}  // namespace nnops
