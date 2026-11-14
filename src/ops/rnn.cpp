/// @file rnn.cpp
/// @brief RNN operator dispatch.
///
///   H_t = g(clip(X_t W^T + H_{t-1} R^T + Wb + Rb))
///
/// One gate, so the fused epilogue is the shortest of the three: activate, store
/// into H, store into Y. The input projection and the tile geometry come from
/// recurrent_common.hpp; see include/nnops/ops/rnn.hpp.

#include "nnops/ops/rnn.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "recurrent_common.hpp"

#include "backend/cpu/rnn/recurrent.h"

namespace nnops {

struct Rnn::Impl {
    using KernelFn = detail::RecurrentKernelFn;
    KernelFn kernel_fn = nullptr;
};
Rnn::~Rnn() = default;

namespace {

auto resolve_rnn_kernel(Backend backend) -> Rnn::Impl::KernelFn {
    switch (backend) {
    case Backend::CPU: return backend::cpu::recurrent_cpu;
    default:           return nullptr;
    }
}

detail::RecurrentSpec rnn_spec(const RnnAttributes& a) {
    detail::RecurrentSpec s;
    s.mode = RecurrentMode::Rnn;
    s.hidden_size = a.hidden_size;
    s.directions = rnn_num_directions(a.direction);
    s.direction = a.direction;
    // RNN has one activation, and the kernel reads the gate one.
    s.act_f = s.act_g = s.act_h = a.activation;
    s.alpha_f = s.alpha_g = s.alpha_h = a.activation_alpha;
    s.beta_f = s.beta_g = s.beta_h = a.activation_beta;
    s.clip = a.clip;
    return s;
}

}  // anonymous namespace

NNOPS_OP_CREATE(Rnn, RnnAttributes)

Rnn::Rnn(const RnnAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_rnn_kernel(backend);
}

std::vector<TensorDesc> Rnn::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    if (inputs.size() < 3) { return {}; }
    return recurrent_output_shape(attrs_.hidden_size,
                                  rnn_num_directions(attrs_.direction),
                                  /*with_cell=*/false, inputs);
}

Status Rnn::validate(std::span<const TensorDesc> inputs,
                     std::span<const TensorDesc> outputs) const
{
    if (attrs_.layout != 0) { return Status::unsupported; }
    return detail::recurrent_check(backend_, rnn_spec(attrs_), inputs, outputs,
                                   /*with_cell=*/false);
}

Status Rnn::compute(std::span<TensorView> outputs,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace)
{
    (void)workspace;
    NNOPS_TRY(validate_views(outputs, inputs));
    return detail::recurrent_run(backend_, rnn_spec(attrs_), impl_->kernel_fn, outputs,
                                 inputs, ctx);
}

void Rnn::prepackWeights(std::span<const TensorView> inputs,
                        std::span<TensorView> outputs,
                        const ComputeContext& ctx)
{
    detail::recurrent_prepack(backend_, rnn_spec(attrs_), inputs, outputs, ctx);
}

}  // namespace nnops
