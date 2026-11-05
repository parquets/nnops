/// @file rope.cpp
/// @brief RoPE operator dispatch.

#include "nnops/ops/rope.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

#include <cmath>

namespace nnops {

NNOPS_DECLARE_CPU(RoPEAttributes, rope_cpu)

NNOPS_OP_BOILERPLATE(RoPE, RoPEAttributes,
        backend::cpu::rope_cpu, nullptr, nullptr)

std::vector<TensorDesc> RoPE::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return identity_output_shape(inputs);
}

Status RoPE::validate(std::span<const TensorDesc> inputs,
                      std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1));

    const auto& x = inputs[0];
    NNOPS_TRY(detail::check_rank(x.rank, 2));
    const int64_t head_dim = x.dims[static_cast<size_t>(x.rank - 1)];
    if (head_dim % 2 != 0) { return Status::invalid_argument; }  // must be even

    if (!attrs_.mrope_section_dims.empty()) {
        int64_t total = 0;
        for (auto d : attrs_.mrope_section_dims) { total += d; }
        if (total != head_dim) { return Status::invalid_argument; }
        if (!attrs_.mrope_section_bases.empty() &&
            attrs_.mrope_section_bases.size() != attrs_.mrope_section_dims.size()) {
            return Status::invalid_argument;
        }
    }

    if (attrs_.position_offset < 0) { return Status::invalid_argument; }
    if (attrs_.attention_factor < 0.0f) { return Status::invalid_argument; }
    if (attrs_.scaling_type != RoPEScalingType::None) {
        if (!std::isfinite(attrs_.scaling_factor) || attrs_.scaling_factor <= 0.0f) {
            return Status::invalid_argument;
        }
        // Every type but Linear pivots on the pretraining context length.
        if (attrs_.scaling_type != RoPEScalingType::Linear &&
            attrs_.original_max_position <= 0) {
            return Status::invalid_argument;
        }
        if (attrs_.scaling_type == RoPEScalingType::Llama3 &&
            (attrs_.llama3_low_freq_factor <= 0.0f ||
             attrs_.llama3_high_freq_factor <= attrs_.llama3_low_freq_factor)) {
            return Status::invalid_argument;
        }
    }
    return Status::ok;
}

Status RoPE::compute(std::span<TensorView> outputs,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace)
{
    NNOPS_TRY(detail::check_views(outputs, inputs, 1, 1, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
