/// @file rope.cpp
/// @brief RoPE operator dispatch.

#include "nnops/ops/rope.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

#include <algorithm>
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
    // [0] x; [1] optional RMSNorm scale, [2] its optional bias; [3]/[4] an
    // optional precomputed cos/sin table. An omitted slot is a rank-0 desc, so
    // the table pair does not force a dummy scale and bias.
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1, 5, 1));

    const auto& x = inputs[0];
    NNOPS_TRY(detail::check_rank(x.rank, 2));
    if (!detail::dtype_in<DataType::f32, DataType::f16>(backend_, x.dtype)) {
        return Status::unsupported;
    }
    const int64_t head_dim = x.dims[static_cast<size_t>(x.rank - 1)];
    if (head_dim <= 0 || head_dim % 2 != 0) { return Status::invalid_argument; }

    const int64_t rotary_dim =
        attrs_.rotary_dim > 0 ? attrs_.rotary_dim : head_dim;
    if (rotary_dim < 2 || rotary_dim % 2 != 0 || rotary_dim > head_dim) {
        return Status::invalid_argument;
    }

    // The fused pre-norm runs over the whole head, so its vectors are
    // head_dim-wide even when only a prefix rotates.
    for (size_t i = 1; i < std::min<size_t>(inputs.size(), 3); ++i) {
        if (inputs[i].rank == 0) { continue; }
        if (inputs[i].rank != 1 || inputs[i].dims[0] != head_dim ||
            inputs[i].dtype != x.dtype) {
            return Status::invalid_argument;
        }
    }

    // The precomputed tables come as a pair, [S, rotary_dim/2] over the same
    // row order the input uses.
    const bool has_cos = inputs.size() > 3 && inputs[3].rank != 0;
    const bool has_sin = inputs.size() > 4 && inputs[4].rank != 0;
    if (has_cos != has_sin) { return Status::invalid_argument; }
    if (has_cos) {
        const int64_t seq_len = x.dims[static_cast<size_t>(x.rank - 2)];
        for (size_t i = 3; i < 5; ++i) {
            const TensorDesc& t = inputs[i];
            if (t.rank != 2 || t.dims[0] != seq_len || t.dims[1] != rotary_dim / 2 ||
                (t.dtype != DataType::f32 && t.dtype != x.dtype)) {
                return Status::invalid_argument;
            }
        }
    }
    if (!(attrs_.norm_epsilon > 0.0f) || !std::isfinite(attrs_.norm_epsilon)) {
        return Status::invalid_argument;
    }

    if (!attrs_.mrope_section_dims.empty()) {
        int64_t total = 0;
        for (auto d : attrs_.mrope_section_dims) { total += d; }
        if (total != rotary_dim) { return Status::invalid_argument; }
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
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1, 5, 1));
    // Only the activation is mandatory; before the table pair existed the
    // optional slots could simply be dropped, and that has to keep working.
    if (inputs[0].is_empty() || outputs[0].is_empty()) {
        return Status::invalid_argument;
    }
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
