/// @file multi_head_attention.cpp
/// @brief MultiHeadAttention operator dispatch.
///
/// Q, K, V [, mask] in f32 or f16, planar, merged [B, S, H*D] or explicit
/// [B, H, S, D]. The optional mask is positional and may still arrive as an
/// empty view (absent, not malformed). Everything below the attribute copy is
/// the shared CPU attention core.

#include "nnops/ops/multi_head_attention.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "backend/cpu/attention/attention.h"

namespace nnops {

NNOPS_DECLARE_CPU(MultiHeadAttentionAttributes, multi_head_attention_cpu)

NNOPS_OP_BOILERPLATE(MultiHeadAttention, MultiHeadAttentionAttributes,
        backend::cpu::multi_head_attention_cpu, nullptr, nullptr)

std::vector<TensorDesc> MultiHeadAttention::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return {attention_output_shape(inputs)};
}

size_t MultiHeadAttention::getWorkspaceSize(std::span<const TensorDesc> inputs,
                                            std::span<const TensorDesc> outputs) const
{
    (void)inputs;
    (void)outputs;
    // Scratch (scores + packed K^T) is pooled internally by the CPU kernel.
    return 0;
}

Status MultiHeadAttention::validate(std::span<const TensorDesc> inputs,
                                    std::span<const TensorDesc> outputs) const
{
    // Q, K, V [, mask]
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 3, 4, 1));
    if (!is_layout_supported(inputs[0].layout, LayoutSupport::PlanarOnly)) {
        return Status::unsupported;
    }

    const DataType dt = inputs[0].dtype;
    if (!detail::dtype_in<DataType::f32, DataType::f16>(dt)) {
        return Status::unsupported;
    }
    for (size_t i = 1; i <= 2; ++i) {
        if (inputs[i].dtype != dt) { return Status::invalid_argument; }
    }
    if (outputs[0].dtype != dt) { return Status::invalid_argument; }
    if (inputs.size() > 3 && inputs[3].rank != 0 && inputs[3].dtype != dt) {
        return Status::invalid_argument;
    }

    const auto& q = inputs[0];
    const auto& k = inputs[1];
    const auto& v = inputs[2];
    if (q.rank != 3 && q.rank != 4) { return Status::invalid_argument; }
    if (k.rank != q.rank || v.rank != q.rank) { return Status::invalid_argument; }
    if (attrs_.num_heads <= 0) { return Status::invalid_argument; }

    if (q.rank == 3) {
        // [B, S, H*D]: every head shares the row, so K and V must be as wide.
        const int64_t hd = q.dims[2];
        if (hd <= 0 || hd % attrs_.num_heads != 0) { return Status::invalid_argument; }
        if (k.dims[0] != q.dims[0] || k.dims[2] != hd) { return Status::invalid_argument; }
        if (v.dims[0] != q.dims[0] || v.dims[2] != hd) { return Status::invalid_argument; }
        if (v.dims[1] != k.dims[1]) { return Status::invalid_argument; }
    } else {
        const int64_t d = q.dims[3];
        if (d <= 0) { return Status::invalid_argument; }
        if (k.dims[0] != q.dims[0] || k.dims[1] != attrs_.num_heads ||
            k.dims[3] != d) {
            return Status::invalid_argument;
        }
        if (v.dims[0] != q.dims[0] || v.dims[1] != attrs_.num_heads ||
            v.dims[2] != k.dims[2] || v.dims[3] != d) {
            return Status::invalid_argument;
        }
    }

    const int64_t sq = (q.rank == 3) ? q.dims[1] : q.dims[2];
    const int64_t sk = (q.rank == 3) ? k.dims[1] : k.dims[2];
    if (inputs.size() > 3 && inputs[3].rank != 0) {
        const auto& m = inputs[3];
        if (m.rank != 2 || m.dims[0] != sq || m.dims[1] != sk) {
            return Status::invalid_argument;
        }
    }
    return Status::ok;
}

Status MultiHeadAttention::compute(std::span<TensorView> outputs,
                                   std::span<const TensorView> inputs,
                                   const ComputeContext& ctx,
                                   void* workspace)
{
    // The optional mask is allowed to arrive as an empty view (meaning absent),
    // so only the three required inputs are checked for data.
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 3, 4, 1));
    for (size_t i = 0; i < 3; ++i) {
        if (inputs[i].is_empty()) { return Status::invalid_argument; }
    }
    if (outputs[0].is_empty()) { return Status::invalid_argument; }
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
