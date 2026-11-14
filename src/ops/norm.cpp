/// @file norm.cpp
/// @brief Normalization operator dispatch (BatchNorm/LayerNorm/RMSNorm/GroupNorm/L2Norm).

#include "nnops/ops/norm.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(NormAttributes, norm_cpu)
NNOPS_DECLARE_CUDA(NormAttributes, norm_cuda)

NNOPS_OP_BOILERPLATE(Norm, NormAttributes,
        backend::cpu::norm_cpu, NNOPS_IF_CUDA(backend::cuda::norm_cuda), nullptr)

namespace {

/// Expected input count range for each norm type, as {min_inputs, max_inputs}.
/// A pre-add residual occupies one fixed trailing slot, so the count is exact
/// (LayerNorm keeps the bias slot even when it is empty). A gate is one more
/// trailing slot on RMSNorm, after the residual.
std::pair<int, int> norm_input_range(NormType type, bool add_residual, bool gated) {
    switch (type) {
    case NormType::BatchNorm: return {5, 5};
    case NormType::LayerNorm: return add_residual ? std::pair<int, int>{4, 4}
                                                  : std::pair<int, int>{2, 3};
    case NormType::RMSNorm:   return add_residual ? std::pair<int, int>{gated ? 4 : 3, gated ? 4 : 3}
                                                  : std::pair<int, int>{gated ? 3 : 2, gated ? 3 : 2};
    case NormType::GroupNorm: return {2, 3};
    case NormType::L2Norm:    return add_residual ? std::pair<int, int>{2, 2}
                                                  : std::pair<int, int>{1, 1};
    }
    return {0, 0};
}

}  // anonymous namespace

std::vector<TensorDesc> Norm::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    auto out = identity_output_shape(inputs);
    // Quantized X (s8/u8): the output storage type is chosen by the caller —
    // f32/f16 to read back dequantized values, s8/u8 to requantize.
    if (is_quantized_dtype(inputs[0].dtype)) {
        out[0].dtype = attrs_.output_dtype;
    }
    return out;
}

Status Norm::validate(std::span<const TensorDesc> inputs,
                      std::span<const TensorDesc> outputs) const
{
    const bool gated = attrs_.gate_activation != EpilogueActivateType::None;

    // The 1-centered weight is a scale transform: LayerNorm and RMSNorm read
    // scale through norm_apply_row, which applies it. BatchNorm folds ns/nb
    // itself, GroupNorm broadcasts one pair per group, and L2Norm has no
    // weight at all.
    if (attrs_.one_plus_weight && attrs_.type != NormType::LayerNorm &&
        attrs_.type != NormType::RMSNorm) {
        return Status::unsupported;
    }

    // RMSNormGated is the gated form's one model (GatedDeltaNet): the gate is a
    // per-element multiplier over the normalized row, which only the RMSNorm
    // row kernel has.
    if (gated && (attrs_.type != NormType::RMSNorm ||
                  !is_gate_activation(attrs_.gate_activation))) {
        return Status::unsupported;
    }

    const auto [min_n, max_n] =
        norm_input_range(attrs_.type, attrs_.add_residual, gated);
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(),
                                   static_cast<size_t>(min_n),
                                   static_cast<size_t>(max_n), 1));
    if (inputs.empty()) { return Status::invalid_argument; }
    const TensorDesc& x = inputs[0];
    const int64_t rank = x.rank;
    const int64_t axis = attrs_.axis < 0 ? attrs_.axis + rank : attrs_.axis;
    const bool tail = (axis == rank - 1);
    const auto same_as_x = [&](const TensorDesc& t) {
        if (t.rank != rank || t.numel() != x.numel()) { return false; }
        for (int64_t i = 0; i < rank; ++i) {
            if (t.dims[static_cast<size_t>(i)] != x.dims[static_cast<size_t>(i)]) {
                return false;
            }
        }
        return true;
    };

    // The gate is the trailing slot, after the optional residual. It is one
    // element per normalized element, so it needs the same contiguous-tail
    // shape the residual does.
    if (gated) {
        const TensorDesc& g = inputs.back();
        if (is_quantized_dtype(x.dtype) || g.dtype != x.dtype) {
            return Status::unsupported;
        }
        if (!tail) { return Status::unsupported; }
        if (!same_as_x(g)) { return Status::invalid_argument; }
    }

    if (!attrs_.add_residual) { return Status::ok; }

    // The fused `norm(x + r)` is the contiguous-tail case only, on a floating
    // X: the row kernels read both operands as one run.
    if (attrs_.type != NormType::LayerNorm &&
        attrs_.type != NormType::RMSNorm &&
        attrs_.type != NormType::L2Norm) {
        return Status::unsupported;
    }
    const TensorDesc& residual = inputs[inputs.size() - 1 - (gated ? 1 : 0)];
    if (!tail) { return Status::unsupported; }
    if (is_quantized_dtype(x.dtype) || residual.dtype != x.dtype) {
        return Status::unsupported;
    }
    if (!same_as_x(residual)) { return Status::invalid_argument; }
    return Status::ok;
}

Status Norm::compute(std::span<TensorView> outputs,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace)
{
    NNOPS_TRY(detail::check_views(backend_, outputs, inputs, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
