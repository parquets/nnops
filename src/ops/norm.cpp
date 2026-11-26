/// @file norm.cpp
/// @brief Normalization operator dispatch (BatchNorm/LayerNorm/RMSNorm/GroupNorm/L2Norm).

#include "nnops/ops/norm.hpp"
#include "nnops/detail/aliasing.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(NormAttributes, norm_cpu)
NNOPS_DECLARE_CUDA(NormAttributes, norm_cuda)

NNOPS_OP_BOILERPLATE(Norm, NormAttributes, backend::cpu::norm_cpu,
                     NNOPS_IF_CUDA(backend::cuda::norm_cuda), nullptr)

namespace {

/// Expected input count range for each norm type, as {min_inputs, max_inputs}.
/// A pre-add residual occupies one fixed trailing slot, so the count is exact
/// (LayerNorm keeps the bias slot even when it is empty). A gate is one more
/// trailing slot on RMSNorm, after the residual. Modulation appends its
/// (mod_scale, mod_shift) pair, and AdaLN-Zero's gate one more, after
/// everything else.
std::pair<int, int> norm_input_range(NormType type, bool add_residual, bool gated,
                                     bool modulate, bool modulation_gate)
{
    std::pair<int, int> range;
    switch (type) {
    case NormType::BatchNorm:
        range = {5, 5};
        break;
    case NormType::LayerNorm:
        range = add_residual ? std::pair<int, int>{4, 4} : std::pair<int, int>{2, 3};
        break;
    case NormType::RMSNorm:
        range = add_residual ? std::pair<int, int>{gated ? 4 : 3, gated ? 4 : 3} :
                               std::pair<int, int>{gated ? 3 : 2, gated ? 3 : 2};
        break;
    case NormType::GroupNorm:
        range = {2, 3};
        break;
    case NormType::L2Norm:
        range = add_residual ? std::pair<int, int>{2, 2} : std::pair<int, int>{1, 1};
        break;
    default:
        return {0, 0};
    }
    if (modulate) {
        range.first += 2;
        range.second += 2;
        if (modulation_gate) {
            range.first += 1;
            range.second += 1;
        }
    }
    return range;
}

/// NumPy broadcast of one modulation tensor against X over X's leading dims
/// (the normalized axis is the rank-1 tail, so the last dim is the shared
/// width). `last_dim` is the width the modulation's own last axis must match.
bool modulation_shape_ok(const TensorDesc& mod, const TensorDesc& x, int64_t last_dim)
{
    if (mod.rank <= 0 || mod.rank > x.rank) {
        return false;
    }
    if (mod.dims[static_cast<size_t>(mod.rank - 1)] != last_dim) {
        return false;
    }
    const int64_t offset = x.rank - mod.rank;
    const int64_t outer = x.rank > 0 ? x.rank - 1 : 0;
    for (int64_t d = 0; d < outer; ++d) {
        const int64_t md = d - offset;
        if (md < 0) {
            continue;
        }
        const int64_t m = mod.dims[static_cast<size_t>(md)];
        if (m != 1 && m != x.dims[static_cast<size_t>(d)]) {
            return false;
        }
    }
    return true;
}

} // anonymous namespace

std::vector<TensorDesc> Norm::getOutputTensorDesc(std::span<const TensorDesc> inputs) const
{
    auto out = identity_output_shape(inputs);
    // Quantized X (s8/u8): the output storage type is chosen by the caller —
    // f32/f16 to read back dequantized values, s8/u8 to requantize.
    if (is_quantized_dtype(inputs[0].dtype)) {
        out[0].dtype = attrs_.output_dtype;
    }
    return out;
}

Status Norm::validate(std::span<const TensorDesc> inputs, std::span<const TensorDesc> outputs) const
{
    const bool gated = attrs_.gate_activation != EpilogueActivateType::None;
    const bool modulate = attrs_.modulate;
    const bool mod_gate = modulate && attrs_.modulation_gate;

    // The 1-centered weight is a scale transform: LayerNorm and RMSNorm read
    // scale through norm_apply_row, which applies it. BatchNorm folds ns/nb
    // itself, GroupNorm broadcasts one pair per group, and L2Norm has no
    // weight at all.
    if (attrs_.one_plus_weight && attrs_.type != NormType::LayerNorm &&
        attrs_.type != NormType::RMSNorm) {
        return Status::unsupported;
    }

    // RMSNormGated is the gated form's one model (GatedDeltaNet): the gate is a
    // per-element multiplier over the normalized row (`act(g)`, then `add_to`),
    // which only the RMSNorm row kernel has -- the CUDA one included.
    if (gated &&
        (attrs_.type != NormType::RMSNorm || !is_gate_activation(attrs_.gate_activation))) {
        return Status::unsupported;
    }

    // Modulation is the adaptive-normalization form. It needs the row kernel's
    // contiguous tail and has no quantized route, its gate is AdaLN-Zero's
    // (a plain multiply, not an activation), and the two trailing-gate forms
    // would compete for the same slot.
    if (attrs_.modulation_gate && !modulate) {
        return Status::invalid_argument;
    }
    if (modulate &&
        (attrs_.type == NormType::BatchNorm || gated || backend_ != Backend::CPU)) {
        return Status::unsupported;
    }

    const auto [min_n, max_n] =
        norm_input_range(attrs_.type, attrs_.add_residual, gated, modulate, mod_gate);
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), static_cast<size_t>(min_n),
                                   static_cast<size_t>(max_n), 1));
    if (inputs.empty()) {
        return Status::invalid_argument;
    }
    const TensorDesc& x = inputs[0];
    // Spatial BatchNorm and GroupNorm walk the NHWC/NDHWC pixel order; a dense
    // activation has no route.
    if ((attrs_.type == NormType::BatchNorm && attrs_.spatial) ||
        attrs_.type == NormType::GroupNorm) {
        if (!is_channel_last(x.layout)) {
            return Status::unsupported;
        }
    }
    const int64_t rank = x.rank;
    const int64_t axis = attrs_.axis < 0 ? attrs_.axis + rank : attrs_.axis;
    const bool tail = (axis == rank - 1);
    const auto same_as_x = [&](const TensorDesc& t) {
        if (t.rank != rank || t.numel() != x.numel()) {
            return false;
        }
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
        if (!tail) {
            return Status::unsupported;
        }
        if (!same_as_x(g)) {
            return Status::invalid_argument;
        }
    }

    // The modulation pair (then AdaLN-Zero's gate) is the last group of slots,
    // so the type's own slots keep their indices.
    if (modulate) {
        if (is_quantized_dtype(x.dtype) || !tail ||
            (attrs_.type != NormType::LayerNorm && attrs_.type != NormType::RMSNorm &&
             attrs_.type != NormType::GroupNorm && attrs_.type != NormType::L2Norm)) {
            return Status::unsupported;
        }
        const size_t mod_first = inputs.size() - (mod_gate ? 3u : 2u);
        const TensorDesc& ms = inputs[mod_first];
        const TensorDesc& mt = inputs[mod_first + 1];
        if (ms.dtype != x.dtype || mt.dtype != x.dtype) {
            return Status::unsupported;
        }
        if (mod_gate && inputs.back().dtype != x.dtype) {
            return Status::unsupported;
        }

        if (attrs_.type == NormType::GroupNorm) {
            // FiLM: per channel, optionally per sample.
            const int64_t C = x.dims[static_cast<size_t>(rank - 1)];
            const auto group_mod_ok = [&](const TensorDesc& m) {
                if (m.rank < 1 || m.rank > 2) {
                    return false;
                }
                if (m.dims[static_cast<size_t>(m.rank - 1)] != C) {
                    return false;
                }
                return m.rank == 1 || m.dims[0] == 1 || m.dims[0] == x.dims[0];
            };
            if (!group_mod_ok(ms) || !group_mod_ok(mt) ||
                (mod_gate && !group_mod_ok(inputs.back()))) {
                return Status::invalid_argument;
            }
        } else {
            const int64_t width = x.dims[static_cast<size_t>(rank - 1)];
            if (!modulation_shape_ok(ms, x, width) || !modulation_shape_ok(mt, x, width)) {
                return Status::invalid_argument;
            }
            if (mod_gate && !modulation_shape_ok(inputs.back(), x, width)) {
                return Status::invalid_argument;
            }
        }
    }

    if (!attrs_.add_residual) {
        return Status::ok;
    }

    // The fused `norm(x + r)` is the contiguous-tail case only, on a floating
    // X: the row kernels read both operands as one run.
    if (attrs_.type != NormType::LayerNorm && attrs_.type != NormType::RMSNorm &&
        attrs_.type != NormType::L2Norm) {
        return Status::unsupported;
    }
    const size_t trailing = (gated ? 1u : 0u) + (modulate ? (mod_gate ? 3u : 2u) : 0u);
    const TensorDesc& residual = inputs[inputs.size() - 1 - trailing];
    if (!tail) {
        return Status::unsupported;
    }
    if (is_quantized_dtype(x.dtype) || residual.dtype != x.dtype) {
        return Status::unsupported;
    }
    if (!same_as_x(residual)) {
        return Status::invalid_argument;
    }
    return Status::ok;
}

Status Norm::compute(std::span<TensorView> outputs, std::span<const TensorView> inputs,
                     const ComputeContext& ctx, void* workspace)
{
    if (attrs_.modulate) {
        // AdaLN drops the learned affine: `scale`, `bias` and (without
        // add_residual) `residual` may arrive as empty views, so only the
        // slots that hold data are checked.
        if (outputs.empty() || outputs[0].is_empty() || inputs.empty() ||
            inputs[0].is_empty()) {
            return Status::invalid_argument;
        }
        const bool bf16_ok = detail::backend_has_bf16(backend_);
        if (!bf16_ok && outputs[0].data_type() == DataType::bf16) {
            return Status::unsupported;
        }
        for (const auto& v : inputs) {
            if (!v.is_empty() && !bf16_ok && v.data_type() == DataType::bf16) {
                return Status::unsupported;
            }
        }
    } else {
        NNOPS_TRY(detail::check_views(backend_, outputs, inputs, 1));
    }
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) {
        return Status::unsupported;
    }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

NNOPS_INPLACE_ELEMENTWISE(Norm)

} // namespace nnops
