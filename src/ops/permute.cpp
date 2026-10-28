/// @file permute.cpp
/// @brief Permute operator dispatch.

#include "nnops/ops/permute.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(PermuteAttributes, permute_cpu)

NNOPS_OP_BOILERPLATE(Permute, PermuteAttributes,
        backend::cpu::permute_cpu, nullptr, nullptr)

std::vector<TensorDesc> Permute::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return permute_output_shape(attrs_.perm, inputs);
}

Status Permute::validate(std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1));

    const auto& in = inputs[0];
    const auto& out = outputs[0];
    const int64_t rank = in.rank;

    if (static_cast<size_t>(rank) != attrs_.perm.size()) {
        return Status::invalid_argument;
    }
    NNOPS_TRY(detail::check_rank(rank, 1, TensorDesc::kMaxRank));
    if (!detail::dtype_in<DataType::f32, DataType::f16>(in.dtype) || out.dtype != in.dtype) {
        return Status::invalid_argument;
    }
    if (!is_layout_supported(in.layout, LayoutSupport::PlanarOnly) ||
        !is_layout_supported(out.layout, LayoutSupport::PlanarOnly)) {
        return Status::invalid_argument;
    }
    if (out.rank != rank) {
        return Status::invalid_argument;
    }
    for (int64_t i = 0; i < rank; ++i) {
        if (out.dims[static_cast<size_t>(i)] !=
            in.dims[static_cast<size_t>(attrs_.perm[static_cast<size_t>(i)])]) {
            return Status::invalid_argument;
        }
    }
    return Status::ok;
}

Status Permute::compute(std::span<TensorView> outputs,
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
