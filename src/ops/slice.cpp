/// @file slice.cpp
/// @brief Slice operator dispatch.

#include "nnops/ops/slice.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(SliceAttributes, slice_cpu)

NNOPS_OP_BOILERPLATE(Slice, SliceAttributes,
        backend::cpu::slice_cpu, nullptr, nullptr)

std::vector<TensorDesc> Slice::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return slice_output_shape(attrs_.starts, attrs_.ends,
                              attrs_.axes, attrs_.steps, inputs);
}

Status Slice::validate(std::span<const TensorDesc> inputs,
                       std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1));

    const auto& in = inputs[0];
    const auto& out = outputs[0];

    NNOPS_TRY(detail::check_rank(in.rank, 1, TensorDesc::kMaxRank));
    if (!detail::dtype_in<DataType::f32, DataType::f16>(in.dtype) || out.dtype != in.dtype) {
        return Status::invalid_argument;
    }
    if (!is_layout_supported(in.layout, LayoutSupport::PlanarOnly) ||
        !is_layout_supported(out.layout, LayoutSupport::PlanarOnly)) {
        return Status::invalid_argument;
    }

    const size_t n_axes = attrs_.axes.size();
    if (attrs_.starts.size() != n_axes || attrs_.ends.size() != n_axes ||
        (!attrs_.steps.empty() && attrs_.steps.size() != n_axes)) {
        return Status::invalid_argument;
    }

    const auto expected = slice_output_shape(
        attrs_.starts, attrs_.ends, attrs_.axes, attrs_.steps, inputs);
    if (out.rank != expected[0].rank) { return Status::invalid_argument; }
    for (int64_t i = 0; i < out.rank; ++i) {
        if (out.dims[static_cast<size_t>(i)] !=
            expected[0].dims[static_cast<size_t>(i)]) {
            return Status::invalid_argument;
        }
    }
    return Status::ok;
}

Status Slice::compute(std::span<TensorView> outputs,
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
