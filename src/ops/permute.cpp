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

void Permute::compute(std::span<TensorView> outputs,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 1);
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());

    const auto& input = inputs[0];
    const int64_t rank = input.rank();
    NNOPS_ASSERT(static_cast<size_t>(rank) == attrs_.perm.size());

    NNOPS_ASSERT(rank >= 1 && rank <= TensorDesc::kMaxRank);

    NNOPS_ASSERT(input.data_type() == DataType::f32 ||
                 input.data_type() == DataType::f16);

    NNOPS_ASSERT(is_layout_supported(input.layout(), LayoutSupport::PlanarOnly));
    NNOPS_ASSERT(is_layout_supported(output.layout(), LayoutSupport::PlanarOnly));

    NNOPS_ASSERT(output.data_type() == input.data_type());

    NNOPS_ASSERT(output.rank() == rank);
    for (int64_t i = 0; i < rank; ++i) {
        NNOPS_ASSERT(output.shape(i) == input.shape(attrs_.perm[static_cast<size_t>(i)]));
    }

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
