/// @file cumsum.cpp
/// @brief CumSum operator dispatch.

#include "nnops/ops/cumsum.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(CumSumAttributes, cumsum_cpu)
NNOPS_DECLARE_CUDA(CumSumAttributes, cumsum_cuda)

NNOPS_OP_BOILERPLATE(CumSum, CumSumAttributes,
        backend::cpu::cumsum_cpu, NNOPS_IF_CUDA(backend::cuda::cumsum_cuda), nullptr)

std::vector<TensorDesc> CumSum::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return {identity_output_shape(inputs)};
}

void CumSum::compute(std::span<TensorView> outputs,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 1);
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());
    NNOPS_ASSERT(is_layout_supported(inputs[0].layout(), LayoutSupport::PlanarOnly));

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
