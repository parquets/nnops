/// @file resize.cpp
/// @brief Resize operator dispatch.

#include "nnops/ops/resize.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(ResizeAttributes, resize_cpu)

NNOPS_OP_BOILERPLATE(Resize, ResizeAttributes,
        backend::cpu::resize_cpu, nullptr, nullptr)

std::vector<TensorDesc> Resize::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    std::array<int64_t, 3> os = attrs_.output_size;
    return {resize_output_shape(
        std::span<const int64_t>(os.data(), os.size()), inputs)};
}

void Resize::compute(std::span<TensorView> outputs,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 1);
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());
    NNOPS_ASSERT(is_layout_supported(inputs[0].layout(), LayoutSupport::PackedOnly));

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
