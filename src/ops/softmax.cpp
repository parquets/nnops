/// @file softmax.cpp
/// @brief Softmax operator dispatch.

#include "nnops/ops/softmax.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(SoftmaxAttributes, softmax_cpu)
NNOPS_DECLARE_CUDA(SoftmaxAttributes, softmax_cuda)

NNOPS_OP_BOILERPLATE(Softmax, SoftmaxAttributes,
        backend::cpu::softmax_cpu, NNOPS_IF_CUDA(backend::cuda::softmax_cuda), nullptr)

std::vector<TensorDesc> Softmax::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    auto out = identity_output_shape(inputs);
    // Quantized input (s8/u8): softmax output is dequantized float (default f32),
    // not re-quantized int.
    if (is_quantized_dtype(inputs[0].dtype)) {
        out[0].dtype = DataType::f32;
    }
    return out;
}

size_t Softmax::getWorkspaceSize(std::span<const TensorDesc> inputs,
                                 std::span<const TensorDesc> outputs) const
{
    (void)inputs;
    (void)outputs;
    // Quantized dequantization scratch is pooled internally by the CPU kernel.
    return 0;
}

void Softmax::compute(std::span<TensorView> outputs,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 1);
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());

    if (is_quantized_dtype(inputs[0].data_type())) {
        NNOPS_ASSERT(output.data_type() == DataType::f32 ||
                     output.data_type() == DataType::f16);
    }

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
