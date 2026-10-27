/// @file rope.cpp
/// @brief RoPE operator dispatch.

#include "nnops/ops/rope.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(RoPEAttributes, rope_cpu)

NNOPS_OP_BOILERPLATE(RoPE, RoPEAttributes,
        backend::cpu::rope_cpu, nullptr, nullptr)

std::vector<TensorDesc> RoPE::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return identity_output_shape(inputs);
}

void RoPE::compute(std::span<TensorView> outputs,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 1);
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());

    const auto& X = inputs[0];
    const int64_t rank = X.rank();
    NNOPS_ASSERT(rank >= 2);
    NNOPS_ASSERT(X.shape(rank - 1) % 2 == 0);  // head_dim must be even

    if (!attrs_.mrope_section_dims.empty()) {
        int64_t total = 0;
        for (auto d : attrs_.mrope_section_dims) {
            total += d;
        }
        NNOPS_ASSERT(total == X.shape(rank - 1));
        if (!attrs_.mrope_section_bases.empty()) {
            NNOPS_ASSERT(attrs_.mrope_section_bases.size() == attrs_.mrope_section_dims.size());
        }
    }

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
