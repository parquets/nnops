/// @file pooling.cpp
/// @brief Pooling operator dispatch — input/output always NCHWC8/NCDHWC8, direct pass-through.

#include "nnops/ops/pooling.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(PoolingAttributes, pooling_cpu)
NNOPS_DECLARE_CUDA(PoolingAttributes, pooling_cuda)

NNOPS_OP_BOILERPLATE(Pooling, PoolingAttributes,
        backend::cpu::pooling_cpu, NNOPS_IF_CUDA(backend::cuda::pooling_cuda), nullptr)

std::vector<TensorDesc> Pooling::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return {pooling_output_shape(
        attrs_.kernel_shape, attrs_.stride, attrs_.dilation, attrs_.padding,
        static_cast<int>(attrs_.auto_pad), inputs)};
}

void Pooling::compute(std::span<TensorView> outputs,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace)
{
    (void)workspace;
    NNOPS_ASSERT(inputs.size() == 1);
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    auto& input = inputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!input.is_empty());
    NNOPS_ASSERT(is_layout_supported(input.layout(), LayoutSupport::PackedOnly));

    // auto_pad lives in the descriptor; kernels only see explicit leading padding.
    PoolingAttributes attrs = attrs_;
    if (attrs.auto_pad != PoolingAttributes::AutoPad::NOTSET) {
        const int ap = static_cast<int>(attrs.auto_pad);
        const int64_t srank = PoolingAttributes::spatial_rank(input.rank());
        for (int i = 0; i < srank; ++i) {
            // 2D uses [1],[2]; 3D uses [0],[1],[2].
            const size_t ai = static_cast<size_t>(3 - srank + i);
            attrs.padding[ai] = resolve_leading_padding(
                input.shape(2 + i), attrs.kernel_shape[ai], attrs.stride[ai],
                attrs.dilation[ai], attrs.padding[ai], ap);
        }
        attrs.auto_pad = PoolingAttributes::AutoPad::NOTSET;
    }

    impl_->kernel_fn(attrs, output, inputs, ctx, nullptr);
}

}  // namespace nnops
