/// @file matmul.cpp
/// @brief MatMul operator dispatch.

#include "nnops/ops/matmul.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

#include "backend/cpu/matmul.h"

namespace nnops {

namespace backend::cpu::reference {
    void matmul_ref(const MatMulAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

struct MatMul::Impl {
    using KernelFn = void (*)(const MatMulAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};
MatMul::~MatMul() = default;


namespace {
auto resolve_matmul_kernel(Backend backend) -> MatMul::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::matmul_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return nullptr;
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return nullptr;
#endif
    }
    return nullptr;
}
}  // anonymous namespace

std::unique_ptr<MatMul> MatMul::create(const MatMulAttributes& attrs,
                                        Backend backend)
{
    return std::unique_ptr<MatMul>(new MatMul(attrs, backend));
}

MatMul::MatMul(const MatMulAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_matmul_kernel(backend);
}

size_t MatMul::getWorkspaceSize(std::span<const TensorDesc> inputs,
                                std::span<const TensorDesc> outputs) const
{
    (void)inputs;
    (void)outputs;
    // Packed-B (and int8 accumulator) scratch is pooled internally by the CPU kernel.
    return 0;
}

std::vector<TensorDesc> MatMul::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    auto out = matmul_output_shape(attrs_.transpose_a, attrs_.transpose_b, inputs);

    // Integer (s8×s8) matmul: the output follows output_dtype (s32 accumulate or
    // requantized s8), not the input dtype.
    if (inputs[0].dtype == DataType::s8 && inputs[1].dtype == DataType::s8) {
        out[0].dtype = attrs_.output_dtype;
    }

    return out;
}

void MatMul::compute(std::span<TensorView> outputs,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace)
{
    NNOPS_ASSERT(inputs.size() >= 2 && inputs.size() <= 3);  // A, B, [bias]
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());
    NNOPS_ASSERT(!inputs[1].is_empty());
    NNOPS_ASSERT(is_layout_supported(inputs[0].layout(), LayoutSupport::PlanarOnly));

    const auto& a = inputs[0];
    const auto& b = inputs[1];
    const int64_t a_rank = a.rank();
    const int64_t b_rank = b.rank();
    NNOPS_ASSERT(a_rank >= 2);
    NNOPS_ASSERT(b_rank >= 2);

    const int64_t Ka = attrs_.transpose_a ? a.shape(a_rank - 2) : a.shape(a_rank - 1);
    const int64_t Kb = attrs_.transpose_b ? b.shape(b_rank - 1) : b.shape(b_rank - 2);
    NNOPS_ASSERT(Ka == Kb);

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
