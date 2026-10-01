/// @file softmax.cpp
/// @brief Softmax operator dispatch.

#include "nnops/ops/softmax.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

namespace backend::cpu {
    void softmax_cpu(const SoftmaxAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);
}

namespace backend::cpu::reference {
    void softmax_ref(const SoftmaxAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void softmax_cuda(const SoftmaxAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace);
}
#endif

// ============================================================
// Impl
// ============================================================
struct Softmax::Impl {
    using KernelFn = void (*)(const SoftmaxAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};
Softmax::~Softmax() = default;


namespace {
auto resolve_softmax_kernel(Backend backend) -> Softmax::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::softmax_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::softmax_cuda;
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return nullptr;
#endif
    }
    return nullptr;
}
}  // anonymous namespace

std::unique_ptr<Softmax> Softmax::create(const SoftmaxAttributes& attrs,
                                          Backend backend)
{
    return std::unique_ptr<Softmax>(new Softmax(attrs, backend));
}

Softmax::Softmax(const SoftmaxAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_softmax_kernel(backend);
}

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

    // Quantized input (s8/u8): output must be float (f32 or f16), not int.
    if (is_quantized_dtype(inputs[0].data_type())) {
        NNOPS_ASSERT(output.data_type() == DataType::f32 ||
                     output.data_type() == DataType::f16);
    }

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
