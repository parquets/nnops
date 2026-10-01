/// @file quant_linear.cpp
/// @brief QuantizeLinear / DequantizeLinear operator dispatch.

#include "nnops/ops/quant_linear.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

namespace backend::cpu {
    void quantize_linear_cpu(const QuantLinearAttributes& attrs,
                             TensorView& output,
                             std::span<const TensorView> inputs,
                             const ComputeContext& ctx,
                             void* workspace);
    void dequantize_linear_cpu(const QuantLinearAttributes& attrs,
                               TensorView& output,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* workspace);
}

namespace backend::cpu::reference {
    void quantize_linear_ref(const QuantLinearAttributes& attrs,
                             TensorView& output,
                             std::span<const TensorView> inputs,
                             const ComputeContext& ctx,
                             void* workspace);
    void dequantize_linear_ref(const QuantLinearAttributes& attrs,
                               TensorView& output,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void quantize_linear_cuda(const QuantLinearAttributes& attrs,
                              TensorView& output,
                              std::span<const TensorView> inputs,
                              const ComputeContext& ctx,
                              void* workspace);
    void dequantize_linear_cuda(const QuantLinearAttributes& attrs,
                                TensorView& output,
                                std::span<const TensorView> inputs,
                                const ComputeContext& ctx,
                                void* workspace);
}
#endif

// ============================================================
// QuantizeLinear
// ============================================================

struct QuantizeLinear::Impl {
    using KernelFn = void (*)(const QuantLinearAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};
QuantizeLinear::~QuantizeLinear() = default;


namespace {
auto resolve_quantize_kernel(Backend backend) -> QuantizeLinear::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::quantize_linear_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::quantize_linear_cuda;
#endif
    default:
        return nullptr;
    }
}
}

std::unique_ptr<QuantizeLinear> QuantizeLinear::create(
    const QuantLinearAttributes& attrs, Backend backend)
{
    return std::unique_ptr<QuantizeLinear>(new QuantizeLinear(attrs, backend));
}

QuantizeLinear::QuantizeLinear(const QuantLinearAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_quantize_kernel(backend);
}

std::vector<TensorDesc> QuantizeLinear::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return quantize_linear_output_shape(inputs, attrs_.output_dtype);
}

void QuantizeLinear::compute(std::span<TensorView> outputs,
                              std::span<const TensorView> inputs,
                              const ComputeContext& ctx,
                              void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 3);
    NNOPS_ASSERT(outputs.size() == 1);
    NNOPS_ASSERT(!outputs[0].is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());

    impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

// ============================================================
// DequantizeLinear
// ============================================================

struct DequantizeLinear::Impl {
    using KernelFn = void (*)(const QuantLinearAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};
DequantizeLinear::~DequantizeLinear() = default;


namespace {
auto resolve_dequantize_kernel(Backend backend) -> DequantizeLinear::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::dequantize_linear_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::dequantize_linear_cuda;
#endif
    default:
        return nullptr;
    }
}
}

std::unique_ptr<DequantizeLinear> DequantizeLinear::create(
    const QuantLinearAttributes& attrs, Backend backend)
{
    return std::unique_ptr<DequantizeLinear>(new DequantizeLinear(attrs, backend));
}

DequantizeLinear::DequantizeLinear(const QuantLinearAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_dequantize_kernel(backend);
}

std::vector<TensorDesc> DequantizeLinear::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return dequantize_linear_output_shape(inputs, attrs_.output_dtype);
}

void DequantizeLinear::compute(std::span<TensorView> outputs,
                                std::span<const TensorView> inputs,
                                const ComputeContext& ctx,
                                void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 3);
    NNOPS_ASSERT(outputs.size() == 1);
    NNOPS_ASSERT(!outputs[0].is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());

    impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
