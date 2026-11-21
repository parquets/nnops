/// @file quant_linear.cpp
/// @brief QuantizeLinear / DequantizeLinear operator dispatch.

#include "nnops/ops/quant_linear.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

namespace backend::cpu {
Status quantize_linear_cpu(const QuantLinearAttributes& attrs, TensorView& output,
                           std::span<const TensorView> inputs, const ComputeContext& ctx,
                           void* workspace);
Status dequantize_linear_cpu(const QuantLinearAttributes& attrs, TensorView& output,
                             std::span<const TensorView> inputs, const ComputeContext& ctx,
                             void* workspace);
} // namespace backend::cpu

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
Status quantize_linear_cuda(const QuantLinearAttributes& attrs, TensorView& output,
                            std::span<const TensorView> inputs, const ComputeContext& ctx,
                            void* workspace);
Status dequantize_linear_cuda(const QuantLinearAttributes& attrs, TensorView& output,
                              std::span<const TensorView> inputs, const ComputeContext& ctx,
                              void* workspace);
} // namespace backend::cuda
#endif

struct QuantizeLinear::Impl {
    using KernelFn = Status (*)(const QuantLinearAttributes&, TensorView&,
                                std::span<const TensorView>, const ComputeContext&, void*);
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
} // namespace

NNOPS_OP_CREATE(QuantizeLinear, QuantLinearAttributes)

QuantizeLinear::QuantizeLinear(const QuantLinearAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_quantize_kernel(backend);
}

std::vector<TensorDesc>
QuantizeLinear::getOutputTensorDesc(std::span<const TensorDesc> inputs) const
{
    return quantize_linear_output_shape(inputs, attrs_.output_dtype);
}

namespace {

/// The two directions share a shape: a float side, an s8/u8 side, an f32 scale
/// and an s8/u8 zero point. The kernel reads the scale as f32 and the zero point
/// through its own storage tag, so a mismatch misreads the bytes.
Status check_quant_linear_dtypes(const Backend be, std::span<const TensorDesc> inputs,
                                 std::span<const TensorDesc> outputs, bool quantize)
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 3, 3, 1));
    const DataType flt = quantize ? inputs[0].dtype : outputs[0].dtype;
    const DataType qnt = quantize ? outputs[0].dtype : inputs[0].dtype;
    if (!detail::dtype_in<DataType::f32, DataType::f16>(be, flt)) {
        return Status::unsupported;
    }
    if (qnt != DataType::s8 && qnt != DataType::u8) {
        return Status::invalid_argument;
    }
    if (inputs[1].dtype != DataType::f32) {
        return Status::invalid_argument;
    }
    if (inputs[2].dtype != DataType::s8 && inputs[2].dtype != DataType::u8) {
        return Status::invalid_argument;
    }
    return Status::ok;
}

} // namespace

Status QuantizeLinear::validate(std::span<const TensorDesc> inputs,
                                std::span<const TensorDesc> outputs) const
{
    return check_quant_linear_dtypes(backend_, inputs, outputs, true);
}

Status QuantizeLinear::compute(std::span<TensorView> outputs, std::span<const TensorView> inputs,
                               const ComputeContext& ctx, void* workspace)
{
    NNOPS_TRY(detail::check_views(backend_, outputs, inputs, 3, 3, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) {
        return Status::unsupported;
    }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

struct DequantizeLinear::Impl {
    using KernelFn = Status (*)(const QuantLinearAttributes&, TensorView&,
                                std::span<const TensorView>, const ComputeContext&, void*);
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
} // namespace

NNOPS_OP_CREATE(DequantizeLinear, QuantLinearAttributes)

DequantizeLinear::DequantizeLinear(const QuantLinearAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_dequantize_kernel(backend);

    // The shared attribute default is s8 (QuantizeLinear's output); a
    // dequantize output is always float, so an integer request means f32.
    if (attrs_.output_dtype != DataType::f32 && attrs_.output_dtype != DataType::f16) {
        attrs_.output_dtype = DataType::f32;
    }
}

std::vector<TensorDesc>
DequantizeLinear::getOutputTensorDesc(std::span<const TensorDesc> inputs) const
{
    return dequantize_linear_output_shape(inputs, attrs_.output_dtype);
}

Status DequantizeLinear::validate(std::span<const TensorDesc> inputs,
                                  std::span<const TensorDesc> outputs) const
{
    return check_quant_linear_dtypes(backend_, inputs, outputs, false);
}

Status DequantizeLinear::compute(std::span<TensorView> outputs, std::span<const TensorView> inputs,
                                 const ComputeContext& ctx, void* workspace)
{
    NNOPS_TRY(detail::check_views(backend_, outputs, inputs, 3, 3, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) {
        return Status::unsupported;
    }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

} // namespace nnops
