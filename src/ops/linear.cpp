/// @file linear.cpp
/// @brief Linear operator dispatch.

#include "nnops/ops/linear.hpp"
#include "nnops/ops/matmul.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

namespace backend::cpu {
    void linear_cpu(const LinearAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

namespace backend::cpu::reference {
    void linear_ref(const LinearAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

struct Linear::Impl {
    using KernelFn = void (*)(const LinearAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};
Linear::~Linear() = default;


namespace {
auto resolve_linear_kernel(Backend backend) -> Linear::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::linear_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return nullptr;
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return nullptr;
#endif
    default:
        return nullptr;
    }
}
}  // anonymous namespace

std::unique_ptr<Linear> Linear::create(const LinearAttributes& attrs,
                                        Backend backend)
{
    return std::unique_ptr<Linear>(new Linear(attrs, backend));
}

Linear::Linear(const LinearAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_linear_kernel(backend);
}

std::vector<TensorDesc> Linear::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    auto out = linear_output_shape(inputs);

    // The quantized path's output follows output_dtype: f32/f16 dequantizes,
    // s32/s8/u8 keeps the result integral; the fp path follows the input dtype
    // (set by the shape helper already).
    if (is_int8_matmul(inputs[0].dtype, inputs[1].dtype)) {
        const DataType odt = attrs_.output_dtype;
        const bool known = odt == DataType::f16 || odt == DataType::f32 ||
                           odt == DataType::s32 || odt == DataType::s8 ||
                           odt == DataType::u8;
        out[0].dtype = known ? odt : DataType::f32;
    }

    return out;
}

void Linear::compute(std::span<TensorView> outputs,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace)
{
    NNOPS_ASSERT(inputs.size() >= 2);
    NNOPS_ASSERT(inputs.size() <= 3);
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());
    NNOPS_ASSERT(!inputs[1].is_empty());
    NNOPS_ASSERT(is_layout_supported(inputs[0].layout(), LayoutSupport::PlanarOnly));

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
