/// @file norm.cpp
/// @brief Normalization operator dispatch (BatchNorm/LayerNorm/RMSNorm/GroupNorm/L2Norm).

#include "nnops/ops/norm.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

namespace backend::cpu::reference {
    void norm_ref(const NormAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx,
                  void* workspace);
}

namespace backend::cpu {
    void norm_cpu(const NormAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx,
                  void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void norm_cuda(const NormAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx,
                   void* workspace);
}
#endif

struct Norm::Impl {
    using KernelFn = void (*)(const NormAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};
Norm::~Norm() = default;


namespace {
auto resolve_norm_kernel(Backend backend) -> Norm::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::norm_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::norm_cuda;
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return nullptr;
#endif
    default:
        return nullptr;
    }
}

/// Expected input count range for each norm type, as {min_inputs, max_inputs}.
std::pair<int, int> norm_input_range(NormType type) {
    switch (type) {
    case NormType::BatchNorm: return {5, 5};
    case NormType::LayerNorm: return {2, 3};
    case NormType::RMSNorm:   return {2, 2};
    case NormType::GroupNorm: return {2, 3};
    case NormType::L2Norm:    return {1, 1};
    }
    return {0, 0};
}
}  // anonymous namespace

std::unique_ptr<Norm> Norm::create(const NormAttributes& attrs, Backend backend)
{
    return std::unique_ptr<Norm>(new Norm(attrs, backend));
}

Norm::Norm(const NormAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_norm_kernel(backend);
}

std::vector<TensorDesc> Norm::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    auto out = identity_output_shape(inputs);
    // Quantized X (s8/u8): the output storage type is chosen by the caller —
    // f32/f16 to read back dequantized values, s8/u8 to requantize.
    if (is_quantized_dtype(inputs[0].dtype)) {
        out[0].dtype = attrs_.output_dtype;
    }
    return out;
}

void Norm::compute(std::span<TensorView> outputs,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace)
{
    const auto [min_n, max_n] = norm_input_range(attrs_.type);
    NNOPS_ASSERT(inputs.size() >= static_cast<size_t>(min_n));
    NNOPS_ASSERT(inputs.size() <= static_cast<size_t>(max_n));
    NNOPS_ASSERT(outputs.size() == 1);

    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());  // X

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
