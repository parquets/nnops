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

// ============================================================
// Impl
// ============================================================
struct Norm::Impl {
    using KernelFn = void (*)(const NormAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

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
    }
    return nullptr;
}

/// Expected input count range for each norm type.
/// Returns {min_inputs, max_inputs}. max_inputs == -1 means unbounded (but
/// practically bounded by the batch-norm 5-input signature).
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
    return {identity_output_shape(inputs)};
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

// ============================================================
// Functional API
// ============================================================

void norm(const TensorView& x,
          TensorView& output,
          const NormAttributes& attrs,
          const ComputeContext& ctx)
{
    auto op = Norm::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {x};
    op->compute(output, ins, ctx, nullptr);
}

void norm(const TensorView& x,
          const TensorView& scale,
          TensorView& output,
          const NormAttributes& attrs,
          const ComputeContext& ctx)
{
    auto op = Norm::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {x, scale};
    op->compute(output, ins, ctx, nullptr);
}

void norm(const TensorView& x,
          const TensorView& scale,
          const TensorView& bias,
          TensorView& output,
          const NormAttributes& attrs,
          const ComputeContext& ctx)
{
    auto op = Norm::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {x, scale, bias};
    op->compute(output, ins, ctx, nullptr);
}

void norm(const TensorView& x,
          const TensorView& scale,
          const TensorView& bias,
          const TensorView& mean,
          const TensorView& var,
          TensorView& output,
          const NormAttributes& attrs,
          const ComputeContext& ctx)
{
    auto op = Norm::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {x, scale, bias, mean, var};
    op->compute(output, ins, ctx, nullptr);
}

}  // namespace nnops
