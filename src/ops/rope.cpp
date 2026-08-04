/// @file rope.cpp
/// @brief RoPE operator dispatch.

#include "nnops/ops/rope.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

namespace backend::cpu::reference {
    void rope_ref(const RoPEAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx,
                  void* workspace);
}

namespace backend::cpu {
    void rope_cpu(const RoPEAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx,
                  void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void rope_cuda(const RoPEAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx,
                   void* workspace);
}
#endif

// ============================================================
// Impl
// ============================================================
struct RoPE::Impl {
    using KernelFn = void (*)(const RoPEAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_rope_kernel(Backend backend) -> RoPE::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::rope_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::rope_cuda;
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return nullptr;
#endif
    }
    return nullptr;
}
}  // anonymous namespace

std::unique_ptr<RoPE> RoPE::create(const RoPEAttributes& attrs,
                                    Backend backend)
{
    return std::unique_ptr<RoPE>(new RoPE(attrs, backend));
}

RoPE::RoPE(const RoPEAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_rope_kernel(backend);
}

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

    // Validate mRoPE section dims if present
    if (!attrs_.mrope_section_dims.empty()) {
        int64_t total = 0;
        for (auto d : attrs_.mrope_section_dims) total += d;
        NNOPS_ASSERT(total == X.shape(rank - 1));
        if (!attrs_.mrope_section_bases.empty()) {
            NNOPS_ASSERT(attrs_.mrope_section_bases.size() == attrs_.mrope_section_dims.size());
        }
    }

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

// Functional API
void rope(const TensorView& x,
          TensorView& output,
          const RoPEAttributes& attrs,
          const ComputeContext& ctx)
{
    auto op = RoPE::create(attrs, ctx.expected_backend);
    op->compute(output, {&x, 1}, ctx, nullptr);
}

}  // namespace nnops
