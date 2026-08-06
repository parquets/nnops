/// @file embed.cpp
/// @brief Embed operator dispatch.

#include "nnops/ops/embed.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

// Forward declarations of backend kernel entry points
namespace backend::cpu::reference {
    void embed_ref(const EmbedAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

namespace backend::cpu {
    void embed_cpu(const EmbedAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void embed_cuda(const EmbedAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace);
}
#endif

// ============================================================
// Impl
// ============================================================
struct Embed::Impl {
    using KernelFn = void (*)(const EmbedAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_embed_kernel(Backend backend) -> Embed::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::embed_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::embed_cuda;
#endif
    }
    return nullptr;
}
}  // anonymous namespace

std::unique_ptr<Embed> Embed::create(const EmbedAttributes& attrs,
                                       Backend backend)
{
    return std::unique_ptr<Embed>(new Embed(attrs, backend));
}

Embed::Embed(const EmbedAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_embed_kernel(backend);
}

std::vector<TensorDesc> Embed::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return embed_output_shape(inputs);
}

void Embed::compute(std::span<TensorView> outputs,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 2);
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];

    const auto& weight  = inputs[0];
    const auto& indices = inputs[1];

    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!weight.is_empty());
    NNOPS_ASSERT(!indices.is_empty());

    // Validate indices dtype
    NNOPS_ASSERT(indices.data_type() == DataType::i64 ||
                 indices.data_type() == DataType::i32);

    // Validate weight rank >= 2
    NNOPS_ASSERT(weight.rank() >= 2);

    // Validate weight dtype is f32 or f16
    NNOPS_ASSERT(weight.data_type() == DataType::f32 ||
                 weight.data_type() == DataType::f16);

    // Validate planar layout
    NNOPS_ASSERT(is_layout_supported(weight.layout(), LayoutSupport::PlanarOnly));
    NNOPS_ASSERT(is_layout_supported(indices.layout(), LayoutSupport::PlanarOnly));

    // Validate output dtype matches weight
    NNOPS_ASSERT(output.data_type() == weight.data_type());

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

// Functional API
void embed(const TensorView& weight,
           const TensorView& indices,
           TensorView& output,
           const EmbedAttributes& attrs,
           const ComputeContext& ctx)
{
    auto op = Embed::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {weight, indices};
    op->compute(output, ins, ctx, nullptr);
}

}  // namespace nnops
