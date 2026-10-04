/// @file moe.cpp
/// @brief MoE (Mixture-of-Experts) operator dispatch.

#include "nnops/ops/moe.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

namespace backend::cpu::reference {
    void moe_ref(const MoEAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx,
                 void* workspace);
}

namespace backend::cpu {
    void moe_cpu(const MoEAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx,
                 void* workspace);
}

struct MoE::Impl {
    using KernelFn = void (*)(const MoEAttributes&,
                              TensorView&,
                              std::span<const TensorView>,
                              const ComputeContext&,
                              void*);
    KernelFn kernel_fn = nullptr;
};
MoE::~MoE() = default;


namespace {
auto resolve_moe_kernel(Backend backend) -> MoE::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::moe_cpu;
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

std::unique_ptr<MoE> MoE::create(const MoEAttributes& attrs, Backend backend)
{
    return std::unique_ptr<MoE>(new MoE(attrs, backend));
}

MoE::MoE(const MoEAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_moe_kernel(backend);
}

std::vector<TensorDesc> MoE::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return moe_output_shape(inputs);
}

size_t MoE::getWorkspaceSize(std::span<const TensorDesc> /*inputs*/,
                             std::span<const TensorDesc> /*outputs*/) const
{
    // Staging (gathered top-K tokens, per-expert activations) comes from the
    // backend's internal pool, like MatMul/Attention/Conv2D.
    return 0;
}

void MoE::compute(std::span<TensorView> outputs,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx,
                  void* workspace)
{
    // At least 5 inputs (input, router_probs, fc1_w, fc1_b, fc2_w); fc1_b may
    // be empty. The fc2_b and SwiGLU fc3 pair / router_weights tail are also
    // optional and may be omitted entirely or passed as an empty TensorView.
    NNOPS_ASSERT(inputs.size() >= 5);
    NNOPS_ASSERT(inputs.size() <= 9);
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].is_empty()) { continue; }
        NNOPS_ASSERT(inputs[i].data_type() == inputs[0].data_type());
    }
    NNOPS_ASSERT(inputs[0].data_type() == DataType::f32 ||
                 inputs[0].data_type() == DataType::f16);
    NNOPS_ASSERT(!inputs[0].is_empty());
    NNOPS_ASSERT(!inputs[1].is_empty());
    NNOPS_ASSERT(!inputs[2].is_empty());
    NNOPS_ASSERT(!inputs[4].is_empty());
    NNOPS_ASSERT(is_layout_supported(inputs[0].layout(), LayoutSupport::PlanarOnly));

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
