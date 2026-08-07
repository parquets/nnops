/// @file group_norm.cpp
/// @brief GroupNorm operator dispatch.

#include "nnops/ops/group_norm.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

namespace backend::cpu::reference {
    void group_norm_ref(const GroupNormAttributes& attrs,
                        TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx,
                        void* workspace);
}

namespace backend::cpu {
    void group_norm_cpu(const GroupNormAttributes& attrs,
                        TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx,
                        void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void group_norm_cuda(const GroupNormAttributes& attrs,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         void* workspace);
}
#endif

// ============================================================
// Impl
// ============================================================
struct GroupNorm::Impl {
    using KernelFn = void (*)(const GroupNormAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_group_norm_kernel(Backend backend) -> GroupNorm::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::group_norm_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::group_norm_cuda;
#endif
    }
    return nullptr;
}
}  // anonymous namespace

std::unique_ptr<GroupNorm> GroupNorm::create(const GroupNormAttributes& attrs,
                                              Backend backend)
{
    return std::unique_ptr<GroupNorm>(new GroupNorm(attrs, backend));
}

GroupNorm::GroupNorm(const GroupNormAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_group_norm_kernel(backend);
}

std::vector<TensorDesc> GroupNorm::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return identity_output_shape(inputs);
}

void GroupNorm::compute(std::span<TensorView> outputs,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         void* workspace)
{
    NNOPS_ASSERT(inputs.size() >= 2);
    NNOPS_ASSERT(inputs.size() <= 3);  // x, scale [, bias]
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());  // X
    NNOPS_ASSERT(!inputs[1].is_empty());  // scale

    const auto& input = inputs[0];
    const int64_t rank = input.rank();
    NNOPS_ASSERT(rank >= 2);  // at least [N, C]

    // Validate planar layout
    NNOPS_ASSERT(is_layout_supported(input.layout(), LayoutSupport::PlanarOnly));
    NNOPS_ASSERT(is_layout_supported(output.layout(), LayoutSupport::PlanarOnly));

    // Validate dtype
    NNOPS_ASSERT(input.data_type() == DataType::f32 ||
                 input.data_type() == DataType::f16);

    // Validate groups
    const int64_t C = input.shape(1);
    int64_t G = attrs_.num_groups;
    if (G <= 0) G = 1;
    NNOPS_ASSERT(C % G == 0);  // C must be divisible by num_groups

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

// Functional API (without bias)
void group_norm(const TensorView& x,
                const TensorView& scale,
                TensorView& output,
                const GroupNormAttributes& attrs,
                const ComputeContext& ctx)
{
    auto op = GroupNorm::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {x, scale};
    op->compute(output, ins, ctx, nullptr);
}

// Functional API (with bias)
void group_norm(const TensorView& x,
                const TensorView& scale,
                const TensorView& bias,
                TensorView& output,
                const GroupNormAttributes& attrs,
                const ComputeContext& ctx)
{
    auto op = GroupNorm::create(attrs, ctx.expected_backend);
    const TensorView ins[] = {x, scale, bias};
    op->compute(output, ins, ctx, nullptr);
}

}  // namespace nnops
