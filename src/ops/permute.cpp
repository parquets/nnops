/// @file permute.cpp
/// @brief Permute operator dispatch.

#include "nnops/ops/permute.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

// Forward declarations of backend kernel entry points
namespace backend::cpu::reference {
    void permute_ref(const PermuteAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);
}

namespace backend::cpu {
    void permute_cpu(const PermuteAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void permute_cuda(const PermuteAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace);
}
#endif

// ============================================================
// Impl
// ============================================================
struct Permute::Impl {
    using KernelFn = void (*)(const PermuteAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};
Permute::~Permute() = default;

namespace {
auto resolve_permute_kernel(Backend backend) -> Permute::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::permute_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::permute_cuda;
#endif
    }
    return nullptr;
}
}  // anonymous namespace

std::unique_ptr<Permute> Permute::create(const PermuteAttributes& attrs,
                                           Backend backend)
{
    return std::unique_ptr<Permute>(new Permute(attrs, backend));
}

Permute::Permute(const PermuteAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_permute_kernel(backend);
}

std::vector<TensorDesc> Permute::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return permute_output_shape(attrs_.perm, inputs);
}

void Permute::compute(std::span<TensorView> outputs,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 1);
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());

    const auto& input = inputs[0];
    const int64_t rank = input.rank();
    NNOPS_ASSERT(static_cast<size_t>(rank) == attrs_.perm.size());

    // Validate rank
    NNOPS_ASSERT(rank >= 1 && rank <= TensorDesc::kMaxRank);

    // Validate input dtype
    NNOPS_ASSERT(input.data_type() == DataType::f32 ||
                 input.data_type() == DataType::f16);

    // Validate planar layout
    NNOPS_ASSERT(is_layout_supported(input.layout(), LayoutSupport::PlanarOnly));
    NNOPS_ASSERT(is_layout_supported(output.layout(), LayoutSupport::PlanarOnly));

    // Validate output dtype matches input
    NNOPS_ASSERT(output.data_type() == input.data_type());

    // Validate output shape matches permuted input shape
    NNOPS_ASSERT(output.rank() == rank);
    for (int64_t i = 0; i < rank; ++i) {
        NNOPS_ASSERT(output.shape(i) == input.shape(attrs_.perm[static_cast<size_t>(i)]));
    }

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
