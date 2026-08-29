/// @file grid_sample.cpp
/// @brief GridSample operator dispatch.

#include "nnops/ops/grid_sample.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

namespace backend::cpu {
    void grid_sample_cpu(const GridSampleAttributes& attrs,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         void* workspace);
}

// ============================================================
// Impl
// ============================================================
struct GridSample::Impl {
    using KernelFn = void (*)(const GridSampleAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_grid_sample_kernel(Backend backend) -> GridSample::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::grid_sample_cpu;
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

std::unique_ptr<GridSample> GridSample::create(const GridSampleAttributes& attrs,
                                                 Backend backend)
{
    return std::unique_ptr<GridSample>(new GridSample(attrs, backend));
}

GridSample::GridSample(const GridSampleAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_grid_sample_kernel(backend);
}

std::vector<TensorDesc> GridSample::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return grid_sample_output_shape(inputs);
}

void GridSample::compute(std::span<TensorView> outputs,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace)
{
    NNOPS_ASSERT(inputs.size() == 2);
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());
    NNOPS_ASSERT(!inputs[1].is_empty());
    NNOPS_ASSERT(is_layout_supported(inputs[0].layout(), LayoutSupport::PackedOnly));

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
