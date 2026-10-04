/// @file slice.cpp
/// @brief Slice operator dispatch.

#include "nnops/ops/slice.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

namespace backend::cpu::reference {
    void slice_ref(const SliceAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

namespace backend::cpu {
    void slice_cpu(const SliceAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void slice_cuda(const SliceAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);
}
#endif

struct Slice::Impl {
    using KernelFn = void (*)(const SliceAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};
Slice::~Slice() = default;

namespace {
auto resolve_slice_kernel(Backend backend) -> Slice::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::slice_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::slice_cuda;
#endif
    default:
        return nullptr;
    }
}
}  // anonymous namespace

std::unique_ptr<Slice> Slice::create(const SliceAttributes& attrs,
                                       Backend backend)
{
    return std::unique_ptr<Slice>(new Slice(attrs, backend));
}

Slice::Slice(const SliceAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_slice_kernel(backend);
}

std::vector<TensorDesc> Slice::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return slice_output_shape(attrs_.starts, attrs_.ends,
                              attrs_.axes, attrs_.steps, inputs);
}

void Slice::compute(std::span<TensorView> outputs,
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

    NNOPS_ASSERT(rank >= 1 && rank <= TensorDesc::kMaxRank);

    NNOPS_ASSERT(input.data_type() == DataType::f32 ||
                 input.data_type() == DataType::f16);

    NNOPS_ASSERT(is_layout_supported(input.layout(), LayoutSupport::PlanarOnly));
    NNOPS_ASSERT(is_layout_supported(output.layout(), LayoutSupport::PlanarOnly));

    NNOPS_ASSERT(output.data_type() == input.data_type());

    const size_t n_axes = attrs_.axes.size();
    NNOPS_ASSERT(attrs_.starts.size() == n_axes);
    NNOPS_ASSERT(attrs_.ends.size() == n_axes);
    NNOPS_ASSERT(attrs_.steps.empty() || attrs_.steps.size() == n_axes);

    const TensorDesc in_desc_arr[] = {input.desc()};
    const auto expected = slice_output_shape(
        attrs_.starts, attrs_.ends, attrs_.axes, attrs_.steps, in_desc_arr);
    NNOPS_ASSERT(output.rank() == expected[0].rank);
    for (int64_t i = 0; i < output.rank(); ++i) {
        NNOPS_ASSERT(output.shape(i) == expected[0].dims[static_cast<size_t>(i)]);
    }

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
