/// @file topk.cpp
/// @brief TopK operator dispatch.

#include "nnops/ops/topk.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

namespace backend::cpu {
    Status topk_cpu(const TopKAttributes& attrs,
                  TensorView& values,
                  TensorView& indices,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx,
                  void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    Status topk_cuda(const TopKAttributes& attrs,
                   TensorView& values,
                   TensorView& indices,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx,
                   void* workspace);
}
#endif

struct TopK::Impl {
    using KernelFn = Status (*)(const TopKAttributes&,
                               TensorView&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};
TopK::~TopK() = default;


namespace {
auto resolve_topk_kernel(Backend backend) -> TopK::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::topk_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::topk_cuda;
#endif
    default:
        return nullptr;
    }
}
}  // anonymous namespace

NNOPS_OP_CREATE(TopK, TopKAttributes)

TopK::TopK(const TopKAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_topk_kernel(backend);
}

std::vector<TensorDesc> TopK::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return topk_output_shape(attrs_.axis, attrs_.k, inputs);
}

Status TopK::validate(std::span<const TensorDesc> inputs,
                      std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1, 1, 2));

    const auto& in = inputs[0];
    NNOPS_TRY(detail::check_rank(in.rank, 1, TensorDesc::kMaxRank));
    if (attrs_.k < 1) { return Status::invalid_argument; }

    // A negative axis wraps from the end, as everywhere else in the library.
    const int64_t ax = attrs_.axis < 0 ? attrs_.axis + in.rank : attrs_.axis;
    if (ax < 0 || ax >= in.rank) { return Status::invalid_argument; }
    // Selecting more elements than the axis holds would read past the row.
    if (attrs_.k > in.dims[static_cast<size_t>(ax)]) {
        return Status::invalid_argument;
    }

    return detail::require(
        is_layout_supported(in.layout, LayoutSupport::PlanarOnly));
}

Status TopK::compute(std::span<TensorView> outputs,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace)
{
    NNOPS_TRY(detail::check_views(outputs, inputs, 1, 1, 2));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], outputs[1], inputs, ctx, workspace);
}

}  // namespace nnops
