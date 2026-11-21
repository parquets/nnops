/// @file non_max_suppression.cpp
/// @brief NonMaxSuppression operator dispatch (two outputs, like TopK).

#include "nnops/ops/non_max_suppression.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"

#include <algorithm>

namespace nnops {

namespace backend::cpu {
Status nms_cpu(const NonMaxSuppressionAttributes& attrs, TensorView& indices,
               TensorView& num_selected, std::span<const TensorView> inputs,
               const ComputeContext& ctx, void* workspace);
}

struct NonMaxSuppression::Impl {
    using KernelFn = Status (*)(const NonMaxSuppressionAttributes&, TensorView&, TensorView&,
                                std::span<const TensorView>, const ComputeContext&, void*);
    KernelFn kernel_fn = nullptr;
};
NonMaxSuppression::~NonMaxSuppression() = default;

namespace {
auto resolve_nms_kernel(Backend backend) -> NonMaxSuppression::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::nms_cpu;
    default:
        return nullptr;
    }
}
} // anonymous namespace

NNOPS_OP_CREATE(NonMaxSuppression, NonMaxSuppressionAttributes)

NonMaxSuppression::NonMaxSuppression(const NonMaxSuppressionAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_nms_kernel(backend);
}

std::vector<TensorDesc> NonMaxSuppression::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    // Suggested capacity: every (batch, class) can select at most its candidate
    // count, and top_k caps that. The count output says what was used.
    int64_t cap = 0;
    if (inputs.size() >= 2 && inputs[0].rank == 3 && inputs[1].rank == 3) {
        const int64_t B = inputs[0].dims[0];
        const int64_t N = inputs[0].dims[1];
        const int64_t C = inputs[1].dims[1];
        const int64_t per_class = attrs_.top_k > 0 ? std::min(attrs_.top_k, N) : N;
        cap = B * C * per_class;
    }

    TensorDesc indices;
    indices.rank = 2;
    indices.dtype = DataType::s64;
    indices.layout = TensorLayout::Dense;
    indices.dims.resize(2);
    indices.dims[0] = cap;
    indices.dims[1] = 3;

    TensorDesc count;
    count.rank = 1;
    count.dtype = DataType::s64;
    count.layout = TensorLayout::Dense;
    count.dims.resize(1);
    count.dims[0] = 1;

    return {indices, count};
}

namespace {

/// A [1] scalar slot. Rank 0 is how an absent (empty) optional view reaches
/// validate(), so it is accepted and the kernel's is_empty() decides.
bool scalar_slot_ok(const TensorDesc& d, Backend be, bool integral)
{
    if (d.rank <= 0) {
        return true;  // absent
    }
    if (d.rank != 1 || d.dims[0] != 1) {
        return false;
    }
    return integral ? detail::dtype_in<DataType::s64, DataType::s32>(be, d.dtype)
                    : d.dtype == DataType::f32;
}

} // anonymous namespace

Status NonMaxSuppression::validate(std::span<const TensorDesc> inputs,
                                   std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 2, 5, 2));
    if (attrs_.center_point_box != 0 && attrs_.center_point_box != 1) {
        return Status::invalid_argument;
    }
    if (attrs_.top_k < 0) {
        return Status::invalid_argument;
    }

    const TensorDesc& boxes = inputs[0];
    const TensorDesc& scores = inputs[1];
    NNOPS_TRY(detail::check_rank(boxes.rank, 3, 3));
    NNOPS_TRY(detail::check_rank(scores.rank, 3, 3));
    if (boxes.dims[2] != 4) {
        return Status::invalid_argument;
    }
    if (boxes.dims[0] != scores.dims[0] || scores.dims[2] != boxes.dims[1]) {
        return Status::invalid_argument;
    }
    if (!detail::dtype_in<DataType::f32, DataType::f16>(backend_, boxes.dtype) ||
        scores.dtype != boxes.dtype) {
        return Status::unsupported;
    }

    // Optional scalars: max_output (integral), iou / score (float).
    if (inputs.size() >= 3 && !scalar_slot_ok(inputs[2], backend_, true)) {
        return Status::invalid_argument;
    }
    if (inputs.size() >= 4 && !scalar_slot_ok(inputs[3], backend_, false)) {
        return Status::invalid_argument;
    }
    if (inputs.size() >= 5 && !scalar_slot_ok(inputs[4], backend_, false)) {
        return Status::invalid_argument;
    }

    if (outputs[0].rank != 2 || outputs[0].dims[1] != 3 ||
        outputs[0].dtype != DataType::s64 || outputs[0].layout != TensorLayout::Dense) {
        return Status::invalid_argument;
    }
    if (outputs[1].rank != 1 || outputs[1].dims[0] != 1 ||
        outputs[1].dtype != DataType::s64 || outputs[1].layout != TensorLayout::Dense) {
        return Status::invalid_argument;
    }
    return Status::ok;
}

Status NonMaxSuppression::compute(std::span<TensorView> outputs,
                                  std::span<const TensorView> inputs, const ComputeContext& ctx,
                                  void* workspace)
{
    // The three optional scalar slots may arrive as empty views.
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 2, 5, 2));
    if (inputs[0].is_empty() || inputs[1].is_empty()) {
        return Status::invalid_argument;
    }
    if (outputs[0].is_empty() || outputs[1].is_empty()) {
        return Status::invalid_argument;
    }
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) {
        return Status::unsupported;
    }
    return impl_->kernel_fn(attrs_, outputs[0], outputs[1], inputs, ctx, workspace);
}

} // namespace nnops
