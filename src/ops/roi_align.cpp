/// @file roi_align.cpp
/// @brief RoiAlign operator dispatch.

#include "nnops/ops/roi_align.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(RoiAlignAttributes, roi_align_cpu)
NNOPS_DECLARE_CUDA(RoiAlignAttributes, roi_align_cuda)

NNOPS_OP_BOILERPLATE(RoiAlign, RoiAlignAttributes, backend::cpu::roi_align_cpu,
                     NNOPS_IF_CUDA(backend::cuda::roi_align_cuda), nullptr)

std::vector<TensorDesc> RoiAlign::getOutputTensorDesc(std::span<const TensorDesc> inputs) const
{
    TensorDesc out;
    if (inputs.empty()) {
        return {out};
    }
    const TensorDesc& x = inputs[0];
    out.dtype = x.dtype;
    out.layout = x.layout;
    out.rank = 4;
    out.dims.resize(4);
    out.dims[0] = (inputs.size() >= 2 && inputs[1].rank >= 1) ? inputs[1].dims[0] : 0;
    out.dims[1] = attrs_.output_height;
    out.dims[2] = attrs_.output_width;
    out.dims[3] = x.rank >= 4 ? x.dims[3] : 0;
    return {out};
}

Status RoiAlign::validate(std::span<const TensorDesc> inputs,
                          std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 2, 3, 1));

    const TensorDesc& x = inputs[0];
    NNOPS_TRY(detail::check_rank(x.rank, 4, 4));
    if (!is_channel_last(x.layout)) {
        return Status::unsupported;
    }
    if (!detail::dtype_in<DataType::f32, DataType::f16>(backend_, x.dtype)) {
        return Status::unsupported;
    }
    if (attrs_.output_height < 1 || attrs_.output_width < 1 || attrs_.sampling_ratio < 0 ||
        !(attrs_.spatial_scale > 0.0f)) {
        return Status::invalid_argument;
    }
    for (int64_t d = 0; d < 4; ++d) {
        if (x.dims[static_cast<size_t>(d)] < 1) {
            return Status::invalid_argument;
        }
    }

    const TensorDesc& rois = inputs[1];
    NNOPS_TRY(detail::check_rank(rois.rank, 2, 2));
    if (rois.dims[1] != 4) {
        return Status::invalid_argument;
    }
    const int64_t num_rois = rois.dims[0];
    if (rois.dtype != DataType::f32) {
        return Status::invalid_argument;
    }

    // The batch index list is optional, and a trailing slot when present.
    if (inputs.size() >= 3) {
        const TensorDesc& b = inputs[2];
        NNOPS_TRY(detail::check_rank(b.rank, 1, 1));
        if (b.dims[0] != num_rois) {
            return Status::invalid_argument;
        }
        if (!detail::dtype_in<DataType::s32, DataType::s64>(backend_, b.dtype)) {
            return Status::invalid_argument;
        }
    }

    const TensorDesc& y = outputs[0];
    if (y.rank != 4 || y.dims[0] != num_rois || y.dims[1] != attrs_.output_height ||
        y.dims[2] != attrs_.output_width || y.dims[3] != x.dims[3]) {
        return Status::invalid_argument;
    }
    if (y.dtype != x.dtype || y.layout != x.layout) {
        return Status::invalid_argument;
    }
    return Status::ok;
}

Status RoiAlign::compute(std::span<TensorView> outputs, std::span<const TensorView> inputs,
                         const ComputeContext& ctx, void* workspace)
{
    NNOPS_TRY(detail::check_views(backend_, outputs, inputs, 2, 3, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) {
        return Status::unsupported;
    }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

} // namespace nnops
