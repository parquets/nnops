/// @file pad.cpp
/// @brief Pad operator dispatch.

#include "nnops/ops/pad.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"

namespace nnops {

NNOPS_DECLARE_CPU(PadAttributes, pad_cpu)
NNOPS_DECLARE_CUDA(PadAttributes, pad_cuda)

NNOPS_OP_BOILERPLATE(Pad, PadAttributes, backend::cpu::pad_cpu,
        NNOPS_IF_CUDA(backend::cuda::pad_cuda), nullptr)

std::vector<TensorDesc> Pad::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    const auto& in = inputs[0];
    TensorDesc out = in;
    for (int64_t d = 0; d < in.rank; ++d) {
        const size_t i = static_cast<size_t>(d);
        if (attrs_.pads.size() < static_cast<size_t>(2 * in.rank)) { break; }
        out.dims[i] = in.dims[i] + attrs_.pads[i] +
                      attrs_.pads[static_cast<size_t>(in.rank) + i];
    }
    return {out};
}

Status Pad::validate(std::span<const TensorDesc> inputs,
                     std::span<const TensorDesc> outputs) const
{
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 1, 1, 1));
    const auto& in = inputs[0];
    if (!is_layout_supported(in.layout, LayoutSupport::ChannelFirstOnly)) {
        return Status::unsupported;
    }
    if (!detail::dtype_in<DataType::f32, DataType::f16, DataType::s8,
                          DataType::u8, DataType::s32, DataType::s64>(backend_, in.dtype)) {
        return Status::unsupported;
    }
    if (attrs_.pads.size() != static_cast<size_t>(2 * in.rank)) {
        return Status::invalid_argument;
    }

    for (int64_t d = 0; d < in.rank; ++d) {
        const int64_t b = attrs_.pads[static_cast<size_t>(d)];
        const int64_t e = attrs_.pads[static_cast<size_t>(in.rank + d)];
        const int64_t n = in.dims[static_cast<size_t>(d)];
        if (n + b + e < 0) { return Status::invalid_argument; }
        // A reflection must see the axis inside its own extent, so no window
        // wraps around twice.
        if (attrs_.mode == PadAttributes::Mode::Reflect &&
            (b >= n || e >= n)) {
            return Status::invalid_argument;
        }
    }
    if (outputs.size() == 1) {
        if (outputs[0].dtype != in.dtype) { return Status::invalid_argument; }
        for (int64_t d = 0; d < in.rank; ++d) {
            const int64_t want = in.dims[static_cast<size_t>(d)] +
                                 attrs_.pads[static_cast<size_t>(d)] +
                                 attrs_.pads[static_cast<size_t>(in.rank + d)];
            if (outputs[0].dims[static_cast<size_t>(d)] != want) {
                return Status::invalid_argument;
            }
        }
    }
    return Status::ok;
}

Status Pad::compute(std::span<TensorView> outputs,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace)
{
    NNOPS_TRY(detail::check_views(backend_, outputs, inputs, 1, 1, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    if (!impl_->kernel_fn) { return Status::unsupported; }
    return impl_->kernel_fn(attrs_, outputs[0], inputs, ctx, workspace);
}

}  // namespace nnops
