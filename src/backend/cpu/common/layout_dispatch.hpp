#pragma once
/// @file layout_dispatch.hpp
/// @brief LayoutDispatch — routes an operator's compute() through automatic
///        pack/unpack around a single NCHW kernel.
///
/// Use it in src/ops/<name>.cpp: build a LayoutDispatch from the input and
/// output layouts, call dispatch_layout(ld, impl_->kernel, outputs[0], inputs,
/// ctx, workspace), and size the workspace with
/// layout_dispatch_workspace(inputs[0].desc(), outputs[0].desc()).

#include "nnops/core/tensor_view.hpp"
#include "nnops/core/compute_context.hpp"
#include "backend/cpu/layout_convert.hpp"

#include <cstring>
#include <span>

namespace nnops::backend::cpu {

struct LayoutDispatch {
    bool in_packed;
    bool out_packed;

    explicit LayoutDispatch(TensorLayout in, TensorLayout out)
        : in_packed(is_channel_packed(in))
        , out_packed(is_channel_packed(out)) {}

    bool needs_input_unpack() const { return in_packed; }

    bool needs_output_pack() const { return out_packed; }

    bool needs_workspace() const { return in_packed || out_packed; }
};

/// Route a compute call through layout conversion as needed.
///
/// `kernel` always receives NCHW (unpacked) inputs and produces an NCHW output:
/// this unpacks the channel-packed inputs[0] first and packs the output after
/// if a packed output was requested. Workspace layout when conversion is needed:
///   [temp_input (if in_packed)] [temp_output (if out_packed)]
/// That workspace is consumed here, so the kernel is called with nullptr; with
/// no conversion it gets the caller's workspace untouched.
template <typename KernelFn>
void dispatch_layout(const LayoutDispatch& ld,
                     KernelFn kernel,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace)
{
    if (!ld.needs_input_unpack() && !ld.needs_output_pack()) {
        kernel(output, inputs, ctx, workspace);
        return;
    }

    auto* ws = static_cast<char*>(workspace);
    const int64_t elem_size = static_cast<int64_t>(
        data_type_size(inputs[0].data_type()));

    std::span<const TensorView> effective_inputs = inputs;
    if (ld.needs_input_unpack()) {
        const auto& input = inputs[0];
        int64_t in_bytes = input.numel() * elem_size;
        TensorView temp_in(input.shape_span(), input.data_type(),
                           ws, TensorLayout::NCHW);
        unpack_nchwc8_to_nchw(input, temp_in, ctx);
        effective_inputs = {&temp_in, 1};
        ws += in_bytes;
    }

    TensorView effective_output = output;
    if (ld.needs_output_pack()) {
        int64_t out_bytes = output.numel() * elem_size;
        effective_output = TensorView(output.shape_span(),
                                       output.data_type(),
                                       ws, TensorLayout::NCHW);
    }

    kernel(effective_output, effective_inputs, ctx, nullptr);

    if (ld.needs_output_pack()) {
        pack_nchw_to_nchwc8(effective_output, output, ctx);
    }
}

/// Workspace bytes for the layout-conversion temp buffers.
inline size_t layout_dispatch_workspace(const TensorDesc& input,
                                         const TensorDesc& output) {
    LayoutDispatch ld(input.layout, output.layout);
    if (!ld.needs_workspace()) {
        return 0;
    }

    const auto elem_size = data_type_size(input.dtype);
    size_t ws = 0;

    if (ld.needs_input_unpack()) {
        ws += static_cast<size_t>(input.numel()) * elem_size;
    }

    if (ld.needs_output_pack()) {
        ws += static_cast<size_t>(output.numel()) * elem_size;
    }

    return ws;
}

}  // namespace nnops::backend::cpu
