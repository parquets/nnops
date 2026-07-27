#pragma once
/// @file layout_dispatch.hpp
/// @brief LayoutDispatch — shared utility for routing operator compute()
///        calls with automatic pack/unpack around a single NCHW kernel.
///
/// Usage (in src/ops/<name>.cpp):
///
///   #include "backend/cpu/common/layout_dispatch.hpp"
///
///   void Operator::compute(span<TensorView> outputs,
///                          span<const TensorView> inputs,
///                          const ComputeContext& ctx, void* workspace) {
///       LayoutDispatch ld(inputs[0].layout(), outputs[0].layout());
///       dispatch_layout(ld, impl_->kernel, outputs[0], inputs, ctx, workspace);
///   }
///
///   size_t Operator::getWorkspaceSize(...) const override {
///       return layout_dispatch_workspace(inputs[0], outputs[0]);
///   }

#include "nnops/core/tensor_view.hpp"
#include "nnops/core/compute_context.hpp"
#include "backend/cpu/layout_convert.hpp"

#include <cstring>
#include <span>

namespace nnops::backend::cpu {

// ============================================================
// LayoutDispatch — encodes the 4-path dispatch decision
// ============================================================

struct LayoutDispatch {
    bool in_packed;
    bool out_packed;

    explicit LayoutDispatch(TensorLayout in, TensorLayout out)
        : in_packed(is_channel_packed(in))
        , out_packed(is_channel_packed(out)) {}

    /// Does the dispatch need to unpack the input before calling the kernel?
    bool needs_input_unpack() const { return in_packed; }

    /// Does the dispatch need to pack the output after calling the kernel?
    bool needs_output_pack() const { return out_packed; }

    /// Does this combination require workspace temp buffers?
    bool needs_workspace() const { return in_packed || out_packed; }
};

// ============================================================
// dispatch_layout — single-kernel dispatch with auto-conversion
// ============================================================

/// Route a compute call through layout conversion as needed.
///
/// `kernel` is the operator's compute function. It always receives NCHW
/// (unpacked) inputs and produces NCHW outputs. This function transparently
/// unpacks channel-packed inputs before calling the kernel and packs the
/// output after if the user requested a packed output layout.
///
/// Workspace layout (when needed):
///   [temp_input buffer (if in_packed)] [temp_output buffer (if out_packed)]
///
/// @tparam KernelFn  void(TensorView& output, span<const TensorView> inputs,
///                         const ComputeContext& ctx, void* workspace)
template <typename KernelFn>
void dispatch_layout(const LayoutDispatch& ld,
                     KernelFn kernel,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace)
{
    if (!ld.needs_input_unpack() && !ld.needs_output_pack()) {
        // NCHW → NCHW: direct call, zero overhead
        kernel(output, inputs, ctx, workspace);
        return;
    }

    auto* ws = static_cast<char*>(workspace);
    const int64_t elem_size = static_cast<int64_t>(
        data_type_size(inputs[0].data_type()));

    // --- Unpack input if needed ---
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

    // --- Allocate temp output if needed ---
    TensorView effective_output = output;
    if (ld.needs_output_pack()) {
        int64_t out_bytes = output.numel() * elem_size;
        effective_output = TensorView(output.shape_span(),
                                       output.data_type(),
                                       ws, TensorLayout::NCHW);
    }

    // --- Compute in NCHW space ---
    kernel(effective_output, effective_inputs, ctx,
           /* remaining workspace — only for kernel's own use */ nullptr);

    // --- Pack output if needed ---
    if (ld.needs_output_pack()) {
        pack_nchw_to_nchwc8(effective_output, output, ctx);
    }
}

// ============================================================
// Workspace sizing
// ============================================================

/// Compute workspace bytes needed for layout conversion temp buffers.
inline size_t layout_dispatch_workspace(const TensorDesc& input,
                                         const TensorDesc& output) {
    LayoutDispatch ld(input.layout, output.layout);
    if (!ld.needs_workspace()) return 0;

    const auto elem_size = data_type_size(input.dtype);
    size_t ws = 0;

    // Temp buffer for unpacking input (if input is packed)
    if (ld.needs_input_unpack()) {
        ws += static_cast<size_t>(input.numel()) * elem_size;
    }

    // Temp buffer for packing output (if output is packed)
    if (ld.needs_output_pack()) {
        ws += static_cast<size_t>(output.numel()) * elem_size;
    }

    return ws;
}

}  // namespace nnops::backend::cpu
