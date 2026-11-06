/// @file conv1d.cpp
/// @brief Conv1D on the 2D implicit-GEMM kernel — a rank-4 lift and nothing else.
///
/// With a 1-wide height the kernel's derived channel, batch and output strides
/// are the 1D ones, so the GEMM sees exactly the geometry a native 1D
/// implementation builds (channel-first M = OC, N = L, K = IC/G·K; channel-last
/// pixel-major). Scheduling is the 2D kernel's, not this file's: a one-row
/// output has no oh axis, so the 2D plan cuts the columns instead of the filter
/// (wide_split_chunks in conv2d_implicit_gemm.cpp), which is what makes a 1D
/// call faster than the same values lifted by hand.
///
/// Both layouts run on the rank-3 view: NCW is channel-first with an OIHW
/// filter, NWC channel-last with HWIO — the memory order only, the logical dims
/// stay [N, C, L].
///
/// The one thing that cannot be delegated: a depthwise geometry takes the 2D
/// op's routing to the depthwise kernels, and this file bypasses the op.

#include "nnops/ops/conv1d.hpp"
#include "nnops/ops/depthwise_conv.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "backend/cpu/conv/conv2d_implicit_gemm.h"
#include "conv1d_common.h"

#include <span>

namespace nnops::backend::cpu {

Status depthwise_conv_cpu(const DepthwiseConvAttributes&, TensorView&,
                          std::span<const TensorView>, const ComputeContext&,
                          void*);

namespace {

bool is_fp(DataType dtype) noexcept
{
    return dtype == DataType::f32 || dtype == DataType::f16;
}

/// The lifted filter is the 2D order, so this is Conv2D's predicate verbatim.
bool is_depthwise_geometry(const Conv2DAttributes& attrs, const TensorView& w)
{
    if (attrs.groups <= 1) { return false; }
    const bool hwio = is_channel_last_weight(w.layout());
    return w.shape(hwio ? 2 : 1) == 1 && w.shape(hwio ? 3 : 0) == attrs.groups;
}

}  // anonymous namespace

Status conv1d_cpu(const Conv1DAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx,
                  void* workspace)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];

    Conv2DAttributes a{};
    a.kernel_size   = {1, attrs.kernel_size};
    a.stride        = {1, attrs.stride};
    a.dilation      = {1, attrs.dilation};
    a.padding       = {0, attrs.padding};
    a.groups        = attrs.groups;
    a.epilogue      = attrs.epilogue;
    a.add_to        = attrs.add_to;
    a.output_dtype  = attrs.output_dtype;
    a.act_quant_dtype = attrs.act_quant_dtype;
    // No 2D Winograd: its square tile has no 1-wide analogue, and Auto would
    // otherwise resolve to it for a k3s1 filter.
    a.algorithm     = ConvAlgorithm::ImplicitGemm;

    if (attrs.auto_pad != Conv1DAttributes::AutoPad::NOTSET) {
        a.padding[1] = resolve_leading_padding(
            input.shape(2), attrs.kernel_size, attrs.stride, attrs.dilation,
            attrs.padding, static_cast<int>(attrs.auto_pad));
    }

    TensorView in4  = lift_wide(input);
    TensorView out4 = lift_wide(output);
    TensorView w4   = lift_filter_wide(weight);

    TensorView operands[3] = {in4, w4, TensorView{}};
    const size_t n_in = inputs.size();
    if (n_in > 2) { operands[2] = inputs[2]; }
    const std::span<const TensorView> ins(operands, n_in);

    if (is_fp(input.data_type()) && is_fp(weight.data_type()) &&
        is_depthwise_geometry(a, w4)) {
        DepthwiseConvAttributes dw;
        dw.kernel_size = {1, 1, attrs.kernel_size};
        dw.stride      = {1, 1, attrs.stride};
        dw.dilation    = {1, 1, attrs.dilation};
        dw.padding     = {0, 0, a.padding[1]};
        dw.epilogue    = attrs.epilogue;
        dw.add_to      = attrs.add_to;
        return depthwise_conv_cpu(dw, out4, ins, ctx, workspace);
    }

    return conv2d_implicit_gemm_kernel(a, out4, ins, ctx, workspace);
}

}  // namespace nnops::backend::cpu
