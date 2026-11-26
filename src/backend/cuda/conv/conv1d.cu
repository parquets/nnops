/// @file conv1d.cu
/// @brief CUDA Conv1D on the 2D kernel — a rank-4 lift and nothing else.
///
/// With a 1-wide height the 2D kernel's derived channel, batch and output
/// strides are the 1D ones, so the implicit GEMM sees exactly the geometry a
/// native 1D implementation builds (pixel-major). Scheduling is the 2D
/// kernel's, not this file's.
///
/// The rank-3 view is NWC with an HWIO filter — the memory order only, the
/// logical dims stay [N, C, L].
///
/// The one thing that cannot be delegated: a depthwise geometry takes the 2D
/// op's routing to the depthwise kernel, and this file bypasses the op.

#include "nnops/core/tensor_layout.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "nnops/ops/conv1d.hpp"
#include "nnops/ops/conv2d.hpp"
#include "nnops/ops/depthwise_conv.hpp"

#include <span>

namespace nnops::backend::cuda {

// The two entries this file lifts into. Both take the OpBase signature.
Status conv2d_cuda(const Conv2DAttributes&, TensorView&, std::span<const TensorView>,
                   const ComputeContext&, void*);
Status depthwise_conv_cuda(const DepthwiseConvAttributes&, TensorView&,
                           std::span<const TensorView>, const ComputeContext&, void*);

namespace {

/// The view's storage as a writable pointer, for a TensorView that only reads
/// through the 2D entry's non-const signature.
void* raw(const TensorView& v) noexcept
{
    return const_cast<void*>(v.ptr<void>());
}

bool is_fp(DataType dtype) noexcept
{
    return dtype == DataType::f32 || dtype == DataType::f16 || dtype == DataType::bf16;
}

/// A rank-3 NWC signal as the 2D kernel's [N, 1, L, C] view. A row is L·C
/// either way, so the pitch carries over untouched, and the fabricated H axis
/// has stride L·C, which the gather reads only when h moves (it cannot).
TensorView lift_wide(const TensorView& v)
{
    const int64_t s[4] = {v.shape(0), 1, v.shape(1), v.shape(2)};
    return TensorView(std::span<const int64_t>(s, 4), v.data_type(), raw(v), v.pitch(),
                      v.layout());
}

/// An HWIO [K, IC/G, OC] filter as [1, K, IC/G, OC]. The innermost row is OC
/// either way, so the strides the kernel walks are the rank-3 ones.
TensorView lift_filter_wide(const TensorView& w)
{
    const int64_t s[4] = {1, w.shape(0), w.shape(1), w.shape(2)};
    return TensorView(std::span<const int64_t>(s, 4), w.data_type(), raw(w), w.pitch(),
                      w.layout());
}

/// The lifted filter is the 2D HWIO order, so this is Conv2D's predicate
/// verbatim: one input channel a group on both sides.
bool is_depthwise_geometry(const Conv2DAttributes& attrs, const TensorView& w)
{
    if (attrs.groups <= 1 || !is_channel_last_weight(w.layout())) {
        return false;
    }
    return w.shape(2) == 1 && w.shape(3) == attrs.groups;
}

} // anonymous namespace

Status conv1d_cuda(const Conv1DAttributes& attrs, TensorView& output,
                   std::span<const TensorView> inputs, const ComputeContext& ctx, void* workspace)
{
    const auto& input = inputs[0];
    const auto& weight = inputs[1];

    // The quantized route and the prepacked blob have no 1D CUDA form: the
    // packed blob describes a 2D kernel, and conv2d_cuda refuses it anyway.
    if (is_quantized_conv1d(weight.data_type()) || !is_fp(input.data_type()) ||
        !is_fp(weight.data_type()) || input.data_type() != weight.data_type() ||
        output.data_type() != input.data_type()) {
        return Status::unsupported;
    }
    if (output.numel() == 0) {
        return Status::ok;
    }

    Conv2DAttributes a{};
    a.kernel_size = {1, attrs.kernel_size};
    a.stride = {1, attrs.stride};
    a.dilation = {1, attrs.dilation};
    a.padding = {0, attrs.padding};
    a.groups = attrs.groups;
    a.epilogue = attrs.epilogue;
    a.add_to = attrs.add_to;
    a.output_dtype = attrs.output_dtype;
    a.act_quant_dtype = attrs.act_quant_dtype;
    // No 2D Winograd: its square tile has no 1-wide analogue.
    a.algorithm = ConvAlgorithm::ImplicitGemm;

    // auto_pad is still in the descriptor here — the CPU entry resolves it in
    // the kernel the same way, and Conv2D::compute is bypassed.
    if (attrs.auto_pad != Conv1DAttributes::AutoPad::NOTSET) {
        a.padding[1] = resolve_leading_padding(input.shape(1), attrs.kernel_size, attrs.stride,
                                               attrs.dilation, attrs.padding,
                                               static_cast<int>(attrs.auto_pad));
    }
    TensorView in4 = lift_wide(input);
    TensorView out4 = lift_wide(output);
    TensorView w4 = lift_filter_wide(weight);

    TensorView operands[3] = {in4, w4, TensorView{}};
    const size_t n_in = inputs.size();
    if (n_in > 2) {
        operands[2] = inputs[2];
    }
    const std::span<const TensorView> ins(operands, n_in);

    if (is_depthwise_geometry(a, w4)) {
        DepthwiseConvAttributes dw;
        dw.kernel_size = {1, 1, attrs.kernel_size};
        dw.stride = {1, 1, attrs.stride};
        dw.dilation = {1, 1, attrs.dilation};
        dw.padding = {0, 0, a.padding[1]};
        dw.epilogue = attrs.epilogue;
        dw.add_to = attrs.add_to;
        return depthwise_conv_cuda(dw, out4, ins, ctx, workspace);
    }

    return conv2d_cuda(a, out4, ins, ctx, workspace);
}

} // namespace nnops::backend::cuda
