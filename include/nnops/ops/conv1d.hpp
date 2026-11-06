#pragma once
/// @file conv1d.hpp
/// @brief 1D convolution — channel-first or channel-last, with groups and
///        dilation.
///
/// 1D is the 2D convolution with a 1-wide second spatial axis, and that is
/// exactly how it runs: the activation is lifted from [N, C, L] to
/// [N, C, 1, L] and the Conv2D implicit-GEMM kernel is called with a kernel of
/// {1, K}. The lift keeps the caller's row pitch and layout, and because the
/// lifted height is one the derived channel/batch/output strides are the 1D
/// ones — so the GEMM geometry is the one a native 1D conv would have
/// (channel-first M = OC, N = L, K = IC/G·K; channel-last pixel-major).
///
/// Activation [N, IC, L], weight [OC, IC/G, K] (OIHW) channel-first or
/// [K, IC/G, OC] (HWIO) channel-last, bias [OC] optional, output [N, OC, OL].
///
/// Data types: f32/f16 on every tensor (the floating-point route), or the
/// quantized route — an s8 weight with per-channel QuantParams against a
/// u8/s8/f16/f32 activation with per-tensor QuantParams, producing fp or s8/u8
/// output (see output_dtype and is_quantized_conv1d()). The bias dtype follows
/// the output, as on Conv2D.
///
/// Reused from Conv2D rather than reimplemented: the im2col/GEMM tiling, the
/// int8 route, and the depthwise hand-off (a group of one input and one output
/// channel runs the depthwise kernels on the raw filter, either layout).
/// Not offered: Winograd (its square tile has no 1-wide analogue) and
/// prepacked weights (the filter already is the [OC][IC/G·K] matrix the GEMM
/// wants, so prepackWeights() answers nothing).
///
/// Parallelism: the 2D plan splits over (batch, group, output-row chunk, filter
/// chunk) and a 1D tensor has exactly one output row, so on a wide pool a 1D
/// call can leave most workers idle — a channel-first int8 conv at N = 1 has a
/// single task. The CPU driver splits the L axis into contiguous chunks when
/// the plan under-subscribes the pool, each chunk calling the same kernel on a
/// sub-view; the chunks write disjoint output positions and read the same
/// filter, so the result is bit-exact whatever the split. See
/// src/backend/cpu/conv/conv1d.cpp for the rule.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/epilogue.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// True when the weight dtype selects the quantized route; the CPU backend
/// dispatches on this predicate.
inline constexpr bool is_quantized_conv1d(DataType weight_dtype) noexcept {
    return weight_dtype == DataType::s8;
}

/// Attributes describing a 1D convolution.
struct Conv1DAttributes {
    int64_t kernel_size = 1;
    int64_t stride = 1;
    int64_t dilation = 1;
    int64_t padding = 0;
    int64_t groups = 1;

    enum class AutoPad : uint8_t {
        NOTSET = 0,
        SAME_UPPER,
        SAME_LOWER,
        VALID,
    };
    AutoPad auto_pad = AutoPad::NOTSET;

    /// Write-back post-processing. On an s8/u8 destination only Relu applies,
    /// to the int32 accumulate before the requant; the clip range and the other
    /// activations are fp-epilogue only and are ignored there.
    Epilogue epilogue{};

    /// Accumulate into the existing output instead of overwriting (residual
    /// connections without a separate add kernel).
    bool add_to = false;

    // ---- quantized route (is_quantized_conv1d(weight): s8 weights) ----------
    // Same names and meanings as Conv2DAttributes; ignored on the fp route.

    /// Quantized-route output dtype.
    ///   f32 (default) / f16: dequantize the int32 accumulate. With an fp
    ///                        activation the output follows the activation's
    ///                        dtype and this value is ignored; with a
    ///                        pre-quantized (u8/s8) activation it is honoured.
    ///   s8 / u8:             requantize with the output's QuantParams. Honoured
    ///                        for every activation dtype; the bias is then int32
    ///                        rather than the output dtype.
    /// Any other value is refused by the kernel plan.
    DataType output_dtype = DataType::f32;

    /// Integer storage an fp activation is quantized to inside the kernel, per
    /// im2col tile: s8 (default) or u8. Only read when the activation is f32/f16.
    DataType act_quant_dtype = DataType::s8;
};

/// [N, IC, L] → [N, OC, OL], either layout.
class Conv1D : public OpBase {
public:
    static std::unique_ptr<Conv1D> create(const Conv1DAttributes& attrs,
                                          Backend backend = Backend::CPU);

    static std::unique_ptr<Conv1D> create(Backend backend = Backend::CPU) {
        return create(Conv1DAttributes{}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    Status validate(std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs) const override;

    /// Always 0 — the CPU kernels pool their scratch internally.
    size_t getWorkspaceSize(std::span<const TensorDesc> inputs,
                            std::span<const TensorDesc> outputs) const override;

    /// inputs: [0] input, [1] weight (raw filter), [2] optional bias [OC].
    using OpBase::compute;

    Status compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Conv1D; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override
    {
        return LayoutSupport::ChannelFirstOrChannelLast;
    }

    const Conv1DAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in conv1d.cpp
    ~Conv1D();

private:
    Conv1D(const Conv1DAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    Conv1DAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
