#pragma once
/// @file conv3d.hpp
/// @brief 3D convolution over NDHWC activations.
///
/// Input [N, ID, IH, IW, IC] (channel-last), weight [KD, KH, KW, IC/G, OC] —
/// TensorFlow's conv3d filter, the shape *is* the filter and the output
/// channel is contiguous — output [N, OD, OH, OW, OC], optional bias [OC].
/// Grouped and dilated. The activation and the filter order must pair (see
/// validate()).
///
/// compute() takes the raw filter, or the blob prepackWeights() built for it
/// (PackedDHWIO). The prepack is the caller's decision because it allocates
/// the buffer; it is what lets a repeated call skip the per-call B pack, and a
/// prepacked filter is refused by every input it was not packed for.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/epilogue.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// True when the weight dtype selects the quantized route; the CPU backend
/// dispatches on this predicate.
inline constexpr bool is_quantized_conv3d(DataType weight_dtype) noexcept {
    return weight_dtype == DataType::s8;
}

/// Attributes describing a 3D convolution.
struct Conv3DAttributes {
    std::array<int64_t, 3> kernel_size = {1, 1, 1};
    std::array<int64_t, 3> stride = {1, 1, 1};
    std::array<int64_t, 3> dilation = {1, 1, 1};
    std::array<int64_t, 3> padding = {0, 0, 0};
    int64_t groups = 1;

    enum class AutoPad : uint8_t {
        NOTSET = 0,
        SAME_UPPER,
        SAME_LOWER,
        VALID,
    };
    AutoPad auto_pad = AutoPad::NOTSET;

    Epilogue epilogue{};

    /// Accumulate into the existing output instead of overwriting.
    bool add_to = false;

    // ---- quantized route (is_quantized_conv3d(weight): s8 weights) ----------
    // Same contract as Conv2D's: same field names, same meaning, ignored on the
    // floating-point route.

    /// Quantized-route output dtype.
    ///   f32 (default) / f16: dequantize the int32 accumulate. With an fp
    ///                        activation the output follows the activation's
    ///                        dtype and this value is ignored — the two must be
    ///                        one float type, so an f16 activation against a
    ///                        f32 output (or the reverse) is unsupported, not
    ///                        converted. With a pre-quantized (u8/s8) activation
    ///                        this value is honoured.
    ///   s8 / u8:             requantize with the output's QuantParams (scale
    ///                        s_w·s_x/out_scale, round half away from zero,
    ///                        clamped to the storage range; the product is f32).
    ///                        Honoured for every activation dtype; the bias is
    ///                        then int32 rather than the output dtype.
    /// Any other value is refused (s32 included: the raw MatMulInteger
    /// destination exists on MatMul, not here).
    DataType output_dtype = DataType::f32;

    /// Integer storage an fp activation is quantized to inside the kernel, per
    /// im2col tile: s8 (default) or u8, selecting the clamp range.
    DataType act_quant_dtype = DataType::s8;
};

class Conv3D : public OpBase {
public:
    static std::unique_ptr<Conv3D> create(const Conv3DAttributes& attrs,
                                          Backend backend = Backend::CPU);

    static std::unique_ptr<Conv3D> create(Backend backend = Backend::CPU) {
        return create(Conv3DAttributes{}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    Status validate(std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs) const override;

    /// inputs: [0] input, [1] weight (raw DHWIO filter, or the blob
    /// prepackWeights() produced), [2] optional bias [OC].
    using OpBase::compute;

    Status compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    /// Pack the filter into the GEMM's B once, for compute() to read in place
    /// (layout PackedDHWIO): an empty outputs[0] queries the descriptor, a
    /// populated one receives the pack. inputs[0] is the raw filter; the query
    /// needs no activation, since the geometry follows from the attributes and
    /// the filter's own dims. f32/f16 and s8 filters only — any other dtype
    /// queries an empty view.
    void prepackWeights(std::span<const TensorView> inputs,
                        std::span<TensorView> outputs,
                        const ComputeContext& ctx = {}) override;

    OpType  getOpType()  const override { return OpType::Conv3D; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override
    {
        return LayoutSupport::ChannelLastOnly;
    }

    const Conv3DAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in conv3d.cpp (Pimpl pattern)
    ~Conv3D();

private:
    Conv3D(const Conv3DAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    Conv3DAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
