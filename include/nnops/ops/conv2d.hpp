#pragma once
/// @file conv2d.hpp
/// @brief Conv2D operator — 2D convolution with NCHW layout.

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

/// Convolution algorithm selection for the CPU backend.
///
/// ImplicitGemm is the general tiled im2col + GEMM path and handles any
/// kernel size / stride / dilation / groups. Winograd is the F(2×2, 3×3)
/// minimal-filtering fast path: it is only valid for k3s1 (dilation 1) and
/// requires the caller to run prepackWeights first (compute then takes the
/// transformed weight U as inputs[1], not the raw weight).
enum class ConvAlgorithm : uint8_t {
    ImplicitGemm = 0,  ///< Tiled im2col + GEMM (default; supports any geometry)
    Winograd,          ///< F(2×2, 3×3) Winograd, k3s1 only, requires prepacked U
};

/// Attributes describing a 2D convolution.
struct Conv2DAttributes {
    /// Kernel size along height and width.
    std::array<int64_t, 2> kernel_size = {1, 1};

    /// Stride along height and width.
    std::array<int64_t, 2> stride = {1, 1};

    /// Dilation along height and width.
    std::array<int64_t, 2> dilation = {1, 1};

    /// Symmetric zero-padding along height and width.
    std::array<int64_t, 2> padding = {0, 0};

    /// Number of groups for grouped / depthwise convolution.
    int64_t groups = 1;

    /// Convenience enum for automatic padding computation.
    enum class AutoPad : uint8_t {
        NOTSET = 0,
        SAME_UPPER,
        SAME_LOWER,
        VALID,
    };
    AutoPad auto_pad = AutoPad::NOTSET;

    /// Post-processing applied during output write-back (default: identity).
    Epilogue epilogue{};

    /// If true, add result to existing output buffer instead of overwriting.
    /// Enables residual connections without a separate add kernel:
    ///   output += Conv(input)  rather than  output = Conv(input)
    bool add_to = false;

    /// Kernel-selection hint for the CPU backend (default preserves today's
    /// tiled im2col path, so existing callers are unaffected).
    ///
    /// Winograd is only honoured for k3s1 (KH == KW == 3, stride 1, dilation 1)
    /// with a raw f32/f16 weight, and it requires the transformed weight U —
    /// produced by prepackWeights — as inputs[1] of compute(). Shape inference
    /// is unaffected: getOutputTensorDesc still reads the *raw* weight desc, so
    /// the caller passes the raw weight for shape inference and U for compute.
    ConvAlgorithm algorithm = ConvAlgorithm::ImplicitGemm;
};

/// Conv2D operator (class-based API).
///
/// Supports NCHW layout with grouped and dilated convolution.
/// Input: [N, IC, IH, IW], Weight: [OC, IC/G, KH, KW], Output: [N, OC, OH, OW].
/// Optional bias: [OC].
class Conv2D : public OpBase {
public:
    static std::unique_ptr<Conv2D> create(const Conv2DAttributes& attrs,
                                          Backend backend = Backend::CPU);

    static std::unique_ptr<Conv2D> create(Backend backend = Backend::CPU) {
        return create(Conv2DAttributes{}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    /// Always 0: the CPU tiled im2col + GEMM kernel pools its scratch
    /// internally.
    size_t getWorkspaceSize(std::span<const TensorDesc> inputs,
                            std::span<const TensorDesc> outputs) const override;

    /// inputs[0] = input tensor (NCHW)
    /// inputs[1] = weight tensor [OC, IC/G, KH, KW] — ONNX-style, also when
    ///             grouped. When attrs.algorithm == Winograd this is instead the
    ///             prepacked transform-domain weight U [16, OC, IC/G] returned
    ///             by prepackWeights.
    /// inputs[2] = bias tensor [OC] (optional)
    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    /// Winograd weight transform (U = G·g·Gᵀ) — two-step contract like
    /// DepthwiseConv:
    ///   Step 1 — query: outputs[0] empty. Fills shape [16, OC, IC/G], the
    ///            weight's dtype, layout PackedWinoWeight, and an explicit
    ///            pitch of IC/G × elem_size.
    ///   Step 2 — pack: outputs[0] pre-allocated to the queried size. Writes U.
    ///
    /// Requires attrs.algorithm == Winograd and k3s1 (KH == KW == 3, stride 1,
    /// dilation 1); asserts otherwise, since a later compute() receives U and
    /// cannot fall back to the raw-weight im2col path.
    void prepackWeights(std::span<const TensorView> inputs,
                        std::span<TensorView> outputs,
                        const ComputeContext& ctx = {}) override;

    OpType  getOpType()  const override { return OpType::Conv2D; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::PlanarOnly; }

    const Conv2DAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in conv2d.cpp (Pimpl pattern)
    ~Conv2D();

private:
    Conv2D(const Conv2DAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    Conv2DAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
