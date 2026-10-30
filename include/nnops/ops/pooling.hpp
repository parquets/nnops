#pragma once
/// @file pooling.hpp
/// @brief Pooling operator — 2D/3D spatial pooling on either a planar
///        (NCHW/NCDHW) or a channel-last (NHWC/NDHWC) tensor.
///
/// The input's layout is the output's: NCHW stays NCHW. Planar input supports
/// f32/f16; quantized (s8/u8) pooling is channel-last only — the quantization
/// axis is the channel, which a planar plane does not keep together.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Supported pooling types.
enum class PoolingType : uint8_t {
    Max,                 ///< Maximum value in window
    Average,             ///< Average of values in window
};

/// Attributes for the Pooling operator (supports both 2D and 3D).
///
/// For 2D pooling (rank-4 input): only the last 2 elements of each array are
/// used (H, W dimensions). The first element (D) is ignored.
///
/// For 3D pooling (rank-5 input): all 3 elements are used (D, H, W).
struct PoolingAttributes {
    PoolingType type = PoolingType::Max;

    /// Kernel shape: [KD, KH, KW]. For 2D, only KH/KW are used.
    std::array<int64_t, 3> kernel_shape = {2, 2, 2};

    /// Stride: [SD, SH, SW]. For 2D, only SH/SW are used.
    std::array<int64_t, 3> stride = {1, 1, 1};

    /// Padding: [PD, PH, PW]. For 2D, only PH/PW are used.
    std::array<int64_t, 3> padding = {0, 0, 0};

    /// Dilation: [DD, DH, DW]. For 2D, only DH/DW are used.
    std::array<int64_t, 3> dilation = {1, 1, 1};

    /// If true, exclude padded positions from average computation
    /// (only meaningful when type == Average).
    bool exclude_pad = false;

    /// Auto-padding mode.
    enum class AutoPad : uint8_t {
        NOTSET = 0,
        SAME_UPPER,
        SAME_LOWER,
        VALID,
    };
    AutoPad auto_pad = AutoPad::NOTSET;

    /// If true, add result to existing output buffer instead of overwriting.
    bool add_to = false;

    /// Global pooling: the window is the whole spatial extent, so every spatial
    /// axis of the output is 1 (PyTorch's AdaptiveAvgPool2d(1), ONNX's
    /// GlobalAveragePool/GlobalMaxPool). kernel_shape, stride, padding,
    /// dilation and auto_pad are not consulted; @c type still says max or
    /// average. The CPU backend scans each plane directly for f32/f16.
    bool global = false;

    /// Returns the spatial rank (2 or 3) inferred from the input tensor rank.
    /// For a 4D input (NCHW), returns 2. For a 5D input (NCDHW), returns 3.
    static constexpr int64_t spatial_rank(int64_t input_rank) noexcept {
        return input_rank - 2;  // N, C + spatial dims
    }
};

/// Pooling operator (class-based API).
///
/// 2D: Input [N, C, IH, IW] → Output [N, C, OH, OW]
/// 3D: Input [N, C, ID, IH, IW] → Output [N, C, OD, OH, OW]
/// Global: Output [N, C, 1, 1] (or [N, C, 1, 1, 1] in 3D).
class Pooling : public OpBase {
public:
    static std::unique_ptr<Pooling> create(const PoolingAttributes& attrs,
                                           Backend backend = Backend::CPU);

    static std::unique_ptr<Pooling> create(Backend backend = Backend::CPU) {
        return create(PoolingAttributes{}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    Status validate(std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs) const override;

    using OpBase::compute;

    Status compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Pooling; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::Any; }

    const PoolingAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in pooling.cpp (Pimpl pattern)
    ~Pooling();

private:
    Pooling(const PoolingAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    PoolingAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
