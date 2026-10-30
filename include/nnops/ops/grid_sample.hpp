#pragma once
/// @file grid_sample.hpp
/// @brief GridSample operator — 2D/3D sampling at grid points, packed
///        NCHWC8/NCDHWC8 layout only.
///
/// The grid holds normalized coordinates in [-1, 1]: -1 is the outer
/// left/top/front border and 1 the right/bottom/back one. f32/f16 only.
///
/// 2D: Input [N, C, IH, IW], Grid [N, OH, OW, 2] → Output [N, C, OH, OW]
/// 3D: Input [N, C, ID, IH, IW], Grid [N, OD, OH, OW, 3] → Output [N, C, OD, OH, OW]
///
/// Reference: ONNX GridSample operator (since opset 16).

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

enum class GridSampleMode : uint8_t {
    Bilinear,   ///< Bilinear (2D) interpolation
    Nearest,    ///< Nearest-neighbor interpolation
};

enum class GridSamplePaddingMode : uint8_t {
    Zeros,      ///< Fill with zero for out-of-bounds locations
    Border,     ///< Clamp to the nearest border pixel
    Reflection, ///< Reflect coordinates at the boundaries
};

struct GridSampleAttributes {
    GridSampleMode mode = GridSampleMode::Bilinear;
    GridSamplePaddingMode padding_mode = GridSamplePaddingMode::Zeros;

    /// If true, -1 and 1 in the grid map to the centers of the corner pixels;
    /// if false (default), to their outer edges.
    bool align_corners = false;

    /// Accumulate into the existing output instead of overwriting (residual).
    bool add_to = false;

    static constexpr int64_t spatial_rank(int64_t input_rank) noexcept {
        return input_rank - 2;  // N, C + spatial dims
    }
};

class GridSample : public OpBase {
public:
    static std::unique_ptr<GridSample> create(const GridSampleAttributes& attrs,
                                                Backend backend = Backend::CPU);

    static std::unique_ptr<GridSample> create(Backend backend = Backend::CPU) {
        return create(GridSampleAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::GridSample; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::PackedOnly; }

    const GridSampleAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~GridSample();

private:
    GridSample(const GridSampleAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    GridSampleAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
