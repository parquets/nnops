#pragma once
/// @file grid_sample.hpp
/// @brief GridSample operator — 2D/3D grid-based spatial sampling with NCHW/NCDHW layout.
///
/// Given an input tensor and a flow-field grid, GridSample computes the output by
/// sampling the input at locations defined by the grid. The grid contains normalized
/// coordinates in [-1, 1], where -1 maps to the left/top/front border and 1 maps to
/// the right/bottom/back border.
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

/// Interpolation mode for grid sampling.
enum class GridSampleMode : uint8_t {
    Bilinear,   ///< Bilinear (2D) interpolation
    Nearest,    ///< Nearest-neighbor interpolation
};

/// Padding mode for out-of-bounds grid coordinates.
enum class GridSamplePaddingMode : uint8_t {
    Zeros,      ///< Fill with zero for out-of-bounds locations
    Border,     ///< Clamp to the nearest border pixel
    Reflection, ///< Reflect coordinates at the boundaries
};

/// Attributes for the GridSample operator (supports both 2D and 3D).
struct GridSampleAttributes {
    GridSampleMode mode = GridSampleMode::Bilinear;
    GridSamplePaddingMode padding_mode = GridSamplePaddingMode::Zeros;

    /// If true, -1 and 1 in the grid map to the centers of the corner pixels.
    /// If false (default), -1 and 1 map to the outer edges of the corner pixels.
    bool align_corners = false;

    /// If true, add result to existing output buffer instead of overwriting.
    bool add_to = false;

    /// Returns the spatial rank (2 or 3) inferred from the input tensor rank.
    /// For a 4D input (NCHW), returns 2. For a 5D input (NCDHW), returns 3.
    static constexpr int64_t spatial_rank(int64_t input_rank) noexcept {
        return input_rank - 2;  // N, C + spatial dims
    }
};

/// GridSample operator (class-based API).
///
/// Samples values from an input tensor at grid-defined locations.
///
/// 2D: Input [N, C, IH, IW], Grid [N, OH, OW, 2] → Output [N, C, OH, OW]
/// 3D: Input [N, C, ID, IH, IW], Grid [N, OD, OH, OW, 3] → Output [N, C, OD, OH, OW]
class GridSample : public OpBase {
public:
    /// Create a GridSample operator for the specified backend.
    static std::unique_ptr<GridSample> create(const GridSampleAttributes& attrs,
                                                Backend backend = Backend::CPU);

    /// Create with defaults.
    static std::unique_ptr<GridSample> create(Backend backend = Backend::CPU) {
        return create(GridSampleAttributes{}, backend);
    }

    // ---- OpBase interface ----
    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::GridSample; }
    Backend getBackend() const override { return backend_; }

    const GridSampleAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in grid_sample.cpp (Pimpl pattern)

private:
    GridSample(const GridSampleAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    GridSampleAttributes attrs_;
    Backend backend_;
};

/// Functional grid_sample.
void grid_sample(const TensorView& input,
                 const TensorView& grid,
                 TensorView& output,
                 const GridSampleAttributes& attrs,
                 const ComputeContext& ctx = {});

}  // namespace nnops
