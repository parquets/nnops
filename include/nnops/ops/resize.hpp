#pragma once
/// @file resize.hpp
/// @brief Resize operator — 2D/3D spatial interpolation, packed NCHWC8/NCDHWC8 layout.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Supported interpolation modes.
enum class ResizeMode : uint8_t {
    Nearest,  ///< Nearest-neighbor interpolation
    Linear,   ///< Bilinear (2D) / Trilinear (3D) interpolation
};

/// Coordinate transformation mode for mapping output coords to source coords.
enum class CoordinateTransformMode : uint8_t {
    HalfPixel,    ///< src = (dst + 0.5) * scale - 0.5
    AlignCorners, ///< src = dst * (src_size - 1) / (dst_size - 1)
    Asymmetric,   ///< src = dst * scale
};

/// Attributes for the Resize operator (supports both 2D and 3D).
///
/// For 2D resize (NCHWC8 input, rank=4): output_size[1]=OH, output_size[2]=OW.
///   output_size[0] is ignored.
///
/// For 3D resize (NCDHWC8 input, rank=5): all 3 elements are used (OD, OH, OW).
struct ResizeAttributes {
    ResizeMode mode = ResizeMode::Nearest;
    CoordinateTransformMode coord_mode = CoordinateTransformMode::HalfPixel;

    /// Output spatial size: [OD, OH, OW]. For 2D, only OH/OW are used.
    std::array<int64_t, 3> output_size = {0, 0, 0};

    /// If true, add result to existing output buffer instead of overwriting.
    bool add_to = false;

    /// Crop region in input pixel coordinates.
    /// crop_start = {start_d, start_h, start_w} — inclusive start.
    /// crop_end   = {end_d,   end_h,   end_w}   — exclusive end.
    ///
    /// When crop_end[d] == 0, no crop is applied for that dimension
    /// (the full input spatial extent is used).
    ///
    /// For 2D input (NCHWC8, rank=4), index 0 (D) is ignored
    ///   — only crop_start[1]/[2] and crop_end[1]/[2] (H, W) are used.
    /// For 3D input (NCDHWC8, rank=5), all three indices are used.
    ///
    /// Example: input [N,C,8,16,16], crop_start={2,4,4}, crop_end={6,12,12}
    ///   → crops region d=[2,6), h=[4,12), w=[4,12), then resizes to output_size.
    std::array<int64_t, 3> crop_start = {0, 0, 0};
    std::array<int64_t, 3> crop_end   = {0, 0, 0};

    /// Returns true if any crop dimension is active (crop_end > 0).
    bool has_crop() const noexcept {
        return crop_end[0] > 0 || crop_end[1] > 0 || crop_end[2] > 0;
    }

    /// Returns the spatial rank (2 or 3) inferred from the input tensor rank.
    /// For a 4D input (NCHWC8), returns 2. For a 5D input (NCDHWC8), returns 3.
    static constexpr int64_t spatial_rank(int64_t input_rank) noexcept {
        return input_rank - 2;  // N, C + spatial dims
    }
};

/// Resize operator (class-based API).
///
/// 2D: Input [N, C, IH, IW] → Output [N, C, OH, OW]
/// 3D: Input [N, C, ID, IH, IW] → Output [N, C, OD, OH, OW]
class Resize : public OpBase {
public:
    static std::unique_ptr<Resize> create(const ResizeAttributes& attrs,
                                           Backend backend = Backend::CPU);

    static std::unique_ptr<Resize> create(Backend backend = Backend::CPU) {
        return create(ResizeAttributes{}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Resize; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::PackedOnly; }

    const ResizeAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in resize.cpp (Pimpl pattern)
    ~Resize();

private:
    Resize(const ResizeAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    ResizeAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
