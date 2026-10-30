#pragma once
/// @file resize.hpp
/// @brief Resize operator — 2D/3D spatial interpolation, packed NCHWC8/NCDHWC8.
///
/// Per-axis arrays are [D, H, W]; a rank-4 (2D) input reads only H/W (index 0
/// is ignored), a rank-5 input all three.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

enum class ResizeMode : uint8_t {
    Nearest,  ///< Nearest-neighbor
    Linear,   ///< Bilinear (2D) / trilinear (3D)
};

/// Coordinate transformation mode for mapping output coords to source coords.
enum class CoordinateTransformMode : uint8_t {
    HalfPixel,    ///< src = (dst + 0.5) * scale - 0.5
    AlignCorners, ///< src = dst * (src_size - 1) / (dst_size - 1)
    Asymmetric,   ///< src = dst * scale
};

struct ResizeAttributes {
    ResizeMode mode = ResizeMode::Nearest;
    CoordinateTransformMode coord_mode = CoordinateTransformMode::HalfPixel;

    /// Output spatial size [OD, OH, OW]; 2D uses OH/OW.
    std::array<int64_t, 3> output_size = {0, 0, 0};

    /// If true, add into the existing output buffer instead of overwriting.
    bool add_to = false;

    /// Crop region in input pixel coords: start inclusive, end exclusive;
    /// crop_end[d] == 0 leaves that dimension uncropped. 2D uses H/W only.
    std::array<int64_t, 3> crop_start = {0, 0, 0};
    std::array<int64_t, 3> crop_end   = {0, 0, 0};

    /// True if any crop dimension is active (crop_end > 0).
    bool has_crop() const noexcept {
        return crop_end[0] > 0 || crop_end[1] > 0 || crop_end[2] > 0;
    }

    /// 2 for a rank-4 input, 3 for rank-5.
    static constexpr int64_t spatial_rank(int64_t input_rank) noexcept {
        return input_rank - 2;  // N, C + spatial dims
    }
};

/// 2D: [N, C, IH, IW] → [N, C, OH, OW]; 3D adds a D axis.
class Resize : public OpBase {
public:
    static std::unique_ptr<Resize> create(const ResizeAttributes& attrs,
                                           Backend backend = Backend::CPU);

    static std::unique_ptr<Resize> create(Backend backend = Backend::CPU) {
        return create(ResizeAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Resize; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::PackedOnly; }

    const ResizeAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in resize.cpp
    ~Resize();

private:
    Resize(const ResizeAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    ResizeAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
