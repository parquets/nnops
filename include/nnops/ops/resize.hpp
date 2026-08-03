#pragma once
/// @file resize.hpp
/// @brief Resize operator — 2D/3D spatial interpolation with NCHW/NCDHW layout.

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
/// For 2D resize (NCHW input, rank=4): output_size[1]=OH, output_size[2]=OW.
///   output_size[0] is ignored.
///
/// For 3D resize (NCDHW input, rank=5): all 3 elements are used (OD, OH, OW).
struct ResizeAttributes {
    ResizeMode mode = ResizeMode::Nearest;
    CoordinateTransformMode coord_mode = CoordinateTransformMode::HalfPixel;

    /// Output spatial size: [OD, OH, OW]. For 2D, only OH/OW are used.
    std::array<int64_t, 3> output_size = {0, 0, 0};

    /// If true, add result to existing output buffer instead of overwriting.
    bool add_to = false;

    /// Returns the spatial rank (2 or 3) inferred from the input tensor rank.
    /// For a 4D input (NCHW), returns 2. For a 5D input (NCDHW), returns 3.
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
    /// Create a Resize operator for the specified backend.
    static std::unique_ptr<Resize> create(const ResizeAttributes& attrs,
                                           Backend backend = Backend::CPU);

    /// Create with defaults.
    static std::unique_ptr<Resize> create(Backend backend = Backend::CPU) {
        return create(ResizeAttributes{}, backend);
    }

    // ---- OpBase interface ----
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

private:
    Resize(const ResizeAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    ResizeAttributes attrs_;
    Backend backend_;
};

/// Functional resize.
void resize(const TensorView& input,
            TensorView& output,
            const ResizeAttributes& attrs,
            const ComputeContext& ctx = {});

}  // namespace nnops
