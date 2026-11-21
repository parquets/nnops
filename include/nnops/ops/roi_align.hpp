#pragma once
/// @file roi_align.hpp
/// @brief RoiAlign — bilinear ROI pooling over an NHWC/NDHWC-style feature map.
///
/// ONNX's RoiAlign, channel-last: every ROI is cut from one feature map,
/// divided into an OH × OW grid of bins, and each bin averages (or maxes) a
/// regular grid of bilinear samples:
///
///   X     [N, H, W, C]        f32 | f16
///   rois  [R, 4]              f32, corner form (x1, y1, x2, y2)
///   batch [R]                 s32 | s64, optional — all zeros when absent
///   Y     [R, OH, OW, C]      X's dtype, NHWC
///
/// Per ROI:
///   start = rois[r] · spatial_scale, and with HalfPixel the four corners are
///   pulled in by 0.5 (the aligned form of torchvision's roi_align); the bin
///   size is (end - start) / outputs, and bin (ph, pw) samples
///   grid_h × grid_w points, with grid = sampling_ratio when it is positive
///   and ceil(roi size / output size) when it is 0.
///
/// The bilinear tap is the ONNX one: a sample outside `[-1, H] × [-1, W]` is
/// zero, a sample clamped to the far border takes the border pixel, and the
/// four weights are the usual separable products. Rois may overlap, and a
/// batch index outside [0, N) writes zeros for that ROI rather than aborting
/// the call — batch indices live in data, which validate() cannot see.
///
/// This is the detection / crop head a 3D-perception or VLA pipeline needs
/// between the backbone and the policy: crop boxes out of a feature pyramid,
/// pool them to a fixed grid, and hand [R, OH, OW, C] to the next stage.
/// The 3D sibling (trilinear ROI pooling over NDHWC) is not implemented.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// How the samples inside one bin are combined.
enum class RoiAlignMode : uint8_t {
    Avg,  ///< mean over the sampling grid
    Max,  ///< max over the sampling grid
};

/// How ROI coordinates map into the feature map.
enum class RoiAlignCoordinateMode : uint8_t {
    HalfPixel,       ///< subtract 0.5 from every corner (torchvision aligned=True)
    OutputHalfPixel, ///< legacy: use the corners as given (aligned=False)
};

struct RoiAlignAttributes {
    RoiAlignMode mode = RoiAlignMode::Avg;
    RoiAlignCoordinateMode coordinate_mode = RoiAlignCoordinateMode::HalfPixel;

    /// Pooled grid size; both must be >= 1.
    int64_t output_height = 1;
    int64_t output_width = 1;

    /// Samples per bin side; 0 means adaptive (ceil(roi size / output size)).
    int64_t sampling_ratio = 0;

    /// Feature-map units per ROI unit (the stride of the pyramid level the
    /// ROIs were detected on); must be > 0.
    float spatial_scale = 1.0f;

    /// Accumulate into the existing output instead of overwriting.
    bool add_to = false;
};

class RoiAlign : public OpBase {
public:
    static std::unique_ptr<RoiAlign> create(const RoiAlignAttributes& attrs,
                                            Backend backend = Backend::CPU);

    static std::unique_ptr<RoiAlign> create(Backend backend = Backend::CPU) {
        return create(RoiAlignAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::RoiAlign; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::ChannelLastOnly;
    }

    const RoiAlignAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~RoiAlign();

private:
    RoiAlign(const RoiAlignAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    RoiAlignAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
