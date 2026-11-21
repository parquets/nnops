#pragma once
/// @file non_max_suppression.hpp
/// @brief NonMaxSuppression — greedy IoU suppression per (batch, class).
///
/// ONNX NonMaxSuppression with the usual fused TopK pre-selection:
///
///   boxes  [B, N, 4]        f32 | f16
///   scores [B, C, N]        f32 | f16
///   max_output_boxes_per_class [1]  s32 | s64   optional
///   iou_threshold              [1]  f32         optional
///   score_threshold            [1]  f32         optional
///   -> indices [cap, 3] s64, num_selected [1] s64
///
/// The optional inputs are positional, as in ONNX, and may arrive as empty
/// views; a missing slot takes ONNX's default — 0 for all three, so an absent
/// max_output_boxes_per_class selects nothing. `iou_threshold` suppresses a
/// box when its IoU with an already-selected box is strictly greater.
///
/// Every row of the output is (batch_index, class_index, box_index), emitted in
/// (batch, class) order and, inside a class, in selection order (highest score
/// first). The second output is the number of rows written, `num_selected[0]`:
/// the selected count depends on data, and this library sizes every output at
/// plan time, so the caller allocates a capacity and reads back how much of it
/// the call used. `cap` in the suggested descriptor is the upper bound
/// B·C·min(N, top_k); a caller is free to allocate less, and the kernel stops
/// at the capacity it is given.
///
/// `top_k` is the fused TopK step: before the O(n²) suppression, each
/// (batch, class) keeps only its k highest-scoring candidates above the score
/// threshold. That is what turns NMS from unusable to routine on a detector
/// with tens of thousands of anchors, and 0 disables it (all candidates).
///
/// `center_point_box` picks the box encoding: 0 is ONNX's corner form
/// [y1, x1, y2, x2], 1 its center form [x_center, y_center, width, height].
/// Boxes and scores must share a dtype; the arithmetic is f32 either way.
///
/// The suppression runs one (batch, class) at a time on the calling thread —
/// the output positions of class k depend on how many rows classes 0..k-1
/// wrote, so classes cannot append concurrently without a second compaction
/// pass. Parallelise across images at the call site when B > 1.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

struct NonMaxSuppressionAttributes {
    /// 0: corner boxes [y1, x1, y2, x2] (ONNX); 1: center boxes
    /// [x_center, y_center, width, height].
    int64_t center_point_box = 0;

    /// Candidates kept per (batch, class) before suppression — the fused TopK
    /// step. 0 keeps every candidate above the score threshold.
    int64_t top_k = 0;
};

class NonMaxSuppression : public OpBase {
public:
    static std::unique_ptr<NonMaxSuppression> create(
        const NonMaxSuppressionAttributes& attrs = {},
        Backend backend = Backend::CPU);

    static std::unique_ptr<NonMaxSuppression> create(Backend backend) {
        return create(NonMaxSuppressionAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::NonMaxSuppression; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::Any; }

    const NonMaxSuppressionAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~NonMaxSuppression();

private:
    NonMaxSuppression(const NonMaxSuppressionAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    NonMaxSuppressionAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
