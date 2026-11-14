#pragma once
/// @file norm.hpp
/// @brief Normalization operators — BatchNorm, LayerNorm, RMSNorm, GroupNorm,
///        L2Norm.
///
/// One NormAttributes struct and one Norm class, distinguished by NormType; the
/// SIMD kernels are consolidated too (simd_kernel/simd_norm.hpp).
///
/// Inputs by type: BatchNorm X,scale,bias,mean,var (5); LayerNorm X,scale[,bias]
/// (2-3); RMSNorm X,scale (2); GroupNorm X,scale[,bias] (2-3); L2Norm X (1).
/// With `add_residual`, one trailing input R is pre-added to X (see the
/// attribute). The output Y has X's shape.
///
/// LayerNorm and RMSNorm take a 1-centered weight (`(1+w)`, `one_plus_weight`):
/// Gemma's zero-centered norm. The `+1` is applied in the row kernel rather than
/// folded into the stored weight, so an f16/bf16 weight keeps its small-|w|
/// correction.
///
/// RMSNorm takes a gate (`gate_activation`), which turns it into the fused
/// RMSNormGated of the linear-attention blocks: `y = rms_norm(x) · act(g)` with
/// the gate as a trailing input. See the attribute for the slot order.
///
/// Quantized input (CPU): an s8/u8 X with PerTensor or PerToken params, for
/// LayerNorm / RMSNorm / L2Norm with axis == rank-1 only — one normalization
/// row is then exactly one quantization row. Y is f32/f16 (dequantized) or
/// s8/u8 (requantized) per output_dtype, while scale and bias stay f32.
/// BatchNorm and GroupNorm have no quantized route, and add_to is rejected on
/// it.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/data_type.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/epilogue.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

enum class NormType : uint8_t {
    BatchNorm = 0,   ///< per-channel, with pre-computed mean/var
    LayerNorm = 1,   ///< over [axis, rank), mean + variance
    RMSNorm = 2,     ///< root mean square, no mean subtraction
    GroupNorm = 3,   ///< per group of channels + spatial dims
    L2Norm = 4,      ///< L2 norm, no scale/bias
};

/// Only the fields relevant to the selected `type` are used; the rest are
/// ignored.
struct NormAttributes {
    NormType type = NormType::LayerNorm;

    /// First axis to normalize over (LayerNorm / RMSNorm / L2Norm); -1 is the
    /// last axis.
    int64_t axis = -1;

    /// GroupNorm: number of channel groups. C must be divisible by it; 1 is
    /// LayerNorm-like, C is InstanceNorm-like.
    int64_t num_groups = 1;

    float epsilon = 1e-5f;

    /// BatchNorm: per-channel statistics (true) vs per-element (false).
    bool spatial = true;

    /// Output storage on the quantized path: f32/f16 dequantizes, s8/u8
    /// requantizes with the output view's params. Ignored for float inputs,
    /// whose output follows X's dtype.
    DataType output_dtype = DataType::f32;

    /// Accumulate into the existing output instead of overwriting; rejected on
    /// the quantized-input path.
    bool add_to = false;

    /// 1-centered weight: the effective scale is `(1 + w)` instead of `w`. This
    /// is Gemma's zero-centered RMSNorm (and the Qwen3-Next / 3.5 / 4 norms);
    /// the stored weight is the *offset* from one, so exporting it into a plain
    /// `w` tensor would lose the small-|w| correction in f16/bf16. The `+1`
    /// happens inside the row kernel, so the cost is one vector add per row.
    /// LayerNorm and RMSNorm only — the other types reject it.
    bool one_plus_weight = false;

    /// Pre-add a residual before normalizing: `y = norm(X + R)`, the transformer
    /// block's `norm(x + sublayer(x))` in one pass. R is the trailing input, in
    /// X's slot order:
    ///   LayerNorm  [X, scale, bias?, residual]  (4 inputs; bias may be empty)
    ///   RMSNorm    [X, scale, residual]         (3)
    ///   L2Norm     [X, residual]                (1)
    /// R must match X's dims and dtype, and the normalized axis must be the
    /// rank-1 contiguous tail; BatchNorm/GroupNorm and quantized X are rejected.
    /// `add_to` still applies to the output, after the normalization.
    bool add_residual = false;

    /// RMSNormGated (GatedDeltaNet's output norm): the result is multiplied by
    /// `act(gate)`, a trailing input shaped like X:
    ///   RMSNorm  [X, scale, gate]              (3 inputs)
    ///   RMSNorm  [X, scale, residual, gate]    (4, with add_residual)
    /// The multiply rides on the row the norm kernel just wrote, so the fused
    /// call replaces Norm -> Activation -> Eltwise with one pass over the
    /// hidden. None disables; RMSNorm only (the other types' scale/group
    /// geometry has no gated consumer, and the quantized path rejects it).
    EpilogueActivateType gate_activation = EpilogueActivateType::None;
};

class Norm : public OpBase {
public:
    static std::unique_ptr<Norm> create(const NormAttributes& attrs = {},
                                        Backend backend = Backend::CPU);

    static std::unique_ptr<Norm> create(Backend backend) {
        return create(NormAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Norm; }
    Backend getBackend() const override { return backend_; }

    const NormAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~Norm();

private:
    Norm(const NormAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    NormAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
