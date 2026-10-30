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
/// The output Y has X's shape.
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
