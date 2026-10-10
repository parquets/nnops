#pragma once
/// @file norm.hpp
/// @brief Normalization operators — BatchNorm, LayerNorm, RMSNorm, GroupNorm, L2Norm.
///
/// All normalization operators share a single NormAttributes struct and a
/// single Norm class, distinguished by NormType. The SIMD kernels are also
/// consolidated (simd_kernel/simd_norm.hpp).

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/data_type.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Normalization operator type.
enum class NormType : uint8_t {
    /// BatchNorm: per-channel normalization with pre-computed mean/var (5 inputs).
    BatchNorm = 0,
    /// LayerNorm: normalize over [axis, rank), mean + variance (2-3 inputs).
    LayerNorm = 1,
    /// RMSNorm: normalize by root mean square, no mean subtraction (2 inputs).
    RMSNorm = 2,
    /// GroupNorm: normalize per-group of channels + spatial dims (2-3 inputs).
    GroupNorm = 3,
    /// L2Norm: normalize by L2 norm, no scale/bias (1 input).
    L2Norm = 4,
};

/// Attributes shared by all normalization operators.
///
/// Only the fields relevant to the selected `type` are used; the rest are
/// ignored.
struct NormAttributes {
    /// Normalization type.
    NormType type = NormType::LayerNorm;

    /// First axis to normalize over (LayerNorm / RMSNorm / L2Norm).
    /// Default -1 normalizes the last dimension only.
    int64_t axis = -1;

    /// Number of channel groups (GroupNorm). C must be divisible by num_groups.
    /// 1 = LayerNorm-like, C = InstanceNorm-like.
    int64_t num_groups = 1;

    /// Small constant for numerical stability.
    float epsilon = 1e-5f;

    /// BatchNorm: per-channel statistics (true) vs per-element (false).
    bool spatial = true;

    /// Storage type of the output when X is quantized (s8/u8): f32/f16 for a
    /// dequantized output, s8/u8 for a requantized one (the quantized output's
    /// scale/zero_point come from the output TensorView). Ignored for float
    /// inputs, whose output follows X's dtype. Default f32.
    DataType output_dtype = DataType::f32;

    /// If true, add result to existing output buffer instead of overwriting.
    /// Not supported on the quantized-input path.
    bool add_to = false;
};

/// Normalization operator (class-based API).
///
/// Dispatches on NormAttributes::type to the appropriate normalization kernel.
///
/// Inputs by type:
///   BatchNorm  (5): X, scale, bias, mean, var
///   LayerNorm  (2-3): X, scale [, bias]
///   RMSNorm    (2): X, scale
///   GroupNorm  (2-3): X, scale [, bias]
///   L2Norm     (1): X
///
/// Output (1): Y with the same shape as X.
///
/// Quantized input (CPU): X may be s8/u8 with PerTensor or PerToken quant
/// params, in which case Y's dtype is NormAttributes::output_dtype. Supported
/// for LayerNorm / RMSNorm / L2Norm with axis == rank-1 only.
class Norm : public OpBase {
public:
    static std::unique_ptr<Norm> create(const NormAttributes& attrs = {},
                                        Backend backend = Backend::CPU);

    static std::unique_ptr<Norm> create(Backend backend) {
        return create(NormAttributes{}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Norm; }
    Backend getBackend() const override { return backend_; }

    const NormAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in norm.cpp (Pimpl pattern)
    ~Norm();

private:
    Norm(const NormAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    NormAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
