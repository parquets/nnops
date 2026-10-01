#pragma once
/// @file argminmax.hpp
/// @brief ArgMax / ArgMin operators — indices of max/min along an axis.
///
/// Computes the index (as int64) of the maximum or minimum value
/// along a specified axis, with optional keepdims.
///
/// Supports NCHW, NCDHW, NCHWC8, NCDHWC8 layouts.
/// F32 and F16 data types are supported.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Operation type for arg reduction.
enum class ArgMinMaxType : uint8_t {
    Max,  ///< ArgMax: index of maximum value
    Min,  ///< ArgMin: index of minimum value
};

/// Attributes shared by ArgMax and ArgMin.
struct ArgMinMaxAttributes {
    /// Reduction operation type.
    ArgMinMaxType type = ArgMinMaxType::Max;

    /// Axis along which to compute argmax/argmin.
    /// Negative values wrap from the end. Default: 0.
    int64_t axis = 0;

    /// If true, keep the reduced axis as a size-1 dimension.
    bool keepdims = false;
};

/// ArgMax operator (class-based API).
///
/// Computes the index of the maximum value along the specified axis.
///
/// Input:  X [*]  (f32 or f16)
/// Output: Y [*]  (int64, axis reduced or size-1 if keepdims)
class ArgMax : public OpBase {
public:
    static std::unique_ptr<ArgMax> create(const ArgMinMaxAttributes& attrs = {},
                                           Backend backend = Backend::CPU);

    static std::unique_ptr<ArgMax> create(Backend backend) {
        return create(ArgMinMaxAttributes{}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::ArgMax; }
    Backend getBackend() const override { return backend_; }

    const ArgMinMaxAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~ArgMax();

private:
    ArgMax(const ArgMinMaxAttributes& attrs, Backend backend);
    std::unique_ptr<Impl> impl_;
    ArgMinMaxAttributes attrs_;
    Backend backend_;
};

/// ArgMin operator (class-based API).
///
/// Computes the index of the minimum value along the specified axis.
///
/// Input:  X [*]  (f32 or f16)
/// Output: Y [*]  (int64, axis reduced or size-1 if keepdims)
class ArgMin : public OpBase {
public:
    static std::unique_ptr<ArgMin> create(const ArgMinMaxAttributes& attrs = {},
                                           Backend backend = Backend::CPU);

    static std::unique_ptr<ArgMin> create(Backend backend) {
        return create(ArgMinMaxAttributes{ArgMinMaxType::Min}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::ArgMin; }
    Backend getBackend() const override { return backend_; }

    const ArgMinMaxAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~ArgMin();

private:
    ArgMin(const ArgMinMaxAttributes& attrs, Backend backend);
    std::unique_ptr<Impl> impl_;
    ArgMinMaxAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
