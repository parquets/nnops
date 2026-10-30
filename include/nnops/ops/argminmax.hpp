#pragma once
/// @file argminmax.hpp
/// @brief ArgMax / ArgMin — index of the max/min along one axis.
///
/// Returns the index as int64, with optional keepdims. Walks any strided
/// layout via stride_elems; f32 and f16.

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
    Max,
    Min,
};

/// Attributes shared by ArgMax and ArgMin.
struct ArgMinMaxAttributes {
    ArgMinMaxType type = ArgMinMaxType::Max;

    /// Axis; negative values wrap from the end.
    int64_t axis = 0;

    /// Keep the reduced axis as size 1.
    bool keepdims = false;
};

class ArgMax : public OpBase {
public:
    static std::unique_ptr<ArgMax> create(const ArgMinMaxAttributes& attrs = {},
                                           Backend backend = Backend::CPU);

    static std::unique_ptr<ArgMax> create(Backend backend) {
        return create(ArgMinMaxAttributes{}, backend);
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

class ArgMin : public OpBase {
public:
    static std::unique_ptr<ArgMin> create(const ArgMinMaxAttributes& attrs = {},
                                           Backend backend = Backend::CPU);

    static std::unique_ptr<ArgMin> create(Backend backend) {
        return create(ArgMinMaxAttributes{ArgMinMaxType::Min}, backend);
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
