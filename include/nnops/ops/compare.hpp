#pragma once
/// @file compare.hpp
/// @brief Compare (Equal/Greater/Less/…) and Logical (And/Or/Xor/Not).
///
/// Both broadcast their operands under the NumPy rules and emit u8, the
/// library's bool: 0 or 1 per element. Compare takes two operands of one dtype
/// (f32/f16/s32/s64); a NaN compares equal to nothing and greater/less to
/// nothing, so a NaN input yields 0 for every relation. Logical takes one
/// operand (Not) or two, treats a nonzero byte as true, and accepts s8/u8/s32/
/// s64. Channel-first.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

enum class CompareType : uint8_t {
    Equal,
    Greater,
    Less,
    GreaterOrEqual,
    LessOrEqual,
};

struct CompareAttributes {
    CompareType type = CompareType::Equal;
};

class Compare : public OpBase {
public:
    static std::unique_ptr<Compare> create(const CompareAttributes& attrs,
                                           Backend backend = Backend::CPU);
    static std::unique_ptr<Compare> create(Backend backend = Backend::CPU) {
        return create(CompareAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Compare; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::ChannelFirstOnly;
    }

    struct Impl;
    ~Compare();

private:
    Compare(const CompareAttributes& attrs, Backend backend);
    std::unique_ptr<Impl> impl_;
    CompareAttributes attrs_;
    Backend backend_;
};

enum class LogicalType : uint8_t {
    And,
    Or,
    Xor,
    Not,  ///< one operand
};

struct LogicalAttributes {
    LogicalType type = LogicalType::And;
};

class Logical : public OpBase {
public:
    static std::unique_ptr<Logical> create(const LogicalAttributes& attrs,
                                           Backend backend = Backend::CPU);
    static std::unique_ptr<Logical> create(Backend backend = Backend::CPU) {
        return create(LogicalAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Logical; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::ChannelFirstOnly;
    }

    struct Impl;
    ~Logical();

private:
    Logical(const LogicalAttributes& attrs, Backend backend);
    std::unique_ptr<Impl> impl_;
    LogicalAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
