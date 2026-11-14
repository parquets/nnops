#pragma once
/// @file cast.hpp
/// @brief Element-wise dtype conversion (ONNX Cast).
///
/// output[i] = (to)input[i]; shape is preserved and the target dtype comes from
/// the attributes, so the operator can describe its own output.
///
///   f32 / f16 / bf16  <->  f32 / f16 / bf16
///   f32 / f16 / bf16  <->  s32 / s64 / s8 / u8
///   s32 / s64 / s8 / u8 <-> s32 / s64 / s8 / u8
///
/// Rules, matching onnxruntime:
///   - float -> integer truncates toward zero and saturates to the destination
///     range; NaN maps to 0. (ONNX leaves both undefined; a fixed answer keeps
///     a graph reproducible instead of depending on the host's cvttsd2si.)
///   - integer -> float rounds to nearest as the hardware converts.
///   - integer -> integer goes through int64, so an index wider than f32's
///     24-bit mantissa survives; float -> integer clamps at ±2^63 before the
///     truncation, so no conversion is undefined.
///   - float -> float rounds to nearest even in the destination format.
///
/// bf16 follows the backend gate: an x86-64 CPU build has no bf16 kernel, so a
/// bf16 operand there returns Status::unsupported (see dtype_in).

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

struct CastAttributes {
    /// Destination dtype. The caller still allocates the output with this dtype;
    /// validate() rejects a mismatch rather than reinterpreting the buffer.
    DataType to = DataType::f32;
};

class Cast : public OpBase {
public:
    static std::unique_ptr<Cast> create(const CastAttributes& attrs,
                                        Backend backend = Backend::CPU);

    static std::unique_ptr<Cast> create(Backend backend = Backend::CPU) {
        return create(CastAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Cast; }
    Backend getBackend() const override { return backend_; }

    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::Any;
    }

    const CastAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in cast.cpp
    ~Cast();

private:
    Cast(const CastAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    CastAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
