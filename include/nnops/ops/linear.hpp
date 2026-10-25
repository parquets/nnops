#pragma once
/// @file linear.hpp
/// @brief Linear (fully-connected) operator: output = input × weight^T + bias.
///
/// Supported operand combinations:
///   - f32 × f32, f16 × f16                 → output follows the input dtype
///   - u8  × s8,  s8 × s8 (quantized)       → output is always f32 or f16,
///                                             selected by `output_dtype`
///
/// The quantized path dequantizes each int32 accumulate with the QuantParams
/// carried by the two input TensorViews: X's per-tensor / per-token parameters
/// and W's per-tensor / per-channel parameters. With inactive parameters the
/// scales are 1 and the zero points 0, i.e. a plain integer product.
///
/// The weight is stored [N, K] and the multiplication is X × W^T. The optional
/// third input is a per-output-channel bias of shape [N] in the output dtype.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/epilogue.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes for the Linear (fully-connected) operator.
struct LinearAttributes {
    /// Post-processing applied during output write-back (default: identity).
    Epilogue epilogue{};

    /// If true, add result to existing output buffer instead of overwriting.
    bool add_to = false;

    /// Output data type for the quantized (u8|s8 × s8) path. The output is
    /// always floating point: f16 selects the half output, anything else
    /// (including the default f32) selects float.
    ///
    /// Ignored for f32/f16 operands, whose output always follows the input
    /// dtype (fp in, fp out).
    DataType output_dtype = DataType::f32;
};

/// Linear / fully-connected operator (class-based API).
///
/// Computes: output = epilogue(input × weight^T + bias)
///   input:  [..., K]     (f32 | f16 | u8 | s8)
///   weight: [N, K]       (f32 | f16, or s8 for the quantized path)
///   bias:   [N] (optional, output dtype)
///   output: [..., N]     (leading dims preserved; only the last becomes N)
///
/// Transposed weight is implicit — the weight is stored as [N, K] and
/// the multiplication is input × weight^T. Internally this drives the MatMul
/// kernel with `transpose_b`; bias, the epilogue and add_to are applied on the
/// output write-back.
class Linear : public OpBase {
public:
    static std::unique_ptr<Linear> create(const LinearAttributes& attrs,
                                          Backend backend = Backend::CPU);

    static std::unique_ptr<Linear> create(Backend backend = Backend::CPU) {
        return create(LinearAttributes{}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Linear; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::PlanarOnly; }

    const LinearAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in linear.cpp (Pimpl pattern)
    ~Linear();

private:
    Linear(const LinearAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    LinearAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
