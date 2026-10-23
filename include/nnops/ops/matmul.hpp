#pragma once
/// @file matmul.hpp
/// @brief MatMul operator: C = A × B  (matrix multiplication).

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/epilogue.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes for matrix multiplication.
///
/// Semantics: C = alpha × (A × B) + beta × bias
/// where bias is an optional third input broadcast over M rows.
struct MatMulAttributes {
    bool transpose_a = false;
    bool transpose_b = false;

    /// Scale factor for the A×B matrix product (fused into pack — zero overhead).
    float alpha = 1.0f;

    /// Scale factor for old C values: C_out = alpha×A×B + bias + beta×C_old.
    /// beta=0 (default) overwrites C; beta=1 adds A×B to existing C.
    float beta = 0.0f;

    /// Output data type for the integer (s8|u8 × s8) matmul path.
    ///
    ///   s32 (default): raw int32 accumulate with zero-point compensation
    ///                  (MatMulInteger semantics).
    ///   s8:            requantize the accumulate to int8.
    ///   f32 / f16:     dequantize the accumulate back to the given float type
    ///                  (valid when A carries active QuantParams).
    ///
    /// Ignored for f32/f16 × f32/f16 inputs, where the output follows the input
    /// dtype (including the fp-activation path, whose output follows A).
    /// alpha/beta are also ignored on the integer path.
    DataType output_dtype = DataType::s32;

    /// Integer storage for the activation quantization on the
    /// fp-activation × s8-weight path: the in-kernel pass writes A's activation
    /// values as `act_quant_dtype` (s8 default, or u8) before the int8 GEMM.
    /// Only read when is_fp_quant_matmul() holds; ignored everywhere else.
    DataType act_quant_dtype = DataType::s8;

    /// Post-processing applied during output write-back (default: identity).
    Epilogue epilogue{};
};

/// True for the integer GEMM: an int8 or uint8 A against an int8 B — the
/// MatMulInteger operand combinations — whose raw int32 accumulate
/// `output_dtype` then either returns as s32 or requantizes to s8. The operator
/// and the CPU backend must agree on this pair (the backend fuses exactly the
/// cases whose output dtype the operator resolves), so it lives here.
inline constexpr bool is_int8_matmul(DataType a, DataType b) noexcept {
    return (a == DataType::s8 || a == DataType::u8) && b == DataType::s8;
}

/// True for the offline-quantized-activation GEMM: a floating-point A carrying
/// active (per-tensor or per-token) quantization parameters against an int8 B.
///
/// The kernel quantizes A into `MatMulAttributes::act_quant_dtype` on the fly,
/// runs the int8 GEMM, and dequantizes the int32 accumulate back to A's dtype.
/// With inactive QuantParams the pair has no defined semantics — a float A is
/// not a quantized tensor to dequantize — so the backend rejects it rather than
/// reading B as floats.
inline bool is_fp_quant_matmul(DataType a, DataType b, const QuantParams& a_qp) noexcept {
    return (a == DataType::f32 || a == DataType::f16) && b == DataType::s8 && a_qp.is_active();
}

/// Matrix multiplication operator (class-based API).
///
/// Supports N-D batch matmul with numpy-style broadcasting:
///   A: [..., M, K]  (or [..., K, M] if transpose_a)
///   B: [..., K, N]  (or [..., N, K] if transpose_b)
///   C: [..., M, N]
///
/// Leading batch dimensions are broadcast-compatible: each dim must be equal,
/// or one of them must be 1 (broadcast to the other), or one tensor may have
/// fewer dimensions (missing dims are treated as 1).
///
/// Examples:
///   - 2D × 2D:    [M,K] × [K,N] → [M,N]
///   - 3D × 3D:    [B,M,K] × [B,K,N] → [B,M,N]
///   - 2D × 3D:    [M,K] × [B,K,N] → [B,M,N]   (broadcast A across batch)
///   - 3D × 2D:    [B,M,K] × [K,N] → [B,M,N]   (broadcast B across batch)
///   - 4D × 3D:    [2,1,M,K] × [3,K,N] → [2,3,M,N]
class MatMul : public OpBase {
public:
    static std::unique_ptr<MatMul> create(const MatMulAttributes& attrs = {},
                                          Backend backend = Backend::CPU);

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    size_t getWorkspaceSize(std::span<const TensorDesc> inputs,
                            std::span<const TensorDesc> outputs) const override;

    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::MatMul; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::PlanarOnly; }

    const MatMulAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in matmul.cpp (Pimpl pattern)
    ~MatMul();

private:
    MatMul(const MatMulAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    MatMulAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
