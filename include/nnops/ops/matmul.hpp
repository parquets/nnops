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
/// Semantics: C = epilogue(alpha × (A × B) + bias + beta × C_old)
/// where bias is an optional third input of shape [N], broadcast over the M
/// rows. It sits between the product and the activation, so `epilogue` sees
/// `product + bias`.
///
/// Every route takes a bias, and its dtype follows the output: the output's own
/// fp dtype on an fp destination (fp × fp, Q4_1 weights, the dequantizing
/// int8 route), int32 on an integer one (output_dtype s32 / s8 / u8). Integer
/// destinations add it to the compensated int32 accumulate — before the
/// activation, and never scaled by alpha, which the integer routes ignore.
struct MatMulAttributes {
    bool transpose_a = false;
    bool transpose_b = false;

    /// Scale factor for the A×B matrix product only — bias and beta×C_old stay
    /// unscaled (see the semantics above). Floating-point routes fold it into
    /// the RHS pack (f32/f16 × f32/f16, Q4_1 weights), so it costs nothing; the
    /// small-M kernels apply it to the accumulator before the beta add.
    ///
    /// The integer routes (output_dtype s32 / s8 / u8) ignore alpha: their
    /// accumulator is int32 and has no scale of its own — requantization
    /// belongs in the output QuantParams. The dequantizing int8 route ignores
    /// it too, for the same reason (its scale is per-tensor/per-channel, not
    /// a single float).
    float alpha = 1.0f;

    /// Scale factor for old C values: C_out = alpha×A×B + bias + beta×C_old.
    /// beta=0 (default) overwrites C; beta=1 adds A×B to existing C.
    float beta = 0.0f;

    /// Output data type for the integer (s8|u8 × s8) matmul path.
    ///
    ///   s32 (default): raw int32 accumulate with zero-point compensation
    ///                  (MatMulInteger semantics).
    ///   s8 / u8:       requantize the accumulate with the output's QuantParams
    ///                  (scale s_a·s_b/out_scale, round half away from zero,
    ///                  clamped to the storage range; the product is f32, as in
    ///                  ONNX Runtime's requantize). The bias is int32.
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
/// `output_dtype` then returns as s32 or requantizes to s8 / u8. The operator
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

    Status validate(std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs) const override;

    size_t getWorkspaceSize(std::span<const TensorDesc> inputs,
                            std::span<const TensorDesc> outputs) const override;

    using OpBase::compute;

    Status compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    /// Pack the B operand once, for compute() to read in place — the two-step
    /// contract Conv2D and Linear use (empty outputs[0] = query, filled = pack,
    /// the buffer stays the caller's).
    ///
    /// inputs[0] is the raw B, in the order `attributes().transpose_b` says.
    /// The blob is indexed only by a prepacked-B compute(): it leads with N
    /// ([N, ⌈elems / N⌉]), which is the number shape inference reads, and the
    /// panel geometry follows from N, K and the dtype. f32/f16 only; any other
    /// dtype queries an empty view. alpha must be 1 for a prepacked product.
    void prepackWeights(std::span<const TensorView> inputs,
                        std::span<TensorView> outputs,
                        const ComputeContext& ctx = {}) override;

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
