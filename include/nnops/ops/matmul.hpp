#pragma once
/// @file matmul.hpp
/// @brief MatMul operator: C = A × B.
///
/// C = epilogue(alpha × (A × B) + bias + beta × C_old). The optional third
/// input is a per-column [N] bias; it sits between the product and the
/// activation and is never scaled by alpha. Its dtype follows the output — the
/// output's fp dtype on an fp destination, int32 on an integer one, where it is
/// added to the compensated int32 accumulate before the activation.
///
/// Routes: f32/f16 × f32/f16 (including Q4_1 block weights), s8|u8 × s8 with
/// output_dtype s32/s8/u8 (integral) or f32/f16 (dequantized), and an fp
/// activation carrying QuantParams × s8 (quantized on the fly). N-D batched with
/// numpy-style broadcasting on the leading dims. alpha rides the RHS pack on the
/// fp routes; the integer routes ignore it, since requantization belongs in the
/// output QuantParams.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/epilogue.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

struct MatMulAttributes {
    bool transpose_a = false;
    bool transpose_b = false;

    /// Scale for the A×B product only — bias and beta×C_old stay unscaled.
    /// The fp routes fold it into the RHS pack; the integer routes ignore it
    /// (an int32 accumulate has no scale of its own, and requantization belongs
    /// in the output QuantParams).
    float alpha = 1.0f;

    /// Scale for old C; beta = 0 (default) overwrites C, beta = 1 accumulates.
    float beta = 0.0f;

    /// Integer-path output type: s32 (default) is the raw int32 accumulate
    /// (MatMulInteger semantics); s8/u8 requantize it with the output's
    /// QuantParams (round half away from zero, clamped, f32 product) and take
    /// an int32 bias; f32/f16 dequantize it. Ignored for fp × fp, whose output
    /// follows the input dtype.
    DataType output_dtype = DataType::s32;

    /// Integer storage for the activation quantization on the fp-activation ×
    /// s8 path (s8 default, or u8); only read when is_fp_quant_matmul() holds.
    DataType act_quant_dtype = DataType::s8;

    Epilogue epilogue{};
};

/// True for the integer GEMM — an s8/u8 A against an s8 B (the MatMulInteger
/// operand pairs), whose raw int32 accumulate `output_dtype` returns as s32 or
/// requantizes to s8/u8. The operator and the CPU backend must agree on this
/// pair (the backend fuses exactly the cases whose output dtype the operator
/// resolves), so the test lives here.
inline constexpr bool is_int8_matmul(DataType a, DataType b) noexcept {
    return (a == DataType::s8 || a == DataType::u8) && b == DataType::s8;
}

/// True for the offline-quantized-activation GEMM: a floating-point A carrying
/// active (per-tensor or per-token) QuantParams against an s8 B. The kernel
/// quantizes A to `act_quant_dtype` on the fly, runs the int8 GEMM, and
/// dequantizes the int32 accumulate back to A's dtype. With inactive QuantParams
/// the pair has no defined semantics, so the backend rejects it rather than
/// reading B as floats.
inline bool is_fp_quant_matmul(DataType a, DataType b, const QuantParams& a_qp) noexcept {
    return (a == DataType::f32 || a == DataType::f16) && b == DataType::s8 && a_qp.is_active();
}

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

    /// Pack the B operand once for compute() to read in place — the two-step
    /// contract Conv2D and Linear use (empty outputs[0] = query, filled =
    /// pack, the buffer stays the caller's). inputs[0] is the raw B, in the
    /// order `attributes().transpose_b` says. The blob leads with N
    /// ([N, ⌈elems / N⌉]), which is the number shape inference reads; the panel
    /// geometry follows from N, K and the dtype. f32/f16 only (any other dtype
    /// queries an empty view), and alpha must be 1.
    void prepackWeights(std::span<const TensorView> inputs,
                        std::span<TensorView> outputs,
                        const ComputeContext& ctx = {}) override;

    OpType  getOpType()  const override { return OpType::MatMul; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::PlanarOnly; }

    const MatMulAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~MatMul();

private:
    MatMul(const MatMulAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    MatMulAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
