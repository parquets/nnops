#pragma once
/// @file linear.hpp
/// @brief Linear (fully-connected) operator: output = input × weight^T + bias.
///
///   f32 × f32, f16 × f16             → output follows the input dtype
///   f32 : q4_1, f16 : q4_1           → the same, with a block-packed weight
///                                      (32 values per 20-byte block, decoded
///                                      inside the RHS pack or the fused kernel)
///   u8 × s8, s8 × s8 (quantized)     → output follows `output_dtype`:
///                                      f32/f16 dequantizes, s32/s8/u8 stays
///                                      integral
///
/// The quantized path folds the QuantParams carried by the two input
/// TensorViews into each int32 accumulate — X's per-tensor / per-token, W's
/// per-tensor / per-channel. Inactive parameters are scale 1 and zero point 0,
/// i.e. a plain integer product. An s8/u8 output requantizes with the output's
/// QuantParams, clamped to the destination's storage range.
///
/// The weight is [N, K] and the product is X × W^T, i.e. the MatMul kernel with
/// `transpose_b`; the input is [..., K] and the output [..., N], leading dims
/// preserved. Bias, epilogue and add_to ride the output write-back; the
/// residual is added after the epilogue unless `residual_pre_activation` folds
/// it into the sum the epilogue sees. The optional [N] bias follows the output
/// dtype — int32 on an integer destination. With `gated`, the weight carries
/// the gate and value halves ([2N, K]) and the output is the SwiGLU combine of
/// the two.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/epilogue.hpp"
#include <cstdint>
#include <limits>
#include <memory>
#include <span>

namespace nnops {

struct LinearAttributes {
    Epilogue epilogue{};

    /// Accumulate into the existing output instead of overwriting (residual).
    bool add_to = false;

    /// Order of the add_to term relative to the epilogue:
    ///   false: y = epilogue(x·wᵀ + bias) + residual
    ///   true:  y = epilogue(x·wᵀ + bias + residual)
    /// The pre-activation form is the ResNet block's order: the residual joins
    /// the sum MatMul's clamp and activation see, so it folds into the GEMM call
    /// even with an active epilogue, on every CPU route — fp, q4_1 and int8. An
    /// identity epilogue folds either way. Only meaningful with add_to; the
    /// gated combine and the CUDA kernel refuse it rather than apply the
    /// residual on the wrong side.
    bool residual_pre_activation = false;

    /// Quantized-path output type: f32/f16 dequantize, s32/s8/u8 stay integral
    /// (s8/u8 requantize with the output's QuantParams). Anything else selects
    /// f32. Ignored for fp operands, whose output follows the input dtype.
    ///
    /// An integer output takes its bias as int32; see MatMulAttributes.
    DataType output_dtype = DataType::f32;

    // ---- gated (SwiGLU) form ---------------------------------------------
    // The weight is [2N, K] and the bias [2N]: rows [0, N) are the gate, rows
    // [N, 2N) the value, and the output is [..., N]:
    //
    //   y = epilogue( gate · sigmoid(alpha·gate) · (value + beta) )
    //
    // with both halves clamped to ±limit when limit is finite. alpha 1, beta 0,
    // limit +inf is standard SwiGLU; GPT-OSS uses 1.702 / 1.0 / 7.0. This is
    // the same combine the MoE operator applies inside an expert, so a dense
    // FFN's gate/up pair costs one GEMM and one output pass instead of a GEMM,
    // a split and two elementwise passes. fp operands only.
    bool gated = false;
    float gate_alpha = 1.0f;
    float gate_beta = 0.0f;
    float gate_limit = std::numeric_limits<float>::infinity();
};

class Linear : public OpBase {
public:
    static std::unique_ptr<Linear> create(const LinearAttributes& attrs,
                                          Backend backend = Backend::CPU);

    static std::unique_ptr<Linear> create(Backend backend = Backend::CPU) {
        return create(LinearAttributes{}, backend);
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

    /// Pack the weight once for compute() to read in place — the same two-step
    /// contract as Conv2D: an empty outputs[0] is the query, a filled one the
    /// pack, and the buffer stays the caller's. inputs[0] is the raw [N, K]
    /// weight. For f32/f16 the blob is the GEMM's uniform nr-wide panels over
    /// the whole K, which is what removes the per-call RHS pack — and why alpha
    /// must be 1 (a prepacked B cannot carry a per-call scale). An s8 weight is
    /// packed into the same panels cut into ⌈K/512⌉ k segments, read by an s8
    /// activation only (alpha is ignored there). Any other dtype — u8 and the
    /// block-packed q4_1 among them — has no packed form and the query returns
    /// an empty view: a q4_1 weight is already its own compact form, and the
    /// only blob that would remove its per-call decode is a dense one, which
    /// would have to name the activation type it is decoded for.
    void prepackWeights(std::span<const TensorView> inputs,
                        std::span<TensorView> outputs,
                        const ComputeContext& ctx = {}) override;

    OpType  getOpType()  const override { return OpType::Linear; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::ChannelFirstOnly; }

    const LinearAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~Linear();

private:
    Linear(const LinearAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    LinearAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
