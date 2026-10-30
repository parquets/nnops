#pragma once
/// @file linear.hpp
/// @brief Linear (fully-connected) operator: output = input × weight^T + bias.
///
///   f32 × f32, f16 × f16             → output follows the input dtype
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
/// preserved. Bias, epilogue and add_to ride the output write-back. The
/// optional [N] bias follows the output dtype — int32 on an integer
/// destination.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/epilogue.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

struct LinearAttributes {
    Epilogue epilogue{};

    /// Accumulate into the existing output instead of overwriting (residual).
    bool add_to = false;

    /// Quantized-path output type: f32/f16 dequantize, s32/s8/u8 stay integral
    /// (s8/u8 requantize with the output's QuantParams). Anything else selects
    /// f32. Ignored for fp operands, whose output follows the input dtype.
    ///
    /// An integer output takes its bias as int32; see MatMulAttributes.
    DataType output_dtype = DataType::f32;
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
    /// weight. The blob is the GEMM's uniform nr-wide panels over the whole K,
    /// which is what removes the per-call RHS pack — and why alpha must be 1
    /// (a prepacked B cannot carry a per-call scale). f32/f16 only; a quantized
    /// weight has no packed form (its panels would need the int32 column sums)
    /// and the query returns an empty view for it.
    void prepackWeights(std::span<const TensorView> inputs,
                        std::span<TensorView> outputs,
                        const ComputeContext& ctx = {}) override;

    OpType  getOpType()  const override { return OpType::Linear; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::PlanarOnly; }

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
