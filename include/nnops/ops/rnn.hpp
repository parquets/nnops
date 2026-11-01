#pragma once
/// @file rnn.hpp
/// @brief RNN operator (ONNX RNN): one vanilla recurrent layer, f32 or f16.
///
///   H_t = activation(clip(X_t W^T + H_{t-1} R^T + Wb + Rb))
///
/// with X_t [B, I], W [G*H = H, I], R [H, H] and `clip` applied to the
/// pre-activation input of the activation (0 = no clipping). Both weights are
/// packed into the GEMM's uniform RHS panels once per compute().
///
/// The layer is laid out to keep the matmul hot. The input projection
/// X[0..T] × W^T runs as ONE GEMM per chunk of timesteps — M = chunk·B, so its
/// arithmetic intensity is the GEMM's, not a GEMV's — and each timestep then
/// adds H_{t-1} × R^T straight into that buffer, with the activation, the
/// sequence-length mask and the Y write fused into the tile epilogue. So a
/// timestep costs one accumulate pass over [B, H] and never materializes a
/// separate gate buffer.
///
/// `direction = Bidirectional` runs the second direction over the timesteps in
/// reverse; Y still holds time t at index t.
///
/// Inputs (ONNX order, an absent optional is an empty view):
///   0 X [T,B,I]  1 W [D,H,I]  2 R [D,H,H]  3 B [D,2H]
///   4 sequence_lens [B] int32  5 initial_h [D,B,H]
/// Outputs: 0 Y [T,D,B,H]  1 Y_h [D,B,H]. Rows whose sequence_lens is 0 (or
/// shorter than t) keep their state and write Y = 0.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/ops/recurrent.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace nnops {

struct RnnAttributes {
    /// Hidden width; 0 infers it from R.
    int64_t hidden_size = 0;

    RnnDirection direction = RnnDirection::Forward;

    /// The single gate's activation.
    RnnActivation activation = RnnActivation::Tanh;
    float activation_alpha = 0.0f;
    float activation_beta = 0.0f;

    /// Clip the pre-activation to [-clip, +clip]; 0 disables it.
    float clip = 0.0f;

    /// 0 = [T, D, B, ·] (all the kernel implements); 1 is rejected.
    int64_t layout = 0;
};

class Rnn : public OpBase {
public:
    static std::unique_ptr<Rnn> create(const RnnAttributes& attrs,
                                       Backend backend = Backend::CPU);

    static std::unique_ptr<Rnn> create(Backend backend = Backend::CPU) {
        return create(RnnAttributes{}, backend);
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

    /// The recurrent family's prepacked weights, two-step like Conv2D's and
    /// MatMul's: an empty outputs[0] queries the blob's descriptor, a
    /// preallocated one receives W and R interleaved and packed, plus this
    /// call's bias when inputs[3] is present. The blob is then the whole
    /// weight — a prepacked compute() adds the bias it was packed with, not
    /// the one in its own inputs. inputs are this op's own list; weights a
    /// kernel cannot pack leave the query empty.
    void prepackWeights(std::span<const TensorView> inputs,
                        std::span<TensorView> outputs,
                        const ComputeContext& ctx = {}) override;

    OpType  getOpType()  const override { return OpType::Rnn; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::PlanarOnly; }

    const RnnAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~Rnn();

private:
    Rnn(const RnnAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    RnnAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
