#pragma once
/// @file gru.hpp
/// @brief GRU operator (ONNX GRU): gated recurrent unit, f32 or f16.
///
///   z_t = f(clip(X_t Wz^T + H_{t-1} Rz^T + Wbz + Rbz))
///   r_t = f(clip(X_t Wr^T + H_{t-1} Rr^T + Wbr + Rbr))
///   h~_t = g(clip(X_t Wh^T + Wbh
///                 + ((r_t ⊙ H_{t-1}) Rh^T + Rbh      linear_before_reset = 0
///                 +  r_t ⊙ (H_{t-1} Rh^T + Rbh))))   linear_before_reset ≠ 0
///   H_t = (1 - z_t) ⊙ h~_t + z_t ⊙ H_{t-1}
///
/// The two readings differ only in where the per-hidden reset multiply sits, so
/// they cost the same. `clip` (0 = off) bounds the pre-activation of f and g.
///
/// The layer needs r before the h-gate's recurrent product, so a timestep runs
/// in two phases: the z and r gates (one GEMM, one fused epilogue that writes
/// z and r⊙H_{t-1} or r), then the h gate keyed off that buffer. Both phases
/// ride the same interleaved gate layout and the same tile epilogue as RNN and
/// LSTM; see rnn.hpp for the chunked input projection.
///
/// Inputs (ONNX order, an absent optional is an empty view):
///   0 X [T,B,I]  1 W [D,3H,I]  2 R [D,3H,H]  3 B [D,6H]
///   4 sequence_lens [B] int32  5 initial_h [D,B,H]
/// Outputs: 0 Y [T,D,B,H]  1 Y_h [D,B,H].

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

struct GruAttributes {
    /// Hidden width; 0 infers it from R.
    int64_t hidden_size = 0;

    RnnDirection direction = RnnDirection::Forward;

    /// Gate activations: f for z and r, g for the candidate h~.
    RnnActivation activation_f = RnnActivation::Sigmoid;
    RnnActivation activation_g = RnnActivation::Tanh;
    float alpha_f = 0.0f;
    float beta_f = 0.0f;
    float alpha_g = 0.0f;
    float beta_g = 0.0f;

    /// The h-gate reading above; both cost the same here.
    bool linear_before_reset = false;

    /// Clip the pre-activation to [-clip, +clip]; 0 disables it.
    float clip = 0.0f;

    /// 0 = [T, D, B, ·] (all the kernel implements); 1 is rejected.
    int64_t layout = 0;
};

class Gru : public OpBase {
public:
    static std::unique_ptr<Gru> create(const GruAttributes& attrs,
                                       Backend backend = Backend::CPU);

    static std::unique_ptr<Gru> create(Backend backend = Backend::CPU) {
        return create(GruAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Gru; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::PlanarOnly; }

    const GruAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~Gru();

private:
    Gru(const GruAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    GruAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
