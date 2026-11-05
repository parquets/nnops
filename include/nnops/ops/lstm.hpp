#pragma once
/// @file lstm.hpp
/// @brief LSTM operator (ONNX LSTM): f32 or f16, with an optional cell output.
///
///   i_t = f(clip(X_t Wi^T + H_{t-1} Ri^T + Wbi + Rbi))
///   o_t = f(clip(X_t Wo^T + H_{t-1} Ro^T + Wbo + Rbo))
///   f_t = f(clip(X_t Wf^T + H_{t-1} Rf^T + Wbf + Rbf)),  or 1 - i_t with
///                                                        `input_forget`
///   c~_t = g(clip(X_t Wc^T + H_{t-1} Rc^T + Wbc + Rbc))
///   C_t = f_t ⊙ C_{t-1} + i_t ⊙ c~_t
///   H_t = o_t ⊙ h(C_t)
///
/// Gate order is ONNX's i, o, f, c. `clip` (0 = off) bounds every gate's
/// pre-activation.
///
/// All four gates of a hidden unit are adjacent columns of the fused gate
/// tile, so one pass over a tile produces C and H without a second sweep; see
/// rnn.hpp for the chunked input projection and the tile epilogue.
///
/// An s8 W and R select the quantized route (W8A8): both products of every gate
/// are int8 MMAs against per-row dynamically quantized activations, dequantized
/// into the compute dtype before the cell update, which is unchanged. The
/// contract — symmetric only, the s8 activation option, the prepacked blob — is
/// rnn.hpp's.
///
/// Inputs (ONNX order, an absent optional is an empty view):
///   0 X [T,B,I]  1 W [D,4H,I]  2 R [D,4H,H]  3 B [D,8H]
///   4 sequence_lens [B] int32  5 initial_h [D,B,H]  6 initial_c [D,B,H]
///   7 P (peepholes) — not implemented; a non-empty view is rejected
/// Outputs: 0 Y [T,D,B,H]  1 Y_h [D,B,H]  2 Y_c [D,B,H].

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

struct LstmAttributes {
    /// Hidden width; 0 infers it from R.
    int64_t hidden_size = 0;

    RnnDirection direction = RnnDirection::Forward;

    /// Gate activations: f for i/o/f, g for the candidate c~, h for the cell.
    RnnActivation activation_f = RnnActivation::Sigmoid;
    RnnActivation activation_g = RnnActivation::Tanh;
    RnnActivation activation_h = RnnActivation::Tanh;
    float alpha_f = 0.0f;
    float beta_f = 0.0f;
    float alpha_g = 0.0f;
    float beta_g = 0.0f;
    float alpha_h = 0.0f;
    float beta_h = 0.0f;

    /// f_t = 1 - i_t instead of its own gate.
    bool input_forget = false;

    /// Clip every gate's pre-activation to [-clip, +clip]; 0 disables it.
    float clip = 0.0f;

    /// 0 = [T, D, B, ·] (all the kernel implements); 1 is rejected.
    int64_t layout = 0;
};

class Lstm : public OpBase {
public:
    static std::unique_ptr<Lstm> create(const LstmAttributes& attrs,
                                        Backend backend = Backend::CPU);

    static std::unique_ptr<Lstm> create(Backend backend = Backend::CPU) {
        return create(LstmAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Lstm; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::ChannelFirstOnly; }

    const LstmAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~Lstm();

private:
    Lstm(const LstmAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    LstmAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
