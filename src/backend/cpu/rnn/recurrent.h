#pragma once
/// @file recurrent.h
/// @brief The CPU recurrent kernel — RNN, GRU and LSTM behind one driver.
///
/// The three operators share everything but the gate arithmetic, and the shared
/// part is where the time goes:
///
///  - The input projection X[0..T] × W^T runs as one GEMM per chunk of
///    timesteps (M = chunk·B), so its arithmetic intensity is the GEMM's.
///  - Each timestep then adds H_{t-1} × R^T into that buffer with the packed R
///    panels, and the tile epilogue that follows fuses the activation, the cell
///    update, the sequence-length mask and the Y write. A timestep therefore
///    costs one accumulate pass over [B, gates·H] and never materializes a gate
///    buffer that a second pass would have to re-read.
///  - Both weights are prepacked into the GEMM's uniform nr-wide RHS panels;
///    the gate layout is interleaved so that one SIMD vector holds eight
///    hidden units of one gate, which is what makes the epilogue branch-free.
///    That work is per call unless the caller runs prepackWeights() once and
///    passes the blob back as W — see recurrent_pack_weights below, and note
///    that the fixed cost is most of a one-step call.
///
/// Quantized weights (W8A8, `weight_dtype == s8`). Both products run on the
/// int8 MMA: A is quantized per row and always normalised to the signed
/// convention, B is the symmetric s8 weight, and the int32 accumulate is
/// dequantized back to `dtype` before the epilogue — so the gate arithmetic,
/// the cell update and the outputs are untouched, and the only numbers that
/// carry scales are the two products. Symmetric only (zero_point 0): the
/// dequant is one multiply per element, with no zero-point correction terms.
///
///   proj[m][n] = C_w[m][n] · s_x[m] · s_w[n]
///   rec [m][n] = C_r[m][n] · s_h[m] · s_r[n]
///
/// s_x and s_h are dynamic (per row, from that row's max) when the activation
/// arrives in fp, and come from the tensor's QuantParams when it arrives as s8.
/// s_w and s_r are per gate row, given per row of W and R and interleaved with
/// them. The weight panels are per k-block slabs, since the int8 MMA takes no
/// RHS stride: a blob packed once is readable by any batch, sequence or team.
///
/// RecurrentParams is filled by the operator's dispatch unit (which owns the
/// ONNX attribute and shape rules); this header carries no attribute includes.
///
/// Inputs, ONNX order (an absent optional is an empty view):
///   0 X [T,B,I]  1 W [D,G*H,I]  2 R [D,G*H,H]  3 B [D,2*G*H]
///   4 sequence_lens [B] int32  5 initial_h [D,B,H]  6 initial_c [D,B,H]
/// Outputs: 0 Y [T,D,B,H]  1 Y_h [D,B,H]  2 Y_c [D,B,H]

#include "nnops/core/compute_context.hpp"
#include "nnops/core/data_type.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/status.hpp"
#include "nnops/ops/recurrent.hpp"

#include <span>

namespace nnops::backend::cpu {

/// A recurrent layer fully resolved: no attributes, no inference, just the
/// numbers the kernel needs. The dispatch unit guarantees the inputs match.
struct RecurrentParams {
    RecurrentMode mode = RecurrentMode::Rnn;

    int64_t seq = 0;         ///< T
    int64_t batch = 0;       ///< B
    int64_t input = 0;       ///< I
    int64_t hidden = 0;      ///< H
    int64_t directions = 1;  ///< D
    /// Only one direction is ambiguous about its timeline: with D = 2 the first
    /// is forward and the second reverse by ONNX's rule, but with D = 1 this is
    /// the answer.
    RnnDirection direction = RnnDirection::Forward;
    /// The compute dtype: the gate buffer, the cell and hidden state and the
    /// outputs. f32 or f16.
    DataType dtype = DataType::f32;
    /// W's and R's storage. s8 selects the quantized route, whose activations
    /// are still `dtype` (see the header note); anything else must equal
    /// `dtype`.
    DataType weight_dtype = DataType::f32;

    RnnActivation act_f = RnnActivation::Sigmoid;
    RnnActivation act_g = RnnActivation::Tanh;
    RnnActivation act_h = RnnActivation::Tanh;
    float alpha_f = 0.0f, beta_f = 0.0f;
    float alpha_g = 0.0f, beta_g = 0.0f;
    float alpha_h = 0.0f, beta_h = 0.0f;

    /// Bounds every gate's pre-activation; 0 disables it.
    float clip = 0.0f;

    bool input_forget = false;        ///< LSTM: f = 1 - i
    bool linear_before_reset = false; ///< GRU: see gru.hpp
};

Status recurrent_cpu(const RecurrentParams& params,
                     std::span<TensorView> outputs,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx);

/// Element count of the blob recurrent_pack_weights() produces for these
/// parameters — what a prepack query answers with. Independent of the batch, the
/// sequence length and the worker count, which is why it can be built once.
int64_t recurrent_packed_numel(const RecurrentParams& params);

/// Pack @p w and @p r (and @p bias, when present) for every direction into
/// @p dst, which must hold recurrent_packed_numel() elements. A later
/// recurrent_cpu() handed the blob as its W skips the interleave and both pack
/// passes and reads them in place.
Status recurrent_pack_weights(const RecurrentParams& params,
                              const TensorView& w, const TensorView& r,
                              const TensorView* bias, TensorView& dst,
                              const ComputeContext& ctx);

}  // namespace nnops::backend::cpu
