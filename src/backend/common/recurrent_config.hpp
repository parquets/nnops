#pragma once
/// @file recurrent_config.hpp
/// @brief A recurrent layer fully resolved — the description the CPU driver and
///        the CUDA timeline both read.
///
/// RNN, GRU and LSTM differ only in gates and activation; the operator front end
/// (src/ops/recurrent_common.hpp) reduces ONNX's attributes and shapes to this
/// block, and neither backend's kernel knows which operator called it. The
/// per-backend headers add their own entry points: the CPU header also declares
/// the packed-weight blob, which only the CPU packer and its reader share.

#include "nnops/core/data_type.hpp"
#include "nnops/ops/recurrent.hpp"

#include <cstdint>

namespace nnops::backend {

/// A recurrent layer fully resolved: no attributes, no inference, just the
/// numbers the kernel needs. The dispatch unit guarantees the inputs match.
struct RecurrentParams {
    RecurrentMode mode = RecurrentMode::Rnn;

    int64_t seq = 0;        ///< T
    int64_t batch = 0;      ///< B
    int64_t input = 0;      ///< I
    int64_t hidden = 0;     ///< H
    int64_t directions = 1; ///< D
    /// Only one direction is ambiguous about its timeline: with D = 2 the first
    /// is forward and the second reverse by ONNX's rule, but with D = 1 this is
    /// the answer.
    RnnDirection direction = RnnDirection::Forward;
    /// The compute dtype: the gate buffer, the cell and hidden state and the
    /// outputs. f32 or f16.
    DataType dtype = DataType::f32;
    /// W's and R's storage. s8 selects the quantized route, whose activations
    /// are still `dtype` (see the CPU header's note); anything else must equal
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

} // namespace nnops::backend
