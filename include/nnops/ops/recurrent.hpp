#pragma once
/// @file recurrent.hpp
/// @brief Shared vocabulary of the recurrent family — RNN, GRU and LSTM.
///
/// The three operators share one kernel: they differ only in how many gates a
/// hidden unit has and how those gates combine. Shapes are ONNX's, seq-major
/// (the `layout = 0` reading):
///
///   X   [T, B, I]         one [B, I] matrix per timestep
///   W   [D, G*H, I]       gate 0 .. G-1 of every hidden unit, in ONNX order
///   R   [D, G*H, H]
///   B   [D, 2*G*H]        Wb[G*H] then Rb[G*H] — added, not concatenated
///   Y   [T, D, B, H],  Y_h [D, B, H],  Y_c [D, B, H] (LSTM only)
///
/// with D = num_directions (1 or 2) and G = recurrent_gates(mode) = 1 / 3 / 4.
/// Gate order per mode is ONNX's: RNN has the one activation gate; GRU is
/// z, r, h; LSTM is i, o, f, c.

#include <cstdint>

namespace nnops {

/// ONNX `direction`.
enum class RnnDirection : uint8_t {
    Forward       = 0,
    Reverse       = 1,
    Bidirectional = 2,
};

/// Per-gate activation (`activations`). ONNX also lists Affine,
/// ThresholdedRelu, ScaledTanh, HardSigmoid, Softsign and Softplus; those have
/// no kernel here and are rejected at validate().
enum class RnnActivation : uint8_t {
    Sigmoid   = 0,
    Tanh      = 1,
    Relu      = 2,
    LeakyRelu = 3,
    Elu       = 4,
};

/// 1 for a unidirectional layer, 2 for a bidirectional one.
inline constexpr int64_t rnn_num_directions(RnnDirection direction) noexcept {
    return direction == RnnDirection::Bidirectional ? 2 : 1;
}

/// True when @p a has an activation kernel.
inline constexpr bool rnn_activation_supported(RnnActivation a) noexcept {
    return static_cast<uint8_t>(a) <= static_cast<uint8_t>(RnnActivation::Elu);
}

/// The three operators, which is really "how many gates per hidden unit".
enum class RecurrentMode : uint8_t {
    Rnn  = 0,
    Gru  = 1,
    Lstm = 2,
};

/// Gates per hidden unit: the gate axis of W and R.
inline constexpr int64_t recurrent_gates(RecurrentMode mode) noexcept {
    return mode == RecurrentMode::Rnn ? 1 : (mode == RecurrentMode::Gru ? 3 : 4);
}

}  // namespace nnops
