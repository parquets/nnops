#pragma once
/// @file op_type.hpp
/// @brief OpType enum identifying different operator types.
///
/// Loosely tracks ONNX op naming, but several entries have no ONNX op behind
/// them (LinearAttention, MoE, Norm, GroupNorm, …). Nothing in
/// the library maps these to ONNX names; the enum is this library's own.

#include <cstdint>

namespace nnops {

/// Operator type identifiers.
enum class OpType : uint8_t {
    Conv2D      = 0,
    Activation  = 1,
    Pooling     = 2,
    Linear      = 3,
    MatMul      = 4,
    MultiHeadAttention = 5,
    Softmax     = 6,
    CumSum      = 7,
    BatchNorm   = 8,
    LayerNorm   = 9,
    RMSNorm     = 10,
    Conv3D          = 11,
    DepthwiseConv = 12,
    Eltwise         = 13,
    Unary           = 14,
    Reduce          = 15,
    Resize          = 16,
    GridSample      = 17,
    LayoutConvert   = 18,
    QuantizeLinear  = 19,
    DequantizeLinear = 20,
    Concat          = 21,
    ArgMax          = 22,
    ArgMin          = 23,
    TopK            = 24,
    RoPE            = 25,
    Flatten         = 26,
    Clamp           = 27,
    Embed           = 28,
    Permute         = 29,
    Slice           = 30,
    GroupNorm       = 31,
    TransposeConv2D = 32,
    LinearAttention = 33,
    Norm            = 34,
    MoE             = 35,
    Rnn             = 36,
    Gru             = 37,
    Lstm            = 38,
    Pad             = 39,
    Gather          = 40,
    GatherElements  = 41,
    GatherND        = 42,
    ScatterElements = 43,
    ScatterND       = 44,
    Split           = 45,
    Tile            = 46,
    Expand          = 47,
    Where           = 48,
    Compare         = 49,
    Logical         = 50,
    GroupQueryAttention = 51,
    MultiLatentAttention = 52,
    DepthwiseTransposeConv2D = 53,
    Conv1D          = 54,
    Pool1D          = 56,
};

}  // namespace nnops
