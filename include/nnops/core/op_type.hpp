#pragma once
/// @file op_type.hpp
/// @brief OpType enum identifying different operator types (aligned with ONNX op types).

#include <cstdint>

namespace nnops {

/// Operator type identifiers (corresponds to ONNX OpType).
enum class OpType : uint8_t {
    Conv2D      = 0,
    Activation  = 1,
    Pooling     = 2,
    Linear      = 3,
    MatMul      = 4,
    Attention   = 5,
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
    CausalAttention = 27,
    Clamp           = 28,
    Embed           = 29,
    Permute         = 30,
    Slice           = 31,
};

}  // namespace nnops
