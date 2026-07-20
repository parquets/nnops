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
    Conv3D      = 11,
};

}  // namespace nnops
