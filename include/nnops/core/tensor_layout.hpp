#pragma once
/// @file tensor_layout.hpp
/// @brief TensorLayout enum for describing data arrangement in memory.

#include <cstdint>

namespace nnops {

/// Memory layout of tensor data.
enum class TensorLayout : uint8_t {
    NCHW    = 0,  ///< Standard: [batch, channels, height, width]
    NHWC    = 1,  ///< Channels-last: [batch, height, width, channels]
    NCHWC4  = 2,  ///< Pack 4 channels into inner dimension
    NCHWC8  = 3,  ///< Pack 8 channels into inner dimension
    NCHWC16 = 4,  ///< Pack 16 channels into inner dimension
    NCHWC32 = 5,  ///< Pack 32 channels into inner dimension
};

/// Returns the channel block size for blocked layouts, or 1 for plain layouts.
constexpr int64_t layout_channel_block(TensorLayout layout) noexcept {
    switch (layout) {
    case TensorLayout::NCHW:    return 1;
    case TensorLayout::NHWC:    return 1;
    case TensorLayout::NCHWC4:  return 4;
    case TensorLayout::NCHWC8:  return 8;
    case TensorLayout::NCHWC16: return 16;
    case TensorLayout::NCHWC32: return 32;
    }
    return 1;
}

}  // namespace nnops
