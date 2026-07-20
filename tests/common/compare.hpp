#pragma once
/// @file compare.hpp
/// @brief Tensor comparison utilities for testing.

#include "nnops/core/tensor_view.hpp"

#include <cmath>
#include <cstdint>
#include <string>

namespace nnops::test {

/// Compare two float tensor views element-wise with tolerance.
/// Returns true if all elements match within the given tolerances.
inline bool allclose(const TensorView& a, const TensorView& b,
                     float rtol = 1e-4f, float atol = 1e-6f)
{
    if (a.numel() != b.numel()) return false;
    if (a.data_type() != b.data_type()) return false;

    const int64_t N = a.numel();
    const auto* ap = a.data_as<float>();
    const auto* bp = b.data_as<float>();

    for (int64_t i = 0; i < N; ++i) {
        float diff = std::abs(ap[i] - bp[i]);
        float threshold = atol + rtol * std::max(std::abs(ap[i]), std::abs(bp[i]));
        if (diff > threshold) {
            return false;
        }
    }
    return true;
}

/// Returns the maximum absolute difference between two tensors.
inline float max_diff(const TensorView& a, const TensorView& b)
{
    if (a.numel() != b.numel()) return std::numeric_limits<float>::infinity();
    if (a.data_type() != b.data_type()) return std::numeric_limits<float>::infinity();

    const int64_t N = a.numel();
    const auto* ap = a.data_as<float>();
    const auto* bp = b.data_as<float>();

    float max_d = 0.0f;
    for (int64_t i = 0; i < N; ++i) {
        float diff = std::abs(ap[i] - bp[i]);
        if (diff > max_d) max_d = diff;
    }
    return max_d;
}

}  // namespace nnops::test
