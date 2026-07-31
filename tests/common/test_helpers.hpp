/// @file test_helpers.hpp
/// @brief MSVC-compatible helpers for constructing TensorViews from TensorDescs
/// and invoking operator APIs with explicit C arrays (MSVC lacks init-list → span).
#pragma once

#include "nnops/core/tensor_view.hpp"
#include <span>
#include <vector>

namespace nnops {
namespace test {

/// Create a planar TensorView from a TensorDesc.
inline TensorView make_planar(const TensorDesc& desc, void* data) {
    auto shape = std::span<const int64_t>(desc.dims.data(), static_cast<size_t>(desc.rank));
    return TensorView(shape, desc.dtype, data, desc.layout);
}

/// Create a packed TensorView from a TensorDesc (uses row_pitch for the pitch parameter).
inline TensorView make_packed(const TensorDesc& desc, void* data, int64_t alignment = 32) {
    auto shape = std::span<const int64_t>(desc.dims.data(), static_cast<size_t>(desc.rank));
    return TensorView(shape, desc.dtype, data, desc.row_pitch(alignment), desc.layout);
}

}  // namespace test
}  // namespace nnops
