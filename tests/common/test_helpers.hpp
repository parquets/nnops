/// @file test_helpers.hpp
/// @brief MSVC-compatible helpers for constructing TensorViews from TensorDescs
/// and invoking operator APIs with explicit C arrays (MSVC lacks init-list → span).
#pragma once

#include "nnops/core/tensor_view.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "random_tensor.hpp"
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

// ============================================================
// f16 test helpers
// ============================================================

/// Convert f32 vector to f16 (half) vector.
inline std::vector<nnops::backend::cpu::half> f32_to_f16(const std::vector<float>& src) {
    std::vector<nnops::backend::cpu::half> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        simd::s_store(&dst[i], src[i]);
    }
    return dst;
}

/// Convert f16 (half) vector to f32 vector.
inline std::vector<float> f16_to_f32(const std::vector<nnops::backend::cpu::half>& src) {
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = simd::s_load(&src[i]);
    }
    return dst;
}

/// Create a random f16 tensor. Returns (f16_buffer, TensorView) pair.
inline std::pair<std::vector<nnops::backend::cpu::half>, TensorView>
make_random_f16_tensor(std::span<const int64_t> shape,
                        float min = -1.0f, float max = 1.0f,
                        uint64_t seed = 12345)
{
    // Generate f32 random data first, then convert
    auto [f32_buf, f32_view] = make_random_tensor(shape, min, max, seed);
    auto f16_buf = f32_to_f16(f32_buf);
    TensorView view(shape, DataType::f16, f16_buf.data(), f32_view.layout());
    return {std::move(f16_buf), view};
}

/// Overload for initializer_list convenience.
inline std::pair<std::vector<nnops::backend::cpu::half>, TensorView>
make_random_f16_tensor(std::initializer_list<int64_t> shape,
                        float min = -1.0f, float max = 1.0f,
                        uint64_t seed = 12345)
{
    return make_random_f16_tensor(std::span<const int64_t>(shape.begin(), shape.size()),
                                   min, max, seed);
}

}  // namespace test
}  // namespace nnops
