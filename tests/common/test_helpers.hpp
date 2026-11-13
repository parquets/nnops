/// @file test_helpers.hpp
/// @brief MSVC-compatible helpers for constructing TensorViews from TensorDescs
/// and invoking operator APIs with explicit C arrays (MSVC lacks init-list → span).
///
/// kF16AccumTol is arch-specific because the f16 GEMM micro-kernels accumulate
/// in different precisions: x86_64 widens to fp32 (F16C), adds and narrows back,
/// so the error is ~fp32, while aarch64 keeps native f16 accumulators
/// (vfmaq_laneq_f16) and the error grows with K to ~0.09 at K=256 (≈3-4 f16 ulp).
/// The x86 value would be far too tight on aarch64 past a few dozen K. Shared by
/// the f16 GEMM-family tests (matmul, conv2d, attention), which all compare an f16
/// op against the f32 reference computed on the same input data.
#pragma once

#include "nnops/core/tensor_view.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "random_tensor.hpp"
#include <span>
#include <vector>

namespace nnops {
namespace test {

/// Create a channel-first TensorView from a TensorDesc.
inline TensorView make_channel_first(const TensorDesc& desc, void* data) {
    auto shape = std::span<const int64_t>(desc.dims.data(), static_cast<size_t>(desc.rank));
    return TensorView(shape, desc.dtype, data, desc.layout);
}

/// Create a pitched TensorView from a TensorDesc (uses row_pitch for the pitch
/// parameter); needed by channel-last tests, whose row is one spatial line.
inline TensorView make_packed(const TensorDesc& desc, void* data, int64_t alignment = 32) {
    auto shape = std::span<const int64_t>(desc.dims.data(), static_cast<size_t>(desc.rank));
    return TensorView(shape, desc.dtype, data, desc.row_pitch(alignment), desc.layout);
}

// ============================================================
// f16 accumulation tolerance
// ============================================================
#if defined(NNOPS_ARCH_AARCH64)
inline constexpr float kF16AccumTol = 2e-1f;
#else
inline constexpr float kF16AccumTol = 5e-2f;
#endif

// ============================================================
// f16 test helpers
// ============================================================

inline std::vector<nnops::backend::cpu::half> f32_to_f16(const std::vector<float>& src) {
    std::vector<nnops::backend::cpu::half> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        simd::s_store(&dst[i], src[i]);
    }
    return dst;
}

inline std::vector<float> f16_to_f32(const std::vector<nnops::backend::cpu::half>& src) {
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = simd::s_load(&src[i]);
    }
    return dst;
}

/// Returns (buffer, view); the buffer owns the data.
inline std::pair<std::vector<nnops::backend::cpu::half>, TensorView>
make_random_f16_tensor(std::span<const int64_t> shape,
                        float min = -1.0f, float max = 1.0f,
                        uint64_t seed = 12345)
{
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

// ============================================================
// bf16 accumulation tolerance
// ============================================================
// The aarch64 bf16 GEMM accumulates in f32 inside a tile (BFMMLA) but the
// kc-blocked loop carries partial sums through C, which is bf16, so the error
// grows with K the same way the f16 kernels' does — only against a 8-bit
// mantissa instead of 10, so the same shapes land ~1-3x further out (measured:
// 0.014 vs f16's 0.013 at K = 40). x86 has no bf16 route at all; the value is
// only there to keep the helper compiling where the dtype is refused.
#if defined(NNOPS_ARCH_AARCH64)
inline constexpr float kBF16AccumTol = 4e-1f;
#else
inline constexpr float kBF16AccumTol = 5e-2f;
#endif

// ============================================================
// bf16 test helpers
// ============================================================

inline std::vector<nnops::backend::cpu::bfloat16> f32_to_bf16(const std::vector<float>& src) {
    std::vector<nnops::backend::cpu::bfloat16> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        simd::s_store(&dst[i], src[i]);
    }
    return dst;
}

inline std::vector<float> bf16_to_f32(const std::vector<nnops::backend::cpu::bfloat16>& src) {
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = simd::s_load(&src[i]);
    }
    return dst;
}

/// Returns (buffer, view); the buffer owns the data.
inline std::pair<std::vector<nnops::backend::cpu::bfloat16>, TensorView>
make_random_bf16_tensor(std::span<const int64_t> shape,
                        float min = -1.0f, float max = 1.0f,
                        uint64_t seed = 12345)
{
    auto [f32_buf, f32_view] = make_random_tensor(shape, min, max, seed);
    auto bf16_buf = f32_to_bf16(f32_buf);
    TensorView view(shape, DataType::bf16, bf16_buf.data(), f32_view.layout());
    return {std::move(bf16_buf), view};
}

/// Overload for initializer_list convenience.
inline std::pair<std::vector<nnops::backend::cpu::bfloat16>, TensorView>
make_random_bf16_tensor(std::initializer_list<int64_t> shape,
                        float min = -1.0f, float max = 1.0f,
                        uint64_t seed = 12345)
{
    return make_random_bf16_tensor(std::span<const int64_t>(shape.begin(), shape.size()),
                                   min, max, seed);
}

}  // namespace test
}  // namespace nnops
