#pragma once
/// @file simd_tconv_row.hpp
/// @brief The row primitives both transposed-conv kernels accumulate with — one
///        contiguous run added, added as int32, FMA'd or filled, a vector at a
///        time.
///
/// Every route of the operator — channel-first, channel-last, depthwise, 2-D or
/// 3-D, fp or int8 — reduces to "add this tap's run into that row", so these are
/// the leaves of all of them. The walks that produce the runs are
/// simd_col2im.hpp; the merge that ends a channel-last row is route glue and
/// lives in conv/tconv_cl.h.

#include "nnops/detail/simd/simd.hpp"
#include "simd_permute.hpp"

#include <cstdint>

namespace nnops::kernel {

using namespace nnops::simd;

/// row[0, n) += src[0, n), element-wise.
template <typename T>
void row_add(T* dst, const T* src, int64_t n)
{
    constexpr int L = simd_lane_for<T>;
    int64_t i = 0;
    for (; i + L <= n; i += L) {
        v_store(dst + i, v_add(v_load(dst + i), v_load(src + i)));
    }
    for (; i < n; ++i) { s_store(dst + i, s_load(dst + i) + s_load(src + i)); }
}

/// row[0, n) += src[0, n) · w[0, n).
template <typename T>
void row_fma(T* dst, const T* src, const T* w, int64_t n)
{
    constexpr int L = simd_lane_for<T>;
    int64_t i = 0;
    for (; i + L <= n; i += L) {
        v_store(dst + i, v_fmadd(v_load(src + i), v_load(w + i), v_load(dst + i)));
    }
    for (; i < n; ++i) { s_store(dst + i, s_load(dst + i) + s_load(src + i) * s_load(w + i)); }
}

/// row[0, n) += src[0, n) · w, one weight over the whole run. The channel-first
/// depthwise shape: its vectors run over pixels, so a tap's weight is a scalar.
template <typename T>
void row_fma_scalar(T* dst, const T* src, float w, int64_t n)
{
    constexpr int L = simd_lane_for<T>;
    const T* tag = nullptr;
    const auto wv = v_set1(tag, w);
    int64_t i = 0;
    for (; i + L <= n; i += L) {
        v_store(dst + i, v_fmadd(v_load(src + i), wv, v_load(dst + i)));
    }
    for (; i < n; ++i) { s_store(dst + i, s_load(dst + i) + s_load(src + i) * w); }
}

/// row[0, n) += src[0, n) over the int32 col. The portable wrapper carries no
/// integer vector — the int8 accumulator is the only int32 row an operator
/// accumulates — so this one is native.
inline void row_add_i32(int32_t* dst, const int32_t* src, int64_t n)
{
    int64_t i = 0;
#if defined(NNOPS_ARCH_X86_64)
    for (; i + 8 <= n; i += 8) {
        const __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
        const __m256i s = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_add_epi32(d, s));
    }
#elif defined(NNOPS_ARCH_AARCH64)
    for (; i + 4 <= n; i += 4) {
        vst1q_s32(dst + i, vaddq_s32(vld1q_s32(dst + i), vld1q_s32(src + i)));
    }
#endif
    for (; i < n; ++i) { dst[i] += src[i]; }
}

/// row[0, n) = v.
template <typename T>
void row_fill(T* dst, int64_t n, float v)
{
    constexpr int L = simd_lane_for<T>;
    const T* tag = nullptr;
    const auto fill = v_set1(tag, v);
    int64_t i = 0;
    for (; i + L <= n; i += L) { v_store(dst + i, fill); }
    for (; i < n; ++i) { s_store(dst + i, v); }
}

/// [rows][taps] → [taps][rows], eight lanes at a time. A tap has to be
/// contiguous before a scatter can add one run or one oc vector, so every route
/// that permutes a filter — or a col that comes out of the GEMM oriented the
/// same way — transposes this one shape.
template <typename T>
void permute_taps(const T* src, T* dst, int64_t rows, int64_t taps)
{
    tiled_transpose_2d(src, dst, rows, taps, taps, rows);
}

}  // namespace nnops::kernel
