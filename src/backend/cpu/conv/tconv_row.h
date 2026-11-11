#pragma once
/// @file tconv_row.h
/// @brief The row primitives both transposed-conv kernels accumulate with — one
///        contiguous run added, added as int32, FMA'd or filled, a vector at a
///        time — and the merge that turns an accumulated channel-last row into
///        output.
///
/// Every route — channel-first, channel-last, depthwise, 2-D or 3-D — reduces to
/// "add this tap's run into that row", so these are the leaves of all of them.

#include "nnops/detail/simd/simd.hpp"
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"
#include "backend/cpu/simd_kernel/simd_permute.hpp"

#include <cstdint>

namespace nnops::backend::cpu {

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
    nnops::kernel::tiled_transpose_2d(src, dst, rows, taps, taps, rows);
}

/// Merge one channel-last scratch row into the output: bias, then the
/// pre-activation residual, the epilogue and add_to, over `nc` of the row's
/// channels starting at @p oc_base. The caller has this row to itself, which is
/// what keeps add_to's original value — this is the only read and the only write.
///
/// `gm` is duck-typed on OW, OC, epilogue, pre_act and add_to, so the 2-D and
/// 3-D geometries both fit without either kernel learning the other's type.
template <typename T, typename G>
void merge_oc_row(T* orow, const T* row, const T* bias, int64_t oc_base,
                  const G& gm, int64_t nc)
{
    const T* type_tag = nullptr;
    constexpr int L = simd_lane_for<T>;
    for (int64_t ow = 0; ow < gm.OW; ++ow) {
        const T* rp = row + ow * nc;
        T* dp = orow + ow * gm.OC;
        int64_t c = 0;
        for (; c + L <= nc; c += L) {
            vec_for<T> acc = v_load(rp + c);
            if (bias) { acc = v_add(acc, v_load(bias + oc_base + c)); }
            if (gm.pre_act) { acc = v_add(acc, v_load(dp + c)); }
            acc = apply_epilogue_vec<T>(gm.epilogue, type_tag, acc);
            if (gm.add_to && !gm.pre_act) { acc = v_add(acc, v_load(dp + c)); }
            v_store(dp + c, acc);
        }
        for (; c < nc; ++c) {
            float v = s_load(rp + c);
            if (bias) { v += s_load(bias + oc_base + c); }
            if (gm.pre_act) { v += s_load(dp + c); }
            v = nnops::apply_epilogue(gm.epilogue, v, oc_base + c);
            if (gm.add_to && !gm.pre_act) { v += s_load(dp + c); }
            s_store(dp + c, v);
        }
    }
}

}  // namespace nnops::backend::cpu
