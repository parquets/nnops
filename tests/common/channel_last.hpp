#pragma once
/// @file channel_last.hpp
/// @brief NCHW ↔ NHWC and OIHW ↔ HWIO bridges for the channel-last Conv2D
///        tests and benchmarks.
///
/// The channel-last routes take their activations stored [N][H][W][C] and their
/// weights stored [KH][KW][IC/G][OC] — TensorFlow's filter order, referred to
/// here as HWIO (TensorLayout::HWIO) — while every reference implementation in
/// this directory reads the planar orders. The tests therefore keep one copy in
/// each order and move between them with these helpers; the benches use them to
/// prepare the inputs once, outside the timed region.
///
/// All conversions are plain element copies — tens of microseconds against a
/// convolution — so a test can convert per case without hiding anything.

#include <span>
#include <vector>

namespace nnops::test {

/// NCHW [N][C][H][W] → NHWC [N][H][W][C].
template <class T>
std::vector<T> nchw_to_nhwc(std::span<const T> src,
                            int64_t N, int64_t C, int64_t H, int64_t W)
{
    std::vector<T> dst(src.size());
    const int64_t HW = H * W;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const T* s = src.data() + (n * C + c) * HW;
            T* d = dst.data() + n * HW * C + c;
            for (int64_t hw = 0; hw < HW; ++hw) { d[hw * C] = s[hw]; }
        }
    }
    return dst;
}

/// NHWC [N][H][W][C] → NCHW [N][C][H][W].
template <class T>
std::vector<T> nhwc_to_nchw(std::span<const T> src,
                            int64_t N, int64_t C, int64_t H, int64_t W)
{
    std::vector<T> dst(src.size());
    const int64_t HW = H * W;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            T* d = dst.data() + (n * C + c) * HW;
            const T* s = src.data() + n * HW * C + c;
            for (int64_t hw = 0; hw < HW; ++hw) { d[hw] = s[hw * C]; }
        }
    }
    return dst;
}

/// OIHW [OC][ICG][KH][KW] → HWIO [KH][KW][ICG][OC] (TensorFlow's filter).
template <class T>
std::vector<T> oihw_to_hwio(std::span<const T> src,
                            int64_t OC, int64_t ICG, int64_t KH, int64_t KW)
{
    std::vector<T> dst(src.size());
    const int64_t karea = KH * KW;
    for (int64_t oc = 0; oc < OC; ++oc) {
        const T* s = src.data() + oc * ICG * karea;
        for (int64_t ic = 0; ic < ICG; ++ic) {
            for (int64_t t = 0; t < karea; ++t) {
                dst[(t * ICG + ic) * OC + oc] = s[ic * karea + t];
            }
        }
    }
    return dst;
}

/// HWIO [KH][KW][ICG][OC] → OIHW [OC][ICG][KH][KW].
template <class T>
std::vector<T> hwio_to_oihw(std::span<const T> src,
                            int64_t OC, int64_t ICG, int64_t KH, int64_t KW)
{
    std::vector<T> dst(src.size());
    const int64_t karea = KH * KW;
    for (int64_t oc = 0; oc < OC; ++oc) {
        T* d = dst.data() + oc * ICG * karea;
        for (int64_t ic = 0; ic < ICG; ++ic) {
            for (int64_t t = 0; t < karea; ++t) {
                d[ic * karea + t] = src[(t * ICG + ic) * OC + oc];
            }
        }
    }
    return dst;
}

}  // namespace nnops::test
