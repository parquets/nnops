#pragma once
/// @file depthwise_transpose_conv_internal.h
/// @brief Internals shared by the two DepthwiseTransposeConv2D translation
/// units.
///
/// Split one file per arithmetic family, as Conv2D and TransposeConv2D are:
///
///   - depthwise_transpose_conv.cpp    — the float routes (the per-tap channel
///                                       form and the scalar gather) and the
///                                       entry point;
///   - i8depthwise_transpose_conv.cpp  — the quantized route (s8 weight,
///                                       u8/s8/f16/f32 activation, int32
///                                       accumulate) and its merges.
///
/// The per-class facts — which (kh, ih) rows land on an output row, and which
/// (kw, shift) taps land on one output-column residue — are tap lists, not
/// arithmetic, and both families walk them unchanged. They live here.

#include "nnops/core/compute_context.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/epilogue.hpp"
#include "nnops/ops/depthwise_transpose_conv.hpp"

#include <cstdint>
#include <span>

namespace nnops::backend::cpu {

struct DwtConvGeom {
    int64_t N, C, IH, IW; // input
    int64_t OH, OW;       // output
    int64_t KH, KW;       // kernel
    int64_t SH, SW;       // stride
    int64_t DH, DW;       // dilation
    int64_t PH, PW;       // padding
    Epilogue epilogue;
    bool add_to;
    bool pre_act; ///< fold add_to's old value in before the epilogue
};

/// The tap lists hold at most ceil(K) entries; a taller kernel takes the scalar
/// route.
constexpr int64_t kMaxTaps = 16;

struct TapList {
    int64_t n = 0;
    int64_t a[kMaxTaps] = {}; // kernel index (kh or kw)
    int64_t b[kMaxTaps] = {}; // input index (ih) or column shift
};

/// The (kh, ih) pairs that land on output row @p oh.
inline TapList row_taps(const DwtConvGeom& gm, int64_t oh) noexcept
{
    TapList t;
    for (int64_t kh = 0; kh < gm.KH && t.n < kMaxTaps; ++kh) {
        const int64_t num = oh + gm.PH - kh * gm.DH;
        if (num < 0 || num % gm.SH != 0) {
            continue;
        }
        const int64_t ih = num / gm.SH;
        if (ih >= gm.IH) {
            continue;
        }
        t.a[t.n] = kh;
        t.b[t.n] = ih;
        ++t.n;
    }
    return t;
}

/// The (kw, shift) pairs that land on output-column residue @p r: the class
/// step t reads input column shift + t. Pairs that no step of the class can
/// reach are dropped.
inline TapList col_taps(const DwtConvGeom& gm, int64_t r, int64_t mic) noexcept
{
    TapList t;
    for (int64_t kw = 0; kw < gm.KW && t.n < kMaxTaps; ++kw) {
        const int64_t num = r + gm.PW - kw * gm.DW;
        if (num % gm.SW != 0) {
            continue;
        }
        const int64_t shift = num / gm.SW;
        if (shift >= gm.IW || shift <= -mic) {
            continue;
        }
        t.a[t.n] = kw;
        t.b[t.n] = shift;
        ++t.n;
    }
    return t;
}

/// The quantized family's entry (i8depthwise_transpose_conv.cpp): which
/// activation/weight/output/bias combinations have an arithmetic there.
Status depthwise_transpose_conv_int8_cpu(const DepthwiseTransposeConv2DAttributes& attrs,
                                         const DwtConvGeom& gm, TensorView& output,
                                         std::span<const TensorView> inputs,
                                         const ComputeContext& ctx);

} // namespace nnops::backend::cpu
