/// @file pooling.cpp
/// @brief SIMD-optimized CPU implementation of 2D/3D spatial pooling (NCHW + NCHWC8).
///
/// Design combines best practices from two reference codebases:
///
///   nn_compute (d:\vscode\nn_compute\src\cpu\kernel\pooling\):
///     - Height-4 blocking: process 4 output rows simultaneously for Max/Avg
///     - Pre-positioned pointer passing: caller pre-positions input+output pointers,
///       SIMD kernels use only relative offsets — eliminates positional bugs
///     - Trust boundary: SIMD kernels called ONLY for interior valid region;
///       NO per-element bounds checks inside SIMD kernels
///     - Region splitting: pad-depth/top/bottom/front/back → scalar; interior → SIMD
///
///   onnxruntime MLAS (d:\git\onnxruntime\onnxruntime\core\mlas\lib\pooling.cpp):
///     - Two-phase reduction (vertical then horizontal) — noted for future reference
///     - Identity pre-fill for edge handling (reduction buffer approach)
///     - Global pooling fast path (dedicated kernel when kernel == input spatial shape)
///
/// Key design decisions for nnops:
///   1. SIMD kernels accept pre-positioned pointers, use only relative offsets
///   2. No per-element validity checks in SIMD — caller guarantees valid interior region
///   3. Width-SIMD via v_f32x8/v_f16x8 (simd_lane_for<T>-wide; AVX2 native, SSE/NEON emulated)
///   4. SW == 1: v_load (contiguous). SW == 2: v_load_even (stride-2 gather via LD2/UNPCK).
///   5. AverageExcludePad and Lp pooling: scalar-only (per-element counting needed)
///   6. Unified 2D/3D code path: KD=1/SD=1/DD=1/PD=0/ID=1/OD=1 for 2D
///   7. add_to support: branch-hoisted; one extra load per v_store when enabled
///   8. NCHW + NCHWC8 unified via Packed template: same SIMD kernels, different offset math
///   9. Per-(sample,channel_block) parallel dispatch via parallel_for (N*C or N*C8 tasks)

#include "nnops/ops/pooling.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/core/tensor_layout.hpp"
#include "nnops/detail/simd/simd.hpp"

#include "pooling_detail.hpp"

#include <algorithm>
#include <cmath>

namespace nnops::backend::cpu {

using namespace nnops::simd;
using namespace pooling_detail;

namespace {

// ============================================================
// Interior region helper — shared between NCHW and NCHWC8
// ============================================================

/// First output position where kernel start >= 0 (after padding): ceil(P / S).
inline int64_t interior_beg(int64_t pad, int64_t stride) {
    if (stride <= 0) { return 0; }
    return static_cast<int64_t>(
        std::ceil(static_cast<float>(pad) / static_cast<float>(stride)));
}

/// Last output position where kernel end is within input: ceil((I + P - kernel_span) / S).
inline int64_t interior_end(int64_t input_extent, int64_t pad,
                             int64_t kernel_span, int64_t stride, int64_t beg) {
    if (stride <= 0) { return input_extent; }
    const float num = static_cast<float>(input_extent) + static_cast<float>(pad)
                      - static_cast<float>(kernel_span);
    return std::max(static_cast<int64_t>(std::ceil(num / static_cast<float>(stride))), beg);
}

// ============================================================
// Unified pooling implementation — Packed=false for NCHW,
// Packed=true for NCHWC8.
// ============================================================

template <typename T, bool Packed>
void pooling_impl(const PoolingAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t rank  = input.rank();
    const int64_t srank = PoolingAttributes::spatial_rank(rank);

    // ----- Batch / channel dimensions -----
    const int64_t N  = input.shape(0);
    const int64_t C  = input.shape(1);
    const int64_t C8 = input.num_channel_blocks();
    const int64_t C_parallel = Packed ? C8 : C;  // N*C8 or N*C parallel tasks

    // ----- Spatial input dimensions -----
    const int64_t ID = (srank == 3) ? input.shape(2) : 1;
    const int64_t IH = input.shape(srank);
    const int64_t IW = input.shape(srank + 1);

    // ----- Spatial output dimensions -----
    const int64_t OD = (srank == 3) ? output.shape(2) : 1;
    const int64_t OH = output.shape(srank);
    const int64_t OW = output.shape(srank + 1);

    // ----- Kernel / stride / padding / dilation -----
    const int64_t KD = (srank == 3) ? attrs.kernel_shape[0] : 1;
    const int64_t KH = attrs.kernel_shape[1];
    const int64_t KW = attrs.kernel_shape[2];

    const int64_t SD = (srank == 3) ? attrs.stride[0] : 1;
    const int64_t SH = attrs.stride[1];
    const int64_t SW = attrs.stride[2];

    const int64_t DD_ = (srank == 3) ? attrs.dilation[0] : 1;
    const int64_t DH  = attrs.dilation[1];
    const int64_t DW  = attrs.dilation[2];

    const int64_t PD = (srank == 3) ? attrs.padding[0] : 0;
    const int64_t PH = attrs.padding[1];
    const int64_t PW = attrs.padding[2];

    auto* out_ptr = output.ptr<T>();
    const auto* in_ptr = input.ptr<T>();

    // ----- Strides -----
    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t in_d_stride    = IH * in_row_stride;
    const int64_t in_ch_stride   = Packed ? input.channel_block_stride_elems()  : ID * in_d_stride;
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_d_stride   = OH * out_row_stride;
    const int64_t out_ch_stride  = Packed ? output.channel_block_stride_elems() : OD * out_d_stride;
    const int64_t out_w_stride   = 1;

    // ----- SIMD eligibility -----
    const bool simd_eligible =
        (attrs.type == PoolingType::Max || attrs.type == PoolingType::Average)
        && (SW == 1 || SW == 2);

    // ----- Interior valid region -----
    const int64_t od_beg = interior_beg(PD, SD);
    const int64_t od_end = interior_end(ID, PD, (KD - 1) * DD_ + 1, SD, od_beg);
    const int64_t oh_beg = interior_beg(PH, SH);
    const int64_t oh_end = interior_end(IH, PH, (KH - 1) * DH + 1, SH, oh_beg);
    const int64_t ow_beg = interior_beg(PW, SW);
    const int64_t ow_end = interior_end(IW, PW, (KW - 1) * DW + 1, SW, ow_beg);

    // Clamp to output bounds
    const int64_t od_beg_c = std::max<int64_t>(0, std::min(od_beg, OD));
    const int64_t od_end_c = std::max<int64_t>(od_beg_c, std::min(od_end, OD));
    const int64_t oh_beg_c = std::max<int64_t>(0, std::min(oh_beg, OH));
    const int64_t oh_end_c = std::max<int64_t>(oh_beg_c, std::min(oh_end, OH));
    const int64_t ow_beg_c = std::max<int64_t>(0, std::min(ow_beg, OW));
    const int64_t ow_end_c = std::max<int64_t>(ow_beg_c, std::min(ow_end, OW));

    // SIMD-aligned interior OW range
    // NCHW (Packed=false): align to simd_lane_for<T> for v_load blocks
    // NCHWC8 (Packed=true): every W position processes 8 channels — no alignment needed
    const int64_t ow_simd_beg = ow_beg_c;
    const int64_t ow_simd_end = Packed
        ? ow_end_c
        : (simd_eligible
            ? ow_simd_beg + ((ow_end_c - ow_simd_beg) / simd_lane_for<T>) * simd_lane_for<T>
            : ow_beg_c);

    // AvgPool scale
    const float avg_scale = 1.0f / static_cast<float>(KD * KH * KW);

    // ----- Function pointer selection (hoist SW/type/Packed dispatch) -----
    using PoolingFn = void (*)(T*, const T*,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
        int64_t, int64_t, int64_t, int64_t, int64_t,
        float, bool);

    PoolingFn pool_h4_fn, pool_h1_fn;
    if (attrs.type == PoolingType::Max) {
        if constexpr (Packed) {
            pool_h4_fn = (SW == 2) ? maxpool_h4_simd<T, 2, true> : maxpool_h4_simd<T, 1, true>;
            pool_h1_fn = (SW == 2) ? maxpool_h1_simd<T, 2, true> : maxpool_h1_simd<T, 1, true>;
        } else {
            pool_h4_fn = (SW == 2) ? maxpool_h4_simd<T, 2, false> : maxpool_h4_simd<T, 1, false>;
            pool_h1_fn = (SW == 2) ? maxpool_h1_simd<T, 2, false> : maxpool_h1_simd<T, 1, false>;
        }
    } else {
        if constexpr (Packed) {
            pool_h4_fn = (SW == 2) ? avgpool_h4_simd<T, 2, true> : avgpool_h4_simd<T, 1, true>;
            pool_h1_fn = (SW == 2) ? avgpool_h1_simd<T, 2, true> : avgpool_h1_simd<T, 1, true>;
        } else {
            pool_h4_fn = (SW == 2) ? avgpool_h4_simd<T, 2, false> : avgpool_h4_simd<T, 1, false>;
            pool_h1_fn = (SW == 2) ? avgpool_h1_simd<T, 2, false> : avgpool_h1_simd<T, 1, false>;
        }
    }

    // ----- Per-work-item compute lambda -----
    // Packed=false: c_idx is a channel (0..C-1), valid_lanes always 1
    // Packed=true:  c_idx is a C8 block (0..C8-1), valid_lanes = 1..8
    const auto compute_item = [&](int64_t n, int64_t c_idx) {
        const int64_t c_base = Packed ? c_idx * 8 : c_idx;
        const int64_t valid_lanes = Packed ? std::min<int64_t>(8, C - c_base) : 1;
        const bool use_simd = Packed ? (simd_eligible && valid_lanes == 8) : simd_eligible;

        const T* in_base  = in_ptr + n * C_parallel * in_ch_stride + c_idx * in_ch_stride;
        T* out_base = out_ptr + n * C_parallel * out_ch_stride + c_idx * out_ch_stride;

        // ---- Pad-Depth Front ----
        for (int64_t od = 0; od < od_beg_c; ++od) {
            T* out_d = out_base + od * out_d_stride;
            for (int64_t oh = 0; oh < OH; ++oh) {
                pooling_scalar_row<T, Packed>(
                    out_d + oh * out_row_stride, in_base,
                    ID, IH, IW, od, oh, 0, OW,
                    KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                    in_d_stride, in_row_stride, valid_lanes,
                    attrs.type, attrs.add_to, attrs.p_norm);
            }
        }

        // ---- Interior Depth ----
        for (int64_t od = od_beg_c; od < od_end_c; ++od) {
            T* out_d = out_base + od * out_d_stride;
            const int64_t id_base = od * SD - PD;

            // -- Pad-Top --
            for (int64_t oh = 0; oh < oh_beg_c; ++oh) {
                pooling_scalar_row<T, Packed>(
                    out_d + oh * out_row_stride, in_base,
                    ID, IH, IW, od, oh, 0, OW,
                    KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                    in_d_stride, in_row_stride, valid_lanes,
                    attrs.type, attrs.add_to, attrs.p_norm);
            }

            // -- Interior Height --
            if (use_simd) {
                // --- h4 blocks ---
                int64_t oh = oh_beg_c;
                for (; oh + 3 < oh_end_c; oh += 4) {
                    const int64_t ih_base = oh * SH - PH;

                    // Pad-left
                    if (ow_beg_c > 0) {
                        for (int64_t r = 0; r < 4; ++r) {
                            pooling_scalar_row<T, Packed>(
                                out_d + (oh + r) * out_row_stride, in_base,
                                ID, IH, IW, od, oh + r, 0, ow_beg_c,
                                KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                                in_d_stride, in_row_stride, valid_lanes,
                                attrs.type, attrs.add_to, attrs.p_norm);
                        }
                    }

                    // SIMD h4 interior
                    const int64_t ow_simd_elems = ow_simd_end - ow_simd_beg;
                    if (ow_simd_elems > 0) {
                        const int64_t ow_mul = Packed ? 8 : out_w_stride;
                        T* out_simd = out_d + oh * out_row_stride + ow_simd_beg * ow_mul;
                        const T* in_simd = in_base
                            + id_base * in_d_stride
                            + ih_base * in_row_stride
                            + (ow_simd_beg * SW - PW) * ow_mul;

                        pool_h4_fn(out_simd, in_simd,
                                   KD, KH, KW, SH,
                                   DD_, DH, DW,
                                   ow_simd_elems,
                                   out_row_stride, out_w_stride,
                                   in_d_stride, in_row_stride,
                                   avg_scale, attrs.add_to);
                    }

                    // SIMD tail (NCHW only: interior W cols not simd-aligned)
                    if (ow_simd_end < ow_end_c) {
                        for (int64_t r = 0; r < 4; ++r) {
                            pooling_scalar_row<T, Packed>(
                                out_d + (oh + r) * out_row_stride, in_base,
                                ID, IH, IW, od, oh + r, ow_simd_end, ow_end_c,
                                KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                                in_d_stride, in_row_stride, valid_lanes,
                                attrs.type, attrs.add_to, attrs.p_norm);
                        }
                    }

                    // Pad-right
                    if (ow_end_c < OW) {
                        for (int64_t r = 0; r < 4; ++r) {
                            pooling_scalar_row<T, Packed>(
                                out_d + (oh + r) * out_row_stride, in_base,
                                ID, IH, IW, od, oh + r, ow_end_c, OW,
                                KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                                in_d_stride, in_row_stride, valid_lanes,
                                attrs.type, attrs.add_to, attrs.p_norm);
                        }
                    }
                }

                // --- h1 blocks (remaining interior rows) ---
                for (; oh < oh_end_c; ++oh) {
                    const int64_t ih_base = oh * SH - PH;

                    // Pad-left
                    if (ow_beg_c > 0) {
                        pooling_scalar_row<T, Packed>(
                            out_d + oh * out_row_stride, in_base,
                            ID, IH, IW, od, oh, 0, ow_beg_c,
                            KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                            in_d_stride, in_row_stride, valid_lanes,
                            attrs.type, attrs.add_to, attrs.p_norm);
                    }

                    // SIMD h1 interior
                    const int64_t ow_simd_elems = ow_simd_end - ow_simd_beg;
                    if (ow_simd_elems > 0) {
                        const int64_t ow_mul = Packed ? 8 : out_w_stride;
                        T* out_simd = out_d + oh * out_row_stride + ow_simd_beg * ow_mul;
                        const T* in_simd = in_base
                            + id_base * in_d_stride
                            + ih_base * in_row_stride
                            + (ow_simd_beg * SW - PW) * ow_mul;

                        pool_h1_fn(out_simd, in_simd,
                                   KD, KH, KW, SH,
                                   DD_, DH, DW,
                                   ow_simd_elems,
                                   out_row_stride, out_w_stride,
                                   in_d_stride, in_row_stride,
                                   avg_scale, attrs.add_to);
                    }

                    // SIMD tail (NCHW only)
                    if (ow_simd_end < ow_end_c) {
                        pooling_scalar_row<T, Packed>(
                            out_d + oh * out_row_stride, in_base,
                            ID, IH, IW, od, oh, ow_simd_end, ow_end_c,
                            KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                            in_d_stride, in_row_stride, valid_lanes,
                            attrs.type, attrs.add_to, attrs.p_norm);
                    }

                    // Pad-right
                    if (ow_end_c < OW) {
                        pooling_scalar_row<T, Packed>(
                            out_d + oh * out_row_stride, in_base,
                            ID, IH, IW, od, oh, ow_end_c, OW,
                            KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                            in_d_stride, in_row_stride, valid_lanes,
                            attrs.type, attrs.add_to, attrs.p_norm);
                    }
                }
            } else {
                // -- Non-SIMD interior: all-scalar --
                for (int64_t oh = oh_beg_c; oh < oh_end_c; ++oh) {
                    pooling_scalar_row<T, Packed>(
                        out_d + oh * out_row_stride, in_base,
                        ID, IH, IW, od, oh, 0, OW,
                        KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                        in_d_stride, in_row_stride, valid_lanes,
                        attrs.type, attrs.add_to, attrs.p_norm);
                }
            }

            // -- Pad-Bottom --
            for (int64_t oh = oh_end_c; oh < OH; ++oh) {
                pooling_scalar_row<T, Packed>(
                    out_d + oh * out_row_stride, in_base,
                    ID, IH, IW, od, oh, 0, OW,
                    KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                    in_d_stride, in_row_stride, valid_lanes,
                    attrs.type, attrs.add_to, attrs.p_norm);
            }
        }

        // ---- Pad-Depth Back ----
        for (int64_t od = od_end_c; od < OD; ++od) {
            T* out_d = out_base + od * out_d_stride;
            for (int64_t oh = 0; oh < OH; ++oh) {
                pooling_scalar_row<T, Packed>(
                    out_d + oh * out_row_stride, in_base,
                    ID, IH, IW, od, oh, 0, OW,
                    KD, KH, KW, SD, SH, SW, DD_, DH, DW, PD, PH, PW,
                    in_d_stride, in_row_stride, valid_lanes,
                    attrs.type, attrs.add_to, attrs.p_norm);
            }
        }
    };

    // ----- Parallel dispatch (N * C_parallel) -----
    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, N * C_parallel,
            [&](int64_t tid) {
                compute_item(tid / C_parallel, tid % C_parallel);
            });
    } else {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t ci = 0; ci < C_parallel; ++ci) {
                compute_item(n, ci);
            }
        }
    }
}

}  // anonymous namespace

// ============================================================
// Entry point — dtype + layout dispatch
// ============================================================

void pooling_cpu(const PoolingAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx,
                  void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    const bool packed = is_channel_packed(inputs[0].layout())
                     && is_channel_packed(output.layout());

    switch (dtype) {
    case DataType::f32:
        if (packed) {
            pooling_impl<float, true>(attrs, output, inputs, ctx);
        } else {
            pooling_impl<float, false>(attrs, output, inputs, ctx);
        }
        return;
    case DataType::f16:
        if (packed) {
            pooling_impl<half, true>(attrs, output, inputs, ctx);
        } else {
            pooling_impl<half, false>(attrs, output, inputs, ctx);
        }
        return;
    default:
        NNOPS_ASSERT(!"pooling_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
