/// @file pooling.cpp
/// @brief SIMD-optimized CPU implementation of 2D/3D spatial pooling (NCHWC8 only).
///
/// Key design decisions:
///   1. Full C8 -> SIMD h4/h1 kernels (all types). SIMD kernels handle exclude_pad
///      via in-kernel per-position valid_count.
///   2. Partial C8 -> scalar fallback (valid_lanes only).
///      SIMD kernels ARE correct for partial C8 (nchwc8 tests prove it), but
///      the NCHW->NCHWC8 auto-conversion path has a subtle interaction — the
///      valid_lanes==8 gate keeps both paths correct. TODO: fix auto-conversion.
///   3. Height-4 blocking: process 4 output rows at once (h4), remainder with h1.
///   4. In-kernel bounds checking for all kernel positions — no pre-splitting.
///   5. All SW values supported — offset math is just integer arithmetic.
///   6. N*C8 parallel dispatch.
///   7. Params passed by const reference (unified across all kernels).
///
/// NCHW->NCHWC8 conversion is handled at the operator dispatch layer (src/ops/pooling.cpp).

#include "nnops/ops/pooling.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/core/tensor_layout.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <algorithm>
#include <limits>

namespace nnops::backend::cpu {
namespace pooling_detail {

using namespace nnops::simd;

// ============================================================
// Kernel parameters — set once, od/oh updated per call
// ============================================================

struct PoolingKernelParams {
    int64_t ID, IH, IW;          // input spatial dims
    int64_t OH, OW;              // output spatial dims (for store bounds)
    int64_t od, oh;              // current output position (set before call)
    int64_t KD, KH, KW;          // kernel shape
    int64_t SD, SH, SW;          // stride
    int64_t DD, DH, DW;          // dilation
    int64_t PD, PH, PW;          // padding
    int64_t out_row_stride;      // output row stride (elems)
    int64_t in_d_stride;         // input depth stride (elems)
    int64_t in_row_stride;       // input row stride (elems)
    bool exclude_pad = false;    // AvgPool: divide by valid count instead of K_total
};

// ============================================================
// MaxPooling SIMD kernels (bounds-checked, any SW)
// ============================================================

template <typename T>
inline void maxpool_h4_simd(
    T* output, const T* input,
    const PoolingKernelParams& p,
    float /*scale*/, bool add_to)
{
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();
    const auto vinit = v_set1(input, neg_inf);
    const auto vzero = v_set1(input, 0.0f);

    for (int64_t ow = 0; ow < p.OW; ++ow) {
        auto vacc0 = vinit, vacc1 = vinit, vacc2 = vinit, vacc3 = vinit;
        bool any0 = false, any1 = false, any2 = false, any3 = false;

        for (int64_t kd = 0; kd < p.KD; ++kd) {
            int64_t id = p.od * p.SD + kd * p.DD - p.PD;
            if (id < 0 || id >= p.ID) continue;
            int64_t d_off = id * p.in_d_stride;

            for (int64_t kh = 0; kh < p.KH; ++kh) {
                int64_t ih0 = (p.oh + 0) * p.SH + kh * p.DH - p.PH;
                int64_t ih1 = (p.oh + 1) * p.SH + kh * p.DH - p.PH;
                int64_t ih2 = (p.oh + 2) * p.SH + kh * p.DH - p.PH;
                int64_t ih3 = (p.oh + 3) * p.SH + kh * p.DH - p.PH;
                bool v0 = (p.oh + 0 < p.OH) && ih0 >= 0 && ih0 < p.IH;
                bool v1 = (p.oh + 1 < p.OH) && ih1 >= 0 && ih1 < p.IH;
                bool v2 = (p.oh + 2 < p.OH) && ih2 >= 0 && ih2 < p.IH;
                bool v3 = (p.oh + 3 < p.OH) && ih3 >= 0 && ih3 < p.IH;

                for (int64_t kw = 0; kw < p.KW; ++kw) {
                    int64_t iw = ow * p.SW + kw * p.DW - p.PW;
                    if (iw < 0 || iw >= p.IW) continue;
                    int64_t w_off = iw * 8;

                    if (v0) { vacc0 = v_max(vacc0, v_load(input + d_off + ih0 * p.in_row_stride + w_off)); any0 = true; }
                    if (v1) { vacc1 = v_max(vacc1, v_load(input + d_off + ih1 * p.in_row_stride + w_off)); any1 = true; }
                    if (v2) { vacc2 = v_max(vacc2, v_load(input + d_off + ih2 * p.in_row_stride + w_off)); any2 = true; }
                    if (v3) { vacc3 = v_max(vacc3, v_load(input + d_off + ih3 * p.in_row_stride + w_off)); any3 = true; }
                }
            }
        }

        if (!any0) vacc0 = vzero;
        if (!any1) vacc1 = vzero;
        if (!any2) vacc2 = vzero;
        if (!any3) vacc3 = vzero;

        if (p.oh + 0 < p.OH) {
            T* out0 = output + 0 * p.out_row_stride + ow * 8;
            if (add_to) vacc0 = v_add(vacc0, v_load(out0));
            v_store(out0, vacc0);
        }
        if (p.oh + 1 < p.OH) {
            T* out1 = output + 1 * p.out_row_stride + ow * 8;
            if (add_to) vacc1 = v_add(vacc1, v_load(out1));
            v_store(out1, vacc1);
        }
        if (p.oh + 2 < p.OH) {
            T* out2 = output + 2 * p.out_row_stride + ow * 8;
            if (add_to) vacc2 = v_add(vacc2, v_load(out2));
            v_store(out2, vacc2);
        }
        if (p.oh + 3 < p.OH) {
            T* out3 = output + 3 * p.out_row_stride + ow * 8;
            if (add_to) vacc3 = v_add(vacc3, v_load(out3));
            v_store(out3, vacc3);
        }
    }
}

template <typename T>
inline void maxpool_h1_simd(
    T* output, const T* input,
    const PoolingKernelParams& p,
    float /*scale*/, bool add_to)
{
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();
    const auto vinit = v_set1(input, neg_inf);
    const auto vzero = v_set1(input, 0.0f);

    for (int64_t ow = 0; ow < p.OW; ++ow) {
        auto vacc = vinit;
        bool any = false;

        for (int64_t kd = 0; kd < p.KD; ++kd) {
            int64_t id = p.od * p.SD + kd * p.DD - p.PD;
            if (id < 0 || id >= p.ID) continue;
            int64_t d_off = id * p.in_d_stride;

            for (int64_t kh = 0; kh < p.KH; ++kh) {
                int64_t ih = p.oh * p.SH + kh * p.DH - p.PH;
                if (ih < 0 || ih >= p.IH) continue;

                for (int64_t kw = 0; kw < p.KW; ++kw) {
                    int64_t iw = ow * p.SW + kw * p.DW - p.PW;
                    if (iw < 0 || iw >= p.IW) continue;
                    vacc = v_max(vacc, v_load(input + d_off + ih * p.in_row_stride + iw * 8));
                    any = true;
                }
            }
        }

        if (!any) vacc = vzero;

        T* out_r = output + ow * 8;
        if (add_to) vacc = v_add(vacc, v_load(out_r));
        v_store(out_r, vacc);
    }
}

// ============================================================
// AvgPooling SIMD kernels (bounds-checked, any SW)
// ============================================================

template <typename T>
inline void avgpool_h4_simd(
    T* output, const T* input,
    const PoolingKernelParams& p,
    float scale, bool add_to)
{
    const auto vscale = v_set1(input, scale);
    const auto vzero  = v_set1(input, 0.0f);

    for (int64_t ow = 0; ow < p.OW; ++ow) {
        auto vacc0 = vzero, vacc1 = vzero, vacc2 = vzero, vacc3 = vzero;
        int64_t cnt0 = 0, cnt1 = 0, cnt2 = 0, cnt3 = 0;

        for (int64_t kd = 0; kd < p.KD; ++kd) {
            int64_t id = p.od * p.SD + kd * p.DD - p.PD;
            if (id < 0 || id >= p.ID) continue;
            int64_t d_off = id * p.in_d_stride;

            for (int64_t kh = 0; kh < p.KH; ++kh) {
                int64_t ih0 = (p.oh + 0) * p.SH + kh * p.DH - p.PH;
                int64_t ih1 = (p.oh + 1) * p.SH + kh * p.DH - p.PH;
                int64_t ih2 = (p.oh + 2) * p.SH + kh * p.DH - p.PH;
                int64_t ih3 = (p.oh + 3) * p.SH + kh * p.DH - p.PH;
                bool v0 = (p.oh + 0 < p.OH) && ih0 >= 0 && ih0 < p.IH;
                bool v1 = (p.oh + 1 < p.OH) && ih1 >= 0 && ih1 < p.IH;
                bool v2 = (p.oh + 2 < p.OH) && ih2 >= 0 && ih2 < p.IH;
                bool v3 = (p.oh + 3 < p.OH) && ih3 >= 0 && ih3 < p.IH;

                for (int64_t kw = 0; kw < p.KW; ++kw) {
                    int64_t iw = ow * p.SW + kw * p.DW - p.PW;
                    if (iw < 0 || iw >= p.IW) continue;
                    int64_t w_off = iw * 8;

                    if (v0) { vacc0 = v_add(vacc0, v_load(input + d_off + ih0 * p.in_row_stride + w_off)); cnt0++; }
                    if (v1) { vacc1 = v_add(vacc1, v_load(input + d_off + ih1 * p.in_row_stride + w_off)); cnt1++; }
                    if (v2) { vacc2 = v_add(vacc2, v_load(input + d_off + ih2 * p.in_row_stride + w_off)); cnt2++; }
                    if (v3) { vacc3 = v_add(vacc3, v_load(input + d_off + ih3 * p.in_row_stride + w_off)); cnt3++; }
                }
            }
        }

        // Apply scale: exclude_pad → divide by valid count; else use fixed scale
        if (p.exclude_pad) {
            if (cnt0 > 0) vacc0 = v_mul(vacc0, v_set1(input, 1.0f / static_cast<float>(cnt0)));
            if (cnt1 > 0) vacc1 = v_mul(vacc1, v_set1(input, 1.0f / static_cast<float>(cnt1)));
            if (cnt2 > 0) vacc2 = v_mul(vacc2, v_set1(input, 1.0f / static_cast<float>(cnt2)));
            if (cnt3 > 0) vacc3 = v_mul(vacc3, v_set1(input, 1.0f / static_cast<float>(cnt3)));
        } else {
            vacc0 = v_mul(vacc0, vscale);
            vacc1 = v_mul(vacc1, vscale);
            vacc2 = v_mul(vacc2, vscale);
            vacc3 = v_mul(vacc3, vscale);
        }

        if (p.oh + 0 < p.OH) {
            T* out0 = output + 0 * p.out_row_stride + ow * 8;
            if (add_to) vacc0 = v_add(vacc0, v_load(out0));
            v_store(out0, vacc0);
        }
        if (p.oh + 1 < p.OH) {
            T* out1 = output + 1 * p.out_row_stride + ow * 8;
            if (add_to) vacc1 = v_add(vacc1, v_load(out1));
            v_store(out1, vacc1);
        }
        if (p.oh + 2 < p.OH) {
            T* out2 = output + 2 * p.out_row_stride + ow * 8;
            if (add_to) vacc2 = v_add(vacc2, v_load(out2));
            v_store(out2, vacc2);
        }
        if (p.oh + 3 < p.OH) {
            T* out3 = output + 3 * p.out_row_stride + ow * 8;
            if (add_to) vacc3 = v_add(vacc3, v_load(out3));
            v_store(out3, vacc3);
        }
    }
}

template <typename T>
inline void avgpool_h1_simd(
    T* output, const T* input,
    const PoolingKernelParams& p,
    float scale, bool add_to)
{
    const auto vscale = v_set1(input, scale);
    const auto vzero  = v_set1(input, 0.0f);

    for (int64_t ow = 0; ow < p.OW; ++ow) {
        auto vacc = vzero;
        int64_t valid_count = 0;

        for (int64_t kd = 0; kd < p.KD; ++kd) {
            int64_t id = p.od * p.SD + kd * p.DD - p.PD;
            if (id < 0 || id >= p.ID) continue;
            int64_t d_off = id * p.in_d_stride;

            for (int64_t kh = 0; kh < p.KH; ++kh) {
                int64_t ih = p.oh * p.SH + kh * p.DH - p.PH;
                if (ih < 0 || ih >= p.IH) continue;

                for (int64_t kw = 0; kw < p.KW; ++kw) {
                    int64_t iw = ow * p.SW + kw * p.DW - p.PW;
                    if (iw < 0 || iw >= p.IW) continue;
                    vacc = v_add(vacc, v_load(input + d_off + ih * p.in_row_stride + iw * 8));
                    valid_count++;
                }
            }
        }

        if (p.exclude_pad && valid_count > 0) {
            vacc = v_mul(vacc, v_set1(input, 1.0f / static_cast<float>(valid_count)));
        } else if (!p.exclude_pad) {
            vacc = v_mul(vacc, vscale);
        }

        T* out_r = output + ow * 8;
        if (add_to) vacc = v_add(vacc, v_load(out_r));
        v_store(out_r, vacc);
    }
}

}  // namespace pooling_detail

using namespace nnops::simd;
using namespace pooling_detail;

namespace {

template <typename T>
void pooling_impl(const PoolingAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t rank  = input.rank();
    const int64_t srank = PoolingAttributes::spatial_rank(rank);

    const int64_t N  = input.shape(0);
    const int64_t C  = input.shape(1);
    const int64_t C8 = input.num_channel_blocks();

    const int64_t ID = (srank == 3) ? input.shape(2) : 1;
    const int64_t IH = input.shape(srank);
    const int64_t IW = input.shape(srank + 1);

    const int64_t OD = (srank == 3) ? output.shape(2) : 1;
    const int64_t OH = output.shape(srank);
    const int64_t OW = output.shape(srank + 1);

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

    // Kernel params (shared across all calls, od/oh updated per call)
    PoolingKernelParams p{};
    p.ID = ID; p.IH = IH; p.IW = IW;
    p.OH = OH; p.OW = OW;
    p.KD = KD; p.KH = KH; p.KW = KW;
    p.SD = SD; p.SH = SH; p.SW = SW;
    p.DD = DD_; p.DH = DH; p.DW = DW;
    p.PD = PD; p.PH = PH; p.PW = PW;
    p.out_row_stride = output.row_stride_elems();
    p.in_d_stride    = IH * input.row_stride_elems();
    p.in_row_stride  = input.row_stride_elems();
    p.exclude_pad    = attrs.exclude_pad;

    const int64_t in_ch_stride  = input.channel_block_stride_elems();
    const int64_t out_ch_stride = output.channel_block_stride_elems();
    const float avg_scale = 1.0f / static_cast<float>(KD * KH * KW);

    // Select SIMD kernels (all C8 blocks; partial C8 also correct but auto-
    // conversion path has subtle interaction — keep valid_lanes==8 gate for now)
    using PoolingFn = void (*)(T*, const T*, const PoolingKernelParams&,
                                float, bool);
    const PoolingFn pool_h4_fn = (attrs.type == PoolingType::Max)
        ? maxpool_h4_simd<T> : avgpool_h4_simd<T>;
    const PoolingFn pool_h1_fn = (attrs.type == PoolingType::Max)
        ? maxpool_h1_simd<T> : avgpool_h1_simd<T>;

    // Per-C8-block compute lambda
    const auto compute_c8 = [&](int64_t n, int64_t c8) {
        const int64_t c_base = c8 * 8;
        const int64_t valid_lanes = std::min<int64_t>(8, C - c_base);

        const T* in_base  = in_ptr + n * C8 * in_ch_stride + c8 * in_ch_stride;
        T* out_base = out_ptr + n * C8 * out_ch_stride + c8 * out_ch_stride;

        if (valid_lanes == 8) {
            // SIMD path: full C8, all types
            for (int64_t od = 0; od < OD; ++od) {
                T* out_d = out_base + od * p.out_row_stride * OH;

                int64_t oh = 0;
                for (; oh + 3 < OH; oh += 4) {
                    p.od = od;
                    p.oh = oh;
                    pool_h4_fn(out_d + oh * p.out_row_stride, in_base, p, avg_scale, attrs.add_to);
                }
                for (; oh < OH; ++oh) {
                    p.od = od;
                    p.oh = oh;
                    pool_h1_fn(out_d + oh * p.out_row_stride, in_base,
                               p, avg_scale, attrs.add_to);
                }
            }
        } else {
            // Scalar path: partial C8 — same SIMD kernel logic but per-lane
            constexpr int64_t ws = 8;
            const bool is_max  = (attrs.type == PoolingType::Max);
            const bool excl_pad = attrs.exclude_pad;
            const float inv_k = 1.0f / static_cast<float>(KD * KH * KW);
            const int64_t K_total = KD * KH * KW;

            for (int64_t od = 0; od < OD; ++od) {
                T* out_d = out_base + od * p.out_row_stride * OH;
                for (int64_t oh = 0; oh < OH; ++oh) {
                    p.od = od;
                    p.oh = oh;
                    T* out_row = out_d + oh * p.out_row_stride;

                    for (int64_t ow = 0; ow < p.OW; ++ow) {
                        float result[8], pad_count[8] = {};
                        bool any_valid[8] = {};
                        if (is_max) {
                            for (int l = 0; l < valid_lanes; ++l)
                                result[l] = -std::numeric_limits<float>::infinity();
                        } else {
                            for (int l = 0; l < valid_lanes; ++l)
                                result[l] = 0.0f;
                        }

                        for (int64_t kd = 0; kd < p.KD; ++kd) {
                            int64_t id = p.od * p.SD + kd * p.DD - p.PD;
                            for (int64_t kh = 0; kh < p.KH; ++kh) {
                                int64_t ih = p.oh * p.SH + kh * p.DH - p.PH;
                                for (int64_t kw = 0; kw < p.KW; ++kw) {
                                    int64_t iw = ow * p.SW + kw * p.DW - p.PW;
                                    if (id < 0 || id >= p.ID
                                        || ih < 0 || ih >= p.IH
                                        || iw < 0 || iw >= p.IW) {
                                        for (int l = 0; l < valid_lanes; ++l) pad_count[l] += 1;
                                        continue;
                                    }
                                    int64_t w_off = iw * ws;
                                    if (is_max) {
                                        for (int l = 0; l < valid_lanes; ++l) {
                                            float val = s_load(&in_base[id * p.in_d_stride
                                                + ih * p.in_row_stride + w_off + l]);
                                            if (val > result[l]) result[l] = val;
                                            any_valid[l] = true;
                                        }
                                    } else {
                                        for (int l = 0; l < valid_lanes; ++l)
                                            result[l] += s_load(&in_base[id * p.in_d_stride
                                                + ih * p.in_row_stride + w_off + l]);
                                    }
                                }
                            }
                        }

                        if (is_max) {
                            for (int l = 0; l < valid_lanes; ++l)
                                if (!any_valid[l]) result[l] = 0.0f;
                        } else if (excl_pad) {
                            for (int l = 0; l < valid_lanes; ++l) {
                                int64_t vc = K_total - static_cast<int64_t>(pad_count[l]);
                                result[l] = (vc > 0) ? result[l] / static_cast<float>(vc) : 0.0f;
                            }
                        } else {
                            for (int l = 0; l < valid_lanes; ++l)
                                result[l] *= inv_k;
                        }

                        for (int l = 0; l < valid_lanes; ++l) {
                            const int64_t out_idx = ow * ws + l;
                            if (attrs.add_to)
                                s_store(&out_row[out_idx], s_load(&out_row[out_idx]) + result[l]);
                            else
                                s_store(&out_row[out_idx], result[l]);
                        }
                    }
                }
            }
        }
    };

    // Parallel dispatch (N * C8)
    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, N * C8,
            [&](int64_t tid) { compute_c8(tid / C8, tid % C8); });
    } else {
        for (int64_t n = 0; n < N; ++n)
            for (int64_t c8 = 0; c8 < C8; ++c8)
                compute_c8(n, c8);
    }
}

}  // anonymous namespace

// ============================================================
// Entry point — dtype dispatch
// ============================================================

void pooling_cpu(const PoolingAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx,
                  void* /*workspace*/)
{
    switch (inputs[0].data_type()) {
    case DataType::f32:
        pooling_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        pooling_impl<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"pooling_cpu: unsupported data type");
    }
}

}  // namespace nnops::backend::cpu
