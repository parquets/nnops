/// @file depthwise_conv.cpp
/// @brief SIMD-optimized CPU implementation of 2D/3D depthwise convolution (NCHWC8/NCDHWC8 only).
///
/// Input/output are always channel-packed (NCHWC8 or NCDHWC8). Weights are prepacked
/// by the operator dispatch layer into dense [C8, (KD,) KH, KW, 8] format before
/// reaching this backend.
///
/// Spatial rank is auto-detected from input tensor rank (4→2D, 5→3D).
/// A single unified kernel handles both 2D and 3D: the KD loop is always present;
/// for 2D, KD=1 so it degenerates to a single no-op iteration.
///
/// Key design decisions (following the pooling.cpp pattern):
///   1. Single DwConvParams struct with full 3D fields — 2D uses KD=1/OD=1/ID=1.
///   2. Single set of h4/h1 SIMD kernels with built-in KD loop.
///   3. Dispatch has od loop; for 2D, OD=1 so it runs once.
///   4. All C8 blocks use SIMD — pad channels are zero.
///   5. N*C8 parallel dispatch.
///   6. Weights loaded as vectors (8 channels at once) from prepacked buffer.

#include "nnops/ops/depthwise_conv.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/core/tensor_layout.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "epilogue_impl.hpp"


namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

// ============================================================
// Kernel parameters — set once per C8 block, od/oh updated per call
// ============================================================

struct DwConvParams {
    int64_t ID, IH, IW;           // input spatial dims
    int64_t OH, OW;               // output spatial dims (for store bounds)
    int64_t od, oh;               // current output position (set before call)
    int64_t KD, KH, KW;           // kernel shape
    int64_t SD, SH, SW;           // stride
    int64_t DD, DH, DW;           // dilation
    int64_t PD, PH, PW;           // padding
    int64_t out_row_stride;       // output row stride (elems)
    int64_t in_d_stride;          // input depth stride (elems)
    int64_t in_row_stride;        // input row stride (elems)
};

// ============================================================
// H4 SIMD kernel: process 4 output rows × all OW columns
// ============================================================

template <typename T>
inline void dwconv_h4_simd(
    T* output, const T* input, const T* w_base,
    const T* bias_vec,            // nullptr or pointer to 8 bias values
    const DwConvParams& p,
    const Epilogue& epilogue,
    bool add_to)
{
    const T* type_tag = output;
    const auto vbias = bias_vec ? v_load(bias_vec) : v_zero(type_tag);

    for (int64_t ow = 0; ow < p.OW; ++ow) {
        auto vacc0 = vbias;
        auto vacc1 = vbias;
        auto vacc2 = vbias;
        auto vacc3 = vbias;

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

                    auto vk = v_load(&w_base[(kd * p.KH * p.KW + kh * p.KW + kw) * 8]);

                    if (v0) vacc0 = v_fmadd(v_load(input + d_off + ih0 * p.in_row_stride + w_off), vk, vacc0);
                    if (v1) vacc1 = v_fmadd(v_load(input + d_off + ih1 * p.in_row_stride + w_off), vk, vacc1);
                    if (v2) vacc2 = v_fmadd(v_load(input + d_off + ih2 * p.in_row_stride + w_off), vk, vacc2);
                    if (v3) vacc3 = v_fmadd(v_load(input + d_off + ih3 * p.in_row_stride + w_off), vk, vacc3);
                }
            }
        }

        // SIMD epilogue
        vacc0 = apply_epilogue_vec(epilogue, type_tag, vacc0);
        vacc1 = apply_epilogue_vec(epilogue, type_tag, vacc1);
        vacc2 = apply_epilogue_vec(epilogue, type_tag, vacc2);
        vacc3 = apply_epilogue_vec(epilogue, type_tag, vacc3);

        // Store
        if (p.oh + 0 < p.OH) v_store_add(output + 0 * p.out_row_stride + ow * 8, vacc0, add_to);
        if (p.oh + 1 < p.OH) v_store_add(output + 1 * p.out_row_stride + ow * 8, vacc1, add_to);
        if (p.oh + 2 < p.OH) v_store_add(output + 2 * p.out_row_stride + ow * 8, vacc2, add_to);
        if (p.oh + 3 < p.OH) v_store_add(output + 3 * p.out_row_stride + ow * 8, vacc3, add_to);
    }
}

// ============================================================
// H1 SIMD kernel: process 1 output row × all OW columns
// ============================================================

template <typename T>
inline void dwconv_h1_simd(
    T* output, const T* input, const T* w_base,
    const T* bias_vec,
    const DwConvParams& p,
    const Epilogue& epilogue,
    bool add_to)
{
    const T* type_tag = output;
    const auto vbias = bias_vec ? v_load(bias_vec) : v_zero(type_tag);

    for (int64_t ow = 0; ow < p.OW; ++ow) {
        auto vacc = vbias;

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
                    int64_t w_off = iw * 8;

                    auto vk = v_load(&w_base[(kd * p.KH * p.KW + kh * p.KW + kw) * 8]);
                    vacc = v_fmadd(v_load(input + d_off + ih * p.in_row_stride + w_off), vk, vacc);
                }
            }
        }

        vacc = apply_epilogue_vec(epilogue, type_tag, vacc);
        v_store_add(output + ow * 8, vacc, add_to);
    }
}

}  // anonymous namespace

// ============================================================
// Main depthwise conv implementation (unified 2D/3D)
// ============================================================

template <typename T>
void dwconv_impl_nchwc8(
    const DepthwiseConvAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& packed_weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t rank  = input.rank();
    const int64_t srank = DepthwiseConvAttributes::spatial_rank(rank);

    const int64_t N  = input.shape(0);
    const int64_t C8 = input.num_channel_blocks();

    // Spatial dims (auto-detect 2D vs 3D)
    const int64_t ID  = (srank == 3) ? input.shape(2) : 1;
    const int64_t IH  = input.shape(srank);
    const int64_t IW  = input.shape(srank + 1);

    const int64_t KD  = attrs.kernel_size[0];
    const int64_t KH  = attrs.kernel_size[1];
    const int64_t KW  = attrs.kernel_size[2];

    const int64_t OD  = (srank == 3) ? output.shape(2) : 1;
    const int64_t OH  = output.shape(srank);
    const int64_t OW  = output.shape(srank + 1);

    const int64_t SD  = attrs.stride[0];
    const int64_t SH  = attrs.stride[1];
    const int64_t SW  = attrs.stride[2];
    const int64_t DD  = attrs.dilation[0];
    const int64_t DH  = attrs.dilation[1];
    const int64_t DW  = attrs.dilation[2];
    const int64_t PD  = attrs.padding[0];
    const int64_t PH  = attrs.padding[1];
    const int64_t PW  = attrs.padding[2];

    auto* out_ptr = output.ptr<T>();
    const auto* in_ptr  = input.ptr<T>();
    const auto* pw_ptr  = packed_weight.ptr<T>();
    const auto* pb_ptr  = has_bias ? inputs[2].ptr<T>() : nullptr;

    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t in_d_stride    = IH * in_row_stride;
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_d_stride   = OH * out_row_stride;
    const int64_t in_ch_stride   = input.channel_block_stride_elems();
    const int64_t out_ch_stride  = output.channel_block_stride_elems();

    // Kernel params (od/oh updated per call)
    DwConvParams p{};
    p.ID = ID; p.IH = IH; p.IW = IW;
    p.OH = OH; p.OW = OW;
    p.KD = KD; p.KH = KH; p.KW = KW;
    p.SD = SD; p.SH = SH; p.SW = SW;
    p.DD = DD; p.DH = DH; p.DW = DW;
    p.PD = PD; p.PH = PH; p.PW = PW;
    p.out_row_stride = out_row_stride;
    p.in_d_stride    = in_d_stride;
    p.in_row_stride  = in_row_stride;

    // Per-C8-block compute lambda (all C8 blocks use SIMD — pad channels are zero)
    const auto compute_c8 = [&](int64_t n, int64_t c8) {
        const T* in_base  = in_ptr  + n * C8 * in_ch_stride  + c8 * in_ch_stride;
        T*       out_base = out_ptr + n * C8 * out_ch_stride + c8 * out_ch_stride;
        const T* w_base   = pw_ptr + c8 * KD * KH * KW * 8;
        const T* bias_vec = pb_ptr ? pb_ptr + c8 * 8 : nullptr;

        for (int64_t od = 0; od < OD; ++od) {
            T* out_d = out_base + od * out_d_stride;

            int64_t oh = 0;
            for (; oh + 3 < OH; oh += 4) {
                p.od = od;
                p.oh = oh;
                dwconv_h4_simd(out_d + oh * p.out_row_stride, in_base, w_base, bias_vec,
                               p, attrs.epilogue, attrs.add_to);
            }
            for (; oh < OH; ++oh) {
                p.od = od;
                p.oh = oh;
                dwconv_h1_simd(out_d + oh * p.out_row_stride, in_base, w_base, bias_vec,
                               p, attrs.epilogue, attrs.add_to);
            }
        }
    };

    // Parallel dispatch (N * C8)
    const int64_t N_C8 = N * C8;
    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, N_C8,
            [&](int64_t tid) { compute_c8(tid / C8, tid % C8); });
    } else {
        for (int64_t n = 0; n < N; ++n)
            for (int64_t c8 = 0; c8 < C8; ++c8)
                compute_c8(n, c8);
    }
}

// ============================================================
// Entry point with dtype dispatch
// ============================================================

void depthwise_conv_cpu(const DepthwiseConvAttributes& attrs,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        dwconv_impl_nchwc8<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        dwconv_impl_nchwc8<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"depthwise_conv_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
