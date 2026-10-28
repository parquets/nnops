/// @file pooling.cpp
/// @brief SIMD-optimized CPU implementation of 2D/3D spatial pooling, in both
///        the packed (NCHWC8/NCDHWC8) and the planar (NCHW/NCDHW) layout.
///
/// Key design decisions:
///   1. The layout picks the vector lane. Packed: all C8 blocks (including
///      partial) use SIMD h4/h1 kernels — pad channels in NCHWC8 input are
///      zero, so SIMD operates correctly on all 8 lanes (max ignores pad
///      zeros, avg accumulates zero = no contribution). Planar: one channel
///      plane at a time, the lane runs along output width.
///   2. SIMD kernels handle exclude_pad via in-kernel per-position valid_count.
///   3. Height-4 blocking for packed rows (4 output rows per call), remainder
///      with h1; a planar row has no h4 form.
///   4. In-kernel bounds checking for all kernel positions — no pre-splitting.
///   5. All SW values supported — offset math is just integer arithmetic.
///   6. N*blocks parallel dispatch (blocks = C8 for packed, C for planar).
///   7. Params passed by const reference (unified across all kernels).
///
/// Quantized pooling is packed-only; layout conversion is the caller's job.

#include "nnops/ops/pooling.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "simd_kernel/simd_pooling.hpp"

#include <cstdint>
#include <type_traits>

namespace nnops::backend::cpu {

using namespace nnops::simd;
namespace k = nnops::kernel;

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

    // Channel blocks: C8 blocks when packed, one per channel when planar. The
    // block strides below describe both, so only the row kernel differs.
    const int64_t N      = input.shape(0);
    const int64_t blocks = input.num_channel_blocks();
    const bool packed    = is_channel_packed(input.layout());

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
    k::PoolingKernelParams p{};
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

    // Select SIMD kernels (all C8 blocks use SIMD)
    using PoolingFn = void (*)(T*, const T*, const k::PoolingKernelParams&,
                                float, bool);
    const bool is_max = (attrs.type == PoolingType::Max);
    const PoolingFn pool_h4_fn = is_max ? k::maxpool_h4<T> : k::avgpool_h4<T>;
    // A planar row is vectorized along W, so it has no h4 form: the h4 slot is
    // only reached when packed, and the h1 slot carries the layout's kernel.
    const PoolingFn pool_h1_fn = packed
        ? (is_max ? k::maxpool_h1<T> : k::avgpool_h1<T>)
        : (is_max ? k::maxpool_planar<T> : k::avgpool_planar<T>);

    // Per-block compute lambda (all C8 blocks use SIMD — pad channels are zero)
    const auto compute_block = [&](int64_t n, int64_t blk) {
        const T* in_base  = in_ptr + n * blocks * in_ch_stride + blk * in_ch_stride;
        T* out_base = out_ptr + n * blocks * out_ch_stride + blk * out_ch_stride;

        // Per-task copy: `p` is shared by every worker, so od/oh cannot live in it.
        k::PoolingKernelParams pc = p;

        for (int64_t od = 0; od < OD; ++od) {
            T* out_d = out_base + od * pc.out_row_stride * OH;

            int64_t oh = 0;
            if (packed) {
                for (; oh + 3 < OH; oh += 4) {
                    pc.od = od;
                    pc.oh = oh;
                    pool_h4_fn(out_d + oh * pc.out_row_stride, in_base, pc, avg_scale,
                               attrs.add_to);
                }
            }
            for (; oh < OH; ++oh) {
                pc.od = od;
                pc.oh = oh;
                pool_h1_fn(out_d + oh * pc.out_row_stride, in_base,
                           pc, avg_scale, attrs.add_to);
            }
        }
    };

    ctx.cpu.run(0, N * blocks,
            [&](int64_t tid) { compute_block(tid / blocks, tid % blocks); });
}

// ============================================================
// Global pooling — one output per channel plane, no window walk
// ============================================================

template <bool IsMax, typename T>
void pooling_global_impl(const PoolingAttributes& attrs,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t srank = PoolingAttributes::spatial_rank(input.rank());
    const int64_t N      = input.shape(0);
    const int64_t blocks = input.num_channel_blocks();
    const bool packed    = is_channel_packed(input.layout());

    k::PoolingKernelParams p{};
    p.ID = (srank == 3) ? input.shape(2) : 1;
    p.IH = input.shape(srank);
    p.IW = input.shape(srank + 1);
    p.in_d_stride   = p.IH * input.row_stride_elems();
    p.in_row_stride = input.row_stride_elems();

    const double count = static_cast<double>(p.ID) * p.IH * p.IW;
    const float scale = (count > 0.0) ? static_cast<float>(1.0 / count) : 0.0f;

    const int64_t in_ch_stride  = input.channel_block_stride_elems();
    const int64_t out_ch_stride = output.channel_block_stride_elems();
    const auto* in_ptr = input.ptr<T>();
    auto* out_ptr = output.ptr<T>();

    ctx.cpu.run(0, N * blocks, [&](int64_t tid) {
        const T* in_base = in_ptr + tid * in_ch_stride;
        T* out_base = out_ptr + tid * out_ch_stride;

        if (packed) {
            k::global_store(out_base, k::global_packed<IsMax>(in_base, p, scale),
                            attrs.add_to);
        } else {
            s_store_add(out_base, k::global_planar<IsMax>(in_base, p, scale),
                        attrs.add_to);
        }
    });
}

/// The fp driver: a native plane scan when global, the window walk otherwise.
template <typename T>
void pooling_fp(const PoolingAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx)
{
    if (!attrs.global) {
        pooling_impl<T>(attrs, output, inputs, ctx);
    } else if (attrs.type == PoolingType::Max) {
        pooling_global_impl<true, T>(attrs, output, inputs, ctx);
    } else {
        pooling_global_impl<false, T>(attrs, output, inputs, ctx);
    }
}

/// A global window spelled out as the full extent. The quantized kernels have
/// no whole-plane form, but reducing a window that covers the input is the
/// same reduction — the walk just costs its per-tap bounds checks.
inline PoolingAttributes full_extent_window(PoolingAttributes attrs,
                                            const TensorView& in)
{
    const int64_t srank = PoolingAttributes::spatial_rank(in.rank());
    const auto first = static_cast<size_t>(3 - srank);
    for (int64_t d = 0; d < srank; ++d) {
        attrs.kernel_shape[first + static_cast<size_t>(d)] = in.shape(2 + d);
        attrs.stride[first + static_cast<size_t>(d)]   = 1;
        attrs.padding[first + static_cast<size_t>(d)]  = 0;
        attrs.dilation[first + static_cast<size_t>(d)] = 1;
    }
    attrs.global = false;
    return attrs;
}

// ============================================================
// Quantized pooling (s8/u8 input/output)
// ============================================================
//
// Activation is PerTensor-quantized (broadcast scale/zero_point). Values are
// dequantized to float, pooled, and requantized on store (round-to-nearest-even).
// MaxPooling additionally takes a raw-integer fast path when input and output
// share the same dtype, scale and zero_point (and !add_to) — then `max` in the
// integer domain matches `max` in the float domain exactly.

template <typename InT, typename OutT>
void pooling_impl_quant(const PoolingAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t rank  = input.rank();
    const int64_t srank = PoolingAttributes::spatial_rank(rank);

    const int64_t N  = input.shape(0);
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

    const auto& in_qp  = input.quant_params();
    const auto& out_qp = output.quant_params();
    const float in_scale  = in_qp.scale;
    const float in_zero   = static_cast<float>(in_qp.zero_point);
    const float out_scale = out_qp.scale;
    const float out_zero  = static_cast<float>(out_qp.zero_point);

    const bool use_fast = (attrs.type == PoolingType::Max)
                       && std::is_same_v<InT, OutT>
                       && !attrs.add_to
                       && (in_qp.scale == out_qp.scale)
                       && (in_qp.zero_point == out_qp.zero_point);

    auto* out_ptr = output.ptr<OutT>();
    const auto* in_ptr = input.ptr<InT>();

    k::PoolingKernelParams p{};
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

    const auto compute_c8 = [&](int64_t n, int64_t c8) {
        const InT* in_base  = in_ptr + n * C8 * in_ch_stride + c8 * in_ch_stride;
        OutT* out_base = out_ptr + n * C8 * out_ch_stride + c8 * out_ch_stride;

        // Per-task copy (see above).
        k::PoolingKernelParams pc = p;

        for (int64_t od = 0; od < OD; ++od) {
            OutT* out_d = out_base + od * pc.out_row_stride * OH;

            int64_t oh = 0;
            if (use_fast) {
                // Only instantiated when InT == OutT (guaranteed by use_fast).
                if constexpr (std::is_same_v<InT, OutT>) {
                    for (; oh + 3 < OH; oh += 4) {
                        pc.od = od;
                        pc.oh = oh;
                        k::maxpool_h4_quant_fast<InT>(out_d + oh * pc.out_row_stride,
                                                   in_base, pc, out_zero);
                    }
                    for (; oh < OH; ++oh) {
                        pc.od = od;
                        pc.oh = oh;
                        k::maxpool_h1_quant_fast<InT>(out_d + oh * pc.out_row_stride,
                                                   in_base, pc, out_zero);
                    }
                }
            } else if (attrs.type == PoolingType::Max) {
                for (; oh + 3 < OH; oh += 4) {
                    pc.od = od;
                    pc.oh = oh;
                    k::maxpool_h4_quant<InT, OutT>(out_d + oh * pc.out_row_stride, in_base, pc,
                                               in_scale, in_zero, out_scale, out_zero,
                                               attrs.add_to);
                }
                for (; oh < OH; ++oh) {
                    pc.od = od;
                    pc.oh = oh;
                    k::maxpool_h1_quant<InT, OutT>(out_d + oh * pc.out_row_stride, in_base, pc,
                                               in_scale, in_zero, out_scale, out_zero,
                                               attrs.add_to);
                }
            } else {
                for (; oh + 3 < OH; oh += 4) {
                    pc.od = od;
                    pc.oh = oh;
                    k::avgpool_h4_quant<InT, OutT>(out_d + oh * pc.out_row_stride, in_base, pc,
                                               avg_scale, in_scale, in_zero, out_scale,
                                               out_zero, attrs.add_to);
                }
                for (; oh < OH; ++oh) {
                    pc.od = od;
                    pc.oh = oh;
                    k::avgpool_h1_quant<InT, OutT>(out_d + oh * pc.out_row_stride, in_base, pc,
                                               avg_scale, in_scale, in_zero, out_scale,
                                               out_zero, attrs.add_to);
                }
            }
        }
    };

    ctx.cpu.run(0, N * C8,
            [&](int64_t tid) { compute_c8(tid / C8, tid % C8); });
}

}  // anonymous namespace

Status pooling_cpu(const PoolingAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx,
                  void* /*workspace*/)
{
    const auto dtype  = inputs[0].data_type();
    const auto odtype = output.data_type();
    const bool quant = (dtype == DataType::s8 || dtype == DataType::u8);

    // Quantized pooling is packed-only: its dequantize-on-load kernels place
    // the lane on a C8 channel block, which a planar plane does not have.
    if (quant && !is_channel_packed(inputs[0].layout())) {
        return Status::unsupported;
    }
    // ...and it has no whole-plane form, so a global request takes the window
    // walk with the window spelled out as the full extent.
    const PoolingAttributes qattrs =
        (quant && attrs.global) ? full_extent_window(attrs, inputs[0]) : attrs;

    switch (dtype) {
    case DataType::f32:
        pooling_fp<float>(attrs, output, inputs, ctx);
        return Status::ok;
    case DataType::f16:
        pooling_fp<half>(attrs, output, inputs, ctx);
        return Status::ok;
    case DataType::s8:
        if (odtype == DataType::s8) {
            pooling_impl_quant<int8_t, int8_t>(qattrs, output, inputs, ctx);
        } else if (odtype == DataType::u8) {
            pooling_impl_quant<int8_t, uint8_t>(qattrs, output, inputs, ctx);
        } else {
            NNOPS_ASSERT(!"pooling_cpu: s8 input requires s8/u8 output");
        }
        return Status::ok;
    case DataType::u8:
        if (odtype == DataType::s8) {
            pooling_impl_quant<uint8_t, int8_t>(qattrs, output, inputs, ctx);
        } else if (odtype == DataType::u8) {
            pooling_impl_quant<uint8_t, uint8_t>(qattrs, output, inputs, ctx);
        } else {
            NNOPS_ASSERT(!"pooling_cpu: u8 input requires s8/u8 output");
        }
        return Status::ok;
    default:
        NNOPS_ASSERT(!"pooling_cpu: unsupported data type");
    }
    return Status::ok;
}

}  // namespace nnops::backend::cpu
