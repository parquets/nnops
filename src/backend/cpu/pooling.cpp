/// @file pooling.cpp
/// @brief SIMD-optimized CPU implementation of 2D/3D spatial pooling (NCHWC8 only).
///
/// Key design decisions:
///   1. All C8 blocks (including partial) use SIMD h4/h1 kernels — pad channels in
///      NCHWC8 input are zero, so SIMD operates correctly on all 8 lanes
///      (max ignores pad zeros, avg accumulates zero = no contribution).
///   2. SIMD kernels handle exclude_pad via in-kernel per-position valid_count.
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
#include "simd_kernel/simd_pooling.hpp"

#include <limits>

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
    const PoolingFn pool_h4_fn = (attrs.type == PoolingType::Max)
        ? k::maxpool_h4<T> : k::avgpool_h4<T>;
    const PoolingFn pool_h1_fn = (attrs.type == PoolingType::Max)
        ? k::maxpool_h1<T> : k::avgpool_h1<T>;

    // Per-C8-block compute lambda (all C8 blocks use SIMD — pad channels are zero)
    const auto compute_c8 = [&](int64_t n, int64_t c8) {
        const T* in_base  = in_ptr + n * C8 * in_ch_stride + c8 * in_ch_stride;
        T* out_base = out_ptr + n * C8 * out_ch_stride + c8 * out_ch_stride;

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
    };

    // Parallel dispatch (N * C8)
    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, N * C8,
            [&](int64_t tid) { compute_c8(tid / C8, tid % C8); });
    } else {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c8 = 0; c8 < C8; ++c8) {
                compute_c8(n, c8);
            }
        }
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
