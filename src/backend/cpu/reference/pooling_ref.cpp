/// @file pooling_ref.cpp
/// @brief Naive CPU reference implementation of 2D/3D spatial pooling (NCHW/NCDHW layout).
///
/// Supports both f32 and f16 data types. Computation always done in fp32;
/// dtype-specific load/store handled via overloaded helpers.
///
/// Parallelism is over N*C tasks (sample x channel), allowing better thread utilization
/// especially for small-batch inference (N=1).

#include "nnops/ops/pooling.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <cmath>
#include <algorithm>
#include <limits>

namespace nnops::backend::cpu::reference {

using nnops::simd::s_load;
using nnops::simd::s_store;

namespace {

/// Compute the NCHW offset for a 4D or 5D tensor using pitch.
/// row_stride is the element stride between rows (>= W for padded data).
inline int64_t nchw_offset(int64_t n, int64_t c,
                           int64_t d, int64_t h, int64_t w,
                           int64_t C, int64_t D, int64_t H,
                           int64_t row_stride) {
    return (((n * C + c) * D + d) * H + h) * row_stride + w;
}

}  // namespace

template <typename T>
void pooling_impl_ref(const PoolingAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();
    const int64_t srank = PoolingAttributes::spatial_rank(rank);  // 2 or 3

    const int64_t N = input.shape(0);
    const int64_t C = input.shape(1);

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

    const int64_t PD = (srank == 3) ? attrs.padding[0] : 0;
    const int64_t PH = attrs.padding[1];
    const int64_t PW = attrs.padding[2];

    const auto* in_ptr  = input.ptr<T>();
    auto* out_ptr = output.ptr<T>();

    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t out_row_stride = output.row_stride_elems();

    const int64_t K_total = KD * KH * KW;

    // Per-channel compute lambda (N*C parallel)
    const auto compute_channel = [&](int64_t n, int64_t c) {
        for (int64_t od = 0; od < OD; ++od) {
        for (int64_t oh = 0; oh < OH; ++oh) {
        for (int64_t ow = 0; ow < OW; ++ow) {
            float result = 0.0f;

            switch (attrs.type) {
            case PoolingType::Max: {
                float max_val = -std::numeric_limits<float>::infinity();
                bool any = false;
                for (int64_t kd = 0; kd < KD; ++kd) {
                const int64_t id = od * SD + kd - PD;
                if (id < 0 || id >= ID) continue;
                for (int64_t kh = 0; kh < KH; ++kh) {
                const int64_t ih = oh * SH + kh - PH;
                if (ih < 0 || ih >= IH) continue;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    const int64_t iw = ow * SW + kw - PW;
                    if (iw < 0 || iw >= IW) continue;
                    const int64_t in_idx =
                        nchw_offset(n, c, id, ih, iw, C, ID, IH, in_row_stride);
                    max_val = std::max(max_val, s_load(&in_ptr[in_idx]));
                    any = true;
                }}}
                result = any ? max_val : 0.0f;
                break;
            }
            case PoolingType::Average:
            case PoolingType::AverageExcludePad: {
                float sum = 0.0f;
                int64_t pad_count = 0;
                for (int64_t kd = 0; kd < KD; ++kd) {
                const int64_t id = od * SD + kd - PD;
                for (int64_t kh = 0; kh < KH; ++kh) {
                const int64_t ih = oh * SH + kh - PH;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    const int64_t iw = ow * SW + kw - PW;
                    if (id < 0 || id >= ID || ih < 0 || ih >= IH ||
                        iw < 0 || iw >= IW) {
                        ++pad_count;
                        continue;
                    }
                    const int64_t in_idx =
                        nchw_offset(n, c, id, ih, iw, C, ID, IH, in_row_stride);
                    sum += s_load(&in_ptr[in_idx]);
                }}}
                if (attrs.type == PoolingType::AverageExcludePad) {
                    const int64_t valid = K_total - pad_count;
                    result = valid > 0 ? sum / static_cast<float>(valid) : 0.0f;
                } else {
                    result = sum / static_cast<float>(K_total);
                }
                break;
            }
            case PoolingType::Lp: {
                float sum = 0.0f;
                const int64_t p = attrs.p_norm;
                for (int64_t kd = 0; kd < KD; ++kd) {
                const int64_t id = od * SD + kd - PD;
                if (id < 0 || id >= ID) continue;
                for (int64_t kh = 0; kh < KH; ++kh) {
                const int64_t ih = oh * SH + kh - PH;
                if (ih < 0 || ih >= IH) continue;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    const int64_t iw = ow * SW + kw - PW;
                    if (iw < 0 || iw >= IW) continue;
                    const int64_t in_idx =
                        nchw_offset(n, c, id, ih, iw, C, ID, IH, in_row_stride);
                    sum += std::pow(std::abs(s_load(&in_ptr[in_idx])),
                                    static_cast<float>(p));
                }}}
                result = std::pow(sum, 1.0f / static_cast<float>(p));
                break;
            }
            }

            const int64_t out_idx =
                nchw_offset(n, c, od, oh, ow, C, OD, OH, out_row_stride);
            if (attrs.add_to) {
                s_store(&out_ptr[out_idx], s_load(&out_ptr[out_idx]) + result);
            } else {
                s_store(&out_ptr[out_idx], result);
            }
        }}}
    };

    const int64_t NC = N * C;
    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, NC,
            [&](int64_t tid) {
                int64_t n = tid / C;
                int64_t c = tid % C;
                compute_channel(n, c);
            });
    } else {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                compute_channel(n, c);
            }
        }
    }
}

void pooling_ref(const PoolingAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx,
                 void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        pooling_impl_ref<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        pooling_impl_ref<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"pooling_ref: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu::reference
