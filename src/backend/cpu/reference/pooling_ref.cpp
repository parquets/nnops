/// @file pooling_ref.cpp
/// @brief Naive CPU reference implementation of 2D/3D spatial pooling (NCHW/NCDHW layout).

#include "nnops/ops/pooling.hpp"
#include "nnops/core/parallel_for.hpp"

#include <cmath>
#include <algorithm>
#include <limits>

namespace nnops::backend::cpu::reference {

namespace {

/// Compute the NCHW offset for a 4D or 5D tensor.
inline int64_t nchw_offset(int64_t n, int64_t c,
                           int64_t d, int64_t h, int64_t w,
                           int64_t C, int64_t D, int64_t H, int64_t W) {
    return (((n * C + c) * D + d) * H + h) * W + w;
}

}  // namespace

void pooling_ref(const PoolingAttributes& attrs,
                 const TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx,
                 void* /*workspace*/)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();
    const int64_t srank = PoolingAttributes::spatial_rank(rank);  // 2 or 3

    // Common dimensions
    const int64_t N = input.shape(0);
    const int64_t C = input.shape(1);

    // Spatial input dimensions: D, H, W (D=1 for 2D)
    const int64_t ID = (srank == 3) ? input.shape(2) : 1;
    const int64_t IH = input.shape(srank);      // shape[2] for 2D, shape[3] for 3D
    const int64_t IW = input.shape(srank + 1);  // shape[3] for 2D, shape[4] for 3D

    // Spatial output dimensions
    const int64_t OD = (srank == 3) ? output.shape(2) : 1;
    const int64_t OH = output.shape(srank);
    const int64_t OW = output.shape(srank + 1);

    // Attribute layout: kernel_shape/stride/padding = [KD, KH, KW]
    // KH always at index [1], KW always at index [2].
    // KD at index [0] only used for 3D (srank=3).
    const int64_t KD = (srank == 3) ? attrs.kernel_shape[0] : 1;
    const int64_t KH = attrs.kernel_shape[1];
    const int64_t KW = attrs.kernel_shape[2];

    const int64_t SD = (srank == 3) ? attrs.stride[0] : 1;
    const int64_t SH = attrs.stride[1];
    const int64_t SW = attrs.stride[2];

    const int64_t PD = (srank == 3) ? attrs.padding[0] : 0;
    const int64_t PH = attrs.padding[1];
    const int64_t PW = attrs.padding[2];

    const auto* in_ptr  = input.data_as<float>();
    auto* out_ptr = output.data_as<float>();

    const int64_t K_total = KD * KH * KW;

    // Per-sample compute lambda (handles both 2D and 3D via D loops)
    const auto compute_sample = [&](int64_t n) {
        for (int64_t c = 0; c < C; ++c) {
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
                            nchw_offset(n, c, id, ih, iw, C, ID, IH, IW);
                        max_val = std::max(max_val, in_ptr[in_idx]);
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
                            nchw_offset(n, c, id, ih, iw, C, ID, IH, IW);
                        sum += in_ptr[in_idx];
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
                            nchw_offset(n, c, id, ih, iw, C, ID, IH, IW);
                        sum += std::pow(std::abs(in_ptr[in_idx]),
                                        static_cast<float>(p));
                    }}}
                    result = std::pow(sum, 1.0f / static_cast<float>(p));
                    break;
                }
                }

                const int64_t out_idx =
                    nchw_offset(n, c, od, oh, ow, C, OD, OH, OW);
                out_ptr[out_idx] = result;
            }}}
        }
    };

    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, N,
            [&](int64_t n) { compute_sample(n); });
    } else {
        for (int64_t n = 0; n < N; ++n) {
            compute_sample(n);
        }
    }
}

}  // namespace nnops::backend::cpu::reference
