/// @file pixel_shuffle.cpp
/// @brief CPU SpaceToDepth / PixelShuffle — strided element permutations.
///
/// One task per output row: SpaceToDepth walks the b×b source pixels of each
/// output pixel and scatters their C channels with stride b² into the row (the
/// channel order is c·b² + bh·b + bw, a b²×C transpose); PixelShuffle walks the
/// b×b output pixels of each source pixel and either gathers with stride b²
/// (DCR) or copies contiguously (CRD). The channel run is contiguous in both,
/// and every dimension step comes from stride_elems, so a pitched NHWC view
/// works unchanged.

#include "nnops/ops/pixel_shuffle.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "common/dtype_dispatch.hpp"

#include <cstdint>

namespace nnops::backend::cpu {

namespace {

template <typename T>
void space_to_depth_impl(const SpaceToDepthAttributes& attrs, TensorView& output,
                         const TensorView& input, const ComputeContext& ctx)
{
    const int64_t N = input.shape(0);
    const int64_t H = input.shape(1);
    const int64_t W = input.shape(2);
    const int64_t C = input.shape(3);
    const int64_t b = attrs.block_size;
    const int64_t Ho = H / b;
    const int64_t Wo = W / b;

    const auto* x = input.ptr<T>();
    auto* y = output.ptr<T>();
    const int64_t xs0 = input.stride_elems(0);
    const int64_t xs1 = input.stride_elems(1);
    const int64_t xs2 = input.stride_elems(2);
    const int64_t ys0 = output.stride_elems(0);
    const int64_t ys1 = output.stride_elems(1);
    const int64_t ys2 = output.stride_elems(2);
    const int64_t bb = b * b;

    ctx.cpu.run(0, N * Ho, [&](int64_t row) {
        const int64_t n = row / Ho;
        const int64_t ho = row % Ho;
        T* y_row = y + n * ys0 + ho * ys1;
        for (int64_t wo = 0; wo < Wo; ++wo) {
            T* dst = y_row + wo * ys2;
            for (int64_t bh = 0; bh < b; ++bh) {
                for (int64_t bw = 0; bw < b; ++bw) {
                    const T* src =
                        x + n * xs0 + (ho * b + bh) * xs1 + (wo * b + bw) * xs2;
                    int64_t o = bh * b + bw;
                    for (int64_t c = 0; c < C; ++c, o += bb) {
                        dst[o] = src[c];
                    }
                }
            }
        }
    });
}

template <typename T>
void pixel_shuffle_impl(const PixelShuffleAttributes& attrs, TensorView& output,
                        const TensorView& input, const ComputeContext& ctx)
{
    const int64_t N = input.shape(0);
    const int64_t H = input.shape(1);
    const int64_t W = input.shape(2);
    const int64_t b = attrs.block_size;
    const int64_t C_out = input.shape(3) / (b * b);
    const int64_t H_out = H * b;
    const int64_t W_out = W * b;
    const bool dcr = attrs.mode == PixelShuffleMode::Dcr;

    const auto* x = input.ptr<T>();
    auto* y = output.ptr<T>();
    const int64_t xs0 = input.stride_elems(0);
    const int64_t xs1 = input.stride_elems(1);
    const int64_t xs2 = input.stride_elems(2);
    const int64_t ys0 = output.stride_elems(0);
    const int64_t ys1 = output.stride_elems(1);
    const int64_t ys2 = output.stride_elems(2);
    const int64_t bb = b * b;

    ctx.cpu.run(0, N * H_out, [&](int64_t row) {
        const int64_t n = row / H_out;
        const int64_t ho = row % H_out;
        const int64_t h = ho / b;
        const int64_t bh = ho % b;
        T* y_row = y + n * ys0 + ho * ys1;
        for (int64_t wo = 0; wo < W_out; ++wo) {
            const int64_t w = wo / b;
            const int64_t bw = wo % b;
            const int64_t q = bh * b + bw;
            T* dst = y_row + wo * ys2;
            const T* src = x + n * xs0 + h * xs1 + w * xs2;
            if (dcr) {
                int64_t s = q;
                for (int64_t c = 0; c < C_out; ++c, s += bb) {
                    dst[c] = src[s];
                }
            } else {
                const T* s = src + q * C_out;
                for (int64_t c = 0; c < C_out; ++c) {
                    dst[c] = s[c];
                }
            }
        }
    });
}

} // anonymous namespace

Status space_to_depth_cpu(const SpaceToDepthAttributes& attrs, TensorView& output,
                          std::span<const TensorView> inputs, const ComputeContext& ctx,
                          void* /*workspace*/)
{
    for_each_data_type(inputs[0].data_type(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        space_to_depth_impl<T>(attrs, output, inputs[0], ctx);
    });
    return Status::ok;
}

Status pixel_shuffle_cpu(const PixelShuffleAttributes& attrs, TensorView& output,
                         std::span<const TensorView> inputs, const ComputeContext& ctx,
                         void* /*workspace*/)
{
    for_each_data_type(inputs[0].data_type(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        pixel_shuffle_impl<T>(attrs, output, inputs[0], ctx);
    });
    return Status::ok;
}

} // namespace nnops::backend::cpu
