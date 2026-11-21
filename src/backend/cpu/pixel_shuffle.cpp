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
///
/// The 2×2 f32 case — the ViT / VLM patch merge every model actually runs — is
/// a 4×4 in-register transpose per four channels: four 128-bit loads, one
/// v_transpose_4x4_f32x4, four 128-bit stores. That turns the stride-4 scatter
/// (SpaceToDepth) and gather (PixelShuffle DCR) into contiguous vector traffic,
/// ~4x and ~2x the scalar forms on an M4. Other block sizes and element types
/// keep the generic walk.

#include "nnops/ops/pixel_shuffle.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/vec_f32x4.hpp"
#include "common/dtype_dispatch.hpp"

#include <cstdint>
#include <type_traits>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

/// b = 2, f32: out[c·4 + q] = in_q[c], with in_0..in_3 the four source pixels
/// of one output pixel. Four channels at a time become one 4x4 transpose and
/// four contiguous 16-byte stores.
void space_to_depth_b2_f32(const TensorView& input, TensorView& output,
                           const ComputeContext& ctx)
{
    const int64_t N = input.shape(0);
    const int64_t H = input.shape(1);
    const int64_t W = input.shape(2);
    const int64_t C = input.shape(3);
    const int64_t Ho = H / 2;
    const int64_t Wo = W / 2;

    const auto* x = input.ptr<float>();
    auto* y = output.ptr<float>();
    const int64_t xs0 = input.stride_elems(0);
    const int64_t xs1 = input.stride_elems(1);
    const int64_t xs2 = input.stride_elems(2);
    const int64_t ys0 = output.stride_elems(0);
    const int64_t ys1 = output.stride_elems(1);
    const int64_t ys2 = output.stride_elems(2);

    ctx.cpu.run(0, N * Ho, [&](int64_t row) {
        const int64_t n = row / Ho;
        const int64_t ho = row % Ho;
        float* y_row = y + n * ys0 + ho * ys1;
        for (int64_t wo = 0; wo < Wo; ++wo) {
            const float* r0 = x + n * xs0 + (2 * ho) * xs1 + (2 * wo) * xs2;
            const float* r1 = r0 + xs2;
            const float* r2 = r0 + xs1;
            const float* r3 = r2 + xs2;
            float* dst = y_row + wo * ys2;

            int64_t c = 0;
            for (; c + 4 <= C; c += 4) {
                auto v0 = v_load_f32x4(r0 + c);
                auto v1 = v_load_f32x4(r1 + c);
                auto v2 = v_load_f32x4(r2 + c);
                auto v3 = v_load_f32x4(r3 + c);
                v_transpose_4x4_f32x4(v0, v1, v2, v3);
                v_store(dst + 4 * c + 0, v0);
                v_store(dst + 4 * c + 4, v1);
                v_store(dst + 4 * c + 8, v2);
                v_store(dst + 4 * c + 12, v3);
            }
            for (; c < C; ++c) {
                dst[4 * c + 0] = r0[c];
                dst[4 * c + 1] = r1[c];
                dst[4 * c + 2] = r2[c];
                dst[4 * c + 3] = r3[c];
            }
        }
    });
}

/// b = 2, f32, DCR: for one source pixel, channels c·4 + q map to output pixel
/// q, channel c. Four output channels at a time come from one 4x4 transpose of
/// four source channel groups, one 16-byte store per output pixel.
void pixel_shuffle_dcr_b2_f32(const TensorView& input, TensorView& output,
                              const ComputeContext& ctx)
{
    const int64_t N = input.shape(0);
    const int64_t H = input.shape(1);
    const int64_t W = input.shape(2);
    const int64_t C_out = input.shape(3) / 4;
    const int64_t H_out = H * 2;
    const int64_t W_out = W * 2;

    const auto* x = input.ptr<float>();
    auto* y = output.ptr<float>();
    const int64_t xs0 = input.stride_elems(0);
    const int64_t xs1 = input.stride_elems(1);
    const int64_t xs2 = input.stride_elems(2);
    const int64_t ys0 = output.stride_elems(0);
    const int64_t ys1 = output.stride_elems(1);
    const int64_t ys2 = output.stride_elems(2);

    ctx.cpu.run(0, N * H, [&](int64_t row) {
        const int64_t n = row / H;
        const int64_t h = row % H;
        for (int64_t w = 0; w < W; ++w) {
            const float* src = x + n * xs0 + h * xs1 + w * xs2;
            float* d0 = y + n * ys0 + (2 * h) * ys1 + (2 * w) * ys2;
            float* d1 = d0 + ys2;
            float* d2 = d0 + ys1;
            float* d3 = d2 + ys2;

            int64_t c = 0;
            for (; c + 4 <= C_out; c += 4) {
                auto v0 = v_load_f32x4(src + 4 * (c + 0));
                auto v1 = v_load_f32x4(src + 4 * (c + 1));
                auto v2 = v_load_f32x4(src + 4 * (c + 2));
                auto v3 = v_load_f32x4(src + 4 * (c + 3));
                v_transpose_4x4_f32x4(v0, v1, v2, v3);
                v_store(d0 + c, v0);
                v_store(d1 + c, v1);
                v_store(d2 + c, v2);
                v_store(d3 + c, v3);
            }
            for (; c < C_out; ++c) {
                d0[c] = src[4 * c + 0];
                d1[c] = src[4 * c + 1];
                d2[c] = src[4 * c + 2];
                d3[c] = src[4 * c + 3];
            }
        }
    });
}

template <typename T>
void space_to_depth_impl(const SpaceToDepthAttributes& attrs, TensorView& output,
                         const TensorView& input, const ComputeContext& ctx)
{
    const int64_t b = attrs.block_size;
    if constexpr (std::is_same_v<T, float>) {
        if (b == 2) {
            space_to_depth_b2_f32(input, output, ctx);
            return;
        }
    }

    const int64_t N = input.shape(0);
    const int64_t H = input.shape(1);
    const int64_t W = input.shape(2);
    const int64_t C = input.shape(3);
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
    const int64_t b = attrs.block_size;
    const bool dcr = attrs.mode == PixelShuffleMode::Dcr;
    if constexpr (std::is_same_v<T, float>) {
        if (b == 2 && dcr) {
            pixel_shuffle_dcr_b2_f32(input, output, ctx);
            return;
        }
    }

    const int64_t N = input.shape(0);
    const int64_t H = input.shape(1);
    const int64_t W = input.shape(2);
    const int64_t C_out = input.shape(3) / (b * b);
    const int64_t H_out = H * b;
    const int64_t W_out = W * b;

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
    if (!for_each_data_type(inputs[0].data_type(), [&](auto tag) {
            using T = typename decltype(tag)::type;
            space_to_depth_impl<T>(attrs, output, inputs[0], ctx);
        })) {
        return Status::unsupported;
    }
    return Status::ok;
}

Status pixel_shuffle_cpu(const PixelShuffleAttributes& attrs, TensorView& output,
                         std::span<const TensorView> inputs, const ComputeContext& ctx,
                         void* /*workspace*/)
{
    if (!for_each_data_type(inputs[0].data_type(), [&](auto tag) {
            using T = typename decltype(tag)::type;
            pixel_shuffle_impl<T>(attrs, output, inputs[0], ctx);
        })) {
        return Status::unsupported;
    }
    return Status::ok;
}

} // namespace nnops::backend::cpu
