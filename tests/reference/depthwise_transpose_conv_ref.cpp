/// @file depthwise_transpose_conv_ref.cpp
/// @brief Naive CPU reference for depthwise transposed 2D convolution
///        (channel-first).
///
/// The correctness baseline for both the channel-first and the channel-last
/// kernels: plain nested loops over (batch, channel, output pixel, tap), one
/// task per (batch, channel) pair. The channel-last tests feed the same values
/// in the other order and compare after converting the result back, so this
/// file only ever reads the channel-first [C, 1, KH, KW] filter.
///
/// The gather form matches the kernel's indexing exactly: output (oh, ow) reads
/// input (ih, iw) where oh + PH - kh·DH == ih·SH, so no output is ever visited
/// twice and add_to needs no save/restore pass. f32/f16 go through s_load /
/// s_store, and the bias, the epilogue and add_to run in the output's domain.

#include "nnops/ops/depthwise_transpose_conv.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "simd_kernel/simd_epilogue.hpp"

#include <cstdint>

namespace nnops::backend::cpu::reference {

using nnops::simd::s_load;
using nnops::simd::s_store;

namespace {

template <typename T>
void dwtconv_ref_impl(const DepthwiseTransposeConv2DAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t N  = input.shape(0);
    const int64_t C  = input.shape(1);
    const int64_t IH = input.shape(2);
    const int64_t IW = input.shape(3);

    const int64_t OH = output.shape(2);
    const int64_t OW = output.shape(3);

    const int64_t KH = attrs.kernel_size[0];
    const int64_t KW = attrs.kernel_size[1];
    const int64_t SH = attrs.stride[0];
    const int64_t SW = attrs.stride[1];
    const int64_t DH = attrs.dilation[0];
    const int64_t DW = attrs.dilation[1];
    const int64_t PH = attrs.padding[0];
    const int64_t PW = attrs.padding[1];

    const int64_t in_n   = input.stride_elems(0);
    const int64_t in_row = input.row_stride_elems();
    const int64_t in_ch  = input.channel_block_stride_elems();
    const int64_t out_n   = output.stride_elems(0);
    const int64_t out_row = output.row_stride_elems();
    const int64_t out_ch  = output.channel_block_stride_elems();
    const int64_t w_ch = weight.stride_elems(0);
    const int64_t w_kh = weight.stride_elems(2);
    const int64_t w_kw = weight.stride_elems(3);

    const T* in_ptr  = input.ptr<T>();
    const T* w_ptr   = weight.ptr<T>();
    const T* b_ptr   = has_bias ? inputs[2].ptr<T>() : nullptr;
    T* out_ptr = output.ptr<T>();

    const auto compute_channel = [&](int64_t n, int64_t c) {
        const T* in_c  = in_ptr + n * in_n + c * in_ch;
        T* out_c = out_ptr + n * out_n + c * out_ch;
        const T* w_c = w_ptr + c * w_ch;
        const float bias = has_bias ? s_load(&b_ptr[c]) : 0.0f;

        for (int64_t oh = 0; oh < OH; ++oh) {
            T* orow = out_c + oh * out_row;
            for (int64_t ow = 0; ow < OW; ++ow) {
                float acc = 0.0f;
                for (int64_t kh = 0; kh < KH; ++kh) {
                    const int64_t num_h = oh + PH - kh * DH;
                    if (num_h < 0 || num_h % SH != 0) { continue; }
                    const int64_t ih = num_h / SH;
                    if (ih >= IH) { continue; }
                    const T* xrow = in_c + ih * in_row;
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        const int64_t num_w = ow + PW - kw * DW;
                        if (num_w < 0 || num_w % SW != 0) { continue; }
                        const int64_t iw = num_w / SW;
                        if (iw >= IW) { continue; }
                        acc += s_load(&xrow[iw]) *
                               s_load(&w_c[kh * w_kh + kw * w_kw]);
                    }
                }
                acc += bias;
                acc = nnops::apply_epilogue(attrs.epilogue, acc);
                if (attrs.add_to) { acc += s_load(&orow[ow]); }
                s_store(&orow[ow], acc);
            }
        }
    };

    ctx.cpu.run(0, N * C, [&](int64_t task) {
        compute_channel(task / C, task % C);
    });
}

}  // anonymous namespace

void depthwise_transpose_conv_ref(const DepthwiseTransposeConv2DAttributes& attrs,
                                  TensorView& output,
                                  std::span<const TensorView> inputs,
                                  const ComputeContext& ctx,
                                  void* /*workspace*/)
{
    if (inputs[0].data_type() == DataType::f16) {
        dwtconv_ref_impl<half>(attrs, output, inputs, ctx);
    } else {
        dwtconv_ref_impl<float>(attrs, output, inputs, ctx);
    }
}

}  // namespace nnops::backend::cpu::reference
