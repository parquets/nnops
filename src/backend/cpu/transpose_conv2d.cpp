/// @file transpose_conv2d.cpp
/// @brief NCHWC8 SIMD-accelerated CPU implementation of 2D transposed convolution.
///
/// Input/output are channel-packed (NCHWC8). Weights are planar NCHW [IC, OC/G, KH, KW].
/// Algorithm: scatter-add (input-driven) — each input pixel x kernel weight is
/// accumulated into the appropriate output region. This is the natural transpose
/// of regular convolution's gather approach.
///
/// Per-(N, C8) parallel dispatch, following the depthwise_conv.cpp pattern.

#include "nnops/ops/transpose_conv2d.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "simd_kernel/simd_epilogue.hpp"
#include "common/dtype_dispatch.hpp"

namespace nnops::backend::cpu {

using namespace nnops::simd;

template <typename T>
void tconv2d_impl_nchwc8(
    const TransposeConv2DAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    // Input: NCHWC8 [N, ceil(IC/8), IH, IW, 8]
    const int64_t N   = input.shape(0);
    const int64_t IC8 = input.num_channel_blocks();
    const int64_t IH  = input.shape(2);
    const int64_t IW  = input.shape(3);
    const int64_t in_pack  = input.channel_pack_size();   // 8
    const int64_t in_row_stride  = input.row_stride_elems();

    // Weight: planar NCHW [IC, OC/G, KH, KW]
    // Derive actual IC/OC from weight shape (more reliable than C8 * pack_size)
    const int64_t IC       = weight.shape(0);
    const int64_t OC_per_G = weight.shape(1);
    const int64_t KH = attrs.kernel_size[0];
    const int64_t KW = attrs.kernel_size[1];

    const int64_t G = attrs.groups;
    const int64_t OC = OC_per_G * G;
    const int64_t IC_per_G = IC / G;

    // Output: NCHWC8 [N, ceil(OC/8), OH, OW, 8]
    const int64_t OC8 = output.num_channel_blocks();
    const int64_t OH  = output.shape(2);
    const int64_t OW  = output.shape(3);
    const int64_t out_pack = output.channel_pack_size();  // 8
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_ch_stride  = output.channel_block_stride_elems();

    const int64_t SH = attrs.stride[0];
    const int64_t SW = attrs.stride[1];
    const int64_t DH = attrs.dilation[0];
    const int64_t DW = attrs.dilation[1];
    const int64_t PH = attrs.padding[0];
    const int64_t PW = attrs.padding[1];

    auto* out_ptr = output.ptr<T>();
    const auto* in_ptr  = input.ptr<T>();
    const auto* w_ptr   = weight.ptr<T>();
    const auto* b_ptr   = has_bias ? inputs[2].ptr<T>() : nullptr;

    const T* type_tag = nullptr;  // for v_zero / v_set1 overload resolution

    // Per-(N, C8) compute lambda
    const auto compute_sample_c8 = [&](int64_t n, int64_t oc8) {
        const int64_t oc_start = oc8 * out_pack;

        // Determine output channel range for this C8 block
        const int64_t oc_end = std::min(oc_start + out_pack, OC);
        const int64_t num_oc = oc_end - oc_start;  // may be < out_pack for last block

        // ---- Zero-initialize output region for this (n, oc8) block ----
        {
            T* out_base = out_ptr + n * OC8 * out_ch_stride + oc8 * out_ch_stride;
            for (int64_t oh = 0; oh < OH; ++oh) {
                T* out_row = out_base + oh * out_row_stride;
                for (int64_t ow = 0; ow < OW; ++ow) {
                    for (int64_t l = 0; l < out_pack; ++l) {
                        out_row[ow * out_pack + l] = T{};
                    }
                }
            }
        }

        // ---- Scatter-add ----
        // For each group, iterate over all input channels and scatter to output
        for (int64_t g = 0; g < G; ++g) {
            const int64_t ic_start = g * IC_per_G;
            const int64_t g_oc_start = g * OC_per_G;

            for (int64_t ic = ic_start; ic < ic_start + IC_per_G; ++ic) {
                const int64_t ic8 = ic / in_pack;
                const int64_t in_lane = ic % in_pack;

                for (int64_t oc_local = 0; oc_local < OC_per_G; ++oc_local) {
                    const int64_t oc = g_oc_start + oc_local;

                    if (oc < oc_start || oc >= oc_end) { continue; }

                    const int64_t w_base = ((ic * OC_per_G + oc_local) * KH) * KW;

                    for (int64_t ih = 0; ih < IH; ++ih) {
                        const int64_t in_row = (n * IC8 + ic8) * IH + ih;

                        for (int64_t iw = 0; iw < IW; ++iw) {
                            const float in_val = s_load(&in_ptr[
                                in_row * in_row_stride + iw * in_pack + in_lane]);
                            if (in_val == 0.0f) { continue; }

                            for (int64_t kh = 0; kh < KH; ++kh) {
                                const int64_t oh = ih * SH + kh * DH - PH;
                                if (oh < 0 || oh >= OH) { continue; }

                                for (int64_t kw = 0; kw < KW; ++kw) {
                                    const int64_t ow = iw * SW + kw * DW - PW;
                                    if (ow < 0 || ow >= OW) { continue; }

                                    const float w_val = s_load(&w_ptr[w_base + kh * KW + kw]);
                                    const int64_t out_lane = oc % out_pack;
                                    const int64_t out_idx =
                                        (n * OC8 + oc8) * OH * out_row_stride
                                        + oh * out_row_stride
                                        + ow * out_pack + out_lane;
                                    float acc = s_load(&out_ptr[out_idx]);
                                    acc += in_val * w_val;
                                    s_store(&out_ptr[out_idx], acc);
                                }
                            }
                        }
                    }
                }
            }
        }

        // ---- Bias + Epilogue (SIMD for full C8 blocks, scalar for tail) ----
        T* out_base = out_ptr + n * OC8 * out_ch_stride + oc8 * out_ch_stride;

        if (num_oc == out_pack) {
            // Full C8 block — use SIMD
            auto vbias = b_ptr
                ? v_load(b_ptr + oc8 * out_pack)
                : v_zero(type_tag);

            for (int64_t oh = 0; oh < OH; ++oh) {
                T* out_row = out_base + oh * out_row_stride;
                for (int64_t ow = 0; ow < OW; ++ow) {
                    T* out_elm = out_row + ow * out_pack;
                    auto vout = v_load(out_elm);
                    vout = v_add(vout, vbias);
                    vout = apply_epilogue_vec(attrs.epilogue, type_tag, vout);
                    v_store(out_elm, vout);
                }
            }
        } else {
            // Partial C8 block — scalar
            for (int64_t loc = 0; loc < num_oc; ++loc) {
                const int64_t oc = oc_start + loc;
                const float bval = b_ptr ? s_load(&b_ptr[oc]) : 0.0f;

                for (int64_t oh = 0; oh < OH; ++oh) {
                    T* out_row = out_base + oh * out_row_stride;
                    for (int64_t ow = 0; ow < OW; ++ow) {
                        T* out_elm = out_row + ow * out_pack + loc;
                        float val = s_load(out_elm) + bval;
                        val = apply_epilogue(attrs.epilogue, val, oc);
                        s_store(out_elm, val);
                    }
                }
            }
        }
    };

    // ---- add_to handling ----
    // Scatter-add is inherently accumulation-based, so we handle add_to by
    // saving/restoring the output region. For the typical case (add_to=false),
    // zeroing is sufficient.
    //
    // For add_to=true: save existing, zero, scatter, then add bias+epilogue result
    // to the saved original. This is done per-(n, oc8) to keep temp buffer small.
    if (attrs.add_to) {
        const int64_t out_block_elems = OH * out_row_stride;
        std::vector<T> saved(static_cast<size_t>(out_block_elems));

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t oc8 = 0; oc8 < OC8; ++oc8) {
                const int64_t oc_start = oc8 * out_pack;
                const int64_t oc_end = std::min(oc_start + out_pack, OC);
                const int64_t num_oc = oc_end - oc_start;

                T* out_block = out_ptr + n * OC8 * out_ch_stride + oc8 * out_ch_stride;
                for (int64_t i = 0; i < out_block_elems; ++i) {
                    saved[i] = out_block[i];
                }

                compute_sample_c8(n, oc8);

                for (int64_t oh = 0; oh < OH; ++oh) {
                    T* out_row = out_block + oh * out_row_stride;
                    const T* saved_row = saved.data() + oh * out_row_stride;
                    for (int64_t ow = 0; ow < OW; ++ow) {
                        for (int64_t l = 0; l < num_oc; ++l) {
                            const int64_t idx = ow * out_pack + l;
                            float val = s_load(&out_row[idx]) + s_load(&saved_row[idx]);
                            s_store(&out_row[idx], val);
                        }
                    }
                }
            }
        }
        return;
    }

    // ---- Normal case (add_to=false): parallel over N * OC8 ----
    const int64_t N_OC8 = N * OC8;
    ctx.cpu.run(0, N_OC8,
            [&](int64_t tid) {
                compute_sample_c8(tid / OC8, tid % OC8);
            });
}

// ============================================================
// Entry point with dtype dispatch
// ============================================================

void transpose_conv2d_cpu(const TransposeConv2DAttributes& attrs,
                           TensorView& output,
                           std::span<const TensorView> inputs,
                           const ComputeContext& ctx,
                           void* /*workspace*/)
{
    dispatch_f32_f16(inputs[0].data_type(), "transpose_conv2d_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        tconv2d_impl_nchwc8<T>(attrs, output, inputs, ctx);
    });
}

}  // namespace nnops::backend::cpu
