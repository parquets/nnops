/// @file depthwise_conv.cpp
/// @brief DepthwiseConv operator dispatch — connects class API to backend implementations.
///
/// Input/output are always NCHWC8/NCDHWC8. Weights are prepacked via prepackWeights()
/// before compute(). The prepack step converts raw NCHW/NCDHW weight [C, 1, (KD,) KH, KW]
/// to dense [C8, (KD,) KH, KW, 8] format for efficient SIMD access.
///
/// Spatial rank is auto-detected from input tensor rank (4→2D, 5→3D).

#include "nnops/ops/depthwise_conv.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace nnops {

using namespace simd;

// Forward declarations of backend kernel entry points
namespace backend::cpu::reference {
    void depthwise_conv_ref(const DepthwiseConvAttributes& attrs,
                             TensorView& output,
                             std::span<const TensorView> inputs,
                             const ComputeContext& ctx,
                             void* workspace);
}

namespace backend::cpu {
    void depthwise_conv_cpu(const DepthwiseConvAttributes& attrs,
                             TensorView& output,
                             std::span<const TensorView> inputs,
                             const ComputeContext& ctx,
                             void* workspace);
}

#ifdef NNOPS_HAS_CUDA
namespace backend::cuda {
    void depthwise_conv_cuda(const DepthwiseConvAttributes& attrs,
                              TensorView& output,
                              std::span<const TensorView> inputs,
                              const ComputeContext& ctx,
                              void* workspace);
}
#endif

// ============================================================
// Impl — holds the backend-bound kernel function pointer
// ============================================================
struct DepthwiseConv::Impl {
    using KernelFn = void (*)(const DepthwiseConvAttributes&,
                               TensorView&,
                               std::span<const TensorView>,
                               const ComputeContext&,
                               void*);
    KernelFn kernel_fn = nullptr;
};

namespace {
auto resolve_depthwise_conv_kernel(Backend backend) -> DepthwiseConv::Impl::KernelFn
{
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::depthwise_conv_cpu;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return backend::cuda::depthwise_conv_cuda;
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return nullptr;
#endif
    }
    return nullptr;
}

// ============================================================
// Prepack helpers — SIMD-optimized weight & bias packing (2D + 3D)
// ============================================================
//
// Uses 8×8 SIMD transpose (same pattern as layout_convert.cpp):
//   For each C8 block, process 8 kw positions at once:
//     v_load 8 consecutive kw from each of 8 channels → v_transpose_8x8 → v_store.
//   Partial C8 / remainder kw fall back to scalar per-position.

template <typename T>
void prepack_dwconv_weight_2d(const TensorView& weight_nchw,
                                TensorView& packed_out)
{
    const int64_t C  = weight_nchw.shape(0);
    const int64_t KH = weight_nchw.shape(2);
    const int64_t KW = weight_nchw.shape(3);
    const int64_t C8 = (C + 7) / 8;
    const int64_t ch_stride = KH * KW;

    const auto* src = weight_nchw.ptr<T>();
    auto* dst = packed_out.ptr<T>();
    auto vzero = v_set1(src, 0);

    for (int64_t c8 = 0; c8 < C8; ++c8) {
        int64_t c_base = c8 * 8;
        for (int64_t kh = 0; kh < KH; ++kh) {
            int64_t kw = 0;
            // Process 8 kw at a time with 8×8 transpose
            for (; kw + 8 <= KW; kw += 8) {
                auto c0 = c_base + 0 < C ? v_load(&src[(c_base + 0) * ch_stride + kh * KW + kw]) : vzero;
                auto c1 = c_base + 1 < C ? v_load(&src[(c_base + 1) * ch_stride + kh * KW + kw]) : vzero;
                auto c2 = c_base + 2 < C ? v_load(&src[(c_base + 2) * ch_stride + kh * KW + kw]) : vzero;
                auto c3 = c_base + 3 < C ? v_load(&src[(c_base + 3) * ch_stride + kh * KW + kw]) : vzero;
                auto c4 = c_base + 4 < C ? v_load(&src[(c_base + 4) * ch_stride + kh * KW + kw]) : vzero;
                auto c5 = c_base + 5 < C ? v_load(&src[(c_base + 5) * ch_stride + kh * KW + kw]) : vzero;
                auto c6 = c_base + 6 < C ? v_load(&src[(c_base + 6) * ch_stride + kh * KW + kw]) : vzero;
                auto c7 = c_base + 7 < C ? v_load(&src[(c_base + 7) * ch_stride + kh * KW + kw]) : vzero;

                v_transpose_8x8(c0, c1, c2, c3, c4, c5, c6, c7);

                v_store(&dst[(c8 * KH * KW + kh * KW + kw + 0) * 8], c0);
                v_store(&dst[(c8 * KH * KW + kh * KW + kw + 1) * 8], c1);
                v_store(&dst[(c8 * KH * KW + kh * KW + kw + 2) * 8], c2);
                v_store(&dst[(c8 * KH * KW + kh * KW + kw + 3) * 8], c3);
                v_store(&dst[(c8 * KH * KW + kh * KW + kw + 4) * 8], c4);
                v_store(&dst[(c8 * KH * KW + kh * KW + kw + 5) * 8], c5);
                v_store(&dst[(c8 * KH * KW + kh * KW + kw + 6) * 8], c6);
                v_store(&dst[(c8 * KH * KW + kh * KW + kw + 7) * 8], c7);
            }
            // Remainder kw: scalar per-position
            for (; kw < KW; ++kw) {
                T tmp[8] = {};
                for (int64_t lane = 0; lane < 8; ++lane) {
                    int64_t c = c_base + lane;
                    if (c < C) {
                        tmp[lane] = src[c * ch_stride + kh * KW + kw];
                    }
                }
                v_store(&dst[(c8 * KH * KW + kh * KW + kw) * 8], v_load(tmp));
            }
        }
    }
}

template <typename T>
void prepack_dwconv_weight_3d(const TensorView& weight_ncdhw,
                                TensorView& packed_out)
{
    const int64_t C  = weight_ncdhw.shape(0);
    const int64_t KD = weight_ncdhw.shape(2);
    const int64_t KH = weight_ncdhw.shape(3);
    const int64_t KW = weight_ncdhw.shape(4);
    const int64_t C8 = (C + 7) / 8;
    const int64_t ch_stride = KD * KH * KW;

    const auto* src = weight_ncdhw.ptr<T>();
    auto* dst = packed_out.ptr<T>();
    auto vzero = v_set1(src, 0);

    for (int64_t c8 = 0; c8 < C8; ++c8) {
        int64_t c_base = c8 * 8;
        for (int64_t kd = 0; kd < KD; ++kd) {
            for (int64_t kh = 0; kh < KH; ++kh) {
                int64_t kw = 0;
                for (; kw + 8 <= KW; kw += 8) {
                    auto c0 = c_base + 0 < C ? v_load(&src[(c_base + 0) * ch_stride + kd * KH * KW + kh * KW + kw]) : vzero;
                    auto c1 = c_base + 1 < C ? v_load(&src[(c_base + 1) * ch_stride + kd * KH * KW + kh * KW + kw]) : vzero;
                    auto c2 = c_base + 2 < C ? v_load(&src[(c_base + 2) * ch_stride + kd * KH * KW + kh * KW + kw]) : vzero;
                    auto c3 = c_base + 3 < C ? v_load(&src[(c_base + 3) * ch_stride + kd * KH * KW + kh * KW + kw]) : vzero;
                    auto c4 = c_base + 4 < C ? v_load(&src[(c_base + 4) * ch_stride + kd * KH * KW + kh * KW + kw]) : vzero;
                    auto c5 = c_base + 5 < C ? v_load(&src[(c_base + 5) * ch_stride + kd * KH * KW + kh * KW + kw]) : vzero;
                    auto c6 = c_base + 6 < C ? v_load(&src[(c_base + 6) * ch_stride + kd * KH * KW + kh * KW + kw]) : vzero;
                    auto c7 = c_base + 7 < C ? v_load(&src[(c_base + 7) * ch_stride + kd * KH * KW + kh * KW + kw]) : vzero;

                    v_transpose_8x8(c0, c1, c2, c3, c4, c5, c6, c7);

                    v_store(&dst[(c8 * KD * KH * KW + kd * KH * KW + kh * KW + kw + 0) * 8], c0);
                    v_store(&dst[(c8 * KD * KH * KW + kd * KH * KW + kh * KW + kw + 1) * 8], c1);
                    v_store(&dst[(c8 * KD * KH * KW + kd * KH * KW + kh * KW + kw + 2) * 8], c2);
                    v_store(&dst[(c8 * KD * KH * KW + kd * KH * KW + kh * KW + kw + 3) * 8], c3);
                    v_store(&dst[(c8 * KD * KH * KW + kd * KH * KW + kh * KW + kw + 4) * 8], c4);
                    v_store(&dst[(c8 * KD * KH * KW + kd * KH * KW + kh * KW + kw + 5) * 8], c5);
                    v_store(&dst[(c8 * KD * KH * KW + kd * KH * KW + kh * KW + kw + 6) * 8], c6);
                    v_store(&dst[(c8 * KD * KH * KW + kd * KH * KW + kh * KW + kw + 7) * 8], c7);
                }
                for (; kw < KW; ++kw) {
                    T tmp[8] = {};
                    for (int64_t lane = 0; lane < 8; ++lane) {
                        int64_t c = c_base + lane;
                        if (c < C) {
                            tmp[lane] = src[c * ch_stride + kd * KH * KW + kh * KW + kw];
                        }
                    }
                    v_store(&dst[(c8 * KD * KH * KW + kd * KH * KW + kh * KW + kw) * 8], v_load(tmp));
                }
            }
        }
    }
}

// ============================================================
// Byte-copy prepack for s8/u8 weights (8 channels = 8 bytes)
// ============================================================
//
// int8/uint8 `v_load` returns 16-lane vectors, so the 8×8 float transpose
// above is not valid. These paths copy 8 channel bytes per kernel position
// directly. Pad lanes (c >= C) are filled with 0 — they are neutralized by the
// quantized backend via w_scale8[lane] = 0, so the byte value is irrelevant.

template <typename T>
void prepack_dwconv_weight_2d_byte(const TensorView& weight_nchw,
                                   TensorView& packed_out)
{
    const int64_t C  = weight_nchw.shape(0);
    const int64_t KH = weight_nchw.shape(2);
    const int64_t KW = weight_nchw.shape(3);
    const int64_t C8 = (C + 7) / 8;
    const int64_t ch_stride = KH * KW;

    const auto* src = weight_nchw.ptr<T>();
    auto* dst = packed_out.ptr<T>();

    for (int64_t c8 = 0; c8 < C8; ++c8) {
        const int64_t c_base = c8 * 8;
        for (int64_t kh = 0; kh < KH; ++kh) {
            for (int64_t kw = 0; kw < KW; ++kw) {
                T* out = &dst[(c8 * KH * KW + kh * KW + kw) * 8];
                for (int64_t lane = 0; lane < 8; ++lane) {
                    const int64_t c = c_base + lane;
                    out[lane] = (c < C)
                        ? src[c * ch_stride + kh * KW + kw]
                        : static_cast<T>(0);
                }
            }
        }
    }
}

template <typename T>
void prepack_dwconv_weight_3d_byte(const TensorView& weight_ncdhw,
                                   TensorView& packed_out)
{
    const int64_t C  = weight_ncdhw.shape(0);
    const int64_t KD = weight_ncdhw.shape(2);
    const int64_t KH = weight_ncdhw.shape(3);
    const int64_t KW = weight_ncdhw.shape(4);
    const int64_t C8 = (C + 7) / 8;
    const int64_t ch_stride = KD * KH * KW;

    const auto* src = weight_ncdhw.ptr<T>();
    auto* dst = packed_out.ptr<T>();

    for (int64_t c8 = 0; c8 < C8; ++c8) {
        const int64_t c_base = c8 * 8;
        for (int64_t kd = 0; kd < KD; ++kd) {
            for (int64_t kh = 0; kh < KH; ++kh) {
                for (int64_t kw = 0; kw < KW; ++kw) {
                    T* out = &dst[(c8 * KD * KH * KW + kd * KH * KW + kh * KW + kw) * 8];
                    for (int64_t lane = 0; lane < 8; ++lane) {
                        const int64_t c = c_base + lane;
                        out[lane] = (c < C)
                            ? src[c * ch_stride + kd * KH * KW + kh * KW + kw]
                            : static_cast<T>(0);
                    }
                }
            }
        }
    }
}

template <typename T>
void prepack_dwconv_bias_impl(const TensorView& bias_nchw,
                                TensorView& packed_out)
{
    const int64_t C  = bias_nchw.shape(0);
    const int64_t C8 = (C + 7) / 8;

    const auto* src = bias_nchw.ptr<T>();
    auto* dst = packed_out.ptr<T>();

    for (int64_t c8 = 0; c8 < C8; ++c8) {
        int64_t c_base = c8 * 8;
        int64_t valid = std::min<int64_t>(8, C - c_base);
        if (valid == 8) {
            // Full C8: single vector load/store
            auto v = v_load(&src[c_base]);
            v_store(&dst[c8 * 8], v);
        } else {
            // Partial C8: scalar fallback
            T tmp[8] = {};
            for (int64_t lane = 0; lane < valid; ++lane) {
                tmp[lane] = src[c_base + lane];
            }
            v_store(&dst[c8 * 8], v_load(tmp));
        }
    }
}

void prepack_dwconv_weight(const TensorView& weight,
                            TensorView& packed_out,
                            int64_t srank)
{
    switch (weight.data_type()) {
    case DataType::f32:
        if (srank == 3) {
            prepack_dwconv_weight_3d<float>(weight, packed_out);
        }
        else {
            prepack_dwconv_weight_2d<float>(weight, packed_out);
        }
        return;
    case DataType::f16:
        if (srank == 3) {
            prepack_dwconv_weight_3d<backend::cpu::half>(weight, packed_out);
        }
        else {
            prepack_dwconv_weight_2d<backend::cpu::half>(weight, packed_out);
        }
        return;
    case DataType::s8:
        if (srank == 3) {
            prepack_dwconv_weight_3d_byte<int8_t>(weight, packed_out);
        }
        else {
            prepack_dwconv_weight_2d_byte<int8_t>(weight, packed_out);
        }
        return;
    case DataType::u8:
        if (srank == 3) {
            prepack_dwconv_weight_3d_byte<uint8_t>(weight, packed_out);
        }
        else {
            prepack_dwconv_weight_2d_byte<uint8_t>(weight, packed_out);
        }
        return;
    default:
        NNOPS_ASSERT(!"prepack_dwconv_weight: unsupported dtype");
    }
}

void prepack_dwconv_bias(const TensorView& bias_nchw,
                          TensorView& packed_out)
{
    switch (bias_nchw.data_type()) {
    case DataType::f32:
        prepack_dwconv_bias_impl<float>(bias_nchw, packed_out);
        return;
    case DataType::f16:
        prepack_dwconv_bias_impl<backend::cpu::half>(bias_nchw, packed_out);
        return;
    default:
        NNOPS_ASSERT(!"prepack_dwconv_bias: unsupported dtype");
    }
}

}  // anonymous namespace

// ============================================================
// Factory
// ============================================================
std::unique_ptr<DepthwiseConv> DepthwiseConv::create(
    const DepthwiseConvAttributes& attrs, Backend backend)
{
    return std::unique_ptr<DepthwiseConv>(new DepthwiseConv(attrs, backend));
}

// ============================================================
// Constructor
// ============================================================
DepthwiseConv::DepthwiseConv(const DepthwiseConvAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_depthwise_conv_kernel(backend);
}

// ============================================================
// getOutputTensorDesc
// ============================================================
std::vector<TensorDesc> DepthwiseConv::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    return {depthwise_conv_output_shape(
        attrs_.kernel_size, attrs_.stride, attrs_.dilation, attrs_.padding,
        0 /* NOTSET */, inputs)};
}

// ============================================================
// prepackWeights — dual-behavior: empty → metadata, filled → pack
// ============================================================
void DepthwiseConv::prepackWeights(std::span<const TensorView> inputs,
                                    std::span<TensorView> outputs,
                                    const ComputeContext& ctx)
{
    (void)ctx;
    NNOPS_ASSERT(inputs.size() >= 1 && inputs.size() <= 2);
    const bool has_bias = inputs.size() > 1;
    const int64_t num_outputs = has_bias ? 2 : 1;
    NNOPS_ASSERT(outputs.size() == static_cast<size_t>(num_outputs));

    const auto& weight = inputs[0];
    const int64_t C  = weight.shape(0);
    const int64_t wrank = weight.rank();
    const int64_t srank = wrank - 2;  // 2D: rank 4, 3D: rank 5
    const int64_t KD = (srank == 3) ? weight.shape(2) : 1;
    const int64_t KH = weight.shape(srank);
    const int64_t KW = weight.shape(srank + 1);
    const int64_t C8 = (C + 7) / 8;
    const DataType dtype = weight.data_type();
    const int64_t elem_size = static_cast<int64_t>(data_type_size(dtype));
    // Bias keeps its own dtype (f32 for quantized int8 convs).
    const DataType bdtype = has_bias ? inputs[1].data_type() : dtype;
    const int64_t belem_size = static_cast<int64_t>(data_type_size(bdtype));
    // Preserve weight PerChannel quant params on the packed weight so the
    // backend can index scale_data/zero_point_data by input channel C.
    const QuantParams w_qp = weight.quant_params();

    if (outputs[0].is_empty()) {
        // ---- Query mode: fill metadata so caller can allocate ----

        if (srank == 3) {
            // Packed weight: [C8, KD, KH, KW, 8]
            const int64_t w_shape[] = {C8, KD, KH, KW, 8};
            const int64_t w_pitch = 8 * elem_size;  // innermost dim = 8 lanes
            outputs[0] = TensorView(
                std::span<const int64_t>(w_shape, 5), dtype,
                nullptr, w_pitch, TensorLayout::PackedWeight, w_qp);
        } else {
            // Packed weight: [C8, KH, KW, 8]
            const int64_t w_shape[] = {C8, KH, KW, 8};
            const int64_t w_pitch = 8 * elem_size;
            outputs[0] = TensorView(
                std::span<const int64_t>(w_shape, 4), dtype,
                nullptr, w_pitch, TensorLayout::PackedWeight, w_qp);
        }

        // Packed bias: [C8, 8]
        if (has_bias) {
            const int64_t b_shape[] = {C8, 8};
            const int64_t b_pitch = 8 * belem_size;
            outputs[1] = TensorView(
                std::span<const int64_t>(b_shape, 2), bdtype,
                nullptr, b_pitch, TensorLayout::PackedWeight);
        }
    } else {
        // ---- Pack mode: fill the buffer ----
        prepack_dwconv_weight(weight, outputs[0], srank);

        if (has_bias) {
            prepack_dwconv_bias(inputs[1], outputs[1]);
        }
    }
}

// ============================================================
// compute — direct dispatch to backend (input/output already NCHWC8/NCDHWC8)
// ============================================================
void DepthwiseConv::compute(std::span<TensorView> outputs,
                             std::span<const TensorView> inputs,
                             const ComputeContext& ctx,
                             void* workspace)
{
    NNOPS_ASSERT(inputs.size() >= 2);
    NNOPS_ASSERT(inputs.size() <= 3);  // input, prepacked_weight [, prepacked_bias]
    NNOPS_ASSERT(outputs.size() == 1);
    auto& output = outputs[0];
    NNOPS_ASSERT(!output.is_empty());
    NNOPS_ASSERT(!inputs[0].is_empty());
    NNOPS_ASSERT(!inputs[1].is_empty());
    NNOPS_ASSERT(is_layout_supported(inputs[0].layout(), LayoutSupport::PackedOnly));

    impl_->kernel_fn(attrs_, output, inputs, ctx, workspace);
}

}  // namespace nnops
