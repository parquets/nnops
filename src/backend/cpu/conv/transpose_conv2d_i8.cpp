/// @file transpose_conv2d_i8.cpp
/// @brief CPU TransposeConv2D on NHWC — the quantized routes.
///
/// The same output-driven gather as the float route, with an s8 weight
/// (per-channel) against a u8/s8/f32/f16 activation and an int32 accumulate:
/// the quantized GEMM (MatMulInteger) reduces the input channels and withdraws
/// both zero points, so the scatter adds raw int32 and the merge does what the
/// fp merge did — bias, clip, epilogue — under a dequantize or a requantize.
/// The geometry does not change, which is why each scatter here mirrors the fp
/// scatter element for element.
///
/// Three routes, chosen per call:
///   - the depthwise pixel kernel (one channel per group), which needs neither a
///     col matrix nor a weight repack;
///   - the col-matrix GEMM (one per (group, oc chunk, batch)), the dense filter's
///     route;
///   - the plain scalar gather, correct for any shape, which takes whatever the
///     two above decline.
///
/// The activation is normalised to one s8 plane per call so the GEMM sees a
/// single zero-point convention: an fp activation is quantized into it (its
/// act_quant_dtype picks the clamp range), a u8 activation is shifted to s8 with
/// its zero point moved by -128, and a dense s8 activation is read in place.
///
/// The weight keeps its per-channel parameters. The GEMM's reduction axis is the
/// input channel, so an output channel owns karea consecutive rows (taps-major)
/// or columns (NHWC) and its (scale, zero point) has to be replicated across
/// them — the one thing the operand views need that the raw tensors do not have.
/// Replication is cheap and per chunk: the taps of one channel always sit
/// together.
///
/// The float routes and the entry point are transpose_conv2d.cpp; the geometry
/// both families read is transpose_conv2d_internal.h.

#include "transpose_conv2d_internal.h"
#include "backend/cpu/simd_kernel/simd_col2im.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"
#include "backend/cpu/simd_kernel/simd_quant.hpp" // quant_kernel::quantization
#include "backend/cpu/matmul/matmul.h"
#include "backend/cpu/common/dtype_dispatch.hpp"
#include "backend/cpu/common/memory_pool.hpp"

#include <algorithm>
#include <cstring>
#include <type_traits>

namespace nnops::backend::cpu {

using namespace nnops::simd;
using namespace nnops::kernel;

namespace {

// =========================================================================
//  Quantized route — s8 weight (per-channel), u8/s8/f32/f16 activation
// =========================================================================
//
// The float route's two col-matrix forms, with the int32 accumulate as the col: the
// quantized GEMM (MatMulInteger) reduces the input channels and withdraws both
// zero points, so the scatter adds raw int32 and the merge does what the fp
// merge did — bias, clip, epilogue — under a dequantize or a requantize. The
// geometry does not change, which is why each scatter here is the fp scatter
// element for element.
//
// The activation is normalised to one s8 plane per call so the GEMM sees a
// single zero-point convention: an fp activation is quantized into it (its
// act_quant_dtype picks the clamp range), and a u8 activation is shifted to s8
// with its zero point moved by -128 — the same value, one convention. A
// pre-quantized s8 activation whose rows are dense is read in place.
//
// The weight keeps its per-channel parameters. The GEMM's reduction axis is the
// input channel, so an output channel owns karea consecutive rows (taps-major)
// or columns (NHWC) and its (scale, zero point) has to be replicated across
// them — the one thing the operand views need that the raw tensors do not
// have. Replication is cheap and per chunk: the taps of one
// channel always sit together.

namespace k = nnops::kernel;

/// One (scale, zero point) for the whole tensor.
inline QuantParams qp_per_tensor(float scale, int32_t zero)
{
    QuantParams q;
    q.scale = scale;
    q.zero_point = zero;
    return q;
}

/// One (scale, zero point) per row ([OC/G][taps]) or column (NHWC).
inline QuantParams qp_per_channel(const float* scale, const int32_t* zero, int64_t n)
{
    QuantParams q;
    q.granularity = QuantGranularity::PerChannel;
    q.scale_data = scale;
    q.zero_point_data = zero;
    q.num_scales = n;
    return q;
}

/// The activation the int8 GEMM reads: an s8 plane (or the input itself when it
/// is already s8 and dense) and the single (scale, zero point) it carries.
struct ActPlane {
    const int8_t* ptr = nullptr;
    int64_t n_stride = 0; ///< elements per batch item
    float scale = 1.0f;
    int32_t zero = 0;
};

/// Quantize/shift the activation into a dense pixel-major s8 plane
/// [N][IH·IW][IC]. @p zero is the plane's zero point (-128 from the stored one
/// when the storage was unsigned).
template <class TAct>
void build_activation_plane(const TensorView& in, int8_t* dst, float scale, int32_t zero,
                            int64_t IC, int64_t IH, int64_t IW, const ComputeContext& ctx)
{
    const int64_t N = in.shape(0);
    const int64_t rs = in.row_stride_elems();
    const int64_t ns = in.stride_elems(0);
    const int64_t row_len = IW * IC;
    const int64_t rows_n = IH;
    const int64_t total = N * rows_n;
    const TAct* src = in.ptr<TAct>();
    const float s = scale;
    const float z = static_cast<float>(zero);

    const auto run_rows = [&](int64_t beg, int64_t end) {
        for (int64_t r = beg; r < end; ++r) {
            const int64_t nb = r / rows_n;
            const int64_t i = r - nb * rows_n;
            const TAct* sp = src + nb * ns + i * rs;
            int8_t* dp = dst + (nb * rows_n + i) * row_len;
            if constexpr (std::is_same_v<TAct, int8_t>) {
                std::memcpy(dp, sp, static_cast<size_t>(row_len));
            } else if constexpr (std::is_same_v<TAct, uint8_t>) {
                for (int64_t k = 0; k < row_len; ++k) {
                    dp[k] = static_cast<int8_t>(static_cast<uint8_t>(sp[k]) ^ 0x80u);
                }
            } else {
                k::quant_kernel::quantization<int8_t>(
                    static_cast<int>(1), static_cast<int>(row_len), dp, 1, sp, 1, &s, &z);
            }
        }
    };

    // Below this a dispatch costs more than the pass (see quantize_plane).
    constexpr int64_t kInlineMax = int64_t{1} << 19;
    const int64_t nt = std::max<int64_t>(ctx.cpu.thread_count(), 1);
    if (nt == 1 || total <= kInlineMax) {
        run_rows(0, total);
        return;
    }
    const int64_t slab = (total + nt - 1) / nt;
    const int64_t nslab = (total + slab - 1) / slab;
    ctx.cpu.run(0, nslab, [&](int64_t i) { run_rows(i * slab, std::min(i * slab + slab, total)); });
}

/// The int32 col row, whether the GEMM wrote it or the depthwise FMA pass did.
/// The depthwise sum is exact in f32 (|q − zp| ≤ 255 on both operands), so the
/// two forms are the same integer and the passes are shared.
inline v_f32x8 v_acc_to_f32(const int32_t* p)
{
#if defined(NNOPS_ARCH_X86_64)
    return v_f32x8(_mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))));
#elif defined(NNOPS_ARCH_AARCH64)
    return v_f32x8(vcvtq_f32_s32(vld1q_s32(p)), vcvtq_f32_s32(vld1q_s32(p + 4)));
#endif
}
inline v_f32x8 v_acc_to_f32(const float* p) { return v_load(p); }

inline float acc_at(int32_t v) { return static_cast<float>(v); }
inline float acc_at(float v) { return v; }

#if defined(NNOPS_ARCH_X86_64)
/// Rounded int32 lanes: a float row is exact here, so the convert is a no-op.
template <class Src> inline __m256i v_acc_to_i32(const Src* p)
{
    if constexpr (std::is_same_v<Src, float>) {
        return _mm256_cvtps_epi32(_mm256_loadu_ps(p));
    } else {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
}
inline int32_t acc_to_i32(int32_t v) { return v; }
inline int32_t acc_to_i32(float v) { return static_cast<int32_t>(v); }
#elif defined(NNOPS_ARCH_AARCH64)
template <class Src> inline int32x4_t v_acc_to_i32(const Src* p)
{
    if constexpr (std::is_same_v<Src, float>) {
        return vcvtnq_s32_f32(vld1q_f32(p));
    } else {
        return vld1q_s32(p);
    }
}
inline int32_t acc_to_i32(int32_t v) { return v; }
inline int32_t acc_to_i32(float v) { return static_cast<int32_t>(v); }
#endif

/// One pixel's `nc` integer output channels: bias, Relu, then round(acc·req) +
/// zp_out, `req` carrying the per-lane requant scale. Eight lanes at a time,
/// matching matmul_int8_requant's clamp-round-then-bias order.
template <typename TOut, class Src>
void requant_oc_row_i8(TOut* dp, const Src* rp, int64_t nc, const int32_t* bias, bool relu,
                       const float* req, int32_t zp_out)
{
    constexpr bool kU8 = std::is_same_v<TOut, uint8_t>;
    constexpr float kMin = kU8 ? 0.0f : -128.0f;
    constexpr float kMax = kU8 ? 255.0f : 127.0f;
    const float zpf = static_cast<float>(zp_out);
    int64_t c = 0;
#if defined(NNOPS_ARCH_X86_64)
    const __m256i vzp = _mm256_set1_epi32(zp_out);
    const __m256 vmin = _mm256_set1_ps(kMin - zpf);
    const __m256 vmax = _mm256_set1_ps(kMax - zpf);
    const __m256 vsign = _mm256_set1_ps(-0.0f);
    const __m256 vhalf = _mm256_set1_ps(0.5f);
    for (; c + 8 <= nc; c += 8) {
        __m256i v = v_acc_to_i32(rp + c);
        if (bias) {
            v = _mm256_add_epi32(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + c)));
        }
        if (relu) {
            v = _mm256_max_epi32(v, _mm256_setzero_si256());
        }
        __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(v), _mm256_loadu_ps(req + c));
        f = _mm256_min_ps(_mm256_max_ps(f, vmin), vmax);
        f = _mm256_add_ps(f, _mm256_or_ps(_mm256_and_ps(f, vsign), vhalf));
        const __m256i q = _mm256_add_epi32(_mm256_cvttps_epi32(f), vzp);
        const __m128i i16 =
            _mm_packs_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
        const __m128i v8 = kU8 ? _mm_packus_epi16(i16, i16) : _mm_packs_epi16(i16, i16);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dp + c), v8);
    }
#elif defined(NNOPS_ARCH_AARCH64)
    const int32x4_t vzp = vdupq_n_s32(zp_out);
    const float32x4_t vmin = vdupq_n_f32(kMin - zpf);
    const float32x4_t vmax = vdupq_n_f32(kMax - zpf);
    for (; c + 4 <= nc; c += 4) {
        int32x4_t v = v_acc_to_i32(rp + c);
        if (bias) {
            v = vaddq_s32(v, vld1q_s32(bias + c));
        }
        if (relu) {
            v = vmaxq_s32(v, vdupq_n_s32(0));
        }
        float32x4_t f = vmulq_f32(vcvtq_f32_s32(v), vld1q_f32(req + c));
        f = vminq_f32(vmaxq_f32(f, vmin), vmax);
        const int32x4_t q = vaddq_s32(vcvtaq_s32_f32(f), vzp);
        const int16x4_t i16 = vqmovn_s32(q);
        if constexpr (kU8) {
            const uint8x8_t v8 = vqmovun_s16(vcombine_s16(i16, i16));
            vst1_lane_u32(reinterpret_cast<uint32_t*>(dp + c), vreinterpret_u32_u8(v8), 0);
        } else {
            const int8x8_t v8 = vqmovn_s16(vcombine_s16(i16, i16));
            vst1_lane_s32(reinterpret_cast<int32_t*>(dp + c), vreinterpret_s32_s8(v8), 0);
        }
    }
#endif
    for (; c < nc; ++c) {
        int32_t v = acc_to_i32(rp[c]);
        if (bias) {
            v += bias[c];
        }
        if (relu) {
            v = std::max(v, 0);
        }
        dp[c] = matmul_int8_requant<TOut>(v, req[c], zp_out);
    }
}

/// The same row, element at a time: for an f16 destination the wrapper has no
/// f32→f16 row store, and the quant stage indexes its table by channel, which
/// the vector passes do not carry.
template <typename TOut, class Src>
void dequant_oc_row_scalar(TOut* orow, const Src* row, int64_t OW, int64_t nc, int64_t OC,
                           const TOut* bo, const QuantParams& qw, int64_t oc_base, float s_x,
                           const Epilogue& ep)
{
    for (int64_t ow = 0; ow < OW; ++ow) {
        const Src* rp = row + ow * nc;
        TOut* dp = orow + ow * OC;
        for (int64_t c = 0; c < nc; ++c) {
            const int64_t oc = oc_base + c;
            float f = acc_at(rp[c]) * (quant_scale_at(qw, oc) * s_x);
            if (bo) {
                f += s_load(&bo[c]);
            }
            f = nnops::apply_epilogue(ep, f, oc);
            s_store(&dp[c], f);
        }
    }
}

/// Vector-exact epilogues: the ones that decompose into min/max alone, so the
/// vector pass matches the scalar expression bit for bit. A transcendental
/// activation (Sigmoid, Tanh, Gelu, Silu, Elu, HardSwish) is a different function
/// in its vector approximation, and LeakyRelu's blend turns a −0 into a +0, where
/// the scalar `x > 0 ? x : alpha·x` keeps the sign. The integer routes are
/// compared byte for byte against the scalar reference, so those keep the element
/// loop.
inline bool epilogue_is_vector_exact(const Epilogue& ep) noexcept
{
    switch (ep.type) {
    case EpilogueActivateType::None:
    case EpilogueActivateType::Relu:
    case EpilogueActivateType::Relu6:
        return ep.quant == EpilogueQuantType::None;
    default:
        return false;
    }
}

/// The fp form: dequantize with the per-lane s_w·s_x, add the bias, then the
/// clip and the activation the fp route applies.
template <class Src>
void dequant_oc_row_f32(float* dp, const Src* rp, int64_t nc, const float* w_sx, const float* bias,
                        const Epilogue& ep)
{
    constexpr int L = simd_default_lane_f32;
    const float* tag = nullptr;
    int64_t c = 0;
    for (; c + L <= nc; c += L) {
        vec_for<float> acc = v_mul(v_acc_to_f32(rp + c), v_load(w_sx + c));
        if (bias) {
            acc = v_add(acc, v_load(bias + c));
        }
        acc = apply_epilogue_vec<float>(ep, tag, acc);
        v_store(dp + c, acc);
    }
    for (; c < nc; ++c) {
        float f = acc_at(rp[c]) * w_sx[c];
        if (bias) {
            f += bias[c];
        }
        dp[c] = nnops::apply_epilogue(ep, f);
    }
}

/// Depthwise quantized. One channel per group, so group g is
/// channel g and the vector axis is the channel — the fp depthwise route's shape,
/// with the col being the input plane itself. The GEMM route computes
/// Σ (q_x − zp_x)·(q_w − zp_w) in int32; here the same sum is accumulated in f32
/// over operands whose magnitude is at most 255, so every product and partial sum
/// is an integer below 2^24 and the conversion back is exact. The plane is
/// therefore held as (q_x − zp_x) and the encode stage is one row pass, no col
/// and no GEMM.
template <typename TOut>
bool tconv2d_int8_depthwise(const ConvChannels& gm, TensorView& output,
                                         std::span<const TensorView> inputs, const ActPlane& act,
                                         const ComputeContext& ctx)
{
    constexpr bool kIntOut = std::is_same_v<TOut, int8_t> || std::is_same_v<TOut, uint8_t>;

    if (gm.IC_per_G != 1 || gm.OC_per_G != 1) {
        return false;
    }

    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;
    if (weight.row_stride_elems() != gm.KW) {
        return false;
    }

    const int64_t karea = gm.KH * gm.KW;
    const int64_t C = gm.IC;
    const int64_t npix = gm.IH * gm.IW;
    const int64_t slots = std::max(1, ctx.cpu.worker_count());
    const int64_t row_len = gm.OW * C;
    const int64_t plane_elems = gm.N * npix * C;
    // 255·255·karea has to stay an exact integer in f32.
    if (karea > 256) {
        return false;
    }

    const int64_t elem_cap = kColCapBytes / static_cast<int64_t>(sizeof(float));
    if (C <= 0 || npix <= 0 || slots * row_len + plane_elems > elem_cap) {
        return false;
    }

    PoolArray<float> wd(static_cast<size_t>(karea * C));
    PoolArray<float> rows(static_cast<size_t>(slots * row_len));
    PoolArray<float> plane(static_cast<size_t>(plane_elems));
    PoolArray<float> lane_scale(static_cast<size_t>(C));
    if (!wd || !rows || !plane || !lane_scale) {
        return false;
    }

    const auto* w_ptr = weight.ptr<int8_t>();
    const QuantParams& qw = weight.quant_params();
    const auto* b_i32 = (has_bias && kIntOut) ? inputs[2].ptr<int32_t>() : nullptr;
    const TOut* b_fp = (has_bias && !kIntOut) ? inputs[2].ptr<TOut>() : nullptr;
    const float s_x = act.scale;
    const QuantParams& qo = output.quant_params();
    const float out_scale = kIntOut ? quant_scale_at(qo, 0) : 0.0f;
    const int32_t zp_out = kIntOut ? quant_zp_at(qo, 0) : 0;
    const bool relu = gm.epilogue.type == EpilogueActivateType::Relu;

    // [KH·KW][C] of (q_w − zp_w): the tap vector an output row is FMA'd with.
    for (int64_t c = 0; c < C; ++c) {
        const float s_w = quant_scale_at(qw, c);
        const float zw = static_cast<float>(quant_zp_at(qw, c));
        lane_scale.data()[c] = kIntOut ? matmul_int8_requant_scale(s_w, s_x, out_scale) : s_w * s_x;
        for (int64_t t = 0; t < karea; ++t) {
            wd.data()[t * C + c] = static_cast<float>(w_ptr[c * karea + t]) - zw;
        }
    }

    // The plane: (q_x − zp_x) in NHWC order, which is the layout the pixel-major
    // walk reads; one pass, shared by every tap of every output row.
    {
        const int8_t* src = act.ptr;
        const float zx = static_cast<float>(act.zero);
        const float* tag = nullptr;
        const vec_for<float> zvec = v_set1(tag, zx);
        const int64_t row_width = gm.IW * C;
        ctx.cpu.run(0, gm.N * gm.IH, [&](int64_t r) {
            const int64_t nb = r / gm.IH;
            const int8_t* sp = src + nb * act.n_stride + (r % gm.IH) * row_width;
            float* dp = plane.data() + r * row_width;
            int64_t i = 0;
            for (; i + simd_default_lane_f32 <= row_width; i += simd_default_lane_f32) {
                v_store(dp + i, v_sub(v_cvt_s8_to_f32(sp + i), zvec));
            }
            for (; i < row_width; ++i) {
                dp[i] = static_cast<float>(sp[i]) - zx;
            }
        });
    }

    auto* out_ptr = output.ptr<TOut>();
    const int64_t out_n_stride = output.stride_elems(0);
    const int64_t out_row_stride = output.row_stride_elems();
    const Tconv2dColGeom cg = gm.col_geom();

    ctx.cpu.run(0, gm.N * gm.OH, [&](int64_t task) {
        const int64_t n = task / gm.OH;
        const int64_t oh = task % gm.OH;
        float* row = rows.data() + static_cast<int64_t>(ctx.cpu.current_thread_id()) * row_len;
        const float* in_n = plane.data() + n * act.n_stride;

        row_fill(row, row_len, 0.0f);

        tconv2d_col2im_pixel_major<float>(
            cg, in_n, C, C, 0, row, oh,
            [&](float* dst, const float* src, int64_t len, int64_t tap) {
                row_fma(dst, src, wd.data() + tap * C, len);
            });

        TOut* orow = out_ptr + n * out_n_stride + oh * out_row_stride;
        for (int64_t ow = 0; ow < gm.OW; ++ow) {
            TOut* dp = orow + ow * gm.OC;
            const float* rp = row + ow * C;
            if constexpr (kIntOut) {
                requant_oc_row_i8<TOut>(dp, rp, C, b_i32, relu, lane_scale.data(), zp_out);
            } else if constexpr (std::is_same_v<TOut, float>) {
                if (epilogue_is_vector_exact(gm.epilogue)) {
                    dequant_oc_row_f32(dp, rp, C, lane_scale.data(), b_fp, gm.epilogue);
                } else {
                    dequant_oc_row_scalar(dp, rp, 1, C, gm.OC, b_fp, qw,
                                          /*oc_base=*/0, s_x, gm.epilogue);
                }
            } else {
                dequant_oc_row_scalar(dp, rp, 1, C, gm.OC, b_fp, qw, /*oc_base=*/0, s_x,
                                      gm.epilogue);
            }
        }
    });
    return true;
}

/// Quantized route: one int8 GEMM per (group, oc chunk, batch)
/// gives [pixel] × [tap·oc], and the fp route's stride-residue scatter adds the
/// int32 columns before the merge. The filter is always permuted here, unlike
/// the fp route's in-place orientation: its columns carry the per-channel
/// parameters, which the raw [oc][tap] order does not line up with.
template <typename TOut>
bool tconv2d_int8(const ConvChannels& gm, TensorView& output,
                               std::span<const TensorView> inputs, const ActPlane& act,
                               const ComputeContext& ctx)
{
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;
    constexpr bool kIntOut = std::is_same_v<TOut, int8_t> || std::is_same_v<TOut, uint8_t>;

    if (weight.row_stride_elems() != gm.KW) {
        return false;
    }

    const int64_t karea = gm.KH * gm.KW;
    const int64_t npix = gm.IH * gm.IW;
    const int64_t icg = gm.IC_per_G;
    const int64_t ocg = gm.OC_per_G;
    if (npix <= 0 || karea <= 0 || icg <= 0) {
        return false;
    }

    const int64_t out_n_stride = output.stride_elems(0);
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t slots = std::max(1, ctx.cpu.worker_count());
    const Tconv2dColGeom cg = gm.col_geom();

    // The col and the per-worker row scratch both scale with the oc chunk.
    const int64_t elem_cap = kColCapBytes / static_cast<int64_t>(sizeof(int32_t));
    int64_t cc = ocg;
    cc = std::min(cc, std::max<int64_t>(1, elem_cap / (npix * karea)));
    cc = std::min(cc, std::max<int64_t>(1, elem_cap / (slots * gm.OW)));

    PoolArray<int32_t> col(static_cast<size_t>(npix * karea * cc));
    PoolArray<int8_t> aux(static_cast<size_t>(icg * karea * cc));
    PoolArray<int32_t> rows(static_cast<size_t>(slots * gm.OW * cc));
    PoolArray<float> b_scale(static_cast<size_t>(karea * cc));
    PoolArray<int32_t> b_zp(static_cast<size_t>(karea * cc));
    PoolArray<float> lane_scale(static_cast<size_t>(cc));
    if (!col || !aux || !rows || !b_scale || !b_zp || !lane_scale) {
        return false;
    }

    const auto* w_ptr = weight.ptr<int8_t>();
    const QuantParams& qw = weight.quant_params();
    const auto* b_i32 = (has_bias && kIntOut) ? inputs[2].ptr<int32_t>() : nullptr;
    const TOut* b_fp = (has_bias && !kIntOut) ? inputs[2].ptr<TOut>() : nullptr;
    const float s_x = act.scale;
    const QuantParams act_qp = qp_per_tensor(s_x, act.zero);

    const QuantParams& qo = output.quant_params();
    const float out_scale = kIntOut ? quant_scale_at(qo, 0) : 0.0f;
    const int32_t zp_out = kIntOut ? quant_zp_at(qo, 0) : 0;
    const bool relu = gm.epilogue.type == EpilogueActivateType::Relu;

    auto* out_ptr = output.ptr<TOut>();

    MatMulAttributes mm{};

    for (int64_t g = 0; g < gm.G; ++g) {
        for (int64_t oc0 = 0; oc0 < ocg; oc0 += cc) {
            const int64_t nc = std::min(cc, ocg - oc0);
            if (nc <= 0) {
                continue;
            }
            const int64_t m = nc * karea;

            // Per-column (tap, oc) parameters: B's columns are the
            // (kh, kw, oc) triples of one output channel. The merge's own
            // per-lane factor is the same scale folded once per channel.
            for (int64_t o = 0; o < nc; ++o) {
                const int64_t oc = g * ocg + oc0 + o;
                const float s = quant_scale_at(qw, oc);
                const int32_t z = quant_zp_at(qw, oc);
                lane_scale.data()[o] =
                    kIntOut ? matmul_int8_requant_scale(s, s_x, out_scale) : s * s_x;
                for (int64_t t = 0; t < karea; ++t) {
                    b_scale.data()[t * nc + o] = s;
                    b_zp.data()[t * nc + o] = z;
                }
            }

            // B: the group's filter slice as [IC/G] × [KH·KW·nc], a blocked
            // transpose of the [IC/G][OC/G][KH·KW] rows.
            for (int64_t ic = 0; ic < icg; ++ic) {
                const int8_t* src = w_ptr + ((g * icg + ic) * ocg + oc0) * karea;
                int8_t* dst = aux.data() + ic * m;
                constexpr int64_t kB = 8;
                for (int64_t o0 = 0; o0 < nc; o0 += kB) {
                    const int64_t o1 = std::min(o0 + kB, nc);
                    for (int64_t t0 = 0; t0 < karea; t0 += kB) {
                        const int64_t t1 = std::min(t0 + kB, karea);
                        for (int64_t o = o0; o < o1; ++o) {
                            int8_t* d = dst + t0 * nc + o;
                            const int8_t* s = src + o * karea + t0;
                            for (int64_t t = 0; t < t1 - t0; ++t) {
                                d[t * nc] = s[t];
                            }
                        }
                    }
                }
            }

            const int64_t bshape[] = {icg, m};
            TensorView bm(std::span<const int64_t>(bshape, 2), DataType::s8, aux.data(),
                          static_cast<int64_t>(m), TensorLayout::Dense,
                          qp_per_channel(b_scale.data(), b_zp.data(), m));

            for (int64_t n = 0; n < gm.N; ++n) {
                const int8_t* in_n = act.ptr + n * act.n_stride;
                TOut* out_n = out_ptr + n * out_n_stride;

                // A: the NHWC image as [IH·IW] × [IC/G] pixels — one
                // pixel stride of IC, so the group offset is g·IC/G elements in.
                const int64_t ashape[] = {npix, icg};
                TensorView am(std::span<const int64_t>(ashape, 2), DataType::s8,
                              const_cast<int8_t*>(in_n + g * icg), static_cast<int64_t>(gm.IC),
                              TensorLayout::Dense, act_qp);
                const int64_t cshape[] = {npix, m};
                TensorView cm(std::span<const int64_t>(cshape, 2), DataType::s32, col.data(),
                              static_cast<int64_t>(m * sizeof(int32_t)), TensorLayout::Dense);

                const TensorView mins[] = {am, bm};
                matmul_cpu(mm, cm, mins, ctx, nullptr);

                ctx.cpu.run(0, gm.OH, [&](int64_t oh) {
                    int32_t* row = rows.data() +
                                   static_cast<int64_t>(ctx.cpu.current_thread_id()) * (gm.OW * nc);
                    std::fill_n(row, gm.OW * nc, int32_t{0});

                    tconv2d_col2im_pixel_major<int32_t>(
                        cg, col.data(), m, nc, nc, row, oh,
                        [](int32_t* dst, const int32_t* src, int64_t len, int64_t) {
                            row_add_i32(dst, src, len);
                        });

                    TOut* orow = out_n + oh * out_row_stride + g * ocg + oc0;
                    const TOut* bo_fp = b_fp ? b_fp + g * ocg + oc0 : nullptr;
                    const int64_t oc_base = g * ocg + oc0;
                    if constexpr (kIntOut) {
                        const int32_t* bo = b_i32 ? b_i32 + oc_base : nullptr;
                        for (int64_t ow = 0; ow < gm.OW; ++ow) {
                            requant_oc_row_i8<TOut>(orow + ow * gm.OC, row + ow * nc, nc, bo, relu,
                                                    lane_scale.data(), zp_out);
                        }
                    } else if constexpr (std::is_same_v<TOut, float>) {
                        if (epilogue_is_vector_exact(gm.epilogue)) {
                            for (int64_t ow = 0; ow < gm.OW; ++ow) {
                                dequant_oc_row_f32(orow + ow * gm.OC, row + ow * nc, nc,
                                                   lane_scale.data(), bo_fp, gm.epilogue);
                            }
                        } else {
                            dequant_oc_row_scalar(orow, row, gm.OW, nc, gm.OC, bo_fp, qw, oc_base,
                                                  s_x, gm.epilogue);
                        }
                    } else {
                        dequant_oc_row_scalar(orow, row, gm.OW, nc, gm.OC, bo_fp, qw, oc_base, s_x,
                                              gm.epilogue);
                    }
                });
            }
        }
    }
    return true;
}

/// Scalar quantized gather — the total route. Each term is
/// (w - zp_w)·(x - zp_x) in int32, the reference's own arithmetic; used when the
/// operands cannot be read as flat matrices or the kernel is too tall for the
/// per-row tap list.
template <typename TOut>
void tconv2d_int8_scalar(const ConvChannels& gm, TensorView& output, std::span<const TensorView> inputs,
                         const ActPlane& act, const ComputeContext& ctx)
{
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;
    constexpr bool kIntOut = std::is_same_v<TOut, int8_t> || std::is_same_v<TOut, uint8_t>;

    const auto* w_ptr = weight.ptr<int8_t>();
    const QuantParams& qw = weight.quant_params();
    const auto* b_i32 = (has_bias && kIntOut) ? inputs[2].ptr<int32_t>() : nullptr;
    const TOut* b_fp = (has_bias && !kIntOut) ? inputs[2].ptr<TOut>() : nullptr;
    const float s_x = act.scale;
    const QuantParams& qo = output.quant_params();
    const float out_scale = kIntOut ? quant_scale_at(qo, 0) : 0.0f;
    const int32_t zp_out = kIntOut ? quant_zp_at(qo, 0) : 0;
    const bool relu = gm.epilogue.type == EpilogueActivateType::Relu;

    const int64_t in_row_stride = gm.IC;
    const int64_t out_c_stride = output.stride_elems(1);
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_n_stride = output.stride_elems(0);
    auto* out_ptr = output.ptr<TOut>();

    const auto process = [&](int64_t task) {
        const int64_t n = task / gm.OC;
        const int64_t oc = task % gm.OC;
        const int64_t g = oc / gm.OC_per_G;
        const int64_t oc_l = oc % gm.OC_per_G;
        const float s_w = quant_scale_at(qw, oc);
        const int32_t zp_w = quant_zp_at(qw, oc);
        const int8_t* in_n = act.ptr + n * act.n_stride;
        TOut* out_plane = out_ptr + n * out_n_stride + oc * out_c_stride;

        for (int64_t oh = 0; oh < gm.OH; ++oh) {
            TOut* orow = out_plane + oh * out_row_stride;
            for (int64_t ow = 0; ow < gm.OW; ++ow) {
                int32_t acc = 0;
                for (int64_t kh = 0; kh < gm.KH; ++kh) {
                    const int64_t num_h = oh + gm.PH - kh * gm.DH;
                    if (num_h < 0 || num_h % gm.SH != 0) {
                        continue;
                    }
                    const int64_t ih = num_h / gm.SH;
                    if (ih >= gm.IH) {
                        continue;
                    }
                    for (int64_t kw = 0; kw < gm.KW; ++kw) {
                        const int64_t num_w = ow + gm.PW - kw * gm.DW;
                        if (num_w < 0 || num_w % gm.SW != 0) {
                            continue;
                        }
                        const int64_t iw = num_w / gm.SW;
                        if (iw >= gm.IW) {
                            continue;
                        }
                        const int8_t* xp =
                            in_n + (ih * gm.IW + iw) * gm.IC + g * gm.IC_per_G;
                        for (int64_t ic_l = 0; ic_l < gm.IC_per_G; ++ic_l) {
                            const int64_t ic = g * gm.IC_per_G + ic_l;
                            const int32_t xv = static_cast<int32_t>(xp[ic_l]);
                            const int64_t wi =
                                ((ic * gm.OC_per_G + oc_l) * gm.KH + kh) * gm.KW + kw;
                            acc += (static_cast<int32_t>(w_ptr[wi]) - zp_w) * (xv - act.zero);
                        }
                    }
                }

                if constexpr (kIntOut) {
                    int32_t v = acc;
                    if (b_i32) {
                        v += b_i32[oc];
                    }
                    if (relu) {
                        v = std::max(v, int32_t{0});
                    }
                    orow[ow] = matmul_int8_requant<TOut>(
                        v, matmul_int8_requant_scale(s_w, s_x, out_scale), zp_out);
                } else {
                    float f = static_cast<float>(acc) * (s_w * s_x);
                    if (b_fp) {
                        f += s_load(&b_fp[oc]);
                    }
                    apply_epilogue_store<TOut>(&orow[ow], f, gm.epilogue, oc,
                                               /*add_to=*/false, /*pre_act=*/false);
                }
            }
        }
    };

    ctx.cpu.run(0, gm.N * gm.OC, process);
}

template <typename TOut>
void tconv2d_int8_layout(const ConvChannels& gm, TensorView& output, std::span<const TensorView> inputs,
                         const ActPlane& act, const ComputeContext& ctx)
{
    if (tconv2d_int8_depthwise<TOut>(gm, output, inputs, act, ctx)) {
        return;
    }
    if (tconv2d_int8<TOut>(gm, output, inputs, act, ctx)) {
        return;
    }
    tconv2d_int8_scalar<TOut>(gm, output, inputs, act, ctx);
}

/// One activation storage type of the quantized route: build the plane, then
/// dispatch the output dtype.
template <class TAct>
Status tconv2d_int8_act(const TransposeConv2DAttributes& attrs, const ConvChannels& gm, TensorView& output,
                        std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    constexpr bool kFpAct = !std::is_same_v<TAct, int8_t> && !std::is_same_v<TAct, uint8_t>;
    const auto& input = inputs[0];
    const DataType adt = input.data_type();
    const bool target_u8 = kFpAct ? (attrs.act_quant_dtype == DataType::u8) : (adt == DataType::u8);
    const QuantParams& qa = input.quant_params();
    const float s_x = quant_scale_at(qa, 0);
    const int32_t zp_x = quant_zp_at(qa, 0);

    const int64_t npix = gm.IH * gm.IW;
    const bool dense = input.row_stride_elems() == gm.IW * gm.IC;
    // A pre-quantized s8 activation with dense rows needs no plane at all.
    const bool in_place = !kFpAct && adt == DataType::s8 && dense;

    ActPlane act;
    act.scale = s_x;
    act.zero = zp_x - (target_u8 ? 128 : 0);
    // An empty plane (IH or IW 0) leaves the pointer on the input: no pixel of
    // it is ever read, but the batch walk still forms in_n.
    act.ptr = reinterpret_cast<const int8_t*>(input.ptr<TAct>());

    PoolArray<int8_t> plane;
    if (in_place) {
        act.ptr = input.ptr<int8_t>();
        act.n_stride = input.stride_elems(0);
    } else if (npix > 0) {
        const int64_t total = gm.N * gm.IC * npix;
        plane.reset(static_cast<size_t>(total));
        if (!plane) {
            return Status::out_of_memory;
        }
        build_activation_plane<TAct>(input, plane.data(), s_x, act.zero, gm.IC, gm.IH, gm.IW, ctx);
        act.ptr = plane.data();
        act.n_stride = gm.IC * npix;
    }

    switch (output.data_type()) {
    case DataType::f32:
        tconv2d_int8_layout<float>(gm, output, inputs, act, ctx);
        break;
    case DataType::f16:
        tconv2d_int8_layout<half>(gm, output, inputs, act, ctx);
        break;
    case DataType::s8:
        tconv2d_int8_layout<int8_t>(gm, output, inputs, act, ctx);
        break;
    default:
        tconv2d_int8_layout<uint8_t>(gm, output, inputs, act, ctx);
        break;
    }
    return Status::ok;
}

} // anonymous namespace

/// dtype gate of the quantized route, then the activation dispatch.
Status transpose_conv2d_int8_cpu(const TransposeConv2DAttributes& attrs, const ConvChannels& gm,
                                 TensorView& output, std::span<const TensorView> inputs,
                                 const ComputeContext& ctx)
{
    const DataType adt = inputs[0].data_type();
    const DataType odt = output.data_type();
    const bool fp_act = adt == DataType::f32 || adt == DataType::f16;
    const bool int_out = odt == DataType::s8 || odt == DataType::u8;

    if (!fp_act && adt != DataType::s8 && adt != DataType::u8) {
        return Status::unsupported;
    }
    if (!int_out && odt != DataType::f32 && odt != DataType::f16) {
        return Status::unsupported;
    }
    // An fp activation keeps its own dtype on an fp destination; a pre-quantized
    // one is dequantized to whatever output_dtype asked for.
    if (!int_out && fp_act && odt != adt) {
        return Status::unsupported;
    }
    if (inputs.size() > 2) {
        const DataType bdt = inputs[2].data_type();
        if (bdt != (int_out ? DataType::s32 : odt)) {
            return Status::unsupported;
        }
    }
    // add_to would have to keep the pre-existing destination out of the
    // scatter; the integer accumulate has no fold for it.
    if (attrs.add_to) {
        return Status::unsupported;
    }

    if (gm.N == 0 || gm.OC == 0 || gm.OH == 0 || gm.OW == 0) {
        return Status::ok;
    }

    switch (adt) {
    case DataType::f32:
        return tconv2d_int8_act<float>(attrs, gm, output, inputs, ctx);
    case DataType::f16:
        return tconv2d_int8_act<half>(attrs, gm, output, inputs, ctx);
    case DataType::u8:
        return tconv2d_int8_act<uint8_t>(attrs, gm, output, inputs, ctx);
    default:
        return tconv2d_int8_act<int8_t>(attrs, gm, output, inputs, ctx);
    }
}

} // namespace nnops::backend::cpu
