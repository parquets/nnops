/// @file cast.cpp
/// @brief CPU dtype conversion (ONNX Cast).
///
/// A flat element tile is the parallel unit — earlier versions took a row, which
/// left a 1-D tensor (one row) on a single thread however long it was. Every
/// tile walks its own row stride, since source and destination have different
/// element sizes. Conversion goes through f32 for anything touching a float,
/// and through int64 when both sides are integral — an int64 index is not
/// representable in f32 past 2^24, and casting it through the float path would
/// silently round it.
///
/// The float-like pairs are converted eight lanes at a time through the SIMD
/// layer's widening/narrowing load/store (F16C / NEON fcvt; bf16 is a shift),
/// s8/u8 → f32 uses the 8-lane widen, and f32 → s32/s8/u8 uses a vector
/// truncation with the saturation patched in. Only int64 — and the int→int
/// pairs, which go through int64 — keep the scalar body.
///
/// Semantics (ONNX Cast / onnxruntime): float→int truncates toward zero and
/// saturates; NaN maps to 0; int→int goes through int64; float→float rounds to
/// nearest even in the destination format (the hardware conversion's mode).

#include "nnops/ops/cast.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "common/elementwise.hpp"
#include "common/dtype_dispatch.hpp"

#if defined(NNOPS_ARCH_X86_64)
#include <immintrin.h>
#elif defined(NNOPS_ARCH_AARCH64)
#include <arm_neon.h>
#endif

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace nnops::backend::cpu {

namespace {

// ---- load as f32 / store from f32 --------------------------------------

template <class S>
inline float load_f32(const S* p) { return simd::s_load(p); }

template <class S>
inline void store_f32(S* p, float v) { simd::s_store(p, v); }

// ---- the four conversion categories -----------------------------------

template <class Src, class Dst>
inline void cast_element(const Src* src, Dst* dst)
{
    constexpr bool src_int = std::is_integral_v<Src>;
    constexpr bool dst_int = std::is_integral_v<Dst>;

    if constexpr (src_int && dst_int) {
        // int64 end to end: an index past 2^24 must not round.
        const int64_t v = static_cast<int64_t>(*src);
        constexpr int64_t lo = static_cast<int64_t>(std::numeric_limits<Dst>::lowest());
        constexpr int64_t hi = static_cast<int64_t>(std::numeric_limits<Dst>::max());
        *dst = static_cast<Dst>(v < lo ? lo : (v > hi ? hi : v));
    } else if constexpr (src_int) {
        store_f32(dst, static_cast<float>(static_cast<int64_t>(*src)));
    } else if constexpr (dst_int) {
        // f32 end to end. The float limits are exact for s32/u8/s8's bounds
        // (INT_MAX rounds up to 2^31, which the >= test still catches before
        // the conversion), and truncation toward zero is the value conversion
        // itself once the value is inside the destination range — no double,
        // which used to cost ~4x the memory-bound rate on these pairs.
        const float v = load_f32(src);
        if (std::isnan(v)) {
            *dst = Dst{0};
        } else {
            constexpr float lo = static_cast<float>(std::numeric_limits<Dst>::lowest());
            constexpr float hi = static_cast<float>(std::numeric_limits<Dst>::max());
            *dst = v <= lo ? std::numeric_limits<Dst>::lowest()
                  : v >= hi ? std::numeric_limits<Dst>::max()
                            : static_cast<Dst>(v);
        }
    } else {
        store_f32(dst, load_f32(src));
    }
}

/// The types the eight-lane widening/narrowing path covers.
template <typename T>
inline constexpr bool is_float_storage =
    std::is_same_v<T, float> || std::is_same_v<T, half> ||
    std::is_same_v<T, bfloat16>;

// ---- f32 → integer, eight lanes ---------------------------------------
//
// The ONNX rule this backend follows is a truncation toward zero plus a
// saturation to the destination range, with NaN to 0. x86's cvttps raises the
// integer indefinite pattern instead of saturating, so the range is patched
// with two blends; NEON's FCVTZS already saturates and maps NaN to 0, so the
// s8/u8 forms only have to narrow.

#if defined(NNOPS_ARCH_X86_64)
/// Eight floats → eight saturating-truncated int32.
inline __m256i f32_to_i32_sat(const float* p)
{
    const __m256 v = _mm256_loadu_ps(p);
    __m256i i = _mm256_cvttps_epi32(v);
    const __m256 nan   = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
    const __m256 ge_hi = _mm256_cmp_ps(v, _mm256_set1_ps(2147483648.0f), _CMP_GE_OQ);
    const __m256 lt_lo = _mm256_cmp_ps(v, _mm256_set1_ps(-2147483648.0f), _CMP_LT_OQ);
    i = _mm256_blendv_epi8(i, _mm256_set1_epi32(INT32_MAX),
                           _mm256_castps_si256(ge_hi));
    i = _mm256_blendv_epi8(i, _mm256_set1_epi32(INT32_MIN),
                           _mm256_castps_si256(lt_lo));
    return _mm256_andnot_si256(_mm256_castps_si256(nan), i);   // NaN → 0
}

inline void f32_to_i32_sat(const float* s, int32_t* d)
{
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(d), f32_to_i32_sat(s));
}

/// Eight floats → eight saturating int8 / uint8.
template <class Dst>
inline void f32_to_i8_sat(const float* s, Dst* d)
{
    const __m256i i = f32_to_i32_sat(s);
    const __m128i i16 = _mm_packs_epi32(_mm256_castsi256_si128(i),
                                        _mm256_extracti128_si256(i, 1));
    const __m128i i8 = std::is_same_v<Dst, uint8_t>
        ? _mm_packus_epi16(i16, i16) : _mm_packs_epi16(i16, i16);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(d), i8);
}

#elif defined(NNOPS_ARCH_AARCH64)

inline void f32_to_i32_sat(const float* s, int32_t* d)
{
    vst1q_s32(d, vcvtq_s32_f32(vld1q_f32(s)));
    vst1q_s32(d + 4, vcvtq_s32_f32(vld1q_f32(s + 4)));
}

template <class Dst>
inline void f32_to_i8_sat(const float* s, Dst* d)
{
    const int32x4_t a = vcvtq_s32_f32(vld1q_f32(s));
    const int32x4_t b = vcvtq_s32_f32(vld1q_f32(s + 4));
    const int16x8_t i16 = vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
    if constexpr (std::is_same_v<Dst, uint8_t>) {
        vst1_u8(reinterpret_cast<uint8_t*>(d), vqmovun_s16(i16));
    } else {
        vst1_s8(reinterpret_cast<int8_t*>(d), vqmovn_s16(i16));
    }
}

#endif

/// Convert a contiguous run.
template <class Src, class Dst>
inline void cast_range(const Src* s, Dst* d, int64_t n)
{
    int64_t i = 0;
#if defined(NNOPS_ARCH_X86_64) || defined(NNOPS_ARCH_AARCH64)
    if constexpr (std::is_same_v<Src, float> &&
                  (std::is_same_v<Dst, int32_t> || std::is_same_v<Dst, int8_t> ||
                   std::is_same_v<Dst, uint8_t>)) {
        for (; i + 8 <= n; i += 8) {
            if constexpr (std::is_same_v<Dst, int32_t>) {
                f32_to_i32_sat(s + i, d + i);
            } else {
                f32_to_i8_sat(s + i, d + i);
            }
        }
    }
#endif
    if constexpr (is_float_storage<Src> && is_float_storage<Dst>) {
        // Widens on load, rounds once on store, so float→half is one F16C
        // instruction per eight elements rather than one per element.
        constexpr int64_t lane = simd::simd_lane_f32x8;
        for (; i + lane <= n; i += lane) {
            simd::v_store_f32(d + i, simd::v_load_f32(s + i));
        }
    } else if constexpr (std::is_same_v<Src, int8_t> && std::is_same_v<Dst, float>) {
        for (; i + 8 <= n; i += 8) { simd::v_store(d + i, simd::v_cvt_s8_to_f32(s + i)); }
    } else if constexpr (std::is_same_v<Src, uint8_t> && std::is_same_v<Dst, float>) {
        for (; i + 8 <= n; i += 8) { simd::v_store(d + i, simd::v_cvt_u8_to_f32(s + i)); }
    } else if constexpr (std::is_same_v<Src, Dst> &&
                         (std::is_same_v<Src, int8_t> || std::is_same_v<Src, uint8_t>)) {
        for (; i + 16 <= n; i += 16) { simd::v_store(d + i, simd::v_load(s + i)); }
    }
    for (; i < n; ++i) {
        cast_element(s + i, d + i);
    }
}

template <class Src, class Dst>
void cast_rows(const TensorView& in, TensorView& out, const ComputeContext& ctx)
{
    const RowGeometry g = rowwise_geometry(in, out);
    const Src* src = in.template ptr<Src>();
    Dst* dst = out.template ptr<Dst>();

    // Tiles of `kCastTile` elements inside each row. One tile per row would
    // leave a [1, N] or [N, 1] tensor with a single task.
    constexpr int64_t kCastTile = int64_t{1} << 15;
    const int64_t tiles_per_row = (g.last_dim + kCastTile - 1) / kCastTile;
    ctx.cpu.run(0, g.num_rows * tiles_per_row, [&](int64_t t) {
        const int64_t r   = t / tiles_per_row;
        const int64_t off = (t % tiles_per_row) * kCastTile;
        const int64_t len = g.last_dim - off < kCastTile ? g.last_dim - off : kCastTile;
        cast_range(src + r * g.in_stride + off,
                   dst + r * g.out_stride + off, len);
    });
}

template <class Src>
Status cast_from(DataType to, const TensorView& in, TensorView& out,
                 const ComputeContext& ctx)
{
    switch (to) {
    case DataType::f32: cast_rows<Src, float>(in, out, ctx); return Status::ok;
    case DataType::f16: cast_rows<Src, half>(in, out, ctx); return Status::ok;
#ifdef NNOPS_ARCH_AARCH64
    case DataType::bf16: cast_rows<Src, bfloat16>(in, out, ctx); return Status::ok;
#endif
    case DataType::s32: cast_rows<Src, int32_t>(in, out, ctx); return Status::ok;
    case DataType::s64: cast_rows<Src, int64_t>(in, out, ctx); return Status::ok;
    case DataType::s8:  cast_rows<Src, int8_t>(in, out, ctx); return Status::ok;
    case DataType::u8:  cast_rows<Src, uint8_t>(in, out, ctx); return Status::ok;
    default: break;
    }
    return Status::unsupported;
}

}  // anonymous namespace

Status cast_cpu(const CastAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* /*workspace*/)
{
    const TensorView& in = inputs[0];
    switch (in.data_type()) {
    case DataType::f32: return cast_from<float>(attrs.to, in, output, ctx);
    case DataType::f16: return cast_from<half>(attrs.to, in, output, ctx);
#ifdef NNOPS_ARCH_AARCH64
    case DataType::bf16: return cast_from<bfloat16>(attrs.to, in, output, ctx);
#endif
    case DataType::s32: return cast_from<int32_t>(attrs.to, in, output, ctx);
    case DataType::s64: return cast_from<int64_t>(attrs.to, in, output, ctx);
    case DataType::s8:  return cast_from<int8_t>(attrs.to, in, output, ctx);
    case DataType::u8:  return cast_from<uint8_t>(attrs.to, in, output, ctx);
    default: break;
    }
    return Status::unsupported;
}

}  // namespace nnops::backend::cpu
