/// @file cast.cpp
/// @brief CPU dtype conversion (ONNX Cast).
///
/// One row-tiled loop over the pitch-aware row geometry: the source and the
/// destination have different element sizes, so both walk their own row stride.
/// Conversion goes through f32 for anything touching a float, and through int64
/// when both sides are integral — an int64 index is not representable in f32
/// past 2^24, and casting it through the float path would silently round it.
///
/// The scalar body is the whole kernel: the conversion is one instruction (F16C
/// / NEON fcvt) or a saturating clamp, so a hand-rolled vector path would only
/// pay for the shuffle of mismatched lane widths. Rows are the parallel unit.
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

#include <cmath>
#include <limits>

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
        const double v = static_cast<double>(load_f32(src));
        if (std::isnan(v)) {
            *dst = Dst{0};
        } else {
            constexpr double lo = static_cast<double>(std::numeric_limits<Dst>::lowest());
            constexpr double hi = static_cast<double>(std::numeric_limits<Dst>::max());
            const double c = v < lo ? lo : (v > hi ? hi : v);
            *dst = static_cast<Dst>(static_cast<int64_t>(c));
        }
    } else {
        store_f32(dst, load_f32(src));
    }
}

template <class Src, class Dst>
void cast_rows(const TensorView& in, TensorView& out, const ComputeContext& ctx)
{
    const RowGeometry g = rowwise_geometry(in, out);
    const Src* src = in.template ptr<Src>();
    Dst* dst = out.template ptr<Dst>();

    ctx.cpu.run(0, g.num_rows, [&](int64_t r) {
        const Src* s = src + r * g.in_stride;
        Dst* d = dst + r * g.out_stride;
        for (int64_t i = 0; i < g.last_dim; ++i) {
            cast_element(s + i, d + i);
        }
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
