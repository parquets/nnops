/// @file pad.cpp
/// @brief CPU Pad — per-axis source maps, then one strided copy per element.
///
/// Padding is a gather with a per-axis map, so the driver builds each axis's map
/// once (an in-range index, or -1 for a Constant fill) and a chunked odometer
/// walks the output, summing index·stride. Every mode but Constant maps into
/// range, so only Constant needs the marker branch. The element copy is a plain
/// assignment — half bits travel untouched, and int dtypes never round-trip
/// through float.

#include "nnops/ops/pad.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "nnops/core/compute_context.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace nnops::backend::cpu {

namespace {

template <typename T>
inline T constant_of(float v)
{
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16>) {
        // The fp store form covers both 16-bit floats; the integer branch
        // below would read numeric_limits, which neither of them specializes.
        T h;
        simd::s_store(&h, v);
        return h;
    } else {
        const double r = std::nearbyint(static_cast<double>(v));
        return static_cast<T>(std::clamp<double>(
            r, static_cast<double>(std::numeric_limits<T>::lowest()),
            static_cast<double>(std::numeric_limits<T>::max())));
    }
}

inline int64_t map_index(int64_t i, int64_t n, PadAttributes::Mode mode)
{
    switch (mode) {
    case PadAttributes::Mode::Edge:
        return std::clamp<int64_t>(i, 0, n - 1);
    case PadAttributes::Mode::Reflect:
        if (i < 0) { return -i; }
        if (i >= n) { return 2 * (n - 1) - i; }
        return i;
    case PadAttributes::Mode::Wrap:
        return ((i % n) + n) % n;
    case PadAttributes::Mode::Constant:
    default:
        return (i < 0 || i >= n) ? -1 : i;
    }
}

template <typename T>
void pad_impl(const PadAttributes& attrs, TensorView& output,
              std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t rank  = input.rank();
    const int64_t numel = output.numel();
    if (numel == 0) { return; }

    int64_t odims[8];
    int64_t ostr[8];
    int64_t istr[8];
    int64_t in_dims[8];
    for (int64_t d = 0; d < rank; ++d) {
        odims[d]   = output.shape(d);
        ostr[d]    = output.stride_elems(d);
        istr[d]    = input.stride_elems(d);
        in_dims[d] = input.shape(d);
    }

    // One map per axis over that axis's output extent.
    std::vector<int64_t> maps;
    int64_t map_off[8];
    {
        size_t total = 0;
        for (int64_t d = 0; d < rank; ++d) {
            map_off[d] = static_cast<int64_t>(total);
            total += static_cast<size_t>(odims[d]);
        }
        maps.resize(total);
    }
    for (int64_t d = 0; d < rank; ++d) {
        const int64_t begin = attrs.pads[static_cast<size_t>(d)];
        for (int64_t o = 0; o < odims[d]; ++o) {
            maps[static_cast<size_t>(map_off[d] + o)] =
                map_index(o - begin, in_dims[d], attrs.mode);
        }
    }

    const T cval = constant_of<T>(attrs.constant_value);
    const T* src = input.ptr<T>();
    T* dst = output.ptr<T>();

    constexpr int64_t kChunk = 1 << 16;
    const int64_t chunks = (numel + kChunk - 1) / kChunk;

    ctx.cpu.run(0, chunks, [&](int64_t ci) {
        const int64_t begin = ci * kChunk;
        const int64_t end   = std::min(begin + kChunk, numel);

        int64_t idx[8] = {};
        int64_t rem = begin;
        for (int64_t d = rank - 1; d >= 0; --d) {
            idx[d] = (odims[d] > 0) ? rem % odims[d] : 0;
            rem    = (odims[d] > 0) ? rem / odims[d] : 0;
        }

        int64_t oo = 0;
        for (int64_t d = 0; d < rank; ++d) { oo += idx[d] * ostr[d]; }

        for (int64_t n = begin; n < end; ++n) {
            int64_t so = 0;
            bool padded = false;
            for (int64_t d = 0; d < rank; ++d) {
                const int64_t si = maps[static_cast<size_t>(map_off[d] + idx[d])];
                if (si < 0) { padded = true; break; }
                so += si * istr[d];
            }
            dst[oo] = padded ? cval : src[so];

            for (int64_t d = rank - 1; d >= 0; --d) {
                ++idx[d];
                oo += ostr[d];
                if (idx[d] < odims[d]) { break; }
                idx[d] = 0;
                oo -= ostr[d] * odims[d];
            }
        }
    });
}

}  // anonymous namespace

Status pad_cpu(const PadAttributes& attrs,
               TensorView& output,
               std::span<const TensorView> inputs,
               const ComputeContext& ctx,
               void* /*workspace*/)
{
    switch (inputs[0].data_type()) {
    case DataType::f32: pad_impl<float>(attrs, output, inputs, ctx); return Status::ok;
    case DataType::f16: pad_impl<half>(attrs, output, inputs, ctx); return Status::ok;
#ifdef NNOPS_ARCH_AARCH64
    case DataType::bf16: pad_impl<bfloat16>(attrs, output, inputs, ctx); return Status::ok;
#endif
    case DataType::s8:  pad_impl<int8_t>(attrs, output, inputs, ctx); return Status::ok;
    case DataType::u8:  pad_impl<uint8_t>(attrs, output, inputs, ctx); return Status::ok;
    case DataType::s32: pad_impl<int32_t>(attrs, output, inputs, ctx); return Status::ok;
    case DataType::s64: pad_impl<int64_t>(attrs, output, inputs, ctx); return Status::ok;
    default: return Status::unsupported;
    }
}

}  // namespace nnops::backend::cpu
