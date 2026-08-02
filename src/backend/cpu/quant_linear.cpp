/// @file quant_linear.cpp
/// @brief SIMD-accelerated CPU implementation of QuantizeLinear / DequantizeLinear.
///
/// Strategy:
///   - All arithmetic is in f32 (v_f32x8). For f16 input/output, conversion
///     happens at the boundary via v_cvt_f16_to_f32 / v_cvt_f32_to_f16.
///   - int8/uint8 → f32 uses v_cvt_i8_to_f32 / v_cvt_u8_to_f32 (8 lanes).
///   - Packed layouts (NCHWC8/NCDHWC8): per-lane processing — each physical
///     row of W×8 elements has 8 independent lanes processed with float SIMD.
///   - PerTensor scale/zp: broadcast to all SIMD lanes.
///   - PerChannel scale/zp: index per-axis-dimension element.

#include "nnops/ops/quant_linear.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace nnops::backend::cpu {

using namespace nnops::simd;
using nnops::backend::cpu::half;
using nnops::backend::cpu::half_to_float;
using nnops::backend::cpu::float_to_half;

namespace {

constexpr int L = 8;

inline int32_t int_min(DataType dt) { return (dt == DataType::i8) ? -128 : 0; }
inline int32_t int_max(DataType dt) { return (dt == DataType::i8) ? 127 : 255; }

inline void write_int8(void* ptr, DataType dt, int64_t off, int32_t val) {
    int32_t lo = int_min(dt), hi = int_max(dt);
    val = std::max(lo, std::min(hi, val));
    if (dt == DataType::i8)
        static_cast<int8_t*>(ptr)[off] = static_cast<int8_t>(val);
    else
        static_cast<uint8_t*>(ptr)[off] = static_cast<uint8_t>(val);
}

}  // anonymous namespace

// ============================================================
// QuantizeLinear: float → integer
// ============================================================

template <typename T>  // T = float or half (input type)
void quantize_linear_impl(const QuantLinearAttributes& attrs,
                          TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx)
{
    const auto& X     = inputs[0];
    const auto& scale = inputs[1];
    const auto& zp    = inputs[2];

    const int64_t rank = X.rank();
    int64_t axis = attrs.axis;
    if (axis < 0) axis += rank;

    const int64_t D = X.shape(axis);
    const bool is_per_channel = (scale.numel() > 1);
    const DataType out_dtype = output.data_type();

    // Pre-load scale/zp as float
    std::vector<float> s_f32(static_cast<size_t>(D));
    std::vector<float> z_f32(static_cast<size_t>(D));
    for (int64_t k = 0; k < D; ++k) {
        s_f32[static_cast<size_t>(k)] = scale.ptr<float>()[is_per_channel ? k : 0];
        int32_t z = (zp.data_type() == DataType::i8)
            ? static_cast<int32_t>(zp.ptr<int8_t>()[is_per_channel ? k : 0])
            : static_cast<int32_t>(zp.ptr<uint8_t>()[is_per_channel ? k : 0]);
        z_f32[static_cast<size_t>(k)] = static_cast<float>(z);
    }

    const int64_t pack = X.channel_pack_size();

    // Helper: load a float32 vector from input (f32 or f16 → f32)
    auto load_input8 = [](const T* p) -> v_f32x8 {
        if constexpr (std::is_same_v<T, half>) {
            // v_f16x8 → v_f32x8 conversion via scalar buffer (avoid ADL ambiguity)
            alignas(16) half htmp[8];
            v_store(htmp, v_load(p));
            float fbuf[8];
            for (int i = 0; i < 8; ++i) fbuf[i] = half_to_float(htmp[i]);
            return v_load(fbuf);
        } else {
            return v_load(p);
        }
    };

    if (pack > 1 && axis == rank - 1) {
        // ---- Packed SIMD: axis == last dim (W), per-lane processing ----
        const int64_t num_rows = X.total_rows();
        const int64_t x_rs = X.row_stride_elems();
        const int64_t y_rs = output.row_stride_elems();

        const float s_val = s_f32[0];
        const float z_val = z_f32[0];

        const auto process_row = [&](int64_t r) {
            const T* x_row = X.ptr<T>() + r * x_rs;
            void* y_row = static_cast<char*>(output.ptr<void>()) + r * y_rs;

            for (int64_t w = 0; w < D; ++w) {
                int64_t off = w * pack;
                auto vx = load_input8(x_row + off);
                float inv_s = is_per_channel ? (1.0f / s_f32[static_cast<size_t>(w)]) : (1.0f / s_val);
                auto vs = v_set1_f32x8(inv_s);
                auto vzp = v_set1_f32x8(is_per_channel ? z_f32[static_cast<size_t>(w)] : z_val);
                auto vr = v_add(v_mul(vx, vs), vzp);

                alignas(32) float ftmp[8];
                v_store(ftmp, vr);
                for (int lane = 0; lane < pack; ++lane) {
                    int32_t q = static_cast<int32_t>(std::lround(ftmp[lane]));
                    write_int8(y_row, out_dtype, off + lane, q);
                }
            }
        };

        if (ctx.cpu_parallel_for)
            ctx.cpu_parallel_for(0, num_rows, process_row);
        else
            for (int64_t r = 0; r < num_rows; ++r) process_row(r);

    } else if (pack > 1) {
        // ---- Packed general axis: scalar ----
        const int64_t num_outer = [&]() {
            int64_t n = 1;
            for (int64_t d = 0; d < axis; ++d) n *= X.shape(d);
            return n;
        }();
        const int64_t num_inner = [&]() {
            int64_t n = 1;
            for (int64_t d = axis + 1; d < rank; ++d) n *= X.shape(d);
            return n;
        }();
        const int64_t x_outer_stride = (axis > 0) ? X.stride_elems(axis - 1) : 0;
        const int64_t x_axis_stride = X.stride_elems(axis);

        const auto process_outer = [&](int64_t outer) {
            const T* x_base = X.ptr<T>() + outer * x_outer_stride;
            void* y_base = static_cast<char*>(output.ptr<void>()) + outer * x_outer_stride;

            for (int64_t inner = 0; inner < num_inner; ++inner) {
                int64_t inner_off = 0;
                {
                    int64_t rem = inner;
                    for (int64_t d = rank - 1; d > axis; --d) {
                        inner_off += (rem % X.shape(d)) * X.stride_elems(d);
                        rem /= X.shape(d);
                    }
                }
                for (int64_t k = 0; k < D; ++k) {
                    float xv;
                    if constexpr (std::is_same_v<T, half>)
                        xv = half_to_float(x_base[inner_off + k * x_axis_stride]);
                    else
                        xv = static_cast<float>(x_base[inner_off + k * x_axis_stride]);
                    float q = std::round(xv / s_f32[static_cast<size_t>(k)]) + z_f32[static_cast<size_t>(k)];
                    write_int8(y_base, out_dtype, inner_off + k * x_axis_stride, static_cast<int32_t>(q));
                }
            }
        };

        if (ctx.cpu_parallel_for)
            ctx.cpu_parallel_for(0, num_outer, process_outer);
        else
            for (int64_t o = 0; o < num_outer; ++o) process_outer(o);

    } else {
        // ---- Planar layout (NCHW/NCDHW) ----
        int64_t stride_before_axis = 1;
        for (int64_t d = 0; d < axis; ++d) stride_before_axis *= X.shape(d);

        int64_t stride_after_axis = 1;
        for (int64_t d = axis + 1; d < rank; ++d) stride_after_axis *= X.shape(d);

        const auto process_outer = [&](int64_t outer) {
            int64_t base = outer * D * stride_after_axis;

            // Contiguous tail with per-tensor scale: SIMD fast path
            if (!is_per_channel && (stride_after_axis == 1 || axis == rank - 1)) {
                int64_t i = base;
                float inv_s = 1.0f / s_f32[0];
                auto vs = v_set1_f32x8(inv_s);
                auto vzp = v_set1_f32x8(z_f32[0]);
                for (; i + L <= base + D * stride_after_axis; i += L) {
                    auto vx = load_input8(X.ptr<T>() + i);
                    auto vr = v_add(v_mul(vx, vs), vzp);
                    alignas(32) float ftmp[8];
                    v_store(ftmp, vr);
                    for (int lane = 0; lane < L; ++lane) {
                        int32_t q = static_cast<int32_t>(std::lround(ftmp[lane]));
                        write_int8(output.ptr<void>(), out_dtype, i + lane, q);
                    }
                }
                for (; i < base + D * stride_after_axis; ++i) {
                    float xv;
                    if constexpr (std::is_same_v<T, half>)
                        xv = half_to_float(X.ptr<T>()[i]);
                    else
                        xv = X.ptr<float>()[i];
                    float q = std::round(xv * inv_s) + z_f32[0];
                    write_int8(output.ptr<void>(), out_dtype, i, static_cast<int32_t>(q));
                }
            } else {
                // Per-channel or non-contiguous: scalar loop with per-k scale/zp
                for (int64_t k = 0; k < D; ++k) {
                    float inv_s = 1.0f / s_f32[static_cast<size_t>(k)];
                    float z = z_f32[static_cast<size_t>(k)];
                    auto vs = v_set1_f32x8(inv_s);
                    auto vzp = v_set1_f32x8(z);
                    int64_t ks = base + k * X.stride_elems(axis);
                    int64_t i = 0;
                    for (; i + L <= stride_after_axis; i += L) {
                        auto vx = load_input8(X.ptr<T>() + ks + i);
                        auto vr = v_add(v_mul(vx, vs), vzp);
                        alignas(32) float ftmp[8];
                        v_store(ftmp, vr);
                        for (int lane = 0; lane < L; ++lane) {
                            int32_t q = static_cast<int32_t>(std::lround(ftmp[lane]));
                            write_int8(output.ptr<void>(), out_dtype, ks + i + lane, q);
                        }
                    }
                    for (; i < stride_after_axis; ++i) {
                        float xv;
                        if constexpr (std::is_same_v<T, half>)
                            xv = half_to_float(X.ptr<T>()[ks + i]);
                        else
                            xv = X.ptr<float>()[ks + i];
                        float q = std::round(xv * inv_s) + z;
                        write_int8(output.ptr<void>(), out_dtype, ks + i, static_cast<int32_t>(q));
                    }
                }
            }
        };

        if (ctx.cpu_parallel_for)
            ctx.cpu_parallel_for(0, stride_before_axis, process_outer);
        else
            for (int64_t o = 0; o < stride_before_axis; ++o) process_outer(o);
    }
}

// ============================================================
// DequantizeLinear: integer → float
// ============================================================

template <typename T>  // T = float or half (output type)
void dequantize_linear_impl(const QuantLinearAttributes& attrs,
                            TensorView& output,
                            std::span<const TensorView> inputs,
                            const ComputeContext& ctx)
{
    const auto& X     = inputs[0];  // i8 or u8
    const auto& scale = inputs[1];  // f32
    const auto& zp    = inputs[2];  // i8 or u8

    const int64_t rank = X.rank();
    int64_t axis = attrs.axis;
    if (axis < 0) axis += rank;

    const int64_t D = X.shape(axis);
    const bool is_per_channel = (scale.numel() > 1);
    const DataType in_dtype = X.data_type();
    const bool in_is_i8 = (in_dtype == DataType::i8);

    std::vector<float> s_f32(static_cast<size_t>(D));
    std::vector<float> z_f32(static_cast<size_t>(D));
    for (int64_t k = 0; k < D; ++k) {
        s_f32[static_cast<size_t>(k)] = scale.ptr<float>()[is_per_channel ? k : 0];
        int32_t z = (zp.data_type() == DataType::i8)
            ? static_cast<int32_t>(zp.ptr<int8_t>()[is_per_channel ? k : 0])
            : static_cast<int32_t>(zp.ptr<uint8_t>()[is_per_channel ? k : 0]);
        z_f32[static_cast<size_t>(k)] = static_cast<float>(z);
    }

    const int64_t pack = X.channel_pack_size();

    // Helper: load 8 int8/uint8 values and widen to v_f32x8
    auto load_int8_to_f32 = [in_is_i8](const void* p) -> v_f32x8 {
        if (in_is_i8)
            return v_cvt_i8_to_f32(static_cast<const int8_t*>(p));
        else
            return v_cvt_u8_to_f32(static_cast<const uint8_t*>(p));
    };

    // Helper: store v_f32x8 to output (f32 or f16)
    auto store_output8 = [](T* dest, const v_f32x8& vy) {
        if constexpr (std::is_same_v<T, half>) {
            alignas(32) float fbuf[8];
            v_store(fbuf, vy);
            for (int i = 0; i < 8; ++i) dest[i] = float_to_half(fbuf[i]);
        } else {
            v_store(dest, vy);
        }
    };

    if (pack > 1 && axis == rank - 1) {
        // ---- Packed SIMD: axis == last dim (W), per-lane processing ----
        const int64_t num_rows = X.total_rows();
        const int64_t x_rs = X.row_stride_elems();
        const int64_t y_rs = output.row_stride_elems();
        const int64_t W = X.shape(rank - 1);

        const float s_val = s_f32[0];
        const float z_val = z_f32[0];

        const auto process_row = [&](int64_t r) {
            const void* x_row = static_cast<const char*>(X.ptr<void>()) + r * x_rs;
            T* y_row = output.ptr<T>() + r * y_rs;

            for (int64_t w = 0; w < W; ++w) {
                int64_t off = w * pack;
                auto vx = load_int8_to_f32(static_cast<const char*>(x_row) + off);
                float s = is_per_channel ? s_f32[static_cast<size_t>(w)] : s_val;
                float z = is_per_channel ? z_f32[static_cast<size_t>(w)] : z_val;
                auto vs = v_set1_f32x8(s);
                auto vz = v_set1_f32x8(z);
                auto vy = v_mul(v_sub(vx, vz), vs);
                store_output8(y_row + off, vy);
            }
        };

        if (ctx.cpu_parallel_for)
            ctx.cpu_parallel_for(0, num_rows, process_row);
        else
            for (int64_t r = 0; r < num_rows; ++r) process_row(r);

    } else if (pack > 1) {
        // ---- Packed general axis: scalar ----
        const int64_t num_outer = [&]() {
            int64_t n = 1;
            for (int64_t d = 0; d < axis; ++d) n *= X.shape(d);
            return n;
        }();
        const int64_t num_inner = [&]() {
            int64_t n = 1;
            for (int64_t d = axis + 1; d < rank; ++d) n *= X.shape(d);
            return n;
        }();
        const int64_t x_outer_stride = (axis > 0) ? X.stride_elems(axis - 1) : 0;
        const int64_t x_axis_stride = X.stride_elems(axis);

        const auto process_outer = [&](int64_t outer) {
            const void* x_base = static_cast<const char*>(X.ptr<void>()) + outer * x_outer_stride;
            T* y_base = output.ptr<T>() + outer * x_outer_stride;

            for (int64_t inner = 0; inner < num_inner; ++inner) {
                int64_t inner_off = 0;
                {
                    int64_t rem = inner;
                    for (int64_t d = rank - 1; d > axis; --d) {
                        inner_off += (rem % X.shape(d)) * X.stride_elems(d);
                        rem /= X.shape(d);
                    }
                }
                for (int64_t k = 0; k < D; ++k) {
                    int64_t off = inner_off + k * x_axis_stride;
                    int32_t iv = in_is_i8
                        ? static_cast<int32_t>(static_cast<const int8_t*>(x_base)[off])
                        : static_cast<int32_t>(static_cast<const uint8_t*>(x_base)[off]);
                    float yv = (static_cast<float>(iv) - z_f32[static_cast<size_t>(k)]) * s_f32[static_cast<size_t>(k)];
                    if constexpr (std::is_same_v<T, half>)
                        y_base[off] = float_to_half(yv);
                    else
                        y_base[off] = yv;
                }
            }
        };

        if (ctx.cpu_parallel_for)
            ctx.cpu_parallel_for(0, num_outer, process_outer);
        else
            for (int64_t o = 0; o < num_outer; ++o) process_outer(o);

    } else {
        // ---- Planar layout (NCHW/NCDHW) ----
        int64_t stride_before_axis = 1;
        for (int64_t d = 0; d < axis; ++d) stride_before_axis *= X.shape(d);

        int64_t stride_after_axis = 1;
        for (int64_t d = axis + 1; d < rank; ++d) stride_after_axis *= X.shape(d);

        const auto process_outer = [&](int64_t outer) {
            int64_t base = outer * D * stride_after_axis;

            // Per-tensor scale with contiguous tail: SIMD fast path
            if (!is_per_channel && (axis == rank - 1 || stride_after_axis == 1)) {
                float s = s_f32[0];
                float z = z_f32[0];
                const int64_t total = D * stride_after_axis;
                int64_t i = base;
                auto vs = v_set1_f32x8(s);
                auto vz = v_set1_f32x8(z);
                for (; i + L <= base + total; i += L) {
                    auto vx = load_int8_to_f32(static_cast<const char*>(X.ptr<void>()) + i);
                    auto vy = v_mul(v_sub(vx, vz), vs);
                    store_output8(output.ptr<T>() + i, vy);
                }
                for (; i < base + total; ++i) {
                    int32_t iv = in_is_i8
                        ? static_cast<int32_t>(X.ptr<int8_t>()[i])
                        : static_cast<int32_t>(X.ptr<uint8_t>()[i]);
                    float yv = (static_cast<float>(iv) - z) * s;
                    if constexpr (std::is_same_v<T, half>)
                        output.ptr<T>()[i] = float_to_half(yv);
                    else
                        output.ptr<float>()[i] = yv;
                }
            } else {
                // Per-channel or non-contiguous: iterate per axis element
                for (int64_t k = 0; k < D; ++k) {
                    float s = s_f32[static_cast<size_t>(k)];
                    float z = z_f32[static_cast<size_t>(k)];
                    int64_t ks = base + k * X.stride_elems(axis);
                    int64_t i = 0;
                    auto vs = v_set1_f32x8(s);
                    auto vz = v_set1_f32x8(z);
                    for (; i + L <= stride_after_axis; i += L) {
                        auto vx = load_int8_to_f32(static_cast<const char*>(X.ptr<void>()) + ks + i);
                        auto vy = v_mul(v_sub(vx, vz), vs);
                        store_output8(output.ptr<T>() + ks + i, vy);
                    }
                    for (; i < stride_after_axis; ++i) {
                        int32_t iv = in_is_i8
                            ? static_cast<int32_t>(X.ptr<int8_t>()[ks + i])
                            : static_cast<int32_t>(X.ptr<uint8_t>()[ks + i]);
                        float yv = (static_cast<float>(iv) - z) * s;
                        if constexpr (std::is_same_v<T, half>)
                            output.ptr<T>()[ks + i] = float_to_half(yv);
                        else
                            output.ptr<float>()[ks + i] = yv;
                    }
                }
            }
        };

        if (ctx.cpu_parallel_for)
            ctx.cpu_parallel_for(0, stride_before_axis, process_outer);
        else
            for (int64_t o = 0; o < stride_before_axis; ++o) process_outer(o);
    }
}

// ============================================================
// Entry points
// ============================================================

void quantize_linear_cpu(const QuantLinearAttributes& attrs,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         void* /*workspace*/)
{
    switch (inputs[0].data_type()) {
    case DataType::f32:
        quantize_linear_impl<float>(attrs, output, inputs, ctx);
        break;
    case DataType::f16:
        quantize_linear_impl<half>(attrs, output, inputs, ctx);
        break;
    default:
        NNOPS_ASSERT(!"quantize_linear_cpu: unsupported input dtype");
    }
}

void dequantize_linear_cpu(const QuantLinearAttributes& attrs,
                           TensorView& output,
                           std::span<const TensorView> inputs,
                           const ComputeContext& ctx,
                           void* /*workspace*/)
{
    switch (output.data_type()) {
    case DataType::f32:
        dequantize_linear_impl<float>(attrs, output, inputs, ctx);
        break;
    case DataType::f16:
        dequantize_linear_impl<half>(attrs, output, inputs, ctx);
        break;
    default:
        NNOPS_ASSERT(!"dequantize_linear_cpu: unsupported output dtype");
    }
}

}  // namespace nnops::backend::cpu
