/// @file linear_ref.cpp
/// @brief Naive CPU reference implementation of Linear / fully-connected layer.
///
/// Computes: output = epilogue(input × weight^T + bias), optional add_to
///   input:  [..., K], weight: [N, K], bias: [N] (optional), output: [..., N]
///
/// Supported operand combinations, matching the operator's contract:
///   - f32/f16 input and weight          → output follows the input dtype
///   - u8/s8 input × s8 weight (quantized) → output f32/f16, or s32/s8/u8
///
/// An integer output takes an int32 [N] bias and runs MatMul's integer output
/// pass (compensate, bias, Relu, requant). The quantized path uses the same
/// expression as MatMul's int8 epilogue:
///
///   acc = Σ_k (x[m,k] − zp_x[m]) · (w[n,k] − zp_w[n])
///   f   = acc · s_x[m] · s_w[n]
///
/// with per-tensor, per-token (rows) or per-channel (columns) parameters.

#include "nnops/ops/linear.hpp"
#include "nnops/ops/matmul.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"
#include "simd_kernel/simd_epilogue.hpp"

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace nnops::backend::cpu::reference {

namespace {

using nnops::backend::cpu::float_to_half;
using nnops::backend::cpu::half;
using nnops::backend::cpu::half_to_float;

/// Read one operand element as float (half widens through F16C-equivalent).
template <class T>
inline float as_float(T v) noexcept {
    if constexpr (std::is_same_v<T, half>) { return half_to_float(v); }
    else { return static_cast<float>(v); }
}

/// Store one result element in the output type.
template <class T>
inline void store_as(T& dst, float v) noexcept {
    if constexpr (std::is_same_v<T, half>) { dst = float_to_half(v); }
    else { dst = v; }
}

inline int32_t zp_at(const QuantParams& q, int64_t idx) noexcept {
    return (q.zero_point_data != nullptr) ? q.zero_point_data[idx] : q.zero_point;
}
inline float scale_at(const QuantParams& q, int64_t idx) noexcept {
    return (q.scale_data != nullptr) ? q.scale_data[idx] : q.scale;
}

/// XT = input element, WT = weight element (same as XT for fp, int8_t for the
/// quantized path), CT = output element.
template <class XT, class WT, class CT>
void linear_ref_impl(const LinearAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx)
{
    const auto& x = inputs[0];
    const auto& w = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t K = x.shape(x.rank() - 1);
    const int64_t N = w.shape(0);
    const int64_t rows = (x.rank() >= 2) ? x.total_rows() : 1;

    // The element types select the path: integral operands are the quantized
    // (u8|s8 × s8) case, floating-point operands the fp case. An integral CT is
    // the quantized path's integer destination (s32 / s8 / u8), where the bias
    // is an int32 and Relu + the requant replace the fp epilogue.
    constexpr bool quant   = std::is_integral_v<XT>;
    constexpr bool int_out = std::is_integral_v<CT>;
    NNOPS_ASSERT(!has_bias || inputs[2].data_type() ==
                                 (int_out ? DataType::s32 : output.data_type()));
    NNOPS_ASSERT(int_out || output.data_type() == DataType::f32 ||
                 output.data_type() == DataType::f16);
    NNOPS_ASSERT(!attrs.add_to || !int_out);

    const QuantParams& qx = x.quant_params();
    const QuantParams& qw = w.quant_params();
    [[maybe_unused]] const CT* bias =
        (has_bias && !int_out) ? inputs[2].ptr<CT>() : nullptr;
    [[maybe_unused]] const int32_t* int_bias =
        (has_bias && int_out) ? inputs[2].ptr<int32_t>() : nullptr;

    // The output requant's parameters, read only for an integral destination.
    [[maybe_unused]] const QuantParams& qc = output.quant_params();
    [[maybe_unused]] const float out_scale = scale_at(qc, 0);
    [[maybe_unused]] const int32_t zp_out = zp_at(qc, 0);

    // Raw reductions of W: r_b[n] = Σ_k W[n,k]. Needed only on the quantized
    // path (the compensation subtracts zp_w·r_x + zp_x·r_w).
    std::vector<int32_t> r_b;
    if constexpr (quant) {
        r_b.resize(static_cast<size_t>(N));
        for (int64_t n = 0; n < N; ++n) {
            const WT* wr = w.ptr<WT>(static_cast<size_t>(n));
            int32_t s = 0;
            for (int64_t k = 0; k < K; ++k) { s += static_cast<int32_t>(wr[k]); }
            r_b[static_cast<size_t>(n)] = s;
        }
    }

    ctx.cpu.run(0, rows, [&](int64_t row) {
        const XT* xr = x.ptr<XT>(static_cast<size_t>(row));
        CT* orow = output.ptr<CT>(static_cast<size_t>(row));

        int32_t r_a = 0;
        if constexpr (quant) {
            for (int64_t k = 0; k < K; ++k) {
                r_a += static_cast<int32_t>(xr[k]);
            }
        }

        for (int64_t n = 0; n < N; ++n) {
            const WT* wr = w.ptr<WT>(static_cast<size_t>(n));
            float f;

            if constexpr (quant) {
                int32_t raw = 0;
                for (int64_t k = 0; k < K; ++k) {
                    raw += static_cast<int32_t>(xr[k]) *
                           static_cast<int32_t>(wr[k]);
                }
                const int32_t zp_x = zp_at(qx, row);
                const int32_t zp_w = zp_at(qw, n);
                const int32_t acc = matmul_int8_compensate(
                    raw, 0, zp_x, zp_w, r_a, r_b[static_cast<size_t>(n)],
                    static_cast<int32_t>(K));
                if constexpr (int_out) {
                    // Same order as MatMul's integer output pass: compensate,
                    // bias, Relu, requant.
                    int32_t v = acc;
                    if (int_bias != nullptr) { v += int_bias[n]; }
                    if (attrs.epilogue.type == EpilogueActivateType::Relu) {
                        v = std::max(v, 0);
                    }
                    if constexpr (std::is_same_v<CT, int32_t>) {
                        orow[n] = v;
                    } else {
                        const float req = matmul_int8_requant_scale(
                            scale_at(qx, row), scale_at(qw, n), out_scale);
                        orow[n] = matmul_int8_requant<CT>(v, req, zp_out);
                    }
                    continue;
                }
                f = static_cast<float>(acc) *
                    (scale_at(qx, row) * scale_at(qw, n));
            } else {
                float sum = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    sum += as_float(xr[k]) * as_float(wr[k]);
                }
                f = sum;
            }

            if (bias != nullptr) { f += as_float(bias[n]); }
            f = apply_epilogue(attrs.epilogue, f);
            if (attrs.add_to) { f += as_float(orow[n]); }
            store_as(orow[n], f);
        }
    });
}

template <class XT, class WT>
void linear_ref_out(const LinearAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx)
{
    switch (output.data_type()) {
    case DataType::f16: linear_ref_impl<XT, WT, half>(attrs, output, inputs, ctx); break;
    case DataType::s32: linear_ref_impl<XT, WT, int32_t>(attrs, output, inputs, ctx); break;
    case DataType::s8:  linear_ref_impl<XT, WT, int8_t>(attrs, output, inputs, ctx); break;
    case DataType::u8:  linear_ref_impl<XT, WT, uint8_t>(attrs, output, inputs, ctx); break;
    default:            linear_ref_impl<XT, WT, float>(attrs, output, inputs, ctx); break;
    }
}

}  // anonymous namespace

void linear_ref(const LinearAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* /*workspace*/)
{
    NNOPS_ASSERT(inputs.size() >= 2 && inputs.size() <= 3);
    NNOPS_ASSERT(!output.is_empty());

    const DataType x_dt = inputs[0].data_type();
    const DataType w_dt = inputs[1].data_type();

    if (x_dt == DataType::f32) {
        linear_ref_out<float, float>(attrs, output, inputs, ctx);
    } else if (x_dt == DataType::f16) {
        linear_ref_out<half, half>(attrs, output, inputs, ctx);
    } else if (x_dt == DataType::u8) {
        NNOPS_ASSERT(w_dt == DataType::s8);
        linear_ref_out<uint8_t, int8_t>(attrs, output, inputs, ctx);
    } else if (x_dt == DataType::s8) {
        NNOPS_ASSERT(w_dt == DataType::s8);
        linear_ref_out<int8_t, int8_t>(attrs, output, inputs, ctx);
    } else {
        NNOPS_ASSERT(!"linear_ref: unsupported input data type");
    }
}

}  // namespace nnops::backend::cpu::reference
