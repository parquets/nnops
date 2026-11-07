/// @file linear.cpp
/// @brief CPU Linear: output = epilogue(X × Wᵀ + bias), optional add_to.
///
/// Linear is MatMul with transpose_b plus an optional per-column bias, so it
/// drives the MatMul kernel instead of duplicating the GEMM:
///
///   - f32/f16 X and W: MatMul's fp route; the output keeps X's dtype.
///   - f32/f16 X × q4_1 W: MatMul's block-quant routes — the fused small-M
///     kernels decode into the accumulator, the packed plan decodes into its
///     panels — so the weight stays 20 bytes per 32 values. The output keeps
///     X's dtype.
///   - u8|s8 X × s8 W:  MatMul's int8 route, with the output dtype taken from
///     `attrs.output_dtype` — f32/f16 dequantizes every int32 accumulate with
///     X's and W's QuantParams, s32/s8/u8 keeps it integral (requantizing the
///     last two with the output's QuantParams).
///
/// Bias and the activation are MatMul's: it adds the bias between the product
/// and the epilogue (an fp bias on an fp output, an int32 bias on an integer
/// one). Linear forwards its third input and sets the epilogue; what is left
/// here is add_to, which rides MatMul's beta when the operands are fp and the
/// epilogue is identity — the residual must not pass through the activation —
/// and otherwise goes through a compact M×N temporary. add_to on an integer
/// output is not implemented (the residual would need its own requantized add).

#include "nnops/ops/linear.hpp"
#include "nnops/ops/matmul.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"
#include "backend/cpu/matmul/matmul.h"
#include "backend/cpu/common/memory_pool.hpp"
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"
#include "backend/cpu/simd_kernel/tiled_map.hpp"
#include "nnops/detail/validate.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

namespace nnops::backend::cpu {

namespace {

/// fp keeps X's dtype; the quantized path follows attrs.output_dtype, else f32.
DataType linear_output_dtype(const LinearAttributes& attrs, DataType x_dt) noexcept {
    if (x_dt == DataType::f32 || x_dt == DataType::f16) { return x_dt; }
    switch (attrs.output_dtype) {
    case DataType::f16:
    case DataType::s32:
    case DataType::s8:
    case DataType::u8:
        return attrs.output_dtype;
    default:
        return DataType::f32;
    }
}

/// Raw data pointer of a view, for building a promoted (rank-1 → rank-2) view.
void* view_data(const TensorView& t) noexcept {
    return const_cast<void*>(static_cast<const void*>(t.ptr<uint8_t>()));
}

/// The gated form: one MatMul over the [2N, K] weight into a pooled [rows, 2N]
/// tile, then the SwiGLU combine into the caller's [rows, N] output. Both
/// halves must exist before the combine, so the gate cannot ride MatMul's
/// epilogue — but the whole path is one GEMM plus one output pass, where a
/// graph would otherwise run the GEMM, a split, a Silu and a multiply.
template <typename T>
Status linear_gated_impl(const LinearAttributes& attrs, TensorView& output,
                         std::span<const TensorView> inputs, int64_t rows,
                         int64_t n2, const ComputeContext& ctx, void* workspace)
{
    const int64_t n = n2 / 2;
    PoolPtr tmp(static_cast<size_t>(rows) * static_cast<size_t>(n2) * sizeof(T));
    if (!tmp) { return Status::out_of_memory; }
    int64_t tshape[2] = {rows, n2};
    TensorView mt(std::span<const int64_t>(tshape, 2),
                  std::is_same_v<T, float> ? DataType::f32 : DataType::f16,
                  tmp.get(), n2 * static_cast<int64_t>(sizeof(T)));

    MatMulAttributes ma;
    ma.transpose_b = true;
    ma.epilogue = Epilogue{};   // the combine applies Linear's
    NNOPS_TRY(matmul_cpu(ma, mt, inputs, ctx, workspace));

    const float alpha = attrs.gate_alpha;
    const float beta = attrs.gate_beta;
    const float limit = attrs.gate_limit;
    const bool clamp = std::isfinite(limit);
    const Epilogue ep = attrs.epilogue;
    const bool add_to = attrs.add_to;
    const T* tag = output.ptr<T>();

    // gate · sigmoid(alpha·gate) · (value + beta), both halves clamped to
    // ±limit when finite — the MoE operator's combine, applied to one GEMM's
    // two column halves.
    auto vec = [=](auto g, auto v) {
        if (clamp) {
            const auto vlim = simd::v_set1(tag, limit);
            const auto vnlim = simd::v_neg(vlim);
            g = simd::v_max(simd::v_min(g, vlim), vnlim);
            v = simd::v_max(simd::v_min(v, vlim), vnlim);
        }
        const auto one = simd::v_set1(tag, 1.0f);
        const auto sig = simd::v_div(
            one, simd::v_add(one, simd::v_exp(
                                      simd::v_neg(simd::v_mul(g, simd::v_set1(tag, alpha))))));
        const auto r = simd::v_mul(simd::v_mul(g, sig),
                                   simd::v_add(v, simd::v_set1(tag, beta)));
        return apply_epilogue_vec(ep, tag, r);
    };
    auto scalar = [=](float g, float v) {
        if (clamp) {
            g = std::max(std::min(g, limit), -limit);
            v = std::max(std::min(v, limit), -limit);
        }
        const float sig = 1.0f / (1.0f + std::exp(-alpha * g));
        return apply_epilogue(ep, g * sig * (v + beta));
    };

    T* base = mt.ptr<T>();
    kernel::tiled_map_simd<T, 2>({base, base + n}, {n2, n2},
                                 output.ptr<T>(), output.row_stride_elems(),
                                 rows, n, add_to, vec, scalar);
    return Status::ok;
}

/// out += src over every row; MatMul already put bias and activation through `src`.
template <class T>
void residual_add(TensorView& out, const TensorView& src, const ComputeContext& ctx)
{
    const int64_t cols = out.shape(out.rank() - 1);
    const int64_t rows = out.total_rows();
    if (cols <= 0 || rows <= 0) { return; }

    ctx.cpu.run(0, rows, [&](int64_t r) {
        T* d = out.ptr<T>(static_cast<size_t>(r));
        const T* s = src.ptr<T>(static_cast<size_t>(r));
        for (int64_t n = 0; n < cols; ++n) {
            if constexpr (std::is_same_v<T, half>) {
                d[n] = float_to_half(half_to_float(d[n]) + half_to_float(s[n]));
            } else {
                d[n] += s[n];
            }
        }
    });
}

}  // anonymous namespace

Status linear_cpu(const LinearAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* workspace)
{
    NNOPS_ASSERT(inputs.size() >= 2 && inputs.size() <= 3);
    const auto& x = inputs[0];
    const auto& w = inputs[1];
    const bool has_bias = inputs.size() > 2;

    NNOPS_ASSERT(x.rank() >= 1);
    NNOPS_ASSERT(w.rank() == 2);

    const DataType x_dt = x.data_type();
    const DataType w_dt = w.data_type();
    const DataType out_dt = linear_output_dtype(attrs, x_dt);
    NNOPS_ASSERT(output.data_type() == out_dt);

    const bool quant = is_int8_matmul(x_dt, w_dt);
    // A block-packed weight (q4_1) is self-describing and decodes to X's type
    // inside MatMul, so it pairs with a float activation only.
    const bool block_rhs = is_block_packed_dtype(w_dt) &&
                           (x_dt == DataType::f32 || x_dt == DataType::f16);
    NNOPS_ASSERT(quant || block_rhs ||
                 (x_dt == w_dt && (x_dt == DataType::f32 || x_dt == DataType::f16)));
    // Integer output -> int32 bias (MatMul's rule).
    const bool int_out = out_dt == DataType::s32 || out_dt == DataType::s8 ||
                         out_dt == DataType::u8;
    NNOPS_ASSERT_MSG(!(attrs.add_to && int_out),
                     "Linear: add_to needs an fp output");
    if (has_bias) {
        NNOPS_ASSERT(inputs[2].data_type() == (int_out ? DataType::s32 : out_dt));
        NNOPS_ASSERT(inputs[2].numel() >= w.shape(0));
    }

    const int64_t K = x.shape(x.rank() - 1);
    // In the gated form the weight is [2N, K] and the output's last dim is N.
    const int64_t N = attrs.gated ? w.shape(0) / 2 : w.shape(0);

    // A rank-1 X (and its [N] output) is one row over the same buffers.
    std::optional<TensorView> x_row, out_row;
    if (x.rank() == 1) {
        const int64_t xs[2] = {1, K};
        x_row.emplace(std::span<const int64_t>(xs, 2), x_dt, view_data(x),
                      K * static_cast<int64_t>(data_type_size(x_dt)),
                      x.layout(), x.quant_params());
        const int64_t os[2] = {1, N};
        // The quant params ride along: an s8/u8 output requantizes with them.
        out_row.emplace(std::span<const int64_t>(os, 2), out_dt, view_data(output),
                        N * static_cast<int64_t>(data_type_size(out_dt)),
                        output.layout(), output.quant_params());
    }
    TensorView& ov = out_row ? *out_row : output;

    TensorView minputs[3] = {x_row ? *x_row : x, w, TensorView{}};
    if (has_bias) { minputs[2] = inputs[2]; }
    const std::span<const TensorView> min(
        minputs, has_bias ? size_t{3} : size_t{2});

    if (attrs.gated) {
        return x_dt == DataType::f32
                   ? linear_gated_impl<float>(attrs, ov, min, ov.total_rows(),
                                              w.shape(0), ctx, workspace)
                   : linear_gated_impl<half>(attrs, ov, min, ov.total_rows(),
                                             w.shape(0), ctx, workspace);
    }

    MatMulAttributes ma;
    ma.transpose_b = true;
    ma.output_dtype = out_dt;   // read only on the int8 route
    ma.epilogue = attrs.epilogue;

    // add_to can ride beta only for fp operands with an identity epilogue: the
    // residual must not pass through the activation, and int8 zeroes C.
    const bool ep_identity = !epilogue_is_active(attrs.epilogue);
    const bool beta_residual = attrs.add_to && !quant && ep_identity;
    if (beta_residual) { ma.beta = 1.0f; }

    if (attrs.add_to && !beta_residual) {
        // MatMul writes a compact temporary; the residual is added after its epilogue.
        const int64_t out_rows = ov.total_rows();
        PoolPtr tmp(static_cast<size_t>(out_rows) * static_cast<size_t>(N)
                    * data_type_size(out_dt));
        TensorView mt(ov.shape_span(), out_dt, tmp.get());
        matmul_cpu(ma, mt, min, ctx, workspace);

        if (out_dt == DataType::f32) {
            residual_add<float>(ov, mt, ctx);
        } else {
            residual_add<half>(ov, mt, ctx);
        }
        return Status::ok;
    }

    // MatMul's status is Linear's — e.g. a prepacked weight whose K disagrees
    // with the activation is refused there, and the caller must see that.
    return matmul_cpu(ma, ov, min, ctx, workspace);
}

}  // namespace nnops::backend::cpu
