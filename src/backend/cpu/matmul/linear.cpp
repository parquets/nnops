/// @file linear.cpp
/// @brief CPU Linear: output = epilogue(X × Wᵀ + bias), optional add_to.
///
/// Linear is a MatMul with a transposed weight and an optional per-column bias,
/// so it drives the MatMul kernel instead of duplicating the GEMM:
///
///   - f32/f16 X and W: MatMul's fp route; the output follows X.
///   - u8|s8 X × s8 W:  MatMul's int8 route, with the output dtype taken from
///     `output_dtype` — f32/f16 dequantizes every int32 accumulate with X's and
///     W's QuantParams, s32/s8/u8 keeps it integral (requantizing for the last
///     two with the output's QuantParams).
///
/// Bias and the activation are MatMul's now: it adds the bias between the
/// product and the epilogue (an fp bias on the fp outputs, an int32 bias in the
/// int8 output pass on the integer ones). Linear forwards its third input and
/// sets the epilogue; the only thing left here is add_to, which rides MatMul's
/// beta when the residual must not pass through the activation, and otherwise
/// goes through a compact M×N temporary. add_to on an integer output is not
/// implemented (the residual would need its own requantized add).

#include "nnops/ops/linear.hpp"
#include "nnops/ops/matmul.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"
#include "backend/cpu/matmul/matmul.h"
#include "backend/cpu/common/memory_pool.hpp"

#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

namespace nnops::backend::cpu {

namespace {

/// The output type the Linear resolves. fp operands keep their own dtype; the
/// quantized path follows `output_dtype` — f32/f16 dequantizes, s32/s8/u8 keeps
/// the result integral — with anything else falling back to f32.
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

/// out += src, over every output row — the add_to fallback, after MatMul has
/// already applied the bias and the activation to the temporary.
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
    NNOPS_ASSERT(quant || (x_dt == w_dt &&
                           (x_dt == DataType::f32 || x_dt == DataType::f16)));
    // Integer output -> int32 bias (MatMul's rule); fp output -> the output
    // dtype. add_to needs a quantized residual add, which does not exist yet.
    const bool int_out = out_dt == DataType::s32 || out_dt == DataType::s8 ||
                         out_dt == DataType::u8;
    NNOPS_ASSERT_MSG(!(attrs.add_to && int_out),
                     "Linear: add_to needs an fp output");
    if (has_bias) {
        NNOPS_ASSERT(inputs[2].data_type() == (int_out ? DataType::s32 : out_dt));
        NNOPS_ASSERT(inputs[2].numel() >= w.shape(0));
    }

    const int64_t K = x.shape(x.rank() - 1);
    const int64_t N = w.shape(0);

    // MatMul works on matrices: a rank-1 X (and its [N] output) is viewed as
    // one row over the same buffers.
    std::optional<TensorView> x_row, out_row;
    if (x.rank() == 1) {
        const int64_t xs[2] = {1, K};
        x_row.emplace(std::span<const int64_t>(xs, 2), x_dt, view_data(x),
                      K * static_cast<int64_t>(data_type_size(x_dt)),
                      x.layout(), x.quant_params());
        const int64_t os[2] = {1, N};
        // The quant params travel with the view: an s8/u8 output requantizes
        // with them.
        out_row.emplace(std::span<const int64_t>(os, 2), out_dt, view_data(output),
                        N * static_cast<int64_t>(data_type_size(out_dt)),
                        output.layout(), output.quant_params());
    }
    TensorView& ov = out_row ? *out_row : output;

    TensorView minputs[3] = {x_row ? *x_row : x, w, TensorView{}};
    if (has_bias) { minputs[2] = inputs[2]; }
    const std::span<const TensorView> min(
        minputs, has_bias ? size_t{3} : size_t{2});

    MatMulAttributes ma;
    ma.transpose_b = true;
    ma.output_dtype = out_dt;   // read only on the int8 route
    ma.epilogue = attrs.epilogue;

    // add_to can ride beta only for fp operands with an identity epilogue: the
    // residual must not pass through the activation, and the int8 route zeroes
    // C instead of accumulating into it.
    const bool ep_identity = attrs.epilogue.type == EpilogueActivateType::None;
    const bool beta_residual = attrs.add_to && !quant && ep_identity;
    if (beta_residual) { ma.beta = 1.0f; }

    if (attrs.add_to && !beta_residual) {
        // Preserve the old output: the result must pass through the epilogue
        // before the residual is added, so MatMul writes a compact temporary.
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

    matmul_cpu(ma, ov, min, ctx, workspace);
    return Status::ok;
}

}  // namespace nnops::backend::cpu
