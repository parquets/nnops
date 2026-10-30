/// @file embed.cpp
/// @brief CPU embedding lookup — a row copy for fp weights, per-row dequant for
/// int8/uint8.
///
/// output row i is weight[indices[i]], with s32 or s64 indices clamped to
/// [0, V-1]; the indices are the parallel axis, run through ctx.cpu.run.
///
/// An s8/u8 weight dequantizes per row: out[j] = (w[idx,j] - zp[idx]) * scale[idx],
/// via the arch kernels (`quant_kernel::dequantization<T>` in x86_64/quant.hpp /
/// aarch64/quant.hpp); the f16 output overload writes f16 in one pass, with no
/// f32 staging buffer. Quantization must be per-row — PerToken does this
/// directly, and PerChannel on the [V, dim] weight quantizes along axis 0 (the
/// vocab entry), so it is the same thing. Per-tensor and block granularities
/// are rejected.

#include "nnops/ops/embed.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/simd/cpu_features.hpp"

#if defined(NNOPS_ARCH_X86_64)
#include "x86_64/quant.hpp"
#elif defined(NNOPS_ARCH_AARCH64)
#include "aarch64/quant.hpp"
#else
#error "embed: unsupported architecture for quantization kernels"
#endif

#include <cstring>

namespace nnops::backend::cpu {

#if defined(NNOPS_ARCH_X86_64)
namespace quant_kernel = nnops::backend::cpu::x86_64;
#elif defined(NNOPS_ARCH_AARCH64)
namespace quant_kernel = nnops::backend::cpu::aarch64;
#endif

namespace {

template <typename T>
void embed_direct_impl(const EmbedAttributes& /*attrs*/,
                        TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx)
{
    const auto& weight  = inputs[0];
    const auto& indices = inputs[1];

    const int64_t V   = weight.shape(0);
    const int64_t dim = weight.shape(1);
    const int64_t num_indices = indices.numel();

    const T* weight_ptr = weight.ptr<T>();
    const int64_t w_row_stride = (weight.rank() == 2)
        ? dim
        : weight.row_stride_elems();

    T* out_ptr = output.ptr<T>();
    const int64_t out_row_stride = output.row_stride_elems();
    const bool is_i64 = (indices.data_type() == DataType::s64);
    const size_t row_bytes = static_cast<size_t>(dim) * sizeof(T);

    auto body = [&](int64_t n) {
        int64_t idx;
        if (is_i64) {
            idx = static_cast<const int64_t*>(indices.ptr<void>())[n];
        } else {
            idx = static_cast<const int32_t*>(indices.ptr<void>())[n];
        }

        if (idx < 0) {
            idx = 0;
        }
        if (idx >= V) {
            idx = V - 1;
        }

        std::memcpy(out_ptr + n * out_row_stride,
                    weight_ptr + idx * w_row_stride,
                    row_bytes);
    };

    ctx.cpu.run(0, num_indices, body);
}

template <typename T>  // T = weight storage type (int8_t / uint8_t)
void embed_int8_dequant_impl(const EmbedAttributes& /*attrs*/,
                              TensorView& output,
                              std::span<const TensorView> inputs,
                              const ComputeContext& ctx)
{
    const auto& weight  = inputs[0];
    const auto& indices = inputs[1];

    const auto& qp = weight.quant_params();
    const bool per_row = (qp.granularity == QuantGranularity::PerToken)
                      || (qp.granularity == QuantGranularity::PerChannel);
    NNOPS_ASSERT(per_row);
    const bool has_zp = (qp.zero_point_data != nullptr);

    const int64_t V = weight.shape(0);
    const int64_t dim = weight.shape(1);
    const int64_t num_indices = indices.numel();

    const T* weight_ptr = weight.ptr<T>();
    const int64_t w_row_elems = dim;  // int8/uint8: 1 byte per element

    const bool out_f16 = (output.data_type() == DataType::f16);
    const bool is_i64 = (indices.data_type() == DataType::s64);
    const int64_t out_row_stride = output.row_stride_elems();

    const int dim_i = static_cast<int>(dim);

    const auto resolve_idx = [&](int64_t n) -> int64_t {
        int64_t idx = is_i64
            ? static_cast<const int64_t*>(indices.ptr<void>())[n]
            : static_cast<const int32_t*>(indices.ptr<void>())[n];
        if (idx < 0) { idx = 0; }
        if (idx >= V) { idx = V - 1; }
        return idx;
    };

    const auto body = [&](int64_t n) {
        const int64_t idx = resolve_idx(n);
        const T* w_row = weight_ptr + idx * w_row_elems;

        const float s_val = qp.scale_data[idx];
        const float zp_val = has_zp ? static_cast<float>(qp.zero_point_data[idx]) : 0.0f;

        if (out_f16) {
            half* dst = output.ptr<half>() + n * out_row_stride;
            quant_kernel::dequantization<T>(1, dim_i, dst, dim_i,
                                            w_row, dim_i, &s_val, &zp_val);
        } else {
            float* dst = output.ptr<float>() + n * out_row_stride;
            quant_kernel::dequantization<T>(1, dim_i, dst, dim_i,
                                            w_row, dim_i, &s_val, &zp_val);
        }
    };

    ctx.cpu.run(0, num_indices, body);
}

}  // anonymous namespace

Status embed_cpu(const EmbedAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx,
                 void* /*workspace*/)
{
    const auto w_dtype = inputs[0].data_type();
    const auto o_dtype = output.data_type();

    if (w_dtype == DataType::f32) {
        embed_direct_impl<float>(attrs, output, inputs, ctx);
        return Status::ok;
    }
    if (w_dtype == DataType::f16) {
        embed_direct_impl<half>(attrs, output, inputs, ctx);
        return Status::ok;
    }

    NNOPS_ASSERT(o_dtype == DataType::f32 || o_dtype == DataType::f16);
    if (w_dtype == DataType::s8) {
        embed_int8_dequant_impl<int8_t>(attrs, output, inputs, ctx);
    } else if (w_dtype == DataType::u8) {
        embed_int8_dequant_impl<uint8_t>(attrs, output, inputs, ctx);
    } else {
        NNOPS_ASSERT(!"embed_cpu: unsupported weight dtype");
    }
    return Status::ok;
}

}  // namespace nnops::backend::cpu
