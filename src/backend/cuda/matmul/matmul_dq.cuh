#pragma once
/// @file matmul_dq.cuh
/// @brief The offline-quantized-activation GEMM: a floating A carrying
///        QuantParams against an s8 B, quantized to `act_quant_dtype` and run
///        on the int8 tensor-core engine, the int32 accumulate dequantized back
///        into A's dtype.
///
/// MatMul's `is_fp_quant_matmul` pair and Linear (its transposed twin) share
/// this. The CPU's route quantizes A into a dense s8 buffer too whenever it
/// cannot fuse the pack, so the device route starts from the same A_q and rides
/// the engine that is already bit-for-bit with the CPU's integer arithmetic:
///
///     out = (raw − zp_a·R_b[n] − (R_a[m] − K·zp_a)·zp_b[n]) · s_a[m] · s_b[n]
///
/// One product per batch element: the per-token scale table is cut at that
/// batch's first physical row, which is the row base the CPU's batch walk uses.
/// A broadcast A keeps row base 0, since its batch stride is zero.
///
/// A's parameters are the caller's (offline), and may carry a zero point. The
/// activation quantization itself uses the same reciprocal-multiply,
/// round-to-nearest-even, saturating store as the CPU arch kernel.
///
/// The fused epilogue is the engine's (clip + activation); a residual
/// (`Linear`'s add_to) has no fold in the integer engine and is refused here.

#include "matmul_i8.cuh"
#include "matmul_op.cuh"
#include "quant_rows.cuh"
#include "nnops/core/quant_params.hpp"
#include "nnops/core/tensor_view.hpp"

#include <limits>

namespace nnops::backend::cuda {
namespace detail {

/// One batched product's operands and epilogue. Pointers are the operator's;
/// `act_dt` is the integer storage A is quantized into.
struct DqCall {
    const TensorView* a = nullptr;
    const TensorView* b = nullptr;
    TensorView* c = nullptr;
    const TensorView* bias = nullptr;
    bool trans_a = false;
    bool trans_b = false;
    DataType act_dt = DataType::s8;
    EpilogueActivateType epilogue = EpilogueActivateType::None;
    float ep_alpha = 0.0f;
    float ep_beta = 1.0f;
    float min_clip = -std::numeric_limits<float>::infinity();
    float max_clip = std::numeric_limits<float>::infinity();
};

/// The same walk `gemm_batch_offset` does, on the host.
inline int64_t dq_batch_offset(const GemmParams& p, const int64_t* strides, int64_t idx) noexcept
{
    int64_t off = 0;
    for (int d = 0; d < p.batch_rank; ++d) {
        off += (idx % p.batch_size[d]) * strides[d];
        idx /= p.batch_size[d];
    }
    return off;
}

inline Status dense_gemm_dq_status(const DqCall& call, cudaStream_t stream)
{
    const TensorView& a = *call.a;
    const TensorView& b = *call.b;
    TensorView& c = *call.c;

    if (call.trans_a) {
        return Status::unsupported;
    }
    const DataType dt = a.data_type();
    const DataType odt = c.data_type();
    if ((dt != DataType::f32 && dt != DataType::f16) || odt != dt) {
        return Status::unsupported;
    }
    if (call.act_dt != DataType::s8 && call.act_dt != DataType::u8) {
        return Status::unsupported;
    }

    const int64_t M = a.shape(a.rank() - 2);
    const int64_t K = a.shape(a.rank() - 1);
    const int64_t N = call.trans_b ? b.shape(b.rank() - 2) : b.shape(b.rank() - 1);
    if (M <= 0 || N <= 0 || K <= 0) {
        return Status::ok;
    }

    const QuantParams& qa = a.quant_params();
    const int64_t rows = a.total_rows();
    if (rows <= 0) {
        return Status::ok;
    }
    const bool row_scale = qa.scale_data != nullptr;
    const bool row_zp = qa.zero_point_data != nullptr;
    if ((row_scale || row_zp) && qa.num_scales < rows) {
        return Status::invalid_argument;
    }

    // A_q, then the per-row tables the quantization kernel reads. They stay
    // device-resident for the engine's dequantize (`sc_a_device`).
    int64_t cursor = rows * K;
    int64_t sc_off = -1, zp_off = -1;
    if (row_scale) {
        cursor = (cursor + 15) & ~int64_t(15);
        sc_off = cursor;
        cursor += rows * 4;
    }
    if (row_zp) {
        cursor = (cursor + 15) & ~int64_t(15);
        zp_off = cursor;
        cursor += rows * 4;
    }

    const GemmWorkspace ws = gemm_workspace_alloc(cursor, stream);
    if (ws.ptr == nullptr) {
        return Status::out_of_memory;
    }
    char* base = static_cast<char*>(ws.ptr);
    int8_t* aq = reinterpret_cast<int8_t*>(base);

    const float* sc_dev = nullptr;
    const int32_t* zp_dev = nullptr;
    bool staged = true;
    if (row_scale) {
        sc_dev = reinterpret_cast<const float*>(base + sc_off);
        staged = cudaMemcpyAsync(base + sc_off, qa.scale_data, size_t(rows * 4),
                                 cudaMemcpyHostToDevice, stream) == cudaSuccess;
    }
    if (staged && row_zp) {
        zp_dev = reinterpret_cast<const int32_t*>(base + zp_off);
        staged = cudaMemcpyAsync(base + zp_off, qa.zero_point_data, size_t(rows * 4),
                                 cudaMemcpyHostToDevice, stream) == cudaSuccess;
    }

    Status st = Status::internal;
    if (staged) {
        st = qrows::quantize_fixed(dt, call.act_dt, a.ptr<void>(), rows, K, a.row_stride_elems(),
                                   aq, sc_dev, zp_dev, qa.scale, qa.zero_point, stream);
    }
    if (!is_ok(st)) {
        gemm_workspace_free(ws, stream);
        return st;
    }

    // The quantized view D is described by is dense, which is what makes the
    // batch offsets below dense too.
    const TensorView aq_view(a.shape_span(), call.act_dt, aq, TensorLayout::Dense, qa);
    GemmI8Params layout;
    if (!gemm_batch_layout(aq_view, b, c, nullptr, layout.g)) {
        gemm_workspace_free(ws, stream);
        return Status::unsupported;
    }
    const int64_t batch = gemm_batch_count(layout.g);
    const int64_t c_esz = static_cast<int64_t>(data_type_size(odt));

    for (int64_t bi = 0; bi < batch && is_ok(st); ++bi) {
        const int64_t a_off = dq_batch_offset(layout.g, layout.g.a_batch, bi);

        GemmI8Params q;
        q.g.a = aq + a_off;
        q.g.b = static_cast<const int8_t*>(b.ptr<void>()) +
                dq_batch_offset(layout.g, layout.g.b_batch, bi);
        q.g.c = static_cast<char*>(c.ptr<void>()) +
                dq_batch_offset(layout.g, layout.g.c_batch, bi) * c_esz;
        q.bias = call.bias != nullptr ? call.bias->ptr<void>() : nullptr;
        q.g.m = static_cast<int>(M);
        q.g.n = static_cast<int>(N);
        q.g.k = static_cast<int>(K);
        q.g.lda = K;
        q.g.ldb = b.row_stride_elems();
        q.g.ldc = c.row_stride_elems();
        q.g.epilogue = call.epilogue;
        q.g.ep_alpha = call.ep_alpha;
        q.g.ep_beta = call.ep_beta;
        q.g.min_clip = call.min_clip;
        q.g.max_clip = call.max_clip;

        const int64_t row_base = a_off / K;
        q.sc_a = row_scale ? sc_dev + row_base : nullptr;
        q.sc_a_scalar = qa.scale;
        q.sc_a_device = row_scale;
        q.zp_a = row_zp ? zp_dev + row_base : nullptr;
        q.zp_a_scalar = qa.zero_point;
        q.zp_a_device = row_zp;

        const QuantParams& qb = b.quant_params();
        q.zp_b = qb.zero_point_data;
        q.zp_b_scalar = qb.zero_point;
        q.sc_b = qb.scale_data;
        q.sc_b_scalar = qb.scale;

        q.out = odt == DataType::f32 ? I8OutKind::F32 : I8OutKind::F16;
        st = dense_gemm_i8_status(q, call.act_dt, /*trans_a=*/false, call.trans_b, stream);
    }

    gemm_workspace_free(ws, stream);
    return st;
}

} // namespace detail
} // namespace nnops::backend::cuda
