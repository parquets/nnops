/// @file layout_convert.cpp
/// @brief SIMD-accelerated layout conversion between NCHW and NCHWC8.
///
/// Unified implementation: 2D (NCHW↔NCHWC8) and 3D (NCDHW↔NCDHWC8)
/// share a single pack kernel and a single unpack kernel. The only
/// difference between 2D and 3D is num_spatial_rows (H vs D*H).
///
/// Uses an 8×8 SIMD transpose to vectorize both loads AND stores —
/// 8 vector loads + transpose + 8 vector stores processes 8 W-positions
/// at once (64 elements per iteration). Works for both f32 and f16 via
/// generic auto-deduction (v_load/v_store dispatch to the correct SIMD type).
/// Partial (trailing) C8 blocks are zero-padded on pack and truncated on unpack.
///
/// Parallelism: flattens N * ceil(C/8) * [D] * H into a single loop of
/// independent rows dispatched via ctx.cpu_parallel_for; falls back to
/// sequential otherwise. 2D: N*C8*H rows; 3D: N*C8*D*H rows.

#include "backend/cpu/layout_convert.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <algorithm>
#include <cstring>

namespace nnops {

using namespace nnops::simd;

namespace {

// ============================================================
// Per-element helpers — used for f16 and f32 partial/remainder.
// ============================================================

template <typename T>
inline void pack_one_w(const T* in_row, T* out_row,
                        int64_t w, int64_t c_base, int64_t ch_stride,
                        int64_t valid_lanes) {
    T tmp[8] = {};
    for (int64_t lane = 0; lane < valid_lanes; ++lane) {
        tmp[lane] = in_row[(c_base + lane) * ch_stride + w];
    }
    v_store(&out_row[w * 8], v_load(tmp));
}

template <typename T>
inline void unpack_one_w(const T* in_row, T* out_row,
                          int64_t w, int64_t c_base, int64_t ch_stride,
                          int64_t valid_lanes) {
    T tmp[8] = {};
    v_store(tmp, v_load(&in_row[w * 8]));
    for (int64_t lane = 0; lane < valid_lanes; ++lane) {
        out_row[(c_base + lane) * ch_stride + w] = tmp[lane];
    }
}

// ============================================================
// Unified pack kernel — works for both NCHW→NCHWC8 (2D)
// and NCDHW→NCDHWC8 (3D).
// ============================================================

template <typename T>
void pack_impl(const TensorView& src, TensorView& dst,
               const ComputeContext& ctx)
{
    const int64_t rank  = src.rank();
    const int64_t N     = src.shape(0);
    const int64_t C     = src.shape(1);
    const int64_t W     = src.shape(rank - 1);
    const int64_t C8    = (C + 7) / 8;

    // Product of all spatial dims except W (= H for 2D, D*H for 3D)
    int64_t num_spatial_rows = 1;
    for (int64_t d = 2; d < rank - 1; ++d) {
        num_spatial_rows *= src.shape(d);
    }

    const int64_t ch_stride      = src.stride_elems(1);
    const int64_t in_row_stride  = src.row_stride_elems();
    const int64_t out_row_stride = dst.row_stride_elems();
    const int64_t total_rows     = N * C8 * num_spatial_rows;

    const T* in_ptr  = src.ptr<T>();
    T*       out_ptr = dst.ptr<T>();

    auto process_row = [&](int64_t row) {
        // Flattened index: row = n*C8*S + c8*S + sr
        const int64_t n      = row / (C8 * num_spatial_rows);
        const int64_t remain = row % (C8 * num_spatial_rows);
        const int64_t c8     = remain / num_spatial_rows;
        const int64_t sr     = remain % num_spatial_rows;
        const int64_t c_base = c8 * 8;
        const int64_t valid_lanes = std::min<int64_t>(8, C - c_base);

        const T* in_row = in_ptr + n * C * ch_stride + sr * in_row_stride;
        T* out_row = out_ptr + (n * C8 * num_spatial_rows + c8 * num_spatial_rows + sr) * out_row_stride;

        auto vzero = v_set1(in_row, 0);
        int64_t w = 0;
        for (; w + 8 <= W; w += 8) {
            auto c0 = c_base + 0 < C ? v_load(&in_row[(c_base + 0) * ch_stride + w]) : vzero;
            auto c1 = c_base + 1 < C ? v_load(&in_row[(c_base + 1) * ch_stride + w]) : vzero;
            auto c2 = c_base + 2 < C ? v_load(&in_row[(c_base + 2) * ch_stride + w]) : vzero;
            auto c3 = c_base + 3 < C ? v_load(&in_row[(c_base + 3) * ch_stride + w]) : vzero;
            auto c4 = c_base + 4 < C ? v_load(&in_row[(c_base + 4) * ch_stride + w]) : vzero;
            auto c5 = c_base + 5 < C ? v_load(&in_row[(c_base + 5) * ch_stride + w]) : vzero;
            auto c6 = c_base + 6 < C ? v_load(&in_row[(c_base + 6) * ch_stride + w]) : vzero;
            auto c7 = c_base + 7 < C ? v_load(&in_row[(c_base + 7) * ch_stride + w]) : vzero;

            v_transpose_8x8(c0, c1, c2, c3, c4, c5, c6, c7);

            v_store(&out_row[(w + 0) * 8], c0);
            v_store(&out_row[(w + 1) * 8], c1);
            v_store(&out_row[(w + 2) * 8], c2);
            v_store(&out_row[(w + 3) * 8], c3);
            v_store(&out_row[(w + 4) * 8], c4);
            v_store(&out_row[(w + 5) * 8], c5);
            v_store(&out_row[(w + 6) * 8], c6);
            v_store(&out_row[(w + 7) * 8], c7);
        }
        for (; w < W; ++w) {
            pack_one_w<T>(in_row, out_row, w, c_base, ch_stride, valid_lanes);
        }
    };

    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, total_rows, process_row);
    } else {
        for (int64_t i = 0; i < total_rows; ++i) { process_row(i); }
    }
}

// ============================================================
// Unified unpack kernel — works for both NCHWC8→NCHW (2D)
// and NCDHWC8→NCDHW (3D).
// ============================================================

template <typename T>
void unpack_impl(const TensorView& src, TensorView& dst,
                 const ComputeContext& ctx)
{
    const int64_t rank  = src.rank();
    const int64_t N     = src.shape(0);
    const int64_t C     = src.shape(1);
    const int64_t W     = src.shape(rank - 1);
    const int64_t C8    = (C + 7) / 8;

    int64_t num_spatial_rows = 1;
    for (int64_t d = 2; d < rank - 1; ++d) {
        num_spatial_rows *= src.shape(d);
    }

    const int64_t ch_stride      = dst.stride_elems(1);
    const int64_t in_row_stride  = src.row_stride_elems();
    const int64_t out_row_stride = dst.row_stride_elems();
    const int64_t total_rows     = N * C8 * num_spatial_rows;

    const T* in_ptr  = src.ptr<T>();
    T*       out_ptr = dst.ptr<T>();

    auto process_row = [&](int64_t row) {
        const int64_t n      = row / (C8 * num_spatial_rows);
        const int64_t remain = row % (C8 * num_spatial_rows);
        const int64_t c8     = remain / num_spatial_rows;
        const int64_t sr     = remain % num_spatial_rows;
        const int64_t c_base = c8 * 8;
        const int64_t valid_lanes = std::min<int64_t>(8, C - c_base);

        const T* in_row = in_ptr + (n * C8 * num_spatial_rows + c8 * num_spatial_rows + sr) * in_row_stride;
        T* out_row = out_ptr + n * C * ch_stride + sr * out_row_stride;

        // Full SIMD transpose: 8 w at a time (works for f32, f16, partial C8)
        int64_t w = 0;
        for (; w + 8 <= W; w += 8) {
            auto r0 = v_load(&in_row[(w + 0) * 8]);
            auto r1 = v_load(&in_row[(w + 1) * 8]);
            auto r2 = v_load(&in_row[(w + 2) * 8]);
            auto r3 = v_load(&in_row[(w + 3) * 8]);
            auto r4 = v_load(&in_row[(w + 4) * 8]);
            auto r5 = v_load(&in_row[(w + 5) * 8]);
            auto r6 = v_load(&in_row[(w + 6) * 8]);
            auto r7 = v_load(&in_row[(w + 7) * 8]);

            v_transpose_8x8(r0, r1, r2, r3, r4, r5, r6, r7);

            // Store only valid channels (skip pad channels for partial C8)
            if (c_base + 0 < C) v_store(&out_row[(c_base + 0) * ch_stride + w], r0);
            if (c_base + 1 < C) v_store(&out_row[(c_base + 1) * ch_stride + w], r1);
            if (c_base + 2 < C) v_store(&out_row[(c_base + 2) * ch_stride + w], r2);
            if (c_base + 3 < C) v_store(&out_row[(c_base + 3) * ch_stride + w], r3);
            if (c_base + 4 < C) v_store(&out_row[(c_base + 4) * ch_stride + w], r4);
            if (c_base + 5 < C) v_store(&out_row[(c_base + 5) * ch_stride + w], r5);
            if (c_base + 6 < C) v_store(&out_row[(c_base + 6) * ch_stride + w], r6);
            if (c_base + 7 < C) v_store(&out_row[(c_base + 7) * ch_stride + w], r7);
        }
        // Remainder: per-w-element fallback
        for (; w < W; ++w) {
            unpack_one_w<T>(in_row, out_row, w, c_base,
                            ch_stride, valid_lanes);
        }
    };

    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, total_rows, process_row);
    } else {
        for (int64_t i = 0; i < total_rows; ++i) { process_row(i); }
    }
}

}  // anonymous namespace

// ============================================================
// Public entry points — dtype dispatch
// ============================================================

void pack_nchw_to_nchwc8(const TensorView& src, TensorView& dst,
                          const ComputeContext& ctx) {
    NNOPS_ASSERT(src.data_type() == dst.data_type());
    switch (src.data_type()) {
    case DataType::f32: pack_impl<float>(src, dst, ctx); return;
    case DataType::f16: pack_impl<half>(src, dst, ctx);  return;
    default: NNOPS_ASSERT(!"pack_nchw_to_nchwc8: unsupported data type");
    }
}

void unpack_nchwc8_to_nchw(const TensorView& src, TensorView& dst,
                            const ComputeContext& ctx) {
    NNOPS_ASSERT(src.data_type() == dst.data_type());
    switch (src.data_type()) {
    case DataType::f32: unpack_impl<float>(src, dst, ctx); return;
    case DataType::f16: unpack_impl<half>(src, dst, ctx);  return;
    default: NNOPS_ASSERT(!"unpack_nchwc8_to_nchw: unsupported data type");
    }
}

void pack_ncdhw_to_ncdhwc8(const TensorView& src, TensorView& dst,
                            const ComputeContext& ctx) {
    NNOPS_ASSERT(src.data_type() == dst.data_type());
    switch (src.data_type()) {
    case DataType::f32: pack_impl<float>(src, dst, ctx); return;
    case DataType::f16: pack_impl<half>(src, dst, ctx);  return;
    default: NNOPS_ASSERT(!"pack_ncdhw_to_ncdhwc8: unsupported data type");
    }
}

void unpack_ncdhwc8_to_ncdhw(const TensorView& src, TensorView& dst,
                              const ComputeContext& ctx) {
    NNOPS_ASSERT(src.data_type() == dst.data_type());
    switch (src.data_type()) {
    case DataType::f32: unpack_impl<float>(src, dst, ctx); return;
    case DataType::f16: unpack_impl<half>(src, dst, ctx);  return;
    default: NNOPS_ASSERT(!"unpack_ncdhwc8_to_ncdhw: unsupported data type");
    }
}

// ============================================================
// Storage size helper
// ============================================================

size_t nchwc8_storage_bytes(const TensorDesc& logical_desc, int64_t alignment) {
    const int64_t C = logical_desc.dims[1];
    const int64_t W = logical_desc.dims[static_cast<size_t>(logical_desc.rank - 1)];
    const int64_t C8 = (C + 7) / 8;
    const int64_t elem_size = static_cast<int64_t>(data_type_size(logical_desc.dtype));

    int64_t row_bytes = W * 8 * elem_size;
    int64_t aligned_row_bytes = ((row_bytes + alignment - 1) / alignment) * alignment;

    int64_t total_rows = logical_desc.dims[0] * C8;
    for (int64_t d = 2; d < logical_desc.rank - 1; ++d) {
        total_rows *= logical_desc.dims[static_cast<size_t>(d)];
    }

    return static_cast<size_t>(total_rows * aligned_row_bytes);
}

}  // namespace nnops
