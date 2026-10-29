#pragma once
/// @file matmul_internal.h
/// @brief Internals shared by the MatMul translation units.
///
/// matmul.cpp owns the skeleton: the dtype routes, the tile/block kernels, the
/// plan-driven dispatch and the public entry point. The two alternate routes
/// that bypass the plan altogether are self-contained enough to live in their
/// own files — the same variant-per-file split Conv2D uses
/// (conv2d_implicit_gemm / conv2d_winograd):
///
///   - matmul_small.cpp         — the naive SIMD kernel below
///                                GEMM_FAST_PATH_THRESHOLD MACs, and the direct
///                                GEMV at M <= GEMV_M_MAX;
///   - matmul_qblock_fused.cpp  — the x86-only fused Q5_1/Q4_1 decode route
///                                (M <= 16).
///
/// They stay out of the driver because neither packs, blocks, splits or touches
/// the routes: each sweeps the product in one pass. What they do share with the
/// driver is here — the product's dims, the epilogue predicates, and the batch
/// walk that maps a flat batch index to per-operand element offsets.

#include "matmul_helper.h"                 // MatMulPlan geometry, Q5_1 block size
#include "nnops/core/compute_context.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/ops/matmul.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace nnops::backend::cpu {

/// Only the final k-block clamps; partial sums accumulate unclamped.
inline std::pair<float, float> kblock_clamp(const MatMulAttributes& attrs, bool last_k) noexcept {
    if (!last_k) {
        const float inf = std::numeric_limits<float>::infinity();
        return {-inf, inf};
    }
    return {attrs.epilogue.min_clip, attrs.epilogue.max_clip};
}

/// Non-clamp epilogues (Relu is folded into the MMA clamp) run in place.
inline bool has_inplace_epilogue(const MatMulAttributes& attrs) noexcept {
    return attrs.epilogue.type != EpilogueActivateType::None &&
           attrs.epilogue.type != EpilogueActivateType::Relu;
}

/// M, N and K of the product, honouring the transpose flags.
struct MatMulDims { int64_t M, N, K; };

inline MatMulDims matmul_dims(const MatMulAttributes& attrs,
                              const TensorView& a, const TensorView& b) noexcept {
    return { attrs.transpose_a ? a.shape(a.rank() - 1) : a.shape(a.rank() - 2),
             attrs.transpose_b ? b.shape(b.rank() - 2) : b.shape(b.rank() - 1),
             attrs.transpose_b ? b.shape(b.rank() - 1) : b.shape(b.rank() - 2) };
}

/// A block-packed B (q5_1/q4_1) for a batch element. BatchWalk's offsets and
/// the row pitch already count in blocks, so only the block size turns a
/// block offset into bytes.
inline const uint8_t* block_rhs_at(const uint8_t* base, int64_t block_off,
                                   DataType dt) noexcept {
    return base + block_off * static_cast<int64_t>(data_type_stride_unit(dt));
}

/// Broadcast batch dims (as matmul_ref) and map a flat index to per-operand
/// element offsets.
struct BatchWalk {
    const TensorView* a;
    const TensorView* b;
    int64_t ndim  = 0;
    int64_t total = 1;
    std::vector<int64_t> a_shape, b_shape, out_shape;

    BatchWalk(const TensorView& a_, const TensorView& b_) : a(&a_), b(&b_) {
        const int64_t a_dims = a->rank() - 2;
        const int64_t b_dims = b->rank() - 2;
        ndim = std::max(a_dims, b_dims);
        a_shape.assign(static_cast<size_t>(ndim), 1);
        b_shape.assign(static_cast<size_t>(ndim), 1);
        out_shape.assign(static_cast<size_t>(ndim), 1);

        for (int64_t i = 0; i < a_dims; ++i) a_shape[ndim - a_dims + i] = a->shape(i);
        for (int64_t i = 0; i < b_dims; ++i) b_shape[ndim - b_dims + i] = b->shape(i);

        for (int64_t i = 0; i < ndim; ++i) {
            const int64_t da = a_shape[i];
            const int64_t db = b_shape[i];
            if (da == db || db == 1) {
                out_shape[i] = da;
            } else if (da == 1) {
                out_shape[i] = db;
            } else {
                NNOPS_ASSERT(!"MatMul: incompatible batch dimensions for broadcast");
            }
            total *= out_shape[i];
        }
    }

    void offsets(int64_t bi, const TensorView& out,
                 int64_t& a_off, int64_t& b_off, int64_t& c_off) const {
        a_off = 0;
        b_off = 0;
        c_off = 0;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            const int64_t coord = bi % out_shape[d];
            bi /= out_shape[d];

            const int64_t a_dim = d - (ndim - (a->rank() - 2));
            if (a_dim >= 0) {
                const int64_t a_coord = (a_shape[d] == 1) ? 0 : coord;
                a_off += a_coord * a->stride_elems(a_dim);
            }

            const int64_t b_dim = d - (ndim - (b->rank() - 2));
            if (b_dim >= 0) {
                const int64_t b_coord = (b_shape[d] == 1) ? 0 : coord;
                b_off += b_coord * b->stride_elems(b_dim);
            }

            c_off += coord * out.stride_elems(d);
        }
    }
};

/// Dense f32/f16 small products: the naive kernel below GEMM_FAST_PATH_THRESHOLD
/// MACs, the direct GEMV at M <= GEMV_M_MAX. Returns false when the product is
/// not one of those (wrong dtypes, or large enough that the plan should run).
bool matmul_small_fp(const MatMulAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx);

#if defined(NNOPS_ARCH_X86_64)
/// Whole product through the fused Q5_1/Q4_1 kernels; false if not applicable.
bool matmul_qblock_fused(const MatMulAttributes& attrs,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         int M, int N, int K);
#endif

}  // namespace nnops::backend::cpu
