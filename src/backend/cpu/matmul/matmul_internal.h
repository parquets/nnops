#pragma once
/// @file matmul_internal.h
/// @brief Internals shared by the MatMul translation units.
///
/// The skeleton is one body for every route (matmul_tile_impl.h); the routes
/// themselves are split one file per arithmetic family, the same
/// variant-per-file split Conv2D uses:
///
///   - matmul.cpp      — the float routes (f32/f16, and bf16 on aarch64), the
///                       classification and the public entry point;
///   - i8matmul.cpp    — the quantized routes: s8|u8 × s8 into int32 or an
///                       fp dequantize, and the fp-activation twin.
///
/// The two alternate routes that bypass the plan altogether are self-contained
/// enough to live in their own files:
///
///   - matmul_small.cpp         — the naive SIMD kernel below
///                                GEMM_FAST_PATH_THRESHOLD MACs, and the direct
///                                GEMV at M <= GEMV_M_MAX;
///   - matmul_qblock_fused.cpp  — the fused Q4_1 decode route (x86_64 and
///                                aarch64; M <= 16 trans, 8 copy by default).
///
/// The AArch64 SME tiles are not a separate driver: they are a route variant
/// (SmeFpRoute in matmul.cpp, SmeI8Route in i8matmul.cpp) on the same
/// plan-driven skeleton, reached through the tile_pack_*_sme /
/// tile_mma_pack_*_sme layer.
///
/// None of the plan-free routes packs, blocks or splits: each sweeps the
/// product in one pass and never touches the routes. What they do share with
/// the driver is here — the product's dims, the epilogue predicates, the
/// prepacked-RHS view, and the batch walk that maps a flat batch index to
/// per-operand element offsets.

#include "matmul_helper.h" // MatMulPlan geometry, block sizes
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
inline std::pair<float, float> kblock_clamp(const MatMulAttributes& attrs, bool last_k) noexcept
{
    if (!last_k) {
        const float inf = std::numeric_limits<float>::infinity();
        return {-inf, inf};
    }
    return {attrs.epilogue.min_clip, attrs.epilogue.max_clip};
}

/// Non-clamp epilogues (Relu is folded into the MMA clamp) run in place.
inline bool has_inplace_epilogue(const MatMulAttributes& attrs) noexcept
{
    return epilogue_is_active(attrs.epilogue) && attrs.epilogue.type != EpilogueActivateType::Relu;
}

/// M, N and K of the product, honouring the transpose flags.
struct MatMulDims {
    int64_t M, N, K;
};

inline MatMulDims matmul_dims(const MatMulAttributes& attrs, const TensorView& a,
                              const TensorView& b) noexcept
{
    // A packed RHS (see matmul_helper.h) leads with N and carries no K of its
    // own — the blob is addressed from K, not shaped by it — so K comes from A.
    const int64_t k_a = attrs.transpose_a ? a.shape(a.rank() - 2) : a.shape(a.rank() - 1);
    if (is_packed_matmul_b(b.layout())) {
        return {attrs.transpose_a ? a.shape(a.rank() - 1) : a.shape(a.rank() - 2), b.shape(0), k_a};
    }
    return {attrs.transpose_a ? a.shape(a.rank() - 1) : a.shape(a.rank() - 2),
            attrs.transpose_b ? b.shape(b.rank() - 2) : b.shape(b.rank() - 1),
            attrs.transpose_b ? b.shape(b.rank() - 1) : b.shape(b.rank() - 2)};
}

/// A block-packed B (q4_1) at a per-batch offset already counted in blocks.
inline const uint8_t* block_rhs_at(const uint8_t* base, int64_t block_off, DataType dt) noexcept
{
    return base + block_off * static_cast<int64_t>(data_type_stride_unit(dt));
}

/// Broadcast the batch dims and map a flat index to per-operand element offsets.
struct BatchWalk {
    const TensorView* a;
    const TensorView* b;
    int64_t ndim = 0;
    int64_t total = 1;
    std::vector<int64_t> a_shape, b_shape, out_shape;

    BatchWalk(const TensorView& a_, const TensorView& b_) : a(&a_), b(&b_)
    {
        const int64_t a_dims = a->rank() - 2;
        const int64_t b_dims = b->rank() - 2;
        ndim = std::max(a_dims, b_dims);
        a_shape.assign(static_cast<size_t>(ndim), 1);
        b_shape.assign(static_cast<size_t>(ndim), 1);
        out_shape.assign(static_cast<size_t>(ndim), 1);

        for (int64_t i = 0; i < a_dims; ++i)
            a_shape[ndim - a_dims + i] = a->shape(i);
        for (int64_t i = 0; i < b_dims; ++i)
            b_shape[ndim - b_dims + i] = b->shape(i);

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

    void offsets(int64_t bi, const TensorView& out, int64_t& a_off, int64_t& b_off,
                 int64_t& c_off) const
    {
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

/// The caller's prepacked RHS (MatMul/Linear ::prepackWeights) as one object.
/// A float blob is a single whole-K panel sequence; an int8 blob is the same
/// sequence repeated per k segment, because its per-panel correction block is
/// read at the call's kc. Routes turn a (k-block, panel) pair into a pointer and
/// a kernel pitch — the two routes disagree on both.
struct PackedBView {
    const void* base = nullptr;
    int64_t ldd = 0; // panel pitch: elements for fp, bytes for int8
    int64_t np = 0;  // panels per segment (int8 only)
    int64_t seg = 0; // k segment length (int8 only; 0 = the whole K)

    bool ok() const noexcept { return base != nullptr; }
};

// ---- int8 zero-point predicates (shared by the skeleton and its routes) ----

/// True when some zero point is non-zero; @p n is the extent when @p num_scales
/// is 0.
inline bool any_zp_nonzero(const int32_t* zp, int32_t scalar, int64_t num_scales,
                           int64_t n) noexcept
{
    if (zp == nullptr) {
        return scalar != 0;
    }
    const int64_t count = num_scales > 0 ? num_scales : n;
    for (int64_t i = 0; i < count; ++i) {
        if (zp[i] != 0) {
            return true;
        }
    }
    return false;
}

/// True when some zero point differs from @p v, i.e. whether r_b is live.
inline bool any_zp_differs(const int32_t* zp, int32_t scalar, int64_t num_scales, int64_t n,
                           int32_t v) noexcept
{
    if (zp == nullptr) {
        return scalar != v;
    }
    const int64_t count = num_scales > 0 ? num_scales : n;
    for (int64_t i = 0; i < count; ++i) {
        if (zp[i] != v) {
            return true;
        }
    }
    return false;
}

/// Raw int8 reductions for the output pass: r_a[m] = Σ_k A[m,k], r_b[n] =
/// Σ_k B[k,n] (the transposes mirror them). Disjoint slabs, so no atomics; the
/// kernels are in simd_int8_reduce.hpp. @p a_u8 sums A unsigned; a null
/// @p r_a / @p r_b skips that sweep. Defined in i8matmul.cpp.
void compute_int8_reductions(const int8_t* a_ptr, bool a_u8, int64_t lda, const int8_t* b_ptr,
                             int64_t ldb, const MatMulAttributes& attrs, int64_t M, int64_t N,
                             int64_t K, int32_t* r_a, int32_t* r_b, const ComputeContext& ctx);

/// Dense f32/f16 small products: the naive kernel below GEMM_FAST_PATH_THRESHOLD
/// MACs, the direct GEMV at M <= GEMV_M_MAX. Returns false when the product is
/// not one of those (wrong dtypes, or large enough that the plan should run).
bool matmul_small_fp(const MatMulAttributes& attrs, TensorView& output,
                     std::span<const TensorView> inputs, const ComputeContext& ctx);

#if defined(NNOPS_ARCH_X86_64) || defined(NNOPS_ARCH_AARCH64)
/// Whole product through the fused Q4_1 kernels; false if not applicable.
bool matmul_qblock_fused(const MatMulAttributes& attrs, TensorView& output,
                         std::span<const TensorView> inputs, const ComputeContext& ctx, int M,
                         int N, int K);
#endif

/// The quantized family (i8matmul.cpp): s8 weights, with an activation that is
/// already integer or fp carrying the quantization to apply.
Status matmul_dispatch_quant(const MatMulAttributes& attrs, TensorView& output,
                             std::span<const TensorView> inputs, const ComputeContext& ctx);

/// fp activations against s8 weights: quantize A (in the pack, or into A_q),
/// run the int8 GEMM, dequantize into A's dtype.
Status matmul_dq_from_fp(const MatMulAttributes& attrs, TensorView& output,
                         std::span<const TensorView> inputs, const ComputeContext& ctx);

} // namespace nnops::backend::cpu
