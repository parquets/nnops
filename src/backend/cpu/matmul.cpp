/// @file matmul.cpp
/// @brief Tiled matrix multiplication kernel — pack + MMA dispatch.
///
/// f32, f16 and int8 share one skeleton (FpRoute / I8Route): B is always packed
/// into pooled scratch; A is packed too when the route needs it (always int8)
/// or read directly. Tiling, the split direction and the workspace size come
/// from get_matmul_plan(): NKM on an N-split, MKN on an M-split.
///
/// f32/f16 below GEMM_FAST_PATH_THRESHOLD MACs, and every product at M <= 4,
/// use the arch's naive SIMD kernel / direct GEMV instead of the plan; a Q5_1 B
/// at small M decodes straight into its dot product. Those routes sweep the
/// product in one pass and share nothing with the skeleton, so they live in
/// matmul_small.cpp and matmul_q5_fused.cpp (see matmul_internal.h).
///
/// int8 accumulates raw int32 and recovers the product in one epilogue pass.

#include "matmul.h"
#include "matmul_internal.h"      // dims / batch walk shared with the small routes
#include "matmul_helper.h"
#include "simd_kernel/simd_epilogue.hpp"
#include "simd_kernel/simd_int8_reduce.hpp"
#include "simd_kernel/simd_quant.hpp"
#include "common/memory_pool.hpp"           // internal scratch-memory pool
#include "nnops/detail/half.hpp"
#include "nnops/detail/assert.hpp"

#include <algorithm>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

// Forward-declare reference kernels for the fallback paths.
namespace nnops::backend::cpu::reference {
extern void matmul_ref(const MatMulAttributes& attrs,
                       TensorView& output,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace);
extern void matmul_int8_ref(const MatMulAttributes& attrs,
                            TensorView& output,
                            std::span<const TensorView> inputs,
                            const ComputeContext& ctx,
                            void* workspace);
extern void matmul_dq_ref(const MatMulAttributes& attrs,
                          TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx,
                          void* workspace);
}

namespace nnops::backend::cpu {

namespace {

// =========================================================================
//  Dtype routes — the only place a path names its element types
// =========================================================================
//
// Per k-block pack B, per M panel pack A (or read it raw) and call one MMA
// kernel that sweeps the N side. Everything dtype-specific is named here.

/// The fp route names no output pass — its epilogue is in place — but the tiled
/// kernel's parameter type must still exist.
struct NoEpilogueParams {};

template <class T>
struct FpRoute {
    using Elem  = T;
    using AElem = T;   // A's element type; differs from Elem only on the fused dq route
    using Acc   = T;
    using Clamp = float;

    static constexpr bool is_fp            = true;
    static constexpr bool is_dq            = false;
    static constexpr bool fused_a          = false;
    static constexpr bool has_direct_a     = true;   // A may be read through lda
    static constexpr bool hoists_b_m_split = false;  // B packs inside each block

    /// No int8-style output pass on the fp route (its epilogue is in-place).
    using EpilogueParams = NoEpilogueParams;

    static constexpr const MatmulGeom& geom() {
        return std::is_same_v<T, float> ? GEOM_F32 : GEOM_F16;
    }

    /// B for one batch element: a dense view or a Q5_1 block stream.
    struct B {
        const void* ptr;
        int ldb;
        bool q51;
        int64_t row_blocks;
    };

    static B b_source(const TensorView& b, int64_t b_off, int ldb, bool q51) {
        if (q51) {
            return {q5_rhs_at(static_cast<const uint8_t*>(b.ptr<void>()), b_off),
                    ldb, true, static_cast<int64_t>(ldb)};
        }
        return {static_cast<const void*>(b.ptr<T>() + b_off), ldb, false, 0};
    }

    static int ldd_b(int kc) { return static_cast<int>(geom().ldd_b(kc)); }
    static int ldd_a(int kc) { return static_cast<int>(geom().ldd_a(kc)); }

    static std::pair<Clamp, Clamp> clamp(const MatMulAttributes& attrs, bool last_k) {
        return kblock_clamp(attrs, last_k);
    }

    static void pack_a(bool trans, int mc, int kc, Elem* dst, int ldd,
                       const AElem* src, int lds, bool /*a_biased*/,
                       const EpilogueParams* /*ip*/, int /*m*/) {
        tile_pack_lhs(trans, mc, kc, dst, ldd, src, lds, 1.0f);
    }

    static void pack_b(const MatMulAttributes& attrs, const B& b, int k, int kc,
                       int n_start, int nc, Elem* dst, int ldd, bool /*a_biased*/) {
        // The packer takes the tile origin in block units, not elements.
        if (b.q51) {
            tile_pack_rhs_q5_1(attrs.transpose_b, n_start, nc, k, kc, dst, ldd,
                               b.ptr, b.row_blocks);
            return;
        }
        const Elem* base = static_cast<const Elem*>(b.ptr);
        const Elem* src = attrs.transpose_b
            ? base + n_start * b.ldb + k     // N×K: row n, col k
            : base + k * b.ldb + n_start;    // K×N: row k, col n
        tile_pack_rhs(attrs.transpose_b, nc, kc, dst, ldd, src, b.ldb, 1.0f);
    }

    static void mma_pack(int mc, int nc, int kc, Acc* c, int ldc,
                         const Elem* pa, const Elem* pb, Clamp lo, Clamp hi,
                         bool zero_mode) {
        tile_mma_pack(mc, nc, kc, c, ldc, pa, pb, lo, hi, zero_mode);
    }

    static void mma_direct(int mc, int nc, int kc, Acc* c, int ldc,
                           const Elem* a, int lda, const Elem* pb,
                           Clamp lo, Clamp hi, bool zero_mode) {
        tile_mma_direct(mc, nc, kc, c, ldc, a, lda, pb, lo, hi, zero_mode);
    }

    /// C row of a batch element plus the two fp-only in-place passes.
    static Acc* c_base(TensorView& output, int64_t c_off, void* /*accum*/) {
        return output.ptr<T>() + c_off;
    }
    static void scale(Acc* c, int ldc, float beta, int mc, int nc) {
        tile_scale(c, ldc, beta, mc, nc);
    }
    static void epilogue(const MatMulAttributes& attrs, Acc* c, int ldc, int mc, int nc) {
        epilogue_inplace(mc, nc, c, ldc, static_cast<const Elem*>(nullptr),
                         attrs.epilogue);
    }
};

/// Bytes of the compact M×N int32 accumulator an int8 output pass reads (the
/// plan's scratch is followed by exactly this much).
inline size_t accum_bytes(const TensorView& output) noexcept {
    const auto od = output.desc();
    return static_cast<size_t>(od.dims[static_cast<size_t>(od.rank - 2)])
         * static_cast<size_t>(od.dims[static_cast<size_t>(od.rank - 1)])
         * sizeof(int32_t);
}

/// int8 (s8|u8 A × s8 B): no raw A, raw int32 C, epilogue once after all
/// blocks.
struct I8Route {
    using Elem  = int8_t;
    using AElem = int8_t;
    using Acc   = int32_t;
    using Clamp = int32_t;

    static constexpr bool is_fp            = false;
    static constexpr bool is_dq            = false;
    static constexpr bool fused_a          = false;
    static constexpr bool has_direct_a     = false;
    static constexpr bool hoists_b_m_split = true;  // one B pack per k-block

    using EpilogueParams = Int8EpilogueParams;

    static constexpr const MatmulGeom& geom() { return GEOM_I8; }

    struct B { const int8_t* ptr; int ldb; };

    static B b_source(const TensorView& b, int64_t b_off, int ldb, bool /*q51*/) {
        return {b.ptr<int8_t>() + b_off, ldb};
    }

    static int ldd_b(int kc) { return static_cast<int>(geom().ldd_b(kc)); }
    static int ldd_a(int kc) { return static_cast<int>(geom().ldd_a(kc)); }

    static std::pair<Clamp, Clamp> clamp(const MatMulAttributes&, bool) {
        return {INT32_MIN, INT32_MAX};
    }

    static void pack_a(bool trans, int mc, int kc, Elem* dst, int ldd,
                       const AElem* src, int lds, bool a_biased,
                       const EpilogueParams* /*ip*/, int /*m*/) {
        tile_pack_lhs_i8(trans, mc, kc, dst, ldd, src, lds, a_biased);
    }

    /// dtype the int8 kernels see on the A side. Only the fused dq route
    /// differs: it reads fp and quantises to `act_quant_dtype`.
    static DataType a_eff_type(const MatMulAttributes& /*attrs*/, const TensorView& a) {
        return a.data_type();
    }

    static void pack_b(const MatMulAttributes& attrs, const B& b, int k, int kc,
                       int n_start, int nc, Elem* dst, int ldd, bool a_biased) {
        const Elem* src = attrs.transpose_b
            ? b.ptr + n_start * b.ldb + k     // N×K: row n, col k
            : b.ptr + k * b.ldb + n_start;    // K×N: row k, col n
        tile_pack_rhs_i8(attrs.transpose_b, nc, kc, dst, ldd, src, b.ldb, a_biased);
    }

    static void mma_pack(int mc, int nc, int kc, Acc* c, int ldc,
                         const Elem* pa, const Elem* pb, Clamp lo, Clamp hi,
                         bool /*zero_mode*/) {
        tile_mma_pack_i8(mc, nc, kc, c, ldc, pa, pb, lo, hi);
    }

    /// The s32 output, or the M×N temporary an s8 output requantizes from.
    static Acc* c_base(TensorView& output, int64_t c_off, void* accum) {
        return output.data_type() == DataType::s8 ? static_cast<Acc*>(accum)
                                                  : output.ptr<Acc>() + c_off;
    }

    static void zero_c(Acc* c, int ldc, int M, int N) {
        for (int m = 0; m < M; ++m) {
            std::fill(c + m * ldc, c + m * ldc + N, 0);
        }
    }

    /// s8 output only: the accumulator appended after the plan's scratch.
    static size_t extra_workspace(const TensorView& output) {
        return output.data_type() == DataType::s8 ? accum_bytes(output) : 0;
    }

    static void epilogue_tile(const EpilogueParams& p, Acc* c, int ldc,
                              int64_t m0, int n0, int64_t mc, int64_t nc) {
        int8_epilogue_tile(p, c, ldc, m0, n0, mc, nc);
    }
};

/// Offline-quantized fp activations × s8 weights.
///
/// The GEMM is exactly the int8 one: A_q was materialized upstream (same shape,
/// carrying A's QuantParams) and handed in as inputs[0], so the plan, the pack
/// bias and the reductions are all already right for it. What changes is the
/// output pass — the compact int32 accumulator is dequantized into fp.
template <class TOut, class AIn = int8_t>
struct DqRoute : I8Route {
    static constexpr bool is_dq = true;

    /// A arrives as fp and is quantised inside the pack, so no A_q is ever
    /// materialised (MKN, no compensation; see matmul_dq).
    static constexpr bool fused_a = !std::is_same_v<AIn, int8_t>;

    using OutElem = TOut;
    using AElem   = AIn;
    using EpilogueParams = DqEpilogueParams<TOut>;

    static DataType a_eff_type(const MatMulAttributes& attrs, const TensorView& a) {
        if constexpr (fused_a) { return attrs.act_quant_dtype; }
        else { return a.data_type(); }
    }

    /// Fused: quantise each panel row from fp with its offline parameters
    /// (already indexed from the batch's first row) and then pack.
    static void pack_a(bool trans, int mc, int kc, int8_t* dst, int ldd,
                       const AElem* src, int lds, bool a_biased,
                       const EpilogueParams* ip, int m) {
        if constexpr (fused_a) {
            NNOPS_ASSERT(!trans);
            tile_pack_quant_lhs_i8(
                mc, kc, dst, ldd, src, lds,
                ip->s_a ? ip->s_a + m : nullptr,
                ip->zp_a ? ip->zp_a + m : nullptr,
                ip->s_a_scalar, ip->zp_a_scalar, ip->act_u8, a_biased);
        } else {
            tile_pack_lhs_i8(trans, mc, kc, dst, ldd, src, lds, a_biased);
        }
    }

    /// Always the compact accumulator: the destination is fp and is written by
    /// the output pass, never accumulated into.
    static Acc* c_base(TensorView& /*output*/, int64_t /*c_off*/, void* accum) {
        return static_cast<Acc*>(accum);
    }

    /// The dequant pass is unconditional — it is what produces the output.
    static constexpr bool always_epilogue = true;

    /// Always the compact accumulator, whatever the output dtype.
    static size_t extra_workspace(const TensorView& output) {
        return accum_bytes(output);
    }

    static void epilogue_tile(const EpilogueParams& p, Acc* c, int ldc,
                              int64_t m0, int n0, int64_t mc, int64_t nc) {
        dq_epilogue_tile<TOut>(p, c, ldc, m0, n0, mc, nc);
    }
};

// =========================================================================
//  Tile-block kernels
// =========================================================================

/// The row/column range a parallel block owns.
struct BlockRange { int start; int count; };

inline BlockRange m_range(const MatMulPlan& plan, int64_t blk, int64_t num_blocks,
                          int M) noexcept {
    if (plan.split_n) { return {0, M}; }
    const int start = static_cast<int>(blk * M / num_blocks);
    return {start, static_cast<int>((blk + 1) * M / num_blocks) - start};
}

inline BlockRange n_range(const MatMulPlan& plan, int64_t blk, int N) noexcept {
    if (!plan.split_n) { return {0, N}; }
    const int start = static_cast<int>(blk) * static_cast<int>(plan.nc);
    return {start, std::min(static_cast<int>(plan.nc), N - start)};
}

/// Scratch slot: per worker thread when slots < blocks, else per block.
inline int64_t block_slot(int64_t num_slots, int64_t num_blocks, int64_t blk,
                          const ComputeContext& ctx) noexcept {
    return (num_slots < num_blocks)
        ? static_cast<int64_t>(ctx.cpu.current_thread_id())
        : blk;
}

/// Packed-A slot of a block, or null when A is read raw. A has its own slot
/// count (`num_slots_a`), not the B one.
template <class Elem>
inline Elem* pack_a_slot(const MatMulPlan& plan, int64_t num_blocks, int64_t blk,
                         const ComputeContext& ctx, Elem* base) noexcept {
    if (!plan.pack_a) { return nullptr; }
    return base + block_slot(plan.num_slots_a, num_blocks, blk, ctx)
                * plan.np_a * plan.ldd_a;
}

/// Accumulate one k-block into an m×n tile of C.
template <class R>
void matmul_tile(const MatMulAttributes& attrs,
                 typename R::Acc* c_ptr, int ldc,
                 const typename R::AElem* a_ptr, int lda,
                 int m_start, int m_count, int mc,
                 int n_start, int n_count,
                 int k, int actual_kc, bool last_k,
                 bool pack_a, const typename R::Elem* packed_b,
                 typename R::Elem* pack_a_slice, bool a_biased,
                 const typename R::EpilogueParams* ip)
{
    using Acc = typename R::Acc;

    const int ldd_a = R::ldd_a(actual_kc);
    const bool zero_mode = (k == 0 && attrs.beta == 0.0f);  // C is overwritten
    const auto [cmin, cmax] = R::clamp(attrs, last_k);
    // fp-only; compile out of the int8 instantiation.
    const bool scale_c  = R::is_fp && k == 0 && attrs.beta != 1.0f && attrs.beta != 0.0f;
    const bool epilogue = R::is_fp && last_k && has_inplace_epilogue(attrs);
    // int8 does its output pass in the same place fp does: on the tile the
    // last k-block just produced, while it is still in cache.
    const bool int8_ep = !R::is_fp && last_k && (ip != nullptr);

    for (int m = m_start; m < m_start + m_count; m += mc) {
        const int actual_mc = std::min(mc, m_start + m_count - m);
        Acc* c_tile = c_ptr + m * ldc + n_start;

        const typename R::AElem* a_src = attrs.transpose_a
            ? a_ptr + k * lda + m    // A phys is K×M, row k, col m
            : a_ptr + m * lda + k;   // A phys is M×K, row m, col k

        if constexpr (R::is_fp) {
            if (scale_c) { R::scale(c_tile, ldc, attrs.beta, actual_mc, n_count); }
        }
        if (pack_a) {
            R::pack_a(attrs.transpose_a, actual_mc, actual_kc,
                      pack_a_slice, ldd_a, a_src, lda, a_biased, ip, m);
            R::mma_pack(actual_mc, n_count, actual_kc,
                        c_tile, ldc, pack_a_slice, packed_b, cmin, cmax, zero_mode);
        } else if constexpr (R::has_direct_a) {
            R::mma_direct(actual_mc, n_count, actual_kc,
                          c_tile, ldc, a_src, lda, packed_b, cmin, cmax, zero_mode);
        }
        if constexpr (R::is_fp) {
            if (epilogue) { R::epilogue(attrs, c_tile, ldc, actual_mc, n_count); }
        } else {
            // From column 0 of C, not from the tile: the kernels index the row
            // by the absolute output column (they read r_b[n] the same way).
            if (int8_ep) {
                R::epilogue_tile(*ip, c_ptr + m * ldc, ldc, m, n_start,
                                 actual_mc, n_count);
            }
        }
    }
}

/// One block over all K: pack B per k-block, accumulate every m-panel.
template <class R>
void matmul_block(const MatMulAttributes& attrs,
                  typename R::Acc* c_ptr, int ldc,
                  const typename R::AElem* a_ptr, int lda,
                  const typename R::B& b_src,
                  int K, int kc, int mc,
                  int m_start, int m_count, int n_start, int n_count,
                  typename R::Elem* pack_b_slice,
                  typename R::Elem* pack_a_slice,
                  bool pack_a, bool a_biased,
                  const typename R::EpilogueParams* ip)
{
    for (int k = 0; k < K; k += kc) {
        const int actual_kc = std::min(kc, K - k);
        const bool last_k = (k + kc >= K);
        const int ldd_b = R::ldd_b(actual_kc);
        R::pack_b(attrs, b_src, k, actual_kc, n_start, n_count,
                  pack_b_slice, ldd_b, a_biased);
        matmul_tile<R>(attrs, c_ptr, ldc, a_ptr, lda,
                       m_start, m_count, mc, n_start, n_count,
                       k, actual_kc, last_k, pack_a, pack_b_slice, pack_a_slice,
                       a_biased, ip);
    }
}

// =========================================================================
//  int8 — reduction and epilogue passes around the raw accumulate
// =========================================================================

/// Raw int8 reductions for the epilogue: r_a[m] = Σ_k A[m,k], r_b[n] = Σ_k B[k,n]
/// (A[k,m] / the mirror image when transposed). Output slabs are disjoint, so
/// no atomics; int32 accumulation matches the reference bit for bit. The
/// kernels are in simd_int8_reduce.hpp — this only owns the slab split.
///
/// B is always int8; @p a_u8 selects A's element type — a u8 A must be summed
/// unsigned, since the compensation is over the true value.
void compute_int8_reductions(const int8_t* a_ptr, bool a_u8, int64_t lda,
                             const int8_t* b_ptr, int64_t ldb,
                             const MatMulAttributes& attrs,
                             int64_t M, int64_t N, int64_t K,
                             int32_t* r_a, int32_t* r_b,
                             const ComputeContext& ctx) {
    // `rows` is the reduction direction: r[i] = Σ_j p[i*ld+j] when the K axis
    // is contiguous (the transposed layout), else a column slab.
    const auto reduce = [&](const void* p, bool u8, int64_t ld, int64_t n_out,
                            bool rows, int32_t* r) {
        const int64_t tasks = std::min<int64_t>(
            n_out, std::max<int64_t>(ctx.cpu.thread_count(), 1));
        ctx.cpu.run(0, tasks, [&](int64_t t) {
            const int64_t i0 = t * n_out / tasks;
            const int64_t i1 = (t + 1) * n_out / tasks;
            if (rows) {
                for (int64_t i = i0; i < i1; ++i) {
                    r[i] = u8
                        ? kernel::int8_row_sum(
                              static_cast<const uint8_t*>(p) + i * ld, K)
                        : kernel::int8_row_sum(
                              static_cast<const int8_t*>(p) + i * ld, K);
                }
            } else {
                for (int64_t i = i0; i < i1; ++i) { r[i] = 0; }
                if (u8) {
                    kernel::int8_col_sum_slab(static_cast<const uint8_t*>(p), ld,
                                              i0, i1, K, r);
                } else {
                    kernel::int8_col_sum_slab(static_cast<const int8_t*>(p), ld,
                                              i0, i1, K, r);
                }
            }
        });
    };

    reduce(a_ptr, a_u8, lda, M, /*rows=*/!attrs.transpose_a, r_a);
    reduce(b_ptr, /*u8=*/false, ldb, N, /*rows=*/attrs.transpose_b, r_b);
}

/// True when some zero point is non-zero. A per-row / per-column buffer wins
/// over the scalar; @p n is the tensor extent, used when the caller left
/// @p num_scales at zero.
inline bool any_zp_nonzero(const int32_t* zp, int32_t scalar, int64_t num_scales,
                           int64_t n) noexcept {
    if (zp == nullptr) { return scalar != 0; }
    const int64_t count = num_scales > 0 ? num_scales : n;
    for (int64_t i = 0; i < count; ++i) {
        if (zp[i] != 0) { return true; }
    }
    return false;
}

/// True when some zero point differs from @p v — i.e. whether a_bias_offset +
/// zp_a can be non-zero for a row, which is what makes the r_b term live.
inline bool any_zp_differs(const int32_t* zp, int32_t scalar, int64_t num_scales,
                           int64_t n, int32_t v) noexcept {
    if (zp == nullptr) { return scalar != v; }
    const int64_t count = num_scales > 0 ? num_scales : n;
    for (int64_t i = 0; i < count; ++i) {
        if (zp[i] != v) { return true; }
    }
    return false;
}

/// Whether the dq compensation is anything other than the identity, i.e.
/// whether the reduction sweep has to run. True as soon as a row's
/// a_bias_offset + zp_a or any column's zp_b is non-zero.
inline bool dq_needs_compensation(const TensorView& a, const TensorView& b,
                                  int32_t a_bias_offset, int64_t M, int64_t N) noexcept {
    const QuantParams& qa = a.quant_params();
    const QuantParams& qb = b.quant_params();
    return any_zp_differs(qa.zero_point_data, qa.zero_point, qa.num_scales, M,
                          -a_bias_offset) ||
           any_zp_nonzero(qb.zero_point_data, qb.zero_point, qb.num_scales, N);
}

// =========================================================================
//  Plan-driven dispatch — pack + block-granularity parallelism
// =========================================================================
//
// One skeleton for every route: per batch element, split the output on its
// larger dimension and run the blocks on the pool. The dtype islands are
// marked inline; B hoisting on an M-split is a route property.

template <class R>
void matmul_dispatch_tiled(const MatMulAttributes& attrs,
                           TensorView& output,
                           std::span<const TensorView> inputs,
                           const ComputeContext& ctx,
                           const MatMulPlan& plan, void* workspace)
{
    using Elem = typename R::Elem;
    using Acc  = typename R::Acc;

    const auto& a = inputs[0];
    const auto& b = inputs[1];
    const auto [M, N, K] = matmul_dims(attrs, a, b);

    const bool b_q51 = R::is_fp && (b.data_type() == DataType::q5_1);

    if constexpr (R::is_fp) {
        // No scratch to pack into: the reference is the only route left.
        if (workspace == nullptr) {
            reference::matmul_ref(attrs, output, inputs, ctx, nullptr);
            return;
        }
    }

    const int lda = static_cast<int>(a.row_stride_elems());
    const int ldb = static_cast<int>(b.row_stride_elems());
    const int ldc_out = static_cast<int>(output.row_stride_elems());

    // int8 accumulates raw int32: the s32 output in place, or a compact M×N
    // temporary the output pass rewrites — s8 requant, or fp dequant.
    const bool out_s8 = !R::is_fp && output.data_type() == DataType::s8;
    const bool compact_acc = out_s8 || (!R::is_fp && R::is_dq);
    const int ldc = compact_acc ? static_cast<int>(N) : ldc_out;

    const typename R::AElem* a_base = a.ptr<typename R::AElem>();
    // A's hardware shift (see int8_a_pack_bias): the packers take it, the
    // epilogue takes what the kernels did not fold back out themselves. The
    // fused dq route reads fp and quantises, so what matters is the target.
    bool a_biased = false;
    [[maybe_unused]] int32_t a_bias_offset = 0;
    if constexpr (!R::is_fp) {
        const DataType a_dt = R::a_eff_type(attrs, a);
        a_biased = (int8_a_pack_bias(a_dt) != 0);
        a_bias_offset = int8_epilogue_offset(a_dt);
    }

    // plan.ldd_b is the full-Kc stride used for *sizing*; each k-block packs at
    // its own actual_kc stride, which is <= that.
    const int kc = static_cast<int>(plan.kc);
    const int mc = static_cast<int>(plan.mc);
    const int nc = static_cast<int>(plan.nc);
    const int64_t ldd_b_full = plan.ldd_b;
    const int np_full = static_cast<int>(plan.np_slice);

    // Workspace: packed-B panels, then packed A, then (s8 only) the accumulator.
    Elem* pack_b_base = static_cast<Elem*>(workspace);
    auto* pack_a_base = reinterpret_cast<Elem*>(
        static_cast<char*>(workspace) + plan.pack_a_offset);
    Acc* accum = nullptr;
    if constexpr (!R::is_fp) {
        if (compact_acc) {
            accum = reinterpret_cast<Acc*>(pack_b_base + plan.workspace_size);
        }
    }

    const BatchWalk batch(a, b);
    const int Mi = static_cast<int>(M);
    const int Ni = static_cast<int>(N);
    const int Ki = static_cast<int>(K);

    // A block owns an n-block on an N-split or an m-block on an M-split.
    const bool split_n = plan.split_n;
    const int64_t num_blocks = split_n ? split_block_count(Ni, nc)
                                       : split_block_count(Mi, mc);

    // The reduction sweep and the epilogue exist only to apply quantization:
    // with no zero points, no activation, no s8 output and no A shift left to
    // withdraw, both are arithmetic identities (compensate reduces to `raw`).
    // They are not free — a full sweep of A and B up front, then a pass over
    // every element of C — and at shallow K that fixed cost is what keeps the
    // call off the kernel rate.
    // Two separate gates. `int8_ep` says the output pass runs on the last-k
    // tile; `int8_passes` says the reductions that feed its compensation are
    // needed. They differ only on the dq route, whose output pass is the
    // dequant multiply and therefore always runs, while the sweep is pure
    // overhead whenever compensate() is the identity: both subtractive terms
    // vanish when a_bias_offset + zp_a == 0 for every row and every zp_b is
    // zero, which is the ordinary symmetric s8 W8A8 setup.
    bool int8_ep = false;
    bool int8_passes = false;
    std::vector<int32_t> r_a, r_b;
    if constexpr (!R::is_fp) {
        const QuantParams& qa = a.quant_params();
        const QuantParams& qb = b.quant_params();
        if constexpr (R::is_dq) {
            int8_ep = true;
            int8_passes = dq_needs_compensation(a, b, a_bias_offset, Mi, Ni);
            if constexpr (R::fused_a) {
                // matmul_dq only fuses the identity case: the pack has no way
                // to accumulate r_a, and A_q (which the sweep reads) is gone.
                NNOPS_ASSERT(!int8_passes);
            }
        } else {
            int8_passes = out_s8 || a_bias_offset != 0 ||
                          attrs.epilogue.type != EpilogueActivateType::None ||
                          qa.zero_point != 0 || qb.zero_point != 0 ||
                          qa.zero_point_data != nullptr || qb.zero_point_data != nullptr;
            int8_ep = int8_passes;
        }
        if (int8_passes) {
            r_a.resize(static_cast<size_t>(Mi));
            r_b.resize(static_cast<size_t>(Ni));
        }
    }

    // The int8 output pass rides on the last k-block's tile, exactly where fp
    // runs its epilogue, so C is never swept a second time. Its full-K inputs
    // (the reductions) are ready before the blocks start; only the s8
    // destination moves per batch.
    using EpParams = typename R::EpilogueParams;
    EpParams ip{};
    if constexpr (!R::is_fp) {
        if (int8_ep) {
            const QuantParams& qa = a.quant_params();
            const QuantParams& qb = b.quant_params();
            ip.r_a = int8_passes ? r_a.data() : nullptr;
            ip.r_b = int8_passes ? r_b.data() : nullptr;
            ip.s_a = qa.scale_data;
            ip.s_a_scalar = qa.scale;
            ip.zp_a = qa.zero_point_data;
            ip.zp_a_scalar = qa.zero_point;
            ip.s_b = qb.scale_data;
            ip.s_b_scalar = qb.scale;
            ip.zp_b = qb.zero_point_data;
            ip.zp_b_scalar = qb.zero_point;
            ip.a_bias_offset = a_bias_offset;
            ip.k = Ki;
            ip.ldc_out = ldc_out;
            if constexpr (R::is_dq) {
                ip.compensate = int8_passes;
                ip.act_u8 = R::a_eff_type(attrs, a) == DataType::u8;
                ip.ep = &attrs.epilogue;
            } else {
                const QuantParams& qc = output.quant_params();
                ip.out_scale = quant_scale_at(qc, 0);  // requant only
                ip.zp_out = quant_zp_at(qc, 0);
                ip.relu = attrs.epilogue.type == EpilogueActivateType::Relu;
            }
        }
    }
    const EpParams* ip_or_null = int8_ep ? &ip : nullptr;

    for (int64_t bi = 0; bi < batch.total; ++bi) {
        int64_t a_offset, b_offset, c_offset;
        batch.offsets(bi, output, a_offset, b_offset, c_offset);

        const typename R::AElem* a_p = a_base + a_offset;
        const typename R::B b_src = R::b_source(b, b_offset, ldb, b_q51);
        Acc* c_p = R::c_base(output, c_offset, accum);

        if constexpr (!R::is_fp) {
            // Integer matmul ignores alpha/beta, so C starts at zero. fp needs
            // no pass: its first k-block runs the zero_mode kernel.
            R::zero_c(c_p, ldc, Mi, Ni);
            if constexpr (!R::fused_a) {
                if (int8_passes) {
                    compute_int8_reductions(a_p, R::a_eff_type(attrs, a) == DataType::u8,
                                            lda, b_src.ptr, ldb, attrs, M, N, K,
                                            r_a.data(), r_b.data(), ctx);
                }
            }
            if (int8_ep) {
                if constexpr (R::is_dq) {
                    ip.out = output.ptr<typename R::OutElem>() + c_offset;
                    // Per-token arrays carry one entry per global row
                    // (num_scales == total_rows()); the tile indexes its rows
                    // within the batch. A_q is compact, so the batch offset is
                    // a whole number of rows.
                    const QuantParams& qa_b = a.quant_params();
                    const int64_t row_base = (a_offset > 0 && lda > 0) ? a_offset / lda : 0;
                    ip.s_a = qa_b.scale_data ? qa_b.scale_data + row_base : nullptr;
                    ip.zp_a = qa_b.zero_point_data ? qa_b.zero_point_data + row_base
                                                   : nullptr;
                } else if (out_s8) {
                    ip.out = output.ptr<int8_t>() + c_offset;
                }
            }
        }

        if (split_n) {
            // NKM: one n-block per parallel task, every block over all of M.
            ctx.cpu.run(0, num_blocks, [&](int64_t blk) {
                const auto nr = n_range(plan, blk, Ni);
                const int64_t slot = block_slot(plan.num_slots, num_blocks, blk, ctx);
                matmul_block<R>(attrs, c_p, ldc, a_p, lda, b_src,
                                Ki, kc, mc, 0, Mi, nr.start, nr.count,
                                pack_b_base + slot * np_full * ldd_b_full,
                                pack_a_slot(plan, num_blocks, blk, ctx, pack_a_base),
                                plan.pack_a, a_biased,
                                ip_or_null);
            });
        } else if constexpr (R::hoists_b_m_split) {
            // MKN, hoisted: pack the full-N B once per k-block, then split the
            // rows. The one packed-B slice serves every m-block.
            for (int k = 0; k < Ki; k += kc) {
                const int actual_kc = std::min(kc, Ki - k);
                const bool last_k = (k + kc >= Ki);
                R::pack_b(attrs, b_src, k, actual_kc, 0, Ni,
                          pack_b_base, R::ldd_b(actual_kc), a_biased);
                ctx.cpu.run(0, num_blocks, [&](int64_t blk) {
                    const auto mr = m_range(plan, blk, num_blocks, Mi);
                    matmul_tile<R>(attrs, c_p, ldc, a_p, lda,
                                   mr.start, mr.count, mc, 0, Ni,
                                   k, actual_kc, last_k, plan.pack_a, pack_b_base,
                                   pack_a_slot(plan, num_blocks, blk, ctx, pack_a_base),
                                   a_biased, ip_or_null);
                });
            }
        } else {
            // MKN: every m-block packs its own full-N B per k-block.
            ctx.cpu.run(0, num_blocks, [&](int64_t blk) {
                const auto mr = m_range(plan, blk, num_blocks, Mi);
                const int64_t slot = block_slot(plan.num_slots, num_blocks, blk, ctx);
                matmul_block<R>(attrs, c_p, ldc, a_p, lda, b_src,
                                Ki, kc, mc, mr.start, mr.count, 0, Ni,
                                pack_b_base + slot * np_full * ldd_b_full,
                                pack_a_slot(plan, num_blocks, blk, ctx, pack_a_base),
                                plan.pack_a, a_biased,
                                ip_or_null);
            });
        }
    }
}

// =========================================================================
//  Offline-quantized activations — quantize A, then ride the int8 GEMM
// =========================================================================

/// Quantize A into A_q (A's shape, compact) with A's own offline per-tensor /
/// per-token parameters.
///
/// Rows are split across the pool. The quant kernel wants a per-row float
/// zero-point array, so a per-token int32 array (or PerTensor's single scalar)
/// is materialized once per call; per-token arrays are indexed by the global
/// row, which the compact layout preserves.
template <class AT>
void quantize_activation(const TensorView& a, TensorView& aq,
                         const ComputeContext& ctx) {
    const QuantParams& qp = a.quant_params();
    const int64_t rows = a.total_rows();
    const int64_t lda = a.row_stride_elems();
    const int64_t K = a.shape(a.rank() - 1);
    if (rows <= 0 || K <= 0) { return; }

    std::vector<float> scale_buf, zero_buf;
    const float* scale_arr = qp.scale_data;
    if (scale_arr == nullptr) {
        scale_buf.assign(static_cast<size_t>(rows), qp.scale);
        scale_arr = scale_buf.data();
    }
    const float* zero_arr = nullptr;
    if (qp.zero_point_data != nullptr) {
        zero_buf.resize(static_cast<size_t>(rows));
        for (int64_t i = 0; i < rows; ++i) {
            zero_buf[static_cast<size_t>(i)] =
                static_cast<float>(qp.zero_point_data[i]);
        }
        zero_arr = zero_buf.data();
    } else if (qp.zero_point != 0) {
        zero_buf.assign(static_cast<size_t>(rows), static_cast<float>(qp.zero_point));
        zero_arr = zero_buf.data();
    }

    const AT* src = a.ptr<AT>();
    const int n = static_cast<int>(K);
    const int sr = static_cast<int>(lda);
    const bool u8 = aq.data_type() == DataType::u8;
    uint8_t*  d8  = u8 ? aq.ptr<uint8_t>() : nullptr;
    int8_t*   ds8 = u8 ? nullptr : aq.ptr<int8_t>();

    const int64_t tasks = std::min<int64_t>(rows, std::max<int64_t>(ctx.cpu.thread_count(), 1));
    ctx.cpu.run(0, tasks, [&](int64_t t) {
        const int64_t r0 = t * rows / tasks;
        const int64_t r1 = (t + 1) * rows / tasks;
        if (r1 <= r0) { return; }
        const int m = static_cast<int>(r1 - r0);
        const float* scale = scale_arr + r0;
        const float* zero = zero_arr ? zero_arr + r0 : nullptr;
        if (u8) {
            kernel::quant_kernel::quantization<uint8_t>(
                m, n, d8 + r0 * K, n, src + r0 * lda, sr, scale, zero);
        } else {
            kernel::quant_kernel::quantization<int8_t>(
                m, n, ds8 + r0 * K, n, src + r0 * lda, sr, scale, zero);
        }
    });
}

/// The fp-activation entry point: materialize A_q, then run the int8 GEMM with
/// the dequantizing output pass.
///
/// A_q is handed to the plan as inputs[0] on purpose: it carries A's shape and
/// A's QuantParams in the integer storage, so get_matmul_plan, the batch walk,
/// lda, the reductions and the arch pack bias (derived from A_q's dtype) are
/// all correct with no plan changes.
template <class T>
void matmul_dq(const MatMulAttributes& attrs, TensorView& output,
               std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    NNOPS_ASSERT(attrs.act_quant_dtype == DataType::s8 ||
                 attrs.act_quant_dtype == DataType::u8);

    // A_q's shape and parameters, with no storage: the plan only reads the
    // description, and the int8 geometry is chosen by the target dtype.
    const TensorView aq_shape(a.shape_span(), attrs.act_quant_dtype, nullptr,
                              TensorLayout::NCHW, a.quant_params());
    auto plan = get_matmul_plan(attrs, aq_shape.desc(), b.desc(),
                                ctx.cpu.thread_count(),
                                ctx.cpu.thread_slot_count() > 0);
    NNOPS_ASSERT(plan.mc > 0 && plan.nc > 0 && plan.kc > 0);

    // Fused A: nothing ever reads A_q as a whole — no reduction sweep (the
    // compensation is the identity here) and, on an M-split, each A row is
    // packed exactly once — so the pack can quantise straight from A and the
    // M×K intermediate never exists. An N-split re-packs A once per n-block,
    // which would quantise the same row `nt` times; it keeps the materialised
    // form.
    const auto [M, N, K] = matmul_dims(attrs, a, b);
    const int32_t a_bias_offset = int8_epilogue_offset(attrs.act_quant_dtype);
    if (!plan.split_n && !attrs.transpose_a &&
        !dq_needs_compensation(a, b, a_bias_offset, M, N)) {
        const TensorView ins[] = {a, b};
        PoolPtr scratch(static_cast<size_t>(plan.workspace_size)
                        + DqRoute<T, T>::extra_workspace(output));
        matmul_dispatch_tiled<DqRoute<T, T>>(attrs, output, ins, ctx, plan,
                                             scratch.get());
        return;
    }

    PoolPtr aq_buf(static_cast<size_t>(a.numel()));
    TensorView aq(a.shape_span(), attrs.act_quant_dtype, aq_buf.get(),
                  TensorLayout::NCHW, a.quant_params());
    quantize_activation<T>(a, aq, ctx);

    const TensorView ins[] = {aq, b};
    PoolPtr scratch(static_cast<size_t>(plan.workspace_size)
                    + DqRoute<T>::extra_workspace(output));
    matmul_dispatch_tiled<DqRoute<T>>(attrs, output, ins, ctx, plan, scratch.get());
}

}  // anonymous namespace

// =========================================================================
//  Public API
// =========================================================================

void matmul_cpu(const MatMulAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx,
                   void* /*workspace*/)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    NNOPS_ASSERT(a.rank() >= 2);
    NNOPS_ASSERT(b.rank() >= 2);
    NNOPS_ASSERT(inputs.size() >= 2 && inputs.size() <= 3);

    const auto dt_a = a.data_type();
    const auto dt_b = b.data_type();

    // Fold pure-clamp activations into min_clip/max_clip, which kblock_clamp
    // (and the Q5 route) read directly.
    MatMulAttributes attrs_n = attrs;
    switch (attrs_n.epilogue.type) {
    case EpilogueActivateType::Relu:  attrs_n.epilogue.min_clip = 0.0f; break;
    case EpilogueActivateType::Relu6: attrs_n.epilogue.min_clip = 0.0f; attrs_n.epilogue.max_clip = 6.0f; break;
    default: break;
    }

#if defined(NNOPS_ARCH_X86_64)
    // Small-M Q5_1: decode straight into the reduction, before the plan and the
    // scratch. Only a transposed A is out of reach.
    if (dt_a == DataType::f32 && dt_b == DataType::q5_1 && !attrs_n.transpose_a) {
        const auto [M, N, K] = matmul_dims(attrs_n, a, b);
        if (matmul_q5_fused(attrs_n, output, inputs, ctx,
                            static_cast<int>(M), static_cast<int>(N),
                            static_cast<int>(K))) {
            return;
        }
    }
#endif

    // A dense f32/f16 product that is small enough (or narrow enough) skips the
    // plan: no pack, no tiles, no scratch. The thresholds live with the kernels.
    const bool b_q51 = (dt_b == DataType::q5_1);
    const bool float_a = dt_a == DataType::f32 || dt_a == DataType::f16;
    if (matmul_small_fp(attrs_n, output, inputs, ctx)) {
        return;
    }

    // Offline-quantized activations × s8 weights: quantize A with its own
    // per-tensor / per-token parameters, run the int8 GEMM, dequantize into A's
    // dtype. Without active parameters the pair has no defined semantics — a
    // float A is not a quantized tensor to dequantize — so it takes the
    // reference rather than misreading the int8 weights as floats.
    if (float_a && dt_b == DataType::s8) {
        const QuantParams& qp = a.quant_params();
        const bool qp_ok = qp.is_active() &&
                           (qp.granularity == QuantGranularity::PerTensor ||
                            qp.granularity == QuantGranularity::PerToken);
        if (!qp_ok || attrs_n.transpose_a) {
            NNOPS_ASSERT_MSG(qp_ok,
                             "MatMul: fp A x s8 B needs per-tensor or per-token "
                             "QuantParams on A");
            // transpose_a has no packed A_q path yet; the reference is exact.
            reference::matmul_dq_ref(attrs_n, output, inputs, ctx, nullptr);
            return;
        }
        NNOPS_ASSERT(output.data_type() == dt_a);
        if (dt_a == DataType::f32) {
            matmul_dq<float>(attrs_n, output, inputs, ctx);
        } else {
            matmul_dq<half>(attrs_n, output, inputs, ctx);
        }
        return;
    }

    // Worker ids → one scratch slot per thread instead of per m-block.
    auto plan = get_matmul_plan(attrs_n, a.desc(), b.desc(), ctx.cpu.thread_count(),
                                ctx.cpu.thread_slot_count() > 0);

    // int8: tiled for s32 / s8 outputs, reference otherwise. An s8 output adds
    // an int32 M×N accumulator after the scratch.
    if (is_int8_matmul(dt_a, dt_b)) {
        const DataType odt = output.data_type();
        // An fp output dequantizes the same int32 accumulate instead of
        // returning or requantizing it — the "already-quantized A" entry point
        // mirrors the fp-activation one, sharing the epilogue kernel.
        const bool out_fp = (odt == DataType::f32 || odt == DataType::f16);
        const bool out_ok = (odt == DataType::s32) || (odt == DataType::s8) || out_fp;
        if (out_ok) {
            NNOPS_ASSERT(plan.mc > 0 && plan.nc > 0 && plan.kc > 0);
            if (out_fp && odt == DataType::f32) {
                PoolPtr scratch(static_cast<size_t>(plan.workspace_size)
                                + DqRoute<float>::extra_workspace(output));
                matmul_dispatch_tiled<DqRoute<float>>(attrs_n, output, inputs, ctx,
                                                      plan, scratch.get());
            } else if (out_fp) {
                PoolPtr scratch(static_cast<size_t>(plan.workspace_size)
                                + DqRoute<half>::extra_workspace(output));
                matmul_dispatch_tiled<DqRoute<half>>(attrs_n, output, inputs, ctx,
                                                     plan, scratch.get());
            } else {
                PoolPtr scratch(static_cast<size_t>(plan.workspace_size)
                                + I8Route::extra_workspace(output));
                matmul_dispatch_tiled<I8Route>(attrs_n, output, inputs, ctx, plan,
                                               scratch.get());
            }
        } else {
            reference::matmul_int8_ref(attrs_n, output, inputs, ctx, nullptr);
        }
        return;
    }

    // Past the int8 branch: only f32/f16 have a fused kernel, plus a Q5_1 B
    // (its blocks decode to A's type in the RHS pack).
    const bool supported = float_a && (dt_b == dt_a || b_q51);
    if (!supported) {
        reference::matmul_ref(attrs_n, output, inputs, ctx, nullptr);
        return;
    }

    PoolPtr scratch(plan.workspace_size);

    if (dt_a == DataType::f32) {
        matmul_dispatch_tiled<FpRoute<float>>(attrs_n, output, inputs, ctx, plan,
                                              scratch.get());
    } else {
        matmul_dispatch_tiled<FpRoute<half>>(attrs_n, output, inputs, ctx, plan,
                                             scratch.get());
    }
}

}  // namespace nnops::backend::cpu
