/// @file matmul.cpp
/// @brief Tiled matrix multiplication kernel — pack + MMA dispatch.
///
/// f32, f16 and int8 share one skeleton (FpRoute / I8Route): B is always packed
/// into pooled scratch; A is packed too when the route needs it (always int8)
/// or read directly. Tiling, the split direction and the workspace size come
/// from get_matmul_plan(): NKM on an N-split, MKN on an M-split.
///
/// f32/f16 below GEMM_FAST_PATH_THRESHOLD MACs use the arch's naive SIMD kernel
/// instead of the plan. int8 accumulates raw int32 and recovers the product in
/// one epilogue pass.

#include "matmul.h"
#include "matmul_helper.h"
#include "simd_kernel/simd_epilogue.hpp"
#include "common/memory_pool.hpp"           // internal scratch-memory pool
#include "nnops/detail/half.hpp"
#include "nnops/detail/assert.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>
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
}

namespace nnops::backend::cpu {

#if defined(NNOPS_ARCH_X86_64)
/// Benchmark-only cap for the fused Q5_1 route: negative = the limits below,
/// 0 = disabled, else caps M (so one process can A/B both routes).
int q5_fused_max_m_override = -1;
#endif

namespace {

// =========================================================================
//  Shared helpers
// =========================================================================

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

/// Q5_1 B for a batch element; BatchWalk yields block offsets, not bytes.
inline const uint8_t* q5_rhs_at(const uint8_t* base, int64_t block_off) noexcept {
    return base + block_off * static_cast<int64_t>(q5_1_block_bytes);
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

// =========================================================================
//  Small GEMM — naive SIMD, no plan
// =========================================================================
//
// Below GEMM_FAST_PATH_THRESHOLD MACs, packing and tiling cost more than the
// arithmetic. Each batch element runs the arch's naive triple loop, followed by
// one epilogue pass over the (tiny) output.

/// Total MACs below which the small path stays on the calling thread: a pool
/// spawn costs more than the work it would spread.
constexpr int64_t NAIVE_PARALLEL_MACS = 32768;

template <typename T>
void matmul_naive_small(const MatMulAttributes& attrs, TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];
    const auto [M, N, K] = matmul_dims(attrs, a, b);

    const int lda = static_cast<int>(a.row_stride_elems());
    const int ldb = static_cast<int>(b.row_stride_elems());
    const int ldc = static_cast<int>(output.row_stride_elems());
    const int Mi = static_cast<int>(M);
    const int Ni = static_cast<int>(N);
    const int Ki = static_cast<int>(K);

    const T* a_base = a.ptr<T>();
    const T* b_base = b.ptr<T>();
    T* c_base = output.ptr<T>();

    const BatchWalk batch(a, b);
    const auto gemm = [&](int64_t bi) {
        int64_t a_off, b_off, c_off;
        batch.offsets(bi, output, a_off, b_off, c_off);
        arch::matmul_naive(Mi, Ni, Ki,
                           a_base + a_off, lda, attrs.transpose_a,
                           b_base + b_off, ldb, attrs.transpose_b,
                           c_base + c_off, ldc, attrs.beta);
    };

    if (batch.total > 1 && batch.total * M * N * K >= NAIVE_PARALLEL_MACS) {
        ctx.cpu.run(0, batch.total, gemm);
    } else {
        for (int64_t bi = 0; bi < batch.total; ++bi) { gemm(bi); }
    }

    if (attrs.epilogue.type == EpilogueActivateType::None) { return; }
    for (int64_t bi = 0; bi < batch.total; ++bi) {
        int64_t a_off, b_off, c_off;
        batch.offsets(bi, output, a_off, b_off, c_off);
        T* c_p = c_base + c_off;
        for (int m = 0; m < Mi; ++m) {
            for (int n = 0; n < Ni; ++n) {
                T& slot = c_p[m * ldc + n];
                if constexpr (std::is_same_v<T, float>) {
                    slot = apply_epilogue(attrs.epilogue, slot);
                } else {
                    slot = float_to_half(apply_epilogue(attrs.epilogue, half_to_float(slot)));
                }
            }
        }
    }
}

// =========================================================================
//  Dtype routes — the only place a path names its element types
// =========================================================================
//
// Per k-block pack B, per M panel pack A (or read it raw) and call one MMA
// kernel that sweeps the N side. Everything dtype-specific is named here.

template <class T>
struct FpRoute {
    using Elem  = T;
    using Acc   = T;
    using Clamp = float;

    static constexpr bool is_fp            = true;
    static constexpr bool has_direct_a     = true;   // A may be read through lda
    static constexpr bool hoists_b_m_split = false;  // B packs inside each block

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
                       const Elem* src, int lds, bool /*a_biased*/) {
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

/// int8 (s8|u8 A × s8 B): no raw A, raw int32 C, epilogue once after all
/// blocks.
struct I8Route {
    using Elem  = int8_t;
    using Acc   = int32_t;
    using Clamp = int32_t;

    static constexpr bool is_fp            = false;
    static constexpr bool has_direct_a     = false;
    static constexpr bool hoists_b_m_split = true;  // one B pack per k-block

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
                       const Elem* src, int lds, bool a_biased) {
        tile_pack_lhs_i8(trans, mc, kc, dst, ldd, src, lds, a_biased);
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
        if (output.data_type() != DataType::s8) { return 0; }
        const auto od = output.desc();
        return static_cast<size_t>(od.dims[static_cast<size_t>(od.rank - 2)])
             * static_cast<size_t>(od.dims[static_cast<size_t>(od.rank - 1)])
             * sizeof(Acc);
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
                 const typename R::Elem* a_ptr, int lda,
                 int m_start, int m_count, int mc,
                 int n_start, int n_count,
                 int k, int actual_kc, bool last_k,
                 bool pack_a, const typename R::Elem* packed_b,
                 typename R::Elem* pack_a_slice, bool a_biased)
{
    using Elem = typename R::Elem;
    using Acc  = typename R::Acc;

    const int ldd_a = R::ldd_a(actual_kc);
    const bool zero_mode = (k == 0 && attrs.beta == 0.0f);  // C is overwritten
    const auto [cmin, cmax] = R::clamp(attrs, last_k);
    // fp-only; compile out of the int8 instantiation.
    const bool scale_c  = R::is_fp && k == 0 && attrs.beta != 1.0f && attrs.beta != 0.0f;
    const bool epilogue = R::is_fp && last_k && has_inplace_epilogue(attrs);

    for (int m = m_start; m < m_start + m_count; m += mc) {
        const int actual_mc = std::min(mc, m_start + m_count - m);
        Acc* c_tile = c_ptr + m * ldc + n_start;

        const Elem* a_src = attrs.transpose_a
            ? a_ptr + k * lda + m    // A phys is K×M, row k, col m
            : a_ptr + m * lda + k;   // A phys is M×K, row m, col k

        if constexpr (R::is_fp) {
            if (scale_c) { R::scale(c_tile, ldc, attrs.beta, actual_mc, n_count); }
        }
        if (pack_a) {
            R::pack_a(attrs.transpose_a, actual_mc, actual_kc,
                      pack_a_slice, ldd_a, a_src, lda, a_biased);
            R::mma_pack(actual_mc, n_count, actual_kc,
                        c_tile, ldc, pack_a_slice, packed_b, cmin, cmax, zero_mode);
        } else if constexpr (R::has_direct_a) {
            R::mma_direct(actual_mc, n_count, actual_kc,
                          c_tile, ldc, a_src, lda, packed_b, cmin, cmax, zero_mode);
        }
        if constexpr (R::is_fp) {
            if (epilogue) { R::epilogue(attrs, c_tile, ldc, actual_mc, n_count); }
        }
    }
}

/// One block over all K: pack B per k-block, accumulate every m-panel.
template <class R>
void matmul_block(const MatMulAttributes& attrs,
                  typename R::Acc* c_ptr, int ldc,
                  const typename R::Elem* a_ptr, int lda,
                  const typename R::B& b_src,
                  int K, int kc, int mc,
                  int m_start, int m_count, int n_start, int n_count,
                  typename R::Elem* pack_b_slice,
                  typename R::Elem* pack_a_slice,
                  bool pack_a, bool a_biased)
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
                       a_biased);
    }
}


// =========================================================================
//  Fused Q5_1 small-M route — decode straight into the dot product
// =========================================================================
// No packed B, no workspace, no k-blocking; each call sweeps K once.

#if defined(NNOPS_ARCH_X86_64)
/// Largest M routed to the fused Q5_1 kernels, measured at K=N=4096: the dot
/// kernel stays 2x..3.8x ahead to M=16 and loses at 32, where the per-panel
/// re-decode outweighs the packed panel round trip; the outer kernel (2 rows
/// per panel, four accumulators each) wins to M=4.
constexpr int Q5_FUSED_MAX_M_TRANS = 16;
constexpr int Q5_FUSED_MAX_M_COPY  = 4;

/// Two Q5_1 blocks per worker task, which is also the copy kernel's minimum.
constexpr int Q5_FUSED_NTILE = 2 * static_cast<int>(q5_1_block_elems);

/// Call the kernel instantiated for @p MR rows, recursing down to the live count.
template <bool ZERO, bool TRANS, int MR>
inline void q5_fused_rows(int mr, int nc, int K, float* c, int ldc,
                          const float* a, int lda, const uint8_t* B,
                          int64_t row_blocks, int n_start,
                          float cmin, float cmax) {
    if (mr >= MR) {
        if constexpr (TRANS) {
            arch::mma_q5_f32_trans<MR, ZERO>(nc, K, c, ldc, a, lda, B, row_blocks,
                                             n_start, cmin, cmax);
        } else {
            arch::mma_q5_f32_copy<MR, ZERO>(nc, K, c, ldc, a, lda, B, row_blocks,
                                            n_start, cmin, cmax);
        }
    } else if constexpr (MR > 1) {
        q5_fused_rows<ZERO, TRANS, MR - 1>(mr, nc, K, c, ldc, a, lda, B,
                                           row_blocks, n_start, cmin, cmax);
    }
}

/// One m-panel at a time: 8 rows for the dot kernel, 2 for the outer product.
template <bool ZERO, bool TRANS>
inline void q5_fused_dispatch(int M, int nc, int K, float* C, int ldc,
                              const float* A, int lda, const uint8_t* B,
                              int64_t row_blocks, int n_start,
                              float cmin, float cmax) {
    constexpr int kPanel = TRANS ? 8 : 2;
    for (int m = 0; m < M; m += kPanel) {
        q5_fused_rows<ZERO, TRANS, kPanel>(
            std::min(kPanel, M - m), nc, K, C + static_cast<int64_t>(m) * ldc,
            ldc, A + static_cast<int64_t>(m) * lda, lda, B, row_blocks,
            n_start, cmin, cmax);
    }
}

/// Select the kernel family for one n tile; ZERO is the beta == 0 case.
template <bool ZERO>
inline void q5_fused_run(bool trans, int M, int nc, int K, float* C, int ldc,
                         const float* A, int lda, const uint8_t* B,
                         int64_t row_blocks, int n_start,
                         float cmin, float cmax) {
    if (trans) {
        q5_fused_dispatch<ZERO, true>(M, nc, K, C, ldc, A, lda, B, row_blocks,
                                      n_start, cmin, cmax);
    } else {
        q5_fused_dispatch<ZERO, false>(M, nc, K, C, ldc, A, lda, B, row_blocks,
                                       n_start, cmin, cmax);
    }
}

/// Run the whole product through the fused Q5_1 kernels; false if not applicable.
bool matmul_q5_fused(const MatMulAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     int M, int N, int K)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    const bool trans = attrs.transpose_b;
    if (K <= 0 || N <= 0) {
        return false;
    }
    int max_m = trans ? Q5_FUSED_MAX_M_TRANS : Q5_FUSED_MAX_M_COPY;
    if (q5_fused_max_m_override >= 0) {
        max_m = q5_fused_max_m_override;
    }
    if (M > max_m) {
        return false;
    }
    const int lda = static_cast<int>(a.row_stride_elems());
    const int ldc = static_cast<int>(output.row_stride_elems());
    const int64_t row_blocks = b.row_stride_elems();

    const float* a_base = a.ptr<float>();
    const uint8_t* b_base = static_cast<const uint8_t*>(b.ptr<void>());
    float* c_base = output.ptr<float>();

    const auto [cmin, cmax] = kblock_clamp(attrs, /*last_k=*/true);
    const bool epilogue = has_inplace_epilogue(attrs);
    const bool scale_c = attrs.beta != 0.0f && attrs.beta != 1.0f;

    const BatchWalk batch(a, b);
    const int nt = Q5_FUSED_NTILE;
    const int64_t n_tiles = split_block_count(N, nt);

    // One flat task list over (batch, n tile): batch elements need no fork/join.
    ctx.cpu.run(0, batch.total * n_tiles, [&](int64_t task) {
        const int64_t bi = task / n_tiles;
        const int64_t t = task % n_tiles;
        const int n_start = static_cast<int>(t) * nt;
        const int nc = std::min(nt, static_cast<int>(N) - n_start);

        int64_t a_off, b_off, c_off;
        batch.offsets(bi, output, a_off, b_off, c_off);
        const float* a_p = a_base + a_off;
        const uint8_t* b_p = q5_rhs_at(b_base, b_off);
        float* c_p = c_base + c_off + n_start;

        if (scale_c) {
            tile_scale(c_p, ldc, attrs.beta, M, nc);
        }

        if (attrs.beta == 0.0f) {
            q5_fused_run<true>(trans, M, nc, K, c_p, ldc, a_p, lda, b_p,
                               row_blocks, n_start, cmin, cmax);
        } else {
            q5_fused_run<false>(trans, M, nc, K, c_p, ldc, a_p, lda, b_p,
                                row_blocks, n_start, cmin, cmax);
        }

        if (epilogue) {
            epilogue_inplace(M, nc, c_p, ldc,
                             static_cast<const float*>(nullptr), attrs.epilogue);
        }
    });
    return true;
}
#endif  // NNOPS_ARCH_X86_64

// =========================================================================
//  int8 — reduction and epilogue passes around the raw accumulate
// =========================================================================

/// Scalar fallback for PerTensor; PerToken / PerChannel index the buffer at
/// @p idx (output row m for A, output column n for B).
inline int32_t quant_zp_at(const QuantParams& q, int64_t idx) noexcept {
    return (q.zero_point_data != nullptr) ? q.zero_point_data[idx] : q.zero_point;
}
inline float quant_scale_at(const QuantParams& q, int64_t idx) noexcept {
    return (q.scale_data != nullptr) ? q.scale_data[idx] : q.scale;
}

/// r[i] = Σ_j p[j*ld + i]. Sweeping the outer axis keeps the output contiguous;
/// the per-output form would touch a fresh cache line per element.
template <class AT>
void reduce_cols(const AT* p, int64_t ld, int64_t n_out, int64_t n_in,
                 int32_t* r, const ComputeContext& ctx) {
    const int64_t tasks = std::min<int64_t>(n_out,
        std::max<int64_t>(ctx.cpu.thread_count(), 1));
    ctx.cpu.run(0, tasks, [&](int64_t t) {
        const int64_t i0 = t * n_out / tasks;
        const int64_t i1 = (t + 1) * n_out / tasks;
        for (int64_t i = i0; i < i1; ++i) r[i] = 0;
        for (int64_t j = 0; j < n_in; ++j) {
            const AT* row = p + j * ld;
            for (int64_t i = i0; i < i1; ++i) r[i] += static_cast<int32_t>(row[i]);
        }
    });
}

/// r[i] = Σ_j p[i*ld + j]. Each output is already a contiguous row sum.
template <class AT>
void reduce_rows(const AT* p, int64_t ld, int64_t n_out, int64_t n_in,
                 int32_t* r, const ComputeContext& ctx) {
    const int64_t tasks = std::min<int64_t>(n_out,
        std::max<int64_t>(ctx.cpu.thread_count(), 1));
    ctx.cpu.run(0, tasks, [&](int64_t t) {
        const int64_t i0 = t * n_out / tasks;
        const int64_t i1 = (t + 1) * n_out / tasks;
        for (int64_t i = i0; i < i1; ++i) {
            const AT* row = p + i * ld;
            int32_t s = 0;
            for (int64_t j = 0; j < n_in; ++j) s += static_cast<int32_t>(row[j]);
            r[i] = s;
        }
    });
}

/// Raw int8 reductions for the epilogue: r_a[m] = Σ_k A[m,k], r_b[n] = Σ_k B[k,n]
/// (A[k,m] / the mirror image when transposed). Output slabs are disjoint, so
/// no atomics; int32 accumulation matches the reference bit for bit.
///
/// B is always int8; @p a_u8 selects A's element type — a u8 A must be summed
/// unsigned, since the compensation is over the true value.
void compute_int8_reductions(const int8_t* a_ptr, bool a_u8, int64_t lda,
                             const int8_t* b_ptr, int64_t ldb,
                             const MatMulAttributes& attrs,
                             int64_t M, int64_t N, int64_t K,
                             int32_t* r_a, int32_t* r_b,
                             const ComputeContext& ctx) {
    if (a_u8) {
        const uint8_t* au = reinterpret_cast<const uint8_t*>(a_ptr);
        if (attrs.transpose_a) reduce_cols(au, lda, M, K, r_a, ctx);
        else                   reduce_rows(au, lda, M, K, r_a, ctx);
    } else {
        if (attrs.transpose_a) reduce_cols(a_ptr, lda, M, K, r_a, ctx);
        else                   reduce_rows(a_ptr, lda, M, K, r_a, ctx);
    }

    if (attrs.transpose_b) reduce_rows(b_ptr, ldb, N, K, r_b, ctx);
    else                   reduce_cols(b_ptr, ldb, N, K, r_b, ctx);
}

/// int8 epilogue: compensation plus (for an s8 output) requantization, in one
/// pass. Rows are independent, so they split into slabs without atomics.
void matmul_int8_epilogue(const MatMulAttributes& attrs,
                          const TensorView& a, const TensorView& b,
                          TensorView& output,
                          int32_t* c_p, int ldc, int64_t c_offset,
                          int M, int N, int K,
                          const int32_t* r_a, const int32_t* r_b,
                          int32_t a_bias_offset, const ComputeContext& ctx)
{
    const bool out_s8 = (output.data_type() == DataType::s8);
    const int ldc_out = static_cast<int>(output.row_stride_elems());
    int8_t* out_p = out_s8 ? output.ptr<int8_t>() + c_offset : nullptr;

    // Quantization parameters (batch-independent, indexed by m / n / 0).
    const QuantParams& qa = a.quant_params();
    const QuantParams& qb = b.quant_params();
    const QuantParams& qc = output.quant_params();
    const double  scale_out = (qc.scale_data != nullptr) ? static_cast<double>(qc.scale_data[0])
                                                         : static_cast<double>(qc.scale);
    const int32_t zp_out = (qc.zero_point_data != nullptr) ? qc.zero_point_data[0]
                                                           : qc.zero_point;
    const bool relu = (attrs.epilogue.type == EpilogueActivateType::Relu);

    const int64_t ep_tasks = std::min<int64_t>(M, std::max<int64_t>(ctx.cpu.thread_count(), 1));
    ctx.cpu.run(0, ep_tasks, [&](int64_t t) {
        const int64_t m0 = t * M / ep_tasks;
        const int64_t m1 = (t + 1) * M / ep_tasks;
        for (int64_t m = m0; m < m1; ++m) {
            const int32_t zp_a = quant_zp_at(qa, m);
            const double  s_a  = out_s8 ? static_cast<double>(quant_scale_at(qa, m)) : 0.0;
            for (int64_t n = 0; n < N; ++n) {
                // Both arches return the exact s8×s8 product. x86 folds the
                // VNNI bias out in-kernel; aarch64 leaves the u8 A's -128 shift
                // in, withdrawn here with the full-K column sums.
                int32_t v = matmul_int8_compensate(c_p[m * ldc + n], a_bias_offset,
                                                   zp_a, quant_zp_at(qb, n),
                                                   r_a[static_cast<size_t>(m)],
                                                   r_b[static_cast<size_t>(n)],
                                                   K);
                if (relu) {
                    v = std::max(v, 0);
                }
                if (out_s8) {
                    const double req = s_a * static_cast<double>(quant_scale_at(qb, n))
                                     / scale_out;
                    out_p[m * ldc_out + n] = matmul_int8_requant(v, req, zp_out);
                } else {
                    c_p[m * ldc + n] = v;
                }
            }
        }
    });
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

    // int8 accumulates raw int32: the s32 output in place, or an s8 output in
    // a compact M×N temporary the epilogue requantizes.
    const bool out_s8 = !R::is_fp && output.data_type() == DataType::s8;
    const int ldc = out_s8 ? static_cast<int>(N) : ldc_out;

    const Elem* a_base = a.ptr<Elem>();
    // A's hardware shift (see int8_a_pack_bias): the packers take it, the
    // epilogue takes what the kernels did not fold back out themselves.
    bool a_biased = false;
    [[maybe_unused]] int32_t a_bias_offset = 0;
    if constexpr (!R::is_fp) {
        a_biased = (int8_a_pack_bias(a.data_type()) != 0);
        a_bias_offset = int8_epilogue_offset(a.data_type());
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
        if (out_s8) {
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

    std::vector<int32_t> r_a, r_b;
    if constexpr (!R::is_fp) {
        r_a.resize(static_cast<size_t>(Mi));
        r_b.resize(static_cast<size_t>(Ni));
    }

    for (int64_t bi = 0; bi < batch.total; ++bi) {
        int64_t a_offset, b_offset, c_offset;
        batch.offsets(bi, output, a_offset, b_offset, c_offset);

        const Elem* a_p = a_base + a_offset;
        const typename R::B b_src = R::b_source(b, b_offset, ldb, b_q51);
        Acc* c_p = R::c_base(output, c_offset, accum);

        if constexpr (!R::is_fp) {
            // Integer matmul ignores alpha/beta, so C starts at zero. fp needs
            // no pass: its first k-block runs the zero_mode kernel.
            R::zero_c(c_p, ldc, Mi, Ni);
            compute_int8_reductions(a_p, a.data_type() == DataType::u8, lda,
                                    b_src.ptr, ldb, attrs, M, N, K,
                                    r_a.data(), r_b.data(), ctx);
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
                                plan.pack_a, a_biased);
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
                                   a_biased);
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
                                plan.pack_a, a_biased);
            });
        }

        if constexpr (!R::is_fp) {
            matmul_int8_epilogue(attrs, a, b, output, c_p, ldc, c_offset,
                                 Mi, Ni, Ki, r_a.data(), r_b.data(),
                                 a_bias_offset, ctx);
        }
    }
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

    // Small fp GEMMs skip the plan: no pack, no tiles, no scratch.
    const bool b_q51 = (dt_b == DataType::q5_1);
    const bool float_a = dt_a == DataType::f32 || dt_a == DataType::f16;
    if (float_a && dt_b == dt_a) {
        const auto [M, N, K] = matmul_dims(attrs_n, a, b);
        if (gemm_is_small(M, N, K)) {
            if (dt_a == DataType::f32) {
                matmul_naive_small<float>(attrs_n, output, inputs, ctx);
            } else {
                matmul_naive_small<half>(attrs_n, output, inputs, ctx);
            }
            return;
        }
    }

    // Worker ids → one scratch slot per thread instead of per m-block.
    auto plan = get_matmul_plan(attrs_n, a.desc(), b.desc(), ctx.cpu.thread_count(),
                                ctx.cpu.thread_slot_count() > 0);

    // int8: tiled for s32 / s8 outputs, reference otherwise. An s8 output adds
    // an int32 M×N accumulator after the scratch.
    if (is_int8_matmul(dt_a, dt_b)) {
        const bool out_ok = (output.data_type() == DataType::s32) ||
                            (output.data_type() == DataType::s8);
        if (out_ok) {
            NNOPS_ASSERT(plan.mc > 0 && plan.nc > 0 && plan.kc > 0);
            PoolPtr scratch(static_cast<size_t>(plan.workspace_size)
                            + I8Route::extra_workspace(output));
            matmul_dispatch_tiled<I8Route>(attrs_n, output, inputs, ctx, plan,
                                           scratch.get());
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
