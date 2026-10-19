/// @file matmul.cpp
/// @brief Tiled matrix multiplication kernel — pack + MMA dispatch.
///
/// B (rhs) is always packed into pooled scratch; A (lhs) is packed into a slot
/// of that same workspace only when transpose_a forces it or its row stride is
/// page-scattered, else read directly. Tiling, the pack decision, the split
/// direction, and the workspace size all come from get_matmul_plan()
/// (matmul_helper): the loop order is NKM when split on N, MKN when split on M.
///
/// f32 GEMMs below GEMM_FAST_PATH_THRESHOLD MACs route to the reference. Work is
/// split on the larger dimension via ctx.cpu.run; C tiles are disjoint with no
/// reductions, so the result is bit-identical to serial.
///
/// Beta scaling is fused into the first k-block; the clamp (epilogue
/// min_clip/max_clip) is folded into the last k-block and the other activations
/// apply in-place, matching matmul_ref semantics.

#include "matmul.h"
#include "matmul_helper.h"
#include "simd_kernel/simd_epilogue.hpp"
#include "common/memory_pool.hpp"           // internal scratch-memory pool
#include "nnops/detail/half.hpp"
#include "nnops/detail/assert.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
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
/// Diagnostic cap on the fused Q5_1 small-M route: negative uses the limits
/// declared below, 0 disables the route, a non-negative value caps M. Lets a
/// benchmark A/B the fused and packed routes inside one process, which host
/// drift makes otherwise unreliable.
int q5_fused_max_m_override = -1;
#endif

namespace {

// =========================================================================
//  Shared helpers
// =========================================================================

/// Clamp bounds for the current k-block, read directly from the epilogue's
/// explicit min_clip/max_clip (Relu → [0,∞] / Relu6 → [0,6] are folded into
/// those bounds upstream). Only the final k-block clamps; intermediate partial
/// sums must accumulate unclamped.
inline std::pair<float, float> kblock_clamp(const MatMulAttributes& attrs, bool last_k) noexcept {
    if (!last_k) {
        const float inf = std::numeric_limits<float>::infinity();
        return {-inf, inf};
    }
    return {attrs.epilogue.min_clip, attrs.epilogue.max_clip};
}

/// Non-clamp epilogues (Relu is folded into the MMA clamp) are applied
/// in-place after the last k-block of each tile.
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

/// Q5_1 B address for a batch element: BatchWalk yields block offsets, and the
/// stream is addressed in 24-byte blocks.
inline const uint8_t* q5_rhs_at(const uint8_t* base, int64_t block_off) noexcept {
    return base + block_off * static_cast<int64_t>(q5_1_block_bytes);
}

/// Broadcast A's and B's batch dimensions (same rule as matmul_ref) and walk the
/// flattened batch index back out to per-operand element offsets.
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
//  Tile-block kernels — one call processes one (m-range × n-range) tile
// =========================================================================

/// Accumulate one k-block into an m×n tile of C: C += A × packed_B. Shared by
/// both split directions. @p pack_a_slice is untouched when @p pack_a is false
/// (the direct route reads A through @p lda).
template <typename T>
void matmul_tile(const MatMulAttributes& attrs,
                 T* c_ptr, int ldc,
                 const T* a_ptr, int lda,
                 int m_start, int m_count, int mc,
                 int n_start, int n_count,
                 int k, int actual_kc, bool last_k,
                 bool pack_a, const T* packed_b, T* pack_a_slice)
{
    // The tallest A micro-panel this tile can be packed into, which is what sizes
    // the packed-A panel stride below. Routing is irrelevant here: the
    // B-packed-only MMA reads A through lda and never touches ldd_a/pack_a_slice.
    constexpr int mr_max = mr_max_flt<T>();

    const int ldd_a = static_cast<int>(
        mma_pack_align(mr_max * actual_kc * static_cast<int>(sizeof(T)))
        / static_cast<int>(sizeof(T)));
    const bool epilogue = has_inplace_epilogue(attrs);
    // First k-block with beta == 0 overwrites C outright (the kernels skip
    // the C load); every other block accumulates. beta ∉ {0, 1} still needs
    // the in-place scale pass, beta == 1 is a plain accumulate.
    const bool zero_mode = (k == 0 && attrs.beta == 0.0f);

    for (int m = m_start; m < m_start + m_count; m += mc) {
        int actual_mc = std::min(mc, m_start + m_count - m);
        T* c_tile = c_ptr + m * ldc + n_start;

        if (k == 0 && attrs.beta != 1.0f && attrs.beta != 0.0f) {
            tile_scale(c_tile, ldc, attrs.beta, actual_mc, n_count);
        }

        const auto [cmin, cmax] = kblock_clamp(attrs, last_k);

        if (pack_a) {
            const T* a_src = attrs.transpose_a
                ? a_ptr + k * lda + m    // A phys is K×M, row k, col m
                : a_ptr + m * lda + k;   // A phys is M×K, row m, col k
            tile_pack_lhs(attrs.transpose_a, actual_mc, actual_kc,
                          pack_a_slice, ldd_a, a_src, lda, 1.0f);
            tile_mma_pack(actual_mc, n_count, actual_kc,
                                  c_tile, ldc,
                                  pack_a_slice, packed_b,
                                  cmin, cmax, zero_mode);
        } else {
            const T* a_sub = a_ptr + m * lda + k;  // A phys is M×K, row m, col k
            tile_mma_direct(actual_mc, n_count, actual_kc,
                                    c_tile, ldc,
                                    a_sub, lda, packed_b,
                                    cmin, cmax, zero_mode);
        }

        if (epilogue && last_k) {
            epilogue_inplace(actual_mc, n_count, c_tile, ldc,
                             static_cast<const T*>(nullptr), attrs.epilogue);
        }
    }
}

/// Fused tiled kernel: one (m-range × n-range) tile over all K-blocks. B is
/// always packed; A is packed only when @p pack_a, else read raw.
template <typename T>
void matmul_block_fused(const MatMulAttributes& attrs,
                        T* c_ptr, int ldc,
                        const T* a_ptr, int lda,
                        const void* b_ptr, int ldb,
                        bool b_q51, int64_t b_row_blocks,
                        int K, int kc, int mc,
                        int m_start, int m_count,
                        int n_start, int n_count,
                        bool pack_a, T* pack_b_slice, T* pack_a_slice)
{
    constexpr int nr_max = nr_max_flt<T>();

    for (int k = 0; k < K; k += kc) {
        int actual_kc = std::min(kc, K - k);
        const bool last_k = (k + kc >= K);

        // Uniform 64-byte-aligned panel stride for this k-block.
        const int ldd_b = static_cast<int>(
            mma_pack_align(nr_max * actual_kc * static_cast<int>(sizeof(T)))
            / static_cast<int>(sizeof(T)));

        // Uniform nr_max-wide panels (last one zero-filled) for the mrxncxkc
        // MMA kernels. A Q5_1 B is decoded straight into those panels: the
        // packer takes the logical tile origin in block units, so no element
        // pointer arithmetic is possible (or needed) here.
        if (b_q51) {
            tile_pack_rhs_q5_1(attrs.transpose_b, n_start, n_count, k, actual_kc,
                               pack_b_slice, ldd_b, b_ptr, b_row_blocks);
        } else {
            const T* b_t = static_cast<const T*>(b_ptr);
            const T* b_src = attrs.transpose_b
                ? b_t + n_start * ldb + k     // B phys is N×K, row n, col k
                : b_t + k * ldb + n_start;    // B phys is K×N, row k, col n
            tile_pack_rhs(attrs.transpose_b, n_count, actual_kc,
                          pack_b_slice, ldd_b, b_src, ldb, 1.0f);
        }

        matmul_tile<T>(attrs, c_ptr, ldc, a_ptr, lda,
                       m_start, m_count, mc, n_start, n_count,
                       k, actual_kc, last_k, pack_a, pack_b_slice, pack_a_slice);
    }
}


// =========================================================================
//  Fused Q5_1 small-M route — decode straight into the dot product
// =========================================================================

#if defined(NNOPS_ARCH_X86_64)
/// Largest M routed to the fused Q5_1 kernels, measured at K=N=4096: the dot
/// kernel stays 2x..3.8x ahead to M=16 and loses at 32, where the per-panel
/// re-decode outweighs the packed panel round trip; the outer kernel (2 rows
/// per panel, four accumulators each) wins to M=4.
constexpr int Q5_FUSED_MAX_M_TRANS = 16;
constexpr int Q5_FUSED_MAX_M_COPY  = 4;

/// Two Q5_1 blocks per worker task, which is also the copy kernel's minimum.
constexpr int Q5_FUSED_NTILE = 2 * static_cast<int>(q5_1_block_elems);

/// Call the kernel instantiated for exactly @p MR live rows (rows are never
/// duplicated), recursing down to the runtime row count.
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

/// Run the whole product through the fused Q5_1 kernels: no packed B, no
/// workspace, no k-blocking (each call sweeps K once, so it clamps the final
/// value). Returns false when this route does not apply.
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

    // One flat task list over (batch, n tile) — batch elements are independent
    // and need no per-element fork/join.
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

        // beta ∉ {0, 1} scales the tile in place; beta == 0 skips the C load.
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
//  Batched dispatch — pack + block-granularity parallelism
// =========================================================================

template <typename T>
void matmul_dispatch_2d(const MatMulAttributes& attrs,
                        TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx,
                        const MatMulPlan& plan, void* workspace)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    const auto [M, N, K] = matmul_dims(attrs, a, b);

    // A Q5_1 B has no dense element view: no reference fast path, no raw
    // element arithmetic.
    const bool b_q51 = (b.data_type() == DataType::q5_1);

    // Fast path (f32 only): tiny GEMMs (M*N*K < 1024) skip the tiled/packed
    // machinery and go straight to the reference. f16 has no reference kernel,
    // so it always uses the tiled path below.
    if constexpr (std::is_same_v<T, float>) {
        if (gemm_is_small(M, N, K) && !b_q51) {
            reference::matmul_ref(attrs, output, inputs, ctx, nullptr);
            return;
        }
    }

    const int ldc = static_cast<int>(output.row_stride_elems());
    const int lda = static_cast<int>(a.row_stride_elems());
    // For Q5_1 this is the row pitch in blocks, never an element stride.
    const int ldb = static_cast<int>(b.row_stride_elems());

    T* c_base = output.ptr<T>();
    const T* a_base = a.ptr<T>();
    const void* b_base = b.ptr<void>();
    const T* b_base_t = static_cast<const T*>(b_base);
    const int64_t b_row_blocks = b_q51 ? static_cast<int64_t>(ldb) : 0;

    // A is packed only for transpose_a (correctness) or a wide row stride —
    // decided once in get_matmul_plan, not re-derived here.
    const bool pack_a = plan.pack_a;

    if (workspace == nullptr) {
        reference::matmul_ref(attrs, output, inputs, ctx, nullptr);
        return;
    }

    // Tile sizes come from the plan (single source of truth).
    const int kc = static_cast<int>(plan.kc);
    const int mc = static_cast<int>(plan.mc);
    const int nc = static_cast<int>(plan.nc);

    // Full-Kc packed-B panel stride, shared by both split paths.
    const int ldd_b_full = static_cast<int>(plan.ldd_b);

    // Work is split on the larger dimension (mirrors MLAS sgemm/qgemm: 1D
    // partition over N when N > M, else over M). The workspace always holds
    // the packed B for the full N at one k-block stride, so both paths fit in
    // the same buffer.

    const BatchWalk batch(a, b);

    const int Mi = static_cast<int>(M);
    const int Ni = static_cast<int>(N);
    const int Ki = static_cast<int>(K);

    // Split on the larger dimension (as MLAS does): a block owns an n-block
    // (N-split) or an m-block (M-split), each an (m-range × n-range) tile over
    // packed B. One scratch slot per worker thread when there are fewer threads
    // than blocks, else one per block (see MatMulPlan::num_slots).
    const bool split_n = plan.split_n;
    const int64_t num_blocks = split_n ? split_block_count(Ni, nc)
                                       : split_block_count(Mi, mc);
    const int np_full_b = static_cast<int>(plan.np_slice);
    const bool per_thread = plan.num_slots < num_blocks;

    for (int64_t bi = 0; bi < batch.total; ++bi) {
        int64_t a_offset, b_offset, c_offset;
        batch.offsets(bi, output, a_offset, b_offset, c_offset);

        T* c_p = c_base + c_offset;
        const T* a_p = a_base + a_offset;
        // Batch offsets use the tensor's stride unit: elements for dense types,
        // 24-byte blocks for Q5_1.
        const void* b_p = b_q51
            ? static_cast<const void*>(
                  q5_rhs_at(static_cast<const uint8_t*>(b_base), b_offset))
            : static_cast<const void*>(b_base_t + b_offset);

        ctx.cpu.run(0, num_blocks, [&](int64_t blk) {
            int m_start, m_count, n_start, n_count;
            if (split_n) {
                n_start = static_cast<int>(blk) * nc;
                n_count = std::min(nc, Ni - n_start);
                m_start = 0;
                m_count = Mi;
            } else {
                m_start = static_cast<int>(blk * Mi / num_blocks);
                m_count = static_cast<int>((blk + 1) * Mi / num_blocks) - m_start;
                n_start = 0;
                n_count = Ni;
            }
            const int64_t slot = per_thread
                ? static_cast<int64_t>(ctx.cpu.current_thread_id())
                : blk;
            T* pack_b_slice = static_cast<T*>(workspace) + slot * np_full_b * ldd_b_full;

            // A's slot predicate is separate: the int8 M-split hoists its B
            // pack (num_slots == 1) but every m-block still packs its own A.
            T* pack_a_slice = nullptr;
            if (pack_a) {
                const int64_t aslot = (plan.num_slots_a < num_blocks)
                    ? static_cast<int64_t>(ctx.cpu.current_thread_id())
                    : blk;
                auto* a_base = reinterpret_cast<T*>(
                    static_cast<char*>(workspace) + plan.pack_a_offset);
                pack_a_slice = a_base + aslot * plan.np_a * plan.ldd_a;
            }

            matmul_block_fused<T>(attrs, c_p, ldc, a_p, lda, b_p, ldb,
                                  b_q51, b_row_blocks,
                                  Ki, kc, mc, m_start, m_count, n_start, n_count,
                                  pack_a, pack_b_slice, pack_a_slice);
        });
    }
}

// =========================================================================
//  int8 (s8×s8) — fused NKM, always packs A and B
// =========================================================================

/// Zero-point / scale accessor: PerTensor falls back to the scalar, PerToken /
/// PerChannel index the external buffer at @p idx (the output row m for A, the
/// output column n for the weight B).
inline int32_t quant_zp_at(const QuantParams& q, int64_t idx) noexcept {
    return (q.zero_point_data != nullptr) ? q.zero_point_data[idx] : q.zero_point;
}
inline float quant_scale_at(const QuantParams& q, int64_t idx) noexcept {
    return (q.scale_data != nullptr) ? q.scale_data[idx] : q.scale;
}

/// Raw int8 reductions for the epilogue: r_a[m] = Σ_k A[m,k] (A[k,m] when
/// transposed), r_b[n] = Σ_k B[k,n] (mirror image). Slabs of the output index
/// are disjoint, so the pool needs no atomics, and int32 accumulation stays
/// bit-identical to the reference.

/// r[i] = Σ_j p[j*ld + i]. The output index is contiguous, so the sum sweeps
/// the outer axis; the naive per-output form would stride by the row pitch and
/// touch a fresh cache line per element.
void reduce_cols(const int8_t* p, int64_t ld, int64_t n_out, int64_t n_in,
                 int32_t* r, const ComputeContext& ctx) {
    const int64_t tasks = std::min<int64_t>(n_out,
        std::max<int64_t>(ctx.cpu.thread_count(), 1));
    ctx.cpu.run(0, tasks, [&](int64_t t) {
        const int64_t i0 = t * n_out / tasks;
        const int64_t i1 = (t + 1) * n_out / tasks;
        for (int64_t i = i0; i < i1; ++i) r[i] = 0;
        for (int64_t j = 0; j < n_in; ++j) {
            const int8_t* row = p + j * ld;
            for (int64_t i = i0; i < i1; ++i) r[i] += row[i];
        }
    });
}

/// r[i] = Σ_j p[i*ld + j]. Each output is already a contiguous row sum.
void reduce_rows(const int8_t* p, int64_t ld, int64_t n_out, int64_t n_in,
                 int32_t* r, const ComputeContext& ctx) {
    const int64_t tasks = std::min<int64_t>(n_out,
        std::max<int64_t>(ctx.cpu.thread_count(), 1));
    ctx.cpu.run(0, tasks, [&](int64_t t) {
        const int64_t i0 = t * n_out / tasks;
        const int64_t i1 = (t + 1) * n_out / tasks;
        for (int64_t i = i0; i < i1; ++i) {
            const int8_t* row = p + i * ld;
            int32_t s = 0;
            for (int64_t j = 0; j < n_in; ++j) s += row[j];
            r[i] = s;
        }
    });
}

/// A physical is K×M when transposed (sum down a column), M×K otherwise (row
/// sums); B is the mirror image.
void compute_int8_reductions(const int8_t* a_ptr, int64_t lda,
                             const int8_t* b_ptr, int64_t ldb,
                             const MatMulAttributes& attrs,
                             int64_t M, int64_t N, int64_t K,
                             int32_t* r_a, int32_t* r_b,
                             const ComputeContext& ctx) {
    if (attrs.transpose_a) reduce_cols(a_ptr, lda, M, K, r_a, ctx);
    else                   reduce_rows(a_ptr, lda, M, K, r_a, ctx);

    if (attrs.transpose_b) reduce_rows(b_ptr, ldb, N, K, r_b, ctx);
    else                   reduce_cols(b_ptr, ldb, N, K, r_b, ctx);
}

/// Compute the m-panels of a single k-block against a packed-B slice (s8×s8).
/// Shared by the N-split and M-split int8 paths. @p packed_b holds @p n_count
/// columns for this k-block; @p m_start/@p m_count select the C rows.
/// @p pack_a_buf is the caller's packed-A slot.
void matmul_m_panels_i8(const MatMulAttributes& attrs,
                        int32_t* c_ptr, int ldc,
                        const int8_t* a_ptr, int lda,
                        int m_start, int m_count, int mc,
                        int n_start, int n_count,
                        int k, int actual_kc,
                        const int8_t* packed_b,
                        int8_t* pack_a_buf)
{
    // Pack step writes ceil(kc/4) groups per row; stride by the padded byte
    // count to keep adjacent panels from overlapping.
    const int kbytes = (actual_kc + 3) & ~3;
    const int ldd_a = static_cast<int>(mma_pack_align(MR_MAX_I8 * kbytes));

    for (int m = m_start; m < m_start + m_count; m += mc) {
        int actual_mc = std::min(mc, m_start + m_count - m);
        int32_t* c_tile = c_ptr + m * ldc + n_start;

        const int8_t* a_src = attrs.transpose_a
            ? a_ptr + k * lda + m    // A phys is K×M, row k, col m
            : a_ptr + m * lda + k;   // A phys is M×K, row m, col k
        tile_pack_lhs_i8(attrs.transpose_a, actual_mc, actual_kc,
                         pack_a_buf, ldd_a, a_src, lda);

        tile_mma_pack_i8(actual_mc, n_count, actual_kc,
                         c_tile, ldc,
                         pack_a_buf, packed_b,
                         INT32_MIN, INT32_MAX);
    }
}

/// One n-block of the s8×s8 GEMM: packs B once per k-block, packs A per
/// m-panel into @p pack_a_slice, accumulates the raw int32 dot-product into
/// @p c_ptr.
void matmul_block_int8(const MatMulAttributes& attrs,
                       int32_t* c_ptr, int ldc,
                       const int8_t* a_ptr, int lda,
                       const int8_t* b_ptr, int ldb,
                       int M, int K, int kc, int mc,
                       int n, int actual_nc,
                       int8_t* pack_b_slice,
                       int8_t* pack_a_slice)
{
    for (int k = 0; k < K; k += kc) {
        int actual_kc = std::min(kc, K - k);

        // Uniform 64-byte-aligned panel stride (bytes) for this k-block.
        const int kbytes = (actual_kc + 3) & ~3;
        const int ldd_b = static_cast<int>(mma_pack_align(NR_MAX_I8 * kbytes));

        // B panel: packed ([K][nr]) regardless of transpose_b.
        const int8_t* b_src = attrs.transpose_b
            ? b_ptr + n * ldb + k     // B phys is N×K, row n, col k
            : b_ptr + k * ldb + n;    // B phys is K×N, row k, col n
        tile_pack_rhs_i8(attrs.transpose_b, actual_nc, actual_kc,
                         pack_b_slice, ldd_b, b_src, ldb);

        matmul_m_panels_i8(attrs, c_ptr, ldc, a_ptr, lda,
                           0, M, mc, n, actual_nc, k, actual_kc, pack_b_slice,
                           pack_a_slice);
    }
}

/// Plan-driven int8 dispatch: per-token A × per-channel W, raw int32
/// accumulate, then one epilogue pass (zero-point compensation, plus
/// requantization for an s8 output, whose accumulator sits after the packed-B
/// region — it is a property of the output, not of the layout).
void matmul_dispatch_int8(const MatMulAttributes& attrs,
                          TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx,
                          const MatMulPlan& plan, void* workspace)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    const auto [M, N, K] = matmul_dims(attrs, a, b);

    const int lda = static_cast<int>(a.row_stride_elems());
    const int ldb = static_cast<int>(b.row_stride_elems());
    const int ldc_out = static_cast<int>(output.row_stride_elems());

    const int8_t* a_base = a.ptr<int8_t>();
    const int8_t* b_base = b.ptr<int8_t>();

    const bool out_s8 = (output.data_type() == DataType::s8);
    // int32 accumulator: the output itself for s32, a compact M×N temp for s8.
    const int ldc = out_s8 ? static_cast<int>(N) : ldc_out;

    // Tiles come from the plan (M/N/K are shared across batch elements). Its
    // ldd_b is the full-Kc stride used for *sizing*; each k-block below packs at
    // its own actual_kc stride, which is <= that.
    const int kc = static_cast<int>(plan.kc);
    const int mc = static_cast<int>(plan.mc);
    const int nc = static_cast<int>(plan.nc);

    // Workspace: packed-B panels, then packed A, then the s8 accumulator. Only
    // the N-split indexes B by np_slice; the M-split packs the whole full-N
    // slice at the base, one k-block at a time.
    const int np_full = static_cast<int>(plan.np_slice);
    int8_t* pack_b_base = static_cast<int8_t*>(workspace);
    // int8 always packs A, so the region is always present here.
    int8_t* pack_a_base = pack_b_base + plan.pack_a_offset;
    int32_t* accum = nullptr;
    if (out_s8) {
        accum = reinterpret_cast<int32_t*>(pack_b_base + static_cast<size_t>(plan.workspace_size));
    }

    // Compile-time u8 offset baked into the A pack on the x86 VNNI path.
    constexpr int32_t u8_offset = INT8_USE_U8_OFFSET ? 128 : 0;

    // Quantization parameters (batch-independent, indexed by m / n / 0).
    const QuantParams& qa = a.quant_params();
    const QuantParams& qb = b.quant_params();
    const QuantParams& qc = output.quant_params();
    const double  scale_out = (qc.scale_data != nullptr) ? static_cast<double>(qc.scale_data[0]) : static_cast<double>(qc.scale);
    const int32_t zp_out    = (qc.zero_point_data != nullptr) ? qc.zero_point_data[0] : qc.zero_point;
    const bool relu = (attrs.epilogue.type == EpilogueActivateType::Relu);

    const BatchWalk batch(a, b);

    const int Mi = static_cast<int>(M);
    const int Ni = static_cast<int>(N);
    const int Ki = static_cast<int>(K);

    std::vector<int32_t> r_a(static_cast<size_t>(M));
    std::vector<int32_t> r_b(static_cast<size_t>(N));

    for (int64_t bi = 0; bi < batch.total; ++bi) {
        int64_t a_offset, b_offset, c_offset;
        batch.offsets(bi, output, a_offset, b_offset, c_offset);

        const int8_t* a_p = a_base + a_offset;
        const int8_t* b_p = b_base + b_offset;

        // Zero the int32 accumulator: the MMA accumulates, and the integer path
        // has overwrite semantics (alpha/beta ignored).
        int32_t* c_p;
        if (out_s8) {
            c_p = accum;  // reused per batch element
            std::fill(accum, accum + M * N, 0);
        } else {
            c_p = output.ptr<int32_t>() + c_offset;
            for (int64_t m = 0; m < M; ++m) {
                std::fill(c_p + m * ldc, c_p + m * ldc + N, 0);
            }
        }

        compute_int8_reductions(a_p, lda, b_p, ldb, attrs, M, N, K,
                                r_a.data(), r_b.data(), ctx);

        if (plan.split_n) {
            // N-split: one n-block per parallel task (N is the larger dim).
            const int64_t num_blocks = (Ni + nc - 1) / nc;
            auto run_block = [&](int64_t blk) {
                const int n = static_cast<int>(blk) * nc;
                const int actual_nc = std::min(nc, Ni - n);
                // A slice is one worst-case tile wide (np_full panels) at the
                // plan's sizing stride, which is >= every per-k-block stride
                // packed into it below.
                int8_t* pack_b_slice = pack_b_base + blk * np_full * plan.ldd_b;
                const int64_t a_slot = (plan.num_slots_a < num_blocks)
                    ? static_cast<int64_t>(ctx.cpu.current_thread_id())
                    : blk;
                int8_t* pack_a_slice = pack_a_base + a_slot * plan.np_a * plan.ldd_a;
                matmul_block_int8(attrs, c_p, ldc, a_p, lda, b_p, ldb,
                                  Mi, Ki, kc, mc, n, actual_nc, pack_b_slice,
                                  pack_a_slice);
            };

            ctx.cpu.run(0, num_blocks, run_block);
        } else {
            // M-split: pack B once per k-block (full N), then split the M rows
            // into equal contiguous slabs.
            const int64_t num_m_blocks = split_block_count(Mi, mc);

            for (int k = 0; k < Ki; k += kc) {
                const int actual_kc = std::min(kc, Ki - k);
                const int kbytes = (actual_kc + 3) & ~3;
                const int ldd_b = static_cast<int>(mma_pack_align(NR_MAX_I8 * kbytes));
                const int8_t* b_src = attrs.transpose_b
                    ? b_p + k                  // B phys is N×K, col k over full N
                    : b_p + k * ldb;           // B phys is K×N, row k over full N
                tile_pack_rhs_i8(attrs.transpose_b, Ni, actual_kc,
                                 pack_b_base, ldd_b, b_src, ldb);

                auto run_m = [&](int64_t blk) {
                    const int m_start = static_cast<int>(blk * Mi / num_m_blocks);
                    const int m_end   = static_cast<int>((blk + 1) * Mi / num_m_blocks);
                    // B is shared here; every m-block packs its own A, so
                    // num_slots_a (not num_slots) decides the slot.
                    const int64_t a_slot = (plan.num_slots_a < num_m_blocks)
                        ? static_cast<int64_t>(ctx.cpu.current_thread_id())
                        : blk;
                    int8_t* pack_a_slice = pack_a_base + a_slot * plan.np_a * plan.ldd_a;
                    matmul_m_panels_i8(attrs, c_p, ldc, a_p, lda,
                                       m_start, m_end - m_start, mc,
                                       0, Ni, k, actual_kc, pack_b_base,
                                       pack_a_slice);
                };
                ctx.cpu.run(0, num_m_blocks, run_m);
            }
        }

        // Epilogue: zero-point compensation (+ requantization for s8). Rows are
        // independent (own C row, r_a[m], read-only r_b), so they split into
        // slabs without reductions or atomics.
        const int64_t ep_rows = std::max<int64_t>(ctx.cpu.thread_count(), 1);
        const int64_t ep_tasks = std::min<int64_t>(M, ep_rows);
        int8_t* out_p = out_s8 ? output.ptr<int8_t>() + c_offset : nullptr;
        ctx.cpu.run(0, ep_tasks, [&](int64_t t) {
            const int64_t m0 = t * M / ep_tasks;
            const int64_t m1 = (t + 1) * M / ep_tasks;
            for (int64_t m = m0; m < m1; ++m) {
                const int32_t zp_a = quant_zp_at(qa, m);
                const double  s_a  = out_s8 ? static_cast<double>(quant_scale_at(qa, m)) : 0.0;
                for (int64_t n = 0; n < N; ++n) {
                    int32_t v = matmul_int8_compensate(c_p[m * ldc + n], u8_offset,
                                                       zp_a, quant_zp_at(qb, n),
                                                       r_a[static_cast<size_t>(m)],
                                                       r_b[static_cast<size_t>(n)],
                                                       Ki);
                    if (relu) {
                        v = std::max(v, 0);
                    }
                    if (out_s8) {
                        const double req = s_a * static_cast<double>(quant_scale_at(qb, n)) / scale_out;
                        out_p[m * ldc_out + n] = matmul_int8_requant(v, req, zp_out);
                    } else {
                        c_p[m * ldc + n] = v;
                    }
                }
            }
        });
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

    // Fold the pure-clamp activations into the epilogue's explicit clip bounds
    // so the tiled kernel's kblock_clamp reads min_clip/max_clip directly.
    MatMulAttributes attrs_n = attrs;
    switch (attrs_n.epilogue.type) {
    case EpilogueActivateType::Relu:  attrs_n.epilogue.min_clip = 0.0f; break;
    case EpilogueActivateType::Relu6: attrs_n.epilogue.min_clip = 0.0f; attrs_n.epilogue.max_clip = 6.0f; break;
    default: break;
    }

#if defined(NNOPS_ARCH_X86_64)
    // Small-M Q5_1: decode straight into the reduction, before the plan and the
    // pooled scratch so the route pays for neither. Only a transposed A is out
    // of reach (the kernels index A[m][k] through any row stride).
    if (dt_a == DataType::f32 && dt_b == DataType::q5_1 && !attrs_n.transpose_a) {
        const auto [M, N, K] = matmul_dims(attrs_n, a, b);
        if (matmul_q5_fused(attrs_n, output, inputs, ctx,
                            static_cast<int>(M), static_cast<int>(N),
                            static_cast<int>(K))) {
            return;
        }
    }
#endif

    // Worker ids available → the packed-B scratch is one slot per worker thread
    // instead of one per m-block (see MatMulPlan::num_slots).
    auto plan = get_matmul_plan(attrs_n, a.desc(), b.desc(), ctx.cpu.thread_count(),
                                ctx.cpu.thread_slot_count() > 0);

    // int8 (s8×s8): fused tiled kernel for s32 / s8 outputs. The tiled path
    // always packs B into pooled scratch; an s8 output additionally needs an
    // int32 M×N accumulator, which is a property of the output rather than of
    // the packed-B layout, so it is sized here from the output's last two dims
    // (the output shape is [..., M, N]) and appended after the plan's scratch.
    if (dt_a == DataType::s8 && dt_b == DataType::s8) {
        const bool out_ok = (output.data_type() == DataType::s32) ||
                            (output.data_type() == DataType::s8);
        if (out_ok) {
            NNOPS_ASSERT(plan.mc > 0 && plan.nc > 0 && plan.kc > 0);
            size_t ws = static_cast<size_t>(plan.workspace_size);
            if (output.data_type() == DataType::s8) {
                const auto od = output.desc();
                ws += static_cast<size_t>(od.dims[static_cast<size_t>(od.rank - 2)])
                    * static_cast<size_t>(od.dims[static_cast<size_t>(od.rank - 1)])
                    * sizeof(int32_t);
            }
            PoolPtr scratch(ws);
            matmul_dispatch_int8(attrs_n, output, inputs, ctx, plan, scratch.get());
        } else {
            reference::matmul_int8_ref(attrs_n, output, inputs, ctx, nullptr);
        }
        return;
    }

    // Past the int8 branch: only f32/f16 have a fused kernel, plus a Q5_1 B
    // (its blocks decode to A's type in the RHS pack).
    const bool b_q51 = (dt_b == DataType::q5_1);
    const bool float_a = dt_a == DataType::f32 || dt_a == DataType::f16;
    const bool supported = float_a && (dt_b == dt_a || b_q51);
    if (!supported) {
        reference::matmul_ref(attrs_n, output, inputs, ctx, nullptr);
        return;
    }

    PoolPtr scratch(plan.workspace_size);

    if (dt_a == DataType::f32) {
        matmul_dispatch_2d<float>(attrs_n, output, inputs, ctx, plan, scratch.get());
    } else {
        matmul_dispatch_2d<half>(attrs_n, output, inputs, ctx, plan, scratch.get());
    }
}

}  // namespace nnops::backend::cpu
