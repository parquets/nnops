/// @file matmul.cpp
/// @brief Tiled matrix multiplication kernel — pack + MMA dispatch.
///
/// B (rhs) is always packed into pooled scratch; A (lhs) is packed on the
/// per-thread stack only when transpose_a forces it or its row stride is
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

// =========================================================================
//  Tile-block kernels — one call processes one (m-range × n-range) tile
// =========================================================================

/// Accumulate one k-block into an m×n tile of C: C += A × packed_B. Shared by
/// the N-split (m over full M) and M-split (n over full N) paths so both use
/// identical pack/MMA/epilogue logic. @p packed_b holds @p n_count columns for
/// this k-block (its panel stride is derived internally from @p actual_kc);
/// @p m_start/@p m_count and @p n_start/@p n_count select the C tile.
template <typename T>
void matmul_tile(const MatMulAttributes& attrs,
                 T* c_ptr, int ldc,
                 const T* a_ptr, int lda,
                 int m_start, int m_count, int mc,
                 int n_start, int n_count,
                 int k, int actual_kc, bool last_k,
                 bool pack_a, const T* packed_b)
{
    // The tallest A micro-panel this tile can be packed into, which is what sizes
    // the packed-A panel stride below. Routing is irrelevant here: tile_mma_direct
    // reads A through lda and never touches ldd_a/pack_a_buf.
    constexpr int mr_max = mr_max_flt<T>();

    // Packed A lives on the kernel stack (84 KB f32 / 42 KB f16 on aarch64,
    // 64-byte aligned). Sized by num_panels_max, not num_panels(MC_TARGET, ..):
    // this tile's mc is a per-block height <= MC_TARGET whose greedy
    // decomposition can need more panels than MC_TARGET's (see num_panels_max).
    alignas(PANEL_ALIGN_BYTES) T pack_a_buf[pack_a_stack_elems<T>()];

    const int ldd_a = align_up<PANEL_ALIGN_BYTES>(mr_max * actual_kc * static_cast<int>(sizeof(T)))
                      / static_cast<int>(sizeof(T));
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
                          pack_a_buf, ldd_a, a_src, lda, 1.0f);
            tile_mma_pack(actual_mc, n_count, actual_kc,
                          c_tile, ldc,
                          pack_a_buf, packed_b, -1,
                          cmin, cmax, zero_mode);
        } else {
            const T* a_sub = a_ptr + m * lda + k;  // A phys is M×K, row m, col k
            tile_mma_direct(actual_mc, n_count, actual_kc,
                            c_tile, ldc,
                            a_sub, lda, packed_b, -1,
                            cmin, cmax, zero_mode);
        }

        if (epilogue && last_k) {
            epilogue_inplace(actual_mc, n_count, c_tile, ldc,
                             static_cast<const T*>(nullptr), attrs.epilogue);
        }
    }
}

/// Fused tiled kernel: B is always packed into @p pack_b_slice; A is packed on
/// the stack only when @p pack_a, else read raw. Processes one (m-range ×
/// n-range) tile over all K-blocks. Shared by the N-split path (m over full M,
/// n = one n-block) and the M-split path (m = one m-block, n over full N).
template <typename T>
void matmul_block_fused(const MatMulAttributes& attrs,
                        T* c_ptr, int ldc,
                        const T* a_ptr, int lda,
                        const T* b_ptr, int ldb,
                        int K, int kc, int mc,
                        int m_start, int m_count,
                        int n_start, int n_count,
                        bool pack_a, T* pack_b_slice)
{
    constexpr int nr_max = nr_max_flt<T>();

    for (int k = 0; k < K; k += kc) {
        int actual_kc = std::min(kc, K - k);
        const bool last_k = (k + kc >= K);

        // Uniform 64-byte-aligned panel stride for this k-block.
        const int ldd_b = align_up<PANEL_ALIGN_BYTES>(nr_max * actual_kc * static_cast<int>(sizeof(T)))
                          / static_cast<int>(sizeof(T));

        // B panel: packed ([K][nr]) regardless of transpose_b.
        const T* b_src = attrs.transpose_b
            ? b_ptr + n_start * ldb + k     // B phys is N×K, row n, col k
            : b_ptr + k * ldb + n_start;    // B phys is K×N, row k, col n
        tile_pack_rhs(attrs.transpose_b, n_count, actual_kc,
                      pack_b_slice, ldd_b, b_src, ldb, 1.0f);

        matmul_tile<T>(attrs, c_ptr, ldc, a_ptr, lda,
                       m_start, m_count, mc, n_start, n_count,
                       k, actual_kc, last_k, pack_a, pack_b_slice);
    }
}


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

    const int64_t a_rank = a.rank();
    const int64_t b_rank = b.rank();
    const int64_t M  = attrs.transpose_a ? a.shape(a_rank - 1) : a.shape(a_rank - 2);
    const int64_t N  = attrs.transpose_b ? b.shape(b_rank - 2) : b.shape(b_rank - 1);
    const int64_t K  = attrs.transpose_b ? b.shape(b_rank - 1) : b.shape(b_rank - 2);

    // Fast path (f32 only): tiny GEMMs (M*N*K < 1024) skip the tiled/packed
    // machinery and go straight to the reference. f16 has no reference kernel,
    // so it always uses the tiled path below.
    if constexpr (std::is_same_v<T, float>) {
        if (gemm_is_small(M, N, K)) {
            reference::matmul_ref(attrs, output, inputs, ctx, nullptr);
            return;
        }
    }

    const int ldc = static_cast<int>(output.row_stride_elems());
    const int lda = static_cast<int>(a.row_stride_elems());
    const int ldb = static_cast<int>(b.row_stride_elems());

    T* c_base = output.ptr<T>();
    const T* a_base = a.ptr<T>();
    const T* b_base = b.ptr<T>();

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

    // Broadcast batch dimensions (same logic as matmul_ref).
    const int64_t batch_a_dims = a_rank - 2;
    const int64_t batch_b_dims = b_rank - 2;
    const int64_t batch_ndim   = std::max(batch_a_dims, batch_b_dims);

    std::vector<int64_t> batch_a_shape(batch_ndim, 1);
    std::vector<int64_t> batch_b_shape(batch_ndim, 1);
    std::vector<int64_t> batch_out_shape(batch_ndim, 1);

    for (int64_t i = 0; i < batch_a_dims; ++i) {
        batch_a_shape[batch_ndim - batch_a_dims + i] = a.shape(i);
    }
    for (int64_t i = 0; i < batch_b_dims; ++i) {
        batch_b_shape[batch_ndim - batch_b_dims + i] = b.shape(i);
    }

    int64_t total_batch = 1;
    for (int64_t i = 0; i < batch_ndim; ++i) {
        const int64_t da = batch_a_shape[i];
        const int64_t db = batch_b_shape[i];
        if (da == db) {
            batch_out_shape[i] = da;
        } else if (da == 1) {
            batch_out_shape[i] = db;
        } else if (db == 1) {
            batch_out_shape[i] = da;
        } else {
            NNOPS_ASSERT(!"MatMul: incompatible batch dimensions for broadcast");
        }
        total_batch *= batch_out_shape[i];
    }

    const int Mi = static_cast<int>(M);
    const int Ni = static_cast<int>(N);
    const int Ki = static_cast<int>(K);

    for (int64_t bi = 0; bi < total_batch; ++bi) {
        // Unflatten batch index → multi-dimensional coords.
        int64_t rem = bi;
        int64_t a_offset = 0;
        int64_t b_offset = 0;
        int64_t c_offset = 0;

        for (int64_t d = batch_ndim - 1; d >= 0; --d) {
            const int64_t coord = rem % batch_out_shape[d];
            rem /= batch_out_shape[d];

            const int64_t a_dim = d - (batch_ndim - batch_a_dims);
            if (a_dim >= 0) {
                const int64_t a_coord = (batch_a_shape[d] == 1) ? 0 : coord;
                a_offset += a_coord * a.stride_elems(a_dim);
            }

            const int64_t b_dim = d - (batch_ndim - batch_b_dims);
            if (b_dim >= 0) {
                const int64_t b_coord = (batch_b_shape[d] == 1) ? 0 : coord;
                b_offset += b_coord * b.stride_elems(b_dim);
            }

            c_offset += coord * output.stride_elems(d);
        }

        T* c_p = c_base + c_offset;
        const T* a_p = a_base + a_offset;
        const T* b_p = b_base + b_offset;

        // One block either owns an n-block (N-split) or an m-block (M-split); each
        // reduces to an (m-range × n-range) tile over packed B.
        const bool split_n = plan.split_n;
        const int64_t num_blocks = split_n
            ? (Ni + nc - 1) / nc
            : split_block_count(Mi, mc);
        const int np_full_b = num_panels(split_n ? nc : Ni,
                                         std::is_same_v<T, float> ? NR_F32 : NR_F16);

        // Scratch slots: the plan allocates one packed-B slice per worker thread
        // when the backend reports ids and that is fewer than the block count
        // (M-split only — N-split's blocks tile a single full-N slice), else one
        // per block. A thread packs one slice at a time and blocks are
        // independent, so reusing a slot across every block a thread claims is
        // safe and the result is unchanged.
        const bool per_thread = plan.num_slots < num_blocks;

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
            matmul_block_fused<T>(attrs, c_p, ldc, a_p, lda, b_p, ldb,
                                  Ki, kc, mc, m_start, m_count, n_start, n_count,
                                  pack_a, pack_b_slice);
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

/// Raw int8 row/column reductions needed by the epilogue:
///   r_a[m] = Σ_k A[m,k] (or A[k,m] if transpose_a)
///   r_b[n] = Σ_k B[k,n] (or B[n,k] if transpose_b)
///
/// When the sum runs *down a column* of the physical layout (transpose_a, or a
/// non-transposed B) the naive per-output form strides by the row pitch, so
/// every element is a fresh cache line. Sweeping along the row and accumulating
/// instead — the transposed layout's axis order — touches the same bytes but
/// contiguously. The sum is unchanged: int32 accumulates of int8 products
/// cannot overflow for any K this kernel accepts, and integer addition is
/// associative, so the result stays bit-identical to the reference.
///
/// The output index is split into contiguous slabs so the whole thing runs on
/// the pool; slabs are disjoint, so no atomics or per-thread partials are
/// needed.
void compute_int8_reductions(const int8_t* a_ptr, int64_t lda,
                             const int8_t* b_ptr, int64_t ldb,
                             const MatMulAttributes& attrs,
                             int64_t M, int64_t N, int64_t K,
                             int32_t* r_a, int32_t* r_b,
                             const ComputeContext& ctx) {
    const int64_t nt = std::max<int64_t>(ctx.cpu.thread_count(), 1);

    // ---- r_a: one output per A row ----
    if (attrs.transpose_a) {
        // A physical is K×M: the sum runs down a column, so sweep k and
        // accumulate across a contiguous slab of m.
        const int64_t tasks = std::min<int64_t>(M, nt);
        ctx.cpu.run(0, tasks, [&](int64_t t) {
            const int64_t m0 = t * M / tasks;
            const int64_t m1 = (t + 1) * M / tasks;
            for (int64_t m = m0; m < m1; ++m) r_a[m] = 0;
            for (int64_t k = 0; k < K; ++k) {
                const int8_t* row = a_ptr + k * lda;
                for (int64_t m = m0; m < m1; ++m) r_a[m] += row[m];
            }
        });
    } else {
        // A physical is M×K: each row is already contiguous.
        const int64_t tasks = std::min<int64_t>(M, nt);
        ctx.cpu.run(0, tasks, [&](int64_t t) {
            const int64_t m0 = t * M / tasks;
            const int64_t m1 = (t + 1) * M / tasks;
            for (int64_t m = m0; m < m1; ++m) {
                const int8_t* row = a_ptr + m * lda;
                int32_t s = 0;
                for (int64_t k = 0; k < K; ++k) s += row[k];
                r_a[m] = s;
            }
        });
    }

    // ---- r_b: one output per B column ----
    if (attrs.transpose_b) {
        // B physical is N×K: each row is already contiguous.
        const int64_t tasks = std::min<int64_t>(N, nt);
        ctx.cpu.run(0, tasks, [&](int64_t t) {
            const int64_t n0 = t * N / tasks;
            const int64_t n1 = (t + 1) * N / tasks;
            for (int64_t n = n0; n < n1; ++n) {
                const int8_t* row = b_ptr + n * ldb;
                int32_t s = 0;
                for (int64_t k = 0; k < K; ++k) s += row[k];
                r_b[n] = s;
            }
        });
    } else {
        // B physical is K×N: the sum runs down a column, so sweep k and
        // accumulate across a contiguous slab of n.
        const int64_t tasks = std::min<int64_t>(N, nt);
        ctx.cpu.run(0, tasks, [&](int64_t t) {
            const int64_t n0 = t * N / tasks;
            const int64_t n1 = (t + 1) * N / tasks;
            for (int64_t n = n0; n < n1; ++n) r_b[n] = 0;
            for (int64_t k = 0; k < K; ++k) {
                const int8_t* row = b_ptr + k * ldb;
                for (int64_t n = n0; n < n1; ++n) r_b[n] += row[n];
            }
        });
    }
}

/// Compute the m-panels of a single k-block against a packed-B slice (s8×s8).
/// Shared by the N-split and M-split int8 paths. @p packed_b holds @p n_count
/// columns for this k-block; @p m_start/@p m_count select the C rows.
void matmul_m_panels_i8(const MatMulAttributes& attrs,
                        int32_t* c_ptr, int ldc,
                        const int8_t* a_ptr, int lda,
                        int m_start, int m_count, int mc,
                        int n_start, int n_count,
                        int k, int actual_kc,
                        const int8_t* packed_b)
{
    alignas(PANEL_ALIGN_BYTES) int8_t pack_a_buf[PACK_A_STACK_I8];

    // Pack step writes ceil(kc/4) groups per row; stride by the padded byte
    // count to keep adjacent panels from overlapping.
    const int kbytes = (actual_kc + 3) & ~3;
    const int ldd_a = align_up<PANEL_ALIGN_BYTES>(MR_MAX_I8 * kbytes);

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
/// m-panel on the stack, accumulates the raw int32 dot-product into @p c_ptr.
void matmul_block_int8(const MatMulAttributes& attrs,
                       int32_t* c_ptr, int ldc,
                       const int8_t* a_ptr, int lda,
                       const int8_t* b_ptr, int ldb,
                       int M, int K, int kc, int mc,
                       int n, int actual_nc,
                       int8_t* pack_b_slice)
{
    for (int k = 0; k < K; k += kc) {
        int actual_kc = std::min(kc, K - k);

        // Uniform 64-byte-aligned panel stride (bytes) for this k-block.
        const int kbytes = (actual_kc + 3) & ~3;
        const int ldd_b = align_up<PANEL_ALIGN_BYTES>(NR_MAX_I8 * kbytes);

        // B panel: packed ([K][nr]) regardless of transpose_b.
        const int8_t* b_src = attrs.transpose_b
            ? b_ptr + n * ldb + k     // B phys is N×K, row n, col k
            : b_ptr + k * ldb + n;    // B phys is K×N, row k, col n
        tile_pack_rhs_i8(attrs.transpose_b, actual_nc, actual_kc,
                         pack_b_slice, ldd_b, b_src, ldb);

        matmul_m_panels_i8(attrs, c_ptr, ldc, a_ptr, lda,
                           0, M, mc, n, actual_nc, k, actual_kc, pack_b_slice);
    }
}

/// Plan-driven int8 dispatch: per-token A × per-channel W, raw int32 accumulate,
/// then a single epilogue pass that applies zero-point compensation (and, for an
/// s8 output, requantization). Packed B lives in the plan's workspace; for an s8
/// output the int32 accumulator sits after it (sized by matmul_cpu — it is a
/// property of the output, not of the packed-B layout).
void matmul_dispatch_int8(const MatMulAttributes& attrs,
                          TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx,
                          const MatMulPlan& plan, void* workspace)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    const int64_t a_rank = a.rank();
    const int64_t b_rank = b.rank();
    const int64_t M  = attrs.transpose_a ? a.shape(a_rank - 1) : a.shape(a_rank - 2);
    const int64_t N  = attrs.transpose_b ? b.shape(b_rank - 2) : b.shape(b_rank - 1);
    const int64_t K  = attrs.transpose_b ? b.shape(b_rank - 1) : b.shape(b_rank - 2);

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

    // Workspace: packed-B panels at the base, then the s8 accumulator.
    const int np_full = num_panels4(nc, NR_I8);
    int8_t* pack_b_base = static_cast<int8_t*>(workspace);
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

    // ---- broadcast batch dims (same logic as matmul_dispatch_2d) ----
    const int64_t batch_a_dims = a_rank - 2;
    const int64_t batch_b_dims = b_rank - 2;
    const int64_t batch_ndim   = std::max(batch_a_dims, batch_b_dims);

    std::vector<int64_t> batch_a_shape(batch_ndim, 1);
    std::vector<int64_t> batch_b_shape(batch_ndim, 1);
    std::vector<int64_t> batch_out_shape(batch_ndim, 1);

    for (int64_t i = 0; i < batch_a_dims; ++i) {
        batch_a_shape[batch_ndim - batch_a_dims + i] = a.shape(i);
    }
    for (int64_t i = 0; i < batch_b_dims; ++i) {
        batch_b_shape[batch_ndim - batch_b_dims + i] = b.shape(i);
    }

    int64_t total_batch = 1;
    for (int64_t i = 0; i < batch_ndim; ++i) {
        const int64_t da = batch_a_shape[i];
        const int64_t db = batch_b_shape[i];
        if (da == db) {
            batch_out_shape[i] = da;
        } else if (da == 1) {
            batch_out_shape[i] = db;
        } else if (db == 1) {
            batch_out_shape[i] = da;
        } else {
            NNOPS_ASSERT(!"MatMul: incompatible batch dimensions for broadcast");
        }
        total_batch *= batch_out_shape[i];
    }

    const int Mi = static_cast<int>(M);
    const int Ni = static_cast<int>(N);
    const int Ki = static_cast<int>(K);

    std::vector<int32_t> r_a(static_cast<size_t>(M));
    std::vector<int32_t> r_b(static_cast<size_t>(N));

    for (int64_t bi = 0; bi < total_batch; ++bi) {
        int64_t rem = bi;
        int64_t a_offset = 0;
        int64_t b_offset = 0;
        int64_t c_offset = 0;

        for (int64_t d = batch_ndim - 1; d >= 0; --d) {
            const int64_t coord = rem % batch_out_shape[d];
            rem /= batch_out_shape[d];

            const int64_t a_dim = d - (batch_ndim - batch_a_dims);
            if (a_dim >= 0) {
                const int64_t a_coord = (batch_a_shape[d] == 1) ? 0 : coord;
                a_offset += a_coord * a.stride_elems(a_dim);
            }

            const int64_t b_dim = d - (batch_ndim - batch_b_dims);
            if (b_dim >= 0) {
                const int64_t b_coord = (batch_b_shape[d] == 1) ? 0 : coord;
                b_offset += b_coord * b.stride_elems(b_dim);
            }

            c_offset += coord * output.stride_elems(d);
        }

        const int8_t* a_p = a_base + a_offset;
        const int8_t* b_p = b_base + b_offset;

        // int32 accumulator for this batch element. The MMA micro-kernels
        // accumulate (C += A·B), and alpha/beta are ignored on the integer
        // path (C = A·B, overwrite semantics), so zero the tile first.
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
                // Slices are laid out at the plan's sizing stride (full Kc, or K
                // when K < Kc), which is >= the per-k-block stride the block
                // below packs at — that is what makes the offset an upper bound.
                int8_t* pack_b_slice = pack_b_base + blk * np_full * plan.ldd_b;
                matmul_block_int8(attrs, c_p, ldc, a_p, lda, b_p, ldb,
                                  Mi, Ki, kc, mc, n, actual_nc, pack_b_slice);
            };

            ctx.cpu.run(0, num_blocks, run_block);
        } else {
            // M-split: pack B once per k-block (full N), then parallelize the M
            // rows into equal contiguous slabs. Mirrors MLAS's "M >= N →
            // partition M" rule; the block count is ceil(M/mc).
            const int64_t num_m_blocks = split_block_count(Mi, mc);

            for (int k = 0; k < Ki; k += kc) {
                const int actual_kc = std::min(kc, Ki - k);
                const int kbytes = (actual_kc + 3) & ~3;
                const int ldd_b = align_up<PANEL_ALIGN_BYTES>(NR_MAX_I8 * kbytes);
                const int8_t* b_src = attrs.transpose_b
                    ? b_p + k                  // B phys is N×K, col k over full N
                    : b_p + k * ldb;           // B phys is K×N, row k over full N
                tile_pack_rhs_i8(attrs.transpose_b, Ni, actual_kc,
                                 pack_b_base, ldd_b, b_src, ldb);

                auto run_m = [&](int64_t blk) {
                    const int m_start = static_cast<int>(blk * Mi / num_m_blocks);
                    const int m_end   = static_cast<int>((blk + 1) * Mi / num_m_blocks);
                    matmul_m_panels_i8(attrs, c_p, ldc, a_p, lda,
                                       m_start, m_end - m_start, mc,
                                       0, Ni, k, actual_kc, pack_b_base);
                };
                ctx.cpu.run(0, num_m_blocks, run_m);
            }
        }

        // Epilogue: zero-point compensation (+ requantization for s8). Each
        // output row depends only on its own C row, r_a[m] and the (read-only)
        // r_b, so rows are independent and can be split into slabs across the
        // pool — no reduction, no atomics. The per-row terms (zp_a, s_a) are
        // hoisted to the slab's row loop exactly as before.
        const int64_t ep_rows = std::max<int64_t>(ctx.cpu.thread_count(), 1);
        const int64_t ep_tasks = std::min<int64_t>(M, ep_rows);
        if (out_s8) {
            int8_t* out_p = output.ptr<int8_t>() + c_offset;
            ctx.cpu.run(0, ep_tasks, [&](int64_t t) {
                const int64_t m0 = t * M / ep_tasks;
                const int64_t m1 = (t + 1) * M / ep_tasks;
                for (int64_t m = m0; m < m1; ++m) {
                    const int32_t zp_a = quant_zp_at(qa, m);
                    const double  s_a  = static_cast<double>(quant_scale_at(qa, m));
                    for (int64_t n = 0; n < N; ++n) {
                        int32_t v = matmul_int8_compensate(c_p[m * ldc + n], u8_offset,
                                                           zp_a, quant_zp_at(qb, n),
                                                           r_a[static_cast<size_t>(m)],
                                                           r_b[static_cast<size_t>(n)],
                                                           Ki);
                        if (relu) {
                            v = std::max(v, 0);
                        }
                        const double req = s_a * static_cast<double>(quant_scale_at(qb, n)) / scale_out;
                        out_p[m * ldc_out + n] = matmul_int8_requant(v, req, zp_out);
                    }
                }
            });
        } else {
            ctx.cpu.run(0, ep_tasks, [&](int64_t t) {
                const int64_t m0 = t * M / ep_tasks;
                const int64_t m1 = (t + 1) * M / ep_tasks;
                for (int64_t m = m0; m < m1; ++m) {
                    const int32_t zp_a = quant_zp_at(qa, m);
                    for (int64_t n = 0; n < N; ++n) {
                        int32_t v = matmul_int8_compensate(c_p[m * ldc + n], u8_offset,
                                                           zp_a, quant_zp_at(qb, n),
                                                           r_a[static_cast<size_t>(m)],
                                                           r_b[static_cast<size_t>(n)],
                                                           Ki);
                        if (relu) {
                            v = std::max(v, 0);
                        }
                        c_p[m * ldc + n] = v;
                    }
                }
            });
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

    // Fold the pure-clamp activations into the epilogue's explicit clip bounds
    // so the tiled kernel's kblock_clamp reads min_clip/max_clip directly.
    MatMulAttributes attrs_n = attrs;
    switch (attrs_n.epilogue.type) {
    case EpilogueActivateType::Relu:  attrs_n.epilogue.min_clip = 0.0f; break;
    case EpilogueActivateType::Relu6: attrs_n.epilogue.min_clip = 0.0f; attrs_n.epilogue.max_clip = 6.0f; break;
    default: break;
    }

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

    // Past the int8 branch: only f32 and f16 have a fused kernel, the rest use
    // the reference.
    const bool supported = (dt_a == DataType::f32 && dt_b == DataType::f32) ||
                           (dt_a == DataType::f16 && dt_b == DataType::f16);
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
