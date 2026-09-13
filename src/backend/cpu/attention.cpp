/// @file attention.cpp
/// @brief Tiled multi-head scaled dot-product attention — fused QK^T + softmax + AV.
///
/// For each (batch, head) — parallel via ctx.cpu.run (serial fallback):
///   scores = Q @ K^T * scale   (transpose-B GEMM: K^T packed into the workspace)
///   scores += mask             (optional, flat [Sq, Sk] as in the reference)
///   attn   = softmax(scores)   (simd_softmax, per contiguous row, in-place)
///   output = attn @ V          (direct GEMM, V raw K×N)
///
/// Both GEMMs reuse the tile_mma_direct / tile_pack_rhs micro-kernel dispatch
/// from matmul_helper.h. A Q/K/V/O head slice is an M×K / N×K / K×N / M×N
/// submatrix of the planar tensor, for both merged [B, S, H*D] and explicit
/// [B, H, S, D] layouts (a head's D columns are contiguous within each row).
///
/// The scale factor is fused into the GEMM1 RHS pack, so scores accumulate
/// Q·K^T·scale directly. The GEMM kernels accumulate C += A×B, so the scores
/// buffer is zeroed before GEMM1 and the output head is zeroed before GEMM2
/// (attention has no beta / add_to; the reference overwrites the output).
///
/// When the per-head scores matrix is large (Sq × Sk ≥ kFlashMinScores), the
/// tiled online-softmax FlashAttention path (attention_flash_impl) is used
/// instead, streaming KV in Br×Bc tiles so the full scores matrix is never
/// materialized (MLAS FlashAttention-1 algorithm, adapted to these kernels).
/// The flash path parallelizes over batch × head × query-block (B × H × NQ),
/// mirroring MLAS's q_chunk_count task decomposition, so long-sequence prefill
/// still exposes ample parallelism even when B × H is small.
///
/// Grouped-query attention (attrs.num_group) is not yet handled — this runs
/// standard MHA (one head block per (batch, head)), matching the reference.

#include "attention.h"
#include "matmul_helper.h"                  // tile_mma_direct / tile_pack_rhs + panel constants
#include "simd_kernel/simd_softmax.hpp"     // softmax + FlashAttention row kernels
#include "common/memory_pool.hpp"           // internal scratch-memory pool
#include "nnops/detail/assert.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace nnops::backend::cpu {

// Forward-declare the reference kernel for the fallback paths.
namespace reference {
void attention_ref(const AttentionAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx,
                   void* workspace);
}

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr float kPosInf =  std::numeric_limits<float>::infinity();

// ---- head-shape extraction (merged [B,S,H*D] vs explicit [B,H,S,D]) ----

struct HeadShape {
    int64_t B, H, Sq, Sk, Sv, D;
    bool merged;
};

inline HeadShape resolve_head_shape(const AttentionAttributes& attrs,
                                    std::span<const TensorView> inputs)
{
    const auto& Q = inputs[0];
    const auto& K = inputs[1];
    const auto& V = inputs[2];

    HeadShape s;
    s.B = Q.shape(0);
    s.H = attrs.num_heads;
    s.merged = (Q.rank() == 3);
    if (s.merged) {
        s.Sq = Q.shape(1);
        s.Sk = K.shape(1);
        s.Sv = V.shape(1);
        s.D  = Q.shape(2) / s.H;
    } else {
        s.Sq = Q.shape(2);
        s.Sk = K.shape(2);
        s.Sv = V.shape(2);
        s.D  = Q.shape(3);
    }
    return s;
}

// ---- per-(batch,head) tiled kernel ---------------------------------------

template <class T>
void attention_impl(const AttentionAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace)
{
    const auto& Q = inputs[0];
    const auto& K = inputs[1];
    const auto& V = inputs[2];
    const bool has_mask = inputs.size() > 3;

    const HeadShape hs = resolve_head_shape(attrs, inputs);
    const int64_t B = hs.B, H = hs.H, Sq = hs.Sq, Sk = hs.Sk, D = hs.D;
    const bool merged = hs.merged;

    const float scale = (attrs.scale == 0.0f)
        ? (1.0f / std::sqrt(static_cast<float>(D)))
        : attrs.scale;

    const T* q_ptr = Q.ptr<T>();
    const T* k_ptr = K.ptr<T>();
    const T* v_ptr = V.ptr<T>();
    const T* mask_ptr = has_mask ? inputs[3].ptr<T>() : nullptr;
    T* out_ptr = output.ptr<T>();

    // Row strides in elements (from pitch, accounts for padding).
    const int64_t q_rs = Q.row_stride_elems();
    const int64_t k_rs = K.row_stride_elems();
    const int64_t v_rs = V.row_stride_elems();
    const int64_t o_rs = output.row_stride_elems();

    // Tile sizes (f32 only — the fast path is gated on dtype in attention_kernel).
    constexpr int mr_max = mr_max_flt<T>();
    constexpr int nr_max = nr_max_flt<T>();
    constexpr int kc     = KC_F32;

    int mc1, nc1;
    resolve_tile_sizes(static_cast<int>(Sq), static_cast<int>(Sk),
                       mr_max, nr_max, kc, static_cast<int>(sizeof(T)), mc1, nc1);
    int mc2, nc2;
    resolve_tile_sizes(static_cast<int>(Sq), static_cast<int>(D),
                       mr_max, nr_max, kc, static_cast<int>(sizeof(T)), mc2, nc2);

    // Uniform 64-byte-aligned panel stride at full Kc (sizing upper bound).
    const int ldd_b_full = align_up<PANEL_ALIGN_BYTES>(nr_max * kc * static_cast<int>(sizeof(T)))
                           / static_cast<int>(sizeof(T));
    const int np_full = num_panels(nc1, NR_F32);

    const int64_t scores_elems = Sq * Sk;
    const int64_t pack_elems   = static_cast<int64_t>(np_full) * ldd_b_full;
    const int64_t per_head     = scores_elems + pack_elems;

    const int64_t NG = B * H;
    T* ws_base = static_cast<T*>(workspace);

    const auto run = [&](int64_t idx) {
        const int64_t b = idx / H;
        const int64_t h = idx % H;

        // Head base pointers (head D columns are contiguous within each row).
        const T* q_head;
        const T* k_head;
        const T* v_head;
        T* o_head;
        if (merged) {
            q_head = q_ptr + b * Sq * q_rs + h * D;
            k_head = k_ptr + b * Sk * k_rs + h * D;
            v_head = v_ptr + b * hs.Sv * v_rs + h * D;
            o_head = out_ptr + b * Sq * o_rs + h * D;
        } else {
            q_head = q_ptr + (b * H + h) * Sq * q_rs;
            k_head = k_ptr + (b * H + h) * Sk * k_rs;
            v_head = v_ptr + (b * H + h) * hs.Sv * v_rs;
            o_head = out_ptr + (b * H + h) * Sq * o_rs;
        }

        T* scores = ws_base + idx * per_head;
        T* pack_b = scores + scores_elems;

        // ---- GEMM1: scores = Q @ K^T * scale --------------------------
        // First k-block writes with add_to=false — no pre-zeroing pass.
        for (int64_t k0 = 0; k0 < D; k0 += kc) {
            const int actual_kc = static_cast<int>(std::min<int64_t>(kc, D - k0));
            const int ldd_b = align_up<PANEL_ALIGN_BYTES>(nr_max * actual_kc * static_cast<int>(sizeof(T)))
                              / static_cast<int>(sizeof(T));
            for (int64_t n = 0; n < Sk; n += nc1) {
                const int actual_nc = static_cast<int>(std::min<int64_t>(nc1, Sk - n));
                tile_pack_rhs(true, actual_nc, actual_kc,
                              pack_b, ldd_b,
                              k_head + n * k_rs + k0, static_cast<int>(k_rs), scale);
                for (int64_t m = 0; m < Sq; m += mc1) {
                    const int actual_mc = static_cast<int>(std::min<int64_t>(mc1, Sq - m));
                    tile_mma_direct(actual_mc, actual_nc, actual_kc,
                                    scores + m * Sk + n, static_cast<int>(Sk),
                                    q_head + m * q_rs + k0, static_cast<int>(q_rs),
                                    pack_b, -1,
                                    kNegInf, kPosInf, k0 == 0);
                }
            }
        }

        // ---- mask + softmax over the last dim, per row, in-place ------
        // The additive mask is fused into the softmax 3-pass (max → exp+sum →
        // normalize), so scores is never rewritten before normalization — one
        // fewer full-buffer pass versus an explicit `scores += mask`.
        for (int64_t i = 0; i < Sq; ++i) {
            if (mask_ptr) {
                kernel::mask_softmax_process_standard_row<T>(
                    scores + i * Sk, mask_ptr + i * Sk, scores + i * Sk, Sk,
                    /*log_softmax=*/false, 1.0f);
            } else {
                kernel::softmax_process_standard_row<T>(
                    scores + i * Sk, scores + i * Sk, Sk, /*log_softmax=*/false, 1.0f);
            }
        }

        // ---- GEMM2: output = attn @ V (first block writes, rest accumulate) ----
        for (int64_t k0 = 0; k0 < Sk; k0 += kc) {
            const int actual_kc = static_cast<int>(std::min<int64_t>(kc, Sk - k0));
            for (int64_t n = 0; n < D; n += nc2) {
                const int actual_nc = static_cast<int>(std::min<int64_t>(nc2, D - n));
                for (int64_t m = 0; m < Sq; m += mc2) {
                    const int actual_mc = static_cast<int>(std::min<int64_t>(mc2, Sq - m));
                    tile_mma_direct(actual_mc, actual_nc, actual_kc,
                                    o_head + m * o_rs + n, static_cast<int>(o_rs),
                                    scores + m * Sk + k0, static_cast<int>(Sk),
                                    v_head + k0 * v_rs + n, static_cast<int>(v_rs),
                                    kNegInf, kPosInf, k0 == 0);
                }
            }
        }
    };

    ctx.cpu.run(0, NG, run);
}

// ---- FlashAttention routing + tiling -------------------------------------
//
// The tiled online-softmax (FlashAttention) path is chosen once the per-head
// scores matrix (Sq × Sk floats) is large enough that the standard path —
// materializing the full scores plus the fused mask/softmax passes — would
// thrash cache. Br / Bc follow the MLAS L2 heuristic:
//   Bc ≈ L2 / (4 · 4 · (qk_head_size + v_head_size)),  Br = min(Bc, 2D).

constexpr int64_t kFlashMinScores = 1 << 16;  // Sq*Sk threshold (≈ 256 KiB f32)

inline bool should_use_flash_attention(int64_t Sq, int64_t Sk)
{
    return Sq * Sk >= kFlashMinScores;
}

inline void resolve_flash_tile_sizes(int64_t Sq, int64_t Sk, int64_t D, int& Br, int& Bc)
{
    const size_t l2 = simd::CpuFeatures::get().l2_cache_size();
    const size_t l2_eff = l2 > 0 ? l2 : (1024u * 1024u);  // 1 MiB fallback
    const int64_t head_pairs = D + D;  // qk_head_size + v_head_size (both == D here)
    Bc = static_cast<int>(l2_eff / (sizeof(float) * 4 * head_pairs));
    if (Bc < 1) { Bc = 1; }
    Br = static_cast<int>(std::min<int64_t>(Bc, head_pairs));
    Bc = static_cast<int>(std::min<int64_t>(Bc, Sk));
    Br = static_cast<int>(std::min<int64_t>(Br, Sq));
    if (Br < 1) { Br = 1; }
    if (Bc < 1) { Bc = 1; }
}

// Per-task scratch (elements) — one Br-row query block — mirroring the layout
// inside attention_flash_impl:
//   [m:Br][l:Br][s:Br*Bc][o_acc:Br*D][pack_b:num_panels(Bc)*ldd_b_full]
inline int64_t flash_per_task_elems(int Br, int Bc, int64_t D)
{
    const int ldd_b_full = align_up<PANEL_ALIGN_BYTES>(NR_MAX_F32 * KC_F32 * static_cast<int>(sizeof(float)))
                           / static_cast<int>(sizeof(float));
    const int pack_elems = num_panels(Bc, NR_F32) * ldd_b_full;
    return static_cast<int64_t>(Br) * 2
         + static_cast<int64_t>(Br) * Bc
         + static_cast<int64_t>(Br) * D
         + pack_elems;
}

// ---- tiled FlashAttention kernel -----------------------------------------
//
// One parallel task = one query block (Br rows) of one (batch, head): it
// streams the KV rows in Bc-sized blocks, keeping a running max/sum and an
// output accumulator so the full Sq×Sk scores matrix is never materialized
// (standard online-softmax / FlashAttention-1 forward pass). Query blocks are
// folded into the parallel dimension (B × H × NQ) so long sequences still
// saturate the thread pool even when B × H is small. Reuses tile_mma_direct /
// tile_pack_rhs.

template <class T>
void attention_flash_impl(const AttentionAttributes& attrs,
                          TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx,
                          void* workspace)
{
    const auto& Q = inputs[0];
    const auto& K = inputs[1];
    const auto& V = inputs[2];
    const bool has_mask = inputs.size() > 3;

    const HeadShape hs = resolve_head_shape(attrs, inputs);
    const int64_t B = hs.B, H = hs.H, Sq = hs.Sq, Sk = hs.Sk, D = hs.D;
    const bool merged = hs.merged;

    const float scale = (attrs.scale == 0.0f)
        ? (1.0f / std::sqrt(static_cast<float>(D)))
        : attrs.scale;

    const T* q_ptr = Q.ptr<T>();
    const T* k_ptr = K.ptr<T>();
    const T* v_ptr = V.ptr<T>();
    const T* mask_ptr = has_mask ? inputs[3].ptr<T>() : nullptr;
    T* out_ptr = output.ptr<T>();

    const int64_t q_rs = Q.row_stride_elems();
    const int64_t k_rs = K.row_stride_elems();
    const int64_t v_rs = V.row_stride_elems();
    const int64_t o_rs = output.row_stride_elems();

    constexpr int nr_max = nr_max_flt<T>();
    constexpr int kc = KC_F32;
    constexpr int L = simd::simd_lane_for<T>;

    int Br, Bc;
    resolve_flash_tile_sizes(Sq, Sk, D, Br, Bc);
    const int64_t per_task = flash_per_task_elems(Br, Bc, D);

    const int64_t NQ = (Sq + Br - 1) / Br;  // number of Br-row query blocks
    const int64_t NG = B * H * NQ;
    T* ws_base = static_cast<T*>(workspace);

    const auto run = [&](int64_t idx) {
        const int64_t qi = idx % NQ;
        const int64_t b  = (idx / NQ) / H;
        const int64_t h  = (idx / NQ) % H;
        const int64_t i0 = qi * Br;
        const int actual_br = static_cast<int>(std::min<int64_t>(Br, Sq - i0));

        const T* q_head;
        const T* k_head;
        const T* v_head;
        T* o_head;
        if (merged) {
            q_head = q_ptr + b * Sq * q_rs + h * D;
            k_head = k_ptr + b * Sk * k_rs + h * D;
            v_head = v_ptr + b * hs.Sv * v_rs + h * D;
            o_head = out_ptr + b * Sq * o_rs + h * D;
        } else {
            q_head = q_ptr + (b * H + h) * Sq * q_rs;
            k_head = k_ptr + (b * H + h) * Sk * k_rs;
            v_head = v_ptr + (b * H + h) * hs.Sv * v_rs;
            o_head = out_ptr + (b * H + h) * Sq * o_rs;
        }

        T* m      = ws_base + idx * per_task;
        T* l      = m + Br;
        T* s      = l + Br;
        T* o_acc  = s + static_cast<int64_t>(Br) * Bc;
        T* pack_b = o_acc + static_cast<int64_t>(Br) * D;

        // reset the running online-softmax state for this query block
        for (int i = 0; i < actual_br; ++i) {
            m[i] = -std::numeric_limits<T>::infinity();
            l[i] = 0.0f;
        }
        // o_acc is written by the first KV block with add_to=false — no
        // pre-zeroing pass. (The exp_diff rescale below may touch it first on
        // j0 == 0; that garbage is fully overwritten by the first MMA.)
        for (int64_t j0 = 0; j0 < Sk; j0 += Bc) {
            const int actual_bc = static_cast<int>(std::min<int64_t>(Bc, Sk - j0));

            // ---- S = Q[i0:i0+Br] @ K[j0:j0+Bc]^T * scale ----------
            // First k-block writes with add_to=false — no pre-zeroing pass.
            for (int64_t k0 = 0; k0 < D; k0 += kc) {
                const int actual_kc = static_cast<int>(std::min<int64_t>(kc, D - k0));
                const int ldd_b = align_up<PANEL_ALIGN_BYTES>(nr_max * actual_kc * static_cast<int>(sizeof(T)))
                                  / static_cast<int>(sizeof(T));
                tile_pack_rhs(true, actual_bc, actual_kc,
                              pack_b, ldd_b,
                              k_head + j0 * k_rs + k0, static_cast<int>(k_rs), scale);
                tile_mma_direct(actual_br, actual_bc, actual_kc,
                                s, actual_bc,
                                q_head + i0 * q_rs + k0, static_cast<int>(q_rs),
                                pack_b, -1, kNegInf, kPosInf, k0 == 0);
            }

            // ---- additive mask (flat [Sq, Sk], shared across heads) ----
            if (mask_ptr) {
                for (int i = 0; i < actual_br; ++i) {
                    const T* mrow = mask_ptr + (i0 + i) * Sk + j0;
                    T* srow = s + i * actual_bc;
                    int j = 0;
                    for (; j + L <= actual_bc; j += L) {
                        simd::v_store(srow + j, simd::v_add(simd::v_load(srow + j), simd::v_load(mrow + j)));
                    }
                    for (; j < actual_bc; ++j) {
                        simd::s_store(&srow[j], simd::s_load(&srow[j]) + simd::s_load(&mrow[j]));
                    }
                }
            }

            // ---- online-softmax update + accumulate into o_acc -----
            for (int i = 0; i < actual_br; ++i) {
                T* srow = s + i * actual_bc;
                const float rowmax = kernel::softmax_row_max<T>(srow, actual_bc);
                const float old_m = static_cast<float>(m[i]);
                const float new_m = std::max(old_m, rowmax);
                const float exp_diff = std::exp(old_m - new_m);  // 0 on the first KV block
                m[i] = static_cast<T>(new_m);
                const float rowsum = kernel::softmax_row_exp_sum<T>(srow, srow, actual_bc, new_m, 1.0f);
                l[i] = static_cast<T>(exp_diff * static_cast<float>(l[i]) + rowsum);

                // rescale the accumulated output by exp_diff (skip work when it is 1.0)
                if (exp_diff != 1.0f) {
                    T* orow = o_acc + i * D;
                    auto v_ed = simd::v_set1(orow, exp_diff);
                    int64_t d = 0;
                    for (; d + L <= D; d += L) {
                        simd::v_store(orow + d, simd::v_mul(simd::v_load(orow + d), v_ed));
                    }
                    for (; d < D; ++d) {
                        simd::s_store(&orow[d], simd::s_load(&orow[d]) * exp_diff);
                    }
                }
            }

            // ---- o_acc += S @ V[j0:j0+Bc] --------------------------
            for (int64_t k0 = 0; k0 < actual_bc; k0 += kc) {
                const int actual_kc = static_cast<int>(std::min<int64_t>(kc, actual_bc - k0));
                tile_mma_direct(actual_br, static_cast<int>(D), actual_kc,
                                o_acc, static_cast<int>(D),
                                s + k0, actual_bc,
                                v_head + (j0 + k0) * v_rs, static_cast<int>(v_rs),
                                kNegInf, kPosInf, j0 == 0 && k0 == 0);
            }
        }

        // ---- final normalize: output = o_acc / l -------------------
        for (int i = 0; i < actual_br; ++i) {
            const float inv_l = 1.0f / static_cast<float>(l[i]);
            const T* src = o_acc + i * D;
            T* dst = o_head + (i0 + i) * o_rs;
            auto v_inv = simd::v_set1(src, inv_l);
            int64_t d = 0;
            for (; d + L <= D; d += L) {
                simd::v_store(dst + d, simd::v_mul(simd::v_load(src + d), v_inv));
            }
            for (; d < D; ++d) {
                simd::s_store(&dst[d], simd::s_load(&src[d]) * inv_l);
            }
        }
    };

    ctx.cpu.run(0, NG, run);
}

}  // anonymous namespace

// =========================================================================
//  Workspace sizing — mirrors the kernel's tiling exactly
// =========================================================================

size_t attention_get_workspace_size(const AttentionAttributes& attrs,
                                    std::span<const TensorDesc> inputs,
                                    std::span<const TensorDesc> /*outputs*/)
{
    NNOPS_ASSERT(inputs.size() >= 3);
    NNOPS_ASSERT(inputs.size() <= 4);

    const auto& q = inputs[0];
    const auto& k = inputs[1];
    const auto& v = inputs[2];

    // f32-only fast path (reference is f32-only).
    if (q.dtype != DataType::f32 || k.dtype != DataType::f32 ||
        v.dtype != DataType::f32) {
        return 0;
    }

    const int64_t H = attrs.num_heads;
    const bool merged = (q.rank == 3);
    int64_t B, Sq, Sk, D;
    if (merged) {
        B  = q.dims[0];
        Sq = q.dims[1];
        Sk = k.dims[1];
        D  = q.dims[2] / H;
    } else {
        B  = q.dims[0];
        Sq = q.dims[2];
        Sk = k.dims[2];
        D  = q.dims[3];
    }

    // FlashAttention path — sized with the same Br/Bc the kernel resolves, and
    // with one scratch slot per query block (tasks are B × H × NQ).
    if (should_use_flash_attention(Sq, Sk)) {
        int Br, Bc;
        resolve_flash_tile_sizes(Sq, Sk, D, Br, Bc);
        const int64_t per_task = flash_per_task_elems(Br, Bc, D);
        const int64_t NQ = (Sq + Br - 1) / Br;
        return static_cast<size_t>(B * H * NQ) * static_cast<size_t>(per_task) * sizeof(float);
    }

    constexpr int mr_max = MR_MAX_F32;
    constexpr int nr_max = NR_MAX_F32;
    constexpr int kc     = KC_F32;

    int mc1, nc1;
    resolve_tile_sizes(static_cast<int>(Sq), static_cast<int>(Sk),
                       mr_max, nr_max, kc, static_cast<int>(sizeof(float)),
                       mc1, nc1);

    const int ldd_b_full = align_up<PANEL_ALIGN_BYTES>(nr_max * kc * static_cast<int>(sizeof(float)))
                           / static_cast<int>(sizeof(float));
    const int np_full = num_panels(nc1, NR_F32);

    const int64_t scores_elems = Sq * Sk;
    const int64_t pack_elems   = static_cast<int64_t>(np_full) * ldd_b_full;
    const int64_t per_head     = scores_elems + pack_elems;

    return static_cast<size_t>(B * H) * static_cast<size_t>(per_head) * sizeof(float);
}

// =========================================================================
//  Public API
// =========================================================================

void attention_kernel(const AttentionAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* /*workspace*/)
{
    const DataType dt = inputs[0].data_type();

    const bool dtypes_ok = (dt == DataType::f32) &&
                           inputs[1].data_type() == dt &&
                           inputs[2].data_type() == dt &&
                           output.data_type() == dt &&
                           (inputs.size() <= 3 || inputs[3].data_type() == dt);

    if (dtypes_ok) {
        // Scratch (scores + packed K^T) is pooled internally; the caller-provided
        // workspace is obsolete (getWorkspaceSize returns 0). Allocated once here,
        // before the parallel dispatch, so no per-task allocation happens inside
        // the parallel body.
        const TensorDesc descs[] = {inputs[0].desc(), inputs[1].desc(), inputs[2].desc()};
        const TensorDesc outs[]  = {output.desc()};
        const size_t ws = attention_get_workspace_size(attrs, descs, outs);
        PoolPtr scratch(ws);

        const HeadShape hs = resolve_head_shape(attrs, inputs);
        if (should_use_flash_attention(hs.Sq, hs.Sk)) {
            attention_flash_impl<float>(attrs, output, inputs, ctx, scratch.as<float>());
        } else {
            attention_impl<float>(attrs, output, inputs, ctx, scratch.as<float>());
        }
        return;
    }

    // Mismatched dtypes: the reference is the correctness baseline (no scratch).
    reference::attention_ref(attrs, output, inputs, ctx, nullptr);
}

}  // namespace nnops::backend::cpu
