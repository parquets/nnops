/// @file attention.cpp
/// @brief Tiled multi-head scaled dot-product attention — fused QK^T + softmax + AV.
///
/// For each (batch, head) — parallel via ctx.cpu_parallel_for (serial fallback):
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
/// Grouped-query attention (attrs.num_group) is not yet handled — this runs
/// standard MHA (one head block per (batch, head)), matching the reference.

#include "attention.h"
#include "matmul_helper.h"                  // tile_mma_direct / tile_pack_rhs + panel constants
#include "simd_kernel/simd_softmax.hpp"     // softmax_process_standard_row
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
        std::memset(scores, 0, static_cast<size_t>(scores_elems) * sizeof(T));
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
                                    kNegInf, kPosInf);
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

        // ---- GEMM2: output = attn @ V (zero-init, then accumulate) ----
        for (int64_t m = 0; m < Sq; ++m) {
            std::memset(o_head + m * o_rs, 0, static_cast<size_t>(D) * sizeof(T));
        }
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
                                    kNegInf, kPosInf);
                }
            }
        }
    };

    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, NG, run);
    } else {
        for (int64_t idx = 0; idx < NG; ++idx) {
            run(idx);
        }
    }
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
                      void* workspace)
{
    const DataType dt = inputs[0].data_type();

    const bool dtypes_ok = (dt == DataType::f32) &&
                           inputs[1].data_type() == dt &&
                           inputs[2].data_type() == dt &&
                           output.data_type() == dt &&
                           (inputs.size() <= 3 || inputs[3].data_type() == dt);

    if (dtypes_ok && workspace != nullptr) {
        attention_impl<float>(attrs, output, inputs, ctx, workspace);
        return;
    }

    // f32 without a workspace (or mismatched dtypes): the reference is the
    // correctness baseline and needs no scratch.
    reference::attention_ref(attrs, output, inputs, ctx, workspace);
}

}  // namespace nnops::backend::cpu
