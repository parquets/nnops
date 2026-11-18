/// @file matmul_sme_i8.cpp
/// @brief Whole-product int8 MatMul through the AArch64 SME SMOPA kernels.
///
/// The int8 sibling of matmul_sme.cpp: no plan, its own pack layouts
/// ([m/16][k/4][64] for A, [n/64][k/4][256] for B) and its own k-block, tuned
/// around the 4-way dot's 1024 MACs per instruction. Threads split N, each
/// worker packs its own A slab and, without a prepacked B, its own B slab.
///
/// This route owns only the integer destinations — s32's raw accumulate, or
/// s8/u8's requantize. The int8 output pass is the planned route's
/// (simd_epilogue.hpp): the kernel leaves the raw accumulator in C, and the
/// last k-block's tile runs the zero-point compensation, the bias, the clamp and
/// the requantize exactly as the tiled route does. An fp destination keeps the
/// dequantizing route: it reads a different B layout, so an s8 SME blob is
/// decoded to a dense matrix for it (matmul_packed_b_sme_i8_unpack).
///
/// A prepacked s8 blob is [ceil(N/64)][ceil(K/4)][256], the kernel's per-call
/// block layout stretched over the whole K: one k-block is an offset into it,
/// and no per-call B pack runs. The blob is only produced when the prepack call
/// asked for an integral output (see matmul_packed_b_desc).
///
/// The route declines what it cannot fill or what another route owns: an fp
/// activation (the dq route quantizes and dequantizes), M <= 4 or N < 16 on a
/// raw B, a NEON panel blob, and a destination that is neither integral nor the
/// blob's raw matrix form.

#include "matmul_internal.h"

#if defined(NNOPS_ARCH_SME)

#include "backend/cpu/aarch64/mma_i8_sme.hpp"
#include "backend/cpu/aarch64/mma_sme.hpp"
#include "backend/cpu/aarch64/pack_i8_sme.hpp"
#include "backend/cpu/common/memory_pool.hpp"
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"
#include "backend/cpu/simd_kernel/simd_int8_reduce.hpp"

#include <algorithm>
#include <cstdint>

namespace nnops::backend::cpu {

using namespace aarch64;

namespace {

constexpr int kSmeRows = 16;  // rows in one ZA tile
constexpr int kSmeMSmall = 4; // MatMul's small-M routes own M at or below it
constexpr int kSmeMc = 64;    // rows per A slab: four ZA tiles
/// Widest column block one worker takes.
constexpr int kSmeNMax = 1024;
/// K per block. A multiple of the 4-way dot's group, so a prepacked blob's
/// block offset stays group-aligned; long enough that the ZA read-back and the
/// stores amortise, while a worker's B slab stays 2 MiB at the widest block.
constexpr int kSmeKc = 2048;

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
/// Σ_k B[k,n] (the transposes mirror them). The same kernels and slab split the
/// planned route uses, so the compensation is bit-identical.
void i8_reductions(const int8_t* a_ptr, bool a_u8, int64_t lda, const int8_t* b_ptr,
                   int64_t ldb, const MatMulAttributes& attrs, int64_t M, int64_t N,
                   int64_t K, int32_t* r_a, int32_t* r_b, const ComputeContext& ctx)
{
    const auto reduce = [&](const void* p, bool u8, int64_t ld, int64_t n_out, bool rows,
                            int32_t* r) {
        const int64_t tasks =
            std::min<int64_t>(n_out, std::max<int64_t>(ctx.cpu.thread_count(), 1));
        ctx.cpu.run(0, tasks, [&](int64_t t) {
            const int64_t i0 = t * n_out / tasks;
            const int64_t i1 = (t + 1) * n_out / tasks;
            if (rows) {
                for (int64_t i = i0; i < i1; ++i) {
                    r[i] = u8 ? kernel::int8_row_sum(static_cast<const uint8_t*>(p) + i * ld, K) :
                                kernel::int8_row_sum(static_cast<const int8_t*>(p) + i * ld, K);
                }
            } else {
                for (int64_t i = i0; i < i1; ++i) {
                    r[i] = 0;
                }
                if (u8) {
                    kernel::int8_col_sum_slab(static_cast<const uint8_t*>(p), ld, i0, i1, K, r);
                } else {
                    kernel::int8_col_sum_slab(static_cast<const int8_t*>(p), ld, i0, i1, K, r);
                }
            }
        });
    };

    if (r_a != nullptr) {
        reduce(a_ptr, a_u8, lda, M, /*rows=*/!attrs.transpose_a, r_a);
    }
    if (r_b != nullptr) {
        reduce(b_ptr, /*u8=*/false, ldb, N, /*rows=*/attrs.transpose_b, r_b);
    }
}

/// Σ_k B[k,n] read out of an SME-layout s8 blob. Padding (the columns past N
/// and a short final k group) is zero, so it contributes nothing.
void sme_i8_blob_col_sums(const int8_t* blob, int64_t N, int64_t K, int32_t* r_b,
                          const ComputeContext& ctx)
{
    const int64_t kg = (K + 3) / 4;
    const int64_t nr = sme_nr_max;
    const int64_t za = sme_za_n;
    const int64_t n_blocks = (N + nr - 1) / nr;
    ctx.cpu.run(0, n_blocks, [&](int64_t nb) {
        const int64_t n0 = nb * nr;
        const int64_t cols = std::min(nr, N - n0);
        const int8_t* const blk = blob + nb * kg * (4 * nr);
        for (int64_t j = 0; j < cols; ++j) {
            const int8_t* const cell0 = blk + (j / za) * (4 * za) + (j % za) * 4;
            int32_t s = 0;
            for (int64_t g = 0; g < kg; ++g) {
                const int8_t* const cell = cell0 + g * (4 * nr);
                s += cell[0] + cell[1] + cell[2] + cell[3];
            }
            r_b[n0 + j] = s;
        }
    });
}

/// The output pass's parameters plus the buffers that feed it. Empty when the
/// pass is the identity (the raw accumulate is the answer).
struct I8SmeEpilogue {
    Int8EpilogueParams p{};
    PoolArray<int32_t> r_a, r_b, r_fold;
    bool active = false;
    bool out_q = false;
    bool fold_row = false;
    bool sweep_a = false;
    bool sweep_b = false;
};

void setup_epilogue(I8SmeEpilogue& e, const MatMulAttributes& attrs, const TensorView& a,
                    const TensorView& b, const TensorView& output, const int32_t* int_bias,
                    int64_t M, int64_t N, int64_t K, int64_t ldc_out)
{
    const DataType odt = output.data_type();
    e.out_q = odt == DataType::s8 || odt == DataType::u8;
    const QuantParams& qa = a.quant_params();
    const QuantParams& qb = b.quant_params();
    const int32_t a_bias_offset = int8_epilogue_offset(a.data_type());
    const bool col_term =
        any_zp_differs(qa.zero_point_data, qa.zero_point, qa.num_scales, M, -a_bias_offset);
    const bool row_half = any_zp_nonzero(qb.zero_point_data, qb.zero_point, qb.num_scales, N);
    e.fold_row = row_half && qb.zero_point_data == nullptr;
    e.sweep_a = row_half;
    e.sweep_b = col_term;
    e.active = e.sweep_a || e.sweep_b || e.out_q || int_bias != nullptr ||
               epilogue_is_active(attrs.epilogue);
    if (!e.active) {
        return;
    }
    if (e.sweep_a) {
        e.r_a.reset(static_cast<size_t>(M));
    }
    if (e.sweep_b) {
        e.r_b.reset(static_cast<size_t>(N));
    }
    if (e.fold_row) {
        e.r_fold.reset(static_cast<size_t>(M));
    }

    Int8EpilogueParams& p = e.p;
    p.r_a = e.sweep_a ? e.r_a.data() : nullptr;
    p.r_b = e.sweep_b ? e.r_b.data() : nullptr;
    p.s_a = qa.scale_data;
    p.s_a_scalar = qa.scale;
    p.zp_a = qa.zero_point_data;
    p.zp_a_scalar = qa.zero_point;
    p.s_b = qb.scale_data;
    p.s_b_scalar = qb.scale;
    p.zp_b = qb.zero_point_data;
    p.zp_b_scalar = qb.zero_point;
    p.a_bias_offset = a_bias_offset;
    p.k = static_cast<int32_t>(K);
    p.ldc_out = static_cast<int>(ldc_out);
    p.compensate = e.sweep_a && !e.fold_row;
    p.col_term = col_term;
    p.relu = attrs.epilogue.type == EpilogueActivateType::Relu;
    p.col_bias = int_bias;
    p.bias = e.fold_row ? e.r_fold.data() : nullptr;
    if (e.out_q) {
        const QuantParams& qc = output.quant_params();
        p.out_scale = quant_scale_at(qc, 0);
        p.zp_out = quant_zp_at(qc, 0);
        if (attrs.epilogue.quant == EpilogueQuantType::Requantize &&
            attrs.epilogue.quant_param_count > 0) {
            p.row_out_scales = attrs.epilogue.quant_scales;
            p.row_out_zps = attrs.epilogue.quant_zero_points;
            p.out_scale = epilogue_quant_scale(attrs.epilogue, 0);
            p.zp_out = epilogue_quant_zero(attrs.epilogue, 0);
        }
    }
}

/// One worker's column block over the whole product: pack B per k-block (unless
/// the caller prepacked it), pack A per row slab, accumulate, and run the
/// output pass on the last k-block's tiles.
void sme_i8_block(const int8_t* a, int lda, const int8_t* b_raw, int ldb, int M, int n0,
                  int nw, int K, const int8_t* prepacked, bool trans_a, bool trans_b,
                  bool a_biased, int32_t* c_base, int ldc, const Int8EpilogueParams* ip)
{
    PoolArray<int8_t> pack_b(prepacked != nullptr ?
                                 0 :
                                 static_cast<size_t>(packed_b_elems_i8_sme(nw, kSmeKc)));
    PoolArray<int8_t> pack_a(static_cast<size_t>(packed_a_elems_i8_sme(kSmeMc, kSmeKc)));
    const int64_t blob_stride = prepacked != nullptr ?
                                    static_cast<int64_t>((K + 3) / 4) * (4 * sme_nr_max) :
                                    0;
    const int64_t blob_block0 = static_cast<int64_t>(n0 / sme_nr_max) * blob_stride;

    for (int k = 0; k < K; k += kSmeKc) {
        const int kc = std::min(kSmeKc, K - k);
        const bool last_k = (k + kc >= K);
        if (prepacked == nullptr) {
            const int8_t* src = trans_b ? b_raw + static_cast<int64_t>(n0) * ldb + k :
                                          b_raw + static_cast<int64_t>(k) * ldb + n0;
            pack_b_i8_sme(nw, kc, pack_b.data(), src, ldb, trans_b);
        }

        for (int m0 = 0; m0 < M; m0 += kSmeMc) {
            const int mc = std::min(kSmeMc, M - m0);
            int32_t* c_tile = c_base + static_cast<int64_t>(m0) * ldc + n0;
            const int8_t* a_src = trans_a ? a + static_cast<int64_t>(k) * lda + m0 :
                                            a + static_cast<int64_t>(m0) * lda + k;
            pack_a_i8_sme(mc, kc, pack_a.data(), a_src, lda, trans_a, a_biased);

            const int8_t* b_tile =
                prepacked != nullptr ?
                    prepacked + blob_block0 + static_cast<int64_t>(k / 4) * (4 * sme_nr_max) :
                    pack_b.data();
            mma_i8_sme(mc, nw, kc, c_tile, ldc, pack_a.data(), b_tile, /*zero_mode=*/k == 0,
                        INT32_MIN, INT32_MAX, blob_stride);

            if (last_k && ip != nullptr) {
                // The tile body is indexed from its own first row; the per-row
                // arrays (r_a, r_fold, bias) by the absolute m0.
                int8_epilogue_tile(*ip, c_base + static_cast<int64_t>(m0) * ldc, ldc, m0, n0,
                                   mc, nw);
            }
        }
    }
}

} // namespace

bool matmul_sme_i8(const MatMulAttributes& attrs, TensorView& output,
                   std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    if (!available_sme() || inputs.size() < 2) {
        return false;
    }
    const auto& a = inputs[0];
    const auto& b = inputs[1];
    const DataType dt_a = a.data_type();
    const DataType odt = output.data_type();
    if ((dt_a != DataType::s8 && dt_a != DataType::u8) || b.data_type() != DataType::s8) {
        return false;
    }
    if (odt != DataType::s32 && odt != DataType::s8 && odt != DataType::u8) {
        return false;
    }
    // An integral destination has no fp sum to add the residual to.
    if (attrs.residual_pre_activation) {
        return false;
    }
    // The NEON panel blob is unreadable here; the SME blob is ours.
    if (is_packed_matmul_b(b.layout()) && b.layout() != TensorLayout::PackedBSme) {
        return false;
    }
    const bool has_blob = b.layout() == TensorLayout::PackedBSme;

    const auto [M64, N64, K64] = matmul_dims(attrs, a, b);
    if (M64 <= 0 || N64 <= 0 || K64 <= 0 || M64 > INT32_MAX || N64 > INT32_MAX ||
        K64 > INT32_MAX) {
        return false;
    }
    // Raw operands leave the small shapes to the planned route; a prepacked
    // blob has no other reader, so every shape takes this one.
    if (!has_blob && (M64 <= kSmeMSmall || N64 < kSmeRows)) {
        return false;
    }
    const int M = static_cast<int>(M64);
    const int N = static_cast<int>(N64);
    const int K = static_cast<int>(K64);

    const int64_t n_blocks_total = (N64 + sme_nr_max - 1) / sme_nr_max;
    const int workers = static_cast<int>(
        std::max<int64_t>(1, std::min<int64_t>(ctx.cpu.thread_count(), n_blocks_total)));
    int nw = std::min(kSmeNMax, (N + workers - 1) / workers);
    nw = ((nw + sme_nr_max - 1) / sme_nr_max) * sme_nr_max;
    const int blocks = (N + nw - 1) / nw;

    const int lda = static_cast<int>(a.row_stride_elems());
    const int ldb = static_cast<int>(b.row_stride_elems());
    const int ldc_out = static_cast<int>(output.row_stride_elems());

    const int32_t* int_bias = inputs.size() > 2 ? inputs[2].ptr<int32_t>() : nullptr;
    I8SmeEpilogue ep;
    setup_epilogue(ep, attrs, a, b, output, int_bias, M, N, K, ldc_out);

    const bool out_q = ep.out_q;
    const int ldc = out_q ? N : ldc_out;
    PoolArray<int32_t> accum(out_q ? static_cast<size_t>(M) * static_cast<size_t>(N) : 0);

    const int8_t* const a_base = a.ptr<int8_t>();
    const int8_t* const b_base = b.ptr<int8_t>();
    const bool a_biased = dt_a == DataType::u8;
    const bool trans_b_raw = attrs.transpose_b && !has_blob;

    // A prepacked blob does not vary per batch: its column sums are read once.
    if (has_blob && ep.sweep_b) {
        sme_i8_blob_col_sums(b_base, N, K, ep.r_b.data(), ctx);
    }

    const BatchWalk batch(a, b);
    for (int64_t bi = 0; bi < batch.total; ++bi) {
        int64_t a_off, b_off, c_off;
        batch.offsets(bi, output, a_off, b_off, c_off);

        const int8_t* a_p = a_base + a_off;
        const int8_t* b_p = b_base + b_off;
        int32_t* c_p = out_q ? accum.data() : output.ptr<int32_t>() + c_off;

        if (ep.sweep_a || (ep.sweep_b && !has_blob)) {
            i8_reductions(a_p, a_biased, lda, b_p, ldb, attrs, M, N, K,
                          ep.sweep_a ? ep.r_a.data() : nullptr,
                          (ep.sweep_b && !has_blob) ? ep.r_b.data() : nullptr, ctx);
        }
        if (ep.active) {
            if (out_q) {
                const bool as_u8 = odt == DataType::u8;
                ep.p.out = as_u8 ? nullptr : output.ptr<int8_t>() + c_off;
                ep.p.out_u8 = as_u8 ? output.ptr<uint8_t>() + c_off : nullptr;
            }
            if (ep.fold_row) {
                for (int m = 0; m < M; ++m) {
                    const int32_t zp_a_m = int8_zp_at(ep.p.zp_a, ep.p.zp_a_scalar, m);
                    ep.r_fold[static_cast<size_t>(m)] =
                        -ep.p.zp_b_scalar *
                        (ep.r_a[static_cast<size_t>(m)] - ep.p.k * zp_a_m);
                }
            }
        }

        ctx.cpu.run(0, blocks, [&](int64_t blk) {
            const int n0 = static_cast<int>(blk) * nw;
            sme_i8_block(a_p, lda, b_p, ldb, M, n0, std::min(nw, N - n0), K,
                         has_blob ? b_p : nullptr, attrs.transpose_a, trans_b_raw, a_biased, c_p,
                         ldc, ep.active ? &ep.p : nullptr);
        });
    }
    return true;
}

} // namespace nnops::backend::cpu

#else // no SME kernels compiled in

namespace nnops::backend::cpu {

bool matmul_sme_i8(const MatMulAttributes&, TensorView&, std::span<const TensorView>,
                   const ComputeContext&)
{
    return false;
}

} // namespace nnops::backend::cpu

#endif
