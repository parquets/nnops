#pragma once
/// @file matmul_tile_impl.h
/// @brief The plan-driven tile skeleton, shared by the float routes
/// (matmul.cpp) and the quantized ones (matmul_i8.cpp).
///
/// One body for every route: per batch element, split the output on its larger
/// dimension and run the blocks on the pool. The route is a type — its packs,
/// MMA, clamp, epilogue and output rules are the only dtype-specific part; the
/// two routes disagree on B hoisting, on the output pass, and on where C lives,
/// and say so through their own constants.
///
/// Everything here is a template (or a helper an instantiation reads), so each
/// translation unit instantiates exactly the routes it names and pays nothing
/// for the other family.

#include "matmul_internal.h" // dims, batch walk, PackedBView, route declarations
#include "matmul_helper.h"   // MatMulPlan, pack/MMA layer, panel geometry
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"
#include "backend/cpu/common/memory_pool.hpp"

#include <algorithm>
#include <cstdint>
#include <span>
#include <utility>

namespace nnops::backend::cpu {

namespace {

// =========================================================================
//  Tile-block kernels
// =========================================================================

struct BlockRange {
    int start;
    int count;
};

inline BlockRange m_range(const MatMulPlan& plan, int64_t blk, int64_t num_blocks, int M) noexcept
{
    if (plan.split_n) {
        return {0, M};
    }
    const int start = static_cast<int>(blk * M / num_blocks);
    return {start, static_cast<int>((blk + 1) * M / num_blocks) - start};
}

inline BlockRange n_range(const MatMulPlan& plan, int64_t blk, int N) noexcept
{
    if (!plan.split_n) {
        return {0, N};
    }
    const int start = static_cast<int>(blk) * static_cast<int>(plan.nc);
    return {start, std::min(static_cast<int>(plan.nc), N - start)};
}

/// Scratch slot: per worker thread when slots < blocks, else per block.
inline int64_t block_slot(int64_t num_slots, int64_t num_blocks, int64_t blk,
                          const ComputeContext& ctx) noexcept
{
    return (num_slots < num_blocks) ? static_cast<int64_t>(ctx.cpu.current_thread_id()) : blk;
}

/// Packed-A slot of a block, or null when A is read raw; uses num_slots_a.
template <class Elem>
inline Elem* pack_a_slot(const MatMulPlan& plan, int64_t num_blocks, int64_t blk,
                         const ComputeContext& ctx, Elem* base) noexcept
{
    if (!plan.pack_a) {
        return nullptr;
    }
    return base + block_slot(plan.num_slots_a, num_blocks, blk, ctx) * plan.np_a * plan.ldd_a;
}

template <class R>
void matmul_tile(const MatMulAttributes& attrs, typename R::Acc* c_ptr, int ldc,
                 const typename R::AElem* a_ptr, int lda, int m_start, int m_count, int mc,
                 int n_start, int n_count, int k, int actual_kc, bool last_k, bool pack_a,
                 const typename R::Elem* packed_b, typename R::Elem* pack_a_slice, bool a_biased,
                 const typename R::EpilogueParams* ip, const typename R::Elem* bias,
                 const typename R::Elem* col_bias, const PackedBView& prepacked)
{
    using Acc = typename R::Acc;

    const int ldd_a = R::ldd_a(actual_kc);
    // The first k-block primes C with beta·C_old + bias; every k-block then
    // only accumulates the product. A bias forces accumulate mode — the
    // overwriting kernel would drop it — even when beta is 0.
    //
    // When there is no old C to fold (beta 0) the bias becomes the accumulator's
    // initial value and the priming pass disappears.
    //
    // The pre-activation residual is the beta = 1 order — C keeps its own
    // contents, the first block accumulates onto them, and the last block
    // clamps and activates the finished sum — so it substitutes the scale and
    // leaves MatMul's beta untouched. A bias is then added onto C instead of
    // seeding the accumulator, which would drop the residual. The int8 routes
    // keep beta out of it: their accumulator is not C, and the dequant pass
    // adds C_old itself.
    const float beta = (R::is_fp && attrs.residual_pre_activation) ? 1.0f : attrs.beta;
    const bool seed_bias = R::is_fp && k == 0 && col_bias != nullptr && beta == 0.0f;
    const bool prime_c = R::is_fp && k == 0 && bias != nullptr && !seed_bias;
    const bool zero_mode = (k == 0 && beta == 0.0f && bias == nullptr);
    const auto [cmin, cmax] = R::clamp(attrs, last_k);
    // fp-only; compile out of the int8 instantiation.
    const bool scale_c = R::is_fp && k == 0 && bias == nullptr && beta != 1.0f && beta != 0.0f;
    const bool epilogue = R::is_fp && last_k && has_inplace_epilogue(attrs);
    const bool int8_ep = !R::is_fp && last_k && (ip != nullptr);

    for (int m = m_start; m < m_start + m_count; m += mc) {
        const int actual_mc = std::min(mc, m_start + m_count - m);
        Acc* c_tile = c_ptr + m * ldc + n_start;

        const typename R::AElem* a_src = attrs.transpose_a ?
                                             a_ptr + k * lda + m // A phys is K×M, row k, col m
                                             :
                                             a_ptr + m * lda + k; // A phys is M×K, row m, col k

        if constexpr (R::is_fp) {
            if (prime_c) {
                R::scale_add_bias(c_tile, ldc, beta, bias + n_start, actual_mc, n_count);
            } else if (scale_c) {
                R::scale(c_tile, ldc, beta, actual_mc, n_count);
            }
        }
        // The RHS the MMA reads: the k-block's slice of the packed scratch, or
        // the blob's own panel for this block, which one kernel call per panel
        // walks at the pitch the route names (see PackedBView).
        const typename R::Elem* b_tile = packed_b;
        int64_t b_ldb = R::ldb_packed;
        if (prepacked.ok()) {
            const auto at = R::prepacked_addr(prepacked, n_start / R::b_panel_w(), k);
            b_tile = at.first;
            b_ldb = at.second;
        }

        if (pack_a) {
            R::pack_a(attrs.transpose_a, actual_mc, actual_kc, pack_a_slice, ldd_a, a_src, lda,
                      a_biased, ip, m);
            R::mma_pack(actual_mc, n_count, actual_kc, c_tile, ldc, pack_a_slice, b_tile, cmin,
                        cmax, zero_mode, b_ldb, seed_bias ? col_bias + n_start : nullptr);
        } else if constexpr (R::has_direct_a) {
            R::mma_direct(actual_mc, n_count, actual_kc, c_tile, ldc, a_src, lda, b_tile, cmin,
                          cmax, zero_mode, b_ldb, seed_bias ? col_bias + n_start : nullptr);
        }
        if constexpr (R::is_fp) {
            if (epilogue) {
                R::epilogue(attrs, c_tile, ldc, actual_mc, n_count);
            }
        } else {
            // From column 0 of C, not from the tile: the kernels index the row
            // by the absolute output column (they read r_b[n] the same way).
            if (int8_ep) {
                R::epilogue_tile(*ip, c_ptr + m * ldc, ldc, m, n_start, actual_mc, n_count);
            }
        }
    }
}

/// One block over all K: pack B per k-block, accumulate every m-panel.
template <class R>
void matmul_block(const MatMulAttributes& attrs, typename R::Acc* c_ptr, int ldc,
                  const typename R::AElem* a_ptr, int lda, const typename R::B& b_src, int K,
                  int kc, int mc, int m_start, int m_count, int n_start, int n_count,
                  typename R::Elem* pack_b_slice, typename R::Elem* pack_a_slice, bool pack_a,
                  bool a_biased, const typename R::EpilogueParams* ip, const typename R::Elem* bias,
                  const typename R::Elem* col_bias, const PackedBView& prepacked)
{
    for (int k = 0; k < K; k += kc) {
        const int actual_kc = std::min(kc, K - k);
        const bool last_k = (k + kc >= K);
        // A prepacked RHS is already packed for every k-block: no pack here.
        if (!prepacked.ok()) {
            const int ldd_b = R::ldd_b(actual_kc);
            R::pack_b(attrs, b_src, k, actual_kc, n_start, n_count, pack_b_slice, ldd_b, a_biased);
        }
        matmul_tile<R>(attrs, c_ptr, ldc, a_ptr, lda, m_start, m_count, mc, n_start, n_count, k,
                       actual_kc, last_k, pack_a, pack_b_slice, pack_a_slice, a_biased, ip, bias,
                       col_bias, prepacked);
    }
}


// =========================================================================
//  Plan-driven dispatch — pack + block-granularity parallelism
// =========================================================================
//
// One skeleton for every route: per batch element, split the output on its
// larger dimension and run the blocks on the pool; B hoisting on an M-split is
// a route property.

template <class R>
Status matmul_dispatch_tiled(const MatMulAttributes& attrs, TensorView& output,
                             std::span<const TensorView> inputs, const ComputeContext& ctx,
                             const MatMulPlan& plan, void* workspace)
{
    using Elem = typename R::Elem;
    using Acc = typename R::Acc;

    const auto& a = inputs[0];
    const auto& b = inputs[1];
    const auto [M, N, K] = matmul_dims(attrs, a, b);

    // Only the fp route has a block-packed B; the tag names the format the
    // shared packer decodes.
    const DataType b_block_dt = R::is_fp ? b.data_type() : DataType::f32;

    // Optional fp bias: primed into C by the first k-block, not swept after.
    // With no old C to fold it can seed the accumulator instead, which drops
    // that pass — but the kernel reads a whole panel of lanes, so only a bias
    // whose last panel is full can be read in place.
    const typename R::Elem* bias = nullptr;
    const typename R::Elem* col_bias = nullptr;
    if constexpr (R::is_fp) {
        if (inputs.size() > 2) {
            bias = inputs[2].ptr<typename R::Elem>();
            const int64_t pw = nr_max_flt<typename R::Elem>();
            if (attrs.beta == 0.0f && N % pw == 0) {
                col_bias = bias;
            }
        }
    }

    // The fp routes pack into the scratch; a null buffer here means the pool
    // refused it, and the first pack would fault rather than report. A prepacked
    // RHS with no packed A has no scratch at all.
    if constexpr (R::is_fp) {
        if (plan.workspace_size > 0 && workspace == nullptr) {
            return Status::out_of_memory;
        }
    }

    PackedBView prepacked;
    if (is_packed_matmul_b(b.layout())) {
        // The SME blob is a different walk: 64-column blocks over the whole K.
        const MatMulPackedB bg = b.layout() == TensorLayout::PackedBSme ?
                                     matmul_packed_b_sme_geom(N, K, b.data_type()) :
                                     matmul_packed_b_geom(N, K, b.data_type());
        prepacked = {b.ptr<void>(), bg.ldd, bg.np, bg.seg};
    }

    const int lda = static_cast<int>(a.row_stride_elems());
    const int ldb = static_cast<int>(b.row_stride_elems());
    const int ldc_out = static_cast<int>(output.row_stride_elems());

    // int8 accumulates raw int32: the s32 output in place, or a compact M×N
    // temporary the output pass rewrites — s8/u8 requant, or fp dequant.
    const DataType odt_t = output.data_type();
    const bool out_q = !R::is_fp && (odt_t == DataType::s8 || odt_t == DataType::u8);
    const bool compact_acc = out_q || (!R::is_fp && R::is_dq);
    const int ldc = compact_acc ? static_cast<int>(N) : ldc_out;

    const typename R::AElem* a_base = a.ptr<typename R::AElem>();
    // A's hardware shift (see int8_a_pack_bias): the packers take it, the
    // epilogue takes back whatever the kernels did not fold out. The fused dq
    // route reads fp and quantises, so the target dtype decides.
    bool a_biased = false;
    [[maybe_unused]] int32_t a_bias_offset = 0;
    if constexpr (!R::is_fp) {
        const DataType a_dt = R::a_eff_type(attrs, a);
        a_biased = (int8_a_pack_bias(a_dt) != 0);
        a_bias_offset = int8_epilogue_offset(a_dt);
    }

    // The integer bias: an int32 [N] column vector the output pass adds after
    // the zero-point compensation and before Relu, not scaled by alpha (which
    // only rides the RHS pack). The dequantizing route has no int32 bias — its
    // bias is the fp-typed inputs[2].
    [[maybe_unused]] const int32_t* int_bias = nullptr;
    if constexpr (!R::is_fp && !R::is_dq) {
        if (inputs.size() > 2) {
            int_bias = inputs[2].ptr<int32_t>();
        }
    }

    // plan.ldd_b is the full-Kc stride used for *sizing*; each k-block packs at
    // its own actual_kc stride, which is <= that.
    const int kc = static_cast<int>(plan.kc);
    const int mc = static_cast<int>(plan.mc);
    const int nc = static_cast<int>(plan.nc);
    const int64_t ldd_b_full = plan.ldd_b;
    const int np_full = static_cast<int>(plan.np_slice);

    // Workspace: packed-B panels, then packed A, then the compact accumulator.
    Elem* pack_b_base = static_cast<Elem*>(workspace);
    auto* pack_a_base = reinterpret_cast<Elem*>(static_cast<char*>(workspace) + plan.pack_a_offset);
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
    const int64_t num_blocks = split_n ? split_block_count(Ni, nc) : split_block_count(Mi, mc);

    // The reduction sweep and the output pass exist only to apply quantization.
    // With no zero point, no activation, no integer destination and no A shift
    // left to withdraw, the pass would just copy the raw accumulate and the
    // sweeps are identities — and neither is free: a sweep of A and B up front,
    // then a pass over every element of C.
    //
    // The compensation has two independent halves, each needing its own sweep:
    //   col_term → −(a_bias_offset + zp_a)·r_b[n]   sweeps B
    //   row half → −(r_a − k·zp_a)·zp_b[n]          sweeps A
    // A per-tensor zp_b makes the row half a row constant, folded into one int32
    // per row (r_fold); a per-channel zp_b has no such fold.
    //
    // int8_ep says the output pass runs on the last-k tile; the sweep flags say
    // which reductions feed it. They differ on the dq route, whose output pass
    // is the dequant multiply and always runs, so a sweep is paid only when its
    // half is live.
    bool int8_ep = false;
    bool col_term = false; // the column half is non-zero for some row
    bool sweep_a = false;  // Σ_k A[m,k] is needed
    bool sweep_b = false;  // Σ_k B[k,n] is needed
    bool fold_row = false; // the row half is folded into r_fold
    PoolArray<int32_t> r_a, r_b, r_fold;
    if constexpr (!R::is_fp) {
        const QuantParams& qa = a.quant_params();
        const QuantParams& qb = b.quant_params();
        col_term =
            any_zp_differs(qa.zero_point_data, qa.zero_point, qa.num_scales, Mi, -a_bias_offset);
        const bool row_half = any_zp_nonzero(qb.zero_point_data, qb.zero_point, qb.num_scales, Ni);
        // One int32 per row needs one zp_b: a per-channel array folds only when
        // every channel shares its value, which is not worth scanning for.
        fold_row = row_half && qb.zero_point_data == nullptr;
        sweep_a = row_half;
        sweep_b = col_term;
        if constexpr (R::is_dq) {
            int8_ep = true;
            if constexpr (R::fused_a) {
                // matmul_dq only fuses the identity case: the pack has no way
                // to accumulate r_a, and A_q (which the sweep reads) is gone.
                NNOPS_ASSERT(!sweep_a && !sweep_b);
            }
        } else {
            // The output pass has its own reasons to run when no sweep does:
            // an integer destination, the bias and Relu live there. An integer
            // destination alone is no reason to sweep — a symmetric s8 W8A8
            // setup requantizes the raw accumulate.
            int8_ep = sweep_a || sweep_b || out_q || int_bias != nullptr ||
                      epilogue_is_active(attrs.epilogue);
        }
        if (sweep_a) {
            r_a.reset(static_cast<size_t>(Mi));
        }
        if (sweep_b) {
            r_b.reset(static_cast<size_t>(Ni));
        }
        if (fold_row) {
            r_fold.reset(static_cast<size_t>(Mi));
        }
    }

    // The int8 output pass rides on the last k-block's tile, exactly where fp
    // runs its epilogue, so C is never swept twice; its full-K inputs are ready
    // before the blocks start, only the destination moves per batch.
    using EpParams = typename R::EpilogueParams;
    EpParams ip{};
    if constexpr (!R::is_fp) {
        if (int8_ep) {
            const QuantParams& qa = a.quant_params();
            const QuantParams& qb = b.quant_params();
            ip.r_a = sweep_a ? r_a.data() : nullptr;
            ip.r_b = sweep_b ? r_b.data() : nullptr;
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
                ip.compensate = sweep_a && !fold_row; // row half, per element
                ip.col_term = col_term;
                ip.row_off = fold_row ? r_fold.data() : nullptr;
                ip.act_u8 = R::a_eff_type(attrs, a) == DataType::u8;
                ip.ep = &attrs.epilogue;
                // The int8 accumulator is its own buffer, so the destination
                // still holds the residual when the pass writes it.
                ip.residual = attrs.residual_pre_activation;
                // The dequantize pass is this route's accumulator; the bias
                // enters there, between product and clip.
                if (inputs.size() > 2) {
                    ip.col_bias = inputs[2].ptr<typename R::OutElem>();
                }
            } else {
                const QuantParams& qc = output.quant_params();
                ip.out_scale = quant_scale_at(qc, 0); // requant only
                ip.zp_out = quant_zp_at(qc, 0);
                // The epilogue's Requantize stage carries a per-row table; it
                // overrides the destination's per-tensor parameters.
                if (attrs.epilogue.quant == EpilogueQuantType::Requantize &&
                    attrs.epilogue.quant_param_count > 0) {
                    ip.row_out_scales = attrs.epilogue.quant_scales;
                    ip.row_out_zps = attrs.epilogue.quant_zero_points;
                    ip.out_scale = epilogue_quant_scale(attrs.epilogue, 0);
                    ip.zp_out = epilogue_quant_zero(attrs.epilogue, 0);
                }
                ip.relu = attrs.epilogue.type == EpilogueActivateType::Relu;
                ip.compensate = sweep_a && !fold_row; // row half, per element
                ip.col_term = col_term;
                ip.col_bias = int_bias; // per-column: MatMul's [N] bias
                // The folded row half is an int32 added before the requant —
                // exactly the per-row bias slot, free here because MatMul's own
                // bias is per-column.
                ip.bias = fold_row ? r_fold.data() : nullptr;
            }
        }
    }
    const EpParams* ip_or_null = int8_ep ? &ip : nullptr;

    // A prepacked B has no raw matrix left to sweep, so its column sums are read
    // out of the panels themselves — once, since B does not vary per batch.
    const bool sweep_b_raw = sweep_b && !prepacked.ok();
    if constexpr (!R::is_fp) {
        if (sweep_b && prepacked.ok()) {
            matmul_packed_b_i8_col_sums(b, Ki, r_b.data(), ctx);
        }
    }

    for (int64_t bi = 0; bi < batch.total; ++bi) {
        int64_t a_offset, b_offset, c_offset;
        batch.offsets(bi, output, a_offset, b_offset, c_offset);

        const typename R::AElem* a_p = a_base + a_offset;
        const typename R::B b_src = R::b_source(b, b_offset, ldb, b_block_dt);
        Acc* c_p = R::c_base(output, c_offset, accum);

        if constexpr (!R::is_fp) {
            // C starts at zero. With beta 0 the first k-block runs in zero_mode
            // and overwrites every tile, so only a beta-scaled C needs the
            // pass — it is a whole additional write of the M×N accumulator, 6%
            // of a 1024^3 int8 product. (fp needs no pass either way: its first
            // k-block is zero_mode whenever there is no bias to prime.)
            if (attrs.beta != 0.0f) {
                R::zero_c(c_p, ldc, Mi, Ni);
            }
            if constexpr (!R::fused_a) {
                if (sweep_a || sweep_b_raw) {
                    compute_int8_reductions(a_p, R::a_eff_type(attrs, a) == DataType::u8, lda,
                                            b_src.ptr, ldb, attrs, M, N, K,
                                            sweep_a ? r_a.data() : nullptr,
                                            sweep_b_raw ? r_b.data() : nullptr, ctx);
                }
            }
            if (int8_ep) {
                if constexpr (R::is_dq) {
                    ip.out = output.ptr<typename R::OutElem>() + c_offset;
                    // Per-token arrays carry one entry per global row
                    // (num_scales == total_rows()); the tile indexes rows
                    // within the batch, and a compact A_q makes the batch
                    // offset a whole number of rows.
                    const QuantParams& qa_b = a.quant_params();
                    const int64_t row_base = (a_offset > 0 && lda > 0) ? a_offset / lda : 0;
                    ip.s_a = qa_b.scale_data ? qa_b.scale_data + row_base : nullptr;
                    ip.zp_a = qa_b.zero_point_data ? qa_b.zero_point_data + row_base : nullptr;
                } else if (out_q) {
                    const bool as_u8 = (odt_t == DataType::u8);
                    ip.out = as_u8 ? nullptr : output.ptr<int8_t>() + c_offset;
                    ip.out_u8 = as_u8 ? output.ptr<uint8_t>() + c_offset : nullptr;
                }
                if (fold_row) {
                    // −zp_b·(r_a[m] − k·zp_a[m]) for this batch's rows, read
                    // exactly as the epilogue would, so the fold is
                    // bit-identical to the per-element term; a per-tensor zp_b
                    // is the whole reason it exists.
                    for (int64_t m = 0; m < Mi; ++m) {
                        const int32_t zp_a_m = int8_zp_at(ip.zp_a, ip.zp_a_scalar, m);
                        r_fold[static_cast<size_t>(m)] =
                            -ip.zp_b_scalar * (r_a[static_cast<size_t>(m)] - ip.k * zp_a_m);
                    }
                }
            }
        }

        // A block-packed B is decoded, not copied, so one full-N pack per
        // k-block can serve every m-block. int8 always hoists; fp does it only
        // for a block-packed B (dense B's copy is cheap and its full-N slice is
        // large). The ZA route opts out: its kc makes that slice tens of MB.
        const bool hoist_b =
            R::hoists_b_m_split ||
            (R::is_fp && R::hoists_block_b && is_block_packed_dtype(b_block_dt));

        if (split_n) {
            // NKM: one n-block per parallel task, every block over all of M.
            ctx.cpu.run(0, num_blocks, [&](int64_t blk) {
                const auto nr = n_range(plan, blk, Ni);
                const int64_t slot = block_slot(plan.num_slots, num_blocks, blk, ctx);
                matmul_block<R>(attrs, c_p, ldc, a_p, lda, b_src, Ki, kc, mc, 0, Mi, nr.start,
                                nr.count, pack_b_base + slot * np_full * ldd_b_full,
                                pack_a_slot(plan, num_blocks, blk, ctx, pack_a_base), plan.pack_a,
                                a_biased, ip_or_null, bias, col_bias, prepacked);
            });
        } else if (hoist_b) {
            // MKN, hoisted: pack the full-N B once per k-block, then split the
            // rows. The one packed-B slice serves every m-block.
            for (int k = 0; k < Ki; k += kc) {
                const int actual_kc = std::min(kc, Ki - k);
                const bool last_k = (k + kc >= Ki);
                if (!prepacked.ok()) {
                    R::pack_b(attrs, b_src, k, actual_kc, 0, Ni, pack_b_base, R::ldd_b(actual_kc),
                              a_biased);
                }
                ctx.cpu.run(0, num_blocks, [&](int64_t blk) {
                    const auto mr = m_range(plan, blk, num_blocks, Mi);
                    matmul_tile<R>(attrs, c_p, ldc, a_p, lda, mr.start, mr.count, mc, 0, Ni, k,
                                   actual_kc, last_k, plan.pack_a, pack_b_base,
                                   pack_a_slot(plan, num_blocks, blk, ctx, pack_a_base), a_biased,
                                   ip_or_null, bias, col_bias, prepacked);
                });
            }
        } else {
            // MKN: every m-block packs its own full-N B per k-block.
            ctx.cpu.run(0, num_blocks, [&](int64_t blk) {
                const auto mr = m_range(plan, blk, num_blocks, Mi);
                const int64_t slot = block_slot(plan.num_slots, num_blocks, blk, ctx);
                matmul_block<R>(attrs, c_p, ldc, a_p, lda, b_src, Ki, kc, mc, mr.start, mr.count, 0,
                                Ni, pack_b_base + slot * np_full * ldd_b_full,
                                pack_a_slot(plan, num_blocks, blk, ctx, pack_a_base), plan.pack_a,
                                a_biased, ip_or_null, bias, col_bias, prepacked);
            });
        }
    }
    return Status::ok;
}

/// The plan's scratch plus whatever the route appends to it, then the shared
/// skeleton: batch walk, block grid, per-route scratch slots.
template <class R>
Status run_route(const MatMulAttributes& attrs, TensorView& output,
                 std::span<const TensorView> inputs, const ComputeContext& ctx,
                 const MatMulPlan& plan)
{
    PoolPtr scratch(static_cast<size_t>(plan.workspace_size) + R::extra_workspace(output));
    return matmul_dispatch_tiled<R>(attrs, output, inputs, ctx, plan, scratch.get());
}

/// The plan for one call: worker ids turn scratch slots per m-block into slots
/// per thread.
inline MatMulPlan make_plan(const MatMulAttributes& attrs, const TensorView& a, const TensorView& b,
                            const ComputeContext& ctx)
{
    return get_matmul_plan(attrs, a.desc(), b.desc(), ctx.cpu.worker_count(),
                           ctx.cpu.thread_slot_count() > 0);
}

} // anonymous namespace

} // namespace nnops::backend::cpu
