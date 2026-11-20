/// @file matmul_i8.cpp
/// @brief The quantized MatMul routes — s8 weights against an integer or
/// fp-quantized activation.
///
/// Same skeleton, plan and scratch layout as the float route
/// (matmul_tile_impl.h); everything dtype-specific is named here.
///
///   - I8Route      s8|u8 A × s8 B, raw int32 through the requantizing output
///                  pass (or an s32 destination straight into C);
///   - DqRoute      the same accumulate dequantized into fp — A already s8, or
///                  an fp A quantized inside the pack (fused), so no A_q is
///                  materialized for an M-split;
///   - SmeI8Route   the AArch64 ZA tiles, on the same skeleton with the ZA pack
///                  and kernel (FEAT_SME only).
///
/// int8 accumulates raw int32 and recovers the product in one output pass on the
/// last k-block's tile, while C is still in cache. That pass needs the two halves
/// of the zero-point compensation, each fed by its own reduction sweep (r_a over
/// A, r_b over B); a half that is zero skips its sweep.
///
/// The fp-activation entry point materializes A_q, then runs the int8 GEMM with
/// the dequantizing output pass — up to two steps the caller never sees.

#include "matmul_tile_impl.h" // skeleton, plan plumbing, route view
#include "matmul_internal.h"  // dims, batch walk, the quantized entry points
#include "backend/cpu/simd_kernel/simd_int8_reduce.hpp"
#include "backend/cpu/simd_kernel/simd_quant.hpp"
#include "backend/cpu/common/memory_pool.hpp"
#if defined(NNOPS_ARCH_SME)
#include "backend/cpu/aarch64/sme/mma_sme.hpp" // available_sme — the ZA route gate
#endif
#include "nnops/detail/half.hpp"
#include "nnops/detail/assert.hpp"

#include <algorithm>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>

namespace nnops::backend::cpu {

namespace {


/// Bytes of the compact M×N int32 accumulator an int8 output pass reads.
inline size_t accum_bytes(const TensorView& output) noexcept
{
    const auto od = output.desc();
    return static_cast<size_t>(od.dims[static_cast<size_t>(od.rank - 2)]) *
           static_cast<size_t>(od.dims[static_cast<size_t>(od.rank - 1)]) * sizeof(int32_t);
}

/// int8 (s8|u8 A × s8 B): A always packed, raw int32 C, one output pass.
struct I8Route {
    using Elem = int8_t;
    using AElem = int8_t;
    using Acc = int32_t;
    using Clamp = int32_t;

    static constexpr bool is_fp = false;
    static constexpr bool is_dq = false;
    static constexpr bool fused_a = false;
    static constexpr bool has_direct_a = false;
    static constexpr bool hoists_b_m_split = true; // one B pack per k-block

    using EpilogueParams = Int8EpilogueParams;

    static const MatmulGeom& geom() { return GEOM_I8; }

    struct B {
        const int8_t* ptr;
        int ldb;
    };

    static B b_source(const TensorView& b, int64_t b_off, int ldb, DataType /*block_dt*/)
    {
        return {b.ptr<int8_t>() + b_off, ldb};
    }

    static int ldd_b(int kc) { return static_cast<int>(geom().ldd_b(kc)); }
    static int ldd_a(int kc) { return static_cast<int>(geom().ldd_a(kc)); }

    /// One RHS panel — the unit a prepacked B is walked in.
    static constexpr int b_panel_w() { return NR_MAX_I8; }

    /// B pitch the pack wrote; 0 is the kernel's own slot, the per-call form.
    static constexpr int64_t ldb_packed = 0;

    /// The blob's panel for (k block, panel) and the pitch to walk it: segment
    /// `k / seg` owns a contiguous run of np panels, because the correction
    /// block a kernel reads at kc = seg lives inside each panel.
    static std::pair<const int8_t*, int64_t> prepacked_addr(const PackedBView& pb, int64_t panel0,
                                                            int k)
    {
        const int64_t seg = k / pb.seg;
        return {static_cast<const int8_t*>(pb.base) + (seg * pb.np + panel0) * pb.ldd, pb.ldd};
    }

    static std::pair<Clamp, Clamp> clamp(const MatMulAttributes&, bool)
    {
        return {INT32_MIN, INT32_MAX};
    }

    static void pack_a(bool trans, int mc, int kc, Elem* dst, int ldd, const AElem* src, int lds,
                       bool a_biased, const EpilogueParams* /*ip*/, int /*m*/)
    {
        tile_pack_lhs_i8(trans, mc, kc, dst, ldd, src, lds, a_biased);
    }

    /// The A dtype the int8 kernels see; only the fused dq route differs.
    static DataType a_eff_type(const MatMulAttributes& /*attrs*/, const TensorView& a)
    {
        return a.data_type();
    }

    static void pack_b(const MatMulAttributes& attrs, const B& b, int k, int kc, int n_start,
                       int nc, Elem* dst, int ldd, bool a_biased)
    {
        const Elem* src =
            attrs.transpose_b ? b.ptr + n_start * b.ldb + k : b.ptr + k * b.ldb + n_start;
        tile_pack_rhs_i8(attrs.transpose_b, nc, kc, dst, ldd, src, b.ldb, a_biased);
    }

    static void mma_pack(int mc, int nc, int kc, Acc* c, int ldc, const Elem* pa, const Elem* pb,
                         Clamp lo, Clamp hi, bool zero_mode, int64_t ldb = 0,
                         const Elem* /*col_bias*/ = nullptr)
    {
        tile_mma_pack_i8(mc, nc, kc, c, ldc, pa, pb, static_cast<int>(ldb), lo, hi, zero_mode);
    }

    /// The s32 output, or the M×N temporary an s8/u8 output requantizes from.
    static Acc* c_base(TensorView& output, int64_t c_off, void* accum)
    {
        const DataType dt = output.data_type();
        return (dt == DataType::s8 || dt == DataType::u8) ? static_cast<Acc*>(accum) :
                                                            output.ptr<Acc>() + c_off;
    }

    static void zero_c(Acc* c, int ldc, int M, int N)
    {
        for (int m = 0; m < M; ++m) {
            std::fill(c + m * ldc, c + m * ldc + N, 0);
        }
    }

    /// An s8/u8 output only: the accumulator appended after the plan's scratch.
    static size_t extra_workspace(const TensorView& output)
    {
        const DataType dt = output.data_type();
        return (dt == DataType::s8 || dt == DataType::u8) ? accum_bytes(output) : 0;
    }

    static void epilogue_tile(const EpilogueParams& p, Acc* c, int ldc, int64_t m0, int n0,
                              int64_t mc, int64_t nc)
    {
        int8_epilogue_tile(p, c, ldc, m0, n0, mc, nc);
    }
};

/// The int8 GEMM with a dequantizing fp output pass.
template <class TOut, class AIn = int8_t> struct DqRoute : I8Route {
    static constexpr bool is_dq = true;

    /// A arrives as fp and is quantised in the pack, so no A_q is materialised.
    static constexpr bool fused_a = !std::is_same_v<AIn, int8_t>;

    using OutElem = TOut;
    using AElem = AIn;
    using EpilogueParams = DqEpilogueParams<TOut>;

    static DataType a_eff_type(const MatMulAttributes& attrs, const TensorView& a)
    {
        if constexpr (fused_a) {
            return attrs.act_quant_dtype;
        } else {
            return a.data_type();
        }
    }

    /// Fused: quantise each panel row from fp with its offline parameters.
    static void pack_a(bool trans, int mc, int kc, int8_t* dst, int ldd, const AElem* src, int lds,
                       bool a_biased, const EpilogueParams* ip, int m)
    {
        if constexpr (fused_a) {
            NNOPS_ASSERT(!trans);
            tile_pack_quant_lhs_i8(mc, kc, dst, ldd, src, lds, ip->s_a ? ip->s_a + m : nullptr,
                                   ip->zp_a ? ip->zp_a + m : nullptr, ip->s_a_scalar,
                                   ip->zp_a_scalar, ip->act_u8, a_biased);
        } else {
            tile_pack_lhs_i8(trans, mc, kc, dst, ldd, src, lds, a_biased);
        }
    }

    /// Always the compact accumulator: the fp output is written, not accumulated.
    static Acc* c_base(TensorView& /*output*/, int64_t /*c_off*/, void* accum)
    {
        return static_cast<Acc*>(accum);
    }

    /// The dequant pass is unconditional — it is what produces the output.
    static constexpr bool always_epilogue = true;

    static size_t extra_workspace(const TensorView& output) { return accum_bytes(output); }

    static void epilogue_tile(const EpilogueParams& p, Acc* c, int ldc, int64_t m0, int n0,
                              int64_t mc, int64_t nc)
    {
        dq_epilogue_tile<TOut>(p, c, ldc, m0, n0, mc, nc);
    }
};

#if defined(NNOPS_ARCH_SME)
// =========================================================================
//  SME route variant — the same skeleton, the ZA tiles' pack layout
// =========================================================================
//
// Everything but the packs, the MMA and the B blob's walk comes from I8Route:
// the batch walk, the zero-point compensation and the requantize are the
// planned route's. A is always packed, so there is no direct-A arm. The float
// ZA route (SmeFpRoute) lives with the float family.

/// The int8 SMOPA kernel's block: 64 columns.
constexpr int kSmeI8PanelN = 64;

struct SmeI8Route : I8Route {
    static int ldd_b(int kc) { return static_cast<int>(sme_b_panel_stride(DataType::s8, kc)); }
    static int ldd_a(int kc) { return static_cast<int>(sme_a_panel_stride(DataType::s8, kc)); }

    static constexpr int b_panel_w() { return kSmeI8PanelN; }
    static constexpr int64_t ldb_packed = 0;

    static std::pair<const int8_t*, int64_t> prepacked_addr(const PackedBView& pb,
                                                            int64_t panel0, int k)
    {
        return {static_cast<const int8_t*>(pb.base) + panel0 * pb.ldd +
                    sme_b_k_offset(DataType::s8, k),
                pb.ldd};
    }

    static void pack_a(bool trans, int mc, int kc, Elem* dst, int /*ldd*/, const AElem* src,
                       int lds, bool a_biased, const EpilogueParams* /*ip*/, int /*m*/)
    {
        tile_pack_lhs_i8_sme(trans, mc, kc, dst, src, lds, a_biased);
    }

    static void pack_b(const MatMulAttributes& attrs, const B& b, int k, int kc, int n_start,
                       int nc, Elem* dst, int /*ldd*/, bool /*a_biased*/)
    {
        const Elem* src = attrs.transpose_b ? b.ptr + n_start * b.ldb + k
                                            : b.ptr + k * b.ldb + n_start;
        tile_pack_rhs_i8_sme(attrs.transpose_b, nc, kc, dst, src, b.ldb);
    }

    static void mma_pack(int mc, int nc, int kc, Acc* c, int ldc, const Elem* pa, const Elem* pb,
                         Clamp lo, Clamp hi, bool zero_mode, int64_t ldb = 0,
                         const Elem* /*col_bias*/ = nullptr)
    {
        tile_mma_pack_i8_sme(mc, nc, kc, c, ldc, pa, pb, ldb, lo, hi, zero_mode);
    }
};
#endif  // NNOPS_ARCH_SME

} // anonymous namespace

// =========================================================================
//  int8 — reduction and epilogue passes around the raw accumulate
// =========================================================================

/// Raw int8 reductions for the output pass: r_a[m] = Σ_k A[m,k], r_b[n] =
/// Σ_k B[k,n] (the transposes mirror them). Disjoint slabs, so no atomics; the
/// kernels are in simd_int8_reduce.hpp. @p a_u8 sums A unsigned; a null
/// @p r_a / @p r_b skips that sweep.
void compute_int8_reductions(const int8_t* a_ptr, bool a_u8, int64_t lda, const int8_t* b_ptr,
                             int64_t ldb, const MatMulAttributes& attrs, int64_t M, int64_t N,
                             int64_t K, int32_t* r_a, int32_t* r_b, const ComputeContext& ctx)
{
    // `rows` is the reduction direction: r[i] = Σ_j p[i*ld+j] when the K axis
    // is contiguous (the transposed layout), else a column slab.
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

namespace {

/// Whether the dq compensation is not the identity, i.e. the sweep must run.
inline bool dq_needs_compensation(const TensorView& a, const TensorView& b, int32_t a_bias_offset,
                                  int64_t M, int64_t N) noexcept
{
    const QuantParams& qa = a.quant_params();
    const QuantParams& qb = b.quant_params();
    return any_zp_differs(qa.zero_point_data, qa.zero_point, qa.num_scales, M, -a_bias_offset) ||
           any_zp_nonzero(qb.zero_point_data, qb.zero_point, qb.num_scales, N);
}

// =========================================================================
//  Offline-quantized activations — quantize A, then ride the int8 GEMM
// =========================================================================

/// Quantize A into A_q (same shape, compact) with A's own per-tensor/per-token
/// parameters. The kernel wants a per-row float zero-point array, so one is
/// materialized per call from a per-token int32 array or PerTensor's scalar;
/// rows split across the pool.
template <class AT>
void quantize_activation(const TensorView& a, TensorView& aq, const ComputeContext& ctx)
{
    const QuantParams& qp = a.quant_params();
    const int64_t rows = a.total_rows();
    const int64_t lda = a.row_stride_elems();
    const int64_t K = a.shape(a.rank() - 1);
    if (rows <= 0 || K <= 0) {
        return;
    }

    PoolArray<float> scale_buf, zero_buf;
    const float* scale_arr = qp.scale_data;
    if (scale_arr == nullptr) {
        scale_buf.assign(static_cast<size_t>(rows), qp.scale);
        scale_arr = scale_buf.data();
    }
    const float* zero_arr = nullptr;
    if (qp.zero_point_data != nullptr) {
        zero_buf.reset(static_cast<size_t>(rows));
        for (int64_t i = 0; i < rows; ++i) {
            zero_buf[static_cast<size_t>(i)] = static_cast<float>(qp.zero_point_data[i]);
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
    uint8_t* d8 = u8 ? aq.ptr<uint8_t>() : nullptr;
    int8_t* ds8 = u8 ? nullptr : aq.ptr<int8_t>();

    const int64_t tasks = std::min<int64_t>(rows, std::max<int64_t>(ctx.cpu.thread_count(), 1));
    ctx.cpu.run(0, tasks, [&](int64_t t) {
        const int64_t r0 = t * rows / tasks;
        const int64_t r1 = (t + 1) * rows / tasks;
        if (r1 <= r0) {
            return;
        }
        const int m = static_cast<int>(r1 - r0);
        const float* scale = scale_arr + r0;
        const float* zero = zero_arr ? zero_arr + r0 : nullptr;
        if (u8) {
            kernel::quant_kernel::quantization<uint8_t>(m, n, d8 + r0 * K, n, src + r0 * lda, sr,
                                                        scale, zero);
        } else {
            kernel::quant_kernel::quantization<int8_t>(m, n, ds8 + r0 * K, n, src + r0 * lda, sr,
                                                       scale, zero);
        }
    });
}

/// The fp-activation entry point: materialize A_q, then run the int8 GEMM with
/// the dequantizing output pass.
///
/// A_q goes to the plan as inputs[0] on purpose: with A's shape and QuantParams
/// in integer storage, the plan, batch walk, lda, reductions and arch pack bias
/// all come out right unchanged.
///
/// Fused A skips A_q entirely: nothing reads it as a whole — no sweep, and each
/// row is packed once on an M-split — so the pack quantises straight from A. An
/// N-split re-packs A per n-block and keeps the materialised form.
template <class T>
Status matmul_dq(const MatMulAttributes& attrs, TensorView& output,
                 std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    NNOPS_ASSERT(attrs.act_quant_dtype == DataType::s8 || attrs.act_quant_dtype == DataType::u8);

    // A_q's shape and parameters with no storage: the plan reads only the
    // description, and the target dtype picks the int8 geometry.
    const TensorView aq_shape(a.shape_span(), attrs.act_quant_dtype, nullptr, TensorLayout::Dense,
                              a.quant_params());
    auto plan = get_matmul_plan(attrs, aq_shape.desc(), b.desc(), ctx.cpu.worker_count(),
                                ctx.cpu.thread_slot_count() > 0);
    NNOPS_ASSERT(plan.mc > 0 && plan.nc > 0 && plan.kc > 0);

    const auto [M, N, K] = matmul_dims(attrs, a, b);
    const int32_t a_bias_offset = int8_epilogue_offset(attrs.act_quant_dtype);
    // The optional bias rides `inputs` through both branches: the fused one
    // passes the array through, the materialised one substitutes A_q for A.
    if (!plan.split_n && !attrs.transpose_a && !dq_needs_compensation(a, b, a_bias_offset, M, N)) {
        PoolPtr scratch(static_cast<size_t>(plan.workspace_size) +
                        DqRoute<T, T>::extra_workspace(output));
        return matmul_dispatch_tiled<DqRoute<T, T>>(attrs, output, inputs, ctx, plan,
                                                    scratch.get());
    }

    PoolPtr aq_buf(static_cast<size_t>(a.numel()));
    TensorView aq(a.shape_span(), attrs.act_quant_dtype, aq_buf.get(), TensorLayout::Dense,
                  a.quant_params());
    quantize_activation<T>(a, aq, ctx);

    PoolPtr scratch(static_cast<size_t>(plan.workspace_size) + DqRoute<T>::extra_workspace(output));
    if (inputs.size() > 2) {
        const TensorView ins[] = {aq, b, inputs[2]};
        return matmul_dispatch_tiled<DqRoute<T>>(attrs, output, ins, ctx, plan, scratch.get());
    }
    const TensorView ins[] = {aq, b};
    return matmul_dispatch_tiled<DqRoute<T>>(attrs, output, ins, ctx, plan, scratch.get());
}

} // anonymous namespace

/// Offline-quantized activations × s8 weights: quantize A with its own
/// per-tensor/per-token parameters, run the int8 GEMM, dequantize into A's
/// dtype. Without active parameters the pair has no defined semantics — a float
/// A is not a quantized tensor to dequantize — so it is rejected rather than
/// reading the int8 weights as floats.
Status matmul_dq_from_fp(const MatMulAttributes& attrs, TensorView& output,
                         std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& a = inputs[0];
    const QuantParams& qp = a.quant_params();
    const bool quantized = qp.is_active() && (qp.granularity == QuantGranularity::PerTensor ||
                                              qp.granularity == QuantGranularity::PerToken);
    if (!quantized) {
        return Status::invalid_argument;
    }
    if (attrs.transpose_a) {
        return Status::unsupported;
    }
    NNOPS_ASSERT(output.data_type() == a.data_type());
    return a.data_type() == DataType::f32 ? matmul_dq<float>(attrs, output, inputs, ctx) :
                                            matmul_dq<half>(attrs, output, inputs, ctx);
}


/// s8/u8 activations × s8 weights. The destination alone picks the variant: raw
/// int32 through the requantizing pass (every destination but s32 takes the
/// compact accumulator after the scratch), or the same accumulate dequantized
/// into fp — DqRoute's A is already s8, so only the output type differs.
Status matmul_int8_tiled(const MatMulAttributes& attrs, TensorView& output,
                         std::span<const TensorView> inputs, const ComputeContext& ctx,
                         const MatMulPlan& plan)
{
    const DataType odt = output.data_type();
    const bool out_fp = odt == DataType::f32 || odt == DataType::f16;
    // The residual folds into the fp output pass; an integral destination has no
    // fp sum to add it to.
    if (attrs.residual_pre_activation && !out_fp) {
        return Status::unsupported;
    }
    NNOPS_ASSERT(plan.mc > 0 && plan.nc > 0 && plan.kc > 0);
    switch (odt) {
    case DataType::f32:
        return run_route<DqRoute<float>>(attrs, output, inputs, ctx, plan);
    case DataType::f16:
        return run_route<DqRoute<half>>(attrs, output, inputs, ctx, plan);
    case DataType::s32:
    case DataType::s8:
    case DataType::u8:
        return run_route<I8Route>(attrs, output, inputs, ctx, plan);
    default:
        return Status::unsupported;
    }
}

/// The quantized family: s8 weights, and an activation that is either already
/// integer (the integer GEMM's operand pair) or fp carrying QuantParams, which
/// the A pack applies on the fly — so both ends of the chain may be fp while the
/// arithmetic is int8.
Status matmul_dispatch_quant(const MatMulAttributes& attrs, TensorView& output,
                             std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];
    const DataType dt_a = a.data_type();
#if defined(NNOPS_ARCH_SME)
    // The integer pair on the ZA tiles, integral destinations only. A raw B
    // keeps the small-shape gate; a prepacked blob has no other reader. The
    // fp-output and fp-A cases stay on the planned engine below, with an SME
    // blob decoded first.
    const DataType odt = output.data_type();
    const bool sme_blob = b.layout() == TensorLayout::PackedBSme;
    if ((dt_a == DataType::s8 || dt_a == DataType::u8) && b.data_type() == DataType::s8 &&
        (odt == DataType::s32 || odt == DataType::s8 || odt == DataType::u8) &&
        !attrs.residual_pre_activation && aarch64::available_sme() &&
        (!is_packed_matmul_b(b.layout()) || sme_blob)) {
        const auto [M, N, K] = matmul_dims(attrs, a, b);
        // A raw B needs four full 64-column ZA blocks and enough K to repay the
        // per-call packing; a prepacked blob already holds B, so the shape gate
        // does not apply. The small-K transpose-conv GEMMs measured ~20-35%
        // behind the planned route at K < 256 (and N < 256 for the same reason).
        const bool wide =
            sme_blob || (M > 4 && N >= 4 * aarch64::sme_nr_max && K >= 256);
        if (wide && K > 0) {
            const MatMulPlan plan =
                get_matmul_plan_sme(attrs, a.desc(), b.desc(), ctx.cpu.worker_count(),
                                    ctx.cpu.thread_slot_count() > 0);
            if (plan.mc > 0) {
                return run_route<SmeI8Route>(attrs, output, inputs, ctx, plan);
            }
        }
    }
#endif
    if (b.layout() == TensorLayout::PackedBSme) {
        // An fp destination decodes the SME blob once — the combination only
        // arises when a weight prepacked for an integral output is dequantized.
        const int64_t N = b.shape(0);
        const int64_t K = matmul_dims(attrs, a, b).K;
        PoolArray<int8_t> dense_buf(static_cast<size_t>(K) * static_cast<size_t>(N));
        matmul_packed_b_sme_i8_unpack(b, K, dense_buf.data(), ctx);
        const int64_t shape[] = {K, N};
        const TensorView dense(shape, DataType::s8, dense_buf.data(), TensorLayout::Dense,
                               b.quant_params());
        MatMulAttributes attrs_n = attrs;
        attrs_n.transpose_b = false;
        const TensorView ins[] = {a, dense};
        if (dt_a == DataType::f32 || dt_a == DataType::f16) {
            return matmul_dq_from_fp(attrs_n, output, ins, ctx);
        }
        const auto plan = make_plan(attrs_n, a, dense, ctx);
        return matmul_int8_tiled(attrs_n, output, ins, ctx, plan);
    }
    // fp activations: quantize (in the pack, or into A_q), then dequantize out.
    if (dt_a == DataType::f32 || dt_a == DataType::f16) {
        return matmul_dq_from_fp(attrs, output, inputs, ctx);
    }
    // Integer activations: the destination dtype picks the output pass.
    const auto plan = make_plan(attrs, a, b, ctx);
    return matmul_int8_tiled(attrs, output, inputs, ctx, plan);
}

} // namespace nnops::backend::cpu
