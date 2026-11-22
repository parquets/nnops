/// @file matmul.cpp
/// @brief Tiled matrix multiplication — the float routes and the entry point.
///
/// The skeleton is one body for every route and lives in matmul_tile_impl.h,
/// as does the plan (get_matmul_plan: NKM on an N-split, MKN on an M-split).
/// B is packed into pooled scratch per k-block unless the caller supplied a
/// prepacked blob, which is walked in place. A is packed per M panel when the
/// plan asks for it — for a transposed A or K past PACK_A_K_THRESHOLD here, and
/// always on int8 — and read through lda otherwise.
///
/// This file owns the float family (FpRoute, plus FpRoute over the AArch64 ZA
/// tiles) and the classification: matmul_dispatch_impl validates the call and
/// forks once on the compute dtype, sending s8 weights to matmul_i8.cpp
/// (matmul_dispatch_quant) and everything else to matmul_dispatch_fp below.
///
/// An optional fp [N] bias reaches the output in the first k-block: with no old
/// C to fold it is the accumulator's initial value (no pass at all), and with a
/// beta·C_old term the tile is primed with both before the MMA.
///
/// f32/f16 below GEMM_FAST_PATH_THRESHOLD MACs, or at M <= 4 (bar a transposed A
/// against a transposed B), and a Q4_1 B at small M, skip the plan and sweep the
/// product in one pass — matmul_small.cpp and matmul_qblock_fused.cpp (see
/// matmul_internal.h).

#include "matmul.h"
#include "matmul_internal.h"  // dims / batch walk / PackedBView, shared declarations
#include "matmul_tile_impl.h" // the plan-driven skeleton both families instantiate
#include "matmul_helper.h"
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"
#include "backend/cpu/common/memory_pool.hpp" // internal scratch-memory pool
#if defined(NNOPS_ARCH_SME)
#include "backend/cpu/aarch64/sme/mma_sme.hpp" // available_sme — the ZA route gate
#endif
#include "nnops/detail/half.hpp"
#include "nnops/detail/assert.hpp"

#include <algorithm>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace nnops::backend::cpu {

namespace {

// =========================================================================
//  Dtype routes — the only place a path names its element types
// =========================================================================
//
// Per k-block pack B, per M panel pack A (or read it raw), and one MMA kernel
// sweeps the N side. Everything dtype-specific is named here.

/// The fp route has no output pass; the tiled kernel still needs a type.
struct NoEpilogueParams {};

template <class T> struct FpRoute {
    using Elem = T;
    using AElem = T; // A's element type; differs from Elem only on the fused dq route
    using Acc = T;
    using Clamp = float;

    static constexpr bool is_fp = true;
    static constexpr bool is_dq = false;
    static constexpr bool fused_a = false;
    // A may be read through lda — except for bf16, whose GEOM packs it always.
    static constexpr bool has_direct_a = !std::is_same_v<T, bfloat16>;
    static constexpr bool hoists_b_m_split = false; // B packs inside each block
    // A block-packed B is decoded, not copied, so a full-N pack per k-block is
    // hoisted above the M split: one decode serves every m-block.
    static constexpr bool hoists_block_b = true;

    using EpilogueParams = NoEpilogueParams;

    static constexpr const MatmulGeom& geom()
    {
        if constexpr (std::is_same_v<T, float>) {
            return GEOM_F32;
        }
#ifdef NNOPS_ARCH_AARCH64
        else if constexpr (std::is_same_v<T, bfloat16>) {
            return GEOM_BF16;
        }
#endif
        else {
            return GEOM_F16;
        }
    }

    /// One batch element's B: a dense view, or a Q4_1 block stream.
    struct B {
        const void* ptr;
        int ldb;
        DataType block_dt; ///< f32/f16 for a dense view
        int64_t row_blocks;

        bool blocked() const noexcept { return is_block_packed_dtype(block_dt); }
    };

    static B b_source(const TensorView& b, int64_t b_off, int ldb, DataType block_dt)
    {
        if (is_block_packed_dtype(block_dt)) {
            return {block_rhs_at(static_cast<const uint8_t*>(b.ptr<void>()), b_off, block_dt), ldb,
                    block_dt, static_cast<int64_t>(ldb)};
        }
        return {static_cast<const void*>(b.ptr<T>() + b_off), ldb,
                std::is_same_v<T, float> ? DataType::f32 : DataType::f16, 0};
    }

    static int ldd_b(int kc) { return static_cast<int>(geom().ldd_b(kc)); }
    static int ldd_a(int kc) { return static_cast<int>(geom().ldd_a(kc)); }

    /// Width of one RHS panel — the unit a prepacked B is walked in.
    static constexpr int b_panel_w() { return nr_max_flt<T>(); }

    /// B pitch for the panels this call packed. A prepacked blob walks one
    /// whole-K panel per call, negated so the tile reads it as a blob.
    static constexpr int64_t ldb_packed = -1;

    /// The blob's panel for (k block, panel) and the pitch to walk it: the
    /// k-block's slice of a k-major panel, panels `ldd` apart.
    static std::pair<const T*, int64_t> prepacked_addr(const PackedBView& pb, int64_t panel0, int k)
    {
        return {static_cast<const T*>(pb.base) + panel0 * pb.ldd +
                    static_cast<int64_t>(k) * b_panel_w(),
                -pb.ldd};
    }

    static std::pair<Clamp, Clamp> clamp(const MatMulAttributes& attrs, bool last_k)
    {
        return kblock_clamp(attrs, last_k);
    }

    static void pack_a(bool trans, int mc, int kc, Elem* dst, int ldd, const AElem* src, int lds,
                       bool /*a_biased*/, const EpilogueParams* /*ip*/, int /*m*/)
    {
        tile_pack_lhs(trans, mc, kc, dst, ldd, src, lds, 1.0f);
    }

    static void pack_b(const MatMulAttributes& attrs, const B& b, int k, int kc, int n_start,
                       int nc, Elem* dst, int ldd, bool /*a_biased*/)
    {
        // k and n_start are element indices even for a block-packed B; only
        // its row stride counts blocks.
        if (b.blocked()) {
            // The destination type picks the panel layout: f32/f16 the uniform
            // [k][nr] rectangle, bf16 the BFMMLA operand layout (see
            // pack_q4_rhs); b.block_dt picks Q4_0 or Q4_1.
            tile_pack_rhs_q4(attrs.transpose_b, n_start, nc, k, kc, dst, ldd, b.ptr, b.row_blocks,
                             attrs.alpha, b.block_dt);
            return;
        }
        const Elem* base = static_cast<const Elem*>(b.ptr);
        const Elem* src = attrs.transpose_b ? base + n_start * b.ldb + k // N×K: row n, col k
                                              :
                                              base + k * b.ldb + n_start; // K×N: row k, col n
        // alpha rides the RHS pack: every k-block scales, so the sum does too.
        tile_pack_rhs(attrs.transpose_b, nc, kc, dst, ldd, src, b.ldb, attrs.alpha);
    }

    /// @p ldb -1: the panels this call packed; below that, a prepacked blob's
    /// panel stride negated (only `tile_mma_direct` also takes B in place, which
    /// no MatMul route does). @p col_bias, when set, becomes the accumulator's
    /// initial value instead of C being read: the first k-block builds the tile
    /// from it.
    static void mma_pack(int mc, int nc, int kc, Acc* c, int ldc, const Elem* pa, const Elem* pb,
                         Clamp lo, Clamp hi, bool zero_mode, int64_t ldb = -1,
                         const Elem* col_bias = nullptr)
    {
        tile_mma_pack(mc, nc, kc, c, ldc, pa, pb, ldb, lo, hi, zero_mode, nullptr, col_bias);
    }

    static void mma_direct(int mc, int nc, int kc, Acc* c, int ldc, const Elem* a, int lda,
                           const Elem* pb, Clamp lo, Clamp hi, bool zero_mode, int64_t ldb = -1,
                           const Elem* col_bias = nullptr)
    {
        tile_mma_direct(mc, nc, kc, c, ldc, a, lda, pb, ldb, nullptr, lo, hi, zero_mode, nullptr,
                        col_bias);
    }

    /// The output is C itself: nothing is appended to the plan's scratch.
    static constexpr size_t extra_workspace(const TensorView&) { return 0; }

    /// C row of a batch element plus the two fp-only in-place passes.
    static Acc* c_base(TensorView& output, int64_t c_off, void* /*accum*/)
    {
        return output.ptr<T>() + c_off;
    }
    static void scale(Acc* c, int ldc, float beta, int mc, int nc)
    {
        tile_scale(c, ldc, beta, mc, nc);
    }
    static void scale_add_bias(Acc* c, int ldc, float beta, const Elem* bias, int mc, int nc)
    {
        tile_scale_add_bias(c, ldc, beta, bias, mc, nc);
    }
    static void epilogue(const MatMulAttributes& attrs, Acc* c, int ldc, int mc, int nc)
    {
        epilogue_inplace(mc, nc, c, ldc, static_cast<const Elem*>(nullptr), attrs.epilogue);
    }
};

#if defined(NNOPS_ARCH_SME)
// =========================================================================
//  SME route variants — the same skeleton, the ZA tiles' pack layouts
// =========================================================================
//
// Everything but the packs, the MMA and the B blob's walk comes from FpRoute:
// the batch walk, beta and bias, and the fp epilogue are the planned route's.
// A is always packed, so there is no direct-A arm. The int8 ZA route
// (SmeI8Route) lives with the quantized family.

/// One float ZA B block: 32 columns (the 2x2 tile). The prepacked blob is
/// walked in these units.
constexpr int kSmeFpPanelN = 32;

template <class T> struct SmeFpRoute : FpRoute<T> {
    using B = typename FpRoute<T>::B;

    static constexpr bool has_direct_a = false;
    // kc is 1024 here, so the full-N B slice is tens of MB; the ZA plan keeps
    // packing per m-block instead.
    static constexpr bool hoists_block_b = false;

    static int ldd_b(int kc)
    {
        return static_cast<int>(sme_b_panel_stride(
            std::is_same_v<T, float> ? DataType::f32 : DataType::f16, kc));
    }
    static int ldd_a(int kc)
    {
        return static_cast<int>(sme_a_panel_stride(
            std::is_same_v<T, float> ? DataType::f32 : DataType::f16, kc));
    }

    static constexpr int b_panel_w() { return kSmeFpPanelN; }
    static constexpr int64_t ldb_packed = 0;

    static std::pair<const T*, int64_t> prepacked_addr(const PackedBView& pb, int64_t panel0,
                                                       int k)
    {
        const DataType dt = std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
        return {static_cast<const T*>(pb.base) + panel0 * pb.ldd + sme_b_k_offset(dt, k),
                pb.ldd};
    }

    static void pack_a(bool trans, int mc, int kc, T* dst, int /*ldd*/, const T* src, int lds,
                       bool /*a_biased*/, const NoEpilogueParams* /*ip*/, int /*m*/)
    {
        tile_pack_lhs_sme(trans, mc, kc, dst, src, lds);
    }

    static void pack_b(const MatMulAttributes& attrs, const B& b, int k, int kc, int n_start,
                       int nc, T* dst, int /*ldd*/, bool /*a_biased*/)
    {
        // A Q4 B is a block stream: the decode fuses into the panel pack, so
        // the kernels still read rows of T.
        if (b.blocked()) {
            tile_pack_rhs_q4_sme(attrs.transpose_b, n_start, nc, k, kc, dst, b.ptr, b.row_blocks,
                                 attrs.alpha, b.block_dt);
            return;
        }
        const T* base = static_cast<const T*>(b.ptr);
        const T* src = attrs.transpose_b ? base + n_start * b.ldb + k : base + k * b.ldb + n_start;
        tile_pack_rhs_sme(attrs.transpose_b, nc, kc, dst, src, b.ldb);
    }

    static void mma_pack(int mc, int nc, int kc, T* c, int ldc, const T* pa, const T* pb, float lo,
                         float hi, bool zero_mode, int64_t ldb = 0, const T* col_bias = nullptr)
    {
        tile_mma_pack_sme(mc, nc, kc, c, ldc, pa, pb, ldb, lo, hi, zero_mode, col_bias);
    }
};
#endif // NNOPS_ARCH_SME

} // anonymous namespace

// =========================================================================
//  Public API
// =========================================================================

namespace {

// =========================================================================
//  Route selection — the only place an operand pair names a route
// =========================================================================
//
// Each helper below instantiates the tiled skeleton once, so the entry point is
// a table of operand pairs and output dtypes rather than a sequence of tests.

/// A prepacked RHS (MatMul/Linear ::prepackWeights) is a float-panel blob —
/// same dtype as A, f32 or f16, alpha 1 (alpha rode the per-call RHS pack, and
/// a load-time blob has no slot for it) — or an int8 one. The segmented int8
/// blob is s8-only and alpha-free: its panel correction block is the +128 A
/// shift, which neither a u8 activation nor a fused fp one would reproduce. The
/// SME int8 blob is raw B in the ZA layout, so it takes u8 too; an fp activation
/// decodes it to a dense matrix for the dequantizing route. Its numel is the one
/// thing that can be wrong — a blob packed against another K — so it is checked
/// rather than trusted.
Status check_packed_b(const MatMulAttributes& attrs, const TensorView& a, const TensorView& b)
{
    const DataType dt_a = a.data_type();
    if (b.data_type() == DataType::s8) {
        const bool sme_blob = b.layout() == TensorLayout::PackedBSme;
        const bool fp_a = dt_a == DataType::f32 || dt_a == DataType::f16;
        const bool int_a = dt_a == DataType::s8 || dt_a == DataType::u8;
        // s8 is the only A the segmented panels' folded shift reproduces.
        if (!(dt_a == DataType::s8 || (sme_blob && (int_a || fp_a)))) {
            return Status::unsupported;
        }
    } else {
        const bool float_blob =
            dt_a == b.data_type() && (dt_a == DataType::f32 || dt_a == DataType::f16 ||
                                      (nnops::detail::bf16_cpu_enabled && dt_a == DataType::bf16));
        if (!float_blob) {
            return Status::unsupported;
        }
        if (attrs.alpha != 1.0f) {
            return Status::unsupported;
        }
    }
    if (!matmul_packed_b_matches(b.desc(), matmul_dims(attrs, a, b).K)) {
        return Status::invalid_argument;
    }
    return Status::ok;
}

/// True for the quantized operand pairs: s8 weights against an activation that
/// is either integer already (s8/u8) or fp and carrying the quantization to
/// apply. Every other pair runs the fp kernels, or has no kernel at all.
inline bool is_quantized_pair(DataType dt_a, DataType dt_b) noexcept
{
    if (dt_b != DataType::s8) {
        return false;
    }
    switch (dt_a) {
    case DataType::f32:
    case DataType::f16:
    case DataType::s8:
    case DataType::u8:
        return true;
    default:
        return false;
    }
}


/// The float family: a dense f32/f16 product, with B the same type or a Q4_1
/// block stream that decodes to it in the RHS pack. On aarch64 bf16 joins
/// both: a dense bf16 product, or a Q4_1 B decoded into the BFMMLA layout.
/// Two routes here never build a plan — the fused Q4_1 decode and the
/// small-product sweep. The AArch64 SME tiles are a third route body, on the
/// same planned skeleton (SmeFpRoute), with a plan of their own geometry.
Status matmul_dispatch_fp(const MatMulAttributes& attrs, TensorView& output,
                          std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];
    const DataType dt_a = a.data_type();
    const DataType dt_b = b.data_type();
    const bool float_a = dt_a == DataType::f32 || dt_a == DataType::f16 ||
                         (nnops::detail::bf16_cpu_enabled && dt_a == DataType::bf16);
    const bool block_b = is_block_packed_dtype(dt_b);
    const bool same_type = dt_b == dt_a;
    // A bf16 B has one packed form, but a Q4_1 B decodes into bf16 on aarch64
    // (tile_pack_rhs_q4's bf16 overload); anything else is not a pair.
    if (dt_a == DataType::bf16 && !same_type && !block_b) {
        return Status::unsupported;
    }

    // A dense f32/f16 product that is small enough (or narrow enough) skips the
    // plan: no pack, no tiles, no scratch. The thresholds live with the kernels,
    // and the route rejects every operand pair that is not a same-type fp one.
    if (matmul_small_fp(attrs, output, inputs, ctx)) {
        return Status::ok;
    }

    // Past those, only f32/f16 have a fused kernel, plus a block-packed B, whose
    // blocks decode to A's type in the RHS pack.
    if (!float_a || (!same_type && !block_b)) {
        return Status::unsupported;
    }

#if defined(NNOPS_ARCH_SME)
    // Dense f32/f16 on the ZA tiles, before the plan, plus a Q4_0/Q4_1 B
    // decoded straight into those panels. The route packs A, so a transposed A
    // is fine; alpha rides the generic RHS pack, which this route does not
    // have — except for a Q4 decode, which folds it into every weight as the
    // NEON route does. A raw B keeps the small-shape gate, but a prepacked blob
    // has no other reader. This runs before the fused qblock route so the ZA
    // tiles own the Q4 shapes they admit.
    const bool sme_blob = b.layout() == TensorLayout::PackedBSme;
    const bool sme_pair = (dt_a == DataType::f32 || dt_a == DataType::f16) &&
                          (same_type || block_b);
    if (sme_pair && (attrs.alpha == 1.0f || block_b) && aarch64::available_sme() &&
        (!is_packed_matmul_b(b.layout()) || sme_blob)) {
        const auto [M, N, K] = matmul_dims(attrs, a, b);
        if ((sme_blob || (M > 4 && N >= 16)) && K > 0) {
            const MatMulPlan plan =
                get_matmul_plan_sme(attrs, a.desc(), b.desc(), ctx.cpu.worker_count(),
                                    ctx.cpu.thread_slot_count() > 0);
            if (plan.mc > 0) {
                return dt_a == DataType::f32 ?
                           run_route<SmeFpRoute<float>>(attrs, output, inputs, ctx, plan) :
                           run_route<SmeFpRoute<half>>(attrs, output, inputs, ctx, plan);
            }
        }
    }
#endif
    // An SME-layout blob has no other reader: the panel route would misread it.
    if (b.layout() == TensorLayout::PackedBSme) {
        return Status::unsupported;
    }

#if defined(NNOPS_ARCH_X86_64) || defined(NNOPS_ARCH_AARCH64)
    // Small-M block quant: decode straight into the product, before the plan
    // and the scratch. On aarch64 the ZA route above owns the shapes it admits
    // (M > 4, N >= 16), where it is several times faster than this one; the
    // fused kernel is the M <= 4 route and the fallback for shapes ZA declines.
    // Only a transposed A is out of reach.
    if (float_a && block_b && !attrs.transpose_a) {
        const auto [M, N, K] = matmul_dims(attrs, a, b);
        if (matmul_qblock_fused(attrs, output, inputs, ctx, static_cast<int>(M),
                                static_cast<int>(N), static_cast<int>(K))) {
            return Status::ok;
        }
    }
#endif

    const auto plan = make_plan(attrs, a, b, ctx);
    if (dt_a == DataType::f32) {
        return run_route<FpRoute<float>>(attrs, output, inputs, ctx, plan);
    }
#ifdef NNOPS_ARCH_AARCH64
    if (dt_a == DataType::bf16) {
        return run_route<FpRoute<bfloat16>>(attrs, output, inputs, ctx, plan);
    }
#endif
    return run_route<FpRoute<half>>(attrs, output, inputs, ctx, plan);
}

/// An optional third input (a per-column bias) rides the routes that write fp
/// output: the fp kernels seed their accumulator with it, the dequantizing route
/// adds it in its output pass. Integer outputs take an int32 bias instead (see
/// matmul_cpu).
///
/// This function only classifies the call — validate a prepacked B, normalize
/// the attributes, pick a family. Each family owns its operands, its plan-free
/// routes and its own rejections: matmul_dispatch_quant and
/// matmul_dispatch_fp above.
Status matmul_dispatch_impl(const MatMulAttributes& attrs, TensorView& output,
                            std::span<const TensorView> inputs, const ComputeContext& ctx,
                            void* /*workspace*/)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    NNOPS_ASSERT(a.rank() >= 2);
    NNOPS_ASSERT(b.rank() >= 2);
    NNOPS_ASSERT(inputs.size() >= 2 && inputs.size() <= 3);

    if (is_packed_matmul_b(b.layout())) {
        const Status st = check_packed_b(attrs, a, b);
        if (!is_ok(st)) {
            return st;
        }
    }

    // Fold pure-clamp activations into min_clip/max_clip, which kblock_clamp
    // (and the Q4_1 routes) read directly. A normalization of the attributes
    // that every route shares, so it happens before the family is picked.
    MatMulAttributes attrs_n = attrs;
    switch (attrs_n.epilogue.type) {
    case EpilogueActivateType::Relu:
        attrs_n.epilogue.min_clip = 0.0f;
        break;
    case EpilogueActivateType::Relu6:
        attrs_n.epilogue.min_clip = 0.0f;
        attrs_n.epilogue.max_clip = 6.0f;
        break;
    default:
        break;
    }

    // The only fork left: quantized compute or float compute. s8 weights are the
    // whole of the first family — the activation may be integer or fp, so the
    // dtype pair alone does not say where the arithmetic happens.
    return is_quantized_pair(a.data_type(), b.data_type()) ?
               matmul_dispatch_quant(attrs_n, output, inputs, ctx) :
               matmul_dispatch_fp(attrs_n, output, inputs, ctx);
}

} // anonymous namespace

Status matmul_cpu(const MatMulAttributes& attrs, TensorView& output,
                  std::span<const TensorView> inputs, const ComputeContext& ctx, void* workspace)
{
    NNOPS_ASSERT(inputs.size() >= 2 && inputs.size() <= 3);
    if (inputs.size() < 3) {
        return matmul_dispatch_impl(attrs, output, inputs, ctx, workspace);
    }

    // An optional [N] bias, typed by the output branch. An fp output (f32/f16)
    // carries an fp bias, consumed where the accumulator is seeded — before the
    // first k-block on the fp kernels, between dequantize and clip on the
    // dequantizing one. An integer output (s32/s8/u8) takes an int32 bias, added
    // by the int8 output pass after the compensation and before Relu. Neither is
    // scaled by alpha, which only rides the RHS pack.
    const TensorView& bias = inputs[2];
    const DataType odt = output.data_type();
    const bool int_out = odt == DataType::s32 || odt == DataType::s8 || odt == DataType::u8;
    if (int_out) {
        NNOPS_ASSERT_MSG(bias.data_type() == DataType::s32,
                         "MatMul: an integer output (s32/s8/u8) takes an int32 bias");
    } else {
        NNOPS_ASSERT_MSG(odt == DataType::f32 || odt == DataType::f16 || odt == DataType::bf16,
                         "MatMul: the bias needs an fp or integer output");
        NNOPS_ASSERT(bias.data_type() == odt);
    }
    NNOPS_ASSERT(bias.numel() >= output.shape(output.rank() - 1));

    return matmul_dispatch_impl(attrs, output, inputs, ctx, workspace);
}

} // namespace nnops::backend::cpu
