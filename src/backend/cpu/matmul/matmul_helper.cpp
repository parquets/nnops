/// @file matmul_helper.cpp
/// @brief Arch-neutral pack/MMA layer between the tiled drivers (MatMul,
/// Conv2D, attention) and the per-arch kernels under x86_64/ and aarch64/ — the
/// float half, plus the geometry and plan every dtype shares.
///
/// The per-arch pack/MMA fork lives here: the tables and TileTraits<> below are
/// selected by NNOPS_ARCH_*, so no caller names an arch.
///
/// Invariants:
///  - A decomposes into the arch's MR cascade (MR_F32 / MR_F16), and the split
///    minimises the number of panels, not the largest first: every panel
///    re-streams all of B, so 8 rows is 4+4 rather than 6+1+1. A short tail rides
///    a wider kernel rather than a narrow one (3 rows through the 4-wide kernel),
///    since the float kernels alias rows past the live count — one redundant row
///    is cheaper than a second B pass. The int8 packers take no live width, so
///    their ladder keeps exact panels (see matmul_helper_i8.cpp).
///  - B is uniform: ceil(nc/nr_max) nr_max-wide panels at one 64-byte-aligned
///    stride. The MMA reads whole lanes, so callers pass the live width and
///    each kernel's N-tail path drops the lanes past it.
///  - Panels sit at the caller's `ldd`, never end to end.
///
/// Two RHS modes: a per-call pack whose stride comes from the k-block length,
/// and a MatMulPackedB blob packed outside the call. Those strides disagree, so
/// the blob route hands the kernel one panel per call and steps panels itself —
/// except on the 1-row rung, where one call takes two panels and reads the
/// caller's stride (a negative ldb below -1) instead of its own.
/// The int8 blob adds a second axis — its panels come in k segments, because the
/// per-panel correction block is read at the call's kc (see matmul_packed_b_geom).
///
/// tile_mma_direct's bias seeds the accumulator instead of reading C: `bias` is
/// one value per output row, `col_bias` one per output column (NHWC
/// Conv2D, where a row is a pixel). Either one builds the tile, so it belongs on
/// the first k-block only.

#include "matmul_helper.h"
#include "matmul_pack_walk.h"

#include <array>

#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/q4_1.hpp"
#include "nnops/detail/simd.hpp"
#include "backend/cpu/aarch64/sme/mma_sme.hpp"
#if defined(NNOPS_ARCH_SME)
#include "backend/cpu/aarch64/sme/mma_f16_sme.hpp"
#include "backend/cpu/aarch64/sme/mma_f32_sme.hpp"
#include "backend/cpu/aarch64/sme/pack_f16_sme.hpp"
#include "backend/cpu/aarch64/sme/pack_f32_sme.hpp"
#endif

#ifdef NNOPS_ARCH_X86_64
using namespace nnops::backend::cpu::x86_64;
#elif defined(NNOPS_ARCH_AARCH64)
using namespace nnops::backend::cpu::aarch64;
#endif

namespace nnops::backend::cpu {

// Trailing live width: one kernel serves a full panel and a tail.

template <class T>
using PackFn = void (*)(T* NNOPS_RESTRICT output, const T* NNOPS_RESTRICT input, int ir_step, int K,
                        float scale, int w);

// Pack tables (largest-first): row 0 = LHS (MR), row 1 = RHS (NR).
constexpr std::array<std::array<PackFn<float>, MR_LEVELS>, 2> pack_trans_f32_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {pack_trans_n6_f32, pack_trans_n4_f32, pack_trans_n2_f32, pack_trans_n1_f32},
    {pack_trans_n16_f32, pack_trans_n8_f32, nullptr, pack_trans_n1_f32},
#elif defined(NNOPS_ARCH_AARCH64)
    {pack_trans_n8_f32, pack_trans_n4_f32, pack_trans_n1_f32, nullptr},
    {pack_trans_n12_f32, pack_trans_n4_f32, nullptr, pack_trans_n1_f32},
#endif
}};

constexpr std::array<std::array<PackFn<float>, MR_LEVELS>, 2> pack_copy_f32_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {pack_copy_n6_f32, pack_copy_n4_f32, pack_copy_n2_f32, pack_copy_n1_f32},
    {pack_copy_n16_f32, pack_copy_n8_f32, nullptr, pack_copy_n1_f32},
#elif defined(NNOPS_ARCH_AARCH64)
    {pack_copy_n8_f32, pack_copy_n4_f32, pack_copy_n1_f32, nullptr},
    {pack_copy_n12_f32, pack_copy_n4_f32, nullptr, pack_copy_n1_f32},
#endif
}};

constexpr std::array<std::array<PackFn<half>, MR_LEVELS>, 2> pack_trans_f16_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {pack_trans_n6_f16, pack_trans_n4_f16, pack_trans_n2_f16, pack_trans_n1_f16},
    {pack_trans_n16_f16, pack_trans_n8_f16, nullptr, pack_trans_n1_f16},
#elif defined(NNOPS_ARCH_AARCH64)
    {pack_trans_n8_f16, pack_trans_n4_f16, pack_trans_n1_f16, nullptr},
    {pack_trans_n16_f16, pack_trans_n8_f16, nullptr, pack_trans_n1_f16},
#endif
}};

constexpr std::array<std::array<PackFn<half>, MR_LEVELS>, 2> pack_copy_f16_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {pack_copy_n6_f16, pack_copy_n4_f16, pack_copy_n2_f16, pack_copy_n1_f16},
    {pack_copy_n16_f16, pack_copy_n8_f16, nullptr, pack_copy_n1_f16},
#elif defined(NNOPS_ARCH_AARCH64)
    {pack_copy_n8_f16, pack_copy_n4_f16, pack_copy_n1_f16, nullptr},
    {pack_copy_n16_f16, pack_copy_n8_f16, nullptr, pack_copy_n1_f16},
#endif
}};

#ifdef NNOPS_ARCH_AARCH64
// bf16 has one ladder and one B layout; the A side is dense rows, so the
// transposing and copying packers differ only in where the k axis comes from.
constexpr std::array<std::array<PackFn<bfloat16>, MR_LEVELS>, 2> pack_trans_bf16_fn = {{
    {pack_trans_n8_bf16, pack_trans_n4_bf16, pack_trans_n1_bf16, nullptr},
    {pack_trans_n16_bf16, nullptr, nullptr, nullptr},
}};

constexpr std::array<std::array<PackFn<bfloat16>, MR_LEVELS>, 2> pack_copy_bf16_fn = {{
    {pack_copy_n8_bf16, pack_copy_n4_bf16, pack_copy_n1_bf16, nullptr},
    {pack_copy_n16_bf16, nullptr, nullptr, nullptr},
}};
#endif

template <class T> struct TileTraits;

template <> struct TileTraits<float> {
    using Pack = PackFn<float>;
    static constexpr const int* mr() { return MR_F32; }
    static constexpr const int* nr() { return NR_F32; }
    static constexpr const auto& trans() { return pack_trans_f32_fn; }
    static constexpr const auto& copy() { return pack_copy_f32_fn; }
    template <bool Z, int S = 0> static constexpr const auto& mma_pack()
    {
        return mma_pack_fns_f32<Z, S>;
    }
    template <bool Z, int S> static constexpr const auto& mma_direct()
    {
        return mma_direct_fns_f32<Z, S>;
    }
};

template <> struct TileTraits<half> {
    using Pack = PackFn<half>;
    static constexpr const int* mr() { return MR_F16; }
    static constexpr const int* nr() { return NR_F16; }
    static constexpr const auto& trans() { return pack_trans_f16_fn; }
    static constexpr const auto& copy() { return pack_copy_f16_fn; }
    template <bool Z, int S = 0> static constexpr const auto& mma_pack()
    {
        return mma_pack_fns_f16<Z, S>;
    }
    template <bool Z, int S> static constexpr const auto& mma_direct()
    {
        return mma_direct_fns_f16<Z, S>;
    }
};

#ifdef NNOPS_ARCH_AARCH64
template <> struct TileTraits<bfloat16> {
    using Pack = PackFn<bfloat16>;
    static constexpr const int* mr() { return MR_BF16; }
    static constexpr const int* nr() { return NR_BF16; }
    static constexpr const auto& trans() { return pack_trans_bf16_fn; }
    static constexpr const auto& copy() { return pack_copy_bf16_fn; }
    template <bool Z, int S = 0> static constexpr const auto& mma_pack()
    {
        return mma_pack_fns_bf16<Z, S>;
    }
    template <bool Z, int S> static constexpr const auto& mma_direct()
    {
        return mma_direct_fns_bf16<Z, S>;
    }
};
#endif

/// 64-byte-aligned panel stride, in elements.
template <class T> constexpr int panel_stride(int rows, int kc)
{
    return static_cast<int>(mma_pack_align(rows * kc * static_cast<int>(sizeof(T))) /
                            static_cast<int>(sizeof(T)));
}


template <class T>
void tile_pack_lhs_impl(bool trans, int mc, int kc, T* dst, int ldd, const T* src, int lds,
                        float scale)
{
    using Tr = TileTraits<T>;
    using Fn = typename Tr::Pack;
    // transposed source is a plain copy; row-major needs the transposing kernel.
    const std::array<Fn, MR_LEVELS>& fns = trans ? Tr::copy()[0] : Tr::trans()[0];
    const int pack_lds = trans ? 1 : lds;

    int i = 0;
    walk_levels(i, mc, Tr::mr(), [&](int level, int s) {
        fns[level](dst, src, lds, kc, scale, s);
        dst += ldd;
        src += s * pack_lds;
    });
}

void tile_pack_lhs(bool trans, int mc, int kc, float* dst, int ldd, const float* src, int lds,
                   float scale)
{
    tile_pack_lhs_impl(trans, mc, kc, dst, ldd, src, lds, scale);
}

void tile_pack_lhs(bool trans, int mc, int kc, half* dst, int ldd, const half* src, int lds,
                   float scale)
{
    tile_pack_lhs_impl(trans, mc, kc, dst, ldd, src, lds, scale);
}

#ifdef NNOPS_ARCH_AARCH64
void tile_pack_lhs(bool trans, int mc, int kc, bfloat16* dst, int ldd, const bfloat16* src, int lds,
                   float scale)
{
    tile_pack_lhs_impl(trans, mc, kc, dst, ldd, src, lds, scale);
}
#endif

// =========================================================================
//  Uniform-width RHS panels
// =========================================================================

template <class T>
void tile_pack_rhs_impl(bool trans, int nc, int kc, T* dst, int ldd, const T* src, int lds,
                        float scale)
{
    using Tr = TileTraits<T>;
    using Fn = typename Tr::Pack;
    constexpr int nr_max = Tr::nr()[0];
    const Fn fn = trans ? Tr::trans()[1][0] : Tr::copy()[1][0];
    // transposed source: a panel is nr_max rows of K; row-major: nr_max columns.
    const int step = trans ? lds : 1;

    for_each_n_panel(nc, nr_max, [&](int w) {
        fn(dst, src, lds, kc, scale, w);
        dst += ldd;
        src += w * step;
    });
}

/// One M panel over B's three forms (see the header): `ldb == -1`, the kernel
/// walks the N side itself over the panels this call packed; `ldb < -1`, one
/// call per nr_max-wide panel of the caller's blob, the panels -ldb apart.
/// `Seed` picks the bias axis: 1 is a per-row vector indexed from @p m, 2 a
/// per-column one that steps with the panel.
template <class T, bool Zero, int Seed>
void tile_mma_pack_impl(int Mc, int Nc, int Kc, T* c, int ldc, const T* packed_a, const T* b,
                        int64_t ldb, float clamp_min, float clamp_max, const T* bias,
                        const T* col_bias)
{
    using Tr = TileTraits<T>;
    const auto& fns = Tr::template mma_pack<Zero, Seed>();
    constexpr int nr = Tr::nr()[0];
    const int ldd_a = panel_stride<T>(Tr::mr()[0], Kc);

    const auto seed_at = [&](int m, int n) {
        return (Seed == 1) ? bias + m : (Seed == 2 ? col_bias + n : nullptr);
    };

    int m = 0;
    walk_levels(m, Mc, Tr::mr(), [&](int level, int s) {
        if (ldb == -1) {
            fns[level](s, Nc, Kc, c + m * ldc, ldc, packed_a, b, clamp_min, clamp_max,
                       seed_at(m, 0));
        } else {
            for (int n = 0; n < Nc; n += nr) {
                const int w = std::min<int>(nr, Nc - n);
                fns[level](s, w, Kc, c + m * ldc + n, ldc, packed_a,
                           b + (static_cast<int64_t>(n) / nr) * -ldb, clamp_min, clamp_max,
                           seed_at(m, n));
            }
        }
        packed_a += ldd_a;
    });
}

/// Same walk, A read raw through @p lda, plus B's third form: `ldb > 0` reads
/// the caller's matrix in place, panel p at b + p·nr_max with rows `ldb` apart.
/// The partial last panel is packed into @p stage, since the kernel reads a
/// panel whole.
template <class T, bool Zero, int Seed>
void tile_mma_direct_impl(int Mc, int Nc, int Kc, T* c, int ldc, const T* a, int lda, const T* b,
                          int64_t ldb, T* stage, float clamp_min, float clamp_max, const T* bias,
                          const T* col_bias)
{
    using Tr = TileTraits<T>;
    const auto& fns = Tr::template mma_direct<Zero, Seed>();
    constexpr int nr = Tr::nr()[0];
    // Every float 1-row kernel takes two packed panels per call.
    constexpr int wide_nr = 2 * nr;

    const auto seed_at = [&](int m, int n) {
        return (Seed == 1) ? bias + m : (Seed == 2 ? col_bias + n : nullptr);
    };

    int m = 0;
    walk_levels(m, Mc, Tr::mr(), [&](int level, int s) {
        const T* const a_m = a + m * lda;
        if (ldb == -1) {
            fns[level](s, Nc, Kc, c + m * ldc, ldc, a_m, lda, b, /*ldb=*/-1, clamp_min, clamp_max,
                       seed_at(m, 0));
        } else if (ldb < 0) {
            // A 1-row panel is a pure weight stream, so its kernel takes two
            // packed panels there and its one A scalar feeds two panel widths;
            // the packing is uniform, so any other rung stays one panel.
            const int step = (Tr::mr()[level] == 1) ? wide_nr : nr;
            const int64_t b_ld = (step > nr) ? ldb : -1;
            for (int n = 0; n < Nc; n += step) {
                const int w = std::min<int>(step, Nc - n);
                fns[level](s, w, Kc, c + m * ldc + n, ldc, a_m, lda,
                           b + (static_cast<int64_t>(n) / nr) * -ldb, b_ld, clamp_min, clamp_max,
                           seed_at(m, n));
            }
        } else {
            const int ld = static_cast<int>(ldb);
            int n = 0;
            for (; n + nr <= Nc; n += nr) {
                fns[level](s, nr, Kc, c + m * ldc + n, ldc, a_m, lda, b + n, ld, clamp_min,
                           clamp_max, seed_at(m, n));
            }
            if (n < Nc) {
                const int w = Nc - n;
                // bf16's in-place read is the caller's own matrix at a row
                // pitch, not the f16-style [k][nr] panel, so its kernel takes
                // the partial column block directly: packing it into `stage`
                // and handing it back at `ldb == nr` would misread the layout
                // (and bf16's panel is ⌈kc/4⌉ groups, wider than nr·kc).
                if constexpr (std::is_same_v<T, bfloat16>) {
                    fns[level](s, w, Kc, c + m * ldc + n, ldc, a_m, lda, b + n, ld, clamp_min,
                               clamp_max, seed_at(m, n));
                } else {
                    tile_pack_rhs(false, w, Kc, stage, nr * Kc, b + n, ld, 1.0f);
                    fns[level](s, w, Kc, c + m * ldc + n, ldc, a_m, lda, stage, nr, clamp_min,
                               clamp_max, seed_at(m, n));
                }
            }
        }
    });
}

template <class T>
void dispatch_mma_pack(bool zero_mode, int Mc, int Nc, int Kc, T* c, int ldc, const T* a,
                       const T* b, int64_t ldb, float lo, float hi, const T* bias,
                       const T* col_bias)
{
    // Either bias builds the tile, so there is nothing in C to accumulate onto.
    if (col_bias != nullptr) {
        tile_mma_pack_impl<T, true, 2>(Mc, Nc, Kc, c, ldc, a, b, ldb, lo, hi, nullptr, col_bias);
    } else if (bias != nullptr) {
        tile_mma_pack_impl<T, true, 1>(Mc, Nc, Kc, c, ldc, a, b, ldb, lo, hi, bias, nullptr);
    } else if (zero_mode) {
        tile_mma_pack_impl<T, true, 0>(Mc, Nc, Kc, c, ldc, a, b, ldb, lo, hi, nullptr, nullptr);
    } else {
        tile_mma_pack_impl<T, false, 0>(Mc, Nc, Kc, c, ldc, a, b, ldb, lo, hi, nullptr, nullptr);
    }
}

template <class T>
void dispatch_mma_direct(bool zero_mode, int Mc, int Nc, int Kc, T* c, int ldc, const T* a, int lda,
                         const T* b, int64_t ldb, T* stage, float lo, float hi, const T* bias,
                         const T* col_bias)
{
    // Either bias builds the tile, so there is nothing in C to accumulate onto.
    if (col_bias != nullptr) {
        tile_mma_direct_impl<T, true, 2>(Mc, Nc, Kc, c, ldc, a, lda, b, ldb, stage, lo, hi, nullptr,
                                         col_bias);
    } else if (bias != nullptr) {
        tile_mma_direct_impl<T, true, 1>(Mc, Nc, Kc, c, ldc, a, lda, b, ldb, stage, lo, hi, bias,
                                         nullptr);
    } else if (zero_mode) {
        tile_mma_direct_impl<T, true, 0>(Mc, Nc, Kc, c, ldc, a, lda, b, ldb, stage, lo, hi, nullptr,
                                         nullptr);
    } else {
        tile_mma_direct_impl<T, false, 0>(Mc, Nc, Kc, c, ldc, a, lda, b, ldb, stage, lo, hi,
                                          nullptr, nullptr);
    }
}

void tile_pack_rhs(bool trans, int nc, int kc, float* dst, int ldd, const float* src, int lds,
                   float scale)
{
    tile_pack_rhs_impl(trans, nc, kc, dst, ldd, src, lds, scale);
}

void tile_pack_rhs(bool trans, int nc, int kc, half* dst, int ldd, const half* src, int lds,
                   float scale)
{
    tile_pack_rhs_impl(trans, nc, kc, dst, ldd, src, lds, scale);
}

#ifdef NNOPS_ARCH_AARCH64
void tile_pack_rhs(bool trans, int nc, int kc, bfloat16* dst, int ldd, const bfloat16* src, int lds,
                   float scale)
{
    tile_pack_rhs_impl(trans, nc, kc, dst, ldd, src, lds, scale);
}
#endif

// ---- Block-packed RHS (q4_1) --------------------------------------------
// The f32/f16 panel layout matches tile_pack_rhs; bf16 uses the BFMMLA layout
// pack_bf16.hpp defines. The MMA route never sees the blocks either way.

static_assert(q4_1_block_elems == 32, "the block RHS packers assume 32-value blocks");

#if defined(NNOPS_ARCH_X86_64) || defined(NNOPS_ARCH_AARCH64)

void tile_pack_rhs_q4_1(bool trans, int n_start, int nc, int k_start, int kc, float* dst, int ldd,
                        const void* src, int64_t src_row_blocks, float scale)
{
    arch::pack_q4_1_rhs<float>(trans, n_start, nc, k_start, kc, dst, ldd, src, src_row_blocks,
                               scale);
}

void tile_pack_rhs_q4_1(bool trans, int n_start, int nc, int k_start, int kc, half* dst, int ldd,
                        const void* src, int64_t src_row_blocks, float scale)
{
    arch::pack_q4_1_rhs<half>(trans, n_start, nc, k_start, kc, dst, ldd, src, src_row_blocks,
                              scale);
}

#ifdef NNOPS_ARCH_AARCH64
void tile_pack_rhs_q4_1(bool trans, int n_start, int nc, int k_start, int kc, bfloat16* dst,
                        int ldd, const void* src, int64_t src_row_blocks, float scale)
{
    arch::pack_q4_1_rhs(trans, n_start, nc, k_start, kc, dst, ldd, src, src_row_blocks, scale);
}
#endif

#else

/// Portable fallback for arches without a vectorized block pack.
///
/// The width must be the arch's float RHS panel width — the workspace stride
/// and the MMA are sized from it — and that is 12 for aarch64 f32, 16 elsewhere.
/// The arch packers write this same layout, but decode a block once per row.
template <class T> constexpr int block_panel_nr() noexcept
{
    if constexpr (std::is_same_v<T, float>) {
        return NR_MAX_F32;
    } else {
        return NR_MAX_F16;
    }
}

/// Values per block, for the indexing below; must equal q4_1_block_elems.
constexpr int kBlockElems = 32;

/// One block-packed format's row decoder: unpack(block_ptr, float out[]).
template <class T, class Unpack>
void tile_pack_rhs_block_scalar(bool trans, int n_start, int nc, int k_start, int kc, T* dst,
                                int ldd, const void* src, int64_t row_blocks, float scale,
                                size_t block_bytes, Unpack unpack)
{
    constexpr int NR = block_panel_nr<T>();
    const uint8_t* base = static_cast<const uint8_t*>(src);
    const int64_t row_bytes = row_blocks * static_cast<int64_t>(block_bytes);

    auto store = [](T* p, float v) {
        if constexpr (std::is_same_v<T, float>) {
            *p = v;
        } else {
            *p = float_to_half(v);
        }
    };

    int n = 0;
    while (n < nc) {
        const int nr = std::min(NR, nc - n);
        for (int k = 0; k < kc; ++k) {
            T* o = dst + static_cast<int64_t>(k) * NR;
            for (int j = 0; j < nr; ++j) {
                const int row = trans ? (n_start + n + j) : (k_start + k);
                const int col = trans ? (k_start + k) : (n_start + n + j);
                float vals[kBlockElems];
                unpack(base + static_cast<int64_t>(row) * row_bytes +
                           static_cast<int64_t>(col / kBlockElems) * block_bytes,
                       vals);
                store(&o[j], vals[col % kBlockElems] * scale);
            }
            for (int j = nr; j < NR; ++j) {
                o[j] = T{};
            }
        }
        dst += ldd;
        n += nr;
    }
}

void tile_pack_rhs_q4_1(bool trans, int n_start, int nc, int k_start, int kc, float* dst, int ldd,
                        const void* src, int64_t src_row_blocks, float scale)
{
    tile_pack_rhs_block_scalar<float>(trans, n_start, nc, k_start, kc, dst, ldd, src,
                                      src_row_blocks, scale, q4_1_block_bytes, q4_1_unpack_block);
}

void tile_pack_rhs_q4_1(bool trans, int n_start, int nc, int k_start, int kc, half* dst, int ldd,
                        const void* src, int64_t src_row_blocks, float scale)
{
    tile_pack_rhs_block_scalar<half>(trans, n_start, nc, k_start, kc, dst, ldd, src, src_row_blocks,
                                     scale, q4_1_block_bytes, q4_1_unpack_block);
}

#endif

void tile_mma_pack(int Mc, int Nc, int Kc, float* c, int ldc, const float* packed_a, const float* b,
                   int64_t ldb, float clamp_min, float clamp_max, bool zero_mode, const float* bias,
                   const float* col_bias)
{
    dispatch_mma_pack(zero_mode, Mc, Nc, Kc, c, ldc, packed_a, b, ldb, clamp_min, clamp_max, bias,
                      col_bias);
}

void tile_mma_pack(int Mc, int Nc, int Kc, half* c, int ldc, const half* packed_a, const half* b,
                   int64_t ldb, float clamp_min, float clamp_max, bool zero_mode, const half* bias,
                   const half* col_bias)
{
    dispatch_mma_pack(zero_mode, Mc, Nc, Kc, c, ldc, packed_a, b, ldb, clamp_min, clamp_max, bias,
                      col_bias);
}

void tile_mma_direct(int Mc, int Nc, int Kc, float* c, int ldc, const float* a, int lda,
                     const float* b, int64_t ldb, float* stage, float clamp_min, float clamp_max,
                     bool zero_mode, const float* bias, const float* col_bias)
{
    dispatch_mma_direct(zero_mode, Mc, Nc, Kc, c, ldc, a, lda, b, ldb, stage, clamp_min, clamp_max,
                        bias, col_bias);
}

void tile_mma_direct(int Mc, int Nc, int Kc, half* c, int ldc, const half* a, int lda,
                     const half* b, int64_t ldb, half* stage, float clamp_min, float clamp_max,
                     bool zero_mode, const half* bias, const half* col_bias)
{
    dispatch_mma_direct(zero_mode, Mc, Nc, Kc, c, ldc, a, lda, b, ldb, stage, clamp_min, clamp_max,
                        bias, col_bias);
}

#ifdef NNOPS_ARCH_AARCH64
void tile_mma_pack(int Mc, int Nc, int Kc, bfloat16* c, int ldc, const bfloat16* packed_a,
                   const bfloat16* b, int64_t ldb, float clamp_min, float clamp_max, bool zero_mode,
                   const bfloat16* bias, const bfloat16* col_bias)
{
    dispatch_mma_pack(zero_mode, Mc, Nc, Kc, c, ldc, packed_a, b, ldb, clamp_min, clamp_max, bias,
                      col_bias);
}

void tile_mma_direct(int Mc, int Nc, int Kc, bfloat16* c, int ldc, const bfloat16* a, int lda,
                     const bfloat16* b, int64_t ldb, bfloat16* stage, float clamp_min,
                     float clamp_max, bool zero_mode, const bfloat16* bias,
                     const bfloat16* col_bias)
{
    dispatch_mma_direct(zero_mode, Mc, Nc, Kc, c, ldc, a, lda, b, ldb, stage, clamp_min, clamp_max,
                        bias, col_bias);
}
#endif

// ---- the blob ---------------------------------------------------------

MatMulPackedB matmul_packed_b_geom(int64_t N, int64_t K, DataType dtype) noexcept
{
    MatMulPackedB g;
    if (N <= 0 || K <= 0) {
        return g;
    }

    if (dtype == DataType::s8) {
        // One slot per panel per k segment: the kernel reads a panel's
        // correction block at 16·⌈kc/4⌉ past its start, i.e. at the call's kc,
        // so a segment is as much as one call can consume. The last segment
        // keeps a whole slot too — the padding after its sums is never read, and
        // a uniform pitch is what lets one kernel call walk a segment's panels.
        g.nr = NR_MAX_I8;
        g.seg = std::min<int64_t>(K, KC_CAP_I8);
        g.ldd = int8_panel_stride(g.seg);
        g.np = (N + g.nr - 1) / g.nr;
        g.total = ((K + g.seg - 1) / g.seg) * g.np * g.ldd;
        return g;
    }

#ifdef NNOPS_ARCH_AARCH64
    if (dtype == DataType::bf16) {
        // Panels are BFMMLA's operand layout, so the slot is the pack's own
        // stride: ⌈K/4⌉ k-groups of 64 elements, independent of K's parity.
        g.nr = NR_MAX_BF16;
        g.ldd = geom_ldd_bf16_b(K);
        g.np = (N + g.nr - 1) / g.nr;
        g.total = g.np * g.ldd;
        return g;
    }
#endif

    if (dtype != DataType::f32 && dtype != DataType::f16) {
        return g;
    }

    const int64_t nr = (dtype == DataType::f16) ? NR_MAX_F16 : NR_MAX_F32;
    const int64_t esz = static_cast<int64_t>(data_type_size(dtype));
    g.nr = nr;
    // The same aligned slot the per-call pack uses, only at the whole K.
    g.ldd = static_cast<int64_t>(
                mma_pack_align(static_cast<size_t>(nr * K) * static_cast<size_t>(esz))) /
            esz;
    g.np = (N + nr - 1) / nr;
    g.total = g.np * g.ldd;
    return g;
}

MatMulPackedB matmul_packed_b_sme_geom(int64_t N, int64_t K, DataType dtype) noexcept
{
    MatMulPackedB g;
    if (N <= 0 || K <= 0 ||
        (dtype != DataType::f32 && dtype != DataType::f16 && dtype != DataType::s8) ||
        !aarch64::available_sme()) {
        return g;
    }
    // One slot per n tile, contiguous, the layout the kernels read in place:
    // f32 [K][32], f16 [ceil(K/2)][64] with each column's K pair adjacent, and
    // s8 [ceil(K/4)][256] with each column's four K values adjacent. The float
    // slots are the 2x2 ZA tile's 32 columns, so a tile's stream is sequential;
    // at 64 the kernel read half of every K row.
    g.nr = dtype == DataType::s8 ? aarch64::sme_nr_max : aarch64::sme_fp_nr;
    if (dtype == DataType::f16) {
        g.ldd = ((K + 1) / 2) * (2 * g.nr);
    } else if (dtype == DataType::s8) {
        g.ldd = ((K + 3) / 4) * (4 * g.nr);
    } else {
        g.ldd = K * g.nr;
    }
    g.np = (N + g.nr - 1) / g.nr;
    g.total = g.np * g.ldd;
    return g;
}

int64_t sme_b_panel_stride(DataType dtype, int64_t kc) noexcept
{
    const int64_t nr = dtype == DataType::s8 ? aarch64::sme_nr_max : aarch64::sme_fp_nr;
    if (dtype == DataType::f32) {
        return kc * nr;
    }
    if (dtype == DataType::f16) {
        return ((kc + 1) / 2) * (2 * nr);
    }
    return ((kc + 3) / 4) * (4 * nr);
}

int64_t sme_a_panel_stride(DataType dtype, int64_t kc) noexcept
{
    // fp A blocks are the kernel's 32-row (2x2 ZA) tile so its two vectors are
    // one pair load; int8 keeps the 16-row SMOPA block.
    const int64_t mr = 2 * aarch64::sme_mr;
    if (dtype == DataType::f32) {
        return kc * mr;
    }
    if (dtype == DataType::f16) {
        return ((kc + 1) / 2) * (2 * mr);
    }
    return ((kc + 3) / 4) * (4 * aarch64::sme_mr);
}

int64_t sme_b_k_offset(DataType dtype, int64_t k) noexcept
{
    const int64_t nr = dtype == DataType::s8 ? aarch64::sme_nr_max : aarch64::sme_fp_nr;
    if (dtype == DataType::f32) {
        return k * nr;
    }
    if (dtype == DataType::f16) {
        return (k / 2) * (2 * nr);
    }
    return (k / 4) * (4 * nr);
}

int64_t sme_kc_for(DataType dtype) noexcept
{
    return dtype == DataType::f32 ? 1024 : 2048;
}

namespace {

/// N and K of a raw B, in the order @p transpose_b describes.
inline void raw_b_extent(const TensorDesc& raw, bool transpose_b, int64_t& N, int64_t& K) noexcept
{
    const size_t r = static_cast<size_t>(raw.rank);
    N = transpose_b ? raw.dims[r - 2] : raw.dims[r - 1];
    K = transpose_b ? raw.dims[r - 1] : raw.dims[r - 2];
}

/// ⌈total / N⌉ × N — the descriptor's numel, and so the blob's allocation.
inline int64_t packed_b_numel(const MatMulPackedB& g, int64_t N) noexcept
{
    return ((g.total + N - 1) / N) * N;
}

} // namespace

TensorDesc matmul_packed_b_desc(const TensorDesc& raw, bool transpose_b, bool sme_int8)
{
    TensorDesc d;
    if (raw.rank < 2) {
        return d;
    }

    int64_t N = 0, K = 0;
    raw_b_extent(raw, transpose_b, N, K);
    // On an SME machine a dense f32/f16 weight is packed into the ZA kernels'
    // own layout: those run the product as soon as M is past the small routes,
    // and a NEON panel blob would be unreadable there. Decided once, at load
    // time. An s8 weight joins only when the caller asked for an integral
    // output: an fp output keeps the segmented blob its quantizing route reads.
    const bool sme_fp = raw.dtype == DataType::f32 || raw.dtype == DataType::f16;
    const bool sme_i8 = raw.dtype == DataType::s8 && sme_int8;
    const bool sme_blob = (sme_fp || sme_i8) && aarch64::available_sme();
    const MatMulPackedB g = sme_blob ? matmul_packed_b_sme_geom(N, K, raw.dtype) :
                                       matmul_packed_b_geom(N, K, raw.dtype);
    if (!g.ok()) {
        return d;
    }

    d.rank = 2;
    d.dtype = raw.dtype;
    d.layout = sme_blob ? TensorLayout::PackedBSme : TensorLayout::PackedB;
    d.dims.resize(2);
    d.dims[0] = N;
    d.dims[1] = packed_b_numel(g, N) / N;
    d.row_stride_elems = d.dims[1];
    return d;
}

bool matmul_packed_b_matches(const TensorDesc& packed, int64_t K) noexcept
{
    if (!is_packed_matmul_b(packed.layout) || packed.rank < 2) {
        return false;
    }
    const int64_t N = packed.dims[0];
    const MatMulPackedB g = packed.layout == TensorLayout::PackedBSme ?
                                matmul_packed_b_sme_geom(N, K, packed.dtype) :
                                matmul_packed_b_geom(N, K, packed.dtype);
    return g.ok() && packed.numel() == packed_b_numel(g, N);
}

void matmul_packed_b_pack(const TensorView& raw, TensorView& packed, bool transpose_b,
                          const ComputeContext& ctx)
{
    const DataType dt = packed.data_type();
    const bool trans = transpose_b;
    const int64_t N = trans ? raw.shape(raw.rank() - 2) : raw.shape(raw.rank() - 1);
    const int64_t K = trans ? raw.shape(raw.rank() - 1) : raw.shape(raw.rank() - 2);
    if (packed.layout() == TensorLayout::PackedBSme) {
#if defined(NNOPS_ARCH_SME)
        if (dt == DataType::s8) {
            // The int8 ZA packer lives with the int8 half of this layer.
            matmul_packed_b_sme_i8_pack(raw, packed, trans);
        } else if (dt == DataType::f16) {
            pack_b_f16_sme(static_cast<int>(N), static_cast<int>(K), packed.ptr<half>(),
                           raw.ptr<half>(), raw.row_stride_elems(), trans);
        } else {
            pack_b_f32_sme(static_cast<int>(N), static_cast<int>(K), packed.ptr<float>(),
                           raw.ptr<float>(), raw.row_stride_elems(), trans);
        }
#endif
        return;
    }
    const MatMulPackedB g = matmul_packed_b_geom(N, K, dt);
    NNOPS_ASSERT(g.ok() && "matmul_packed_b_pack: no packed form for this B");
    NNOPS_ASSERT(packed.numel() == packed_b_numel(g, N) &&
                 "matmul_packed_b_pack: the buffer does not match the geometry");

    const int64_t lds = raw.row_stride_elems();
    const int ldd = static_cast<int>(g.ldd);
    const int lds_i = static_cast<int>(lds);
    const int kc = static_cast<int>(K);

    if (dt == DataType::s8) {
        // One cell per (k segment, panel). The correction block is the s8 A
        // shift, so the blob is packed as if A were s8 on every arch — which is
        // also the only A dtype the compute side accepts for this blob.
        const int64_t nb = (K + g.seg - 1) / g.seg;
        const int64_t cells = nb * g.np;
        const bool a_biased = int8_a_pack_bias(DataType::s8) != 0;
        const int8_t* const base = raw.ptr<int8_t>();
        int8_t* const blob = packed.ptr<int8_t>();
        const int64_t tasks =
            std::min<int64_t>(cells, std::max<int64_t>(ctx.cpu.thread_count(), 1));
        ctx.cpu.run(0, tasks, [&](int64_t t) {
            const int64_t c0 = t * cells / tasks;
            const int64_t c1 = (t + 1) * cells / tasks;
            for (int64_t c = c0; c < c1; ++c) {
                const int64_t b = c / g.np;
                const int64_t p = c % g.np;
                const int64_t k0 = b * g.seg;
                const int64_t seg_kc = std::min(g.seg, K - k0);
                const int64_t n_start = p * g.nr;
                const int w = static_cast<int>(std::min(g.nr, N - n_start));
                const int64_t src_off = trans ? n_start * lds + k0 : k0 * lds + n_start;
                tile_pack_rhs_i8(trans, w, static_cast<int>(seg_kc), blob + (b * g.np + p) * g.ldd,
                                 ldd, base + src_off, lds_i, a_biased);
            }
        });
        return;
    }

    // Panels are independent: one contiguous range per task, each panel packed
    // by the same kernel the per-call route runs on a k-block.
    const int64_t tasks = std::min<int64_t>(g.np, std::max<int64_t>(ctx.cpu.thread_count(), 1));
    ctx.cpu.run(0, tasks, [&](int64_t t) {
        const int64_t p0 = t * g.np / tasks;
        const int64_t p1 = (t + 1) * g.np / tasks;
        for (int64_t p = p0; p < p1; ++p) {
            const int64_t n_start = p * g.nr;
            const int w = static_cast<int>(std::min<int64_t>(g.nr, N - n_start));
            const int64_t src_off = trans ? n_start * lds : n_start;
            if (dt == DataType::f32) {
                const float* src = raw.ptr<float>() + src_off;
                float* dst = packed.ptr<float>() + p * g.ldd;
                tile_pack_rhs(trans, w, kc, dst, ldd, src, lds_i, 1.0f);
            } else if (dt == DataType::f16) {
                const half* src = raw.ptr<half>() + src_off;
                half* dst = packed.ptr<half>() + p * g.ldd;
                tile_pack_rhs(trans, w, kc, dst, ldd, src, lds_i, 1.0f);
            }
#ifdef NNOPS_ARCH_AARCH64
            else {
                const bfloat16* src = raw.ptr<bfloat16>() + src_off;
                bfloat16* dst = packed.ptr<bfloat16>() + p * g.ldd;
                tile_pack_rhs(trans, w, kc, dst, ldd, src, lds_i, 1.0f);
            }
#endif
        }
    });
}



#if defined(NNOPS_ARCH_SME)
void tile_pack_lhs_sme(bool trans, int mc, int kc, float* dst, const float* src, int lds) noexcept
{
    aarch64::pack_a_f32_sme(mc, kc, dst, src, lds, trans);
}

void tile_pack_lhs_sme(bool trans, int mc, int kc, half* dst, const half* src, int lds) noexcept
{
    aarch64::pack_a_f16_sme(mc, kc, dst, src, lds, trans);
}

void tile_pack_rhs_sme(bool trans, int nc, int kc, float* dst, const float* src, int lds) noexcept
{
    aarch64::pack_b_f32_sme(nc, kc, dst, src, lds, trans);
}

void tile_pack_rhs_sme(bool trans, int nc, int kc, half* dst, const half* src, int lds) noexcept
{
    aarch64::pack_b_f16_sme(nc, kc, dst, src, lds, trans);
}

void tile_mma_pack_sme(int mc, int nc, int kc, float* c, int ldc, const float* packed_a,
                       const float* b, int64_t block_stride, float clamp_min, float clamp_max,
                       bool zero_mode, const float* col_bias)
{
    aarch64::mma_f32_sme(mc, nc, kc, c, ldc, packed_a, b, zero_mode, clamp_min, clamp_max,
                         col_bias, block_stride > 0 ? block_stride : 0);
}

void tile_mma_pack_sme(int mc, int nc, int kc, half* c, int ldc, const half* packed_a,
                       const half* b, int64_t block_stride, float clamp_min, float clamp_max,
                       bool zero_mode, const half* col_bias)
{
    aarch64::mma_f16_sme(mc, nc, kc, c, ldc, packed_a, b, zero_mode, clamp_min, clamp_max,
                         col_bias, block_stride > 0 ? block_stride : 0);
}

#endif

// ---- tile scale -----------------------------------------------------------

template <typename T> void tile_scale_impl(T* c, int ldc, float scale, int M, int N)
{
    using namespace nnops::simd;
    constexpr int L = simd_lane_for<T>;
    for (int m = 0; m < M; ++m) {
        T* row = c + m * ldc;
        auto v_scale = v_set1(row, scale);
        int i = 0;
        for (; i + L <= N; i += L) {
            v_store(row + i, v_mul(v_load(row + i), v_scale));
        }
        for (; i < N; ++i) {
            s_store(&row[i], s_load(&row[i]) * scale);
        }
    }
}

void tile_scale(float* c, int ldc, float scale, int M, int N)
{
    tile_scale_impl(c, ldc, scale, M, N);
}

void tile_scale(half* c, int ldc, float scale, int M, int N)
{
    tile_scale_impl(c, ldc, scale, M, N);
}

#ifdef NNOPS_ARCH_AARCH64
void tile_scale(bfloat16* c, int ldc, float scale, int M, int N)
{
    tile_scale_impl(c, ldc, scale, M, N);
}
#endif

/// C = scale·C + bias. A zero scale replaces the tile, so it stores the bias
/// rather than trusting 0·C: the old C is arbitrary (a pooled output buffer),
/// and 0·±inf or 0·NaN is NaN, which would then ride the whole k sweep.
template <typename T>
void tile_scale_add_bias_impl(T* c, int ldc, float scale, const T* bias, int M, int N)
{
    using namespace nnops::simd;
    constexpr int L = simd_lane_for<T>;
    const bool replace = (scale == 0.0f);
    for (int m = 0; m < M; ++m) {
        T* row = c + m * ldc;
        auto v_scale = v_set1(row, scale);
        int i = 0;
        for (; i + L <= N; i += L) {
            auto v = replace ? v_load(bias + i) :
                               v_add(v_mul(v_load(row + i), v_scale), v_load(bias + i));
            v_store(row + i, v);
        }
        for (; i < N; ++i) {
            s_store(&row[i],
                    replace ? s_load(&bias[i]) : s_load(&row[i]) * scale + s_load(&bias[i]));
        }
    }
}

void tile_scale_add_bias(float* c, int ldc, float scale, const float* bias, int M, int N)
{
    tile_scale_add_bias_impl(c, ldc, scale, bias, M, N);
}

void tile_scale_add_bias(half* c, int ldc, float scale, const half* bias, int M, int N)
{
    tile_scale_add_bias_impl(c, ldc, scale, bias, M, N);
}

#ifdef NNOPS_ARCH_AARCH64
void tile_scale_add_bias(bfloat16* c, int ldc, float scale, const bfloat16* bias, int M, int N)
{
    tile_scale_add_bias_impl(c, ldc, scale, bias, M, N);
}
#endif

MatMulPlan get_matmul_plan(const MatMulAttributes& attrs, const TensorDesc& a_desc,
                           const TensorDesc& b_desc, int num_threads, bool use_thread_slots)
{
    MatMulPlan plan{}; // the dtype-reject returns below promise an all-zero plan

    const auto dt_a = a_desc.dtype;
    const auto dt_b = b_desc.dtype;

    // A block-packed B (q4_1) is self-describing, so A only has to be a float
    // type. bf16 joins on aarch64, where it decodes into the BFMMLA layout.
    const bool block_rhs = is_block_packed_dtype(dt_b) &&
                           (dt_a == DataType::f32 || dt_a == DataType::f16 ||
                            (nnops::detail::bf16_cpu_enabled && dt_a == DataType::bf16));
    const bool int8_pair = is_int8_matmul(dt_a, dt_b);
    if (dt_a != dt_b && !block_rhs && !int8_pair) {
        return plan;
    }

    const MatmulGeom* geom = nullptr;
    if (int8_pair) {
        geom = &GEOM_I8;
    } else if (dt_a == DataType::f32) {
        geom = &GEOM_F32;
    } else if (dt_a == DataType::f16) {
        geom = &GEOM_F16;
    }
#ifdef NNOPS_ARCH_AARCH64
    else if (dt_a == DataType::bf16) {
        geom = &GEOM_BF16;
    }
#endif
    else {
        return plan;
    }
    num_threads = std::max(num_threads, 1);

    const int64_t a_rank = a_desc.rank;
    const int64_t b_rank = b_desc.rank;
    NNOPS_ASSERT(a_rank >= 2 && b_rank >= 2);

    // A PackedB leads with N and carries no K (the blob is addressed from K,
    // not shaped by it), so N comes from the descriptor and K from A — which
    // the product requires to agree anyway.
    const bool packed_b = is_packed_matmul_b(b_desc.layout);
    // The packed RHS is a float-panel product, or an int8 one whose activation
    // is s8 — that blob carries the s8 A shift in its correction block, which a
    // u8 activation would not undo.
    const bool packed_i8 = packed_b && dt_a == DataType::s8 && dt_b == DataType::s8;
    if (packed_b && !packed_i8 &&
        !(dt_a == DataType::f32 || dt_a == DataType::f16 ||
          (nnops::detail::bf16_cpu_enabled && dt_a == DataType::bf16))) {
        return plan;
    }

    const int64_t M = attrs.transpose_a ? a_desc.dims[static_cast<size_t>(a_rank - 1)] :
                                          a_desc.dims[static_cast<size_t>(a_rank - 2)];
    const int64_t N = packed_b          ? b_desc.dims[0] :
                      attrs.transpose_b ? b_desc.dims[static_cast<size_t>(b_rank - 2)] :
                                          b_desc.dims[static_cast<size_t>(b_rank - 1)];
    const int64_t K = packed_b ?
                          (attrs.transpose_a ? a_desc.dims[static_cast<size_t>(a_rank - 2)] :
                                               a_desc.dims[static_cast<size_t>(a_rank - 1)]) :
                          (attrs.transpose_b ? b_desc.dims[static_cast<size_t>(b_rank - 1)] :
                                               b_desc.dims[static_cast<size_t>(b_rank - 2)]);

    // Pack A for correctness (transpose_a), for int8, or for a deep K.
    plan.pack_a = geom->packs_a_always || attrs.transpose_a || (K > PACK_A_K_THRESHOLD);

    // Parallel split, as MLAS: the larger dimension.
    plan.split_n = (N > M);
    const bool split_m = !plan.split_n;

    // Uniform panels for every dtype: the count is monotonic.
    auto panel_count = [&](int n) { return (n + geom->nr_max - 1) / geom->nr_max; };

    const auto& features = simd::CpuFeatures::get();
    if (packed_i8) {
        // The blob's segments are cut at KC_CAP_I8 and each panel's correction
        // block sits at its segment's kc, so one k-block is one segment: kc is
        // frozen at the segment length and mc/nc are sized around it.
        const MatMulPackedB bg = matmul_packed_b_geom(N, K, dt_b);
        NNOPS_ASSERT(bg.ok());
        choose_tile_impl(M, N, K, geom->mr_max, geom->nr_max, geom->elem, num_threads, plan.split_n,
                         geom->kc_cap, bg.seg, features.l1_cache_size(),
                         features.l2_shared_cache_size(), plan.mc, plan.nc, plan.kc);
    } else {
        choose_matmul_tile(M, N, K, geom->mr_max, geom->nr_max, geom->elem, num_threads,
                           plan.split_n, geom->kc_cap, features.l1_cache_size(),
                           features.l2_shared_cache_size(), plan.mc, plan.nc, plan.kc);
    }

    // Sizing stride at full Kc.
    const int64_t ldd_b = geom->ldd_b(plan.kc);
    plan.ldd_b = ldd_b;

    // Packed-B scratch. N-split: one slice per n-block. M-split: one per
    // m-block, or per worker thread when ids are available and blocks outnumber
    // them (see MatMulPlan::num_slots). int8 hoists B, so one slot serves all.
    const int64_t slice_bytes =
        static_cast<int64_t>(panel_count(static_cast<int>(N))) * ldd_b * geom->elem;

    const int64_t num_m_blocks = split_block_count(M, plan.mc);
    const int64_t num_n_blocks = (N + plan.nc - 1) / plan.nc;

    if (packed_b) {
        // The B side is the caller's blob: no slice, no slots, nothing to pack.
        plan.num_slots = 0;
        plan.np_slice = 0;
        plan.workspace_size = 0;
    } else if (split_m) {
        if (geom->hoists_b_m_split) {
            plan.num_slots = 1;
        } else {
            const int64_t nt = std::max<int64_t>(num_threads, 1);
            plan.num_slots = (use_thread_slots && nt < num_m_blocks) ? nt : num_m_blocks;
        }
        plan.np_slice = panel_count(static_cast<int>(N));
        plan.workspace_size = plan.num_slots * slice_bytes;
    } else {
        plan.num_slots = num_n_blocks;
        plan.np_slice = panel_count(static_cast<int>(plan.nc));
        plan.workspace_size = plan.num_slots * plan.np_slice * ldd_b * geom->elem;
    }

    // Packed-A scratch follows the packed-B region.
    plan.pack_a_offset = plan.workspace_size;
    if (plan.pack_a) {
        const int64_t nconcurrent = plan.split_n ? num_n_blocks : num_m_blocks;
        const int64_t nt = std::max<int64_t>(num_threads, 1);
        if (nt <= 1) {
            plan.num_slots_a = 1;
        } else if (use_thread_slots && nt < nconcurrent) {
            plan.num_slots_a = nt;
        } else {
            plan.num_slots_a = nconcurrent;
        }

        plan.ldd_a = geom->ldd_a(plan.kc);
        plan.np_a = num_panels_max(static_cast<int>(plan.mc), geom->mr_levels);

        plan.workspace_size =
            plan.pack_a_offset + plan.num_slots_a * plan.np_a * plan.ldd_a * geom->elem;
    }

    return plan;
}

#if defined(NNOPS_ARCH_SME)
MatMulPlan get_matmul_plan_sme(const MatMulAttributes& attrs, const TensorDesc& a_desc,
                               const TensorDesc& b_desc, int num_threads, bool use_thread_slots)
{
    MatMulPlan plan{};

    const DataType dt_a = a_desc.dtype;
    const DataType dt_b = b_desc.dtype;
    const bool i8_pair = is_int8_matmul(dt_a, dt_b);
    const bool fp_pair =
        (dt_a == DataType::f32 || dt_a == DataType::f16) && dt_b == dt_a;
    if ((!i8_pair && !fp_pair) || a_desc.rank < 2 || b_desc.rank < 2) {
        return plan;
    }

    const bool packed_b = is_packed_matmul_b(b_desc.layout);
    // Only the ZA blob is readable here; a NEON panel blob is another route's.
    if (packed_b && b_desc.layout != TensorLayout::PackedBSme) {
        return plan;
    }
    // A block-quantised B has no SME kernel.
    if (!packed_b && is_block_packed_dtype(dt_b)) {
        return plan;
    }

    const int64_t a_rank = a_desc.rank;
    const int64_t b_rank = b_desc.rank;
    const int64_t M = attrs.transpose_a ? a_desc.dims[static_cast<size_t>(a_rank - 1)] :
                                          a_desc.dims[static_cast<size_t>(a_rank - 2)];
    const int64_t N = packed_b ? b_desc.dims[0] :
                      attrs.transpose_b ? b_desc.dims[static_cast<size_t>(b_rank - 2)] :
                                          b_desc.dims[static_cast<size_t>(b_rank - 1)];
    const int64_t K = packed_b ?
                          (attrs.transpose_a ? a_desc.dims[static_cast<size_t>(a_rank - 2)] :
                                               a_desc.dims[static_cast<size_t>(a_rank - 1)]) :
                          (attrs.transpose_b ? b_desc.dims[static_cast<size_t>(b_rank - 1)] :
                                               b_desc.dims[static_cast<size_t>(b_rank - 2)]);
    if (M <= 0 || N <= 0 || K <= 0) {
        return plan;
    }

    const int64_t elem = static_cast<int64_t>(data_type_size(dt_a));
    const int64_t kc = sme_kc_for(dt_a);
    // The kernel's n-tile loop reloads B for every m-block, so a small mc is
    // paid in B traffic: 64 columns x kc of B per block. 256 rows keeps the
    // packed A slab at mc*kc elements (1-2 MB) while amortising B 4x better.
    const int64_t mc = 64;
    const int64_t nr = i8_pair ? aarch64::sme_nr_max : aarch64::sme_fp_nr;
    num_threads = std::max(num_threads, 1);

    // One N block per worker, rounded to the 64-column grid and capped so a
    // worker's B slab stays small; the driver splits N over the pool.
    const int64_t n_total_blocks = (N + nr - 1) / nr;
    const int64_t workers =
        std::max<int64_t>(1, std::min<int64_t>(num_threads, n_total_blocks));
    int64_t nc = std::min<int64_t>(1024, (N + workers - 1) / workers);
    nc = std::max<int64_t>(nr, ((nc + nr - 1) / nr) * nr);

    const int64_t np_slice = (nc + nr - 1) / nr;
    const int64_t num_n_blocks = (N + nc - 1) / nc;

    plan.mc = mc;
    plan.nc = nc;
    plan.kc = kc;
    plan.split_n = true;
    plan.pack_a = true;
    plan.ldd_b = sme_b_panel_stride(dt_b, kc);
    plan.ldd_a = sme_a_panel_stride(dt_a, kc);
    plan.np_a = i8_pair ? (mc + aarch64::sme_mr - 1) / aarch64::sme_mr :
                          (mc + 2 * aarch64::sme_mr - 1) / (2 * aarch64::sme_mr);

    if (packed_b) {
        plan.num_slots = 0;
        plan.np_slice = 0;
        plan.workspace_size = 0;
    } else {
        plan.num_slots = num_n_blocks;
        plan.np_slice = np_slice;
        plan.workspace_size = plan.num_slots * np_slice * plan.ldd_b * elem;
    }

    plan.pack_a_offset = plan.workspace_size;
    const int64_t nt = std::max<int64_t>(num_threads, 1);
    if (nt <= 1) {
        plan.num_slots_a = 1;
    } else if (use_thread_slots && nt < num_n_blocks) {
        plan.num_slots_a = nt;
    } else {
        plan.num_slots_a = num_n_blocks;
    }
    plan.workspace_size =
        plan.pack_a_offset + plan.num_slots_a * plan.np_a * plan.ldd_a * elem;
    return plan;
}
#endif

} // namespace nnops::backend::cpu
