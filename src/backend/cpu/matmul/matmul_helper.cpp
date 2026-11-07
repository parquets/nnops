/// @file matmul_helper.cpp
/// @brief Arch-neutral pack/MMA dispatch between the tiled drivers (MatMul,
/// Conv2D, attention) and the per-arch kernels under x86_64/ and aarch64/.
///
/// The per-arch pack/MMA fork lives here: the tables and TileTraits<> below are
/// selected by NNOPS_ARCH_*, so no caller names an arch.
///
/// Invariants:
///  - A decomposes into the arch's MR cascade (MR_F32 / MR_F16 / MR_I8),
///    largest level first; a leftover runs at the next, smaller level.
///  - B is uniform: ceil(nc/nr_max) nr_max-wide panels at one 64-byte-aligned
///    stride. The MMA reads whole lanes, so callers pass the live width and
///    each kernel's N-tail path drops the lanes past it.
///  - Panels sit at the caller's `ldd`, never end to end.
///
/// Two RHS modes: a per-call pack whose stride comes from the k-block length,
/// and a MatMulPackedB blob packed outside the call. Those strides disagree, so
/// the blob route hands the kernel one panel per call and steps panels itself.
/// The int8 blob adds a second axis — its panels come in k segments, because the
/// per-panel correction block is read at the call's kc (see matmul_packed_b_geom).
///
/// tile_mma_direct's bias seeds the accumulator instead of reading C: `bias` is
/// one value per output row, `col_bias` one per output column (channel-last
/// Conv2D, where a row is a pixel). Either one builds the tile, so it belongs on
/// the first k-block only.

#include "matmul_helper.h"

#include <array>

#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/q4_1.hpp"
#include "nnops/detail/simd.hpp"
#include "backend/cpu/simd_kernel/simd_quant.hpp"

#ifdef NNOPS_ARCH_X86_64
using namespace nnops::backend::cpu::x86_64;
#elif defined(NNOPS_ARCH_AARCH64)
using namespace nnops::backend::cpu::aarch64;
#endif

namespace nnops::backend::cpu {

// Trailing live width: one kernel serves a full panel and a tail.

template <class T>
using PackFn = void (*)(T* NNOPS_RESTRICT output, const T* NNOPS_RESTRICT input,
                        int ir_step, int K, float scale, int w);

// Pack tables (largest-first): row 0 = LHS (MR), row 1 = RHS (NR).
constexpr std::array<std::array<PackFn<float>, 3>, 2> pack_trans_f32_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {pack_trans_n6_f32, pack_trans_n4_f32, pack_trans_n1_f32},
    {pack_trans_n16_f32, pack_trans_n8_f32, pack_trans_n1_f32},
#elif defined(NNOPS_ARCH_AARCH64)
    {pack_trans_n8_f32, pack_trans_n4_f32, pack_trans_n1_f32},
    {pack_trans_n12_f32, pack_trans_n4_f32, pack_trans_n1_f32},
#endif
}};

constexpr std::array<std::array<PackFn<float>, 3>, 2> pack_copy_f32_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {pack_copy_n6_f32, pack_copy_n4_f32, pack_copy_n1_f32},
    {pack_copy_n16_f32, pack_copy_n8_f32, pack_copy_n1_f32},
#elif defined(NNOPS_ARCH_AARCH64)
    {pack_copy_n8_f32, pack_copy_n4_f32, pack_copy_n1_f32},
    {pack_copy_n12_f32, pack_copy_n4_f32, pack_copy_n1_f32},
#endif
}};

constexpr std::array<std::array<PackFn<half>, 3>, 2> pack_trans_f16_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {pack_trans_n6_f16, pack_trans_n4_f16, pack_trans_n1_f16},
    {pack_trans_n16_f16, pack_trans_n8_f16, pack_trans_n1_f16},
#elif defined(NNOPS_ARCH_AARCH64)
    {pack_trans_n8_f16, pack_trans_n4_f16, pack_trans_n1_f16},
    {pack_trans_n16_f16, pack_trans_n8_f16, pack_trans_n1_f16},
#endif
}};

constexpr std::array<std::array<PackFn<half>, 3>, 2> pack_copy_f16_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {pack_copy_n6_f16, pack_copy_n4_f16, pack_copy_n1_f16},
    {pack_copy_n16_f16, pack_copy_n8_f16, pack_copy_n1_f16},
#elif defined(NNOPS_ARCH_AARCH64)
    {pack_copy_n8_f16, pack_copy_n4_f16, pack_copy_n1_f16},
    {pack_copy_n16_f16, pack_copy_n8_f16, pack_copy_n1_f16},
#endif
}};

template <class T> struct TileTraits;

template <> struct TileTraits<float> {
    using Pack = PackFn<float>;
    static constexpr const int* mr() { return MR_F32; }
    static constexpr const int* nr() { return NR_F32; }
    static constexpr const auto& trans() { return pack_trans_f32_fn; }
    static constexpr const auto& copy()  { return pack_copy_f32_fn; }
    template <bool Z, int S = 0> static constexpr const auto& mma_pack() { return mma_pack_fns_f32<Z, S>; }
    template <bool Z, int S> static constexpr const auto& mma_direct() { return mma_direct_fns_f32<Z, S>; }
};

template <> struct TileTraits<half> {
    using Pack = PackFn<half>;
    static constexpr const int* mr() { return MR_F16; }
    static constexpr const int* nr() { return NR_F16; }
    static constexpr const auto& trans() { return pack_trans_f16_fn; }
    static constexpr const auto& copy()  { return pack_copy_f16_fn; }
    template <bool Z, int S = 0> static constexpr const auto& mma_pack() { return mma_pack_fns_f16<Z, S>; }
    template <bool Z, int S> static constexpr const auto& mma_direct() { return mma_direct_fns_f16<Z, S>; }
};

/// 64-byte-aligned panel stride, in elements.
template <class T>
constexpr int panel_stride(int rows, int kc) {
    return static_cast<int>(
        mma_pack_align(rows * kc * static_cast<int>(sizeof(T)))
        / static_cast<int>(sizeof(T)));
}

template <int Level, class Emit>
inline void panel_level(int& i, int count, const int* sizes, Emit&& emit) {
    const int s = sizes[Level];
    for (; i + s <= count; i += s) { emit(Level, s); }
}

template <class Emit>
inline void walk_levels(int& i, int count, const int* levels, Emit&& emit) {
    panel_level<0>(i, count, levels, emit);
    panel_level<1>(i, count, levels, emit);
    panel_level<2>(i, count, levels, emit);
}

template <class Pack>
inline void for_each_n_panel(int nc, int nr_max, Pack&& pack) {
    int n = 0;
    for (; n + nr_max <= nc; n += nr_max) { pack(nr_max); }
    if (n < nc) { pack(nc - n); }
}

template <class T>
void tile_pack_lhs_impl(bool trans, int mc, int kc, T* dst, int ldd,
                        const T* src, int lds, float scale) {
    using Tr = TileTraits<T>;
    using Fn = typename Tr::Pack;
    // transposed source is a plain copy; row-major needs the transposing kernel.
    const std::array<Fn, 3>& fns = trans ? Tr::copy()[0] : Tr::trans()[0];
    const int pack_lds = trans ? 1 : lds;

    int i = 0;
    walk_levels(i, mc, Tr::mr(), [&](int level, int s) {
        fns[level](dst, src, lds, kc, scale, s);
        dst += ldd;
        src += s * pack_lds;
    });
}

void tile_pack_lhs(bool trans, int mc, int kc,
                   float* dst, int ldd, const float* src, int lds, float scale) {
    tile_pack_lhs_impl(trans, mc, kc, dst, ldd, src, lds, scale);
}

void tile_pack_lhs(bool trans, int mc, int kc,
                   half* dst, int ldd, const half* src, int lds, float scale) {
    tile_pack_lhs_impl(trans, mc, kc, dst, ldd, src, lds, scale);
}

// =========================================================================
//  Uniform-width RHS panels
// =========================================================================

template <class T>
void tile_pack_rhs_impl(bool trans, int nc, int kc, T* dst, int ldd,
                        const T* src, int lds, float scale) {
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
void tile_mma_pack_impl(int Mc, int Nc, int Kc, T* c, int ldc,
                        const T* packed_a, const T* b, int64_t ldb,
                        float clamp_min, float clamp_max,
                        const T* bias, const T* col_bias) {
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
            fns[level](s, Nc, Kc, c + m * ldc, ldc, packed_a, b,
                       clamp_min, clamp_max, seed_at(m, 0));
        } else {
            for (int n = 0; n < Nc; n += nr) {
                const int w = std::min<int>(nr, Nc - n);
                fns[level](s, w, Kc, c + m * ldc + n, ldc, packed_a,
                           b + (static_cast<int64_t>(n) / nr) * -ldb,
                           clamp_min, clamp_max, seed_at(m, n));
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
void tile_mma_direct_impl(int Mc, int Nc, int Kc, T* c, int ldc,
                          const T* a, int lda, const T* b, int64_t ldb,
                          T* stage, float clamp_min, float clamp_max,
                          const T* bias, const T* col_bias) {
    using Tr = TileTraits<T>;
    const auto& fns = Tr::template mma_direct<Zero, Seed>();
    constexpr int nr = Tr::nr()[0];

    const auto seed_at = [&](int m, int n) {
        return (Seed == 1) ? bias + m : (Seed == 2 ? col_bias + n : nullptr);
    };

    int m = 0;
    walk_levels(m, Mc, Tr::mr(), [&](int level, int s) {
        const T* const a_m = a + m * lda;
        if (ldb == -1) {
            fns[level](s, Nc, Kc, c + m * ldc, ldc, a_m, lda, b, /*ldb=*/-1,
                       clamp_min, clamp_max, seed_at(m, 0));
        } else if (ldb < 0) {
            for (int n = 0; n < Nc; n += nr) {
                const int w = std::min<int>(nr, Nc - n);
                fns[level](s, w, Kc, c + m * ldc + n, ldc, a_m, lda,
                           b + (static_cast<int64_t>(n) / nr) * -ldb, /*ldb=*/-1,
                           clamp_min, clamp_max, seed_at(m, n));
            }
        } else {
            const int ld = static_cast<int>(ldb);
            int n = 0;
            for (; n + nr <= Nc; n += nr) {
                fns[level](s, nr, Kc, c + m * ldc + n, ldc, a_m, lda, b + n, ld,
                           clamp_min, clamp_max, seed_at(m, n));
            }
            if (n < Nc) {
                const int w = Nc - n;
                tile_pack_rhs(false, w, Kc, stage, nr * Kc, b + n, ld, 1.0f);
                fns[level](s, w, Kc, c + m * ldc + n, ldc, a_m, lda, stage, nr,
                           clamp_min, clamp_max, seed_at(m, n));
            }
        }
    });
}

template <class T>
void dispatch_mma_pack(bool zero_mode, int Mc, int Nc, int Kc, T* c, int ldc,
                       const T* a, const T* b, int64_t ldb, float lo, float hi,
                       const T* bias, const T* col_bias) {
    // Either bias builds the tile, so there is nothing in C to accumulate onto.
    if (col_bias != nullptr) { tile_mma_pack_impl<T, true, 2>(Mc, Nc, Kc, c, ldc, a, b, ldb, lo, hi, nullptr, col_bias); }
    else if (bias != nullptr) { tile_mma_pack_impl<T, true, 1>(Mc, Nc, Kc, c, ldc, a, b, ldb, lo, hi, bias, nullptr); }
    else if (zero_mode)  { tile_mma_pack_impl<T, true, 0>(Mc, Nc, Kc, c, ldc, a, b, ldb, lo, hi, nullptr, nullptr); }
    else                 { tile_mma_pack_impl<T, false, 0>(Mc, Nc, Kc, c, ldc, a, b, ldb, lo, hi, nullptr, nullptr); }
}

template <class T>
void dispatch_mma_direct(bool zero_mode, int Mc, int Nc, int Kc, T* c, int ldc,
                         const T* a, int lda, const T* b, int64_t ldb, T* stage,
                         float lo, float hi, const T* bias, const T* col_bias) {
    // Either bias builds the tile, so there is nothing in C to accumulate onto.
    if (col_bias != nullptr) { tile_mma_direct_impl<T, true, 2>(Mc, Nc, Kc, c, ldc, a, lda, b, ldb, stage, lo, hi, nullptr, col_bias); }
    else if (bias != nullptr) { tile_mma_direct_impl<T, true, 1>(Mc, Nc, Kc, c, ldc, a, lda, b, ldb, stage, lo, hi, bias, nullptr); }
    else if (zero_mode)  { tile_mma_direct_impl<T, true, 0>(Mc, Nc, Kc, c, ldc, a, lda, b, ldb, stage, lo, hi, nullptr, nullptr); }
    else                 { tile_mma_direct_impl<T, false, 0>(Mc, Nc, Kc, c, ldc, a, lda, b, ldb, stage, lo, hi, nullptr, nullptr); }
}

void tile_pack_rhs(bool trans, int nc, int kc,
                   float* dst, int ldd, const float* src, int lds, float scale) {
    tile_pack_rhs_impl(trans, nc, kc, dst, ldd, src, lds, scale);
}

void tile_pack_rhs(bool trans, int nc, int kc,
                   half* dst, int ldd, const half* src, int lds, float scale) {
    tile_pack_rhs_impl(trans, nc, kc, dst, ldd, src, lds, scale);
}

// ---- Block-packed RHS (q4_1) --------------------------------------------
// The panel layout matches tile_pack_rhs; the MMA route never sees the blocks.

static_assert(q4_1_block_elems == 32,
              "the block RHS packers assume 32-value blocks");

#if defined(NNOPS_ARCH_X86_64) || defined(NNOPS_ARCH_AARCH64)

void tile_pack_rhs_q4_1(bool trans, int n_start, int nc, int k_start, int kc,
                        float* dst, int ldd, const void* src, int64_t src_row_blocks,
                        float scale) {
    arch::pack_q4_1_rhs<float>(trans, n_start, nc, k_start, kc, dst, ldd, src,
                               src_row_blocks, scale);
}

void tile_pack_rhs_q4_1(bool trans, int n_start, int nc, int k_start, int kc,
                        half* dst, int ldd, const void* src, int64_t src_row_blocks,
                        float scale) {
    arch::pack_q4_1_rhs<half>(trans, n_start, nc, k_start, kc, dst, ldd, src,
                              src_row_blocks, scale);
}

#else

/// Portable fallback for arches without a vectorized block pack.
///
/// The width must be the arch's float RHS panel width — the workspace stride
/// and the MMA are sized from it — and that is 12 for aarch64 f32, 16 elsewhere.
/// The arch packers write this same layout, but decode a block once per row.
template <class T>
constexpr int block_panel_nr() noexcept {
    if constexpr (std::is_same_v<T, float>) { return NR_MAX_F32; }
    else { return NR_MAX_F16; }
}

/// Values per block, for the indexing below; must equal q4_1_block_elems.
constexpr int kBlockElems = 32;

/// One block-packed format's row decoder: unpack(block_ptr, float out[]).
template <class T, class Unpack>
void tile_pack_rhs_block_scalar(bool trans, int n_start, int nc, int k_start, int kc,
                                T* dst, int ldd, const void* src, int64_t row_blocks,
                                float scale, size_t block_bytes, Unpack unpack) {
    constexpr int NR = block_panel_nr<T>();
    const uint8_t* base = static_cast<const uint8_t*>(src);
    const int64_t row_bytes = row_blocks * static_cast<int64_t>(block_bytes);

    auto store = [](T* p, float v) {
        if constexpr (std::is_same_v<T, float>) { *p = v; }
        else { *p = float_to_half(v); }
    };

    int n = 0;
    while (n < nc) {
        const int nr = std::min(NR, nc - n);
        for (int k = 0; k < kc; ++k) {
            T* o = dst + static_cast<int64_t>(k) * NR;
            for (int j = 0; j < nr; ++j) {
                const int row = trans ? (n_start + n + j) : (k_start + k);
                const int col = trans ? (k_start + k)     : (n_start + n + j);
                float vals[kBlockElems];
                unpack(base + static_cast<int64_t>(row) * row_bytes
                            + static_cast<int64_t>(col / kBlockElems) * block_bytes,
                       vals);
                store(&o[j], vals[col % kBlockElems] * scale);
            }
            for (int j = nr; j < NR; ++j) { o[j] = T{}; }
        }
        dst += ldd;
        n += nr;
    }
}

void tile_pack_rhs_q4_1(bool trans, int n_start, int nc, int k_start, int kc,
                        float* dst, int ldd, const void* src, int64_t src_row_blocks,
                        float scale) {
    tile_pack_rhs_block_scalar<float>(trans, n_start, nc, k_start, kc, dst, ldd,
                                      src, src_row_blocks, scale,
                                      q4_1_block_bytes, q4_1_unpack_block);
}

void tile_pack_rhs_q4_1(bool trans, int n_start, int nc, int k_start, int kc,
                        half* dst, int ldd, const void* src, int64_t src_row_blocks,
                        float scale) {
    tile_pack_rhs_block_scalar<half>(trans, n_start, nc, k_start, kc, dst, ldd,
                                     src, src_row_blocks, scale,
                                     q4_1_block_bytes, q4_1_unpack_block);
}

#endif

void tile_mma_pack(int Mc, int Nc, int Kc, float* c, int ldc,
                   const float* packed_a, const float* b, int64_t ldb,
                   float clamp_min, float clamp_max, bool zero_mode,
                   const float* bias, const float* col_bias) {
    dispatch_mma_pack(zero_mode, Mc, Nc, Kc, c, ldc, packed_a, b, ldb,
                      clamp_min, clamp_max, bias, col_bias);
}

void tile_mma_pack(int Mc, int Nc, int Kc, half* c, int ldc,
                   const half* packed_a, const half* b, int64_t ldb,
                   float clamp_min, float clamp_max, bool zero_mode,
                   const half* bias, const half* col_bias) {
    dispatch_mma_pack(zero_mode, Mc, Nc, Kc, c, ldc, packed_a, b, ldb,
                      clamp_min, clamp_max, bias, col_bias);
}

void tile_mma_direct(int Mc, int Nc, int Kc, float* c, int ldc,
                     const float* a, int lda, const float* b, int64_t ldb,
                     float* stage, float clamp_min, float clamp_max,
                     bool zero_mode, const float* bias, const float* col_bias) {
    dispatch_mma_direct(zero_mode, Mc, Nc, Kc, c, ldc, a, lda, b, ldb,
                        stage, clamp_min, clamp_max, bias, col_bias);
}

void tile_mma_direct(int Mc, int Nc, int Kc, half* c, int ldc,
                     const half* a, int lda, const half* b, int64_t ldb,
                     half* stage, float clamp_min, float clamp_max,
                     bool zero_mode, const half* bias, const half* col_bias) {
    dispatch_mma_direct(zero_mode, Mc, Nc, Kc, c, ldc, a, lda, b, ldb,
                        stage, clamp_min, clamp_max, bias, col_bias);
}

// ---- the blob ---------------------------------------------------------

MatMulPackedB matmul_packed_b_geom(int64_t N, int64_t K, DataType dtype) noexcept {
    MatMulPackedB g;
    if (N <= 0 || K <= 0) { return g; }

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

    if (dtype != DataType::f32 && dtype != DataType::f16) { return g; }

    const int64_t nr = (dtype == DataType::f16) ? NR_MAX_F16 : NR_MAX_F32;
    const int64_t esz = static_cast<int64_t>(data_type_size(dtype));
    g.nr = nr;
    // The same aligned slot the per-call pack uses, only at the whole K.
    g.ldd = static_cast<int64_t>(
        mma_pack_align(static_cast<size_t>(nr * K) * static_cast<size_t>(esz))) / esz;
    g.np = (N + nr - 1) / nr;
    g.total = g.np * g.ldd;
    return g;
}

namespace {

/// N and K of a raw B, in the order @p transpose_b describes.
inline void raw_b_extent(const TensorDesc& raw, bool transpose_b,
                         int64_t& N, int64_t& K) noexcept {
    const size_t r = static_cast<size_t>(raw.rank);
    N = transpose_b ? raw.dims[r - 2] : raw.dims[r - 1];
    K = transpose_b ? raw.dims[r - 1] : raw.dims[r - 2];
}

/// ⌈total / N⌉ × N — the descriptor's numel, and so the blob's allocation.
inline int64_t packed_b_numel(const MatMulPackedB& g, int64_t N) noexcept {
    return ((g.total + N - 1) / N) * N;
}

}  // namespace

TensorDesc matmul_packed_b_desc(const TensorDesc& raw, bool transpose_b) {
    TensorDesc d;
    if (raw.rank < 2) { return d; }

    int64_t N = 0, K = 0;
    raw_b_extent(raw, transpose_b, N, K);
    const MatMulPackedB g = matmul_packed_b_geom(N, K, raw.dtype);
    if (!g.ok()) { return d; }

    d.rank = 2;
    d.dtype = raw.dtype;
    d.layout = TensorLayout::PackedB;
    d.dims.resize(2);
    d.dims[0] = N;
    d.dims[1] = packed_b_numel(g, N) / N;
    d.row_stride_elems = d.dims[1];
    return d;
}

bool matmul_packed_b_matches(const TensorDesc& packed, int64_t K) noexcept {
    if (!is_packed_matmul_b(packed.layout) || packed.rank < 2) { return false; }
    const int64_t N = packed.dims[0];
    const MatMulPackedB g = matmul_packed_b_geom(N, K, packed.dtype);
    return g.ok() && packed.numel() == packed_b_numel(g, N);
}

void matmul_packed_b_pack(const TensorView& raw, TensorView& packed,
                          bool transpose_b, const ComputeContext& ctx) {
    const DataType dt = packed.data_type();
    const bool trans = transpose_b;
    const int64_t N = trans ? raw.shape(raw.rank() - 2) : raw.shape(raw.rank() - 1);
    const int64_t K = trans ? raw.shape(raw.rank() - 1) : raw.shape(raw.rank() - 2);
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
        const int64_t tasks = std::min<int64_t>(
            cells, std::max<int64_t>(ctx.cpu.thread_count(), 1));
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
                const int64_t src_off = trans ? n_start * lds + k0
                                              : k0 * lds + n_start;
                tile_pack_rhs_i8(trans, w, static_cast<int>(seg_kc),
                                 blob + (b * g.np + p) * g.ldd, ldd,
                                 base + src_off, lds_i, a_biased);
            }
        });
        return;
    }

    // Panels are independent: one contiguous range per task, each panel packed
    // by the same kernel the per-call route runs on a k-block.
    const int64_t tasks = std::min<int64_t>(g.np,
        std::max<int64_t>(ctx.cpu.thread_count(), 1));
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
            } else {
                const half* src = raw.ptr<half>() + src_off;
                half* dst = packed.ptr<half>() + p * g.ldd;
                tile_pack_rhs(trans, w, kc, dst, ldd, src, lds_i, 1.0f);
            }
        }
    });
}

void matmul_packed_b_i8_col_sums(const TensorView& packed, int64_t K,
                                 int32_t* r_b, const ComputeContext& ctx) {
    const int64_t N = packed.shape(0);
    const MatMulPackedB g = matmul_packed_b_geom(N, K, DataType::s8);
    NNOPS_ASSERT(g.ok() && r_b != nullptr);

    // The panel is [k group][nr][4] bytes, so column c of a group is the four
    // bytes at c*4 — the same shape on both arches, only nr differs. A short
    // final group and the columns past N are what the packer zero-filled.
    const int cols = static_cast<int>(g.nr);
    const int64_t nb = (K + g.seg - 1) / g.seg;
    const int8_t* const blob = packed.ptr<int8_t>();
    const int64_t tasks = std::min<int64_t>(g.np,
        std::max<int64_t>(ctx.cpu.thread_count(), 1));
    ctx.cpu.run(0, tasks, [&](int64_t t) {
        const int64_t p0 = t * g.np / tasks;
        const int64_t p1 = (t + 1) * g.np / tasks;
        for (int64_t p = p0; p < p1; ++p) {
            int32_t* const out = r_b + p * cols;
            const int live = static_cast<int>(std::min(g.nr, N - p * g.nr));
            for (int c = 0; c < live; ++c) { out[c] = 0; }
            for (int64_t b = 0; b < nb; ++b) {
                const int kg = static_cast<int>((std::min(g.seg, K - b * g.seg) + 3) / 4);
                const int8_t* panel = blob + (b * g.np + p) * g.ldd;
                for (int grp = 0; grp < kg; ++grp, panel += cols * 4) {
                    for (int c = 0; c < live; ++c) {
                        const int8_t* q = panel + c * 4;
                        out[c] += q[0] + q[1] + q[2] + q[3];
                    }
                }
            }
        }
    });
}

// =========================================================================
//  int8 (s8×s8) pack + MMA dispatch
// =========================================================================
// Same split as the float route; the accumulation is int32.

using PackI8Fn = void (*)(void* NNOPS_RESTRICT output,
                          const void* NNOPS_RESTRICT input,
                          int ir_step, int K, float scale);

using PackI8WideFn = void (*)(void* NNOPS_RESTRICT output,
                              const void* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int nr);

// LHS cascade: a panel is a level's full row count, so no width argument.
constexpr std::array<PackI8Fn, 3> pack_trans_i8_lhs = {{
#ifdef NNOPS_ARCH_X86_64
    pack_trans_n6_dp4a_i8, pack_trans_n4_dp4a_i8, pack_trans_n1_dp4a_i8,
#elif defined(NNOPS_ARCH_AARCH64)
    pack_trans_n8_dp4a_i8, pack_trans_n4_dp4a_i8, pack_trans_n1_dp4a_i8,
#endif
}};

constexpr std::array<PackI8Fn, 3> pack_copy_i8_lhs = {{
#ifdef NNOPS_ARCH_X86_64
    pack_copy_n6_dp4a_i8, pack_copy_n4_dp4a_i8, pack_copy_n1_dp4a_i8,
#elif defined(NNOPS_ARCH_AARCH64)
    pack_copy_n8_dp4a_i8, pack_copy_n4_dp4a_i8, pack_copy_n1_dp4a_i8,
#endif
}};

#ifdef NNOPS_ARCH_X86_64
constexpr PackI8WideFn pack_i8_rhs_trans = pack_trans_n16_dp4a_i8;
constexpr PackI8WideFn pack_i8_rhs_copy  = pack_copy_n16_dp4a_i8;
#elif defined(NNOPS_ARCH_AARCH64)
constexpr PackI8WideFn pack_i8_rhs_trans = pack_trans_n12_dp4a_i8;
constexpr PackI8WideFn pack_i8_rhs_copy  = pack_copy_n12_dp4a_i8;
#endif

void xor0x80_i8(int8_t* p, int nbytes) {
    uint8_t* u = reinterpret_cast<uint8_t*>(p);
    int i = 0;
#if defined(NNOPS_ARCH_X86_64)
    const __m256i mask = _mm256_set1_epi32(0x80808080);
    for (; i + 32 <= nbytes; i += 32) {
        __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(u + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(u + i), _mm256_xor_si256(x, mask));
    }
#elif defined(NNOPS_ARCH_AARCH64)
    const uint8x16_t mask = vdupq_n_u8(0x80);
    for (; i + 16 <= nbytes; i += 16) {
        vst1q_u8(u + i, veorq_u8(vld1q_u8(u + i), mask));
    }
#endif
    for (; i < nbytes; ++i) {
        u[i] ^= 0x80u;
    }
}

/// int8 A pack. With @p a_biased every panel is XOR-biased (see
/// int8_a_pack_bias): s8 → u8 on x86, u8 → s8 on aarch64.
void tile_pack_lhs_i8(bool trans, int mc, int kc, int8_t* dst, int ldd,
                      const int8_t* src, int lds, bool a_biased) {
    // transposed source is a plain copy; row-major needs the transposing kernel.
    const PackI8Fn* fns = trans ? pack_copy_i8_lhs.data() : pack_trans_i8_lhs.data();
    const int pack_lds = trans ? 1 : lds;
    const int kbytes   = (kc + 3) & ~3;  // pack writes ceil(kc/4)*4 bytes per row

    int i = 0;
    walk_levels(i, mc, MR_I8, [&](int level, int s) {
        fns[level](dst, src, lds, kc, 0.0f);
        if (a_biased) { xor0x80_i8(dst, s * kbytes); }
        dst += ldd;
        src += s * pack_lds;
    });
}

/// Quantise one panel's rows into a stack copy, then run the ordinary int8 pack
/// on it — the same bytes the materialise-then-pack route produces, without the
/// full M×K intermediate.
template <class AT, class QT>
void pack_quant_lhs_i8_impl(int mc, int kc, int8_t* dst, int ldd,
                            const AT* src, int lds,
                            const float* scale, const int32_t* zero,
                            float scale_scalar, int32_t zero_scalar,
                            bool a_biased) {
    NNOPS_ASSERT(kc <= KC_CAP_I8);
    const int kbytes = (kc + 3) & ~3;
    // One level at a time: at most MR_MAX_I8 rows of at most kbytes elements.
    QT tmp[MR_MAX_I8 * (static_cast<int>(KC_CAP_I8) + 3)];

    int i = 0;
    walk_levels(i, mc, MR_I8, [&](int level, int s) {
        for (int r = 0; r < s; ++r) {
            const float sc = (scale != nullptr) ? scale[i + r] : scale_scalar;
            // The quant kernel takes a float zero-point array; null means no
            // zero point, as in the materialised route.
            float zf = (zero != nullptr) ? static_cast<float>(zero[i + r])
                                         : static_cast<float>(zero_scalar);
            const float* zp = (zero != nullptr || zero_scalar != 0) ? &zf : nullptr;
            kernel::quant_kernel::quantization<QT>(
                1, kc, tmp + r * kbytes, kbytes, src + (i + r) * lds, lds, &sc, zp);
        }
        pack_trans_i8_lhs[level](dst, tmp, kbytes, kc, 0.0f);
        if (a_biased) { xor0x80_i8(dst, s * kbytes); }
        dst += ldd;
    });
}

template <class AT>
void pack_quant_lhs_i8(int mc, int kc, int8_t* dst, int ldd,
                       const AT* src, int lds,
                       const float* scale, const int32_t* zero,
                       float scale_scalar, int32_t zero_scalar,
                       bool u8, bool a_biased) {
    // Only the clamp range differs between the two quantised storage types.
    if (u8) {
        pack_quant_lhs_i8_impl<AT, uint8_t>(mc, kc, dst, ldd, src, lds, scale, zero,
                                            scale_scalar, zero_scalar, a_biased);
    } else {
        pack_quant_lhs_i8_impl<AT, int8_t>(mc, kc, dst, ldd, src, lds, scale, zero,
                                           scale_scalar, zero_scalar, a_biased);
    }
}

void tile_pack_quant_lhs_i8(int mc, int kc, int8_t* dst, int ldd,
                            const float* src, int lds,
                            const float* scale, const int32_t* zero,
                            float scale_scalar, int32_t zero_scalar,
                            bool u8, bool a_biased) {
    pack_quant_lhs_i8<float>(mc, kc, dst, ldd, src, lds, scale, zero,
                             scale_scalar, zero_scalar, u8, a_biased);
}

void tile_pack_quant_lhs_i8(int mc, int kc, int8_t* dst, int ldd,
                            const half* src, int lds,
                            const float* scale, const int32_t* zero,
                            float scale_scalar, int32_t zero_scalar,
                            bool u8, bool a_biased) {
    pack_quant_lhs_i8<half>(mc, kc, dst, ldd, src, lds, scale, zero,
                            scale_scalar, zero_scalar, u8, a_biased);
}

/// B into uniform panels. On x86 each panel's column sums are appended after
/// its data (zeroed when @p a_biased is false); the MMA seeds from them.
void tile_pack_rhs_i8(bool trans, int nc, int kc, int8_t* dst, int ldd,
                      const int8_t* src, int lds, bool a_biased,
                      bool src_shift) {
    constexpr int nr_max = NR_MAX_I8;
    const PackI8WideFn fn = trans ? pack_i8_rhs_trans : pack_i8_rhs_copy;
    // transposed source: a panel is nr_max rows of K; row-major: nr_max columns.
    const int step   = trans ? lds : 1;
    const int kbytes = (kc + 3) & ~3;

    for_each_n_panel(nc, nr_max, [&](int w) {
        fn(dst, src, lds, kc, 0.0f, w);
        // Only the data lanes are shifted; the correction block after them must
        // then be summed from the shifted bytes, so this has to come first.
        if (src_shift) { xor0x80_i8(dst, nr_max * kbytes); }
        mma_i8_b_panel_sums(dst, kbytes, a_biased);
        dst += ldd;
        src += w * step;
    });
}

/// One MMA call per M panel; the kernel walks the N side itself.
void tile_mma_pack_i8(int mc, int nc, int kc,
                      int32_t* c, int ldc,
                      const int8_t* packed_a, const int8_t* packed_b, int ldb,
                      int32_t clamp_min, int32_t clamp_max, bool zero_mode) {
    const int ldd_a = static_cast<int>(GEOM_I8.ldd_a(kc));
    const auto& fns = zero_mode ? mma_pack_fns_i8<true> : mma_pack_fns_i8<false>;

    int m = 0;
    walk_levels(m, mc, MR_I8, [&](int level, int s) {
        fns[level](s, nc, kc, c + m * ldc, ldc, packed_a, packed_b, ldb,
                   clamp_min, clamp_max);
        packed_a += ldd_a;
    });
}

void tile_mma_direct_i8(int mc, int nc, int kc,
                        int32_t* c, int ldc,
                        const int8_t* a, int lda, const int8_t* packed_b, int ldb,
                        int32_t clamp_min, int32_t clamp_max, bool zero_mode) {
    const auto& fns = zero_mode ? mma_direct_fns_i8<true>
                                : mma_direct_fns_i8<false>;

    // Same MR cascade as the packed route; only the A operand moves.
    int m = 0;
    walk_levels(m, mc, MR_I8, [&](int level, int s) {
        fns[level](s, nc, kc, c + m * ldc, ldc, a + m * lda, lda, packed_b, ldb,
                   clamp_min, clamp_max);
    });
}

// ---- tile scale -----------------------------------------------------------

template <typename T>
void tile_scale_impl(T* c, int ldc, float scale, int M, int N) {
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

void tile_scale(float* c, int ldc, float scale, int M, int N) {
    tile_scale_impl(c, ldc, scale, M, N);
}

void tile_scale(half* c, int ldc, float scale, int M, int N) {
    tile_scale_impl(c, ldc, scale, M, N);
}

/// C = scale·C + bias. A zero scale replaces the tile, so it stores the bias
/// rather than trusting 0·C: the old C is arbitrary (a pooled output buffer),
/// and 0·±inf or 0·NaN is NaN, which would then ride the whole k sweep.
template <typename T>
void tile_scale_add_bias_impl(T* c, int ldc, float scale, const T* bias,
                              int M, int N) {
    using namespace nnops::simd;
    constexpr int L = simd_lane_for<T>;
    const bool replace = (scale == 0.0f);
    for (int m = 0; m < M; ++m) {
        T* row = c + m * ldc;
        auto v_scale = v_set1(row, scale);
        int i = 0;
        for (; i + L <= N; i += L) {
            auto v = replace
                ? v_load(bias + i)
                : v_add(v_mul(v_load(row + i), v_scale), v_load(bias + i));
            v_store(row + i, v);
        }
        for (; i < N; ++i) {
            s_store(&row[i], replace ? s_load(&bias[i])
                                     : s_load(&row[i]) * scale + s_load(&bias[i]));
        }
    }
}

void tile_scale_add_bias(float* c, int ldc, float scale, const float* bias,
                         int M, int N) {
    tile_scale_add_bias_impl(c, ldc, scale, bias, M, N);
}

void tile_scale_add_bias(half* c, int ldc, float scale, const half* bias,
                         int M, int N) {
    tile_scale_add_bias_impl(c, ldc, scale, bias, M, N);
}

MatMulPlan get_matmul_plan(const MatMulAttributes& attrs,
                          const TensorDesc& a_desc,
                          const TensorDesc& b_desc,
                          int num_threads,
                          bool use_thread_slots) {
    MatMulPlan plan{};  // the dtype-reject returns below promise an all-zero plan

    const auto dt_a = a_desc.dtype;
    const auto dt_b = b_desc.dtype;

    // A block-packed B (q4_1) is self-describing, so A only has to be a float type.
    const bool block_rhs = is_block_packed_dtype(dt_b) &&
                           (dt_a == DataType::f32 || dt_a == DataType::f16);
    const bool int8_pair = is_int8_matmul(dt_a, dt_b);
    if (dt_a != dt_b && !block_rhs && !int8_pair) {
        return plan;
    }

    const MatmulGeom* geom = nullptr;
    if (int8_pair)                      { geom = &GEOM_I8; }
    else if (dt_a == DataType::f32)     { geom = &GEOM_F32; }
    else if (dt_a == DataType::f16)     { geom = &GEOM_F16; }
    else                                { return plan; }
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
        !(dt_a == DataType::f32 || dt_a == DataType::f16)) {
        return plan;
    }

    const int64_t M = attrs.transpose_a ? a_desc.dims[static_cast<size_t>(a_rank - 1)]
                                        : a_desc.dims[static_cast<size_t>(a_rank - 2)];
    const int64_t N = packed_b ? b_desc.dims[0]
                   : attrs.transpose_b ? b_desc.dims[static_cast<size_t>(b_rank - 2)]
                                       : b_desc.dims[static_cast<size_t>(b_rank - 1)];
    const int64_t K = packed_b
        ? (attrs.transpose_a ? a_desc.dims[static_cast<size_t>(a_rank - 2)]
                             : a_desc.dims[static_cast<size_t>(a_rank - 1)])
        : (attrs.transpose_b ? b_desc.dims[static_cast<size_t>(b_rank - 1)]
                             : b_desc.dims[static_cast<size_t>(b_rank - 2)]);

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
        choose_tile_impl(M, N, K, geom->mr_max, geom->nr_max, geom->elem,
                         num_threads, plan.split_n, geom->kc_cap, bg.seg,
                         features.l1_cache_size(), features.l2_shared_cache_size(),
                         plan.mc, plan.nc, plan.kc);
    } else {
        choose_matmul_tile(M, N, K, geom->mr_max, geom->nr_max, geom->elem,
                           num_threads, plan.split_n, geom->kc_cap,
                           features.l1_cache_size(), features.l2_shared_cache_size(),
                           plan.mc, plan.nc, plan.kc);
    }

    // Sizing stride at full Kc.
    const int64_t ldd_b = geom->ldd_b(plan.kc);
    plan.ldd_b = ldd_b;

    // Packed-B scratch. N-split: one slice per n-block. M-split: one per
    // m-block, or per worker thread when ids are available and blocks outnumber
    // them (see MatMulPlan::num_slots). int8 hoists B, so one slot serves all.
    const int64_t slice_bytes = static_cast<int64_t>(panel_count(static_cast<int>(N)))
                              * ldd_b * geom->elem;

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
        const int64_t nt          = std::max<int64_t>(num_threads, 1);
        if (nt <= 1) {
            plan.num_slots_a = 1;
        } else if (use_thread_slots && nt < nconcurrent) {
            plan.num_slots_a = nt;
        } else {
            plan.num_slots_a = nconcurrent;
        }

        plan.ldd_a = geom->ldd_a(plan.kc);
        plan.np_a = num_panels_max(static_cast<int>(plan.mc), geom->mr_levels);

        plan.workspace_size = plan.pack_a_offset
                            + plan.num_slots_a * plan.np_a * plan.ldd_a * geom->elem;
    }

    return plan;
}

}  // namespace nnops::backend::cpu
