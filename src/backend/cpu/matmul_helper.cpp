#include "matmul_helper.h"

#include <array>

#include "nnops/detail/half.hpp"
#include "nnops/detail/q5_1.hpp"
#include "nnops/detail/simd.hpp"

#ifdef NNOPS_ARCH_X86_64
using namespace nnops::backend::cpu::x86_64;
#elif defined(NNOPS_ARCH_AARCH64)
using namespace nnops::backend::cpu::aarch64;
#endif

namespace nnops::backend::cpu {

// ---- pack / MMA function pointer types ----------------------------------
//
// Every pack kernel carries a trailing live width so one kernel serves both
// the full panel and a partial one (last-row alias for A, zero fill for B).

template <class T>
using PackFn = void (*)(T* NNOPS_RESTRICT output, const T* NNOPS_RESTRICT input,
                        int ir_step, int K, float scale, int w);

/// A × packed-B (A packed) and A × packed-B (A read raw through lda).
template <class T>
using MmaPackFn = void (*)(int, int, int, T*, int, const T*, const T*, float, float);
template <class T>
using MmaDirectFn = void (*)(int, int, int, T*, int, const T*, int, const T*, float, float);

// ---- pack dispatch tables (largest-first decomposition) -----------------

// f32 dispatch tables
// Row 0: MR-based (LHS packing), Row 1: NR-based (RHS packing)
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

// f16 dispatch tables
// Row 0: MR-based (LHS packing), Row 1: NR-based (RHS packing)
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

// ---- per-dtype tile traits ---------------------------------------------
//
// Everything below that used to be written once for float and once for half
// differs only in the panel array and the dispatch tables, so those are named
// once here and the code is templated on the element type.

// ---- per-dtype tile traits ---------------------------------------------
//
// Rows of the arch MMA tables are the MR levels in MR_F32 / MR_F16 order; each
// kernel takes the runtime panel height, so a remainder panel below its own MR
// runs through the level above it. B is always packed, so the two families
// differ only in whether A is.

template <class T> struct TileTraits;

template <> struct TileTraits<float> {
    using Pack = PackFn<float>;
    static constexpr const int* mr() { return MR_F32; }
    static constexpr const int* nr() { return NR_F32; }
    static constexpr const auto& trans() { return pack_trans_f32_fn; }
    static constexpr const auto& copy()  { return pack_copy_f32_fn; }
    template <bool Z> static constexpr const auto& mma_pack()   { return mma_pack_fns_f32<Z>; }
    template <bool Z> static constexpr const auto& mma_direct() { return mma_direct_fns_f32<Z>; }
};

template <> struct TileTraits<half> {
    using Pack = PackFn<half>;
    static constexpr const int* mr() { return MR_F16; }
    static constexpr const int* nr() { return NR_F16; }
    static constexpr const auto& trans() { return pack_trans_f16_fn; }
    static constexpr const auto& copy()  { return pack_copy_f16_fn; }
    template <bool Z> static constexpr const auto& mma_pack()   { return mma_pack_fns_f16<Z>; }
    template <bool Z> static constexpr const auto& mma_direct() { return mma_direct_fns_f16<Z>; }
};

/// Uniform 64-byte-aligned panel stride, in elements, for a `rows × kc` panel.
template <class T>
constexpr int panel_stride(int rows, int kc) {
    return static_cast<int>(
        mma_pack_align(rows * kc * static_cast<int>(sizeof(T)))
        / static_cast<int>(sizeof(T)));
}

/// Walk one level of a largest-first decomposition: take `sizes[Level]`-sized
/// steps while they fit, handing each to `emit(panel_size, fn)`. Templating on
/// the level keeps `sizes[Level]` and `fns[Level]` compile-time indices, so the
/// three instantiations stay the straight-line loops the pack/MMA code was
/// written as.
template <int Level, class Fn, class Emit>
inline void panel_level(int& i, int count, const int* sizes,
                        const std::array<Fn, 3>& fns, Emit&& emit) {
    const int s = sizes[Level];
    for (; i + s <= count; i += s) { emit(s, fns[Level]); }
}

// ---- tiled pack entry points -------------------------------------------
// A panel's payload is contiguous (mr*kc elements for A, nr*kc for B) but is
// laid down at the caller's uniform 64-byte-aligned stride `ldd`, not packed
// end to end — matching the low-level pack kernels, which write their panel's
// output contiguously.

template <class T>
void tile_pack_lhs_impl(bool trans, int mc, int kc, T* dst, int ldd,
                        const T* src, int lds, float scale) {
    using Tr = TileTraits<T>;
    using Fn = typename Tr::Pack;
    // transpose_a stores A column-major, so packing it is a plain copy; the
    // row-major case is what needs the transposing kernel.
    const std::array<Fn, 3>& fns = trans ? Tr::copy()[0] : Tr::trans()[0];
    const int pack_lds = trans ? 1 : lds;

    int i = 0;
    auto emit = [&](int s, Fn fn) {
        fn(dst, src, lds, kc, scale, s);
        dst += ldd;
        src += s * pack_lds;
    };
    panel_level<0>(i, mc, Tr::mr(), fns, emit);
    panel_level<1>(i, mc, Tr::mr(), fns, emit);
    panel_level<2>(i, mc, Tr::mr(), fns, emit);
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
//  Uniform-width RHS panels — the mrxncxkc kernel route
// =========================================================================
//
// These kernels take a runtime mr/nc and read nr_max B lanes per k at every
// column, so B does not decompose per width: it is ceil(nc/nr_max) uniform
// panels, the last zero-filled past its live width, all at the same aligned
// panel stride. The packer below is the same nr_max-wide kernel for every
// panel: it takes the panel's live width and zero-pads the tail itself.
//
// nr_max is arch::nr_*[0] — 16 on x86_64, 12 for aarch64 f32 and 16 for
// aarch64 f16 — and each kernel family follows its arch's MR levels.

/// Pack B into uniform nr_max-wide panels. The kernel takes each panel's live
/// width, so the last panel needs no separate tail kernel: the transposing
/// kernel aliases its last live row into the tail columns, the row-major copy
/// zero-fills them. Either is fine — the MMA stores only the panel's live
/// columns, so tail garbage never reaches C. Panels advance by @p ldd, the
/// caller's aligned nr_max*kc slot stride — the same one the MMA route walks.
template <class T>
void tile_pack_rhs_impl(bool trans, int nc, int kc, T* dst, int ldd,
                        const T* src, int lds, float scale) {
    using Tr = TileTraits<T>;
    using Fn = typename Tr::Pack;
    constexpr int nr_max = Tr::nr()[0];
    const Fn fn = trans ? Tr::trans()[1][0] : Tr::copy()[1][0];
    // transposed source: a panel is nr_max rows of K; row-major: nr_max columns.
    const int step = trans ? lds : 1;

    int n = 0;
    for (; n + nr_max <= nc; n += nr_max) {
        fn(dst, src, lds, kc, scale, nr_max);
        dst += ldd;
        src += nr_max * step;
    }
    if (n < nc) { fn(dst, src, lds, kc, scale, nc - n); }
}

/// Walk the MR levels of an M tile, handing each panel to `emit(height, fn)`.
/// `m` stays in the caller, which is what advances C and A per panel.
template <class T, class Fn, class Emit>
inline void walk_m_panels(int& m, int mc, const std::array<Fn, 3>& fns, Emit&& emit) {
    const int* mr = TileTraits<T>::mr();
    panel_level<0>(m, mc, mr, fns, emit);
    panel_level<1>(m, mc, mr, fns, emit);
    panel_level<2>(m, mc, mr, fns, emit);
}

/// One M panel of a tile against the packed B. The kernel itself walks the N
/// side — its own panel loop advances B by the same aligned slot stride and
/// handles the last, zero-padded panel (storing only that panel's live
/// columns) — so a tile costs one call per M panel, not one per N panel.
template <class T, bool Zero>
void tile_mma_pack_impl(int Mc, int Nc, int Kc, T* c, int ldc,
                        const T* packed_a, const T* b,
                        float clamp_min, float clamp_max) {
    using Tr = TileTraits<T>;
    using Fn = MmaPackFn<T>;
    const auto& fns = Tr::template mma_pack<Zero>();
    const int ldd_a = panel_stride<T>(Tr::mr()[0], Kc);

    int m = 0;
    walk_m_panels<T>(m, Mc, fns, [&](int s, Fn fn) {
        fn(s, Nc, Kc, c + m * ldc, ldc, packed_a, b, clamp_min, clamp_max);
        packed_a += ldd_a;
    });
}

/// Same walk, A read raw through @p lda.
template <class T, bool Zero>
void tile_mma_direct_impl(int Mc, int Nc, int Kc, T* c, int ldc,
                          const T* a, int lda, const T* b,
                          float clamp_min, float clamp_max) {
    using Tr = TileTraits<T>;
    using Fn = MmaDirectFn<T>;
    const auto& fns = Tr::template mma_direct<Zero>();

    int m = 0;
    walk_m_panels<T>(m, Mc, fns, [&](int s, Fn fn) {
        fn(s, Nc, Kc, c + m * ldc, ldc, a + m * lda, lda, b, clamp_min, clamp_max);
    });
}

template <class T>
void dispatch_mma_pack(bool zero_mode, int Mc, int Nc, int Kc, T* c, int ldc,
                       const T* a, const T* b, float lo, float hi) {
    if (zero_mode) { tile_mma_pack_impl<T, true>(Mc, Nc, Kc, c, ldc, a, b, lo, hi); }
    else           { tile_mma_pack_impl<T, false>(Mc, Nc, Kc, c, ldc, a, b, lo, hi); }
}

template <class T>
void dispatch_mma_direct(bool zero_mode, int Mc, int Nc, int Kc, T* c, int ldc,
                         const T* a, int lda, const T* b, float lo, float hi) {
    if (zero_mode) { tile_mma_direct_impl<T, true>(Mc, Nc, Kc, c, ldc, a, lda, b, lo, hi); }
    else           { tile_mma_direct_impl<T, false>(Mc, Nc, Kc, c, ldc, a, lda, b, lo, hi); }
}

void tile_pack_rhs(bool trans, int nc, int kc,
                   float* dst, int ldd, const float* src, int lds, float scale) {
    tile_pack_rhs_impl(trans, nc, kc, dst, ldd, src, lds, scale);
}

void tile_pack_rhs(bool trans, int nc, int kc,
                   half* dst, int ldd, const half* src, int lds, float scale) {
    tile_pack_rhs_impl(trans, nc, kc, dst, ldd, src, lds, scale);
}

// ---- Q5_1 RHS -----------------------------------------------------------
//
// The fused unpack/dequantize panel pack lives in the arch header; this is
// only the type-erased entry point MatMul calls. The panel layout it writes is
// identical to tile_pack_rhs's, so the MMA route needs no Q5_1 knowledge.

#ifdef NNOPS_ARCH_X86_64

void tile_pack_rhs_q5_1(bool trans, int n_start, int nc, int k_start, int kc,
                        float* dst, int ldd, const void* src, int64_t src_row_blocks) {
    arch::pack_q5_1_rhs<float>(trans, n_start, nc, k_start, kc, dst, ldd, src,
                               src_row_blocks);
}

void tile_pack_rhs_q5_1(bool trans, int n_start, int nc, int k_start, int kc,
                        half* dst, int ldd, const void* src, int64_t src_row_blocks) {
    arch::pack_q5_1_rhs<half>(trans, n_start, nc, k_start, kc, dst, ldd, src,
                              src_row_blocks);
}

#else

/// Portable fallback for arches without a vectorized Q5_1 kernel: same uniform
/// panel layout, decoded one value at a time through the scalar reference in
/// <nnops/detail/q5_1.hpp>. Keeps MatMul correct everywhere; x86_64 uses the
/// SIMD pack in x86_64/pack_q5_1.hpp.
template <class T>
void tile_pack_rhs_q5_1_scalar(bool trans, int n_start, int nc, int k_start, int kc,
                               T* dst, int ldd, const void* src, int64_t row_blocks) {
    constexpr int NR = 16;
    const uint8_t* base = static_cast<const uint8_t*>(src);
    const int64_t row_bytes = row_blocks * static_cast<int64_t>(q5_1_block_bytes);

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
                float vals[q5_1_block_elems];
                q5_1_unpack_block(base + static_cast<int64_t>(row) * row_bytes
                                       + static_cast<int64_t>(col >> 5) * q5_1_block_bytes,
                                  vals);
                store(&o[j], vals[col & 31]);
            }
            for (int j = nr; j < NR; ++j) { o[j] = T{}; }
        }
        dst += ldd;
        n += nr;
    }
}

void tile_pack_rhs_q5_1(bool trans, int n_start, int nc, int k_start, int kc,
                        float* dst, int ldd, const void* src, int64_t src_row_blocks) {
    tile_pack_rhs_q5_1_scalar<float>(trans, n_start, nc, k_start, kc, dst, ldd,
                                     src, src_row_blocks);
}

void tile_pack_rhs_q5_1(bool trans, int n_start, int nc, int k_start, int kc,
                        half* dst, int ldd, const void* src, int64_t src_row_blocks) {
    tile_pack_rhs_q5_1_scalar<half>(trans, n_start, nc, k_start, kc, dst, ldd,
                                    src, src_row_blocks);
}

#endif

void tile_mma_pack(int Mc, int Nc, int Kc, float* c, int ldc,
                   const float* packed_a, const float* b,
                   float clamp_min, float clamp_max, bool zero_mode) {
    dispatch_mma_pack(zero_mode, Mc, Nc, Kc, c, ldc, packed_a, b, clamp_min, clamp_max);
}

void tile_mma_pack(int Mc, int Nc, int Kc, half* c, int ldc,
                   const half* packed_a, const half* b,
                   float clamp_min, float clamp_max, bool zero_mode) {
    dispatch_mma_pack(zero_mode, Mc, Nc, Kc, c, ldc, packed_a, b, clamp_min, clamp_max);
}

void tile_mma_direct(int Mc, int Nc, int Kc, float* c, int ldc,
                     const float* a, int lda, const float* b,
                     float clamp_min, float clamp_max, bool zero_mode) {
    dispatch_mma_direct(zero_mode, Mc, Nc, Kc, c, ldc, a, lda, b, clamp_min, clamp_max);
}

void tile_mma_direct(int Mc, int Nc, int Kc, half* c, int ldc,
                     const half* a, int lda, const half* b,
                     float clamp_min, float clamp_max, bool zero_mode) {
    dispatch_mma_direct(zero_mode, Mc, Nc, Kc, c, ldc, a, lda, b, clamp_min, clamp_max);
}

// =========================================================================
//  int8 (s8×s8) pack + MMA dispatch
// =========================================================================

using PackI8Fn = void (*)(void* NNOPS_RESTRICT output,
                          const void* NNOPS_RESTRICT input,
                          int ir_step, int K, float scale);

#ifdef NNOPS_ARCH_X86_64
using MmaPackI8Fn = void (*)(int32_t* NNOPS_RESTRICT C, int ldc,
                             const uint8_t* NNOPS_RESTRICT A,
                             const int8_t* NNOPS_RESTRICT B,
                             int K, int32_t clamp_min, int32_t clamp_max);
#else
using MmaPackI8Fn = void (*)(int32_t* NNOPS_RESTRICT C, int ldc,
                             const int8_t* NNOPS_RESTRICT A,
                             const int8_t* NNOPS_RESTRICT B,
                             int K, int32_t clamp_min, int32_t clamp_max);
#endif

// LHS (MR-based) / RHS (NR-based) pack tables. The RHS tables have 4 entries
// because NR_I8 has the extra nr=4 level.
constexpr std::array<PackI8Fn, 3> pack_trans_i8_lhs = {{
#ifdef NNOPS_ARCH_X86_64
    pack_trans_n6_dp4a_i8, pack_trans_n4_dp4a_i8, pack_trans_n1_dp4a_i8,
#elif defined(NNOPS_ARCH_AARCH64)
    pack_trans_n8_dp4a_i8, pack_trans_n4_dp4a_i8, pack_trans_n1_dp4a_i8,
#endif
}};

constexpr std::array<PackI8Fn, 4> pack_trans_i8_rhs = {{
#ifdef NNOPS_ARCH_X86_64
    pack_trans_n16_dp4a_i8, pack_trans_n8_dp4a_i8, pack_trans_n4_dp4a_i8, pack_trans_n1_dp4a_i8,
#elif defined(NNOPS_ARCH_AARCH64)
    pack_trans_n12_dp4a_i8, pack_trans_n8_dp4a_i8, pack_trans_n4_dp4a_i8, pack_trans_n1_dp4a_i8,
#endif
}};

constexpr std::array<PackI8Fn, 3> pack_copy_i8_lhs = {{
#ifdef NNOPS_ARCH_X86_64
    pack_copy_n6_dp4a_i8, pack_copy_n4_dp4a_i8, pack_copy_n1_dp4a_i8,
#elif defined(NNOPS_ARCH_AARCH64)
    pack_copy_n8_dp4a_i8, pack_copy_n4_dp4a_i8, pack_copy_n1_dp4a_i8,
#endif
}};

constexpr std::array<PackI8Fn, 4> pack_copy_i8_rhs = {{
#ifdef NNOPS_ARCH_X86_64
    pack_copy_n16_dp4a_i8, pack_copy_n8_dp4a_i8, pack_copy_n4_dp4a_i8, pack_copy_n1_dp4a_i8,
#elif defined(NNOPS_ARCH_AARCH64)
    pack_copy_n12_dp4a_i8, pack_copy_n8_dp4a_i8, pack_copy_n4_dp4a_i8, pack_copy_n1_dp4a_i8,
#endif
}};

// [mr idx][nr idx] — x86 VNNI (u8×s8) or aarch64 SDOT (s8×s8) micro-kernels.
constexpr std::array<std::array<MmaPackI8Fn, 4>, 3> mma_pack_i8_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {mma_pack_6x16_u8s8_vnni, mma_pack_6x8_u8s8_vnni, mma_pack_6x4_u8s8_vnni, mma_pack_6x1_u8s8_vnni},
    {mma_pack_4x16_u8s8_vnni, mma_pack_4x8_u8s8_vnni, mma_pack_4x4_u8s8_vnni, mma_pack_4x1_u8s8_vnni},
    {mma_pack_1x16_u8s8_vnni, mma_pack_1x8_u8s8_vnni, mma_pack_1x4_u8s8_vnni, mma_pack_1x1_u8s8_vnni},
#elif defined(NNOPS_ARCH_AARCH64)
    {mma_pack_8x12_s8s8_dot, mma_pack_8x8_s8s8_dot, mma_pack_8x4_s8s8_dot, mma_pack_8x1_s8s8_dot},
    {mma_pack_4x12_s8s8_dot, mma_pack_4x8_s8s8_dot, mma_pack_4x4_s8s8_dot, mma_pack_4x1_s8s8_dot},
    {mma_pack_1x12_s8s8_dot, mma_pack_1x8_s8s8_dot, mma_pack_1x4_s8s8_dot, mma_pack_1x1_s8s8_dot},
#endif
}};

void xor0x80_i8(int8_t* p, int nbytes) {
    uint8_t* u = reinterpret_cast<uint8_t*>(p);
    int i = 0;
#if defined(NNOPS_ARCH_X86_64)
    // 32 bytes (16 int8 groups) per AVX2 iteration; XOR 0x80 into each byte.
    const __m256i mask = _mm256_set1_epi32(0x80808080);
    for (; i + 32 <= nbytes; i += 32) {
        __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(u + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(u + i), _mm256_xor_si256(x, mask));
    }
#elif defined(NNOPS_ARCH_AARCH64)
    // 16 bytes (8 int8 groups) per NEON iteration.
    const uint8x16_t mask = vdupq_n_u8(0x80);
    for (; i + 16 <= nbytes; i += 16) {
        vst1q_u8(u + i, veorq_u8(vld1q_u8(u + i), mask));
    }
#endif
    // Scalar tail (nbytes is a multiple of 4, so at most 31 / 15 bytes remain).
    for (; i < nbytes; ++i) {
        u[i] ^= 0x80u;
    }
}

/// One int8 pack pass over the MR (A) or NR (B) cascade. The RHS lists have a
/// fourth level ({16,8,4,1} / {12,8,4,1}); only the LHS is XOR-biased for VNNI.
template <bool IsLhs>
void tile_pack_i8_impl(bool trans, int count, int kc, int8_t* dst, int ldd,
                       const int8_t* src, int lds) {
    constexpr int levels = IsLhs ? 3 : 4;
    const PackI8Fn* fns = IsLhs
        ? (trans ? pack_copy_i8_lhs : pack_trans_i8_lhs).data()
        : (trans ? pack_trans_i8_rhs : pack_copy_i8_rhs).data();
    const int* sizes   = IsLhs ? MR_I8 : NR_I8;
    const int pack_lds = IsLhs ? (trans ? 1 : lds) : (trans ? lds : 1);
    const int kbytes   = (kc + 3) & ~3;  // pack writes ceil(kc/4)*4 bytes per row

    int i = 0;
    for (int level = 0; level < levels; ++level) {
        const int s = sizes[level];
        for (; i + s <= count; i += s) {
            fns[level](dst, src, lds, kc, 0.0f);
            if constexpr (IsLhs && INT8_USE_U8_OFFSET) { xor0x80_i8(dst, s * kbytes); }
            dst += ldd;
            src += s * pack_lds;
        }
    }
}

void tile_pack_lhs_i8(bool trans, int mc, int kc, int8_t* dst, int ldd, const int8_t* src, int lds) {
    tile_pack_i8_impl<true>(trans, mc, kc, dst, ldd, src, lds);
}

void tile_pack_rhs_i8(bool trans, int nc, int kc, int8_t* dst, int ldd, const int8_t* src, int lds) {
    tile_pack_i8_impl<false>(trans, nc, kc, dst, ldd, src, lds);
}

void tile_mma_pack_i8(int mc, int nc, int kc,
                      int32_t* c, int ldc,
                      const int8_t* packed_a, const int8_t* packed_b,
                      int32_t clamp_min, int32_t clamp_max) {
    // The pack kernels write ceil(kc/4) groups (4 int8 per int32), zero-padding
    // the final partial group. Round K up so the MMA processes that padded tail
    // group too, and size the panel strides to the padded byte count so adjacent
    // panels never overlap.
    const int kbytes = (kc + 3) & ~3;
    const int ldd_a = static_cast<int>(mma_pack_align(MR_MAX_I8 * kbytes));
    const int ldd_b = static_cast<int>(mma_pack_align(NR_MAX_I8 * kbytes));
    const int k_groups = kbytes / 4;

    int m = 0;
    for (int mi = 0; mi < 3; ++mi) {
        const int mr = MR_I8[mi];
        for (; m + mr <= mc; m += mr) {
            int n = 0;
            const int8_t* b_panel = packed_b;
            for (int ni = 0; ni < 4; ++ni) {
                const int nr = NR_I8[ni];
                for (; n + nr <= nc; n += nr) {
                    int32_t* c_mn = c + m * ldc + n;
#ifdef NNOPS_ARCH_X86_64
                    mma_pack_i8_fn[static_cast<size_t>(mi)][static_cast<size_t>(ni)](
                        c_mn, ldc, reinterpret_cast<const uint8_t*>(packed_a), b_panel,
                        kbytes, clamp_min, clamp_max);
#else
                    mma_pack_i8_fn[static_cast<size_t>(mi)][static_cast<size_t>(ni)](
                        c_mn, ldc, packed_a, b_panel,
                        k_groups, clamp_min, clamp_max);
#endif
                    b_panel += ldd_b;
                }
            }
            packed_a += ldd_a;
        }
    }
}

// ---- tile scale -----------------------------------------------------------
// Uses 8-wide SIMD lanes (v_f32x8 / v_f16x8) with scalar tail for remainder.

template <typename T>
void tile_scale_impl(T* c, int ldc, float scale, int M, int N) {
    using namespace nnops::simd;
    constexpr int L = simd_lane_for<T>;  // 8 for both f32 and f16
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

MatMulPlan get_matmul_plan(const MatMulAttributes& attrs,
                          const TensorDesc& a_desc,
                          const TensorDesc& b_desc,
                          int num_threads,
                          bool use_thread_slots) {
    MatMulPlan plan{};  // the dtype-reject returns below promise an all-zero plan

    const auto dt_a = a_desc.dtype;
    const auto dt_b = b_desc.dtype;

    // A Q5_1 B is self-describing (its per-block d/m live in the data), so the
    // only compatibility requirement is that A has a float kernel to convert
    // the decoded values into: f32 or f16.
    const bool q5_1_rhs = (dt_b == DataType::q5_1) &&
                          (dt_a == DataType::f32 || dt_a == DataType::f16);
    if (dt_a != dt_b && !q5_1_rhs) {
        return plan;
    }

    const bool is_f32 = (dt_a == DataType::f32);
    const bool is_f16 = (dt_a == DataType::f16);
    const bool is_i8  = (dt_a == DataType::s8);
    if (!is_f32 && !is_f16 && !is_i8) {
        return plan;  // other dtypes are dispatched by the reference path
    }
    num_threads = std::max(num_threads, 1);

    const int64_t a_rank = a_desc.rank;
    const int64_t b_rank = b_desc.rank;
    NNOPS_ASSERT(a_rank >= 2 && b_rank >= 2);

    const int64_t M = attrs.transpose_a ? a_desc.dims[static_cast<size_t>(a_rank - 1)]
                                        : a_desc.dims[static_cast<size_t>(a_rank - 2)];
    const int64_t N = attrs.transpose_b ? b_desc.dims[static_cast<size_t>(b_rank - 2)]
                                        : b_desc.dims[static_cast<size_t>(b_rank - 1)];
    const int64_t K = attrs.transpose_b ? b_desc.dims[static_cast<size_t>(b_rank - 1)]
                                        : b_desc.dims[static_cast<size_t>(b_rank - 2)];


    // A is packed for correctness (transpose_a) or a page-scattered row stride.
    // int8 has no direct-A route at all — tile_pack_lhs_i8 is unconditional.
    int64_t lda = a_desc.row_stride_elems;
    if (lda == 0) { lda = attrs.transpose_a ? M : K; }
    plan.pack_a = is_i8 || attrs.transpose_a || (lda > PACK_A_STRIDE_THRESHOLD);

    // Split on the larger dimension (the dispatch's N > M rule).
    plan.split_n = (N > M);
    const bool split_m = !plan.split_n;

    // The M panel height the tile will run — the nc heuristic below charges the
    // L2 working set for it, so it must be the height the route really uses.
    const int  mr_panel = is_i8 ? MR_MAX_I8
                     : (is_f32 ? mr_max_flt<float>() : mr_max_flt<half>());
    const int  nr_max = is_i8 ? NR_MAX_I8 : (is_f32 ? nr_max_flt<float>() : nr_max_flt<half>());
    const int* nr     = is_i8 ? NR_I8 : (is_f32 ? NR_F32 : NR_F16);
    const int  elem   = is_i8 ? 1
                     : (is_f32 ? static_cast<int>(sizeof(float)) : static_cast<int>(sizeof(half)));

    // Panel count of an n-run. fp packs B into uniform nr_max-wide panels;
    // int8 keeps its per-width cascade (four levels: {16,8,4,1} / {12,8,4,1}).
    auto panel_count = [&](int n) {
        return is_i8 ? num_panels4(n, nr) : (n + nr_max - 1) / nr_max;
    };

    // The tile. mc, nc and kc are one problem now, not three — see the constants
    // block in matmul_helper.h for the objective and its constraints. Both L2
    // sources feed the same budget, so they have to be the same number: the
    // shared (performance-cluster) size.
    const auto& features = simd::CpuFeatures::get();
    const int64_t kc_cap = is_i8  ? KC_CAP_I8
                         : is_f32 ? KC_CAP_F32
                                  : KC_CAP_F16;
    choose_matmul_tile(M, N, K, mr_panel, nr_max, elem, num_threads, plan.split_n,
                       kc_cap, features.l1_cache_size(), features.l2_shared_cache_size(),
                       plan.mc, plan.nc, plan.kc);

    // Packed-B stride (elements) at full Kc.
    const int ldd_b = static_cast<int>(
        mma_pack_align(nr_max * static_cast<int>(plan.kc) * elem) / elem);
    plan.ldd_b = ldd_b;

    // Packed-B scratch. One full-N slice is `slice_bytes`; how many the
    // workspace holds, and what indexes them, depends on the split:
    //
    //   N-split  each n-block owns a slice one worst-case tile wide and the
    //            dispatch indexes them by block. Every block but the last packs
    //            exactly nc columns; the last packs `N mod nc`, anywhere in
    //            1..nc. The greedy NR decomposition is not monotonic in n (see
    //            num_panels_max: num_panels(nc - 1) exceeds num_panels(nc)), so
    //            the slice is sized at the largest count any tile of nc columns
    //            can emit — sizing it from num_panels(N) across the whole
    //            buffer, as this used to, is short by those few panels and the
    //            last pack writes past the end.
    //   M-split  every m-block gets its OWN full-N slice, because the blocks
    //            run concurrently and each packs its B per k-block. That is
    //            the duplication. When the backend reports worker ids the
    //            scratch goes one slot per worker thread instead: a thread
    //            packs one slice at a time, so num_threads slots serve
    //            num_m_blocks blocks (see MatMulPlan::num_slots). Every block
    //            packs the full N, so num_panels(N) is exact here.
    //
    // int8 is the exception on the M-split: its dispatch hoists the B pack out
    // of the parallel region (one pack per k-block, then M is partitioned), so
    // every m-block reads the same slice and one slot serves all of them.
    const int64_t slice_bytes = static_cast<int64_t>(panel_count(static_cast<int>(N)))
                              * static_cast<int64_t>(ldd_b) * elem;

    // The block counts of both splits. The packed-A region below needs the same
    // number either way: it is how many A tiles can be live at once.
    const int64_t num_m_blocks = split_block_count(M, plan.mc);
    const int64_t num_n_blocks = (N + plan.nc - 1) / plan.nc;

    if (split_m) {
        if (is_i8) {
            plan.num_slots = 1;
        } else {
            const int64_t nt = std::max<int64_t>(num_threads, 1);
            plan.num_slots = (use_thread_slots && nt < num_m_blocks) ? nt : num_m_blocks;
        }
        // Every m-block packs the full N, so the exact count is the bound.
        plan.np_slice = panel_count(static_cast<int>(N));
        plan.workspace_size = plan.num_slots * slice_bytes;
    } else {
        plan.num_slots = num_n_blocks;
        plan.np_slice = is_i8 ? num_panels_max4(static_cast<int>(plan.nc), nr)
                              : panel_count(static_cast<int>(plan.nc));
        plan.workspace_size = plan.num_slots * plan.np_slice
                            * static_cast<int64_t>(ldd_b) * elem;
    }

    // Packed-A scratch, appended after the packed-B region (whose offsets the
    // dispatch already computes from the workspace base and must keep).
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

        plan.ldd_a = static_cast<int64_t>(is_i8
            ? mma_pack_align(MR_MAX_I8 * ((static_cast<int>(plan.kc) + 3) & ~3))
            : mma_pack_align(mr_panel * static_cast<int>(plan.kc) * elem) / elem);
        const int* mr_a = is_i8 ? MR_I8 : (is_f32 ? MR_F32 : MR_F16);
        plan.np_a = num_panels_max(static_cast<int>(plan.mc), mr_a);

        plan.workspace_size = plan.pack_a_offset
                            + plan.num_slots_a * plan.np_a * plan.ldd_a * elem;
    }

    return plan;
}


}  // namespace nnops::backend::cpu
