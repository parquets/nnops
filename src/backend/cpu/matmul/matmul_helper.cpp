#include "matmul_helper.h"

#include <array>

#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/q4_1.hpp"
#include "nnops/detail/q5_1.hpp"
#include "nnops/detail/simd.hpp"
#include "backend/cpu/simd_kernel/simd_quant.hpp"

#ifdef NNOPS_ARCH_X86_64
using namespace nnops::backend::cpu::x86_64;
#elif defined(NNOPS_ARCH_AARCH64)
using namespace nnops::backend::cpu::aarch64;
#endif

namespace nnops::backend::cpu {

// The trailing live width lets one kernel serve a full panel and a partial one.

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

// MMA rows follow MR_F32 / MR_F16 order; a remainder runs at the level above.

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

// Panels sit at the caller's aligned ldd stride, not end to end.

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
// The MMA reads nr_max B lanes per column, so B is ceil(nc/nr_max) uniform
// panels at one stride; one nr_max-wide pack kernel pads each tail itself.

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

/// One M panel; the kernel walks the N side itself.
template <class T, bool Zero>
void tile_mma_pack_impl(int Mc, int Nc, int Kc, T* c, int ldc,
                        const T* packed_a, const T* b,
                        float clamp_min, float clamp_max) {
    using Tr = TileTraits<T>;
    const auto& fns = Tr::template mma_pack<Zero>();
    const int ldd_a = panel_stride<T>(Tr::mr()[0], Kc);

    int m = 0;
    walk_levels(m, Mc, Tr::mr(), [&](int level, int s) {
        fns[level](s, Nc, Kc, c + m * ldc, ldc, packed_a, b, clamp_min, clamp_max);
        packed_a += ldd_a;
    });
}

/// Same walk, A read raw through @p lda.
template <class T, bool Zero>
void tile_mma_direct_impl(int Mc, int Nc, int Kc, T* c, int ldc,
                          const T* a, int lda, const T* b,
                          float clamp_min, float clamp_max) {
    using Tr = TileTraits<T>;
    const auto& fns = Tr::template mma_direct<Zero>();

    int m = 0;
    walk_levels(m, Mc, Tr::mr(), [&](int level, int s) {
        fns[level](s, Nc, Kc, c + m * ldc, ldc, a + m * lda, lda, b, clamp_min, clamp_max);
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

// ---- Block-packed RHS (q5_1 / q4_1) --------------------------------------
// The panel layout matches tile_pack_rhs, so the MMA route knows no block
// format: it reads the same uniform panels whichever stream they came from.

#ifdef NNOPS_ARCH_X86_64

void tile_pack_rhs_q5_1(bool trans, int n_start, int nc, int k_start, int kc,
                        float* dst, int ldd, const void* src, int64_t src_row_blocks,
                        float scale) {
    arch::pack_q5_1_rhs<float>(trans, n_start, nc, k_start, kc, dst, ldd, src,
                               src_row_blocks, scale);
}

void tile_pack_rhs_q5_1(bool trans, int n_start, int nc, int k_start, int kc,
                        half* dst, int ldd, const void* src, int64_t src_row_blocks,
                        float scale) {
    arch::pack_q5_1_rhs<half>(trans, n_start, nc, k_start, kc, dst, ldd, src,
                              src_row_blocks, scale);
}

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
/// The panel width must be the active arch's float RHS panel width, because the
/// workspace stride (ldd_b) and the MMA kernels are sized from it — it is not
/// the same on every arch (aarch64 f32 uses nr_f32[0] = 12, both x86 widths and
/// the aarch64 f16 one are 16).
template <class T>
constexpr int block_panel_nr() noexcept {
    if constexpr (std::is_same_v<T, float>) { return NR_MAX_F32; }
    else { return NR_MAX_F16; }
}

/// Values per block of both formats. The scalar pack below indexes blocks with
/// shifts, which is the 32-value layout q5_1 and q4_1 share.
constexpr int kBlockElems = 32;
static_assert(q5_1_block_elems == kBlockElems && q4_1_block_elems == kBlockElems,
              "tile_pack_rhs_block_scalar indexes 32-value blocks");

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

void tile_pack_rhs_q5_1(bool trans, int n_start, int nc, int k_start, int kc,
                        float* dst, int ldd, const void* src, int64_t src_row_blocks,
                        float scale) {
    tile_pack_rhs_block_scalar<float>(trans, n_start, nc, k_start, kc, dst, ldd,
                                      src, src_row_blocks, scale,
                                      q5_1_block_bytes, q5_1_unpack_block);
}

void tile_pack_rhs_q5_1(bool trans, int n_start, int nc, int k_start, int kc,
                        half* dst, int ldd, const void* src, int64_t src_row_blocks,
                        float scale) {
    tile_pack_rhs_block_scalar<half>(trans, n_start, nc, k_start, kc, dst, ldd,
                                     src, src_row_blocks, scale,
                                     q5_1_block_bytes, q5_1_unpack_block);
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

void tile_pack_rhs_block(DataType dt, bool trans, int n_start, int nc, int k_start,
                         int kc, float* dst, int ldd, const void* src,
                         int64_t src_row_blocks, float scale) {
    if (dt == DataType::q4_1) {
        tile_pack_rhs_q4_1(trans, n_start, nc, k_start, kc, dst, ldd, src,
                           src_row_blocks, scale);
    } else {
        tile_pack_rhs_q5_1(trans, n_start, nc, k_start, kc, dst, ldd, src,
                           src_row_blocks, scale);
    }
}

void tile_pack_rhs_block(DataType dt, bool trans, int n_start, int nc, int k_start,
                         int kc, half* dst, int ldd, const void* src,
                         int64_t src_row_blocks, float scale) {
    if (dt == DataType::q4_1) {
        tile_pack_rhs_q4_1(trans, n_start, nc, k_start, kc, dst, ldd, src,
                           src_row_blocks, scale);
    } else {
        tile_pack_rhs_q5_1(trans, n_start, nc, k_start, kc, dst, ldd, src,
                           src_row_blocks, scale);
    }
}

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
// Same split as the float route. x86's VNNI bias lives entirely in the packed
// layouts; both arches return the exact s8×s8 product.

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

/// int8 A pack. When A needs the hardware shift (see int8_a_pack_bias) every
/// panel is XOR-biased: s8 → u8 for VNNI, u8 → s8 for SDOT.
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

/// Quantise one panel's rows into a stack copy and run the ordinary int8 pack
/// on it. The panel layout and the rounding are therefore the same bytes the
/// materialise-then-pack route produces; only the full M×K intermediate is gone.
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
            // The quant kernel takes a float zero-point array; a null pointer
            // means "no zero point", exactly as in the materialised route.
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
                      const int8_t* packed_a, const int8_t* packed_b,
                      int32_t clamp_min, int32_t clamp_max) {
    const int ldd_a = static_cast<int>(GEOM_I8.ldd_a(kc));
    const auto& fns = mma_pack_fns_i8<false>;

    int m = 0;
    walk_levels(m, mc, MR_I8, [&](int level, int s) {
        fns[level](s, nc, kc, c + m * ldc, ldc, packed_a, packed_b, clamp_min, clamp_max);
        packed_a += ldd_a;
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

template <typename T>
void tile_scale_add_bias_impl(T* c, int ldc, float scale, const T* bias,
                              int M, int N) {
    using namespace nnops::simd;
    constexpr int L = simd_lane_for<T>;
    for (int m = 0; m < M; ++m) {
        T* row = c + m * ldc;
        auto v_scale = v_set1(row, scale);
        int i = 0;
        for (; i + L <= N; i += L) {
            auto v = v_add(v_mul(v_load(row + i), v_scale), v_load(bias + i));
            v_store(row + i, v);
        }
        for (; i < N; ++i) {
            s_store(&row[i], s_load(&row[i]) * scale + s_load(&bias[i]));
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

    // A block-packed B (q5_1/q4_1) is self-describing, so A only has to be a
    // float type.
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
    else                                { return plan; }  // the reference path takes the rest
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

    // Pack A for correctness (transpose_a), for int8, or for a deep K.
    plan.pack_a = geom->packs_a_always || attrs.transpose_a || (K > PACK_A_K_THRESHOLD);

    // Parallel split, as MLAS: the larger dimension.
    plan.split_n = (N > M);
    const bool split_m = !plan.split_n;

    // Uniform panels for every dtype: the count is monotonic.
    auto panel_count = [&](int n) { return (n + geom->nr_max - 1) / geom->nr_max; };

    const auto& features = simd::CpuFeatures::get();
    choose_matmul_tile(M, N, K, geom->mr_max, geom->nr_max, geom->elem, num_threads,
                       plan.split_n, geom->kc_cap,
                       features.l1_cache_size(), features.l2_shared_cache_size(),
                       plan.mc, plan.nc, plan.kc);

    // Sizing stride at full Kc.
    const int64_t ldd_b = geom->ldd_b(plan.kc);
    plan.ldd_b = ldd_b;

    // Packed-B scratch. N-split: one slice per n-block. M-split: one per
    // m-block, or one per worker thread when ids are available (see
    // MatMulPlan::num_slots). int8 hoists the pack, so one slot serves all.
    const int64_t slice_bytes = static_cast<int64_t>(panel_count(static_cast<int>(N)))
                              * ldd_b * geom->elem;

    const int64_t num_m_blocks = split_block_count(M, plan.mc);
    const int64_t num_n_blocks = (N + plan.nc - 1) / plan.nc;

    if (split_m) {
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
