/// @file matmul_helper_i8.cpp
/// @brief The int8 half of the arch-neutral pack/MMA layer: the s8 packers, the
/// integer MMA dispatch, and the int8 prepacked-RHS readers.
///
/// Same contract as matmul_helper.cpp — A decomposes into the arch's MR cascade,
/// B is uniform nr_max-wide panels at one 64-byte-aligned stride, panels sit at
/// the caller's `ldd` — with int32 accumulators and the group-major layout: four
/// int8 per int32 along K, A as [row0(4B) row1(4B) …] and B as [col0(4B) …] per
/// group, a partial final group zero-filled. On x86 each B panel carries a
/// trailing correction block (column sums) at int8_panel_stride; aarch64 has
/// none. FEAT_I8MM switches aarch64 to SMMLA (8-k groups) at runtime inside this
/// layer, so no caller forks.
///
/// The float twin is matmul_helper.cpp; the declarations both sides share are in
/// matmul_helper.h.

#include "matmul_helper.h"
#include "matmul_pack_walk.h"

#include <array>

#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/simd.hpp"
#include "backend/cpu/aarch64/sme/mma_sme.hpp"
#include "backend/cpu/simd_kernel/simd_quant.hpp"
#if defined(NNOPS_ARCH_SME)
#include "backend/cpu/aarch64/sme/mma_i8_sme.hpp"
#include "backend/cpu/aarch64/sme/pack_i8_sme.hpp"
#endif

#ifdef NNOPS_ARCH_X86_64
using namespace nnops::backend::cpu::x86_64;
#elif defined(NNOPS_ARCH_AARCH64)
using namespace nnops::backend::cpu::aarch64;
#endif

namespace nnops::backend::cpu {

void matmul_packed_b_sme_i8_pack(const TensorView& raw, TensorView& packed, bool transpose_b)
{
#if defined(NNOPS_ARCH_SME)
    const int64_t N = transpose_b ? raw.shape(raw.rank() - 2) : raw.shape(raw.rank() - 1);
    const int64_t K = transpose_b ? raw.shape(raw.rank() - 1) : raw.shape(raw.rank() - 2);
    aarch64::pack_b_i8_sme(static_cast<int>(N), static_cast<int>(K), packed.ptr<int8_t>(),
                           raw.ptr<int8_t>(), raw.row_stride_elems(), transpose_b);
#else
    (void)raw;
    (void)packed;
    (void)transpose_b;
    NNOPS_ASSERT(!"matmul_packed_b_sme_i8_pack: SME blob on a non-SME build");
#endif
}

void matmul_packed_b_sme_i8_unpack(const TensorView& packed, int64_t K, int8_t* dst,
                                   const ComputeContext& ctx)
{
    const int64_t N = packed.shape(0);
    const int64_t kg = (K + 3) / 4;
    const int64_t nr = aarch64::sme_nr_max;
    const int64_t za = aarch64::sme_za_n;
    const int8_t* const blob = packed.ptr<int8_t>();
    const int64_t n_blocks = (N + nr - 1) / nr;
    ctx.cpu.run(0, n_blocks, [&](int64_t nb) {
        const int64_t n0 = nb * nr;
        const int64_t cols = std::min(nr, N - n0);
        const int8_t* const blk = blob + nb * kg * (4 * nr);
        for (int64_t g = 0; g < kg; ++g) {
            const int8_t* const grp = blk + g * (4 * nr);
            for (int64_t j = 0; j < cols; ++j) {
                const int8_t* const cell = grp + (j / za) * (4 * za) + (j % za) * 4;
                for (int p = 0; p < 4; ++p) {
                    const int64_t k = 4 * g + p;
                    if (k < K) {
                        dst[k * N + n0 + j] = cell[p];
                    }
                }
            }
        }
    });
}

void matmul_packed_b_i8_col_sums(const TensorView& packed, int64_t K, int32_t* r_b,
                                 const ComputeContext& ctx)
{
    const int64_t N = packed.shape(0);
    NNOPS_ASSERT(r_b != nullptr);

    if (packed.layout() == TensorLayout::PackedBSme) {
        // [n/64][k/4][256]: column j's four K values are contiguous at its
        // lane slot, panels of 16 columns 64 bytes apart.
        const int64_t nr = aarch64::sme_nr_max;
        const int64_t za = aarch64::sme_za_n;
        const int64_t kg = (K + 3) / 4;
        const int8_t* const blob = packed.ptr<int8_t>();
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
        return;
    }

    const MatMulPackedB g = matmul_packed_b_geom(N, K, DataType::s8);
    NNOPS_ASSERT(g.ok());

    // The panel is [k group][nr][group] bytes: four on SDOT (x86 too), eight on
    // i8mm. Column c of a group is that many contiguous bytes at c*group. A
    // short final group and the columns past N are what the packer zero-filled.
    const int group = i8mm_enabled() ? 8 : 4;
    const int cols = static_cast<int>(g.nr);
    const int64_t nb = (K + g.seg - 1) / g.seg;
    const int8_t* const blob = packed.ptr<int8_t>();
    const int64_t tasks = std::min<int64_t>(g.np, std::max<int64_t>(ctx.cpu.thread_count(), 1));
    ctx.cpu.run(0, tasks, [&](int64_t t) {
        const int64_t p0 = t * g.np / tasks;
        const int64_t p1 = (t + 1) * g.np / tasks;
        for (int64_t p = p0; p < p1; ++p) {
            int32_t* const out = r_b + p * cols;
            const int live = static_cast<int>(std::min(g.nr, N - p * g.nr));
            for (int c = 0; c < live; ++c) {
                out[c] = 0;
            }
            for (int64_t b = 0; b < nb; ++b) {
                const int kseg = static_cast<int>(std::min(g.seg, K - b * g.seg));
                const int kg = (kseg + group - 1) / group;
                const int8_t* panel = blob + (b * g.np + p) * g.ldd;
                for (int grp = 0; grp < kg; ++grp, panel += cols * group) {
                    for (int c = 0; c < live; ++c) {
                        const int8_t* q = panel + c * group;
                        for (int i = 0; i < group; ++i) {
                            out[c] += q[i];
                        }
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

using PackI8Fn = void (*)(void* NNOPS_RESTRICT output, const void* NNOPS_RESTRICT input,
                          int ir_step, int K, float scale);

using PackI8WideFn = void (*)(void* NNOPS_RESTRICT output, const void* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int nr);

// LHS cascade: a panel is a level's full row count, so no width argument.
constexpr std::array<PackI8Fn, 3> pack_trans_i8_lhs = {{
#ifdef NNOPS_ARCH_X86_64
    pack_trans_n6_dp4a_i8,
    pack_trans_n4_dp4a_i8,
    pack_trans_n1_dp4a_i8,
#elif defined(NNOPS_ARCH_AARCH64)
    pack_trans_n8_dp4a_i8,
    pack_trans_n4_dp4a_i8,
    pack_trans_n1_dp4a_i8,
#endif
}};

constexpr std::array<PackI8Fn, 3> pack_copy_i8_lhs = {{
#ifdef NNOPS_ARCH_X86_64
    pack_copy_n6_dp4a_i8,
    pack_copy_n4_dp4a_i8,
    pack_copy_n1_dp4a_i8,
#elif defined(NNOPS_ARCH_AARCH64)
    pack_copy_n8_dp4a_i8,
    pack_copy_n4_dp4a_i8,
    pack_copy_n1_dp4a_i8,
#endif
}};

#ifdef NNOPS_ARCH_AARCH64
// The SMMLA route's LHS ladder: same rungs, eight-k groups, one 16-byte
// fragment per row pair.
constexpr std::array<PackI8Fn, 3> pack_trans_i8_lhs_i8mm = {{
    pack_trans_n8_i8mm,
    pack_trans_n4_i8mm,
    pack_trans_n1_i8mm,
}};
constexpr std::array<PackI8Fn, 3> pack_copy_i8_lhs_i8mm = {{
    pack_copy_n8_i8mm,
    pack_copy_n4_i8mm,
    pack_copy_n1_i8mm,
}};
#endif

#ifdef NNOPS_ARCH_X86_64
constexpr PackI8WideFn pack_i8_rhs_trans = pack_trans_n16_dp4a_i8;
constexpr PackI8WideFn pack_i8_rhs_copy = pack_copy_n16_dp4a_i8;
#elif defined(NNOPS_ARCH_AARCH64)
constexpr PackI8WideFn pack_i8_rhs_trans = pack_trans_n12_dp4a_i8;
constexpr PackI8WideFn pack_i8_rhs_copy = pack_copy_n12_dp4a_i8;
#endif

#if defined(NNOPS_ARCH_SME)
void tile_pack_lhs_i8_sme(bool trans, int mc, int kc, int8_t* dst, const int8_t* src,
                          int lds, bool a_biased) noexcept
{
    aarch64::pack_a_i8_sme(mc, kc, dst, src, lds, trans, a_biased);
}

void tile_pack_rhs_i8_sme(bool trans, int nc, int kc, int8_t* dst, const int8_t* src,
                          int lds) noexcept
{
    aarch64::pack_b_i8_sme(nc, kc, dst, src, lds, trans);
}

void tile_mma_pack_i8_sme(int mc, int nc, int kc, int32_t* c, int ldc, const int8_t* packed_a,
                          const int8_t* b, int64_t block_stride, int32_t clamp_min,
                          int32_t clamp_max, bool zero_mode)
{
    aarch64::mma_i8_sme(mc, nc, kc, c, ldc, packed_a, b, zero_mode, clamp_min, clamp_max,
                        block_stride > 0 ? block_stride : 0);
}

void tile_mma_pack_i8_sme_epi(int mc, int nc, int kc, float* out, int64_t ldm,
                              const int8_t* packed_a, const int8_t* b, int64_t block_stride,
                              const float* s_b, float s_b_scalar, float s_a,
                              const float* col_bias, float lo, float hi)
{
    aarch64::mma_i8_sme_epi<float>(mc, nc, kc, out, ldm, packed_a, b,
                                   block_stride > 0 ? block_stride : 0, s_b, s_b_scalar, s_a,
                                   col_bias, lo, hi);
}

void tile_mma_pack_i8_sme_epi_i8(int mc, int nc, int kc, int8_t* out, int64_t ldm,
                                 const int8_t* packed_a, const int8_t* b, int64_t block_stride,
                                 const float* s_b, float s_b_scalar, float row_k,
                                 const int32_t* col_bias, int32_t zp_out, bool relu)
{
    aarch64::mma_i8_sme_epi_i8<int8_t>(mc, nc, kc, out, ldm, packed_a, b,
                                       block_stride > 0 ? block_stride : 0, s_b, s_b_scalar,
                                       row_k, col_bias, zp_out, relu);
}

void tile_mma_pack_i8_sme_epi_i8(int mc, int nc, int kc, uint8_t* out, int64_t ldm,
                                 const int8_t* packed_a, const int8_t* b, int64_t block_stride,
                                 const float* s_b, float s_b_scalar, float row_k,
                                 const int32_t* col_bias, int32_t zp_out, bool relu)
{
    aarch64::mma_i8_sme_epi_i8<uint8_t>(mc, nc, kc, out, ldm, packed_a, b,
                                        block_stride > 0 ? block_stride : 0, s_b, s_b_scalar,
                                        row_k, col_bias, zp_out, relu);
}

void tile_mma_pack_i8_sme_epi(int mc, int nc, int kc, half* out, int64_t ldm,
                              const int8_t* packed_a, const int8_t* b, int64_t block_stride,
                              const float* s_b, float s_b_scalar, float s_a,
                              const float* col_bias, float lo, float hi)
{
    aarch64::mma_i8_sme_epi<half>(mc, nc, kc, out, ldm, packed_a, b,
                                  block_stride > 0 ? block_stride : 0, s_b, s_b_scalar, s_a,
                                  col_bias, lo, hi);
}

#endif  // NNOPS_ARCH_SME

void xor0x80_i8(int8_t* p, int nbytes)
{
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
void tile_pack_lhs_i8(bool trans, int mc, int kc, int8_t* dst, int ldd, const int8_t* src, int lds,
                      bool a_biased)
{
    // transposed source is a plain copy; row-major needs the transposing kernel.
#ifdef NNOPS_ARCH_AARCH64
    const bool i8mm = i8mm_enabled();
    const PackI8Fn* fns =
        i8mm ? (trans ? pack_copy_i8_lhs_i8mm.data() : pack_trans_i8_lhs_i8mm.data()) :
               (trans ? pack_copy_i8_lhs.data() : pack_trans_i8_lhs.data());
    const int kbytes = i8mm ? ((kc + 7) & ~7) : ((kc + 3) & ~3);
#else
    const PackI8Fn* fns = trans ? pack_copy_i8_lhs.data() : pack_trans_i8_lhs.data();
    const int kbytes = (kc + 3) & ~3; // pack writes ceil(kc/4)*4 bytes per row
#endif
    const int pack_lds = trans ? 1 : lds;

    int i = 0;
    walk_levels_exact(i, mc, MR_I8, [&](int level, int s) {
        fns[level](dst, src, lds, kc, 0.0f);
        if (a_biased) {
            xor0x80_i8(dst, s * kbytes);
        }
        dst += ldd;
        src += s * pack_lds;
    });
}

/// Quantise one panel's rows into a stack copy, then run the ordinary int8 pack
/// on it — the same bytes the materialise-then-pack route produces, without the
/// full M×K intermediate.
template <class AT, class QT>
void pack_quant_lhs_i8_impl(int mc, int kc, int8_t* dst, int ldd, const AT* src, int lds,
                            const float* scale, const int32_t* zero, float scale_scalar,
                            int32_t zero_scalar, bool a_biased)
{
    NNOPS_ASSERT(kc <= KC_CAP_I8);
#ifdef NNOPS_ARCH_AARCH64
    const bool i8mm = i8mm_enabled();
    const int kbytes = i8mm ? ((kc + 7) & ~7) : ((kc + 3) & ~3);
    const PackI8Fn* const packers = i8mm ? pack_trans_i8_lhs_i8mm.data() : pack_trans_i8_lhs.data();
#else
    const int kbytes = (kc + 3) & ~3;
    const PackI8Fn* const packers = pack_trans_i8_lhs.data();
#endif
    // One level at a time: at most MR_MAX_I8 rows of at most kbytes elements.
    QT tmp[MR_MAX_I8 * (static_cast<int>(KC_CAP_I8) + 8)];

    int i = 0;
    walk_levels_exact(i, mc, MR_I8, [&](int level, int s) {
        for (int r = 0; r < s; ++r) {
            const float sc = (scale != nullptr) ? scale[i + r] : scale_scalar;
            // The quant kernel takes a float zero-point array; null means no
            // zero point, as in the materialised route.
            float zf = (zero != nullptr) ? static_cast<float>(zero[i + r]) :
                                           static_cast<float>(zero_scalar);
            const float* zp = (zero != nullptr || zero_scalar != 0) ? &zf : nullptr;
            kernel::quant_kernel::quantization<QT>(1, kc, tmp + r * kbytes, kbytes,
                                                   src + (i + r) * lds, lds, &sc, zp);
        }
        packers[level](dst, tmp, kbytes, kc, 0.0f);
        if (a_biased) {
            xor0x80_i8(dst, s * kbytes);
        }
        dst += ldd;
    });
}

template <class AT>
void pack_quant_lhs_i8(int mc, int kc, int8_t* dst, int ldd, const AT* src, int lds,
                       const float* scale, const int32_t* zero, float scale_scalar,
                       int32_t zero_scalar, bool u8, bool a_biased)
{
    // Only the clamp range differs between the two quantised storage types.
    if (u8) {
        pack_quant_lhs_i8_impl<AT, uint8_t>(mc, kc, dst, ldd, src, lds, scale, zero, scale_scalar,
                                            zero_scalar, a_biased);
    } else {
        pack_quant_lhs_i8_impl<AT, int8_t>(mc, kc, dst, ldd, src, lds, scale, zero, scale_scalar,
                                           zero_scalar, a_biased);
    }
}

void tile_pack_quant_lhs_i8(int mc, int kc, int8_t* dst, int ldd, const float* src, int lds,
                            const float* scale, const int32_t* zero, float scale_scalar,
                            int32_t zero_scalar, bool u8, bool a_biased)
{
    pack_quant_lhs_i8<float>(mc, kc, dst, ldd, src, lds, scale, zero, scale_scalar, zero_scalar, u8,
                             a_biased);
}

void tile_pack_quant_lhs_i8(int mc, int kc, int8_t* dst, int ldd, const half* src, int lds,
                            const float* scale, const int32_t* zero, float scale_scalar,
                            int32_t zero_scalar, bool u8, bool a_biased)
{
    pack_quant_lhs_i8<half>(mc, kc, dst, ldd, src, lds, scale, zero, scale_scalar, zero_scalar, u8,
                            a_biased);
}

void tile_pack_quant_lhs_i8(int mc, int kc, int8_t* dst, int ldd, const bfloat16* src, int lds,
                            const float* scale, const int32_t* zero, float scale_scalar,
                            int32_t zero_scalar, bool u8, bool a_biased)
{
    pack_quant_lhs_i8<bfloat16>(mc, kc, dst, ldd, src, lds, scale, zero, scale_scalar, zero_scalar,
                                u8, a_biased);
}

/// B into uniform panels. On x86 each panel's column sums are appended after
/// its data (zeroed when @p a_biased is false); the MMA seeds from them.
void tile_pack_rhs_i8(bool trans, int nc, int kc, int8_t* dst, int ldd, const int8_t* src, int lds,
                      bool a_biased, bool src_shift)
{
    constexpr int nr_max = NR_MAX_I8;
#ifdef NNOPS_ARCH_AARCH64
    // 8-k groups on the SMMLA route, 4-k on SDOT; the panel width is the same.
    const bool i8mm = i8mm_enabled();
    const PackI8WideFn fn = i8mm ? (trans ? pack_trans_n12_i8mm : pack_copy_n12_i8mm) :
                                   (trans ? pack_i8_rhs_trans : pack_i8_rhs_copy);
    const int kbytes = i8mm ? ((kc + 7) & ~7) : ((kc + 3) & ~3);
#else
    const PackI8WideFn fn = trans ? pack_i8_rhs_trans : pack_i8_rhs_copy;
    const int kbytes = (kc + 3) & ~3;
#endif
    // transposed source: a panel is nr_max rows of K; row-major: nr_max columns.
    const int step = trans ? lds : 1;

    for_each_n_panel(nc, nr_max, [&](int w) {
        fn(dst, src, lds, kc, 0.0f, w);
        // Only the data lanes are shifted; the correction block after them must
        // then be summed from the shifted bytes, so this has to come first.
        if (src_shift) {
            xor0x80_i8(dst, nr_max * kbytes);
        }
        mma_i8_b_panel_sums(dst, kbytes, a_biased);
        dst += ldd;
        src += w * step;
    });
}

using MmaPackI8Fn = void (*)(int, int, int, int32_t*, int, const int8_t*, const int8_t*, int,
                             int32_t, int32_t);
using MmaDirectI8Fn = void (*)(int, int, int, int32_t*, int, const int8_t*, int, const int8_t*, int,
                               int32_t, int32_t);

/// The rung's kernel for the route this process runs. Both routes share the MR
/// ladder and the panel width; only the bytes inside a B panel differ.
inline MmaPackI8Fn mma_pack_i8_fn(int level, bool zero_mode)
{
#ifdef NNOPS_ARCH_AARCH64
    if (i8mm_enabled()) {
        return zero_mode ? mma_pack_fns_i8mm<true>[level] : mma_pack_fns_i8mm<false>[level];
    }
#endif
    return zero_mode ? mma_pack_fns_i8<true>[level] : mma_pack_fns_i8<false>[level];
}

inline MmaDirectI8Fn mma_direct_i8_fn(int level, bool zero_mode)
{
#ifdef NNOPS_ARCH_AARCH64
    if (i8mm_enabled()) {
        return zero_mode ? mma_direct_fns_i8mm<true>[level] : mma_direct_fns_i8mm<false>[level];
    }
#endif
    return zero_mode ? mma_direct_fns_i8<true>[level] : mma_direct_fns_i8<false>[level];
}

/// One MMA call per M panel; the kernel walks the N side itself.
void tile_mma_pack_i8(int mc, int nc, int kc, int32_t* c, int ldc, const int8_t* packed_a,
                      const int8_t* packed_b, int ldb, int32_t clamp_min, int32_t clamp_max,
                      bool zero_mode)
{
    const int ldd_a = static_cast<int>(GEOM_I8.ldd_a(kc));

    int m = 0;
    walk_levels_exact(m, mc, MR_I8, [&](int level, int s) {
        mma_pack_i8_fn(level, zero_mode)(s, nc, kc, c + m * ldc, ldc, packed_a, packed_b, ldb,
                                         clamp_min, clamp_max);
        packed_a += ldd_a;
    });
}

void tile_mma_direct_i8(int mc, int nc, int kc, int32_t* c, int ldc, const int8_t* a, int lda,
                        const int8_t* packed_b, int ldb, int32_t clamp_min, int32_t clamp_max,
                        bool zero_mode)
{
    // Same MR cascade as the packed route; only the A operand moves.
    int m = 0;
    walk_levels_exact(m, mc, MR_I8, [&](int level, int s) {
        mma_direct_i8_fn(level, zero_mode)(s, nc, kc, c + m * ldc, ldc, a + m * lda, lda, packed_b,
                                           ldb, clamp_min, clamp_max);
    });
}

} // namespace nnops::backend::cpu
