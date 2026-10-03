#include "matmul_helper.h"

#include <array>

#include "nnops/detail/half.hpp"
#include "nnops/detail/simd.hpp"

#ifdef NNOPS_ARCH_X86_64
using namespace nnops::backend::cpu::x86_64;
#elif defined(NNOPS_ARCH_AARCH64)
using namespace nnops::backend::cpu::aarch64;
#endif

namespace nnops::backend::cpu {

// =========================================================================
//  Workspace sizing — int8 path only (fp sizing lives in get_matmul_plan)
// =========================================================================

size_t matmul_get_workspace_size(const MatMulAttributes& attrs,
                                 const TensorDesc& a_desc,
                                 const TensorDesc& b_desc,
                                 const TensorDesc& c_desc)
{
    // int8 (s8×s8): the tiled path always packs B, and an s8 output also needs
    // an int32 accumulator. Both live in the workspace. (The fp path computes
    // its own workspace through get_matmul_plan(), so this is never called for
    // f32/f16 and returns 0 there.)
    if (a_desc.dtype == DataType::s8 && b_desc.dtype == DataType::s8) {
        const int64_t M = attrs.transpose_a ? a_desc.dims[static_cast<size_t>(a_desc.rank - 1)]
                                            : a_desc.dims[static_cast<size_t>(a_desc.rank - 2)];
        const int64_t N = attrs.transpose_b ? b_desc.dims[static_cast<size_t>(b_desc.rank - 2)]
                                            : b_desc.dims[static_cast<size_t>(b_desc.rank - 1)];
        const int ldd_b = align_up<PANEL_ALIGN_BYTES>(NR_MAX_I8 * KC_I8);
        size_t total = static_cast<size_t>(num_panels4(static_cast<int>(N), NR_I8))
                     * static_cast<size_t>(ldd_b);
        if (c_desc.dtype == DataType::s8) {
            total += static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(int32_t);
        }
        return total;
    }
    return 0;
}

// ---- pack function pointer types ---------------------------------------

using PackF32Fn = void (*)(float* NNOPS_RESTRICT output,
                           const float* NNOPS_RESTRICT input,
                           int ir_step, int K, float scale);

using PackF16Fn = void (*)(half* NNOPS_RESTRICT output,
                           const half* NNOPS_RESTRICT input,
                           int ir_step, int K, float scale);

// ---- MMA-pack function pointer types -----------------------------------

using MmaPackF32Fn = void (*)(float* NNOPS_RESTRICT C, int ldc,
                              const float* NNOPS_RESTRICT A,
                              const float* NNOPS_RESTRICT B,
                              int ldb, int K,
                              float clamp_min, float clamp_max);

using MmaPackF16Fn = void (*)(half* NNOPS_RESTRICT C, int ldc,
                              const half* NNOPS_RESTRICT A,
                              const half* NNOPS_RESTRICT B,
                              int ldb, int K,
                              float clamp_min, float clamp_max);

// ---- MMA-direct function pointer types ---------------------------------

using MmaDirectF32Fn = void (*)(float* NNOPS_RESTRICT C, int ldc,
                                const float* NNOPS_RESTRICT A, int lda,
                                const float* NNOPS_RESTRICT B, int ldb,
                                int K, float clamp_min, float clamp_max);

using MmaDirectF16Fn = void (*)(half* NNOPS_RESTRICT C, int ldc,
                                const half* NNOPS_RESTRICT A, int lda,
                                const half* NNOPS_RESTRICT B, int ldb,
                                int K, float clamp_min, float clamp_max);



// ---- pack dispatch tables (largest-first decomposition) -----------------

// f32 dispatch tables
// Row 0: MR-based (LHS packing), Row 1: NR-based (RHS packing)
constexpr std::array<std::array<PackF32Fn, 3>, 2> pack_trans_f32_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {pack_trans_n6_f32, pack_trans_n4_f32, pack_trans_n1_f32},
    {pack_trans_n16_f32, pack_trans_n8_f32, pack_trans_n1_f32},
#elif defined(NNOPS_ARCH_AARCH64)
    {pack_trans_n8_f32, pack_trans_n4_f32, pack_trans_n1_f32},
    {pack_trans_n12_f32, pack_trans_n4_f32, pack_trans_n1_f32},
#endif
}};

constexpr std::array<std::array<PackF32Fn, 3>, 2> pack_copy_f32_fn = {{
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
constexpr std::array<std::array<PackF16Fn, 3>, 2> pack_trans_f16_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {pack_trans_n6_f16, pack_trans_n4_f16, pack_trans_n1_f16},
    {pack_trans_n16_f16, pack_trans_n8_f16, pack_trans_n1_f16},
#elif defined(NNOPS_ARCH_AARCH64)
    {pack_trans_n8_f16, pack_trans_n4_f16, pack_trans_n1_f16},
    {pack_trans_n16_f16, pack_trans_n8_f16, pack_trans_n1_f16},
#endif
}};

constexpr std::array<std::array<PackF16Fn, 3>, 2> pack_copy_f16_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {pack_copy_n6_f16, pack_copy_n4_f16, pack_copy_n1_f16},
    {pack_copy_n16_f16, pack_copy_n8_f16, pack_copy_n1_f16},
#elif defined(NNOPS_ARCH_AARCH64)
    {pack_copy_n8_f16, pack_copy_n4_f16, pack_copy_n1_f16},
    {pack_copy_n16_f16, pack_copy_n8_f16, pack_copy_n1_f16},
#endif
}};

// ---- MMA dispatch tables -----------------------------------------------

template <bool zero_mode>
constexpr std::array<std::array<MmaPackF32Fn, 3>, 3> mma_pack_f32_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {{mma_pack_6x16_f32<zero_mode>, mma_pack_6x8_f32<zero_mode>, mma_pack_6x1_f32<zero_mode>}},
    {{mma_pack_4x16_f32<zero_mode>, mma_pack_4x8_f32<zero_mode>, mma_pack_4x1_f32<zero_mode>}},
    {{mma_pack_1x16_f32<zero_mode>, mma_pack_1x8_f32<zero_mode>, mma_pack_1x1_f32<zero_mode>}},
#elif defined(NNOPS_ARCH_AARCH64)
    {{mma_pack_8x12_f32<zero_mode>, mma_pack_8x4_f32<zero_mode>, mma_pack_8x1_f32<zero_mode>}},
    {{mma_pack_4x12_f32<zero_mode>, mma_pack_4x4_f32<zero_mode>, mma_pack_4x1_f32<zero_mode>}},
    {{mma_pack_1x12_f32<zero_mode>, mma_pack_1x4_f32<zero_mode>, mma_pack_1x1_f32<zero_mode>}},
#endif
}};

template <bool zero_mode>
constexpr std::array<std::array<MmaPackF16Fn, 3>, 3> mma_pack_f16_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {{mma_pack_6x16_f16<zero_mode>, mma_pack_6x8_f16<zero_mode>, mma_pack_6x1_f16<zero_mode>}},
    {{mma_pack_4x16_f16<zero_mode>, mma_pack_4x8_f16<zero_mode>, mma_pack_4x1_f16<zero_mode>}},
    {{mma_pack_1x16_f16<zero_mode>, mma_pack_1x8_f16<zero_mode>, mma_pack_1x1_f16<zero_mode>}},
#elif defined(NNOPS_ARCH_AARCH64)
    {{mma_pack_8x16_f16<zero_mode>, mma_pack_8x8_f16<zero_mode>, mma_pack_8x1_f16<zero_mode>}},
    {{mma_pack_4x16_f16<zero_mode>, mma_pack_4x8_f16<zero_mode>, mma_pack_4x1_f16<zero_mode>}},
    {{mma_pack_1x16_f16<zero_mode>, mma_pack_1x8_f16<zero_mode>, mma_pack_1x1_f16<zero_mode>}},
#endif
}};

template <bool zero_mode>
constexpr std::array<std::array<MmaDirectF32Fn, 3>, 3> mma_direct_f32_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {{mma_direct_6x16_f32<zero_mode>, mma_direct_6x8_f32<zero_mode>, mma_direct_6x1_f32<zero_mode>}},
    {{mma_direct_4x16_f32<zero_mode>, mma_direct_4x8_f32<zero_mode>, mma_direct_4x1_f32<zero_mode>}},
    {{mma_direct_1x16_f32<zero_mode>, mma_direct_1x8_f32<zero_mode>, mma_direct_1x1_f32<zero_mode>}},
#elif defined(NNOPS_ARCH_AARCH64)
    // mr=6, not 8: the 8-row direct kernel keeps 35 vectors live and spills on
    // every k. Rows here must stay consistent with MR_F32_DIRECT.
    {{mma_direct_6x12_f32<zero_mode>, mma_direct_6x4_f32<zero_mode>, mma_direct_6x1_f32<zero_mode>}},
    {{mma_direct_4x12_f32<zero_mode>, mma_direct_4x4_f32<zero_mode>, mma_direct_4x1_f32<zero_mode>}},
    {{mma_direct_1x12_f32<zero_mode>, mma_direct_1x4_f32<zero_mode>, mma_direct_1x1_f32<zero_mode>}},
#endif
}};

template <bool zero_mode>
constexpr std::array<std::array<MmaDirectF16Fn, 3>, 3> mma_direct_f16_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {{mma_direct_6x16_f16<zero_mode>, mma_direct_6x8_f16<zero_mode>, mma_direct_6x1_f16<zero_mode>}},
    {{mma_direct_4x16_f16<zero_mode>, mma_direct_4x8_f16<zero_mode>, mma_direct_4x1_f16<zero_mode>}},
    {{mma_direct_1x16_f16<zero_mode>, mma_direct_1x8_f16<zero_mode>, mma_direct_1x1_f16<zero_mode>}},
#elif defined(NNOPS_ARCH_AARCH64)
    {{mma_direct_8x16_f16<zero_mode>, mma_direct_8x8_f16<zero_mode>, mma_direct_8x1_f16<zero_mode>}},
    {{mma_direct_4x16_f16<zero_mode>, mma_direct_4x8_f16<zero_mode>, mma_direct_4x1_f16<zero_mode>}},
    {{mma_direct_1x16_f16<zero_mode>, mma_direct_1x8_f16<zero_mode>, mma_direct_1x1_f16<zero_mode>}},
#endif
}};


// ---- tiled pack entry points -------------------------------------------
// Panels are packed densely (contiguous); panel p of size mr/nr occupies the
// next mr*kc / nr*kc elements. This matches the nn_compute reference and the
// low-level pack kernels (which write output contiguously).

void tile_pack_lhs(bool trans, int mc, int kc,
                   float* dst, int ldd, const float* src, int lds, float scale) {
    int pack_lds = trans ? 1 : lds;
    auto& pack_fns = trans ? pack_copy_f32_fn[0] : pack_trans_f32_fn[0];
    int m = 0;
    for(; m + MR_F32[0] <= mc; m += MR_F32[0]) {
        pack_fns[0](dst, src, lds, kc, scale);
        dst += ldd;
        src += MR_F32[0] * pack_lds;
    }
    for(; m + MR_F32[1] <= mc; m += MR_F32[1]) {
        pack_fns[1](dst, src, lds, kc, scale);
        dst += ldd;
        src += MR_F32[1] * pack_lds;
    }
    for(; m + MR_F32[2] <= mc; m += MR_F32[2]) {
        pack_fns[2](dst, src, lds, kc, scale);
        dst += ldd;
        src += MR_F32[2] * pack_lds;
    }
}

void tile_pack_lhs(bool trans, int mc, int kc,
                   half* dst, int ldd, const half* src, int lds, float scale) {
    int pack_lds = trans ? 1 : lds;
    auto& pack_fns = trans ? pack_copy_f16_fn[0] : pack_trans_f16_fn[0];
    int m = 0;
    for(; m + MR_F16[0] <= mc; m += MR_F16[0]) {
        pack_fns[0](dst, src, lds, kc, scale);
        dst += ldd;
        src += MR_F16[0] * pack_lds;
    }
    for(; m + MR_F16[1] <= mc; m += MR_F16[1]) {
        pack_fns[1](dst, src, lds, kc, scale);
        dst += ldd;
        src += MR_F16[1] * pack_lds;
    }
    for(; m + MR_F16[2] <= mc; m += MR_F16[2]) {
        pack_fns[2](dst, src, lds, kc, scale);
        dst += ldd;
        src += MR_F16[2] * pack_lds;
    }
}

void tile_pack_rhs(bool trans, int nc, int kc,
                   float* dst, int ldd, const float* src, int lds, float scale) {
    int pack_lds = trans ? lds : 1;
    auto& pack_fns = trans ? pack_trans_f32_fn[1] : pack_copy_f32_fn[1];
    int n = 0;
    for(; n + NR_F32[0] <= nc; n += NR_F32[0]) {
        pack_fns[0](dst, src, lds, kc, scale);
        dst += ldd;
        src += NR_F32[0] * pack_lds;
    }
    for(; n + NR_F32[1] <= nc; n += NR_F32[1]) {
        pack_fns[1](dst, src, lds, kc, scale);
        dst += ldd;
        src += NR_F32[1] * pack_lds;
    }
    for(; n + NR_F32[2] <= nc; n += NR_F32[2]) {
        pack_fns[2](dst, src, lds, kc, scale);
        dst += ldd;
        src += NR_F32[2] * pack_lds;
    }
}

void tile_pack_rhs(bool trans, int nc, int kc,
                   half* dst, int ldd, const half* src, int lds, float scale) {
    int pack_lds = trans ? lds : 1;
    auto& pack_fns = trans ? pack_trans_f16_fn[1] : pack_copy_f16_fn[1];
    int n = 0;
    for(; n + NR_F16[0] <= nc; n += NR_F16[0]) {
        pack_fns[0](dst, src, lds, kc, scale);
        dst += ldd;
        src += NR_F16[0] * pack_lds;
    }
    for(; n + NR_F16[1] <= nc; n += NR_F16[1]) {
        pack_fns[1](dst, src, lds, kc, scale);
        dst += ldd;
        src += NR_F16[1] * pack_lds;
    }
    for(; n + NR_F16[2] <= nc; n += NR_F16[2]) {
        pack_fns[2](dst, src, lds, kc, scale);
        dst += ldd;
        src += NR_F16[2] * pack_lds;
    }
}


// ---- mma entry points ---------------------------------------------------
//
// `maybe_packed_b`/`ldb` describe B for the pack path:
//   ldb < 0  → B is packed ([K][nr], row stride nr, panels advance nr*K)
//   ldb >= 0 → B is raw (row stride ldb, advance nr per panel)
// Packed A panels advance contiguously by mr*Kc.

// f32 — packed A, packed/raw B
void mrkcnc_mma_pack(int Nc, int Kc,
                     float* c, int ldc,
                     const float* packed_a,
                     const float* maybe_packed_b, int ldb,
                     float clamp_min, float clamp_max,
                     const std::array<MmaPackF32Fn, 3>& mma_pack_f32_fn) {
    const bool packed = (ldb < 0);
    const int ldd_b = align_up<PANEL_ALIGN_BYTES>(NR_F32[0] * Kc * static_cast<int>(sizeof(float))) / static_cast<int>(sizeof(float));
    int n = 0;
    for(; n + NR_F32[0] <= Nc; n += NR_F32[0]) {
        mma_pack_f32_fn[0](c + n, ldc, packed_a, maybe_packed_b,
                           packed ? NR_F32[0] : ldb, Kc, clamp_min, clamp_max);
        maybe_packed_b += packed ? ldd_b : NR_F32[0];
    }
    for(; n + NR_F32[1] <= Nc; n += NR_F32[1]) {
        mma_pack_f32_fn[1](c + n, ldc, packed_a, maybe_packed_b,
                           packed ? NR_F32[1] : ldb, Kc, clamp_min, clamp_max);
        maybe_packed_b += packed ? ldd_b : NR_F32[1];
    }
    for(; n + NR_F32[2] <= Nc; n += NR_F32[2]) {
        mma_pack_f32_fn[2](c + n, ldc, packed_a, maybe_packed_b,
                           packed ? NR_F32[2] : ldb, Kc, clamp_min, clamp_max);
        maybe_packed_b += packed ? ldd_b : NR_F32[2];
    }
}

// f32 — raw A, packed/raw B (ldb < 0 signals packed, as in the pack path)
void mrkcnc_mma_direct(int Nc, int Kc,
                       float* c, int ldc,
                       const float* a, int lda,
                       const float* b, int ldb,
                       float clamp_min, float clamp_max,
                       const std::array<MmaDirectF32Fn, 3>& mma_direct_f32_fn) {
    const bool packed = (ldb < 0);
    const int ldd_b = align_up<PANEL_ALIGN_BYTES>(NR_F32[0] * Kc * static_cast<int>(sizeof(float))) / static_cast<int>(sizeof(float));
    int n = 0;
    for(; n + NR_F32[0] <= Nc; n += NR_F32[0]) {
        mma_direct_f32_fn[0](c + n, ldc, a, lda, b,
                             packed ? NR_F32[0] : ldb, Kc, clamp_min, clamp_max);
        b += packed ? ldd_b : NR_F32[0];
    }
    for(; n + NR_F32[1] <= Nc; n += NR_F32[1]) {
        mma_direct_f32_fn[1](c + n, ldc, a, lda, b,
                             packed ? NR_F32[1] : ldb, Kc, clamp_min, clamp_max);
        b += packed ? ldd_b : NR_F32[1];
    }
    for(; n + NR_F32[2] <= Nc; n += NR_F32[2]) {
        mma_direct_f32_fn[2](c + n, ldc, a, lda, b,
                             packed ? NR_F32[2] : ldb, Kc, clamp_min, clamp_max);
        b += packed ? ldd_b : NR_F32[2];
    }
}

// f16 — packed A, packed/raw B
void mrkcnc_mma_pack(int Nc, int Kc,
                     half* c, int ldc,
                     const half* packed_a, const half* maybe_packed_b, int ldb,
                     float clamp_min, float clamp_max,
                     const std::array<MmaPackF16Fn, 3>& mma_pack_f16_fn) {
    const bool packed = (ldb < 0);
    const int ldd_b = align_up<PANEL_ALIGN_BYTES>(NR_F16[0] * Kc * static_cast<int>(sizeof(half))) / static_cast<int>(sizeof(half));
    int n = 0;
    for(; n + NR_F16[0] <= Nc; n += NR_F16[0]) {
        mma_pack_f16_fn[0](c + n, ldc, packed_a, maybe_packed_b,
                           packed ? NR_F16[0] : ldb, Kc, clamp_min, clamp_max);
        maybe_packed_b += packed ? ldd_b : NR_F16[0];
    }
    for(; n + NR_F16[1] <= Nc; n += NR_F16[1]) {
        mma_pack_f16_fn[1](c + n, ldc, packed_a, maybe_packed_b,
                           packed ? NR_F16[1] : ldb, Kc, clamp_min, clamp_max);
        maybe_packed_b += packed ? ldd_b : NR_F16[1];
    }
    for(; n + NR_F16[2] <= Nc; n += NR_F16[2]) {
        mma_pack_f16_fn[2](c + n, ldc, packed_a, maybe_packed_b,
                           packed ? NR_F16[2] : ldb, Kc, clamp_min, clamp_max);
        maybe_packed_b += packed ? ldd_b : NR_F16[2];
    }
}

// f16 — raw A, packed/raw B (ldb < 0 signals packed, as in the pack path)
void mrkcnc_mma_direct(int Nc, int Kc,
                       half* c, int ldc,
                       const half* a, int lda,
                       const half* b, int ldb,
                       float clamp_min, float clamp_max,
                       const std::array<MmaDirectF16Fn, 3>& mma_direct_f16_fn) {
    const bool packed = (ldb < 0);
    const int ldd_b = align_up<PANEL_ALIGN_BYTES>(NR_F16[0] * Kc * static_cast<int>(sizeof(half))) / static_cast<int>(sizeof(half));
    int n = 0;
    for(; n + NR_F16[0] <= Nc; n += NR_F16[0]) {
        mma_direct_f16_fn[0](c + n, ldc, a, lda, b,
                             packed ? NR_F16[0] : ldb, Kc, clamp_min, clamp_max);
        b += packed ? ldd_b : NR_F16[0];
    }
    for(; n + NR_F16[1] <= Nc; n += NR_F16[1]) {
        mma_direct_f16_fn[1](c + n, ldc, a, lda, b,
                             packed ? NR_F16[1] : ldb, Kc, clamp_min, clamp_max);
        b += packed ? ldd_b : NR_F16[1];
    }
    for(; n + NR_F16[2] <= Nc; n += NR_F16[2]) {
        mma_direct_f16_fn[2](c + n, ldc, a, lda, b,
                             packed ? NR_F16[2] : ldb, Kc, clamp_min, clamp_max);
        b += packed ? ldd_b : NR_F16[2];
    }
}

// f32 — packed: packed_a advances contiguously by mr*Kc
void tile_mma_pack(int Mc, int Nc, int Kc,
                   float* c, int ldc,
                   const float* packed_a, const float* maybe_packed_b, int ldb,
                   float clamp_min, float clamp_max, bool zero_mode) {
    const auto& fns = zero_mode ? mma_pack_f32_fn<true> : mma_pack_f32_fn<false>;
    const int ldd_a = align_up<PANEL_ALIGN_BYTES>(MR_F32[0] * Kc * static_cast<int>(sizeof(float))) / static_cast<int>(sizeof(float));
    int m = 0;
    for(; m + MR_F32[0] <= Mc; m += MR_F32[0]) {
        mrkcnc_mma_pack(Nc, Kc, c + m * ldc, ldc,
                        packed_a, maybe_packed_b, ldb,
                        clamp_min, clamp_max, fns[0]);
        packed_a += ldd_a;
    }
    for(; m + MR_F32[1] <= Mc; m += MR_F32[1]) {
        mrkcnc_mma_pack(Nc, Kc, c + m * ldc, ldc,
                        packed_a, maybe_packed_b, ldb,
                        clamp_min, clamp_max, fns[1]);
        packed_a += ldd_a;
    }
    for(; m + MR_F32[2] <= Mc; m += MR_F32[2]) {
        mrkcnc_mma_pack(Nc, Kc, c + m * ldc, ldc,
                        packed_a, maybe_packed_b, ldb,
                        clamp_min, clamp_max, fns[2]);
        packed_a += ldd_a;
    }
}

// f32 — direct: A raw via lda, B raw/packed via ldb
void tile_mma_direct(int Mc, int Nc, int Kc,
                     float* c, int ldc,
                     const float* a, int lda,
                     const float* b, int ldb,
                     float clamp_min, float clamp_max, bool zero_mode) {
    const auto& fns = zero_mode ? mma_direct_f32_fn<true> : mma_direct_f32_fn<false>;
    int m = 0;
    for(; m + MR_F32_DIRECT[0] <= Mc; m += MR_F32_DIRECT[0]) {
        mrkcnc_mma_direct(Nc, Kc, c + m * ldc, ldc,
                          a + m * lda, lda, b, ldb,
                          clamp_min, clamp_max, fns[0]);
    }
    for(; m + MR_F32_DIRECT[1] <= Mc; m += MR_F32_DIRECT[1]) {
        mrkcnc_mma_direct(Nc, Kc, c + m * ldc, ldc,
                          a + m * lda, lda, b, ldb,
                          clamp_min, clamp_max, fns[1]);
    }
    for(; m + MR_F32_DIRECT[2] <= Mc; m += MR_F32_DIRECT[2]) {
        mrkcnc_mma_direct(Nc, Kc, c + m * ldc, ldc,
                          a + m * lda, lda, b, ldb,
                          clamp_min, clamp_max, fns[2]);
    }
}

// f16 — packed: packed_a advances contiguously by mr*Kc
void tile_mma_pack(int Mc, int Nc, int Kc,
                   half* c, int ldc,
                   const half* packed_a, const half* maybe_packed_b, int ldb,
                   float clamp_min, float clamp_max, bool zero_mode) {
    const auto& fns = zero_mode ? mma_pack_f16_fn<true> : mma_pack_f16_fn<false>;
    const int ldd_a = align_up<PANEL_ALIGN_BYTES>(MR_F16[0] * Kc * static_cast<int>(sizeof(half))) / static_cast<int>(sizeof(half));
    int m = 0;
    for(; m + MR_F16[0] <= Mc; m += MR_F16[0]) {
        mrkcnc_mma_pack(Nc, Kc, c + m * ldc, ldc,
                        packed_a, maybe_packed_b, ldb,
                        clamp_min, clamp_max, fns[0]);
        packed_a += ldd_a;
    }
    for(; m + MR_F16[1] <= Mc; m += MR_F16[1]) {
        mrkcnc_mma_pack(Nc, Kc, c + m * ldc, ldc,
                        packed_a, maybe_packed_b, ldb,
                        clamp_min, clamp_max, fns[1]);
        packed_a += ldd_a;
    }
    for(; m + MR_F16[2] <= Mc; m += MR_F16[2]) {
        mrkcnc_mma_pack(Nc, Kc, c + m * ldc, ldc,
                        packed_a, maybe_packed_b, ldb,
                        clamp_min, clamp_max, fns[2]);
        packed_a += ldd_a;
    }
}

// f16 — direct: A raw via lda, B raw/packed via ldb
void tile_mma_direct(int Mc, int Nc, int Kc,
                     half* c, int ldc,
                     const half* a, int lda,
                     const half* b, int ldb,
                     float clamp_min, float clamp_max, bool zero_mode) {
    const auto& fns = zero_mode ? mma_direct_f16_fn<true> : mma_direct_f16_fn<false>;
    const int ldd_a = align_up<PANEL_ALIGN_BYTES>(MR_F16[0] * Kc * static_cast<int>(sizeof(half))) / static_cast<int>(sizeof(half));
    int m = 0;
    for(; m + MR_F16[0] <= Mc; m += MR_F16[0]) {
        mrkcnc_mma_direct(Nc, Kc, c + m * ldc, ldc,
                          a + m * lda, lda, b, ldb,
                          clamp_min, clamp_max, fns[0]);
    }
    for(; m + MR_F16[1] <= Mc; m += MR_F16[1]) {
        mrkcnc_mma_direct(Nc, Kc, c + m * ldc, ldc,
                          a + m * lda, lda, b, ldb,
                          clamp_min, clamp_max, fns[1]);
    }
    for(; m + MR_F16[2] <= Mc; m += MR_F16[2]) {
        mrkcnc_mma_direct(Nc, Kc, c + m * ldc, ldc,
                          a + m * lda, lda, b, ldb,
                          clamp_min, clamp_max, fns[2]);
    }
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

void tile_pack_lhs_i8(bool trans, int mc, int kc, int8_t* dst, int ldd, const int8_t* src, int lds) {
    const int pack_lds = trans ? 1 : lds;
    const auto& pack_fns = trans ? pack_copy_i8_lhs : pack_trans_i8_lhs;
    const int kbytes = (kc + 3) & ~3;  // pack writes ceil(kc/4)*4 bytes per row
    int m = 0;
    for (int mi = 0; mi < 3; ++mi) {
        const int mr = MR_I8[mi];
        for (; m + mr <= mc; m += mr) {
            pack_fns[static_cast<size_t>(mi)](dst, src, lds, kc, 0.0f);
            if constexpr (INT8_USE_U8_OFFSET) {
                xor0x80_i8(dst, mr * kbytes);
            }
            dst += ldd;
            src += mr * pack_lds;
        }
    }
}

void tile_pack_rhs_i8(bool trans, int nc, int kc, int8_t* dst, int ldd, const int8_t* src, int lds) {
    const int pack_lds = trans ? lds : 1;
    const auto& pack_fns = trans ? pack_trans_i8_rhs : pack_copy_i8_rhs;
    int n = 0;
    for (int ni = 0; ni < 4; ++ni) {
        const int nr = NR_I8[ni];
        for (; n + nr <= nc; n += nr) {
            pack_fns[static_cast<size_t>(ni)](dst, src, lds, kc, 0.0f);
            dst += ldd;
            src += nr * pack_lds;
        }
    }
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
    const int ldd_a = align_up<PANEL_ALIGN_BYTES>(MR_MAX_I8 * kbytes);
    const int ldd_b = align_up<PANEL_ALIGN_BYTES>(NR_MAX_I8 * kbytes);
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
// SIMD-accelerated in-place scale: C[i] *= scale
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
                          int num_threads) {
    MatMulPlan plan;

    const auto dt_a = a_desc.dtype;
    const auto dt_b = b_desc.dtype;
    if (dt_a != b_desc.dtype) {
        return plan;
    }

    const bool is_f32 = (dt_a == DataType::f32);
    const bool is_f16 = (dt_a == DataType::f16);
    if (!is_f32 && !is_f16) {
        return plan;  // int8 / other dtypes are dispatched elsewhere
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
    int64_t lda = a_desc.row_stride_elems;
    if (lda == 0) { lda = attrs.transpose_a ? M : K; }
    plan.pack_a = attrs.transpose_a || (lda > PACK_A_STRIDE_THRESHOLD);

    // Split on the larger dimension (the dispatch's N > M rule).
    plan.split_n = (N > M);
    const bool split_m = !plan.split_n;

    plan.kc = std::min<int64_t>(is_f32 ? KC_F32 : KC_F16, K);

    // The M panel height the route will actually run — the nc heuristic below
    // charges the L2 working set for it, so a pack_a=0 GEMM must be sized with
    // the direct height, not the packed one.
    const int  mr_panel = is_f32 ? mr_max_flt<float>(plan.pack_a)
                                 : mr_max_flt<half>(plan.pack_a);
    const int  nr_max = is_f32 ? nr_max_flt<float>() : nr_max_flt<half>();
    const int* nr     = is_f32 ? NR_F32 : NR_F16;
    const int  elem   = is_f32 ? static_cast<int>(sizeof(float)) : static_cast<int>(sizeof(half));

    plan.mc = (split_m && num_threads > 1) ? (M / num_threads) : MC_TARGET;
    plan.mc = std::max<int64_t>(std::min<int64_t>(plan.mc, MC_TARGET), 1);

    // N tile bounded by the L2 cache, rounded down to a nr_max multiple so the
    // per-block packed-B slices tile the workspace exactly, clamped to [nr_max, N].
    const size_t l2 = simd::CpuFeatures::get().l2_cache_size();
    const int nc_l2 = compute_nc(mr_panel, static_cast<int>(plan.kc), elem, l2);
    const int nc_cap = clamp_nc(round_down_nc(nc_l2, nr_max), nr_max, static_cast<int>(N));

    int64_t nc = nc_cap;
    if (plan.split_n && num_threads > 1) {
        nc = std::min<int64_t>(N / num_threads, nc_cap);
    }
    plan.nc = clamp_nc(round_down_nc(static_cast<int>(nc), nr_max), nr_max, static_cast<int>(N));

    // Packed-B stride (elements) at full Kc.
    const int ldd_b = align_up<PANEL_ALIGN_BYTES>(nr_max * static_cast<int>(plan.kc) * elem) / elem;
    plan.ldd_b = ldd_b;

    // One full-N packed-B slice (bytes). N-split tiles this once across the
    // n-blocks (thread-count invariant); M-split (MKN) gives every m-block its
    // own slice, so the workspace is num_m_blocks × this.
    const size_t packed_b_bytes = static_cast<size_t>(num_panels(static_cast<int>(N), nr))
                                * static_cast<size_t>(ldd_b) * static_cast<size_t>(elem);
    if (split_m) {
        const int64_t num_m_blocks = split_block_count(M, plan.mc);
        plan.workspace_size = num_m_blocks * static_cast<int64_t>(packed_b_bytes);
    } else {
        plan.workspace_size = static_cast<int64_t>(packed_b_bytes);
    }

    return plan;
}


}  // namespace nnops::backend::cpu
