#include "matmul_helper.h"

#include <array>

#ifdef NNOPS_ARCH_X86_64
using namespace nnops::backend::cpu::x86_64;
#elif defined(NNOPS_ARCH_AARCH64)
using namespace nnops::backend::cpu::aarch64;
#endif

namespace nnops::backend::cpu {

// =========================================================================
//  Workspace sizing — same tiling as the matmul.cpp kernels
// =========================================================================

size_t matmul_get_workspace_size(const MatMulAttributes& attrs,
                                 const TensorDesc& a_desc,
                                 const TensorDesc& b_desc,
                                 const TensorDesc& /*c_desc*/)
{
    const auto dt_a = a_desc.dtype;
    if (dt_a != b_desc.dtype) {
        return 0;
    }

    const int64_t a_rank = a_desc.rank;
    const int64_t b_rank = b_desc.rank;
    NNOPS_ASSERT(a_rank >= 2 && b_rank >= 2);

    const int64_t M = attrs.transpose_a ? a_desc.dims[static_cast<size_t>(a_rank - 1)]
                                        : a_desc.dims[static_cast<size_t>(a_rank - 2)];
    const int64_t N = attrs.transpose_b ? b_desc.dims[static_cast<size_t>(b_rank - 2)]
                                        : b_desc.dims[static_cast<size_t>(b_rank - 1)];

    // Physical row strides (elements): use the desc pitch when present, else the
    // compact last-dim size. This keeps sizing consistent with the kernel's
    // runtime lda/ldb even for padded, non-transposed matrices.
    const int64_t lda = (a_desc.row_stride_elems > 0)
        ? a_desc.row_stride_elems : a_desc.dims[static_cast<size_t>(a_rank - 1)];
    const int64_t ldb = (b_desc.row_stride_elems > 0)
        ? b_desc.row_stride_elems : b_desc.dims[static_cast<size_t>(b_rank - 1)];

    int kc, nr_max;
    size_t elem;
    const int* nr;
    if (dt_a == DataType::f32) {
        kc = KC_F32;  nr_max = NR_MAX_F32;  elem = sizeof(float);  nr = NR_F32;
    } else if (dt_a == DataType::f16) {
        kc = KC_F16;  nr_max = NR_MAX_F16;  elem = sizeof(half);   nr = NR_F16;
    } else {
        return 0;  // unsupported dtype → reference fallback, no workspace
    }

    // Global plan (full M×N) mirrors the kernel's dispatch: a workspace is
    // needed iff pack_b. The size is thread-count invariant — per-block panel
    // slices tile the buffer exactly, so only the global shape matters here.
    const bool pack_b = (dt_a == DataType::f32)
        ? make_pack_plan<float>(attrs, M, N, lda, ldb).pack_b
        : make_pack_plan<half>(attrs, M, N, lda, ldb).pack_b;
    if (!pack_b) {
        return 0;
    }

    // Packed B panels sit at the workspace base at a uniform 64-byte-aligned
    // full-Kc stride. Each n-block owns num_panels(nc, nr) panels at this
    // stride and the blocks tile the buffer exactly, so the total is
    // num_panels(N, nr) panels — independent of nc and of the thread count.
    const int ldd_b = align_up<PANEL_ALIGN_BYTES>(nr_max * kc * static_cast<int>(elem)) / static_cast<int>(elem);

    return static_cast<size_t>(num_panels(static_cast<int>(N), nr)) * static_cast<size_t>(ldd_b) * elem;
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

constexpr std::array<std::array<MmaPackF32Fn, 3>, 3> mma_pack_f32_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {{mma_pack_6x16_f32, mma_pack_6x8_f32, mma_pack_6x1_f32}},
    {{mma_pack_4x16_f32, mma_pack_4x8_f32, mma_pack_4x1_f32}},
    {{mma_pack_1x16_f32, mma_pack_1x8_f32, mma_pack_1x1_f32}},
#elif defined(NNOPS_ARCH_AARCH64)
    {{mma_pack_8x12_f32, mma_pack_8x4_f32, mma_pack_8x1_f32}},
    {{mma_pack_4x12_f32, mma_pack_4x4_f32, mma_pack_4x1_f32}},
    {{mma_pack_1x12_f32, mma_pack_1x4_f32, mma_pack_1x1_f32}},
#endif
}};

constexpr std::array<std::array<MmaPackF16Fn, 3>, 3> mma_pack_f16_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {{mma_pack_6x16_f16, mma_pack_6x8_f16, mma_pack_6x1_f16}},
    {{mma_pack_4x16_f16, mma_pack_4x8_f16, mma_pack_4x1_f16}},
    {{mma_pack_1x16_f16, mma_pack_1x8_f16, mma_pack_1x1_f16}},
#elif defined(NNOPS_ARCH_AARCH64)
    {{mma_pack_8x16_f16, mma_pack_8x8_f16, mma_pack_8x1_f16}},
    {{mma_pack_4x16_f16, mma_pack_4x8_f16, mma_pack_4x1_f16}},
    {{mma_pack_1x16_f16, mma_pack_1x8_f16, mma_pack_1x1_f16}},
#endif
}};

constexpr std::array<std::array<MmaDirectF32Fn, 3>, 3> mma_direct_f32_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {{mma_direct_6x16_f32, mma_direct_6x8_f32, mma_direct_6x1_f32}},
    {{mma_direct_4x16_f32, mma_direct_4x8_f32, mma_direct_4x1_f32}},
    {{mma_direct_1x16_f32, mma_direct_1x8_f32, mma_direct_1x1_f32}},
#elif defined(NNOPS_ARCH_AARCH64)
    {{mma_direct_8x12_f32, mma_direct_8x4_f32, mma_direct_8x1_f32}},
    {{mma_direct_4x12_f32, mma_direct_4x4_f32, mma_direct_4x1_f32}},
    {{mma_direct_1x12_f32, mma_direct_1x4_f32, mma_direct_1x1_f32}},
#endif
}};

constexpr std::array<std::array<MmaDirectF16Fn, 3>, 3> mma_direct_f16_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {{mma_direct_6x16_f16, mma_direct_6x8_f16, mma_direct_6x1_f16}},
    {{mma_direct_4x16_f16, mma_direct_4x8_f16, mma_direct_4x1_f16}},
    {{mma_direct_1x16_f16, mma_direct_1x8_f16, mma_direct_1x1_f16}},
#elif defined(NNOPS_ARCH_AARCH64)
    {{mma_direct_8x16_f16, mma_direct_8x8_f16, mma_direct_8x1_f16}},
    {{mma_direct_4x16_f16, mma_direct_4x8_f16, mma_direct_4x1_f16}},
    {{mma_direct_1x16_f16, mma_direct_1x8_f16, mma_direct_1x1_f16}},
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
                   float clamp_min, float clamp_max) {
    const int ldd_a = align_up<PANEL_ALIGN_BYTES>(MR_F32[0] * Kc * static_cast<int>(sizeof(float))) / static_cast<int>(sizeof(float));
    int m = 0;
    for(; m + MR_F32[0] <= Mc; m += MR_F32[0]) {
        mrkcnc_mma_pack(Nc, Kc, c + m * ldc, ldc,
                        packed_a, maybe_packed_b, ldb,
                        clamp_min, clamp_max, mma_pack_f32_fn[0]);
        packed_a += ldd_a;
    }
    for(; m + MR_F32[1] <= Mc; m += MR_F32[1]) {
        mrkcnc_mma_pack(Nc, Kc, c + m * ldc, ldc,
                        packed_a, maybe_packed_b, ldb,
                        clamp_min, clamp_max, mma_pack_f32_fn[1]);
        packed_a += ldd_a;
    }
    for(; m + MR_F32[2] <= Mc; m += MR_F32[2]) {
        mrkcnc_mma_pack(Nc, Kc, c + m * ldc, ldc,
                        packed_a, maybe_packed_b, ldb,
                        clamp_min, clamp_max, mma_pack_f32_fn[2]);
        packed_a += ldd_a;
    }
}

// f32 — direct: A raw via lda, B raw/packed via ldb
void tile_mma_direct(int Mc, int Nc, int Kc,
                     float* c, int ldc,
                     const float* a, int lda,
                     const float* b, int ldb,
                     float clamp_min, float clamp_max) {
    const int ldd_a = align_up<PANEL_ALIGN_BYTES>(MR_F32[0] * Kc * static_cast<int>(sizeof(float))) / static_cast<int>(sizeof(float));
    int m = 0;
    for(; m + MR_F32[0] <= Mc; m += MR_F32[0]) {
        mrkcnc_mma_direct(Nc, Kc, c + m * ldc, ldc,
                          a + m * lda, lda, b, ldb,
                          clamp_min, clamp_max, mma_direct_f32_fn[0]);
    }
    for(; m + MR_F32[1] <= Mc; m += MR_F32[1]) {
        mrkcnc_mma_direct(Nc, Kc, c + m * ldc, ldc,
                          a + m * lda, lda, b, ldb,
                          clamp_min, clamp_max, mma_direct_f32_fn[1]);
    }
    for(; m + MR_F32[2] <= Mc; m += MR_F32[2]) {
        mrkcnc_mma_direct(Nc, Kc, c + m * ldc, ldc,
                          a + m * lda, lda, b, ldb,
                          clamp_min, clamp_max, mma_direct_f32_fn[2]);
    }
}

// f16 — packed: packed_a advances contiguously by mr*Kc
void tile_mma_pack(int Mc, int Nc, int Kc,
                   half* c, int ldc,
                   const half* packed_a, const half* maybe_packed_b, int ldb,
                   float clamp_min, float clamp_max) {
    const int ldd_a = align_up<PANEL_ALIGN_BYTES>(MR_F16[0] * Kc * static_cast<int>(sizeof(half))) / static_cast<int>(sizeof(half));
    int m = 0;
    for(; m + MR_F16[0] <= Mc; m += MR_F16[0]) {
        mrkcnc_mma_pack(Nc, Kc, c + m * ldc, ldc,
                        packed_a, maybe_packed_b, ldb,
                        clamp_min, clamp_max, mma_pack_f16_fn[0]);
        packed_a += ldd_a;
    }
    for(; m + MR_F16[1] <= Mc; m += MR_F16[1]) {
        mrkcnc_mma_pack(Nc, Kc, c + m * ldc, ldc,
                        packed_a, maybe_packed_b, ldb,
                        clamp_min, clamp_max, mma_pack_f16_fn[1]);
        packed_a += ldd_a;
    }
    for(; m + MR_F16[2] <= Mc; m += MR_F16[2]) {
        mrkcnc_mma_pack(Nc, Kc, c + m * ldc, ldc,
                        packed_a, maybe_packed_b, ldb,
                        clamp_min, clamp_max, mma_pack_f16_fn[2]);
        packed_a += ldd_a;
    }
}

// f16 — direct: A raw via lda, B raw/packed via ldb
void tile_mma_direct(int Mc, int Nc, int Kc,
                     half* c, int ldc,
                     const half* a, int lda,
                     const half* b, int ldb,
                     float clamp_min, float clamp_max) {
    const int ldd_a = align_up<PANEL_ALIGN_BYTES>(MR_F16[0] * Kc * static_cast<int>(sizeof(half))) / static_cast<int>(sizeof(half));
    int m = 0;
    for(; m + MR_F16[0] <= Mc; m += MR_F16[0]) {
        mrkcnc_mma_direct(Nc, Kc, c + m * ldc, ldc,
                          a + m * lda, lda, b, ldb,
                          clamp_min, clamp_max, mma_direct_f16_fn[0]);
    }
    for(; m + MR_F16[1] <= Mc; m += MR_F16[1]) {
        mrkcnc_mma_direct(Nc, Kc, c + m * ldc, ldc,
                          a + m * lda, lda, b, ldb,
                          clamp_min, clamp_max, mma_direct_f16_fn[1]);
    }
    for(; m + MR_F16[2] <= Mc; m += MR_F16[2]) {
        mrkcnc_mma_direct(Nc, Kc, c + m * ldc, ldc,
                          a + m * lda, lda, b, ldb,
                          clamp_min, clamp_max, mma_direct_f16_fn[2]);
    }
}


#include "nnops/detail/half.hpp"
#include "nnops/detail/simd.hpp"

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



}  // namespace nnops::backend::cpu
