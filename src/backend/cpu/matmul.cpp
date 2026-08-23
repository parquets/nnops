/// @file matmul.cpp
/// @brief Tiled matrix multiplication kernel — NKM loop with matmul_helper pack + MMA.
///
/// Determines tile sizes Mc, Nc from the L2 cache constraint:
///   (mr × Kc + Nc × Kc + mr × Nc) × 2 × elem_size < L2_SIZE
/// where mr is the micro-kernel M panel (not Mc — only the active mr panel
/// is in the L2 working set during MMA).

#include "matmul.h"
#include "matmul_helper.h"
#include "simd_kernel/simd_epilogue.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/simd/cpu_features.hpp"
#include "nnops/detail/assert.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

// Forward-declare reference kernel for batched fallback.
namespace nnops::backend::cpu::reference {
extern void matmul_ref(const MatMulAttributes& attrs,
                       TensorView& output,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace);
}

// =========================================================================
//  Panel-size compile-time constants from the active arch namespace.
// =========================================================================
// These come from matmul_helper.h — arch::mr_f32 / arch::nr_f32 etc. are
// defined in the arch-specific pack_f32.hpp / pack_f16.hpp headers and
// exposed through the `arch` namespace alias.

// arch::mr_f32 = {8,4,1} (aarch64) or {6,4,1} (x86_64)
// arch::nr_f32 = {12,4,1} (aarch64) or {16,8,1} (x86_64)
// arch::mr_f16 = {8,4,1} (aarch64) or {6,4,1} (x86_64)
// arch::nr_f16 = {16,8,1} (aarch64) or {16,8,1} (x86_64)

namespace nnops::backend::cpu {
namespace {

// =========================================================================
//  Tiling constants
// =========================================================================

constexpr int KC_F32   = 128;
constexpr int KC_F16   = 256;
constexpr int KC_I8    = 512;   // int8: larger Kc since elements are 1 byte
constexpr int KC_F16I4 = 256;   // fp16×int4: placeholder (future hardware)

constexpr int MC_TARGET = 192;  // 192/6=32 (x86), 192/8=24 (aarch64)

// Placeholder max panel sizes for integer kernels (no SIMD kernels yet).
// MR_MAX_F32 / NR_MAX_F32 / MR_MAX_F16 / NR_MAX_F16 are in matmul_helper.h.
constexpr int MR_MAX_I8    = 4;
constexpr int NR_MAX_I8    = 4;
constexpr int MR_MAX_F16I4 = 4;
constexpr int NR_MAX_F16I4 = 4;

// 64-byte aligned panel strides (in elements): ldd = align_up(mr_max * kc * sizeof(T), 64) / sizeof(T)
constexpr int LDD_A_F32 = (MR_MAX_F32 * KC_F32 * 4 + 63) / 64 * 16;
constexpr int LDD_B_F32 = (NR_MAX_F32 * KC_F32 * 4 + 63) / 64 * 16;
constexpr int LDD_A_F16 = (MR_MAX_F16 * KC_F16 * 2 + 63) / 64 * 32;
constexpr int LDD_B_F16 = (NR_MAX_F16 * KC_F16 * 2 + 63) / 64 * 32;


// =========================================================================
//  Nc from L2 constraint
// =========================================================================
// From: (mr × Kc + Nc × Kc + mr × Nc) × 2 × elem_size < L2_SIZE
// → Nc < (L2_SIZE / (2 × elem_size) - mr × Kc) / (Kc + mr)

inline int compute_nc(int mr_max, int kc, int elem_bytes, size_t l2_size) noexcept {
    int denom = 2 * elem_bytes;
    int rhs   = static_cast<int>(l2_size / denom) - mr_max * kc;
    denom = kc + mr_max;
    return (rhs / denom) - 1;  // -1 for safety margin
}

inline int round_down_nc(int nc, int nr_max) noexcept {
    return (nc / nr_max) * nr_max;
}


// Workspace = packed A + packed B with 64-byte-aligned panel strides.
// ldd_a/b are in elements: align_up(mr_max * kc * sizeof(T), 64) / sizeof(T)
inline size_t workspace_bytes(int mc, int nc, int mr_max, int nr_max,
                              int ldd_a, int ldd_b, size_t elem) noexcept {
    int num_panels_a = (mc + mr_max - 1) / mr_max;
    int num_panels_b = (nc + nr_max - 1) / nr_max;
    return (static_cast<size_t>(num_panels_a) * static_cast<size_t>(ldd_a) +
            static_cast<size_t>(num_panels_b) * static_cast<size_t>(ldd_b)) * elem;
}


// =========================================================================
//  Kernel stubs — dispatched by (A_dtype, B_dtype) pair
// =========================================================================

// =========================================================================
//  Generic 2D GEMM kernel — NKM tiled loop with mma_direct.
//  Used for non-transposed cases.  Takes raw pointers for batch reuse.
// =========================================================================
template <typename T>
void matmul_kernel_2d_direct_flt(const MatMulAttributes& attrs,
                             T* c_ptr, int ldc,
                             const T* a_ptr, int lda,
                             const T* b_ptr, int ldb,
                             int M, int N, int K,
                             int kc, int clamp_min, int clamp_max)
{
    // ---- Clamp for epilogue ----
    float clamp_min = -std::numeric_limits<float>::infinity();
    float clamp_max =  std::numeric_limits<float>::infinity();
    if (attrs.epilogue.type == EpilogueActivateType::Relu) {
        clamp_min = 0.0f;
    }

    constexpr int mr_max = mr_max_flt<T>();
    constexpr int nr_max = nr_max_flt<T>();

    // ---- Compute tile sizes ----
    int mc = std::min(MC_TARGET, M);
    size_t l2_size = simd::CpuFeatures::get().l2_cache_size();
    int nc = round_down_nc(compute_nc(mr_max, kc, static_cast<int>(sizeof(T)), l2_size), nr_max);
    nc = std::min(nc, N);

    bool requires_epilogue = (attrs.epilogue.type != EpilogueActivateType::None &&
                         attrs.epilogue.type != EpilogueActivateType::Relu);

    // ---- NKM tiled loop (direct path) ----
    // Beta scaling fused into first k-block; epilogue fused after last k-block.
    for (int n = 0; n < N; n += nc) {
        int actual_nc = std::min(nc, N - n);
        for (int k = 0; k < K; k += kc) {
            int actual_kc = std::min(kc, K - k);

            const T* b_sub = b_ptr + k * ldb + n;  // B is K×N, row k, col n

            for (int m = 0; m < M; m += mc) {
                int actual_mc = std::min(mc, M - m);

                const T* a_sub = a_ptr + m * lda + k;  // A is M×K, row m, col k

                // Fuse beta scaling into first k-block (subsequent blocks accumulate)
                if (k == 0 && attrs.beta != 1.0f) {
                    tile_scale(c_ptr + m * ldc + n, ldc, attrs.beta, actual_mc, actual_nc);
                }

                tile_mma_direct(actual_mc, actual_nc, actual_kc,
                                c_ptr + m * ldc + n, ldc,
                                a_sub, lda, b_sub, ldb,
                                clamp_min, clamp_max);

                // Fuse epilogue after last k-block (C tile still in L1 cache)
                if (requires_epilogue && k + kc >= K) {
                    epilogue_inplace(actual_mc, actual_nc,
                                            c_ptr + m * ldc + n, ldc,
                                            static_cast<const T*>(nullptr), attrs.epilogue);
                }
            }
        }
    }
}

// =========================================================================
//  Generic 2D GEMM kernel — NKM tiled loop with pack + mma_pack.
//  Used for transposed cases (handles strided column access).
// =========================================================================
template <typename T>
void matmul_kernel_2d_packed_flt(const MatMulAttributes& attrs,
                             T* c_ptr, int ldc,
                             const T* a_ptr, int lda,
                             const T* b_ptr, int ldb,
                             int M, int N, int K, int kc, 
                             void* workspace)
{
    // ---- Clamp for epilogue ----
    float clamp_min = -std::numeric_limits<float>::infinity();
    float clamp_max =  std::numeric_limits<float>::infinity();
    if (attrs.epilogue.type == EpilogueActivateType::Relu) {
        clamp_min = 0.0f;
    }

    constexpr int mr_max = mr_max_flt<T>();
    constexpr int nr_max = nr_max_flt<T>();
    // ---- Compute tile sizes ----
    int mc = std::min(MC_TARGET, M);
    size_t l2_size = simd::CpuFeatures::get().l2_cache_size();
    int nc = round_down_nc(compute_nc(mr_max, kc, static_cast<int>(sizeof(T)), l2_size), nr_max);
    nc = std::min(nc, N);

    // ---- Workspace layout ----
    T* pack_a = static_cast<T*>(workspace);
    int num_panels_a = (mc + mr_max - 1) / mr_max;
    T* pack_b = pack_a + num_panels_a * ldd_a;

    // ---- NKM tiled loop (packed path) ----
    // Beta scaling fused into first k-block; epilogue fused after last k-block.
    for (int n = 0; n < N; n += nc) {
        int actual_nc = std::min(nc, N - n);
        for (int k = 0; k < K; k += kc) {
            int actual_kc = std::min(kc, K - k);

            int ldd_b = align_up<PANEL_ALIGN_BYTES>(nr_max * actual_kc * sizeof(T)) / sizeof(T);
            // Pack B panel
            const T* b_src = attrs.transpose_b
                ? b_ptr + n * ldb + k   // B phys is N×K, row n, col k
                : b_ptr + k * ldb + n;  // B phys is K×N, row k, col n
            tile_pack_rhs(attrs.transpose_b, actual_nc, actual_kc,
                          pack_b, ldd_b, b_src, ldb, 1.0f);

            for (int m = 0; m < M; m += mc) {
                int actual_mc = std::min(mc, M - m);

                // Pack A panel
                const T* a_src = attrs.transpose_a
                    ? a_ptr + k * lda + m   // A phys is K×M, row k, col m
                    : a_ptr + m * lda + k;  // A phys is M×K, row m, col k

                int ldd_a = align_up<PANEL_ALIGN_BYTES>(mr_max * actual_kc * sizeof(T)) / sizeof(T);
                tile_pack_lhs(attrs.transpose_a, actual_mc, actual_kc,
                              pack_a, ldd_a, a_src, lda, 1.0f);

                // Fuse beta scaling into first k-block (subsequent blocks accumulate)
                if (k == 0 && attrs.beta != 1.0f) {
                    tile_scale(c_ptr + m * ldc + n, ldc, attrs.beta, actual_mc, actual_nc);
                }

                // MMA
                tile_mma_pack(actual_mc, actual_nc, actual_kc,
                              c_ptr + m * ldc + n, ldc,
                              pack_a, pack_b, ldd_b,
                              clamp_min, clamp_max);

                // Fuse epilogue after last k-block (C tile still in L1 cache)
                if (requires_epilogue && k + kc >= K) {
                    epilogue_inplace(actual_mc, actual_nc,
                                            c_ptr + m * ldc + n, ldc,
                                            static_cast<const T*>(nullptr), attrs.epilogue);
                }
            }
        }
    }
}

void matmul_kernel_f32(const MatMulAttributes& attrs,
                       TensorView& output,
                       std::span<const TensorView> inputs,
                       void* workspace)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    const int64_t a_rank = a.rank();
    const int64_t b_rank = b.rank();
    const int64_t M  = attrs.transpose_a ? a.shape(a_rank - 1) : a.shape(a_rank - 2);
    const int64_t N  = attrs.transpose_b ? b.shape(b_rank - 2) : b.shape(b_rank - 1);
    const int64_t K  = attrs.transpose_b ? b.shape(b_rank - 1) : b.shape(b_rank - 2);

    auto* c_ptr = output.ptr<float>();
    const auto* a_ptr = a.ptr<float>();
    const auto* b_ptr = b.ptr<float>();
    int ldc = static_cast<int>(output.row_stride_elems());
    int lda = static_cast<int>(a.row_stride_elems());
    int ldb = static_cast<int>(b.row_stride_elems());
    int iM = static_cast<int>(M), iN = static_cast<int>(N), iK = static_cast<int>(K);

    if (attrs.transpose_a || attrs.transpose_b) {
        matmul_kernel_2d_packed<float>(attrs, c_ptr, ldc, a_ptr, lda, b_ptr, ldb,
                                       iM, iN, iK, KC_F32, MR_MAX_F32, NR_MAX_F32, workspace);
    } else {
        matmul_kernel_2d_direct<float>(attrs, c_ptr, ldc, a_ptr, lda, b_ptr, ldb,
                                       iM, iN, iK, KC_F32, MR_MAX_F32, NR_MAX_F32);
    }
}

void matmul_kernel_f16(const MatMulAttributes& attrs,
                       TensorView& output,
                       std::span<const TensorView> inputs,
                       void* workspace)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    const int64_t a_rank = a.rank();
    const int64_t b_rank = b.rank();
    const int64_t M  = attrs.transpose_a ? a.shape(a_rank - 1) : a.shape(a_rank - 2);
    const int64_t N  = attrs.transpose_b ? b.shape(b_rank - 2) : b.shape(b_rank - 1);
    const int64_t K  = attrs.transpose_b ? b.shape(b_rank - 1) : b.shape(b_rank - 2);

    auto* c_ptr = output.ptr<half>();
    const auto* a_ptr = a.ptr<half>();
    const auto* b_ptr = b.ptr<half>();
    int ldc = static_cast<int>(output.row_stride_elems());
    int lda = static_cast<int>(a.row_stride_elems());
    int ldb = static_cast<int>(b.row_stride_elems());
    int iM = static_cast<int>(M), iN = static_cast<int>(N), iK = static_cast<int>(K);

    if (attrs.transpose_a || attrs.transpose_b) {
        matmul_kernel_2d_packed<half>(attrs, c_ptr, ldc, a_ptr, lda, b_ptr, ldb,
                                      iM, iN, iK, KC_F16, MR_MAX_F16, NR_MAX_F16, workspace);
    } else {
        matmul_kernel_2d_direct<half>(attrs, c_ptr, ldc, a_ptr, lda, b_ptr, ldb,
                                      iM, iN, iK, KC_F16, MR_MAX_F16, NR_MAX_F16);
    }
}


// ---- int8 kernels ----------------------------------------------------------
// A (activation) may be u8 or s8; B (weight) is typically s8.
// Output is s32 accumulator → stored as output dtype after epilogue.

void matmul_kernel_u8i8(const MatMulAttributes& /*attrs*/,
                        TensorView& /*output*/,
                        std::span<const TensorView> /*inputs*/,
                        void* /*workspace*/)
{
    // TODO: u8 activation × s8 weight → s32 accumulator
}

void matmul_kernel_i8i8(const MatMulAttributes& /*attrs*/,
                        TensorView& /*output*/,
                        std::span<const TensorView> /*inputs*/,
                        void* /*workspace*/)
{
    // TODO: s8 activation × s8 weight → s32 accumulator
}


// ---- fp16×int4 kernel (future hardware) -----------------------------------

void matmul_kernel_f16i4(const MatMulAttributes& /*attrs*/,
                         TensorView& /*output*/,
                         std::span<const TensorView> /*inputs*/,
                         void* /*workspace*/)
{
    // TODO: fp16 activation × s4 weight (sub-byte packing required)
}

}  // anonymous namespace

// =========================================================================
//  Public API
// =========================================================================

size_t matmul_get_workspace_size(const MatMulAttributes& attrs,
                                 const TensorDesc& a_desc,
                                 const TensorDesc& b_desc,
                                 const TensorDesc& /*c_desc*/)
{
    // int64_t a_rank = a_desc.rank;
    // int64_t b_rank = b_desc.rank;
    // // M = rows of A in its logical (non-transposed) layout
    // // N = cols of B in its logical (non-transposed) layout
    // int64_t M = attrs.transpose_a ? a_desc.dims[static_cast<size_t>(a_rank - 1)]
    //                               : a_desc.dims[static_cast<size_t>(a_rank - 2)];
    // int64_t N = attrs.transpose_b ? b_desc.dims[static_cast<size_t>(b_rank - 2)]
    //                               : b_desc.dims[static_cast<size_t>(b_rank - 1)];

    // size_t l2_size = simd::CpuFeatures::get().l2_cache_size();

    // auto dtype_a = a_desc.dtype;
    // auto dtype_b = b_desc.dtype;

    // // ---- f16 ----------------------------------------------------------
    // if (dtype_a == DataType::f16 && dtype_b == DataType::f16) {
    //     int nc = round_down_nc(compute_nc(MR_MAX_F16, KC_F16, 2, l2_size), NR_MAX_F16);
    //     int mc = std::min(MC_TARGET, static_cast<int>(M));
    //     nc = std::min(nc, static_cast<int>(N));
    //     return workspace_bytes(mc, nc, KC_F16, 2);
    // }

    // // ---- f32 ----------------------------------------------------------
    // if (dtype_a == DataType::f32 && dtype_b == DataType::f32) {
    //     int nc = round_down_nc(compute_nc(MR_MAX_F32, KC_F32, 4, l2_size), NR_MAX_F32);
    //     int mc = std::min(MC_TARGET, static_cast<int>(M));
    //     nc = std::min(nc, static_cast<int>(N));
    //     return workspace_bytes(mc, nc, KC_F32, 4);
    // }

    // // ---- int8 variants (u8×s8, i8×s8) ---------------------------------
    // if ((dtype_a == DataType::u8 || dtype_a == DataType::s8) && dtype_b == DataType::s8) {
    //     // Placeholder: use MR_MAX_I8 / NR_MAX_I8 until SIMD kernels define real panels.
    //     int nc = round_down_nc(compute_nc(MR_MAX_I8, KC_I8, 1, l2_size), NR_MAX_I8);
    //     int mc = std::min(MC_TARGET, static_cast<int>(M));
    //     nc = std::min(nc, static_cast<int>(N));
    //     return workspace_bytes(mc, nc, KC_I8, 1);  // 1 byte per element
    // }

    // // ---- fp16×int4 (future) ------------------------------------------
    // if (dtype_a == DataType::f16 && dtype_b == DataType::s8) {
    //     // s4 weights are packed 2× per s8 byte — placeholder sizing.
    //     int nc = round_down_nc(compute_nc(MR_MAX_F16I4, KC_F16I4, 2, l2_size), NR_MAX_F16I4);
    //     int mc = std::min(MC_TARGET, static_cast<int>(M));
    //     nc = std::min(nc, static_cast<int>(N));
    //     return workspace_bytes(mc, nc, KC_F16I4, 2);  // fp16 = 2 bytes per element
    // }

    // // ---- unsupported combination — fallback won't pack, just return minimal ----
    return 0;
}

void matmul_kernel(const MatMulAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx,
                   void* workspace)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    NNOPS_ASSERT(a.rank() >= 2);
    NNOPS_ASSERT(b.rank() >= 2);
    NNOPS_ASSERT(inputs.size() >= 2 && inputs.size() <= 3);

    const auto dt_a = a.data_type();
    const auto dt_b = b.data_type();

    // ---- Resolve dtype-specific kernel parameters ----
    int kc, mr_max, nr_max;
    bool supported = true;

    if (dt_a == DataType::f32 && dt_b == DataType::f32) {
        kc = KC_F32; mr_max = MR_MAX_F32; nr_max = NR_MAX_F32;
    } else if (dt_a == DataType::f16 && dt_b == DataType::f16) {
        kc = KC_F16; mr_max = MR_MAX_F16; nr_max = NR_MAX_F16;
    } else {
        // Unsupported dtype — fall back to reference for all ranks
        supported = false;
    }

    if (!supported) {
        reference::matmul_ref(attrs, output, inputs, ctx, workspace);
        return;
    }

    // ---- Extract matrix dimensions ----
    const int64_t a_rank = a.rank();
    const int64_t b_rank = b.rank();
    const int64_t M  = attrs.transpose_a ? a.shape(a_rank - 1) : a.shape(a_rank - 2);
    const int64_t N  = attrs.transpose_b ? b.shape(b_rank - 2) : b.shape(b_rank - 1);
    const int64_t K  = attrs.transpose_b ? b.shape(b_rank - 1) : b.shape(b_rank - 2);

    const int64_t lda = a.row_stride_elems();
    const int64_t ldb = b.row_stride_elems();
    const int64_t ldc = output.row_stride_elems();

    // ---- 2D case: dispatch directly ----
    if (a_rank == 2 && b_rank == 2) {
        bool use_packed = attrs.transpose_a || attrs.transpose_b;
        if (dt_a == DataType::f32) {
            if (use_packed) {
                matmul_kernel_2d_packed<float>(attrs,
                    output.ptr<float>(), static_cast<int>(ldc),
                    a.ptr<float>(), static_cast<int>(lda),
                    b.ptr<float>(), static_cast<int>(ldb),
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                    kc, mr_max, nr_max, workspace);
            } else {
                matmul_kernel_2d_direct<float>(attrs,
                    output.ptr<float>(), static_cast<int>(ldc),
                    a.ptr<float>(), static_cast<int>(lda),
                    b.ptr<float>(), static_cast<int>(ldb),
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                    kc, mr_max, nr_max);
            }
        } else {
            if (use_packed) {
                matmul_kernel_2d_packed<half>(attrs,
                    output.ptr<half>(), static_cast<int>(ldc),
                    a.ptr<half>(), static_cast<int>(lda),
                    b.ptr<half>(), static_cast<int>(ldb),
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                    kc, mr_max, nr_max, workspace);
            } else {
                matmul_kernel_2d_direct<half>(attrs,
                    output.ptr<half>(), static_cast<int>(ldc),
                    a.ptr<half>(), static_cast<int>(lda),
                    b.ptr<half>(), static_cast<int>(ldb),
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                    kc, mr_max, nr_max);
            }
        }
        return;
    }

    // ---- Batched case (rank > 2): numpy-style broadcasting ----
    // Compute broadcast batch dimensions (same logic as matmul_ref)
    const int64_t batch_a_dims = a_rank - 2;
    const int64_t batch_b_dims = b_rank - 2;
    const int64_t batch_ndim   = std::max(batch_a_dims, batch_b_dims);

    std::vector<int64_t> batch_a_shape(batch_ndim, 1);
    std::vector<int64_t> batch_b_shape(batch_ndim, 1);
    std::vector<int64_t> batch_out_shape(batch_ndim, 1);

    for (int64_t i = 0; i < batch_a_dims; ++i) {
        batch_a_shape[batch_ndim - batch_a_dims + i] = a.shape(i);
    }
    for (int64_t i = 0; i < batch_b_dims; ++i) {
        batch_b_shape[batch_ndim - batch_b_dims + i] = b.shape(i);
    }

    int64_t total_batch = 1;
    for (int64_t i = 0; i < batch_ndim; ++i) {
        const int64_t da = batch_a_shape[i];
        const int64_t db = batch_b_shape[i];
        if (da == db) {
            batch_out_shape[i] = da;
        } else if (da == 1) {
            batch_out_shape[i] = db;
        } else if (db == 1) {
            batch_out_shape[i] = da;
        } else {
            NNOPS_ASSERT(!"MatMul: incompatible batch dimensions for broadcast");
        }
        total_batch *= batch_out_shape[i];
    }

    // Dispatch per-batch-element 2D GEMM
    bool use_packed = attrs.transpose_a || attrs.transpose_b;
    auto dispatch_2d = [&](auto* type_tag) {
        using T = std::decay_t<decltype(*type_tag)>;
        auto* c_base = output.ptr<T>();
        const auto* a_base = a.ptr<T>();
        const auto* b_base = b.ptr<T>();

        for (int64_t bi = 0; bi < total_batch; ++bi) {
            int64_t rem = bi;
            int64_t a_offset = 0;
            int64_t b_offset = 0;
            int64_t c_offset = 0;

            for (int64_t d = batch_ndim - 1; d >= 0; --d) {
                const int64_t coord = rem % batch_out_shape[d];
                rem /= batch_out_shape[d];

                const int64_t a_dim = d - (batch_ndim - batch_a_dims);
                if (a_dim >= 0) {
                    const int64_t a_coord = (batch_a_shape[d] == 1) ? 0 : coord;
                    a_offset += a_coord * a.stride_elems(a_dim);
                }

                const int64_t b_dim = d - (batch_ndim - batch_b_dims);
                if (b_dim >= 0) {
                    const int64_t b_coord = (batch_b_shape[d] == 1) ? 0 : coord;
                    b_offset += b_coord * b.stride_elems(b_dim);
                }

                c_offset += coord * output.stride_elems(d);
            }

            if (use_packed) {
                matmul_kernel_2d_packed<T>(attrs,
                    c_base + c_offset, static_cast<int>(ldc),
                    a_base + a_offset, static_cast<int>(lda),
                    b_base + b_offset, static_cast<int>(ldb),
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                    kc, mr_max, nr_max, workspace);
            } else {
                matmul_kernel_2d_direct<T>(attrs,
                    c_base + c_offset, static_cast<int>(ldc),
                    a_base + a_offset, static_cast<int>(lda),
                    b_base + b_offset, static_cast<int>(ldb),
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                    kc, mr_max, nr_max);
            }
        }
    };

    if (dt_a == DataType::f32) {
        dispatch_2d(static_cast<const float*>(nullptr));
    } else {
        dispatch_2d(static_cast<const half*>(nullptr));
    }
}

}  // namespace nnops::backend::cpu

