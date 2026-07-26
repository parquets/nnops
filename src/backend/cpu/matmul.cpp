/// @file matmul.cpp
/// @brief Tiled matrix multiplication kernel — NKM loop with imatmul pack + MMA.
///
/// Determines tile sizes Mc, Nc from the L2 cache constraint:
///   (mr × Kc + Nc × Kc + mr × Nc) × 2 × elem_size < L2_SIZE
/// where mr is the micro-kernel M panel (not Mc — only the active mr panel
/// is in the L2 working set during MMA).

#include "matmul.h"
#include "imatmul.h"
#include "nnops/detail/half.hpp"
#include "nnops/detail/simd/cpu_features.hpp"
#include "nnops/detail/assert.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

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
// These come from imatmul.h — arch::mr_f32 / arch::nr_f32 etc. are
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
// MR_MAX_F32 / NR_MAX_F32 / MR_MAX_F16 / NR_MAX_F16 are in imatmul.h.
constexpr int MR_MAX_I8    = 4;
constexpr int NR_MAX_I8    = 4;
constexpr int MR_MAX_F16I4 = 4;
constexpr int NR_MAX_F16I4 = 4;


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


inline size_t workspace_bytes(int mc, int nc, int kc, size_t elem) noexcept {
    return static_cast<size_t>(mc + nc) * static_cast<size_t>(kc) * elem;
}


// =========================================================================
//  Kernel stubs — dispatched by (A_dtype, B_dtype) pair
// =========================================================================

void matmul_kernel_f32(const MatMulAttributes& /*attrs*/,
                       TensorView& /*output*/,
                       std::span<const TensorView> /*inputs*/,
                       void* /*workspace*/)
{
    // TODO: NKM tiled loop over Mc×Nc×Kc with imatmul pack + mma_pack
}

void matmul_kernel_f16(const MatMulAttributes& /*attrs*/,
                       TensorView& /*output*/,
                       std::span<const TensorView> /*inputs*/,
                       void* /*workspace*/)
{
    // TODO: NKM tiled loop with imatmul f16 pack + mma_pack
}


// ---- int8 kernels ----------------------------------------------------------
// A (activation) may be u8 or i8; B (weight) is typically i8.
// Output is i32 accumulator → stored as output dtype after epilogue.

void matmul_kernel_u8i8(const MatMulAttributes& /*attrs*/,
                        TensorView& /*output*/,
                        std::span<const TensorView> /*inputs*/,
                        void* /*workspace*/)
{
    // TODO: u8 activation × i8 weight → i32 accumulator
}

void matmul_kernel_i8i8(const MatMulAttributes& /*attrs*/,
                        TensorView& /*output*/,
                        std::span<const TensorView> /*inputs*/,
                        void* /*workspace*/)
{
    // TODO: i8 activation × i8 weight → i32 accumulator
}


// ---- fp16×int4 kernel (future hardware) -----------------------------------

void matmul_kernel_f16i4(const MatMulAttributes& /*attrs*/,
                         TensorView& /*output*/,
                         std::span<const TensorView> /*inputs*/,
                         void* /*workspace*/)
{
    // TODO: fp16 activation × i4 weight (sub-byte packing required)
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
    int64_t a_rank = a_desc.rank;
    int64_t b_rank = b_desc.rank;
    // M = rows of A in its logical (non-transposed) layout
    // N = cols of B in its logical (non-transposed) layout
    int64_t M = attrs.transpose_a ? a_desc.dims[static_cast<size_t>(a_rank - 1)]
                                  : a_desc.dims[static_cast<size_t>(a_rank - 2)];
    int64_t N = attrs.transpose_b ? b_desc.dims[static_cast<size_t>(b_rank - 2)]
                                  : b_desc.dims[static_cast<size_t>(b_rank - 1)];

    size_t l2_size = simd::CpuFeatures::get().l2_cache_size();

    auto dtype_a = a_desc.dtype;
    auto dtype_b = b_desc.dtype;

    // ---- f16 ----------------------------------------------------------
    if (dtype_a == DataType::f16 && dtype_b == DataType::f16) {
        int nc = round_down_nc(compute_nc(MR_MAX_F16, KC_F16, 2, l2_size), NR_MAX_F16);
        int mc = std::min(MC_TARGET, static_cast<int>(M));
        nc = std::min(nc, static_cast<int>(N));
        return workspace_bytes(mc, nc, KC_F16, 2);
    }

    // ---- f32 ----------------------------------------------------------
    if (dtype_a == DataType::f32 && dtype_b == DataType::f32) {
        int nc = round_down_nc(compute_nc(MR_MAX_F32, KC_F32, 4, l2_size), NR_MAX_F32);
        int mc = std::min(MC_TARGET, static_cast<int>(M));
        nc = std::min(nc, static_cast<int>(N));
        return workspace_bytes(mc, nc, KC_F32, 4);
    }

    // ---- int8 variants (u8×i8, i8×i8) ---------------------------------
    if ((dtype_a == DataType::u8 || dtype_a == DataType::i8) && dtype_b == DataType::i8) {
        // Placeholder: use MR_MAX_I8 / NR_MAX_I8 until SIMD kernels define real panels.
        int nc = round_down_nc(compute_nc(MR_MAX_I8, KC_I8, 1, l2_size), NR_MAX_I8);
        int mc = std::min(MC_TARGET, static_cast<int>(M));
        nc = std::min(nc, static_cast<int>(N));
        return workspace_bytes(mc, nc, KC_I8, 1);  // 1 byte per element
    }

    // ---- fp16×int4 (future) ------------------------------------------
    if (dtype_a == DataType::f16 && dtype_b == DataType::i8) {
        // i4 weights are packed 2× per i8 byte — placeholder sizing.
        int nc = round_down_nc(compute_nc(MR_MAX_F16I4, KC_F16I4, 2, l2_size), NR_MAX_F16I4);
        int mc = std::min(MC_TARGET, static_cast<int>(M));
        nc = std::min(nc, static_cast<int>(N));
        return workspace_bytes(mc, nc, KC_F16I4, 2);  // fp16 = 2 bytes per element
    }

    // ---- unsupported combination — fallback won't pack, just return minimal ----
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

    // 2D only for now — batched falls through to reference
    if (a.rank() != 2 || b.rank() != 2) {
        reference::matmul_ref(attrs, output, inputs, ctx, workspace);
        return;
    }

    const auto dt_a = a.data_type();
    const auto dt_b = b.data_type();

    // ---- f32 × f32 ----------------------------------------------------
    if (dt_a == DataType::f32 && dt_b == DataType::f32) {
        matmul_kernel_f32(attrs, output, inputs, workspace);
        return;
    }

    // ---- f16 × f16 ----------------------------------------------------
    if (dt_a == DataType::f16 && dt_b == DataType::f16) {
        matmul_kernel_f16(attrs, output, inputs, workspace);
        return;
    }

    // ---- u8 × i8 (unsigned activation, signed weight) -----------------
    if (dt_a == DataType::u8 && dt_b == DataType::i8) {
        matmul_kernel_u8i8(attrs, output, inputs, workspace);
        return;
    }

    // ---- i8 × i8 ------------------------------------------------------
    if (dt_a == DataType::i8 && dt_b == DataType::i8) {
        matmul_kernel_i8i8(attrs, output, inputs, workspace);
        return;
    }

    // ---- fp16 × int4 (future: i4 weights packed 2× per byte) -----------
    // B dtype is i8 (container for packed i4) — distinction TBD when i4
    // becomes a first-class DataType.
    if (dt_a == DataType::f16 && dt_b == DataType::i8) {
        matmul_kernel_f16i4(attrs, output, inputs, workspace);
        return;
    }

    // Unsupported dtype combination — fall back to reference.
    reference::matmul_ref(attrs, output, inputs, ctx, workspace);
}

}  // namespace nnops::backend::cpu

