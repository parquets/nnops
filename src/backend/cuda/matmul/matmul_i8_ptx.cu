/// @file matmul_i8_ptx.cu
/// @brief The host front of the hand-written int8 tile: the tile menu, the
///        automatic pick, and the launch.
///
/// Split out of the kernel header on purpose. The kernel is a template with
/// three tiles x four poses x two element types x two destination families, and
/// while the dispatch lived inline every translation unit that included
/// matmul_i8.cuh -- matmul.cu, linear.cu and conv2d.cu -- instantiated all
/// forty-eight of them, which tripled a full rebuild. One definition, one set of
/// instantiations, and only this file pays for them.
///
/// See matmul_i8_ptx.cuh for the kernel and the fragment mapping.

#include "matmul_i8_ptx_host.hpp"

#include "matmul_i8_ptx.cuh"
#include "nnops/detail/bench_knobs.hpp"

namespace nnops::backend::cuda {

namespace detail {

/// The candidate tiles. 64x64x64 on four warps is the default: the s8 panels are
/// half the fp16 footprint, so two stages cost 20 KB and four blocks fit an SM.
using I8Tile0 = ManualI8Shape<32, 32, 2, 2, 64, 2, 4>;
/// Twice the rows per CTA: the byte transpose of a transposed operand is paid
/// once per 128 rows instead of per 64, which is what an n-major B wants.
using I8Tile1 = ManualI8Shape<64, 32, 2, 2, 64, 2, 2>;
/// Twice the rows *and* the columns per CTA on eight warps.
using I8Tile2 = ManualI8Shape<64, 32, 2, 4, 64, 2, 2>;

constexpr int kI8TileCount = 3;
constexpr int kI8DefaultTile = 0;

/// A dynamic shared memory request above 48 KB needs the opt-in, once per
/// instantiation; these tiles fit under it, and setting it is still the contract
/// the other hand-written GEMMs share.
template <class S, class AElem, bool Dequant>
bool i8_prepare()
{
    static const bool ok = [] {
        const int bytes = S::kSmemBytes;
        const auto set = [bytes](const void* fn) {
            return cudaFuncSetAttribute(
                       fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) ==
                   cudaSuccess;
        };
        if (!set(reinterpret_cast<const void*>(
                &manual_gemm_i8_kernel<S, AElem, true, true, Dequant>)) ||
            !set(reinterpret_cast<const void*>(
                &manual_gemm_i8_kernel<S, AElem, true, false, Dequant>)) ||
            !set(reinterpret_cast<const void*>(
                &manual_gemm_i8_kernel<S, AElem, false, true, Dequant>)) ||
            !set(reinterpret_cast<const void*>(
                &manual_gemm_i8_kernel<S, AElem, false, false, Dequant>))) {
            cudaGetLastError();
            return false;
        }
        return true;
    }();
    return ok;
}

template <class S, class AElem, bool Dequant>
bool i8_run(GemmI8Params q, bool trans_a, bool trans_b, int batch,
            cudaStream_t stream)
{
    if (!i8_prepare<S, AElem, Dequant>()) { return false; }
    // One dimensional in (m, n): the kernel's L2 raster decodes the pair, so a
    // wave shares an operand band instead of walking the n axis.
    const dim3 grid(unsigned(((q.g.m + S::BM - 1) / S::BM) *
                             ((q.g.n + S::BN - 1) / S::BN)),
                    1, unsigned(batch));
    constexpr int kSmem = S::kSmemBytes;
    if (trans_a && trans_b) {
        manual_gemm_i8_kernel<S, AElem, true, true, Dequant>
            <<<grid, S::threads, kSmem, stream>>>(q);
    } else if (trans_a) {
        manual_gemm_i8_kernel<S, AElem, true, false, Dequant>
            <<<grid, S::threads, kSmem, stream>>>(q);
    } else if (trans_b) {
        manual_gemm_i8_kernel<S, AElem, false, true, Dequant>
            <<<grid, S::threads, kSmem, stream>>>(q);
    } else {
        manual_gemm_i8_kernel<S, AElem, false, false, Dequant>
            <<<grid, S::threads, kSmem, stream>>>(q);
    }
    return cudaGetLastError() == cudaSuccess;
}

template <class AElem, bool Dequant>
bool i8_run_tile(GemmI8Params q, bool trans_a, bool trans_b, int batch,
                 cudaStream_t stream, int tile)
{
    switch (tile) {
    case 0: return i8_run<I8Tile0, AElem, Dequant>(q, trans_a, trans_b, batch,
                                                   stream);
    case 1: return i8_run<I8Tile1, AElem, Dequant>(q, trans_a, trans_b, batch,
                                                   stream);
    default: return i8_run<I8Tile2, AElem, Dequant>(q, trans_a, trans_b, batch,
                                                    stream);
    }
}

}  // namespace detail

/// Which menu row the automatic path takes. Both choices are the fill's: the two
/// poses differ in which operand arrives along the wrong axis, and a transposed
/// operand can only be staged synchronously, so its cost per MAC is 1/tile rows.
/// A k-contiguous B (NT) leaves only A's vector copy, so the widest tile wins as
/// soon as it has a wave to fill (4096^3: 1.79 ms against 2.33 for the 128x64
/// tile, and 2.31 for the CuTe engine it replaces). An n-major B (NN) pays the
/// byte transpose on every tile, and the 128-row tile is what halves it -- but
/// only once the grid is deep enough to keep the machine busy without the extra
/// rows (2.30 against 3.29 there, while at 512^3 the smaller tile is ahead).
int i8_auto_tile(const GemmParams& g, bool trans_b)
{
    const int sm = detail::device_sm_count();
    if (trans_b) {
        const int64_t blocks = int64_t((g.m + 127) / 128) * ((g.n + 127) / 128);
        return (sm <= 0 || blocks >= int64_t(sm)) ? 2 : 0;
    }
    const int64_t blocks = int64_t((g.m + 127) / 128) * ((g.n + 63) / 64);
    return (sm <= 0 || blocks >= int64_t(sm) * 4) ? 1 : 0;
}

/// The hand-written int8 tile. `tile` picks a candidate explicitly (-1 takes the
/// default row); false means the caller keeps its own engine, which is the
/// contract for a pose this kernel does not cover.
bool manual_dense_i8_tile(GemmI8Params q, DataType a_dt, bool trans_a,
                          bool trans_b, cudaStream_t stream, int tile)
{
    if (a_dt != DataType::s8 && a_dt != DataType::u8) { return false; }
    const GemmParams& g = q.g;
    if (g.m <= 0 || g.n <= 0 || g.k <= 0) { return false; }
    if (q.r_a == nullptr || q.r_b == nullptr) { return false; }
    // The transposed-side fills read 4-byte words out of a source whose
    // contiguous run is along the other axis, so both the base and the row pitch
    // have to be word-addressable. The k-contiguous fills have their own byte
    // path and need no alignment.
    const auto word_ok = [](const void* p, int64_t ld) {
        return (reinterpret_cast<uintptr_t>(p) % 4 == 0) && (ld % 4 == 0);
    };
    if (trans_a && !word_ok(g.a, g.lda)) { return false; }
    if (!trans_b && !word_ok(g.b, g.ldb)) { return false; }
    const int batch = detail::gemm_batch_count(g);
    if (batch <= 0) { return false; }
    if (tile < 0) { tile = i8_auto_tile(g, trans_b); }
    if (tile >= detail::kI8TileCount) { tile = detail::kI8DefaultTile; }
    const bool dequant = q.out == I8OutKind::F32 || q.out == I8OutKind::F16;
    if (a_dt == DataType::u8) {
        return dequant
                   ? detail::i8_run_tile<uint8_t, true>(q, trans_a, trans_b,
                                                        batch, stream, tile)
                   : detail::i8_run_tile<uint8_t, false>(q, trans_a, trans_b,
                                                         batch, stream, tile);
    }
    return dequant
               ? detail::i8_run_tile<int8_t, true>(q, trans_a, trans_b, batch,
                                                   stream, tile)
               : detail::i8_run_tile<int8_t, false>(q, trans_a, trans_b, batch,
                                                    stream, tile);
}
}  // namespace nnops::backend::cuda
