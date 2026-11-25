/// @file bcast.cuh
/// @brief The NumPy-broadcast walk the elementwise operators share.
///
/// Compare, Logical and Where take operands of different ranks and shapes, and
/// the output is the broadcast of all of them. The walk is the movement
/// engine's: the output is flat, each operand's offset for an output row comes
/// from one mixed-radix decode (`move_offset`), and inside the row the operand
/// advances by a constant stride — its last-axis step, which is 0 when that
/// axis was broadcast. So a broadcast operand is one load per row rather than
/// one per element, and a stride-1 pair keeps the 16 B path.
///
/// Axes align from the right, as the broadcasting rules say; an operand axis of
/// extent 1 against a wider output axis gets stride 0.

#pragma once

#include "move_common.cuh"

namespace nnops::backend::cuda {

/// One operand of a broadcast elementwise op.
struct BcastOperand {
    MoveLayout layout; ///< output flat index -> element offset
    int64_t step = 0;  ///< stride along the output's last axis (0 = broadcast)
    /// The layout maps every flat index to itself (a dense, same-shape
    /// operand). The overwhelming majority of calls are this, and then the
    /// per-row offset is the flat index itself — no odometer, and no 128 B
    /// layout struct materialized per thread.
    bool linear = false;
};

/// The odometer, inlined. `move_offset` is `__noinline__` (cicc aborts on the
/// unrolled form inside the movement row kernel), but a kernel that decodes one
/// offset per element tile must not spill the struct to local memory, so the
/// elementwise walk carries its own inline copy.
__device__ inline int64_t bcast_offset(const MoveLayout& l, int64_t flat) noexcept
{
    int64_t off = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int64_t q = flat / l.dims[i];
        off += (flat - q * l.dims[i]) * l.strides[i];
        flat = q;
    }
    return off;
}

/// The output's own axes, padded to the engine's rank 8 with ones.
inline void bcast_out_dims(const TensorView& out, int64_t (&dims)[8])
{
    const int64_t rank = out.rank();
    for (int64_t d = 0; d < rank; ++d) {
        dims[d] = out.shape(d);
    }
    for (int64_t d = rank; d < 8; ++d) {
        dims[d] = 1;
    }
}

/// Walk `v` against an output of `out_rank` axes.
inline BcastOperand make_bcast(const TensorView& v, const int64_t (&out_dims)[8],
                               int64_t out_rank)
{
    const int64_t vr = v.rank();
    int64_t strides[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int64_t off = out_rank - vr;
    for (int64_t d = 0; d < vr; ++d) {
        const int64_t od = off + d;
        if (od >= 0 && v.shape(d) != 1) {
            strides[od] = v.stride_elems(d);
        }
    }
    BcastOperand o;
    o.layout = make_move_layout(out_dims, strides);
    o.step = out_rank > 0 ? strides[out_rank - 1] : 1;
    // Dense and same-shape: every flat index is its own offset.
    bool linear = true;
    int64_t want = 1;
    for (int64_t d = out_rank - 1; d >= 0; --d) {
        if (out_dims[d] != 1 && strides[d] != want) {
            linear = false;
            break;
        }
        want *= out_dims[d];
    }
    o.linear = linear;
    return o;
}

/// The launch shape for a row walk: a block covers a slice of one row, and the
/// rows ride `gridDim.y`. Rows past the 16-bit grid axis are looped.
constexpr int kBcastThreads = 256;

inline dim3 bcast_grid(int64_t cols, int64_t rows, int64_t vec) noexcept
{
    const int64_t per_block = int64_t(kBcastThreads) * vec;
    const unsigned gx = static_cast<unsigned>((cols + per_block - 1) / per_block);
    const unsigned gy = static_cast<unsigned>(std::min<int64_t>(rows, 65535));
    return dim3(gx > 0 ? gx : 1, gy > 0 ? gy : 1);
}

} // namespace nnops::backend::cuda
