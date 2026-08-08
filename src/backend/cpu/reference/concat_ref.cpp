/// @file concat_ref.cpp
/// @brief Naive CPU reference implementation of Concat.
///
/// Correctness baseline. Handles all layouts uniformly (NCHW, NCDHW,
/// NCHWC8, NCDHWC8), including C-axis concatenation with packed layouts
/// which requires C8 lane-level merging.

#include "nnops/ops/concat.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/simd/simd.hpp"
#include <cstring>
#include <vector>

namespace nnops::backend::cpu::reference {

using nnops::simd::s_load;
using nnops::simd::s_store;

// ============================================================
// Packed C-axis concat — lane-level merge within C8 vectors
// ============================================================

template <typename T>
void concat_packed_c_axis_ref(TensorView& output,
                               std::span<const TensorView> inputs)
{
    const int64_t N = static_cast<int64_t>(inputs.size());
    const int64_t rank = output.rank();
    const int64_t pack = output.channel_pack_size();
    const int64_t srank = rank - 2;  // 2 or 3

    // Cumulative logical channel counts per input
    std::vector<int64_t> ch_start(N + 1, 0);
    for (int64_t i = 0; i < N; ++i) {
        ch_start[static_cast<size_t>(i + 1)] =
            ch_start[static_cast<size_t>(i)] + inputs[static_cast<size_t>(i)].shape(1);
    }
    const int64_t total_logical_c = ch_start[static_cast<size_t>(N)];

    const int64_t oN  = output.shape(0);
    const int64_t oC8 = output.num_channel_blocks();
    const int64_t oD  = (srank == 3) ? output.shape(2) : 1;
    const int64_t oH  = output.shape(srank);
    const int64_t oW  = output.shape(rank - 1);

    auto* o_ptr = output.ptr<T>();
    const int64_t o_row_stride = output.row_stride_elems();
    const int64_t o_ch_stride  = output.channel_block_stride_elems();

    // Precompute source mapping for each output (C8 block, lane) pair.
    // LaneSrc::input_idx == -1 means pad lane (past total_logical_c).
    struct LaneSrc { int64_t input_idx; int64_t c8_block; int64_t lane; };
    std::vector<std::vector<LaneSrc>> lane_map(
        static_cast<size_t>(oC8),
        std::vector<LaneSrc>(static_cast<size_t>(pack)));

    for (int64_t c8 = 0; c8 < oC8; ++c8) {
        for (int64_t lane = 0; lane < pack; ++lane) {
            int64_t o_lc = c8 * pack + lane;
            if (o_lc >= total_logical_c) {
                lane_map[static_cast<size_t>(c8)][static_cast<size_t>(lane)] = {-1, -1, -1};
                continue;
            }
            // Find which input provides this logical channel
            int64_t in_idx = 0;
            while (in_idx < N - 1 && o_lc >= ch_start[static_cast<size_t>(in_idx + 1)]) {
                ++in_idx;
            }
            int64_t in_lc = o_lc - ch_start[static_cast<size_t>(in_idx)];
            lane_map[static_cast<size_t>(c8)][static_cast<size_t>(lane)] = {
                in_idx,
                in_lc / pack,
                in_lc % pack
            };
        }
    }

    // Determine which output C8 blocks are a 1:1 copy from a single input block
    // (all lanes from same input, same C8 block, in sequential order).
    std::vector<int64_t> full_copy_src(static_cast<size_t>(oC8), -1);
    for (int64_t c8 = 0; c8 < oC8; ++c8) {
        const auto& map = lane_map[static_cast<size_t>(c8)];
        int64_t src_idx = map[0].input_idx;
        int64_t src_c8  = map[0].c8_block;
        if (src_idx < 0) {
            continue;
        }
        bool is_full = true;
        for (int64_t l = 0; l < pack; ++l) {
            if (map[static_cast<size_t>(l)].input_idx != src_idx ||
                map[static_cast<size_t>(l)].c8_block != src_c8 ||
                map[static_cast<size_t>(l)].lane != l) {
                is_full = false;
                break;
            }
        }
        if (is_full) {
            full_copy_src[static_cast<size_t>(c8)] = src_idx;
        }
    }

    // Iterate over all spatial positions and fill C8 blocks
    for (int64_t n = 0; n < oN; ++n) {
        for (int64_t d = 0; d < oD; ++d) {
            for (int64_t h = 0; h < oH; ++h) {
                // Base pointer for this (n, d, h) spatial row
                T* o_row = o_ptr
                    + n  * output.stride_elems(0)
                    + (srank == 3 ? d * output.stride_elems(2) : int64_t(0))
                    + h  * o_row_stride;

                for (int64_t w = 0; w < oW; ++w) {
                    for (int64_t c8 = 0; c8 < oC8; ++c8) {
                        T* o_vec = o_row + c8 * o_ch_stride + w * pack;
                        int64_t src_idx = full_copy_src[static_cast<size_t>(c8)];

                        if (src_idx >= 0) {
                            // Full C8 block — copy the entire vector
                            const auto& in = inputs[static_cast<size_t>(src_idx)];
                            const T* i_vec = in.ptr<T>()
                                + n  * in.stride_elems(0)
                                + (srank == 3 ? d * in.stride_elems(2) : int64_t(0))
                                + c8 * in.channel_block_stride_elems()
                                + h  * in.row_stride_elems()
                                + w  * pack;
                            for (int64_t l = 0; l < pack; ++l) {
                                o_vec[l] = i_vec[l];
                            }
                        } else {
                            // Partial C8 block — lane-by-lane copy
                            const auto& map = lane_map[static_cast<size_t>(c8)];
                            for (int64_t l = 0; l < pack; ++l) {
                                const auto& src = map[static_cast<size_t>(l)];
                                if (src.input_idx < 0) {
                                    o_vec[l] = T(0);  // pad
                                    continue;
                                }
                                const auto& in = inputs[static_cast<size_t>(src.input_idx)];
                                const T* i_vec = in.ptr<T>()
                                    + n  * in.stride_elems(0)
                                    + (srank == 3 ? d * in.stride_elems(2) : int64_t(0))
                                    + src.c8_block * in.channel_block_stride_elems()
                                    + h  * in.row_stride_elems()
                                    + w  * pack;
                                o_vec[l] = i_vec[src.lane];
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================
// General concat path (planar all axes + packed non-C axes)
// ============================================================

template <typename T>
void concat_impl_ref(const ConcatAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs)
{
    const int64_t N = static_cast<int64_t>(inputs.size());
    if (N == 0) {
        return;
    }

    const int64_t rank = inputs[0].rank();
    const int64_t ax = (attrs.axis < 0) ? attrs.axis + rank : attrs.axis;
    const int64_t pack = inputs[0].channel_pack_size();

    // ---- Packed C-axis concat (requires lane-level merge) ----
    if (ax == 1 && pack > 1 && rank >= 3) {
        concat_packed_c_axis_ref<T>(output, inputs);
        return;
    }

    // ---- Compute axis offsets (physical dims for packed C axis) ----
    std::vector<int64_t> axis_offset(static_cast<size_t>(N + 1), 0);
    for (int64_t i = 0; i < N; ++i) {
        axis_offset[static_cast<size_t>(i + 1)] =
            axis_offset[static_cast<size_t>(i)] + inputs[static_cast<size_t>(i)].shape(ax);
    }

    const bool is_rank1 = (rank < 2);
    const int64_t num_rows = is_rank1 ? 1 : output.total_rows();
    const int64_t last_dim = is_rank1
        ? output.numel()
        : output.shape(rank - 1) * output.channel_pack_size();
    const int64_t o_row_stride = is_rank1 ? 1 : output.row_stride_elems();

    const bool axis_splits_row = (ax >= rank - 1);
    auto* o_ptr = output.ptr<T>();

    if (axis_splits_row) {
        // ---- Within-row concat: each input provides a segment of each row ----
        for (int64_t r = 0; r < num_rows; ++r) {
            T* o_row = o_ptr + r * o_row_stride;
            int64_t o_col = 0;
            for (int64_t n = 0; n < N; ++n) {
                const auto& in = inputs[static_cast<size_t>(n)];
                const int64_t in_row_stride = is_rank1 ? 1 : in.row_stride_elems();
                const int64_t in_last_dim = is_rank1
                    ? in.numel()
                    : in.shape(rank - 1) * in.channel_pack_size();
                const T* i_row = in.ptr<T>() + r * in_row_stride;
                for (int64_t c = 0; c < in_last_dim; ++c) {
                    s_store(&o_row[o_col++], s_load(&i_row[c]));
                }
            }
        }
    } else {
        // ---- Outer axis concat: rows/blocks from different inputs ----
        const int64_t n_pre = ax;  // number of dimensions before the concat axis
        const int64_t phys_inner = is_rank1 ? output.numel() : output.stride_elems(ax);

        if (n_pre == 0) {
            // ax == 0: each input contributes entire rank-1..rank-N "blocks"
            for (int64_t n = 0; n < N; ++n) {
                const auto& in = inputs[static_cast<size_t>(n)];
                const int64_t in_axis_dim = in.shape(ax);
                const int64_t i_stride_ax = is_rank1 ? 1 : in.stride_elems(ax);
                const T* i_ptr = in.ptr<T>();

                for (int64_t k = 0; k < in_axis_dim; ++k) {
                    int64_t o_pos = (axis_offset[static_cast<size_t>(n)] + k) * phys_inner;
                    int64_t i_pos = k * i_stride_ax;
                    std::memcpy(o_ptr + o_pos, i_ptr + i_pos,
                                static_cast<size_t>(phys_inner) * sizeof(T));
                }
            }
        } else {
            // n_pre >= 1: iterate over all outer dimensions with an odometer
            // Use physical channel block count for packed layouts
            std::vector<int64_t> pre_sizes(static_cast<size_t>(n_pre));
            std::vector<int64_t> pre_strides(static_cast<size_t>(n_pre));
            for (int64_t d = 0; d < n_pre; ++d) {
                if (d == 1 && pack > 1) {
                    pre_sizes[static_cast<size_t>(d)] = output.num_channel_blocks();
                }
                else {
                    pre_sizes[static_cast<size_t>(d)] = output.shape(d);
                }
                pre_strides[static_cast<size_t>(d)] = output.stride_elems(d);
            }

            // Odometer iteration
            std::vector<int64_t> idx(static_cast<size_t>(n_pre), 0);
            do {
                // Compute output base offset for this outer position
                int64_t o_base = 0;
                for (int64_t d = 0; d < n_pre; ++d) {
                    o_base += idx[static_cast<size_t>(d)]
                            * pre_strides[static_cast<size_t>(d)];

                }
                for (int64_t n = 0; n < N; ++n) {
                    const auto& in = inputs[static_cast<size_t>(n)];
                    const int64_t in_axis_dim = in.shape(ax);
                    const int64_t i_stride_ax = is_rank1 ? 1 : in.stride_elems(ax);
                    const T* i_ptr = in.ptr<T>();

                    // Compute input base offset
                    int64_t i_base = 0;
                    for (int64_t d = 0; d < n_pre; ++d) {
                        i_base += idx[static_cast<size_t>(d)]
                                * in.stride_elems(d);

                    }
                    int64_t o_pos = o_base
                        + axis_offset[static_cast<size_t>(n)] * phys_inner;
                    int64_t i_pos = i_base;

                    for (int64_t k = 0; k < in_axis_dim; ++k) {
                        std::memcpy(o_ptr + o_pos + k * phys_inner,
                                    i_ptr + i_pos + k * phys_inner,
                                    static_cast<size_t>(phys_inner) * sizeof(T));
                    }
                }

                // Increment odometer
                int64_t d = n_pre - 1;
                while (d >= 0 && ++idx[static_cast<size_t>(d)]
                                  == pre_sizes[static_cast<size_t>(d)])
                    idx[static_cast<size_t>(d--)] = 0;
                if (d < 0) {
                    break;
                }
            } while (true);
        }
    }
}

void concat_ref(const ConcatAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& /*ctx*/,
                 void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        concat_impl_ref<float>(attrs, output, inputs);
        return;
    case DataType::f16:
        concat_impl_ref<half>(attrs, output, inputs);
        return;
    default:
        NNOPS_ASSERT(!"concat_ref: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu::reference
