#pragma once
/// @file conv1d_common.h
/// @brief The rank-3 → rank-4 view lifts that let the 1D operators run on the
///        2D kernels with a 1-wide spatial axis.
///
/// A 1D activation is [N, C, L] and maps onto 2D as [N, C, 1, L]: H is 1 and W
/// is the signal. Every 2D kernel reads H from dim 2 and W from dim 3, so the
/// conv gather, the pooling window and the im2col N axis all see the signal as
/// one row. The lift keeps the caller's pitch, so the derived channel and batch
/// strides are the 1D ones and a sub-view that starts mid-row still addresses
/// right.
///
/// The channel-last form works through the same call because a rank-3 view
/// keeps its logical dims [N, C, L] whatever the layout says; [N, L, C] in
/// memory lifts to [N, H=1, W=L, C].
///
/// Conv1D and Pool1D both use this lift, and neither owns a kernel.

#include "nnops/core/tensor_view.hpp"

#include <cstdint>
#include <span>

namespace nnops::backend::cpu {

/// A view's data pointer as a writable handle; TensorView has no data().
inline void* view_data(const TensorView& v) noexcept
{
    return const_cast<void*>(static_cast<const void*>(v.ptr<uint8_t>()));
}

/// [N, C, L] → [N, C, 1, L], pitch and layout preserved.
inline TensorView lift_wide(const TensorView& v)
{
    const int64_t s[4] = {v.shape(0), v.shape(1), 1, v.shape(2)};
    return TensorView(std::span<const int64_t>(s, 4), v.data_type(), view_data(v),
                      v.pitch(), v.layout(), v.quant_params());
}

/// A conv filter (rank 3) → its 2D order: OIHW [OC, IC/G, K] → [OC, IC/G, 1, K],
/// HWIO [K, IC/G, OC] → [1, K, IC/G, OC].
///
/// An OIHW filter's row is its last axis, so the caller's pitch carries over.
/// An HWIO one's rank-3 row is IC/G·OC, which is not its rank-4 row (K·OC), so
/// the lifted view is built dense — a filter is addressed by its dims anyway,
/// never by its pitch.
inline TensorView lift_filter_wide(const TensorView& w)
{
    const bool hwio = is_channel_last_weight(w.layout());
    const int64_t s[4] = {
        hwio ? 1 : w.shape(0),
        w.shape(hwio ? 0 : 1),
        hwio ? w.shape(1) : 1,
        w.shape(2),
    };
    if (hwio) {
        return TensorView(std::span<const int64_t>(s, 4), w.data_type(), view_data(w),
                          w.layout(), w.quant_params());
    }
    return TensorView(std::span<const int64_t>(s, 4), w.data_type(), view_data(w),
                      w.pitch(), w.layout(), w.quant_params());
}

}  // namespace nnops::backend::cpu
