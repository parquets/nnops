#pragma once
/// @file conv1d_common.h
/// @brief The rank-3 → rank-4 view lifts that let the 1D operators run on the
///        2D kernels with a 1-wide spatial axis.
///
/// A channel-last 1D activation is [N, L, C] and maps onto 2D as
/// [N, 1, L, C]: H is 1 and W is the signal. Every 2D kernel reads H from axis
/// 1 and W from axis 2, so the conv gather, the pooling window and the im2col
/// N axis all see the signal as one row. The lift keeps the caller's pitch, so
/// the derived channel and batch strides are the 1D ones and a sub-view that
/// starts mid-row still addresses right.
///
/// A dense (channel-first) activation is not a 1D operator input: Conv1D and
/// Pool1D take channel-last tensors only.
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

/// [N, L, C] → [N, 1, L, C], pitch and layout preserved.
inline TensorView lift_wide(const TensorView& v)
{
    const int64_t s[4] = {v.shape(0), 1, v.shape(1), v.shape(2)};
    return TensorView(std::span<const int64_t>(s, 4), v.data_type(), view_data(v), v.pitch(),
                      v.layout(), v.quant_params());
}

/// An NHWC conv filter (rank 3) → its 2D order:
/// HWIO [K, IC/G, OC] → [1, K, IC/G, OC].
///
/// The rank-3 row is IC/G·OC, which is not the rank-4 row (K·OC), so the lifted
/// view is built dense — a filter is addressed by its dims anyway, never by its
/// pitch.
inline TensorView lift_filter_wide(const TensorView& w)
{
    const int64_t s[4] = {1, w.shape(0), w.shape(1), w.shape(2)};
    return TensorView(std::span<const int64_t>(s, 4), w.data_type(), view_data(w), w.layout(),
                      w.quant_params());
}

} // namespace nnops::backend::cpu
