#pragma once
/// @file row_scales.hpp
/// @brief A quantized tensor's per-row scales — a caller table or the
///        per-tensor scalar — the form every W8A8 route resolves first.
///
/// QuantParams carries one scale and, optionally, a table with its own length.
/// A route that walks rows wants that choice made once, in one place: the table
/// is used only when it is at least as long as the row count the route will
/// index (`num_scales >= rows`), so a table that describes fewer rows than the
/// tensor has falls back to the scalar instead of being over-read. The CPU and
/// CUDA recurrent routes resolve it identically; the int8 GEMM engines read the
/// same pair as `sc_a`/`sc_a_scalar`.

#include "nnops/core/quant_params.hpp"

#include <cstdint>

namespace nnops::backend {

struct RowScales {
    const float* rows = nullptr; ///< null: per-tensor, use `scalar`
    float scalar = 1.0f;
};

inline RowScales row_scales_of(const QuantParams& q, int64_t rows) noexcept
{
    RowScales r;
    if (q.scale_data != nullptr && q.num_scales >= rows && rows > 0) {
        r.rows = q.scale_data;
    } else {
        r.scalar = q.scale;
    }
    return r;
}

} // namespace nnops::backend
