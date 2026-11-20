#pragma once
/// @file pooling_internal.h
/// @brief Internals shared by the two Pooling translation units.
///
/// Split one file per arithmetic family:
///
///   - pooling.cpp    — the float routes (spatial, global, adaptive), the Pool1D
///                      lift and the entry point;
///   - pooling_i8.cpp — quantized pooling (s8/u8 in and out), which runs the
///                      same four-row grid over dequantize/requantize kernels.
///
/// Both take the same PoolingAttributes; the quantized family has no
/// whole-plane form, so the driver turns a global request into the full-extent
/// window walk before calling in.

#include "nnops/core/compute_context.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/ops/pooling.hpp"

#include <span>

namespace nnops::backend::cpu {

/// The quantized family's entry (pooling_i8.cpp): s8/u8 input, s8/u8 output.
void pooling_quant_run(const PoolingAttributes& attrs, TensorView& output,
                       std::span<const TensorView> inputs, const ComputeContext& ctx);

} // namespace nnops::backend::cpu
