#pragma once
/// @file recurrent.h
/// @brief The CUDA recurrent kernel — RNN, GRU and LSTM behind one driver.
///
/// The CPU route's decomposition is the one the device keeps: the input
/// projection X[0..T]·W^T is a real GEMM over a chunk of timesteps, and the
/// serial part is only the per-timestep recurrent product H_{t-1}·R^T plus the
/// gate arithmetic. Both products go through the dense GEMM engine
/// (matmul_op.cuh), so the weights are read straight out of the caller's [G·H, K]
/// rows — they are already the kernel's k-major B pose and need no pack.
///
/// The gate arithmetic is a separate elementwise pass over [B, gates·H] rather
/// than a fused epilogue: a tile epilogue would be one more hand-written MMA
/// pipeline, while the gate row is small enough that its launch and read cost is
/// the step's noise beside the two products.
///
/// f32, f16 and bf16 on the fp engine, and the CPU's W8A8 form: an s8 W and R
/// put both products on the int8 MMA (matmul_i8.cuh), with the activation
/// quantized per row from its own max — or read from QuantParams when X arrives
/// as s8 — and the int32 accumulate dequantized into the compute dtype before
/// the same gate pass. A prepacked (PackedRecurrent) W is refused: it is a
/// CPU-packer blob this driver has no reader for.
///
/// RecurrentParams is backend/common/recurrent_config.hpp's, because the
/// operator dispatch layer resolves one kernel signature for every backend;
/// this file only adds the device entry point.

#include "backend/common/recurrent_config.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/status.hpp"
#include "nnops/core/tensor_view.hpp"

#include <span>

namespace nnops::backend::cuda {

/// Run one recurrent layer over device-resident views. Every optional input is
/// absent as an empty view; the outputs may be empty individually, as the
/// operator documents.
Status recurrent_cuda(const RecurrentParams& params, std::span<TensorView> outputs,
                      std::span<const TensorView> inputs, const ComputeContext& ctx);

} // namespace nnops::backend::cuda
