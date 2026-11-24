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
/// The fp route only: f32, f16 and bf16. An s8 W — the CPU's W8A8 route — is
/// answered with Status::unsupported, as is a prepacked (PackedRecurrent) W,
/// which the CPU packer produces and this driver never reads.
///
/// RecurrentParams is the CPU header's, because the operator dispatch layer
/// resolves one kernel signature for every backend; this file only adds the
/// device entry point.

#include "backend/cpu/rnn/recurrent.h"

#include <span>

namespace nnops::backend::cuda {

/// Run one recurrent layer over device-resident views. Every optional input is
/// absent as an empty view; the outputs may be empty individually, as the
/// operator documents.
Status recurrent_cuda(const backend::cpu::RecurrentParams& params, std::span<TensorView> outputs,
                      std::span<const TensorView> inputs, const ComputeContext& ctx);

} // namespace nnops::backend::cuda
