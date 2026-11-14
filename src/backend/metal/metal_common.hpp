#pragma once
/// @file metal_common.hpp
/// @brief Shared Metal runtime for the unary/eltwise example backend.
///
/// One device, one command queue and one MSL library per process, compiled from
/// source on first use. Every kernel takes its operation as a constant, so one
/// pipeline per (function, dtype) serves all of the ops. Buffers are handed in
/// through ComputeContext::metal_buffers, mirroring the Vulkan path.

#include <cstdint>

#include "nnops/core/status.hpp"

namespace nnops {
struct ComputeContext;
}

namespace nnops::backend::metal {

enum class KernelFn : uint8_t {
    UnaryF32,
    UnaryF16,
    EltwiseF32,
    EltwiseF16,
};

/// Dispatch `fn` over `total` elements, reading `n_tensor_buffers` MTLBuffers
/// from `ctx.metal_buffers` (inputs first, output last). Returns
/// Status::unsupported when Metal is unavailable or the context carries no
/// buffers, so a Metal op degrades instead of aborting the process.
Status dispatch(KernelFn fn, int32_t op, bool add_to, const ComputeContext& ctx,
                int64_t total, int n_tensor_buffers);

}  // namespace nnops::backend::metal
