/// @file not_implemented.cu
/// @brief The CUDA entry points of the operators this backend has no kernel
///        for yet.
///
/// An operator's factory asks for the CUDA function pointer whenever the build
/// has NNOPS_HAS_CUDA, and the CPU backend links the CUDA library, so every one
/// of them has to exist or the whole tree fails to link — including the CPU
/// tests, which never call a CUDA route. Defining them here keeps the operator
/// interface uniform (a graph can ask for the backend and get a Status) while
/// the real work stays in the per-operator files.
///
/// Each returns Status::unsupported without touching its arguments, which is
/// what the CPU consumers already expect from an unimplemented backend.

#include "nnops/core/status.hpp"
#include "nnops/ops/linear_attention.hpp"
#include "nnops/ops/quant_linear.hpp"
#include "nnops/ops/topk.hpp"

namespace nnops::backend::cuda {

Status linear_attention_cuda(const LinearAttentionAttributes&,
                             std::span<TensorView>,
                             std::span<const TensorView>,
                             const ComputeContext&,
                             void*)
{
    return Status::unsupported;
}

Status quantize_linear_cuda(const QuantLinearAttributes&, TensorView&,
                            std::span<const TensorView>, const ComputeContext&,
                            void*)
{
    return Status::unsupported;
}

Status dequantize_linear_cuda(const QuantLinearAttributes&, TensorView&,
                              std::span<const TensorView>, const ComputeContext&,
                              void*)
{
    return Status::unsupported;
}

Status topk_cuda(const TopKAttributes&, TensorView&, TensorView&,
                 std::span<const TensorView>, const ComputeContext&, void*)
{
    return Status::unsupported;
}

}  // namespace nnops::backend::cuda
