/// @file epilogue_impl.cpp
/// @brief Non-inline implementations for the CPU epilogue backend.
///
/// All scalar and SIMD apply_epilogue implementations are inline in
/// epilogue_impl.hpp. This file exists for:
///   - Explicit template instantiations of matmul_epilogue_inplace
///   - Future non-inline epilogue functions (e.g. quantize/dequantize)

#include "epilogue_impl.hpp"

namespace nnops::backend::cpu {

// Explicit instantiations for matmul_epilogue_inplace (f32 and f16).
// These ensure the symbol is emitted in this translation unit so that
// callers from other .cpp files can link against it.

template void matmul_epilogue_inplace<float>(
    int M, int N, float* data, int ld, const float* bias, const Epilogue& epilogue);

template void matmul_epilogue_inplace<half>(
    int M, int N, half* data, int ld, const half* bias, const Epilogue& epilogue);

}  // namespace nnops::backend::cpu
