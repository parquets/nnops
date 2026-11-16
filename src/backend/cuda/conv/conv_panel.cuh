/// @file conv_panel.cuh
/// @brief The shared-memory panel indexers the conv gathers write through.
///
/// A gather is written against a (row, col, stage) indexer -- the shape CuTe's
/// shared tensor used to provide. These are the same interface over the
/// hand-written panels: the k-major swizzle the 16-bit and int8 mainloops read,
/// and the padded rows the f32 one reads. The base pointer already carries the
/// stage, so the third argument is kept only to match the gathers' signatures.

#pragma once

#include "../matmul/matmul_f16.cuh" // manual_swz

namespace nnops::backend::cuda::conv {

template <class S, class Elem>
struct KMajorPanel {
    Elem* base;
    __device__ __forceinline__ Elem& operator()(int row, int col, int) const
    {
        return base[manual_swz<S>(row, col)];
    }
};

template <int LDA>
struct PaddedPanel {
    float* base;
    __device__ __forceinline__ float& operator()(int row, int col, int) const
    {
        return base[row * LDA + col];
    }
};

} // namespace nnops::backend::cuda::conv
