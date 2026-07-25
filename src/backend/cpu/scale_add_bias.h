#pragma once

#include "nnops/detail/simd/simd.hpp"

namespace nnops::backend::cpu {

using namespace nnops::simd;

template<class T>
void scale_add_bias(T* data, int ld, const T* bias, int mc, int nc, float beta) {
    
}


}