/// @file activation.cpp
/// @brief SIMD-optimized CPU implementation of activation functions.
///
/// All 8 activation types are vectorized with the nnops SIMD abstraction layer.
/// On x86_64 this compiles to AVX2+FMA by default (/arch:AVX2); 8-wide SIMD
/// with scalar tail handles arbitrary element counts.
///
/// Design decisions:
///   1. No blend/select needed — all conditionals are decomposed via v_max/v_min
///   2. 8-wide SIMD + scalar tail for maximum throughput with minimal code
///   3. add_to support: load + accumulate + store instead of plain store

#include "nnops/ops/activation.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <cmath>

namespace nnops::backend::cpu {

using namespace nnops::simd;

// ============================================================
// Main entry point
// ============================================================

void activation_cpu(const ActivationAttributes& attrs,
                     const TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& /*ctx*/,
                     void* /*workspace*/)
{
    const auto& input = inputs[0];
    const int64_t N = input.numel();
    const auto* in_ptr  = input.data_as<float>();
    auto* out_ptr = output.data_as<float>();
    const bool add_to = attrs.add_to;

    int64_t i = 0;

    switch (attrs.type) {

    // ---- Relu: max(x, 0) ----
    case ActivationType::Relu: {
        if (add_to) {
            for (; i + 8 <= N; i += 8) {
                v_f32x8 x = v_load_f32x8(in_ptr + i);
                v_f32x8 r = v_max(x, v_zero_f32x8());
                v_store(out_ptr + i, v_add(v_load_f32x8(out_ptr + i), r));
            }
            for (; i < N; ++i) {
                float v = in_ptr[i];
                out_ptr[i] += (v > 0.0f ? v : 0.0f);
            }
        } else {
            for (; i + 8 <= N; i += 8) {
                v_f32x8 x = v_load_f32x8(in_ptr + i);
                v_store(out_ptr + i, v_max(x, v_zero_f32x8()));
            }
            for (; i < N; ++i) {
                float v = in_ptr[i];
                out_ptr[i] = (v > 0.0f ? v : 0.0f);
            }
        }
        break;
    }

    // ---- LeakyRelu: max(x,0) + alpha * min(x,0) ----
    case ActivationType::LeakyRelu: {
        const float alpha = attrs.alpha;
        const v_f32x8 a8 = v_set1_f32x8(alpha);

        if (add_to) {
            for (; i + 8 <= N; i += 8) {
                v_f32x8 x = v_load_f32x8(in_ptr + i);
                v_f32x8 r = v_add(v_max(x, v_zero_f32x8()),
                                  v_mul(a8, v_min(x, v_zero_f32x8())));
                v_store(out_ptr + i, v_add(v_load_f32x8(out_ptr + i), r));
            }
            for (; i < N; ++i) {
                float v = in_ptr[i];
                out_ptr[i] += (v > 0.0f ? v : alpha * v);
            }
        } else {
            for (; i + 8 <= N; i += 8) {
                v_f32x8 x = v_load_f32x8(in_ptr + i);
                v_f32x8 r = v_add(v_max(x, v_zero_f32x8()),
                                  v_mul(a8, v_min(x, v_zero_f32x8())));
                v_store(out_ptr + i, r);
            }
            for (; i < N; ++i) {
                float v = in_ptr[i];
                out_ptr[i] = (v > 0.0f ? v : alpha * v);
            }
        }
        break;
    }

    // ---- Sigmoid: 1 / (1 + exp(-x)) ----
    case ActivationType::Sigmoid: {
        const v_f32x8 one8 = v_set1_f32x8(1.0f);

        if (add_to) {
            for (; i + 8 <= N; i += 8) {
                v_f32x8 x = v_load_f32x8(in_ptr + i);
                v_f32x8 r = v_div(one8, v_add(one8, v_exp(v_neg(x))));
                v_store(out_ptr + i, v_add(v_load_f32x8(out_ptr + i), r));
            }
            for (; i < N; ++i) {
                out_ptr[i] += 1.0f / (1.0f + std::exp(-in_ptr[i]));
            }
        } else {
            for (; i + 8 <= N; i += 8) {
                v_f32x8 x = v_load_f32x8(in_ptr + i);
                v_store(out_ptr + i, v_div(one8, v_add(one8, v_exp(v_neg(x)))));
            }
            for (; i < N; ++i) {
                out_ptr[i] = 1.0f / (1.0f + std::exp(-in_ptr[i]));
            }
        }
        break;
    }

    // ---- Tanh ----
    case ActivationType::Tanh: {
        if (add_to) {
            for (; i + 8 <= N; i += 8) {
                v_f32x8 x = v_load_f32x8(in_ptr + i);
                v_f32x8 r = v_tanh(x);
                v_store(out_ptr + i, v_add(v_load_f32x8(out_ptr + i), r));
            }
            for (; i < N; ++i) {
                out_ptr[i] += std::tanh(in_ptr[i]);
            }
        } else {
            for (; i + 8 <= N; i += 8) {
                v_f32x8 x = v_load_f32x8(in_ptr + i);
                v_store(out_ptr + i, v_tanh(x));
            }
            for (; i < N; ++i) {
                out_ptr[i] = std::tanh(in_ptr[i]);
            }
        }
        break;
    }

    // ---- GELU: 0.5 * x * (1 + tanh(c * (x + 0.044715 * x^3)))
    //       where c = sqrt(2/pi)
    case ActivationType::Gelu: {
        const v_f32x8 half8  = v_set1_f32x8(0.5f);
        const v_f32x8 one8   = v_set1_f32x8(1.0f);
        const v_f32x8 c8     = v_set1_f32x8(0.7978845608028654f);
        const v_f32x8 coeff8 = v_set1_f32x8(0.044715f);

        auto simd8 = [&](v_f32x8 x) {
            v_f32x8 x3 = v_mul(v_mul(x, x), x);
            v_f32x8 inner = v_mul(c8, v_add(x, v_mul(coeff8, x3)));
            return v_mul(v_mul(half8, x), v_add(one8, v_tanh(inner)));
        };

        if (add_to) {
            for (; i + 8 <= N; i += 8) {
                v_f32x8 x = v_load_f32x8(in_ptr + i);
                v_store(out_ptr + i, v_add(v_load_f32x8(out_ptr + i), simd8(x)));
            }
            for (; i < N; ++i) {
                float x = in_ptr[i];
                out_ptr[i] += 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
            }
        } else {
            for (; i + 8 <= N; i += 8) {
                v_store(out_ptr + i, simd8(v_load_f32x8(in_ptr + i)));
            }
            for (; i < N; ++i) {
                float x = in_ptr[i];
                out_ptr[i] = 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
            }
        }
        break;
    }

    // ---- SiLU (Swish): x / (1 + exp(-x)) ----
    case ActivationType::Silu: {
        const v_f32x8 one8 = v_set1_f32x8(1.0f);

        if (add_to) {
            for (; i + 8 <= N; i += 8) {
                v_f32x8 x = v_load_f32x8(in_ptr + i);
                v_f32x8 r = v_div(x, v_add(one8, v_exp(v_neg(x))));
                v_store(out_ptr + i, v_add(v_load_f32x8(out_ptr + i), r));
            }
            for (; i < N; ++i) {
                float x = in_ptr[i];
                out_ptr[i] += x / (1.0f + std::exp(-x));
            }
        } else {
            for (; i + 8 <= N; i += 8) {
                v_f32x8 x = v_load_f32x8(in_ptr + i);
                v_store(out_ptr + i, v_div(x, v_add(one8, v_exp(v_neg(x)))));
            }
            for (; i < N; ++i) {
                float x = in_ptr[i];
                out_ptr[i] = x / (1.0f + std::exp(-x));
            }
        }
        break;
    }

    // ---- HardSwish: x * relu6(x+3) * (beta/6)
    //       relu6(y) = min(max(y, 0), 6)
    case ActivationType::HardSwish: {
        const float bd6 = attrs.beta / 6.0f;
        const v_f32x8 three8 = v_set1_f32x8(3.0f);
        const v_f32x8 six8   = v_set1_f32x8(6.0f);
        const v_f32x8 scale8 = v_set1_f32x8(bd6);

        auto simd8 = [&](v_f32x8 x) {
            v_f32x8 relu6 = v_min(v_max(v_add(x, three8), v_zero_f32x8()), six8);
            return v_mul(v_mul(x, relu6), scale8);
        };

        if (add_to) {
            for (; i + 8 <= N; i += 8) {
                v_f32x8 x = v_load_f32x8(in_ptr + i);
                v_store(out_ptr + i, v_add(v_load_f32x8(out_ptr + i), simd8(x)));
            }
            for (; i < N; ++i) {
                float x = in_ptr[i];
                out_ptr[i] += x * std::min(std::max(x + 3.0f, 0.0f), 6.0f) * bd6;
            }
        } else {
            for (; i + 8 <= N; i += 8) {
                v_store(out_ptr + i, simd8(v_load_f32x8(in_ptr + i)));
            }
            for (; i < N; ++i) {
                float x = in_ptr[i];
                out_ptr[i] = x * std::min(std::max(x + 3.0f, 0.0f), 6.0f) * bd6;
            }
        }
        break;
    }

    // ---- ELU: max(x,0) + alpha * (exp(min(x,0)) - 1) ----
    case ActivationType::Elu: {
        const float alpha = attrs.alpha;
        const v_f32x8 a8   = v_set1_f32x8(alpha);
        const v_f32x8 one8 = v_set1_f32x8(1.0f);

        auto simd8 = [&](v_f32x8 x) {
            v_f32x8 neg_part = v_mul(a8, v_sub(v_exp(v_min(x, v_zero_f32x8())), one8));
            return v_add(v_max(x, v_zero_f32x8()), neg_part);
        };

        if (add_to) {
            for (; i + 8 <= N; i += 8) {
                v_f32x8 x = v_load_f32x8(in_ptr + i);
                v_store(out_ptr + i, v_add(v_load_f32x8(out_ptr + i), simd8(x)));
            }
            for (; i < N; ++i) {
                float x = in_ptr[i];
                out_ptr[i] += (x > 0.0f ? x : alpha * (std::exp(x) - 1.0f));
            }
        } else {
            for (; i + 8 <= N; i += 8) {
                v_store(out_ptr + i, simd8(v_load_f32x8(in_ptr + i)));
            }
            for (; i < N; ++i) {
                float x = in_ptr[i];
                out_ptr[i] = (x > 0.0f ? x : alpha * (std::exp(x) - 1.0f));
            }
        }
        break;
    }

    }  // switch
}

}  // namespace nnops::backend::cpu
