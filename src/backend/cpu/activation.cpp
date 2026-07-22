/// @file activation.cpp
/// @brief SIMD-optimized CPU implementation of activation functions.
///
/// All 8 activation types are vectorized with the nnops SIMD abstraction layer.
/// Supports both f32 (v_f32x8) and f16 (v_f16x8) via a single templated implementation
/// that uses the generic v_load/v_store/v_set1/s_load/s_store API.
///
/// On x86_64 this compiles to AVX2+FMA (f32) or F16C+AVX2 (f16); on AArch64 to
/// NEON (v_f32x8 emulated, v_f16x8 native on ARMv8.2+); on RISC-V to V extension.
///
/// Design decisions:
///   1. No blend/select needed — all conditionals are decomposed via v_max/v_min
///   2. Row-by-row processing respects pitch (non-contiguous tensors supported)
///   3. simd_lane_for<T> selects lane count (8 for both f32 and f16)
///   4. add_to is checked inside the loop body to avoid duplicating the entire
///      row-processing structure (branch predictor handles the invariant branch).

#include "nnops/ops/activation.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <cmath>

namespace nnops::backend::cpu {

using namespace nnops::simd;

// ============================================================
// Templated implementation (f32 and f16)
// ============================================================

template <typename T>
void activation_impl(const ActivationAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs)
{
    const auto& input = inputs[0];
    const int64_t total = input.numel();
    if (total == 0) return;

    const int64_t rank = input.rank();

    // Layout: treat the innermost dimension as contiguous "row" elements,
    // and advance by row_stride_elems between rows (accounts for pitch padding).
    const int64_t last_dim = (rank >= 1) ? input.shape(rank - 1) : 1;
    const int64_t num_rows = total / last_dim;
    const int64_t in_row_stride = input.row_stride_elems();
    const int64_t out_row_stride = output.row_stride_elems();

    const auto* in_ptr  = input.ptr<T>();
    auto* out_ptr = output.ptr<T>();
    const bool add_to = attrs.add_to;

    constexpr int L = simd_lane_for<T>;
    const auto vzero = v_zero(in_ptr);

    switch (attrs.type) {

    // ---- Relu: max(x, 0) ----
    case ActivationType::Relu: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_row_stride;
            T* out_row = out_ptr + r * out_row_stride;
            int64_t i = 0;
            for (; i + L <= last_dim; i += L) {
                auto rv = v_max(v_load(in_row + i), vzero);
                if (add_to) {
                    v_store(out_row + i, v_add(v_load(out_row + i), rv));
                } else {
                    v_store(out_row + i, rv);
                }
            }
            for (; i < last_dim; ++i) {
                float v = s_load(&in_row[i]);
                float rv = (v > 0.0f ? v : 0.0f);
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    // ---- LeakyRelu: max(x,0) + alpha * min(x,0) ----
    case ActivationType::LeakyRelu: {
        const float alpha = attrs.alpha;
        const auto a8 = v_set1(in_ptr, alpha);

        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_row_stride;
            T* out_row = out_ptr + r * out_row_stride;
            int64_t i = 0;
            for (; i + L <= last_dim; i += L) {
                auto x = v_load(in_row + i);
                auto rv = v_add(v_max(x, vzero),
                              v_mul(a8, v_min(x, vzero)));
                if (add_to) {
                    v_store(out_row + i, v_add(v_load(out_row + i), rv));
                } else {
                    v_store(out_row + i, rv);
                }
            }
            for (; i < last_dim; ++i) {
                float v = s_load(&in_row[i]);
                float rv = (v > 0.0f ? v : alpha * v);
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    // ---- Sigmoid: 1 / (1 + exp(-x)) ----
    case ActivationType::Sigmoid: {
        const auto one8 = v_set1(in_ptr, 1.0f);

        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_row_stride;
            T* out_row = out_ptr + r * out_row_stride;
            int64_t i = 0;
            for (; i + L <= last_dim; i += L) {
                auto x = v_load(in_row + i);
                auto rv = v_div(one8, v_add(one8, v_exp(v_neg(x))));
                if (add_to) {
                    v_store(out_row + i, v_add(v_load(out_row + i), rv));
                } else {
                    v_store(out_row + i, rv);
                }
            }
            for (; i < last_dim; ++i) {
                float v = s_load(&in_row[i]);
                float rv = 1.0f / (1.0f + std::exp(-v));
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    // ---- Tanh ----
    case ActivationType::Tanh: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_row_stride;
            T* out_row = out_ptr + r * out_row_stride;
            int64_t i = 0;
            for (; i + L <= last_dim; i += L) {
                auto rv = v_tanh(v_load(in_row + i));
                if (add_to) {
                    v_store(out_row + i, v_add(v_load(out_row + i), rv));
                } else {
                    v_store(out_row + i, rv);
                }
            }
            for (; i < last_dim; ++i) {
                float rv = std::tanh(s_load(&in_row[i]));
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    // ---- GELU: 0.5 * x * (1 + tanh(c * (x + 0.044715 * x^3)))
    //       where c = sqrt(2/pi) ----
    case ActivationType::Gelu: {
        const auto half8  = v_set1(in_ptr, 0.5f);
        const auto one8   = v_set1(in_ptr, 1.0f);
        const auto c8     = v_set1(in_ptr, 0.7978845608028654f);
        const auto coeff8 = v_set1(in_ptr, 0.044715f);

        auto simd = [&](auto x) {
            auto x3 = v_mul(v_mul(x, x), x);
            auto inner = v_mul(c8, v_add(x, v_mul(coeff8, x3)));
            return v_mul(v_mul(half8, x), v_add(one8, v_tanh(inner)));
        };

        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_row_stride;
            T* out_row = out_ptr + r * out_row_stride;
            int64_t i = 0;
            for (; i + L <= last_dim; i += L) {
                auto rv = simd(v_load(in_row + i));
                if (add_to) {
                    v_store(out_row + i, v_add(v_load(out_row + i), rv));
                } else {
                    v_store(out_row + i, rv);
                }
            }
            for (; i < last_dim; ++i) {
                float x = s_load(&in_row[i]);
                float rv = 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    // ---- SiLU (Swish): x / (1 + exp(-x)) ----
    case ActivationType::Silu: {
        const auto one8 = v_set1(in_ptr, 1.0f);

        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_row_stride;
            T* out_row = out_ptr + r * out_row_stride;
            int64_t i = 0;
            for (; i + L <= last_dim; i += L) {
                auto x = v_load(in_row + i);
                auto rv = v_div(x, v_add(one8, v_exp(v_neg(x))));
                if (add_to) {
                    v_store(out_row + i, v_add(v_load(out_row + i), rv));
                } else {
                    v_store(out_row + i, rv);
                }
            }
            for (; i < last_dim; ++i) {
                float x = s_load(&in_row[i]);
                float rv = x / (1.0f + std::exp(-x));
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    // ---- HardSwish: x * relu6(x+3) * (beta/6)
    //       relu6(y) = min(max(y, 0), 6) ----
    case ActivationType::HardSwish: {
        const float bd6 = attrs.beta / 6.0f;
        const auto three8 = v_set1(in_ptr, 3.0f);
        const auto six8   = v_set1(in_ptr, 6.0f);
        const auto scale8 = v_set1(in_ptr, bd6);

        auto simd = [&](auto x) {
            auto relu6 = v_min(v_max(v_add(x, three8), vzero), six8);
            return v_mul(v_mul(x, relu6), scale8);
        };

        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_row_stride;
            T* out_row = out_ptr + r * out_row_stride;
            int64_t i = 0;
            for (; i + L <= last_dim; i += L) {
                auto rv = simd(v_load(in_row + i));
                if (add_to) {
                    v_store(out_row + i, v_add(v_load(out_row + i), rv));
                } else {
                    v_store(out_row + i, rv);
                }
            }
            for (; i < last_dim; ++i) {
                float x = s_load(&in_row[i]);
                float rv = x * std::min(std::max(x + 3.0f, 0.0f), 6.0f) * bd6;
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    // ---- ELU: max(x,0) + alpha * (exp(min(x,0)) - 1) ----
    case ActivationType::Elu: {
        const float alpha = attrs.alpha;
        const auto a8   = v_set1(in_ptr, alpha);
        const auto one8 = v_set1(in_ptr, 1.0f);

        auto simd = [&](auto x) {
            auto neg_part = v_mul(a8, v_sub(v_exp(v_min(x, vzero)), one8));
            return v_add(v_max(x, vzero), neg_part);
        };

        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_row_stride;
            T* out_row = out_ptr + r * out_row_stride;
            int64_t i = 0;
            for (; i + L <= last_dim; i += L) {
                auto rv = simd(v_load(in_row + i));
                if (add_to) {
                    v_store(out_row + i, v_add(v_load(out_row + i), rv));
                } else {
                    v_store(out_row + i, rv);
                }
            }
            for (; i < last_dim; ++i) {
                float x = s_load(&in_row[i]);
                float rv = (x > 0.0f ? x : alpha * (std::exp(x) - 1.0f));
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    }  // switch
}

// ============================================================
// Main entry point with dtype dispatch
// ============================================================

void activation_cpu(const ActivationAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& /*ctx*/,
                     void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        activation_impl<float>(attrs, output, inputs);
        return;
    case DataType::f16:
        activation_impl<half>(attrs, output, inputs);
        return;
    default:
        NNOPS_ASSERT(!"activation_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
