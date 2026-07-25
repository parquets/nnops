/// @file unary.cpp
/// @brief SIMD-optimized CPU implementation of element-wise unary operations.
///
/// Supports both f32 and f16 via a single templated implementation.
/// Row-by-row pitch-aware processing with SIMD inner loop + scalar tail.
/// All 9 operations (Exp/Log/Sin/Cos/Tan/Tanh/Abs/Neg/Sqrt) are vectorized.
///
/// Design:
///   1. Row-by-row processing via row_stride_elems() respects pitch padding
///   2. simd_lane_for<T> selects lane count (8 for both f32 and f16)
///   3. Branch on op type is hoisted outside the row loop
///   4. add_to is checked inside the loop body (branch predictor handles it)

#include "nnops/ops/unary.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <cmath>

namespace nnops::backend::cpu {

using namespace nnops::simd;

// ============================================================
// Templated implementation (f32 and f16)
// ============================================================

template <typename T>
void unary_impl(const UnaryAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& /*ctx*/)
{
    const auto& input = inputs[0];
    const int64_t total = input.numel();
    if (total == 0) return;

    const int64_t rank = input.rank();
    NNOPS_ASSERT(output.numel() == total);

    // Row-by-row layout (pitch-aware)
    const int64_t last_dim = (rank >= 1) ? input.shape(rank - 1) : 1;
    const int64_t num_rows = total / last_dim;
    const int64_t in_row_stride = input.row_stride_elems();
    const int64_t out_row_stride = output.row_stride_elems();

    const auto* in_ptr  = input.ptr<T>();
    auto*       out_ptr = output.ptr<T>();
    const bool add_to = attrs.add_to;

    constexpr int L = simd_lane_for<T>;

    switch (attrs.type) {

    // ---- Exp: y = exp(x) ----
    case UnaryType::Exp: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_row_stride;
            T* out_row = out_ptr + r * out_row_stride;
            int64_t i = 0;
            for (; i + L <= last_dim; i += L) {
                auto rv = v_exp(v_load(in_row + i));
                if (add_to) {
                    v_store(out_row + i, v_add(v_load(out_row + i), rv));
                } else {
                    v_store(out_row + i, rv);
                }
            }
            for (; i < last_dim; ++i) {
                float rv = std::exp(s_load(&in_row[i]));
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    // ---- Log: y = ln(x) ----
    case UnaryType::Log: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_row_stride;
            T* out_row = out_ptr + r * out_row_stride;
            int64_t i = 0;
            for (; i + L <= last_dim; i += L) {
                auto rv = v_log(v_load(in_row + i));
                if (add_to) {
                    v_store(out_row + i, v_add(v_load(out_row + i), rv));
                } else {
                    v_store(out_row + i, rv);
                }
            }
            for (; i < last_dim; ++i) {
                float rv = std::log(s_load(&in_row[i]));
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    // ---- Sin: y = sin(x) ----
    case UnaryType::Sin: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_row_stride;
            T* out_row = out_ptr + r * out_row_stride;
            int64_t i = 0;
            for (; i + L <= last_dim; i += L) {
                auto rv = v_sin(v_load(in_row + i));
                if (add_to) {
                    v_store(out_row + i, v_add(v_load(out_row + i), rv));
                } else {
                    v_store(out_row + i, rv);
                }
            }
            for (; i < last_dim; ++i) {
                float rv = std::sin(s_load(&in_row[i]));
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    // ---- Cos: y = cos(x) ----
    case UnaryType::Cos: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_row_stride;
            T* out_row = out_ptr + r * out_row_stride;
            int64_t i = 0;
            for (; i + L <= last_dim; i += L) {
                auto rv = v_cos(v_load(in_row + i));
                if (add_to) {
                    v_store(out_row + i, v_add(v_load(out_row + i), rv));
                } else {
                    v_store(out_row + i, rv);
                }
            }
            for (; i < last_dim; ++i) {
                float rv = std::cos(s_load(&in_row[i]));
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    // ---- Tan: y = tan(x) ----
    case UnaryType::Tan: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_row_stride;
            T* out_row = out_ptr + r * out_row_stride;
            int64_t i = 0;
            for (; i + L <= last_dim; i += L) {
                auto rv = v_tan(v_load(in_row + i));
                if (add_to) {
                    v_store(out_row + i, v_add(v_load(out_row + i), rv));
                } else {
                    v_store(out_row + i, rv);
                }
            }
            for (; i < last_dim; ++i) {
                float rv = std::tan(s_load(&in_row[i]));
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    // ---- Tanh: y = tanh(x) ----
    case UnaryType::Tanh: {
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

    // ---- Abs: y = |x| ----
    case UnaryType::Abs: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_row_stride;
            T* out_row = out_ptr + r * out_row_stride;
            int64_t i = 0;
            for (; i + L <= last_dim; i += L) {
                auto rv = v_abs(v_load(in_row + i));
                if (add_to) {
                    v_store(out_row + i, v_add(v_load(out_row + i), rv));
                } else {
                    v_store(out_row + i, rv);
                }
            }
            for (; i < last_dim; ++i) {
                float v = s_load(&in_row[i]);
                float rv = (v < 0.0f ? -v : v);
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    // ---- Neg: y = -x ----
    case UnaryType::Neg: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_row_stride;
            T* out_row = out_ptr + r * out_row_stride;
            int64_t i = 0;
            for (; i + L <= last_dim; i += L) {
                auto rv = v_neg(v_load(in_row + i));
                if (add_to) {
                    v_store(out_row + i, v_add(v_load(out_row + i), rv));
                } else {
                    v_store(out_row + i, rv);
                }
            }
            for (; i < last_dim; ++i) {
                float rv = -s_load(&in_row[i]);
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    // ---- Sqrt: y = sqrt(x) ----
    case UnaryType::Sqrt: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_row_stride;
            T* out_row = out_ptr + r * out_row_stride;
            int64_t i = 0;
            for (; i + L <= last_dim; i += L) {
                auto rv = v_sqrt(v_load(in_row + i));
                if (add_to) {
                    v_store(out_row + i, v_add(v_load(out_row + i), rv));
                } else {
                    v_store(out_row + i, rv);
                }
            }
            for (; i < last_dim; ++i) {
                float rv = std::sqrt(s_load(&in_row[i]));
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
// Entry point with dtype dispatch
// ============================================================

void unary_cpu(const UnaryAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx,
                 void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        unary_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        unary_impl<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"unary_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
