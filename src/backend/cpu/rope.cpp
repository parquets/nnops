/// @file rope.cpp
/// @brief SIMD-optimized CPU implementation of RoPE (Rotary Position Embedding).
///
/// Pre-computes cos/sin tables for all positions (shared across heads and
/// batch), then applies the rotation row-by-row using SIMD-accelerated
/// kernels from simd_kernel/simd_rope.hpp.
///
/// Supports f32 and f16 via dtype dispatch.
///
/// mRoPE mode: when mrope_section_dims is non-empty, head_dim is partitioned
/// into sections, each with its own base frequency.

#include "nnops/ops/rope.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "simd_kernel/simd_rope.hpp"
#include "common/dtype_dispatch.hpp"

#include <cmath>
#include <vector>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

// ============================================================
// Cos/Sin table computation (float, shared across heads/batch)
// ============================================================

struct RoPESection {
    int64_t dim_start;   // inclusive
    int64_t dim_end;     // exclusive (must be even)
    float   base;
};

/// Build section table.
std::vector<RoPESection> build_sections(const RoPEAttributes& attrs, int64_t head_dim) {
    std::vector<RoPESection> secs;
    if (attrs.mrope_section_dims.empty()) {
        secs.push_back({0, head_dim, attrs.base});
    } else {
        int64_t off = 0;
        for (size_t i = 0; i < attrs.mrope_section_dims.size(); ++i) {
            float b = (i < attrs.mrope_section_bases.size())
                ? attrs.mrope_section_bases[i] : attrs.base;
            secs.push_back({off, off + attrs.mrope_section_dims[i], b});
            off += attrs.mrope_section_dims[i];
        }
    }
    return secs;
}

/// Pre-compute cos and sin tables for all positions.
/// Layout: table[pos * half + pair_idx] where half = head_dim / 2.
/// For interleaved:    pair_idx = d/2      (pair (2*pair, 2*pair+1))
/// For split-half:     pair_idx = d        (pair (d, d+half))
void compute_cos_sin_tables(
    std::vector<float>& cos_table,
    std::vector<float>& sin_table,
    int64_t seq_len,
    int64_t head_dim,
    const std::vector<RoPESection>& sections)
{
    const int64_t half = head_dim / 2;
    const size_t  total = static_cast<size_t>(seq_len) * static_cast<size_t>(half);
    cos_table.resize(total);
    sin_table.resize(total);

    for (int64_t pos = 0; pos < seq_len; ++pos) {
        const float pos_f = static_cast<float>(pos);
        float* cos_row = cos_table.data() + pos * half;
        float* sin_row = sin_table.data() + pos * half;

        int64_t pair_idx = 0;
        for (const auto& sec : sections) {
            int64_t sec_len    = sec.dim_end - sec.dim_start;
            int64_t sec_pairs  = sec_len / 2;
            float   sec_dim_f  = static_cast<float>(sec_len);

            for (int64_t p = 0; p < sec_pairs; ++p) {
                float theta = pos_f / std::pow(sec.base,
                    2.0f * static_cast<float>(p) / sec_dim_f);
                cos_row[pair_idx] = std::cos(theta);
                sin_row[pair_idx] = std::sin(theta);
                ++pair_idx;
            }
        }
    }
}

}  // anonymous namespace

// ============================================================
// Templated implementation (f32 and f16)
// ============================================================

template <typename T>
void rope_impl(const RoPEAttributes& attrs,
               TensorView& output,
               std::span<const TensorView> inputs,
               const ComputeContext& ctx)
{
    const auto& X = inputs[0];
    const int64_t rank = X.rank();
    NNOPS_ASSERT(rank >= 2);

    const int64_t seq_len = X.shape(rank - 2);
    const int64_t head_dim = X.shape(rank - 1);
    NNOPS_ASSERT(head_dim % 2 == 0);

    const int64_t half = head_dim / 2;

    // Flatten leading dims
    int64_t outer_count = 1;
    for (int64_t i = 0; i < rank - 2; ++i) {
        outer_count *= X.shape(i);
    }

    // ---- Build sections and pre-compute cos/sin tables ----
    auto sections = build_sections(attrs, head_dim);
    std::vector<float> cos_table;
    std::vector<float> sin_table;
    compute_cos_sin_tables(cos_table, sin_table, seq_len, head_dim, sections);

    // ---- Data pointers ----
    const T* x_ptr   = X.ptr<T>();
    T*       y_ptr   = output.ptr<T>();
    const int64_t row_stride  = X.row_stride_elems();
    const int64_t out_row_stride = output.row_stride_elems();
    const bool    interleaved = attrs.interleaved;

    // ---- Row-by-row rotation ----
    const auto process_outer = [&](int64_t outer) {
        const T* x_outer = x_ptr + outer * seq_len * row_stride;
        T*       y_outer = y_ptr + outer * seq_len * out_row_stride;

        for (int64_t pos = 0; pos < seq_len; ++pos) {
            const T* x_row = x_outer + pos * row_stride;
            T*       y_row = y_outer + pos * out_row_stride;
            const float* cos_row = cos_table.data() + pos * half;
            const float* sin_row = sin_table.data() + pos * half;

            kernel::rope_process_row<T>(
                y_row, x_row, head_dim, cos_row, sin_row,
                interleaved);
        }
    };

    ctx.cpu.run(0, outer_count, process_outer);
}

// ============================================================
// Entry point with dtype dispatch
// ============================================================

void rope_cpu(const RoPEAttributes& attrs,
              TensorView& output,
              std::span<const TensorView> inputs,
              const ComputeContext& ctx,
              void* /*workspace*/)
{
    dispatch_f32_f16(inputs[0].data_type(), "rope_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        rope_impl<T>(attrs, output, inputs, ctx);
    });
}

}  // namespace nnops::backend::cpu
