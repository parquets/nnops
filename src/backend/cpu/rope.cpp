/// @file rope.cpp
/// @brief SIMD-optimized CPU implementation of RoPE (Rotary Position Embedding).
///
/// Builds cos/sin tables once for all positions (shared across heads and
/// batches), then rotates row by row with the kernels in
/// simd_kernel/simd_rope.hpp. f32 and f16 via dtype dispatch.
///
/// Table layout: table[pos * half + pair_idx], half = rotary_dim/2. Interleaved:
/// pair_idx = d/2, the pair is (2p, 2p+1). Split-half: pair_idx = d, the pair
/// is (d, d+half). Lanes past rotary_dim are copied through unrotated.
///
/// mRoPE: when mrope_section_dims is non-empty, rotary_dim is partitioned into
/// sections, each with its own base frequency.
///
/// Fused pre-norm: an optional scale/bias pair (inputs [1]/[2]) RMS-normalizes
/// each whole head before the rotation, writing into the output so the rotate
/// pass runs in place on it.
///
/// Position offset: row s is at absolute position position_offset + s.
///
/// A caller-supplied cos/sin pair (inputs [3]/[4]) replaces the table build
/// outright — no section planning, no scaling — and is read in row order. The
/// pair may be f32 or the activation dtype; the latter is widened into an f32
/// copy once per call.
///
/// Scaling: Linear divides the angle by the factor; Dynamic NTK grows the
/// section's base with the run's end position past original_max_position; YaRN
/// mixes interpolated and extrapolated angles over a correction ramp and
/// multiplies cos/sin by the attention magnitude; Llama3 splits on the pair's
/// wavelength, scaling only the long end and blending the band between. All
/// four follow the HuggingFace rope_scaling formulas.

#include "backend/common/rope_plan.hpp"
#include "nnops/ops/rope.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "simd_kernel/simd_norm.hpp"
#include "simd_kernel/simd_rope.hpp"
#include "common/dtype_dispatch.hpp"
#include "common/memory_pool.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

void compute_cos_sin_tables(PoolArray<float>& cos_table, PoolArray<float>& sin_table,
                            int64_t blocks, int64_t seq_len, int64_t rotary_dim,
                            const int64_t* starts, const RoPEAttributes& attrs,
                            const ComputeContext& ctx)
{
    const int64_t half = rotary_dim / 2;
    const int64_t rows = blocks * seq_len;
    const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(half);
    cos_table.reset(total);
    sin_table.reset(total);

    // One block of seq_len consecutive positions per batch (or one block when
    // every batch shares a start). Dynamic NTK pivots on the furthest end.
    int64_t max_start = starts[0];
    for (int64_t blk = 1; blk < blocks; ++blk) {
        max_start = std::max(max_start, starts[blk]);
    }
    const auto plans = plan_rope_sections(attrs, rotary_dim, seq_len, max_start);
    const auto pairs = build_rope_pair_plans(plans, attrs, half);

    const bool linear = attrs.scaling_type == RoPEScalingType::Linear;
    const bool yarn = attrs.scaling_type == RoPEScalingType::YaRN;
    const bool llama3 = attrs.scaling_type == RoPEScalingType::Llama3;
    const float factor = attrs.scaling_factor;
    const float magnitude = yarn ? yarn_attention_scale(attrs) : 1.0f;

    // One row per (batch, position), and each row is independent — so the
    // table builds across the pool like the rotation pass does.
    ctx.cpu.run(0, rows, [&](int64_t i) {
        const int64_t blk = i / seq_len;
        const int64_t r = i % seq_len;
        const float pos_f = static_cast<float>(starts[blk] + r);
        float* cos_row = cos_table.data() + i * half;
        float* sin_row = sin_table.data() + i * half;

        for (int64_t p = 0; p < half; ++p) {
            const RopePairPlan& pp = pairs[static_cast<size_t>(p)];
            const float extrap = pos_f / pp.ratio;

            float theta = extrap;
            if (linear) {
                theta = extrap / factor;
            } else if (yarn) {
                const float interp = extrap / factor;
                theta = interp * pp.ramp + extrap * (1.0f - pp.ramp);
            } else if (llama3) {
                // Short wavelengths are untouched, long ones divide by the
                // factor, the band between them blends the two.
                if (pp.long_wl) {
                    theta = extrap / factor;
                } else if (pp.smooth >= 0.0f) {
                    theta = extrap * ((1.0f - pp.smooth) / factor + pp.smooth);
                }
            }

            cos_row[p] = std::cos(theta) * magnitude;
            sin_row[p] = std::sin(theta) * magnitude;
        }
    });
}

} // anonymous namespace

template <typename T>
void rope_impl(const RoPEAttributes& attrs, TensorView& output, std::span<const TensorView> inputs,
               const ComputeContext& ctx)
{
    const auto& X = inputs[0];
    const int64_t rank = X.rank();
    NNOPS_ASSERT(rank >= 2);

    const int64_t seq_len = X.shape(rank - 2);
    const int64_t head_dim = X.shape(rank - 1);
    NNOPS_ASSERT(head_dim % 2 == 0);

    const int64_t rotary_dim = attrs.rotary_dim > 0 ? attrs.rotary_dim : head_dim;
    NNOPS_ASSERT(rotary_dim >= 2 && rotary_dim % 2 == 0 && rotary_dim <= head_dim);

    const int64_t half = rotary_dim / 2;

    int64_t outer_count = 1;
    for (int64_t i = 0; i < rank - 2; ++i) {
        outer_count *= X.shape(i);
    }

    // A supplied cos/sin pair replaces the angle computation: its rows are
    // already the absolute positions the caller wants, so position_offset,
    // position_ids, the bases, the mRoPE sections and the scaling types do not
    // apply. A table in the activation dtype is widened once into f32, the same
    // layout the computed tables use.
    const bool has_table = inputs.size() > 3 && !inputs[3].is_empty();

    // Where each batch's row 0 sits: the per-batch int64 vector [5] (axis 0 of
    // x is the batch axis) or the scalar attribute. Equal starts share one
    // block of table rows.
    const int64_t batch = (rank >= 3) ? X.shape(0) : 1;
    std::vector<int64_t> starts;
    int64_t blocks = 1;
    if (!has_table) {
        const bool has_pos = inputs.size() > 5 && !inputs[5].is_empty();
        if (has_pos) {
            const int64_t* ids = inputs[5].ptr<int64_t>();
            if (inputs[5].numel() == 1) {
                starts.push_back(ids[0]);
            } else {
                starts.assign(ids, ids + batch);
                bool uniform = true;
                for (int64_t b = 1; b < batch; ++b) {
                    uniform = uniform && starts[static_cast<size_t>(b)] == starts[0];
                }
                if (uniform) {
                    starts.resize(1);
                } else {
                    blocks = batch;
                }
            }
        } else {
            starts.push_back(attrs.position_offset);
        }
    }

    const float* cos_src = nullptr;
    const float* sin_src = nullptr;
    int64_t tbl_stride = half;
    PoolArray<float> cos_table;
    PoolArray<float> sin_table;

    if (has_table) {
        const auto& ct = inputs[3];
        const auto& st = inputs[4];
        if (ct.data_type() == DataType::f32) {
            cos_src = ct.ptr<float>();
            sin_src = st.ptr<float>();
            tbl_stride = ct.row_stride_elems();
        } else {
            const size_t total = static_cast<size_t>(seq_len) * static_cast<size_t>(half);
            cos_table.reset(total);
            sin_table.reset(total);
            const T* cs = ct.ptr<T>();
            const T* ss = st.ptr<T>();
            const int64_t cs_stride = ct.row_stride_elems();
            const int64_t ss_stride = st.row_stride_elems();
            for (int64_t i = 0; i < seq_len; ++i) {
                for (int64_t p = 0; p < half; ++p) {
                    cos_table[static_cast<size_t>(i * half + p)] =
                        simd::s_load(cs + i * cs_stride + p);
                    sin_table[static_cast<size_t>(i * half + p)] =
                        simd::s_load(ss + i * ss_stride + p);
                }
            }
            cos_src = cos_table.data();
            sin_src = sin_table.data();
        }
    } else {
        compute_cos_sin_tables(cos_table, sin_table, blocks, seq_len, rotary_dim, starts.data(),
                               attrs, ctx);
        cos_src = cos_table.data();
        sin_src = sin_table.data();
    }

    const T* x_ptr = X.ptr<T>();
    T* y_ptr = output.ptr<T>();
    const int64_t row_stride = X.row_stride_elems();
    const int64_t out_row_stride = output.row_stride_elems();
    const bool interleaved = attrs.interleaved;

    const T* gamma = inputs.size() > 1 && !inputs[1].is_empty() ? inputs[1].ptr<T>() : nullptr;
    const T* beta = inputs.size() > 2 && !inputs[2].is_empty() ? inputs[2].ptr<T>() : nullptr;
    const float norm_eps = attrs.norm_epsilon;

    // Rows are independent, so the pool is filled across (head, row tile):
    // a [1, 8, S, D] tensor has only 8 heads, which left 12 of 20 threads idle
    // when one task was a whole head.
    constexpr int64_t kRowsPerTask = 32;
    const int64_t pos_tiles = (seq_len + kRowsPerTask - 1) / kRowsPerTask;

    const auto process_tile = [&](int64_t task) {
        const int64_t outer = task / pos_tiles;
        const int64_t pos0 = (task % pos_tiles) * kRowsPerTask;
        const int64_t pos1 = std::min(pos0 + kRowsPerTask, seq_len);
        // Table rows are batch-major: block b starts at b * seq_len.
        const int64_t tbl_base =
            (blocks > 1) ? (outer / (outer_count / batch)) * seq_len : 0;

        const T* x_outer = x_ptr + outer * seq_len * row_stride;
        T* y_outer = y_ptr + outer * seq_len * out_row_stride;

        for (int64_t pos = pos0; pos < pos1; ++pos) {
            const T* x_row = x_outer + pos * row_stride;
            T* y_row = y_outer + pos * out_row_stride;
            const float* cos_row = cos_src + (tbl_base + pos) * tbl_stride;
            const float* sin_row = sin_src + (tbl_base + pos) * tbl_stride;

            if (gamma) {
                // Normalize the whole head into the output, then rotate the
                // prefix in place — one fewer tensor pass than norm+rope.
                auto [sum, sum_sq] = kernel::norm_reduce_sum_sq<T>(x_row, head_dim);
                (void)sum;
                const float inv_rms =
                    1.0f / std::sqrt(sum_sq / static_cast<float>(head_dim) + norm_eps);
                if (beta) {
                    kernel::norm_apply_row<T, false, true, true>(
                        x_row, y_row, head_dim, 0.0f, inv_rms, gamma, beta, false, false);
                } else {
                    kernel::norm_apply_row<T, false, true, false>(
                        x_row, y_row, head_dim, 0.0f, inv_rms, gamma, nullptr, false, false);
                }
                kernel::rope_process_row<T>(y_row, y_row, head_dim, rotary_dim, cos_row, sin_row,
                                            interleaved);
            } else {
                kernel::rope_process_row<T>(y_row, x_row, head_dim, rotary_dim, cos_row, sin_row,
                                            interleaved);
            }
        }
    };

    ctx.cpu.run(0, outer_count * pos_tiles, process_tile);
}

Status rope_cpu(const RoPEAttributes& attrs, TensorView& output, std::span<const TensorView> inputs,
                const ComputeContext& ctx, void* /*workspace*/)
{
    if (!dispatch_f32_f16(inputs[0].data_type(), "rope_cpu", [&](auto tag) {
            using T = typename decltype(tag)::type;
            rope_impl<T>(attrs, output, inputs, ctx);
        })) {
        return Status::unsupported;
    }
    return Status::ok;
}

} // namespace nnops::backend::cpu
