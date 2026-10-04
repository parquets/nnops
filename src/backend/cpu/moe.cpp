/// @file moe.cpp
/// @brief CPU Mixture-of-Experts kernel — fused router + per-expert FFN.
///
/// Semantics follow onnxruntime's `com.microsoft::MoE` CPU kernel: the router
/// logits are softmaxed over *all* experts, the top-K are selected (ties toward
/// the lower expert index), optionally renormalized, and then each selected
/// expert's FFN runs and contributes `w * expert_out` to the token's output.
///
/// The work is split in two phases so that neither races:
///
///   A. per expert — gather its routed tokens, FC1 -> activation -> FC2, and
///      write the result to a private per-route slot. Different experts own
///      disjoint slots, so this parallelizes over experts.
///   B. per token — sum the token's K contributions in a fixed order and write
///      the output row. Each token is owned by exactly one worker.
///
/// Accumulating `out[token] += ...` directly in phase A would be a lost update:
/// with top-K > 1 a token belongs to K *different* experts, which run on
/// different workers.
///
/// The per-expert GEMMs reuse the library's tiled `matmul_cpu` with a
/// *default-constructed* ComputeContext, so the inner GEMM stays sequential and
/// does not nest a second parallel region inside phase A.

#include "nnops/ops/moe.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/simd/simd.hpp"

#include "matmul.h"
#include "common/memory_pool.hpp"
#include "simd_kernel/simd_activation.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

// Forward-declare the reference kernel for the unsupported-dtype fallback.
namespace nnops::backend::cpu::reference {
extern void moe_ref(const MoEAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

namespace nnops::backend::cpu {
namespace {

using nnops::backend::cpu::half;

/// Absent optional inputs arrive either as an omitted argument or as an empty
/// TensorView (data == nullptr) — treat both as "not provided".
inline bool has_input(std::span<const TensorView> inputs, size_t i) noexcept
{
    return i < inputs.size() && !inputs[i].is_empty();
}

template <typename T>
inline float ldf(const T* p) noexcept { return simd::s_load(p); }

template <typename T>
inline void stf(T* p, float v) noexcept { simd::s_store(p, v); }

/// SwiGLU gate/value combine, matching moe.hpp's documented formula:
///   h = gate * sigmoid(alpha * gate) * (value + beta)
/// with both halves clamped to +/- limit first when limit is finite.
inline float swiglu_combine(float gate, float value, float alpha, float beta,
                            float limit) noexcept
{
    if (std::isfinite(limit)) {
        gate  = gate  < -limit ? -limit : (gate  > limit ? limit : gate);
        value = value < -limit ? -limit : (value > limit ? limit : value);
    }
    const float sig = 1.0f / (1.0f + std::exp(-alpha * gate));
    return gate * sig * (value + beta);
}

/// Run one `[M,K] x [N,K]^T -> [M,N]` GEMM into @p out, sequentially.
///
/// B is `transpose_b`, i.e. stored `[out_features, in_features]` — the layout the
/// stacked expert weights already use. The output is zeroed first: below
/// GEMM_FAST_PATH_THRESHOLD the f32 path routes to matmul_ref, which evaluates
/// `sum + beta * (*dst)` and therefore *reads* C even when beta == 0 — pooled
/// scratch recycled from another op can hold NaN, and `0 * NaN` would poison the
/// whole tile. The tiled path avoids this via zero_mode, so the zeroing only
/// matters for small experts, which is exactly where it is cheap.
template <typename T>
void expert_gemm(const T* a, int64_t M, int64_t K,
                 const T* b, int64_t N,
                 T* out, DataType dt)
{
    std::memset(out, 0, static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(T));
    if (M == 0 || N == 0 || K == 0) { return; }

    const int64_t a_shape[] = {M, K};
    const int64_t b_shape[] = {N, K};
    const int64_t c_shape[] = {M, N};

    // TensorView takes void*; the GEMM reads B and never writes it.
    TensorView a_view(a_shape, dt, const_cast<T*>(a));
    TensorView b_view(b_shape, dt, const_cast<T*>(b));
    TensorView c_view(c_shape, dt, out);

    MatMulAttributes mm{};
    mm.transpose_b = true;

    const TensorView ins[] = {a_view, b_view};
    const ComputeContext serial_ctx;  // no hooks -> inner GEMM runs sequentially
    matmul_cpu(mm, c_view, ins, serial_ctx, nullptr);
}

/// Add a per-expert bias vector to every row of an [M,N] block, in f32.
template <typename T>
void add_bias_rows(T* block, int64_t M, int64_t N, const T* bias) noexcept
{
    if (bias == nullptr) { return; }
    for (int64_t r = 0; r < M; ++r) {
        T* row = block + r * N;
        for (int64_t n = 0; n < N; ++n) {
            stf(row + n, ldf(row + n) + ldf(bias + n));
        }
    }
}

template <typename T>
void moe_cpu_impl(const MoEAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx)
{
    const auto& input        = inputs[0];
    const auto& router_probs = inputs[1];
    const auto& fc1_w        = inputs[2];
    const auto& fc2_w        = inputs[4];

    const bool has_fc1_bias = has_input(inputs, 3);
    const bool has_fc2_bias = has_input(inputs, 5);
    const bool has_fc3_w    = has_input(inputs, 6);
    const bool has_fc3_bias = has_input(inputs, 7);
    const bool has_rweights = has_input(inputs, 8);

    const DataType dt = input.data_type();

    const int64_t H = input.shape(input.rank() - 1);
    const int64_t num_tokens = input.numel() / H;
    const int64_t E = router_probs.shape(1);
    const int64_t I = fc2_w.shape(2);
    const int64_t fc1_out = fc1_w.shape(1);
    const int64_t k = attrs.k;

    const bool is_swiglu = (attrs.activation == MoEActivation::SwiGLU);

    NNOPS_ASSERT(!attrs.use_sparse_mixer);
    NNOPS_ASSERT(k >= 1 && k <= E);
    NNOPS_ASSERT(router_probs.shape(0) == num_tokens);
    NNOPS_ASSERT(fc1_w.shape(0) == E && fc2_w.shape(0) == E);
    NNOPS_ASSERT(fc2_w.shape(1) == H);
    if (is_swiglu) {
        // Only the Separate layout is implemented: the value half comes from
        // fc3, so fc1 is the gate alone (fc1_out == I) and fc3 is required.
        // Interleaved/Block would pack gate+value into fc1 (fc1_out == 2*I).
        NNOPS_ASSERT_MSG(attrs.swiglu_layout == SwiGLULayout::Separate,
                         "moe_cpu: only the Separate SwiGLU layout is supported");
        NNOPS_ASSERT(has_fc3_w);
        NNOPS_ASSERT(fc1_w.shape(1) == I);
        NNOPS_ASSERT(inputs[6].shape(0) == E && inputs[6].shape(1) == I &&
                     inputs[6].shape(2) == H);
    } else {
        NNOPS_ASSERT(fc1_out == I);
    }

    const T* in_ptr      = input.ptr<T>();
    const T* rout_ptr    = router_probs.ptr<T>();
    const T* fc1_ptr     = fc1_w.ptr<T>();
    const T* fc2_ptr     = fc2_w.ptr<T>();
    const T* fc1b_ptr    = has_fc1_bias ? inputs[3].ptr<T>() : nullptr;
    const T* fc2b_ptr    = has_fc2_bias ? inputs[5].ptr<T>() : nullptr;
    const T* fc3_ptr     = has_fc3_w ? inputs[6].ptr<T>() : nullptr;
    const T* fc3b_ptr    = has_fc3_bias ? inputs[7].ptr<T>() : nullptr;
    const T* rw_ptr      = has_rweights ? inputs[8].ptr<T>() : nullptr;

    const int64_t in_row_stride   = input.row_stride_elems();
    const int64_t rout_row_stride = router_probs.row_stride_elems();
    const int64_t rw_row_stride   = has_rweights ? inputs[8].row_stride_elems() : 0;
    const int64_t out_row_stride  = output.row_stride_elems();

    const int64_t total = k * num_tokens;  // number of (token, expert) routes

    // fc1_out only ever appears in the guards above. Below, FC1 is driven with
    // fc1_out = I: the gate buffer is laid out as [n_e, I], so a Release build
    // (where NNOPS_ASSERT compiles away) fed fc1_out != I would silently
    // mis-stride the activation and FC2 rather than fail. That mismatch is
    // outside the op's contract — no layout in moe.hpp produces it.
    (void)fc1_out;

    // ------------------------------------------------------------------
    // Routing: softmax over all experts, then top-K. Serial on purpose —
    // O(T*E) is negligible next to the GEMMs, and a fixed order keeps the
    // tie-break and the weights bit-reproducible.
    // ------------------------------------------------------------------
    std::vector<int32_t> route_expert(static_cast<size_t>(total));
    std::vector<float>   route_weight(static_cast<size_t>(total));
    {
        std::vector<float> probs(static_cast<size_t>(E));
        std::vector<char>  used(static_cast<size_t>(E));

        for (int64_t t = 0; t < num_tokens; ++t) {
            const T* logits = rout_ptr + t * rout_row_stride;

            float max_logit = ldf(logits);
            for (int64_t e = 1; e < E; ++e) {
                const float v = ldf(logits + e);
                max_logit = v > max_logit ? v : max_logit;
            }
            float sum_exp = 0.0f;
            for (int64_t e = 0; e < E; ++e) {
                probs[static_cast<size_t>(e)] = std::exp(ldf(logits + e) - max_logit);
                sum_exp += probs[static_cast<size_t>(e)];
            }
            const float inv_sum = 1.0f / sum_exp;
            for (int64_t e = 0; e < E; ++e) {
                probs[static_cast<size_t>(e)] *= inv_sum;
            }

            std::fill(used.begin(), used.end(), 0);
            for (int64_t j = 0; j < k; ++j) {
                // Scan ascending with a strict '>', so an exact tie keeps the
                // lower expert index — matches reference::moe_ref.
                int64_t best = -1;
                for (int64_t e = 0; e < E; ++e) {
                    if (used[static_cast<size_t>(e)]) { continue; }
                    if (best < 0 ||
                        probs[static_cast<size_t>(e)] > probs[static_cast<size_t>(best)]) {
                        best = e;
                    }
                }
                used[static_cast<size_t>(best)] = 1;
                const int64_t idx = t * k + j;
                route_expert[static_cast<size_t>(idx)] = static_cast<int32_t>(best);
                route_weight[static_cast<size_t>(idx)] = has_rweights
                    ? ldf(rw_ptr + t * rw_row_stride + best)
                    : probs[static_cast<size_t>(best)];
            }

            if (attrs.normalize_routing_weights) {
                float wsum = 0.0f;
                for (int64_t j = 0; j < k; ++j) {
                    wsum += route_weight[static_cast<size_t>(t * k + j)];
                }
                const float inv_wsum = (wsum != 0.0f) ? 1.0f / wsum : 0.0f;
                for (int64_t j = 0; j < k; ++j) {
                    route_weight[static_cast<size_t>(t * k + j)] *= inv_wsum;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Group the routes by expert: counts -> exclusive prefix sum -> slots.
    // Routes of one expert end up contiguous, and `route_row[s]` maps a slot
    // back to the token it came from.
    // ------------------------------------------------------------------
    std::vector<int32_t> expert_off(static_cast<size_t>(E) + 1, 0);
    for (int64_t r = 0; r < total; ++r) {
        ++expert_off[static_cast<size_t>(route_expert[static_cast<size_t>(r)]) + 1];
    }
    for (int64_t e = 0; e < E; ++e) {
        expert_off[static_cast<size_t>(e) + 1] += expert_off[static_cast<size_t>(e)];
    }

    std::vector<int32_t> slot(static_cast<size_t>(total));
    std::vector<int32_t> route_row(static_cast<size_t>(total));
    {
        std::vector<int32_t> cursor(expert_off.begin(), expert_off.end() - 1);
        for (int64_t t = 0; t < num_tokens; ++t) {
            for (int64_t j = 0; j < k; ++j) {
                const int64_t idx = t * k + j;
                const int64_t e = route_expert[static_cast<size_t>(idx)];
                const int32_t s = cursor[static_cast<size_t>(e)]++;
                slot[static_cast<size_t>(idx)] = s;
                route_row[static_cast<size_t>(s)] = static_cast<int32_t>(t);
            }
        }
    }

    auto& pool = MemoryPool::instance();
    PoolPtr gather_owner(static_cast<size_t>(total) * static_cast<size_t>(H) * sizeof(T));
    PoolPtr eout_owner(static_cast<size_t>(total) * static_cast<size_t>(H) * sizeof(T));
    T* gather = gather_owner.as<T>();
    T* eout   = eout_owner.as<T>();

    // Gather the routed token rows into contiguous [total, H] — every slot is
    // written by exactly one loop iteration, so this parallelizes cleanly.
    const size_t row_bytes = static_cast<size_t>(H) * sizeof(T);
    ctx.cpu.run(0, total, [&](int64_t s) {
        std::memcpy(gather + s * H,
                    in_ptr + static_cast<size_t>(route_row[static_cast<size_t>(s)]) * in_row_stride,
                    row_bytes);
    });

    // ------------------------------------------------------------------
    // Phase A: one expert per task. Slots are disjoint, so no synchronization
    // is needed — but note the output tensor is *not* touched here.
    // ------------------------------------------------------------------
    ctx.cpu.run(0, E, [&](int64_t e) {
        const int64_t begin = expert_off[static_cast<size_t>(e)];
        const int64_t n_e = expert_off[static_cast<size_t>(e) + 1] - begin;
        if (n_e == 0) { return; }

        const T* a = gather + begin * H;
        T* out_slice = eout + begin * H;

        PoolPtr gate_owner(static_cast<size_t>(n_e) * static_cast<size_t>(I) * sizeof(T));
        T* gate = gate_owner.as<T>();

        // FC1: [n_e, H] x [I, H]^T -> [n_e, I] (the gate half).
        expert_gemm<T>(a, n_e, H, fc1_ptr + e * I * H, I, gate, dt);
        add_bias_rows<T>(gate, n_e, I, fc1b_ptr ? fc1b_ptr + e * I : nullptr);

        if (is_swiglu) {
            PoolPtr value_owner(static_cast<size_t>(n_e) * static_cast<size_t>(I) * sizeof(T));
            T* value = value_owner.as<T>();

            // FC3: the value half, same shape.
            expert_gemm<T>(a, n_e, H, fc3_ptr + e * I * H, I, value, dt);
            add_bias_rows<T>(value, n_e, I, fc3b_ptr ? fc3b_ptr + e * I : nullptr);

            const int64_t n_elems = n_e * I;
            for (int64_t i = 0; i < n_elems; ++i) {
                stf(gate + i, swiglu_combine(ldf(gate + i), ldf(value + i),
                                             attrs.activation_alpha, attrs.activation_beta,
                                             attrs.swiglu_limit));
            }
        } else {
            // Same activations as the reference (GELU is the tanh approximation
            // in both). Identity is a no-op, so skip the pass entirely.
            using namespace nnops::kernel;
            switch (attrs.activation) {
            case MoEActivation::Relu: relu<T>(gate, gate, n_e, I, I, I, false); break;
            case MoEActivation::Gelu: gelu<T>(gate, gate, n_e, I, I, I, false); break;
            case MoEActivation::Silu: silu<T>(gate, gate, n_e, I, I, I, false); break;
            case MoEActivation::Identity: break;
            default:
                NNOPS_ASSERT(!"moe_cpu: unsupported activation");
                break;
            }
        }

        // FC2: [n_e, I] x [H, I]^T -> [n_e, H], straight into the expert's slots.
        expert_gemm<T>(gate, n_e, I, fc2_ptr + e * H * I, H, out_slice, dt);
        add_bias_rows<T>(out_slice, n_e, H, fc2b_ptr ? fc2b_ptr + e * H : nullptr);
    });

    // ------------------------------------------------------------------
    // Phase B: one token per task. Sum its K contributions in a fixed order
    // (j ascending) and write the row. f32 accumulation: for f16 this is the
    // only place precision is under our control, since the per-expert GEMMs
    // accumulate in f16.
    // ------------------------------------------------------------------
    ctx.cpu.run(0, num_tokens, [&](int64_t t) {
        T* out_row = output.ptr<T>() + t * out_row_stride;
        for (int64_t n = 0; n < H; ++n) {
            float acc = 0.0f;
            for (int64_t j = 0; j < k; ++j) {
                const int64_t idx = t * k + j;
                const float w = route_weight[static_cast<size_t>(idx)];
                const T* er = eout + static_cast<size_t>(slot[static_cast<size_t>(idx)]) * H;
                acc += w * ldf(er + n);
            }
            stf(out_row + n, acc);
        }
    });
}

}  // anonymous namespace

void moe_cpu(const MoEAttributes& attrs,
             TensorView& output,
             std::span<const TensorView> inputs,
             const ComputeContext& ctx,
             void* workspace)
{
    switch (inputs[0].data_type()) {
    case DataType::f32: moe_cpu_impl<float>(attrs, output, inputs, ctx); return;
    case DataType::f16: moe_cpu_impl<half>(attrs, output, inputs, ctx);  return;
    default: break;
    }
    // Only f32/f16 have kernels; everything else falls back to the reference
    // (which asserts in debug and is the correctness baseline in release).
    reference::moe_ref(attrs, output, inputs, ctx, workspace);
}

}  // namespace nnops::backend::cpu
