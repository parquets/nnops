/// @file moe.cpp
/// @brief CPU Mixture-of-Experts — fused router + per-expert FFN.
///
/// Semantics follow onnxruntime's `com.microsoft::MoE`: the logits are scored
/// over *all* experts, top-K are selected (ties toward the lower expert index),
/// optionally renormalized and scaled, then each selected expert's FFN
/// contributes `w * expert_out` to the token's output. `attrs.router_gating`
/// picks the scoring function — softmax by default (what ORT does), or sigmoid
/// / sqrt-softplus following llama.cpp's `expert_gating_func`; all three are
/// monotonic in the logits, so the selected experts are the same and only the
/// mixing weights change.
///
/// Two phases, so that neither races:
///   A. per (expert, output-channel block) — FC1 -> activation -> FC2 over an
///      expert's routed tokens, into that expert's private per-route slots. The
///      slots are disjoint, so this never touches the output tensor, and the N
///      split is what keeps the pool busy when there are fewer experts than
///      workers. It runs as two dispatches: FC1 writes the gate with its bias
///      and activation folded into the GEMM's output pass, then FC2 reads the
///      whole gate row and so needs a barrier.
///   B. per token — sum the token's K contributions in a fixed order and write
///      the output row; each token has exactly one owner.
/// A direct `out[token] += ...` in phase A would be a lost update: with
/// top-K > 1 a token belongs to K *different* experts, which run on different
/// workers.
///
/// The inner GEMMs use a *default-constructed* ComputeContext: phase A is
/// already spread over the pool, and a nested region would either
/// oversubscribe the machine or serialize inside the outer team.

#include "nnops/ops/moe.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/simd/simd.hpp"

#include "backend/cpu/matmul/matmul.h"
#include "common/memory_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace nnops::backend::cpu {
namespace {

using nnops::backend::cpu::half;

/// Optional inputs may be omitted or passed empty; either means "not provided".
inline bool has_input(std::span<const TensorView> inputs, size_t i) noexcept
{
    return i < inputs.size() && !inputs[i].is_empty();
}

template <typename T>
inline float ldf(const T* p) noexcept { return simd::s_load(p); }

template <typename T>
inline void stf(T* p, float v) noexcept { simd::s_store(p, v); }

/// Elementwise router scores for the gatings that do not need the whole row
/// (softmax is normalized by the caller). Sigmoid saturates on its own;
/// softplus needs the guard at 20 that ggml uses — beyond it log1p(exp(x))
/// already equals x in f32, and exp(x) overflows f32 near 88.
inline float gating_sigmoid(float x) noexcept
{
    return 1.0f / (1.0f + std::exp(-x));
}

inline float gating_sqrt_softplus(float x) noexcept
{
    return std::sqrt(x > 20.0f ? x : std::log1p(std::exp(x)));
}

/// The expert activation as a MatMul epilogue. GELU is the same tanh
/// approximation on both sides; SwiGLU has no epilogue form because it needs
/// the value half too, and the combine runs as its own pass.
inline EpilogueActivateType epilogue_of(MoEActivation act) noexcept
{
    switch (act) {
    case MoEActivation::Relu: return EpilogueActivateType::Relu;
    case MoEActivation::Gelu: return EpilogueActivateType::Gelu;
    case MoEActivation::Silu: return EpilogueActivateType::Silu;
    case MoEActivation::Identity:
    case MoEActivation::SwiGLU: break;
    }
    return EpilogueActivateType::None;
}

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

/// Run one `[M,K] x [N,K]^T -> [M,N]` GEMM into @p out, whose rows are @p ld
/// apart. B is `transpose_b` — the stacked expert weights are already
/// `[out, in]`, so an output-channel slice is a contiguous run of rows, and
/// @p bias / @p ep are that same slice's bias and activation. Both ride the
/// kernel's own output pass — the last k-block's tile, still in cache — so the
/// gate is written once, already final, instead of being swept twice more.
///
/// @p b is raw weight storage of @p bdt, which may be block-packed: the caller
/// offsets it by whole rows either way, and MatMul is what knows the format.
template <typename T>
void expert_gemm(const T* a, int64_t M, int64_t K,
                 const void* b, int64_t N, DataType bdt,
                 const T* bias, const Epilogue& ep,
                 T* out, int64_t ld,
                 DataType dt, const ComputeContext& ctx)
{
    if (M == 0 || N == 0 || K == 0) { return; }

    const int64_t a_shape[] = {M, K};
    const int64_t b_shape[] = {N, K};
    const int64_t c_shape[] = {M, N};

    TensorView a_view(a_shape, dt, const_cast<T*>(a));
    TensorView b_view(b_shape, bdt, const_cast<void*>(b));
    // beta = 0: the GEMM overwrites its slice, so no clear pass is needed.
    TensorView c_view(c_shape, dt, out, ld * static_cast<int64_t>(sizeof(T)));

    MatMulAttributes mm{};
    mm.transpose_b = true;
    mm.epilogue = ep;

    TensorView ins[3] = {a_view, b_view, TensorView{}};
    if (bias != nullptr) {
        const int64_t bias_shape[] = {N};
        ins[2] = TensorView(bias_shape, dt, const_cast<T*>(bias));
    }
    matmul_cpu(mm, c_view, std::span<const TensorView>(ins, bias != nullptr ? 3 : 2),
               ctx, nullptr);
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
        // Only Separate is implemented: fc1 is the gate alone and fc3 holds the
        // value; the fused layouts pack both into fc1 (fc1_out == 2 * I).
        NNOPS_ASSERT_MSG(attrs.swiglu_layout == SwiGLULayout::Separate,
                         "moe_cpu: only the Separate SwiGLU layout is supported");
        NNOPS_ASSERT(has_fc3_w);
        NNOPS_ASSERT(fc1_w.shape(1) == I);
        NNOPS_ASSERT(inputs[6].shape(0) == E && inputs[6].shape(1) == I &&
                     inputs[6].shape(2) == H);
    } else {
        NNOPS_ASSERT(fc1_out == I);
    }

    // The weights may be block-packed, so they are addressed as bytes: one row
    // is H (or I) values, which is `pitch()` bytes — 20 per 32 values for Q4_1.
    const DataType fc1_dt = fc1_w.data_type();
    const DataType fc2_dt = fc2_w.data_type();
    const DataType fc3_dt = has_fc3_w ? inputs[6].data_type() : fc1_dt;
    const auto* fc1_base = static_cast<const uint8_t*>(fc1_w.ptr<void>());
    const auto* fc2_base = static_cast<const uint8_t*>(fc2_w.ptr<void>());
    const auto* fc3_base = has_fc3_w ? static_cast<const uint8_t*>(inputs[6].ptr<void>())
                                     : nullptr;
    const int64_t fc1_row_bytes = fc1_w.pitch();
    const int64_t fc2_row_bytes = fc2_w.pitch();
    const int64_t fc3_row_bytes = has_fc3_w ? inputs[6].pitch() : 0;

    const T* in_ptr      = input.ptr<T>();
    const T* rout_ptr    = router_probs.ptr<T>();
    const T* fc1b_ptr    = has_fc1_bias ? inputs[3].ptr<T>() : nullptr;
    const T* fc2b_ptr    = has_fc2_bias ? inputs[5].ptr<T>() : nullptr;
    const T* fc3b_ptr    = has_fc3_bias ? inputs[7].ptr<T>() : nullptr;
    const T* rw_ptr      = has_rweights ? inputs[8].ptr<T>() : nullptr;

    const int64_t in_row_stride   = input.row_stride_elems();
    const int64_t rout_row_stride = router_probs.row_stride_elems();
    const int64_t rw_row_stride   = has_rweights ? inputs[8].row_stride_elems() : 0;
    const int64_t out_row_stride  = output.row_stride_elems();

    const int64_t total = k * num_tokens;  // number of (token, expert) routes

    // fc1_out is used only by the guards above; FC1 below is driven with
    // fc1_out = I. A Release build (NNOPS_ASSERT compiled out) fed fc1_out != I
    // would silently mis-stride the gate buffer and FC2 rather than fail — no
    // layout in moe.hpp produces that.
    (void)fc1_out;

    // Routing, serial on purpose: O(T*E) is negligible next to the GEMMs, and
    // the fixed order keeps the tie-break and the weights bit-reproducible.
    // Anything that finalizes route_weight must happen here — phase B reads it
    // from worker threads.
    PoolArray<int32_t> route_expert(static_cast<size_t>(total));
    PoolArray<float>   route_weight(static_cast<size_t>(total));
    {
        PoolArray<float> probs(static_cast<size_t>(E));
        PoolArray<char>  used(static_cast<size_t>(E));

        for (int64_t t = 0; t < num_tokens; ++t) {
            const T* logits = rout_ptr + t * rout_row_stride;

            switch (attrs.router_gating) {
            case MoERouterGating::Softmax: {
                // Max-subtracted for stability; softmax covers the full row.
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
            } break;
            case MoERouterGating::Sigmoid:
                // Not max-subtracted: sigmoid is not shift-invariant.
                for (int64_t e = 0; e < E; ++e) {
                    probs[static_cast<size_t>(e)] = gating_sigmoid(ldf(logits + e));
                }
                break;
            case MoERouterGating::SqrtSoftplus:
                for (int64_t e = 0; e < E; ++e) {
                    probs[static_cast<size_t>(e)] = gating_sqrt_softplus(ldf(logits + e));
                }
                break;
            }

            std::fill(used.begin(), used.end(), 0);
            for (int64_t j = 0; j < k; ++j) {
                // Ascending scan, strict '>': an exact tie keeps the lower
                // expert index.
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

            // Routed scaling (llama.cpp's w_scale, DeepSeek-V3's 2.5) goes
            // last, so it composes with the renormalization above.
            if (attrs.routed_scaling_factor != 1.0f) {
                for (int64_t j = 0; j < k; ++j) {
                    route_weight[static_cast<size_t>(t * k + j)] *= attrs.routed_scaling_factor;
                }
            }
        }
    }

    // Group the routes by expert: counts -> exclusive prefix sum -> slots, so
    // one expert's routes are contiguous; route_row[s] maps a slot back to the
    // token it came from.
    PoolArray<int32_t> expert_off;
    expert_off.zero(static_cast<size_t>(E) + 1);
    for (int64_t r = 0; r < total; ++r) {
        ++expert_off[static_cast<size_t>(route_expert[static_cast<size_t>(r)]) + 1];
    }
    for (int64_t e = 0; e < E; ++e) {
        expert_off[static_cast<size_t>(e) + 1] += expert_off[static_cast<size_t>(e)];
    }

    PoolArray<int32_t> slot(static_cast<size_t>(total));
    PoolArray<int32_t> route_row(static_cast<size_t>(total));
    {
        PoolArray<int32_t> cursor;
        cursor.reset(static_cast<size_t>(E));
        std::memcpy(cursor.data(), expert_off.data(), static_cast<size_t>(E) * sizeof(int32_t));
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

    // FC1 output for every route. The two GEMM stages are separated by a
    // barrier — a second-stage task reads a whole gate row, which several
    // first-stage tasks may have written — so this spans all routes instead of
    // living inside one expert's task.
    PoolPtr gate_owner(static_cast<size_t>(total) * static_cast<size_t>(I) * sizeof(T));
    T* gates = gate_owner.as<T>();
    PoolPtr value_owner;
    T* values = nullptr;
    if (is_swiglu) {
        value_owner = PoolPtr(static_cast<size_t>(total) * static_cast<size_t>(I) * sizeof(T));
        values = value_owner.as<T>();
    }

    const size_t row_bytes = static_cast<size_t>(H) * sizeof(T);
    ctx.cpu.run(0, total, [&](int64_t s) {
        std::memcpy(gather + s * H,
                    in_ptr + static_cast<size_t>(route_row[static_cast<size_t>(s)]) * in_row_stride,
                    row_bytes);
    });

    // Phase A: one task per (expert, output-channel block), over both GEMM
    // stages. Splitting the N side is the only split that does not re-read an
    // expert's weights: an output-channel slice of B is a contiguous run of
    // rows and each task packs only its own, while splitting M would pack the
    // whole weight once per chunk.
    //
    // The block count targets sixteen tasks per worker across all experts —
    // enough to absorb both an uneven route histogram and the static task
    // order. With fewer experts than workers, expert-parallel alone leaves the
    // pool half idle; with more, it already saturates it and the blocks only
    // add granularity. Two floors keep the slice a GEMM rather than a
    // reduction in disguise: never below 128 channels, and the span is rounded
    // to the 16-wide pack width so every slice but the last is a whole panel.
    const int64_t workers = std::max<int64_t>(1, ctx.cpu.worker_count());
    constexpr int64_t kTasksPerWorker = 16;
    constexpr int64_t kMinSlice = 128;
    constexpr int64_t kSliceAlign = 16;

    const auto blocks_for = [&](int64_t N) {
        int64_t b = (kTasksPerWorker * workers + E - 1) / E;
        const int64_t widest = (N + kMinSlice - 1) / kMinSlice;
        if (b > widest) { b = widest; }
        return std::max<int64_t>(b, 1);
    };
    // Slice width: the block count's share of N, rounded up to a whole panel.
    const auto span_of = [](int64_t N, int64_t b) {
        const int64_t share = (N + b - 1) / b;
        return (share + kSliceAlign - 1) / kSliceAlign * kSliceAlign;
    };
    const int64_t ib = blocks_for(I);
    const int64_t hb = blocks_for(H);

    // The inner GEMM stays serial: this stage is already spread over the pool,
    // and a nested parallel region would either oversubscribe or serialize
    // inside the outer team.
    const ComputeContext serial_ctx;

    // FC1's bias and activation fold into the GEMM's output pass; the SwiGLU
    // halves take neither, since the combine pass reads the gate raw.
    Epilogue act_ep{};
    act_ep.type = is_swiglu ? EpilogueActivateType::None
                            : epilogue_of(attrs.activation);

    // Stage 1: FC1 (and FC3 for SwiGLU) on one output-channel slice, then — for
    // SwiGLU only — the gate×value combine, confined to that slice.
    ctx.cpu.run(0, E * ib, [&](int64_t task) {
        const int64_t e = task / ib;
        const int64_t begin = expert_off[static_cast<size_t>(e)];
        const int64_t n_e = expert_off[static_cast<size_t>(e) + 1] - begin;
        if (n_e == 0) { return; }

        const int64_t span = span_of(I, ib);
        const int64_t n0 = (task % ib) * span;
        if (n0 >= I) { return; }
        const int64_t n1 = std::min(I, n0 + span);
        const int64_t w = n1 - n0;

        const T* a = gather + begin * H;
        T* gate_row = gates + begin * I;
        T* gate = gate_row + n0;

        // FC1: [n_e, H] x [w, H]^T -> [n_e, w] (the gate half).
        expert_gemm<T>(a, n_e, H, fc1_base + (e * I + n0) * fc1_row_bytes, w, fc1_dt,
                       fc1b_ptr ? fc1b_ptr + e * I + n0 : nullptr, act_ep,
                       gate, I, dt, serial_ctx);

        if (is_swiglu) {
            T* value_row = values + begin * I;
            T* value = value_row + n0;

            // FC3: the value half, same shape.
            expert_gemm<T>(a, n_e, H, fc3_base + (e * I + n0) * fc3_row_bytes, w, fc3_dt,
                           fc3b_ptr ? fc3b_ptr + e * I + n0 : nullptr, Epilogue{},
                           value, I, dt, serial_ctx);

            for (int64_t r = 0; r < n_e; ++r) {
                T* g = gate_row + r * I + n0;
                const T* v = value_row + r * I + n0;
                for (int64_t c = 0; c < w; ++c) {
                    stf(g + c, swiglu_combine(ldf(g + c), ldf(v + c),
                                              attrs.activation_alpha, attrs.activation_beta,
                                              attrs.swiglu_limit));
                }
            }
        }
    });

    // Stage 2: FC2 over an output-channel slice of H, straight into the
    // expert's routes.
    ctx.cpu.run(0, E * hb, [&](int64_t task) {
        const int64_t e = task / hb;
        const int64_t begin = expert_off[static_cast<size_t>(e)];
        const int64_t n_e = expert_off[static_cast<size_t>(e) + 1] - begin;
        if (n_e == 0) { return; }

        const int64_t span = span_of(H, hb);
        const int64_t n0 = (task % hb) * span;
        if (n0 >= H) { return; }
        const int64_t n1 = std::min(H, n0 + span);
        const int64_t w = n1 - n0;

        T* out_slice = eout + begin * H;
        // [n_e, I] x [w, I]^T -> [n_e, w].
        expert_gemm<T>(gates + begin * I, n_e, I, fc2_base + (e * H + n0) * fc2_row_bytes,
                       w, fc2_dt, fc2b_ptr ? fc2b_ptr + e * H + n0 : nullptr, Epilogue{},
                       out_slice + n0, H, dt, serial_ctx);
    });

    // Phase B: one token per task, its K contributions summed in j order. On the
    // f16 path this f32 accumulator is the only precision under our control —
    // each expert's GEMM result arrives stored as f16.
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

Status moe_cpu(const MoEAttributes& attrs,
             TensorView& output,
             std::span<const TensorView> inputs,
             const ComputeContext& ctx,
             void* workspace)
{
    // Sigmoid scores sit in (0, 1) and do not sum to 1 (sqrt-softplus scores
    // are unbounded and likewise do not sum to 1), so leaving them unnormalized
    // scales the routed output down. That is well defined and occasionally
    // intended, hence a warning rather than an assertion — but it is far more
    // often a forgotten `normalize_routing_weights`. It cannot fire when
    // `router_weights` supplies the mixing weights, since then the scores only
    // select.
    NNOPS_WARN_MSG(attrs.router_gating == MoERouterGating::Softmax ||
                       attrs.normalize_routing_weights ||
                       has_input(inputs, 8),
                   "moe_cpu: non-softmax router gating without "
                   "normalize_routing_weights: the routed output will be "
                   "systematically smaller than under softmax");

    switch (inputs[0].data_type()) {
    case DataType::f32: moe_cpu_impl<float>(attrs, output, inputs, ctx); return Status::ok;
    case DataType::f16: moe_cpu_impl<half>(attrs, output, inputs, ctx);  return Status::ok;
    default: break;
    }
    return Status::unsupported;
}

}  // namespace nnops::backend::cpu
