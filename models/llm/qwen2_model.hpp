#pragma once
/// @file qwen2_model.hpp
/// @brief A Qwen2/Qwen2.5 decoder driven by this repository's operators, with
///        fake q4_1 weights and a dense (continuous) q8 KV cache.
///
/// The forward pass is the HF/llama.cpp graph, one operator per step:
///
///   x = embed(tokens)
///   per layer:
///     q,k,v = Linear(norm(x)) (+ q/k/v bias)      q4_1, activation dtype
///     q,k   = RoPE(permute(q,k), pos, NeoX)        whole head, base 1e6
///     cache = append(q8(k), q8(v))                per-token symmetric scale
///     a     = GQA-causal(q, k_cache, v_cache, seqlens, pos)
///     x     = x + Linear_o(permute(a))
///     x     = x + Linear_down(silu(Linear_gate(norm(x))) * Linear_up(norm(x)))
///   logits = Linear_lm(norm(x[last]))
///
/// Two things are deliberately outside the operator set: the q8 quantization
/// of the cache rows (the cache is a plain dense buffer the caller owns, and
/// the operator reads it as an s8 tensor with f32 scales) and the swiglu
/// multiply, which is Eltwise(Mul) because Linear's fused gated form accepts
/// only matching fp operands, not a q4_1 weight. Everything else is an op.
///
/// The KV cache is continuous: one [H_kv, capacity, D] plane per layer, back to
/// back, with a [H_kv, capacity] f32 scale plane beside it. There is no block
/// table, so a step writes its rows at `length` directly and the attention op
/// sees the whole plane with `kv_seqlens` bounding the valid prefix.

#include "models/llm/qwen2_config.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/data_type.hpp"
#include "nnops/core/status.hpp"
#include "nnops/core/tensor_view.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace nnops::llm {

/// A CUDA allocation owned by a Buffer or a Q4Matrix. Empty in a build without
/// the CUDA backend (and in the CPU run of a build that has it).
struct DeviceBlock {
    void* ptr = nullptr;
    size_t bytes = 0;

    DeviceBlock() = default;
    ~DeviceBlock();
    DeviceBlock(const DeviceBlock&) = delete;
    DeviceBlock& operator=(const DeviceBlock&) = delete;
    DeviceBlock(DeviceBlock&& other) noexcept;
    DeviceBlock& operator=(DeviceBlock&& other) noexcept;

    bool ok() const noexcept { return ptr != nullptr; }
    void reset() noexcept;
};

/// One llama.cpp Q4_1 matrix: `rows` rows of `cols` values, each row padded to
/// whole 32-value blocks, 20 bytes a block.
struct Q4Matrix {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t blocks = 0;
    std::vector<uint8_t> bytes; ///< host copy
    DeviceBlock dev;            ///< device copy, when uploaded

    void* data() const noexcept { return dev.ok() ? dev.ptr : const_cast<uint8_t*>(bytes.data()); }

    TensorView view() const noexcept
    {
        const int64_t shape[2] = {rows, cols};
        return TensorView(shape, DataType::q4_1, data());
    }

    size_t nbytes() const noexcept
    {
        return dev.ok() ? dev.bytes : bytes.size();
    }

    /// Copy the block stream to a device allocation; no-op without CUDA.
    bool to_device();
};

/// A dense row-major tensor in the activation dtype (f32 or f16) — the
/// embedding table, the normalization weights, the q/k/v biases and every
/// activation scratch.
struct Buffer {
    std::vector<int64_t> shape;
    DataType dtype = DataType::f32;
    std::vector<uint8_t> bytes; ///< host storage (empty for a device buffer)
    DeviceBlock dev;            ///< device storage, when allocated on the device

    /// Host allocation, or a device one when @p on_device asks for it.
    void alloc(std::vector<int64_t> shape_, DataType dtype_, bool on_device = false);
    int64_t numel() const noexcept;
    size_t nbytes() const noexcept;
    bool empty() const noexcept { return numel() == 0; }
    void* data() noexcept { return dev.ok() ? dev.ptr : bytes.data(); }
    const void* data() const noexcept
    {
        return dev.ok() ? const_cast<const void*>(dev.ptr) : bytes.data();
    }
    TensorView view() noexcept;
    TensorView view() const noexcept;

    /// Host <-> device copies of the whole buffer (no-op without CUDA).
    bool upload(const void* src);
    bool download(void* dst) const;
};

struct LayerWeights {
    Q4Matrix q_proj; ///< [H*D, hidden]
    Q4Matrix k_proj; ///< [H_kv*D, hidden]
    Q4Matrix v_proj; ///< [H_kv*D, hidden]
    Q4Matrix o_proj; ///< [hidden, H*D]
    Q4Matrix gate_proj; ///< [inter, hidden]
    Q4Matrix up_proj;   ///< [inter, hidden]
    Q4Matrix down_proj; ///< [hidden, inter]

    Buffer q_bias; ///< [H*D], activation dtype, when attention_bias
    Buffer k_bias; ///< [H_kv*D]
    Buffer v_bias; ///< [H_kv*D]

    Buffer attn_norm; ///< [hidden]
    Buffer ffn_norm;  ///< [hidden]
};

/// The whole fake checkpoint: q4_1 projections and a q4_1 embedding table,
/// which the lookup decodes a row at a time (Embed itself has no q4_1 form).
/// Only the norm weights and the q/k/v biases stay dense.
struct Qwen2Weights {
    Qwen2Config cfg;
    DataType act_dtype = DataType::f32;
    Q4Matrix embed;    ///< [vocab, hidden]
    Buffer final_norm; ///< [hidden]
    std::vector<LayerWeights> layers;
    Q4Matrix lm_head;  ///< [vocab, hidden]; empty when the table is tied

    /// Copy every weight to the device; false when a CUDA call fails.
    bool to_device();

    size_t q4_nbytes() const noexcept;
    size_t dense_nbytes() const noexcept;
    size_t total_nbytes() const noexcept { return q4_nbytes() + dense_nbytes(); }
};

/// Build random q4_1 projections and a random dense embedding table. The same
/// seed and config produce the same weights; `act_dtype` selects the dense
/// tensors' storage only (the q4_1 bytes are identical for f32 and f16).
Qwen2Weights make_fake_weights(const Qwen2Config& cfg, DataType act_dtype, uint64_t seed);

/// Wall time spent in each phase, accumulated by run() while profiling is on.
/// The operator calls are blocking, so the time around each call is the whole
/// kernel; the phases partition the forward pass except for the top-level
/// bookkeeping.
struct Qwen2Profile {
    double embed_ms = 0.0;
    double norm_ms = 0.0;
    double qkv_ms = 0.0;
    double permute_ms = 0.0;
    double rope_ms = 0.0;
    double cache_ms = 0.0;
    double attention_ms = 0.0;
    double o_proj_ms = 0.0;
    double residual_ms = 0.0;
    double ffn_gate_ms = 0.0;
    double ffn_up_ms = 0.0;
    double ffn_mul_ms = 0.0;
    double ffn_down_ms = 0.0;
    double lm_head_ms = 0.0;
    double total_ms = 0.0; ///< sum of the phases above
};

/// One conversation's mutable state: the dense q8 KV cache, the activation
/// scratch, and the operator instances the forward pass reuses.
class Qwen2State {
public:
    /// @p backend runs the graph where the operators execute; a device run
    /// needs @p stream (a cudaStream_t), f16 activations, and weights that went
    /// through Qwen2Weights::to_device().
    Qwen2State(const Qwen2Config& cfg, DataType act_dtype, int64_t capacity,
               Backend backend = Backend::CPU, void* stream = nullptr);
    ~Qwen2State();

    Qwen2State(const Qwen2State&) = delete;
    Qwen2State& operator=(const Qwen2State&) = delete;

    /// Forget every cached row; the next run starts at position 0.
    void reset() noexcept;

    int64_t length() const noexcept;
    int64_t capacity() const noexcept;
    size_t cache_nbytes() const noexcept;
    Backend backend() const noexcept;

    /// Run @p tokens at the current position, appending their K/V to the cache.
    /// Prefill and decode are the same call: S > 1 is a prefill block, S == 1 a
    /// decode step. @p last_logits is resized to the vocabulary and filled with
    /// the last token's f32 logits (whatever the activation dtype).
    Status run(const Qwen2Weights& weights, std::span<const int64_t> tokens,
               const ComputeContext& ctx, std::vector<float>& last_logits);

    /// Per-phase wall-time accounting; off means run() skips the clock reads.
    void set_profile(bool on) noexcept;
    void reset_profile() noexcept;
    const Qwen2Profile& profile() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nnops::llm
